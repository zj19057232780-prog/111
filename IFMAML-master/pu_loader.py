import os

import numpy as np
import scipy.io as scio


def _unwrap_scalar_object(value):
    """Repeatedly unwrap MATLAB scalar object cells."""
    x = value
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    return x


def _field_to_text(cell):
    """Read MATLAB char/object fields as a plain Python string."""
    try:
        x = _unwrap_scalar_object(np.asarray(cell))
        if isinstance(x, str):
            return x.strip()

        arr = np.asarray(x)
        if arr.size == 0:
            return ''
        if arr.dtype.kind in 'US':
            return ''.join(arr.ravel().astype(str)).strip()
        if arr.dtype == object:
            parts = [_field_to_text(v) for v in arr.ravel()]
            return ' '.join(p for p in parts if p).strip()
        if arr.size == 1:
            return str(arr.ravel()[0]).strip()
    except Exception:
        return ''
    return ''


def _numeric_to_1d(x):
    x = _unwrap_scalar_object(np.asarray(x))
    if isinstance(x, np.ndarray) and x.dtype == object:
        x = _unwrap_scalar_object(x)
    arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != np.bool_:
        raise TypeError(f'non-numeric array dtype={arr.dtype}')
    return arr.astype(np.float64).ravel()


def _looks_time_like(sig):
    """Detect monotonic time/rpm-like ramps so they are not picked as signals."""
    sig = np.asarray(sig, dtype=np.float64).ravel()
    if sig.size < 8:
        return True
    diffs = np.diff(sig[:min(sig.size, 4096)])
    if diffs.size == 0:
        return True
    mono_ratio = max(np.mean(diffs >= -1e-12), np.mean(diffs <= 1e-12))
    return mono_ratio > 0.995


def _channel_score(name, sig):
    lower = name.lower()
    score = float(np.log1p(np.asarray(sig).size))

    for key, bonus in (
        ('vibration', 120),
        ('accelerometer', 110),
        ('acceleration', 105),
        ('beschleunigung', 105),
        ('schwing', 100),
        ('bearing', 80),
        ('acc', 70),
    ):
        if key in lower:
            score += bonus

    for key, penalty in (
        ('time', 80),
        ('speed', 50),
        ('temp', 45),
        ('torque', 35),
        ('current', 30),
        ('force', 10),
    ):
        if key in lower:
            score -= penalty

    if _looks_time_like(sig):
        score -= 60
    if np.std(sig) < 1e-12:
        score -= 100
    return score


def _extract_from_field(cell):
    x = _unwrap_scalar_object(np.asarray(cell))
    if isinstance(x, np.ndarray) and x.dtype == object:
        x = _unwrap_scalar_object(x)

    if isinstance(x, np.ndarray) and x.dtype.names:
        return _structured_to_1d(x)
    if hasattr(x, 'dtype') and x.dtype.names:
        return _structured_to_1d(x)
    return _numeric_to_1d(x)


def _select_best_structured_channel(channels):
    """Pick the most likely vibration channel from a MATLAB channel array."""
    arr = np.asarray(channels)
    if arr.dtype.names is None or 'Data' not in arr.dtype.names:
        raise ValueError('not a channel array')

    candidates = []
    for idx, item in enumerate(arr.reshape(-1)):
        try:
            sig = _extract_from_field(item['Data'])
        except (TypeError, ValueError):
            continue
        if sig.size < 16:
            continue
        name = _field_to_text(item['Name']) if 'Name' in item.dtype.names else ''
        candidates.append((_channel_score(name, sig), idx, name, sig))

    if not candidates:
        raise ValueError('no numeric channel data found')

    candidates.sort(key=lambda x: (x[0], x[3].size), reverse=True)
    return candidates[0][3]


def _structured_to_1d(void_or_row):
    """Extract one likely signal vector from a scipy-loaded MATLAB struct."""
    if void_or_row is None:
        raise ValueError('empty struct')

    if isinstance(void_or_row, np.ndarray) and void_or_row.dtype.names is None:
        return _numeric_to_1d(void_or_row)

    if isinstance(void_or_row, np.ndarray) and void_or_row.dtype.names:
        if 'Data' in void_or_row.dtype.names:
            return _select_best_structured_channel(void_or_row)
        elem = void_or_row.reshape(-1)[0]
        names = elem.dtype.names
        base = elem
    elif hasattr(void_or_row, 'dtype') and void_or_row.dtype.names:
        base = void_or_row
        names = void_or_row.dtype.names
    else:
        return _numeric_to_1d(void_or_row)

    if not names:
        return _numeric_to_1d(void_or_row)

    if 'Data' in names:
        return _extract_from_field(base['Data'])

    for field in ('Y', 'DE_time', 'FE_time', 'Data', 'data', 'vibration', 'sig', 'signal', 'X'):
        if field not in names:
            continue
        try:
            return _extract_from_field(base[field])
        except (TypeError, ValueError):
            continue

    for field in names:
        if field in ('Info', 'Description', 'Label', 'label'):
            continue
        try:
            return _extract_from_field(base[field])
        except (TypeError, ValueError):
            continue

    raise ValueError('no usable numeric channel found in MATLAB struct')


def _extract_signal_from_value(value):
    """Parse one scipy.loadmat variable into a float64 1-D signal."""
    v = np.asarray(value)

    if v.dtype.names:
        if v.ndim >= 1:
            flat = v.reshape(-1)
            if len(flat) == 1:
                return _structured_to_1d(flat[0])
        return _structured_to_1d(v)

    if v.dtype == object:
        inner = _unwrap_scalar_object(v)
        if inner is v:
            raise TypeError('unresolved object array')
        return _extract_signal_from_value(inner)

    if v.ndim == 2 and v.shape[1] == 1:
        return _numeric_to_1d(v[:, 0])
    if v.ndim == 1:
        return _numeric_to_1d(v)
    if v.ndim == 2:
        return _numeric_to_1d(v[:, 0])

    return _numeric_to_1d(v.ravel())


class PULoader:
    """Loader for the normalized PU .mat layout: root/condition/fault.mat."""

    def __init__(self, root_path):
        self.root_path = root_path
        self.file_map = {}
        self._scan_files()

    def _scan_files(self):
        if not os.path.isdir(self.root_path):
            return
        for condition in os.listdir(self.root_path):
            cond_path = os.path.join(self.root_path, condition)
            if not os.path.isdir(cond_path):
                continue
            self.file_map[condition] = {}
            for fname in os.listdir(cond_path):
                if fname.endswith('.mat'):
                    fault_name = fname.replace('.mat', '')
                    self.file_map[condition][fault_name] = os.path.join(cond_path, fname)

    def load_file(self, file_path):
        return scio.loadmat(file_path)

    def get_signal(self, mat_dict):
        preferred_keys = (
            'DE_time',
            'FE_time',
            'Data',
            'data',
            'vibration',
            'sig',
            'signal',
            'measurement',
            'Y',
            'X',
        )
        keys = [k for k in mat_dict.keys() if not k.startswith('__')]

        def try_keys(order):
            for key in order:
                if key not in mat_dict:
                    continue
                try:
                    return _extract_signal_from_value(mat_dict[key])
                except (TypeError, ValueError):
                    continue
            return None

        sig = try_keys([k for k in preferred_keys if k in keys])
        if sig is not None:
            return sig

        sig = try_keys(sorted(keys))
        if sig is not None:
            return sig

        raise KeyError('unknown .mat signal format: no usable numeric variable found')

    def load_condition(self, condition_name):
        if condition_name not in self.file_map:
            raise KeyError(f'unknown condition: {condition_name}')

        fault_signals = {}
        for fault_name, file_path in self.file_map[condition_name].items():
            mat_dict = self.load_file(file_path)
            fault_signals[fault_name] = self.get_signal(mat_dict)
        return fault_signals

    def get_condition_names(self):
        return list(self.file_map.keys())

    def get_fault_names(self, condition_name):
        if condition_name not in self.file_map:
            return []
        return list(self.file_map[condition_name].keys())
