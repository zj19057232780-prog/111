"""Shared raw/STFT windows with domain isolation and deterministic paired episodes."""
from fractions import Fraction
from pathlib import Path
import json
import math
import cv2
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from config_cross_dataset import fingerprint
from cwru_preprocess import load_metadata_ids, load_drive_end_signal
from pu_loader import PULoader
from pu_preprocess import compute_stft_log_image
from hust_preprocess import digest, window_starts
from hust_dataset import HUSTTasks

DATA_KEYS = ('protocol', 'samples_per_block', 'pu_record', 'stft_fs', 'window_size',
             'img_size', 'normalize', 'stft_nperseg', 'stft_noverlap', 'stft_window',
             'stft_bandpass', 'stft_log_eps', 'stft_p_low', 'stft_p_high', 'stft_gamma')


def data_signature(cfg):
    return {k: cfg[k] for k in DATA_KEYS}


def records(cfg):
    result = []
    root = Path(cfg['cwru_raw_root'])
    ids = load_metadata_ids(root)
    for cls, subset, name in [('N', 'NormalBaseline', 'Normal.mat'),
                              ('I', '48DriveEndFault', '0.007-InnerRace.mat'),
                              ('O', '48DriveEndFault', '0.007-OuterRace6.mat')]:
        x, variable, path = load_drive_end_signal(root, subset, '1730', name, ids, cfg)
        result.append(('A', cls, path, 48000, variable, x))
    loader = PULoader(cfg['pu_raw_root'])
    for cls, bearing in [('N', 'K001'), ('I', 'KI01'), ('O', 'KA01')]:
        filename = f"N15_M01_F10_{bearing}_{cfg['pu_record']}.mat"
        matches = list(Path(cfg['pu_raw_root']).rglob(filename))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one PU file {filename}, found {matches}')
        path = matches[0]
        x = loader.get_signal(loadmat(path))
        result.append(('B', cls, path, 64000, 'vibration_1 via PULoader', x))
    for cls in ('N', 'I', 'O'):
        matches = list(Path(cfg['hust_raw_root']).rglob(cls+'500.mat'))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one HUST {cls}500.mat, found {matches}')
        path = matches[0]
        x = np.asarray(loadmat(path, variable_names=['data'])['data']).squeeze()
        result.append(('C', cls, path, 51200, 'data', x))
    for domain, cls, path, fs, variable, x in result:
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError(f'Invalid vibration record: {path}')
    return result


def preprocess_cross(cfg):
    root = Path(cfg['processed_root'])
    manifest_path = root / 'manifest.json'
    source = records(cfg)
    raw = {f'{d}/{c}': dict(path=str(p.resolve()), sha256=digest(p), fs=fs,
                           variable=v, length=len(x)) for d, c, p, fs, v, x in source}
    if root.exists() and any(root.iterdir()):
        if not manifest_path.exists():
            raise ValueError('Incomplete output; choose a new --processed-root (no automatic deletion)')
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['signature'] != data_signature(cfg) or old['records'] != raw:
            raise ValueError('Existing cross data differs; choose a new --processed-root')
        for row in old['windows']:
            for field in ('image', 'raw'):
                if digest(root / row[field]) != row[field+'_sha256']:
                    raise ValueError('Existing cross artifact corrupted')
        return old
    # Split each original record before filtering/resampling; guard prevents boundary mixing.
    prepared = []
    for domain, cls, path, fs, variable, x in source:
        ratio = Fraction(cfg['stft_fs'], fs)
        mid, gap = len(x)//2, math.ceil(cfg['window_size'] * fs/cfg['stft_fs'])
        for block, (begin, end) in enumerate(((0, mid), (mid+gap, len(x)))):
            signal = resample_poly(x[begin:end].astype(np.float64), ratio.numerator, ratio.denominator)
            starts = window_starts(0, len(signal), cfg['window_size'], cfg['samples_per_block'])
            for index, start in enumerate(starts):
                prepared.append((domain, cls, block, index, int(start), begin, end, fs,
                                 signal[start:start+cfg['window_size']].copy()))
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain, cls, block, index, start, begin, end, fs, segment in prepared:
        name = f'{domain}/{cls}/b{block}_{index:03d}'
        image_path, raw_path = root / (name+'.png'), root / (name+'.npy')
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = compute_stft_log_image(segment, cfg)
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise IOError('PNG encoding failed')
        image_path.write_bytes(encoded.tobytes())
        np.save(raw_path, segment, allow_pickle=False)
        rows.append(dict(domain=domain, fault=cls, block=block, record=f'{domain}/{cls}',
                         original_begin=begin, original_end=end, original_fs=fs,
                         start=start, end=start+cfg['window_size'],
                         image=name+'.png', raw=name+'.npy',
                         image_sha256=digest(image_path), raw_sha256=digest(raw_path)))
    manifest = dict(signature=data_signature(cfg), records=raw, windows=rows)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Cross preprocessing: {len(rows)} aligned raw/STFT windows, {root}', flush=True)
    return manifest


class CrossDataset:
    def __init__(self, cfg):
        self.cfg, self.root = cfg, Path(cfg['processed_root'])
        manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['signature'] != data_signature(cfg):
            raise ValueError('Preprocessing signature mismatch; use matching cross data')
        self.signature = fingerprint(dict(data=digest(self.root/'manifest.json'), task=cfg['task']))
        self.source_classes = cfg['class_names']
        rows = manifest['windows']
        seen, grouped = set(), {}
        for row in rows:
            if row['image'] in seen or row['domain'] not in ('A', 'B', 'C') or row['fault'] not in self.source_classes:
                raise ValueError('Invalid or duplicate cross window')
            if row['record'] != row['domain']+'/'+row['fault']:
                raise ValueError('Window record belongs to a different domain/class')
            seen.add(row['image'])
            record = manifest['records'][row['record']]
            mid = record['length']//2
            gap = math.ceil(cfg['window_size']*record['fs']/cfg['stft_fs'])
            expected = (0, mid) if row['block'] == 0 else (mid+gap, record['length'])
            if row['block'] not in (0, 1) or (row['original_begin'], row['original_end']) != expected:
                raise ValueError('Target block crosses isolation boundary')
            capacity = math.ceil((expected[1]-expected[0])*cfg['stft_fs']/record['fs'])
            if row['end']-row['start'] != cfg['window_size'] or not 0 <= row['start'] < row['end'] <= capacity:
                raise ValueError('Cross window out of bounds')
            grouped.setdefault((row['domain'], row['fault'], row['block']), []).append(row)
        for d in 'ABC':
            for c in self.source_classes:
                for block in (0, 1):
                    group = sorted(grouped.get((d, c, block), []), key=lambda r: r['start'])
                    if len(group) != cfg['samples_per_block'] or any(a['end'] > b['start'] for a, b in zip(group, group[1:])):
                        raise ValueError('Wrong cross counts or overlapping windows')
        self.by_split, self._cache = {}, {}
        for split in ('train', 'validation', 'test'):
            domain = cfg['source_domain'] if split == 'train' else cfg['target_domain']
            selected = [r for r in rows if r['domain'] == domain and
                        (split == 'train' or r['block'] == (0 if split == 'validation' else 1))]
            self.by_split[split] = {c: [r for r in selected if r['fault'] == c] for c in self.source_classes}

    def images(self, split):
        if split not in self._cache:
            groups = {}
            for cls, rows in self.by_split[split].items():
                values = []
                for row in rows:
                    field = 'raw' if self.cfg['method'] == 'tl_wdcnn' else 'image'
                    path = self.root / row[field]
                    if digest(path) != row[field+'_sha256']:
                        raise ValueError(f'Corrupt cross artifact: {path}')
                    if field == 'raw':
                        x = np.load(path, allow_pickle=False)
                        if x.shape != (self.cfg['window_size'],) or not np.isfinite(x).all():
                            raise ValueError('Invalid raw window')
                        x = x - x.mean()
                        if self.cfg['normalize'] == 'zscore':
                            x = x/(x.std()+1e-8)
                    else:
                        x = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 0)
                        if x is None or x.shape != (self.cfg['img_size'],)*2:
                            raise ValueError('Invalid cross STFT image')
                        x = x.astype(np.float32)/255
                    values.append(x[None].astype(np.float32))
                groups[cls] = np.stack(values)
            self._cache[split] = groups
        return self._cache[split]


class CrossTasks(HUSTTasks):
    """Reuse generic episode mechanics; both representations share image IDs."""
    pass
