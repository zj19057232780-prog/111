"""Record-indexed, nonoverlapping HUST STFT data preparation."""
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from pu_preprocess import compute_stft_log_image

PATTERN = re.compile(r'^(N|I|O|B|IO|IB|OB)([4-8])0([024])$', re.I)
DATA_KEYS = ('protocol', 'class_names', 'bearing_types', 'source_condition',
             'target_condition', 'source_samples', 'validation_samples', 'test_samples',
             'window_size', 'img_size', 'normalize', 'stft_fs', 'stft_nperseg',
             'stft_noverlap', 'stft_window', 'stft_bandpass', 'stft_log_eps',
             'stft_p_low', 'stft_p_high', 'stft_gamma')


def data_signature(cfg):
    return {k: cfg[k] for k in DATA_KEYS}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover_records(cfg):
    records, excluded = {}, []
    for path in sorted(Path(cfg['raw_data_root']).rglob('*.mat')):
        match = PATTERN.fullmatch(path.stem)
        if not match:
            raise ValueError(f'Unrecognized HUST MAT filename: {path}')
        fault, bearing, load = match.groups()
        bearing, load = 6200 + int(bearing), int(load) * 100
        if bearing not in cfg['bearing_types']:
            excluded.append(path.name)
            continue
        key = (fault.upper(), bearing, load)
        if key in records:
            raise ValueError(f'Duplicate record identity: {key}')
        records[key] = path
    expected = {(c, b, load) for c in cfg['class_names']
                for b in cfg['bearing_types'] for load in (0, 200, 400)}
    if set(records) != expected:
        raise ValueError(f'HUST record matrix mismatch; missing={expected-set(records)}, '
                         f'extra={set(records)-expected}')
    return records, excluded


def read_signal(path):
    value = loadmat(path, variable_names=['data']).get('data')
    if value is None:
        raise ValueError(f'Missing steady vibration data: {path}')
    x = np.asarray(value).squeeze()
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f'Expected finite single-channel data: {path}')
    return x.astype(np.float64)


def window_starts(begin, end, width, count):
    if count <= 0 or end-begin < width*count:
        raise ValueError(f'Insufficient disjoint-window capacity: {begin}:{end}, {count}x{width}')
    starts = np.linspace(begin, end-width, count, dtype=np.int64)
    if len(starts) > 1 and np.any(np.diff(starts) < width):
        raise ValueError('Window overlap')
    return starts


def preprocess_hust(cfg):
    records, excluded = discover_records(cfg)
    root = Path(cfg['root_path'])
    manifest_path = root / 'manifest.json'
    if root.exists() and any(root.iterdir()):
        if not manifest_path.exists():
            raise ValueError(f'Incomplete output; use a new --processed-root: {root}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['signature'] != data_signature(cfg):
            raise ValueError('Existing HUST data configuration differs; use a new --processed-root')
        current = {p.name: digest(p) for p in records.values()}
        if current != manifest['raw_sha256']:
            raise ValueError('Raw HUST records changed; use a new --processed-root')
        for row in manifest['windows']:
            if digest(root / row['image']) != row['sha256']:
                raise ValueError(f'Image checksum mismatch: {row["image"]}')
        print(f"Verified existing {cfg['task']}: {len(manifest['windows'])} images")
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    rows, raw_hashes = [], {}
    width = cfg['window_size']
    for (fault, bearing, load), path in sorted(records.items()):
        x = read_signal(path)
        raw_hashes[path.name] = digest(path)
        if load in cfg['source_condition']:
            splits = [('train', 0, len(x), cfg['source_samples'])]
        else:
            mid = len(x)//2
            splits = [('validation', 0, mid, cfg['validation_samples']),
                      ('test', mid+width, len(x), cfg['test_samples'])]
        for split, begin, end, count in splits:
            for start in window_starts(begin, end, width, count):
                start = int(start)
                relative = f'{split}/{fault}/{path.stem}_{start:07d}.png'
                output = root / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                image = compute_stft_log_image(x[start:start+width], cfg)
                ok, encoded = cv2.imencode('.png', image)
                if not ok:
                    raise IOError(f'Failed to encode: {output}')
                output.write_bytes(encoded.tobytes())
                rows.append(dict(image=relative, record=path.name, bearing=bearing,
                                 load=load, fault=fault, split=split, start=start,
                                 end=start+width, length=len(x), sha256=digest(output)))
    manifest = dict(signature=data_signature(cfg), excluded_6204=excluded,
                    raw_sha256=raw_hashes, windows=rows)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Prepared {cfg['task']}: {len(records)} records, {len(rows)} images; excluded={len(excluded)}")
    return manifest
