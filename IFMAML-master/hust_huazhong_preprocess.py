"""Strict parser for the instrument TSV exports bearing an .xls suffix."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pu_preprocess import compute_stft_log_image, save_grayscale_png
from config_hust_huazhong import CONDITIONS, CLASSES, PREFIXES, configuration, CACHE_PROTOCOL


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_signal(path, channel='X'):
    path = Path(path)
    if channel not in ('X', 'Y', 'Z'):
        raise ValueError('Only vibration channels X/Y/Z are allowed, never Speed/time')
    with path.open(encoding='utf-8-sig') as stream:
        header = []
        for line in stream:
            if line.strip() == 'Data':
                break
            header.append(line.strip())
        else:
            raise ValueError(f'Missing Data marker: {path}')
        legend = [s.split('\t')[1:] for s in header if s.startswith('Legend\t')]
        if legend != [['Speed', 'X', 'Y', 'Z']]:
            raise ValueError(f'Unexpected channel header: {path}: {legend}')
        if header[0].split() != ['Title:', path.stem]:
            raise ValueError(f'Title/filename mismatch: {path}')
        declared = [int(s.split('\t')[1]) for s in header if s.startswith('Total Data Rows\t')]
        data = np.loadtxt(stream, delimiter='\t', dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 5 or declared != [len(data)] or len(data) != 262144:
        raise ValueError(f'Unexpected dimensions/record length: {path}: {data.shape}, {declared}')
    if not np.isfinite(data).all():
        raise ValueError(f'Nonfinite data: {path}')
    # Export timestamps are rounded to six decimals, so check against the full grid.
    if not np.allclose(data[:, 0], np.arange(len(data))/25600, atol=0.00000051, rtol=0):
        raise ValueError(f'Time column is not a contiguous 25.6 kHz grid: {path}')
    signal = data[:, {'X':2, 'Y':3, 'Z':4}[channel]]
    if np.std(signal) == 0:
        raise ValueError(f'Constant vibration channel: {path}')
    return signal


def split_indices(count, fraction=.5, gap=1):
    boundary = int(count*fraction)
    train = list(range(boundary-gap))
    validation = list(range(boundary, count))
    if not train or not validation:
        raise ValueError('Record too short for disjoint target validation/test')
    return train, validation


def preprocessing_identity(cfg):
    keys = ('window_size', 'img_size', 'channel', 'normalize', 'target_validation_fraction', 'split_gap_windows')
    return {k:v for k,v in cfg.items() if k in keys or k.startswith('stft_')}


def verify_cache(root, cfg=None):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol'] != CACHE_PROTOCOL or (cfg and manifest['preprocessing'] != preprocessing_identity(cfg)):
        raise ValueError('Preprocessing configuration mismatch; use a fresh processed directory')
    expected = {f'{c}/{cls}/w{i:03d}.png' for c in CONDITIONS for cls in CLASSES for i in range(64)}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*.png')}
    if actual != expected or {x['image'] for x in manifest['windows']} != expected:
        raise ValueError('Missing/extra cached images')
    if len(manifest['windows']) != len(expected):
        raise ValueError('Duplicate window entries')
    if (manifest['source_train_indices'] != list(range(64)) or
        manifest['target_validation_indices'] != split_indices(64)[0] or
        manifest['target_test_indices'] != split_indices(64)[1]):
        raise ValueError('Invalid split manifest')
    for row in manifest['windows']:
        if row['class_name'] not in CLASSES or row['condition'] not in map(str, CONDITIONS):
            raise ValueError('Invalid class or condition')
        if (row['raw_file'] != f"{PREFIXES[row['class_name']]}_{row['condition']}Hz.xls" or
            row['image'] != f"{row['condition']}/{row['class_name']}/w{row['index']:03d}.png" or
            row['channel'] != manifest['preprocessing']['channel']):
            raise ValueError('Wrong class/channel/window mapping')
        if row['stop']-row['start'] != 4096 or row['start'] != row['index']*4096:
            raise ValueError('Invalid/overlapping window manifest')
        if sha256(root/row['image']) != row['image_sha256']:
            raise ValueError('Cached image hash mismatch: '+row['image'])
    return manifest


def preprocess(cfg):
    raw = Path(cfg['raw_root'])/'Raw data (原始数据)'
    root = Path(cfg['root_path'])
    files = {(str(c), cls):raw/f'{PREFIXES[cls]}_{c}Hz.xls' for c in CONDITIONS for cls in CLASSES}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(str(missing))
    hashes = {p.name:sha256(p) for p in files.values()}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError('Duplicate raw records across classes/conditions')
    if root.exists():
        manifest = verify_cache(root, cfg)
        if manifest['raw_sha256'] != hashes:
            raise ValueError('Raw data changed; choose a fresh processed directory')
        return manifest
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for (condition, cls), path in files.items():
        signal = read_signal(path, cfg['channel'])
        for i, start in enumerate(range(0, len(signal)-4096+1, 4096)):
            relative = f'{condition}/{cls}/w{i:03d}.png'
            dest = root/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            save_grayscale_png(compute_stft_log_image(signal[start:start+4096], cfg), str(dest))
            rows.append(dict(condition=condition, class_name=cls, raw_file=path.name,
                channel=cfg['channel'], index=i, start=start, stop=start+4096,
                image=relative, image_sha256=sha256(dest)))
        print(f'[preprocess] {condition}Hz {cls}: {len(signal)//4096} nonoverlapping windows', flush=True)
    manifest = dict(protocol=CACHE_PROTOCOL, preprocessing=preprocessing_identity(cfg),
        raw_sha256=hashes, windows=rows, source_train_indices=list(range(64)),
        target_validation_indices=split_indices(64)[0], target_test_indices=split_indices(64)[1])
    save_json(root/'manifest.json', manifest)
    verify_cache(root, cfg)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path)
    p.add_argument('--processed-root', type=Path)
    p.add_argument('--channel', choices=('X','Y','Z'), default='X')
    args=p.parse_args(); cfg=configuration('H1','full',3,channel=args.channel)
    if args.raw_root: cfg['raw_root']=str(args.raw_root.resolve())
    if args.processed_root: cfg['root_path']=str(args.processed_root.resolve())
    preprocess(cfg)

if __name__ == '__main__':
    main()
