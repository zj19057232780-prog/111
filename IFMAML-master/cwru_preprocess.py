"""Create leakage-controlled STFT images from the local CWRU archive."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import scipy.io as scio

from config_cwru import CWRU_CONFIG
from pu_preprocess import compute_stft_log_image, sample_starts


METADATA_RE = re.compile(r'^(?P<label>.+?)\s+https?://.+/(?P<file_id>\d+)\.mat$')


def load_metadata_ids(raw_root):
    metadata_path = Path(raw_root) / '__file__' / 'metadata.txt'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'CWRU metadata not found: {metadata_path}')

    mapping = {}
    for line in metadata_path.read_text(encoding='utf-8').splitlines():
        match = METADATA_RE.match(line.strip())
        if match is None:
            continue
        label = f'{match.group("label")}.mat'
        mapping[label] = match.group('file_id')
    return mapping


def _metadata_label(subset, speed, filename):
    return f'{subset} {speed} {filename}'


def resolve_signal_variable(mat_path, subset, speed, filename, metadata_ids, cfg):
    relative_key = f'{subset}/{speed}/{filename}'
    override = cfg.get('variable_overrides', {}).get(relative_key)
    mat_dict = scio.loadmat(mat_path)

    if override is not None:
        if override not in mat_dict:
            raise KeyError(f'Configured CWRU variable {override} not found in {mat_path}')
        return override, mat_dict

    label = _metadata_label(subset, speed, filename)
    file_id = metadata_ids.get(label)
    if file_id is None:
        raise KeyError(f'CWRU file is missing from metadata.txt: {label}')

    expected = f'X{int(file_id):03d}_DE_time'
    if expected not in mat_dict:
        available = sorted(k for k in mat_dict if k.endswith('_DE_time'))
        raise KeyError(
            f'Expected {expected} in {mat_path}, available DE variables: {available}'
        )
    return expected, mat_dict


def load_drive_end_signal(raw_root, subset, speed, filename, metadata_ids, cfg):
    mat_path = Path(raw_root) / subset / speed / filename
    if not mat_path.is_file():
        raise FileNotFoundError(f'CWRU source file not found: {mat_path}')

    variable, mat_dict = resolve_signal_variable(
        mat_path,
        subset,
        speed,
        filename,
        metadata_ids,
        cfg,
    )
    signal = np.asarray(mat_dict[variable], dtype=np.float64).ravel()
    if len(signal) < int(cfg['window_size']):
        raise ValueError(
            f'CWRU signal is shorter than one sample window: '
            f'{mat_path} ({len(signal)} < {cfg["window_size"]})'
        )
    return signal, variable, mat_path


def _save_split_images(signal, count, prefix, output_dir, cfg, rng):
    starts = sample_starts(len(signal), int(cfg['window_size']), int(count), rng)
    if len(starts) != int(count):
        raise RuntimeError(
            f'Expected {count} CWRU windows, generated {len(starts)} under {output_dir}'
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(starts):
        segment = signal[int(start):int(start) + int(cfg['window_size'])]
        image = compute_stft_log_image(segment, cfg)
        output_path = output_dir / f'{prefix}_{index:03d}.png'
        if not cv2.imwrite(str(output_path), image):
            raise IOError(f'Failed to write CWRU STFT image: {output_path}')
    return starts


def _clean_output_root(output_root):
    output_root = Path(output_root).resolve()
    project_root = Path(__file__).resolve().parent
    if output_root == project_root or project_root not in output_root.parents:
        raise ValueError(
            f'Refusing to recursively clean a directory outside the project: {output_root}'
        )
    if output_root.exists():
        shutil.rmtree(output_root)


def preprocess_cwru(cfg=None, clean=False):
    cfg = dict(CWRU_CONFIG if cfg is None else cfg)
    raw_root = Path(cfg['raw_data_root'])
    output_root = Path(cfg['root_path'])
    if not raw_root.is_dir():
        raise FileNotFoundError(f'CWRU raw data root not found: {raw_root}')

    if clean:
        _clean_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_ids = load_metadata_ids(raw_root)
    rng = np.random.default_rng(int(cfg.get('preprocess_seed', 24)))
    source_conditions = set(cfg['source_condition'])
    target_condition = cfg['target_condition']
    condition_speeds = cfg['condition_speeds']
    manifest_rows = []

    for condition, speed in condition_speeds.items():
        if condition not in source_conditions and condition != target_condition:
            continue
        is_target = condition == target_condition

        for class_name, spec in cfg['class_specs'].items():
            signal, variable, mat_path = load_drive_end_signal(
                raw_root=raw_root,
                subset=spec['subset'],
                speed=speed,
                filename=spec['filename'],
                metadata_ids=metadata_ids,
                cfg=cfg,
            )
            class_dir = output_root / condition / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for old_image in class_dir.glob('*.png'):
                old_image.unlink()

            if is_target:
                midpoint = len(signal) // 2
                val_signal = signal[:midpoint]
                test_signal = signal[midpoint:]
                split_specs = (
                    (
                        'val',
                        val_signal,
                        int(cfg['target_val_samples_per_class']),
                        0,
                        midpoint,
                    ),
                    (
                        'test',
                        test_signal,
                        int(cfg['target_test_samples_per_class']),
                        midpoint,
                        len(signal),
                    ),
                )
            else:
                split_specs = (
                    (
                        'source',
                        signal,
                        int(cfg['source_samples_per_class']),
                        0,
                        len(signal),
                    ),
                )

            for prefix, split_signal, count, range_start, range_end in split_specs:
                starts = _save_split_images(
                    split_signal,
                    count,
                    prefix,
                    class_dir,
                    cfg,
                    rng,
                )
                manifest_rows.append({
                    'condition': condition,
                    'speed_folder': speed,
                    'class_name': class_name,
                    'split': prefix,
                    'source_mat': str(mat_path),
                    'mat_variable': variable,
                    'signal_range_start': range_start,
                    'signal_range_end': range_end,
                    'first_window_start': range_start + int(starts.min()),
                    'last_window_start': range_start + int(starts.max()),
                    'image_count': len(starts),
                })
            print(
                f'[{condition}] {class_name}: '
                f'{"20 val + 20 test" if is_target else "40 source"} images'
            )

    manifest_path = output_root / 'manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_images = sum(int(row['image_count']) for row in manifest_rows)
    print(f'Done. Generated {total_images} CWRU STFT images.')
    print(f'Output root: {output_root.resolve()}')
    print(f'Manifest: {manifest_path.resolve()}')
    return total_images, manifest_path


def main():
    parser = argparse.ArgumentParser(description='CWRU raw MAT to STFT PNG preprocessing.')
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    preprocess_cwru(CWRU_CONFIG, clean=args.clean)


if __name__ == '__main__':
    main()
