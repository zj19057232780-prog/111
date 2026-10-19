"""
Prepare the normalized PU .mat layout for the STFT-CNN4-MAML baseline.

Input layout:
    raw_root/
      K001/K001/N09_M07_F10_K001_1.mat
      K001/K001/N15_M01_F10_K001_1.mat
      ...

Output layout:
    pu_data_mat/
      N09_M07_F10/K001.mat
      N15_M01_F10/K001.mat
      ...
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from config_pu import PU_CONFIG


MAT_NAME_RE = re.compile(
    r'^(?P<condition>N\d+_M\d+_F\d+)_(?P<fault>K[A-Z]?\d+)_(?P<trial>\d+)\.mat$',
    re.IGNORECASE,
)


def _condition_list(value):
    if isinstance(value, str):
        return [value]
    return list(value)


def discover_raw_mats(raw_root: Path):
    """Return mapping: fault -> condition -> trial -> path."""
    mapping: dict[str, dict[str, dict[int, Path]]] = {}
    for fault_dir in sorted(raw_root.iterdir()):
        if not fault_dir.is_dir():
            continue
        inner = fault_dir / fault_dir.name
        search_dir = inner if inner.is_dir() else fault_dir
        for mat_path in sorted(search_dir.glob('*.mat')):
            match = MAT_NAME_RE.match(mat_path.name)
            if not match:
                continue
            condition = match.group('condition')
            fault = match.group('fault')
            trial = int(match.group('trial'))
            mapping.setdefault(fault, {}).setdefault(condition, {})[trial] = mat_path
    return mapping


def pick_trial(trials: dict[int, Path], trial_pick: str) -> Path | None:
    if not trials:
        return None
    if trial_pick == 'min':
        return trials[min(trials)]
    if trial_pick == 'max':
        return trials[max(trials)]
    return trials.get(int(trial_pick))


def build_pu_mat_layout(
    raw_root: str | os.PathLike,
    mat_root: str | os.PathLike,
    conditions: list[str],
    trial_pick: str = 'min',
    overwrite: bool = True,
) -> tuple[int, int]:
    raw_root = Path(raw_root)
    mat_root = Path(mat_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f'Raw PU root not found: {raw_root}')

    mapping = discover_raw_mats(raw_root)
    if not mapping:
        raise RuntimeError(f'No PU .mat files found under: {raw_root}')

    copied = 0
    skipped = 0
    mat_root.mkdir(parents=True, exist_ok=True)

    for fault in sorted(mapping):
        for condition in conditions:
            src = pick_trial(mapping[fault].get(condition, {}), trial_pick)
            if src is None:
                print(f'[skip] missing {condition}/{fault}')
                skipped += 1
                continue
            out_dir = mat_root / condition
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / f'{fault}.mat'
            if dst.exists() and not overwrite:
                skipped += 1
                continue
            shutil.copy2(src, dst)
            copied += 1
    return copied, skipped


def main() -> None:
    cfg = PU_CONFIG
    default_conditions = sorted(
        set(_condition_list(cfg['source_condition']) + [cfg['target_condition']])
    )

    parser = argparse.ArgumentParser(description='Prepare PU mat layout for baseline.')
    parser.add_argument('--raw_root', default=cfg['raw_data_root'])
    parser.add_argument('--mat_root', default=cfg['mat_root'])
    parser.add_argument('--conditions', nargs='*', default=default_conditions)
    parser.add_argument('--trial_pick', default=cfg.get('trial_pick', 'min'))
    parser.add_argument('--no_overwrite', action='store_true')
    args = parser.parse_args()

    copied, skipped = build_pu_mat_layout(
        raw_root=args.raw_root,
        mat_root=args.mat_root,
        conditions=args.conditions,
        trial_pick=args.trial_pick,
        overwrite=not args.no_overwrite,
    )
    print(f'Done. Copied {copied} files, skipped {skipped}.')
    print(f'MAT root: {Path(args.mat_root).resolve()}')


if __name__ == '__main__':
    main()
