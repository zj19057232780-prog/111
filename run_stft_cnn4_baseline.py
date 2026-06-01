"""
One-command helper for the STFT-CNN4-MAML baseline.

Examples:
    python run_stft_cnn4_baseline.py --prepare --preprocess
    python run_stft_cnn4_baseline.py --train --test --quick
"""

from __future__ import annotations

import argparse
import os

from config_pu import PU_CONFIG
from prepare_pu_baseline_data import build_pu_mat_layout
from pu_preprocess import process_mat_root_to_png


def main() -> None:
    parser = argparse.ArgumentParser(description='Run STFT-CNN4-MAML baseline steps.')
    parser.add_argument('--prepare', action='store_true', help='Build pu_data_mat layout.')
    parser.add_argument('--preprocess', action='store_true', help='Generate STFT PNG data.')
    parser.add_argument('--clean_png', action='store_true', help='Delete old pu_data_processed first.')
    parser.add_argument('--train', action='store_true', help='Train CNN4-MAML.')
    parser.add_argument('--test', action='store_true', help='Test the best checkpoint.')
    parser.add_argument('--quick', action='store_true', help='Use a short smoke-test run.')
    args = parser.parse_args()

    cfg = PU_CONFIG
    conditions = sorted(set(list(cfg['source_condition']) + [cfg['target_condition']]))

    if args.prepare:
        copied, skipped = build_pu_mat_layout(
            raw_root=cfg['raw_data_root'],
            mat_root=cfg['mat_root'],
            conditions=conditions,
            trial_pick=cfg.get('trial_pick', 'min'),
            overwrite=cfg.get('overwrite_mat_layout', True),
        )
        print(f'Prepared mat layout: copied={copied}, skipped={skipped}')

    if args.preprocess:
        process_mat_root_to_png(
            mat_root=cfg['mat_root'],
            png_root=cfg['root_path'],
            window_size=cfg['window_size'],
            img_size=cfg['img_size'],
            samples_per_class=cfg['samples_per_class'],
            cfg=cfg,
            seed=cfg.get('preprocess_seed', 24),
            clean=args.clean_png,
        )

    if args.train or args.test:
        from maml_train_pic import MAML_learner
        from my_utils.init_utils import seed_torch

        seed_torch(cfg.get('seed', 24))
        net = MAML_learner(ways=cfg['n_way'], pu_config=cfg)
        save_path = os.path.join('.', 'model_save', 'STFT_CNN4_MAML')
        best_path = save_path + '_best'

        if args.train:
            best_path = net.train(save_path, shots=cfg['k_shot'], quick_test=args.quick)
        if args.test:
            net.test(
                best_path,
                inner_steps=cfg.get('test_inner_steps', 10),
                shots=cfg['k_shot'],
                meta_batch_size=4 if args.quick else None,
            )

    if not any([args.prepare, args.preprocess, args.train, args.test]):
        parser.print_help()


if __name__ == '__main__':
    main()
