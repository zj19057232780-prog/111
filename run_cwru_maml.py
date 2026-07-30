"""Independent CWRU preprocessing, training, and testing entry point."""

from __future__ import annotations

import argparse
import os

from config_cwru import CWRU_CONFIG, QUICK_TEST
from cwru_dataset import CWRUMetaDataset
from cwru_preprocess import preprocess_cwru
from l2l_shim import MetaDataset, TaskDataset
from maml_train_pic import MAML_learner, device
from my_utils.init_utils import seed_torch


class CWRUMAMLLearner(MAML_learner):
    """Reuse the model/MAML implementation while replacing only the dataset."""

    def __init__(self, ways, cwru_config=None):
        super().__init__(ways=ways, pu_config=cwru_config or CWRU_CONFIG)
        self._cwru_storage = None

    def _get_pu_storage(self):
        if self._cwru_storage is None:
            cfg = self.pu_config
            self._cwru_storage = CWRUMetaDataset(
                root_path=cfg['root_path'],
                source_condition=cfg['source_condition'],
                target_condition=cfg['target_condition'],
                class_names=cfg['class_names'],
                img_size=cfg.get('img_size', 64),
            )
        return self._cwru_storage

    def build_tasks(
        self,
        mode='train',
        ways=5,
        shots=1,
        queries=1,
        num_tasks=100,
        filter_labels=None,
    ):
        cfg = self.pu_config
        storage = self._get_pu_storage()
        dataset = CWRUMetaDataset(
            root_path=cfg['root_path'],
            source_condition=cfg['source_condition'],
            target_condition=cfg['target_condition'],
            class_names=cfg['class_names'],
            img_size=cfg.get('img_size', 64),
            share_storage=storage,
        )
        dataset.set_mode(mode)
        meta_dataset = MetaDataset(dataset)
        task_ways = len(filter_labels) if filter_labels is not None else ways
        return TaskDataset(
            meta_dataset,
            n_way=task_ways,
            k_shot=shots,
            q_query=queries,
            num_tasks=num_tasks,
            filter_labels=filter_labels,
        )


def main():
    cfg = CWRU_CONFIG
    parser = argparse.ArgumentParser(
        description='CWRU 10-state cross-condition STFT-MAML experiment.'
    )
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument(
        '--model_path',
        default=os.path.join('.', 'model_save', cfg['model_name']),
    )
    args = parser.parse_args()

    if args.clean and not args.preprocess:
        parser.error('--clean must be used together with --preprocess')

    seed_torch(cfg.get('seed', 24))
    print(f'Device: {device}')
    print(
        f'CWRU protocol: {cfg["source_condition"]} -> {cfg["target_condition"]}'
    )
    print(
        f'Classification: {cfg["classification_preset"]} | '
        f'pool={len(cfg["class_names"])} classes'
    )
    print(
        f'Episode: {cfg["n_way"]}-way {cfg["k_shot"]}-shot '
        f'{cfg["q_query"]}-query'
    )
    print(f'Backbone: {cfg.get("backbone", "cnn4")}')
    print(f'Frequency module: {cfg.get("frequency_module", "none")}')
    print(f'Denoise module: {cfg.get("denoise_module", "none")}')
    print(f'Attention module: {cfg.get("attention_module", "none")}')

    if args.preprocess:
        preprocess_cwru(cfg, clean=args.clean)

    quick = args.quick or QUICK_TEST
    if args.train or args.test:
        learner = CWRUMAMLLearner(
            ways=cfg['n_way'],
            cwru_config=cfg,
        )
        if args.train:
            best_path = learner.train(
                args.model_path,
                shots=cfg['k_shot'],
                quick_test=quick,
            )
        else:
            best_path = args.model_path + '_best'

        if args.test:
            if not os.path.exists(best_path):
                raise FileNotFoundError(f'CWRU model not found: {best_path}')
            learner.test(
                best_path,
                inner_steps=cfg.get('test_inner_steps', 10),
                shots=cfg['k_shot'],
                meta_batch_size=4 if quick else None,
            )

    if not args.preprocess and not args.train and not args.test:
        print('Nothing to do. Use --preprocess, --train, and/or --test.')


if __name__ == '__main__':
    main()
