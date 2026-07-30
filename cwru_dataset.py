"""CWRU STFT image dataset with fixed target validation/test partitions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data


def _read_images(directory, img_size, prefix=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'CWRU image directory not found: {directory}')

    pattern = f'{prefix}_*.png' if prefix else '*.png'
    paths = sorted(directory.glob(pattern))
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Failed to read CWRU image: {path}')
        if image.shape != (img_size, img_size):
            image = cv2.resize(
                image,
                (img_size, img_size),
                interpolation=cv2.INTER_LINEAR,
            )
        images.append(image.astype(np.float32) / 255.0)

    if not images:
        raise ValueError(
            f'No CWRU images matching {pattern} under directory: {directory}'
        )
    return np.asarray(images, dtype=np.float32).reshape(-1, 1, img_size, img_size)


class CWRUMetaDataset(data.Dataset):
    """In-memory 10-state CWRU dataset for the existing MAML task sampler."""

    def __init__(
        self,
        root_path,
        source_condition,
        target_condition,
        class_names,
        img_size=64,
        share_storage=None,
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.source_conditions = (
            [source_condition] if isinstance(source_condition, str)
            else list(source_condition)
        )
        self.target_condition = target_condition
        self.source_classes = list(class_names)
        self.target_classes = list(class_names)
        self.num_classes = len(self.source_classes)
        self.height = int(img_size)
        self.width = int(img_size)
        self.source_data = {}
        self.target_val_data = {}
        self.target_test_data = {}
        self._x = np.zeros((0, 1, self.height, self.width), dtype=np.float32)
        self._y = np.zeros((0,), dtype=np.int64)

        if share_storage is None:
            self.prepare_data()
        else:
            self.source_conditions = list(share_storage.source_conditions)
            self.target_condition = share_storage.target_condition
            self.source_classes = list(share_storage.source_classes)
            self.target_classes = list(share_storage.target_classes)
            self.num_classes = share_storage.num_classes
            self.source_data = share_storage.source_data
            self.target_val_data = share_storage.target_val_data
            self.target_test_data = share_storage.target_test_data

    def prepare_data(self):
        for class_name in self.source_classes:
            source_parts = []
            for condition in self.source_conditions:
                class_dir = self.root_path / condition / class_name
                source_parts.append(
                    _read_images(class_dir, self.height, prefix='source')
                )
            self.source_data[class_name] = np.concatenate(source_parts, axis=0)

            target_dir = self.root_path / self.target_condition / class_name
            self.target_val_data[class_name] = _read_images(
                target_dir,
                self.height,
                prefix='val',
            )
            self.target_test_data[class_name] = _read_images(
                target_dir,
                self.height,
                prefix='test',
            )

        print(
            f'CWRU source conditions {self.source_conditions} -> '
            f'target {self.target_condition}; classes={self.num_classes}'
        )

    def to_maml_dataset_format(self, mode='train'):
        if mode == 'train':
            data_source = self.source_data
        elif mode == 'validation':
            data_source = self.target_val_data
        elif mode == 'test':
            data_source = self.target_test_data
        else:
            raise ValueError(f'Unknown CWRU dataset mode: {mode}')

        all_x = []
        all_y = []
        for class_index, class_name in enumerate(self.source_classes):
            class_data = data_source[class_name]
            all_x.append(class_data)
            all_y.append(
                np.full(len(class_data), class_index, dtype=np.int64)
            )
        return (
            np.concatenate(all_x, axis=0).astype(np.float32),
            np.concatenate(all_y, axis=0).astype(np.int64),
        )

    def set_mode(self, mode):
        self._x, self._y = self.to_maml_dataset_format(mode)

    def __getitem__(self, item):
        image = torch.from_numpy(np.ascontiguousarray(self._x[item]))
        label = torch.tensor(int(self._y[item]), dtype=torch.long)
        return image, label

    def __len__(self):
        return int(self._x.shape[0])
