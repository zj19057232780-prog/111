from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils import data


def read_directory(directory_name, height, width):
    """Load grayscale images as [N, 1, H, W] float32 tensors in numpy form."""
    file_list = sorted(
        f
        for f in os.listdir(directory_name)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    )
    imgs = []
    for each_file in file_list:
        path = os.path.join(directory_name, each_file)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if gray.shape != (height, width):
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
        imgs.append(gray.astype(np.float32))

    if not imgs:
        return np.zeros((0, 1, height, width), dtype=np.float32)
    arr = np.asarray(imgs, dtype=np.float32) / 255.0
    return arr.reshape(-1, 1, height, width)


def _normalize_source_condition(source_condition):
    if isinstance(source_condition, str):
        return [source_condition]
    if isinstance(source_condition, (list, tuple)):
        return list(source_condition)
    raise TypeError(f'source_condition must be str/list, got {type(source_condition)}')


def _split_target_class_data(data_arr, val_ratio=0.5, seed=24):
    n = len(data_arr)
    if n < 2:
        return data_arr, data_arr
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_count = int(round(n * val_ratio))
    val_count = min(max(val_count, 1), n - 1)
    val_idx = np.sort(indices[:val_count])
    test_idx = np.sort(indices[val_count:])
    return data_arr[val_idx], data_arr[test_idx]


def _normalize_class_groups(class_groups):
    if class_groups is None:
        return None
    if not isinstance(class_groups, dict) or not class_groups:
        raise ValueError('class_groups must be a non-empty dict or None')

    normalized = {}
    used_bearings = set()
    for class_name, bearing_ids in class_groups.items():
        members = list(bearing_ids)
        if not members:
            raise ValueError(f'Classification group is empty: {class_name}')
        duplicates = used_bearings.intersection(members)
        if duplicates:
            raise ValueError(
                f'Bearing IDs assigned to multiple groups: {sorted(duplicates)}'
            )
        used_bearings.update(members)
        normalized[str(class_name)] = members
    return normalized


class PUMetaDataset(data.Dataset):
    """
    In-memory PU STFT image dataset for MAML.

    train: source conditions
    validation: held-out split of the target condition
    test: separate held-out split of the target condition
    """

    def __init__(
        self,
        root_path,
        source_condition,
        target_condition,
        img_size=64,
        target_val_ratio=0.5,
        split_seed=24,
        class_groups=None,
        classification_name=None,
        share_storage=None,
    ):
        super().__init__()
        self.root_path = root_path
        self.source_conditions = _normalize_source_condition(source_condition)
        self.target_condition = target_condition
        self.height = img_size
        self.width = img_size
        self.target_val_ratio = target_val_ratio
        self.split_seed = split_seed
        self.class_groups = _normalize_class_groups(class_groups)
        self.classification_name = classification_name or 'bearing_id'
        self._share = share_storage

        self.source_classes = []
        self.target_classes = []
        self.source_data = {}
        self.target_val_data = {}
        self.target_test_data = {}
        self.num_classes = 0
        self._x = np.zeros((0, 1, img_size, img_size), dtype=np.float32)
        self._y = np.zeros((0,), dtype=np.int64)

        if self._share is not None:
            self.source_conditions = list(self._share.source_conditions)
            self.source_classes = list(self._share.source_classes)
            self.target_classes = list(self._share.target_classes)
            self.source_data = self._share.source_data
            self.target_val_data = self._share.target_val_data
            self.target_test_data = self._share.target_test_data
            self.num_classes = self._share.num_classes
            self.class_groups = self._share.class_groups
            self.classification_name = self._share.classification_name
        else:
            self.prepare_data()

    @staticmethod
    def _list_class_dirs(base):
        if not os.path.isdir(base):
            return []
        return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))

    def prepare_data(self):
        if self._share is not None:
            return

        target_dir = os.path.join(self.root_path, self.target_condition)
        target_bearings = self._list_class_dirs(target_dir)
        if not target_bearings:
            raise FileNotFoundError(f'No target class directories found: {target_dir}')

        if self.class_groups is None:
            self.class_groups = {
                bearing_id: [bearing_id] for bearing_id in target_bearings
            }
        required_bearings = {
            bearing_id
            for bearing_ids in self.class_groups.values()
            for bearing_id in bearing_ids
        }
        missing_target = sorted(required_bearings.difference(target_bearings))
        if missing_target:
            raise ValueError(
                f'Target condition {self.target_condition} is missing bearings: '
                f'{missing_target}'
            )

        self.target_classes = list(self.class_groups)
        self.num_classes = len(self.target_classes)

        for cond in self.source_conditions:
            src_dir = os.path.join(self.root_path, cond)
            if not os.path.isdir(src_dir):
                raise FileNotFoundError(f'Source condition directory not found: {src_dir}')
            src_bearings = self._list_class_dirs(src_dir)
            missing_source = sorted(required_bearings.difference(src_bearings))
            if missing_source:
                raise ValueError(
                    f'Source condition {cond} is missing bearings: {missing_source}'
                )

        self.source_classes = list(self.target_classes)
        self.source_data = {}
        self.target_val_data = {}
        self.target_test_data = {}

        for class_seed, cls_name in enumerate(self.target_classes):
            bearing_ids = self.class_groups[cls_name]
            merged_source = []
            for cond in self.source_conditions:
                for bearing_id in bearing_ids:
                    src_cls_dir = os.path.join(self.root_path, cond, bearing_id)
                    cls_data = read_directory(src_cls_dir, self.height, self.width)
                    if len(cls_data) == 0:
                        raise ValueError(
                            f'No images under source bearing directory: {src_cls_dir}'
                        )
                    merged_source.append(cls_data)
            self.source_data[cls_name] = np.concatenate(merged_source, axis=0)

            target_group = []
            for bearing_id in bearing_ids:
                tgt_cls_dir = os.path.join(target_dir, bearing_id)
                bearing_data = read_directory(tgt_cls_dir, self.height, self.width)
                if len(bearing_data) == 0:
                    raise ValueError(
                        f'No images under target bearing directory: {tgt_cls_dir}'
                    )
                target_group.append(bearing_data)
            tgt_data = np.concatenate(target_group, axis=0)
            val_data, test_data = _split_target_class_data(
                tgt_data,
                val_ratio=self.target_val_ratio,
                seed=self.split_seed + class_seed,
            )
            self.target_val_data[cls_name] = val_data
            self.target_test_data[cls_name] = test_data

        print(
            f'Source conditions {self.source_conditions} -> target {self.target_condition}; '
            f'classification={self.classification_name}; classes={self.num_classes}'
        )

    def to_maml_dataset_format(self, mode='train'):
        if mode == 'train':
            data_source = self.source_data
        elif mode == 'validation':
            data_source = self.target_val_data
        elif mode == 'test':
            data_source = self.target_test_data
        else:
            raise ValueError(f'Unknown mode: {mode}')

        all_x = []
        all_y = []
        for cls_idx, cls_name in enumerate(self.source_classes):
            cls_data = data_source[cls_name]
            all_x.append(cls_data)
            all_y.append(np.full(len(cls_data), cls_idx, dtype=np.int64))

        x = np.concatenate(all_x, axis=0).astype(np.float32)
        y = np.concatenate(all_y, axis=0).astype(np.int64)
        return x, y

    def set_mode(self, mode):
        self._x, self._y = self.to_maml_dataset_format(mode)

    def __getitem__(self, item):
        x = torch.from_numpy(np.ascontiguousarray(self._x[item]))
        y = int(self._y[item])
        return x, torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return int(self._x.shape[0])
