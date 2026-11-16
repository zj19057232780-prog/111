"""Disjoint target-validation adapter reusing PUMetaDataset and the original trainer."""
from pathlib import Path
import numpy as np
from pu_dataset import PUMetaDataset, read_directory
from maml_train_pic import MAML_learner
from config_hust_huazhong import CLASSES
from hust_huazhong_preprocess import verify_cache, split_indices


class HUSTMetaDataset(PUMetaDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.manifest = verify_cache(cfg['root_path'], cfg)
        self.target_loaded = False
        super().__init__(cfg['root_path'], cfg['source_condition'], cfg['target_condition'],
            img_size=64, class_groups={c:[c] for c in CLASSES}, classification_name='huazhong_9state')

    def prepare_data(self):
        self.source_classes = list(CLASSES)
        self.target_classes = list(CLASSES)
        self.num_classes = len(CLASSES)
        if self.target_condition in self.source_conditions:
            raise ValueError('Source/target condition leakage')
        valid, test = split_indices(64)
        for cls in CLASSES:
            arrays = [read_directory(str(Path(self.root_path)/c/cls),64,64) for c in self.source_conditions]
            if any(len(a)!=64 for a in arrays):
                raise ValueError('Unreadable source images')
            self.source_data[cls] = np.concatenate(arrays)
            target = read_directory(str(Path(self.root_path)/self.target_condition/cls),64,64)
            if len(target)!=64:
                raise ValueError('Unreadable target images')
            self.target_val_data[cls] = target[valid]
            self.target_test_data[cls] = target[test]
        self.target_loaded = True



class HUSTMAMLLearner(MAML_learner):
    def _get_pu_storage(self):
        if self._pu_storage is None:
            self._pu_storage=HUSTMetaDataset(self.pu_config)
        return self._pu_storage

    def build_tasks(self, mode='train', ways=5, shots=1, queries=1, num_tasks=100, filter_labels=None):
        return super().build_tasks(mode,ways,shots,queries,num_tasks,filter_labels)
