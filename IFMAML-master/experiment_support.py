"""Data verification and evaluation adapters; no replacement learning algorithms."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np

from raw_transfer import RawTransferLearner


class STFTTransferLearner(RawTransferLearner):
    """Existing WDCNN consumes flattened STFT64, matching its archived cross adapter."""
    def __init__(self, ways, cfg, dataset):
        super().__init__(ways, cfg)
        self.dataset = dataset
        self.pu_config['input_representation'] = 'stft_flattened_4096'

    def _get_raw_storage(self):
        if self._pu_storage is None:
            cfg = self.pu_config
            if self.dataset == 'cwru':
                from cwru_dataset import CWRUMetaDataset
                data = CWRUMetaDataset(cfg['root_path'], cfg['source_condition'],
                                      cfg['target_condition'], cfg['class_names'], cfg['img_size'])
            else:
                from pu_dataset import PUMetaDataset
                data = PUMetaDataset(cfg['root_path'], cfg['source_condition'], cfg['target_condition'],
                    img_size=cfg['img_size'], target_val_ratio=cfg['target_val_ratio'], split_seed=cfg['seed'],
                    class_groups=cfg['class_groups'], classification_name=cfg['classification_preset'])
            from raw_signal_dataset import RawSignalDataset
            flattened = RawSignalDataset.__new__(RawSignalDataset)
            flattened.source_classes = list(data.source_classes)
            for key in ('source_data', 'target_val_data', 'target_test_data'):
                values = getattr(data, key)
                setattr(flattened, key, {name: array.reshape(len(array), 1, -1) for name, array in values.items()})
            flattened.set_mode('train')
            self._pu_storage = flattened
        return self._pu_storage


def data_signature(storage, cfg: dict) -> dict:
    result = {}
    for split, key in [('train', 'source_data'), ('validation', 'target_val_data'), ('test', 'target_test_data')]:
        arrays = getattr(storage, key)
        result[split] = {}
        for name in storage.source_classes:
            array = np.ascontiguousarray(arrays[name])
            if len(array) < cfg['k_shot'] + cfg['q_query'] or not np.isfinite(array).all():
                raise ValueError(f'{split}/{name}: insufficient or non-finite samples')
            result[split][name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                                      sha256=hashlib.sha256(array.tobytes()).hexdigest())
    return result


def evaluate_classic(learner, best: Path, cfg: dict, out: Path) -> dict:
    import maml_train_pic as training
    from my_utils.init_utils import seed_torch
    from sklearn.metrics import f1_score
    captured = {}
    original = training._save_test_visualizations

    def capture(*args, **kwargs):
        payload = inspect.signature(original).bind(*args, **kwargs).arguments
        rows = payload['rows']
        expected = cfg['test_meta_batch_size'] * cfg['n_way'] * cfg['q_query']
        if len(rows) != expected:
            raise ValueError(f'Incomplete predictions: {len(rows)} != {expected}')
        counts = {}
        for row in rows:
            counts[row['episode']] = counts.get(row['episode'], 0) + 1
        if len(counts) != cfg['test_meta_batch_size'] or set(counts.values()) != {cfg['n_way'] * cfg['q_query']}:
            raise ValueError('Invalid episode structure')
        with (out/'test_predictions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        y_true, y_pred = payload['y_true'], payload['y_pred']
        captured.update(accuracy_pct=float(100*np.mean(y_true == y_pred)),
            macro_f1_pct=float(100*f1_score(y_true, y_pred, average='macro', zero_division=0)),
            test_episodes=len(counts), query_count=len(rows))

    training._save_test_visualizations = capture
    try:
        seed_torch(100000 + cfg['seed'])
        learner.test(str(best), shots=cfg['k_shot'], inner_steps=cfg['test_inner_steps'],
                     meta_batch_size=cfg['test_meta_batch_size'])
    finally:
        training._save_test_visualizations = original
    if not captured:
        raise RuntimeError('Test did not produce predictions')
    print(f"[TEST] Accuracy={captured['accuracy_pct']:.4f}% Macro-F1={captured['macro_f1_pct']:.4f}%", flush=True)
    return captured
