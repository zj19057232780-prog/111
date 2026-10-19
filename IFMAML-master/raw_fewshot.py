"""1D adaptations of ProtoNet/MatchingNet/RelationNet/MAML for vibration.

Algorithm/loss/outer-loop reuse the project implementations. These are 1D
adaptations, not verbatim reproductions of the original image benchmarks.
No STFT or PNG loading is performed by DirectRawDataset.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn

from classic_config import resolve_method_config
from classic_models import ClassicMetricModel
from maml_train_pic import MAML_learner, device
from raw_signal_dataset import RawSignalDataset


class Conv4Raw(nn.Sequential):
    def __init__(self, channels=64, bins=4):
        blocks = []
        for i in range(4):
            conv = nn.Conv1d(1 if i == 0 else channels, channels, 3, padding=1)
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
            norm = nn.BatchNorm1d(channels)
            nn.init.uniform_(norm.weight)
            nn.init.zeros_(norm.bias)
            blocks.append(nn.Sequential(conv, norm, nn.ReLU(), nn.MaxPool1d(2)))
        blocks.append(nn.AdaptiveAvgPool1d(bins))
        super().__init__(*blocks)


class RawMAMLModel(nn.Module):
    def __init__(self, ways, channels=64, bins=4):
        super().__init__()
        self.encoder = Conv4Raw(channels, bins)
        self.classifier = nn.Linear(channels * bins, ways)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        features = self.encoder(x).flatten(1)
        return features, self.classifier(features)


class RawMetricModel(ClassicMetricModel):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.method = cfg['method']
        channels = int(cfg.get('hidden_size', 64))
        bins = int(cfg.get('raw_feature_bins', 4))
        self.encoder = Conv4Raw(channels, bins)
        dimension = channels * bins
        self.temperature = float(cfg.get('metric_temperature', 1.0))
        if self.temperature <= 0:
            raise ValueError('metric_temperature must be positive')
        self.fce = bool(cfg.get('matching_fce', True))
        self.fce_steps = int(cfg.get('matching_fce_steps', 5))
        if self.method == 'matchingnet' and self.fce:
            if self.fce_steps < 1:
                raise ValueError('matching_fce_steps must be positive')
            self.support_lstm = nn.LSTM(dimension, dimension, batch_first=True, bidirectional=True)
            self.query_lstm = nn.LSTMCell(2 * dimension, dimension)
        if self.method == 'relationnet':
            self.relation = nn.Sequential(
                nn.Conv1d(2 * channels, channels, 3, padding=1),
                nn.BatchNorm1d(channels), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.BatchNorm1d(channels), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                nn.Flatten(), nn.Linear(channels, int(cfg.get('relation_hidden_size', 8))),
                nn.ReLU(), nn.Linear(int(cfg.get('relation_hidden_size', 8)), 1), nn.Sigmoid())

    def episode(self, support, labels, query):
        if self.method != 'relationnet':
            return super().episode(support, labels, query)
        classes = torch.unique(labels, sorted=True)
        if not torch.equal(classes, torch.arange(len(classes), device=labels.device)):
            raise ValueError('Support labels must be contiguous from zero')
        sf, qf = self.encoder(support), self.encoder(query)
        prototypes = torch.stack([sf[labels == c].sum(0) for c in classes])
        pairs = torch.cat([prototypes.unsqueeze(0).expand(len(qf), -1, -1, -1),
                           qf.unsqueeze(1).expand(-1, len(classes), -1, -1)], dim=2)
        scores = self.relation(pairs.flatten(0, 1)).reshape(len(qf), len(classes))
        return qf.flatten(1), scores


class DirectRawDataset(RawSignalDataset):
    """Reuse MAT parsing, sample positions and splits; replace PNG verification.

    CWRU: source windows from the whole trace; target val/test from separate halves.
    PU: replay preprocessing RNG across all sorted MAT files; use existing seed split.
    Only segmentation and per-window z-score remain; no frequency transformation.
    """
    def __init__(self, cfg):
        self.source_classes = list(cfg.get('class_groups', cfg.get('class_names', [])))
        self.source_data, self.target_val_data, self.target_test_data = {}, {}, {}
        self.cfg = cfg
        self.window_manifest = []
        self.verified_windows = 0
        self._load_pu() if 'class_groups' in cfg else self._load_cwru()
        self.set_mode('train')
        target = cfg['k_shot'] + cfg['q_query']
        for mode, split in [('train', self.source_data), ('validation', self.target_val_data),
                            ('test', self.target_test_data)]:
            for name in self.source_classes:
                if len(split[name]) < target:
                    raise ValueError(f'{mode}/{name} requires {target} samples, got {len(split[name])}')
        path = cfg.get('raw_window_manifest')
        if path:
            Path(path).write_text(json.dumps(self.window_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Direct raw 1D: {self.verified_windows} vibration windows; no STFT/PNG processing')

    def _windows(self, signal, starts, directory, prefix):
        windows = []
        size = int(self.cfg['window_size'])
        for i, start in enumerate(starts):
            raw = np.asarray(signal[int(start):int(start) + size], dtype=np.float64)
            if len(raw) != size or not np.isfinite(raw).all():
                raise ValueError(f'Invalid raw vibration window: {directory}/{prefix}/{i}')
            windows.append(((raw - raw.mean()) / (raw.std() + 1e-8)).astype(np.float32)[None])
            self.window_manifest.append(dict(
                condition=Path(directory).parent.name, class_or_bearing=Path(directory).name,
                role=prefix, index=i, start_in_segment=int(start), length=size,
                segment='second_half' if prefix == 'test' else 'first_half' if prefix == 'val' else 'whole',
                raw_sha256=hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest()))
            self.verified_windows += 1
        return np.stack(windows)


class RawFewShotLearner(MAML_learner):
    def __init__(self, ways, pu_config):
        cfg = resolve_method_config(pu_config)
        if cfg['method'] not in ('maml', 'protonet', 'matchingnet', 'relationnet'):
            raise ValueError('RawFewShotLearner supports only four classical algorithms')
        cfg.update(input_representation='raw_1d', backbone='cnn4_1d',
                   raw_normalization='per_window_zscore', raw_feature_bins=4)
        self.pu_config, self.method, self.ways = cfg, cfg['method'], ways
        self._pu_storage = None
        self.backbone_name = 'cnn4_1d'
        self.frequency_module = self.denoise_module = self.attention_module = 'none'
        self.model = (RawMAMLModel(ways, int(cfg.get('hidden_size', 64)), 4)
                      if self.method == 'maml' else RawMetricModel(cfg)).to(device)

    def _get_raw_storage(self):
        if self._pu_storage is None:
            self._pu_storage = DirectRawDataset(self.pu_config)
        return self._pu_storage

    def _get_pu_storage(self):
        return self._get_raw_storage()

    def build_tasks(self, mode='train', ways=5, shots=1, queries=1,
                    num_tasks=100, filter_labels=None):
        return self._raw_tasks(mode, ways, shots, queries, num_tasks, filter_labels)
