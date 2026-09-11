"""Replay original preprocessing windows, verify against PNGs, then use raw 1D.

No image inversion, random new windows, image overwrites or test-data fitting.
"""
import copy
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pu_dataset import _split_target_class_data
from pu_preprocess import sample_starts, compute_stft_log_image


class RawSignalDataset(Dataset):
    def __init__(self, cfg):
        self.source_classes = list(cfg.get('class_groups', cfg.get('class_names', [])))
        self.source_data, self.target_val_data, self.target_test_data = {}, {}, {}
        self.verified_windows = 0
        self.cfg = cfg
        if 'class_groups' in cfg:
            self._load_pu()
        else:
            self._load_cwru()
        self.set_mode('train')
        print(f'Raw signal alignment verified: {self.verified_windows} PNG/window pairs')

    def _windows(self, signal, starts, directory, prefix):
        cfg = self.cfg
        paths = sorted(Path(directory).glob(prefix + '_*.png'))
        if len(paths) != len(starts):
            raise ValueError(f'Raw/PNG count mismatch: {directory}/{prefix}; check preprocessing config')
        windows = []
        for i, start in enumerate(starts):
            path = Path(directory) / f'{prefix}_{i:03d}.png'
            raw = np.asarray(signal[start:start + int(cfg['window_size'])], dtype=np.float64)
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or not np.array_equal(image, compute_stft_log_image(raw, cfg)):
                raise ValueError(f'Raw window does not match {path}. Check MAT source, preprocess_seed '
                                 'and STFT settings; no data were overwritten.')
            # Per-window z-score uses no statistics from other samples/splits.
            windows.append(((raw - raw.mean()) / (raw.std() + 1e-8)).astype(np.float32)[None])
            self.verified_windows += 1
        return np.stack(windows)

    def _load_pu(self):
        from pu_loader import PULoader
        cfg = self.cfg
        loader = PULoader(cfg['mat_root'])
        rng = np.random.default_rng(cfg.get('preprocess_seed', 24))
        source = cfg['source_condition']
        source = [source] if isinstance(source, str) else list(source)
        wanted = {b for group in cfg['class_groups'].values() for b in group}
        data = {}
        # Consume RNG for ALL files in the original sorted preprocessing order.
        for condition in sorted(loader.get_condition_names()):
            signals = loader.load_condition(condition)
            for bearing in sorted(signals):
                raw = np.asarray(signals[bearing]).ravel()
                starts = sample_starts(len(raw), cfg['window_size'], cfg['samples_per_class'], rng)
                if bearing in wanted and condition in source + [cfg['target_condition']]:
                    directory = Path(cfg['root_path']) / condition / bearing
                    data[condition, bearing] = self._windows(raw, starts, directory, 'stft')
        for index, (name, bearings) in enumerate(cfg['class_groups'].items()):
            self.source_data[name] = np.concatenate([data[c, b] for c in source for b in bearings])
            target = np.concatenate([data[cfg['target_condition'], b] for b in bearings])
            val, test = _split_target_class_data(target, cfg.get('target_val_ratio', .5),
                                                 cfg.get('seed', 24) + index)
            self.target_val_data[name], self.target_test_data[name] = val, test

    def _load_cwru(self):
        from cwru_preprocess import load_metadata_ids, load_drive_end_signal
        cfg = self.cfg
        metadata = load_metadata_ids(Path(cfg['raw_data_root']))
        rng = np.random.default_rng(cfg.get('preprocess_seed', 24))
        source_parts = {name: {} for name in self.source_classes}
        sources = cfg['source_condition']
        sources = [sources] if isinstance(sources, str) else list(sources)
        for condition, speed in cfg['condition_speeds'].items():
            if condition not in sources + [cfg['target_condition']]:
                continue
            for name, spec in cfg['class_specs'].items():
                signal, _, _ = load_drive_end_signal(cfg['raw_data_root'], spec['subset'], speed,
                                                      spec['filename'], metadata, cfg)
                mid = len(signal) // 2
                specs = [('val', signal[:mid], cfg['target_val_samples_per_class']),
                         ('test', signal[mid:], cfg['target_test_samples_per_class'])] if condition == cfg['target_condition'] else [
                         ('source', signal, cfg['source_samples_per_class'])]
                for prefix, raw, count in specs:
                    starts = sample_starts(len(raw), cfg['window_size'], count, rng)
                    if name not in self.source_classes:
                        continue
                    windows = self._windows(raw, starts, Path(cfg['root_path']) / condition / name, prefix)
                    if prefix == 'source':
                        source_parts[name][condition] = windows
                    else:
                        (self.target_val_data if prefix == 'val' else self.target_test_data)[name] = windows
        for name, parts in source_parts.items():
            self.source_data[name] = np.concatenate([parts[c] for c in sources])

    def for_mode(self, mode):
        result = copy.copy(self)
        result.set_mode(mode)
        return result

    def set_mode(self, mode):
        data = {'train': self.source_data, 'validation': self.target_val_data,
                'test': self.target_test_data}[mode]
        self._x = np.concatenate([data[c] for c in self.source_classes])
        self._y = np.concatenate([np.full(len(data[c]), i, dtype=np.int64)
                                  for i, c in enumerate(self.source_classes)])

    def __len__(self):
        return len(self._y)

    def __getitem__(self, index):
        return torch.from_numpy(self._x[index]), int(self._y[index])
