"""Read project raw windows and reshape 1444 consecutive samples to 38 x 38."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from raw_signal_dataset import RawSignalDataset

IMAGE_SIZE = 38
SAMPLE_POINTS = IMAGE_SIZE ** 2
REPRESENTATION = 'raw_reshape_2d'


def reshape_samples(raw: np.ndarray) -> np.ndarray:
    """Preserve temporal order and amplitudes; only cast to float32 and reshape."""
    values = np.asarray(raw)
    if values.ndim != 1 or len(values) != SAMPLE_POINTS or not np.isfinite(values).all():
        raise ValueError('AMAML requires exactly 1444 finite consecutive raw samples')
    return values.astype(np.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE)


class RawReshapeDataset(RawSignalDataset):
    """Reuse MAT parsing, RNG order, condition roles and PU target split unchanged.

    cfg.window_size stays at the project's original sampling footprint (4096 for
    PU / 3072 for CWRU). Taking its first 1444 points preserves window identities
    and cannot cross the original CWRU target half boundaries. No PNG is read,
    no STFT is computed, and no normalization or interpolation is applied.
    """
    def __init__(self, cfg: dict) -> None:
        if cfg.get('input_representation') != REPRESENTATION or cfg.get('img_size') != IMAGE_SIZE:
            raise ValueError('Expected raw_reshape_2d input with img_size=38')
        if int(cfg['window_size']) < SAMPLE_POINTS:
            raise ValueError('Project sampling window is shorter than 1444 points')
        self.cfg = cfg
        self.source_classes = list(cfg.get('class_groups', cfg.get('class_names', [])))
        self.source_data, self.target_val_data, self.target_test_data = {}, {}, {}
        self.window_manifest: list[dict] = []
        self.verified_windows = 0
        self._load_pu() if cfg['dataset'] == 'pu' else self._load_cwru()
        self.set_mode('train')
        print(f'Raw reshape: {self.verified_windows} windows, 1444 -> [1,38,38]; no STFT/PNG')

    def _windows(self, signal: np.ndarray, starts: np.ndarray, directory: Path,
                 prefix: str) -> np.ndarray:
        windows = []
        for index, start in enumerate(starts):
            start = int(start)
            raw = np.asarray(signal[start:start + SAMPLE_POINTS], dtype=np.float64)
            windows.append(reshape_samples(raw))
            self.window_manifest.append(dict(
                condition=Path(directory).parent.name,
                class_or_bearing=Path(directory).name,
                role='pool' if prefix == 'stft' else prefix,
                index=index, start_in_segment=start, length=SAMPLE_POINTS,
                sampling_window_size=int(self.cfg['window_size']),
                segment='second_half' if prefix == 'test' else 'first_half' if prefix == 'val' else 'whole',
                raw_sha256=hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest(),
            ))
            self.verified_windows += 1
        return np.stack(windows)

    def to_maml_dataset_format(self, mode: str = 'train') -> tuple[np.ndarray, np.ndarray]:
        view = self.for_mode(mode)
        return view._x, view._y
