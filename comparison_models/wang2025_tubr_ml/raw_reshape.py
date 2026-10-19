"""Read project raw windows and reshape 7056 consecutive samples to normalized 84 x 84."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from raw_signal_dataset import RawSignalDataset

IMAGE_SIZE = 84
SAMPLE_POINTS = IMAGE_SIZE ** 2
REPRESENTATION = 'raw7056_minmax_reshape84'


def reshape_samples(raw: np.ndarray) -> np.ndarray:
    """Normalize each window independently to [0,1], reshape and repeat grayscale 3x."""
    values = np.asarray(raw)
    if values.ndim != 1 or len(values) != SAMPLE_POINTS or not np.isfinite(values).all():
        raise ValueError('Paper input requires exactly 7056 finite consecutive raw samples')
    values = values.astype(np.float64)
    span = values.max() - values.min()
    normalized = (values - values.min()) / span if span > 0 else np.zeros_like(values)
    return np.repeat(normalized.astype(np.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE), 3, axis=0)


class RawReshapeDataset(RawSignalDataset):
    """Reuse project MAT parsing, classes, counts and split roles.

    Sampling footprint is 7056 rather than the old STFT footprint. The same
    sampling function/seed/order is used with the new valid start range. CWRU
    target halves are split BEFORE windowing, so a long window cannot cross.
    PU retains the existing per-class window split; overlapping windows across
    PU validation/test remain an inherited risk (not a bearing-level split).
    No PNG is read and no offline dataset is written.
    """
    def __init__(self, cfg: dict) -> None:
        if cfg.get('input_representation') != REPRESENTATION or cfg.get('img_size') != IMAGE_SIZE:
            raise ValueError('Expected raw7056_minmax_reshape84 input with img_size=84')
        if int(cfg['window_size']) != SAMPLE_POINTS:
            raise ValueError('Paper raw input requires window_size=7056')
        self.cfg = cfg
        self.source_classes = list(cfg.get('class_groups', cfg.get('class_names', [])))
        self.source_data, self.target_val_data, self.target_test_data = {}, {}, {}
        self.window_manifest: list[dict] = []
        self.verified_windows = 0
        self._load_pu() if cfg['dataset'] == 'pu' else self._load_cwru()
        self.set_mode('train')
        print(f'Raw reshape: {self.verified_windows} windows, 7056 -> minmax -> [3,84,84]; no STFT/PNG')

    def _windows(self, signal: np.ndarray, starts: np.ndarray, directory: Path,
                 prefix: str) -> np.ndarray:
        windows = []
        for index, start in enumerate(starts):
            start = int(start)
            if start < 0 or start + SAMPLE_POINTS > len(signal):
                raise ValueError('Raw window exceeds its source/validation/test segment')
            raw = np.asarray(signal[start:start + SAMPLE_POINTS], dtype=np.float64)
            windows.append(reshape_samples(raw))
            self.window_manifest.append(dict(
                condition=Path(directory).parent.name,
                class_or_bearing=Path(directory).name,
                role='pool' if prefix == 'stft' else prefix,
                index=index, start_in_segment=start, length=SAMPLE_POINTS,
                end_in_segment=start + SAMPLE_POINTS, segment_length=len(signal),
                sampling_window_size=int(self.cfg['window_size']),
                segment='second_half' if prefix == 'test' else 'first_half' if prefix == 'val' else 'whole',
                raw_sha256=hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest(),
            ))
            self.verified_windows += 1
        return np.stack(windows)

    def to_maml_dataset_format(self, mode: str = 'train') -> tuple[np.ndarray, np.ndarray]:
        view = self.for_mode(mode)
        return view._x, view._y
