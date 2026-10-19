"""Reuse project sampling positions; return 2000 real samples for AdaptiveSquare."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from raw_signal_dataset import RawSignalDataset

IMAGE_SIZE = 45
SAMPLE_POINTS = 2000
REPRESENTATION = 'raw2000_adaptive_square45'


def raw_samples(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != SAMPLE_POINTS or not np.isfinite(values).all():
        raise ValueError('MFEML input requires exactly 2000 finite consecutive raw samples')
    return values.astype(np.float32)[None, :]


class PaperRawDataset(RawSignalDataset):
    """Reuse source conditions, MAT parsers, classes, RNG and split definitions.

    Keep the project's sampling footprint (PU4096/CWRU3072) and start positions;
    select its first 2000 points. No normalization is specified by the paper, so
    preserve amplitudes. Model performs padding/reshape online. No STFT/PNG writes.
    """
    def __init__(self, cfg: dict) -> None:
        if cfg.get('input_representation') != REPRESENTATION or cfg.get('img_size') != IMAGE_SIZE:
            raise ValueError('Expected paper raw2000 input and square image_size=45')
        if int(cfg['window_size']) < SAMPLE_POINTS:
            raise ValueError('Project sampling footprint cannot be shorter than 2000')
        self.cfg = cfg
        self.source_classes = list(cfg.get('class_groups', cfg.get('class_names', [])))
        self.source_data, self.target_val_data, self.target_test_data = {}, {}, {}
        self.window_manifest: list[dict] = []
        self.verified_windows = 0
        self._load_pu() if cfg['dataset'] == 'pu' else self._load_cwru()
        self.set_mode('train')
        print(f'Raw MFEML input: {self.verified_windows} windows, [1,2000]; model pads to 45x45')

    def _windows(self, signal: np.ndarray, starts: np.ndarray, directory: Path,
                 prefix: str) -> np.ndarray:
        windows = []
        for index, start in enumerate(starts):
            start = int(start)
            if start < 0 or start + SAMPLE_POINTS > len(signal):
                raise ValueError('Raw window crosses its source/validation/test segment boundary')
            raw = np.asarray(signal[start:start + SAMPLE_POINTS], dtype=np.float64)
            windows.append(raw_samples(raw))
            self.window_manifest.append(dict(
                condition=Path(directory).parent.name, class_or_bearing=Path(directory).name,
                role='pool' if prefix == 'stft' else prefix, index=index,
                start_in_segment=start, end_in_segment=start + SAMPLE_POINTS,
                length=SAMPLE_POINTS, segment_length=len(signal),
                sampling_window_size=int(self.cfg['window_size']),
                segment='second_half' if prefix == 'test' else 'first_half' if prefix == 'val' else 'whole',
                raw_sha256=hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest(),
            ))
            self.verified_windows += 1
        return np.stack(windows)

    def to_maml_dataset_format(self, mode: str = 'train') -> tuple[np.ndarray, np.ndarray]:
        view = self.for_mode(mode)
        return view._x, view._y
