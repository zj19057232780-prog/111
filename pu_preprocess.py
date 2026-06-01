"""
Offline STFT preprocessing for the PU baseline.

Normalized .mat layout:
    mat_root/condition/fault.mat

Output PNG layout:
    png_root/condition/fault/stft_000.png
"""

from __future__ import annotations

import argparse
import os
import shutil

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, stft

from config_pu import PU_CONFIG
from pu_loader import PULoader


def to_uint8_robust(S, p_low=1, p_high=99, gamma=1.0, eps=1e-8):
    lo, hi = np.percentile(S, [p_low, p_high])
    if hi - lo < eps:
        lo, hi = S.min(), S.max()
    if hi - lo < eps:
        return np.zeros_like(S, dtype=np.uint8)
    S = np.clip(S, lo, hi)
    S = (S - lo) / (hi - lo + eps)
    if gamma != 1.0 and gamma > 0:
        S = np.power(S, gamma)
    return (S * 255).astype(np.uint8)


def bandpass_filter(x, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)


def compute_stft_log_image(signal_segment, cfg):
    x = np.asarray(signal_segment, dtype=np.float64).ravel()

    if cfg.get('normalize', 'demean') == 'zscore':
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    else:
        x = x - np.mean(x)

    bandpass = cfg.get('stft_bandpass')
    fs = cfg.get('stft_fs', 64000)
    if bandpass is not None and len(bandpass) == 2:
        x = bandpass_filter(x, bandpass[0], bandpass[1], fs)

    _, _, Zxx = stft(
        x,
        fs=fs,
        window=cfg.get('stft_window', 'hann'),
        nperseg=cfg.get('stft_nperseg', 256),
        noverlap=cfg.get('stft_noverlap', 192),
        boundary=None,
        padded=False,
    )
    S = np.log10(np.abs(Zxx) + cfg.get('stft_log_eps', 1e-8))
    img = to_uint8_robust(
        S,
        p_low=cfg.get('stft_p_low', 5),
        p_high=cfg.get('stft_p_high', 95),
        gamma=cfg.get('stft_gamma', 0.7),
    )

    img_size = cfg.get('img_size', 64)
    if img.shape != (img_size, img_size):
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


def sample_starts(signal_len: int, window_size: int, count: int, rng) -> np.ndarray:
    if signal_len < window_size:
        return np.zeros((0,), dtype=np.int64)
    max_start = signal_len - window_size
    if count <= 1:
        return np.array([max_start // 2], dtype=np.int64)

    # A deterministic grid keeps the whole signal covered; a tiny random jitter
    # avoids generating exactly identical starts across classes with equal length.
    base = np.linspace(0, max_start, num=count, dtype=np.int64)
    step = max(1, max_start // max(count - 1, 1))
    jitter = rng.integers(-step // 4, step // 4 + 1, size=count)
    starts = np.clip(base + jitter, 0, max_start)
    return starts.astype(np.int64)


def save_grayscale_png(img2d, save_path):
    ok = cv2.imwrite(save_path, img2d)
    if not ok:
        raise IOError(f'Failed to write image: {save_path}')


def process_mat_root_to_png(
    mat_root,
    png_root,
    window_size=4096,
    img_size=64,
    samples_per_class=40,
    cfg=None,
    seed=24,
    clean=False,
):
    if cfg is None:
        cfg = dict(PU_CONFIG)
    cfg = dict(cfg)
    cfg['img_size'] = img_size

    if clean and os.path.isdir(png_root):
        shutil.rmtree(png_root)
    os.makedirs(png_root, exist_ok=True)

    loader = PULoader(mat_root)
    rng = np.random.default_rng(seed)
    total = 0

    for condition in sorted(loader.get_condition_names()):
        fault_signals = loader.load_condition(condition)
        out_cond = os.path.join(png_root, condition)
        os.makedirs(out_cond, exist_ok=True)

        for fault_name in sorted(fault_signals):
            sig = np.asarray(fault_signals[fault_name], dtype=np.float64).ravel()
            starts = sample_starts(len(sig), window_size, samples_per_class, rng)
            if starts.size == 0:
                print(f'[skip] {condition}/{fault_name}: signal shorter than window')
                continue

            fault_dir = os.path.join(out_cond, fault_name)
            os.makedirs(fault_dir, exist_ok=True)

            for idx, start in enumerate(starts):
                segment = sig[start:start + window_size]
                img = compute_stft_log_image(segment, cfg)
                save_path = os.path.join(fault_dir, f'stft_{idx:03d}.png')
                save_grayscale_png(img, save_path)
                total += 1
            print(f'[{condition}] {fault_name}: generated {len(starts)} STFT images')

    print(f'Done. Generated {total} PNG files under {os.path.abspath(png_root)}')


def main() -> None:
    cfg = PU_CONFIG
    parser = argparse.ArgumentParser(description='PU .mat -> STFT PNG preprocessing')
    parser.add_argument('--mat_root', default=cfg['mat_root'])
    parser.add_argument('--png_root', default=cfg['root_path'])
    parser.add_argument('--window', type=int, default=cfg['window_size'])
    parser.add_argument('--img_size', type=int, default=cfg['img_size'])
    parser.add_argument('--samples_per_class', type=int, default=cfg['samples_per_class'])
    parser.add_argument('--seed', type=int, default=cfg['preprocess_seed'])
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()

    process_mat_root_to_png(
        mat_root=args.mat_root,
        png_root=args.png_root,
        window_size=args.window,
        img_size=args.img_size,
        samples_per_class=args.samples_per_class,
        cfg=cfg,
        seed=args.seed,
        clean=args.clean,
    )


if __name__ == '__main__':
    main()
