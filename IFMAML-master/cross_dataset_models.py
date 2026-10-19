"""Thin adapters over the existing project models and reconstructed algorithms."""
from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.func import functional_call

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
PAPERS = {'amaml': 'comparison_models.mu2025_amaml.model',
          'tubr_ml': 'comparison_models.wang2025_tubr_ml.tubr_model',
          'mfeml': 'comparison_models.li2025_mfeml.mfeml_model'}
METRICS = ('protonet', 'matchingnet', 'relationnet')


@lru_cache(None)
def paper_module(method: str):
    return importlib.import_module(PAPERS[method])


def native_raw_input(values, method: str, domain: str):
    """Use the exact existing raw adapters, without frequency conversion/resampling."""
    import numpy as np
    if method == 'amaml':
        from comparison_models.mu2025_amaml.raw_reshape import reshape_samples
        return reshape_samples(values[:1444])
    if method == 'tubr_ml':
        from comparison_models.wang2025_tubr_ml.raw_reshape import reshape_samples
        return reshape_samples(values[:7056])
    if method == 'mfeml':
        from comparison_models.li2025_mfeml.raw_input import raw_samples
        return raw_samples(values[:2000])
    if method == 'configured_maml':
        raise ValueError('SCG-MAML supports STFT only')
    length = 3072 if domain == 'A' else 4096
    raw = np.asarray(values[:length], dtype=np.float64)
    return ((raw - raw.mean()) / (raw.std() + 1e-8)).astype(np.float32)[None]


def build_model(cfg: dict, device: torch.device):
    """Construct existing modules unchanged; no replacement neural architectures."""
    from maml_train_pic import MAML_learner
    from raw_fewshot import RawMAMLModel, RawMetricModel
    method = cfg['method']
    if method in PAPERS:
        mod = paper_module(method)
        options = mod.MethodConfig(**cfg['method_options'])
        side = {'amaml': 38, 'tubr_ml': 84, 'mfeml': 45}[method] if cfg['input_mode'] == 'nostft' else 64
        if method == 'amaml':
            model = mod.AMAMLNetwork(cfg['n_way'], side)
        elif method == 'tubr_ml':
            model = mod.TUBRNetwork(cfg['n_way'], side, options,
                                    input_channels=3 if cfg['input_mode'] == 'nostft' else 1)
        else:
            model = mod.MFEMLNetwork(cfg['n_way'], side, options)
    elif cfg['input_mode'] == 'nostft' and method in (*METRICS, 'maml'):
        model = (RawMAMLModel(cfg['n_way'], cfg['hidden_size'], cfg['raw_feature_bins'])
                 if method == 'maml' else RawMetricModel(cfg))
    else:
        model = MAML_learner(cfg['n_way'], cfg).model
    return model.to(device)


def episode(model, batch: tuple, cfg: dict, training: bool, steps: int):
    """Preserve each method's own adaptation, training objective and loss."""
    from run_hust_maml import HUSTMAML
    from transfer_models import TransferEpisode
    method = cfg['method']
    device = next(model.parameters()).device
    sx, sy, qx, qy = (t.to(device) for t in batch[:4])
    if method in METRICS:
        with torch.set_grad_enabled(training):
            features, scores = model.episode(sx, sy, qx)
            loss = model.episode_loss(scores, qy)
    elif method == 'tl_wdcnn':
        features, scores = TransferEpisode(model, cfg).evaluate_episode(sx, sy, qx)
        loss = F.cross_entropy(scores, qy)
    elif method in PAPERS:
        mod = paper_module(method)
        options = mod.MethodConfig(**cfg['method_options'])
        # TUBR's existing implementation disables structured discard at evaluation.
        previous = model.training
        if method == 'tubr_ml':
            model.train(training)
        try:
            with torch.enable_grad():
                if method == 'amaml':
                    params = mod.adapt(model, sx, sy, steps, cfg['inner_lr'], options, training)
                elif method == 'mfeml':
                    params = mod.adapt(model, sx, sy, steps, cfg['inner_lr'], training)
                else:
                    before = model(qx)[1] if training else None
                    params, _ = mod.adapt(model, sx, sy, steps, cfg['inner_lr'], options, training)
            with torch.set_grad_enabled(training):
                features, scores = functional_call(model.sdfe if method == 'tubr_ml' else model,
                                                   params, (qx,))
                if training and method == 'amaml':
                    loss = mod.adaptive_loss(scores, qy, options)[0]
                elif training and method == 'tubr_ml':
                    loss = mod.turl_loss(before, scores, qy, options)[0]
                else:
                    loss = F.cross_entropy(scores, qy)
        finally:
            model.train(previous)
    else:
        clone = HUSTMAML(model, cfg['inner_lr']).clone()
        with torch.enable_grad():
            for _ in range(steps):
                clone.adapt(F.cross_entropy(clone(sx)[1], sy))
            features, scores = clone(qx)
            loss = F.cross_entropy(scores, qy)
    accuracy = (scores.argmax(1) == qy).float().mean()
    return loss, accuracy, features, qy, scores


def query_ce(scores, labels, method: str) -> float:
    """Comparable checkpoint tie-breaker even when the native loss is MSE."""
    if method == 'matchingnet':
        return float(F.nll_loss(scores, labels))
    if method == 'relationnet':
        p = scores.clamp_min(1e-12)
        return float(F.nll_loss((p / p.sum(1, keepdim=True)).log(), labels))
    return float(F.cross_entropy(scores, labels))
