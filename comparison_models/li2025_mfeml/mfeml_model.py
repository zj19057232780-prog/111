"""Li et al. (2025), MFEML: source-grounded project-protocol reconstruction."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F


@dataclass(frozen=True)
class MethodConfig:
    # Not specified in the MFEML paper: fixed implementation choices, not tuned.
    simam_lambda: float = 1e-4
    eca_kernel: int = 3
    attention_fusion: str = 'parallel'  # Fig. 2: GAP/ECA and SimAM on the same input.

    def __post_init__(self) -> None:
        if not math.isfinite(self.simam_lambda) or self.simam_lambda <= 0:
            raise ValueError('simam_lambda must be positive and finite')
        if type(self.eca_kernel) is not int or self.eca_kernel < 1 or self.eca_kernel % 2 != 1:
            raise ValueError('eca_kernel must be a positive odd integer')
        if self.attention_fusion not in ('parallel', 'sequential'):
            raise ValueError('attention_fusion must be parallel or sequential')


class AdaptiveSquare(nn.Module):
    """Section 3.1: keep all signal values, append zeros to ceil(sqrt(L))**2."""
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 4:
            if x.shape[1] != 1 or x.shape[-1] != x.shape[-2]:
                raise ValueError('Existing images must be single-channel square tensors')
            return x
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] < 1:
            raise ValueError('Expected raw [B,1,L] / [B,L], or image [B,1,H,H]')
        length = x.shape[-1]
        side = math.isqrt(length)
        side += side * side < length
        return F.pad(x, (0, side * side - length)).reshape(x.shape[0], 1, side, side)


class MetaFeatureEnhancement(nn.Module):
    """Eqs. 3-5 / Fig. 2: energy gate (SimAM) times learned ECA channel gate."""
    def __init__(self, cfg: MethodConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.channel_conv = nn.Conv1d(1, 1, cfg.eca_kernel,
                                      padding=cfg.eca_kernel // 2, bias=False)

    def spatial_gate(self, x: Tensor) -> Tensor:
        deviations = (x - x.mean(dim=(-2, -1), keepdim=True)).square()
        count = x.shape[-2] * x.shape[-1]
        variance = deviations.sum(dim=(-2, -1), keepdim=True) / max(count - 1, 1)
        inverse_energy = deviations / (4 * (variance + self.cfg.simam_lambda)) + 0.5
        return inverse_energy.sigmoid()

    def channel_gate(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=(-2, -1)).unsqueeze(1)  # [B,1,C], convolve over channels.
        return self.channel_conv(pooled).sigmoid().squeeze(1).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: Tensor) -> Tensor:
        enhanced = x * self.spatial_gate(x)
        descriptor_input = x if self.cfg.attention_fusion == 'parallel' else enhanced
        return enhanced * self.channel_gate(descriptor_input)


class MFEMLNetwork(nn.Module):
    """Table 2: 4x64 Conv3 blocks and 256-dimensional flattened FC input."""
    def __init__(self, ways: int = 5, image_size: int = 45,
                 cfg: MethodConfig | None = None) -> None:
        super().__init__()
        if image_size < 16 or ways < 2:
            raise ValueError('Require four-pool-compatible input and at least two classes')
        cfg = cfg or MethodConfig()
        self.square = AdaptiveSquare()
        self.blocks = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(1 if i == 0 else 64, 64, 3, padding=1),
                          nn.BatchNorm2d(64, track_running_stats=False),
                          nn.ReLU(), nn.MaxPool2d(2), MetaFeatureEnhancement(cfg))
            for i in range(4)
        ])
        # Raw2000 -> 45 -> 22 -> 11 -> 5 -> 2: this is identity for the paper input.
        # STFT64 -> 4x4: pool to 2x2 to retain Table 2's 256-dimensional head.
        self.head_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Linear(256, ways)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        image = self.square(x)
        if min(image.shape[-2:]) < 16:
            raise ValueError('Input is too short for four max-pooling blocks')
        features = self.head_pool(self.blocks(image)).flatten(1)
        return features, self.classifier(features)


def adapt(model: MFEMLNetwork, x: Tensor, y: Tensor, steps: int, lr: float,
          meta_grad: bool) -> OrderedDict[str, Tensor]:
    """Eq. 6: task-local SGD on support CE; retain exact second-order meta gradient."""
    if steps < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Invalid adaptation steps or learning rate')
    params = OrderedDict(model.named_parameters())
    if not meta_grad:
        params = OrderedDict((name, value.detach().clone().requires_grad_())
                             for name, value in params.items())
    for _ in range(steps):
        with torch.enable_grad():
            logits = functional_call(model, params, (x,))[1]
            gradients = torch.autograd.grad(F.cross_entropy(logits, y), tuple(params.values()),
                                            create_graph=meta_grad)
            params = OrderedDict((name, value - lr * gradient)
                                 for (name, value), gradient in zip(params.items(), gradients))
        if not meta_grad:
            params = OrderedDict((name, value.detach().requires_grad_())
                                 for name, value in params.items())
    return params
