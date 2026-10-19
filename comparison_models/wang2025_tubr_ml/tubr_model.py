"""Wang et al. (2025) TUBR-ML under the project protocol; see notes.md."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F


@dataclass(frozen=True)
class MethodConfig:
    eta: float = 0.005  # Table 7; section 4.2 instead says 0.05.
    tau: float = 2.0
    keep_prob: float = 0.9
    block_size: int = 3
    leaky_slope: float = 0.01  # Not specified by the paper.
    attenuation_reduce: str = 'sum'
    detach_prior: bool = False  # Literal two-network loss in Eqs. 8/11.
    inner_decay: float = 0.9
    decay_after: int = 10

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'Non-finite method option: {name}')
        if self.eta < 0 or self.tau < 0 or not 0 < self.keep_prob <= 1:
            raise ValueError('Require eta/tau >= 0 and 0 < keep_prob <= 1')
        if type(self.block_size) is not int or self.block_size < 1:
            raise ValueError('block_size must be a positive integer')
        if self.attenuation_reduce not in ('sum', 'mean'):
            raise ValueError('attenuation_reduce must be sum or mean')
        if type(self.detach_prior) is not bool or not 0 <= self.leaky_slope < 1:
            raise ValueError('Invalid detach_prior/leaky_slope')
        if not 0 < self.inner_decay <= 1 or type(self.decay_after) is not int or self.decay_after < 0:
            raise ValueError('Invalid inner learning-rate schedule')


class StructuredDiscard(nn.Module):
    """DropBlock with valid block origins and Eq. 14 seed probability.

    For inputs with maps smaller than c, cap c to the smaller spatial dimension.
    Use a per-batch scaling factor; an all-dropped map stays finite/zero.
    """
    def __init__(self, keep_prob: float = 0.9, block_size: int = 3) -> None:
        super().__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.keep_prob == 1:
            return x
        n, channels, height, width = x.shape
        block = min(self.block_size, height, width)
        gamma = ((1 - self.keep_prob) * height * width /
                 (block ** 2 * (height - block + 1) * (width - block + 1)))
        seeds = (torch.rand((n * channels, 1, height - block + 1, width - block + 1),
                            device=x.device) < min(gamma, 1.)).to(x.dtype)
        # Expanding valid origins handles odd/even sizes without asymmetric padding.
        dropped = F.conv_transpose2d(seeds, x.new_ones((1, 1, block, block))) > 0
        keep = (~dropped).reshape_as(x).to(x.dtype)
        return x * keep * (keep.numel() / keep.sum().clamp_min(1))


class SDFE(nn.Module):
    """Table 6 / Fig. 4: (Conv-BN-Pool-LeakyReLU-DropBlock)x4, GAP, FC."""
    def __init__(self, ways: int, cfg: MethodConfig, input_channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(input_channels if i == 0 else 64, 64, 3, padding=1, bias=True),
                nn.BatchNorm2d(64, track_running_stats=False),
                nn.MaxPool2d(2), nn.LeakyReLU(cfg.leaky_slope),
                StructuredDiscard(cfg.keep_prob, cfg.block_size),
            ) for i in range(4)
        ])
        self.classifier = nn.Linear(64, ways)
        self.input_channels = input_channels

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 4 or x.shape[1] != self.input_channels or min(x.shape[-2:]) < 16:
            raise ValueError('SDFE input channels/shape do not match the selected input mode')
        features = self.blocks(x).mean(dim=(-2, -1))
        return features, self.classifier(features)


class TUBRNetwork(nn.Module):
    def __init__(self, ways: int = 5, image_size: int = 84,
                 cfg: MethodConfig | None = None, input_channels: int = 3) -> None:
        super().__init__()
        if image_size < 16:
            raise ValueError('Four pooling blocks require image_size >= 16')
        cfg = cfg or MethodConfig()
        self.sdfe = SDFE(ways, cfg, input_channels)
        # 4*(conv weight/bias + BN weight/bias) + FC weight/bias = 18.
        self.gradient_names = tuple(name for name, _ in self.sdfe.named_parameters())
        if len(self.gradient_names) != 18:
            raise RuntimeError('Attenuator expects the documented 18 parameter tensors')
        self.attenuator = nn.Sequential(nn.Linear(18, 18), nn.ReLU(),
                                       nn.Linear(18, 18), nn.Sigmoid())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.sdfe(x)

    def attenuation(self, gradients: tuple[Tensor, ...], cfg: MethodConfig) -> Tensor:
        # Paper does not define tensor -> scalar reduction; explicitly use signed mean.
        descriptor = torch.stack([gradient.mean() for gradient in gradients])
        gates = self.attenuator(descriptor)
        return gates.sum() if cfg.attenuation_reduce == 'sum' else gates.mean()


def entropy(logits: Tensor) -> Tensor:
    """Mean sample entropy, natural logarithm (Eq. 7)."""
    logp = F.log_softmax(logits, dim=1)
    return -(logp.exp() * logp).sum(dim=1).mean()


def turl_loss(before: Tensor, after: Tensor, labels: Tensor,
              cfg: MethodConfig) -> tuple[Tensor, dict[str, Tensor]]:
    """Eqs. 8/10/11: CE + tau*(1-p_true) + eta*(H_before-H_after).

    The signed entropy difference is NOT KL, absolute difference, or a square.
    """
    logp = F.log_softmax(after, dim=1)
    logpt = logp.gather(1, labels[:, None]).squeeze(1)
    poly = (-logpt + cfg.tau * (1 - logpt.exp())).mean()
    prior = entropy(before)
    if cfg.detach_prior:
        prior = prior.detach()
    posterior = entropy(after)
    tul = prior - posterior
    return poly + cfg.eta * tul, dict(poly1=poly, tul=tul,
                                     prior_entropy=prior, adapted_entropy=posterior)


def inner_rate(lr: float, step: int, cfg: MethodConfig) -> float:
    """Zero-based step: first 10 unchanged, step 10 uses 0.9*lr (section 4.2)."""
    return lr * cfg.inner_decay ** max(0, step + 1 - cfg.decay_after)


def adapt(model: TUBRNetwork, x: Tensor, y: Tensor, steps: int, lr: float,
          cfg: MethodConfig, meta_grad: bool) -> tuple[OrderedDict[str, Tensor], Tensor]:
    """Eq. 5, CE support loss; attenuator is meta-learned only in outer loop.

    Recompute omega each step (Table 1). Keep full second-order gradients through
    both support gradients and their descriptor. Evaluation uses support only.
    """
    params = OrderedDict(model.sdfe.named_parameters())
    if not meta_grad:
        params = OrderedDict((name, value.detach().clone().requires_grad_())
                             for name, value in params.items())
    omegas = []
    for step in range(steps):
        with torch.enable_grad():
            logits = functional_call(model.sdfe, params, (x,))[1]
            gradients = torch.autograd.grad(F.cross_entropy(logits, y), tuple(params.values()),
                                            create_graph=meta_grad)
            with torch.set_grad_enabled(meta_grad):
                omega = model.attenuation(gradients, cfg)
            rate = inner_rate(lr, step, cfg)
            params = OrderedDict((name, value - rate * (gradient + omega * value))
                                 for (name, value), gradient in zip(params.items(), gradients))
        omegas.append(omega.detach())
        if not meta_grad:
            params = OrderedDict((name, value.detach().requires_grad_())
                                 for name, value in params.items())
    return params, torch.stack(omegas).mean() if omegas else x.new_zeros(())
