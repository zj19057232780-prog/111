"""Mu et al. (2025), project-protocol reconstruction; see notes.md for ambiguities."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.func import functional_call


@dataclass(frozen=True)
class MethodConfig:
    # Fig. 4 fixes channels=64 and bottleneck=16. Other choices are documented.
    radius: float = 1e-3
    alc_a: float = 0.3
    alc_b: float = 3.0
    alc_g1: float = 0.01
    alc_g2: float = 0.01
    epsilon_min: float = 0.2
    epsilon_max: float = 0.3
    outer_loss: str = 'eq27'
    differentiate_epsilon: bool = True
    differentiate_perturbation: bool = False
    evaluation_amos: bool = True

    def __post_init__(self) -> None:
        import math
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'Non-finite method option: {key}')
        if self.radius < 0 or not 0 <= self.epsilon_min <= self.epsilon_max <= 1:
            raise ValueError('Invalid radius or epsilon range')
        if not 0 <= self.alc_a <= 1 or self.outer_loss not in ('eq27', 'eq19'):
            raise ValueError('Invalid ALC configuration')


class SBCA(nn.Module):
    """Directional pooling and bilateral gates, Eqs. (7)-(12) / Fig. 4."""
    def __init__(self, channels: int = 64, hidden: int = 16) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(channels, hidden, 1)
        # Episode-local batch statistics match the project's train-mode BN protocol.
        self.bn = nn.BatchNorm2d(hidden, track_running_stats=False)
        self.height_gate = nn.Conv2d(hidden, channels, 1)
        self.width_gate = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        height = x.mean(dim=3, keepdim=True)
        width = x.mean(dim=2, keepdim=True).transpose(2, 3)
        encoded = F.relu(self.bn(self.reduce(torch.cat((height, width), dim=2))))
        height, width = encoded.split((h, w), dim=2)
        gh = self.height_gate(height).sigmoid()
        gw = self.width_gate(width.transpose(2, 3)).sigmoid()
        return x * gh * gw


class AMAMLNetwork(nn.Module):
    def __init__(self, ways: int = 5, image_size: int = 38) -> None:
        super().__init__()
        if image_size < 16:
            raise ValueError('Four pooling blocks require image_size >= 16')
        blocks = []
        for i, kernel in enumerate((5, 5, 3, 3)):
            blocks.append(nn.Sequential(
                nn.Conv2d(1 if i == 0 else 64, 64, kernel, padding=kernel // 2),
                nn.BatchNorm2d(64, track_running_stats=False),
                nn.ReLU(), nn.MaxPool2d(2), SBCA(),
            ))
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(64 * (image_size // 16) ** 2, ways)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.blocks(x).flatten(1)
        return features, self.classifier(features)


def adaptive_loss(logits: Tensor, labels: Tensor, cfg: MethodConfig) -> tuple[Tensor, Tensor]:
    """Literal Eq. (15); Eq. (27) default, Eq. (19) product interpretation optional."""
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    lam = -(p * logp).sum()
    target = p.gather(1, labels[:, None]).squeeze(1)
    non_target = p.masked_fill(F.one_hot(labels, logits.shape[1]).bool(), -torch.inf).max(1).values
    eta = -(non_target - target).abs().sum()
    eps = (cfg.alc_a * torch.sigmoid(cfg.alc_g1 * eta - cfg.alc_g2 * lam + cfg.alc_b))
    eps = eps.clamp(cfg.epsilon_min, cfg.epsilon_max)
    if not cfg.differentiate_epsilon:
        eps = eps.detach()
    hard = -(1 - eps) * logp.gather(1, labels[:, None]).mean()
    uniform = -eps * logp.mean()
    loss = hard + uniform
    if cfg.outer_loss == 'eq19':
        loss = loss + hard * uniform
    return loss, eps


def adapt(model: nn.Module, x: Tensor, y: Tensor, steps: int, lr: float,
          cfg: MethodConfig, meta_grad: bool) -> OrderedDict[str, Tensor]:
    """Eq. (25)-(26), with a global vector norm and optional perturbation derivative."""
    params = OrderedDict(model.named_parameters())
    if not meta_grad:
        params = OrderedDict((k, v.detach().clone().requires_grad_()) for k, v in params.items())
    amos = meta_grad or cfg.evaluation_amos
    for _ in range(steps):
        support_loss = F.cross_entropy(functional_call(model, params, (x,))[1], y)
        through_delta = meta_grad and cfg.differentiate_perturbation
        gradients = torch.autograd.grad(support_loss, tuple(params.values()),
                                        create_graph=through_delta if amos and cfg.radius else meta_grad)
        if amos and cfg.radius:
            norm = torch.stack([(p.abs() * g).square().sum()
                                for p, g in zip(params.values(), gradients)]).sum().sqrt().clamp_min(1e-12)
            perturbed = OrderedDict()
            for (name, param), grad in zip(params.items(), gradients):
                delta = cfg.radius * param.square() * grad / norm
                perturbed[name] = param + (delta if through_delta else delta.detach())
            shifted_loss = F.cross_entropy(functional_call(model, perturbed, (x,))[1], y)
            gradients = torch.autograd.grad(shifted_loss, tuple(perturbed.values()), create_graph=meta_grad)
        params = OrderedDict((name, param - lr * grad)
                             for (name, param), grad in zip(params.items(), gradients))
        if not meta_grad:
            params = OrderedDict((k, v.detach().requires_grad_()) for k, v in params.items())
    return params
