"""Project-native classical metric baselines (not verbatim official code).

ProtoNet: squared Euclidean prototypes + CE; MatchingNet: cosine attention,
optional support BiLSTM / query attentive-LSTM full-context embeddings + NLL;
RelationNet: class-summed spatial features, learned sigmoid relations + MSE.
All share this project's four-block CNN encoder and episodic data protocol.
"""
import torch
from torch import nn
from torch.nn import functional as F
from maml_model import ConvBase


class ClassicMetricModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.method = cfg['method']
        channels = int(cfg.get('hidden_size', 64))
        self.encoder = ConvBase(channels, cfg.get('in_channels', 1), 4)
        dimension = channels * (int(cfg.get('img_size', 64)) // 16) ** 2
        self.temperature = float(cfg.get('metric_temperature', 1.0))
        if self.temperature <= 0:
            raise ValueError('metric_temperature must be positive')
        self.fce = bool(cfg.get('matching_fce', True))
        self.fce_steps = int(cfg.get('matching_fce_steps', 5))
        if self.method == 'matchingnet' and self.fce:
            if self.fce_steps < 1:
                raise ValueError('matching_fce_steps must be >= 1')
            self.support_lstm = nn.LSTM(dimension, dimension, batch_first=True,
                                        bidirectional=True)
            self.query_lstm = nn.LSTMCell(2 * dimension, dimension)
        if self.method == 'relationnet':
            width = int(cfg.get('relation_hidden_size', 8))
            self.relation = nn.Sequential(
                nn.Conv2d(2 * channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                nn.Flatten(), nn.Linear(channels, width), nn.ReLU(),
                nn.Linear(width, 1), nn.Sigmoid())

    def episode(self, support, labels, query):
        classes = torch.unique(labels, sorted=True)
        if not torch.equal(classes, torch.arange(len(classes), device=labels.device)):
            raise ValueError('Episode support labels must be contiguous starting at zero')
        ways = len(classes)
        # Batch-query BN matches the existing evaluation convention.
        sf, qf = self.encoder(support), self.encoder(query)
        s, q = sf.flatten(1), qf.flatten(1)
        if self.method == 'protonet':
            prototypes = torch.stack([s[labels == c].mean(0) for c in classes])
            scores = -(q[:, None] - prototypes[None]).square().sum(-1) / self.temperature
        elif self.method == 'matchingnet':
            if self.fce:
                contextual, _ = self.support_lstm(s.unsqueeze(0))
                forward, backward = contextual.squeeze(0).chunk(2, dim=-1)
                s = s + forward + backward
                hidden, cell = q, torch.zeros_like(q)
                for _ in range(self.fce_steps):
                    read = (hidden @ s.T).softmax(-1) @ s
                    hidden, cell = self.query_lstm(torch.cat([q, read], -1), (hidden, cell))
                    hidden = hidden + q
                q = hidden
            attention = (F.normalize(q, dim=-1) @ F.normalize(s, dim=-1).T
                         / self.temperature).softmax(-1)
            probabilities = attention @ F.one_hot(labels, ways).to(attention.dtype)
            scores = probabilities.clamp_min(1e-12).log()
        else:
            prototypes = torch.stack([sf[labels == c].sum(0) for c in classes])
            pairs = torch.cat([prototypes.unsqueeze(0).expand(len(q), -1, -1, -1, -1),
                               qf.unsqueeze(1).expand(-1, ways, -1, -1, -1)], dim=2)
            scores = self.relation(pairs.flatten(0, 1)).reshape(len(q), ways)
        return q, scores

    def episode_loss(self, scores, query_labels):
        if self.method == 'relationnet':
            return F.mse_loss(scores, F.one_hot(query_labels, scores.shape[1]).to(scores.dtype))
        if self.method == 'matchingnet':
            return F.nll_loss(scores, query_labels)
        return F.cross_entropy(scores, query_labels)


class MetricAlgorithm:
    """Reuse the outer-loop interface, without any support gradient adaptation."""
    def __init__(self, model):
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def clone(self):
        # episode() has no persistent support state or parameter updates. BN uses
        # batch statistics, as in the existing MAML training/testing protocol.
        return self.model
