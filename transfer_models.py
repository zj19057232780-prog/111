"""Supervised source training and independent support-only fine tuning.

Project adaptations: one-channel ResNet18 (no ImageNet weights); WDCNN with
five 1D convolutions and adaptive final pooling for PU/CWRU window lengths.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(z)) + self.shortcut(x))


class TransferModel(nn.Module):
    def __init__(self, method, num_classes):
        super().__init__()
        if method in ('resnet18', 'resnet18_ft'):
            layers = [nn.Conv2d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
                      nn.ReLU(), nn.MaxPool2d(3, 2, 1)]
            cin = 64
            for index, cout in enumerate((64, 128, 256, 512)):
                layers += [BasicBlock(cin, cout, 1 if index == 0 else 2), BasicBlock(cout, cout)]
                cin = cout
            layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
            self.feature_dim = 512
        elif method == 'tl_wdcnn':
            layers = []
            cin = 1
            for index, cout in enumerate((16, 32, 64, 64, 64)):
                layers += [nn.Conv1d(cin, cout, 64 if index == 0 else 3,
                                     stride=16 if index == 0 else 1,
                                     padding=24 if index == 0 else 1),
                           nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2)]
                cin = cout
            layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 100), nn.ReLU()]
            self.feature_dim = 100
        else:
            raise ValueError(f'Unknown transfer model {method}')
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return features, self.classifier(features)


class TransferAlgorithm:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg

    def clone(self):
        return TransferEpisode(self.model, self.cfg)


class TransferEpisode:
    def __init__(self, model, cfg):
        self.base, self.cfg = model, cfg

    def evaluate_episode(self, support, labels, query):
        classes = torch.unique(labels, sorted=True)
        if not torch.equal(classes, torch.arange(len(classes), device=labels.device)):
            raise ValueError('Support labels must be local contiguous labels')
        # Deepcopy includes buffers. No test task modifies the pretrained model.
        model = copy.deepcopy(self.base)
        model.classifier = nn.Linear(model.feature_dim, len(classes)).to(support.device)
        if self.cfg.get('ft_scope', 'all') == 'head':
            for p in model.encoder.parameters():
                p.requires_grad_(False)
        model.train()
        if self.cfg.get('ft_bn_mode', 'batch') == 'frozen':
            for layer in model.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                    lr=float(self.cfg.get('ft_lr', 0.01)))
        with torch.enable_grad():
            for _ in range(int(self.cfg.get('ft_steps', 10))):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(support)[1], labels)
                loss.backward()
                optimizer.step()
        # Query labels are intentionally absent from this interface.
        with torch.no_grad():
            return model(query)
