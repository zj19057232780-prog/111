"""Raw vibration adaptations of ResNet18, ResNet18-FT and TL-WDCNN.

No STFT/PNG access. ResNet18 is a 1D adaptation, not an image-model reproduction.
"""
import json
import time
from pathlib import Path
import torch
from torch import nn
import maml_train_pic as training
from classic_config import resolve_method_config
from raw_fewshot import RawFewShotLearner
from transfer_models import TransferModel
from supervised_training import data_loader, evaluate


class BasicBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv1d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm1d(cout), nn.ReLU(), nn.Conv1d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm1d(cout))
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        return torch.relu(self.branch(x) + self.shortcut(x))


class ResNet18Raw(nn.Module):
    def __init__(self, classes):
        super().__init__()
        layers = [nn.Conv1d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64),
                  nn.ReLU(), nn.MaxPool1d(3, 2, 1)]
        cin = 64
        for index, cout in enumerate((64, 128, 256, 512)):
            layers += [BasicBlock1D(cin, cout, 1 if index == 0 else 2), BasicBlock1D(cout, cout)]
            cin = cout
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.feature_dim = 512
        self.classifier = nn.Linear(512, classes)

    def forward(self, x):
        features = self.encoder(x)
        return features, self.classifier(features)


class RawTransferLearner(RawFewShotLearner):
    def __init__(self, ways, pu_config):
        cfg = resolve_method_config(pu_config)
        if cfg['method'] not in ('resnet18', 'resnet18_ft', 'tl_wdcnn'):
            raise ValueError('Unsupported raw transfer method')
        cfg.update(input_representation='raw_1d', raw_normalization='per_window_zscore',
                   backbone='wdcnn' if cfg['method'] == 'tl_wdcnn' else 'resnet18_1d')
        self.pu_config, self.method, self.ways = cfg, cfg['method'], ways
        self._pu_storage = None
        self.backbone_name = cfg['backbone']
        self.frequency_module = self.denoise_module = self.attention_module = 'none'
        classes = len(cfg.get('class_groups', cfg.get('class_names', [])))
        self.model = (TransferModel('tl_wdcnn', classes) if self.method == 'tl_wdcnn'
                      else ResNet18Raw(classes)).to(training.device)

    def train(self, save_path, shots=1, quick_test=False):
        if self.method != 'resnet18':
            return super().train(save_path, shots, quick_test)
        cfg, model = self.pu_config, self.model
        storage = self._get_raw_storage()
        classes = list(storage.source_classes)
        size = int(cfg.get('supervised_batch_size', 64))
        batches = data_loader(storage.source_data, classes, size, shuffle=True)
        validation = data_loader(storage.target_val_data, classes, size)
        epochs = 1 if quick_test else int(cfg['supervised_epochs'])
        patience = 0 if quick_test else int(cfg.get('early_stop_patience', 100))
        limit = 2 if quick_test else int(cfg.get('supervised_max_batches', 0))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get('supervised_lr', .001)),
                                     weight_decay=float(cfg.get('supervised_weight_decay', .0001)))
        device = next(model.parameters()).device
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        best_path = save_path + '_best'
        metadata = dict(cfg, source_classes=classes, quick_test=quick_test,
            training_protocol='source_supervised_direct_target', evaluation_bn='frozen_source')
        Path(best_path + '.config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        best, stale, history = -1., 0, []
        for epoch in range(epochs):
            started = time.time()
            model.train()
            total_loss = count = correct = 0
            for index, (x, y) in enumerate(batches):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)[1]
                loss = nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite supervised loss')
                loss.backward()
                optimizer.step()
                count += len(y)
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                if limit and index + 1 >= limit:
                    break
            val_loss, val_acc, _, _, _ = evaluate(model, validation)
            if val_acc > best:
                best, stale = val_acc, 0
                torch.save(model.state_dict(), best_path)
            else:
                stale += 1
            history.append(dict(epoch=epoch+1, train_loss=total_loss/count, train_acc=correct/count,
                valid_loss=val_loss, valid_acc=val_acc, best_valid_acc=best,
                epochs_without_improvement=stale, epoch_seconds=time.time()-started))
            print(f'Epoch {epoch+1}: source acc={correct/count:.4f}, direct validation acc={val_acc:.4f}, stale={stale}/{patience}', flush=True)
            if patience and stale >= patience:
                print('Early stopping: no validation improvement.', flush=True)
                break
        training._save_training_history(history, save_path, cfg)
        return best_path
