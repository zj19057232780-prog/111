"""Closed-set source-supervised ResNet18, with direct target inference."""
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def data_loader(data, classes, batch_size, shuffle=False):
    if set(data) != set(classes) or any(len(data[c]) == 0 for c in classes):
        raise ValueError('Supervised ResNet18 requires nonempty matching source/target classes')
    x = np.concatenate([data[c] for c in classes])
    y = np.concatenate([np.full(len(data[c]), i, dtype=np.int64) for i, c in enumerate(classes)])
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=batch_size, shuffle=shuffle)


def evaluate(model, batches):
    model.eval()  # Fixed source BN statistics; no support or target adaptation.
    device = next(model.parameters()).device
    features, truths, predictions = [], [], []
    total_loss = count = 0
    with torch.no_grad():
        for x, y in batches:
            f, logits = model(x.to(device))
            total_loss += torch.nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
            count += len(y)
            features.append(f.cpu().numpy())
            truths.append(y.numpy())
            predictions.append(logits.argmax(1).cpu().numpy())
    truth, pred = np.concatenate(truths), np.concatenate(predictions)
    return total_loss / count, float(np.mean(truth == pred)), np.concatenate(features), truth, pred


def train_supervised(owner, save_path, quick_test=False):
    from maml_train_pic import _save_training_history
    cfg, model = owner.pu_config, owner.model
    storage = owner._get_pu_storage()
    classes = list(storage.source_classes)
    size = int(cfg.get('supervised_batch_size', 64))
    training = data_loader(storage.source_data, classes, size, shuffle=True)
    validation = data_loader(storage.target_val_data, classes, size)
    epochs = 1 if quick_test else int(cfg.get('supervised_epochs', 100))
    limit = 2 if quick_test else int(cfg.get('supervised_max_batches', 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get('supervised_lr', .001)),
                                 weight_decay=float(cfg.get('supervised_weight_decay', .0001)))
    device = next(model.parameters()).device
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    best_path = save_path + '_best'
    metadata = dict(cfg, source_classes=classes, quick_test=quick_test,
                    training_protocol='source_supervised_direct_target', evaluation_bn='frozen_source')
    best, history = -1., []
    for epoch in range(epochs):
        started = time.time()
        model.train()
        count = correct = 0
        total_loss = 0.
        for index, (x, y) in enumerate(training):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[1]
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite supervised loss')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
            if limit and index + 1 >= limit:
                break
        val_loss, val_acc, _, _, _ = evaluate(model, validation)
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), best_path)
            with open(best_path + '.config.json', 'w', encoding='utf-8') as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2, default=str)
        history.append(dict(epoch=epoch+1, train_loss=total_loss/count, train_acc=correct/count,
                            valid_loss=val_loss, valid_acc=val_acc, best_valid_acc=best,
                            epoch_seconds=time.time()-started))
        print(f'ResNet18 epoch {epoch+1}: source acc={correct/count:.4f}, '
              f'direct target validation acc={val_acc:.4f} ({len(classes)} classes)')
    _save_training_history(history, save_path, cfg)
    return best_path


def test_supervised(owner, load_path):
    from maml_train_pic import _save_test_visualizations
    cfg, model = owner.pu_config, owner.model
    storage = owner._get_pu_storage()
    classes = list(storage.source_classes)
    with open(str(load_path) + '.config.json', encoding='utf-8') as stream:
        saved = json.load(stream)
    if saved.get('method') != 'resnet18' or saved.get('source_classes') != classes:
        raise ValueError('Checkpoint method or class order differs from current ResNet18 config')
    model.load_state_dict(torch.load(load_path, map_location=next(model.parameters()).device, weights_only=True))
    batches = data_loader(storage.target_test_data, classes, int(cfg.get('supervised_batch_size', 64)))
    loss, accuracy, features, truth, pred = evaluate(model, batches)
    rows = [dict(sample=i, true_class_index=int(t), pred_class_index=int(p),
                 true_class_name=classes[t], pred_class_name=classes[p])
            for i, (t, p) in enumerate(zip(truth, pred))]
    _save_test_visualizations(features=features, y_true=truth, y_pred=pred, rows=rows,
                              class_names=classes, load_path=load_path, cfg=cfg, tsne_payload=None)
    print(f'ResNet18 direct test: {len(classes)} classes, {len(truth)} samples, '
          f'loss={loss:.4f}, accuracy={accuracy:.4f}; no support adaptation')
    return dict(loss=loss, accuracy=accuracy, samples=len(truth), classes=classes)
