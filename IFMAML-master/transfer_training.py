"""Ordinary minibatch source pretraining; validation selects pretrained weights."""
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def train_transfer(owner, save_path, shots, quick_test=False):
    from maml_train_pic import _save_training_history, capture_rng_state, restore_rng_state
    cfg, model = owner.pu_config, owner.model
    device = next(model.parameters()).device
    storage = owner._get_pu_storage()
    classes = storage.source_classes
    x = np.concatenate([storage.source_data[c] for c in classes])
    y = np.concatenate([np.full(len(storage.source_data[c]), i, dtype=np.int64)
                        for i, c in enumerate(classes)])
    batches = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                         batch_size=int(cfg.get('ft_pretrain_batch_size', 64)), shuffle=True)
    epochs = 1 if quick_test else int(cfg.get('ft_pretrain_epochs', 100))
    limit = 2 if quick_test else int(cfg.get('ft_pretrain_max_batches', 0))
    episodes = 2 if quick_test else int(cfg.get('ft_validation_episodes', 16))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get('ft_pretrain_lr', .001)),
                           weight_decay=float(cfg.get('ft_weight_decay', 1e-4)))
    validation = owner.build_tasks('validation', owner.ways, shots, cfg.get('q_query', 15), episodes)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    best_path = save_path + '_best'
    metadata = dict(cfg, actual_shots=shots, quick_test=quick_test,
                    source_classes=classes, training_protocol='supervised_source_then_support_finetune')
    if hasattr(storage, 'verified_windows'):
        metadata['raw_png_verified_windows'] = storage.verified_windows
    with open(best_path + '.config.json', 'w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, default=str)
    history, best, stale = [], -1., 0
    patience = int(cfg.get('ft_patience', 0))
    print(f'Train {owner.method}: supervised {len(classes)}-class source pretraining; '
          f'{len(y)} samples, {epochs} epochs; validation {owner.ways}-way {shots}-shot')
    for epoch in range(epochs):
        started = time.time()
        model.train()
        total_loss = correct = count = 0
        for batch_index, (bx, by) in enumerate(batches):
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            scores = model(bx)[1]
            loss = torch.nn.functional.cross_entropy(scores, by)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite supervised pretraining loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('outer_grad_clip_norm') or 1.,
                                           error_if_nonfinite=True)
            opt.step()
            total_loss += loss.item() * len(by)
            correct += (scores.argmax(-1) == by).sum().item()
            count += len(by)
            if limit and batch_index + 1 >= limit:
                break
        rng = capture_rng_state()
        try:
            # Repeat the same validation episodes and random local heads each epoch.
            seed = int(cfg.get('ft_validation_seed', 2026))
            np.random.seed(seed)
            torch.manual_seed(seed)
            val_loss, val_acc = owner.evaluate_meta_tasks(validation, owner._algorithm(),
                torch.nn.CrossEntropyLoss(), 0, episodes)
        finally:
            restore_rng_state(rng)
        if val_acc > best:
            best, stale = val_acc, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
        history.append(dict(epoch=epoch + 1, train_loss=total_loss / count,
                            train_acc=correct / count, valid_loss=val_loss, valid_acc=val_acc,
                            epoch_seconds=time.time() - started, best_valid_acc=best))
        print(f'Epoch {epoch+1:03d} | source loss={total_loss/count:.4f} acc={correct/count:.4f} '
              f'| support-FT validation loss={val_loss:.4f} acc={val_acc:.4f} | best={best:.4f}')
        if patience and stale >= patience:
            break
    _save_training_history(history, save_path, cfg)
    return best_path
