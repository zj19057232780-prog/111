from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

from config_pu import PU_CONFIG, QUICK_TEST
from l2l_shim import MAML as MAMLAlgo
from l2l_shim import MetaDataset, TaskDataset
from maml_model import Net4CNN, Net4LSK
from my_utils.init_utils import seed_torch
from my_utils.train_utils import accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _artifact_name(path):
    name = os.path.basename(os.path.normpath(path))
    if name.endswith('_best'):
        name = name[:-5]
    return name or 'STFT_CNN4_MAML'


def _visual_dir(cfg, artifact_path=None):
    out_dir = cfg.get('visualization_dir', './results')
    if artifact_path:
        out_dir = os.path.join(out_dir, _artifact_name(artifact_path))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _save_training_history(history, save_path, cfg):
    if not history or not cfg.get('save_visualizations', True):
        return

    out_dir = _visual_dir(cfg, save_path)
    name = _artifact_name(save_path)
    csv_path = os.path.join(out_dir, f'{name}_training_history.csv')
    curve_path = os.path.join(out_dir, f'{name}_training_curves.png')

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = [row['epoch'] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)

        axes[0].plot(epochs, [row['train_loss'] for row in history], label='train')
        axes[0].plot(epochs, [row['valid_loss'] for row in history], label='valid')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('MAML Loss')
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].plot(epochs, [row['train_acc'] for row in history], label='train')
        axes[1].plot(epochs, [row['valid_acc'] for row in history], label='valid')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1)
        axes[1].set_title('MAML Accuracy')
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(curve_path)
        plt.close(fig)
        print(f'Saved training history: {csv_path}')
        print(f'Saved training curves: {curve_path}')
    except Exception as exc:
        print(f'[warn] Failed to save training curve PNG: {exc}')
        print(f'Saved training history: {csv_path}')


def _save_prediction_rows(rows, out_dir, name):
    csv_path = os.path.join(out_dir, f'{name}_test_predictions.csv')
    if not rows:
        return csv_path
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _save_confusion_matrix(y_true, y_pred, class_names, out_dir, name):
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_i, pred_i in zip(y_true, y_pred):
        if 0 <= true_i < n_classes and 0 <= pred_i < n_classes:
            cm[true_i, pred_i] += 1

    csv_path = os.path.join(out_dir, f'{name}_confusion_matrix_counts.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['true\\pred'] + class_names)
        for idx, row in enumerate(cm):
            writer.writerow([class_names[idx]] + row.tolist())

    png_path = os.path.join(out_dir, f'{name}_confusion_matrix.png')
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, np.maximum(row_sum, 1), where=row_sum != 0)

        fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Recall-normalized count')
        ax.set_title('Meta-test Confusion Matrix')
        ax.set_xlabel('Predicted fault class')
        ax.set_ylabel('True fault class')
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
    except Exception as exc:
        print(f'[warn] Failed to save confusion matrix PNG: {exc}')

    return csv_path, png_path


def _save_tsne(features, labels, class_names, out_dir, name, cfg):
    if features.size == 0:
        return None

    selected_classes = cfg.get('tsne_selected_classes')
    if selected_classes:
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        selected_indices = [
            class_to_idx[cls_name]
            for cls_name in selected_classes
            if cls_name in class_to_idx
        ]
        missing = [cls_name for cls_name in selected_classes if cls_name not in class_to_idx]
        if missing:
            print(f'[warn] t-SNE selected classes not found: {missing}')
        if not selected_indices:
            print('[warn] No valid selected classes for t-SNE; skip t-SNE.')
            return None

        selected_set = set(selected_indices)
        mask = np.array([label in selected_set for label in labels], dtype=bool)
        features = features[mask]
        labels = labels[mask]
        class_names = [class_names[idx] for idx in selected_indices]
        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        labels = np.array([remap[int(label)] for label in labels], dtype=np.int64)

        if features.size == 0:
            print('[warn] No query features matched selected t-SNE classes.')
            return None

    max_points = int(cfg.get('tsne_max_points', 1000))
    if len(features) > max_points:
        rng = np.random.default_rng(cfg.get('seed', 24))
        indices = rng.choice(len(features), size=max_points, replace=False)
        features = features[indices]
        labels = labels[indices]

    if len(features) < 4:
        print('[warn] Not enough samples for t-SNE visualization.')
        return None

    png_path = os.path.join(out_dir, f'{name}_tsne.png')
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))
        from sklearn.manifold import TSNE

        perplexity = min(float(cfg.get('tsne_perplexity', 30)), max(2.0, (len(features) - 1) / 3.0))
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto',
            random_state=cfg.get('seed', 24),
        ).fit_transform(features)

        fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
        cmap = plt.get_cmap('tab10')
        for cls_idx, cls_name in enumerate(class_names):
            mask = labels == cls_idx
            if not np.any(mask):
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=12,
                alpha=0.75,
                color=cmap(cls_idx % 10),
                label=cls_name,
                linewidths=0,
            )
        title_suffix = 'Selected Classes' if selected_classes else 'All Classes'
        ax.set_title(f't-SNE of Meta-test Query Features ({title_suffix})')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(alpha=0.18)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=8, fontsize=5, frameon=False)
        fig.tight_layout()
        fig.savefig(png_path, bbox_inches='tight')
        plt.close(fig)
        return png_path
    except Exception as exc:
        print(f'[warn] Failed to save t-SNE PNG: {exc}')
        return None


def _save_test_visualizations(
    features,
    y_true,
    y_pred,
    rows,
    class_names,
    load_path,
    cfg,
    tsne_payload=None,
):
    if not cfg.get('save_visualizations', True):
        return

    out_dir = _visual_dir(cfg, load_path)
    name = _artifact_name(load_path)
    prediction_csv = _save_prediction_rows(rows, out_dir, name)
    cm_csv, cm_png = _save_confusion_matrix(y_true, y_pred, class_names, out_dir, name)
    if tsne_payload is not None:
        tsne_features, tsne_labels, tsne_class_names = tsne_payload
        tsne_png = _save_tsne(tsne_features, tsne_labels, tsne_class_names, out_dir, name, cfg)
    else:
        tsne_png = _save_tsne(features, y_true, class_names, out_dir, name, cfg)

    print(f'Saved test predictions: {prediction_csv}')
    print(f'Saved confusion matrix CSV: {cm_csv}')
    print(f'Saved confusion matrix PNG: {cm_png}')
    if tsne_png:
        print(f'Saved t-SNE PNG: {tsne_png}')


class MAML_learner(object):
    def __init__(self, ways, pu_config=None):
        cfg = pu_config or {}
        hidden_size = cfg.get('hidden_size', 64)
        layers = 4
        img_size = cfg.get('img_size', 64)
        in_channels = cfg.get('in_channels', 1)
        backbone = cfg.get('backbone', 'cnn4').lower()

        if backbone in ('lsk', 'lsk_lite', 'lsklite'):
            self.model = Net4LSK(
                output_size=ways,
                channels=in_channels,
                stage_channels=cfg.get('lsk_stage_channels', (32, 64, 96)),
                stage_depths=cfg.get('lsk_stage_depths', (1, 1, 1)),
                mlp_ratio=cfg.get('lsk_mlp_ratio', 2),
                img_size=img_size,
                frequency_module=cfg.get('frequency_module', 'none'),
                gfnet_depth=cfg.get('gfnet_depth', 1),
                gfnet_mlp_ratio=cfg.get('gfnet_mlp_ratio', 2),
                gfnet_weight_scale=cfg.get('gfnet_weight_scale', 0.02),
                gfnet_layer_scale_init=cfg.get('gfnet_layer_scale_init', 1e-2),
                denoise_module=cfg.get('denoise_module', 'none'),
                arsm_reduction=cfg.get('arsm_reduction', 4),
                arsm_blend_init=cfg.get('arsm_blend_init', 1e-3),
                attention_module=cfg.get('attention_module', 'none'),
                ema_factor=cfg.get('ema_factor', 8),
                ema_layer_scale_init=cfg.get('ema_layer_scale_init', 1e-3),
                gcnet_ratio=cfg.get('gcnet_ratio', 0.25),
                gcnet_pooling_type=cfg.get('gcnet_pooling_type', 'att'),
                gcnet_fusion_types=cfg.get('gcnet_fusion_types', ('channel_add',)),
                gcnet_layer_scale_init=cfg.get('gcnet_layer_scale_init', 1e-4),
            ).to(device)
            self.frequency_module = getattr(self.model, 'frequency_module', 'none')
            self.denoise_module = getattr(self.model, 'denoise_module', 'none')
            self.attention_module = getattr(self.model, 'attention_module', 'none')
            self.backbone_name = 'lsk_lite'
            if self.frequency_module != 'none':
                self.backbone_name = f'{self.backbone_name}_{self.frequency_module}'
            if self.denoise_module != 'none':
                self.backbone_name = f'{self.backbone_name}_{self.denoise_module}'
            if self.attention_module != 'none':
                self.backbone_name = f'{self.backbone_name}_{self.attention_module}'
        elif backbone == 'cnn4':
            feat_size = hidden_size * (img_size // (2 ** layers)) ** 2
            self.model = Net4CNN(
                output_size=ways,
                hidden_size=hidden_size,
                layers=layers,
                channels=in_channels,
                embedding_size=feat_size,
            ).to(device)
            self.backbone_name = 'cnn4'
            self.frequency_module = 'none'
            self.denoise_module = 'none'
            self.attention_module = 'none'
        else:
            raise ValueError(f'Unknown backbone: {backbone}')

        self.ways = ways
        self.pu_config = pu_config or PU_CONFIG
        self._pu_storage = None

    def _get_pu_storage(self):
        if self._pu_storage is None:
            from pu_dataset import PUMetaDataset

            cfg = self.pu_config
            self._pu_storage = PUMetaDataset(
                root_path=cfg['root_path'],
                source_condition=cfg['source_condition'],
                target_condition=cfg['target_condition'],
                img_size=cfg.get('img_size', 64),
                target_val_ratio=cfg.get('target_val_ratio', 0.5),
                split_seed=cfg.get('seed', 24),
                class_groups=cfg.get('class_groups'),
                classification_name=cfg.get('classification_preset'),
            )
        return self._pu_storage

    def build_tasks(self, mode='train', ways=5, shots=1, queries=1,
                    num_tasks=100, filter_labels=None):
        from pu_dataset import PUMetaDataset

        cfg = self.pu_config
        storage = self._get_pu_storage()
        ds = PUMetaDataset(
            root_path=cfg['root_path'],
            source_condition=cfg['source_condition'],
            target_condition=cfg['target_condition'],
            img_size=cfg.get('img_size', 64),
            target_val_ratio=cfg.get('target_val_ratio', 0.5),
            split_seed=cfg.get('seed', 24),
            class_groups=cfg.get('class_groups'),
            classification_name=cfg.get('classification_preset'),
            share_storage=storage,
        )
        ds.set_mode(mode)
        dataset = MetaDataset(ds)
        new_ways = len(filter_labels) if filter_labels is not None else ways
        return TaskDataset(
            dataset,
            n_way=new_ways,
            k_shot=shots,
            q_query=queries,
            num_tasks=num_tasks,
            filter_labels=filter_labels,
        )

    def build_fixed_selected_class_task(self, shots=1):
        cfg = self.pu_config
        selected_classes = cfg.get('tsne_selected_classes')
        if not selected_classes:
            return None

        storage = self._get_pu_storage()
        rng = np.random.default_rng(cfg.get('seed', 24))
        support_x, support_y = [], []
        query_x, query_y = [], []
        used_classes = []

        for cls_name in selected_classes:
            cls_data = storage.target_test_data.get(cls_name)
            if cls_data is None:
                print(f'[warn] fixed t-SNE class not found in target test data: {cls_name}')
                continue
            if len(cls_data) <= shots:
                print(f'[warn] fixed t-SNE class has too few samples: {cls_name}')
                continue

            new_label = len(used_classes)
            used_classes.append(cls_name)
            indices = np.arange(len(cls_data))
            rng.shuffle(indices)
            support_indices = indices[:shots]
            query_indices = indices[shots:]

            for idx in support_indices:
                support_x.append(torch.from_numpy(np.ascontiguousarray(cls_data[idx])).float())
                support_y.append(new_label)
            for idx in query_indices:
                query_x.append(torch.from_numpy(np.ascontiguousarray(cls_data[idx])).float())
                query_y.append(new_label)

        if len(used_classes) < 2 or not query_x:
            print('[warn] fixed selected-class t-SNE task is too small; skip fixed t-SNE.')
            return None

        return (
            torch.stack(support_x),
            torch.tensor(support_y, dtype=torch.long),
            torch.stack(query_x),
            torch.tensor(query_y, dtype=torch.long),
            used_classes,
        )

    def fixed_selected_class_tsne_payload(self, inner_steps=10, shots=1):
        fixed_task = self.build_fixed_selected_class_task(shots=shots)
        if fixed_task is None:
            return None

        support_x, support_y, query_x, query_y, used_classes = fixed_task
        task = (support_x, support_y, query_x, query_y)
        cfg = self.pu_config
        maml = MAMLAlgo(self.model, lr=cfg.get('inner_lr', 0.05))
        loss = torch.nn.CrossEntropyLoss(reduction='mean')
        learner = maml.clone()
        _, fixed_acc, features, labels, _ = self.fast_adapt(
            task,
            learner,
            loss,
            inner_steps,
            return_predictions=True,
        )
        print(
            f'Fixed selected-class t-SNE task: classes={used_classes}, '
            f'query_samples={len(query_y)}, acc={fixed_acc.item():.4f}'
        )
        return (
            features.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
            used_classes,
        )

    @staticmethod
    def fast_adapt(batch, learner, loss, adaptation_steps, return_predictions=False):
        support_data, support_labels, query_data, query_labels = batch
        support_data = support_data.to(device)
        support_labels = support_labels.to(device)
        query_data = query_data.to(device)
        query_labels = query_labels.to(device)

        for _ in range(adaptation_steps):
            train_error = loss(learner(support_data)[1], support_labels)
            learner.adapt(train_error)

        features, predictions = learner(query_data)
        valid_error = loss(predictions, query_labels)
        valid_accuracy = accuracy(predictions, query_labels)
        if return_predictions:
            return valid_error, valid_accuracy, features, query_labels, predictions
        return valid_error, valid_accuracy, features, query_labels

    def train(self, save_path, shots=1, quick_test=False):
        cfg = self.pu_config
        meta_lr = cfg.get('outer_lr', 0.005)
        fast_lr = cfg.get('inner_lr', 0.05)
        queries = cfg.get('q_query', 1)
        maml = MAMLAlgo(self.model, lr=fast_lr)
        opt = torch.optim.Adam(maml.parameters(), meta_lr)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        train_ways = valid_ways = self.ways
        num_tasks = 20 if quick_test else cfg.get('tasks_per_epoch', 1000)
        epochs = 3 if quick_test else cfg.get('epochs', 250)
        meta_batch_size = 4 if quick_test else cfg.get('meta_batch_size', 16)
        adaptation_steps = cfg.get('adaptation_steps', {}).get(shots, 1)
        grad_clip_norm = cfg.get('outer_grad_clip_norm')
        scheduler_name = str(cfg.get('outer_lr_scheduler', 'none')).lower()
        if scheduler_name in ('none', 'off', 'false'):
            scheduler = None
        elif scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=max(epochs, 1),
                eta_min=cfg.get('outer_lr_min', 0.0),
            )
        else:
            raise ValueError(f'Unknown outer_lr_scheduler: {scheduler_name}')

        print(
            f'Train STFT-{self.backbone_name}-MAML: {train_ways}-way {shots}-shot '
            f'{queries}-query, epochs={epochs}, meta_batch={meta_batch_size}, '
            f'outer_lr={meta_lr}, scheduler={scheduler_name}, grad_clip={grad_clip_norm}'
        )

        train_tasks = self.build_tasks('train', train_ways, shots, queries, num_tasks)
        valid_tasks = self.build_tasks('validation', valid_ways, shots, queries, num_tasks)

        best_valid_acc = -1.0
        best_epoch = 0
        best_path = save_path + '_best'
        history = []

        for ep in range(epochs):
            t0 = time.time()
            current_lr = opt.param_groups[0]['lr']
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            opt.zero_grad()
            for _ in range(meta_batch_size):
                learner = maml.clone()
                task = train_tasks.sample()
                evaluation_error, evaluation_accuracy, _, _ = self.fast_adapt(
                    task, learner, loss, adaptation_steps
                )
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                learner = maml.clone()
                task = valid_tasks.sample()
                evaluation_error, evaluation_accuracy, _, _ = self.fast_adapt(
                    task, learner, loss, adaptation_steps
                )
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            for p in maml.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / meta_batch_size)
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    maml.parameters(),
                    max_norm=float(grad_clip_norm),
                    error_if_nonfinite=False,
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    maml.parameters(),
                    max_norm=float('inf'),
                    error_if_nonfinite=False,
                )

            grad_norm_value = float(grad_norm.detach().cpu())
            if np.isfinite(grad_norm_value):
                opt.step()
            else:
                print(
                    f'[warn] Epoch {ep + 1}: non-finite outer gradient; '
                    'optimizer step skipped.'
                )
            if scheduler is not None:
                scheduler.step()

            avg_train_acc = meta_train_accuracy / meta_batch_size
            avg_valid_acc = meta_valid_accuracy / meta_batch_size
            avg_train_err = meta_train_error / meta_batch_size
            avg_valid_err = meta_valid_error / meta_batch_size
            epoch_seconds = time.time() - t0

            history.append({
                'epoch': ep + 1,
                'train_loss': avg_train_err,
                'train_acc': avg_train_acc,
                'valid_loss': avg_valid_err,
                'valid_acc': avg_valid_acc,
                'epoch_seconds': epoch_seconds,
                'outer_lr': current_lr,
                'outer_grad_norm': grad_norm_value,
                'best_valid_acc': max(best_valid_acc, avg_valid_acc),
                'best_epoch': ep + 1 if avg_valid_acc > best_valid_acc else best_epoch,
            })

            print(
                f'Epoch {ep + 1:03d} | '
                f'time={epoch_seconds:.2f}s | '
                f'train_loss={avg_train_err:.4f} acc={avg_train_acc:.4f} | '
                f'valid_loss={avg_valid_err:.4f} acc={avg_valid_acc:.4f} | '
                f'lr={current_lr:.6f} grad={grad_norm_value:.3f} | '
                f'best={best_valid_acc:.4f}@{best_epoch}'
            )

            if avg_valid_acc > best_valid_acc:
                best_valid_acc = avg_valid_acc
                best_epoch = ep + 1
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), best_path)
                print(f'  saved best model: {best_path}')

        _save_training_history(history, save_path, cfg)
        return best_path

    def test(self, load_path, inner_steps=10, shots=1, meta_batch_size=None):
        try:
            state = torch.load(load_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(load_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)

        cfg = self.pu_config
        queries = cfg.get('q_query', 1)
        fast_lr = cfg.get('inner_lr', 0.05)
        test_tasks = self.build_tasks('test', self.ways, shots, queries, 1000)
        class_names = list(self._get_pu_storage().source_classes)
        maml = MAMLAlgo(self.model, lr=fast_lr)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        meta_batch_size = meta_batch_size or cfg.get('test_meta_batch_size', 100)
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        all_features = []
        all_true = []
        all_pred = []
        prediction_rows = []
        t0 = time.time()

        for episode_idx in range(meta_batch_size):
            learner = maml.clone()
            if hasattr(test_tasks, 'sample_with_original_labels'):
                full_task = test_tasks.sample_with_original_labels()
                task = full_task[:4]
                query_orig_labels = full_task[5].cpu().numpy()
                chosen_orig_labels = full_task[6].cpu().numpy()
            else:
                task = test_tasks.sample()
                query_orig_labels = None
                chosen_orig_labels = None

            evaluation_error, evaluation_accuracy, features, query_labels, predictions = self.fast_adapt(
                task, learner, loss, inner_steps, return_predictions=True
            )
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

            pred_local = predictions.detach().argmax(dim=1).cpu().numpy()
            true_local = query_labels.detach().cpu().numpy()

            if chosen_orig_labels is not None and query_orig_labels is not None:
                pred_orig = chosen_orig_labels[pred_local]
                true_orig = query_orig_labels
            else:
                pred_orig = pred_local
                true_orig = true_local

            all_features.append(features.detach().cpu().numpy())
            all_true.append(true_orig.astype(np.int64))
            all_pred.append(pred_orig.astype(np.int64))

            for sample_idx, (tl, pl, to, po) in enumerate(zip(true_local, pred_local, true_orig, pred_orig)):
                prediction_rows.append({
                    'episode': episode_idx,
                    'sample': sample_idx,
                    'true_local_label': int(tl),
                    'pred_local_label': int(pl),
                    'true_class_index': int(to),
                    'pred_class_index': int(po),
                    'true_class_name': class_names[int(to)] if int(to) < len(class_names) else str(to),
                    'pred_class_name': class_names[int(po)] if int(po) < len(class_names) else str(po),
                })

        print(
            f'Test {self.ways}-way {shots}-shot {queries}-query | '
            f'tasks={meta_batch_size} | time={time.time() - t0:.2f}s'
        )
        print(f'Meta Test Error: {meta_test_error / meta_batch_size:.4f}')
        print(f'Meta Test Accuracy: {meta_test_accuracy / meta_batch_size:.4f}')

        if all_features:
            tsne_payload = self.fixed_selected_class_tsne_payload(
                inner_steps=inner_steps,
                shots=shots,
            )
            _save_test_visualizations(
                features=np.concatenate(all_features, axis=0),
                y_true=np.concatenate(all_true, axis=0),
                y_pred=np.concatenate(all_pred, axis=0),
                rows=prediction_rows,
                class_names=class_names,
                load_path=load_path,
                cfg=cfg,
                tsne_payload=tsne_payload,
            )


def main():
    parser = argparse.ArgumentParser(description='Train/test STFT-CNN4-MAML baseline.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--quick', action='store_true')
    default_model_name = PU_CONFIG.get('model_name', 'STFT_CNN4_MAML')
    parser.add_argument('--model_path', default=os.path.join('.', 'model_save', default_model_name))
    args = parser.parse_args()

    seed_torch(PU_CONFIG.get('seed', 24))
    print(f'Device: {device}')
    print(
        f'Input shape: [batch, {PU_CONFIG["in_channels"]}, '
        f'{PU_CONFIG["img_size"]}, {PU_CONFIG["img_size"]}]'
    )
    print(f'Backbone: {PU_CONFIG.get("backbone", "cnn4")}')
    print(f'Frequency module: {PU_CONFIG.get("frequency_module", "none")}')
    print(f'Denoise module: {PU_CONFIG.get("denoise_module", "none")}')
    print(f'Attention module: {PU_CONFIG.get("attention_module", "none")}')
    print(
        f'Classification preset: {PU_CONFIG.get("classification_preset")} | '
        f'pool={len(PU_CONFIG.get("class_groups", {}))} classes'
    )

    net = MAML_learner(ways=PU_CONFIG['n_way'], pu_config=PU_CONFIG)
    shots = PU_CONFIG['k_shot']
    quick = args.quick or QUICK_TEST

    if args.train:
        best_path = net.train(args.model_path, shots=shots, quick_test=quick)
    else:
        best_path = args.model_path + '_best'

    if args.test:
        if not os.path.exists(best_path):
            raise FileNotFoundError(f'Model not found: {best_path}')
        net.test(
            best_path,
            inner_steps=PU_CONFIG.get('test_inner_steps', 10),
            shots=shots,
            meta_batch_size=4 if quick else None,
        )

    if not args.train and not args.test:
        print('Nothing to do. Use --train and/or --test.')


if __name__ == '__main__':
    main()
