"""HUST training/test CLI. Explicit --test never starts training."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from config_hust import BASE, TASKS, resolve_hust_config
from hust_dataset import HUSTMetaDataset, HUSTTasks
from hust_preprocess import preprocess_hust
from l2l_shim import MAML, _MAMLClone, functional_call
from maml_train_pic import (MAML_learner, device, compute_ggm_rw_weights,
                            capture_rng_state, restore_rng_state, freeze_bn_running_stats)
from my_utils.init_utils import seed_torch


class HUSTClone(_MAMLClone):
    def __init__(self, model, lr):
        super().__init__(model, lr)
        self.buffers = {n: b.clone() for n, b in model.named_buffers()}

    def __call__(self, x):
        # Preserve batch-query BN, while preventing state carryover between episodes.
        return functional_call(self._model, {**self.fast_params, **self.buffers}, x)


class HUSTMAML(MAML):
    def clone(self):
        return HUSTClone(self.model, self.lr)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class HUSTMAMLLearner(MAML_learner):
    def __init__(self, cfg):
        super().__init__(cfg['n_way'], cfg)
        self.pu_config['model_name'] = cfg['model_name']
        self.storage = HUSTMetaDataset(cfg)

    def _algorithm(self):
        return HUSTMAML(self.model, self.pu_config['inner_lr'])

    def build_tasks(self, mode='train', ways=5, shots=1, queries=15,
                    num_tasks=1000, filter_labels=None):
        if filter_labels is not None:
            raise ValueError('HUST protocol draws five random classes from all seven')
        cfg = self.pu_config
        seed = cfg['seed'] if mode == 'train' else cfg[f'{"validation" if mode == "validation" else "test"}_episode_seed']
        return HUSTTasks(self.storage, mode, ways, shots, queries, num_tasks, seed,
                         fixed=mode != 'train')

    def train(self, save_path, shots=1, quick_test=False):
        cfg = self.pu_config
        best_path = str(save_path)+'_best'
        if Path(best_path).exists():
            raise FileExistsError(f'Checkpoint already exists; choose a new --model-path: {best_path}')
        Path(best_path).parent.mkdir(parents=True, exist_ok=True)
        algo = self._algorithm()
        opt = torch.optim.Adam(algo.parameters(), lr=cfg['outer_lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg['epochs'], eta_min=cfg['outer_lr_min'])
        loss = torch.nn.CrossEntropyLoss()
        train = self.build_tasks('train', shots=shots)
        val = self.build_tasks('validation', shots=shots,
                               num_tasks=cfg['validation_episodes']+cfg['early_stop_confirmation_episodes'])
        best_acc, history = -1.0, []
        steps = cfg['adaptation_steps'][shots]
        self.model.train()
        for epoch in range(cfg['epochs']):
            started = time.time()
            batch = [train.sample() for _ in range(cfg['meta_batch_size'])]
            weights = torch.full((len(batch),), 1/len(batch), device=device)
            if cfg['task_weighting_mode'] == 'ggm_rw':
                rng = capture_rng_state()
                try:
                    with freeze_bn_running_stats(self.model):
                        gaps = [self.pilot_generalization_gap(t, self.model, loss, steps,
                                                              cfg['inner_lr']) for t in batch]
                finally:
                    restore_rng_state(rng)
                weights, _ = compute_ggm_rw_weights(
                    gaps=gaps, epoch=epoch, batch_size=len(batch),
                    temperature=cfg['task_weight_temperature'], alpha_max=cfg['task_weight_alpha_max'],
                    warmup_epochs=cfg['task_weight_warmup_epochs'], z_clip=cfg['task_weight_z_clip'],
                    max_ratio=cfg['task_weight_max_ratio'], eps=cfg['task_weight_eps'])
            opt.zero_grad()
            train_acc = []
            for weight, task in zip(weights, batch):
                error, acc, _, _ = self.fast_adapt(task, algo.clone(), loss, steps)
                if not torch.isfinite(error):
                    raise FloatingPointError('Nonfinite training loss')
                (weight*error).backward()
                train_acc.append(acc.item())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg['outer_grad_clip_norm'],
                                           error_if_nonfinite=True)
            opt.step()
            # Score exactly the parameters being saved, after the outer update.
            val.cursor = 0
            val_loss, val_acc = self.evaluate_meta_tasks(
                val, algo, loss, steps, cfg['validation_episodes'])
            if not np.isfinite(val_loss):
                raise FloatingPointError('Nonfinite target-validation loss')
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), best_path)
                write_json(best_path+'.config.json', dict(cfg, actual_shots=shots,
                           data_sha256=self.storage.signature, best_epoch=epoch+1,
                           validation_accuracy=best_acc, quick_test=quick_test))
            row = dict(epoch=epoch+1, train_accuracy=float(np.mean(train_acc)),
                       validation_loss=val_loss, validation_accuracy=val_acc,
                       lr=opt.param_groups[0]['lr'], seconds=time.time()-started)
            history.append(row)
            write_json(str(save_path)+'_history.json', history)
            print(f"epoch {epoch+1}/{cfg['epochs']} train={row['train_accuracy']:.4f} "
                  f"target-val={val_acc:.4f} best={best_acc:.4f}", flush=True)
            scheduler.step()
            if cfg['early_stop_on_perfect_validation'] and not quick_test and val_acc >= cfg['early_stop_trigger_accuracy']:
                _, confirmation = self.evaluate_meta_tasks(
                    val, algo, loss, steps, cfg['early_stop_confirmation_episodes'])
                row['confirmation_accuracy'] = confirmation
                write_json(str(save_path)+'_history.json', history)
                if confirmation >= cfg['early_stop_confirmation_accuracy']:
                    break
        return best_path

    def test(self, load_path, inner_steps=10, shots=1, meta_batch_size=None):
        cfg = self.pu_config
        saved = json.loads(Path(str(load_path)+'.config.json').read_text(encoding='utf-8'))
        keys = ('protocol', 'task', 'method', 'n_way', 'k_shot', 'q_query',
                'backbone', 'attention_module', 'task_weighting_mode', 'inner_lr')
        if any(saved[k] != cfg[k] for k in keys) or saved['data_sha256'] != self.storage.signature:
            raise ValueError('Checkpoint task/method/episode/data signature differs from HUST config')
        self.model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
        self.model.train()
        count = meta_batch_size or cfg['test_meta_batch_size']
        tasks = self.build_tasks('test', shots=shots, num_tasks=count)
        algo, loss = self._algorithm(), torch.nn.CrossEntropyLoss()
        accuracies, losses, true, pred, rows = [], [], [], [], []
        started = time.time()
        for i in range(count):
            task = tasks.sample_with_original_labels()
            error, acc, _, _, logits = self.fast_adapt(task[:4], algo.clone(), loss,
                                                       inner_steps, return_predictions=True)
            if not torch.isfinite(error):
                raise FloatingPointError('Nonfinite HUST test loss')
            predicted = task[6][logits.detach().argmax(1).cpu()].tolist()
            actual = task[5].tolist()
            true.extend(actual); pred.extend(predicted)
            accuracies.append(acc.item()); losses.append(error.item())
            ids = [image for group in tasks.plans[i]['selections'] for image in group['query_ids']]
            rows.extend(dict(episode=i, image=image, true_label=a, predicted_label=p)
                        for image, a, p in zip(ids, actual, predicted))
        result = dict(task=cfg['task'], method=cfg['method'], shots=shots,
                      seed=saved['seed'], episodes=count, query_per_class=cfg['q_query'],
                      accuracy=float(np.mean(accuracies)), loss=float(np.mean(losses)),
                      macro_f1=float(f1_score(true, pred, labels=list(range(7)), average='macro', zero_division=0)),
                      confusion_matrix=confusion_matrix(true, pred, labels=list(range(7))).tolist(),
                      class_names=cfg['class_names'], episode_accuracies=accuracies,
                      checkpoint=str(load_path), episode_manifest=str(tasks.path),
                      data_sha256=self.storage.signature, seconds=time.time()-started,
                      quick_checkpoint=saved.get('quick_test', False))
        output = Path(str(load_path)+f'_test_{count}eps')
        write_json(str(output)+'.json', result)
        with open(str(output)+'.csv', 'w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=['episode', 'image', 'true_label', 'predicted_label'])
            writer.writeheader(); writer.writerows(rows)
        print(f"HUST test: {count} episodes, accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=[*TASKS, 'all'], default='HUST-T1')
    parser.add_argument('--method', choices=['configured_maml', 'maml'], default='configured_maml')
    parser.add_argument('--shots', type=int, choices=[1, 5], default=1)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--raw-root')
    parser.add_argument('--processed-root')
    parser.add_argument('--model-path', '--model_path', dest='model_path')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--quick', action='store_true', help='1 outer update, 1 train/val episode, 2 test episodes; isolated checkpoint')
    parser.add_argument('--test-episodes', type=int)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.task == 'all' and args.model_path:
        parser.error('--model-path requires one task')
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if not (args.preprocess or args.train or args.test):
        args.train = args.test = True
    for task in TASKS if args.task == 'all' else [args.task]:
        overrides = dict(task=task, method=args.method, k_shot=args.shots, seed=args.seed)
        if args.raw_root:
            overrides['raw_data_root'] = args.raw_root
        if args.processed_root:
            overrides['processed_root'] = args.processed_root
        if args.quick:
            overrides.update(epochs=1, meta_batch_size=1, validation_episodes=1, test_meta_batch_size=2)
        if args.test_episodes is not None:
            overrides['test_meta_batch_size'] = args.test_episodes
        cfg = resolve_hust_config(overrides)
        seed_torch(cfg['seed'])
        name = cfg['model_name'] + ('_quick' if args.quick else '')
        path = args.model_path or str(BASE / 'model_save' / 'hust' / name)
        print(f"{task}: {cfg['source_condition']} W -> {cfg['target_condition']} W; "
              f"7-class pool, 5-way {args.shots}-shot; device={device}; "
              f"preprocess={args.preprocess}, train={args.train}, test={args.test}", flush=True)
        if args.preprocess:
            preprocess_hust(cfg)
        if args.train or args.test:
            learner = HUSTMAMLLearner(cfg)
            best = learner.train(path, shots=args.shots, quick_test=args.quick) if args.train else path+'_best'
            if args.test:
                learner.test(best, inner_steps=cfg['test_inner_steps'], shots=args.shots)


if __name__ == '__main__':
    main()
