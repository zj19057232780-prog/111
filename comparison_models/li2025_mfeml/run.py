"""MFEML under project PU/CWRU tasks: raw2000 adaptive square or existing STFT64."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import random
import statistics
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F

from mfeml_model import MFEMLNetwork, MethodConfig, adapt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent / 'IFMAML-master'
sys.path.insert(0, str(PROJECT))
from l2l_shim import MetaDataset, TaskDataset
from run_table_row import TARGETS, SEEDS, configure
from raw_input import IMAGE_SIZE, SAMPLE_POINTS, REPRESENTATION, PaperRawDataset


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def code_hashes() -> dict[str, str]:
    paths = [HERE / name for name in ('run.py', 'mfeml_model.py', 'raw_input.py')] + [PROJECT / name for name in (
        'pu_dataset.py', 'cwru_dataset.py', 'config_pu.py', 'config_cwru.py',
        'classic_config.py', 'l2l_shim.py', 'run_table_row.py',
        'raw_signal_dataset.py', 'pu_loader.py', 'pu_preprocess.py', 'cwru_preprocess.py')]
    return {str(p.relative_to(HERE.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class RecordedMetaDataset(MetaDataset):
    def __init__(self, dataset: list) -> None:
        self.record = False
        self.picked: list[int] = []
        super().__init__(dataset)
        self.record = True

    def __getitem__(self, index: int):
        if self.record:
            self.picked.append(int(index))
        return super().__getitem__(index)


class ProjectData:
    def __init__(self, cfg: dict, output: Path | None = None) -> None:
        if cfg['input_representation'] == REPRESENTATION:
            storage = PaperRawDataset(cfg)
            if output:
                save_json(output / 'raw_window_manifest.json', storage.window_manifest)
        elif cfg['input_representation'] == 'stft':
            if cfg['dataset'] == 'pu':
                from pu_dataset import PUMetaDataset
                storage = PUMetaDataset(
                    cfg['root_path'], cfg['source_condition'], cfg['target_condition'],
                    img_size=cfg['img_size'], target_val_ratio=cfg['target_val_ratio'],
                    split_seed=cfg['seed'], class_groups=cfg['class_groups'],
                    classification_name=cfg['classification_preset'])
            else:
                from cwru_dataset import CWRUMetaDataset
                storage = CWRUMetaDataset(
                    cfg['root_path'], cfg['source_condition'], cfg['target_condition'],
                    cfg['class_names'], img_size=cfg['img_size'])
        else:
            raise ValueError('Unknown input representation')
        self.names = storage.source_classes
        self.cfg = cfg
        self.tasks = {}
        self.signature = {}
        for split in ('train', 'validation', 'test'):
            x, y = storage.to_maml_dataset_format(split)
            self.signature[split] = {
                'shape': list(x.shape), 'labels': self.names,
                'counts': {self.names[int(c)]: int((y == c).sum()) for c in np.unique(y)},
                'sha256': hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
            }
            meta = RecordedMetaDataset(list(zip(x, y)))
            self.tasks[split] = TaskDataset(meta, cfg['n_way'], cfg['k_shot'], cfg['q_query'])
        self.stream = (output / 'episodes.jsonl').open('w', encoding='utf-8') if output else None
        self.counter = {key: 0 for key in self.tasks}

    def sample(self, split: str) -> tuple:
        task = self.tasks[split]
        task.meta.picked.clear()
        full = task.sample_with_original_labels()
        if self.stream:
            self.stream.write(json.dumps({
                'split': split, 'index': self.counter[split],
                'classes': full[6].tolist(), 'indices': task.meta.picked,
            }) + '\n')
        self.counter[split] += 1
        return full

    def close(self) -> None:
        if self.stream:
            self.stream.close()


def episode(model: MFEMLNetwork, batch: tuple, cfg: dict, method: MethodConfig,
            device: torch.device, training: bool, steps: int) -> tuple:
    sx, sy, qx, qy = (t.to(device) for t in batch[:4])
    params = adapt(model, sx, sy, steps, cfg['inner_lr'], meta_grad=training)
    with torch.set_grad_enabled(training):
        _, logits = functional_call(model, params, (qx,))
        loss = F.cross_entropy(logits, qy)
    return loss, (logits.argmax(1) == qy).float().mean(), logits.detach().argmax(1).cpu(), {}



def evaluate(model: MFEMLNetwork, data: ProjectData, cfg: dict, method: MethodConfig,
             device: torch.device, split: str, count: int, steps: int,
             output: Path | None = None) -> dict:
    correct = 0
    total = 0
    losses = []
    matrix = np.zeros((len(data.names), len(data.names)), dtype=np.int64)
    stream = (output / 'test_predictions.csv').open('w', newline='', encoding='utf-8-sig') if output else None
    try:
        writer = csv.writer(stream) if stream else None
        if writer:
            writer.writerow(['episode', 'sample', 'true_local', 'pred_local', 'true_global', 'pred_global'])
        for i in range(count):
            batch = data.sample(split)
            loss, _, pred, _ = episode(model, batch, cfg, method, device, False, steps)
            true = batch[3]
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite {split} loss at episode {i}')
            losses.append(float(loss))
            correct += int((true == pred).sum())
            total += len(true)
            orig_true, orig_pred = batch[5].numpy(), batch[6][pred].numpy()
            np.add.at(matrix, (orig_true, orig_pred), 1)
            if writer:
                writer.writerows((i, j, int(t), int(p), int(ot), int(op))
                                 for j, (t, p, ot, op) in enumerate(zip(true, pred, orig_true, orig_pred)))
    finally:
        if stream:
            stream.close()
    tp = matrix.diagonal().astype(float)
    denom = matrix.sum(0) + matrix.sum(1)
    macro_f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom != 0).mean()
    metrics = dict(accuracy_pct=100 * correct / total, macro_f1_pct=100 * macro_f1,
                   loss=statistics.mean(losses), episodes=count, query_count=total,
                   confusion_matrix=matrix.tolist())
    if split == 'test':
        label = 'SMOKE TEST' if cfg.get('smoke', False) else 'TEST'
        print(f"[{label}] MFEML dataset={cfg['dataset']} task={cfg['task']} "
              f"seed={cfg['seed']} shot={cfg['k_shot']} input={cfg['input_tag']} "
              f"accuracy={metrics['accuracy_pct']:.2f}% "
              f"macro_f1={metrics['macro_f1_pct']:.2f}% "
              f"episodes={count} queries={total}", flush=True)
        if output is not None:
            print(f"[{label}] Output: {output.resolve()}", flush=True)
    return metrics


def train_one(cfg: dict, method: MethodConfig, device: torch.device, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'resolved_config.json', cfg)
    save_json(out / 'method_config.json', asdict(method))
    hashes = code_hashes()
    save_json(out / 'code_sha256.json', hashes)
    for relative in hashes:
        destination = out / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((HERE.parent.parent / relative).read_bytes())
    seed_all(cfg['seed'])
    data = ProjectData(cfg, out)
    save_json(out / 'data_signature.json', data.signature)
    model = MFEMLNetwork(cfg['n_way'], cfg['img_size'], method).to(device)
    save_json(out / 'architecture.json', {
        'network': str(model), 'feature_dimension': 256,
        'input_channels': 1, 'attention_fusion': method.attention_fusion,
        'bn_mode': 'episode_batch_statistics_no_running_buffers',
        'square_padding': 'append_zeros_to_ceil_sqrt_squared',
        'stft_head_adapter': 'adaptive_average_pool_4x4_to_2x2',
    })
    opt = torch.optim.Adam(model.parameters(), lr=cfg['outer_lr'])
    best, best_epoch = -1., 0
    history = []
    stop_reason = 'epochs_limit'
    try:
        for epoch in range(1, cfg['epochs'] + 1):
            start = perf_counter()
            opt.zero_grad(set_to_none=True)
            train_losses, train_accs = [], []
            for _ in range(cfg['meta_batch_size']):
                batch = data.sample('train')
                loss, acc, _, _ = episode(model, batch, cfg, method, device, True, cfg['train_steps'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite meta-training loss')
                (loss / cfg['meta_batch_size']).backward()
                train_losses.append(float(loss.detach()))
                train_accs.append(float(acc))
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.get('outer_grad_clip_norm') or float('inf'), error_if_nonfinite=True)
            opt.step()
            # Validate the exact parameters that will be saved (fixes legacy one-step lag).
            valid = evaluate(model, data, cfg, method, device, 'validation',
                             cfg['validation_episodes'], cfg['test_inner_steps'])
            improved = valid['accuracy_pct'] > best
            if improved:
                best, best_epoch = valid['accuracy_pct'], epoch
                torch.save(model.state_dict(), out / 'best.pt')
            row = dict(epoch=epoch, train_loss=statistics.mean(train_losses),
                       train_acc=statistics.mean(train_accs), validation_accuracy_pct=valid['accuracy_pct'],
                       grad_norm=float(grad_norm),
                       outer_lr=opt.param_groups[0]['lr'], best_epoch=best_epoch,
                       seconds=perf_counter() - start)
            history.append(row)
            with (out / 'history.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader(); writer.writerows(history)
            print(f"{cfg['task']} seed={cfg['seed']} epoch={epoch} "
                  f"train={row['train_acc']:.4f} val={valid['accuracy_pct']:.2f}% "
                  f"time={row['seconds']:.1f}s", flush=True)
            if cfg['early_stop_patience'] and epoch - best_epoch >= cfg['early_stop_patience']:
                stop_reason = 'validation_patience'
                break
        model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
        seed_all(100000 + cfg['seed'])
        result = evaluate(model, data, cfg, method, device, 'test',
                          cfg['test_meta_batch_size'], cfg['test_inner_steps'], out)
        result.update(status='SMOKE_ONLY' if cfg['smoke'] else 'OK', dataset=cfg['dataset'],
                      task=cfg['task'], seed=cfg['seed'], shot=cfg['k_shot'],
                      method=f"MFEML-{cfg['input_tag']}-project-reconstruction", best_epoch=best_epoch,
                      input_representation=cfg['input_representation'], input_mode=cfg['input_mode'],
                      epochs_run=len(history), stop_reason=stop_reason,
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      checkpoint_sha256=hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest(),
                      runtime_device=str(device), torch_version=torch.__version__)
        save_json(out / 'result.json', result)
        return result
    finally:
        data.close()


def configuration(args: argparse.Namespace, task: str, seed: int) -> dict:
    mod = importlib.import_module('config_' + args.dataset)
    base = copy.deepcopy(getattr(mod, args.dataset.upper() + '_CONFIG'))
    target = dict(TARGETS[args.dataset])[task]
    cfg = configure(base, 'cnn4maml', target, args.dataset, args.shot, seed, args.epochs)
    is_raw = args.input in ('no_stft', 'nostft')
    if args.data_map and is_raw:
        raise ValueError('--data-map applies only to --input stft')
    if args.data_root and args.data_map:
        raise ValueError('Use either --data-root or --data-map')
    root_key = ('mat_root' if args.dataset == 'pu' else 'raw_data_root') if is_raw else 'root_path'
    root = Path(args.data_root) if args.data_root else Path(cfg[root_key])
    if args.data_map:
        mapping = json.loads(Path(args.data_map).read_text(encoding='utf-8-sig'))
        if task not in mapping:
            raise ValueError(f'Missing {task} in STFT data map')
        root = Path(mapping[task])
        if not root.is_absolute():
            root = Path(args.data_map).resolve().parent / root
    if not root.is_absolute():
        root = (Path.cwd() if args.data_root else PROJECT) / root
    cfg[root_key] = str(root.resolve())
    cfg.update(root_path=str(root.resolve()), dataset=args.dataset, task=task,
               method='li2025_mfeml', backbone='cnn4_mfe', attention_module='simam_eca',
               outer_loss='cross_entropy', protocol='project_multisource_targetval_5way',
               input_mode='no_stft' if is_raw else 'stft', input_tag='raw2000' if is_raw else 'stft64',
               input_representation=REPRESENTATION if is_raw else 'stft',
               img_size=IMAGE_SIZE if is_raw else 64, input_channels=1,
               sample_points=SAMPLE_POINTS if is_raw else None,
               sample_shape=[1, SAMPLE_POINTS] if is_raw else [1, 64, 64],
               raw_normalization='none' if is_raw else None,
               raw_crop='first_2000_of_project_window' if is_raw else None,
               square_padding_points=25 if is_raw else 0,
               inner_lr=args.inner_lr, outer_lr=args.outer_lr,
               outer_lr_scheduler='none', outer_lr_min=args.outer_lr,
               outer_grad_clip_norm=args.grad_clip,
               train_steps=args.train_steps, test_inner_steps=args.test_steps,
               adaptation_steps={shot: args.train_steps for shot in (1, 3, 5)},
               meta_batch_size=args.meta_batch_size, validation_episodes=args.meta_batch_size,
               smoke=args.action == 'smoke', early_stop_patience=args.patience,
               early_stop_on_perfect_validation=False)
    if args.action == 'smoke':
        cfg.update(epochs=1, meta_batch_size=1, validation_episodes=1, test_meta_batch_size=2)

    return cfg


def test_saved(path: Path, device: torch.device) -> None:
    cfg = json.loads((path / 'resolved_config.json').read_text(encoding='utf-8'))
    expected_size = {REPRESENTATION: IMAGE_SIZE, 'stft': 64}.get(cfg.get('input_representation'))
    if expected_size is None or cfg.get('img_size') != expected_size:
        raise ValueError('Checkpoint input representation/size is invalid')
    expected_channels = 1
    if cfg.get('input_channels') != expected_channels:
        raise ValueError('Checkpoint input channels are invalid')
    method = MethodConfig(**json.loads((path / 'method_config.json').read_text(encoding='utf-8')))
    if json.loads((path / 'code_sha256.json').read_text(encoding='utf-8')) != code_hashes():
        raise RuntimeError('Source changed since training; evaluate with the archived code version')
    result = json.loads((path / 'result.json').read_text(encoding='utf-8'))
    if hashlib.sha256((path / 'best.pt').read_bytes()).hexdigest() != result['checkpoint_sha256']:
        raise RuntimeError('Checkpoint hash mismatch')
    out = path / ('eval_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    data = ProjectData(cfg, out)
    try:
        if data.signature != json.loads((path / 'data_signature.json').read_text(encoding='utf-8')):
            raise RuntimeError('Data changed since training')
        model = MFEMLNetwork(cfg['n_way'], cfg['img_size'], method).to(device)
        model.load_state_dict(torch.load(path / 'best.pt', map_location=device, weights_only=True))
        seed_all(100000 + cfg['seed'])
        metrics = evaluate(model, data, cfg, method, device, 'test',
                           cfg['test_meta_batch_size'], cfg['test_inner_steps'], out)
        metrics.update(runtime_device=str(device), torch_version=torch.__version__,
                       input_representation=cfg['input_representation'],
                       training_device=result['runtime_device'],
                       same_device_type=str(device) == result['runtime_device'],
                       checkpoint_sha256=result['checkpoint_sha256'])
        save_json(out / 'metrics.json', metrics)
        print(json.dumps(metrics, ensure_ascii=False))
    finally:
        data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('pu', 'cwru'), default='pu')
    parser.add_argument('--input', choices=('stft', 'no_stft', 'nostft'), default='no_stft',
                        help='stft: existing 64x64 images; no_stft (default): raw2000 -> zero-pad to 45x45')
    parser.add_argument('--action', choices=('check', 'smoke', 'train', 'test'), default='check')
    parser.add_argument('--tasks', nargs='+', help='P1 P2 P3 P4 or C0 C1 C2 C3; all; default=current config target')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--shot', type=int, choices=(1, 3, 5), default=1)
    parser.add_argument('--epochs', type=int, default=500, help='Maximum meta updates: default 500 (paper Table 2: 1000)')
    parser.add_argument('--inner-lr', type=float, default=0.1, help='Table 2 base-learning rate: 0.1')
    parser.add_argument('--outer-lr', type=float, default=0.01, help='Table 2 meta-learning rate: 0.01')
    parser.add_argument('--meta-batch-size', type=int, default=25)
    parser.add_argument('--train-steps', type=int, default=10)
    parser.add_argument('--test-steps', type=int, default=10)
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Supplemental stability limit; 0 disables')
    parser.add_argument('--patience', type=int, default=100, help='Validation epochs without strict accuracy improvement: default 100; 0 disables')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--data-root', help='no_stft: raw MAT root; stft: existing image root')
    parser.add_argument('--data-map', help='STFT only: JSON mapping tasks to existing image roots')
    parser.add_argument('--method-config', help='JSON overrides for MethodConfig; see notes.md')
    parser.add_argument('--checkpoint-dir', type=Path, help='Completed run directory for --action test')
    args = parser.parse_args()
    if args.input == 'nostft':
        args.input = 'no_stft'
    if args.data_map and (args.input != 'stft' or args.data_root):
        parser.error('--data-map requires --input stft and cannot be combined with --data-root')
    if args.epochs <= 0 or args.patience < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('epochs > 0, patience >= 0 and unique seeds required')
    import math
    if any(not math.isfinite(v) or v <= 0 for v in (args.inner_lr, args.outer_lr)):
        parser.error('Learning rates must be positive and finite')
    if min(args.meta_batch_size, args.train_steps, args.test_steps) < 1:
        parser.error('Task batch size and inner steps must be positive')
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        parser.error('grad-clip must be finite and nonnegative')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.action == 'test':
        if not args.checkpoint_dir:
            parser.error('--action test requires --checkpoint-dir')
        test_saved(args.checkpoint_dir.resolve(), device)
        return
    base_module = importlib.import_module('config_' + args.dataset)
    current = getattr(base_module, args.dataset.upper() + '_CONFIG')['target_condition']
    tasks = args.tasks or [t for t, target in TARGETS[args.dataset] if target == current]
    if tasks == ['all']:
        tasks = [t for t, _ in TARGETS[args.dataset]]
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - dict(TARGETS[args.dataset]).keys():
        parser.error('Invalid or duplicate tasks')
    method = MethodConfig()
    if args.method_config:
        method = MethodConfig(**(asdict(method) | json.loads(Path(args.method_config).read_text(encoding='utf-8-sig'))))
    try:
        jobs = [configuration(args, task, seed) for task in tasks for seed in args.seeds]
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    # All jobs must pass data checks before starting any expensive training.
    for cfg in jobs:
        try:
            data = ProjectData(cfg)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            parser.error(f"{cfg['task']} data check failed: {exc}\n"
                         'No input files were generated or overwritten. Check the selected input mode and data root.')
        print(json.dumps({'task': cfg['task'], 'seed': cfg['seed'], 'root': cfg['root_path'],
                          'input_representation': cfg['input_representation'], 'epochs': cfg['epochs'],
                          'inner_lr': cfg['inner_lr'], 'outer_lr': cfg['outer_lr'],
                          'meta_batch_size': cfg['meta_batch_size'],
                          'train_steps': cfg['train_steps'], 'test_steps': cfg['test_inner_steps'],
                          'sample_shape': cfg['sample_shape'],
                          'counts': {k: v['counts'] for k, v in data.signature.items()}}, ensure_ascii=False))
        data.close()
    if args.action == 'check':
        print(f'CHECK PASSED: {args.input} inputs and support/query budgets verified; no offline preprocessing.')
        return
    batch = HERE / ('smoke_runs' if args.action == 'smoke' else 'runs') / (
        f"{args.dataset}_{jobs[0]['input_tag']}_{args.shot}s_" + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    batch.mkdir(parents=True)
    print(f'Output: {batch}', flush=True)
    results = []
    for cfg in jobs:
        result = train_one(cfg, method, device, batch / f"{cfg['task']}_s{cfg['seed']}")
        results.append(result)
        save_json(batch / 'seed_results.json', results)
    summary = {}
    for task in tasks:
        values = [r['accuracy_pct'] for r in results if r['task'] == task and r['status'] == 'OK']
        summary[task] = ({'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
                         if len(values) == 3 and set(args.seeds) == set(SEEDS) else 'incomplete_or_smoke')
    if len(tasks) == 4 and len(results) == 12 and all(r['status'] == 'OK' for r in results) and set(args.seeds) == set(SEEDS):
        means = [statistics.mean(r['accuracy_pct'] for r in results if r['seed'] == seed) for seed in SEEDS]
        summary['all_conditions'] = dict(mean=statistics.mean(means), sample_sd=statistics.stdev(means))
    save_json(batch / 'summary.json', summary)
    print(f'Finished: {batch}', flush=True)


if __name__ == '__main__':
    main()
