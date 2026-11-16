"""Unified CWRU/PU experiments using the existing model and training implementations."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

from run_table_row import TARGETS, SEEDS, configure, select_device

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
PAPERS = {'amaml': 'mu2025_amaml', 'tubr_ml': 'wang2025_tubr_ml', 'mfeml': 'li2025_mfeml'}
BASELINES = ('protonet', 'matchingnet', 'relationnet', 'maml', 'tl_wdcnn')
ABLATIONS = {'a0': 'cnn4maml', 'a1': 'lskmaml', 'a2': 'lskgcmaml', 'a3': 'final'}
METHODS = (*BASELINES, *PAPERS, 'scg_maml', *ABLATIONS)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict:
    paths = [p for p in PROJECT.glob('*.py') if not p.name.startswith('test_')]
    paths += list((PROJECT/'my_utils').glob('*.py'))
    paths += list((PROJECT/'optional_models').glob('*.py'))
    for folder in PAPERS.values():
        paths += list((ROOT/'comparison_models'/folder).glob('*.py'))
        paths += list((ROOT/'comparison_models'/folder).glob('*.json'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def method_name(value: str) -> str:
    value = value.lower().replace('-', '_')
    return {'configured_maml': 'scg_maml', 'final': 'scg_maml', 'scgmaml': 'scg_maml',
            'tubrml': 'tubr_ml', 'tlwdcnn': 'tl_wdcnn'}.get(value, value)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--dataset', choices=tuple(TARGETS))
    p.add_argument('--method', type=method_name, choices=METHODS)
    p.add_argument('--tasks', '--task', nargs='+', default=['all'])
    p.add_argument('--seeds', '--seed', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--shots', '--shot', type=int, choices=(1, 3, 5), default=1)
    p.add_argument('--input', choices=('auto', 'stft', 'nostft'), default='auto')
    p.add_argument('--action', choices=('check', 'smoke', 'train', 'train-test', 'test'), default='check')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--data-root', type=Path, help='STFT: prepared images; nostft: raw MAT layout')
    p.add_argument('--data-map', type=Path, help='STFT JSON mapping each task to its image directory')
    p.add_argument('--raw-root', type=Path, help='CWRU raw root for generating a separate task STFT cache')
    p.add_argument('--output-root', type=Path, default=PROJECT/'experiment_results')
    p.add_argument('--checkpoint-dir', type=Path, help='A completed run directory; required for test only')
    p.add_argument('--dry-run', action='store_true', help='Print jobs without loading data or writing files')
    args = p.parse_args(argv)
    if args.worker:
        return args
    if args.epochs < 1 or args.patience < 0 or args.threads < 1:
        p.error('epochs/threads must be positive and patience non-negative')
    if args.action == 'test':
        if not args.checkpoint_dir:
            p.error('--action test requires --checkpoint-dir; saved settings determine dataset and method')
        if any((args.dataset, args.method, args.data_root, args.data_map, args.raw_root)):
            p.error('test uses saved configuration; omit dataset/method/data overrides')
        return args
    if not args.dataset or not args.method:
        p.error('--dataset and --method are required')
    tasks = [t.upper() for item in args.tasks for t in item.split(',')]
    args.tasks = list(dict(TARGETS[args.dataset])) if tasks == ['ALL'] else tasks
    if not args.tasks or len(set(args.tasks)) != len(args.tasks) or set(args.tasks)-dict(TARGETS[args.dataset]).keys():
        p.error('Invalid or duplicate tasks for this dataset')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        p.error('Seeds must be unique integers in [0, 2**32)')
    if args.epochs < 1 or args.patience < 0 or args.threads < 1:
        p.error('epochs/threads must be positive and patience non-negative')
    if args.input == 'auto':
        args.input = 'nostft' if args.method in (*BASELINES, 'amaml') else 'stft'
    if args.method in ('scg_maml', *ABLATIONS) and args.input != 'stft':
        p.error('SCG-MAML and A0-A3 require STFT input')
    if args.method in ABLATIONS and args.shots not in (1, 3):
        p.error('The table ablations A0-A3 support 1-shot and 3-shot')
    if args.data_map and (args.data_root or args.input != 'stft'):
        p.error('--data-map requires STFT and cannot be combined with --data-root')
    if args.raw_root and (args.dataset != 'cwru' or args.input != 'stft' or args.data_root or args.data_map):
        p.error('--raw-root is only for generating CWRU STFT caches without an existing image root/map')
    for key in ('data_root', 'data_map', 'raw_root', 'output_root', 'checkpoint_dir'):
        if getattr(args, key, None) is not None:
            setattr(args, key, getattr(args, key).resolve())
    return args


def base_config(args, task: str, seed: int) -> dict:
    mod = importlib.import_module('config_' + args.dataset)
    base = copy.deepcopy(getattr(mod, args.dataset.upper()+'_CONFIG'))
    if args.dataset == 'pu':
        base = mod.resolve_pu_config(dict(base, classification_preset='artificial_9state'))
    method = ABLATIONS.get(args.method, 'final' if args.method == 'scg_maml' else args.method)
    cfg = configure(base, method if method not in PAPERS else 'cnn4maml',
                    dict(TARGETS[args.dataset])[task], args.dataset, args.shots, seed, args.epochs)
    cfg['input_representation'] = 'stft' if args.input == 'stft' else 'raw_1d'
    patience = args.patience
    cfg.update(early_stop_patience=patience, ft_patience=patience,
               early_stop_on_perfect_validation=False, smoke=args.action == 'smoke',
               test_meta_batch_size=1000, tsne_selected_classes=[], save_visualizations=True)
    if args.method == 'tl_wdcnn':
        cfg['ft_pretrain_epochs'] = min(args.epochs, 100)
    for key in ('root_path', 'mat_root', 'raw_data_root'):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((PROJECT/cfg[key]).resolve())
    root = args.data_root
    if args.data_map:
        mapping = json.loads(args.data_map.read_text(encoding='utf-8-sig'))
        root = Path(mapping[task])
        if not root.is_absolute():
            root = args.data_map.parent/root
    if root:
        key = 'root_path' if args.input == 'stft' else 'mat_root' if args.dataset == 'pu' else 'raw_data_root'
        cfg[key] = str(root.resolve())
    elif args.input == 'stft' and args.dataset == 'cwru':
        if args.raw_root:
            cfg['raw_data_root'] = str(args.raw_root)
        cfg['preprocess_seed'] = 24
        cache_cfg = {k: v for k, v in cfg.items() if k in (
            'raw_data_root', 'source_condition', 'target_condition', 'class_specs',
            'window_size', 'preprocess_seed', 'stft_fs', 'stft_nperseg', 'stft_noverlap',
            'stft_window', 'normalize', 'stft_gamma', 'stft_p_low', 'stft_p_high')}
        tag = hashlib.sha256(json.dumps(cache_cfg, sort_keys=True).encode()).hexdigest()[:12]
        cfg['root_path'] = str(args.output_root/'_data'/f'cwru_{task}_{tag}')
    if cfg['smoke']:
        cfg.update(epochs=1, meta_batch_size=1, validation_episodes=1, tasks_per_epoch=2,
                   test_meta_batch_size=2, ft_pretrain_epochs=1, ft_pretrain_max_batches=2,
                   ft_validation_episodes=1, ft_steps=1, test_inner_steps=1,
                   adaptation_steps={1:1, 3:1, 5:1}, early_stop_patience=0, ft_patience=0)
    return cfg


def prepare_images(args, cfg: dict) -> None:
    if args.dataset == 'cwru' and args.input == 'stft' and not (args.data_root or args.data_map):
        from cwru_preprocess import preprocess_cwru
        root = Path(cfg['root_path'])
        keys = ('raw_data_root', 'source_condition', 'target_condition', 'class_specs',
                'condition_speeds', 'window_size', 'img_size', 'preprocess_seed', 'normalize',
                'source_samples_per_class', 'target_val_samples_per_class', 'target_test_samples_per_class')
        identity = {k: v for k, v in cfg.items() if k in keys or k.startswith('stft_')}
        identity['raw_files'] = {p.relative_to(Path(cfg['raw_data_root'])).as_posix(): digest(p)
                                 for p in sorted(Path(cfg['raw_data_root']).rglob('*.mat'))}
        marker = root/'unified_cache.json'
        if root.exists():
            if not marker.is_file():
                raise FileExistsError(f'Unverified/incomplete STFT cache: {root}; choose a new output root')
            saved = json.loads(marker.read_text(encoding='utf-8'))
            images = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*.png'))}
            if saved != dict(identity=identity, images=images):
                raise ValueError(f'STFT cache or source changed: {root}; choose a new output root')
            return
        preprocess_cwru(cfg, clean=False)
        save_json(marker, dict(identity=identity,
            images={p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*.png'))}))


def load_paper(method: str):
    folder = ROOT/'comparison_models'/PAPERS[method]
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('paper_runner', folder/'run.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_learner(cfg: dict, dataset: str, input_mode: str):
    import maml_train_pic as training
    from raw_fewshot import RawFewShotLearner
    from raw_transfer import RawTransferLearner
    from run_cwru_maml import CWRUMAMLLearner
    if cfg['method'] == 'tl_wdcnn' and input_mode == 'stft':
        from experiment_support import STFTTransferLearner
        return STFTTransferLearner(5, cfg, dataset)
    if cfg['method'] == 'tl_wdcnn':
        return RawTransferLearner(5, cfg)
    if input_mode == 'nostft':
        return RawFewShotLearner(5, cfg)
    return CWRUMAMLLearner(5, cfg) if dataset == 'cwru' else training.MAML_learner(5, cfg)


def run_classic(args, cfg: dict, out: Path, source: Path | None = None) -> dict:
    from experiment_support import data_signature, evaluate_classic
    from my_utils.init_utils import seed_torch
    cfg['visualization_dir'] = str(out/'artifacts')
    learner = create_learner(cfg, args.dataset, args.input)
    signature = data_signature(learner._get_pu_storage(), cfg)
    if source and signature != json.loads((source/'data_signature.json').read_text(encoding='utf-8')):
        raise ValueError('Data changed since training; use the original data and source snapshot')
    if args.action == 'check':
        return dict(status='CHECKED', signature=signature)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'data_signature.json', signature)
    save_json(out/'resolved_config.json', learner.pu_config)
    if source:
        saved = json.loads((source/'result.json').read_text(encoding='utf-8'))
        best = source/'model_best'
        if digest(best) != saved['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
    else:
        seed_torch(cfg['seed'])
        best = Path(learner.train(str(out/'model'), shots=cfg['k_shot'], quick_test=False))
    result = dict(status='TRAINED', dataset=args.dataset, task=args.task, seed=cfg['seed'],
                  method=args.method, shot=cfg['k_shot'], input_mode=args.input,
                  input_representation=learner.pu_config['input_representation'],
                  checkpoint_sha256=digest(best), checkpoint=str(best),
                  runtime_device=str(next(learner.model.parameters()).device))
    # Keep a recoverable training result even if subsequent evaluation fails.
    save_json(out/'result.json', result)
    if args.action != 'train':
        # New instance proves checkpoint reconstruction rather than evaluating in-memory weights.
        learner = create_learner(cfg, args.dataset, args.input)
        result.update(evaluate_classic(learner, best, cfg, out))
        result['status'] = 'SMOKE_ONLY' if cfg['smoke'] else 'OK'
    save_json(out/'result.json', result)
    return result


def worker(job: dict) -> None:
    import torch
    args = SimpleNamespace(**job['args'])
    for key in ('data_root', 'data_map', 'raw_root', 'output_root'):
        if getattr(args, key, None):
            setattr(args, key, Path(getattr(args, key)))
    os.chdir(PROJECT)
    device = torch.device(select_device(args.device))
    torch.set_num_threads(args.threads)
    import maml_train_pic as training
    training.device = device
    from my_utils.init_utils import seed_torch
    seed_torch(args.seed)
    out = Path(job['out'])
    source = Path(job['source']) if job.get('source') else None
    if source:
        if code_hashes() != json.loads((source/'unified_code_sha256.json').read_text(encoding='utf-8')):
            raise ValueError('Source changed since training; use the saved source snapshot')
        cfg = json.loads((source/'resolved_config.json').read_text(encoding='utf-8'))
        cfg['adaptation_steps'] = {int(k): v for k, v in cfg['adaptation_steps'].items()}
    else:
        cfg = base_config(args, args.task, args.seed)
        prepare_images(args, cfg)
    try:
        result = execute_backend(args, cfg, out, source, device)
    finally:
        # Preserve the exact source also on interrupted/failed evaluation.
        if args.action != 'check' and out.is_dir():
            hashes = code_hashes()
            save_json(out/'unified_code_sha256.json', hashes)
            save_json(out/'run_identity.json', vars(args))
            for relative in hashes:
                dest = out/'source_snapshot'/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((ROOT/relative).read_bytes())
    print(json.dumps(result, ensure_ascii=False), flush=True)


def execute_backend(args, cfg, out, source, device):
    if args.method in PAPERS:
        module = load_paper(args.method)
        if source:
            module.test_saved(source, device)
            return
        raw = args.input == 'nostft'
        paper_args = SimpleNamespace(dataset=args.dataset, shot=args.shots, epochs=args.epochs,
            input='no_stft' if raw else 'stft', action=args.action,
            data_root=cfg['mat_root' if args.dataset == 'pu' else 'raw_data_root'] if raw else cfg['root_path'],
            data_map=None, patience=args.patience, inner_lr=.1 if args.method == 'mfeml' else .01,
            outer_lr=.01 if args.method == 'mfeml' else .001, grad_clip=1.,
            train_steps=10, test_steps=10, meta_batch_size=25)
        cfg = module.configuration(paper_args, args.task, args.seed)
        cfg.update(early_stop_on_perfect_validation=False, train_only=args.action == 'train')
        if args.action == 'check':
            data = module.ProjectData(cfg)
            try:
                result = dict(status='CHECKED', signature=data.signature)
            finally:
                data.close()
        else:
            result = module.train_one(cfg, module.MethodConfig(), device, out)
    else:
        result = run_classic(args, cfg, out, source)
    return result


def summarize(records: list, tasks: list) -> dict:
    good = {(r['task'], r['seed']): r for r in records if r.get('status') == 'OK'}
    summary = {}
    for metric in ('accuracy_pct', 'macro_f1_pct'):
        by_task = {}
        for task in tasks:
            vals = [good[(task, s)][metric] for s in SEEDS if (task, s) in good and metric in good[(task, s)]]
            by_task[task] = dict(mean=statistics.mean(vals), sample_sd=statistics.stdev(vals)) if len(vals)==3 else 'incomplete'
        summary[metric] = by_task
        if all((t, s) in good and metric in good[(t, s)] for t in tasks for s in SEEDS):
            vals = [statistics.mean(good[(t, s)][metric] for t in tasks) for s in SEEDS]
            summary[metric]['all_conditions'] = dict(mean=statistics.mean(vals), sample_sd=statistics.stdev(vals))
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker:
        worker(json.loads(args.worker.read_text(encoding='utf-8')))
        return 0
    if args.action == 'test':
        source = args.checkpoint_dir.resolve()
        saved = json.loads((source/'run_identity.json').read_text(encoding='utf-8'))
        saved.update(action='test', device=args.device, threads=args.threads)
        job = dict(args=saved, source=str(source), out=str(source/('eval_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
        if args.dry_run:
            print(json.dumps(job, ensure_ascii=False, indent=2))
        else:
            worker(job)
        return 0
    jobs = [dict(vars(args), task=t, seed=s) for t in args.tasks for s in args.seeds]
    if args.dry_run:
        print(json.dumps(jobs, ensure_ascii=False, indent=2, default=str))
        return 0
    group = 'ablation' if args.method in ABLATIONS else 'comparison'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    batch = args.output_root/('_smoke' if args.action=='smoke' else '_checks' if args.action=='check' else 'formal')/args.dataset/group/args.input/args.method/f'{args.shots}shot'/stamp
    batch.mkdir(parents=True, exist_ok=False)
    records = []
    failed = False
    for item in jobs:
        out = batch/f'seed{item["seed"]}'/item['task']
        job_path = batch/f'{item["task"]}_seed{item["seed"]}.job.json'
        save_json(job_path, dict(args=item, out=str(out)))
        log_path = job_path.with_suffix('.log')
        print(f'[{args.action}] {args.dataset} {item["task"]} {args.method} {args.input} seed={item["seed"]}', flush=True)
        with log_path.open('w', encoding='utf-8') as log:
            env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
            process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker', str(job_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            failed = True
            records.append(dict(task=item['task'], seed=item['seed'], status='FAILED', log=str(log_path)))
        elif (out/'result.json').is_file():
            records.append(json.loads((out/'result.json').read_text(encoding='utf-8')))
        save_json(batch/'runs.json', records)
    save_json(batch/'summary.json', summarize(records, list(dict(TARGETS[args.dataset]))))
    print(f'Output: {batch}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    raise SystemExit(main())
