"""Run one CWRU/PU table row: four targets x three seeds, then summarize.

Run beside config_cwru.py / config_pu.py on Windows or Linux.
Uses existing train/test APIs; all outputs are under table_row_results/.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime

SEEDS = (3, 24, 38)
RAW_METHODS = ('protonet', 'matchingnet', 'relationnet', 'maml')
TRANSFER_METHODS = ('resnet18', 'resnet18_ft', 'tl_wdcnn')
RAW_INPUT_METHODS = RAW_METHODS + TRANSFER_METHODS
TARGETS = {
    'cwru': [('C0', 'C0_1797rpm'), ('C1', 'C1_1772rpm'),
             ('C2', 'C2_1750rpm'), ('C3', 'C3_1730rpm')],
    'pu': [('P1', 'N09_M07_F10'), ('P2', 'N15_M01_F10'),
           ('P3', 'N15_M07_F04'), ('P4', 'N15_M07_F10')],
}
METHODS = ('protonet', 'matchingnet', 'relationnet', 'maml',
           'cnn4maml', 'lskmaml', 'lskgcmaml', 'final') + TRANSFER_METHODS
LABELS = dict(protonet='ProtoNet (1D)', matchingnet='MatchingNet (1D)', relationnet='RelationNet (1D)',
              maml='MAML (1D)', cnn4maml='STFT-CNN4-MAML', lskmaml='LSK-MAML',
              lskgcmaml='LSK-GC-MAML', final='SCG-MAML', resnet18='ResNet18 (1D, all classes)',
              resnet18_ft='ResNet18-FT (1D)', tl_wdcnn='TL-WDCNN (1D)')

# Verified before/after versions of the patience-only trainer change.
# Do not exempt arbitrary future training-code edits from compatibility checks.
PATIENCE_TRAINER_HASHES = {
    '92086551fbe4cba53d1f8269b3629f68e459efc011faccf831ef62d4fc843e63',
    'be537df3aa89b0c4a89b95aff1e1c15253c59b3bd8efba94522d1d723b963de3',
}


def comparable_code(hashes: dict, method: str = '') -> dict:
    code = {k: v for k, v in hashes.items() if k != 'run_table_row.py'}
    if method not in TRANSFER_METHODS:
        code.pop('raw_transfer.py', None)  # New module is not used by existing methods.
    if code.get('maml_train_pic.py') in PATIENCE_TRAINER_HASHES:
        code['maml_train_pic.py'] = 'verified-patience-compatible-trainer'
    return code


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def select_device(requested: str) -> str:
    import torch
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定了 CUDA，但当前 Python 环境无法使用 CUDA；可改用 --device cpu。')
    return 'cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu'


def comparable_config(job: dict) -> dict:
    cfg = copy.deepcopy(job['config'])
    for key in ('model_name', 'visualization_dir', 'save_visualizations',
                'raw_window_manifest', 'runtime_device'):
        cfg.pop(key, None)
    # User explicitly permits completed comparison results with earlier patience settings.
    if job['method'] in RAW_METHODS:
        cfg.pop('early_stop_patience', None)
    # This directory is regenerated for each CWRU STFT batch.
    if job['dataset'] == 'cwru' and cfg.get('input_representation') == 'stft':
        cfg.pop('root_path', None)
    return json.loads(json.dumps(cfg, default=str))


def find_saved_result(results_root: Path, job: dict, hashes: dict) -> dict | None:
    """Reuse the newest compatible completed run, never select by accuracy."""
    code = comparable_code(hashes, job['method'])
    for path in sorted(results_root.glob(f'*/{job["run_name"]}/result.json'), reverse=True):
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
            old_job = json.loads(path.with_name('job.json').read_text(encoding='utf-8'))
            old_code = json.loads((path.parent.parent / 'code_sha256.json').read_text(encoding='utf-8'))
            if comparable_code(old_code, job['method']) != code:
                continue
            if comparable_config(old_job) != comparable_config(job):
                continue
            if old_job.get('image_sha256') != job.get('image_sha256'):
                continue
            expected = dict(status='OK', dataset=job['dataset'], method=job['method'],
                            task=job['task'], seed=job['config']['seed'],
                            shot=None if job['method'] == 'resnet18' else job['config']['k_shot'],
                            input_representation=job['config']['input_representation'])
            if job['method'] == 'resnet18':
                expected.update(evaluation_protocol='direct_all_classes', test_episodes=0)
                if result.get('query_count', 0) <= 0:
                    continue
            else:
                expected.update(query_count=75000, test_episodes=1000)
            if any(result.get(k) != v for k, v in expected.items()):
                continue
            acc = result['accuracy_pct']
            if isinstance(acc, bool) or not math.isfinite(acc) or not 0 <= acc <= 100:
                continue
            if not Path(result['checkpoint']).is_file():
                continue
            # Reused runs retain a pointer to the original predictions.
            original = Path(result.get('reused_from', str(path)))
            if not original.with_name('test_predictions.csv').is_file():
                continue
            original_job = json.loads(original.with_name('job.json').read_text(encoding='utf-8'))
            return dict(result, reused_from=str(original.resolve()),
                        early_stop_patience=original_job['config'].get('early_stop_patience', 0))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            # Partial writes, old formats and damaged records must not block a rerun.
            continue
    return None


def configure(base: dict, method: str, target: str, dataset: str,
              shot: int, seed: int, epochs: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg['input_representation'] = 'raw_1d' if method in RAW_INPUT_METHODS else 'stft'
    cfg.update(source_condition=[v for _, v in TARGETS[dataset] if v != target],
               target_condition=target, n_way=5, k_shot=shot, q_query=15, seed=seed,
               epochs=epochs, test_meta_batch_size=1000, meta_batch_size=16,
               test_inner_steps=10, adaptation_steps={1: 3, 3: 3, 5: 3},
               run_mode='train_test', noise_test=False,
               frequency_module='none', denoise_module='none')
    if method in ('lskmaml', 'lskgcmaml', 'final'):
        cfg.update(method='configured_maml', backbone='lsk_lite',
                   attention_module='none' if method == 'lskmaml' else 'gcnet',
                   task_weighting_mode='ggm_rw' if method == 'final' else 'none')
    else:
        cfg.update(method='maml' if method in ('maml', 'cnn4maml') else method,
                   backbone='cnn4', attention_module='none', task_weighting_mode='none')
    if method in TRANSFER_METHODS:
        cfg.update(supervised_epochs=epochs, ft_pretrain_epochs=epochs,
                   ft_patience=cfg.get('early_stop_patience', 100))
    return cfg


def summarize(records: list[dict], dataset: str, method: str, shot: int) -> tuple[list[str], list[str]]:
    index = {(r['task'], r['seed']): r for r in records if r['status'] == 'OK'}
    def fmt(values: list[float]) -> str:
        return f'{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}'
    cells = []
    for task, _ in TARGETS[dataset]:
        vals = [index[(task, s)]['accuracy_pct'] for s in SEEDS if (task, s) in index]
        cells.append(fmt(vals) if len(vals) == 3 else f'待补({len(vals)}/3)')
    # Whole-condition SD is computed across three seed-level means.
    if len(index) == 12:
        by_seed = [statistics.mean(index[(t, s)]['accuracy_pct'] for t, _ in TARGETS[dataset])
                   for s in SEEDS]
        cells.append(fmt(by_seed))
    else:
        cells.append(f'待补({len(index)}/12)')
    return (['shot', '方法'] + [t for t, _ in TARGETS[dataset]] + ['全工况平均'],
            ['—' if method == 'resnet18' else str(shot), LABELS[method]] + cells)


def worker(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding='utf-8'))
    cfg = job['config']
    cfg['adaptation_steps'] = {int(k): v for k, v in cfg['adaptation_steps'].items()}
    project = Path(job['project'])
    sys.path.insert(0, str(project))
    os.chdir(project)
    import torch
    selected_device = select_device(job.get('device', 'auto'))
    import maml_train_pic as training
    training.device = torch.device(selected_device)
    cfg['runtime_device'] = selected_device
    print(f'Python: {sys.executable}\n设备: {selected_device}', flush=True)
    from my_utils.init_utils import seed_torch
    raw_input = cfg.get('input_representation') == 'raw_1d'
    if job['method'] in TRANSFER_METHODS:
        from raw_transfer import RawTransferLearner
        cls = RawTransferLearner
    elif raw_input:
        from raw_fewshot import RawFewShotLearner
        cls = RawFewShotLearner
    elif job['dataset'] == 'cwru':
        from run_cwru_maml import CWRUMAMLLearner
        cls = CWRUMAMLLearner
    else:
        cls = training.MAML_learner
    run_dir = Path(job['run_dir'])
    seed_torch(cfg['seed'])
    learner = cls(ways=5, **({'cwru_config': cfg} if job['dataset'] == 'cwru' and not raw_input else {'pu_config': cfg}))
    save_json(run_dir / 'resolved_config.json', learner.pu_config)
    # Capture the original test predictions at the existing output hook. No log rounding.
    captured = {}
    original = training._save_test_visualizations
    def capture(*args, **kwargs):
        import inspect
        payload = inspect.signature(original).bind(*args, **kwargs).arguments
        rows = payload['rows']
        direct = job['method'] == 'resnet18'
        expected = (sum(len(v) for v in learner._get_raw_storage().target_test_data.values())
                    if direct else cfg['test_meta_batch_size'] * cfg['n_way'] * cfg['q_query'])
        if len(rows) != expected:
            raise RuntimeError(f'测试查询数不符：{len(rows)} != {expected}')
        episodes = {}
        correct = 0
        for row in rows:
            if direct:
                correct += row['true_class_index'] == row['pred_class_index']
            else:
                episodes[row['episode']] = episodes.get(row['episode'], 0) + 1
                correct += row['true_local_label'] == row['pred_local_label']
        if not direct and (len(episodes) != 1000 or set(episodes.values()) != {75}):
            raise RuntimeError('测试episode结构与1000个5-way/15-query任务不一致')
        with (run_dir / 'test_predictions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        captured.update(accuracy_pct=100.0 * correct / len(rows), query_count=len(rows),
                        test_episodes=len(episodes))
        captured['evaluation_protocol'] = 'direct_all_classes' if direct else '5way_support_query'
        # The visualization helper uses load_path only to name its artifacts.
        # Avoid repeating the long experiment name twice (Windows MAX_PATH).
        payload['load_path'] = str(run_dir / 'eval_best')
        try:
            original(**payload)
        except OSError as exc:
            # Exact predictions and metrics are already saved above.
            captured['visualization_warning'] = str(exc)
            print(f'[warn] 测试预测已完整保存，附加图表保存失败：{exc}', flush=True)
    training._save_test_visualizations = capture
    prefix = run_dir / ('model' if job['method'] in TRANSFER_METHODS else job['run_name'])
    best = learner.train(str(prefix), shots=cfg['k_shot'], quick_test=False)
    if not best or not Path(best).is_file():
        raise RuntimeError('训练没有返回有效best checkpoint')
    # Shared deterministic test seed across methods, independent of training duration.
    seed_torch(100000 + cfg['seed'])
    learner.test(best, inner_steps=10, shots=cfg['k_shot'], meta_batch_size=1000)
    if not captured:
        raise RuntimeError('没有捕获测试预测；服务器版本可能与本地接口不同')
    captured.update(status='OK', checkpoint=str(best), task=job['task'], seed=cfg['seed'],
                    dataset=job['dataset'], method=job['method'],
                    shot=None if job['method'] == 'resnet18' else cfg['k_shot'],
                    input_representation=cfg['input_representation'],
                    early_stop_patience=cfg.get('early_stop_patience', 0))
    save_json(run_dir / 'result.json', captured)
    print(json.dumps(captured, ensure_ascii=False), flush=True)


def run_row(args: argparse.Namespace) -> int:
    project = Path(args.project).resolve()
    sys.path.insert(0, str(project))
    os.chdir(project)
    selected_device = select_device(args.device)
    mod = importlib.import_module('config_' + args.dataset)
    base = copy.deepcopy(getattr(mod, args.dataset.upper() + '_CONFIG'))
    raw_input = args.method in RAW_INPUT_METHODS
    if raw_input:
        base['early_stop_patience'] = args.patience
    if args.dataset == 'pu':
        base['classification_preset'] = 'artificial_9state'
        base = mod.resolve_pu_config(base)
        if args.pu_images:
            base['root_path'] = str(Path(args.pu_images).resolve())
        if args.pu_mat:
            base['mat_root'] = str(Path(args.pu_mat).resolve())
        if raw_input and not Path(base['mat_root']).is_dir():
            raise RuntimeError(f"PU原始MAT布局不存在：{base['mat_root']}；用 --pu-mat 指定规范化MAT目录。")
        # Validate every target before spending GPU time.
        for _, target in ([] if raw_input else TARGETS['pu']):
            for bearings in base['class_groups'].values():
                for bearing in bearings:
                    directory = Path(base['root_path']) / target / bearing
                    if len(list(directory.glob('*.png'))) < 2 * (args.shot + 15):
                        raise RuntimeError(f'PU图像不足或目录不存在：{directory}；需先完成STFT预处理。')
    else:
        if args.cwru_raw:
            base['raw_data_root'] = str(Path(args.cwru_raw).resolve())
        if not Path(base['raw_data_root']).is_dir():
            raise RuntimeError(f"CWRU原始目录不存在：{base['raw_data_root']}；用 --cwru-raw 指定含48DriveEndFault和NormalBaseline的目录。")
    if args.method in ('lskgcmaml', 'final'):
        required = ('gcnet_ratio', 'gcnet_pooling_type', 'gcnet_fusion_types', 'gcnet_layer_scale_init')
        if any(k not in base for k in required):
            raise RuntimeError('服务器配置缺少GCNet参数，请同步本地项目代码。')
    representation = 'raw1d' if raw_input else 'stft'
    print(f'Python: {sys.executable}\n设备: {selected_device}；输入: {representation}', flush=True)
    if args.check_only:
        if args.method in TRANSFER_METHODS:
            from raw_transfer import RawTransferLearner
        elif raw_input:
            from raw_fewshot import RawFewShotLearner
        else:
            import maml_train_pic
        print('环境导入与数据根目录检查通过（未启动训练，未逐个核验数据文件）。', flush=True)
        return 0
    shot_tag = 'allclasses' if args.method == 'resnet18' else f'{args.shot}shot'
    batch = project / 'table_row_results' / f'{args.dataset}_{representation}_{args.method}_{shot_tag}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    batch.mkdir(parents=True, exist_ok=False)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in project.glob('*.py')}
    save_json(batch / 'code_sha256.json', hashes)
    test_description = '目标域全类别直接测试（无support）' if args.method == 'resnet18' else '1000测试episodes'
    print(f'输出目录：{batch}\n输入：{representation}；4工况 × 3seed = 12次训练测试；epochs上限={args.epochs}；{test_description}。', flush=True)
    records = []
    for task, target in TARGETS[args.dataset]:
        common = configure(base, args.method, target, args.dataset, args.shot, 3, args.epochs)
        if args.dataset == 'cwru' and not raw_input:
            # Each leave-one-condition protocol needs source_ vs val_/test_ image roles.
            from cwru_preprocess import preprocess_cwru
            common['root_path'] = str(batch / 'data' / task)
            common['preprocess_seed'] = 24
            preprocess_cwru(common, clean=False)
        root = Path(common['root_path'])
        fingerprint = hashlib.sha256()
        for image in ([] if raw_input else sorted(root.rglob('*.png'))):
            fingerprint.update(image.relative_to(root).as_posix().encode())
            fingerprint.update(hashlib.sha256(image.read_bytes()).digest())
        for seed in SEEDS:
            cfg = copy.deepcopy(common)
            cfg['seed'] = seed
            run_name = f'{args.dataset.upper()}_{representation}_{args.method}_{task}_5w{args.shot}s_seed{seed}'
            if args.method == 'resnet18':
                run_name = f'{args.dataset.upper()}_{representation}_{args.method}_{task}_allclasses_seed{seed}'
            run_dir = batch / run_name
            run_dir.mkdir()
            cfg.update(model_name=run_name, visualization_dir=str(run_dir / 'figures'),
                       save_visualizations=True)
            if raw_input:
                cfg['raw_window_manifest'] = str(run_dir / 'raw_window_manifest.json')
            job = dict(project=str(project), dataset=args.dataset, method=args.method, device=selected_device,
                       task=task, run_name=run_name, run_dir=str(run_dir), config=cfg,
                       image_sha256=None if raw_input else fingerprint.hexdigest())
            save_json(run_dir / 'job.json', job)
            print(f'\n[{len(records)+1}/12] {run_name}', flush=True)
            reused = None if args.force_rerun else find_saved_result(batch.parent, job, hashes)
            if reused is not None:
                print(f'跳过已完成组：{reused["accuracy_pct"]:.5f}%；原patience={reused["early_stop_patience"]}（0=未启用）；来源：{reused["reused_from"]}', flush=True)
                save_json(run_dir / 'result.json', reused)
                records.append(reused)
                save_json(batch / 'runs.json', records)
                headers, row = summarize(records, args.dataset, args.method, args.shot)
                with (batch / 'row_summary.tsv').open('w', newline='', encoding='utf-8-sig') as stream:
                    csv.writer(stream, delimiter='\t').writerows([headers, row])
                continue
            with (run_dir / 'run.log').open('w', encoding='utf-8') as log:
                env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
                if selected_device == 'cpu':
                    env['CUDA_VISIBLE_DEVICES'] = '-1'
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                         '--worker', str(run_dir / 'job.json')],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding='utf-8', errors='replace', env=env)
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = proc.wait()
            result_path = run_dir / 'result.json'
            if code == 0 and result_path.is_file():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if not math.isfinite(result['accuracy_pct']) or not 0 <= result['accuracy_pct'] <= 100:
                    raise RuntimeError('非法准确率')
            else:
                result = dict(task=task, seed=seed, status='FAILED', accuracy_pct=None,
                              log=str(run_dir / 'run.log'), exit_code=code)
            records.append(result)
            save_json(batch / 'runs.json', records)
            headers, row = summarize(records, args.dataset, args.method, args.shot)
            with (batch / 'row_summary.tsv').open('w', newline='', encoding='utf-8-sig') as stream:
                csv.writer(stream, delimiter='\t').writerows([headers, row])
    with (batch / 'seed_results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['task', 'seed', 'status', 'accuracy_pct',
                                                  'early_stop_patience', 'reused_from'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print('\n========== 该行结果汇总（Accuracy %；Mean ± sample SD） ==========')
    print('\t'.join(headers))
    print('\t'.join(row))
    print(f'\n汇总：{batch / "row_summary.tsv"}\n逐seed结果：{batch / "seed_results.csv"}')
    return 0 if all(r['status'] == 'OK' for r in records) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--dataset', choices=list(TARGETS))
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--shot', type=int, choices=(1, 3, 5), default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100,
                        help='Raw comparisons: stop after this many epochs without validation accuracy improvement; 0 disables')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--check-only', action='store_true', help='Check imports, device and data roots without training')
    parser.add_argument('--force-rerun', action='store_true', help='Ignore saved results and rerun every group')
    parser.add_argument('--project', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--cwru-raw', help='CWRU raw root containing 48DriveEndFault / NormalBaseline')
    parser.add_argument('--pu-images', help='Prepared PU PNG root, normally ./pu_data_processed')
    parser.add_argument('--pu-mat', help='Prepared PU MAT layout, normally ./pu_data_mat; used for 1D baselines')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    if not args.dataset or not args.method:
        parser.error('--dataset and --method are required')
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    if args.patience < 0:
        parser.error('--patience must be non-negative')
    if args.method == 'resnet18':
        args.shot = 1  # Canonical job identity; supervised baseline has no shot dimension.
    return run_row(args)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    # Some cloud images export a malformed OpenMP setting; repair only invalid values.
    omp_value = os.environ.get('OMP_NUM_THREADS', '1')
    if not omp_value.isdecimal() or int(omp_value) < 1:
        os.environ['OMP_NUM_THREADS'] = '1'
    raise SystemExit(main())
