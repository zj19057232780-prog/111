"""Draw a t-SNE figure of AMAML meta-test query features from a finished run.

Read-only with respect to training artifacts: reads a completed run directory and
writes only into <run_dir>/tsne/. No retraining, no checkpoint overwrite, no STFT PNG.

Required inputs (all already produced by run.py):
  <run_dir>/best.pt                 trained weights (the validation-best checkpoint)
  <run_dir>/resolved_config.json    full runtime config, including data root and class_groups
  <run_dir>/method_config.json      MethodConfig for adapt()
  <run_dir>/data_signature.json     used to verify the reloaded split matches training
plus the raw MAT data root recorded in resolved_config.json (override with --data-root).

Two modes:

  --mode fixed    (default) Class-balanced fixed task over --classes (default
                  tsne_selected_classes, i.e. the 5 semantic classes used by the project's
                  existing figures). k-shot support, every remaining window of those classes
                  as query, one adaptation, features before the linear classifier.
                  Comparable with IFMAML-master/maml_train_pic.py's fixed-class t-SNE.

  --mode episodes Pool the query features of the run's own meta-test episodes, so every
                  class in the split can be shown (--classes all). The episodes are replayed
                  exactly as run.py built them: the test split is re-created, the RNG is
                  re-seeded with 100000 + seed, and the same TaskDataset sampler is used.
                  Pooled accuracy is cross-checked against result.json. Only --split test is
                  supported: the training-time validation episodes are not reproducible from
                  a fixed seed because they depend on the training RNG history.

Caveats inherited from the model, both modes:
  * BatchNorm uses episode-local batch statistics (track_running_stats=False), so a feature
    comes from the whole query batch in one forward, not from a single sample.
  * In --mode episodes, different episodes come from different adapted parameter sets, so the
    pooled cloud mixes adaptation contexts. Treat cluster distances as indicative only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

# joblib/loky probes the physical core count through a subprocess that fails on this
# machine and then dumps a raw traceback to stderr. Capping the worker count below the
# logical core count makes joblib take its "respect user setting" early return.
# Must happen before sklearn imports joblib. Costs one worker thread; irrelevant here.
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(max((os.cpu_count() or 2) - 1, 1)))

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent / 'IFMAML-master'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT))

from model import AMAMLNetwork, MethodConfig, adapt
from l2l_shim import MetaDataset, TaskDataset
from raw_reshape import RawReshapeDataset

SPLITS = ('test', 'validation', 'train')
SPLIT_DATA = {
    'train': 'source_data',
    'validation': 'target_val_data',
    'test': 'target_test_data',
}


def load_cfg(directory: Path, data_root: str | None) -> dict:
    cfg = json.loads((directory / 'resolved_config.json').read_text(encoding='utf-8'))
    key = 'mat_root' if cfg['dataset'] == 'pu' else 'raw_data_root'
    root = Path(data_root) if data_root else Path(cfg[key])
    if not root.is_absolute():
        root = (Path.cwd() if data_root else PROJECT) / root
    if not root.is_dir():
        raise FileNotFoundError(f'Raw MAT root not found: {root}')
    cfg[key] = str(root.resolve())
    cfg['root_path'] = str(root.resolve())
    return cfg


def check_code_hashes(directory: Path) -> None:
    recorded = json.loads((directory / 'code_sha256.json').read_text(encoding='utf-8'))
    for relative, expected in recorded.items():
        path = HERE.parent.parent / relative
        if not path.exists():
            print(f'[warn] source file missing: {relative}')
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            print(f'[warn] source changed since training: {relative} '
                  '(features may not match the archived checkpoint)')


def seed_all(seed: int) -> None:
    """Same seeding order as run.py, so the test episodes can be replayed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(directory: Path, cfg: dict, device: torch.device) -> AMAMLNetwork:
    model = AMAMLNetwork(cfg['n_way'], cfg['img_size']).to(device)
    model.load_state_dict(torch.load(directory / 'best.pt', map_location=device, weights_only=True))
    model.eval()
    return model


def verify_split(storage: RawReshapeDataset, split: str, recorded: dict) -> tuple:
    names = storage.source_classes
    x, y = storage.to_maml_dataset_format(split)
    actual = {
        'shape': list(x.shape),
        'labels': names,
        'counts': {names[int(c)]: int((y == c).sum()) for c in np.unique(y)},
        'sha256': hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
    }
    if actual != recorded.get(split):
        raise RuntimeError(
            f'Reloaded {split} split differs from training-time data_signature.json. '
            'Check --data-root, preprocess_seed and class_groups before trusting this figure.')
    print(f'{split} split verified against data_signature.json: shape={actual["shape"]}')
    return x, y


def resolve_classes(storage: RawReshapeDataset, cfg: dict, requested: list[str] | None) -> list[str]:
    wanted = requested if requested else cfg.get('tsne_selected_classes')
    if not wanted:
        raise ValueError('No classes given and resolved_config.json has no tsne_selected_classes')
    if len(wanted) == 1 and wanted[0].lower() == 'all':
        return list(storage.source_classes)
    unknown = [name for name in wanted if name not in storage.source_classes]
    if unknown:
        raise ValueError(f'Unknown classes: {unknown}; available: {storage.source_classes}')
    return list(wanted)


def adapt_and_forward(model: AMAMLNetwork, sx, sy, qx, cfg: dict, method: MethodConfig,
                      device: torch.device, steps: int):
    params = adapt(model, sx.to(device), sy.to(device), steps, cfg['inner_lr'], method,
                   meta_grad=False)
    with torch.no_grad():
        features, logits = functional_call(model, params, (qx.to(device),))
    return features.cpu().numpy(), logits.cpu()


def build_fixed_task(storage: RawReshapeDataset, split: str, selected: list[str],
                     shots: int, rng: np.random.Generator) -> tuple:
    """Class-balanced task: k-shot support plus every remaining window as query."""
    per_class = getattr(storage, SPLIT_DATA[split])
    support_x, support_y, query_x, query_y, used = [], [], [], [], []
    for name in selected:
        data = per_class.get(name)
        if data is None:
            print(f'[warn] class not present in {split} split: {name}')
            continue
        if len(data) <= shots:
            print(f'[warn] class has too few samples for {shots}-shot: {name} ({len(data)})')
            continue
        label = len(used)
        used.append(name)
        order = np.arange(len(data))
        rng.shuffle(order)
        for index in order[:shots]:
            support_x.append(torch.from_numpy(np.ascontiguousarray(data[index])).float())
            support_y.append(label)
        for index in order[shots:]:
            query_x.append(torch.from_numpy(np.ascontiguousarray(data[index])).float())
            query_y.append(label)
    if len(used) < 2 or not query_x:
        raise RuntimeError('Fixed t-SNE task has fewer than 2 classes or no query samples')
    return (torch.stack(support_x), torch.tensor(support_y, dtype=torch.long),
            torch.stack(query_x), torch.tensor(query_y, dtype=torch.long), used)


def replay_test_episodes(storage: RawReshapeDataset, x: np.ndarray, y: np.ndarray, cfg: dict):
    """Rebuild the run's own meta-test episode sampler. Call seed_all first."""
    meta = MetaDataset(list(zip(x, y)))
    return TaskDataset(meta, cfg['n_way'], cfg['k_shot'], cfg['q_query'])


def pooled_episode_features(model: AMAMLNetwork, task: TaskDataset, cfg: dict,
                            method: MethodConfig, device: torch.device, episodes: int,
                            steps: int) -> tuple:
    """Collect adapted query features over episodes.

    Accuracy needs the local 5-way labels the adapted head predicts; colouring needs the
    global class index, because every episode re-labels its own 5 chosen classes 0..4.
    """
    all_features, all_local, all_global, all_pred = [], [], [], []
    for index in range(episodes):
        sx, sy, qx, qy_local, _, qy_global, _ = task.sample_with_original_labels()
        features, logits = adapt_and_forward(model, sx, sy, qx, cfg, method, device, steps)
        all_features.append(features)
        all_local.append(qy_local.numpy())
        all_global.append(qy_global.numpy())
        all_pred.append(logits.argmax(1).numpy())
        if (index + 1) % 100 == 0:
            print(f'  episode {index + 1}/{episodes}', flush=True)
    features = np.concatenate(all_features)
    labels = np.concatenate(all_global)
    accuracy = float((np.concatenate(all_local) == np.concatenate(all_pred)).mean()) * 100
    return features, labels, accuracy


def subsample(features: np.ndarray, labels: np.ndarray, limit: int, seed: int) -> tuple:
    if len(features) <= limit:
        return features, labels
    picked = np.random.default_rng(seed).choice(len(features), size=limit, replace=False)
    return features[picked], labels[picked]


def draw(features: np.ndarray, labels: np.ndarray, names: list[str], out_png: Path,
         perplexity: float, seed: int, title: str) -> np.ndarray:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    perplexity = min(float(perplexity), max(2.0, (len(features) - 1) / 3.0))
    embedding = TSNE(n_components=2, perplexity=perplexity, init='pca',
                     learning_rate='auto', random_state=seed).fit_transform(features)

    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=180)
    cmap = plt.get_cmap('tab10')
    for index, name in enumerate(names):
        mask = labels == index
        if not np.any(mask):
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=14, alpha=0.75,
                   color=cmap(index % 10), label=f'{name} (n={int(mask.sum())})', linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(alpha=0.18)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=min(5, max(2, (len(names) + 1) // 2)), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    return embedding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint-dir', type=Path, required=True,
                        help='Finished run directory containing best.pt and result.json')
    parser.add_argument('--mode', choices=('fixed', 'episodes'), default='fixed')
    parser.add_argument('--split', choices=SPLITS, default='test')
    parser.add_argument('--classes', nargs='+',
                        help='Class names, or "all" for every class in the split; '
                             'default = tsne_selected_classes in resolved_config.json')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='--mode episodes: number of test episodes to pool')
    parser.add_argument('--shots', type=int, default=None,
                        help='Support shots for --mode fixed; default = k_shot of the run')
    parser.add_argument('--inner-steps', type=int, default=None,
                        help='Adaptation steps; default = test_inner_steps of the run')
    parser.add_argument('--points', type=int, default=1000, help='t-SNE point cap')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--data-root', help='Override the raw MAT root recorded in the run')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--out', type=Path, help='Output directory; default <run_dir>/tsne')
    args = parser.parse_args()

    directory = args.checkpoint_dir.resolve()
    if not (directory / 'best.pt').is_file():
        parser.error(f'No best.pt in {directory}')
    if args.episodes <= 0:
        parser.error('--episodes must be > 0')
    if args.mode == 'episodes' and args.split != 'test':
        parser.error('--mode episodes only supports --split test; the training-time validation '
                     'episodes depend on the training RNG history and cannot be replayed')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')

    cfg = load_cfg(directory, args.data_root)
    check_code_hashes(directory)
    method = MethodConfig(**json.loads((directory / 'method_config.json').read_text(encoding='utf-8')))
    steps = args.inner_steps if args.inner_steps is not None else cfg.get('test_inner_steps', 10)
    result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))

    storage = RawReshapeDataset(cfg)
    x, y = verify_split(storage, args.split, json.loads(
        (directory / 'data_signature.json').read_text(encoding='utf-8')))
    selected = resolve_classes(storage, cfg, args.classes)
    model = load_model(directory, cfg, device)

    if args.mode == 'fixed':
        if len(selected) > cfg['n_way']:
            parser.error(f'--mode fixed adapts a {cfg["n_way"]}-way classifier, so at most '
                         f'{cfg["n_way"]} classes; {len(selected)} classes need --mode episodes')
        shots = args.shots if args.shots is not None else cfg['k_shot']
        task = build_fixed_task(storage, args.split, selected, shots,
                                np.random.default_rng(int(cfg.get('seed', 24))))
        features, logits = adapt_and_forward(model, *task[:3], cfg, method, device, steps)
        labels = task[3].numpy()
        names = task[4]
        accuracy = float((logits.argmax(1) == task[3]).float().mean()) * 100
        print(f'Fixed task: {args.split}, classes={names}, shots={shots}, '
              f'query={len(labels)}, inner_steps={steps}')
        print(f'Fixed-task accuracy: {accuracy:.2f}%  feature dim: {features.shape[1]}')
        print('Note: this task is not the run\'s 1000-episode meta-test; do not quote it as '
              f'the reported accuracy ({result["accuracy_pct"]:.2f}%).')
        tag = f'{args.split}_{shots}s_fixed{len(names)}c'
    else:
        seed_all(100000 + int(cfg['seed']))
        sampler = replay_test_episodes(storage, x, y, cfg)
        print(f'Replaying {args.episodes} meta-test episodes '
              f'({cfg["n_way"]}-way {cfg["k_shot"]}-shot {cfg["q_query"]}-query, '
              f'seed_all(100000 + {cfg["seed"]}))')
        features, labels, accuracy = pooled_episode_features(
            model, sampler, cfg, method, device, args.episodes, steps)
        names = list(storage.source_classes)
        if args.classes and not (len(args.classes) == 1 and args.classes[0].lower() == 'all'):
            keep = [names.index(name) for name in selected]
            mask = np.isin(labels, keep)
            features, labels = features[mask], labels[mask]
            remap = {old: new for new, old in enumerate(keep)}
            labels = np.array([remap[int(label)] for label in labels], dtype=np.int64)
            names = selected
        print(f'Pooled query accuracy: {accuracy:.2f}%  '
              f'(result.json: {result["accuracy_pct"]:.2f}%)  feature dim: {features.shape[1]}')
        if abs(accuracy - result['accuracy_pct']) > 0.01 and args.episodes == 1000:
            print('[warn] pooled accuracy differs from result.json; episodes may not have been '
                  'replayed exactly. Check the device and --data-root.')
        counts = {names[i]: int((labels == i).sum()) for i in range(len(names))}
        print('points per class:', counts)
        tag = f'{args.split}_{cfg["k_shot"]}s_ep{args.episodes}_all{len(names)}c'

    features, labels = subsample(features, labels, args.points, int(cfg.get('seed', 24)))
    out_dir = args.out.resolve() if args.out else directory / 'tsne'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'{directory.name}_{tag}_{stamp}'
    png = out_dir / f'{stem}_tsne.png'

    embedding = draw(features, labels, names, png, args.perplexity, int(cfg.get('seed', 24)),
                     title=f'AMAML t-SNE of meta-{args.split} query features '
                           f'({cfg["dataset"].upper()} {cfg["task"]} seed={cfg["seed"]}, '
                           f'{len(names)} classes, mode={args.mode})')
    with (out_dir / f'{stem}_points.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        stream.write('tsne1,tsne2,label_index,label_name\n')
        for (px, py), label in zip(embedding, labels):
            stream.write(f'{px:.6f},{py:.6f},{int(label)},{names[int(label)]}\n')
    (out_dir / f'{stem}_meta.json').write_text(json.dumps({
        'source_run': str(directory), 'mode': args.mode, 'split': args.split,
        'classes': names, 'points': int(len(features)), 'perplexity': float(args.perplexity),
        'shots': cfg['k_shot'] if args.mode == 'episodes' else shots,
        'inner_steps': steps, 'episodes': args.episodes if args.mode == 'episodes' else None,
        'accuracy_pct': accuracy,
        'reported_test_accuracy_pct': result['accuracy_pct'],
        'checkpoint_sha256': hashlib.sha256((directory / 'best.pt').read_bytes()).hexdigest(),
        'device': str(device),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved t-SNE PNG: {png}')


if __name__ == '__main__':
    main()
