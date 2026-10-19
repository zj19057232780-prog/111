"""Test a completed cross-dataset run using its verified saved source, never train.

On AutoDL upload this helper to the parent of IFMAML-master, not inside that
folder's top level: top-level Python files participate in its code fingerprint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Original IFMAML-master directory')
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument('--model-name', help='Exact checkpoint basename without _best')
    identity.add_argument('--checkpoint', type=Path, help='Exact _best file, if multiple copies exist')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.checkpoint and not args.checkpoint.name.endswith('_best'):
        parser.error('--checkpoint must name a _best file')
    name = args.model_name or args.checkpoint.name[:-5]
    if Path(name).name != name or any(c in name for c in '*?[]/\\'):
        parser.error('--model-name must be an exact basename')
    matches = [args.checkpoint] if args.checkpoint else list((root/'model_save').rglob(name+'_best'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one checkpoint, found {len(matches)}: {matches}')
    checkpoint = matches[0].resolve()
    base = checkpoint.with_name(name)
    saved = json.loads(Path(str(checkpoint)+'.config.json').read_text(encoding='utf-8'))
    cfg = json.loads(Path(str(base)+'_resolved_config.json').read_text(encoding='utf-8'))
    history = json.loads(Path(str(base)+'_history.json').read_text(encoding='utf-8'))
    if not history or [r['epoch'] for r in history] != list(range(1,len(history)+1)):
        raise ValueError('Missing or non-contiguous training history')
    limit = cfg['ft_pretrain_epochs'] if cfg['method']=='tl_wdcnn' else cfg['epochs']
    full_budget = len(history) == limit
    stopped = (saved.get('training_complete') is True and
               saved.get('stop_reason') == 'validation_patience' and
               saved.get('epochs_run') == len(history) and
               history[-1].get('patience_stop_triggered') is True)
    if not (full_budget or stopped):
        raise ValueError('Training is incomplete; refusing to label an interrupted best checkpoint as complete')
    if cfg['experiment_signature'] != saved['experiment_signature']:
        raise ValueError('Resolved config and checkpoint metadata signatures differ')
    if not 1 <= saved['best_epoch'] <= len(history):
        raise ValueError('Invalid best epoch')
    if digest(checkpoint) != saved['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    snapshot = Path(str(base)+'_source')
    if not snapshot.is_dir():
        snapshot = Path(saved['code_snapshot'])
    snapshot = snapshot.resolve()
    for relative, expected in saved['code_hashes'].items():
        source = (snapshot/relative).resolve()
        if snapshot not in source.parents or not source.is_file() or digest(source) != expected:
            raise ValueError(f'Missing or modified source snapshot: {relative}')
    snapshot_main = snapshot/'IFMAML-master'
    if not (snapshot_main/'run_cross_dataset.py').is_file():
        raise FileNotFoundError('Snapshot entry point missing')
    # Absolute old run paths and relative paths retain their original meaning.
    os.chdir(root)
    sys.path[:0] = [str(snapshot_main), str(snapshot)]
    import run_cross_dataset as runner
    if Path(runner.__file__).resolve() != snapshot_main/'run_cross_dataset.py':
        raise RuntimeError('Imported the wrong runner; execute this helper in a fresh Python process')
    if runner.code_files() != saved['code_hashes']:
        raise ValueError('Snapshot contains a different set of code files')
    print(f'Verified: {checkpoint}\nTrained epochs: {len(history)}; best epoch: {saved["best_epoch"]}\nSource: {snapshot}',flush=True)
    if args.check_only:
        return
    runner.torch.set_num_threads(4)
    runner.seed_torch(cfg['seed'])
    # Do not resolve cfg against new defaults; use the original resolved config.
    learner = runner.CrossLearner(cfg)
    learner.test(str(checkpoint))


if __name__ == '__main__':
    main()
