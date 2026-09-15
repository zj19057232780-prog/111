"""Aggregate completed independent seeds; exclude smoke checkpoints by default."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from config_cross_dataset import CROSS_CONFIG, TASKS


def aggregate(root, seeds=(3, 24, 38), episodes=1000):
    latest = {}
    for path in sorted(Path(root).glob('*/*/summary.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        if r.get('quick_checkpoint') or r['episodes'] != episodes or r['seed'] not in seeds:
            continue
        key = (r['method'], r['shots'], r['task'], r['seed'])
        if key not in latest or path.stat().st_mtime_ns > latest[key][0].stat().st_mtime_ns:
            latest[key] = (path, r)
    # A table may not mix different preprocessing or episode manifests for the same task/shot.
    for task in TASKS:
        for shots in (1, 5):
            runs = [r for _, r in latest.values() if r['task'] == task and r['shots'] == shots]
            signatures = {(r['data_sha256'], r['query_per_class'],
                           Path(r['episode_manifest']).name) for r in runs}
            if len(signatures) > 1:
                raise ValueError(f'Incomparable data/episode settings at {task}/{shots}-shot')
    rows = []
    def display(values):
        return f'{np.mean(values)*100:.2f} +/- {np.std(values, ddof=1)*100:.2f}'
    for method, shots in sorted({(k[0], k[1]) for k in latest}):
        row = dict(Method=method, shots=shots)
        for task in TASKS:
            cells = [latest.get((method, shots, task, seed)) for seed in seeds]
            row[task] = display([v[1]['accuracy'] for v in cells]) if all(cells) else 'incomplete'
        complete = all((method, shots, t, s) in latest for t in TASKS for s in seeds)
        row['Average'] = display([np.mean([latest[method, shots, t, s][1]['accuracy']
                                 for t in TASKS]) for s in seeds]) if complete else 'incomplete'
        rows.append(row)
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['Method', 'shots', *TASKS, 'Average'])
        writer.writeheader(); writer.writerows(rows)
    (output/'comparison_sources.json').write_text(json.dumps(
        [dict(summary=str(p.resolve()), method=r['method'], task=r['task'], shots=r['shots'], seed=r['seed'])
         for p, r in latest.values()], ensure_ascii=False, indent=2), encoding='utf-8')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=CROSS_CONFIG['result_root'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[3, 24, 38])
    parser.add_argument('--episodes', type=int, default=1000)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error('At least two distinct training seeds are required for sample SD')
    rows = aggregate(args.root, args.seeds, args.episodes)
    print(f'{len(rows)} method/shot rows -> {Path(args.root)/"comparison.csv"}; incomplete cells are not averaged')
