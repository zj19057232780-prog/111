"""Summarize AB five-class results, separately by representation/direction/metric."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from config_cross_dataset import CROSS_CONFIG, TASKS, TASK_GROUPS, CLASS_NAMES, CWRU_CONDITIONS, PU_CONDITIONS


def aggregate(root, seeds=(3,24,38), episodes=1000):
    if len(set(seeds))!=len(seeds) or len(seeds)<2:
        raise ValueError('At least two distinct training seeds required')
    latest={}
    for path in sorted(Path(root).glob('*/*/summary.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        if (r.get('protocol')!=CROSS_CONFIG['protocol'] or r.get('n_way')!=5 or
            r.get('class_names')!=CLASS_NAMES or r.get('quick_checkpoint') or
            r.get('episodes')!=episodes or r.get('seed') not in seeds):
            continue
        if r.get('input_mode') not in ('stft','nostft') or r.get('task') not in TASKS or r.get('shots') not in (1,3,5):
            raise ValueError(f'Invalid experiment identity: {path}')
        source, target, condition = TASKS[r['task']]
        source_conditions = list(CWRU_CONDITIONS if source=='A' else PU_CONDITIONS)
        if (r.get('source')!=source or r.get('target')!=target or
            r.get('source_conditions')!=source_conditions or r.get('target_condition')!=condition):
            raise ValueError(f'Wrong multi-source/single-target roles: {path}')
        key=(r['input_mode'],r['method'],r['shots'],r['task'],r['seed'])
        if key in latest and latest[key][1]['experiment_signature']!=r['experiment_signature']:
            raise ValueError(f'Multiple incompatible training configs for {key}; use separate roots')
        if key not in latest or path.stat().st_mtime_ns>latest[key][0].stat().st_mtime_ns:
            latest[key]=(path,r)
    # Underlying native anchors / labels / episode choices must match for comparisons.
    for task in TASKS:
        for shots in (1,3,5):
            rs=[r for _,r in latest.values() if r['task']==task and r['shots']==shots]
            sig={(r['data_sha256'],r['query_per_class'],r['episode_sha256']) for r in rs}
            if len(sig)>1:
                raise ValueError(f'Incomparable data/episode settings: {task}/{shots}')
    def display(values):
        return f'{np.mean(values)*100:.2f} +/- {np.std(values,ddof=1)*100:.2f}'
    rows=[]
    for mode,method,shots in sorted({k[:3] for k in latest}):
        row=dict(Input=mode,Method=method,shots=shots)
        for task in TASKS:
            cells=[latest.get((mode,method,shots,task,s)) for s in seeds]
            for metric in ('accuracy','macro_f1'):
                row[f'{task}_{metric}']=display([c[1][metric] for c in cells]) if all(cells) else 'incomplete'
        # Average the four target conditions within each seed, then compute seed SD.
        for direction, tasks in TASK_GROUPS.items():
            complete = all((mode,method,shots,t,s) in latest for t in tasks for s in seeds)
            for metric in ('accuracy','macro_f1'):
                row[f'{direction}_{metric}'] = display([
                    np.mean([latest[mode,method,shots,t,s][1][metric] for t in tasks])
                    for s in seeds]) if complete else 'incomplete'
        rows.append(row)
    output=Path(root);output.mkdir(parents=True,exist_ok=True)
    fields=['Input','Method','shots',*[f'{t}_{m}' for t in [*TASKS,*TASK_GROUPS] for m in ('accuracy','macro_f1')]]
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (output/'comparison_sources.json').write_text(json.dumps([
        dict(summary=str(p.resolve()),method=r['method'],input_mode=r['input_mode'],task=r['task'],
             source_conditions=r['source_conditions'],target_condition=r['target_condition'],
             shots=r['shots'],seed=r['seed']) for p,r in latest.values()],ensure_ascii=False,indent=2),encoding='utf-8')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=CROSS_CONFIG['result_root'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[3,24,38])
    parser.add_argument('--episodes',type=int,default=1000)
    args=parser.parse_args()
    rows=aggregate(args.root,args.seeds,args.episodes)
    print(f'{len(rows)} representation/method/shot rows -> {Path(args.root)/"comparison.csv"}')
