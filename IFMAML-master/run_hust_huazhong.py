"""Run Huazhong HUSTbearing single-source H1-H6 using the unchanged PU MAML trainer."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from config_hust_huazhong import TASKS, MODELS, SEEDS, PAPERS, BASELINES, configuration
from hust_huazhong_preprocess import preprocess, verify_cache, sha256, save_json

ROOT=Path(__file__).resolve().parent


def source_hashes():
    paths=[p for p in ROOT.glob('*.py') if not p.name.startswith('test_')]
    paths+=list((ROOT/'my_utils').glob('*.py'))+list((ROOT/'optional_models').glob('*.py'))
    for folder in PAPERS.values():
        paths+=list((ROOT.parent/'comparison_models'/folder).glob('*.py'))
        paths+=list((ROOT.parent/'comparison_models'/folder).glob('*.json'))
    return {p.relative_to(ROOT.parent).as_posix():sha256(p) for p in sorted(paths)}


def snapshot(out):
    hashes=source_hashes()
    save_json(out/'hust_code_sha256.json',hashes)
    for relative in hashes:
        dest=out/'source_snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT.parent/relative).read_bytes())


def paper_config(cfg,action):
    from types import SimpleNamespace
    from run_experiment import load_paper
    model=cfg['model_variant'];module=load_paper(model)
    args=SimpleNamespace(dataset='pu',shot=cfg['k_shot'],epochs=cfg['epochs'],
        input='no_stft' if cfg['input_mode']=='nostft' else 'stft',action=action,
        data_root=cfg['raw_root'] if cfg['input_mode']=='nostft' else cfg['root_path'],data_map=None,
        patience=cfg['early_stop_patience'],inner_lr=.1 if model=='mfeml' else .01,
        outer_lr=.01 if model=='mfeml' else .001,grad_clip=1.,train_steps=10,test_steps=10,meta_batch_size=16)
    result=module.configuration(args,'P1',cfg['seed'])
    keys=('dataset','task','protocol','model_variant','source_condition','target_condition','class_names',
          'class_groups','classification_preset','root_path','raw_root','channel','validation_domain',
          'target_validation_fraction','split_gap_windows','normalize','q_query')
    result.update({k:v for k,v in cfg.items() if k in keys or k.startswith('stft_')})
    result.update(train_only=action=='train',early_stop_on_perfect_validation=False)
    return module,result


def worker(job):
    import torch
    import maml_train_pic as training
    from run_table_row import select_device
    from my_utils.init_utils import seed_torch
    training.device=torch.device(select_device(job['device']))
    torch.set_num_threads(job['threads'])
    from hust_huazhong_inputs import create_learner
    from experiment_support import data_signature, evaluate_classic
    cfg=job['config']
    cfg['adaptation_steps']={int(k):v for k,v in cfg['adaptation_steps'].items()}
    seed_torch(cfg['seed'])
    out=Path(job['out'])
    source=Path(job['source']) if job.get('source') else None
    if source and source_hashes()!=json.loads((source/'hust_code_sha256.json').read_text(encoding='utf-8')):
        raise ValueError('Source changed since training; use saved source_snapshot')
    if cfg['model_variant'] in PAPERS:
        if source:
            from run_experiment import load_paper
            load_paper(cfg['model_variant']).test_saved(source,training.device)
            return
        module,cfg=paper_config(cfg,job['action'])
        try:
            result=module.train_one(cfg,module.MethodConfig(),training.device,out)
            result.update(paper_method=result['method'],method=cfg['model_variant'],
                          protocol=cfg['protocol'],validation_domain='target')
            save_json(out/'result.json',result)
        finally:
            if out.exists():snapshot(out)
        return
    out.mkdir(parents=True,exist_ok=False)
    cfg['visualization_dir']=str(out/'artifacts')
    snapshot(out)
    learner=create_learner(cfg)
    signature=data_signature(learner._get_pu_storage(),cfg)
    save_json(out/'data_signature.json',signature)
    save_json(out/'resolved_config.json',learner.pu_config)
    if hasattr(learner._get_pu_storage(),'window_manifest'):
        save_json(out/'raw_window_manifest.json',learner._get_pu_storage().window_manifest)
    else:
        manifest=Path(cfg['root_path'])/'manifest.json'
        save_json(out/'data_manifest_identity.json',{'path':str(manifest),'sha256':sha256(manifest)})
    if source:
        if signature!=json.loads((source/'data_signature.json').read_text(encoding='utf-8')):
            raise ValueError('Data differs from training')
        saved=json.loads((source/'result.json').read_text(encoding='utf-8'))
        best=source/'model_best'
        if sha256(best)!=saved['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
    else:
        best=Path(learner.train(str(out/'model'),shots=cfg['k_shot']))
    result=dict(status='TRAINED',dataset='hust_huazhong',protocol=cfg['protocol'],
        task=cfg['task'],seed=cfg['seed'],method=cfg['model_variant'],shot=cfg['k_shot'],input_mode=cfg['input_mode'],
        checkpoint=str(best),checkpoint_sha256=sha256(best),validation_domain='target')
    save_json(out/'result.json',result)
    if job['action']!='train':
        learner=create_learner(cfg)
        result.update(evaluate_classic(learner,best,cfg,out))
        result['status']='SMOKE_ONLY' if cfg['smoke'] else 'OK'
        save_json(out/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def summary(records):
    from run_experiment import summarize
    return {model:summarize([r for r in records if r['method']==model],list(TASKS)) for model in MODELS}


def write_summary(batch, records):
    report=summary(records)
    save_json(batch/'summary.json',report)
    fields=['model','metric',*TASKS,'all_conditions']
    rows=[]
    for model, metrics in report.items():
        for metric, values in metrics.items():
            row={'model':model,'metric':metric}
            for field in fields[2:]:
                v=values.get(field,'incomplete')
                row[field]=f"{v['mean']:.4f} ± {v['sample_sd']:.4f}" if isinstance(v,dict) else 'incomplete'
            rows.append(row)
    with (batch/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# HUST Huazhong target-validation results', '',
        'Mean ± sample SD across seeds 3/24/38; all_conditions averages tasks within each seed first.', '',
        '| '+' | '.join(fields)+' |','|'+'---|'*len(fields)]
    lines+=['| '+' | '.join(str(row[k]) for k in fields)+' |' for row in rows]
    (batch/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--tasks','--task',nargs='+',default=['all'])
    p.add_argument('--models','--model',nargs='+',default=['full'])
    p.add_argument('--seeds','--seed',nargs='+',type=int,default=list(SEEDS))
    p.add_argument('--shots',type=int,choices=(1,3,5),default=1)
    p.add_argument('--channel',choices=('X','Y','Z'),default='X')
    p.add_argument('--input',choices=('stft','nostft'),default='stft')
    p.add_argument('--patience',type=int,default=100)
    p.add_argument('--raw-root',type=Path)
    p.add_argument('--processed-root',type=Path)
    p.add_argument('--output-root',type=Path,default=ROOT/'hust_huazhong_results')
    p.add_argument('--action',choices=('check','smoke','train','train-test','test'),default='check')
    p.add_argument('--run-dir',type=Path,help='Saved task/seed directory for test only')
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args(argv)
    if a.worker:
        worker(json.loads(a.worker.read_text(encoding='utf-8')));return 0
    if a.epochs<1 or a.threads<1 or any(s<0 or s>=2**32 for s in a.seeds) or len(set(a.seeds))!=len(a.seeds):
        p.error('Invalid epochs/threads/seeds')
    if a.action=='test':
        if not a.run_dir:p.error('--run-dir required for test')
        source=a.run_dir.resolve()
        cfg=json.loads((source/'resolved_config.json').read_text(encoding='utf-8'))
        worker(dict(config=cfg,source=str(source),out=str(source/('eval_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))),
                    action='test',device=a.device,threads=a.threads))
        return 0
    tasks=[t.upper() for value in a.tasks for t in value.split(',')]
    models=[m.lower().replace('-','_') for value in a.models for m in value.split(',')]
    models=['full' if m in ('scg_maml','configured_maml') else m for m in models]
    tasks=list(TASKS) if tasks==['ALL'] else tasks
    if models == ['all']:
        models=[m for m in BASELINES if not (a.shots == 3 and m == 'tubr_ml')]
        models+=['full'] if a.input=='stft' else []
    if set(tasks)-TASKS.keys() or len(set(tasks))!=len(tasks):p.error('Tasks must be H1-H6, unique')
    if set(models)-MODELS.keys() or len(set(models))!=len(models):p.error('Models: cnn4/lsk/lsk_gc/full/all')
    try:
        for model in models:configuration(tasks[0],model,a.seeds[0],a.shots,a.channel,a.input,a.patience)
    except ValueError as exc:p.error(str(exc))
    cfg=configuration(tasks[0],models[0],a.seeds[0],a.shots,a.channel,a.input,a.patience)
    if a.raw_root:cfg['raw_root']=str(a.raw_root.resolve())
    if a.processed_root:cfg['root_path']=str(a.processed_root.resolve())
    if a.input=='stft':preprocess(cfg)
    if a.action=='check':
        from hust_huazhong_inputs import storage_for
        for task in tasks:
            for model in models:
                c=configuration(task,model,a.seeds[0],a.shots,a.channel,a.input,a.patience)
                c.update(root_path=cfg['root_path'],raw_root=cfg['raw_root'])
                ds=storage_for(c)
                counts=[min(map(len,getattr(ds,k).values())) for k in ('source_data','target_val_data','target_test_data')]
                print(task,model,a.input,counts,'sample shape',ds.source_data['healthy'].shape[1:])
        return 0
    batch=a.output_root.resolve()/('_smoke' if a.action=='smoke' else 'formal')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    batch.mkdir(parents=True,exist_ok=False)
    records=[];failed=[]
    for model in models:
        for task in tasks:
            for seed in a.seeds:
                c=configuration(task,model,seed,a.shots,a.channel,a.input,a.patience)
                c.update(root_path=cfg['root_path'],raw_root=cfg['raw_root'],epochs=a.epochs)
                if a.action=='smoke':
                    c.update(smoke=True,epochs=1,meta_batch_size=2,tasks_per_epoch=2,test_meta_batch_size=2,
                             ft_pretrain_epochs=1,ft_pretrain_max_batches=2,ft_validation_episodes=1,ft_steps=1)
                out=batch/a.input/model/task/f'seed{seed}'
                job=dict(config=c,out=str(out),action=a.action,device=a.device,threads=a.threads)
                jobpath=batch/f'{model}_{task}_seed{seed}.job.json';save_json(jobpath,job)
                with jobpath.with_suffix('.log').open('w',encoding='utf-8') as log:
                    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
                    child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',str(jobpath)],
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
                    for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    code=child.wait()
                if code:failed.append(str(jobpath))
                elif (out/'result.json').is_file():records.append(json.loads((out/'result.json').read_text(encoding='utf-8')))
                save_json(batch/'runs.json',records);write_summary(batch,records)
    save_json(batch/'failures.json',failed)
    print(f'Output: {batch}',flush=True)
    return int(bool(failed))

if __name__=='__main__':
    os.environ.setdefault('MPLBACKEND','Agg')
    raise SystemExit(main())
