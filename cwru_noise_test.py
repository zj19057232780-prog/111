"""CWRU-only paired AWGN evaluation; never writes input PNGs or trains weights."""
import csv
import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import torch
from cwru_preprocess import load_metadata_ids, load_drive_end_signal
from pu_preprocess import sample_starts, compute_stft_log_image


def seed_torch(seed):
    # Quiet equivalent of the shared initializer; avoid one log line per episode.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_awgn(signal, snr_db, seed):
    x = np.asarray(signal, dtype=np.float64)
    if snr_db is None:
        return x.copy()
    if not math.isfinite(float(snr_db)):
        raise ValueError('SNR must be finite; use noise_include_clean for no added noise')
    centered = x - x.mean()
    power = np.mean(centered ** 2)
    if power <= 0:
        raise ValueError('Cannot define SNR for a zero-power signal')
    z = np.random.default_rng(seed).standard_normal(x.shape)
    z -= z.mean()
    noise = z * np.sqrt(power / (10 ** (float(snr_db)/10) * np.mean(z*z)))
    return x + noise


def load_target_windows(cfg):
    """Replay ALL original RNG draws; verify clean target windows against original PNGs."""
    metadata = load_metadata_ids(cfg['raw_data_root'])
    rng = np.random.default_rng(cfg.get('preprocess_seed', 24))
    sources = cfg['source_condition']
    sources = [sources] if isinstance(sources,str) else list(sources)
    windows, provenance = {}, []
    for condition, speed in cfg['condition_speeds'].items():
        if condition not in sources + [cfg['target_condition']]:
            continue
        for name, spec in cfg['class_specs'].items():
            raw, variable, mat = load_drive_end_signal(cfg['raw_data_root'], spec['subset'], speed,
                                                       spec['filename'], metadata, cfg)
            mid = len(raw)//2
            splits = [('val',raw[:mid],cfg['target_val_samples_per_class'],0),
                      ('test',raw[mid:],cfg['target_test_samples_per_class'],mid)] if condition == cfg['target_condition'] else [
                      ('source',raw,cfg['source_samples_per_class'],0)]
            for prefix, signal, count, offset in splits:
                starts = sample_starts(len(signal),cfg['window_size'],count,rng)
                if prefix != 'test' or name not in cfg['class_names']:
                    continue
                directory = Path(cfg['root_path'])/condition/name
                if len(list(directory.glob('test_*.png'))) != len(starts):
                    raise ValueError(f'Window/PNG count mismatch: {directory}')
                values = []
                for i,start in enumerate(starts):
                    x = signal[start:start+cfg['window_size']].copy()
                    path = directory/f'test_{i:03d}.png'
                    image = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                    if image is None or not np.array_equal(image,compute_stft_log_image(x,cfg)):
                        raise ValueError(f'Original raw window does not match {path}; check preprocessing settings')
                    values.append(x)
                    provenance.append(dict(class_name=name,sample=i,mat=str(mat),variable=variable,
                                           start=int(offset+start),length=len(x),png=str(path)))
                windows[name] = np.stack(values)
    if set(windows) != set(cfg['class_names']):
        raise ValueError('Incomplete CWRU target class pool')
    return windows, provenance


def make_manifest(cfg, windows, episodes):
    rng = np.random.default_rng(int(cfg.get('noise_episode_seed',2026)))
    classes = cfg['class_names']
    way, shot, query = (int(cfg[k]) for k in ('n_way','k_shot','q_query'))
    if min(way,shot,query,episodes)<=0 or way>len(classes):
        raise ValueError('Invalid few-shot task dimensions')
    if any(len(windows[c])<shot+query for c in classes):
        raise ValueError('Support plus query exceeds available target samples per class')
    result=[]
    for _ in range(episodes):
        chosen=rng.choice(len(classes),way,replace=False).tolist()
        indices=[rng.choice(len(windows[classes[c]]),shot+query,replace=False).tolist() for c in chosen]
        result.append(dict(classes=chosen,indices=indices))
    return result


def representation(cfg, windows, snr, seed):
    result={}
    for name in cfg['class_names']:
        values=[]
        for i,x in enumerate(windows[name]):
            # Stable per-window key, independent of model, SNR and class iteration order.
            key=f"{seed}|{cfg['target_condition']}|{name}|{i}"
            local_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
            noisy=add_awgn(x,snr,local_seed)
            if cfg['method']=='tl_wdcnn':
                value=((noisy-noisy.mean())/(noisy.std()+1e-8)).astype(np.float32)[None]
            else:
                value=compute_stft_log_image(noisy,cfg).astype(np.float32)[None]/255.
            values.append(value)
        result[name]=np.stack(values)
    return result


def episode_batch(data, item, classes, shot):
    support=[];query=[];sl=[];ql=[];original=[];sample_indices=[]
    for local,(c,idx) in enumerate(zip(item['classes'],item['indices'])):
        support.extend(data[classes[c]][idx[:shot]])
        query.extend(data[classes[c]][idx[shot:]])
        sl.extend([local]*shot); ql.extend([local]*(len(idx)-shot))
        original.extend([c]*(len(idx)-shot)); sample_indices.extend(idx[shot:])
    batch=(torch.from_numpy(np.stack(support)),torch.tensor(sl,dtype=torch.long),
           torch.from_numpy(np.stack(query)),torch.tensor(ql,dtype=torch.long))
    return batch,original,sample_indices


def write_csv(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run_noise_test(owner, checkpoint, quick=False, output_dir=None):
    cfg=owner.pu_config
    if 'class_groups' in cfg or 'class_specs' not in cfg:
        raise ValueError('Noise sweep is supported only for CWRU')
    levels=[]
    for snr in cfg.get('noise_snr_db',[10,8,6,4,2,0,-2,-4,-6,-8,-10]):
        snr=float(snr)
        if not math.isfinite(snr):raise ValueError('Noise SNR must be finite')
        if snr not in levels:levels.append(snr)
    if cfg.get('noise_include_clean',True):levels.insert(0,None)
    seeds=list(dict.fromkeys(int(s) for s in cfg.get('noise_seeds',[101,202,303])))
    if not levels or not seeds or min(seeds)<0:raise ValueError('Provide SNR levels and nonnegative noise seeds')
    if quick:seeds=seeds[:1]
    episodes=4 if quick else int(cfg.get('noise_test_episodes',1000))
    checkpoint=Path(checkpoint)
    metadata=Path(str(checkpoint)+'.config.json')
    if metadata.exists():
        saved=json.loads(metadata.read_text(encoding='utf-8'))
        if saved.get('method','configured_maml')!=cfg['method']:
            raise ValueError('Checkpoint method differs from noise-test method')
        for key in ['n_way','k_shot','class_names','source_condition','target_condition','window_size','stft_fs','q_query','stft_nperseg','stft_noverlap','img_size','preprocess_seed']:
            if key in saved and saved[key]!=cfg[key]:
                raise ValueError(f'Checkpoint {key} differs from noise-test config')
    else:
        print('Checkpoint has no config metadata; ensure task and preprocessing match manually.')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    owner.model.load_state_dict(state)
    windows,provenance=load_target_windows(cfg)
    manifest=[] if owner.method=='resnet18' else make_manifest(cfg,windows,episodes)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=Path(output_dir) if output_dir else Path(cfg.get('visualization_dir','results/cwru'))/'noise'/checkpoint.name/stamp
    out.mkdir(parents=True,exist_ok=False)
    manifest_text=json.dumps(manifest,sort_keys=True)
    digest=hashlib.sha256(manifest_text.encode()).hexdigest()
    (out/'episodes.json').write_text(manifest_text,encoding='utf-8')
    write_csv(out/'windows.csv',provenance)
    run_config=dict(cfg,checkpoint=str(checkpoint.resolve()),quick=quick,episode_manifest_sha256=digest,
                    noise_scope='support_and_query',snr_definition='demeaned_raw_window_power',
                    effective_noise_seeds=seeds,effective_levels=levels,effective_episodes=len(manifest),
                    evaluation='full_class_direct' if owner.method=='resnet18' else 'episodic')
    (out/'config.json').write_text(json.dumps(run_config,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    summaries=[]
    for snr in levels:
        tag='clean' if snr is None else f'{snr:+g}dB'
        for seed in ([seeds[0]] if snr is None else seeds):
            data=representation(cfg,windows,snr,seed)
            # No SNR/seed condition inherits model parameters or running statistics.
            owner.model.load_state_dict(state)
            owner.model.train()
            seed_torch(int(cfg.get('noise_episode_seed',2026)))
            rows=[];loss_sum=0.;count=0;correct=0
            if owner.method=='resnet18':
                from supervised_training import data_loader,evaluate
                loss,acc,_,truth,pred=evaluate(owner.model,data_loader(data,cfg['class_names'],int(cfg.get('supervised_batch_size',64))))
                for i,(t,p) in enumerate(zip(truth,pred)):
                    rows.append(dict(episode=-1,sample=i,true_class_index=int(t),pred_class_index=int(p)))
                loss_sum=loss*len(truth);count=len(truth);correct=int((truth==pred).sum())
            else:
                algorithm=owner._algorithm()
                for e,item in enumerate(manifest):
                    seed_torch(int(cfg.get('noise_episode_seed',2026))+e)
                    batch,truth,indices=episode_batch(data,item,cfg['class_names'],int(cfg['k_shot']))
                    err,_,_,_,scores=owner.fast_adapt(batch,algorithm.clone(),torch.nn.CrossEntropyLoss(),
                        int(cfg.get('test_inner_steps',10)),return_predictions=True)
                    pred=np.array(item['classes'])[scores.detach().argmax(1).cpu().numpy()]
                    loss_sum+=err.item()*len(truth);count+=len(truth);correct+=int((pred==truth).sum())
                    rows.extend(dict(episode=e,sample=int(i),true_class_index=int(t),pred_class_index=int(p))
                                for i,t,p in zip(indices,truth,pred))
                    if (e+1)%100==0: print(f'{tag} seed={seed} episodes={e+1}/{len(manifest)}',flush=True)
            write_csv(out/f'{tag}_seed{seed}_predictions.csv',rows)
            summaries.append(dict(snr_db='clean' if snr is None else snr,noise_seed='' if snr is None else seed,
                accuracy_percent=100*correct/count,loss=loss_sum/count,prediction_count=count,episodes=len(manifest)))
            write_csv(out/'summary.csv',summaries)
            print(f'{owner.method} {tag} seed={seed}: {100*correct/count:.4f}%',flush=True)
    aggregate=[]
    clean=next((r['accuracy_percent'] for r in summaries if r['snr_db']=='clean'),None)
    for snr in levels:
        key='clean' if snr is None else snr
        vals=[r['accuracy_percent'] for r in summaries if r['snr_db']==key]
        mean=float(np.mean(vals))
        aggregate.append(dict(snr_db=key,mean_accuracy_percent=mean,
            noise_seed_std_percent=float(np.std(vals,ddof=1)) if len(vals)>1 else '',
            repeats=len(vals),drop_from_clean_pp='' if clean is None else clean-mean))
    write_csv(out/'summary_mean_std.csv',aggregate)
    owner.model.load_state_dict(state)
    print(f'Noise results: {out.resolve()}')
    return out
