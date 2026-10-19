"""Cross AB: existing nine models, STFT/native raw, 1/3/5-shot; --test never trains."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from config_cross_dataset import BASE, CROSS_CONFIG, TASKS, METHODS, canonical_method, resolve_cross_config
from cross_dataset_data import CrossDataset, CrossTasks, preprocess_cross
from cross_dataset_models import build_model, episode, query_ce
from hust_preprocess import digest
from maml_train_pic import (device, capture_rng_state, restore_rng_state,
                           compute_ggm_rw_weights, freeze_bn_running_stats, MAML_learner)
from my_utils.init_utils import seed_torch


def write_json(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def code_files() -> dict:
    project = BASE.parent
    files = list(BASE.glob('*.py'))
    for folder in ('mu2025_amaml', 'wang2025_tubr_ml', 'li2025_mfeml'):
        files.extend((project/'comparison_models'/folder).glob('*.py'))
    # Include module blocks and the MAML clone implementation, excluding tests/runners
    # unrelated to this experiment. A source snapshot is stored beside every checkpoint.
    files.extend((BASE/'my_utils').glob('*.py'))
    return {str(p.relative_to(project)).replace('\\', '/'): digest(p)
            for p in sorted(set(files)) if not p.name.startswith('test_')}


@contextmanager
def evaluation_state(model, seed: int):
    """Restore all training RNG streams, BN buffers and submodule modes."""
    rng = capture_rng_state()
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    modes = {m: m.training for m in model.modules()}
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.train()
        yield
    finally:
        with torch.no_grad():
            for n, b in model.named_buffers():
                b.copy_(buffers[n])
        for m, flag in modes.items():
            m.training = flag
        restore_rng_state(rng)


class CrossLearner:
    def __init__(self, cfg: dict):
        self.pu_config = cfg
        self.method = cfg['method']
        self.model = build_model(cfg, device)
        self.storage = CrossDataset(cfg)

    def build_tasks(self, mode='train', ways=None, shots=None, queries=None,
                    num_tasks=1000, filter_labels=None):
        cfg = self.pu_config
        if filter_labels is not None or (ways is not None and ways != cfg['n_way']):
            raise ValueError('Cross AB episodes always use the five canonical classes')
        seed = cfg['seed'] if mode=='train' else cfg[f'{mode}_episode_seed']
        return CrossTasks(self.storage, mode, cfg['n_way'], shots or cfg['k_shot'],
                          queries or cfg['q_query'], num_tasks, seed, fixed=mode!='train')

    def run_episode(self, batch, training: bool, steps: int):
        return episode(self.model, batch, self.pu_config, training, steps)

    def evaluate(self, tasks, steps: int) -> tuple[float, float]:
        tasks.cursor = 0
        losses, accuracies = [], []
        for i in range(len(tasks.plans)):
            task = tasks.sample()
            with evaluation_state(self.model, self.pu_config['validation_episode_seed']+i):
                _, acc, _, labels, scores = self.run_episode(task, False, steps)
                losses.append(query_ce(scores.detach(), labels, self.method))
                accuracies.append(float(acc.detach()))
        return float(np.mean(losses)), float(np.mean(accuracies))

    def train(self, save_path, shots=None, quick_test=False) -> str:
        cfg = self.pu_config
        if shots is not None and shots != cfg['k_shot']:
            raise ValueError('Train shot differs from resolved config')
        best_path = Path(str(save_path)+'_best')
        history_path = Path(str(save_path)+'_history.json')
        if best_path.exists() or history_path.exists():
            raise FileExistsError(f'Run already exists: {save_path}; use a fresh path')
        best_path.parent.mkdir(parents=True, exist_ok=True)
        hashes = code_files()
        snapshot = Path(str(save_path)+'_source')
        if snapshot.exists():
            raise FileExistsError(f'Source snapshot already exists: {snapshot}')
        for rel in hashes:
            target = snapshot/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BASE.parent/rel, target)
        write_json(str(save_path)+'_resolved_config.json', cfg)
        transfer = self.method=='tl_wdcnn'
        train = self.build_tasks('train') if not transfer else None
        val = self.build_tasks('validation', num_tasks=cfg['validation_episodes'])
        epochs = cfg['ft_pretrain_epochs'] if transfer else cfg['epochs']
        interval = cfg['ft_validation_interval'] if transfer else cfg['validation_interval']
        patience = 0 if self.method=='configured_maml' else cfg['early_stop_patience']
        optimizer = torch.optim.Adam(self.model.parameters(),
                     lr=cfg['ft_pretrain_lr'] if transfer else cfg['outer_lr'],
                     weight_decay=cfg['ft_weight_decay'] if transfer else 0.)
        scheduler = None if transfer else torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=cfg['outer_lr_min'])
        if transfer:
            groups = self.storage.images('train')
            x = torch.from_numpy(np.concatenate(list(groups.values())))
            y = torch.cat([torch.full((len(v),), i, dtype=torch.long) for i,v in enumerate(groups.values())])
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),
                       batch_size=cfg['ft_pretrain_batch_size'], shuffle=True)
        best_acc, best_loss, history = -1., float('inf'), []
        best_epoch = best_accuracy_epoch = 0
        stop_reason = 'epochs_limit'
        self.model.train()
        for epoch in range(epochs):
            started = time.time()
            accuracies = []
            if transfer:
                for batch_index, (x,y) in enumerate(loader):
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    scores = self.model(x)[1]
                    error = torch.nn.functional.cross_entropy(scores,y)
                    if not torch.isfinite(error):
                        raise FloatingPointError('Nonfinite supervised source loss')
                    error.backward()
                    optimizer.step()
                    accuracies.append(float((scores.argmax(1)==y).float().mean()))
                    if cfg['ft_pretrain_max_batches'] and batch_index+1>=cfg['ft_pretrain_max_batches']:
                        break
            else:
                batch = [train.sample() for _ in range(cfg['meta_batch_size'])]
                weights = torch.full((len(batch),),1/len(batch),device=device)
                if cfg['task_weighting_mode']=='ggm_rw':
                    rng = capture_rng_state()
                    try:
                        with freeze_bn_running_stats(self.model):
                            gaps = [MAML_learner.pilot_generalization_gap(t,self.model,
                                      torch.nn.CrossEntropyLoss(),cfg['adaptation_steps'][cfg['k_shot']],
                                      cfg['inner_lr']) for t in batch]
                    finally:
                        restore_rng_state(rng)
                    weights, _ = compute_ggm_rw_weights(gaps,epoch,len(batch),
                        cfg['task_weight_temperature'],cfg['task_weight_alpha_max'],
                        cfg['task_weight_warmup_epochs'],cfg['task_weight_z_clip'],
                        cfg['task_weight_max_ratio'],cfg['task_weight_eps'])
                optimizer.zero_grad(set_to_none=True)
                for weight, task in zip(weights,batch):
                    error, acc, _, _, _ = self.run_episode(task,True,cfg['adaptation_steps'][cfg['k_shot']])
                    if not torch.isfinite(error):
                        raise FloatingPointError('Nonfinite meta-training loss')
                    (weight*error).backward()
                    accuracies.append(float(acc.detach()))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),cfg['outer_grad_clip_norm'],error_if_nonfinite=True)
                optimizer.step()
            row = dict(epoch=epoch+1,train_accuracy=float(np.mean(accuracies)),
                       lr=optimizer.param_groups[0]['lr'])
            if (epoch+1)%interval==0 or epoch+1==epochs:
                val_loss,val_acc = self.evaluate(val,cfg['validation_inner_steps'])
                if not np.isfinite(val_loss) or not np.isfinite(val_acc):
                    raise FloatingPointError('Nonfinite target validation')
                row.update(validation_accuracy=val_acc, validation_loss=val_loss)
                # Count training epochs, not validation calls. A lower loss at
                # the same accuracy may select a checkpoint but never resets patience.
                if val_acc > best_acc:
                    best_accuracy_epoch = epoch+1
                if val_acc>best_acc or (val_acc==best_acc and val_loss<best_loss):
                    best_acc,best_loss = val_acc,val_loss
                    best_epoch = epoch+1
                    torch.save(self.model.state_dict(),best_path)
                    write_json(str(best_path)+'.config.json',dict(cfg,actual_shots=cfg['k_shot'],
                        data_sha256=self.storage.signature,best_epoch=epoch+1,
                        validation_accuracy=best_acc,validation_loss=best_loss,
                        quick_test=quick_test,code_hashes=hashes,
                        code_snapshot=str(snapshot.resolve()),checkpoint_sha256=digest(best_path)))
            patience_stop = bool(patience and 'validation_accuracy' in row and
                                 epoch+1-best_accuracy_epoch >= patience)
            row.update(best_epoch=best_epoch, best_accuracy_epoch=best_accuracy_epoch,
                       epochs_without_improvement=epoch+1-best_accuracy_epoch,
                       patience_stop_triggered=patience_stop)
            row['seconds'] = time.time()-started
            history.append(row)
            write_json(history_path,history)
            if scheduler is not None:
                scheduler.step()
            if 'validation_accuracy' in row:
                print(f"{cfg['model_name']} update {epoch+1}/{epochs}: val={row['validation_accuracy']:.4f}; "
                      f"best={best_acc:.4f}@{best_epoch}; no_improvement={row['epochs_without_improvement']}; "
                      f"patience={patience}",flush=True)
            if patience_stop:
                stop_reason = 'validation_patience'
                print(f'Early stopping at {epoch+1}: validation accuracy has not strictly improved '
                      f'for {row["epochs_without_improvement"]} epochs; using {best_path}',flush=True)
                break
        training = dict(epochs_run=len(history), epoch_limit=epochs, best_epoch=best_epoch,
                        best_accuracy_epoch=best_accuracy_epoch, early_stop_patience=patience,
                        epochs_without_improvement=history[-1]['epochs_without_improvement'],
                        stop_reason=stop_reason, training_complete=True)
        write_json(str(save_path)+'_training_summary.json',training)
        saved = json.loads(Path(str(best_path)+'.config.json').read_text(encoding='utf-8'))
        write_json(str(best_path)+'.config.json',dict(saved, **training))
        return str(best_path)

    def test(self, load_path) -> dict:
        cfg = self.pu_config
        saved = json.loads(Path(str(load_path)+'.config.json').read_text(encoding='utf-8'))
        if (saved.get('experiment_signature')!=cfg['experiment_signature'] or
            saved.get('data_sha256')!=self.storage.signature or saved.get('actual_shots')!=cfg['k_shot']):
            raise ValueError('Checkpoint/config/data mismatch (including legacy 3-way)')
        if digest(load_path)!=saved['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        if code_files()!=saved['code_hashes']:
            raise ValueError('Source code changed since training; use the saved source snapshot')
        self.model.load_state_dict(torch.load(load_path,map_location=device,weights_only=True))
        tasks = self.build_tasks('test',num_tasks=cfg['test_meta_batch_size'])
        true,pred,rows,accuracies = [],[],[],[]
        started = time.time()
        for i in range(cfg['test_meta_batch_size']):
            task = tasks.sample_with_original_labels()
            with evaluation_state(self.model,cfg['test_episode_seed']+i):
                error,acc,_,_,scores = self.run_episode(task,False,cfg['test_inner_steps'])
                if not torch.isfinite(error):
                    raise FloatingPointError('Nonfinite test loss')
                predicted = task[6][scores.detach().argmax(1).cpu()].tolist()
            actual = task[5].tolist()
            true.extend(actual);pred.extend(predicted);accuracies.append(float(acc.detach()))
            ids = [v for s in tasks.plans[i]['selections'] for v in s['query_ids']]
            rows.extend(dict(episode=i,window=v,true_label=a,predicted_label=p)
                        for v,a,p in zip(ids,actual,predicted))
        labels = list(range(cfg['n_way']))
        result = dict(protocol=cfg['protocol'],task=cfg['task'],source=cfg['source_domain'],
            target=cfg['target_domain'],source_conditions=cfg['source_conditions'],
            target_condition=cfg['target_condition'],target_condition_name=cfg['target_condition_name'],
            method=self.method,input_mode=cfg['input_mode'],
            native_input=cfg['native_input'],shots=cfg['k_shot'],seed=cfg['seed'],
            accuracy=float(np.mean(accuracies)),macro_f1=float(f1_score(true,pred,labels=labels,average='macro',zero_division=0)),
            confusion_matrix=confusion_matrix(true,pred,labels=labels).tolist(),
            episodes=cfg['test_meta_batch_size'],n_way=cfg['n_way'],query_per_class=cfg['q_query'],
            class_names=cfg['class_names'],episode_accuracies=accuracies,
            checkpoint=str(Path(load_path).resolve()),checkpoint_sha256=digest(load_path),
            best_epoch=saved.get('best_epoch'), epochs_run=saved.get('epochs_run'),
            stop_reason=saved.get('stop_reason'), early_stop_patience=saved.get('early_stop_patience'),
            best_accuracy_epoch=saved.get('best_accuracy_epoch'),
            experiment_signature=cfg['experiment_signature'],data_sha256=self.storage.signature,
            code_hashes=saved['code_hashes'],episode_manifest=str(tasks.path.resolve()),
            episode_sha256=digest(tasks.path),quick_checkpoint=saved['quick_test'],seconds=time.time()-started)
        output = Path(cfg['result_root'])/cfg['model_name']/str(time.time_ns())
        write_json(output/'summary.json',result)
        write_json(output/'config.json',cfg)
        with (output/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=['episode','window','true_label','predicted_label'])
            writer.writeheader();writer.writerows(rows)
        print(f"Test {cfg['task']}/{self.method}/{cfg['input_mode']}: accuracy={result['accuracy']:.4f}; {output}",flush=True)
        return result


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=[*TASKS,'all'])
    p.add_argument('--method',type=canonical_method,choices=[*METHODS,'all'])
    p.add_argument('--input',dest='input_mode',choices=['stft','nostft','no_stft'])
    p.add_argument('--shots',type=int,choices=[1,3,5])
    p.add_argument('--seed',type=int)
    p.add_argument('--processed-root');p.add_argument('--checkpoint-root');p.add_argument('--result-root')
    p.add_argument('--model-path',help='Base checkpoint path; _best is appended')
    p.add_argument('--cwru-raw-root');p.add_argument('--pu-raw-root')
    p.add_argument('--preprocess',action='store_true')
    p.add_argument('--train',action='store_true');p.add_argument('--test',action='store_true')
    p.add_argument('--quick',action='store_true',help='Isolated one-update smoke, not paper evidence')
    p.add_argument('--test-episodes',type=int);p.add_argument('--epochs',type=int)
    p.add_argument('--patience',type=int,help='Epochs without strict validation-accuracy improvement; '
                   'comparisons default 100, 0 disables; SCG-MAML always disables')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(argv)
    if args.patience is not None and args.patience < 0:
        p.error('--patience must be non-negative')
    if args.threads<=0 or (args.model_path and (args.task=='all' or args.method=='all')):
        p.error('Positive threads; --model-path requires one task and one method')
    return args


def main(argv=None):
    args=parse_args(argv)
    torch.set_num_threads(args.threads)
    train,test=args.train,args.test
    if not (args.preprocess or train or test):
        train=CROSS_CONFIG['run_mode'] in ('train','train_test')
        test=CROSS_CONFIG['run_mode'] in ('test','train_test')
    overrides={k:v for k,v in dict(seed=args.seed,k_shot=args.shots,input_mode=args.input_mode,
        processed_root=args.processed_root,checkpoint_root=args.checkpoint_root,result_root=args.result_root,
        cwru_raw_root=args.cwru_raw_root,pu_raw_root=args.pu_raw_root,
        test_meta_batch_size=args.test_episodes,epochs=args.epochs,
        early_stop_patience=args.patience).items() if v is not None}
    if args.quick:
        overrides.update(epochs=1,meta_batch_size=1,validation_episodes=1,validation_interval=1,
                         ft_pretrain_epochs=1,ft_validation_interval=1,ft_pretrain_max_batches=1,
                         validation_inner_steps=1,test_inner_steps=1,ft_steps=1,
                         adaptation_steps={1:1,3:1,5:1})
        overrides.setdefault('test_meta_batch_size',2)
    task_choice=args.task or CROSS_CONFIG['task']
    tasks=TASKS if task_choice=='all' else [task_choice]
    method_choice=args.method or CROSS_CONFIG['method']
    methods=list(METHODS) if method_choice=='all' else [method_choice]
    input_mode=overrides.get('input_mode',CROSS_CONFIG['input_mode']).replace('no_stft','nostft')
    if input_mode=='nostft' and method_choice=='all':
        methods.remove('configured_maml')
        print('SCG-MAML omitted from nostft sweep: no existing native implementation',flush=True)
    first=True
    for task in tasks:
        for method in methods:
            cfg=resolve_cross_config(dict(overrides,task=task,method=method))
            print(f"{task}: {cfg['source_domain']} {cfg['source_conditions']} -> "
                  f"{cfg['target_domain']} {cfg['target_condition']} ({cfg['target_condition_name']}); {method}; "
                  f"{cfg['input_mode']}; 5-way {cfg['k_shot']}-shot; device={device}; "
                  f"patience={cfg['early_stop_patience']}",flush=True)
            if args.preprocess and first:
                preprocess_cross(cfg)
            first=False
            if not (train or test):
                continue
            seed_torch(cfg['seed'])
            path=args.model_path or str(Path(cfg['checkpoint_root'])/(cfg['model_name']+('_quick' if args.quick else '')))
            learner=CrossLearner(cfg)
            best=learner.train(path,quick_test=args.quick) if train else path+'_best'
            if test:
                learner.test(best)


if __name__=='__main__':
    main()
