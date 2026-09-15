"""Independent cross-dataset CLI; --test never initiates training."""
import argparse
import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from config_cross_dataset import CROSS_CONFIG, TASKS, METHODS, resolve_cross_config
from cross_dataset_data import CrossDataset, CrossTasks, preprocess_cross
from run_hust_maml import HUSTMAMLLearner, HUSTMAML, write_json
from maml_train_pic import MAML_learner, device, capture_rng_state, restore_rng_state
from classic_models import MetricAlgorithm
from transfer_models import TransferAlgorithm
from my_utils.init_utils import seed_torch


@contextmanager
def evaluation_state(model, seed):
    """Validation/test cannot advance training RNG or change shared BN buffers."""
    rng = capture_rng_state()
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    modes = {m: m.training for m in model.modules()}
    try:
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


class CrossLearner(HUSTMAMLLearner):
    def __init__(self, cfg):
        MAML_learner.__init__(self, cfg['n_way'], cfg)
        self.pu_config['model_name'] = cfg['model_name']
        self.storage = CrossDataset(cfg)

    def _algorithm(self):
        if self.method == 'protonet':
            return MetricAlgorithm(self.model)
        if self.method == 'tl_wdcnn':
            return TransferAlgorithm(self.model, self.pu_config)
        return HUSTMAML(self.model, self.pu_config['inner_lr'])

    def build_tasks(self, mode='train', ways=None, shots=None, queries=None,
                    num_tasks=1000, filter_labels=None):
        cfg = self.pu_config
        if filter_labels is not None or (ways is not None and ways != 3):
            raise ValueError('Cross episodes use exactly N/I/O')
        seed = cfg['seed'] if mode == 'train' else cfg[f'{"validation" if mode == "validation" else "test"}_episode_seed']
        return CrossTasks(self.storage, mode, 3, shots or cfg['k_shot'],
                          queries or cfg['q_query'], num_tasks, seed, fixed=mode != 'train')

    def evaluate_meta_tasks(self, tasks, algorithm, loss, adaptation_steps, num_episodes):
        errors, accuracies = [], []
        for _ in range(num_episodes):
            index = tasks.cursor
            task = tasks.sample()
            with evaluation_state(self.model, self.pu_config['validation_episode_seed']+index):
                error, acc, _, _ = self.fast_adapt(task, algorithm.clone(), loss, adaptation_steps)
                errors.append(error.item()); accuracies.append(acc.item())
        return float(np.mean(errors)), float(np.mean(accuracies))

    def train(self, save_path, shots=None, quick_test=False):
        shots = shots or self.pu_config['k_shot']
        if self.method != 'tl_wdcnn':
            # Existing formal second-order MAML/PAG-R loop, with cross-domain task provider.
            return super().train(save_path, shots, quick_test)
        cfg = self.pu_config
        best_path = str(save_path)+'_best'
        if Path(best_path).exists():
            raise FileExistsError(f'Checkpoint exists: {best_path}; select a new --model-path')
        Path(best_path).parent.mkdir(parents=True, exist_ok=True)
        groups = self.storage.images('train')
        x = torch.from_numpy(np.concatenate(list(groups.values())))
        y = torch.cat([torch.full((len(v),), i, dtype=torch.long) for i, v in enumerate(groups.values())])
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                    batch_size=cfg['ft_pretrain_batch_size'], shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg['ft_pretrain_lr'],
                               weight_decay=cfg['ft_weight_decay'])
        val = self.build_tasks('validation', shots=shots, num_tasks=cfg['validation_episodes'])
        best, history = -1., []
        for epoch in range(cfg['ft_pretrain_epochs']):
            started = time.time()
            self.model.train()
            total, correct = 0, 0
            for step, (bx, by) in enumerate(loader):
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad(set_to_none=True)
                logits = self.model(bx)[1]
                loss = torch.nn.functional.cross_entropy(logits, by)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite source supervised loss')
                loss.backward(); opt.step()
                total += len(by); correct += (logits.argmax(1) == by).sum().item()
                if cfg['ft_pretrain_max_batches'] and step+1 >= cfg['ft_pretrain_max_batches']:
                    break
            val.cursor = 0
            error, acc = self.evaluate_meta_tasks(val, self._algorithm(), torch.nn.CrossEntropyLoss(),
                                                  cfg['ft_steps'], cfg['validation_episodes'])
            if not np.isfinite(error):
                raise FloatingPointError('Nonfinite target validation')
            if acc > best:
                best = acc
                torch.save(self.model.state_dict(), best_path)
                write_json(best_path+'.config.json', dict(cfg, actual_shots=shots,
                    data_sha256=self.storage.signature, best_epoch=epoch+1,
                    validation_accuracy=best, quick_test=quick_test))
            history.append(dict(epoch=epoch+1, train_accuracy=correct/total,
                                validation_accuracy=acc, validation_loss=error,
                                seconds=time.time()-started))
            write_json(str(save_path)+'_history.json', history)
            print(f'TL-WDCNN epoch {epoch+1}/{cfg["ft_pretrain_epochs"]}: source={correct/total:.4f}, target-val={acc:.4f}', flush=True)
        return best_path

    def test(self, load_path):
        cfg = self.pu_config
        saved = json.loads(Path(str(load_path)+'.config.json').read_text(encoding='utf-8'))
        if saved.get('experiment_signature') != cfg['experiment_signature'] or saved['data_sha256'] != self.storage.signature:
            raise ValueError('Checkpoint/config/data mismatch; use its original cross configuration')
        if saved['actual_shots'] != cfg['k_shot']:
            raise ValueError('Checkpoint shot mismatch')
        self.model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
        count = cfg['test_meta_batch_size']
        tasks = self.build_tasks('test', num_tasks=count)
        algorithm, loss = self._algorithm(), torch.nn.CrossEntropyLoss()
        true, pred, rows, accuracies = [], [], [], []
        start = time.time()
        for i in range(count):
            task = tasks.sample_with_original_labels()
            with evaluation_state(self.model, cfg['test_episode_seed']+i):
                error, acc, _, _, scores = self.fast_adapt(task[:4], algorithm.clone(), loss,
                                       cfg['test_inner_steps'], return_predictions=True)
                if not torch.isfinite(error):
                    raise FloatingPointError('Nonfinite cross test loss')
                predicted = task[6][scores.detach().argmax(1).cpu()].tolist()
                actual = task[5].tolist()
            true.extend(actual); pred.extend(predicted); accuracies.append(acc.item())
            ids = [v for s in tasks.plans[i]['selections'] for v in s['query_ids']]
            rows.extend(dict(episode=i, image=v, true_label=a, predicted_label=p)
                        for v, a, p in zip(ids, actual, predicted))
        result = dict(task=cfg['task'], source=cfg['source_domain'], target=cfg['target_domain'],
                      method=cfg['method'], shots=cfg['k_shot'], seed=cfg['seed'],
                      accuracy=float(np.mean(accuracies)), episodes=count,
                      n_way=3, query_per_class=cfg['q_query'],
                      class_names=cfg['class_names'], episode_accuracies=accuracies,
                      macro_f1=float(f1_score(true, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
                      confusion_matrix=confusion_matrix(true, pred, labels=[0, 1, 2]).tolist(),
                      checkpoint=str(Path(load_path).resolve()), data_sha256=self.storage.signature,
                      episode_manifest=str(tasks.path.resolve()),
                      quick_checkpoint=saved.get('quick_test', False), seconds=time.time()-start)
        # Every evaluation gets its own directory; old predictions are never replaced.
        output = Path(cfg['result_root']) / cfg['model_name'] / str(time.time_ns())
        output.mkdir(parents=True)
        write_json(output/'summary.json', result)
        write_json(output/'config.json', cfg)
        with (output/'predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'image', 'true_label', 'predicted_label'])
            writer.writeheader(); writer.writerows(rows)
        print(f'Cross test {cfg["task"]}: accuracy={result["accuracy"]:.4f}; {output}', flush=True)
        return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=[*TASKS, 'all'])
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--shots', type=int, choices=[1, 5])
    p.add_argument('--seed', type=int)
    p.add_argument('--processed-root')
    p.add_argument('--model-path')
    p.add_argument('--preprocess', action='store_true')
    p.add_argument('--train', action='store_true')
    p.add_argument('--test', action='store_true')
    p.add_argument('--quick', action='store_true', help='isolated one-update smoke test, not a paper result')
    p.add_argument('--test-episodes', type=int)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args(argv)
    if args.threads <= 0 or (args.task == 'all' and args.model_path):
        p.error('positive --threads required; --model-path supports one task only')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    task_choice = args.task or CROSS_CONFIG['task']
    train, test = args.train, args.test
    if not (args.preprocess or train or test):
        train = CROSS_CONFIG['run_mode'] in ('train', 'train_test')
        test = CROSS_CONFIG['run_mode'] in ('test', 'train_test')
    overrides = {k: v for k, v in dict(method=args.method, seed=args.seed,
                  k_shot=args.shots, processed_root=args.processed_root,
                  test_meta_batch_size=args.test_episodes).items() if v is not None}
    if args.quick:
        overrides.update(epochs=1, meta_batch_size=1, validation_episodes=1,
                         ft_pretrain_epochs=1, ft_pretrain_max_batches=1,
                         early_stop_confirmation_episodes=1, early_stop_on_perfect_validation=False)
        overrides.setdefault('test_meta_batch_size', 2)
    for number, task in enumerate(TASKS if task_choice == 'all' else [task_choice]):
        cfg = resolve_cross_config(dict(overrides, task=task))
        print(f'{task}: {cfg["source_domain"]}->{cfg["target_domain"]}; {cfg["method"]}; '
              f'3-way {cfg["k_shot"]}-shot, device={device}; '
              f'Actions: preprocess={args.preprocess}, train={train}, test={test}', flush=True)
        if args.preprocess and number == 0:
            preprocess_cross(cfg)
        if not (train or test):
            continue
        seed_torch(cfg['seed'])
        path = args.model_path or str(Path(cfg['checkpoint_root']) / (cfg['model_name']+('_quick' if args.quick else '')))
        learner = CrossLearner(cfg)
        best = learner.train(path, quick_test=args.quick) if train else path+'_best'
        if test:
            learner.test(best)


if __name__ == '__main__':
    main()
