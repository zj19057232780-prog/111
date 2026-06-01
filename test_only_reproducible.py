"""
可复现测试脚本：加载已保存的最优模型进行测试，使用固定随机种子

用法：
    conda activate ifmaml2
    python test_only_reproducible.py

模型路径：
    ./model_save/MAML_PU_test_best
"""

import torch
import numpy as np
import random
import os
import glob

from maml_model import Net4CNN
from l2l_shim import MetaDataset, TaskDataset, MAML as MAMLAlgo
from pu_dataset import PUMetaDataset
from my_utils.train_utils import accuracy
from config_pu import PU_CONFIG

RANDOM_SEED = 75
NUM_SEEDS = 5
SEEDS = [75, 42, 123, 456, 789]  # 5个不同的随机种子
NUM_TASKS = 18 # 每个seed测试的task数（可改为1000）

def set_seed(seed):
    """设置所有随机种子，确保测试结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")
print(f"随机种子: {RANDOM_SEED}")

def build_tasks(mode='train', ways=5, shots=5, num_tasks=1000, filter_labels=None):
    cfg = PU_CONFIG
    base = PUMetaDataset(
        root_path=cfg['root_path'],
        source_condition=cfg['source_condition'],
        target_condition=cfg['target_condition'],
        img_size=cfg.get('img_size', 64),
    )
    base.prepare_data()
    base.set_mode(mode)
    dataset = MetaDataset(base)
    new_ways = len(filter_labels) if filter_labels is not None else ways
    tasks = TaskDataset(dataset, new_ways, 2 * shots,
                       num_tasks=num_tasks, filter_labels=filter_labels)
    return tasks


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data)[1], adaptation_labels)
        learner.adapt(train_error)

    features, predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy, features, evaluation_labels


def run_single_seed(model, load_path, seed, shots, n_way, inner_steps, num_tasks):
    """运行单个seed的测试"""
    set_seed(seed)

    if model is None:
        h_size = 64
        layers = 4
        sample_len = 256
        feat_size = (sample_len // 2**layers) * h_size
        in_channels = PU_CONFIG.get('in_channels', 1)
        model = Net4CNN(output_size=n_way, hidden_size=h_size, layers=layers,
                        channels=in_channels, embedding_size=feat_size).to(device)
        model.load_state_dict(torch.load(load_path, weights_only=True))
        print(f'模型加载成功: {load_path}')

    test_tasks = build_tasks('test', n_way, shots, num_tasks, None)
    maml = MAMLAlgo(model, lr=0.05)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    adaptation_steps = inner_steps
    task_accuracies = []

    for i in range(num_tasks):
        learner = maml.clone()
        task = test_tasks.sample()
        _, evaluation_accuracy, _, _ = fast_adapt(
            task, learner, loss, adaptation_steps, shots, n_way
        )
        task_accuracies.append(evaluation_accuracy.item())

        if (i + 1) % 50 == 0:
            print(f'    Seed {seed}: Task {i+1}/{num_tasks}, Acc = {evaluation_accuracy.item():.4f}')

    mean_acc = np.mean(task_accuracies)
    return mean_acc, model


def test(load_path, inner_steps=10, shots=1, n_way=5):
    h_size = 64
    layers = 4
    sample_len = 256
    feat_size = (sample_len // 2**layers) * h_size
    in_channels = PU_CONFIG.get('in_channels', 1)
    model = Net4CNN(output_size=n_way, hidden_size=h_size, layers=layers,
                    channels=in_channels, embedding_size=feat_size).to(device)

    model.load_state_dict(torch.load(load_path, weights_only=True))
    print(f'模型加载成功: {load_path}')
    print(f'测试配置: {n_way}-way, {shots}-shot, {NUM_TASKS} tasks/seed, {NUM_SEEDS} seeds')
    print()

    all_accuracies = []

    for seed_idx, seed in enumerate(SEEDS, 1):
        print(f'--- Seed {seed_idx}/{NUM_SEEDS} (seed={seed}) ---')
        mean_acc, model = run_single_seed(
            model, load_path, seed, shots, n_way, inner_steps, NUM_TASKS
        )
        all_accuracies.append(mean_acc)
        print(f'  Seed {seed} Mean Accuracy: {mean_acc:.4f}')
        print()

    all_accuracies = np.array(all_accuracies)
    mean_overall = np.mean(all_accuracies)
    std_overall = np.std(all_accuracies)

    from scipy import stats
    se = std_overall / np.sqrt(NUM_SEEDS)
    ci_95 = se * stats.t.ppf(0.975, NUM_SEEDS - 1)

    print('=' * 70)
    print('多Seed统计结果')
    print('=' * 70)
    print(f'各Seed准确率: {[f"{acc:.4f}" for acc in all_accuracies]}')
    print(f'Mean ± Std: {mean_overall:.4f} ± {std_overall:.4f}')
    print(f'95% CI: [{mean_overall - ci_95:.4f}, {mean_overall + ci_95:.4f}]')
    print(f'(95% CI = Mean ± {ci_95:.4f})')
    print('=' * 70)


if __name__ == '__main__':
    best_path = r'.\model_save\MAML_PU_test_best'

    if not os.path.exists(best_path):
        candidates = glob.glob(r'.\model_save\MAML_PU_test*')
        if candidates:
            print(f'最优模型不存在，使用最新模型: {candidates[-1]}')
            best_path = candidates[-1]
        else:
            print('错误: 未找到任何已保存的模型，请先完成训练。')
            exit(1)

    print(f'目标工况: {PU_CONFIG["target_condition"]}')
    print(f'内循环步数: 10')
    print(f'随机种子: {NUM_SEEDS} seeds = {SEEDS}')
    print(f'每Seed测试任务数: {NUM_TASKS}')
    print('=' * 70)
    print()
    test(best_path, inner_steps=10, shots=PU_CONFIG['k_shot'], n_way=PU_CONFIG['n_way'])
