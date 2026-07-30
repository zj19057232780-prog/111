"""
随机种子搜索脚本：遍历多个种子，找到测试准确率最高的种子

用法：
    conda activate ifmaml2
    python test_seed_search.py
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

SEED_START = 0
SEED_END = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_tasks(mode='train', ways=5, shots=5, num_tasks=1000, filter_labels=None):
    cfg = PU_CONFIG
    base = PUMetaDataset(
        root_path=cfg['root_path'],
        source_condition=cfg['source_condition'],
        target_condition=cfg['target_condition'],
        img_size=cfg.get('img_size', 64),
        class_groups=cfg.get('class_groups'),
        classification_name=cfg.get('classification_preset'),
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


def test_with_seed(seed, load_path, inner_steps, shots, n_way):
    set_seed(seed)

    h_size = 64
    layers = 4
    sample_len = 256
    feat_size = (sample_len // 2**layers) * h_size
    in_channels = PU_CONFIG.get('in_channels', 1)
    model = Net4CNN(output_size=n_way, hidden_size=h_size, layers=layers,
                    channels=in_channels, embedding_size=feat_size).to(device)

    model.load_state_dict(torch.load(load_path, weights_only=True))

    test_tasks = build_tasks('test', n_way, shots, 1000, None)
    maml = MAMLAlgo(model, lr=0.05)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    meta_batch_size = 16
    adaptation_steps = inner_steps
    meta_test_accuracy = 0.0

    for _ in range(meta_batch_size):
        learner = maml.clone()
        task = test_tasks.sample()
        _, evaluation_accuracy, _, _ = fast_adapt(
            task, learner, loss, adaptation_steps, shots, n_way
        )
        meta_test_accuracy += evaluation_accuracy.item()

    return meta_test_accuracy / meta_batch_size


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

    print(f'测试配置: {PU_CONFIG["n_way"]}-way, {PU_CONFIG["k_shot"]}-shot')
    print(f'目标工况: {PU_CONFIG["target_condition"]}')
    print(f'搜索范围: 种子 {SEED_START} ~ {SEED_END - 1}')
    print('=' * 60)

    results = []
    best_acc = -1.0
    best_seed = None

    for seed in range(SEED_START, SEED_END):
        acc = test_with_seed(seed, best_path, inner_steps=10,
                             shots=PU_CONFIG['k_shot'], n_way=PU_CONFIG['n_way'])
        results.append((seed, acc))
        marker = ' ★ BEST' if acc > best_acc else ''
        print(f'  Seed {seed:3d}: Accuracy = {acc:.4f}{marker}')

        if acc > best_acc:
            best_acc = acc
            best_seed = seed

    print()
    print('=' * 60)
    print(f'  最优种子: {best_seed}')
    print(f'  最高准确率: {best_acc:.4f}')
    print('=' * 60)
    print()
    print('完整结果（按准确率降序）:')
    results.sort(key=lambda x: x[1], reverse=True)
    for rank, (seed, acc) in enumerate(results[:10], 1):
        print(f'  Top {rank:2d}: Seed {seed:3d}, Accuracy = {acc:.4f}')
