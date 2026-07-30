"""
独立测试脚本：加载已保存的最优模型进行测试

用法：
    conda activate ifmaml2
    python test_only.py

模型路径：
    ./model_save/MAML_PU_test_best
"""

import torch
import os
import glob
from maml_model import Net4CNN
from l2l_shim import MetaDataset, TaskDataset, MAML as MAMLAlgo
from pu_dataset import PUMetaDataset
from my_utils.train_utils import accuracy
from config_pu import PU_CONFIG

RANDOM_SEED = 24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

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
    import numpy as np
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

    test_tasks = build_tasks('test', n_way, shots, 1000, None)
    maml = MAMLAlgo(model, lr=0.05)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    meta_batch_size = 16
    adaptation_steps = inner_steps
    meta_test_error = 0.0
    meta_test_accuracy = 0.0

    for i in range(meta_batch_size):
        learner = maml.clone()
        task = test_tasks.sample()
        evaluation_error, evaluation_accuracy, _, _ = fast_adapt(
            task, learner, loss, adaptation_steps, shots, n_way
        )
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
        print(f'  Task {i+1}/{meta_batch_size}: Acc = {evaluation_accuracy.item():.4f}')

    print()
    print('=' * 50)
    print(f'  Meta Test Error: {meta_test_error / meta_batch_size:.4f}')
    print(f'  Meta Test Accuracy: {meta_test_accuracy / meta_batch_size:.4f}')
    print('=' * 50)


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
    print(f'随机种子(训练时): {RANDOM_SEED}')
    print()
    test(best_path, inner_steps=10, shots=PU_CONFIG['k_shot'], n_way=PU_CONFIG['n_way'])
