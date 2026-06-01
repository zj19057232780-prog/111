from __future__ import annotations

from collections import OrderedDict, defaultdict

import numpy as np
import torch

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


class MetaDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self._class_to_indices = defaultdict(list)
        for i in range(len(dataset)):
            _, y = dataset[i]
            self._class_to_indices[int(y)].append(i)
        self.classes = sorted(self._class_to_indices.keys())

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class TaskDataset:
    """Few-shot episode sampler with separate support/query counts."""

    def __init__(
        self,
        meta_dataset,
        n_way,
        k_shot,
        q_query=1,
        num_tasks=100,
        filter_labels=None,
    ):
        self.meta = meta_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_tasks = num_tasks
        self.available = (
            list(filter_labels) if filter_labels is not None else list(meta_dataset.classes)
        )

        if len(self.available) < n_way:
            raise RuntimeError(f'Need {n_way} classes, got {len(self.available)}')

        required = k_shot + q_query
        samples_per_class = min(
            len(self.meta._class_to_indices[c]) for c in self.available
        )
        if required > samples_per_class:
            raise RuntimeError(
                f'Not enough samples/class: need {required}, available {samples_per_class}'
            )

    def _sample_episode(self):
        chosen = np.random.choice(self.available, self.n_way, replace=False)
        support_x, support_y = [], []
        query_x, query_y = [], []
        support_orig_y, query_orig_y = [], []

        for new_label, orig_label in enumerate(chosen):
            indices = self.meta._class_to_indices[orig_label]
            picked = np.random.choice(
                indices,
                self.k_shot + self.q_query,
                replace=False,
            )
            support_indices = picked[:self.k_shot]
            query_indices = picked[self.k_shot:]

            for idx in support_indices:
                x_i, _ = self.meta[idx]
                support_x.append(torch.as_tensor(x_i, dtype=torch.float32))
                support_y.append(new_label)
                support_orig_y.append(orig_label)
            for idx in query_indices:
                x_i, _ = self.meta[idx]
                query_x.append(torch.as_tensor(x_i, dtype=torch.float32))
                query_y.append(new_label)
                query_orig_y.append(orig_label)

        return (
            torch.stack(support_x),
            torch.tensor(support_y, dtype=torch.long),
            torch.stack(query_x),
            torch.tensor(query_y, dtype=torch.long),
            torch.tensor(support_orig_y, dtype=torch.long),
            torch.tensor(query_orig_y, dtype=torch.long),
            torch.tensor(chosen, dtype=torch.long),
        )

    def sample(self):
        episode = self._sample_episode()
        return episode[:4]

    def sample_with_original_labels(self):
        return self._sample_episode()


class MAML:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def clone(self):
        return _MAMLClone(self.model, self.lr)

    def parameters(self):
        return self.model.parameters()


class _MAMLClone:
    def __init__(self, model, lr):
        self._model = model
        self.lr = lr
        self.fast_params = OrderedDict(
            (name, param.clone().requires_grad_(True))
            for name, param in model.named_parameters()
        )

    def __call__(self, x):
        return functional_call(self._model, self.fast_params, x)

    def adapt(self, loss):
        grads = torch.autograd.grad(
            loss,
            self.fast_params.values(),
            create_graph=True,
            retain_graph=True,
        )
        for (name, param), grad in zip(self.fast_params.items(), grads):
            self.fast_params[name] = param - self.lr * grad
