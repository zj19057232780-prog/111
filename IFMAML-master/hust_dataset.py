"""Lazy split loading and persisted HUST validation/test episode manifests."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from hust_preprocess import data_signature, digest


class HUSTMetaDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg['root_path'])
        path = self.root / 'manifest.json'
        self.manifest = json.loads(path.read_text(encoding='utf-8'))
        if self.manifest['signature'] != data_signature(cfg):
            raise ValueError('HUST preprocessing signature differs from config')
        self.signature = digest(path)
        self.source_classes = list(cfg['class_names'])
        self.rows = self.manifest['windows']
        records = defaultdict(list)
        seen = set()
        for row in self.rows:
            if row['image'] in seen:
                raise ValueError('Duplicate HUST window ID')
            seen.add(row['image'])
            if row['split'] not in ('train', 'validation', 'test'):
                raise ValueError('Unknown HUST split')
            if not (0 <= row['start'] < row['end'] <= row['length']):
                raise ValueError('HUST window out of bounds')
            if row['end']-row['start'] != cfg['window_size']:
                raise ValueError('Wrong HUST window length')
            records[row['record']].append(row)
        for rows in records.values():
            ordered = sorted(rows, key=lambda r: r['start'])
            if any(a['end'] > b['start'] for a, b in zip(ordered, ordered[1:])):
                raise ValueError('Overlapping HUST windows across or within splits')
            for row in rows:
                mid = row['length']//2
                if row['split'] == 'validation' and row['end'] > mid:
                    raise ValueError('Validation window outside first target block')
                if row['split'] == 'test' and row['start'] < mid+cfg['window_size']:
                    raise ValueError('Test window crosses target isolation gap')
        self.by_split = {}
        self._cache = {}
        for split, per_record in [('train', cfg['source_samples']),
                                  ('validation', cfg['validation_samples']),
                                  ('test', cfg['test_samples'])]:
            groups = {c: [] for c in self.source_classes}
            loads = cfg['source_condition'] if split == 'train' else [cfg['target_condition']]
            for row in self.rows:
                if row['split'] == split:
                    if row['load'] not in loads or row['bearing'] not in cfg['bearing_types']:
                        raise ValueError('Record assigned to wrong HUST domain')
                    groups[row['fault']].append(row)
            expected = len(loads)*len(cfg['bearing_types'])*per_record
            if any(len(v) != expected for v in groups.values()):
                raise ValueError(f'Wrong HUST class counts: {split}')
            self.by_split[split] = groups

    def images(self, split):
        # Constructing train/validation tasks never opens target-test pixels.
        if split not in self._cache:
            groups = {}
            for name, rows in self.by_split[split].items():
                images = []
                for row in rows:
                    path = self.root / row['image']
                    if digest(path) != row['sha256']:
                        raise ValueError(f'Corrupt HUST image: {path}')
                    value = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
                    if value is None or value.shape != (self.cfg['img_size'],)*2:
                        raise ValueError(f'Bad HUST image: {path}')
                    images.append(value[None].astype(np.float32)/255)
                groups[name] = np.stack(images)
            self._cache[split] = groups
        return self._cache[split]


class HUSTTasks:
    def __init__(self, storage, mode, ways, shots, queries, count, seed, fixed):
        self.storage, self.mode = storage, mode
        self.ways, self.shots, self.queries = ways, shots, queries
        self.images = storage.images(mode)
        self.rng = np.random.default_rng(seed)
        self.cursor = 0
        self.plans = None
        if fixed:
            header = dict(data_sha256=storage.signature, mode=mode, ways=ways,
                          shots=shots, queries=queries, count=count, seed=seed)
            tag = hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()[:16]
            self.path = storage.root / 'episodes' / f'{mode}_{shots}shot_{tag}.json'
            # Always regenerate deterministically and compare, so stale/edited plans fail.
            plans = [self._new_plan() for _ in range(count)]
            payload = dict(header=header, episodes=plans)
            if self.path.exists():
                if json.loads(self.path.read_text(encoding='utf-8')) != payload:
                    raise ValueError(f'Episode manifest mismatch: {self.path}')
            else:
                self.path.parent.mkdir(exist_ok=True)
                self.path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            self.plans = plans

    def _new_plan(self):
        chosen = self.rng.choice(len(self.images), self.ways, replace=False).tolist()
        selections = []
        for cls in chosen:
            name = self.storage.source_classes[cls]
            # Reserve five support candidates for both shot settings: query is paired.
            picks = self.rng.choice(len(self.images[name]), self.queries+5, replace=False)
            q, s = picks[:self.queries].tolist(), picks[self.queries:][:self.shots].tolist()
            rows = self.storage.by_split[self.mode][name]
            selections.append(dict(support=s, query=q,
                                   support_ids=[rows[i]['image'] for i in s],
                                   query_ids=[rows[i]['image'] for i in q]))
        return dict(classes=chosen, selections=selections)

    def sample_with_original_labels(self):
        if self.plans is None:
            plan = self._new_plan()
        else:
            plan = self.plans[self.cursor % len(self.plans)]
            self.cursor += 1
        sx, sy, qx, qy, so, qo = [], [], [], [], [], []
        for local, (orig, picks) in enumerate(zip(plan['classes'], plan['selections'])):
            arr = self.images[self.storage.source_classes[orig]]
            sx.extend(arr[picks['support']]); sy.extend([local]*self.shots); so.extend([orig]*self.shots)
            qx.extend(arr[picks['query']]); qy.extend([local]*self.queries); qo.extend([orig]*self.queries)
        return (torch.from_numpy(np.stack(sx)), torch.tensor(sy),
                torch.from_numpy(np.stack(qx)), torch.tensor(qy),
                torch.tensor(so), torch.tensor(qo), torch.tensor(plan['classes']))

    def sample(self):
        return self.sample_with_original_labels()[:4]
