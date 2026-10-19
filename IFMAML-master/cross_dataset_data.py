"""Native raw anchors, optional STFT cache, and paired cross-dataset episodes."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math

import cv2
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from config_cross_dataset import fingerprint, CLASS_NAMES
from cwru_preprocess import load_metadata_ids, load_drive_end_signal
from pu_loader import PULoader
from pu_preprocess import compute_stft_log_image
from hust_preprocess import digest
from hust_dataset import HUSTTasks
from cross_dataset_models import native_raw_input

DATA_KEYS = ('protocol', 'class_names', 'samples_per_block', 'pu_records', 'anchor_size',
             'cwru_conditions', 'pu_conditions', 'stft_fs', 'window_size', 'img_size',
             'normalize', 'stft_nperseg', 'stft_noverlap', 'stft_window',
             'stft_bandpass', 'stft_log_eps', 'stft_p_low', 'stft_p_high', 'stft_gamma')
CWRU_SPECS = [('N', 'NormalBaseline', 'Normal.mat'),
              ('I-L1', '48DriveEndFault', '0.007-InnerRace.mat'),
              ('I-L2', '48DriveEndFault', '0.021-InnerRace.mat'),
              ('O-L1', '48DriveEndFault', '0.007-OuterRace6.mat'),
              ('O-L2', '48DriveEndFault', '0.021-OuterRace6.mat')]
PU_BEARINGS = dict(zip(CLASS_NAMES, ('K001', 'KI03', 'KI07', 'KA05', 'KA03')))


def data_signature(cfg: dict) -> dict:
    return {k: cfg[k] for k in DATA_KEYS}


def records(cfg: dict) -> list:
    result = []
    root = Path(cfg['cwru_raw_root'])
    ids = load_metadata_ids(root)
    for condition, speed in cfg['cwru_conditions'].items():
        for cls, subset, name in CWRU_SPECS:
            x, variable, path = load_drive_end_signal(root, subset, speed, name, ids, cfg)
            result.append(('A', condition, cls, 0, path, 48000, variable, x))
    root = Path(cfg['pu_raw_root'])
    loader = PULoader(str(root))
    for condition, full_name in cfg['pu_conditions'].items():
        for cls, bearing in PU_BEARINGS.items():
            for rec in cfg['pu_records']:
                filename = f"{full_name}_{bearing}_{rec}.mat"
                matches = list(root.rglob(filename))
                if len(matches) != 1:
                    raise ValueError(f'Expected one PU file {filename}, found {len(matches)}')
                path = matches[0]
                x = np.asarray(loader.get_signal(loadmat(path))).ravel()
                result.append(('B', condition, cls, rec, path, 64000, 'vibration_1 via PULoader', x))
    for domain, condition, cls, rec, path, fs, variable, x in result:
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError(f'Invalid vibration record: {path}')
    return result


def _image(raw: np.ndarray, domain: str, cfg: dict) -> np.ndarray:
    # STFT-only conversion. Native raw mode never calls this function.
    points = cfg['window_size'] if domain == 'A' else cfg['window_size'] * 4 // 3
    values = raw[:points].astype(np.float64)
    if domain == 'B':
        values = resample_poly(values, 3, 4)
    return compute_stft_log_image(values[:cfg['window_size']], cfg)


def preprocess_cross(cfg: dict) -> dict:
    root = Path(cfg['processed_root'])
    path = root / 'manifest.json'
    original = records(cfg)
    raw_records = {f'{d}/{condition}/{c}/r{rec}': dict(path=str(p.resolve()), sha256=digest(p),
                    fs=fs, variable=v, length=len(x)) for d, condition, c, rec, p, fs, v, x in original}
    size, count = cfg['anchor_size'], cfg['samples_per_block']
    rows = []
    segments = []
    for d, condition, cls, rec, p, fs, v, x in original:
        mid = len(x) // 2
        blocks = [(0, 0, mid), (1, mid + size, len(x))] if d == 'A' else [(rec - 1, 0, len(x))]
        for block, begin, end in blocks:
            candidates = np.arange(0, end - begin - size + 1, size, dtype=np.int64)
            if len(candidates) < count:
                raise ValueError(f'{p}/block{block}: only {len(candidates)} disjoint native windows; need {count}')
            chosen = candidates[np.floor(np.linspace(0, len(candidates)-1, count)).astype(int)]
            for i, start in enumerate(chosen):
                name = f'{d}/{condition}/{cls}/b{block}_{i:03d}'
                r = dict(domain=d, condition=condition, fault=cls, block=block, record=f'{d}/{condition}/{cls}/r{rec}',
                         original_begin=begin, original_end=end, original_fs=fs,
                         start=int(begin+start), end=int(begin+start+size),
                         image=name+'.png', raw=name+'.npy')
                rows.append(r)
                segments.append(x[r['start']:r['end']])
    header = dict(signature=data_signature(cfg), records=raw_records)
    exists = root.exists() and any(root.iterdir())
    if exists and not path.is_file():
        raise ValueError('Incomplete cache; select a fresh --processed-root (no automatic deletion)')
    if exists:
        old = json.loads(path.read_text(encoding='utf-8'))
        if any(old.get(k) != v for k, v in header.items()):
            raise ValueError('Raw data/protocol mismatch; choose a fresh --processed-root')
        stripped = [{k: v for k, v in r.items() if k != 'raw_sha256'} for r in old['windows']]
        if stripped != rows:
            raise ValueError('Cached window identities differ from native sampling plan')
        for r in old['windows']:
            if digest(root/r['raw']) != r['raw_sha256']:
                raise ValueError(f"Corrupted raw window: {r['raw']}")
        manifest = old
    else:
        root.mkdir(parents=True, exist_ok=True)
        for r, segment in zip(rows, segments):
            target = root/r['raw']
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, segment.astype(np.float64), allow_pickle=False)
            r['raw_sha256'] = digest(target)
        manifest = dict(header, windows=rows)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    # Separate image index: adding STFT later cannot invalidate raw/episode identity.
    if cfg['input_mode'] == 'stft':
        index_path = root/'stft_index.json'
        image_index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
        for r in manifest['windows']:
            target = root/r['image']
            if r['image'] in image_index:
                if digest(target) != image_index[r['image']]:
                    raise ValueError(f'Corrupted STFT image: {target}')
                continue
            image = _image(np.load(root/r['raw'], allow_pickle=False), r['domain'], cfg)
            ok, encoded = cv2.imencode('.png', image)
            if not ok:
                raise IOError(f'PNG encode failed: {target}')
            target.write_bytes(encoded.tobytes())
            image_index[r['image']] = digest(target)
        index_path.write_text(json.dumps(image_index, indent=2), encoding='utf-8')
    print(f"Cross AB: {len(rows)} native anchors; {cfg['input_mode']}; {root}", flush=True)
    return manifest


class CrossDataset:
    def __init__(self, cfg: dict):
        self.cfg, self.root = cfg, Path(cfg['processed_root'])
        path = self.root/'manifest.json'
        self.manifest = json.loads(path.read_text(encoding='utf-8'))
        if self.manifest['signature'] != data_signature(cfg):
            raise ValueError('Preprocessing signature mismatch; old 3-way data are not accepted')
        self.signature = fingerprint(dict(data=digest(path), task=cfg['task']))
        self.source_classes = list(cfg['class_names'])
        rows = self.manifest['windows']
        grouped, seen, counts = defaultdict(list), set(), defaultdict(int)
        for r in rows:
            if r['domain'] not in ('A', 'B') or r['fault'] not in self.source_classes or r['block'] not in (0, 1):
                raise ValueError('Invalid domain/class/block')
            condition_map = cfg['cwru_conditions'] if r['domain']=='A' else cfg['pu_conditions']
            if r.get('condition') not in condition_map:
                raise ValueError('Unknown condition in original window')
            counts[(r['domain'], r['condition'], r['fault'], r['block'])] += 1
            if r['raw'] in seen or r['end']-r['start'] != cfg['anchor_size']:
                raise ValueError('Duplicate or invalid native window')
            seen.add(r['raw'])
            rec = self.manifest['records'][r['record']]
            expected_rec = f"{r['domain']}/{r['condition']}/{r['fault']}/r{0 if r['domain']=='A' else r['block']+1}"
            if r['record'] != expected_rec or not 0 <= r['start'] < r['end'] <= rec['length']:
                raise ValueError('Window assigned to wrong original record')
            m = rec['length']//2
            bounds = ((0, m) if r['block']==0 else (m+cfg['anchor_size'], rec['length'])) if r['domain']=='A' else (0, rec['length'])
            if (r['original_begin'], r['original_end']) != bounds or not bounds[0] <= r['start'] < r['end'] <= bounds[1]:
                raise ValueError('Window crosses isolation boundary')
            grouped[r['record']].append(r)
        for rs in grouped.values():
            ordered = sorted(rs, key=lambda r:r['start'])
            if any(a['end'] > b['start'] for a, b in zip(ordered, ordered[1:])):
                raise ValueError('Overlapping windows')
        required = {(d, c, fault, block) for d, conditions in
                    [('A', cfg['cwru_conditions']), ('B', cfg['pu_conditions'])]
                    for c in conditions for fault in self.source_classes for block in (0, 1)}
        if set(counts) != required or any(n != cfg['samples_per_block'] for n in counts.values()):
            raise ValueError('Incomplete or unbalanced domain/condition/class/block data')
        expected = 8 * len(self.source_classes) * 2 * cfg['samples_per_block']
        if len(rows) != expected:
            raise ValueError('Wrong total window count')
        self.by_split, self._cache = {}, {}
        for split in ('train', 'validation', 'test'):
            domain = cfg['source_domain'] if split=='train' else cfg['target_domain']
            conditions = cfg['source_conditions'] if split=='train' else [cfg['target_condition']]
            selected = [r for r in rows if r['domain']==domain and r['condition'] in conditions and
                        (split=='train' or r['block']==(0 if split=='validation' else 1))]
            groups = {c:[r for r in selected if r['fault']==c] for c in self.source_classes}
            expected = cfg['samples_per_block'] * (2*len(cfg['source_conditions']) if split=='train' else 1)
            if any(len(v)!=expected for v in groups.values()):
                raise ValueError('Wrong per-class split count')
            self.by_split[split] = groups

    def images(self, split: str) -> dict:
        if split not in self._cache:
            cfg, groups = self.cfg, {}
            index = json.loads((self.root/'stft_index.json').read_text(encoding='utf-8')) if cfg['input_mode']=='stft' else None
            for cls, rows in self.by_split[split].items():
                values = []
                for r in rows:
                    if cfg['input_mode']=='nostft':
                        path = self.root/r['raw']
                        if digest(path)!=r['raw_sha256']:
                            raise ValueError('Corrupted native raw window')
                        raw = np.load(path, allow_pickle=False)
                        if raw.shape!=(cfg['anchor_size'],) or not np.isfinite(raw).all():
                            raise ValueError('Invalid native raw array')
                        value = native_raw_input(raw, cfg['method'], r['domain'])
                    else:
                        path = self.root/r['image']
                        if digest(path)!=index[r['image']]:
                            raise ValueError('Corrupted STFT image')
                        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 0)
                        if image is None or image.shape!=(64,64):
                            raise ValueError('Invalid STFT image')
                        value = image[None].astype(np.float32)/255
                        if cfg['method']=='tl_wdcnn':
                            value = value.reshape(1, -1)
                    values.append(value)
                groups[cls] = np.stack(values)
            self._cache[split] = groups
        return self._cache[split]


class CrossTasks(HUSTTasks):
    def __init__(self, storage, mode, ways, shots, queries, count, seed, fixed):
        if ways != 5 or shots not in (1,3,5) or queries+5 > storage.cfg['samples_per_block']:
            raise ValueError('Invalid Cross AB episode size')
        # Independent target-task streams, paired across methods, inputs and seeds.
        seed = int(seed) + (int(storage.cfg['task'][1:])-1)*100000
        super().__init__(storage, mode, ways, shots, queries, count, seed, fixed)


    def _new_plan(self):
        if self.mode != 'train':
            return super()._new_plan()
        # Every source meta-task comes from one uniformly sampled source condition.
        # This preserves within-domain support/query adaptation across four domains.
        condition = self.rng.choice(self.storage.cfg['source_conditions']).item()
        chosen = self.rng.choice(len(self.images), self.ways, replace=False).tolist()
        selections = []
        for cls in chosen:
            name = self.storage.source_classes[cls]
            rows = self.storage.by_split['train'][name]
            eligible = [i for i, r in enumerate(rows) if r['condition'] == condition]
            picks = self.rng.choice(eligible, self.queries+5, replace=False)
            q, s = picks[:self.queries].tolist(), picks[self.queries:][:self.shots].tolist()
            selections.append(dict(support=s, query=q,
                support_ids=[rows[i]['image'] for i in s], query_ids=[rows[i]['image'] for i in q]))
        return dict(classes=chosen, source_condition=condition, selections=selections)
