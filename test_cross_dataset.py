"""Real-data protocol and model lifecycle tests (preprocess first)."""
import copy
import json
import time
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.signal import resample_poly
from fractions import Fraction
from config_cross_dataset import TASKS, METHODS, resolve_cross_config
from cross_dataset_data import CrossDataset, CrossTasks, records
from pu_preprocess import compute_stft_log_image
from run_cross_dataset import CrossLearner, evaluation_state, parse_args
from my_utils.init_utils import seed_torch


class CrossProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        seed_torch(24)

    def test_all_splits_and_paired_manifest(self):
        for name in TASKS:
            cfg = resolve_cross_config({'task': name})
            ds = CrossDataset(cfg)
            for split, number in [('train', 40), ('validation', 20), ('test', 20)]:
                self.assertTrue(all(len(v) == number for v in ds.by_split[split].values()))
            pools = [{r['image'] for rs in ds.by_split[s].values() for r in rs}
                     for s in ('train', 'validation', 'test')]
            self.assertFalse(pools[0]&pools[1] or pools[0]&pools[2] or pools[1]&pools[2])
            CrossTasks(ds, 'validation', 3, 1, 15, 3, 99, True)
            self.assertNotIn('test', ds._cache)
            one = CrossTasks(ds, 'test', 3, 1, 15, 3, 100, True)
            five = CrossTasks(ds, 'test', 3, 5, 15, 3, 100, True)
            raw = CrossTasks(CrossDataset(resolve_cross_config({'task': name, 'method': 'tl_wdcnn'})),
                             'test', 3, 1, 15, 3, 100, True)
            self.assertEqual(one.plans, raw.plans)
            for p, q in zip(one.plans, five.plans):
                for a, b in zip(p['selections'], q['selections']):
                    self.assertEqual(a['query_ids'], b['query_ids'])
                    self.assertEqual(a['support_ids'], b['support_ids'][:1])
                    self.assertFalse(set(b['query_ids']) & set(b['support_ids']))
            batch = five.sample_with_original_labels()
            self.assertEqual(tuple(batch[0].shape), (15, 1, 64, 64))
            self.assertEqual(tuple(batch[2].shape), (45, 1, 64, 64))
            self.assertTrue(torch.equal(batch[6][batch[3]], batch[5]))
            self.assertEqual(tuple(raw.sample()[2].shape), (45, 1, 3072))

    def test_all_windows_reproduce_from_raw_records(self):
        cfg = resolve_cross_config()
        root = Path(cfg['processed_root'])
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        originals = {d+'/'+c: (fs, x) for d, c, _, fs, _, x in records(cfg)}
        self.assertEqual(len(originals), 9)
        cache = {}
        for r in manifest['windows']:
            key = (r['record'], r['block'])
            if key not in cache:
                fs, x = originals[r['record']]
                ratio = Fraction(cfg['stft_fs'], fs)
                cache[key] = resample_poly(x[r['original_begin']:r['original_end']].astype(np.float64),
                                          ratio.numerator, ratio.denominator)
            raw = cache[key][r['start']:r['end']]
            np.testing.assert_array_equal(raw, np.load(root/r['raw'], allow_pickle=False))
            expected = compute_stft_log_image(raw, cfg)
            image = cv2.imdecode(np.frombuffer((root/r['image']).read_bytes(), np.uint8), 0)
            np.testing.assert_array_equal(expected, image)

    def test_second_order_and_query_label_isolation(self):
        cfg = resolve_cross_config()
        learner = CrossLearner(cfg)
        batch = learner.build_tasks('validation', num_tasks=2).sample()
        before = {n: b.clone() for n, b in learner.model.named_buffers()}
        err, _, _, _ = learner.fast_adapt(batch, learner._algorithm().clone(), torch.nn.CrossEntropyLoss(), 2)
        err.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in learner.model.parameters()))
        self.assertTrue(all(torch.equal(before[n], b) for n, b in learner.model.named_buffers()))
        for method in METHODS:
            obj = CrossLearner(resolve_cross_config({'method': method}))
            task = obj.build_tasks('validation', num_tasks=1).sample()
            state = {n: p.clone() for n, p in obj.model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            with evaluation_state(obj.model, 88):
                a = obj.fast_adapt(task, obj._algorithm().clone(), torch.nn.CrossEntropyLoss(), 1, True)[-1].detach()
            changed = (*task[:3], (task[3]+1)%3)
            with evaluation_state(obj.model, 88):
                b = obj.fast_adapt(changed, obj._algorithm().clone(), torch.nn.CrossEntropyLoss(), 1, True)[-1].detach()
            self.assertTrue(torch.equal(a, b), method)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(state[n], p) for n, p in obj.model.state_dict().items()), method)

    def test_config_cli_and_invalid_protocol(self):
        args = parse_args(['--test'])
        self.assertTrue(args.test)
        self.assertFalse(args.train)
        self.assertIsNone(args.method)  # config selection is not silently overridden
        for change in ({'n_way': 5}, {'k_shot': 10}, {'q_query': 16}, {'method': 'resnet18'}):
            with self.assertRaises(ValueError):
                resolve_cross_config(change)
        self.assertEqual(resolve_cross_config({'method': ' protonet '})['method'], 'protonet')

    def test_summary_averages_tasks_before_seed_sd(self):
        from summarize_cross_dataset import aggregate
        root = Path(__file__).resolve().parent/'tmp'/f'cross_summary_test_{time.time_ns()}'
        for i, task in enumerate(TASKS):
            for seed in (3, 24):
                path = root/'method'/f'{task}_{seed}'
                path.mkdir(parents=True)
                value = .4 if (i % 2 == 0) == (seed == 3) else .6
                row = dict(method='maml', shots=1, task=task, seed=seed, accuracy=value,
                           episodes=1000, quick_checkpoint=False, data_sha256=task,
                           query_per_class=15, episode_manifest=f'{task}.json')
                (path/'summary.json').write_text(json.dumps(row))
        row = aggregate(root, seeds=(3, 24))[0]
        self.assertEqual(row['Average'], '50.00 +/- 0.00')
        self.assertEqual(row['T1'], '50.00 +/- 14.14')
        self.assertEqual(aggregate(root, seeds=(3, 24, 38))[0]['Average'], 'incomplete')

    def test_training_checkpoint_reload_matrix(self):
        root = Path(__file__).resolve().parent/'tmp'/f'cross_smoke_{time.time_ns()}'
        root.mkdir(parents=True)
        matrix = [('T1', m, shot) for m in METHODS for shot in (1, 5)]
        matrix += [(t, 'configured_maml', 1) for t in TASKS if t != 'T1']
        results = []
        for task, method, shots in matrix:
            cfg = resolve_cross_config(dict(task=task, method=method, k_shot=shots,
                  epochs=1, meta_batch_size=1, validation_episodes=1,
                  early_stop_confirmation_episodes=1, early_stop_on_perfect_validation=False,
                  test_meta_batch_size=2, ft_pretrain_epochs=1, ft_pretrain_max_batches=1,
                  result_root=str(root/'results')))
            seed_torch(cfg['seed'])
            obj = CrossLearner(cfg)
            path = root/cfg['model_name']
            checkpoint = obj.train(path, quick_test=True)
            restored = CrossLearner(cfg)
            result = restored.test(checkpoint)
            self.assertTrue(result['quick_checkpoint'])
            self.assertEqual(sum(map(sum, result['confusion_matrix'])), 90)
            self.assertTrue(0 <= result['accuracy'] <= 1)
            with self.assertRaises(FileExistsError):
                obj.train(path, quick_test=True)
            bad = copy.deepcopy(cfg); bad['experiment_signature'] = 'incompatible'
            with self.assertRaises(ValueError):
                CrossLearner(bad).test(checkpoint)
            results.append(result)
        (root/'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print('Verification artifact:', root, flush=True)


if __name__ == '__main__':
    unittest.main()
