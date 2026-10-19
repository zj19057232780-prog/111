"""Real-data HUST protocol verification; run after --task all --preprocess."""
import unittest
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch

from config_hust import TASKS, resolve_hust_config
from hust_dataset import HUSTMetaDataset, HUSTTasks
from hust_preprocess import discover_records, read_signal, window_starts
from pu_preprocess import compute_stft_log_image


class HUSTProtocolTests(unittest.TestCase):
    def test_window_capacity(self):
        with self.assertRaises(ValueError):
            window_starts(0, 4096*19, 4096, 20)
        starts = window_starts(260096, 512000, 4096, 20)
        self.assertTrue(np.all(np.diff(starts) >= 4096))
        self.assertLessEqual(starts[-1]+4096, 512000)

    def test_raw_matrix_and_all_three_tasks(self):
        for task in TASKS:
            cfg = resolve_hust_config({'task': task})
            records, excluded = discover_records(cfg)
            self.assertEqual(len(records), 84)
            self.assertEqual(len(excluded), 15)
            storage = HUSTMetaDataset(cfg)
            self.assertEqual(Counter(r['split'] for r in storage.rows),
                             {'train': 2240, 'validation': 560, 'test': 560})
            grouped = defaultdict(list)
            hashes = defaultdict(set)
            for row in storage.rows:
                grouped[row['record']].append(row)
                hashes[row['split']].add(row['sha256'])
                self.assertNotEqual(row['bearing'], 6204)
            self.assertFalse(hashes['validation'] & hashes['test'])
            self.assertFalse(hashes['train'] & (hashes['validation'] | hashes['test']))
            for record, rows in grouped.items():
                ordered = sorted(rows, key=lambda r: r['start'])
                self.assertEqual(len(rows), 40)
                self.assertTrue(all(a['end'] <= b['start'] for a, b in zip(ordered, ordered[1:])))
                self.assertTrue(all(0 <= r['start'] < r['end'] <= r['length'] for r in rows))
                first = ordered[0]
                path = records[(first['fault'], first['bearing'], first['load'])]
                signal = read_signal(path)
                # Some released records are shorter than nominal ten seconds.
                self.assertEqual(len(signal), first['length'])
                self.assertGreaterEqual(len(signal), 40*cfg['window_size'])
                actual = compute_stft_log_image(signal[first['start']:first['end']], cfg)
                image = cv2.imdecode(np.frombuffer((storage.root/first['image']).read_bytes(), np.uint8), 0)
                np.testing.assert_array_equal(actual, image)

    def test_episode_pairing_and_lazy_test_loading(self):
        cfg = resolve_hust_config()
        storage = HUSTMetaDataset(cfg)
        val = HUSTTasks(storage, 'validation', 5, 1, 15, 12, 101, True)
        self.assertNotIn('test', storage._cache)
        one = HUSTTasks(storage, 'test', 5, 1, 15, 30, 20260914, True)
        five = HUSTTasks(storage, 'test', 5, 5, 15, 30, 20260914, True)
        repeat = HUSTTasks(HUSTMetaDataset(cfg), 'test', 5, 1, 15, 30, 20260914, True)
        self.assertEqual(one.plans, repeat.plans)
        seen = set()
        for a, b in zip(one.plans, five.plans):
            self.assertEqual(a['classes'], b['classes'])
            self.assertEqual(len(set(a['classes'])), 5)
            seen.update(a['classes'])
            for p, q in zip(a['selections'], b['selections']):
                self.assertEqual(p['query_ids'], q['query_ids'])
                self.assertEqual(p['support_ids'], q['support_ids'][:1])
                self.assertFalse(set(q['support_ids']) & set(q['query_ids']))
        self.assertEqual(seen, set(range(7)))
        batch = five.sample_with_original_labels()
        self.assertEqual(batch[0].shape, (25, 1, 64, 64))
        self.assertEqual(batch[2].shape, (75, 1, 64, 64))
        self.assertTrue(torch.equal(batch[6][batch[3]], batch[5]))

    def test_second_order_gradient_and_buffer_isolation(self):
        from run_hust_maml import HUSTMAMLLearner, device
        torch.set_num_threads(4)
        learner = HUSTMAMLLearner(resolve_hust_config())
        before = {n: b.clone() for n, b in learner.model.named_buffers()}
        task = learner.build_tasks('validation', num_tasks=2).sample()
        clone = learner._algorithm().clone()
        error, _, _, _ = learner.fast_adapt(task, clone, torch.nn.CrossEntropyLoss(), 3)
        error.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in learner.model.parameters()))
        self.assertTrue(all(torch.equal(before[n], b) for n, b in learner.model.named_buffers()))
        fresh = learner._algorithm().clone()
        self.assertTrue(all(torch.equal(fresh.buffers[n], b) for n, b in before.items()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
