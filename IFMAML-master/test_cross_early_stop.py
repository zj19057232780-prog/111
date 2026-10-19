"""Exercise the real training loop with controlled validation observations."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from config_cross_dataset import METHODS, resolve_cross_config
from run_cross_dataset import CrossLearner, parse_args


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 5)

    def forward(self, x):
        return x, self.head(x)


class CrossEarlyStopTests(unittest.TestCase):
    def run_training(self, method='maml', patience=100, epochs=130, interval=10,
                     accuracies=None, force_own_patience=False):
        cfg = resolve_cross_config(dict(method=method, early_stop_patience=patience,
            epochs=epochs, validation_interval=interval, ft_pretrain_epochs=epochs,
            ft_validation_interval=interval, meta_batch_size=1))
        # The test isolates budget/validation logic from SCG's pilot weighting.
        cfg['task_weighting_mode'] = 'none'
        if force_own_patience:
            cfg['early_stop_patience'] = 1
        learner = object.__new__(CrossLearner)
        learner.pu_config, learner.method, learner.model = cfg, cfg['method'], TinyModel()
        learner.storage = SimpleNamespace(signature='synthetic-test', images=lambda split:
            {str(i): np.ones((2, 2), dtype=np.float32) for i in range(5)})
        learner.build_tasks = lambda *a, **k: SimpleNamespace(sample=lambda: None)
        learner.run_episode = lambda *a, **k: (
            sum(p.square().sum() for p in learner.model.parameters()), torch.tensor(.5), None, None, None)
        observations, states = [], []

        def evaluate(*args):
            i = len(observations)
            acc = accuracies[min(i, len(accuracies)-1)] if accuracies else .5
            observations.append(acc)
            states.append({k: v.detach().clone() for k, v in learner.model.state_dict().items()})
            return 1 / (i+1), acc  # Falling loss with tied accuracy must not reset patience.

        learner.evaluate = evaluate
        temp_root = Path(__file__).resolve().parent/'tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temp, patch('run_cross_dataset.code_files', return_value={}), \
                patch('run_cross_dataset.device', torch.device('cpu')), contextlib.redirect_stdout(io.StringIO()):
            base = Path(temp)/'run'
            best = learner.train(base)
            saved = json.loads(Path(best+'.config.json').read_text(encoding='utf-8'))
            history = json.loads(Path(str(base)+'_history.json').read_text(encoding='utf-8'))
            summary = json.loads(Path(str(base)+'_training_summary.json').read_text(encoding='utf-8'))
            weights = torch.load(best, map_location='cpu', weights_only=True)
        validation_epochs = [r['epoch'] for r in history if 'validation_accuracy' in r]
        selected = states[validation_epochs.index(saved['best_epoch'])]
        self.assertTrue(all(torch.equal(weights[k], v) for k, v in selected.items()))
        self.assertEqual(saved['epochs_run'], len(history))
        self.assertEqual(saved['stop_reason'], summary['stop_reason'])
        self.assertTrue(saved['training_complete'])
        return saved, history

    def test_defaults_overrides_and_cli(self):
        for method in METHODS:
            with self.subTest(method=method):
                cfg = resolve_cross_config(dict(method=method))
                self.assertEqual(cfg['early_stop_patience'], 0 if method=='configured_maml' else 100)
                self.assertEqual(resolve_cross_config(dict(method=method, early_stop_patience=0))['early_stop_patience'], 0)
        self.assertEqual(resolve_cross_config(dict(method='SCG-MAML',early_stop_patience=5))['early_stop_patience'],0)
        for value in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                resolve_cross_config(dict(early_stop_patience=value))
        self.assertEqual(parse_args(['--patience','12']).patience,12)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--patience','-1'])
        a=resolve_cross_config(dict(method='maml',early_stop_patience=0))
        b=resolve_cross_config(dict(method='maml',early_stop_patience=100))
        self.assertNotEqual(a['experiment_signature'],b['experiment_signature'])

    def test_default_counts_epochs_not_validation_calls(self):
        saved, rows = self.run_training()
        self.assertEqual(saved['epochs_run'],110)
        self.assertEqual(saved['best_accuracy_epoch'],10)
        self.assertEqual(saved['best_epoch'],110)  # Existing loss tie-break still selects best.
        self.assertEqual(saved['stop_reason'],'validation_patience')
        self.assertTrue(rows[-1]['patience_stop_triggered'])
        self.assertEqual(rows[-1]['epochs_without_improvement'],100)

    def test_strict_improvement_resets_and_checks_next_validation(self):
        saved, rows = self.run_training(patience=15,epochs=60,accuracies=[.5,.6,.6,.6])
        self.assertEqual(saved['epochs_run'],40)
        self.assertEqual(saved['best_accuracy_epoch'],20)
        self.assertEqual(rows[19]['epochs_without_improvement'],0)
        self.assertEqual(rows[-1]['epochs_without_improvement'],20)

    def test_disabled_and_final_non_interval_validation(self):
        saved, rows = self.run_training(patience=0,epochs=35)
        self.assertEqual(saved['epochs_run'],35)
        self.assertEqual(saved['stop_reason'],'epochs_limit')
        self.assertIn('validation_accuracy',rows[-1])
        self.assertFalse(any(r['patience_stop_triggered'] for r in rows))

    def test_own_model_never_stops_even_with_forced_patience(self):
        saved, rows = self.run_training(method='configured_maml',patience=1,epochs=35,force_own_patience=True)
        self.assertEqual(saved['epochs_run'],35)
        self.assertEqual(saved['early_stop_patience'],0)
        self.assertFalse(any(r['patience_stop_triggered'] for r in rows))

    def test_transfer_loop_uses_same_rule(self):
        saved, rows = self.run_training(method='tl_wdcnn',patience=4,epochs=12,interval=2)
        self.assertEqual(saved['epochs_run'],6)
        self.assertEqual(saved['best_accuracy_epoch'],2)
        self.assertEqual(saved['stop_reason'],'validation_patience')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
