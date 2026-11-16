"""Regression checks for routing, reporting and compatibility after simplification."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiment as entry


class EntryTests(unittest.TestCase):
    def test_three_shot_all_methods_except_tubr(self):
        for dataset in ('pu','cwru'):
            for method in entry.METHODS:
                if method == 'tubr_ml':
                    continue
                args=entry.parse_args(['--dataset',dataset,'--method',method,'--shots','3','--input','stft'])
                for task in args.tasks:
                    cfg=entry.base_config(args,task,3)
                    self.assertEqual(cfg['k_shot'],3)
                    self.assertEqual(cfg['adaptation_steps'][3],3)

    def test_task_subset_and_aliases(self):
        args = entry.parse_args(['--dataset','cwru','--method','SCG-MAML','--task','C0,C3','--seed','3'])
        self.assertEqual(args.tasks, ['C0','C3'])
        self.assertEqual(args.input, 'stft')
        for task in args.tasks:
            cfg = entry.base_config(args, task, 3)
            self.assertEqual(len(cfg['source_condition']), 3)
            self.assertNotIn(cfg['target_condition'], cfg['source_condition'])
            self.assertEqual((cfg['n_way'],cfg['q_query'],cfg['early_stop_patience']), (5,15,100))
            self.assertFalse(cfg['early_stop_on_perfect_validation'])

    def test_reject_invalid_jobs(self):
        for extra in [ ['--tasks','P1','P1'], ['--tasks','C0'], ['--seeds','3','3'],
                       ['--method','SCG-MAML','--input','nostft'], ['--method','A1','--shots','5'],
                       ['--patience','-1'] ]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                entry.parse_args(['--dataset','pu','--method','maml']+extra)

    def test_presets_and_inputs(self):
        for dataset in ('pu','cwru'):
            for method in entry.METHODS:
                args = entry.parse_args(['--dataset',dataset,'--method',method])
                cfg = entry.base_config(args,args.tasks[0],24)
                self.assertEqual(cfg['test_meta_batch_size'],1000)
                self.assertEqual(cfg['q_query'],15)
                self.assertEqual(cfg['early_stop_patience'],100)
                if method in entry.ABLATIONS:
                    self.assertEqual(cfg['attention_module'],'gcnet' if method in ('a2','a3') else 'none')
                    self.assertEqual(cfg['task_weighting_mode'],'ggm_rw' if method=='a3' else 'none')

    def test_partial_results_never_become_full_table_average(self):
        rows = [dict(task='P1', seed=s, status='OK', accuracy_pct=v, macro_f1_pct=v-1)
                for s,v in zip(entry.SEEDS,(70,80,90))]
        result = entry.summarize(rows,['P1','P2','P3','P4'])
        self.assertEqual(result['accuracy_pct']['P1'],dict(mean=80,sample_sd=10))
        self.assertNotIn('all_conditions',result['accuracy_pct'])
        rows += [dict(row,task=t,accuracy_pct=100) for t in ('P2','P3','P4') for row in rows.copy()]
        result = entry.summarize(rows,['P1','P2','P3','P4'])
        self.assertEqual(result['accuracy_pct']['all_conditions'],dict(mean=95,sample_sd=2.5))

    def test_model_refactor_identical_weights_forward_and_gradients(self):
        import torch
        import maml_model as current
        old_path = entry.ROOT/'archive/code_before_simplification_20261019/IFMAML-master/maml_model.py'
        if not old_path.is_file():
            self.skipTest('Local pre-refactor snapshot is not bundled with deployments')
        spec = importlib.util.spec_from_file_location('original_maml_model',old_path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        torch.set_num_threads(2)
        for options in ({}, {'attention_module':'gcnet'}, {'attention_module':'ema'},
                        {'frequency_module':'gfnet_lite'}, {'denoise_module':'arsm'}):
            with self.subTest(options=options):
                torch.manual_seed(37)
                before = old.LSKLiteBackbone(stage_channels=(8,8,8),**options).eval()
                torch.manual_seed(37)
                after = current.LSKLiteBackbone(stage_channels=(8,8,8),**options).eval()
                self.assertEqual(before.state_dict().keys(),after.state_dict().keys())
                self.assertTrue(all(torch.equal(v,after.state_dict()[k]) for k,v in before.state_dict().items()))
                x = torch.randn(2,1,64,64)
                a,b = before(x),after(x)
                self.assertTrue(torch.equal(a,b))
                a.sum().backward(); b.sum().backward()
                self.assertTrue(all(torch.equal(a.grad,b.grad) for a,b in zip(before.parameters(),after.parameters())))


if __name__ == '__main__':
    unittest.main()
