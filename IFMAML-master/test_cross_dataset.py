"""Real data isolation, native adapters, and all supported model lifecycles."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from config_cross_dataset import METHODS, TASKS, TASK_GROUPS, resolve_cross_config, CLASS_NAMES
from cross_dataset_data import CrossDataset, CrossTasks, preprocess_cross
from cross_dataset_models import native_raw_input
from run_cross_dataset import CrossLearner, evaluation_state, parse_args
from maml_train_pic import capture_rng_state
from my_utils.init_utils import seed_torch


def config(**changes):
    return resolve_cross_config(changes)


class CrossProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        seed_torch(24)
        cls.output=Path(__file__).resolve().parent/'tmp'/f'cross_ab5_ms_v2_verification_{time.time_ns()}'
        cls.output.mkdir(parents=True,exist_ok=True)

    def test_01_config_and_native_no_stft(self):
        self.assertEqual(config(method='SCG-MAML')['method'],'configured_maml')
        self.assertEqual(config(method='TUBR-ML')['method'],'tubr_ml')
        self.assertEqual(config(k_shot=3)['n_way'],5)
        for changes in ({'input_mode':'nostft'},{'n_way':3},{'k_shot':10},
                        {'protocol':'Cross-ABC-3way-targetval-v1'},
                        {'protocol':'Cross-AB-5way-targetval-v1'}, {'q_query':15}):
            with self.assertRaises(ValueError):config(**changes)
        self.assertFalse(parse_args(['--test']).train)
        cfg=config(task='T1',method='maml',input_mode='nostft')
        # No PNG decode, STFT calculation, or resampling even on preprocess/reuse.
        with (patch('cross_dataset_data._image',side_effect=AssertionError('STFT called')),
             patch('cross_dataset_data.resample_poly',side_effect=AssertionError('resample called')),
             patch('cross_dataset_data.cv2.imdecode',side_effect=AssertionError('PNG read'))):
            preprocess_cross(cfg)
            ds=CrossDataset(cfg)
            x=ds.images('validation')
            self.assertEqual(x['N'].shape,(16,1,4096))
            self.assertNotIn('test',ds._cache)

    def test_02_original_adapter_values(self):
        from comparison_models.mu2025_amaml.raw_reshape import reshape_samples as amaml
        from comparison_models.wang2025_tubr_ml.raw_reshape import reshape_samples as tubr
        from comparison_models.li2025_mfeml.raw_input import raw_samples as mfeml
        values=np.linspace(-2,3,7056)
        for method,expected in [('amaml',amaml(values[:1444])),('tubr_ml',tubr(values)),
                                ('mfeml',mfeml(values[:2000]))]:
            np.testing.assert_array_equal(native_raw_input(values,method,'A'),expected)
        for domain,length in [('A',3072),('B',4096)]:
            expected=(values[:length]-values[:length].mean())/(values[:length].std()+1e-8)
            np.testing.assert_array_equal(native_raw_input(values,'maml',domain)[0],expected.astype(np.float32))

    def test_03_splits_paired_shots_and_representations(self):
        for task in TASKS:
            ds=CrossDataset(config(task=task))
            for split,count in [('train',128),('validation',16),('test',16)]:
                self.assertTrue(all(len(v)==count for v in ds.by_split[split].values()))
            pools=[{r['raw'] for rs in ds.by_split[s].values() for r in rs} for s in ('train','validation','test')]
            self.assertFalse(pools[0]&pools[1] or pools[0]&pools[2] or pools[1]&pools[2])
            plans=[]
            for shot in (1,3,5):
                ts=CrossTasks(ds,'test',5,shot,10,3,89,True);plans.append(ts.plans)
                b=ts.sample_with_original_labels()
                self.assertEqual(tuple(b[0].shape),(5*shot,1,64,64))
                self.assertEqual(tuple(b[2].shape),(50,1,64,64))
                self.assertTrue(torch.equal(b[6][b[3]],b[5]))
            raw=CrossDataset(config(task=task,method='tubr_ml',input_mode='nostft'))
            rawtasks=CrossTasks(raw,'test',5,1,10,3,89,True)
            self.assertEqual(plans[0],rawtasks.plans)
            for episodes in zip(*plans):
                for selections in zip(*(e['selections'] for e in episodes)):
                    a,b,c=selections
                    self.assertEqual(a['query_ids'],c['query_ids'])
                    self.assertEqual(b['support_ids'],c['support_ids'][:3])
                    self.assertEqual(a['support_ids'],c['support_ids'][:1])
                    self.assertFalse(set(c['support_ids'])&set(c['query_ids']))
            # Tampered split manifests are rejected, without altering the real cache.
            with tempfile.TemporaryDirectory(dir=self.output) as tmp:
                m=copy.deepcopy(ds.manifest);m['windows'][0]['start']=m['windows'][0]['original_end']
                Path(tmp,'manifest.json').write_text(json.dumps(m),encoding='utf-8')
                bad=config(task=task,processed_root=tmp)
                with self.assertRaises(ValueError):CrossDataset(bad)

    def test_multisource_episode_roles_and_conditions(self):
        for task in TASKS:
            cfg=config(task=task,method='protonet')
            ds=CrossDataset(cfg)
            self.assertEqual(len(cfg['source_conditions']),4)
            self.assertTrue(all(r['domain']==cfg['source_domain'] and r['condition'] in cfg['source_conditions']
                                for rs in ds.by_split['train'].values() for r in rs))
            for split in ('validation','test'):
                self.assertTrue(all(r['domain']==cfg['target_domain'] and r['condition']==cfg['target_condition']
                                    for rs in ds.by_split[split].values() for r in rs))
            train=CrossTasks(ds,'train',5,5,10,100,33,False)
            seen=set()
            for _ in range(80):
                plan=train._new_plan();condition=plan['source_condition'];seen.add(condition)
                for selection in plan['selections']:
                    ids=selection['support_ids']+selection['query_ids']
                    self.assertTrue(all(v.split('/')[1]==condition for v in ids))
            self.assertEqual(seen,set(cfg['source_conditions']))

    def test_04_all_methods_query_label_and_state_isolation(self):
        cases=[(mode,m) for mode in ('stft','nostft') for m in METHODS
               if mode=='stft' or m!='configured_maml']
        for mode,method in cases:
            with self.subTest(input=mode,method=method):
                cfg=config(method=method,input_mode=mode,q_query=2,validation_inner_steps=1,
                           test_inner_steps=1,ft_steps=1)
                obj=CrossLearner(cfg)
                task=obj.build_tasks('validation',num_tasks=1).sample()
                state={n:p.clone() for n,p in obj.model.state_dict().items()}
                rng=torch.get_rng_state().clone()
                with evaluation_state(obj.model,88):
                    a=obj.run_episode(task,False,1)[-1].detach()
                changed=(*task[:3],(task[3]+1)%5)
                with evaluation_state(obj.model,88):
                    b=obj.run_episode(changed,False,1)[-1].detach()
                self.assertTrue(torch.equal(a,b))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertTrue(all(torch.equal(state[n],p) for n,p in obj.model.state_dict().items()))

    def test_05_training_checkpoint_reload_matrix(self):
        results=[]
        for mode in ('stft','nostft'):
            for method in METHODS:
                if mode=='nostft' and method=='configured_maml':continue
                for shot in (1,3,5):
                    task='T1' if shot!=3 else 'T5'
                    with self.subTest(input=mode,method=method,shot=shot):
                        cfg=config(task=task,method=method,input_mode=mode,k_shot=shot,
                            epochs=1,meta_batch_size=1,q_query=2,validation_episodes=1,
                            validation_interval=1,validation_inner_steps=1,test_inner_steps=1,ft_steps=1,
                            adaptation_steps={1:1,3:1,5:1},test_meta_batch_size=2,
                            ft_pretrain_epochs=1,ft_validation_interval=1,ft_pretrain_max_batches=1,
                            result_root=str(self.output/'results'))
                        seed_torch(cfg['seed'])
                        obj=CrossLearner(cfg)
                        initial={n:p.detach().clone() for n,p in obj.model.named_parameters()}
                        path=self.output/cfg['model_name']
                        checkpoint=obj.train(path,quick_test=True)
                        changed=[n for n,p in obj.model.named_parameters() if not torch.equal(initial[n],p)]
                        self.assertTrue(changed,'No trained parameters')
                        for required in ({'amaml':'blocks.0.4', 'tubr_ml':'attenuator',
                                          'mfeml':'channel_conv', 'matchingnet':'lstm',
                                          'relationnet':'relation'}.get(method),):
                            if required:self.assertTrue(any(required in n for n in changed),changed)
                        restored=CrossLearner(cfg)
                        result=restored.test(checkpoint)
                        self.assertTrue(result['quick_checkpoint'])
                        self.assertEqual(len(result['confusion_matrix']),5)
                        self.assertEqual(sum(map(sum,result['confusion_matrix'])),20)
                        self.assertTrue(0<=result['accuracy']<=1)
                        with self.assertRaises(FileExistsError):obj.train(path,quick_test=True)
                        bad=dict(cfg,experiment_signature='legacy-or-incompatible')
                        with self.assertRaises(ValueError):CrossLearner(bad).test(checkpoint)
                        results.append(dict(method=method,input_mode=mode,shot=shot,task=task,
                                            checkpoint=checkpoint,accuracy=result['accuracy'],
                                            trained_parameter_count=len(changed)))
        (self.output/'verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        self.assertEqual(len(results),51)
        print('Verified 51 model/input/shot train-reload-test lifecycles:',self.output,flush=True)

    def test_06_summary_protocol_and_metric_grouping(self):
        from summarize_cross_dataset import aggregate
        with tempfile.TemporaryDirectory(dir=self.output) as tmp:
            for mode in ('stft','nostft'):
                for task in TASKS:
                    for i,seed in enumerate((3,24,38)):
                        r=dict(protocol='Cross-AB-MS-5way-targetval-v2',n_way=5,class_names=CLASS_NAMES,
                            method='maml',input_mode=mode,task=task,shots=3,seed=seed,episodes=1000,
                            quick_checkpoint=False,
                            accuracy=(.5+i*.1 if int(task[1:])%2 else .7-i*.1),
                            macro_f1=(.4+i*.1 if int(task[1:])%2 else .6-i*.1),
                            data_sha256=task,query_per_class=10,episode_sha256=task,
                            experiment_signature=f'{mode}_{seed}')
                        cfg=config(task=task)
                        r.update(source=cfg['source_domain'],target=cfg['target_domain'],
                                 source_conditions=cfg['source_conditions'],target_condition=cfg['target_condition'])
                        path=Path(tmp)/mode/f'{task}_{seed}'/'summary.json';path.parent.mkdir(parents=True)
                        path.write_text(json.dumps(r),encoding='utf-8')
            rows=aggregate(tmp)
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['T1_accuracy'],'60.00 +/- 10.00')
            self.assertEqual(rows[0]['T1_macro_f1'],'50.00 +/- 10.00')
            self.assertNotIn('Average',rows[0])
            self.assertEqual(rows[0]['CWRU_to_PU_accuracy'],'60.00 +/- 0.00')
            self.assertEqual(rows[0]['PU_to_CWRU_macro_f1'],'50.00 +/- 0.00')
            self.assertEqual(aggregate(tmp,seeds=(3,24,38,42))[0]['T1_accuracy'],'incomplete')


if __name__=='__main__':
    unittest.main(verbosity=2)
