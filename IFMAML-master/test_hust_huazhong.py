"""Protocol and compatibility tests; no formal training."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from config_hust_huazhong import configuration, TASKS, TASK_PAIRS, MODELS, CLASSES, PREFIXES
from hust_huazhong_preprocess import read_signal, split_indices, verify_cache
from hust_huazhong_dataset import HUSTMetaDataset, HUSTMAMLLearner
from run_hust_huazhong import summary


class HUSTTests(unittest.TestCase):
    def test_presets_and_fixed_protocol(self):
        from config_pu import PU_CONFIG
        import copy
        original=copy.deepcopy(PU_CONFIG)
        self.assertEqual(TASK_PAIRS, {'H1':(20,40),'H2':(40,20),'H3':(40,60),
                                     'H4':(60,40),'H5':(60,80),'H6':(80,60)})
        for task,target in TASKS.items():
            for model in ('cnn4','lsk','lsk_gc','full'):
                cfg=configuration(task,model,3,5)
                self.assertEqual(cfg['target_condition'],str(target))
                self.assertEqual(cfg['source_condition'],[str(TASK_PAIRS[task][0])])
                self.assertNotIn(str(target),cfg['source_condition'])
                self.assertEqual(len(cfg['class_groups']),9)
                self.assertEqual((cfg['n_way'],cfg['q_query'],cfg['epochs'],cfg['meta_batch_size']),(5,15,500,16))
                self.assertEqual(cfg['task_weighting_mode'],'ggm_rw' if model=='full' else 'none')
                self.assertEqual(cfg['backbone'],'cnn4' if model=='cnn4' else 'lsk_lite')
                self.assertEqual(cfg['attention_module'],'gcnet' if model in ('lsk_gc','full') else 'none')
        self.assertEqual(PU_CONFIG,original)

    def test_disjoint_time_blocks(self):
        val,test=split_indices(64)
        self.assertEqual((len(val),len(test)),(31,32))
        self.assertFalse(set(val)&set(test))
        self.assertEqual(min(test)*4096-(max(val)+1)*4096,4096)
        self.assertGreaterEqual(min(len(val),len(test)),5+15)
        self.assertEqual(PREFIXES['outer_severe'],'O')
        self.assertEqual(PREFIXES['ball_medium'],'0.5X_B')

    def test_real_export_channels(self):
        root=Path(configuration('H1','full',3)['raw_root'])/'Raw data (原始数据)'
        p=root/'H_20Hz.xls'
        if not p.exists():self.skipTest('Local raw data unavailable')
        x=read_signal(p,'X');y=read_signal(p,'Y')
        self.assertEqual(len(x),262144)
        self.assertAlmostEqual(x[0],-.013681)
        self.assertFalse(np.array_equal(x,y))
        with self.assertRaises(ValueError):read_signal(p,'Speed')

    def test_real_task_pools_and_episode_separation(self):
        for task in TASKS:
            cfg=configuration(task,'cnn4',3,5)
            if not Path(cfg['root_path']).exists():self.skipTest('Run preprocessing first')
            ds=HUSTMetaDataset(cfg)
            for cls in CLASSES:
                self.assertEqual(len(ds.source_data[cls]),64)
                self.assertEqual(len(ds.target_val_data[cls]),31)
                self.assertEqual(len(ds.target_test_data[cls]),32)
                a={hashlib.sha256(x.tobytes()).hexdigest() for x in ds.target_val_data[cls]}
                b={hashlib.sha256(x.tobytes()).hexdigest() for x in ds.target_test_data[cls]}
                self.assertFalse(a&b)
            learner=HUSTMAMLLearner(5,cfg);learner._pu_storage=ds
            sampler=learner.build_tasks('test',5,5,15,2)
            original=np.random.choice;draws=[]
            def capture(values,size,replace=False):
                picked=original(values,size,replace=replace);draws.append(picked);return picked
            with patch('numpy.random.choice',side_effect=capture):episode=sampler.sample_with_original_labels()
            self.assertEqual(episode[0].shape,(25,1,64,64))
            self.assertEqual(episode[2].shape,(75,1,64,64))
            self.assertEqual(len(set(episode[6].tolist())),5)
            self.assertTrue(all(len(set(d))==len(d) for d in draws))

    def test_summary_is_seed_level_and_rejects_incomplete(self):
        rows=[dict(method='full',task=t,seed=s,status='OK',accuracy_pct=70+i,macro_f1_pct=60+i)
              for t in TASKS for i,s in enumerate((3,24,38))]
        report=summary(rows)['full']['accuracy_pct']
        self.assertEqual(report['all_conditions'],dict(mean=71,sample_sd=1))
        self.assertNotIn('all_conditions',summary(rows[:-1])['full']['accuracy_pct'])
        import tempfile, csv
        from run_hust_huazhong import write_summary
        with tempfile.TemporaryDirectory() as tmp:
            write_summary(Path(tmp),rows)
            with (Path(tmp)/'summary.csv').open(encoding='utf-8-sig') as f:
                reader=csv.DictReader(f)
                self.assertEqual(reader.fieldnames,['model','metric',*TASKS,'all_conditions'])
                full=next(r for r in reader if r['model']=='full' and r['metric']=='accuracy_pct')
                self.assertNotEqual(full['H6'],'incomplete')
        for r in rows:r['status']='SMOKE_ONLY'
        self.assertEqual(summary(rows)['full']['accuracy_pct']['H1'],'incomplete')

    def test_pu_default_training_unchanged_and_hust_timing(self):
        import contextlib, io, importlib.util, tempfile, copy, csv
        import torch
        import maml_train_pic as current
        from config_pu import PU_CONFIG, resolve_pu_config
        from run_table_row import configure
        from my_utils.init_utils import seed_torch
        root=Path(__file__).resolve().parent
        old_path=root.parent/'archive/code_before_hust_huazhong_20261019/maml_train_pic.py'
        if not old_path.exists():self.skipTest('Local baseline snapshot not installed')
        spec=importlib.util.spec_from_file_location('pu_baseline_trainer',old_path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        old.device=current.device=torch.device('cpu');torch.set_num_threads(2)
        cfg=configure(resolve_pu_config(copy.deepcopy(PU_CONFIG)),'cnn4maml','N09_M07_F10','pu',1,3,1)
        cfg.update(root_path=str(root/'pu_data_processed'),hidden_size=8,meta_batch_size=1,
                   tasks_per_epoch=2,adaptation_steps={1:1},early_stop_on_perfect_validation=False,
                   early_stop_patience=0,save_visualizations=True)
        if not Path(cfg['root_path']).exists():self.skipTest('Local PU data unavailable')
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            cfg['visualization_dir']=tmp
            states=[];histories=[]
            for i,module in enumerate((old,current)):
                seed_torch(3);learner=module.MAML_learner(5,cfg)
                best=learner.train(str(Path(tmp)/str(i)),shots=1)
                states.append(torch.load(best,map_location='cpu',weights_only=True))
                history_path=next(Path(tmp).rglob('*training_history.csv'))
                with history_path.open(encoding='utf-8-sig') as f:
                    histories.append(list(csv.DictReader(f)))
                history_path.unlink()
            self.assertEqual(states[0].keys(),states[1].keys())
            self.assertTrue(all(torch.equal(states[0][k],states[1][k]) for k in states[0]))
            for key in ('train_loss','train_acc','valid_loss','valid_acc','best_epoch'):
                self.assertEqual(histories[0][0][key],histories[1][0][key])
            events=[];original_step=torch.optim.Adam.step
            def step(optimizer,*args,**kwargs):
                events.append('update');return original_step(optimizer,*args,**kwargs)
            class TimedLearner(current.MAML_learner):
                def build_tasks(self,mode='train',*args,**kwargs):
                    tasks=super().build_tasks(mode,*args,**kwargs)
                    original_sample=tasks.sample
                    def sample():events.append(mode);return original_sample()
                    tasks.sample=sample
                    return tasks
            cfg['validate_after_update']=True
            seed_torch(3)
            with patch.object(torch.optim.Adam,'step',step):
                TimedLearner(5,cfg).train(str(Path(tmp)/'timing'),shots=1)
            self.assertLess(events.index('update'),events.index('validation'))

if __name__=='__main__':unittest.main()
