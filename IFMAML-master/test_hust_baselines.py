import unittest
import numpy as np
from config_hust_huazhong import configuration, BASELINES
from hust_huazhong_inputs import native_transform, HUSTRawDataset


class HUSTBaselineTests(unittest.TestCase):
    def test_three_shot(self):
        from config_hust_huazhong import MODELS, TASKS
        for task in TASKS:
            for model in MODELS:
                if model == 'tubr_ml':
                    continue
                for mode in ('stft', 'nostft') if model in BASELINES else ('stft',):
                    cfg=configuration(task,model,3,shots=3,input_mode=mode)
                    self.assertEqual(cfg['k_shot'],3)
                    self.assertEqual(cfg['adaptation_steps'][3],3)
                    self.assertEqual(cfg['q_query'],15)
        with self.assertRaisesRegex(ValueError,'excluded'):
            configuration('H1','tubr_ml',3,shots=3)

    def test_supported_inputs_and_patience(self):
        for model in (*BASELINES,'cnn4','lsk','lsk_gc','full'):
            cfg=configuration('H1',model,3)
            self.assertEqual(cfg['early_stop_patience'],100)
            self.assertEqual(cfg['ft_patience'],100)
        for patience in (0, 20):
            cfg=configuration('H1','full',3,patience=patience)
            self.assertEqual(cfg['early_stop_patience'],patience)
            self.assertEqual(cfg['ft_patience'],patience)
            self.assertFalse(cfg['early_stop_on_perfect_validation'])
        with self.assertRaisesRegex(ValueError,'Invalid input/patience'):
            configuration('H1','full',3,patience=-1)
        with self.assertRaisesRegex(ValueError,'STFT only'):
            configuration('H1','full',3,input_mode='nostft')
        with self.assertRaisesRegex(ValueError,'exceeds'):
            configuration('H1','tubr_ml',3,5,input_mode='nostft')
        self.assertEqual(configuration('H1','tubr_ml',3,5)['q_query'],15)

    def test_native_shapes_and_amplitudes(self):
        expected={'maml':(1,4096),'amaml':(1,38,38),'tubr_ml':(3,84,84),'mfeml':(1,2000)}
        for model,shape in expected.items():
            n,transform=native_transform(model);x=np.arange(n,dtype=np.float64)
            y=transform(x)
            self.assertEqual(y.shape,shape)
            if model in ('amaml','mfeml'):np.testing.assert_array_equal(y.ravel(),x.astype(np.float32))
            elif model=='maml':self.assertAlmostEqual(float(y.mean()),0,places=5)
            else:self.assertEqual((y.min(),y.max()),(0,1))

    def test_tubr_real_interval_isolation(self):
        cfg=configuration('H4','tubr_ml',3,input_mode='nostft')
        ds=HUSTRawDataset(cfg)
        self.assertEqual(ds.source_data['healthy'].shape,(37,3,84,84))
        self.assertEqual(len(ds.target_val_data['healthy']),17)
        self.assertEqual(len(ds.target_test_data['healthy']),18)
        for cls in ds.source_classes:
            rows=[r for r in ds.window_manifest if r['class_name']==cls]
            for split in ('train','validation','test'):
                for condition in cfg['source_condition']+[cfg['target_condition']]:
                    group=sorted((r for r in rows if r['split']==split and r['condition']==condition),key=lambda r:r['start'])
                    self.assertTrue(all(a['stop']<=b['start'] for a,b in zip(group,group[1:])))
            self.assertLess(max(r['stop'] for r in rows if r['split']=='validation'),
                            min(r['start'] for r in rows if r['split']=='test'))

if __name__=='__main__':unittest.main()
