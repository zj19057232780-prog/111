"""HUST inputs; all model-specific transforms are imported from existing implementations."""
from pathlib import Path
import importlib.util
import numpy as np
from raw_signal_dataset import RawSignalDataset
from config_hust_huazhong import CLASSES, PREFIXES, PAPERS
from hust_huazhong_preprocess import read_signal, sha256


def native_transform(method):
    if method not in PAPERS:
        return 4096, lambda x: ((x-x.mean())/(x.std()+1e-8)).astype(np.float32)[None]
    filename='raw_input.py' if method=='mfeml' else 'raw_reshape.py'
    path=Path(__file__).resolve().parent.parent/'comparison_models'/PAPERS[method]/filename
    spec=importlib.util.spec_from_file_location('hust_input_'+method,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.SAMPLE_POINTS, module.raw_samples if method=='mfeml' else module.reshape_samples


class HUSTRawDataset(RawSignalDataset):
    def __init__(self,cfg):
        self.cfg=cfg
        self.source_classes=list(CLASSES)
        self.source_data={};self.target_val_data={};self.target_test_data={}
        self.window_manifest=[]
        points,transform=native_transform(cfg['model_variant'])
        footprint=7056 if cfg['model_variant']=='tubr_ml' else 4096
        raw=Path(cfg['raw_root'])/'Raw data (原始数据)'
        for cls in CLASSES:
            parts=[]
            for condition in cfg['source_condition']+[cfg['target_condition']]:
                path=raw/f'{PREFIXES[cls]}_{condition}Hz.xls'
                signal=read_signal(path,cfg['channel']);digest=sha256(path)
                regions=[('train',0,len(signal))] if condition!=cfg['target_condition'] else [
                    ('validation',0,31*4096),('test',32*4096,len(signal))]
                for split,start,stop in regions:
                    windows=[]
                    for offset in range(start,stop-footprint+1,footprint):
                        windows.append(transform(signal[offset:offset+points]))
                        self.window_manifest.append(dict(condition=condition,class_name=cls,split=split,
                            file=path.name,file_sha256=digest,start=offset,stop=offset+points,
                            footprint_stop=offset+footprint,channel=cfg['channel']))
                    if len(windows)<cfg['k_shot']+cfg['q_query']:
                        raise ValueError(f'{cls}/{split}: insufficient disjoint windows for shot+query')
                    array=np.stack(windows)
                    if split=='train':parts.append(array)
                    else:(self.target_val_data if split=='validation' else self.target_test_data)[cls]=array
            self.source_data[cls]=np.concatenate(parts)
        self.set_mode('train')

    def to_maml_dataset_format(self,mode='train'):
        view=self.for_mode(mode)
        return view._x,view._y


def storage_for(cfg):
    if cfg['input_mode'] in ('nostft','no_stft'):
        return HUSTRawDataset(cfg)
    from hust_huazhong_dataset import HUSTMetaDataset
    return HUSTMetaDataset(cfg)


def create_learner(cfg):
    from maml_train_pic import MAML_learner
    from raw_fewshot import RawFewShotLearner
    from raw_transfer import RawTransferLearner
    storage=storage_for(cfg)
    if cfg['model_variant']=='tl_wdcnn':
        learner=RawTransferLearner(5,cfg)
        if cfg['input_mode']=='stft':
            flat=RawSignalDataset.__new__(RawSignalDataset)
            flat.source_classes=list(storage.source_classes)
            for key in ('source_data','target_val_data','target_test_data'):
                setattr(flat,key,{c:a.reshape(len(a),1,-1) for c,a in getattr(storage,key).items()})
            flat.set_mode('train');storage=flat
            learner.pu_config['input_representation']='stft_flattened_4096'
    elif cfg['input_mode']=='nostft':learner=RawFewShotLearner(5,cfg)
    else:learner=MAML_learner(5,cfg)
    learner._pu_storage=storage
    return learner
