"""CPU smoke tests; random temporary weights are not research results."""
import json
import csv
import tempfile
from pathlib import Path
import numpy as np
import torch
import maml_train_pic as trainer
from config_cwru import CWRU_CONFIG
from run_cwru_maml import CWRUMAMLLearner
from cwru_noise_test import add_awgn,load_target_windows,make_manifest,representation,run_noise_test
from cwru_dataset import CWRUMetaDataset


def main():
    trainer.device=torch.device('cpu');torch.set_num_threads(2)
    cfg=dict(CWRU_CONFIG,noise_test_episodes=2,noise_seeds=[101,202],save_visualizations=False)
    raw,provenance=load_target_windows(cfg)
    assert len(provenance)==200
    x=raw[cfg['class_names'][0]][0]
    for db in cfg['noise_snr_db']:
        noisy=add_awgn(x,db,101)
        actual=10*np.log10(np.mean((x-x.mean())**2)/np.mean((noisy-x)**2))
        assert abs(actual-db)<1e-9
        assert np.array_equal(noisy,add_awgn(x,db,101))
    assert np.array_equal(x,add_awgn(x,None,101))
    clean=representation(dict(cfg,method='protonet'),raw,None,101)
    storage=CWRUMetaDataset(cfg['root_path'],cfg['source_condition'],cfg['target_condition'],cfg['class_names'],cfg['img_size'])
    assert all(np.array_equal(clean[c],storage.target_test_data[c]) for c in cfg['class_names'])
    manifest=make_manifest(cfg,raw,2)
    for item in manifest:
        for idx in item['indices']:assert len(set(idx))==len(idx)
    with tempfile.TemporaryDirectory(prefix='cwru_noise_checks_') as tmp:
        manifest_hash=None
        for method in ['protonet','maml','matchingnet','relationnet','resnet18_ft','tl_wdcnn','resnet18']:
            local=dict(cfg,method=method,ft_steps=1,test_inner_steps=1,noise_snr_db=[0],noise_seeds=[101])
            if method=='protonet':local.update(noise_snr_db=cfg['noise_snr_db'],noise_seeds=[101,202])
            owner=CWRUMAMLLearner(local['n_way'],local)
            before={k:v.clone() for k,v in owner.model.state_dict().items()}
            checkpoint=Path(tmp)/(method+'_best')
            torch.save(owner.model.state_dict(),checkpoint)
            Path(str(checkpoint)+'.config.json').write_text(json.dumps(owner.pu_config),encoding='utf-8')
            out=run_noise_test(owner,checkpoint,output_dir=Path(tmp)/(method+'_out'))
            assert all(torch.equal(before[k],v) for k,v in owner.model.state_dict().items())
            meta=json.loads((out/'config.json').read_text(encoding='utf-8'))
            if method!='resnet18':
                if manifest_hash is None:manifest_hash=meta['episode_manifest_sha256']
                assert manifest_hash==meta['episode_manifest_sha256']
            with (out/'summary.csv').open(encoding='utf-8-sig') as f:summary=list(csv.DictReader(f))
            assert len(summary)==(23 if method=='protonet' else 2)
            assert all(np.isfinite(float(r['accuracy_percent'])) for r in summary)
            print('PASS',method,flush=True)
    print('ALL CWRU NOISE CHECKS PASSED')

if __name__=='__main__':main()
