"""CPU functional checks, no formal experiment artifacts."""
import copy
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
import maml_train_pic as trainer
from config_pu import PU_CONFIG
from config_cwru import CWRU_CONFIG
from classic_config import resolve_method_config
from run_cwru_maml import CWRUMAMLLearner
from transfer_models import TransferModel, TransferAlgorithm
from raw_signal_dataset import RawSignalDataset
from l2l_shim import MetaDataset


def main():
    torch.set_num_threads(2)
    trainer.device = torch.device('cpu')
    for dataset, base, cls in [('PU', PU_CONFIG, trainer.MAML_learner),
                               ('CWRU', CWRU_CONFIG, CWRUMAMLLearner)]:
        raw = RawSignalDataset(base)
        images = cls(3, dict(base, method='maml'))._get_pu_storage()
        # Exact PNG checks ran in RawSignalDataset; verify class/split cardinality too.
        assert raw.source_classes == images.source_classes
        for field in ('source_data', 'target_val_data', 'target_test_data'):
            for label in raw.source_classes:
                assert len(getattr(raw, field)[label]) == len(getattr(images, field)[label])
        for method in ('resnet18_ft', 'tl_wdcnn'):
            cfg = resolve_method_config(dict(base, method=method, n_way=3, k_shot=1,
                q_query=2, ft_pretrain_epochs=1, ft_pretrain_max_batches=1,
                ft_pretrain_batch_size=8, ft_steps=1, ft_validation_episodes=2,
                save_visualizations=False, tsne_selected_classes=[]))
            net = cls(3, cfg)
            if method == 'tl_wdcnn':
                net._pu_storage = raw
            storage = net._get_pu_storage()
            task = net.build_tasks('validation', 3, 1, 2, 2).sample()
            before = {k: v.clone() for k, v in net.model.state_dict().items()}
            torch.manual_seed(18)
            first = net._algorithm().clone().evaluate_episode(*task[:2], task[2])[1]
            torch.manual_seed(18)
            second = net._algorithm().clone().evaluate_episode(*task[:2], task[2])[1]
            assert first.shape == (6,3) and torch.allclose(first, second)
            assert all(torch.equal(before[k], v) for k,v in net.model.state_dict().items())
            assert all(p.grad is None for p in net.model.parameters())
            # Both datasets and methods: real source updates, validation FT, checkpoint reload, test.
            with tempfile.TemporaryDirectory(prefix='transfer_smoke_') as tmp:
                best = net.train(str(Path(tmp)/method), shots=1)
                assert any(not torch.equal(before[k], v) for k,v in net.model.state_dict().items())
                fresh = cls(3, cfg)
                if method == 'tl_wdcnn':
                    fresh._pu_storage = raw
                fresh.test(best, shots=1, meta_batch_size=2)
            print('PASS supervised/train/independent-FT/save/reload/test',dataset,method,flush=True)
    # Head-only mode and frozen BN must not modify encoder, including running buffers.
    model = TransferModel('tl_wdcnn', 9)
    cfg = dict(ft_scope='head', ft_bn_mode='frozen', ft_steps=2, ft_lr=.01)
    before = copy.deepcopy(model.state_dict())
    for shots in (1,5):
        sx=torch.randn(5*shots,1,4096); sy=torch.arange(5).repeat_interleave(shots)
        qx=torch.randn(10,1,4096)
        result=TransferAlgorithm(model,cfg).clone().evaluate_episode(sx,sy,qx)[1]
        assert result.shape==(10,5) and torch.isfinite(result).all()
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    print('ALL TRANSFER CHECKS PASSED',flush=True)

if __name__ == '__main__':
    main()
