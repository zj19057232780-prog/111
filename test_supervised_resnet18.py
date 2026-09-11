"""Real-data smoke test of direct, non-adaptive ResNet18 evaluation."""
import tempfile
from pathlib import Path
import torch
import maml_train_pic as trainer
from config_pu import PU_CONFIG
from config_cwru import CWRU_CONFIG
from run_cwru_maml import CWRUMAMLLearner
from supervised_training import evaluate, data_loader


def main():
    torch.set_num_threads(2)
    trainer.device = torch.device('cpu')
    for name, base, cls in [('PU', PU_CONFIG, trainer.MAML_learner),
                             ('CWRU', CWRU_CONFIG, CWRUMAMLLearner)]:
        cfg = dict(base, method='resnet18', supervised_epochs=1, supervised_batch_size=8,
                   supervised_max_batches=1, save_visualizations=False)
        net = cls(cfg['n_way'], cfg)
        storage = net._get_pu_storage()
        def forbid(*args, **kwargs):
            raise AssertionError('Ordinary ResNet18 must not build support/query tasks')
        net.build_tasks = forbid
        before = {k:v.clone() for k,v in net.model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix='resnet18_direct_') as tmp:
            best = net.train(str(Path(tmp)/'resnet18'), shots=1)
            assert any(not torch.equal(before[k],v) for k,v in net.model.state_dict().items())
            fresh = cls(cfg['n_way'], cfg)
            fresh._pu_storage = storage
            fresh.build_tasks = forbid
            result = fresh.test(best, shots=1)
            state = {k:v.clone() for k,v in fresh.model.state_dict().items()}
            # Shots are ignored; repeated tests preserve all parameters and BN buffers.
            repeated = fresh.test(best, shots=5)
            assert result == repeated
            assert all(torch.equal(state[k],v) for k,v in fresh.model.state_dict().items())
            x = torch.from_numpy(storage.target_test_data[storage.source_classes[0]][:2])
            with torch.no_grad():
                joint = fresh.model(x)[1][0]
                single = fresh.model(x[:1])[1][0]
            assert torch.allclose(joint, single, atol=1e-5, rtol=1e-4)
            assert result['samples'] == sum(map(len, storage.target_test_data.values()))
        print('PASS',name,'train/save/load/direct-test/no-adaptation',flush=True)
    print('ALL SUPERVISED RESNET18 CHECKS PASSED')

if __name__ == '__main__':
    main()
