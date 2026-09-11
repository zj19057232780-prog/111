"""Run: python test_classic_interfaces.py. CPU-only, no formal result files."""
import copy
import tempfile
from pathlib import Path
import torch
import maml_train_pic as trainer
from config_pu import PU_CONFIG
from config_cwru import CWRU_CONFIG
from classic_config import resolve_method_config
from classic_models import ClassicMetricModel
from run_cwru_maml import CWRUMAMLLearner

def main():
    torch.set_num_threads(2)
    trainer.device = torch.device('cpu')
    methods = ('maml', 'protonet', 'matchingnet', 'relationnet')
    for method in methods[1:]:
        cfg = dict(PU_CONFIG, method=method, hidden_size=8)
        model = ClassicMetricModel(cfg)
        for shot in (1, 5):
            torch.manual_seed(17)
            support = torch.randn(3 * shot, 1, 64, 64)
            labels = torch.arange(3).repeat_interleave(shot)
            query = torch.randn(6, 1, 64, 64)
            target = torch.arange(3).repeat_interleave(2)
            before = {k: p.detach().clone() for k,p in model.named_parameters()}
            f, scores = model.episode(support, labels, query)
            assert scores.shape == (6, 3) and torch.isfinite(scores).all()
            assert all(torch.equal(before[k], p) for k,p in model.named_parameters())
            loss = model.episode_loss(scores, target)
            model.zero_grad(); loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            if method == 'matchingnet':
                assert torch.allclose(scores.exp().sum(-1), torch.ones(6), atol=1e-5)
            restored = ClassicMetricModel(cfg)
            restored.load_state_dict(model.state_dict())
            model.eval(); restored.eval()
            assert torch.allclose(model.episode(support, labels, query)[1],
                                  restored.episode(support, labels, query)[1])
            model.train()
        print('PASS algorithm/gradients/checkpoint', method)
    # Both real datasets: bounded 1-epoch train -> best -> independent reload -> test.
    for dataset, base, cls in [('PU', PU_CONFIG, trainer.MAML_learner),
                               ('CWRU', CWRU_CONFIG, CWRUMAMLLearner)]:
        for method in methods:
            cfg = resolve_method_config(dict(base, method=method, hidden_size=8,
                n_way=3, k_shot=1, q_query=2, epochs=1, meta_batch_size=1,
                tasks_per_epoch=2, adaptation_steps={1:1}, save_visualizations=False,
                tsne_selected_classes=[], early_stop_on_perfect_validation=False))
            assert cfg['task_weighting_mode'] == 'none' and cfg['backbone'] == 'cnn4'
            net = cls(3, cfg)
            with tempfile.TemporaryDirectory(prefix='classic_smoke_') as tmp:
                path = str(Path(tmp)/method)
                best = net.train(path, shots=1)
                assert Path(best).exists()
                fresh = cls(3, cfg)
                fresh.test(best, shots=1, inner_steps=1, meta_batch_size=2)
            print('PASS real train/save/reload/test', dataset, method)
    for method in methods:
        cfg = resolve_method_config(dict(PU_CONFIG, method=method))
        assert resolve_method_config(cfg) == cfg
    try:
        resolve_method_config(dict(PU_CONFIG, method='typo'))
        raise AssertionError('invalid method accepted')
    except ValueError:
        pass
    print('ALL CLASSIC INTERFACE CHECKS PASSED')

if __name__ == '__main__':
    main()
