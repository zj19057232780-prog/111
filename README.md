# PU / CWRU few-shot bearing fault diagnosis

Current research project snapshot, including configurable MAML and metric baselines, supervised ResNet18, transfer baselines, and CWRU AWGN evaluation.

## Entry points

- PU: `python maml_train_pic.py`, configured in `config_pu.py`.
- CWRU: `python run_cwru_maml.py`, configured in `config_cwru.py`.
- CWRU noise-only evaluation: `python run_cwru_maml.py --noise-test` (requires matching trained weights).

`method` selects configured_maml, maml, protonet, matchingnet, relationnet, resnet18, resnet18_ft, or tl_wdcnn. `run_mode` selects train, test, or train_test. Check current configuration before running; this snapshot preserves local experiment settings.

## Environment and data

Validated locally with Python/PyTorch in the ifmaml2 environment (PyTorch 2.4.1). Runtime imports include NumPy, SciPy, OpenCV, scikit-learn and matplotlib. The repository includes a local MAML shim; the model implementation does not require downloading pretrained ImageNet weights.

Supply your own local PU/CWRU MAT data and adjust paths in configuration. Original datasets, generated STFT images, model weights and experiment outputs are not included. CWRU source lookup metadata is included at `CWRU/__file__/metadata.txt`.

CWRU preprocessing: `python run_cwru_maml.py --preprocess`. Follow the data paths and preparation details in the project documentation. Do not use an old checkpoint with a different task or preprocessing protocol.

## Documentation

- [项目文档](docs/project/项目文档.md)
- [经典模型配置](docs/project/经典模型配置使用说明.md)
- [普通ResNet18](docs/project/普通ResNet18配置使用说明.md)
- [迁移学习模型](docs/project/迁移学习模型配置使用说明.md)
- [CWRU噪声测试](docs/project/CWRU噪声测试使用说明.md)

`docs/project` is a snapshot of the parent workspace's research notes. Historical proposals and reverted settings are retained for traceability; current configuration and the latest project change log take precedence.

## Functional checks

Run `test_classic_interfaces.py`, `test_transfer_interfaces.py`, `test_supervised_resnet18.py`, or `test_cwru_noise.py` in the configured environment with local data available. These use temporary smoke-test weights, not formal experiment results.
