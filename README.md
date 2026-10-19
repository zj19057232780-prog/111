
# 少样本轴承故障诊断实验代码

当前代码快照：2026-10-19。包含 CWRU、PU、HUST 单库实验，以及 CWRU ↔ PU 多源工况到单目标工况的跨数据集实验。

## 目录

```text
IFMAML-master/       主程序、模型、数据接口、配置及测试
comparison_models/  AMAML、TUBR-ML、MFEML 独立实现与方法配置
tools/              依据保存的源码快照补测的工具
docs/运行说明/      AutoDL 命令与部署说明
docs/历史说明/      原 main 分支历史说明，仅供追溯
```

请保留 `IFMAML-master/` 与 `comparison_models/` 平级结构。数据集、预处理缓存、训练权重、实验结果及私人研究笔记不随代码发布；运行前需自行准备相应数据或缓存。

## 环境与启动

本地核验环境为 Python 3.8、PyTorch 2.4.1 + CUDA 12.1。先安装适合服务器的 PyTorch，再安装其余依赖：

```bash
python -m pip install numpy scipy opencv-python-headless scikit-learn matplotlib
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
cd IFMAML-master
python run_cross_dataset.py --help
```

ONNX / TensorFlow 导出脚本的可选依赖需要单独安装。部分单库默认配置保留本地 Windows 数据路径，迁移时请按运行说明覆盖路径。

## 跨数据集实验

当前协议为 `Cross-AB-MS-5way-targetval-v2`：T1–T4 为 CWRU 全部 C0–C3 工况分别迁移至 PU P1–P4；T5–T8 为 PU 全部 P1–P4 工况分别迁移至 CWRU C0–C3。采用 5-way、1/3/5-shot、每类 10 query、seed 3/24/38；目标验证集选取最佳 checkpoint，正式测试为 1000 episodes。

支持 ProtoNet、MatchingNet、RelationNet、MAML、TL-WDCNN、AMAML、TUBR-ML、MFEML 和 SCG-MAML。提供 STFT 与原生输入接口；SCG-MAML 仅支持 STFT。八个对比模型默认使用 100 轮耐心早停，严格验证准确率改善才重置计数；SCG-MAML 不启用早停。

上传已有 `cross_data_processed_ab5_ms_v2` 缓存后，可在 `IFMAML-master/` 下执行一个模型、一个 seed 的八任务正式训练与测试：

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
python -u run_cross_dataset.py --task all --method maml --input stft \
  --shots 1 --seed 3 --epochs 500 --test-episodes 1000 \
  --processed-root ./cross_data_processed_ab5_ms_v2 \
  --checkpoint-root ./model_save/github_cross_stft \
  --result-root ./results/github_cross_stft --train --test
```

已有缓存时不加 `--preprocess`。同一运行路径已存在时程序会拒绝覆盖；请按运行说明处理已有结果或改用新的输出根目录。训练中的源码不应被覆盖；若已训练完成但源码哈希变化导致测试受阻，可参照 `tools/retest_cross_from_snapshot.py --help` 使用已保存的训练源码快照补测。

完整逐组命令见 [AutoDL 跨数据集正式实验逐组命令](docs/运行说明/AutoDL跨数据集_正式实验逐组命令.md)，部署细节见 [服务器部署清单](docs/运行说明/AutoDL服务器部署清单.md)，协议细节见 [跨数据集 AB5 运行说明](IFMAML-master/docs/跨数据集AB5运行说明.md)。历史说明中的旧协议不代表当前配置。

## 验证

不依赖原始数据的早停与最佳权重选择测试：

```bash
cd IFMAML-master
python -B -m unittest -v test_cross_early_stop
```

其他数据相关测试需要对应的数据文件。接口测试通过不代表已完成正式实验；本仓库不提供未核验的实验结果。
