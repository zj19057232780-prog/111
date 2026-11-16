> 2026-10-19 更新：华中HUST已通过独立入口 run_hust_huazhong.py 接入，采用目标验证，见[HUST说明](../docs/运行说明/HUST华中科技大学_运行说明.md)。下文“第三库尚未接入”为历史记录，已被本更新取代；run_experiment.py仍用于CWRU/PU。

# CWRU / PU 独立跨工况实验

当前统一入口为 `run_experiment.py`，复用已有模型、数据加载器和训练器。

```bash
python run_experiment.py --dataset cwru --tasks all --method scg_maml --input stft --shots 1 --seed 3 --action train-test --device cuda
python run_experiment.py --dataset pu --tasks P1,P3 --method maml --input nostft --shots 5 --seed 24 --action train-test --device cuda
```

方法：`protonet matchingnet relationnet maml amaml tubr_ml mfeml scg_maml`；`tl_wdcnn` 为补充方法。`a0 a1 a2 a3` 为 STFT 1-shot 消融。
SCG-MAML 和消融仅支持 STFT；其他模型的 nostft 保持原生输入。支持 1/3/5-shot，默认 5-way、15 query、1000 测试 episode。

默认训练500轮，对比方法 patience=100；SCG-MAML/A0–A3 不启用早停。TL-WDCNN 监督预训练上限100轮。只有 `--action train-test` 执行完整训练和测试；默认 action=check，smoke 仅作功能验证。

新结果在 `experiment_results/`，按数据集/实验类型/输入/模型/shot/批次/seed/任务分别保存。正式统计只汇总完整的 seed 3/24/38，缺失组标为 incomplete。

完整命令见 [独立数据集正式实验命令](../docs/运行说明/独立数据集_正式实验命令.md)，服务器上传见 [部署清单](../docs/运行说明/AutoDL服务器部署清单.md)。

跨数据集与河内理工大学 HUST 已退出当前入口。华中科技大学 HUST 使用独立的 `run_hust_huazhong.py` 入口及 `hust_huazhong` 数据配置，支持 H1–H6 单源到单目标、目标域验证实验。

兼容入口 `maml_train_pic.py`、`run_cwru_maml.py`、`run_table_row.py` 和三个独立复现入口保留；其历史默认配置不自动变成统一入口的设置。噪声测试、监督/迁移基线及旧模型组件保留兼容。上传代码时必须包含 `optional_models/`。

原始 CWRU/PU 数据及历史正式结果保留。旧权重应使用对应源码快照补测，不能忽略源码或数据哈希不一致。
