# 普通ResNet18配置使用说明（2026-09-10）

当前PU和CWRU配置均已切换为`method='resnet18'`。模型使用单通道STFT、标准[2,2,2,2]残差块，从随机权重开始在配置的全部源域类别上监督训练，不使用外部预训练权重。

```python
'method': 'resnet18',
'run_mode': 'train_test',
'supervised_epochs': 100,
'supervised_batch_size': 64,
'supervised_lr': 0.001,
'supervised_weight_decay': 1e-4,
'supervised_max_batches': 0,
```

在IFMAML-master运行`python maml_train_pic.py`（PU）或`python run_cwru_maml.py`（CWRU）。train/test/train_test仍有效；正式实验关闭QUICK_TEST/--quick，并保持supervised_max_batches=0。`--quick`仅训练1轮、最多2个批次；验证和测试仍覆盖对应划分全量样本。

训练使用Adam与交叉熵，目标验证集直接分类准确率选择best。测试加载best，保留训练好的固定分类头，使用eval()和源域BN运行统计，对目标测试集全部样本直接预测；不划分support/query、不微调、不重建分类头。目标验证标签仅用于选checkpoint，目标测试标签只用于指标计算。源域和目标域必须使用同一组语义类别；标签索引统一由source_classes顺序确定，加载权重时检查类别顺序。

`n_way`、`k_shot`、`q_query`、test_meta_batch_size、ft_*和MAML的内外循环参数不控制普通ResNet18训练/测试。实际输出类别数等于配置的class_groups（PU）或class_names（CWRU）数量。要改变任务类别数，应修改类别配置，而不是只改n_way。

输出仍有训练记录、预测CSV、混淆矩阵和t-SNE（受原save_visualizations开关控制）。测试准确率按完整目标测试样本汇总；训练与验证均为相同全类别分类范围。结果应写为源域监督训练后的全类别跨域直接分类，不应标为5-way 1-shot/5-shot，或与不同候选类别数的元学习结果直接排名。普通ResNet18使用冻结BN，原批量query元学习方法使用的BN协议也应在论文中分别说明。

TL-WDCNN仍沿用原来的预训练＋支持集微调流程。旧resnet18_ft接口保留供复现实验，但当前默认resnet18走独立普通监督路线。两种方法的权重名称隔离，不应复用旧FT权重作为本次从零训练结果。

实现：supervised_training.py，调用共享ResNet18骨干；两数据集统一入口分派到独立监督训练/测试。检查脚本：test_supervised_resnet18.py，覆盖真实PU/CWRU小规模训练、保存、回读、完整测试、无episodic调用、改变shot结果一致及BN/参数不变。功能检查不作为正式论文准确率。
