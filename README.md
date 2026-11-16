# 少样本轴承故障诊断实验代码

当前代码快照：2026-11-16。包含PU、CWRU与华中科技大学HUST跨工况实验。旧跨数据集入口已从当前本地代码移出，可通过Git历史查阅。

保留 IFMAML-master/ 与 comparison_models/ 平级结构。IFMAML-master/optional_models/ 是模型导入必需目录，需一起上传。原始数据、缓存、权重和实验结果不包含在仓库中。

## 运行

安装适合机器的CUDA版PyTorch及numpy、scipy、opencv-python-headless、scikit-learn、matplotlib。数据路径与完整命令见[AutoDL运行清单](docs/运行说明/三数据集_单模型实验命令清单_AutoDL.md)。

```bash
cd IFMAML-master
python run_experiment.py --help
python run_hust_huazhong.py --help
```

PU/CWRU使用run_experiment.py，最终模型为 --method scg_maml；华中HUST使用run_hust_huazhong.py，最终模型为 --model full。

三数据集少样本模型默认连续100轮目标验证准确率无提升早停，--patience 0关闭；--action train-test在训练结束后测试最佳权重。支持的模型与输入组合可使用 --shots 3，包含最终模型及消融；HUST的TUBR-ML不支持3-shot，--models all --shots 3自动排除它。完整模型及结构消融使用STFT。HUST TUBR-ML原生输入不支持5-shot加15-query。

目标验证标签参与选模，测试集独立计分。重复命令创建新批次，不自动续训。功能验证不代表完成正式实验。

## 轻量回归测试

```bash
python -B -m unittest test_experiment_entry.EntryTests.test_presets_and_inputs test_experiment_entry.EntryTests.test_three_shot_all_methods_except_tubr test_hust_baselines.HUSTBaselineTests.test_three_shot test_hust_baselines.HUSTBaselineTests.test_supported_inputs_and_patience -v
```

更多说明见[三数据集命令清单](docs/运行说明/三数据集_单模型实验命令清单.md)。
