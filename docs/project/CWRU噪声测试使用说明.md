# CWRU噪声测试接口

仅修改CWRU入口和配置；PU训练/测试不受影响。噪声是原始振动信号AWGN，按去均值信号功率定义SNR，不是在PNG像素上加噪。

## 运行

在IFMAML-master、ifmaml2环境下，确保method、源目标、way/shot、预处理参数与待测权重一致：

```powershell
# 仅测试：不会重新训练，默认加载配置model_name对应_best
python run_cwru_maml.py --noise-test

# 指定已有模型路径，传不带_best的训练保存前缀
python run_cwru_maml.py --noise-test --model_path ./model_save/你的模型名

# 小规模接口检查：全部噪声等级保留，每级仅一个噪声种子、4个episode
python run_cwru_maml.py --noise-test --quick --model_path ./model_save/你的模型名
```

也可以只改config_cwru.py然后运行原入口：

```python
'run_mode': 'test',
'noise_test': True,
'noise_snr_db': [10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10],
'noise_include_clean': True,
'noise_seeds': [101, 202, 303],
'noise_episode_seed': 2026,
'noise_test_episodes': 1000,
```

noise_test默认False，避免影响原测试流程。`--noise-test`强制测试，除非显式同时提供`--train`，否则不按run_mode=train_test启动训练。配置noise_test=True与run_mode=train_test组合则正常训练后扫描噪声。正式测试不使用QUICK_TEST/--quick。

## 协议与范围

- 0、+0、-0相同，只计算一次。0dB表示添加噪声功率等于去均值信号功率，clean是不额外加噪，两者不同。
- 目标test池support和query都加噪；训练与验证数据不变。各SNR复用同一份权重，每个等级/噪声种子开始前恢复权重和BN状态。
- 重放原预处理随机数与窗口起点，对200个目标原始窗口逐张核对clean STFT与现有PNG；不修改MAT/PNG。窗口、源目标工况、类别、样本数与原配置保持一致。
- 同一噪声种子、类别和窗口使用同一标准高斯噪声向量；各SNR只缩放幅度。不同模型的同一窗口使用相同噪声。
- 用独立noise_episode_seed预先生成固定无放回support/query索引；所有模型/SNR/噪声种子复用同一清单。clean重新评估该清单，不能直接抄旧测试数值。
- 保持原MAML/度量模型的批量query BN和适配机制；FT方法独立support微调；普通ResNet18保持eval直接分类，不建立support/query任务。
- 支持configured_maml、maml、protonet、matchingnet、relationnet、resnet18_ft、tl_wdcnn、resnet18。TL-WDCNN使用同一加噪原始窗口并标准化，不经STFT。
- 有config元信息时检查method、way/shot、类别、域和预处理字段一致性。不匹配需选择对应配置/权重，不能以5-way模型权重宣称10-way实验。

## 输出

每次运行独立保存至results/cwru/noise/<权重名>/<时间戳>/：

- summary.csv：每个SNR、噪声种子的准确率、损失、预测次数、episode数。
- summary_mean_std.csv：各SNR平均准确率、噪声seed间样本标准差、相对clean下降百分点。
- episodes.json：跨SNR/seed/模型可复用的任务索引；config.json内记录清单SHA256。
- windows.csv：每个测试窗口的MAT路径、变量、起点与原PNG对应。
- 各SNR_seed_predictions.csv：逐条预测（类别索引对应config.json中的class_names）。

默认11个噪声等级×3噪声实现，加一次clean，共34次评估，每次1000episode；普通ResNet18每次为目标测试集全量直接推断。标准差衡量固定模型下噪声随机性，不代表独立训练seed的不确定性。只有一次重复时不填写SD，clean只计算一次。

此接口不自动运行多个模型或训练种子，不自动调噪声测试集上的参数。新噪声数据在内存生成，不需要重新执行原预处理，也不覆盖旧结果。

修改前备份：code_backups/20260911_145445_cwru_noise。功能检查：python test_cwru_noise.py；检查权重为临时随机模型，检查准确率不作为论文结果。

验证：11个SNR标定、200个目标窗口clean一致性、七种对比模型与LSK-lite+GCNet MAML测试链路通过；命令行与配置test模式均确认不触发训练。
