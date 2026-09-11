> **已撤回（2026-09-11）**：用户要求全部恢复Gao调整前设置。本文仅作历史记录，当前代码不再使用gao_task或Gao预设；请使用config_cwru.py中的原三源48kHz配置。

# CWRU按Gao任务调整（2026-09-11）

用户确认保留STFT，对齐单源迁移、12kHz、2048点窗口、80点步长；样本预算随后按用户要求恢复为源40/验证20/测试20。模型结构与训练优化器不随论文替换，当前method仍为用户选择的protonet。

## 直接运行

四个任务的数据已生成到各自独立目录，默认T0_T3。在IFMAML-master、ifmaml2环境执行：

```powershell
python run_cwru_maml.py
```

修改config_cwru.py中的选择项：

```python
'cwru_protocol': 'gao2025_stft',
'gao_task': 'T0_T3',  # T0_T1 / T0_T2 / T0_T3 / T2_T3
'k_shot': 1,         # 可选1、5、10
'seed': 24,
'run_mode': 'train_test',
```

| gao_task | 源域 | 目标域 |
|---|---|---|
| T0_T1 | C0，1797 rpm，0 hp | C1，1772 rpm，1 hp |
| T0_T2 | C0，1797 rpm，0 hp | C2，1750 rpm，2 hp |
| T0_T3 | C0，1797 rpm，0 hp | C3，1730 rpm，3 hp |
| T2_T3 | C2，1750 rpm，2 hp | C3，1730 rpm，3 hp |

底部cwru_protocol.resolve_cwru_protocol统一展开任务。该预设会覆盖source_condition/target_condition、数据路径、采样率、窗口、n_way与test_meta_batch_size；切换方向修改gao_task，不要只改source_condition。模型method仍可独立修改。普通ResNet18仍按10类直接分类，不能称作5-way元学习。

## 数据与任务

- 故障类从12DriveEndFault读取DE_time；正常NormalBaseline显式按48kHz读取，scipy.signal.resample_poly抗混叠降采样到12kHz。原MAT不修改，采样率记录于manifest。
- 十类保持Healthy及IR/Ball/OR各三种尺寸，OR使用6点钟方向。
- 单源每类40窗口；每窗口2048点，相邻起点相隔80点，从对应信号段起点连续取样。
- 目标每类共40窗口：原信号前半段取20验证窗口、后半段取20测试窗口；两段之间没有原始采样点交叠。目标验证标签选best，测试标签仅评估。
- 每任务合计800张图：400源训练、200验证、200测试。相邻窗口自身有96.09%重叠；同一目标test池的support/query可能来自高度重叠的窗口，不应把它们解释为独立记录。
- 保留64×64单通道STFT；在12kHz下使用48点窗/36点重叠，对应4ms窗和1ms hop。这是本项目处理方式，Gao采用CWT。
- 元学习5-way，1/5/10-shot；每类15query是项目保留值，论文未明确M。每次训练后测试20个episode。
- Gao报告十次独立试验。gao_trial_seeds列出十个建议种子，单次入口只跑当前seed；该列表不会自动执行十次训练，也不能以20个episode的标准差代替十次独立试验的标准差。正式统计需逐seed从头训练并汇总。

**实现边界：** Gao没有明确独立目标验证集划分，本项目的20验证+20测试是保留验证选权重流程的明确改编；不能称为论文逐项精确复现。此次保留STFT、q_query、原优化器/轮数/学习率与BN协议，不复现其CWT和全部训练细节。原三源48kHz结果不与本协议混排。

## 文件与检查

新数据：IFMAML-master/cwru_data_gao2025_stft/s40_v20_t20/{task}/，每个任务包括manifest.csv与preprocessing_config.json。训练前核对处理签名；修改STFT参数等导致不一致时会报错，可执行：

```powershell
python run_cwru_maml.py --preprocess
```

该命令只重建当前新任务数据。旧cwru_data_processed、model_save与results历史结果未覆盖。新权重使用GaoSTFT前缀与任务/配置签名隔离。TL-WDCNN与STFT共用重采样和窗口起点，并逐图校验raw/PNG一致性。

备份：code_backups/20260911_142455_gao_cwru。检查脚本：python test_cwru_gao_protocol.py。未启动正式长训练，功能测试权重在临时目录清理。

依据：Gao等2025用户提供PDF第7—10页、Tables I—III。官方数据源区分12k/48k驱动端数据：https://engineering.case.edu/bearingdatacenter/download-data-file 。正常数据48kHz分类的交叉参考：https://www.nature.com/articles/s41598-025-19258-2 。

验证完成：四任务共3200个raw/PNG逐张一致，目标验证/测试原始范围不交叠，各任务1/5/10-shot采样通过；真实ProtoNet小规模训练、保存、回读、测试通过。

## 样本预算回调（2026-09-11）

config_cwru.py的source_samples_per_class=40、target_val_samples_per_class=20、target_test_samples_per_class=20直接控制样本量，预设不再强制覆盖200。新目录按样本预算隔离，之前200样本版数据保留但当前不使用。该规模是每个工况每类的原预算；当前单源共400训练样本，并非之前三源共1200训练样本。1/5-shot可保持15query；10-shot时每类只有20个目标样本，需显式将q_query改为10或更小，否则报错，程序不通过重复采样凑数。
