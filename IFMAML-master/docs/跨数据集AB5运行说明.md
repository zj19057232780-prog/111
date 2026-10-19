# 跨数据集 AB5：四源工况到单目标工况、九模型与双输入

2026-10-16。入口为`run_cross_dataset.py`，协议`Cross-AB-MS-5way-targetval-v2`。只包含CWRU↔PU，5-way 1/3/5-shot，10 query/类。默认不自动展开全部实验矩阵。

AutoDL/Linux运行请见[跨数据集实验命令清单](../../docs/运行说明/AutoDL跨数据集实验命令清单.md)，包含缓存上传、自检、单任务/批量命令及结果汇总。

## 全部八任务

| 任务 | 源域（全部四工况） | 单目标工况 |
|---|---|---|
| T1 | CWRU C0+C1+C2+C3 | PU P1：N09_M07_F10 |
| T2 | CWRU C0+C1+C2+C3 | PU P2：N15_M01_F10 |
| T3 | CWRU C0+C1+C2+C3 | PU P3：N15_M07_F04 |
| T4 | CWRU C0+C1+C2+C3 | PU P4：N15_M07_F10 |
| T5 | PU P1+P2+P3+P4 | CWRU C0：0 hp，1797 rpm目录 |
| T6 | PU P1+P2+P3+P4 | CWRU C1：1 hp，1772 rpm目录 |
| T7 | PU P1+P2+P3+P4 | CWRU C2：2 hp，1750 rpm目录 |
| T8 | PU P1+P2+P3+P4 | CWRU C3：3 hp，1730 rpm目录 |

所有源工况共用五个健康类别，并不变成20-way。元训练每episode从四源工况等概率选一个，再在该工况内抽support/query；外循环跨episode聚合四域。TL-WDCNN将四源工况均衡样本合并监督预训练。每个目标工况使用自己的验证池独立选模和测试，不合并其他目标工况。

## 支持范围

| CLI模型名 | stft | nostft原始接口 | 复用实现 |
|---|---|---|---|
| protonet | 单通道64×64 | 原生1D，CWRU3072/PU4096，逐窗z-score | classic_models.py / raw_fewshot.py |
| matchingnet | 同上 | 同上，保留FCE | 同上 |
| relationnet | 同上 | 同上，保留关系网络和MSE目标 | 同上 |
| maml | 同上 | 同上，原生RawMAMLModel | maml_model.py / raw_fewshot.py |
| tl_wdcnn | STFT按行展开为4096点，交给现有1D WDCNN | CWRU3072/PU4096，逐窗z-score | transfer_models.py的TransferModel/TransferEpisode |
| amaml | 单通道64×64 | 1444点顺序reshape为1×38×38，无归一化 | comparison_models/mu2025_amaml/model.py、raw_reshape.py |
| tubr_ml | 单通道64×64 | 7056点逐窗minmax，reshape84×84并重复为3通道 | comparison_models/wang2025_tubr_ml/tubr_model.py、raw_reshape.py |
| mfeml | 单通道64×64 | 2000原始点，无归一化，模型原生AdaptiveSquare补零成45×45 | comparison_models/li2025_mfeml/mfeml_model.py、raw_input.py |
| configured_maml / SCG-MAML | 单通道64×64 | **明确不支持，按用户决定报错** | 现有Net4LSK、GCNet、AGR |

`nostft`保持原始采样率，**不做重采样、STFT或PNG读取**；模型原有归一化/reshape仍保留。跨库时原始序列的物理持续时间可能不同；这是保留原接口的结果。STFT模式统一48 kHz、3072点、64×64，PU先64→48 kHz抗混叠重采样。TL-WDCNN的STFT模式是明确的展开图像序列对照，不是原论文的一维时域输入；主张原始TL-WDCNN时应使用nostft。

## 数据划分与配额

类别N/I-L1/I-L2/O-L1/O-L2沿用：CWRU正常/IR007/IR021/OR007/OR021（外圈6点钟），PU K001/KI03/KI07/KA05/KA03。全部四工况均采用同一类别映射。

CWRU C0原始文件较短（约24.4万点），按7056点长窗和前后时间隔离，两个池最低容量17/16。用户确认所有任务改为每类10 query；每工况每类每池固定**16个**不重叠锚窗口。源域每工况合并两个池32/类，四工况合并**128/类（共640）**；目标单工况验证**16/类（共80）**、测试**16/类（共80）**。

公共原始锚7056点。CWRU每原始文件前后两段隔离，间隔7056点；PU每工况记录1/2隔离。共8工况×5类×2池×16=1280锚。先划分原始段，再以7056步长生成候选窗，均匀选择16个；不足直接报错。不同原生模型仍各取1444/2000/3072/4096/7056点，不改已有输入设置。共用锚ID只保证同一原始起点，不保证模型观察时长相等。

原始缓存和STFT索引分开，nostft预处理不生成图像、不重采样。两个表示及各模型共享窗口ID和任务；1/3/5-shot query完全相同，support为嵌套子集。验证/测试只加载当前单目标工况；其他目标工况虽在公共缓存中，不进入该任务模型的训练或选模。

## PowerShell / PyCharm Terminal

以下在项目根目录执行，使用已验证可用 CUDA 的 `C:\ANACONDA\envs\ifmaml2\python.exe`（PyTorch 2.4.1+cu121，RTX 4060 Laptop GPU）。激活 ifmaml2 后也可直接使用 `python`。基础解释器 `C:\ANACONDA\python.exe` 安装的是 2.10.0+cpu，仅适合此前的 CPU 验证；显式指定该路径会绕过已激活的环境。设备根据 `torch.cuda.is_available()` 自动选择。

```powershell
Set-Location 'C:\Users\JieZhang\Desktop\new-ifmaml\IFMAML-master'
$env:PYTHONIOENCODING = 'utf-8'
```

只建立原始缓存（不生成STFT）：

```powershell
& 'C:\ANACONDA\envs\ifmaml2\python.exe' run_cross_dataset.py --preprocess --task all --method MAML --input nostft
```

建立/补充STFT缓存：

```powershell
& 'C:\ANACONDA\envs\ifmaml2\python.exe' run_cross_dataset.py --preprocess --task all --method MAML --input stft
```

单模型正式训练并测试：

```powershell
& 'C:\ANACONDA\envs\ifmaml2\python.exe' run_cross_dataset.py --task T1 --method AMAML --input nostft --shots 3 --seed 24 --train --test
& 'C:\ANACONDA\envs\ifmaml2\python.exe' run_cross_dataset.py --task T2 --method SCG-MAML --input stft --shots 5 --seed 38 --train --test
```

全部STFT模型、八任务、一个shot/seed组合：

```powershell
& 'C:\ANACONDA\envs\ifmaml2\python.exe' run_cross_dataset.py --task all --method all --input stft --shots 1 --seed 3 --train --test
```

全部原生nostft模型（明确排除SCG-MAML）：

```powershell
& 'C:\ANACONDA\envs\ifmaml2\python.exe' run_cross_dataset.py --task all --method all --input nostft --shots 1 --seed 3 --train --test
```

完整两接口矩阵共 `(9+8)×8任务×3shot×3seed=1224` 个实验单元；按输入分别出表。执行完整矩阵：

```powershell
foreach ($mode in @('stft','nostft')) {
    foreach ($shot in @(1,3,5)) {
        foreach ($seed in @(3,24,38)) {
            & 'C:\ANACONDA\envs\ifmaml2\python.exe' run_cross_dataset.py --task all --method all --input $mode --shots $shot --seed $seed --train --test
            if ($LASTEXITCODE -ne 0) { throw "Cross AB failed: $mode / $shot-shot / seed $seed" }
        }
    }
}
```

重复运行会拒绝覆盖已有checkpoint。已完成任务单独用`--test`重测，不执行全矩阵训练循环；缺失任务单独指定模型/方向/shot/seed运行。`--test`不隐式训练。默认模型路径由完整训练配置签名决定；同一配置直接将命令的`--train --test`改成`--test`即可。若训练修改了配置参数，测试也必须使用相同设置；或用`--model-path`指定相应基础路径，仍检查配置/数据/代码哈希。

冒烟（结果带quick标记，汇总器排除）：

```powershell
& 'C:\ANACONDA\envs\ifmaml2\python.exe' run_cross_dataset.py --task T1 --method TUBR-ML --input nostft --shots 3 --quick --train --test
```

## 选模、统计与目录

2026-10-17：八个对比模型新增默认`--patience 100`，SCG-MAML（configured_maml）始终关闭早停，仍跑满配置上限。目标验证Accuracy严格改善才重置耐心计数，准确率持平或仅loss降低不重置；原有Accuracy/loss选最佳checkpoint规则保留。耐心按训练epoch/外更新计数，在下一次验证点判断（默认元学习每10更新、TL每2 epochs），不是100次验证；可用`--patience 0`关闭对比模型早停。TL-WDCNN保留100轮上限，默认patience=100通常先达到上限。达到100%验证准确率本身不立即停止。

新增`_training_summary.json`，history与best配置记录实际轮数、严格Accuracy改善轮数、最佳checkpoint轮数及停止原因；测试summary也带`best_epoch`、`epochs_run`、`stop_reason`和`early_stop_patience`。早停设置进入训练配置签名，原缓存继续使用；旧固定预算结果及checkpoint不覆盖，旧权重重测仍需旧源码快照。AutoDL新批命令使用`*_es100`输出根（SCG-MAML同批记录patience=0），在当前任务完成后更新源码，再启动新任务。

元学习模型默认500次外更新、每次16任务；TL-WDCNN源域预训练100 epochs。验证固定100任务，元学习每10更新/TL每2 epochs检查；最后一次也验证。依据验证Accuracy、query交叉熵、较早更新顺序选checkpoint。验证和测试MAML类均10步，ProtoNet/MatchingNet/RelationNet按各自度量机制执行。训练保留AMAML的AMOS/ALC、TUBR的梯度衰减及TURL、MFEML的MFE和原生MAML适应函数，不统一替换成普通MAML。

每个任务的目标验证池另有80个带标签样本，不属于单episode 1/3/5-shot的support预算；测试support只更新临时模型。BN为episode隔离的批量query统计口径；TUBR按现有实现测试时关闭结构化丢弃。测试query标签仅评分。固定1000测试任务，汇总三训练seed均值±样本SD；Macro-F1按规范五类汇总query预测计算，不能把1000任务当1000独立采集批次。

新目录：`cross_data_processed_ab5_ms_v2/`、`model_save/cross_dataset_ab5_ms_v2/`、`results/cross_dataset_ab5_ms_v2/`。旧3-way及双任务v1缓存、权重、结果留原地；新入口拒绝旧签名。

每个checkpoint保存代码快照、解析配置、数据签名及哈希、验证历史和最佳更新；每次测试保存episode manifest、逐query预测、5×5混淆矩阵、Accuracy、Macro-F1及来源。源码发生变化时拒绝直接用新代码评测旧checkpoint，需使用保存的源快照。

汇总：

```powershell
& 'C:\ANACONDA\envs\ifmaml2\python.exe' summarize_cross_dataset.py
```

`comparison.csv`按Input/模型/shot分行，T1–T8分别列Accuracy和Macro-F1，并列出CWRU→PU、PU→CWRU两个方向的四目标工况平均值（先按seed平均四工况，再跨seed求均值及样本SD）；缺seed写`incomplete`，不补造均值，不把STFT/nostft或旧协议混在同一单元。

## 验证入口

```powershell
& 'C:\ANACONDA\envs\ifmaml2\python.exe' -m unittest -v test_cross_dataset
```

生命周期测试覆盖9×3个STFT和8×3个nostft组合（51个），交错检查两个方向；仅一更新的小规模功能验证，不作为论文结果。每次测试产物在独立的`tmp/cross_ab5_ms_v2_verification_<timestamp>/`，不进入正式结果目录。当前版本测试来源索引与验证范围见项目输出目录的`multisource_verification.json`及《多源跨库八任务实施记录.md》。
