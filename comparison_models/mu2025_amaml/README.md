# AMAML（Mu 等，2025）在当前项目协议下的复现

独立目录：`C:/Users/JieZhang/Desktop/new-ifmaml/comparison_models/mu2025_amaml`。

现在支持两个输入接口，默认 `--input no_stft`，两种输入默认都训练最多100 epoch。

| 参数 | 模型输入 | 读取数据 | 输出批次标记 |
|---|---|---|---|
| `--input no_stft` | 原始1444点逐行重排38×38 | PU整理后MAT / CWRU原始MAT | raw38 |
| `--input stft` | 单通道64×64 STFT | 项目已有PNG图片 | stft64 |

两者共用CNN＋SBCA、ALC、AMOS和项目任务预算，但分类头按输入大小分别为256/1024维；权重不能交叉使用。训练不会自动生成STFT图片。原文的输入形式对应no_stft；STFT为项目输入适配。ALC原文歧义与工作参数见 [原文与实现映射](notes.md)。

## 保留的项目协议

| 项目 | 本实现 |
|---|---|
| PU | artificial_9state，9 类池中抽 5 类；P1–P4 多源留一目标 |
| CWRU | 10 类池中抽 5 类；C0–C3 多源留一目标 |
| 每个 episode | 5-way，1/3/5-shot，15-query，无放回采样 |
| 三个 seed | 3、24、38 |
| 更新预算 | 每轮 16 个训练任务、16 个验证任务；最多 100 轮 |
| 适应步数 | 训练/验证 3 步；测试 10 步 |
| 优化器 | 项目 Adam、inner_lr=0.05、outer_lr=0.005、余弦调度和梯度裁剪 |
| 选模 | 目标 validation；最佳 checkpoint 最终测试 1000 episodes |
| 早停 | 验证连续 100 轮无严格提升；保留项目高准确率确认早停 |
| 统计 | 三 seed 均值±样本标准差；全工况 SD 按各 seed 的四任务均值计算 |

PU 使用原加载器的 `seed + class_seed` 划分，no_stft的CWRU按目标信号前半验证/后半测试取窗，STFT则按已有source/val/test图片角色读取。目标验证标签参与选模，不能称为目标域标签完全不可用的协议。BN 使用 support/query 各自的批统计，保留项目的批 query 评测口径。

独立训练器在 outer update **之后**验证并保存相同参数，避免旧训练器“更新前验证、更新后保存”的一轮错位；旧代码与旧结果未改动。严格横向比较时应披露该差异，必要时统一各基线选模时序。未加载其他模型的权重或复用其结果。

## 直接运行（PowerShell）

完整PyCharm终端命令、逐任务清单和绿色运行按钮配置见 [PyCharm本地运行命令清单](PyCharm本地运行命令清单.md)。

```powershell
Set-Location 'C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\mu2025_amaml'
$py = 'C:\ANACONDA\envs\ifmaml2\python.exe'
$env:PYTHONIOENCODING = 'utf-8'

# 只检查原始数据与样本预算，不训练、不生成图片
& $py run.py --input no_stft --dataset pu --tasks all --seeds 24 --shot 5 --action check
& $py run.py --input no_stft --dataset cwru --tasks all --seeds 24 --shot 5 --action check

# 小规模真实数据运行验证：1轮、1个训练任务、1个验证任务、2个测试任务
& $py run.py --input no_stft --dataset pu --tasks P1 --seeds 24 --shot 1 --action smoke
& $py run.py --input no_stft --dataset cwru --tasks all --seeds 24 --shot 5 --action smoke

# 正式训练：PU四任务×三个seed
& $py run.py --input no_stft --dataset pu --tasks all --shot 1 --epochs 100 --action train
& $py run.py --input no_stft --dataset pu --tasks all --shot 5 --epochs 100 --action train

# 正式训练：CWRU四任务×三个seed
& $py run.py --input no_stft --dataset cwru --tasks all --shot 1 --epochs 100 --action train
& $py run.py --input no_stft --dataset cwru --tasks all --shot 5 --epochs 100 --action train
```

省略 `--tasks` 时读取当前项目配置中的目标工况。默认 `--action check`，不会意外开始长训练。正式训练支持 `--device cpu/cuda/auto`、`--epochs` 和 `--patience`；更改训练预算后的结果须单独标注。

### STFT入口示例

```powershell
& $py -u run.py --input stft --dataset pu --tasks all --shot 1 --epochs 100 --action train --device cuda
& $py -u run.py --input stft --dataset pu --tasks all --shot 5 --epochs 100 --action train --device cuda
& $py -u run.py --input stft --dataset cwru --tasks C3 --shot 1 --epochs 100 --action train --device cuda
& $py -u run.py --input stft --dataset cwru --tasks C3 --shot 5 --epochs 100 --action train --device cuda
```

### 原始数据路径与窗口

PU默认读取 `IFMAML-master/pu_data_mat`，CWRU读取 `IFMAML-master/CWRU`。`--data-root`现在指定原始MAT根目录：PU为整理后的 `condition/bearing.mat` 目录，CWRU为包含 `NormalBaseline` / `48DriveEndFault` 的目录。以上为no_stft模式。stft模式的 `--data-root` 应指定已有图片根目录；也可用 `--data-map` 提供任务到图片根目录的JSON映射（与data-root互斥，只适用于stft）。

保持项目窗口身份：起点仍由PU4096点、CWRU3072点的原切窗规则生成，每个窗口取前1444点作为实际输入。此截取约定是本项目接入选择，原文未说明具体起点。目标划分与类别、样本预算保持，原文件不变，不生成离线图片。no_stft的CWRU C0–C3均不需要STFT缓存；stft默认缓存只匹配C3，其他目标须提供已有匹配缓存，缺失时停止。

100轮是默认上限，高准确率确认早停仍可提前结束。patience保持100，在100轮预算内通常不会先于上限触发。smoke仍特意缩减为1轮。

## 结果与回读

- `runs/{dataset}_{raw38或stft64}_{shot}s_时间戳/`：正式训练；`smoke_runs/`：功能验证，状态始终为 `SMOKE_ONLY`，不填论文表。
- 每次运行新建目录，不覆盖旧结果。每个任务保存 `best.pt`、`history.csv`、`resolved_config.json`、`method_config.json`、`result.json`、`test_predictions.csv`。
- no_stft额外保存 `raw_window_manifest.json`，记录原窗口起点、1444点实际长度、原采样窗口长度、角色和原始值哈希。
- 保存 `data_signature.json`（各划分内容哈希与类计数）、`code_sha256.json`、`source_snapshot/`、`episodes.jsonl`。后者记录 split 内数组索引和全局类别，每类先 support 索引后 query 索引；按所存 config、代码与相同数据可定位重放。
- 测试用 `100000 + seed` 固定任务随机数，与原逐行入口一致。Macro-F1 先累积全局语义类别混淆矩阵，再按整个类别池平均。
- 三 seed 完整时才填 `summary.json`；不完整或 smoke 均标注 `incomplete_or_smoke`。
- 独立回读：`& $py run.py --action test --checkpoint-dir '某次运行/P1_s24'`，重新核对代码、数据与权重哈希，输出新 `eval_.../`，保留原测试文件。哈希改变时先恢复匹配版本，不能混用新配置测试旧权重。
- **固定运行设备。** 旧STFT版本CWRU小规模检查中，CUDA训练权重在CPU上进行10步适应的预测与CUDA明显不同；同一CUDA设备回读则完全一致。新版同样不宣称CPU/GPU结果等价，不能混入同一精确复现实验。回读记录实际设备、训练设备和框架版本。

## 本轮交付边界

提供实现和运行验证，不启动 100 轮正式实验，不报告可发表的准确率，也不承诺复现论文中的 97.42%。论文未明确部分保留可配置接口；正式训练前应锁定 `method_config.json`，在相同验证预算内决定参数，避免看测试集挑选实现。

两种输入的权重不能混用；单独测试从checkpoint配置自动恢复输入模式，不需要传--input。历史运行还须通过原有代码哈希检查，源码变化时使用其快照。不会把旧结果冒充当前双接口的新结果。

本轮8项数值/配置检查和两数据集×两输入的4组训练/保存/回读检查通过，同CUDA回读逐条预测一致。本轮没有启动正式训练。按用户要求，完成验证后删除冒烟脚本、smoke_runs、原文提取/页面图片和验证过程文件；保留实现代码、说明、既有正式结果和t-SNE代码。
