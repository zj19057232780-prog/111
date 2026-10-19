# AMAML：PyCharm 本地运行命令清单

适用于 Windows 的 PyCharm Terminal（PowerShell）。以下命令对应本目录的 `run.py`，支持 `--input no_stft`（原始1444点重排38×38，默认）和 `--input stft`（已有64×64 STFT图片）。两者默认训练上限100 epoch。

## 1. 终端准备（每次新开终端先运行一次）

```powershell
Set-Location 'C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\mu2025_amaml'
$py = 'C:\ANACONDA\envs\ifmaml2\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
```

这里直接指定 Python 解释器，不需要再执行 `conda activate`。下方命令都使用同一个环境，并固定 `--device cuda`。如果终端不是 PowerShell，请先输入 `powershell` 再运行上述命令；这些命令不要粘贴到 Python Console 的 `>>>` 提示符中。

可检查实际解释器、PyTorch 和 GPU：

```powershell
& $py -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

## 2. 数据检查（不训练）

```powershell
# PU：检查P1–P4，5-shot+15-query
& $py -u run.py --input no_stft --dataset pu --tasks all --seeds 24 --shot 5 --action check --device cuda

# CWRU：检查C0–C3四个目标任务的原始MAT
& $py -u run.py --input no_stft --dataset cwru --tasks all --seeds 24 --shot 5 --action check --device cuda
```

看到 `CHECK PASSED` 即表示原始信号输入与support/query样本预算通过检查。它不代表正式训练完成。

## 3. 小规模运行验证（建议首次运行时执行）

```powershell
# PU P1，1-shot，seed24
& $py -u run.py --input no_stft --dataset pu --tasks P1 --seeds 24 --shot 1 --action smoke --device cuda

# CWRU C3，5-shot，seed24
& $py -u run.py --input no_stft --dataset cwru --tasks C3 --seeds 24 --shot 5 --action smoke --device cuda
```

每条只跑1轮、1个训练任务、1个验证任务和2个测试任务，结果写入 `smoke_runs/`，标记 `SMOKE_ONLY`，不能填入论文结果表。

## 4. 正式运行：no_stft整组命令

每条命令内部串行执行。等一条完成后再运行下一条，避免多个训练进程同时占用显存。

```powershell
# PU：1-shot，P1–P4 × seed3/24/38，共12次训练与测试
& $py -u run.py --input no_stft --dataset pu --tasks all --seeds 3 24 38 --shot 1 --epochs 100 --patience 100 --action train --device cuda

# PU：5-shot，P1–P4 × seed3/24/38，共12次训练与测试
& $py -u run.py --input no_stft --dataset pu --tasks all --seeds 3 24 38 --shot 5 --epochs 100 --patience 100 --action train --device cuda

# CWRU：1-shot，C0–C3 × seed3/24/38，共12次训练与测试
& $py -u run.py --input no_stft --dataset cwru --tasks all --seeds 3 24 38 --shot 1 --epochs 100 --patience 100 --action train --device cuda

# CWRU：5-shot，C0–C3 × seed3/24/38，共12次训练与测试
& $py -u run.py --input no_stft --dataset cwru --tasks all --seeds 3 24 38 --shot 5 --epochs 100 --patience 100 --action train --device cuda
```

**第4节与第5节是两种替代运行方式，选一种即可。** 当前入口不会自动跳过历史已完成结果；重复运行会创建新目录并重新训练，也不支持从中断处恢复训练。

## 5. 正式运行：no_stft按任务逐条执行

以下每条包含seed3/24/38三次训练与测试。与第4节相比，便于手动选择尚未运行的任务；各条结果分别保存，不自动跨目录合并全工况统计。

### PU：1-shot

```powershell
& $py -u run.py --input no_stft --dataset pu --tasks P1 --seeds 3 24 38 --shot 1 --action train --device cuda
& $py -u run.py --input no_stft --dataset pu --tasks P2 --seeds 3 24 38 --shot 1 --action train --device cuda
& $py -u run.py --input no_stft --dataset pu --tasks P3 --seeds 3 24 38 --shot 1 --action train --device cuda
& $py -u run.py --input no_stft --dataset pu --tasks P4 --seeds 3 24 38 --shot 1 --action train --device cuda
```

### PU：5-shot

```powershell
& $py -u run.py --input no_stft --dataset pu --tasks P1 --seeds 3 24 38 --shot 5 --action train --device cuda
& $py -u run.py --input no_stft --dataset pu --tasks P2 --seeds 3 24 38 --shot 5 --action train --device cuda
& $py -u run.py --input no_stft --dataset pu --tasks P3 --seeds 3 24 38 --shot 5 --action train --device cuda
& $py -u run.py --input no_stft --dataset pu --tasks P4 --seeds 3 24 38 --shot 5 --action train --device cuda
```

### CWRU：1-shot

```powershell
& $py -u run.py --input no_stft --dataset cwru --tasks C0 --seeds 3 24 38 --shot 1 --epochs 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset cwru --tasks C1 --seeds 3 24 38 --shot 1 --epochs 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset cwru --tasks C2 --seeds 3 24 38 --shot 1 --epochs 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset cwru --tasks C3 --seeds 3 24 38 --shot 1 --epochs 100 --action train --device cuda
```

### CWRU：5-shot

```powershell
& $py -u run.py --input no_stft --dataset cwru --tasks C0 --seeds 3 24 38 --shot 5 --epochs 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset cwru --tasks C1 --seeds 3 24 38 --shot 5 --epochs 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset cwru --tasks C2 --seeds 3 24 38 --shot 5 --epochs 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset cwru --tasks C3 --seeds 3 24 38 --shot 5 --epochs 100 --action train --device cuda
```

如只运行单个seed，将 `--seeds 3 24 38` 改成 `--seeds 24`。只跑一个seed不会生成完整三seed均值±标准差。

## 5A. 正式运行：STFT输入

PU默认图片支持P1–P4；CWRU默认图片仅支持C3。每条命令默认三个seed。

```powershell
# 先检查，均不生成图片
& $py -u run.py --input stft --dataset pu --tasks all --seeds 24 --shot 5 --action check --device cuda
& $py -u run.py --input stft --dataset cwru --tasks C3 --seeds 24 --shot 5 --action check --device cuda

# 正式训练
& $py -u run.py --input stft --dataset pu --tasks all --seeds 3 24 38 --shot 1 --epochs 100 --action train --device cuda
& $py -u run.py --input stft --dataset pu --tasks all --seeds 3 24 38 --shot 5 --epochs 100 --action train --device cuda
& $py -u run.py --input stft --dataset cwru --tasks C3 --seeds 3 24 38 --shot 1 --epochs 100 --action train --device cuda
& $py -u run.py --input stft --dataset cwru --tasks C3 --seeds 3 24 38 --shot 5 --epochs 100 --action train --device cuda
```

PU若要逐任务运行，把all改成P1/P2/P3/P4。CWRU的C0/C1/C2在STFT模式下必须提供各自已有缓存，不能仅修改任务名。

可用 `--data-root '已有匹配图片目录'` 运行一个任务；或准备JSON文件（键C0/C1/C2/C3，值为真实图片缓存绝对路径），再指定 `--input stft --tasks all --data-map '文件.json'`。脚本不自动制作缺失的图片。

## 6. 数据根目录

当前默认PU路径为 `C:/Users/JieZhang/Desktop/new-ifmaml/IFMAML-master/pu_data_mat`（整理后的condition/bearing.mat），CWRU路径为 `C:/Users/JieZhang/Desktop/new-ifmaml/IFMAML-master/CWRU`。CWRU的C0–C3现在都可直接从原始数据运行，不再要求各任务的STFT缓存。

若数据搬到其他位置，增加 `--data-root '实际MAT根目录'`。这是no_stft的路径规则。stft模式的data-root需填写图片缓存；data-map仅用于stft模式的各任务缓存映射。

保持原切窗起点与任务划分，实际使用每个原窗口前1444点，不做幅值归一化、插值或频谱变换。不会修改MAT或已有图片文件。

## 7. 已训练权重单独测试

将变量值换成实际运行输出中的单个任务目录，目录内应存在 `best.pt`、`result.json` 等文件：

```powershell
# 将此占位路径替换成真实训练目录
$checkpointDir = 'C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\mu2025_amaml\runs\实际批次目录\P1_s24'
& $py -u run.py --action test --checkpoint-dir $checkpointDir --device cuda
```

测试会读取保存时的dataset、shot、seed与方法配置，不能再靠命令行改变它们。正式训练目录默认测试1000个episode，smoke目录只测试2个。新测试文件保存到对应目录的 `eval_时间戳/`。保持与训练相同的CUDA设备；旧STFT版本曾出现CPU/GPU适应预测差异，新版同样不宣称跨设备等价。STFT和no_stft权重不能互换；回读从保存配置恢复模式，无需填写--input。

## 8. 如果使用PyCharm绿色运行按钮

新建一个Python运行配置，填写：

| 字段 | 值 |
|---|---|
| Python解释器 | `C:\ANACONDA\envs\ifmaml2\python.exe` |
| Script path | `C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\mu2025_amaml\run.py` |
| Working directory | `C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\mu2025_amaml` |
| Environment variables | `PYTHONIOENCODING=utf-8;PYTHONUNBUFFERED=1` |
| Parameters（先检查） | `--input no_stft --dataset pu --tasks all --seeds 24 --shot 5 --action check --device cuda` |

正式运行时，只把Parameters替换为所选终端命令中 `run.py` 后面的参数。例如：

```text
--input no_stft --dataset pu --tasks all --seeds 3 24 38 --shot 1 --epochs 100 --patience 100 --action train --device cuda
```

Parameters里不要填写 `& $py`、`python` 或 `run.py`。

## 9. 默认预算与结果位置

默认5-way，15-query；最多100轮，保留patience=100（该预算内通常不会先于上限触发），同时保留高准确率确认早停；每轮16个训练任务、16个验证任务，训练/验证3步适应，测试10步适应。正式训练结束后自动加载最佳权重并测试1000个episode。

正式输出目录为：

```text
C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\mu2025_amaml\runs
```

新批次目录名分别含raw38或stft64，结果按输入分开保存。优先查看每个任务的 `result.json`、`history.csv` 和批次的 `summary.json`。`best.pt`为最佳权重，`test_predictions.csv`为查询样本预测。按整组命令完成四任务×三个seed时，汇总才包含全工况均值与标准差。

本模型仍属于论文方法在项目协议下的重构，公式取舍和暂定超参数见 `notes.md`。本清单仅提供命令，没有启动任何正式训练。

## 10. t-SNE 出图（只读，不训练）

`run.py` 训练完只留下 `best.pt`、`history.csv`、`test_predictions.csv`，特征在 `episode()` 里被丢弃，所以出图需要单独的 `make_tsne.py`。它只读一个已完成的运行目录，不训练、不覆盖权重，输出写到该目录下的 `tsne/`。

需要的输入：

| 文件 | 来源 | 用途 |
|---|---|---|
| `best.pt` | 运行目录 | 验证准确率最高的权重 |
| `resolved_config.json` | 运行目录 | 类别表、数据根、k_shot、inner_lr、seed、size |
| `method_config.json` | 运行目录 | `adapt()` 的 ALC/AMOS 参数 |
| `data_signature.json` | 运行目录 | 重载数据后逐项校验，防止用错数据根 |
| 原始 MAT 根 | `pu_data_mat` / `CWRU` | 实际信号；路径已记录在 config，可用 `--data-root` 覆盖 |

不读 STFT 图片。重载后先比对 shape/labels/counts/sha256，不一致会直接报错。

### 两种模式

**A. 全部类别：`--mode episodes --classes all`**

重放该次运行自己的 meta-test episode，把全部类别的 query 特征汇到一张图。PU 9 类、CWRU 10 类。重放方式与 `run.py` 一致（重建 test split → `seed_all(100000 + seed)` → 同一个 `TaskDataset` 采样器），所以脚本打印的汇总准确率应当与 `result.json` 相同；不一致会告警。

```powershell
Set-Location 'C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\mu2025_amaml'
$py = 'C:\ANACONDA\envs\ifmaml2\python.exe'

# 全部 9 类，重放 1000 个测试 episode（PU P1 seed3 约 4~5 分钟）
& $py -u make_tsne.py --checkpoint-dir 'runs\pu_raw38_1s_20261012_155345_576647\P1_s3' --mode episodes --classes all

# 只是看形状、不想等：抽 200 个 episode，此时准确率不再等于 result.json
& $py -u make_tsne.py --checkpoint-dir 'runs\...\P1_s3' --mode episodes --classes all --episodes 200
```

**B. 5 类固定任务：`--mode fixed`（默认）**

与已有 `results/*_tsne.png` 同口径，便于横向比较。受 5-way 分类头限制，最多 5 类，超过会直接报错。

```powershell
& $py -u make_tsne.py --checkpoint-dir 'runs\...\P1_s3' --mode fixed
& $py -u make_tsne.py --checkpoint-dir 'runs\...\P1_s3' --mode fixed --classes Healthy IR_EDM_L1
```

其他参数：`--points`（t-SNE 点数上限，默认 1000）、`--perplexity`（默认 30）、`--inner-steps`、`--data-root`、`--device`。

### 输出与口径

`<运行目录>\tsne\<运行名>_<tag>_<时间戳>_tsne.png`，同名 `_points.csv`（二维坐标＋类别）和 `_meta.json`（模式、点数、准确率、checkpoint 哈希）。

- `--mode fixed` 的准确率只是该固定任务上的值，**不是** `result.json` 的 1000-episode 成绩，不要引用。
- `--mode episodes` 只支持 `--split test`：训练期 validation episode 依赖训练 RNG 历史，无法从固定种子复现。
- 两模式都沿用该模型的 episode 内 BatchNorm 统计（`track_running_stats=False`），特征来自整批 query 一次前向。
- `--mode episodes` 汇集的点来自**不同的适配参数集**，簇间距离只能当定性参考。

> 出完图请只保留 PNG/CSV/meta 结果文件，删掉临时脚本、smoke 产物和 `__pycache__`。

交付清理：冒烟命令仅为可选手动检查，生成的smoke_runs不属于正式结果；本轮验证产生的文件和独立测试脚本已按要求删除。
