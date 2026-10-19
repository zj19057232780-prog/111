# TUBR-ML：PyCharm 本地运行命令

适用于PyCharm Terminal中的PowerShell；默认内外学习率0.01/0.001、100 epoch、patience100、seed3/24/38。命令只运行本独立模型。

## 1. 工作目录和解释器

```powershell
Set-Location 'C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\wang2025_tubr_ml'
$py = 'C:\ANACONDA\envs\ifmaml2\python.exe'
& $py -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 2. 只检查数据，不训练

```powershell
& $py -u run.py --input no_stft --dataset pu --tasks all --shot 5 --seeds 24 --action check
& $py -u run.py --input no_stft --dataset cwru --tasks all --shot 5 --seeds 24 --action check
& $py -u run.py --input stft --dataset pu --tasks all --shot 5 --seeds 24 --action check
& $py -u run.py --input stft --dataset cwru --tasks C3 --shot 5 --seeds 24 --action check
```

## 3. 原文原始输入：7056点→归一化→84×84

以下每条运行四任务×三seed，并分别输出结果。1-shot与5-shot是独立实验。

```powershell
& $py -u run.py --input no_stft --dataset pu --tasks all --shot 1 --seeds 3 24 38 --epochs 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset pu --tasks all --shot 5 --seeds 3 24 38 --epochs 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset cwru --tasks all --shot 1 --seeds 3 24 38 --epochs 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset cwru --tasks all --shot 5 --seeds 3 24 38 --epochs 100 --action train --device cuda
```

如需逐任务运行，将 `--tasks all` 替换为PU的 `P1`/`P2`/`P3`/`P4`，或CWRU的 `C0`/`C1`/`C2`/`C3`。例如：

```powershell
& $py -u run.py --input no_stft --dataset cwru --tasks C0 --shot 1 --seeds 3 24 38 --epochs 100 --action train --device cuda
```

整组与逐任务二选一，程序不会自动跳过已完成任务。可仅用 `--seeds 24` 开始单seed实验，但结果不能作为三seed mean±SD。

## 4. 项目STFT输入

```powershell
& $py -u run.py --input stft --dataset pu --tasks all --shot 1 --seeds 3 24 38 --epochs 100 --action train --device cuda
& $py -u run.py --input stft --dataset pu --tasks all --shot 5 --seeds 3 24 38 --epochs 100 --action train --device cuda
& $py -u run.py --input stft --dataset cwru --tasks C3 --shot 1 --seeds 3 24 38 --epochs 100 --action train --device cuda
& $py -u run.py --input stft --dataset cwru --tasks C3 --shot 5 --seeds 3 24 38 --epochs 100 --action train --device cuda
```

CWRU当前STFT缓存仅匹配C3。其他任务需已有匹配缓存；单任务可加 `--data-root '实际图片根目录'`，多任务可加 `--data-map '实际映射.json'`，JSON的键为任务名、值为已有缓存路径，相对路径以JSON所在目录为基准。两参数互斥，data-map仅用于STFT；不自动生成图片。

## 5. 数据路径与参数

原始PU默认读取 `IFMAML-master/pu_data_mat`，原始CWRU默认读取 `IFMAML-master/CWRU`。搬动数据时，no_stft的 `--data-root` 应为MAT根目录，stft的同一参数应为图片根目录。

模型默认参数已固定；无需额外加参数即可运行。需要有意更改时：

```powershell
# 显式写出默认学习率
& $py -u run.py --input no_stft --dataset pu --tasks P1 --shot 1 --inner-lr 0.01 --outer-lr 0.001 --epochs 100 --action train --device cuda

# 仅当决定采用论文正文eta=0.05时运行；默认采用表7的0.005
& $py -u run.py --input no_stft --dataset pu --tasks P1 --shot 1 --method-config method_eta_text.json --epochs 100 --action train --device cuda
```

第二条是不同参数实验，不要与默认配置结果混在同一表格单元中。method-config亦支持notes列出的选项；每次实际参数都会写入method_config.json。

## 6. 已完成训练的独立测试

将路径替换为实际生成的单个seed目录；输入模式、shot及方法参数自动从保存配置恢复。

```powershell
& $py -u run.py --action test --checkpoint-dir '.\runs\实际批次目录\P1_s24' --device cuda
```

测试结果另存为该目录下的 `eval_时间戳`。需使用与训练一致的源码与数据；源码改变则按该次source_snapshot恢复版本，不跳过一致性检查。

## 7. PyCharm绿色运行按钮

| 设置项 | 值 |
|---|---|
| Interpreter | `C:\ANACONDA\envs\ifmaml2\python.exe` |
| Script path | `C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\wang2025_tubr_ml\run.py` |
| Working directory | `C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\wang2025_tubr_ml` |
| Parameters | `--input no_stft --dataset pu --tasks all --shot 1 --seeds 3 24 38 --epochs 100 --action train --device cuda` |

选择STFT时改为 `--input stft`；CWRU同时将dataset改cwru，并明确选择当前缓存支持的C3。无CUDA时改为 `--device cpu`，耗时会增加。

正式结果与复现必需的配置/权重保留；本次实现期间的冒烟脚本、日志及结果在交付时清理。
