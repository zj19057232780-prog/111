# MFEML：PyCharm本地运行命令

以下为Windows PyCharm Terminal（PowerShell）命令。按用户要求默认最多500次元更新，验证准确率连续100轮无严格提升则早停（持平计入），停止后使用验证最佳权重测试。每批25任务、内循环训练/测试10步、内外学习率0.1/0.01沿用论文表2。

## 1. 进入目录

```powershell
Set-Location 'C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\li2025_mfeml'
$py = 'C:\ANACONDA\envs\ifmaml2\python.exe'
& $py -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 2. 数据检查

```powershell
& $py -u run.py --input no_stft --dataset pu --tasks all --shot 5 --seeds 24 --action check
& $py -u run.py --input no_stft --dataset cwru --tasks all --shot 5 --seeds 24 --action check
& $py -u run.py --input stft --dataset pu --tasks all --shot 5 --seeds 24 --action check
& $py -u run.py --input stft --dataset cwru --tasks C3 --shot 5 --seeds 24 --action check
```

`nostft`与`no_stft`同义。检查只读数据，不生成STFT或其他离线数据。CWRU图片缓存必须匹配任务：当前直接可运行C3；C0–C2需要已有的匹配缓存，原始模式不依赖图片缓存。

## 3. 按原文原始输入训练

每条命令运行四任务×三seed；1-shot和5-shot分别运行。

```powershell
& $py -u run.py --input no_stft --dataset pu --tasks all --shot 1 --seeds 3 24 38 --epochs 500 --patience 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset pu --tasks all --shot 5 --seeds 3 24 38 --epochs 500 --patience 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset cwru --tasks all --shot 1 --seeds 3 24 38 --epochs 500 --patience 100 --action train --device cuda
& $py -u run.py --input no_stft --dataset cwru --tasks all --shot 5 --seeds 3 24 38 --epochs 500 --patience 100 --action train --device cuda
```

逐任务执行时，将 `--tasks all` 换成PU的P1/P2/P3/P4或CWRU的C0/C1/C2/C3，例如：

```powershell
& $py -u run.py --input nostft --dataset cwru --tasks C0 --shot 1 --seeds 3 24 38 --epochs 500 --patience 100 --action train --device cuda
```

整组和逐任务命令二选一，以免重复运行。只有一个seed不能写三seed标准差。

## 4. 使用项目STFT输入训练

```powershell
& $py -u run.py --input stft --dataset pu --tasks all --shot 1 --seeds 3 24 38 --epochs 500 --patience 100 --action train --device cuda
& $py -u run.py --input stft --dataset pu --tasks all --shot 5 --seeds 3 24 38 --epochs 500 --patience 100 --action train --device cuda
& $py -u run.py --input stft --dataset cwru --tasks C3 --shot 1 --seeds 3 24 38 --epochs 500 --patience 100 --action train --device cuda
& $py -u run.py --input stft --dataset cwru --tasks C3 --shot 5 --seeds 3 24 38 --epochs 500 --patience 100 --action train --device cuda
```

CWRU其他STFT任务：单任务用 `--data-root '已有的匹配图片根目录'`；多任务用 `--data-map '实际映射.json'`，JSON键为C0/C1/C2/C3、值为对应现有缓存目录。相对路径以JSON所在目录为基准，两个参数互斥。程序不补建缺失缓存。

## 5. 覆盖训练预算或路径

只需要100轮时，显式使用 `--epochs 100`，其他论文参数不变：

```powershell
& $py -u run.py --input no_stft --dataset pu --tasks P1 --shot 1 --seeds 3 24 38 --epochs 100 --action train --device cuda
```

可覆盖的参数：`--inner-lr 0.1`、`--outer-lr 0.01`、`--meta-batch-size 25`、`--train-steps 10`、`--test-steps 10`、`--grad-clip 1`、`--patience 100`。上列为默认值，`--patience 0`表示关闭早停，0 grad-clip表示不裁剪。默认没有cosine调度。若需要论文1000轮预算且不早停，显式设置 `--epochs 1000 --patience 0`。

原始数据默认PU为 `IFMAML-master/pu_data_mat`，CWRU为 `IFMAML-master/CWRU`。搬移时用 `--data-root '实际MAT根目录'`；同一参数在STFT模式需指向图片目录。其余实现参数可通过 `--method-config '实际参数.json'` 显式修改，字段见notes.md；所有实际参数会保存，不按测试集调参。

## 6. 独立测试已有checkpoint

替换成实际的单任务、单seed结果目录：

```powershell
& $py -u run.py --action test --checkpoint-dir '.\runs\实际批次目录\P1_s24' --device cuda
```

输入模式、shot、学习率和适应步数由保存配置恢复。新评测写入其 `eval_时间戳/`，不覆盖训练结果；源码/数据/权重哈希必须匹配，源码变动后需使用source_snapshot归档版本。STFT和原始模式不可混用结果。

## 7. PyCharm绿色运行按钮

| 设置 | 值 |
|---|---|
| Interpreter | `C:\ANACONDA\envs\ifmaml2\python.exe` |
| Script path | `C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\li2025_mfeml\run.py` |
| Working directory | `C:\Users\JieZhang\Desktop\new-ifmaml\comparison_models\li2025_mfeml` |
| Parameters | `--input no_stft --dataset pu --tasks all --shot 1 --seeds 3 24 38 --epochs 500 --patience 100 --action train --device cuda` |

切换STFT只需改 `--input stft`；CWRU图片模式同时明确选择当前缓存支持的任务。无CUDA时可改 `--device cpu`，二阶10步训练会更慢。
