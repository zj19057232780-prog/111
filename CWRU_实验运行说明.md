# CWRU 10状态跨工况小样本实验

该实验与现有 PU 实验相互独立，不修改 `config_pu.py`、PU 数据目录或 PU
训练入口。

## 实验设置

- 信号：48 kHz 驱动端 `DE_time`
- 类别：正常状态，以及内圈、滚动体、外圈6点方向的3种损伤尺寸
- 源域：`1797 + 1772 + 1750 rpm`
- 目标域：`1730 rpm`
- 任务：`5-way 1-shot 15-query`
- 每个源工况每类40张图片
- 目标工况每类20张验证图片和20张测试图片
- 目标验证集取原始信号前半段，测试集取后半段

## 文件说明

| 文件 | 作用 |
|---|---|
| `config_cwru.py` | CWRU数据、模型和MAML配置 |
| `cwru_preprocess.py` | MAT信号读取及STFT图片生成 |
| `cwru_dataset.py` | 固定目标验证/测试划分的数据集 |
| `run_cwru_maml.py` | 独立预处理、训练和测试入口 |

## 第一次运行

进入项目目录：

```powershell
cd C:\Users\JieZhang\Desktop\new-ifmaml\IFMAML-master
```

重新生成全部CWRU STFT图片：

```powershell
python run_cwru_maml.py --preprocess --clean
```

正式训练：

```powershell
python run_cwru_maml.py --train
```

使用最佳权重测试：

```powershell
python run_cwru_maml.py --test
```

训练后立即测试：

```powershell
python run_cwru_maml.py --train --test
```

## 快速流程检查

```powershell
python run_cwru_maml.py --train --test --quick
```

快速模式只训练3个epoch并测试4个episode，结果不能作为正式实验结果。

## 输出位置

预处理数据：

```text
cwru_data_processed/
```

模型权重：

```text
model_save/CWRU_STFT_LSKLite_ARSM_MAML_10state_C012_to_C3_best
```

训练曲线、预测结果、混淆矩阵和t-SNE：

```text
results/cwru/
```

数据预处理清单位于：

```text
cwru_data_processed/manifest.csv
```

该清单记录每个类别实际读取的MAT变量和目标域时间范围，可用于检查数据泄漏。
