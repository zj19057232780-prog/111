# STFT-CNN4-MAML 基准模型实验记录表

## 1. 实验基本信息

| 项目 | 记录 |
|---|---|
| 实验名称 | STFT-CNN4-MAML 少样本轴承故障诊断基准模型 |
| 实验日期 | 2026-05-31 |
| 数据集 | 德国帕德博恩轴承数据集（Paderborn University Bearing Dataset） |
| 原始数据路径 | `C:\数据集\德国帕德博恩轴承数据集` |
| 项目路径 | `C:\Users\JieZhang\Desktop\new-ifmaml\IFMAML-master` |
| 当前目标 | 建立“原始振动信号 + STFT 图像 + CNN4 + MAML”的可复现实验基准 |
| 当前状态 | STFT 图像已重新生成并完成合理性检查；正式基准训练需重新运行 |
| 重要说明 | 旧的 `model_save\STFT_CNN4_MAML_best` 若来自通道修复前训练，不作为正式基准结果 |

## 2. 实验目的

| 编号 | 目的 |
|---|---|
| P1 | 建立后续创新模块对比的基础模型，作为论文实验中的 baseline |
| P2 | 验证 STFT 时频图输入在 MAML 少样本故障诊断框架下的基础效果 |
| P3 | 为后续频域增强分支、替换 CNN 特征提取器、注意力模块缝合提供统一对照 |
| P4 | 保持元学习训练框架不变，只替换或增加前端特征模块，方便做消融实验 |

## 3. 数据处理记录

| 项目 | 当前设置 |
|---|---|
| 原始数据格式 | `.mat` |
| 规范化中间数据目录 | `./pu_data_mat` |
| STFT 图像输出目录 | `./pu_data_processed` |
| 原始文件整理方式 | 每个故障类别、每个工况选择 1 个 trial |
| trial 选择策略 | `trial_pick = min` |
| 使用的真实信号通道 | 优先选择 `Y` 中的 `vibration_1` 等振动通道 |
| 已修正问题 | 之前误读到 `X` 时间轴；现已修正为读取非单调振动信号 |
| 每类 STFT 样本数 | `40` |
| 图像尺寸 | `64 x 64` |
| 图像通道 | 单通道灰度图 |
| 图像归一化 | 读取时由 `0-255` 转为 `0-1` float32 |

### 3.1 工况划分

| 用途 | 工况 |
|---|---|
| Meta-train 源域工况 | `N09_M07_F10`, `N15_M07_F04`, `N15_M07_F10` |
| Target 目标工况 | `N15_M01_F10` |
| Target validation/test 划分 | 每类按 `0.5 / 0.5` 划分 |

### 3.2 数据规模

| 数据部分 | 数量 |
|---|---:|
| 工况总数 | 4 |
| 每个工况故障类别数 | 32 |
| 每个类别 STFT 图像数 | 40 |
| STFT 图像总数 | 5120 |
| 训练集图像数 | 3840 |
| 验证集图像数 | 640 |
| 测试集图像数 | 640 |

### 3.3 STFT 图片合理性检查

| 检查项 | 结果 |
|---|---|
| 图像数量 | 5120 张 |
| 图像尺寸 | 全部为 `64 x 64` |
| 空白图 | 0 |
| 低方差异常图 | 0 |
| 精确重复图 | 0 |
| 相邻窗口相关性 | 约 `-0.04 ~ 0.48`，未出现异常高重复 |
| 代表性拼图 | `stft_montage_check_after_fix.png` |
| 结论 | 当前 STFT 图像合理，可用于正式基准实验 |

## 4. STFT 预处理参数

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `preprocess_mode` | `stft_log` | STFT 幅值谱取对数后转灰度图 |
| `window_size` | 4096 | 每个样本截取的原始信号长度 |
| `stft_fs` | 64000 | 采样频率 |
| `stft_nperseg` | 256 | STFT 每段长度 |
| `stft_noverlap` | 192 | STFT 相邻窗重叠长度 |
| `stft_window` | `hann` | STFT 窗函数 |
| `stft_bandpass` | `None` | 当前不使用带通滤波 |
| `stft_log_eps` | `1e-8` | 防止对数计算溢出 |
| `stft_p_low` | 5 | 灰度拉伸下百分位 |
| `stft_p_high` | 95 | 灰度拉伸上百分位 |
| `stft_gamma` | 0.7 | 灰度 gamma 增强 |
| `preprocess_seed` | 24 | 数据截取随机扰动种子 |
| 信号标准化 | 去均值 | `normalize` 未显式设为 `zscore`，默认 demean |

## 5. 模型结构记录

### 5.1 总体流程

```text
原始 PU .mat 振动信号
    -> 选择 vibration 振动通道
    -> 固定窗口截取 4096 点信号片段
    -> STFT + log 幅值谱
    -> 64 x 64 单通道灰度图
    -> CNN4Backbone 特征提取
    -> Linear 分类头
    -> MAML 内外循环训练
```

### 5.2 模块简要介绍

| 模块 | 文件 | 作用 |
|---|---|---|
| 数据整理 | `prepare_pu_baseline_data.py` | 将原始 PU 数据整理为 `condition/fault.mat` 格式 |
| 信号读取 | `pu_loader.py` | 从 `.mat` 文件中读取真实振动通道，避免误读时间轴 |
| STFT 预处理 | `pu_preprocess.py` | 将一维振动信号切片并转换为 STFT 灰度图 |
| 数据集构建 | `pu_dataset.py` | 读取 STFT 图像，构建 train/validation/test 数据 |
| Episode 采样 | `l2l_shim.py` | 构建 N-way K-shot MAML 任务，返回 support/query |
| CNN4 模型 | `maml_model.py` | 使用 4 层卷积块提取图像特征，并接线性分类头 |
| MAML 训练 | `maml_train_pic.py` | 执行 MAML 内循环适应和外循环元优化 |
| 一键运行 | `run_stft_cnn4_baseline.py` | 串联 prepare、preprocess、train、test 流程 |

### 5.3 CNN4Backbone 参数

| 参数 | 当前值 |
|---|---:|
| 输入尺寸 | `[batch, 1, 64, 64]` |
| 卷积层数 | 4 |
| 每层通道数 | 64 |
| 卷积核大小 | `3 x 3` |
| padding | 1 |
| 每层结构 | Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d |
| 下采样过程 | `64 -> 32 -> 16 -> 8 -> 4` |
| 输出特征维度 | `64 x 4 x 4 = 1024` |
| 分类头 | Linear(`1024`, `5`) |
| 初始化方式 | Conv/Linear 使用 MAML 风格 Xavier 初始化 |
| 注意 | `SEAttention` 目前仅在文件中定义，未接入当前基准模型 |

## 6. MAML 训练参数

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `n_way` | 5 | 每个 episode 采样 5 个类别 |
| `k_shot` | 1 | 每类 support 样本数 |
| `q_query` | 15 | 每类 query 样本数 |
| support 总数 | 5 | `5-way x 1-shot` |
| query 总数 | 75 | `5-way x 15-query` |
| `inner_lr` | 0.05 | MAML 内循环学习率 |
| `outer_lr` | 0.005 | MAML 外循环 Adam 学习率 |
| `epochs` | 250 | 正式训练轮数 |
| `tasks_per_epoch` | 1000 | 每轮可采样任务数 |
| `meta_batch_size` | 16 | 每次外循环累积 episode 数 |
| 1-shot adaptation steps | 3 | 训练阶段 1-shot 内循环步数 |
| `test_inner_steps` | 10 | 测试阶段内循环适应步数 |
| `test_meta_batch_size` | 100 | 测试 episode 数 |
| `seed` | 24 | 随机种子 |
| 损失函数 | CrossEntropyLoss | query 集分类损失 |
| 优化器 | Adam | 优化 MAML 外循环参数 |
| 设备 | 自动选择 CUDA/CPU | `torch.cuda.is_available()` |

## 7. 项目运行流程

### 7.1 完整正式流程

```powershell
cd C:\Users\JieZhang\Desktop\new-ifmaml\IFMAML-master

python run_stft_cnn4_baseline.py --prepare --preprocess --clean_png

python run_stft_cnn4_baseline.py --train

python run_stft_cnn4_baseline.py --test
```

### 7.2 快速冒烟测试流程

```powershell
cd C:\Users\JieZhang\Desktop\new-ifmaml\IFMAML-master

python run_stft_cnn4_baseline.py --train --test --quick
```

### 7.3 分步骤运行流程

| 步骤 | 命令 | 作用 |
|---|---|---|
| 1 | `python prepare_pu_baseline_data.py` | 从原始数据复制并整理 `.mat` 文件 |
| 2 | `python pu_preprocess.py --clean` | 删除旧图并重新生成 STFT 图像 |
| 3 | `python maml_train_pic.py --train` | 正式训练 STFT-CNN4-MAML |
| 4 | `python maml_train_pic.py --test` | 加载最佳权重并测试 |
| 5 | `python maml_train_pic.py --train --test --quick` | 快速验证代码是否跑通 |

## 8. 本次基准实验结果记录

| 实验编号 | 日期 | 模型 | 数据版本 | 训练轮数 | Meta-test Accuracy | Meta-test Error | 最佳验证准确率 | 最佳 epoch | 备注 | seed |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| B0 | 2026-05-31 | STFT-CNN4-MAML | STFT 图已修复 | quick=3 | 待重跑 | 待重跑 | 待重跑 | 待重跑 | 通道修复后需重新训练 | 24 |
| B1 |  | STFT-CNN4-MAML | 正式 STFT 数据 | 250 | 92.51 | 0.3084 | 90.33 | 241 | 正式 baseline | 24 |
|  | |  |  | 250 | 95.59 | 0.2693 | 95.5 | 232 | 正式 baseline | 38 |
|  | |  |  | 250 | 83.79 | 0.5003 | 84 | 235 |  | 3 |
|  | |  |  | 500 | 95.32 | 0.1719 | 96.08 | 468 |  | 3 |
|  | |  |  | 500 | 97.55 | 0.1354 | 98.67 | 415 |  | 38 |
|  | |  |  | 500 | 97.07 | 0.1316 | 97.58 | 497 |  | 24 |
| | | | |  |  |  |  |  | |  |
| | | | |  |  |  |  |  | |  |
| | | | |  |  |  |  |  | |  |

## 09. 后续消融实验记录模板

| 实验编号 | 改动模块 | 具体方案 | 是否保持 MAML 不变 | 是否保持 STFT 参数不变 | Accuracy | 提升幅度 | 备注 |
|---|---|---|---|---|---:|---:|---|
| A1 | 频域增强分支 |  | 是 | 是 |  |  |  |
| A2 | 特征提取网络 | 替换 CNN4 | 是 | 是 |  |  |  |
| A3 | 注意力模块 |  | 是 | 是 |  |  |  |
| A4 | 组合模型 | 频域增强 + 新特征提取器 + 注意力 | 是 | 是 |  |  |  |

## 10. 当前结论

当前基准模型的核心链路已经明确：真实振动信号经 STFT 转为 `64 x 64` 单通道时频图，再输入 CNN4Backbone，并通过 MAML 进行 5-way 1-shot 少样本训练。  

当前最重要的注意事项是：STFT 通道读取已修复，因此正式 baseline 必须基于修复后的 `pu_data_processed` 重新训练，不能沿用通道修复前保存的旧权重结果。
