# Wang 2025 TUBR-ML 项目协议复现

独立实现 SDFE（4层CNN＋DropBlock）、梯度驱动的任务自适应参数衰减、Poly-1＋任务熵差正则（TURL）。复用当前项目PU/CWRU任务与数据来源，不修改主模型或AMAML。默认100 epoch；未运行正式100轮实验。

## 两个输入接口

| 参数 | 模型输入 | 数据来源 |
|---|---|---|
| `--input no_stft`（默认） | 7056点→逐窗min-max→84×84→3个相同灰度通道 | 项目已有MAT；按原文输入长度实时构造，不保存离线图片 |
| `--input stft` | 项目已有单通道64×64时频图 | 当前PNG缓存与原加载器 |

原文没有明确归一化公式，min-max是本实现的固定解释。原始输入更长，窗口合法起点范围随之改变，不能称为与旧STFT逐窗对齐。CWRU目标前后半段仍分别用于validation/test，7056点不会跨分段边界；PU保留原有窗口级划分及可能的重叠风险。

当前PU两种模式均可运行P1–P4；CWRU原始模式支持C0–C3，当前默认STFT缓存仅能直接运行C3。其他STFT任务需要事先存在的匹配图片缓存，通过 `--data-root` 或 `--data-map` 指定；脚本不会生成缺失图片。

## 固定默认设置

| 项目 | 设置 |
|---|---|
| 任务 | 多源留一目标；PU9/CWRU10类池随机5-way；1/3/5-shot，15-query |
| 重复 | seeds 3、24、38；1000个测试episode |
| 每轮/适应 | 16训练任务、16目标验证任务；训练内循环3步、测试10步 |
| 训练预算 | 最多100 epoch，patience100，保留项目高准确率确认早停 |
| 优化 | 内循环式5，lr0.01；外层Adam，lr0.001；cosine最低5e-5 |
| SDFE | 4×64通道、3×3卷积，GAP，5-way FC；LeakyReLU斜率0.01 |
| DropBlock | keep_prob0.9，block_size3，评估关闭 |
| TURL | τ=2，η=0.005（表7）；有符号熵差，两分支默认参与梯度 |
| 衰减模块 | 18维梯度均值描述符→18→ReLU→18→Sigmoid→求和 |

参数冲突、18维梯度描述符的解释、BN评估口径和项目/原文差异见 [notes.md](notes.md)。这是论文方法在项目任务上的重构，尚不能称为作者代码或原文数值的严格复现。

## 文件与运行

- `tubr_model.py`：SDFE、DropBlock、衰减模块、内循环与TURL。
- `raw_reshape.py`：7056点输入适配，复用项目MAT解析和数据划分逻辑。
- `run.py`：check/train/test入口，独立checkpoint与结果目录。
- `method_eta_text.json`：显式选择正文η=0.05的备选配置；默认不读取。
- `PyCharm本地运行命令清单.md`：可直接复制的本地命令。

默认动作是 `check`，不启动训练。需要训练请显式加 `--action train`。默认任务读取项目当前目标，正式批量实验建议明确 `--tasks all` 或具体任务名。

输出在本目录 `runs/<dataset>_<raw84或stft64>_<shot>s_<时间>/<任务>_s<seed>/`：包括best.pt、resolved_config、method_config、architecture、code_sha256与source_snapshot、data_signature、history.csv、episodes.jsonl、test_predictions.csv和result.json。原始模式另有raw_window_manifest。只有完整三seed才汇总均值与样本标准差，Macro-F1按汇总混淆矩阵计算。没有自动续训/跳过重复实验功能。

独立测试自动恢复保存时的输入模式，并核对源码、数据和checkpoint哈希。`stft`与`no_stft`首层通道不同，不能交换权重。保持训练与回读使用同类设备，不宣称CPU/CUDA逐位一致。

## 验证与清理

公式及行为检查覆盖：输入重排/归一化、两种输入形状、DropBlock连续区域及边界、TURL数值与先验梯度、式5一步更新与参数隔离、二阶梯度的有限差分比对、query标签不参与目标任务适应。

2026-10-12验证完成：上述7项检查全部通过；seed24下13个任务/输入组合通过数据检查（PU两模式各P1–P4，CWRU原始C0–C3与STFT C3），每类120源域训练/20目标验证/20目标测试样本；原始3×84×84、STFT1×64×64均正确。最终参数下，PU P1 1-shot、CWRU C3 5-shot各跑两种输入，共4组CUDA单轮训练，每组2个测试episode；checkpoint回读逐query预测完全一致。原始窗口7056点全部在各自分段内；禁止STFT生成的检查通过。

这些检查不用于比较诊断性能，也不是100轮正式实验。本轮临时验证脚本、日志、冒烟结果、论文提取/截图与计划按用户要求清理，保留本目录交付代码、参数文件和说明。
