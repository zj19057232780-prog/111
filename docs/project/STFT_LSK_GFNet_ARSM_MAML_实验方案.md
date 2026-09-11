# 基于自适应残差收缩模块的 Freq-LSK-MAML 少样本轴承故障诊断实验方案

## 1. 方案定位

### 1.1 研究对象

本实验面向 Paderborn University（PU）轴承数据集的跨工况少样本故障诊断任务，在当前已经完成的 `STFT-LSKLite-GFNetLite-MAML` 模型基础上，增加一个轻量化的自适应残差收缩模块：

```text
Adaptive Residual Shrinkage Module，ARSM
```

ARSM 通过学习通道相关的软阈值，抑制跨工况迁移过程中产生的背景噪声、弱相关响应和冗余特征，同时保留具有较高判别价值的故障冲击特征。

### 1.2 设计原则

- 保持现有 STFT 数据、PU 数据划分和 MAML 训练框架不变；
- 不再叠加 EMA、GCNet 等通用注意力模块；
- 只在现有骨干中增加一个轻量去噪模块；
- 采用残差连接和较小分支缩放系数，减少训练初期的不稳定；
- 通过严格消融实验判断提升是否来自 ARSM，而不是训练随机性。

### 1.3 推荐模型名称

论文和代码中建议采用以下名称：

```text
STFT-LSK-GFNet-ARSM-MAML
```

中文名称：

```text
基于频域增强与自适应残差收缩的轻量大核 MAML 网络
```

---

## 2. 当前实验依据

现有困难迁移场景 S4 的主要结果如下：

| 模型 | Meta-test Accuracy | 相对 CNN4 |
|---|---:|---:|
| `STFT-CNN4-MAML` | 88.12% | 基线 |
| `STFT-LSKLite-MAML` | 92.11% | +3.99% |
| `STFT-LSKLite-GFNetLite-MAML` | 92.69% | +4.57% |
| `STFT-LSKLite-GFNetLite-EMA-MAML` | 约 90.13% | 下降 |
| `STFT-LSKLite-GFNetLite-GCNet-MAML` | 约 89.66% | 下降 |

上述结果说明：

1. LSK-lite 对 STFT 图中的多尺度冲击纹理有效；
2. GFNet-lite 能进一步补充全局频率建模；
3. 继续加入通用注意力会增加特征重标定冲突；
4. 下一步更适合增加具有明确去噪作用的软阈值模块。

---

## 3. 总体模型结构

### 3.1 推荐结构

```text
PU raw vibration signal
        ↓
Signal segmentation
        ↓
STFT log-magnitude image
        ↓
64 × 64 grayscale image
        ↓
Conv Stem
        ↓
LSK-lite Backbone
        ↓
GFNet-lite Frequency Enhancement
        ↓
Adaptive Residual Shrinkage Module
        ↓
Global Average Pooling
        ↓
Linear Classifier
        ↓
MAML Episodic Training
```

推荐将 ARSM 放在 GFNet-lite 之后。此时模块处理的是经过局部大核提取和全局频率筛选后的高层特征，空间尺寸较小、通道语义较强，增加的计算量也较低。

### 3.2 模块及文献来源总表

| 模块 | 当前模型中的作用 | 主要文献来源 | 本地来源 |
|---|---|---|---|
| STFT 时频变换 | 将一维非平稳振动信号转换为二维时频表示 | MixMamba-Fewshot 等少样本轴承诊断工作采用时频图输入；相关综述整理了 STFT 在旋转机械诊断中的应用 | `C:\Users\JieZhang\Desktop\文献\Than 等 - Mixmamba-fewshot mamba and attention mixer-based method with few-shot learning for bearing fault di.pdf`；`C:\Users\JieZhang\Desktop\文献\s10462-022-10293-3.pdf` |
| LSK-lite | 通过大核与空洞深度卷积选择不同感受野，提取冲击条纹和宽频带结构 | *Large Selective Kernel Network for Remote Sensing Object Detection* | `C:\Users\JieZhang\Desktop\new-ifmaml\模块1LSK\Large Selective Kernel Network for Remote Sensing Object Detection.pdf` |
| GFNet-lite | 在二维傅里叶域中使用可学习复数滤波器建模全局频率依赖 | *Global Filter Networks for Image Classification* | `C:\Users\JieZhang\Desktop\new-ifmaml\模块2gfnet\2107.00645v2.pdf` |
| ARSM 自适应残差收缩 | 学习软阈值，压制噪声和冗余响应，保留显著故障特征 | RSNLRN 中的残差收缩思想；SMLDMN 中的自适应软阈值去噪思想 | `C:\Users\JieZhang\Desktop\文献\Gao 等 - 2025 - Metric-Based Meta-Learning Relation Network for Cross-Domain Few-Shot Bearing Fault Diagnosis.pdf`；`C:\Users\JieZhang\Desktop\文献\Cross-scenarios few-shot fault diagnosis for rolling bearings via.pdf` |
| MAML | 学习可快速适应新工况任务的模型初始参数 | *Few-Shot Bearing Fault Diagnosis Based on Model-Agnostic Meta-Learning*；IFMAML | `C:\Users\JieZhang\Desktop\文献\Few-Shot_Bearing_Fault_Diagnosis_Based_on_Model-Agnostic_Meta-Learning.pdf`；`C:\Users\JieZhang\Desktop\文献\An information fusion-based meta transfer learning method for.pdf` |

> 说明：LSK、GFNet 和软阈值思想均有明确文献来源。本文的可主张创新不是“首次提出这些基础模块”，而是面向跨工况 STFT 少样本诊断，对三类机制进行轻量化适配，并设计适合二阶 MAML 优化的残差收缩接入方式。

---

## 4. 各模块设计

## 4.1 STFT 时频输入模块

### 作用

STFT 将原始振动信号转换为同时包含时间变化和频率分布的二维图像，使卷积网络能够识别：

- 周期性冲击纹理；
- 故障特征频带；
- 谐波与边频结构；
- 不同工况下的能量迁移。

### 当前参数

| 参数 | 设置 |
|---|---:|
| 信号窗口长度 | 4096 |
| 采样频率 | 64000 Hz |
| `nperseg` | 256 |
| `noverlap` | 192 |
| 窗函数 | Hann |
| 图像大小 | `64 × 64` |
| 通道数 | 1 |
| 幅值处理 | `log10(abs(Zxx) + eps)` |
| 分位裁剪 | 5%–95% |
| Gamma | 0.7 |

### 文献来源

1. *MixMamba-Fewshot: Mamba and Attention Mixer-Based Method With Few-Shot Learning for Bearing Fault Diagnosis*。
2. `s10462-022-10293-3.pdf`，旋转机械振动信号深度学习诊断综述。

### 本实验改动

不修改当前 STFT 参数，防止数据预处理变化干扰 ARSM 的有效性判断。

---

## 4.2 LSK-lite 大核选择骨干

### 作用

LSK 模块使用两个不同感受野的深度卷积分支：

```text
5 × 5 depthwise convolution
        ↓
7 × 7 dilated depthwise convolution，dilation = 3
```

随后根据输入内容生成空间选择权重，自适应融合两个尺度的响应。它适合捕捉 STFT 图中的长条频带、局部冲击和跨区域关联。

### 文献来源

```text
Li et al.
Large Selective Kernel Network for Remote Sensing Object Detection
ICCV 2023
```

本地论文：

```text
C:\Users\JieZhang\Desktop\new-ifmaml\模块1LSK\
Large Selective Kernel Network for Remote Sensing Object Detection.pdf
```

本地参考代码：

```text
C:\Users\JieZhang\Desktop\new-ifmaml\模块1LSK\LSKNet-main\
LSKNet-main\mmrotate\models\backbones\lsknet.py
```

### 当前轻量化设置

| 项目 | 设置 |
|---|---|
| Stage 通道数 | `(32, 64, 96)` |
| Stage 深度 | `(1, 1, 1)` |
| MLP ratio | 2 |
| 下采样方式 | Stem 和 Stage 间 stride=2 |
| 最终特征维度 | 96 |

### 本实验改动

不修改 LSK-lite 结构，直接沿用已经验证有效的版本。

---

## 4.3 GFNet-lite 全局频率增强模块

### 作用

GFNet 将特征图转换到二维傅里叶域，通过可学习复数权重对不同频率成分进行全局筛选：

```text
X_f = FFT2(X)
Y_f = X_f ⊙ W
Y   = IFFT2(Y_f)
```

其中，`W` 为可学习的复数频率滤波器。

LSK 主要负责空间域的大感受野选择，GFNet 负责频率域的全局建模，二者作用不同。

### 文献来源

```text
Rao et al.
Global Filter Networks for Image Classification
NeurIPS 2021
```

本地论文：

```text
C:\Users\JieZhang\Desktop\new-ifmaml\模块2gfnet\2107.00645v2.pdf
```

本地参考代码：

```text
C:\Users\JieZhang\Desktop\new-ifmaml\模块2gfnet\
GFNet-master\GFNet-master\gfnet.py
```

### 当前轻量化设置

| 参数 | 设置 |
|---|---:|
| GFNet 深度 | 1 |
| MLP ratio | 2 |
| 复数权重初始化尺度 | 0.02 |
| Layer scale 初值 | `1e-2` |

### 本实验改动

保持当前 GFNet-lite 参数不变，将其作为 ARSM 的前置特征增强模块。

---

## 4.4 ARSM 自适应残差收缩模块

### 4.4.1 文献依据

ARSM 主要借鉴以下两类相关研究。

#### 来源一：RSNLRN 残差收缩模块

```text
Gao et al., 2025
Metric-Based Meta-Learning Relation Network for
Cross-Domain Few-Shot Bearing Fault Diagnosis
```

该研究在特征提取过程中引入残差收缩和软阈值机制，用于过滤复杂工况中的噪声响应。

本地论文：

```text
C:\Users\JieZhang\Desktop\文献\
Gao 等 - 2025 - Metric-Based Meta-Learning Relation Network for
Cross-Domain Few-Shot Bearing Fault Diagnosis.pdf
```

#### 来源二：SMLDMN 自适应软阈值去噪

```text
Cross-scenarios few-shot fault diagnosis for rolling bearings via
inter-domain similarity-guided meta-learning with
dual-attention multiscale feature denoising network
```

该研究使用软阈值收缩抑制跨场景信号中的噪声，并增强少样本故障特征。

本地论文：

```text
C:\Users\JieZhang\Desktop\文献\
Cross-scenarios few-shot fault diagnosis for rolling bearings via.pdf
```

### 4.4.2 软阈值函数

对输入特征 `x`，软阈值操作定义为：

```text
Shrink(x, τ) = sign(x) × max(|x| - τ, 0)
```

其中：

- `τ` 为自适应学习的通道阈值；
- 小于阈值的弱响应被置零；
- 大于阈值的响应被保留并适当收缩；
- 与硬阈值相比，软阈值连续且更适合梯度反向传播。

### 4.4.3 阈值生成

设输入特征为：

```text
x ∈ R^(B×C×H×W)
```

首先计算每个通道的绝对响应均值：

```text
s = GAP(|x|)
```

通过两层轻量 MLP 生成通道阈值比例：

```text
a = sigmoid(FC2(ReLU(FC1(s))))
```

最终阈值：

```text
τ = a × s
```

这种设计使阈值随样本和通道动态变化，而不是使用人工固定阈值。

### 4.4.4 残差接入

推荐采用恒等映射与软阈值输出之间的可学习插值：

```text
y = x + γ × (Shrink(x, τ) - x)
```

其中 `γ` 为限制在 `(0, 1)` 内的可学习通道插值比例，建议初始化为：

```text
γ = 1e-3
```

当 `γ` 接近 0 时，模块近似恒等映射；当 `γ` 接近 1 时，模块接近完整软阈值输出。较小初值使模型训练开始时近似原始 `LSK-GFNet-MAML`，随后逐渐学习去噪强度，可降低二阶 MAML 训练的不稳定性。

### 4.4.5 推荐伪代码

```python
class AdaptiveResidualShrinkage(nn.Module):
    def __init__(self, channels, reduction=4, layer_scale_init=1e-3):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.threshold_net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        blend_logit = torch.logit(
            torch.full((channels,), layer_scale_init)
        )
        self.blend_logit = nn.Parameter(blend_logit)

    def forward(self, x):
        magnitude = x.abs()
        channel_scale = self.pool(magnitude)
        threshold = self.threshold_net(channel_scale) * channel_scale
        shrunk = torch.sign(x) * F.relu(magnitude - threshold)
        gamma = torch.sigmoid(self.blend_logit).view(1, -1, 1, 1)
        return x + gamma * (shrunk - x)
```

### 4.4.6 本实验相对来源文献的改造

| 来源思想 | 本实验改造 |
|---|---|
| 使用软阈值过滤噪声响应 | 保留 |
| 使用网络自适应生成阈值 | 保留 |
| 与非局部模块或双注意力联合 | 不采用，避免继续堆叠注意力 |
| 完整残差收缩网络 | 简化为单个即插即用模块 |
| 普通监督或度量元学习 | 嵌入二阶 MAML 特征提取器 |
| 直接替换原始特征 | 改为小尺度残差接入，提高稳定性 |

因此，ARSM 是基于已有软阈值思想进行的项目适配模块，不应声称软阈值函数本身由本研究首次提出。

---

## 4.5 MAML 元学习模块

### 作用

MAML 通过内外循环优化，学习适合不同故障任务快速适应的初始化参数。

内循环：

```text
θ'_i = θ - α∇θ Lsupport_i(θ)
```

外循环：

```text
θ = θ - β∇θ Σ Lquery_i(θ'_i)
```

### 文献来源

轴承故障诊断直接相关来源：

```text
Few-Shot Bearing Fault Diagnosis Based on Model-Agnostic Meta-Learning
```

本地论文：

```text
C:\Users\JieZhang\Desktop\文献\
Few-Shot_Bearing_Fault_Diagnosis_Based_on_Model-Agnostic_Meta-Learning.pdf
```

跨工况信息增强参考：

```text
Lin et al., 2024
An information fusion-based meta transfer learning method for
few-shot fault diagnosis under varying operating conditions
Mechanical Systems and Signal Processing
```

本地论文：

```text
C:\Users\JieZhang\Desktop\文献\
An information fusion-based meta transfer learning method for.pdf
```

### 本实验改动

MAML 的 episode 采样、内循环学习率、外循环学习率和损失函数均不修改，以确保实验只验证 ARSM 的贡献。

---

## 5. 实现方案

## 5.1 修改文件

| 文件 | 修改内容 |
|---|---|
| `IFMAML-master/maml_model.py` | 新增 `AdaptiveResidualShrinkage`；在 `LSKLiteBackbone` 中增加可配置的去噪模块 |
| `IFMAML-master/config_pu.py` | 增加 ARSM 开关和超参数 |
| `IFMAML-master/maml_train_pic.py` | 读取 ARSM 配置并在模型名称中记录模块状态 |
| `实验结果记录only.md` | 增加 ARSM 消融实验结果 |

不需要修改：

```text
pu_preprocess.py
pu_dataset.py
l2l_shim.py
```

## 5.2 建议配置

```python
# 自适应残差收缩模块
'denoise_module': 'arsm',
'arsm_reduction': 4,
'arsm_layer_scale_init': 1e-3,
'arsm_position': 'after_gfnet',
```

关闭模块时：

```python
'denoise_module': 'none',
```

## 5.3 推荐接入位置

在 `LSKLiteBackbone` 中：

```python
def forward(self, x):
    x = self.stem(x)
    x = self.stages(x)
    x = self.freq_enhance(x)
    x = self.denoise(x)
    x = self.pool(x).flatten(1)
    return x
```

本实验中不再启用：

```python
'attention_module': 'none'
```

---

## 6. 实验设置

### 6.1 数据集

```text
Paderborn University Bearing Dataset
```

### 6.2 任务设置

| 参数 | 设置 |
|---|---:|
| 故障类别池 | 32 类 |
| Episode | 5-way |
| 主要任务 | 1-shot |
| 补充任务 | 5-shot |
| 每类 Query | 15 |
| 测试 Episode | 1000 |
| 目标场景 | 优先 S4 |

### 6.3 训练参数

| 参数 | 设置 |
|---|---:|
| Epoch | 500 |
| Meta batch size | 16 |
| Inner learning rate | 0.05 |
| Outer learning rate | 0.005 |
| 1-shot adaptation steps | 3 |
| Test adaptation steps | 10 |
| 优化器 | Adam |
| 损失函数 | CrossEntropyLoss |

### 6.4 随机种子

正式结论至少使用：

```text
seed = 3, 24, 38
```

建议论文最终补充至 5 个随机种子，并报告：

```text
Mean Accuracy ± Standard Deviation
95% Confidence Interval
```

---

## 7. 实验分组

## 7.1 必做主消融

| 编号 | 模型 | LSK | GFNet | ARSM | 目的 |
|---|---|---:|---:|---:|---|
| B0 | `STFT-CNN4-MAML` | 否 | 否 | 否 | 原始基线 |
| B1 | `STFT-LSK-MAML` | 是 | 否 | 否 | 验证大核选择骨干 |
| B2 | `STFT-LSK-GFNet-MAML` | 是 | 是 | 否 | 当前最佳基线 |
| A1 | `STFT-LSK-ARSM-MAML` | 是 | 否 | 是 | 单独验证软阈值去噪 |
| P1 | `STFT-LSK-GFNet-ARSM-MAML` | 是 | 是 | 是 | 最终候选模型 |

最关键比较：

```text
P1 对比 B2
```

只有 P1 在多个随机种子上稳定优于 B2，才能说明 ARSM 有效。

## 7.2 ARSM 结构消融

| 编号 | 阈值方式 | 残差方式 | 目的 |
|---|---|---|---|
| D0 | 无阈值 | 无 ARSM | 对照 |
| D1 | 固定全局阈值 | 残差 | 验证固定阈值 |
| D2 | 自适应单阈值 | 残差 | 验证样本自适应 |
| D3 | 自适应通道阈值 | 残差 | 推荐完整版本 |
| D4 | 自适应通道阈值 | 非残差直接输出 | 验证残差连接必要性 |

如果算力有限，只运行：

```text
D0、D3、D4
```

## 7.3 插入位置消融

| 编号 | ARSM 位置 | 预期特点 |
|---|---|---|
| P-A | LSK 后、GFNet 前 | 先去噪，再执行频率筛选 |
| P-B | GFNet 后、GAP 前 | 对高层频率增强特征去噪，推荐 |

第一轮只运行 P-B。只有提升不明显时，再测试 P-A。

## 7.4 超参数消融

### Reduction ratio

```text
r ∈ {2, 4, 8}
```

推荐默认：

```text
r = 4
```

### Layer scale 初值

```text
γ0 ∈ {1e-2, 1e-3, 1e-4}
```

推荐默认：

```text
γ0 = 1e-3
```

不要同时大规模搜索两个超参数。先固定 `r=4` 搜索 `γ0`，再根据最优 `γ0` 比较不同 `r`。

---

## 8. 鲁棒性实验

软阈值模块的核心论点是抗噪，因此仅比较干净数据准确率不足以支撑论文贡献。

### 8.1 测试噪声设置

在目标域测试信号生成 STFT 前加入高斯白噪声：

```text
SNR ∈ {10, 6, 2, 0, -2} dB
```

### 8.2 对比模型

```text
STFT-LSK-GFNet-MAML
STFT-LSK-GFNet-ARSM-MAML
```

### 8.3 预期验证

关注以下指标：

- 不同 SNR 下的 Meta-test Accuracy；
- 相对干净数据的准确率下降幅度；
- 各随机种子的标准差；
- 混淆矩阵中易混淆类别的改善情况。

如果 ARSM 仅提高干净数据准确率，却不能降低噪声下的性能衰减，则“自适应去噪”的论证不充分。

---

## 9. 评价指标

### 9.1 分类指标

- Accuracy；
- Precision；
- Recall；
- Macro-F1；
- 每类 Recall；
- 混淆矩阵。

### 9.2 稳定性指标

- 多随机种子均值；
- 标准差；
- 95% 置信区间；
- 最佳验证 epoch；
- 收敛曲线波动。

### 9.3 模型复杂度

- 参数量；
- FLOPs；
- 单个 episode 推理时间；
- MAML 单 epoch 训练时间；
- 相对 B2 增加的参数比例。

ARSM 应当保持轻量。建议新增参数量不超过当前模型的 5%。

---

## 10. 可视化与解释性实验

### 10.1 特征图对比

选择同一目标域样本，分别展示：

```text
GFNet 输出特征
ARSM 阈值
ARSM 收缩后特征
```

观察弱响应是否被抑制、主要故障响应是否被保留。

### 10.2 阈值分布

绘制不同故障类别和不同工况下的通道阈值分布，验证 ARSM 是否能根据输入动态调整。

### 10.3 t-SNE

对比：

```text
LSK-GFNet-MAML
LSK-GFNet-ARSM-MAML
```

观察加入 ARSM 后是否表现为：

- 类内距离减小；
- 类间距离增大；
- S4 中易混淆故障类别分离更清晰。

---

## 11. 结果记录模板

### 11.1 主实验

| 模型 | Seed | S1 | S2 | S3 | S4 | 参数量 | 单 Epoch 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN4-MAML |  |  |  |  |  |  |  |
| LSK-MAML |  |  |  |  |  |  |  |
| LSK-GFNet-MAML |  |  |  |  |  |  |  |
| LSK-ARSM-MAML |  |  |  |  |  |  |  |
| LSK-GFNet-ARSM-MAML |  |  |  |  |  |  |  |

### 11.2 多随机种子统计

| 模型 | S4 Mean Accuracy | Std | 95% CI | Macro-F1 |
|---|---:|---:|---:|---:|
| LSK-GFNet-MAML |  |  |  |  |
| LSK-GFNet-ARSM-MAML |  |  |  |  |

### 11.3 抗噪结果

| 模型 | Clean | 10 dB | 6 dB | 2 dB | 0 dB | -2 dB |
|---|---:|---:|---:|---:|---:|---:|
| LSK-GFNet-MAML |  |  |  |  |  |  |
| LSK-GFNet-ARSM-MAML |  |  |  |  |  |  |

---

## 12. 判定标准

ARSM 可保留为最终论文模块，需要至少满足以下条件中的三项：

1. S4 平均准确率相对 B2 提升不低于 0.8 个百分点；
2. 至少 3 个随机种子中均未出现明显退化；
3. 标准差低于或不高于 B2；
4. 在 0 dB 或 -2 dB 噪声下具有明显优势；
5. 新增参数量不超过 B2 的 5%；
6. 混淆矩阵或 t-SNE 能显示易混淆类别得到改善。

如果 ARSM 在干净数据上的提升低于 0.5 个百分点，但在低信噪比下有稳定优势，仍可将其定位为“鲁棒性增强模块”，而不是单纯的精度增强模块。

---

## 13. 论文创新点建议

论文中可将贡献表述为：

1. 构建轻量化 LSK-GFNet 特征提取器，分别从大感受野空间结构和全局频率响应两个角度提取 STFT 故障特征；
2. 针对跨工况少样本任务中噪声和弱相关响应对快速适应的干扰，设计适用于二阶 MAML 的自适应残差收缩模块；
3. 利用样本与通道相关的动态软阈值执行特征收缩，并通过小尺度残差连接降低去噪模块对元学习内循环的扰动；
4. 在 PU 多工况 5-way 1-shot/5-shot 任务及不同 SNR 条件下验证模型的准确性、稳定性和抗噪性能。

不建议使用以下表述：

```text
“首次提出软阈值去噪”
“首次提出残差收缩网络”
“首次将 LSK 或 GFNet 用于深度学习”
```

更准确的表述是：

```text
“面向 STFT 跨工况少样本诊断设计并轻量化适配”
“提出适合 MAML 双层优化的残差收缩接入策略”
“实现空间大核、全局频率与自适应去噪的协同建模”
```

---

## 14. 推荐执行顺序

```text
步骤 1：实现 ARSM 和配置开关
步骤 2：运行 quick test，检查前向、反向和模型保存
步骤 3：在 S4、seed=24 上运行 LSK-ARSM-MAML
步骤 4：在 S4、seed=24 上运行 LSK-GFNet-ARSM-MAML
步骤 5：与当前 92.69% 的 B2 结果比较
步骤 6：若有效，补充 seed=3、38
步骤 7：执行 0 dB、-2 dB 抗噪实验
步骤 8：最后补充 S1–S3 和 5-shot 实验
```

第一轮不进行大规模超参数搜索，也不加入 EMA 或 GCNet。先回答最核心的问题：

```text
自适应软阈值去噪能否稳定改善当前 LSK-GFNet-MAML
在困难跨工况场景 S4 中的少样本适应性能？
```

---

## 15. 主要参考文献与本地文件

1. Li, Y. et al. *Large Selective Kernel Network for Remote Sensing Object Detection*. ICCV, 2023.  
   本地文件：`C:\Users\JieZhang\Desktop\new-ifmaml\模块1LSK\Large Selective Kernel Network for Remote Sensing Object Detection.pdf`

2. Rao, Y. et al. *Global Filter Networks for Image Classification*. NeurIPS, 2021.  
   本地文件：`C:\Users\JieZhang\Desktop\new-ifmaml\模块2gfnet\2107.00645v2.pdf`

3. Gao et al. *Metric-Based Meta-Learning Relation Network for Cross-Domain Few-Shot Bearing Fault Diagnosis*, 2025.  
   本地文件：`C:\Users\JieZhang\Desktop\文献\Gao 等 - 2025 - Metric-Based Meta-Learning Relation Network for Cross-Domain Few-Shot Bearing Fault Diagnosis.pdf`

4. Huang et al. *Cross-scenarios few-shot fault diagnosis for rolling bearings via inter-domain similarity-guided meta-learning with dual-attention multiscale feature denoising network*.  
   本地文件：`C:\Users\JieZhang\Desktop\文献\Cross-scenarios few-shot fault diagnosis for rolling bearings via.pdf`

5. *Few-Shot Bearing Fault Diagnosis Based on Model-Agnostic Meta-Learning*.  
   本地文件：`C:\Users\JieZhang\Desktop\文献\Few-Shot_Bearing_Fault_Diagnosis_Based_on_Model-Agnostic_Meta-Learning.pdf`

6. Lin, C. et al. *An information fusion-based meta transfer learning method for few-shot fault diagnosis under varying operating conditions*. Mechanical Systems and Signal Processing, 2024.  
   本地文件：`C:\Users\JieZhang\Desktop\文献\An information fusion-based meta transfer learning method for.pdf`

7. Than et al. *MixMamba-Fewshot: Mamba and Attention Mixer-Based Method With Few-Shot Learning for Bearing Fault Diagnosis*.  
   本地文件：`C:\Users\JieZhang\Desktop\文献\Than 等 - Mixmamba-fewshot mamba and attention mixer-based method with few-shot learning for bearing fault di.pdf`
