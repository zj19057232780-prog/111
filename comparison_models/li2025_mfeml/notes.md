# MFEML原文映射、参数与实现边界

来源：Li, Xianze等，2025，*A meta-learning method based on meta-feature enhancement for bearing fault identification under few-sample conditions*，Mechanical Systems and Signal Processing 226, 112370。[出版社页面](https://www.sciencedirect.com/science/article/pii/S0888327025000718)，[DOI](https://doi.org/10.1016/j.ymssp.2025.112370)。核对用户本地Zotero附件 `ZNJQDKA7/1-s2.0-S0888327025000718-main.pdf`，下文为PDF页码。已目视复核第5页公式、第6页图2细节、第8页表2。本实现由论文重构，不声称为作者源码或原文数值严格复现。

## 1. 按论文采用的设置

| 原文位置 | 设置 | 当前实现 |
|---|---|---|
| 第4–5页§3.1，第8页表2 | 2000点，自适应补零成正方形 | no_stft输出原始 `[B,1,2000]`，模型内追加25个零，逐行变为 `[B,1,45,45]`；不截掉真实点、不插值、不计算STFT |
| 第6页图2，第8页表2 | 4个模块、64通道、3×3卷积 | 每块Conv→BN→ReLU→MaxPool→MFE；卷积stride1/pad1，池化2/stride2 |
| 第5页式3/4 | SimAM空间能量权重 | μ取各通道空间均值，方差按M−1分母；`sigmoid((X−μ)^2/(4*(var+λ))+0.5)` |
| 第5页式5，第6页图2 | GAP→1D Conv→Sigmoid通道注意力 | 沿通道维度卷积，输出 `[B,C,1,1]` 权重，乘到SimAM增强特征上 |
| 第8页表2 | 全连接维度256 | 原始45→22→11→5→2，得到64×2×2=256，Flatten→Linear(256,N-way) |
| 第6页式6–8、§3.4 | MAML、交叉熵、Adam | 支撑集CE内循环梯度更新；query CE外循环Adam；保留完整二阶梯度 |
| 第8页表2 | 每批25任务 | 默认meta_batch_size=25，每批损失取均值；逐任务反传、累加元梯度后一次outer更新 |
| 第8页表2 | Epoch=1000 | 用户指定默认最多500次元更新；这里epoch表示一次episodic元批次更新，不是遍历整个数据集 |
| 第8页表2 | Train step=10、Test step=10 | 默认训练/验证/测试均10次support适应 |
| 第8页表2 | Meta-learning rate=0.01；Base-learning rate=0.1 | **outer_lr=0.01，inner_lr=0.1**，分别更新共享初值/任务副本，不能颠倒 |

按用户后续要求，本模型默认最多500次元更新，并启用100轮无提升早停。这两项属于项目训练预算设置；论文表2为1000轮，未说明patience。需要论文训练预算可显式 `--epochs 1000 --patience 0`，结果须单独标注预算与停止规则；其他模型默认值不变。

## 2. 原文未写清楚的地方与固定选择

1. **激活、stride/padding、初始化**：正文提到激活层但未指定类型；图2以σ示意。采用ReLU、Conv stride1/pad1、MaxPool2/stride2、PyTorch默认初始化，Conv保留bias，BN带affine。这使2000点的末端特征数自然等于表2的256。
2. **SimAM的λ及方差分母**：未明确给出，固定λ=1e-4、空间方差分母M−1；小于2个位置时分母截为1以免除零。`MethodConfig.simam_lambda`可显式更改，未进行测试集调参。
3. **通道卷积核**：表2的3×3是二维卷积核，不能直接当作一维ECA参数。ECA核固定3、padding1、bias=False，4个块各有独立的可训练通道卷积。
4. **并行/串行表述**：图2明确画出同一输入分到GAP/C1D与SimAM的两条路径，默认采用parallel：`SimAM(X) * sigmoid(Conv1d(GAP(X)))`。式4将增强结果重新写作X，紧接式5的X也可读成串行，故提供 `attention_fusion=sequential` 备选，但默认严格按图2连线。图中未标合并运算符，采用注意力加权的逐元素相乘，不自行增加残差或拼接。
5. **FC维度**：表2“Dimension=256”没有说明是否隐藏层；结合45×45经四次池化后恰为64×2×2及图2的Flatten→FC，解释为分类器输入256，不额外增加隐藏FC层。
6. **Adam作用范围**：正文说使用Adam，式6/图2则明确写θ′=θ−α∇L。实现内层按式6梯度下降、外层Adam（betas0.9/0.999、eps1e-8、weight_decay0），未把内层默认为未说明的可微Adam。
7. **损失归约与数值稳定**：每个任务CE对query求均值，再对25任务取均值，避免梯度量级随batch改变。全局outer梯度范数裁剪默认1.0，属于补充稳定设置，可 `--grad-clip 0` 关闭；裁剪前范数写入history.csv。
8. **调度/早停**：论文未给学习率调度或patience，默认恒定论文学习率，不加cosine；按用户要求默认patience=100：每轮outer更新后评估目标验证集，准确率严格提升才重置等待计数，持平也计入未提升；连续100轮无提升则停止，随后加载验证最佳权重测试。`--patience 0`关闭此条件。不执行高准确率确认早停。此规则属于项目补充，并非论文参数。
9. **BN与任务状态**：沿用项目episode批统计、`track_running_stats=False`，避免跨任务共享running buffers。query批统计参与评估，属于transductive batch-statistics口径；query标签仅用于计分，不用于support适应。所有模型参数（含ECA）均内外循环更新。

这些选择是可审计的实现补充，不应被转述为作者明确给出的细节。`--method-config`支持JSON覆盖MethodConfig字段，每次运行完整保存实际参数。

## 3. 原始/STFT两接口与项目协议

- `--input no_stft`（同义 `--input nostft`）：读取现有MAT，保留原项目的随机采样起点及4096/3072点采样范围，每窗取前2000点，转float32后交给模型。原文没有写输入幅值归一化，本实现不额外归一化，也不做min-max或z-score；自适应补零是模型前端，无离线文件生成。
- `--input stft`：复用项目原图片加载器、灰度64×64和已有图片归一化。为了保持表2的256维分类头，在四次池化后的4×4特征上增加无参数AdaptiveAvgPool2d(2×2)；原始分支本来就是2×2，该层不改变值。这是STFT接口适配，不是原文额外模块。
- 两种输入结果分目录、数据签名分开记录。即便网络state_dict形状相同，也禁止把一种输入的权重当另一种输入的实验结果；独立测试按保存配置恢复，不允许命令行悄悄换输入。
- PU9类池、CWRU10类池随机5-way，1/3/5-shot、15-query，seed3/24/38，多源留一目标，项目每类120源域训练/20目标验证/20目标测试，1000个测试episode保持不变。validation默认每轮25个任务、按测试适应步数选checkpoint，验证的是outer更新后实际保存的参数。
- 原文CWRU把多因素组合成110类/每类25样本，另用实验室数据；这些类别和划分没有替换你的项目协议。CWRU仍为项目48kHz驱动端，不改变到风扇端、不重采样。
- CWRU原始目标信号先按前/后半分出validation/test，再采窗；前2000点始终在所属分段内。PU仍是项目窗口级目标划分，可能存在跨val/test重叠窗口；本轮未擅自改成轴承级或时间隔离划分。保留采样起点不等于原始2000点与STFT整窗的信息量完全相同。
- CWRU的STFT接口受当前图片缓存工况角色限制：应先 `--action check`，缺失任务可用已有匹配缓存的 `--data-root` 或任务路径映射 `--data-map`；不自动重做预处理。当前实际可用组合以README验证记录为准。

## 4. 结果、验证与清理

每次训练独立保存实际配置、方法配置、architecture、源码哈希与快照、数据签名、最佳checkpoint及其哈希、history、episode清单、逐query预测、结果。原始输入还保存窗口起点、长度与原始值哈希。Macro-F1由累积全局混淆矩阵计算；完整seed3/24/38才汇总mean±sample SD。

本轮只做公式/梯度/真实数据运行检查，不把短测准确率用作模型效果结论，不进行测试集超参数选择。验证完成摘要见README。按用户要求清理过程计划、PDF提取/截图、临时测试脚本与冒烟输出；保留交付源码、说明和用户已有正式结果。
