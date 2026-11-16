# 原文与接入依据

原文：Zotero/storage/FLYYE792/Mu 等 - 2025 - Adaptive model-agnostic meta-learning network for cross-machine fault diagnosis with limited samples.pdf。

DOI：10.1016/j.engappai.2024.109748；Engineering Applications of Artificial Intelligence 141 (2025), 109748。核对来源为用户提供的完整PDF，且已目视检查第4、5、7页图与公式。

## 逐项映射

| 依据（PDF页序） | 实现 | 取舍 |
|---|---|---|
| p4 Fig.3/4，p5 §3.1 | `model.py::SBCA/AMAMLNetwork` | 四块 Conv-BN-ReLU-MaxPool-SBCA，核5/5/3/3，每层64通道，注意力中间16通道；Fig.4明确有BN+ReLU |
| p4 式7–12 | 两个方向的均值池化，共享1×1降维，分支1×1+sigmoid，乘回输入 | 拼接H/W方向，等价方向轴转置；支持非方形特征 |
| p5 式13–15 | `adaptive_loss` | query预测概率熵总和 λ，最大非目标概率与目标概率绝对差的负总和 η；按印刷公式 ε=a·sigmoid(g1η−g2λ+b) |
| p5 式18/19、p7 式27 | 默认 `outer_loss=eq27` | 式27/18为平滑标签CE；式19多出一项乘积，提供 `eq19` 选项解释为 hard+uniform+hard×uniform，两者不称等价 |
| p7 式25–26，Table1 | `adapt` | support CE梯度；δ=ρ θ²g / ||abs(θ)g||₂；在θ+δ处计算梯度，更新从θ出发 |
| p13–14 §4.4 | `MethodConfig` | ε限制0.2–0.3；1-shot半径1e-3、3/5-shot半径1e-4是依据该论文Task5敏感性结果的初始选择，并非本项目已验证最优值 |
| p9 §4.1 | `raw_reshape.py` | 每样本1444点按时间顺序逐行重排为38×38；复用项目窗口起点，取前1444点，不STFT、不归一化、不插值 |

## 必须披露的未决细节

1. 原文式15的符号随熵变化方向与附近文字解释不一致。本实现优先采用印刷公式，不擅自改符号；a=0.3、b=3、g1=g2=0.01是**工程工作值，非论文明确数值**，未调参。额外clip确保0.2–0.3区间。这是可运行重构的主要不确定性。
2. 原文式19与后续式27不一致。默认采用最终优化流程式27；式19选项只是按印刷式的乘积解释，不代表已得到作者确认。
3. 原文未明确ε是否detach，默认保留其梯度；未明确扰动是否参与更高阶导数，默认δ停止梯度，但保留θ+δ位置支持梯度的二阶MAML路径。`differentiate_perturbation=true`可完整求导，已有数值有限差分验证；不把默认实现称为完整扰动导数。
4. 论文对范数写“spectral norm”，对不同层形状参数如何拼接未明确；本实现采用全部参数展平后的统一L2范数，并加1e-12下限。不会分别对每层归一化。
5. same padding、Flatten+线性分类头、PyTorch默认初始化为适配本项目的显式实现选择，原图未完整列出。当前输入38×38，四层池化19→9→4→2，特征256维。重排不改变幅值（仅转float32），原文未明确的归一化不自行添加。
6. 测试阶段只说通过support内优化适应，未明确是否继续AMOS。默认测试/验证沿用AMOS；可在方法JSON设 `evaluation_amos=false` 对照标准GD适应。query标签永不用于内循环。
7. 本项目使用inner/outer=0.05/0.005及现有scheduler；原文§4.4给0.1/0.0014。此次遵照“项目任务、仅替换方法”采用共同训练配置。
8. SBCA的BN采用每批统计，不积累可跨episode污染的running buffers；query间存在无标签批统计交互。独立训练器修正旧训练器选模一轮错位，验证与保存同一组参数；严格比较需披露这一训练器差异。

可用 `--method-config 路径.json` 覆盖MethodConfig字段；最终值自动保存。未使用测试准确率决定上述选项。未获得或运行作者官方源码，不能承诺算法每个未说明细节与作者实现相同。

## 当前项目核验

no_stft分支：从 `run_table_row.py::TARGETS/configure` 读取任务和episode预算，用户指定默认上限100轮。`RawReshapeDataset`复用 `RawSignalDataset` 的MAT解析与取窗起点，不调用其PNG核验函数，重写窗口输出为38×38。PU目标40窗按原seed拆20/20；CWRU目标保留原信号前半验证/后半测试，各20窗，源每工况40窗。PU用4096、CWRU用3072生成原窗口起点，实际每窗只使用前1444点，以保持项目样本身份。这是接入约定而非原文明确参数。PU/CWRU各四任务均已通过检查，每类120训练/20验证/20测试；不再依赖图片角色缓存。

历史：此前STFT版本、500轮预算、仅CWRU C3图片缓存可用的描述已被原始输入修改取代；历史输出和源码快照保留。其他ALC/AMOS取舍不变。


## 双输入接口（当前）

`--input no_stft`默认采用上述原始重排，`--input stft`读取现有64×64图片，均100轮上限。STFT为项目输入适配，并非原文信号重排；网络末端特征1024维，no_stft为256维。两支共享任务与优化设置，输入表示及分类头维度不同，不把该比较称为只改变一个完全等价操作的严格消融。输出按raw38/stft64隔离，测试自动恢复checkpoint的输入模式。STFT的CWRU仍需角色匹配缓存，默认只有C3可用。

双输入已通过8项检查和PU/CWRU各两种输入的训练/保存/回读验证；按用户要求删除验证脚本和中间结果，仅保留该结论与交付文件。
