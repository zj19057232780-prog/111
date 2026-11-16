# TUBR-ML 原文映射与复现边界

来源：Wang 等，2025，Task-adaptive unbiased regularization meta-learning for few-shot cross-domain fault diagnosis，Engineering Applications of Artificial Intelligence 144, 110200。DOI：[10.1016/j.engappai.2025.110200](https://doi.org/10.1016/j.engappai.2025.110200)。核对用户本地 Zotero PDF（6KAL4XS9），页码为PDF页码；已目视复核公式、图4和表6/7。本实现由论文重构，不声称来自作者源码或复现原文跑分。

## 输入与数据协议

- 按用户最新要求，`no_stft` 使用第5页的连续7056点、归一化、逐行84×84灰度重排。表6写3通道，而正文说灰度：本实现将相同灰度复制为3通道以匹配表6首层；不插值、不填充、不把STFT逆变换为信号。
- 原文没有定义归一化公式、量化规则或窗口步长。本实现明确采用**每窗独立min-max到[0,1]，常量窗归零，float32、不量化**；这属于可复核的实现解释，不是已证实的作者细节。
- `stft` 直接使用项目现有单通道64×64图片、原加载器及归一化。SDFE首层相应为1通道，其余结构和方法一致；两个输入模式首层权重不同，不能直接互换。
- 原始分支复用项目MAT解析、类别、工况、多源留一任务和样本数量。窗口长度改为7056，沿用项目随机采样函数/种子/调用顺序，但合法起点范围改变，**不声称与旧4096/3072点STFT窗口逐窗对应**。不重建或覆盖离线数据。
- CWRU先将目标信号划成前半validation/后半test，再在各自分段内采7056点，逐窗检查边界。仍使用项目48kHz DE，而原文CWRU为12kHz；不进行重采样。
- PU仍沿用项目窗口级随机validation/test划分，来自同一轴承的重叠窗口可能跨两侧，这是继承的评测风险，不能表述为轴承隔离或时间不重叠的测试。长窗口也改变重叠程度。未擅自更换项目划分。
- 主协议：PU9类池/CWRU10类池随机5-way，1/3/5-shot，15-query；seed3/24/38；16训练任务/轮、16目标验证任务/轮；训练内循环3步、测试10步；测试1000episode；默认100epoch，patience100，沿用高准确率确认早停。目标validation参与选模，不属于完全不接触目标域的domain generalization协议。
- 原文为4任务、内外lr 0.01/0.001，测试20步且描述两阶段Adam；本实现保留项目任务数量与3/10步预算，采用论文内外学习率0.01/0.001（可用CLI覆盖），外层Adam、内循环按式5梯度下降；外层沿用项目cosine至5e-5。用户已授权采用合理的模型参数。不能把此组合称为论文完整实验设置。

## 模型与公式

| 原文位置 | 实现 | 说明 |
|---|---|---|
| 第4–5页§3.3、图4；第8页表6 | `SDFE` | 4×64通道，3×3/stride1/pad1；Conv→BN→MaxPool2→LeakyReLU→DropBlock，GAP→64维→N-way FC。按图4确定次序，正文列举次序不完全一致。LeakyReLU斜率原文未给，采用0.01。 |
| 第5页式14、表6/7 | `StructuredDiscard` | keep_prob=0.9、block_size=3，每层池化后使用，含第4层。合法区域Bernoulli块起点，gamma=(1-p)/c²·d²/(d-c+1)²；扩张成块、取并集、按保留比例缩放。评估关闭DropBlock。 |
| 第3页式5 | `adapt` | θ′=θ−α·(∇CE_support+ωθ)，每步重算ω，保留二阶梯度。attenuator只做outer更新，不用support直接更新attenuator权重。 |
| 第3页§3.1、图3a；第7页表5 | `TUBRNetwork.attenuation` | 18→18→ReLU→18→Sigmoid；按正文聚合贡献求和。 |
| 第4页式7/8 | `entropy`、`turl_loss` | 每个query样本的自然对数熵，样本取均值；TUL=H(初始query输出)−H(适应后query输出)，保留符号，无平方/绝对值/KL替换。 |
| 第4页式10/11 | `turl_loss` | Poly-1=CE+τ(1−p_true)，TURL=Poly-1+η·TUL；τ=2。 |
| 第7页§4.2 | `inner_rate` | 前10次inner update不衰减，第11次起每次乘0.9。项目默认只有3/10步，因此默认未进入衰减阶段。 |

## 原文不完整或矛盾处与明确选择

1. **η冲突**：第7页正文0.05，第8页表7为0.005。默认按表7 `eta=0.005`；交付 `method_eta_text.json` 可选正文0.05，不声称已通过调参判定哪一个正确。
2. **18维梯度描述符**：正文将无参数池化/DropBlock也计入18层，却没有说明它们的参数梯度如何产生；式6还省略了MLP。采用4块×(Conv权重/偏置、BN权重/偏置)+FC权重/偏置的18个参数张量，逐张量取有符号均值后进入表5 MLP。这是实现假设；参数顺序写入运行的 `architecture.json`。
3. **ω聚合**：按正文和式6的求和方向，默认 `attenuation_reduce=sum`，每个Sigmoid贡献∈[0,1]，**聚合ω∈[0,18]**。可显式切换mean，∈[0,1]；两者是不同实现，结果须分开标注，不能当作同一精确复现。
4. **先验是否detach未交代**：默认 `detach_prior=false`，两分支按式8/11共同求导；可显式切换。先验是每个episode当前初始化θ在同一query上的输出，不是永久冻结的随机网络。只在源域meta-training计算TURL；validation/test仅用support标签适应，不用query标签或query熵更新参数。
5. **熵差的文字解释与数学直觉有冲突**：保留式8的有符号差，不自行修正为常见散度。原文式3/13遗漏θ−，采用常规Adam梯度下降，而不是将梯度赋给参数。
6. **BN与丢弃随机性**：BN沿用项目episode批统计（无跨任务running buffer）；评估仍使用query批统计，是transductive batch-statistics评估。DropBlock仅meta-training打开，先验/适应后独立采样mask；原文没有指定共享mask或推理阶段DropBlock细节。
7. **精确checkpoint选择**：先outer更新，再验证，再保存同一参数；沿用前一个独立AMAML训练器。项目旧主训练器先验证再更新后保存存在一步错位，未修改主训练器，对比时须披露。
8. **学习率与衰减的量级**：默认求和ω初始化约9，内循环包含(1−αω)θ。若机械沿用项目α=0.05，每步约乘0.55，10步后幅度约0.0025，容易出现过度衰减；因此固定采用论文α=0.01，每步约0.91。此决定依据更新式与论文设置，不依据目标测试准确率选优。正式实验前应固定该配置，不能再按test表现调整sum/mean或η。

## 交付与结果规则

独立目录不改AMAML或主模型。结果同时保存输入标识、方法参数、源码快照/哈希、数据签名、checkpoint哈希、episode清单、逐query预测与聚合Macro-F1。只有完整三seed才汇总mean±sample SD；冒烟标记SMOKE_ONLY。测试必须通过源码、数据和权重哈希检查。测试期间临时脚本不计入源码哈希，删除它们不会破坏本模型回读。

验证结论见README；冒烟脚本/产物、PDF提取及截图和计划均按用户要求清理。正式训练由用户运行命令启动，本轮不生成可填论文表的实验结果。
