# 华中科技大学 HUSTbearing 独立跨工况实验

> 2026-11-10更新：PU/CWRU及华中HUST除TUBR-ML外所有模型均可使用3-shot，包含A0–A3和cnn4/lsk/lsk_gc/full。将适用命令的 --shots 1 改为 --shots 3 即可，保留15-query。HUST --models all --shots 3自动排除TUBR-ML；原有STFT/nostft限制不变。下文旧“只支持1/5-shot”和“A0–A3仅1-shot”描述以本条为准；TUBR-ML既有1/5-shot限制保持不变。

本入口仅用于 `IFMAML-master/HUST-bearing-dataset` 中的 ER-16K 九状态数据；不是归档的河内理工大学 HUST。

## 核查与协议

依据随附 `Description file（描述文件）/Readme for HUSTbearing dataset.pdf` 第4–6页：采样率25600Hz，每记录262144点。实际`.xls`是采集仪导出的TSV文本，不是二进制Excel工作簿。解析Data标记及Legend，校验记录长度、时间戳、有限数值和标题，不使用Speed列。默认X为振动通道（第三列），可指定Y/Z；同一实验组固定通道，不按结果择优。

| 标签 | 文件前缀 |
|---|---|
| 正常 | H |
| 中度/重度内圈 | 0.5X_I / I |
| 中度/重度外圈 | 0.5X_O / O |
| 中度/重度滚动体 | 0.5X_B / B |
| 中度/重度复合 | 0.5X_C / C |

复合为内圈+外圈。PDF第6页将O_80Hz例子误写为ball，与编码表矛盾；代码按其明确编码表将O映射外圈。只选20/40/60/80Hz共36记录，其余工况不进入本协议。

| 任务 | 源域训练Hz | 目标验证/测试Hz |
|---|---|---|
| H1 | 20 | 40 |
| H2 | 40 | 20 |
| H3 | 40 | 60 |
| H4 | 60 | 40 |
| H5 | 60 | 80 |
| H6 | 80 | 60 |

**按用户最新修改采用目标域验证**，协议 `HUST-Huazhong-9pool-5way-single-source-targetval-v2`。每记录按4096点连续不重叠切为64个窗口；目标窗口0–30验证，丢弃31，32–63测试（索引从0起），相隔4096点。每类源域64张、目标验证31张、目标测试32张。该协议是同一目标记录内时间隔离，不能表述为独立轴承实体划分，也不是完全不接触目标标签的域泛化。

调用原PU `compute_stft_log_image`：单通道64×64，窗256、重叠192、Hann、无boundary补齐，log10、5/95分位、gamma0.7等沿用PU。STFT内部帧重叠与4096点样本窗口重叠是两件事；样本之间不共享原始点。逐窗口归一化不使用跨池统计。

九类池随机5-way，1-shot/5-shot，15 query；500轮、meta-batch16、测试1000 episodes；seed3/24/38。其余学习率、内循环步数、二阶MAML及AGR复用现有配置。所有HUST模型（含完整SCG-MAML）默认patience100，可用 `--patience 0` 关闭；最多训练500轮（TL-WDCNN监督预训练上限100轮），用目标验证准确率选最佳；HUST启用更新后验证，使checkpoint对应所报验证分数。原PU默认执行顺序和协议不变。

## 目录与命令

```text
项目根目录/
  IFMAML-master/
    HUST-bearing-dataset/
      Description file（描述文件）/
      Raw data (原始数据)/H_20Hz.xls ...
    hust_huazhong_processed/X/20/healthy/w000.png ...
    hust_huazhong_processed/X/manifest.json
    hust_huazhong_results/formal/时间戳/stft或nostft/模型/H1/seed3/
```

在IFMAML-master目录执行，Windows建议使用 `C:/ANACONDA/envs/ifmaml2/python.exe`；Linux使用安装CUDA PyTorch的python。专用配置/预处理/适配/入口均以hust_huazhong命名。

```bash
# 解析预处理（默认X），也可由运行入口自动执行
python hust_huazhong_preprocess.py
# 六任务容量、数据与哈希检查
python run_hust_huazhong.py --tasks all --shots 5 --action check
# 小规模功能检查，不能填论文表
python run_hust_huazhong.py --tasks H1 --models all --seed 3 --action smoke --device cuda
# 九方法 × 六任务 × 三seed，STFT 1-shot实验
python -u run_hust_huazhong.py --tasks all --models all --seeds 3 24 38 --shots 1 --action train-test --device cuda
# 完整5-shot实验
python -u run_hust_huazhong.py --tasks all --models all --seeds 3 24 38 --shots 5 --action train-test --device cuda
# 单模型单seed的六任务
python -u run_hust_huazhong.py --tasks all --model full --seed 3 --shots 1 --action train-test --device cuda
# 任务子集；模型名称cnn4、lsk、lsk_gc、full
python -u run_hust_huazhong.py --tasks H1,H4 --model lsk_gc --seed 24 --shots 1 --action train-test --device cuda
# 只训练 / 对单个已有运行目录补测
python -u run_hust_huazhong.py --tasks H1 --model full --seed 3 --action train --device cuda
python -u run_hust_huazhong.py --action test --run-dir /实际运行目录/full/H1/seed3 --device cuda
```

可用 --raw-root 指向包含Raw data子目录的数据集根目录，--processed-root 指向该通道专用缓存目录，--output-root改变结果根目录。--channel Y/Z使用独立默认缓存。缓存校验失败不覆盖重建，另选新路径。默认action=check。不要在训练过程中修改源码。

结果含best checkpoint、训练历史、resolved_config、数据签名、代码快照和哈希、预测CSV、Accuracy及Macro-F1。每批生成runs.json、summary.json/csv/md和failures.json。各任务只在3/24/38齐全时计算mean±样本标准差；六任务先在每个seed内求均值，再对三seed均值计算mean±std。冒烟/缺失结果不会进入正式汇总。

原始数据不发布到GitHub，旧PU结果不重跑或改写。新数据集只有一种记录/类别/工况，时间隔离不能证明跨实体泛化。当前实现保留原MAML批内BN行为，比较模型时须使用一致协议。


## 全模型双输入扩展（2026-10-19）

新增ProtoNet、MatchingNet、RelationNet、MAML、TL-WDCNN、AMAML、TUBR-ML、MFEML。`--model`名称分别为protonet、matchingnet、relationnet、maml、tl_wdcnn、amaml、tubr_ml、mfeml；完整模型full（别名scg_maml）。`--input stft/nostft`，默认stft。

| 原生输入 | 现有实现复用 |
|---|---|
| 经典四方法/TL-WDCNN | 4096点，逐窗口z-score，[1,4096] |
| AMAML | 原窗口前1444点，无额外归一化，[1,38,38] |
| TUBR-ML | 连续7056点，逐窗口min-max，[3,84,84] |
| MFEML | 原窗口前2000点，无额外归一化，[1,2000]；模型内部补零 |
| SCG-MAML | 仅STFT，不支持nostft |

STFT均沿用64×64缓存；TL-WDCNN按现有适配展平为[1,4096]。三论文模型直接复用各自run.py训练/验证/测试，仅新增HUST数据分支。PU/CWRU分支和模型实现不改。

目标原始时间段固定为验证[0,126976)、测试[131072,262144)，中间4096点隔离。TUBR-ML长窗口不跨边界；每类源37、验证17、测试18。**保留15-query，仅TUBR-ML nostft拒绝5-shot**，其STFT仍支持1/5-shot。其余输入保持每类源64、验证31、测试32。

所有对比模型和cnn4/lsk/lsk_gc/full默认patience100，验证准确率无严格提升才累计；`--patience 0`关闭早停，`--patience 20`等正整数自定义等待轮数。TL-WDCNN默认监督预训练100轮，默认patience100可能不会提前触发，`--patience`可显式调整。测试始终使用最佳验证权重。

```bash
# 单模型单任务单seed，原生输入
python -u run_hust_huazhong.py --model amaml --tasks H1 --seed 3 --input nostft --shots 1 --action train-test --device cuda
# 单模型六任务三seed，STFT输入
python -u run_hust_huazhong.py --model mfeml --tasks all --seeds 3 24 38 --input stft --shots 5 --action train-test --device cuda
# 原四种结构配置（models all现在指九个对比方法）
python -u run_hust_huazhong.py --models cnn4 lsk lsk_gc full --tasks all --seeds 3 24 38 --input stft --shots 1 --action train-test --device cuda
```

逐模型完整命令见[三数据集清单](三数据集_单模型实验命令清单.md)。输入分目录，原生输入直接读Excel文本，不读取STFT图；结果/配置/源码快照与窗口清单成套保存。旧四模型结果使用旧快照补测，不混入不同输入统计。

## 2026-10-20 任务更新

当前为六组单源到单目标迁移，`--tasks all` 会运行 H1–H6；每个模型三个seed共18次训练。旧四任务的编号含义已改变，旧结果不可混入新汇总。仅任务路由改变，现有STFT缓存仍可复用；缓存manifest保留v1标识，实验resolved_config使用上述v2协议。模型、目标验证/测试隔离和早停规则不变。
