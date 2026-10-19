# HUST跨负载少样本实验

在本文件所在的`IFMAML-master`目录运行。只使用Hanoi HUST v3的6205–6208，按N/I/O/B/IO/IB/OB七类合并型号，每个episode随机抽五类。支持当前LSK-lite＋GCNet＋GGM-RW二阶MAML（`configured_maml`）及CNN4标准MAML（`maml`）。PU/CWRU原有入口和配置不变。

| 任务 | 源负载/W | 目标负载/W |
|---|---|---|
| HUST-T1 | 0＋200 | 400 |
| HUST-T2 | 0＋400 | 200 |
| HUST-T3 | 200＋400 | 0 |

源记录每条40窗口；目标记录前段20 validation、后段20 test，中间隔离4096点。每任务2240训练、560验证、560测试图片。目标验证用于选模和早停，测试集仅最终评价。实际记录长度209000–562001点，按各自长度切分，不截断或补齐到512000。

## 运行

首次生成三任务数据（本次已完成；重跑会核对配置、原始记录与图片校验和）：

```powershell
python run_hust_maml.py --task all --preprocess
```

正式训练及测试一个任务，默认当前最终模型、1-shot、seed24：

```powershell
python run_hust_maml.py --task HUST-T1 --train --test
```

更换任务、shot与算法：

```powershell
python run_hust_maml.py --task HUST-T2 --shots 5 --seed 38 --method configured_maml --train --test
python run_hust_maml.py --task HUST-T1 --shots 1 --seed 24 --method maml --train --test
```

仅加载正式权重测试，所有任务/shot/method参数须与训练匹配：

```powershell
python run_hust_maml.py --task HUST-T1 --test
```

`--test`不会启动训练。默认读取`model_save/hust/HUST_HUST-T1_configured_maml_5w1s_seed24_best`。指定`--model-path`时传入不带`_best`的路径前缀。已有同名权重时训练报错，防止覆盖；确需另跑请使用新的前缀。`--task all`依次执行三个任务，不能同时指定单一`--model-path`。

完整三任务×两种shot×三个seed，每方法18次训练：

```powershell
foreach ($hustShot in 1,5) {
    foreach ($hustSeed in 3,24,38) {
        python run_hust_maml.py --task all --shots $hustShot --seed $hustSeed --method configured_maml --train --test
        if ($LASTEXITCODE -ne 0) { throw 'HUST run failed' }
    }
}
```

换成`--method maml`完成另一条主线。默认每次最多500次outer update、meta-batch16、训练适应3步、测试10步、每类15 query、1000测试episode。配置集中于`config_hust.py`；本机检测为CPU，代码会在可用时自动使用CUDA。

快速功能检查采用1次outer update、1个训练/验证episode、2个测试episode，仍使用真实5-way/15 query与3/10适应步数；权重名自动加`_quick`，不能当正式结果：

```powershell
python run_hust_maml.py --task HUST-T3 --shots 5 --train --test --quick
python -m unittest test_hust_protocol -v
```

## 可追溯输出

- `hust_data_processed/HUST-T*/manifest.json`：原始记录SHA256、型号/负载/标签、每个窗口起止点、图片路径及SHA256。
- 同目录`episodes/`：固定目标验证/测试任务。每条包含全局五类、本地标签顺序、support/query窗口ID。清单与方法、训练seed无关；相同任务下1/5-shot共享query，1-shot support是5-shot support子集。
- `model_save/hust/*_best`与`.config.json`：最佳权重、配置、数据签名、选模epoch；`*_history.json`保存每轮训练/目标验证结果。
- `*_best_test_1000eps.json`和`.csv`：平均准确率、Macro-F1、七个全局状态的混淆矩阵、逐episode准确率、逐query预测及来源。分数表示七类池下5-way任务，不是同时七分类。

验证在outer update之后执行，保存的是被验证的那份权重。每个episode的fast parameters和BN缓冲独立，保留整批query统计；验证不会改变基础模型参数或向后续episode传递BN状态。源训练/目标验证的图片按需读取，训练及验证阶段不加载测试图片。

当前已完成真实数据预处理和小规模流程检查，未运行18/36次正式训练。支持范围目前为方案中的两条MAML主线，其他PU基线尚未接入HUST专用入口。
