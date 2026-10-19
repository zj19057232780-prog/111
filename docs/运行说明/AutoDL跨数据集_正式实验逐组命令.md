# AutoDL跨数据集正式实验命令（每组八任务）

更新：2026-10-17。全部命令使用小写内部模型名，明确指定输入、数据与输出目录，不依赖STFT/NOSTFT数组。SCG-MAML对应`configured_maml`，TL-WDCNN对应`tl_wdcnn`，TUBR-ML对应`tubr_ml`。

**须先更新服务器配套代码**：`python run_cross_dataset.py --help`应包含`--input`、`--checkpoint-root`、`--result-root`、`--epochs`、`--patience`，shots支持1/3/5；仅改模型名不能兼容旧入口。

每条命令：**一个模型 × 一个seed × 一个输入 × 一个shot，依次训练并测试T1–T8**。按1/3/5-shot分别列出；每种shot共51组，全部153组、1224次训练。STFT九模型，nostft八模型。

5-way、每类10 query、测试1000 episodes；元学习最多500次外更新，TL-WDCNN源预训练最多100 epochs，测试目标验证最佳模型。八个对比模型默认100轮验证准确率无严格提升早停；SCG-MAML不早停。按训练轮数计数，在验证点判断，准确率持平不重置。TL-WDCNN保留100轮上限，默认patience=100通常先达到上限。每条命令单独复制运行，已有完成或中断记录的组不会自动跳过/续训，请勿重复启动。

新版代码包：`autodl_cross_code_earlystop_20261017.tar.gz`。本版命令使用独立的`*_es100`输出目录，SCG-MAML在同批次中仍为patience=0。旧训练结果保留；旧checkpoint重测使用原源码快照。已有任务运行期间不要覆盖代码，新启动进程才会加载新逻辑。

## 终端初始化（每个新终端执行一次）

```bash
cd /root/autodl-tmp/ifmaml/IFMAML-master  # 按实际上传位置修改
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
```

## 1-shot · stft

### SCG-MAML

```bash
python -u run_cross_dataset.py --task all --method configured_maml --input stft --shots 1 --seed 3 --patience 0 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method configured_maml --input stft --shots 1 --seed 24 --patience 0 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method configured_maml --input stft --shots 1 --seed 38 --patience 0 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### MAML

```bash
python -u run_cross_dataset.py --task all --method maml --input stft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input stft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input stft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### ProtoNet

```bash
python -u run_cross_dataset.py --task all --method protonet --input stft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input stft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input stft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### MatchingNet

```bash
python -u run_cross_dataset.py --task all --method matchingnet --input stft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input stft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input stft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### RelationNet

```bash
python -u run_cross_dataset.py --task all --method relationnet --input stft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input stft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input stft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### TL-WDCNN

```bash
python -u run_cross_dataset.py --task all --method tl_wdcnn --input stft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input stft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input stft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### AMAML

```bash
python -u run_cross_dataset.py --task all --method amaml --input stft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input stft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input stft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### TUBR-ML

```bash
python -u run_cross_dataset.py --task all --method tubr_ml --input stft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input stft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input stft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### MFEML

```bash
python -u run_cross_dataset.py --task all --method mfeml --input stft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input stft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input stft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

## 1-shot · nostft

### MAML

```bash
python -u run_cross_dataset.py --task all --method maml --input nostft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input nostft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input nostft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### ProtoNet

```bash
python -u run_cross_dataset.py --task all --method protonet --input nostft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input nostft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input nostft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### MatchingNet

```bash
python -u run_cross_dataset.py --task all --method matchingnet --input nostft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input nostft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input nostft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### RelationNet

```bash
python -u run_cross_dataset.py --task all --method relationnet --input nostft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input nostft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input nostft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### TL-WDCNN

```bash
python -u run_cross_dataset.py --task all --method tl_wdcnn --input nostft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input nostft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input nostft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### AMAML

```bash
python -u run_cross_dataset.py --task all --method amaml --input nostft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input nostft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input nostft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### TUBR-ML

```bash
python -u run_cross_dataset.py --task all --method tubr_ml --input nostft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input nostft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input nostft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### MFEML

```bash
python -u run_cross_dataset.py --task all --method mfeml --input nostft --shots 1 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input nostft --shots 1 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input nostft --shots 1 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

## 3-shot · stft

### SCG-MAML

```bash
python -u run_cross_dataset.py --task all --method configured_maml --input stft --shots 3 --seed 3 --patience 0 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method configured_maml --input stft --shots 3 --seed 24 --patience 0 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method configured_maml --input stft --shots 3 --seed 38 --patience 0 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### MAML

```bash
python -u run_cross_dataset.py --task all --method maml --input stft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input stft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input stft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### ProtoNet

```bash
python -u run_cross_dataset.py --task all --method protonet --input stft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input stft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input stft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### MatchingNet

```bash
python -u run_cross_dataset.py --task all --method matchingnet --input stft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input stft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input stft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### RelationNet

```bash
python -u run_cross_dataset.py --task all --method relationnet --input stft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input stft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input stft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### TL-WDCNN

```bash
python -u run_cross_dataset.py --task all --method tl_wdcnn --input stft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input stft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input stft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### AMAML

```bash
python -u run_cross_dataset.py --task all --method amaml --input stft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input stft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input stft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### TUBR-ML

```bash
python -u run_cross_dataset.py --task all --method tubr_ml --input stft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input stft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input stft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### MFEML

```bash
python -u run_cross_dataset.py --task all --method mfeml --input stft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input stft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input stft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

## 3-shot · nostft

### MAML

```bash
python -u run_cross_dataset.py --task all --method maml --input nostft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input nostft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input nostft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### ProtoNet

```bash
python -u run_cross_dataset.py --task all --method protonet --input nostft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input nostft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input nostft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### MatchingNet

```bash
python -u run_cross_dataset.py --task all --method matchingnet --input nostft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input nostft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input nostft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### RelationNet

```bash
python -u run_cross_dataset.py --task all --method relationnet --input nostft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input nostft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input nostft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### TL-WDCNN

```bash
python -u run_cross_dataset.py --task all --method tl_wdcnn --input nostft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input nostft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input nostft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### AMAML

```bash
python -u run_cross_dataset.py --task all --method amaml --input nostft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input nostft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input nostft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### TUBR-ML

```bash
python -u run_cross_dataset.py --task all --method tubr_ml --input nostft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input nostft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input nostft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### MFEML

```bash
python -u run_cross_dataset.py --task all --method mfeml --input nostft --shots 3 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input nostft --shots 3 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input nostft --shots 3 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

## 5-shot · stft

### SCG-MAML

```bash
python -u run_cross_dataset.py --task all --method configured_maml --input stft --shots 5 --seed 3 --patience 0 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method configured_maml --input stft --shots 5 --seed 24 --patience 0 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method configured_maml --input stft --shots 5 --seed 38 --patience 0 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### MAML

```bash
python -u run_cross_dataset.py --task all --method maml --input stft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input stft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input stft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### ProtoNet

```bash
python -u run_cross_dataset.py --task all --method protonet --input stft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input stft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input stft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### MatchingNet

```bash
python -u run_cross_dataset.py --task all --method matchingnet --input stft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input stft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input stft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### RelationNet

```bash
python -u run_cross_dataset.py --task all --method relationnet --input stft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input stft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input stft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### TL-WDCNN

```bash
python -u run_cross_dataset.py --task all --method tl_wdcnn --input stft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input stft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input stft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### AMAML

```bash
python -u run_cross_dataset.py --task all --method amaml --input stft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input stft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input stft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### TUBR-ML

```bash
python -u run_cross_dataset.py --task all --method tubr_ml --input stft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input stft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input stft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

### MFEML

```bash
python -u run_cross_dataset.py --task all --method mfeml --input stft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input stft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input stft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_stft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_stft_es100 --train --test
```

## 5-shot · nostft

### MAML

```bash
python -u run_cross_dataset.py --task all --method maml --input nostft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input nostft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method maml --input nostft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### ProtoNet

```bash
python -u run_cross_dataset.py --task all --method protonet --input nostft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input nostft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method protonet --input nostft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### MatchingNet

```bash
python -u run_cross_dataset.py --task all --method matchingnet --input nostft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input nostft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method matchingnet --input nostft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### RelationNet

```bash
python -u run_cross_dataset.py --task all --method relationnet --input nostft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input nostft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method relationnet --input nostft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### TL-WDCNN

```bash
python -u run_cross_dataset.py --task all --method tl_wdcnn --input nostft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input nostft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tl_wdcnn --input nostft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### AMAML

```bash
python -u run_cross_dataset.py --task all --method amaml --input nostft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input nostft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method amaml --input nostft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### TUBR-ML

```bash
python -u run_cross_dataset.py --task all --method tubr_ml --input nostft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input nostft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method tubr_ml --input nostft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

### MFEML

```bash
python -u run_cross_dataset.py --task all --method mfeml --input nostft --shots 5 --seed 3 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input nostft --shots 5 --seed 24 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
python -u run_cross_dataset.py --task all --method mfeml --input nostft --shots 5 --seed 38 --patience 100 --epochs 500 --test-episodes 1000 --processed-root ./cross_data_processed_ab5_ms_v2 --checkpoint-root ./model_save/autodl_cross_ab5_ms_v2_nostft_es100 --result-root ./results/autodl_cross_ab5_ms_v2_nostft_es100 --train --test
```

