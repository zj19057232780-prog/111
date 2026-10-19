# AutoDL 1-shot 执行清单（每模型 × 每 seed = 1 个单元，含 8 任务）

生成：2026-10-17。协议：`Cross-AB-MS-5way-targetval-v2`，5-way，**1-shot**，每类 10 query，测试 1000 episodes，元学习最多500次外更新；八个对比模型默认patience=100，SCG-MAML不早停。TL-WDCNN仍最多100 epochs，默认patience=100通常先达到上限。

| 项 | 值 |
|---|---|
| 单元定义 | 一个模型 × 一个 seed 跑 `--task all`（T1–T8） |
| 单元数 | 17 模型行 × 3 seed = **51 个单元** |
| 训练次数 | 51 × 8 = **408 次** |
| STFT 批次 | 9 模型，RUN_TAG `autodl_cross_ab5_ms_v2_stft_es100` |
| nostft 批次 | 8 模型（无 SCG-MAML），RUN_TAG `autodl_cross_ab5_ms_v2_nostft_es100` |

## 1. 执行清单（勾选完成）

| # | 批次 | 模型 | 输入 | seed | 任务 | 次数 | 完成 |
|---|---|---|---|---|---|---|---|
| 1 | A | SCG-MAML | stft | 3 | T1–T8 | 8 | ☐ |
| 2 | A | SCG-MAML | stft | 24 | T1–T8 | 8 | ☐ |
| 3 | A | SCG-MAML | stft | 38 | T1–T8 | 8 | ☐ |
| 4 | A | MAML | stft | 3 | T1–T8 | 8 | ☐ |
| 5 | A | MAML | stft | 24 | T1–T8 | 8 | ☐ |
| 6 | A | MAML | stft | 38 | T1–T8 | 8 | ☐ |
| 7 | A | ProtoNet | stft | 3 | T1–T8 | 8 | ☐ |
| 8 | A | ProtoNet | stft | 24 | T1–T8 | 8 | ☐ |
| 9 | A | ProtoNet | stft | 38 | T1–T8 | 8 | ☐ |
| 10 | A | MatchingNet | stft | 3 | T1–T8 | 8 | ☐ |
| 11 | A | MatchingNet | stft | 24 | T1–T8 | 8 | ☐ |
| 12 | A | MatchingNet | stft | 38 | T1–T8 | 8 | ☐ |
| 13 | A | RelationNet | stft | 3 | T1–T8 | 8 | ☐ |
| 14 | A | RelationNet | stft | 24 | T1–T8 | 8 | ☐ |
| 15 | A | RelationNet | stft | 38 | T1–T8 | 8 | ☐ |
| 16 | A | TL-WDCNN | stft | 3 | T1–T8 | 8 | ☐ |
| 17 | A | TL-WDCNN | stft | 24 | T1–T8 | 8 | ☐ |
| 18 | A | TL-WDCNN | stft | 38 | T1–T8 | 8 | ☐ |
| 19 | A | AMAML | stft | 3 | T1–T8 | 8 | ☐ |
| 20 | A | AMAML | stft | 24 | T1–T8 | 8 | ☐ |
| 21 | A | AMAML | stft | 38 | T1–T8 | 8 | ☐ |
| 22 | A | TUBR-ML | stft | 3 | T1–T8 | 8 | ☐ |
| 23 | A | TUBR-ML | stft | 24 | T1–T8 | 8 | ☐ |
| 24 | A | TUBR-ML | stft | 38 | T1–T8 | 8 | ☐ |
| 25 | A | MFEML | stft | 3 | T1–T8 | 8 | ☐ |
| 26 | A | MFEML | stft | 24 | T1–T8 | 8 | ☐ |
| 27 | A | MFEML | stft | 38 | T1–T8 | 8 | ☐ |
| 28 | B | MAML | nostft | 3 | T1–T8 | 8 | ☐ |
| 29 | B | MAML | nostft | 24 | T1–T8 | 8 | ☐ |
| 30 | B | MAML | nostft | 38 | T1–T8 | 8 | ☐ |
| 31 | B | ProtoNet | nostft | 3 | T1–T8 | 8 | ☐ |
| 32 | B | ProtoNet | nostft | 24 | T1–T8 | 8 | ☐ |
| 33 | B | ProtoNet | nostft | 38 | T1–T8 | 8 | ☐ |
| 34 | B | MatchingNet | nostft | 3 | T1–T8 | 8 | ☐ |
| 35 | B | MatchingNet | nostft | 24 | T1–T8 | 8 | ☐ |
| 36 | B | MatchingNet | nostft | 38 | T1–T8 | 8 | ☐ |
| 37 | B | RelationNet | nostft | 3 | T1–T8 | 8 | ☐ |
| 38 | B | RelationNet | nostft | 24 | T1–T8 | 8 | ☐ |
| 39 | B | RelationNet | nostft | 38 | T1–T8 | 8 | ☐ |
| 40 | B | TL-WDCNN | nostft | 3 | T1–T8 | 8 | ☐ |
| 41 | B | TL-WDCNN | nostft | 24 | T1–T8 | 8 | ☐ |
| 42 | B | TL-WDCNN | nostft | 38 | T1–T8 | 8 | ☐ |
| 43 | B | AMAML | nostft | 3 | T1–T8 | 8 | ☐ |
| 44 | B | AMAML | nostft | 24 | T1–T8 | 8 | ☐ |
| 45 | B | AMAML | nostft | 38 | T1–T8 | 8 | ☐ |
| 46 | B | TUBR-ML | nostft | 3 | T1–T8 | 8 | ☐ |
| 47 | B | TUBR-ML | nostft | 24 | T1–T8 | 8 | ☐ |
| 48 | B | TUBR-ML | nostft | 38 | T1–T8 | 8 | ☐ |
| 49 | B | MFEML | nostft | 3 | T1–T8 | 8 | ☐ |
| 50 | B | MFEML | nostft | 24 | T1–T8 | 8 | ☐ |
| 51 | B | MFEML | nostft | 38 | T1–T8 | 8 | ☐ |

## 2. 按模型进度表

| 模型 | 输入 | seed 3 | seed 24 | seed 38 | 单元数 |
|---|---|---|---|---|---|
| SCG-MAML | stft | ☐ | ☐ | ☐ | 3 |
| MAML | stft | ☐ | ☐ | ☐ | 3 |
| ProtoNet | stft | ☐ | ☐ | ☐ | 3 |
| MatchingNet | stft | ☐ | ☐ | ☐ | 3 |
| RelationNet | stft | ☐ | ☐ | ☐ | 3 |
| TL-WDCNN | stft | ☐ | ☐ | ☐ | 3 |
| AMAML | stft | ☐ | ☐ | ☐ | 3 |
| TUBR-ML | stft | ☐ | ☐ | ☐ | 3 |
| MFEML | stft | ☐ | ☐ | ☐ | 3 |
| MAML | nostft | ☐ | ☐ | ☐ | 3 |
| ProtoNet | nostft | ☐ | ☐ | ☐ | 3 |
| MatchingNet | nostft | ☐ | ☐ | ☐ | 3 |
| RelationNet | nostft | ☐ | ☐ | ☐ | 3 |
| TL-WDCNN | nostft | ☐ | ☐ | ☐ | 3 |
| AMAML | nostft | ☐ | ☐ | ☐ | 3 |
| TUBR-ML | nostft | ☐ | ☐ | ☐ | 3 |
| MFEML | nostft | ☐ | ☐ | ☐ | 3 |

## 3. 单单元命令（复制即用，替换 MODEL/SEED/MODE）

```bash
cd /root/autodl-tmp/ifmaml/IFMAML-master
export PROJECT_ROOT="$(cd .. && pwd)"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
CROSS_CACHE="$PROJECT_ROOT/IFMAML-master/cross_data_processed_ab5_ms_v2"

# STFT 批次（先跑）
RUN_TAG=autodl_cross_ab5_ms_v2_stft_es100
python -u run_cross_dataset.py \
  --processed-root "$CROSS_CACHE" \
  --checkpoint-root "$PROJECT_ROOT/IFMAML-master/model_save/$RUN_TAG" \
  --result-root "$PROJECT_ROOT/IFMAML-master/results/$RUN_TAG" \
  --task all --method AMAML --input stft --shots 1 --seed 3 --train --test

# nostft 批次（STFT 跑完再跑）
RUN_TAG=autodl_cross_ab5_ms_v2_nostft_es100
python -u run_cross_dataset.py \
  --processed-root "$CROSS_CACHE" \
  --checkpoint-root "$PROJECT_ROOT/IFMAML-master/model_save/$RUN_TAG" \
  --result-root "$PROJECT_ROOT/IFMAML-master/results/$RUN_TAG" \
  --task all --method AMAML --input nostft --shots 1 --seed 3 --train --test
```

## 4. 运行脚本（分两次跑，不要合并）

两个独立脚本，各自一个 RUN_TAG、各自的 results / model_save / logs：

| 脚本 | 批次 | 模型数 | 单元 | 训练次数 |
|---|---|---|---|---|
| `autodl_run_1shot_stft.sh` | A（stft） | 9（含 SCG-MAML） | 27 | 216 |
| `autodl_run_1shot_nostft.sh` | B（nostft） | 8（无 SCG-MAML） | 24 | 192 |

**第一次**先跑 STFT：

```bash
bash autodl_run_1shot_stft.sh
```

等它结束后核对 `216` 个 summary.json，**再第二次**跑 nostft：

```bash
bash autodl_run_1shot_nostft.sh
```

两个脚本内部都是串行单进程；不要同时启动（单卡会抢显存）。后台运行方式见命令清单第 3 节（tmux 或 nohup）。

## 5. 完成后的核对

```bash
CROSS_RESULT="$PROJECT_ROOT/IFMAML-master/results/autodl_cross_ab5_ms_v2_stft_es100"
find "$CROSS_RESULT" -name summary.json | wc -l     # 期望 9 模型 × 8 任务 × 3 seed = 216
CROSS_RESULT="$PROJECT_ROOT/IFMAML-master/results/autodl_cross_ab5_ms_v2_nostft_es100"
find "$CROSS_RESULT" -name summary.json | wc -l     # 期望 8 × 8 × 3 = 192
cat "$PROJECT_ROOT/IFMAML-master/logs/autodl_cross_ab5_ms_v2_stft_es100/failures.log" 2>/dev/null
```
