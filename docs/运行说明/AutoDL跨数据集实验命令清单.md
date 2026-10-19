# AutoDL 跨数据集实验命令清单

更新：2026-10-17。以下是 **Linux Bash** 命令，适用当前 `Cross-AB-MS-5way-targetval-v2`，从头训练服务器实验。已对照现有入口参数核查；尚未在你的 AutoDL 实例执行。

2026-10-17早停更新：服务器先上传配套`autodl_cross_code_earlystop_20261017.tar.gz`；新批次与旧固定预算结果分目录。`--patience 0`可关闭对比模型早停，SCG-MAML始终关闭。

## 0. 快速上手（TL;DR）

```bash
# ① 上传：只需代码 + 缓存，不要传原始 MAT
#   IFMAML-master/*.py + my_utils/ + cross_data_processed_ab5_ms_v2/
#   comparison_models/{mu2025_amaml,wang2025_tubr_ml,li2025_mfeml}/ 的 py（runs 可不传）

# ② 每个新终端执行第 3 节整段（含两段自检）；先 cd 到实际的 IFMAML-master，再自动取父目录
cd /root/autodl-tmp/ifmaml/IFMAML-master        # ← 改成你的实际上传位置
export PROJECT_ROOT="$(cd .. && pwd)"           # 自动推导，不写死路径
export CUDA_VISIBLE_DEVICES=0
export RUN_TAG=autodl_cross_ab5_ms_v2_es100_v1          # 一批一个标签，不要复用
export CROSS_CACHE="$PROJECT_ROOT/IFMAML-master/cross_data_processed_ab5_ms_v2"
export CROSS_CKPT="$PROJECT_ROOT/IFMAML-master/model_save/$RUN_TAG"
export CROSS_RESULT="$PROJECT_ROOT/IFMAML-master/results/$RUN_TAG"
export CROSS_LOG="$PROJECT_ROOT/IFMAML-master/logs/$RUN_TAG"
mkdir -p "$CROSS_LOG"
COMMON=(--processed-root "$CROSS_CACHE" --checkpoint-root "$CROSS_CKPT" --result-root "$CROSS_RESULT")

# ③ 冒烟（第 4 节，只验证链路，不出论文结果）
python -u run_cross_dataset.py --task T1 --method MAML --input stft --shots 1 --seed 3 \
  --processed-root "$CROSS_CACHE" \
  --checkpoint-root "$PROJECT_ROOT/IFMAML-master/tmp/smoke/models" \
  --result-root "$PROJECT_ROOT/IFMAML-master/tmp/smoke/results" --quick --train --test

# ④ 正式：先执行第 6 节的 run_model 函数，再按模型跑（每个模型 72 次训练）
run_model MAML stft

# ⑤ 汇总
python summarize_cross_dataset.py --root "$CROSS_RESULT" --seeds 3 24 38 --episodes 1000
```

**三条铁律**

1. 同一 `RUN_TAG` 只跑一遍：入口**不自动跳过已完成、不续训**，重复执行会撞防覆盖检查。
2. 已上传缓存就**不要**加 `--preprocess`：会重读原始 MAT 并因绝对路径不一致报 `Raw data/protocol mismatch`。
3. 训练完成但测试中断 → 同参数加 `--test` 补；训练本身中断 → 换 `RETRY_TAG` 从头重跑，不混入原批。

建议节奏：第 3 节检查 → 第 4 节冒烟 → 单模型 `run_model MAML stft` 估时 → 其余模型 → 汇总下载。


## 1. 任务与实验规模

| 任务 | 源域 | 目标工况 |
|---|---|---|
| T1 | CWRU全部四工况 | PU P1：N09_M07_F10 |
| T2 | CWRU全部四工况 | PU P2：N15_M01_F10 |
| T3 | CWRU全部四工况 | PU P3：N15_M07_F04 |
| T4 | CWRU全部四工况 | PU P4：N15_M07_F10 |
| T5 | PU全部四工况 | CWRU C0：1797 rpm目录 |
| T6 | PU全部四工况 | CWRU C1：1772 rpm目录 |
| T7 | PU全部四工况 | CWRU C2：1750 rpm目录 |
| T8 | PU全部四工况 | CWRU C3：1730 rpm目录 |

所有任务5-way，1/3/5-shot，每类10 query；seed 3/24/38，测试1000 episodes。
元学习模型500次外更新，每10次更新做一次目标验证；TL-WDCNN源预训练100 epochs，每2 epochs验证。均保存目标验证最佳权重后测试。八个对比模型默认`--patience 100`，SCG-MAML始终为0；严格验证准确率改善才重置，持平或仅loss下降不重置，按训练轮数计数并在验证点判断。TL-WDCNN保留100轮上限，默认耐心100通常先达到上限。`--epochs`只修改元学习外更新上限，不能用它改变TL-WDCNN的预训练轮数。

STFT有9模型，nostft有8模型（SCG-MAML不支持nostft）。每个“模型×输入”完整行包含8任务×3shot×3seed=72次训练；STFT共648次，nostft共576次，总计1224次。只跑1/5-shot则共816次；本文完整循环包含3-shot。

## 2. 上传目录：优先使用已生成缓存

本文以 `/root/autodl-tmp/ifmaml` 为例（也兼容任何其他根目录名）。下方命令先 `cd` 到实际的 `IFMAML-master`，再用 `export PROJECT_ROOT="$(cd .. && pwd)"` 自动推导，**不需要手改路径**；只需确认 `IFMAML-master/` 与 `comparison_models/` 平级。

```text
/root/autodl-tmp/new-ifmaml/
├── IFMAML-master/
│   ├── *.py                           顶层Python文件保持完整
│   ├── my_utils/
│   └── cross_data_processed_ab5_ms_v2/ 整个目录上传
│       ├── manifest.json
│       ├── stft_index.json
│       ├── ... .npy、.png文件
│       └── episodes/                  若已有则一起上传
└── comparison_models/
    ├── mu2025_amaml/                   保留代码与方法资源，runs可不传
    ├── wang2025_tubr_ml/
    └── li2025_mfeml/
```

STFT与nostft都能从这个完整缓存读取。采用缓存运行时，**无需上传原始MAT，也不要执行 `--preprocess`**。`manifest.json`内原始Windows路径是溯源记录，训练使用缓存内的相对路径，不要手动改写它。改写manifest会改变数据签名。

上传缓存后再运行预处理，会重新读MAT并比较原始绝对路径，可能因服务器路径变化报 `Raw data/protocol mismatch`。需从原始MAT重新生成时，使用第10节的独立新缓存目录。

当前本地 `IFMAML-master/pu_data_mat` 是另一种整理格式，128个MAT中没有本协议要求的40个 `Nxx_Mxx_Fxx_轴承编号_1/2.mat`，不能直接当作本协议的预处理原料。

## 3. 每个新终端先执行：路径、GPU和缓存检查

如已安装tmux，可先执行 `tmux new -s cross`，再在会话中执行以下设置；断线后用 `tmux attach -t cross` 返回。不要在多个终端重复启动同一组任务。两段检查均通过后再开始训练。

```bash
cd /root/autodl-tmp/ifmaml/IFMAML-master        # ← 改成实际上传位置
export PROJECT_ROOT="$(cd .. && pwd)"           # 自动取父目录，避免路径写死

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export MPLBACKEND=Agg

# 每批服务器实验固定一个标签；训练、测试、汇总始终使用相同标签。
export RUN_TAG=autodl_cross_ab5_ms_v2_es100_v1
export CROSS_CACHE="$PROJECT_ROOT/IFMAML-master/cross_data_processed_ab5_ms_v2"
export CROSS_CKPT="$PROJECT_ROOT/IFMAML-master/model_save/$RUN_TAG"
export CROSS_RESULT="$PROJECT_ROOT/IFMAML-master/results/$RUN_TAG"
export CROSS_LOG="$PROJECT_ROOT/IFMAML-master/logs/$RUN_TAG"
mkdir -p "$CROSS_LOG"

COMMON=(--processed-root "$CROSS_CACHE" --checkpoint-root "$CROSS_CKPT" --result-root "$CROSS_RESULT")
set -o pipefail

python - <<'PY'
import sys, torch
import numpy, scipy, cv2, pandas, matplotlib, sklearn, PIL
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)
assert torch.cuda.is_available(), '当前Python无法使用CUDA；先检查解释器和PyTorch环境'
print('GPU:', torch.cuda.get_device_name(0))
PY

# 本段仅检查文件与数据结构，不训练、不生成新缓存。
python - <<'PY'
import os, json
from pathlib import Path
from config_cross_dataset import resolve_cross_config
from cross_dataset_data import CrossDataset
from hust_preprocess import digest
root = Path(os.environ['CROSS_CACHE'])
m = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
index = json.loads((root / 'stft_index.json').read_text(encoding='utf-8'))
assert len(m['windows']) == 1280
for row in m['windows']:
    assert digest(root / row['raw']) == row['raw_sha256'], row['raw']
    assert digest(root / row['image']) == index[row['image']], row['image']
for i in range(1, 9):
    cfg = resolve_cross_config(dict(task=f'T{i}', method='maml', input_mode='stft',
                                    processed_root=str(root)))
    CrossDataset(cfg)
print('1280个原始锚与STFT文件哈希、八任务数据划分检查通过')
PY
```

如果缺少普通Python库，可在当前环境安装后重新检查；已有CUDA版PyTorch先保留，不用基础CPU解释器替换它：

```bash
python -m pip install numpy scipy opencv-python-headless pandas matplotlib scikit-learn pillow
```

## 4. 首次部署：小规模检查

独立目录带时间戳，不会占用正式模型路径。这两条只检查运行链路，不能作为论文结果。

```bash
SMOKE_TAG="smoke_$(date +%Y%m%d_%H%M%S)"
python -u run_cross_dataset.py --task T1 --method MAML --input stft --shots 1 --seed 3 \
  --processed-root "$CROSS_CACHE" \
  --checkpoint-root "$PROJECT_ROOT/IFMAML-master/tmp/$SMOKE_TAG/models" \
  --result-root "$PROJECT_ROOT/IFMAML-master/tmp/$SMOKE_TAG/results" \
  --quick --train --test

python -u run_cross_dataset.py --task T5 --method TUBR-ML --input nostft --shots 5 --seed 3 \
  --processed-root "$CROSS_CACHE" \
  --checkpoint-root "$PROJECT_ROOT/IFMAML-master/tmp/$SMOKE_TAG/models" \
  --result-root "$PROJECT_ROOT/IFMAML-master/tmp/$SMOKE_TAG/results" \
  --quick --train --test
```

## 5. 单个任务命令

每次选其中一条运行。下面固定T1、1-shot、seed3；改 `--task T1`、`--shots 1`、`--seed 3` 即可切换任务、shot和seed。原始接口把 `--input stft` 改为 `--input nostft`，SCG-MAML除外。

```bash
python -u run_cross_dataset.py "${COMMON[@]}" --task T1 --method ProtoNet    --input stft --shots 1 --seed 3 --train --test
python -u run_cross_dataset.py "${COMMON[@]}" --task T1 --method MatchingNet --input stft --shots 1 --seed 3 --train --test
python -u run_cross_dataset.py "${COMMON[@]}" --task T1 --method RelationNet --input stft --shots 1 --seed 3 --train --test
python -u run_cross_dataset.py "${COMMON[@]}" --task T1 --method MAML        --input stft --shots 1 --seed 3 --train --test
python -u run_cross_dataset.py "${COMMON[@]}" --task T1 --method TL-WDCNN    --input stft --shots 1 --seed 3 --train --test
python -u run_cross_dataset.py "${COMMON[@]}" --task T1 --method AMAML       --input stft --shots 1 --seed 3 --train --test
python -u run_cross_dataset.py "${COMMON[@]}" --task T1 --method TUBR-ML     --input stft --shots 1 --seed 3 --train --test
python -u run_cross_dataset.py "${COMMON[@]}" --task T1 --method MFEML       --input stft --shots 1 --seed 3 --train --test
python -u run_cross_dataset.py "${COMMON[@]}" --task T1 --method SCG-MAML    --input stft --shots 1 --seed 3 --train --test
```

TL-WDCNN的STFT输入是64×64图像展开为4096点，属于项目适配对照；原生时域输入用nostft。其他nostft模型沿用各自原始长度、归一化或reshape设置，不统一改成STFT。

## 6. 推荐执行单位：单模型完整八任务

先定义函数。它每次顺序跑一个模型、一个输入下的全部1/3/5-shot和3/24/38，共72次训练，日志实时保存。任何一条失败就停止此函数，不跳过未完成任务。

```bash
run_model() {
  local method="$1" mode="$2" shot seed log
  for shot in 1 3 5; do
    for seed in 3 24 38; do
      log="$CROSS_LOG/${method}_${mode}_${shot}shot_seed${seed}.log"
      python -u run_cross_dataset.py "${COMMON[@]}" \
        --task all --method "$method" --input "$mode" --shots "$shot" --seed "$seed" \
        --train --test 2>&1 | tee -a "$log" || return 1
    done
  done
}
```

STFT：按需要逐条选择模型，不必一次全部执行。

```bash
run_model ProtoNet stft
run_model MatchingNet stft
run_model RelationNet stft
run_model MAML stft
run_model TL-WDCNN stft
run_model AMAML stft
run_model TUBR-ML stft
run_model MFEML stft
run_model SCG-MAML stft
```

nostft：八个模型，明确不包含SCG-MAML。

```bash
run_model ProtoNet nostft
run_model MatchingNet nostft
run_model RelationNet nostft
run_model MAML nostft
run_model TL-WDCNN nostft
run_model AMAML nostft
run_model TUBR-ML nostft
run_model MFEML nostft
```

### 只要 1-shot × 3 seed（每模型 24 次训练）

```bash
run_model_1shot() {
  local method="$1" mode="$2" seed log rc
  for seed in 3 24 38; do
    log="$CROSS_LOG/${method}_${mode}_1shot_seed${seed}.log"
    echo "[$(date +%H:%M:%S)] START $method $mode 1shot seed$seed"
    python -u run_cross_dataset.py "${COMMON[@]}" \
      --task all --method "$method" --input "$mode" --shots 1 --seed "$seed" \
      --train --test 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "$(date +%F' '%T) FAILED $method $mode 1shot seed$seed rc=$rc" | tee -a "$CROSS_LOG/failures.log"
      return 1
    fi
  done
}
```

批量启动：**STFT 与 nostft 必须分两个 RUN_TAG、两套输出根**，不要混在同一批。一个模型失败只记 `failures.log`，不阻塞后续模型。

```bash
# ---------- 批次 A：STFT（9 模型 × 24 次）----------
export RUN_TAG=autodl_cross_ab5_ms_v2_stft_es100
export CROSS_CKPT="$PROJECT_ROOT/IFMAML-master/model_save/$RUN_TAG"
export CROSS_RESULT="$PROJECT_ROOT/IFMAML-master/results/$RUN_TAG"
export CROSS_LOG="$PROJECT_ROOT/IFMAML-master/logs/$RUN_TAG"
mkdir -p "$CROSS_LOG"
COMMON=(--processed-root "$CROSS_CACHE" --checkpoint-root "$CROSS_CKPT" --result-root "$CROSS_RESULT")

for m in SCG-MAML MAML ProtoNet MatchingNet RelationNet TL-WDCNN AMAML TUBR-ML MFEML; do
  echo "===== $m / stft ====="; run_model_1shot "$m" stft || true
done
```

```bash
# ---------- 批次 B：nostft（跑完 A 再跑；8 模型 × 24 次，不含 SCG-MAML）----------
export RUN_TAG=autodl_cross_ab5_ms_v2_nostft_es100
export CROSS_CKPT="$PROJECT_ROOT/IFMAML-master/model_save/$RUN_TAG"
export CROSS_RESULT="$PROJECT_ROOT/IFMAML-master/results/$RUN_TAG"
export CROSS_LOG="$PROJECT_ROOT/IFMAML-master/logs/$RUN_TAG"
mkdir -p "$CROSS_LOG"
COMMON=(--processed-root "$CROSS_CACHE" --checkpoint-root "$CROSS_CKPT" --result-root "$CROSS_RESULT")

for m in MAML ProtoNet MatchingNet RelationNet TL-WDCNN AMAML TUBR-ML MFEML; do
  echo "===== $m / nostft ====="; run_model_1shot "$m" nostft || true
done
```

两批共 17 个模型行 × 24 = 408 次训练。两批共用同一 GPU，**串行执行**（先 A 后 B），不要同时开两个终端抢显存；汇总时分别指定各自的 `CROSS_RESULT`。

如只想跑一个shot和seed的全部八任务，可直接运行：

```bash
python -u run_cross_dataset.py "${COMMON[@]}" --task all --method MAML --input stft --shots 1 --seed 3 --train --test
```

## 7. 全矩阵一次运行（只用于尚未训练的新批次）

与第5、6节是不同启动方式，**同一RUN_TAG不要重复执行重叠任务**。以下顺序跑1224个实验单元，SCG-MAML在nostft中由现有入口自动排除。失败即停止函数。

```bash
run_full_matrix() {
  local mode shot seed log
  for mode in stft nostft; do
    for shot in 1 3 5; do
      for seed in 3 24 38; do
        log="$CROSS_LOG/all_${mode}_${shot}shot_seed${seed}.log"
        python -u run_cross_dataset.py "${COMMON[@]}" \
          --task all --method all --input "$mode" --shots "$shot" --seed "$seed" \
          --train --test 2>&1 | tee -a "$log" || return 1
      done
    done
  done
}
run_full_matrix
```

## 8. 已完成、失败和中断后的操作

当前入口**没有自动续训或自动跳过已完成任务**，也没有 `--resume`、`--force-rerun`、`--device` 参数。已有history/权重/源码快照会触发防覆盖检查；不要删除后直接混成同一批结果。

训练完成但测试中断：用相同参数单独测试，重新跑完整1000测试episodes，不重新训练。前提是训练已完整结束；只有中途的best权重不代表完成500次外更新。

```bash
python -u run_cross_dataset.py "${COMMON[@]}" \
  --task T1 --method MAML --input stft --shots 1 --seed 3 --test
```

训练本身中断：可以换独立输出根，从头重跑这个单任务。下例不会复用中途权重；后续重测及汇总须继续使用重跑目录。为避免混入不同重跑配置，原批与重跑批分别汇总，审核后再整合。

```bash
export RETRY_TAG="${RUN_TAG}_retry_$(date +%Y%m%d_%H%M%S)"
python -u run_cross_dataset.py --processed-root "$CROSS_CACHE" \
  --checkpoint-root "$PROJECT_ROOT/IFMAML-master/model_save/$RETRY_TAG" \
  --result-root "$PROJECT_ROOT/IFMAML-master/results/$RETRY_TAG" \
  --task T1 --method MAML --input stft --shots 1 --seed 3 --train --test
```

部分任务已完成时，用第5节单独跑缺失的T编号；再次执行 `--task all` 不会自动跳过已有结果。训练开始后保留代码版本，入口会检查checkpoint对应代码哈希；上传后不要再运行git自动换行格式化或改动训练源码。

## 9. 汇总、查看和下载

```bash
python summarize_cross_dataset.py --root "$CROSS_RESULT" --seeds 3 24 38 --episodes 1000

# 查看当前批次实际产生的测试摘要。
find "$CROSS_RESULT" -name summary.json -print

# 持续查看某个模型批次日志；Ctrl+C只退出tail，不停止另一会话中的训练。
tail -f "$CROSS_LOG/MAML_stft_1shot_seed3.log"
```

输出位置：

- `$CROSS_RESULT/<模型标识>/<测试时间戳>/summary.json`：Accuracy、Macro-F1、混淆矩阵、来源。
- 同目录 `predictions.csv`、`config.json`：预测与设置。
- `$CROSS_CKPT/<模型标识>_best`及`.config.json`：最佳模型与`best_epoch`。
- 同前缀`_history.json`、`_resolved_config.json`、`_source/`：训练记录、配置和源码快照。
- `$CROSS_RESULT/comparison.csv`：T1–T8及双向汇总；缺seed显示`incomplete`。不是自动续训状态表。

完成训练后打包下载本批结果、权重、日志和缓存中的采样依据。下面会打包整个跨库缓存，便于保留同一份manifest/episodes及输入文件：

```bash
tar -czf "$PROJECT_ROOT/${RUN_TAG}_results.tar.gz" \
  -C "$PROJECT_ROOT/IFMAML-master" \
  "results/$RUN_TAG" "model_save/$RUN_TAG" "logs/$RUN_TAG" \
  cross_data_processed_ab5_ms_v2
```

如果第10节改用了新的缓存路径，打包时也要换成对应目录。只打包已停止写入的批次。服务器与本地的硬件、库版本不同，在报告中记录环境；优先在同一服务器环境完成一组seed，不因同名协议就假定数值逐位一致。

## 10. 可选：从原始MAT建立新的服务器缓存

已有完整缓存时跳过本节。确实需要重建时：上传CWRU原目录，以及PU五轴承K001/KI03/KI07/KA05/KA03、四工况、记录1/2，共40个唯一匹配的MAT文件。例如 `N09_M07_F10_KI03_1.mat` 和 `_2.mat`。不要用现有pu_data_mat的重命名MAT代替。

```bash
# 指向真正的PU原始目录。下面路径需与你的上传位置一致。
export PU_ORIGINAL="$PROJECT_ROOT/PU_original"
export CROSS_CACHE="$PROJECT_ROOT/IFMAML-master/cross_data_processed_ab5_ms_v2_server"
# 新缓存使用独立批次；后续直接使用这里的路径，不重新执行第3节的默认赋值。
export RUN_TAG=autodl_cross_ab5_ms_v2_rebuilt_v1
export CROSS_CKPT="$PROJECT_ROOT/IFMAML-master/model_save/$RUN_TAG"
export CROSS_RESULT="$PROJECT_ROOT/IFMAML-master/results/$RUN_TAG"
export CROSS_LOG="$PROJECT_ROOT/IFMAML-master/logs/$RUN_TAG"
mkdir -p "$CROSS_LOG"

python -u run_cross_dataset.py --preprocess --task T1 --method MAML --input stft \
  --cwru-raw-root "$PROJECT_ROOT/IFMAML-master/CWRU" \
  --pu-raw-root "$PU_ORIGINAL" --processed-root "$CROSS_CACHE"

# 重建后更新参数数组；使用这个新缓存跑同一批所有模型。
COMMON=(--processed-root "$CROSS_CACHE" --checkpoint-root "$CROSS_CKPT" --result-root "$CROSS_RESULT")
```

此命令只预处理，不训练。stft会同时生成原始锚和图像缓存，随后两种输入均可用。不同绝对来源路径会改变manifest哈希，服务器重建数据与上传缓存属于不同数据签名，不放进同一个结果根目录混合汇总。重新运行第3节的两段Python检查（保留本节路径变量），通过后选第5、6或7节开始训练。
