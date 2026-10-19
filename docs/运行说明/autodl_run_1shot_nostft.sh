#!/usr/bin/env bash
# 批次 B：nostft，8 模型 × seed3/24/38 × 8 任务 = 192 次训练
# 用法：bash autodl_run_1shot_nostft.sh（等 STFT 批次跑完再执行）
# 对比模型patience=100；入口强制SCG-MAML=0。上传早停新版代码后启动。
set -o pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/ifmaml}"
cd "$PROJECT_DIR/IFMAML-master" || { echo "目录不存在: $PROJECT_DIR/IFMAML-master"; exit 1; }
export PROJECT_ROOT="$(cd .. && pwd)"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg

CROSS_CACHE="$PROJECT_ROOT/IFMAML-master/cross_data_processed_ab5_ms_v2"
test -f "$CROSS_CACHE/manifest.json" || { echo "缓存缺失: $CROSS_CACHE"; exit 1; }

RUN_TAG=autodl_cross_ab5_ms_v2_nostft_es100
CROSS_CKPT="$PROJECT_ROOT/IFMAML-master/model_save/$RUN_TAG"
CROSS_RESULT="$PROJECT_ROOT/IFMAML-master/results/$RUN_TAG"
CROSS_LOG="$PROJECT_ROOT/IFMAML-master/logs/$RUN_TAG"
mkdir -p "$CROSS_LOG"
COMMON=(--processed-root "$CROSS_CACHE" --checkpoint-root "$CROSS_CKPT" --result-root "$CROSS_RESULT")

METHODS=(MAML ProtoNet MatchingNet RelationNet TL-WDCNN AMAML TUBR-ML MFEML)

echo "===== 批次 B（nostft）RUN_TAG=$RUN_TAG，8 模型（不含 SCG-MAML）====="
for m in "${METHODS[@]}"; do
  for s in 3 24 38; do
    log="$CROSS_LOG/${m}_nostft_1shot_seed${s}.log"
    echo "[$(date +%H:%M:%S)] START $m nostft 1shot seed$s (T1-T8)"
    python -u run_cross_dataset.py "${COMMON[@]}" \
      --task all --method "$m" --input nostft --shots 1 --seed "$s" \
      --patience 100 --train --test 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "[$(date '+%F %T')] FAILED $m nostft seed$s rc=$rc" | tee -a "$CROSS_LOG/failures.log"
      echo "[$(date +%H:%M:%S)] FAIL  $m nostft seed$s rc=$rc（继续下一个）"
    else
      echo "[$(date +%H:%M:%S)] DONE  $m nostft seed$s"
    fi
  done
done

echo "[$(date +%H:%M:%S)] 批次 B（nostft）结束"
echo "结果: $CROSS_RESULT"
echo "失败清单:"; cat "$CROSS_LOG/failures.log" 2>/dev/null; true
