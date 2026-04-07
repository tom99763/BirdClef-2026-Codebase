#!/bin/bash
# Launch V2S 4-fold Grand Combo training sequentially across both GPUs.
#
# Schedule:
#   GPU0: fold0 → fold2  (birdclef-gpu0:0 after v29 finishes)
#   GPU1: fold1 → fold3  (birdclef-gpu1:1 after v24/v30 finishes)
#
# Usage:
#   bash scripts/launch_4fold_grand.sh 0   # GPU0 side: fold0 then fold2
#   bash scripts/launch_4fold_grand.sh 1   # GPU1 side: fold1 then fold3
#
# Or run both in separate panes:
#   tmux send-keys -t birdclef-gpu0:0 "bash scripts/launch_4fold_grand.sh 0" Enter
#   tmux send-keys -t birdclef-gpu1:1 "bash scripts/launch_4fold_grand.sh 1" Enter

set -e
cd /home/lab/BirdClef-2026-Codebase

GPU_SIDE=${1:-0}  # 0 or 1

if [ "$GPU_SIDE" = "0" ]; then
    FOLDS="0 2"
    GPU_ID=0
else
    FOLDS="1 3"
    GPU_ID=1
fi

echo "[4fold-launch] GPU$GPU_ID: folds $FOLDS"
echo "[4fold-launch] Start time: $(date)"

for FOLD in $FOLDS; do
    CONFIG="configs/sed_b0_4fold_grand_v1_fold${FOLD}.yaml"
    LOG="outputs/sed-b0-4fold-grand-v1-fold${FOLD}.log"

    echo ""
    echo "==========================================================="
    echo "[4fold-launch] Fold $FOLD  Config=$CONFIG  GPU=$GPU_ID"
    echo "[4fold-launch] Start: $(date)"
    echo "==========================================================="

    CUDA_VISIBLE_DEVICES=$GPU_ID python train_sed.py \
        --config "$CONFIG" \
        --gpu $GPU_ID \
        2>&1 | tee "$LOG"

    echo "[4fold-launch] Fold $FOLD finished: $(date)"
    echo ""

    # Model soup for this fold
    echo "[4fold-launch] Running model soup for fold $FOLD..."
    python scripts/model_soup.py \
        --run "sed-b0-4fold-grand-v1-fold${FOLD}" \
        --config "$CONFIG" \
        2>&1 | tee -a "$LOG" || echo "[4fold-launch] WARN: soup failed for fold$FOLD (non-fatal)"

    echo "[4fold-launch] Fold $FOLD soup done."
done

echo ""
echo "[4fold-launch] All folds (GPU$GPU_ID side) completed: $(date)"
echo "[4fold-launch] Folds run: $FOLDS"
