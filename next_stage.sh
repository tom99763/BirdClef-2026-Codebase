#!/usr/bin/env bash
# next_stage.sh — wait for nohuman extraction, then train next round
# Run after: evaluation and label-head-pseudo training are done
cd /home/lab/BirdClef-2026-Codebase

log() { echo "[$(date '+%H:%M:%S')] $*"; }

EXTRACT_PID=$1
log "Waiting for nohuman extraction (PID=$EXTRACT_PID)..."
while kill -0 $EXTRACT_PID 2>/dev/null; do sleep 60; done
log "Extraction done."

# Add pseudo features to nohuman-label cache
log "Adding pseudo label features to nohuman cache..."
CUDA_VISIBLE_DEVICES=0 python extract_pseudo_label_features.py \
    --gpu 0 \
    --cache_dir outputs/embeddings_cache_nohuman_label \
    >> outputs/extract_nohuman_label.log 2>&1
log "Pseudo done."

# Start next round
log "Starting nohuman-label-head (GPU 0) and nohuman-label-pseudo (GPU 1)..."
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --config configs/exp_nohuman_label_head.yaml \
    > outputs/nohuman-label-head.log 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --config configs/exp_nohuman_label_pseudo.yaml \
    > outputs/nohuman-label-pseudo.log 2>&1 &
PID_B=$!
log "nohuman-label-head PID=$PID_A, nohuman-label-pseudo PID=$PID_B"

# Wait and evaluate
while kill -0 $PID_A 2>/dev/null || kill -0 $PID_B 2>/dev/null; do
    A=$(grep "Epoch " outputs/nohuman-label-head.log 2>/dev/null | grep -oP "Epoch +\K[0-9]+/[0-9]+" | tail -1)
    B=$(grep "Epoch " outputs/nohuman-label-pseudo.log 2>/dev/null | grep -oP "Epoch +\K[0-9]+/[0-9]+" | tail -1)
    log "nohuman-label-head: $A | nohuman-label-pseudo: $B"
    sleep 120
done

log "Round 2 done. Evaluating..."
CUDA_VISIBLE_DEVICES=0 python evaluate_final.py \
    --runs nohuman-label-head nohuman-label-pseudo \
    --gpu 0 >> outputs/evaluate_r2.log 2>&1
log "Evaluation done."
grep -E "Official|Score" outputs/evaluate_r2.log | tail -10
