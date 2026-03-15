#!/usr/bin/env bash
# monitor_and_advance.sh
# Waits for perch-label-head and label-head-pseudo to finish,
# then extracts nohuman label features and starts the next experiments.

set -e
cd "$(dirname "$0")"

PID_LABEL_HEAD=$1
PID_LABEL_PSEUDO=$2

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: wait for both current trainings ───────────────────────────────────
log "Watching perch-label-head (PID=$PID_LABEL_HEAD) and label-head-pseudo (PID=$PID_LABEL_PSEUDO)"

while kill -0 $PID_LABEL_HEAD 2>/dev/null || kill -0 $PID_LABEL_PSEUDO 2>/dev/null; do
    # Show progress every 2 min
    LH=$(grep "Epoch " outputs/perch-label-head.log 2>/dev/null | grep -oP "Epoch +\K[0-9]+/[0-9]+" | tail -1)
    LP=$(grep "Epoch " outputs/label-head-pseudo.log 2>/dev/null | grep -oP "Epoch +\K[0-9]+/[0-9]+" | tail -1)
    log "perch-label-head: $LH | label-head-pseudo: $LP"
    sleep 120
done

log "Both trainings finished. Running evaluation..."

# ── Step 2: evaluate finished runs ───────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python evaluate_final.py \
    --runs perch-label-head label-head-pseudo \
    --gpu 0 \
    >> outputs/evaluate_label_heads.log 2>&1 &
EVAL_PID=$!
log "Evaluation started (PID=$EVAL_PID)"
wait $EVAL_PID
log "Evaluation done."

# Print scores
grep -E "Official|Score" outputs/evaluate_label_heads.log | tail -10

# ── Step 3: extract nohuman label features ───────────────────────────────────
log "Extracting nohuman label features on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python extract_nohuman_label_features.py \
    --config configs/default.yaml \
    --gpu 0 \
    > outputs/extract_nohuman_label.log 2>&1
log "Nohuman label extraction done."

# Add pseudo entries to nohuman-label manifest
log "Extracting pseudo label features for nohuman cache..."
CUDA_VISIBLE_DEVICES=0 python extract_pseudo_label_features.py \
    --gpu 0 \
    --cache_dir outputs/embeddings_cache_nohuman_label \
    >> outputs/extract_nohuman_label.log 2>&1
log "Pseudo label features added to nohuman-label cache."

# ── Step 4: start next experiments ───────────────────────────────────────────
log "Starting nohuman-label-head on GPU 0, nohuman-label-pseudo on GPU 1..."

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --config configs/exp_nohuman_label_head.yaml \
    > outputs/nohuman-label-head.log 2>&1 &
PID_NLH=$!

CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --config configs/exp_nohuman_label_pseudo.yaml \
    > outputs/nohuman-label-pseudo.log 2>&1 &
PID_NLP=$!

log "nohuman-label-head PID=$PID_NLH, nohuman-label-pseudo PID=$PID_NLP"

# ── Step 5: wait and evaluate next round ─────────────────────────────────────
while kill -0 $PID_NLH 2>/dev/null || kill -0 $PID_NLP 2>/dev/null; do
    NLH=$(grep "Epoch " outputs/nohuman-label-head.log 2>/dev/null | grep -oP "Epoch +\K[0-9]+/[0-9]+" | tail -1)
    NLP=$(grep "Epoch " outputs/nohuman-label-pseudo.log 2>/dev/null | grep -oP "Epoch +\K[0-9]+/[0-9]+" | tail -1)
    log "nohuman-label-head: $NLH | nohuman-label-pseudo: $NLP"
    sleep 120
done

log "Round 2 trainings finished. Running evaluation..."
CUDA_VISIBLE_DEVICES=0 python evaluate_final.py \
    --runs nohuman-label-head nohuman-label-pseudo \
    --gpu 0 \
    >> outputs/evaluate_label_heads_r2.log 2>&1
log "Round 2 evaluation done."
grep -E "Official|Score" outputs/evaluate_label_heads_r2.log | tail -10

log "All done! Check outputs/ for results."
