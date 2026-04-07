#!/usr/bin/env bash
# Competitor SED Knowledge Distillation pipeline
# Step 1: Generate competitor pseudo labels on train_audio
# Step 2: Train student SED with KD loss (5-fold)
#
# Usage:
#   nohup bash scripts/run_distill_competitor.sh > outputs/logs/distill_competitor.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
mkdir -p "$LOG" "outputs/competitor_pseudo"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DISTILL-COMP] $*"; }

# ── Step 1: Generate teacher pseudo labels ────────────────────────────────────
PSEUDO_NPZ="outputs/competitor_pseudo/train_audio_probs.npz"
if [ -f "$PSEUDO_NPZ" ]; then
    log "Pseudo labels already exist: $PSEUDO_NPZ — skipping gen"
else
    log "Step 1: Generating competitor SED pseudo labels on train_audio …"
    python3 scripts/gen_competitor_pseudo.py \
        --audio_dir birdclef-2026/train_audio \
        --taxonomy  birdclef-2026/taxonomy.csv \
        --out       "$PSEUDO_NPZ" \
        --batch_size 64 \
        --device    "$DEVICE" \
        > "${LOG}/gen_competitor_pseudo.log" 2>&1
    log "Step 1: Done → $PSEUDO_NPZ"
fi

# ── Step 2: Train 5-fold student SED with KD ─────────────────────────────────
log "Step 2: Training student SED (5-fold KD) …"
python3 train_distill_competitor.py \
    --config configs/distill_competitor_b0_v1.yaml \
    --device "$DEVICE" \
    > "${LOG}/distill_competitor_train.log" 2>&1

log "Pipeline complete. Check outputs/distill-competitor-b0-v1/result.json"
