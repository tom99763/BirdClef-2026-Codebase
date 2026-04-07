#!/usr/bin/env bash
# Master pipeline: Perch head retrain (with aug) → teacher predictions
#                 → pseudo labels → SED-NS 20s sequential chain (R1-R4)
#
# Usage:
#   nohup bash scripts/run_perch_aug_sed_chain.sh > outputs/logs/perch_aug_sed_chain.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
mkdir -p "$LOG" pseudo_labels outputs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AUG-CHAIN] $*"; }

# ── Step 1: Re-finetune Perch head WITH augmentation ─────────────────────────
AUG_CKPT="checkpoints/perch-head-retrain-aug/best_head.weights.h5"
if [ -f "$AUG_CKPT" ]; then
    log "Augmented Perch head already exists: ${AUG_CKPT} — skipping"
else
    log "Training Perch head with augmentation ..."
    python3 train.py \
        --config configs/perch_head_retrain_aug.yaml \
        > "${LOG}/perch_head_retrain_aug.log" 2>&1
    log "Perch head training done → ${AUG_CKPT}"
fi

# ── Step 2: Extract teacher predictions on all soundscapes ───────────────────
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
if [ -f "$TEACHER_CSV" ]; then
    log "Teacher predictions already exist: ${TEACHER_CSV} — skipping"
else
    log "Extracting Perch teacher predictions (~2h) ..."
    python3 scripts/extract_perch_teacher_all_ss.py \
        --output    "$TEACHER_CSV" \
        --config    configs/perch_head_retrain_aug.yaml \
        --checkpoint "$AUG_CKPT" \
        --batch_size 32 \
        > "${LOG}/extract_perch_teacher_aug.log" 2>&1
    log "Teacher extraction done → ${TEACHER_CSV}"
fi

ROWS=$(wc -l < "$TEACHER_CSV" 2>/dev/null || echo 0)
log "Teacher CSV: ${ROWS} rows"

# ── Step 3: Generate round-0 pseudo labels ───────────────────────────────────
PSEUDO_R0="pseudo_labels/ns_r0_perch_aug.csv"
if [ -f "$PSEUDO_R0" ]; then
    log "Round-0 pseudo labels already exist: ${PSEUDO_R0} — skipping"
else
    log "Generating pseudo labels from Perch teacher ..."
    python3 scripts/gen_pseudo_ns.py \
        --round     0 \
        --clip_sec  20 \
        --perch_csv "$TEACHER_CSV" \
        --out       "$PSEUDO_R0" \
        > "${LOG}/gen_pseudo_r0_aug.log" 2>&1
    log "Round-0 pseudo labels → ${PSEUDO_R0}"
fi

PROWS=$(wc -l < "$PSEUDO_R0" 2>/dev/null || echo 0)
log "ns_r0_perch_aug.csv: ${PROWS} rows"

# Patch R1 config to use new pseudo labels
sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO_R0}|" \
    configs/sed_ns_b0_20s_r1.yaml
log "configs/sed_ns_b0_20s_r1.yaml updated → ${PSEUDO_R0}"

# ── Step 4: SED-only noisy student chain (sequential folds) ──────────────────
log "Launching SED-20s NS chain (R1-R4, folds sequential) ..."
bash scripts/auto_sed_ns_20s_full.sh \
    >> "${LOG}/auto_sed_ns_20s_full.log" 2>&1

log "════════════════════════════════════════"
log "  PERCH-AUG → SED CHAIN COMPLETE"
log "════════════════════════════════════════"
