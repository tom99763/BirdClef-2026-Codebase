#!/usr/bin/env bash
# SSM noisy-student chain (10s clips): rounds 1→4, fully independent.
#
# Pseudo labels: SSM-only predictions → pseudo_labels/ssm_10s_r{k}.csv
# Each round uses only the SSM model's predictions (no Perch mixing).
# SSM uses linear head (not cosine). Inference uses 5s stride → 12 rows per soundscape.
#
# Usage:
#   nohup bash scripts/auto_ssm_ns_10s_full.sh > outputs/logs/auto_ssm_ns_10s_full.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SSM-10s] $*"; }
mkdir -p "$LOG"

for R in 1 2 3 4; do
    log "════════════════ Round ${R} ════════════════"
    OUT_DIR="outputs/ssm-ns-b0-10s-r${R}"
    mkdir -p "$OUT_DIR"

    # ── Train folds 0→4 ──────────────────────────────────────────────────────
    for FOLD in 0 1 2 3 4; do
        CKPT="${OUT_DIR}/fold${FOLD}_best.pt"

        if [ -f "$CKPT" ]; then
            log "Fold ${FOLD} checkpoint exists, skipping"
            continue
        fi

        while pgrep -f "train_ssm_ns.py --config configs/ssm_ns_b0_10s_r${R}.yaml --fold ${FOLD}" > /dev/null 2>&1; do
            log "Fold ${FOLD} in progress, waiting..."
            sleep 60
        done

        if [ -f "$CKPT" ]; then
            log "Fold ${FOLD} complete (waited), skipping"
            continue
        fi

        log "Starting fold ${FOLD}"
        python3 train_ssm_ns.py \
            --config configs/ssm_ns_b0_10s_r${R}.yaml \
            --fold   "$FOLD" \
            --device "$DEVICE" \
            > "${LOG}/ssm_ns_10s_r${R}_fold${FOLD}.log" 2>&1
        log "Fold ${FOLD} done"
    done
    log "All folds complete"

    # ── 5-fold ensemble inference on all soundscapes ──────────────────────────
    log "Running infer_all_ss..."
    python3 train_ssm_ns.py \
        --config       configs/ssm_ns_b0_10s_r${R}.yaml \
        --infer_all_ss \
        --device       "$DEVICE" \
        > "${LOG}/ssm_ns_10s_r${R}_infer.log" 2>&1
    log "infer_all_ss done → ${OUT_DIR}/all_ss_probs.npz"

    # ── Generate SSM-only pseudo labels (skip after final round) ─────────────
    if [ "$R" -lt 4 ]; then
        PSEUDO_OUT="pseudo_labels/ssm_10s_r${R}.csv"
        log "Generating pseudo labels → ${PSEUDO_OUT}"
        python3 scripts/gen_pseudo_ns.py \
            --round   "$R" \
            --ssm_dir "$OUT_DIR" \
            --perch_w 0.0 \
            --ssm_w   1.0 \
            --out     "$PSEUDO_OUT" \
            > "${LOG}/gen_pseudo_ssm_10s_r${R}.log" 2>&1
        log "Pseudo labels saved: ${PSEUDO_OUT}"
    fi

    log "Round ${R} complete"
done

log "════════════════════════════════════════"
log "  SSM-10s NS FULL PIPELINE (R1-R4) COMPLETE"
log "════════════════════════════════════════"
