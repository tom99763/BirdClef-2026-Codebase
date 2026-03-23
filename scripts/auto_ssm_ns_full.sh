#!/usr/bin/env bash
# SSM noisy-student chain: rounds 1→4, fully independent (no SED dependency).
#
# Pseudo labels: SSM-only predictions → pseudo_labels/ssm_r{k}.csv
# Each round uses only the SSM model's predictions (no Perch mixing).
#
# Skip logic: checkpoint exists → skip immediately.
# Wait logic: checkpoint missing but process running → wait for it to finish.
# Only waits for SSM processes; never waits for SED.
#
# Usage:
#   nohup bash scripts/auto_ssm_ns_full.sh > outputs/logs/auto_ssm_ns_full.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SSM] $*"; }
mkdir -p "$LOG"

for R in 1 2 3 4; do
    log "════════════════ Round ${R} ════════════════"
    OUT_DIR="outputs/ssm-ns-b0-r${R}"
    mkdir -p "$OUT_DIR"

    # ── Train folds 0→4 ──────────────────────────────────────────────────────
    for FOLD in 0 1 2 3 4; do
        CKPT="${OUT_DIR}/fold${FOLD}_best.pt"

        # 1. Checkpoint exists → skip immediately
        if [ -f "$CKPT" ]; then
            log "Fold ${FOLD} checkpoint exists, skipping"
            continue
        fi

        # 2. SSM process for this exact fold is running → wait for it
        while pgrep -f "train_ssm_ns.py --config configs/ssm_ns_b0_r${R}.yaml --fold ${FOLD}" > /dev/null 2>&1; do
            log "Fold ${FOLD} in progress, waiting..."
            sleep 60
        done

        # 3. Re-check after wait (process may have written checkpoint)
        if [ -f "$CKPT" ]; then
            log "Fold ${FOLD} complete (waited), skipping"
            continue
        fi

        log "Starting fold ${FOLD}"
        python3 train_ssm_ns.py \
            --config configs/ssm_ns_b0_r${R}.yaml \
            --fold   "$FOLD" \
            --device "$DEVICE" \
            > "${LOG}/ssm_ns_r${R}_fold${FOLD}.log" 2>&1
        log "Fold ${FOLD} done"
    done
    log "All folds complete"

    # ── 5-fold ensemble inference on all soundscapes ──────────────────────────
    log "Running infer_all_ss..."
    python3 train_ssm_ns.py \
        --config       configs/ssm_ns_b0_r${R}.yaml \
        --infer_all_ss \
        --device       "$DEVICE" \
        > "${LOG}/ssm_ns_r${R}_infer.log" 2>&1
    log "infer_all_ss done → ${OUT_DIR}/all_ss_probs.npz"

    # ── Generate SSM-only pseudo labels (skip after final round) ─────────────
    if [ "$R" -lt 4 ]; then
        PSEUDO_OUT="pseudo_labels/ssm_r${R}.csv"
        log "Generating pseudo labels → ${PSEUDO_OUT}"
        python3 scripts/gen_pseudo_ns.py \
            --round   "$R" \
            --ssm_dir "$OUT_DIR" \
            --perch_w 0.0 \
            --ssm_w   1.0 \
            --out     "$PSEUDO_OUT" \
            > "${LOG}/gen_pseudo_ssm_r${R}.log" 2>&1
        log "Pseudo labels saved: ${PSEUDO_OUT}"
    fi

    log "Round ${R} complete"
done

log "════════════════════════════════════════"
log "  SSM NS FULL PIPELINE (R1-R4) COMPLETE"
log "════════════════════════════════════════"
