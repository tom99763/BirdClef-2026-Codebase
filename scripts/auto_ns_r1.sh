#!/usr/bin/env bash
# Auto-orchestrator: NS R1
#
# SED chain: folds 1→4 (fold0 already done)
# SSM chain: folds 0→4 (restart with EMA + stable temp lr)
# Each chain runs independently in parallel.
# Sync point: both chains complete → infer_all_ss → gen_pseudo → ns_r1.csv
#
# Usage:
#   nohup bash scripts/auto_ns_r1.sh > outputs/logs/auto_ns_r1.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
R=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
mkdir -p "$LOG" outputs/sed-ns-b0-r${R} outputs/ssm-ns-b0-r${R}

# ── SED chain: folds 0→4 ─────────────────────────────────────────────────────
run_sed_chain() {
    for FOLD in 0 1 2 3 4; do
        log "[SED] Starting fold ${FOLD}"
        python3 train_sed_ns.py \
            --config configs/sed_ns_b0_r${R}.yaml \
            --fold   "$FOLD" \
            --device "$DEVICE" \
            > "${LOG}/sed_ns_r${R}_fold${FOLD}.log" 2>&1
        log "[SED] Fold ${FOLD} done"
    done
    log "[SED] All folds complete"
}

# ── SSM chain: folds 0→4 (full restart with EMA prototypes) ─────────────────
run_ssm_chain() {
    for FOLD in 0 1 2 3 4; do
        log "[SSM] Starting fold ${FOLD}"
        python3 train_ssm_ns.py \
            --config configs/ssm_ns_b0_r${R}.yaml \
            --fold   "$FOLD" \
            --device "$DEVICE" \
            > "${LOG}/ssm_ns_r${R}_fold${FOLD}.log" 2>&1
        log "[SSM] Fold ${FOLD} done"
    done
    log "[SSM] All folds complete"
}

# Launch both chains in parallel
run_sed_chain &
SED_CHAIN_PID=$!
run_ssm_chain &
SSM_CHAIN_PID=$!

log "SED chain PID=$SED_CHAIN_PID (folds 0-4)"
log "SSM chain PID=$SSM_CHAIN_PID (folds 0-4)"

wait $SED_CHAIN_PID || log "WARNING: SED chain exited non-zero"
wait $SSM_CHAIN_PID || log "WARNING: SSM chain exited non-zero"

log "========================================"
log "  Both chains done. Running infer_all_ss"
log "========================================"
python3 train_sed_ns.py \
    --config       configs/sed_ns_b0_r${R}.yaml \
    --infer_all_ss \
    --device       "$DEVICE" \
    > "${LOG}/sed_ns_r${R}_infer.log" 2>&1
log "  Inference done."

log "========================================"
log "  Generating pseudo labels → ns_r${R}.csv"
log "========================================"
python3 scripts/gen_pseudo_ns.py \
    --round     "$R" \
    --perch_csv outputs/perch_teacher_all_ss.csv \
    --sed_dir   outputs/sed-ns-b0-r${R} \
    --ssm_dir   outputs/ssm-ns-b0-r${R} \
    --out       pseudo_labels/ns_r${R}.csv \
    > "${LOG}/gen_pseudo_ns_r${R}.log" 2>&1
log "  Pseudo labels saved: pseudo_labels/ns_r${R}.csv"

log "========================================"
log "  NS R${R} PIPELINE COMPLETE"
log "========================================"
