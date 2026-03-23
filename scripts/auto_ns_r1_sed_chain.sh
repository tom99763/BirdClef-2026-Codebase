#!/usr/bin/env bash
# SED chain folds 1-4 + wait for SSM folds 0-4,
# then both run infer_all_ss, then gen_pseudo
#
# Usage:
#   nohup bash scripts/auto_ns_r1_sed_chain.sh > outputs/logs/auto_ns_r1_sed.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
R=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── SED folds 1→4 ────────────────────────────────────────────────────────────
for FOLD in 1 2 3 4; do
    log "[SED] Starting fold ${FOLD}"
    python3 train_sed_ns.py \
        --config configs/sed_ns_b0_r${R}.yaml \
        --fold   "$FOLD" \
        --device "$DEVICE" \
        > "${LOG}/sed_ns_r${R}_fold${FOLD}.log" 2>&1
    log "[SED] Fold ${FOLD} done"
done
log "[SED] All folds 1-4 complete"

# ── Wait for SSM fold 4 to complete ──────────────────────────────────────────
log "Waiting for SSM fold4 checkpoint..."
while [ ! -f "outputs/ssm-ns-b0-r${R}/fold4_best.pt" ]; do
    sleep 60
done
log "SSM all folds complete"

# ── SED infer_all_ss (5-fold ensemble) ───────────────────────────────────────
log "[SED] Running infer_all_ss (folds 0-4 ensemble)..."
python3 train_sed_ns.py \
    --config       configs/sed_ns_b0_r${R}.yaml \
    --infer_all_ss \
    --device       "$DEVICE" \
    > "${LOG}/sed_ns_r${R}_infer.log" 2>&1
log "[SED] infer_all_ss done → outputs/sed-ns-b0-r${R}/all_ss_probs.npz"

# ── SSM infer_all_ss (5-fold ensemble) ───────────────────────────────────────
log "[SSM] Running infer_all_ss (folds 0-4 ensemble)..."
python3 train_ssm_ns.py \
    --config       configs/ssm_ns_b0_r${R}.yaml \
    --infer_all_ss \
    --device       "$DEVICE" \
    > "${LOG}/ssm_ns_r${R}_infer.log" 2>&1
log "[SSM] infer_all_ss done → outputs/ssm-ns-b0-r${R}/all_ss_probs.npz"

# ── Generate pseudo labels for R2 ────────────────────────────────────────────
log "Generating pseudo labels → pseudo_labels/ns_r${R}.csv"
python3 scripts/gen_pseudo_ns.py \
    --round     "$R" \
    --perch_csv outputs/perch_teacher_all_ss.csv \
    --sed_dir   outputs/sed-ns-b0-r${R} \
    --ssm_dir   outputs/ssm-ns-b0-r${R} \
    --out       pseudo_labels/ns_r${R}.csv \
    > "${LOG}/gen_pseudo_ns_r${R}.log" 2>&1
log "Pseudo labels saved: pseudo_labels/ns_r${R}.csv"

log "========================================"
log "  NS R${R} PIPELINE COMPLETE"
log "========================================"
