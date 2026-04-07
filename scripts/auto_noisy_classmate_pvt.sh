#!/usr/bin/env bash
# Noisy Classmate PVT pipeline: PVT R9→R15
# Uses blended pseudo labels from B0 + PVT chains instead of self-training.
#
# Step 1: Complete B0 R12 fold4 if needed
# Step 2: Generate Noisy Classmate pseudo labels (B0 + PVT blend)
# Step 3: Run PVT R9→R15 using blended pseudo labels
#
# Usage:
#   nohup bash scripts/auto_noisy_classmate_pvt.sh > outputs/logs/auto_noisy_classmate_pvt.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

# Noisy Classmate blend weights
NC_B0_W="0.5"
NC_PVT_W="0.5"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NC-PVT] $*"; }
mkdir -p "$LOG" checkpoints

# ── Step 1: Complete B0 R12 fold4 if needed ─────────────────────────────────
B0_R12_F4="outputs/sed-ns-b0-20s-r12/fold4_best.pt"
if [ ! -f "$B0_R12_F4" ]; then
    log "B0 R12 fold4: starting on GPU1"
    $PYTHON train_sed_ns.py \
        --config configs/sed_ns_b0_20s_r12.yaml \
        --fold 4 \
        --device "$DEVICE" \
        > "${LOG}/sed_ns_20s_r12_fold4.log" 2>&1
    log "B0 R12 fold4: done"
else
    log "B0 R12 fold4: exists, skipping"
fi

# ── Step 2: B0 R12 infer (if needed) ────────────────────────────────────────
B0_R12_NPZ="outputs/sed-ns-b0-20s-r12/all_ss_probs.npz"
if [ ! -f "$B0_R12_NPZ" ]; then
    log "B0 R12: running infer_all_ss"
    $PYTHON train_sed_ns.py \
        --config configs/sed_ns_b0_20s_r12.yaml \
        --infer_all_ss \
        --device "$DEVICE" \
        > "${LOG}/sed_ns_20s_r12_infer.log" 2>&1
    log "B0 R12: infer done"
else
    log "B0 R12: npz exists, skipping infer"
fi

# ── Step 2b: B0 R12 Residual Corrector ──────────────────────────────────────
B0_R12_CORR="outputs/sed-ns-b0-20s-r12/all_ss_probs_corrected.npz"
if [ ! -f "$B0_R12_CORR" ]; then
    log "B0 R12: training Residual Corrector"
    $PYTHON scripts/train_sed_residual_corrector.py \
        --sed_dir  "outputs/sed-ns-b0-20s-r12" \
        --teacher  "$TEACHER_CSV" \
        --round    12 \
        --alpha    "$CORRECTOR_ALPHA" \
        --out_ckpt "checkpoints/sed_corrector_r12.pt" \
        --device   "$DEVICE" \
        > "${LOG}/sed_corrector_r12.log" 2>&1
    log "B0 R12: corrector done"
else
    log "B0 R12: corrected npz exists, skipping"
fi

# ── Functions ─────────────────────────────────────────────────────────────────

gen_noisy_classmate_pseudo() {
    local PVT_R=$1
    local B0_DIR=$2
    local PVT_DIR=$3
    local OUT="pseudo_labels/noisy_classmate_pvt_r${PVT_R}.csv"

    if [ -f "$OUT" ]; then
        log "NC pseudo R${PVT_R}: exists, skipping"
        return 0
    fi

    log "NC pseudo R${PVT_R}: blending B0(${B0_DIR}) + PVT(${PVT_DIR})"
    $PYTHON scripts/gen_noisy_classmate_pseudo.py \
        --chains "b0:${B0_DIR}" "pvt:${PVT_DIR}" \
        --weights $NC_B0_W $NC_PVT_W \
        --percentile 95 \
        --gamma 2.00 \
        --out "$OUT" \
        > "${LOG}/gen_nc_pseudo_pvt_r${PVT_R}.log" 2>&1

    log "NC pseudo R${PVT_R}: → ${OUT}"

    # Update PVT config to use Noisy Classmate pseudo labels
    local CFG="configs/sed_ns_pvt_20s_r${PVT_R}.yaml"
    [ -f "$CFG" ] && sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${OUT}|" "$CFG"
}

train_pvt_fold() {
    local R=$1 F=$2
    local CKPT="outputs/sed-ns-pvt-20s-r${R}/fold${F}_best.pt"
    if [ -f "$CKPT" ]; then
        log "PVT-R${R} fold${F}: exists, skipping"
        return 0
    fi
    log "PVT-R${R} fold${F}: starting"
    $PYTHON train_sed_ns.py \
        --config configs/sed_ns_pvt_20s_r${R}.yaml \
        --fold   "$F" \
        --device "$DEVICE" \
        > "${LOG}/sed_ns_pvt_r${R}_fold${F}.log" 2>&1
    log "PVT-R${R} fold${F}: done"
}

train_residual_corrector_pvt() {
    local R=$1
    local CORR_NPZ="outputs/sed-ns-pvt-20s-r${R}/all_ss_probs_corrected.npz"
    if [ -f "$CORR_NPZ" ]; then
        log "PVT-R${R}: corrected npz exists, skipping"
        return 0
    fi
    [ ! -f "$TEACHER_CSV" ] && { log "PVT-R${R}: no teacher CSV, skipping corrector"; return 0; }
    log "PVT-R${R}: training Residual Corrector"
    $PYTHON scripts/train_sed_residual_corrector.py \
        --sed_dir  "outputs/sed-ns-pvt-20s-r${R}" \
        --teacher  "$TEACHER_CSV" \
        --round    "$R" \
        --alpha    "$CORRECTOR_ALPHA" \
        --out_ckpt "checkpoints/sed_corrector_pvt_r${R}.pt" \
        --device   "$DEVICE" \
        > "${LOG}/sed_corrector_pvt_r${R}.log" 2>&1
    log "PVT-R${R}: corrector done"
}

# ── Step 3: Generate first Noisy Classmate pseudo and start PVT R9→R15 ─────

# For PVT R9: blend B0 R12 (latest) + PVT R8
# For PVT R10+: blend B0 R12 + PVT R(N-1)

# B0 source stays fixed at R12 (latest completed B0 round)
B0_LATEST="outputs/sed-ns-b0-20s-r12"

# PVT source starts at R8
PVT_PREV="outputs/sed-ns-pvt-20s-r8"

for R in 9 10 11 12 13 14 15; do
    log "════════════════ Noisy Classmate PVT Round ${R} ════════════════"
    mkdir -p "outputs/sed-ns-pvt-20s-r${R}"

    # Generate Noisy Classmate pseudo labels
    gen_noisy_classmate_pseudo "$R" "$B0_LATEST" "$PVT_PREV"

    # Train all folds
    for F in 0 1 2 3 4; do
        train_pvt_fold "$R" "$F"
    done
    log "PVT-R${R}: all folds done"

    # Infer all soundscapes
    INFER_NPZ="outputs/sed-ns-pvt-20s-r${R}/all_ss_probs.npz"
    if [ -f "$INFER_NPZ" ]; then
        log "PVT-R${R}: npz exists, skipping infer"
    else
        log "PVT-R${R}: running infer_all_ss"
        $PYTHON train_sed_ns.py \
            --config configs/sed_ns_pvt_20s_r${R}.yaml \
            --infer_all_ss \
            --device "$DEVICE" \
            > "${LOG}/sed_ns_pvt_r${R}_infer.log" 2>&1
        log "PVT-R${R}: infer done"
    fi

    # Residual Corrector
    train_residual_corrector_pvt "$R"

    # Update PVT source for next round
    PVT_PREV="outputs/sed-ns-pvt-20s-r${R}"

    log "PVT-R${R}: complete"
done

log "═══════════ NOISY CLASSMATE PVT R9-R15 PIPELINE COMPLETE ═══════════"
