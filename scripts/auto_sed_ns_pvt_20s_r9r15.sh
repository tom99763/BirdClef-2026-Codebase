#!/usr/bin/env bash
# SED Noisy-Student PVT v2 B0 (20s clips): rounds 9→15
# Waits for B0 R9-R15 pipeline to finish, then runs sequentially on GPU1.
#
# Usage:
#   nohup bash scripts/auto_sed_ns_pvt_20s_r9r15.sh > outputs/logs/auto_sed_ns_pvt_20s_r9r15.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

declare -A PSEUDO_CFG
PSEUDO_CFG[8]="0.00 1.00 95 2.00"
PSEUDO_CFG[9]="0.00 1.00 95 2.00"
PSEUDO_CFG[10]="0.00 1.00 95 2.00"
PSEUDO_CFG[11]="0.00 1.00 95 2.00"
PSEUDO_CFG[12]="0.00 1.00 95 2.00"
PSEUDO_CFG[13]="0.00 1.00 95 2.00"
PSEUDO_CFG[14]="0.00 1.00 95 2.00"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SED-PVT-R9R15] $*"; }
mkdir -p "$LOG" checkpoints

# ── Wait for B0 R9-R15 to finish ─────────────────────────────────────────────
log "Waiting for B0 R9-R15 pipeline to finish..."
while pgrep -f "auto_sed_ns_20s_r9r15.sh" > /dev/null 2>&1; do
    sleep 120
done
log "B0 R9-R15 done — starting PVT R9-R15"

# ── Wait for PVT R5-R8 pipeline to also finish (need R8 pseudo) ──────────────
log "Waiting for PVT R5-R8 pipeline (need R8 pseudo labels)..."
while pgrep -f "auto_sed_ns_pvt_20s_r5r8.sh" > /dev/null 2>&1; do
    sleep 120
done
log "PVT R5-R8 done"

# ── Functions ─────────────────────────────────────────────────────────────────

train_fold() {
    local R=$1 F=$2
    local CKPT="outputs/sed-ns-pvt-20s-r${R}/fold${F}_best.pt"
    if [ -f "$CKPT" ]; then
        log "PVT-R${R} fold${F}: checkpoint exists, skipping"; return 0
    fi
    log "PVT-R${R} fold${F}: starting"
    $PYTHON train_sed_ns.py \
        --config configs/sed_ns_pvt_20s_r${R}.yaml \
        --fold   "$F" \
        --device "$DEVICE" \
        > "${LOG}/sed_ns_pvt_r${R}_fold${F}.log" 2>&1
    log "PVT-R${R} fold${F}: done"
}

train_residual_corrector() {
    local R=$1
    local CORR_NPZ="outputs/sed-ns-pvt-20s-r${R}/all_ss_probs_corrected.npz"
    if [ -f "$CORR_NPZ" ]; then
        log "PVT-R${R}: corrected npz exists, skipping"; return 0
    fi
    [ ! -f "$TEACHER_CSV" ] && { log "PVT-R${R}: no teacher CSV, skipping corrector"; return 0; }
    log "PVT-R${R}: training Residual Corrector..."
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

gen_pseudo() {
    local R=$1
    local PSEUDO_OUT="pseudo_labels/sed_20s_pvt_r${R}.csv"
    if [ -f "$PSEUDO_OUT" ]; then
        log "PVT-R${R}: pseudo labels exist, skipping"; return 0
    fi
    local CFG="${PSEUDO_CFG[$R]}"
    local PERCH_W SED_W THR_PCT GAMMA
    read -r PERCH_W SED_W THR_PCT GAMMA <<< "$CFG"
    log "PVT-R${R}: gen_pseudo pct=${THR_PCT} gamma=${GAMMA}"

    local ORIG_NPZ="outputs/sed-ns-pvt-20s-r${R}/all_ss_probs.npz"
    local CORR_NPZ="outputs/sed-ns-pvt-20s-r${R}/all_ss_probs_corrected.npz"
    local SWAPPED=0
    if [ -f "$CORR_NPZ" ]; then
        cp "$ORIG_NPZ" "outputs/sed-ns-pvt-20s-r${R}/all_ss_probs_orig.npz"
        cp "$CORR_NPZ" "$ORIG_NPZ"
        SWAPPED=1
    fi

    $PYTHON scripts/gen_pseudo_ns.py \
        --round      "$R" \
        --sed_dir    "outputs/sed-ns-pvt-20s-r${R}" \
        --perch_w    "$PERCH_W" \
        --sed_w      "$SED_W" \
        --percentile "$THR_PCT" \
        --gamma      "$GAMMA" \
        --nonaves_perch_only \
        --out        "$PSEUDO_OUT" \
        > "${LOG}/gen_pseudo_pvt_r${R}.log" 2>&1

    if [ "$SWAPPED" -eq 1 ]; then
        cp "outputs/sed-ns-pvt-20s-r${R}/all_ss_probs_orig.npz" "$ORIG_NPZ"
        rm "outputs/sed-ns-pvt-20s-r${R}/all_ss_probs_orig.npz"
    fi

    log "PVT-R${R}: pseudo labels → ${PSEUDO_OUT}"
    local NEXT="configs/sed_ns_pvt_20s_r$(( R+1 )).yaml"
    [ -f "$NEXT" ] && sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO_OUT}|" "$NEXT"
}

# ── Generate PVT R8 pseudo labels first ──────────────────────────────────────
log "Generating PVT R8 pseudo labels (prerequisite for R9)..."
gen_pseudo 8

# ── Main loop R9-R15 ─────────────────────────────────────────────────────────
for R in 9 10 11 12 13 14 15; do
    log "════════════════ PVT Round ${R} ════════════════"
    mkdir -p "outputs/sed-ns-pvt-20s-r${R}"

    for F in 0 1 2 3 4; do
        train_fold "$R" "$F"
    done
    log "PVT-R${R}: all folds done"

    INFER_NPZ="outputs/sed-ns-pvt-20s-r${R}/all_ss_probs.npz"
    if [ -f "$INFER_NPZ" ]; then
        log "PVT-R${R}: all_ss_probs.npz exists, skipping infer"
    else
        log "PVT-R${R}: running infer_all_ss"
        $PYTHON train_sed_ns.py \
            --config       configs/sed_ns_pvt_20s_r${R}.yaml \
            --infer_all_ss \
            --device       "$DEVICE" \
            > "${LOG}/sed_ns_pvt_r${R}_infer.log" 2>&1
        log "PVT-R${R}: infer done"
    fi

    train_residual_corrector "$R"

    if [ "$R" -lt 15 ]; then
        gen_pseudo "$R"
    fi

    log "PVT-R${R}: complete"
done

log "SED PVT R9-R15 PIPELINE COMPLETE"
