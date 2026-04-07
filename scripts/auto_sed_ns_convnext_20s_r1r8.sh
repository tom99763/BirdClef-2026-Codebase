#!/usr/bin/env bash
# SED Noisy-Student ConvNeXt-Tiny (20s clips): rounds 1→8
# Independent backbone series; seeds from B0 R7 pseudo labels.
# Waits for PVT R9-R15 to finish before starting (GPU1 scheduling).
#
# Backbone: convnext_tiny.fb_in22k_ft_in1k (28M params, IN-22K pretrained)
# Output: outputs/sed-ns-cnxt-20s-r{R}/
# Pseudo: pseudo_labels/sed_20s_cnxt_r{N}.csv
#
# Usage:
#   nohup bash scripts/auto_sed_ns_convnext_20s_r1r8.sh > outputs/logs/auto_sed_ns_convnext_20s_r1r8.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

declare -A PSEUDO_CFG
PSEUDO_CFG[1]="0.00 1.00 95 2.00"
PSEUDO_CFG[2]="0.00 1.00 95 2.00"
PSEUDO_CFG[3]="0.00 1.00 95 2.00"
PSEUDO_CFG[4]="0.00 1.00 95 2.00"
PSEUDO_CFG[5]="0.00 1.00 95 2.00"
PSEUDO_CFG[6]="0.00 1.00 95 2.00"
PSEUDO_CFG[7]="0.00 1.00 95 2.00"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SED-CNXT-R1R8] $*"; }
mkdir -p "$LOG" checkpoints

# ── Also wait for B0 R7 pseudo labels (seed) ─────────────────────────────────
log "Waiting for B0 R7 pseudo labels (seed for ConvNeXt R1)..."
while [ ! -f "pseudo_labels/sed_20s_r7.csv" ]; do
    sleep 60
done
log "Seed pseudo labels found"

# ── Functions ─────────────────────────────────────────────────────────────────

train_fold() {
    local R=$1 F=$2
    local CKPT="outputs/sed-ns-cnxt-20s-r${R}/fold${F}_best.pt"
    if [ -f "$CKPT" ]; then
        log "CNXT-R${R} fold${F}: checkpoint exists, skipping"; return 0
    fi
    log "CNXT-R${R} fold${F}: starting"
    $PYTHON train_sed_ns.py \
        --config configs/sed_ns_cnxt_20s_r${R}.yaml \
        --fold   "$F" \
        --device "$DEVICE" \
        > "${LOG}/sed_ns_cnxt_r${R}_fold${F}.log" 2>&1
    log "CNXT-R${R} fold${F}: done"
}

train_residual_corrector() {
    local R=$1
    local CORR_NPZ="outputs/sed-ns-cnxt-20s-r${R}/all_ss_probs_corrected.npz"
    if [ -f "$CORR_NPZ" ]; then
        log "CNXT-R${R}: corrected npz exists, skipping"; return 0
    fi
    [ ! -f "$TEACHER_CSV" ] && { log "CNXT-R${R}: no teacher CSV, skipping corrector"; return 0; }
    log "CNXT-R${R}: training Residual Corrector..."
    $PYTHON scripts/train_sed_residual_corrector.py \
        --sed_dir  "outputs/sed-ns-cnxt-20s-r${R}" \
        --teacher  "$TEACHER_CSV" \
        --round    "$R" \
        --alpha    "$CORRECTOR_ALPHA" \
        --out_ckpt "checkpoints/sed_corrector_cnxt_r${R}.pt" \
        --device   "$DEVICE" \
        > "${LOG}/sed_corrector_cnxt_r${R}.log" 2>&1
    log "CNXT-R${R}: corrector done"
}

gen_pseudo() {
    local R=$1
    local PSEUDO_OUT="pseudo_labels/sed_20s_cnxt_r${R}.csv"
    if [ -f "$PSEUDO_OUT" ]; then
        log "CNXT-R${R}: pseudo labels exist, skipping"; return 0
    fi
    local CFG="${PSEUDO_CFG[$R]}"
    local PERCH_W SED_W THR_PCT GAMMA
    read -r PERCH_W SED_W THR_PCT GAMMA <<< "$CFG"
    log "CNXT-R${R}: gen_pseudo pct=${THR_PCT} gamma=${GAMMA}"

    local ORIG_NPZ="outputs/sed-ns-cnxt-20s-r${R}/all_ss_probs.npz"
    local CORR_NPZ="outputs/sed-ns-cnxt-20s-r${R}/all_ss_probs_corrected.npz"
    local SWAPPED=0
    if [ -f "$CORR_NPZ" ]; then
        cp "$ORIG_NPZ" "outputs/sed-ns-cnxt-20s-r${R}/all_ss_probs_orig.npz"
        cp "$CORR_NPZ" "$ORIG_NPZ"
        SWAPPED=1
    fi

    $PYTHON scripts/gen_pseudo_ns.py \
        --round      "$R" \
        --sed_dir    "outputs/sed-ns-cnxt-20s-r${R}" \
        --perch_w    "$PERCH_W" \
        --sed_w      "$SED_W" \
        --percentile "$THR_PCT" \
        --gamma      "$GAMMA" \
        --nonaves_perch_only \
        --out        "$PSEUDO_OUT" \
        > "${LOG}/gen_pseudo_cnxt_r${R}.log" 2>&1

    if [ "$SWAPPED" -eq 1 ]; then
        cp "outputs/sed-ns-cnxt-20s-r${R}/all_ss_probs_orig.npz" "$ORIG_NPZ"
        rm "outputs/sed-ns-cnxt-20s-r${R}/all_ss_probs_orig.npz"
    fi

    log "CNXT-R${R}: pseudo labels → ${PSEUDO_OUT}"
    local NEXT="configs/sed_ns_cnxt_20s_r$(( R+1 )).yaml"
    [ -f "$NEXT" ] && sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO_OUT}|" "$NEXT"
}

# ── Main loop R1-R8 ───────────────────────────────────────────────────────────
for R in 1 2 3 4 5 6 7 8; do
    log "════════════════ ConvNeXt Round ${R} ════════════════"
    mkdir -p "outputs/sed-ns-cnxt-20s-r${R}"

    for F in 0 1 2 3 4; do
        train_fold "$R" "$F"
    done
    log "CNXT-R${R}: all folds done"

    INFER_NPZ="outputs/sed-ns-cnxt-20s-r${R}/all_ss_probs.npz"
    if [ -f "$INFER_NPZ" ]; then
        log "CNXT-R${R}: all_ss_probs.npz exists, skipping infer"
    else
        log "CNXT-R${R}: running infer_all_ss"
        $PYTHON train_sed_ns.py \
            --config       configs/sed_ns_cnxt_20s_r${R}.yaml \
            --infer_all_ss \
            --device       "$DEVICE" \
            > "${LOG}/sed_ns_cnxt_r${R}_infer.log" 2>&1
        log "CNXT-R${R}: infer done"
    fi

    train_residual_corrector "$R"

    if [ "$R" -lt 8 ]; then
        gen_pseudo "$R"
    fi

    log "CNXT-R${R}: complete"
done

log "SED ConvNeXt-Tiny R1-R8 PIPELINE COMPLETE"
