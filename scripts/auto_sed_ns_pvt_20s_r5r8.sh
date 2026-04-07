#!/usr/bin/env bash
# SED Noisy-Student PVT v2 B0 series (20s clips): rounds 5→8
# Continues from auto_sed_ns_pvt_20s_r1r4.sh (requires R4 pseudo labels)
#
# Usage:
#   nohup bash scripts/auto_sed_ns_pvt_20s_r5r8.sh > outputs/logs/auto_sed_ns_pvt_20s_r5r8.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

declare -A PSEUDO_CFG
PSEUDO_CFG[5]="0.00 1.00 95 2.00"
PSEUDO_CFG[6]="0.00 1.00 95 2.00"
PSEUDO_CFG[7]="0.00 1.00 95 2.00"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SED-PVT-R5R8] $*"; }
mkdir -p "$LOG" checkpoints

# ── Functions ─────────────────────────────────────────────────────────────────

train_fold() {
    local R=$1 F=$2
    local CKPT="outputs/sed-ns-pvt-20s-r${R}/fold${F}_best.pt"
    if [ -f "$CKPT" ]; then
        log "PVT-R${R} fold${F}: checkpoint exists, skipping"
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

visualize_attention() {
    local R=$1 F=$2
    local ATTN_DIR="outputs/sed-ns-pvt-20s-r${R}/attention_maps/fold${F}"
    if [ -d "$ATTN_DIR" ] && [ "$(ls -A "$ATTN_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
        log "PVT-R${R} fold${F}: attention maps exist, skipping"
        return 0
    fi
    log "PVT-R${R} fold${F}: generating attention maps..."
    $PYTHON scripts/visualize_sed_attention.py \
        --config  configs/sed_ns_pvt_20s_r${R}.yaml \
        --fold    "$F" \
        --out_dir "$ATTN_DIR" \
        --n_worst 20 \
        > "${LOG}/attn_pvt_r${R}_fold${F}.log" 2>&1
    log "PVT-R${R} fold${F}: attention maps → ${ATTN_DIR}"
}

train_residual_corrector() {
    local R=$1
    local CORR_NPZ="outputs/sed-ns-pvt-20s-r${R}/all_ss_probs_corrected.npz"
    if [ -f "$CORR_NPZ" ]; then
        log "PVT-R${R}: corrected npz exists, skipping"; return 0
    fi
    if [ ! -f "$TEACHER_CSV" ]; then
        log "PVT-R${R}: no teacher CSV, skipping corrector"; return 0
    fi
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

# ── Main ──────────────────────────────────────────────────────────────────────

# Wait for R4 pseudo labels before starting
log "Waiting for PVT R4 pseudo labels (pseudo_labels/sed_20s_pvt_r4.csv)..."
while [ ! -f "pseudo_labels/sed_20s_pvt_r4.csv" ]; do
    sleep 60
done
log "PVT R4 pseudo labels found — starting R5"

for R in 5 6 7 8; do
    log "════════════════ PVT Round ${R} ════════════════"
    mkdir -p "outputs/sed-ns-pvt-20s-r${R}"

    for F in 0 1 2 3 4; do
        train_fold "$R" "$F"
    done
    log "PVT-R${R}: all folds done"

    visualize_attention "$R" 0

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

    if [ "$R" -lt 8 ]; then
        gen_pseudo "$R"
    fi

    log "PVT-R${R}: complete"
done

log "SED PVT v2 B0 R5-R8 PIPELINE COMPLETE"
