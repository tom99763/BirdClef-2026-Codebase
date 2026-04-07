#!/usr/bin/env bash
# SED Noisy-Student V2 series (20s clips): rounds 1→4
# Forum insights applied:
#   - Wave-level mixup only (use_sumix_freq: false)
#   - Fewer epochs (25) to avoid overconfidence
#   - Attention map visualization after each round
#
# Waits for R8 of original series to complete before starting.
#
# Usage:
#   nohup bash scripts/auto_sed_ns_20s_v2_r1r4.sh > outputs/logs/auto_sed_ns_20s_v2_r1r4.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

# Per-round pseudo label config: "perch_w sed_w threshold_pct gamma"
declare -A PSEUDO_CFG
PSEUDO_CFG[1]="0.00 1.00 95 2.00"
PSEUDO_CFG[2]="0.00 1.00 95 2.00"
PSEUDO_CFG[3]="0.00 1.00 95 2.00"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SED-20s-V2] $*"; }
mkdir -p "$LOG" checkpoints

# ── Wait for R8 to finish ──────────────────────────────────────────────────────

wait_for_r8() {
    log "Waiting for SED NS R8 to finish..."
    local R8_NPZ="outputs/sed-ns-b0-20s-r8/all_ss_probs.npz"
    local R8_PSEUDO="pseudo_labels/sed_20s_r8.csv"

    while true; do
        if [ -f "$R8_PSEUDO" ]; then
            log "R8 pseudo labels found → starting V2 series"
            break
        fi
        if pgrep -f "auto_sed_ns_20s_r5r8.sh" > /dev/null 2>&1; then
            log "R5-R8 pipeline still running... sleeping 5 min"
        else
            log "R5-R8 pipeline not running. Checking for R8 pseudo..."
            if [ ! -f "$R8_PSEUDO" ]; then
                log "WARNING: R8 pseudo labels not found and pipeline not running. Waiting..."
            fi
        fi
        sleep 300
    done
    sleep 10
}

# ── Functions ─────────────────────────────────────────────────────────────────

train_fold() {
    local R=$1 F=$2
    local CKPT="outputs/sed-ns-b0-20s-v2-r${R}/fold${F}_best.pt"
    if [ -f "$CKPT" ]; then
        log "V2-R${R} fold${F}: checkpoint exists, skipping"
        return 0
    fi
    log "V2-R${R} fold${F}: starting"
    $PYTHON train_sed_ns.py \
        --config configs/sed_ns_b0_20s_v2_r${R}.yaml \
        --fold   "$F" \
        --device "$DEVICE" \
        > "${LOG}/sed_ns_20s_v2_r${R}_fold${F}.log" 2>&1
    log "V2-R${R} fold${F}: done"
}

visualize_attention() {
    local R=$1 F=$2
    local ATTN_DIR="outputs/sed-ns-b0-20s-v2-r${R}/attention_maps/fold${F}"
    if [ -d "$ATTN_DIR" ] && [ "$(ls -A "$ATTN_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
        log "V2-R${R} fold${F}: attention maps exist, skipping"
        return 0
    fi
    log "V2-R${R} fold${F}: generating attention maps..."
    $PYTHON scripts/visualize_sed_attention.py \
        --config  configs/sed_ns_b0_20s_v2_r${R}.yaml \
        --fold    "$F" \
        --out_dir "$ATTN_DIR" \
        --n_worst 20 \
        > "${LOG}/attn_v2_r${R}_fold${F}.log" 2>&1
    log "V2-R${R} fold${F}: attention maps saved to ${ATTN_DIR}"
}

train_residual_corrector() {
    local R=$1
    local CORR_NPZ="outputs/sed-ns-b0-20s-v2-r${R}/all_ss_probs_corrected.npz"
    if [ -f "$CORR_NPZ" ]; then
        log "V2-R${R}: corrected npz exists, skipping"
        return 0
    fi
    if [ ! -f "$TEACHER_CSV" ]; then
        log "V2-R${R}: teacher CSV not found, skipping corrector"
        return 0
    fi
    log "V2-R${R}: training Temporal Residual Corrector..."
    $PYTHON scripts/train_sed_residual_corrector.py \
        --sed_dir  "outputs/sed-ns-b0-20s-v2-r${R}" \
        --teacher  "$TEACHER_CSV" \
        --round    "$R" \
        --alpha    "$CORRECTOR_ALPHA" \
        --out_ckpt "checkpoints/sed_corrector_v2_r${R}.pt" \
        --device   "$DEVICE" \
        > "${LOG}/sed_corrector_v2_r${R}.log" 2>&1
    log "V2-R${R}: corrector done"
}

gen_pseudo() {
    local R=$1
    local PSEUDO_OUT="pseudo_labels/sed_20s_v2_r${R}.csv"
    if [ -f "$PSEUDO_OUT" ]; then
        log "V2-R${R}: pseudo labels exist, skipping"
        return 0
    fi

    local CFG="${PSEUDO_CFG[$R]}"
    local PERCH_W SED_W THR_PCT GAMMA
    read -r PERCH_W SED_W THR_PCT GAMMA <<< "$CFG"

    log "V2-R${R}: gen_pseudo — perch_w=${PERCH_W} sed_w=${SED_W} pct=${THR_PCT} gamma=${GAMMA}"

    local ORIG_NPZ="outputs/sed-ns-b0-20s-v2-r${R}/all_ss_probs.npz"
    local CORR_NPZ="outputs/sed-ns-b0-20s-v2-r${R}/all_ss_probs_corrected.npz"
    local SWAPPED=0
    if [ -f "$CORR_NPZ" ]; then
        local BACKUP_NPZ="outputs/sed-ns-b0-20s-v2-r${R}/all_ss_probs_orig.npz"
        cp "$ORIG_NPZ" "$BACKUP_NPZ"
        cp "$CORR_NPZ" "$ORIG_NPZ"
        SWAPPED=1
        log "V2-R${R}: using corrected probs"
    fi

    $PYTHON scripts/gen_pseudo_ns.py \
        --round      "$R" \
        --sed_dir    "outputs/sed-ns-b0-20s-v2-r${R}" \
        --perch_w    "$PERCH_W" \
        --sed_w      "$SED_W" \
        --percentile "$THR_PCT" \
        --gamma      "$GAMMA" \
        --nonaves_perch_only \
        --out        "$PSEUDO_OUT" \
        > "${LOG}/gen_pseudo_sed_20s_v2_r${R}.log" 2>&1

    if [ "$SWAPPED" -eq 1 ] && [ -f "$BACKUP_NPZ" ]; then
        cp "$BACKUP_NPZ" "$ORIG_NPZ"
        rm "$BACKUP_NPZ"
    fi

    log "V2-R${R}: pseudo labels → ${PSEUDO_OUT}"

    local NEXT="configs/sed_ns_b0_20s_v2_r$(( R+1 )).yaml"
    [ -f "$NEXT" ] && sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO_OUT}|" "$NEXT"
}

# ── Main ──────────────────────────────────────────────────────────────────────

wait_for_r8

for R in 1 2 3 4; do
    log "════════════════ V2 Round ${R} ════════════════"
    mkdir -p "outputs/sed-ns-b0-20s-v2-r${R}"

    for F in 0 1 2 3 4; do
        train_fold "$R" "$F"
    done
    log "V2-R${R}: all folds done"

    # Visualize attention maps for fold 0 (representative)
    visualize_attention "$R" 0

    # Infer all soundscapes
    INFER_NPZ="outputs/sed-ns-b0-20s-v2-r${R}/all_ss_probs.npz"
    if [ -f "$INFER_NPZ" ]; then
        log "V2-R${R}: all_ss_probs.npz exists, skipping infer"
    else
        log "V2-R${R}: running infer_all_ss"
        $PYTHON train_sed_ns.py \
            --config       configs/sed_ns_b0_20s_v2_r${R}.yaml \
            --infer_all_ss \
            --device       "$DEVICE" \
            > "${LOG}/sed_ns_20s_v2_r${R}_infer.log" 2>&1
        log "V2-R${R}: infer done"
    fi

    # Residual Corrector
    train_residual_corrector "$R"

    # Generate pseudo labels for next round (skip R4)
    if [ "$R" -lt 4 ]; then
        gen_pseudo "$R"
    fi

    log "V2-R${R}: complete"
done

log "SED-20s V2 R1-R4 PIPELINE COMPLETE"
