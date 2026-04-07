#!/usr/bin/env bash
# SED noisy-student chain (20s clips): rounds 5→8, 接續 R4 完成後繼續。
# 等待 CLAP 訓練完成（PID 檔或 log 結尾標記）後自動啟動。
#
# Round | perch_w | sed_w | threshold_pct | gamma
#   5   |   0.05  |  0.95 |     94        | 2.00
#   6   |   0.02  |  0.98 |     95        | 2.00
#   7   |   0.00  |  1.00 |     95        | 2.00
#   (R8 is last round, no pseudo needed)
#
# Usage:
#   nohup bash scripts/auto_sed_ns_20s_r5r8.sh > outputs/logs/auto_sed_ns_20s_r5r8.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"
CLAP_LOG="${LOG}/clap_v2_supcon_stage1.log"
CLAP_PID_FILE="/tmp/clap_stage1.pid"

# Per-round pseudo label config: "perch_w sed_w threshold_pct gamma"
declare -A PSEUDO_CFG
PSEUDO_CFG[5]="0.05 0.95 94 2.00"
PSEUDO_CFG[6]="0.02 0.98 95 2.00"
PSEUDO_CFG[7]="0.00 1.00 95 2.00"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SED-20s-R5R8] $*"; }
mkdir -p "$LOG" checkpoints

# ── Wait for CLAP to finish ────────────────────────────────────────────────────

wait_for_clap() {
    log "Waiting for CLAP Stage 1 to finish..."
    while true; do
        # Check if CLAP process is still running
        if pgrep -f "train_clap.py" > /dev/null 2>&1; then
            log "CLAP still running... sleeping 5 min"
            sleep 300
        else
            log "CLAP process not found — assuming finished"
            break
        fi

        # Also check log for completion marker
        if [ -f "$CLAP_LOG" ] && grep -q "Stage 1 best checkpoint\|Stage 2 best checkpoint\|stage.*complete\|Epoch.*best" "$CLAP_LOG" 2>/dev/null; then
            log "CLAP completion marker found in log"
            break
        fi
    done
    log "CLAP finished — starting SED NS R5-R8"
    sleep 10  # brief pause before starting
}

# ── Functions ─────────────────────────────────────────────────────────────────

train_fold() {
    local R=$1 F=$2
    local CKPT="outputs/sed-ns-b0-20s-r${R}/fold${F}_best.pt"
    if [ -f "$CKPT" ]; then
        log "R${R} fold${F}: checkpoint exists, skipping"
        return 0
    fi
    log "R${R} fold${F}: starting"
    $PYTHON train_sed_ns.py \
        --config configs/sed_ns_b0_20s_r${R}.yaml \
        --fold   "$F" \
        --device "$DEVICE" \
        > "${LOG}/sed_ns_20s_r${R}_fold${F}.log" 2>&1
    log "R${R} fold${F}: done"
}

train_residual_corrector() {
    local R=$1
    local CORR_NPZ="outputs/sed-ns-b0-20s-r${R}/all_ss_probs_corrected.npz"

    if [ -f "$CORR_NPZ" ]; then
        log "R${R}: corrected npz already exists, skipping corrector"
        return 0
    fi
    if [ ! -f "$TEACHER_CSV" ]; then
        log "R${R}: teacher CSV not found, skipping corrector"
        return 0
    fi

    log "R${R}: training Temporal Residual Corrector ..."
    $PYTHON scripts/train_sed_residual_corrector.py \
        --sed_dir   "outputs/sed-ns-b0-20s-r${R}" \
        --teacher   "$TEACHER_CSV" \
        --round     "$R" \
        --alpha     "$CORRECTOR_ALPHA" \
        --out_ckpt  "checkpoints/sed_corrector_r${R}.pt" \
        --device    "$DEVICE" \
        > "${LOG}/sed_corrector_r${R}.log" 2>&1
    log "R${R}: corrector done → ${CORR_NPZ}"
}

gen_pseudo() {
    local R=$1
    local PSEUDO_OUT="pseudo_labels/sed_20s_r${R}.csv"

    if [ -f "$PSEUDO_OUT" ]; then
        log "R${R}: pseudo labels already exist, skipping"
        return 0
    fi

    local CFG="${PSEUDO_CFG[$R]}"
    local PERCH_W SED_W THR_PCT GAMMA
    read -r PERCH_W SED_W THR_PCT GAMMA <<< "$CFG"

    log "R${R}: gen_pseudo — perch_w=${PERCH_W} sed_w=${SED_W} pct=${THR_PCT} gamma=${GAMMA}"

    # Use corrected npz if available
    local ORIG_NPZ="outputs/sed-ns-b0-20s-r${R}/all_ss_probs.npz"
    local CORR_NPZ="outputs/sed-ns-b0-20s-r${R}/all_ss_probs_corrected.npz"
    local SWAPPED=0
    if [ -f "$CORR_NPZ" ]; then
        local BACKUP_NPZ="outputs/sed-ns-b0-20s-r${R}/all_ss_probs_orig.npz"
        cp "$ORIG_NPZ" "$BACKUP_NPZ"
        cp "$CORR_NPZ" "$ORIG_NPZ"
        SWAPPED=1
        log "R${R}: using corrected probs for pseudo labels"
    fi

    local PERCH_ARG=""
    [ -f "$TEACHER_CSV" ] && PERCH_ARG="--perch_csv ${TEACHER_CSV}"
    [ "$PERCH_W" = "0.00" ] && PERCH_ARG="" && PERCH_W="0.0" && SED_W="1.0"

    $PYTHON scripts/gen_pseudo_ns.py \
        --round      "$R" \
        --sed_dir    "outputs/sed-ns-b0-20s-r${R}" \
        --perch_w    "$PERCH_W" \
        --sed_w      "$SED_W" \
        --percentile "$THR_PCT" \
        --gamma      "$GAMMA" \
        $PERCH_ARG \
        --nonaves_perch_only \
        --out        "$PSEUDO_OUT" \
        > "${LOG}/gen_pseudo_sed_20s_r${R}.log" 2>&1

    # Restore original npz
    if [ "$SWAPPED" -eq 1 ] && [ -f "$BACKUP_NPZ" ]; then
        cp "$BACKUP_NPZ" "$ORIG_NPZ"
        rm "$BACKUP_NPZ"
    fi

    log "R${R}: pseudo labels → ${PSEUDO_OUT}"

    # Update next round config
    local NEXT="configs/sed_ns_b0_20s_r$(( R+1 )).yaml"
    [ -f "$NEXT" ] && sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO_OUT}|" "$NEXT"
}

# ── Main ──────────────────────────────────────────────────────────────────────

wait_for_clap

for R in 5 6 7 8; do
    log "════════════════ Round ${R} ════════════════"
    mkdir -p "outputs/sed-ns-b0-20s-r${R}"

    for F in 0 1 2 3 4; do
        train_fold "$R" "$F"
    done
    log "R${R}: all folds done"

    # Infer all soundscapes
    INFER_NPZ="outputs/sed-ns-b0-20s-r${R}/all_ss_probs.npz"
    if [ -f "$INFER_NPZ" ]; then
        log "R${R}: all_ss_probs.npz exists, skipping infer_all_ss"
    else
        log "R${R}: running infer_all_ss"
        $PYTHON train_sed_ns.py \
            --config       configs/sed_ns_b0_20s_r${R}.yaml \
            --infer_all_ss \
            --device       "$DEVICE" \
            > "${LOG}/sed_ns_20s_r${R}_infer.log" 2>&1
        log "R${R}: infer done"
    fi

    # Residual Corrector
    train_residual_corrector "$R"

    # Generate pseudo labels for next round (skip R8)
    if [ "$R" -lt 8 ]; then
        gen_pseudo "$R"
    fi

    log "R${R}: complete"
done

# Generate final pseudo labels from R8
log "Generating final pseudo labels from R8..."
$PYTHON scripts/gen_pseudo_ns.py \
    --round      8 \
    --sed_dir    "outputs/sed-ns-b0-20s-r8" \
    --perch_w    0.0 \
    --sed_w      1.0 \
    --percentile 95 \
    --gamma      2.00 \
    --nonaves_perch_only \
    --out        "pseudo_labels/sed_20s_r8.csv" \
    > "${LOG}/gen_pseudo_sed_20s_r8.log" 2>&1
log "Final pseudo labels → pseudo_labels/sed_20s_r8.csv"

log "SED-20s NS R5-R8 PIPELINE COMPLETE"
