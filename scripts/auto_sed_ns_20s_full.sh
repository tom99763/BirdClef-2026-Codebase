#!/usr/bin/env bash
# SED noisy-student chain (20s clips): rounds 1→4.
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
# Folds run SEQUENTIALLY (0→1→2→3→4) to avoid GPU memory contention.
#
# Pseudo label strategy (1st-place inspired):
#   - Perch teacher kept in ensemble throughout, weight decreases each round
#   - Dynamic threshold percentile increases each round (stricter = cleaner labels)
#   - Residual Corrector applied before pseudo label generation
#
# Round | perch_w | sed_w | threshold_pct | rationale
#   1   |   0.50  |  0.50 |     92        | SED still weak, lean on teacher
#   2   |   0.30  |  0.70 |     93        | SED improving, reduce teacher weight
#   3   |   0.10  |  0.90 |     94        | SED strong, teacher mainly for anchoring
#   (R4 is last round, no pseudo needed)
#
# Usage:
#   nohup bash scripts/auto_sed_ns_20s_full.sh > outputs/logs/auto_sed_ns_20s_full.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

# Per-round pseudo label config: "perch_w sed_w threshold_pct gamma"
# gamma per 1st place BirdCLEF 2025: R1=1.0 (raw), R2=1/0.65≈1.54, R3=1/0.55≈1.82
declare -A PSEUDO_CFG
PSEUDO_CFG[1]="0.50 0.50 92 1.00"
PSEUDO_CFG[2]="0.30 0.70 93 1.54"
PSEUDO_CFG[3]="0.10 0.90 94 1.82"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SED-20s] $*"; }
mkdir -p "$LOG" checkpoints

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
    local CORR_CKPT="checkpoints/sed_corrector_r${R}.pt"
    local CORR_NPZ="outputs/sed-ns-b0-20s-r${R}/all_ss_probs_corrected.npz"

    if [ -f "$CORR_NPZ" ]; then
        log "R${R}: corrected npz already exists, skipping corrector"
        return 0
    fi
    if [ ! -f "$TEACHER_CSV" ]; then
        log "R${R}: teacher CSV not found (${TEACHER_CSV}), skipping corrector"
        return 0
    fi

    log "R${R}: training Temporal Residual Corrector ..."
    $PYTHON scripts/train_sed_residual_corrector.py \
        --sed_dir   "outputs/sed-ns-b0-20s-r${R}" \
        --teacher   "$TEACHER_CSV" \
        --round     "$R" \
        --alpha     "$CORRECTOR_ALPHA" \
        --out_ckpt  "$CORR_CKPT" \
        --device    "$DEVICE" \
        > "${LOG}/sed_corrector_r${R}.log" 2>&1
    log "R${R}: corrector done → ${CORR_NPZ}"
}

gen_pseudo() {
    local R=$1
    local PSEUDO_OUT="pseudo_labels/sed_20s_r${R}.csv"
    local CORR_NPZ="outputs/sed-ns-b0-20s-r${R}/all_ss_probs_corrected.npz"
    local ORIG_NPZ="outputs/sed-ns-b0-20s-r${R}/all_ss_probs.npz"
    local BACKUP_NPZ="outputs/sed-ns-b0-20s-r${R}/all_ss_probs_original.npz"

    # Parse per-round config
    local CFG="${PSEUDO_CFG[$R]}"
    local PERCH_W SED_W THR_PCT GAMMA
    read -r PERCH_W SED_W THR_PCT GAMMA <<< "$CFG"

    log "R${R}: pseudo config — perch_w=${PERCH_W} sed_w=${SED_W} threshold_pct=${THR_PCT} gamma=${GAMMA}"

    # Swap corrected probs in if available
    local SWAPPED=0
    if [ -f "$CORR_NPZ" ]; then
        cp "$ORIG_NPZ" "$BACKUP_NPZ"
        cp "$CORR_NPZ" "$ORIG_NPZ"
        log "R${R}: using Residual-Corrector probs for pseudo labels"
        SWAPPED=1
    fi

    # Check whether teacher CSV exists for Perch blend
    local PERCH_ARG=""
    if [ -f "$TEACHER_CSV" ]; then
        PERCH_ARG="--perch_csv ${TEACHER_CSV}"
    else
        log "R${R}: WARNING teacher CSV not found — using SED-only pseudo labels"
        PERCH_W="0.0"
        SED_W="1.0"
    fi

    $PYTHON scripts/gen_pseudo_ns.py \
        --round      "$R" \
        --clip_sec   20 \
        --sed_dir    "outputs/sed-ns-b0-20s-r${R}" \
        --perch_w    "$PERCH_W" \
        --sed_w      "$SED_W" \
        --percentile "$THR_PCT" \
        --gamma      "$GAMMA" \
        $PERCH_ARG \
        --aves_only \
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

# ── Main loop ─────────────────────────────────────────────────────────────────

for R in 1 2 3 4; do
    log "════════════════ Round ${R} ════════════════"
    mkdir -p "outputs/sed-ns-b0-20s-r${R}"

    # Sequential folds: 0 → 1 → 2 → 3 → 4
    for F in 0 1 2 3 4; do
        train_fold "$R" "$F"
    done
    log "R${R}: all folds done"

    # 5-fold ensemble inference on all soundscapes
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

    # Train Residual Corrector → all_ss_probs_corrected.npz
    train_residual_corrector "$R"

    # Generate pseudo labels for next round
    if [ "$R" -lt 4 ]; then
        gen_pseudo "$R"
    fi

    log "R${R}: complete"
done

log "SED-20s NS FULL PIPELINE COMPLETE"
