#!/usr/bin/env bash
# SED noisy-student chain (10s clips): rounds 1→4.
# Folds run SEQUENTIALLY (0→1→2→3→4) to avoid GPU memory contention.
#
# Clean design — sources of pseudo label noise removed:
#   - NO power transform (gamma=1.0 → save original probs, no distortion)
#   - NO Residual Corrector (overfit risk, distorts signal)
#   - NO 1-to-1 fixed MixUp (concat + random MixUp instead)
#   - NO per-round gamma schedule
#
# Pseudo label strategy:
#   - Perch teacher blended with SED, weight decreases each round
#   - Dynamic threshold percentile increases each round (stricter = cleaner)
#
# Round | perch_w | sed_w | threshold_pct
#   1   |   0.50  |  0.50 |     92
#   2   |   0.30  |  0.70 |     93
#   3   |   0.10  |  0.90 |     94
#   (R4 is last round, no pseudo needed)
#
# Usage:
#   nohup bash scripts/auto_sed_ns_10s_full.sh > outputs/logs/auto_sed_ns_10s_full.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"

# Per-round pseudo label config: "perch_w sed_w threshold_pct"
declare -A PSEUDO_CFG
PSEUDO_CFG[1]="0.50 0.50 92"
PSEUDO_CFG[2]="0.30 0.70 93"
PSEUDO_CFG[3]="0.10 0.90 94"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SED-10s] $*"; }
mkdir -p "$LOG" checkpoints

# ── Functions ─────────────────────────────────────────────────────────────────

train_fold() {
    local R=$1 F=$2
    local CKPT="outputs/sed-ns-b0-10s-r${R}/fold${F}_best.pt"
    if [ -f "$CKPT" ]; then
        log "R${R} fold${F}: checkpoint exists, skipping"
        return 0
    fi
    log "R${R} fold${F}: starting"
    python3 train_sed_ns.py \
        --config configs/sed_ns_b0_10s_r${R}.yaml \
        --fold   "$F" \
        --device "$DEVICE" \
        > "${LOG}/sed_ns_10s_r${R}_fold${F}.log" 2>&1
    log "R${R} fold${F}: done"
}

gen_pseudo() {
    local R=$1
    local PSEUDO_OUT="pseudo_labels/sed_10s_r${R}.csv"

    local CFG="${PSEUDO_CFG[$R]}"
    local PERCH_W SED_W THR_PCT
    read -r PERCH_W SED_W THR_PCT <<< "$CFG"

    log "R${R}: pseudo config — perch_w=${PERCH_W} sed_w=${SED_W} threshold_pct=${THR_PCT}"

    local PERCH_ARG=""
    if [ -f "$TEACHER_CSV" ]; then
        PERCH_ARG="--perch_csv ${TEACHER_CSV}"
    else
        log "R${R}: WARNING teacher CSV not found — using SED-only pseudo labels"
        PERCH_W="0.0"
        SED_W="1.0"
    fi

    python3 scripts/gen_pseudo_ns.py \
        --round      "$R" \
        --clip_sec   5 \
        --sed_dir    "outputs/sed-ns-b0-10s-r${R}" \
        --perch_w    "$PERCH_W" \
        --sed_w      "$SED_W" \
        --percentile "$THR_PCT" \
        $PERCH_ARG \
        --out        "$PSEUDO_OUT" \
        > "${LOG}/gen_pseudo_sed_10s_r${R}.log" 2>&1

    log "R${R}: pseudo labels → ${PSEUDO_OUT}"

    # Update next round config
    local NEXT="configs/sed_ns_b0_10s_r$(( R+1 )).yaml"
    [ -f "$NEXT" ] && sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO_OUT}|" "$NEXT"
}

# ── Main loop ─────────────────────────────────────────────────────────────────

for R in 1 2 3 4; do
    log "════════════════ Round ${R} ════════════════"
    mkdir -p "outputs/sed-ns-b0-10s-r${R}"

    for F in 0 1 2 3 4; do
        train_fold "$R" "$F"
    done
    log "R${R}: all folds done"

    # 5-fold ensemble inference on all soundscapes
    INFER_NPZ="outputs/sed-ns-b0-10s-r${R}/all_ss_probs.npz"
    if [ -f "$INFER_NPZ" ]; then
        log "R${R}: all_ss_probs.npz exists, skipping infer_all_ss"
    else
        log "R${R}: running infer_all_ss"
        python3 train_sed_ns.py \
            --config       configs/sed_ns_b0_10s_r${R}.yaml \
            --infer_all_ss \
            --device       "$DEVICE" \
            > "${LOG}/sed_ns_10s_r${R}_infer.log" 2>&1
        log "R${R}: infer done"
    fi

    # Generate pseudo labels for next round (skip last round)
    if [ "$R" -lt 4 ]; then
        gen_pseudo "$R"
    fi

    log "R${R}: complete"
done

log "SED-10s NS FULL PIPELINE COMPLETE"
