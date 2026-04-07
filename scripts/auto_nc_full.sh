#!/usr/bin/env bash
# Noisy Classmate Full Automated Pipeline
# Bidirectional co-evolution: PVT and B0 teach each other
#
# All 5 phases active:
#   Phase 1: Ensemble Pseudo Labels
#   Phase 2: Confidence-Aware Blending
#   Phase 3: Disagreement Mining (sample weighting)
#   Phase 4: Soft Label Distillation (KLD loss, beta=0.3)
#   Phase 5: Bidirectional Iterative Co-Evolution
#
# Flow:
#   Gen 1: B0_R11 + PVT_R8 → PVT_R9 → PVT_R10
#   Gen 2: PVT_R10 + B0_R11 → B0_R13 (knowledge flows back!)
#   Gen 3: B0_R13 + PVT_R10 → PVT_R11
#   ...continues alternating
#
# Usage:
#   nohup bash scripts/auto_nc_full.sh > outputs/logs/auto_nc_full.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NC-FULL] $*"; }
mkdir -p "$LOG" checkpoints

# ── Helper Functions ─────────────────────────────────────────────────────────

gen_nc_pseudo() {
    local CHAIN1_NAME=$1 CHAIN1_DIR=$2 CHAIN2_NAME=$3 CHAIN2_DIR=$4 OUT=$5
    if [ -f "$OUT" ]; then
        log "NC pseudo exists: $OUT, skipping"
        return 0
    fi
    local LOGFILE="${LOG}/gen_nc_pseudo_$(basename $OUT .csv).log"
    log "Generating NC pseudo: $CHAIN1_NAME + $CHAIN2_NAME → $OUT"
    $PYTHON scripts/gen_noisy_classmate_pseudo.py \
        --chains "${CHAIN1_NAME}:${CHAIN1_DIR}" "${CHAIN2_NAME}:${CHAIN2_DIR}" \
        --weights 0.5 0.5 \
        --confidence_weighting \
        --disagreement_mining \
        --soft_labels \
        --percentile 95 --gamma 2.0 \
        --out "$OUT" \
        > "$LOGFILE" 2>&1
    log "NC pseudo done: $(wc -l < "$OUT") rows → $OUT"
}

train_all_folds() {
    local CONFIG=$1 ARCH=$2 ROUND=$3 OUT_DIR=$4
    mkdir -p "$OUT_DIR"
    for F in 0 1 2 3 4; do
        local CKPT="${OUT_DIR}/fold${F}_best.pt"
        if [ -f "$CKPT" ]; then
            log "${ARCH}-R${ROUND} fold${F}: exists, skipping"
            continue
        fi
        log "${ARCH}-R${ROUND} fold${F}: starting"
        $PYTHON train_sed_ns.py \
            --config "$CONFIG" \
            --fold "$F" \
            --device "$DEVICE" \
            > "${LOG}/sed_ns_${ARCH}_r${ROUND}_fold${F}.log" 2>&1
        log "${ARCH}-R${ROUND} fold${F}: done"
    done
    log "${ARCH}-R${ROUND}: all folds complete"
}

run_infer() {
    local CONFIG=$1 ARCH=$2 ROUND=$3 OUT_DIR=$4
    local NPZ="${OUT_DIR}/all_ss_probs.npz"
    if [ -f "$NPZ" ]; then
        log "${ARCH}-R${ROUND}: npz exists, skipping infer"
        return 0
    fi
    log "${ARCH}-R${ROUND}: running infer_all_ss"
    $PYTHON train_sed_ns.py \
        --config "$CONFIG" \
        --infer_all_ss \
        --device "$DEVICE" \
        > "${LOG}/sed_ns_${ARCH}_r${ROUND}_infer.log" 2>&1
    log "${ARCH}-R${ROUND}: infer done"
}

run_corrector() {
    local ARCH=$1 ROUND=$2 OUT_DIR=$3
    local CORR="${OUT_DIR}/all_ss_probs_corrected.npz"
    if [ -f "$CORR" ]; then
        log "${ARCH}-R${ROUND}: corrected npz exists, skipping"
        return 0
    fi
    [ ! -f "$TEACHER_CSV" ] && { log "${ARCH}-R${ROUND}: no teacher CSV, skipping corrector"; return 0; }
    log "${ARCH}-R${ROUND}: training Residual Corrector"
    $PYTHON scripts/train_sed_residual_corrector.py \
        --sed_dir  "$OUT_DIR" \
        --teacher  "$TEACHER_CSV" \
        --round    "$ROUND" \
        --alpha    "$CORRECTOR_ALPHA" \
        --out_ckpt "checkpoints/sed_corrector_${ARCH}_r${ROUND}.pt" \
        --device   "$DEVICE" \
        > "${LOG}/sed_corrector_${ARCH}_r${ROUND}.log" 2>&1
    log "${ARCH}-R${ROUND}: corrector done"
}

full_round() {
    local CONFIG=$1 ARCH=$2 ROUND=$3 OUT_DIR=$4 PSEUDO_CSV=$5
    # Update config to use NC pseudo
    sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO_CSV}|" "$CONFIG"
    train_all_folds "$CONFIG" "$ARCH" "$ROUND" "$OUT_DIR"
    run_infer "$CONFIG" "$ARCH" "$ROUND" "$OUT_DIR"
    run_corrector "$ARCH" "$ROUND" "$OUT_DIR"
}

# ── Track latest completed dirs ─────────────────────────────────────────────
B0_LATEST_DIR="outputs/sed-ns-b0-20s-r11"
B0_LATEST_ROUND=11
PVT_LATEST_DIR="outputs/sed-ns-pvt-20s-r8"
PVT_LATEST_ROUND=8

# ── Wait for current PVT R9 training to finish ──────────────────────────────
log "Waiting for PVT R9 training to complete..."
while pgrep -f "auto_nc_pvt_r9" > /dev/null 2>&1; do
    sleep 60
done

# Check if PVT R9 folds are done
R9_DONE=true
for F in 0 1 2 3 4; do
    [ ! -f "outputs/sed-ns-pvt-20s-r9/fold${F}_best.pt" ] && R9_DONE=false
done

if [ "$R9_DONE" = true ]; then
    log "PVT R9 folds already complete"
    # Run infer + corrector if needed
    run_infer "configs/sed_ns_pvt_20s_r9.yaml" "pvt" 9 "outputs/sed-ns-pvt-20s-r9"
    run_corrector "pvt" 9 "outputs/sed-ns-pvt-20s-r9"
    PVT_LATEST_DIR="outputs/sed-ns-pvt-20s-r9"
    PVT_LATEST_ROUND=9
else
    log "PVT R9 not complete, starting from R9"
    gen_nc_pseudo "b0" "$B0_LATEST_DIR" "pvt" "$PVT_LATEST_DIR" \
        "pseudo_labels/noisy_classmate_pvt_r9.csv"
    full_round "configs/sed_ns_pvt_20s_r9.yaml" "pvt" 9 \
        "outputs/sed-ns-pvt-20s-r9" "pseudo_labels/noisy_classmate_pvt_r9.csv"
    PVT_LATEST_DIR="outputs/sed-ns-pvt-20s-r9"
    PVT_LATEST_ROUND=9
fi

# ── Bidirectional Co-Evolution Loop ──────────────────────────────────────────
# Alternate: PVT round → B0 round → PVT round → ...

for GEN in 1 2 3 4 5 6; do
    log "════════════════ NC Generation ${GEN} ════════════════"

    # ── PVT next round ───────────────────────────────────────────────────────
    PVT_NEXT=$((PVT_LATEST_ROUND + 1))
    PVT_CFG="configs/sed_ns_pvt_20s_r${PVT_NEXT}.yaml"
    if [ -f "$PVT_CFG" ]; then
        PVT_PSEUDO="pseudo_labels/noisy_classmate_pvt_r${PVT_NEXT}.csv"
        gen_nc_pseudo "b0" "$B0_LATEST_DIR" "pvt" "$PVT_LATEST_DIR" "$PVT_PSEUDO"
        full_round "$PVT_CFG" "pvt" "$PVT_NEXT" \
            "outputs/sed-ns-pvt-20s-r${PVT_NEXT}" "$PVT_PSEUDO"
        PVT_LATEST_DIR="outputs/sed-ns-pvt-20s-r${PVT_NEXT}"
        PVT_LATEST_ROUND=$PVT_NEXT
    else
        log "PVT R${PVT_NEXT} config not found, skipping PVT"
    fi

    # ── B0 next round (BIDIRECTIONAL: PVT teaches B0!) ───────────────────────
    B0_NEXT=$((B0_LATEST_ROUND + 1))
    B0_CFG="configs/sed_ns_b0_20s_r${B0_NEXT}.yaml"
    if [ -f "$B0_CFG" ]; then
        B0_PSEUDO="pseudo_labels/noisy_classmate_b0_r${B0_NEXT}.csv"
        # Knowledge flows BACK: PVT → B0
        gen_nc_pseudo "pvt" "$PVT_LATEST_DIR" "b0" "$B0_LATEST_DIR" "$B0_PSEUDO"
        full_round "$B0_CFG" "b0" "$B0_NEXT" \
            "outputs/sed-ns-b0-20s-r${B0_NEXT}" "$B0_PSEUDO"
        B0_LATEST_DIR="outputs/sed-ns-b0-20s-r${B0_NEXT}"
        B0_LATEST_ROUND=$B0_NEXT
    else
        log "B0 R${B0_NEXT} config not found, skipping B0"
    fi

    log "Gen ${GEN} complete: PVT=R${PVT_LATEST_ROUND}, B0=R${B0_LATEST_ROUND}"
done

log "═══════════ NOISY CLASSMATE FULL PIPELINE COMPLETE ═══════════"
log "Final state: PVT=R${PVT_LATEST_ROUND}, B0=R${B0_LATEST_ROUND}"
