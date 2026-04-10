#!/usr/bin/env bash
# Noisy Classmate — Dual GPU Pipeline (NC-only outputs, _nc suffix)
# NC checkpoints go to outputs/sed-ns-{arch}-20s-r{R}-nc/ (separate from NS)
# All 5 phases active. Bidirectional co-evolution.
# Folds run in parallel: 2 on GPU0 + 2 on GPU1 + 1 on GPU0.
#
# Usage:
#   nohup bash scripts/auto_nc_dual_gpu.sh > outputs/logs/auto_nc_dual_gpu.log 2>&1 &

set -euo pipefail
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NC-DUAL] $*"; }
mkdir -p "$LOG" checkpoints

# ── NC output directory (separate from NS) ────────────────────────────────────
nc_dir() {
    local ARCH=$1 ROUND=$2
    echo "outputs/sed-ns-${ARCH}-20s-r${ROUND}-nc"
}

# ── Train single fold on specified GPU ────────────────────────────────────────
train_fold_gpu() {
    local CONFIG=$1 ARCH=$2 ROUND=$3 FOLD=$4 GPU=$5
    local OUT_DIR=$(nc_dir "$ARCH" "$ROUND")
    local CKPT="${OUT_DIR}/fold${FOLD}_best.pt"
    if [ -f "$CKPT" ]; then
        log "${ARCH}-R${ROUND}-nc fold${FOLD}: exists, skipping"
        return 0
    fi
    log "${ARCH}-R${ROUND}-nc fold${FOLD}: starting on GPU${GPU}"

    # Copy config to avoid race condition (4 folds modify same file)
    local TMP_CFG="/tmp/nc_${ARCH}_r${ROUND}_fold${FOLD}.yaml"
    cp "$CONFIG" "$TMP_CFG"
    # Only match the output dir line (starts with "  dir:"), not prev_round_dir
    sed -i "/^  dir:/s|dir:.*|dir:          ${OUT_DIR}|" "$TMP_CFG"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_sed_ns.py \
        --config "$TMP_CFG" \
        --fold "$FOLD" \
        --device "cuda:0" \
        > "${LOG}/sed_ns_${ARCH}_r${ROUND}_nc_fold${FOLD}.log" 2>&1

    rm -f "$TMP_CFG"
    log "${ARCH}-R${ROUND}-nc fold${FOLD}: done"
}

# ── Train all 5 folds using both GPUs ─────────────────────────────────────────
train_all_folds_dual() {
    local CONFIG=$1 ARCH=$2 ROUND=$3
    local OUT_DIR=$(nc_dir "$ARCH" "$ROUND")
    mkdir -p "$OUT_DIR"

    log "${ARCH}-R${ROUND}-nc: training 5 folds on dual GPU (1 per GPU + dynamic fold5)"

    # Wave 1: fold0 (GPU0) + fold1 (GPU1)
    train_fold_gpu "$CONFIG" "$ARCH" "$ROUND" 0 0 &
    local PID_A=$!
    train_fold_gpu "$CONFIG" "$ARCH" "$ROUND" 1 1 &
    local PID_B=$!
    log "${ARCH}-R${ROUND}-nc: wave 1 (fold0 GPU0, fold1 GPU1)"
    wait $PID_A $PID_B

    # Wave 2: fold2 (GPU0) + fold3 (GPU1)
    train_fold_gpu "$CONFIG" "$ARCH" "$ROUND" 2 0 &
    PID_A=$!
    train_fold_gpu "$CONFIG" "$ARCH" "$ROUND" 3 1 &
    PID_B=$!
    log "${ARCH}-R${ROUND}-nc: wave 2 (fold2 GPU0, fold3 GPU1)"
    wait $PID_A $PID_B

    # Wave 3: fold4 (GPU0)
    train_fold_gpu "$CONFIG" "$ARCH" "$ROUND" 4 0
    log "${ARCH}-R${ROUND}-nc: wave 3 (fold4 GPU0)"

    # Verify
    local DONE=0
    for F in 0 1 2 3 4; do
        [ -f "${OUT_DIR}/fold${F}_best.pt" ] && DONE=$((DONE + 1))
    done
    if [ $DONE -eq 5 ]; then
        log "${ARCH}-R${ROUND}-nc: ALL 5 folds complete ✓"
    else
        log "${ARCH}-R${ROUND}-nc: ERROR only ${DONE}/5 folds!"
        return 1
    fi
}

# ── Infer (GPU0) ──────────────────────────────────────────────────────────────
run_infer() {
    local CONFIG=$1 ARCH=$2 ROUND=$3
    local OUT_DIR=$(nc_dir "$ARCH" "$ROUND")
    local NPZ="${OUT_DIR}/all_ss_probs.npz"
    if [ -f "$NPZ" ]; then
        log "${ARCH}-R${ROUND}-nc: npz exists, skipping infer"
        return 0
    fi
    # Use tmp config to avoid modifying shared config
    local TMP_CFG="/tmp/nc_${ARCH}_r${ROUND}_infer.yaml"
    cp "$CONFIG" "$TMP_CFG"
    sed -i "/^  dir:/s|dir:.*|dir:          ${OUT_DIR}|" "$TMP_CFG"

    log "${ARCH}-R${ROUND}-nc: running infer_all_ss (GPU0)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON train_sed_ns.py \
        --config "$TMP_CFG" \
        --infer_all_ss \
        --device "cuda:0" \
        > "${LOG}/sed_ns_${ARCH}_r${ROUND}_nc_infer.log" 2>&1

    rm -f "$TMP_CFG"
    log "${ARCH}-R${ROUND}-nc: infer done"
}

# ── Residual Corrector (GPU0) ─────────────────────────────────────────────────
run_corrector() {
    local ARCH=$1 ROUND=$2
    local OUT_DIR=$(nc_dir "$ARCH" "$ROUND")
    local CORR="${OUT_DIR}/all_ss_probs_corrected.npz"
    if [ -f "$CORR" ]; then
        log "${ARCH}-R${ROUND}-nc: corrected npz exists, skipping"
        return 0
    fi
    [ ! -f "$TEACHER_CSV" ] && { log "${ARCH}-R${ROUND}-nc: no teacher, skipping corrector"; return 0; }
    log "${ARCH}-R${ROUND}-nc: training Residual Corrector (GPU0)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sed_residual_corrector.py \
        --sed_dir  "$OUT_DIR" \
        --teacher  "$TEACHER_CSV" \
        --round    "$ROUND" \
        --alpha    "$CORRECTOR_ALPHA" \
        --out_ckpt "checkpoints/sed_corrector_${ARCH}_r${ROUND}_nc.pt" \
        --device   "cuda:0" \
        > "${LOG}/sed_corrector_${ARCH}_r${ROUND}_nc.log" 2>&1
    log "${ARCH}-R${ROUND}-nc: corrector done"
}

# ── NC Pseudo Label Generation ────────────────────────────────────────────────
gen_nc_pseudo() {
    local CHAIN1_NAME=$1 CHAIN1_DIR=$2 CHAIN2_NAME=$3 CHAIN2_DIR=$4 OUT=$5
    if [ -f "$OUT" ]; then
        log "NC pseudo exists: $OUT, skipping"
        return 0
    fi
    log "Generating NC pseudo: $CHAIN1_NAME($CHAIN1_DIR) + $CHAIN2_NAME($CHAIN2_DIR)"
    $PYTHON scripts/gen_noisy_classmate_pseudo.py \
        --chains "${CHAIN1_NAME}:${CHAIN1_DIR}" "${CHAIN2_NAME}:${CHAIN2_DIR}" \
        --weights 0.5 0.5 \
        --confidence_weighting \
        --disagreement_mining \
        --soft_labels \
        --nonaves_perch_only \
        --percentile 95 --gamma 2.0 \
        --out "$OUT" \
        > "${LOG}/gen_nc_pseudo_$(basename $OUT .csv).log" 2>&1
    log "NC pseudo done → $OUT"
}

# ── Full round ────────────────────────────────────────────────────────────────
full_round() {
    local CONFIG=$1 ARCH=$2 ROUND=$3 PSEUDO_CSV=$4 PREV_NC_DIR=${5:-}
    # Update config BEFORE training (train_fold_gpu copies this)
    sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO_CSV}|" "$CONFIG"
    if [ -n "$PREV_NC_DIR" ] && [ -d "$PREV_NC_DIR" ]; then
        sed -i "s|prev_round_dir:.*|prev_round_dir:     ${PREV_NC_DIR}|" "$CONFIG"
        log "EMA inheritance: ${PREV_NC_DIR}"
    fi
    train_all_folds_dual "$CONFIG" "$ARCH" "$ROUND"
    run_infer "$CONFIG" "$ARCH" "$ROUND"
    run_corrector "$ARCH" "$ROUND"
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Start from B0 R12 NC
# ═══════════════════════════════════════════════════════════════════════════════

# NS baselines (used as initial blend sources)
B0_NS_LATEST="outputs/sed-ns-b0-20s-r11"   # Latest NS B0 with npz+corr
PVT_NS_LATEST="outputs/sed-ns-pvt-20s-r8"  # Latest NS PVT with npz+corr

# NC state tracking — start fresh
B0_NC_LATEST="$B0_NS_LATEST"
B0_NC_ROUND=11
PVT_NC_LATEST="$PVT_NS_LATEST"
PVT_NC_ROUND=8

# ── PVT R9 NC (first NC round) ──────────────────────────────────────────────
log "════════════════ PVT R9 NC (First NC Round) ════════════════"
PVT_R9_PSEUDO="pseudo_labels/noisy_classmate_pvt_r9_nc.csv"
gen_nc_pseudo "b0" "$B0_NC_LATEST" "pvt" "$PVT_NC_LATEST" "$PVT_R9_PSEUDO"
full_round "configs/sed_ns_pvt_20s_r9.yaml" "pvt" 9 "$PVT_R9_PSEUDO" "$PVT_NC_LATEST"
PVT_NC_LATEST=$(nc_dir "pvt" 9)
PVT_NC_ROUND=9

# ── PVT R10 NC ──────────────────────────────────────────────────────────────
log "════════════════ PVT R10 NC ════════════════"
PVT_R10_PSEUDO="pseudo_labels/noisy_classmate_pvt_r10_nc.csv"
gen_nc_pseudo "b0" "$B0_NC_LATEST" "pvt" "$PVT_NC_LATEST" "$PVT_R10_PSEUDO"
full_round "configs/sed_ns_pvt_20s_r10.yaml" "pvt" 10 "$PVT_R10_PSEUDO" "$PVT_NC_LATEST"
PVT_NC_LATEST=$(nc_dir "pvt" 10)
PVT_NC_ROUND=10

# ── B0 R12 NC (first knowledge backflow!) ────────────────────────────────────
log "════════════════ B0 R12 NC (First Backflow) ════════════════"
B0_R12_PSEUDO="pseudo_labels/noisy_classmate_b0_r12_nc.csv"
gen_nc_pseudo "pvt" "$PVT_NC_LATEST" "b0" "$B0_NC_LATEST" "$B0_R12_PSEUDO"
full_round "configs/sed_ns_b0_20s_r12.yaml" "b0" 12 "$B0_R12_PSEUDO" "$B0_NS_LATEST"
B0_NC_LATEST=$(nc_dir "b0" 12)
B0_NC_ROUND=12

# ── Bidirectional Co-Evolution Loop ──────────────────────────────────────────
for GEN in 1 2 3 4 5; do
    log "════════════════ NC Generation ${GEN} ════════════════"

    # ── PVT next round ───────────────────────────────────────────────────
    PVT_NEXT=$((PVT_NC_ROUND + 1))
    PVT_CFG="configs/sed_ns_pvt_20s_r${PVT_NEXT}.yaml"
    if [ -f "$PVT_CFG" ]; then
        PVT_PSEUDO="pseudo_labels/noisy_classmate_pvt_r${PVT_NEXT}_nc.csv"
        gen_nc_pseudo "b0" "$B0_NC_LATEST" "pvt" "$PVT_NC_LATEST" "$PVT_PSEUDO"
        full_round "$PVT_CFG" "pvt" "$PVT_NEXT" "$PVT_PSEUDO" "$PVT_NC_LATEST"
        PVT_NC_LATEST=$(nc_dir "pvt" "$PVT_NEXT")
        PVT_NC_ROUND=$PVT_NEXT
    else
        log "PVT R${PVT_NEXT} config not found, skipping"
    fi

    # ── B0 next round (bidirectional) ────────────────────────────────────
    B0_NEXT=$((B0_NC_ROUND + 1))
    B0_CFG="configs/sed_ns_b0_20s_r${B0_NEXT}.yaml"
    if [ -f "$B0_CFG" ]; then
        B0_PSEUDO="pseudo_labels/noisy_classmate_b0_r${B0_NEXT}_nc.csv"
        gen_nc_pseudo "pvt" "$PVT_NC_LATEST" "b0" "$B0_NC_LATEST" "$B0_PSEUDO"
        full_round "$B0_CFG" "b0" "$B0_NEXT" "$B0_PSEUDO" "$B0_NC_LATEST"
        B0_NC_LATEST=$(nc_dir "b0" "$B0_NEXT")
        B0_NC_ROUND=$B0_NEXT
    else
        log "B0 R${B0_NEXT} config not found, skipping"
    fi

    log "Gen ${GEN} done: B0=R${B0_NC_ROUND}-nc, PVT=R${PVT_NC_ROUND}-nc"
done

log "═══════════ NC DUAL-GPU PIPELINE COMPLETE ═══════════"
