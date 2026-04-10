#!/usr/bin/env bash
# Noisy Classmate v2 — Confidence-Preserved Pipeline
# Difference from v1: NO disagreement mining, NO KLD, higher gamma, student-weighted blend
# Output: _nc2 suffix
#
# Usage:
#   nohup bash scripts/auto_nc2_dual_gpu.sh > outputs/logs/auto_nc2_dual_gpu.log 2>&1 &

set -euo pipefail
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NC2-DUAL] $*"; }
mkdir -p "$LOG" checkpoints

nc2_dir() {
    local ARCH=$1 ROUND=$2
    echo "outputs/sed-ns-${ARCH}-20s-r${ROUND}-nc2"
}

train_fold_gpu() {
    local CONFIG=$1 ARCH=$2 ROUND=$3 FOLD=$4 GPU=$5
    local OUT_DIR=$(nc2_dir "$ARCH" "$ROUND")
    local CKPT="${OUT_DIR}/fold${FOLD}_best.pt"
    if [ -f "$CKPT" ]; then
        log "${ARCH}-R${ROUND}-nc2 fold${FOLD}: exists, skipping"
        return 0
    fi
    log "${ARCH}-R${ROUND}-nc2 fold${FOLD}: starting on GPU${GPU}"
    local TMP_CFG="/tmp/nc2_${ARCH}_r${ROUND}_fold${FOLD}.yaml"
    cp "$CONFIG" "$TMP_CFG"
    sed -i "/^  dir:/s|dir:.*|dir:          ${OUT_DIR}|" "$TMP_CFG"
    # Force nc_distill_beta to 0.0 (no Phase 4)
    if grep -q "nc_distill_beta" "$TMP_CFG"; then
        sed -i "s/nc_distill_beta:.*/nc_distill_beta:    0.0/" "$TMP_CFG"
    fi
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_sed_ns.py \
        --config "$TMP_CFG" --fold "$FOLD" --device "cuda:0" \
        > "${LOG}/sed_ns_${ARCH}_r${ROUND}_nc2_fold${FOLD}.log" 2>&1
    rm -f "$TMP_CFG"
    log "${ARCH}-R${ROUND}-nc2 fold${FOLD}: done"
}

train_all_folds_dual() {
    local CONFIG=$1 ARCH=$2 ROUND=$3
    local OUT_DIR=$(nc2_dir "$ARCH" "$ROUND")
    mkdir -p "$OUT_DIR"
    log "${ARCH}-R${ROUND}-nc2: training 5 folds dual GPU"
    # Wave 1
    train_fold_gpu "$CONFIG" "$ARCH" "$ROUND" 0 0 &
    local PA=$!
    train_fold_gpu "$CONFIG" "$ARCH" "$ROUND" 1 1 &
    local PB=$!
    wait $PA $PB
    # Wave 2
    train_fold_gpu "$CONFIG" "$ARCH" "$ROUND" 2 0 &
    PA=$!
    train_fold_gpu "$CONFIG" "$ARCH" "$ROUND" 3 1 &
    PB=$!
    wait $PA $PB
    # Wave 3
    train_fold_gpu "$CONFIG" "$ARCH" "$ROUND" 4 0
    local DONE=0
    for F in 0 1 2 3 4; do [ -f "${OUT_DIR}/fold${F}_best.pt" ] && DONE=$((DONE+1)); done
    [ $DONE -eq 5 ] && log "${ARCH}-R${ROUND}-nc2: ALL 5 folds ✓" || { log "ERROR ${DONE}/5"; return 1; }
}

run_infer() {
    local CONFIG=$1 ARCH=$2 ROUND=$3
    local OUT_DIR=$(nc2_dir "$ARCH" "$ROUND")
    [ -f "${OUT_DIR}/all_ss_probs.npz" ] && { log "${ARCH}-R${ROUND}-nc2: npz exists"; return 0; }
    local TMP="/tmp/nc2_${ARCH}_r${ROUND}_infer.yaml"
    cp "$CONFIG" "$TMP"
    sed -i "/^  dir:/s|dir:.*|dir:          ${OUT_DIR}|" "$TMP"
    log "${ARCH}-R${ROUND}-nc2: infer (GPU0)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON train_sed_ns.py --config "$TMP" --infer_all_ss --device "cuda:0" \
        > "${LOG}/sed_ns_${ARCH}_r${ROUND}_nc2_infer.log" 2>&1
    rm -f "$TMP"
}

run_corrector() {
    local ARCH=$1 ROUND=$2
    local OUT_DIR=$(nc2_dir "$ARCH" "$ROUND")
    [ -f "${OUT_DIR}/all_ss_probs_corrected.npz" ] && { log "${ARCH}-R${ROUND}-nc2: corr exists"; return 0; }
    [ ! -f "$TEACHER_CSV" ] && return 0
    log "${ARCH}-R${ROUND}-nc2: corrector (GPU0)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sed_residual_corrector.py \
        --sed_dir "$OUT_DIR" --teacher "$TEACHER_CSV" --round "$ROUND" \
        --alpha "$CORRECTOR_ALPHA" --out_ckpt "checkpoints/sed_corrector_${ARCH}_r${ROUND}_nc2.pt" \
        --device "cuda:0" > "${LOG}/sed_corrector_${ARCH}_r${ROUND}_nc2.log" 2>&1
}

gen_nc2_pseudo() {
    local C1N=$1 C1D=$2 C2N=$3 C2D=$4 OUT=$5
    [ -f "$OUT" ] && { log "NC2 pseudo exists: $OUT"; return 0; }
    log "NC2 pseudo: $C1N + $C2N → $OUT (gamma=3.0, weights=0.3/0.7, NO disagreement mining)"
    $PYTHON scripts/gen_noisy_classmate_pseudo.py \
        --chains "${C1N}:${C1D}" "${C2N}:${C2D}" \
        --weights 0.3 0.7 \
        --confidence_weighting \
        --nonaves_perch_only \
        --percentile 95 --gamma 3.0 \
        --out "$OUT" \
        > "${LOG}/gen_nc2_pseudo_$(basename $OUT .csv).log" 2>&1
    # NOTE: NO --disagreement_mining, NO --soft_labels
}

full_round() {
    local CONFIG=$1 ARCH=$2 ROUND=$3 PSEUDO=$4 PREV=${5:-}
    sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO}|" "$CONFIG"
    [ -n "$PREV" ] && [ -d "$PREV" ] && { sed -i "s|prev_round_dir:.*|prev_round_dir:     ${PREV}|" "$CONFIG"; log "EMA: $PREV"; }
    train_all_folds_dual "$CONFIG" "$ARCH" "$ROUND"
    run_infer "$CONFIG" "$ARCH" "$ROUND"
    run_corrector "$ARCH" "$ROUND"
}

# ═══════════════════════════════════════════════════════════════════
# MAIN — NC v2 from scratch
# ═══════════════════════════════════════════════════════════════════

B0_SRC="outputs/sed-ns-b0-20s-r11"      # NS B0 R11
PVT_SRC="outputs/sed-ns-pvt-20s-r8"     # NS PVT R8

# PVT R9 NC2
log "════════ PVT R9 NC2 ════════"
P="pseudo_labels/nc2_pvt_r9.csv"
gen_nc2_pseudo "b0" "$B0_SRC" "pvt" "$PVT_SRC" "$P"
full_round "configs/sed_ns_pvt_20s_r9.yaml" "pvt" 9 "$P" "$PVT_SRC"
PVT_NC2=$(nc2_dir "pvt" 9)

# PVT R10 NC2
log "════════ PVT R10 NC2 ════════"
P="pseudo_labels/nc2_pvt_r10.csv"
gen_nc2_pseudo "b0" "$B0_SRC" "pvt" "$PVT_NC2" "$P"
full_round "configs/sed_ns_pvt_20s_r10.yaml" "pvt" 10 "$P" "$PVT_NC2"
PVT_NC2=$(nc2_dir "pvt" 10)

# B0 R12 NC2 (backflow)
log "════════ B0 R12 NC2 (Backflow) ════════"
P="pseudo_labels/nc2_b0_r12.csv"
gen_nc2_pseudo "pvt" "$PVT_NC2" "b0" "$B0_SRC" "$P"
full_round "configs/sed_ns_b0_20s_r12.yaml" "b0" 12 "$P" "$B0_SRC"
B0_NC2=$(nc2_dir "b0" 12)

# Gen 1: PVT R11 NC2
log "════════ Gen 1: PVT R11 NC2 ════════"
P="pseudo_labels/nc2_pvt_r11.csv"
gen_nc2_pseudo "b0" "$B0_NC2" "pvt" "$PVT_NC2" "$P"
full_round "configs/sed_ns_pvt_20s_r11.yaml" "pvt" 11 "$P" "$PVT_NC2"
PVT_NC2=$(nc2_dir "pvt" 11)

# Gen 1: B0 R13 NC2
log "════════ Gen 1: B0 R13 NC2 ════════"
P="pseudo_labels/nc2_b0_r13.csv"
gen_nc2_pseudo "pvt" "$PVT_NC2" "b0" "$B0_NC2" "$P"
full_round "configs/sed_ns_b0_20s_r13.yaml" "b0" 13 "$P" "$B0_NC2"

log "═══════ NC2 PIPELINE COMPLETE ═══════"
