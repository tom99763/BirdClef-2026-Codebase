#!/usr/bin/env bash
# Noisy Classmate v3 — 3-Architecture Co-evolution
# Architectures: ConvNeXt-Femto, FastViT-T8, RegNetY-008
# Teachers (Phase 0): NS B0 R11 + NS PVT R8
# Features: disagreement mining, soft distillation, confidence weighting, nonaves_perch_only
#
# Usage:
#   nohup bash scripts/auto_nc3_dual_gpu.sh > outputs/logs/auto_nc3_dual_gpu.log 2>&1 &

set -euo pipefail
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
LOG="outputs/logs"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [NC3] $*"; }
mkdir -p "$LOG" checkpoints pseudo_labels

# ── Architecture definitions ──
CNXTF_BACKBONE="convnext_femto.d1_in1k"
FVIT_BACKBONE="fastvit_t8.apple_dist_in1k"
REGY_BACKBONE="regnety_008.pycls_in1k"

nc3_dir() {
    local ARCH=$1 ROUND=$2
    echo "outputs/sed-ns-${ARCH}-20s-r${ROUND}-nc3"
}

train_fold_gpu() {
    local CONFIG=$1 ARCH=$2 ROUND=$3 FOLD=$4 GPU=$5
    local OUT_DIR=$(nc3_dir "$ARCH" "$ROUND")
    local CKPT="${OUT_DIR}/fold${FOLD}_best.pt"
    if [ -f "$CKPT" ]; then
        log "${ARCH}-R${ROUND}-nc3 fold${FOLD}: exists, skipping"
        return 0
    fi
    log "${ARCH}-R${ROUND}-nc3 fold${FOLD}: starting on GPU${GPU}"
    local TMP_CFG="/tmp/nc3_${ARCH}_r${ROUND}_fold${FOLD}.yaml"
    cp "$CONFIG" "$TMP_CFG"
    sed -i "/^  dir:/s|dir:.*|dir:          ${OUT_DIR}|" "$TMP_CFG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_sed_ns.py \
        --config "$TMP_CFG" --fold "$FOLD" --device "cuda:0" \
        > "${LOG}/sed_ns_${ARCH}_r${ROUND}_nc3_fold${FOLD}.log" 2>&1
    rm -f "$TMP_CFG"
    log "${ARCH}-R${ROUND}-nc3 fold${FOLD}: done"
}

train_all_folds_dual() {
    local CONFIG=$1 ARCH=$2 ROUND=$3
    local OUT_DIR=$(nc3_dir "$ARCH" "$ROUND")
    mkdir -p "$OUT_DIR"
    log "${ARCH}-R${ROUND}-nc3: training 5 folds dual GPU"
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
    [ $DONE -eq 5 ] && log "${ARCH}-R${ROUND}-nc3: ALL 5 folds ✓" || { log "ERROR ${DONE}/5"; return 1; }
}

run_infer() {
    local CONFIG=$1 ARCH=$2 ROUND=$3
    local OUT_DIR=$(nc3_dir "$ARCH" "$ROUND")
    [ -f "${OUT_DIR}/all_ss_probs.npz" ] && { log "${ARCH}-R${ROUND}-nc3: npz exists"; return 0; }
    local TMP="/tmp/nc3_${ARCH}_r${ROUND}_infer.yaml"
    cp "$CONFIG" "$TMP"
    sed -i "/^  dir:/s|dir:.*|dir:          ${OUT_DIR}|" "$TMP"
    log "${ARCH}-R${ROUND}-nc3: infer (GPU0)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON train_sed_ns.py --config "$TMP" --infer_all_ss --device "cuda:0" \
        > "${LOG}/sed_ns_${ARCH}_r${ROUND}_nc3_infer.log" 2>&1
    rm -f "$TMP"
}

run_corrector() {
    local ARCH=$1 ROUND=$2
    local OUT_DIR=$(nc3_dir "$ARCH" "$ROUND")
    [ -f "${OUT_DIR}/all_ss_probs_corrected.npz" ] && { log "${ARCH}-R${ROUND}-nc3: corr exists"; return 0; }
    [ ! -f "$TEACHER_CSV" ] && return 0
    log "${ARCH}-R${ROUND}-nc3: corrector (GPU0)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sed_residual_corrector.py \
        --sed_dir "$OUT_DIR" --teacher "$TEACHER_CSV" --round "$ROUND" \
        --alpha "$CORRECTOR_ALPHA" --out_ckpt "checkpoints/sed_corrector_${ARCH}_r${ROUND}_nc3.pt" \
        --device "cuda:0" > "${LOG}/sed_corrector_${ARCH}_r${ROUND}_nc3.log" 2>&1
}

gen_nc3_pseudo() {
    local OUT=$1; shift
    # Remaining args are chain pairs: "name:dir" ...
    [ -f "$OUT" ] && { log "NC3 pseudo exists: $OUT"; return 0; }
    log "NC3 pseudo: → $OUT"
    $PYTHON scripts/gen_noisy_classmate_pseudo.py \
        --chains "$@" \
        --weights 0.5 0.5 \
        --confidence_weighting \
        --disagreement_mining \
        --soft_labels \
        --nonaves_perch_only \
        --percentile 95 --gamma 2.0 \
        --out "$OUT" \
        > "${LOG}/gen_nc3_pseudo_$(basename $OUT .csv).log" 2>&1
}

gen_nc3_pseudo_3way() {
    local OUT=$1; shift
    [ -f "$OUT" ] && { log "NC3 pseudo exists: $OUT"; return 0; }
    log "NC3 3-way pseudo: → $OUT"
    $PYTHON scripts/gen_noisy_classmate_pseudo.py \
        --chains "$@" \
        --confidence_weighting \
        --disagreement_mining \
        --soft_labels \
        --nonaves_perch_only \
        --percentile 95 --gamma 2.0 \
        --out "$OUT" \
        > "${LOG}/gen_nc3_pseudo_$(basename $OUT .csv).log" 2>&1
}

full_round() {
    local CONFIG=$1 ARCH=$2 ROUND=$3 PSEUDO=$4 PREV=${5:-}
    sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO}|" "$CONFIG"
    if [ -n "$PREV" ] && [ -d "$PREV" ]; then
        sed -i "s|prev_round_dir:.*|prev_round_dir:     ${PREV}|" "$CONFIG"
        log "EMA: $PREV"
    else
        # Remove prev_round_dir for from-scratch training
        sed -i '/prev_round_dir/d' "$CONFIG"
        log "From scratch (no EMA)"
    fi
    train_all_folds_dual "$CONFIG" "$ARCH" "$ROUND"
    run_infer "$CONFIG" "$ARCH" "$ROUND"
    run_corrector "$ARCH" "$ROUND"
}

# ═══════════════════════════════════════════════════════════════════
# PHASE 0 — Bootstrap: B0+PVT as teachers
# ═══════════════════════════════════════════════════════════════════

B0_SRC="outputs/sed-ns-b0-20s-r11"
PVT_SRC="outputs/sed-ns-pvt-20s-r8"

log "═══════ PHASE 0: Bootstrap (B0+PVT → 3 new archs) ═══════"
P0="pseudo_labels/nc3_r0.csv"
gen_nc3_pseudo "$P0" "b0:${B0_SRC}" "pvt:${PVT_SRC}"

# ConvNeXt-Femto R1 (from scratch)
log "════════ ConvNeXt-Femto R1 ════════"
full_round "configs/sed_ns_cnxtf_20s_r1.yaml" "cnxtf" 1 "$P0" ""
CNXTF=$(nc3_dir "cnxtf" 1)

# FastViT-T8 R1 (from scratch)
log "════════ FastViT-T8 R1 ════════"
full_round "configs/sed_ns_fvit_20s_r1.yaml" "fvit" 1 "$P0" ""
FVIT=$(nc3_dir "fvit" 1)

# RegNetY-008 R1 (from scratch)
log "════════ RegNetY-008 R1 ════════"
full_round "configs/sed_ns_regy_20s_r1.yaml" "regy" 1 "$P0" ""
REGY=$(nc3_dir "regy" 1)

# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — First NC: B0+PVT(0.3) + new models(0.7) blend
# ═══════════════════════════════════════════════════════════════════

log "═══════ PHASE 1: First NC (teacher 0.3 + students 0.7) ═══════"
# Generate pseudo from all 5 models (B0, PVT, Cnxt, Fvit, Reg)
P1="pseudo_labels/nc3_r1.csv"
$PYTHON scripts/gen_noisy_classmate_pseudo.py \
    --chains "b0:${B0_SRC}" "pvt:${PVT_SRC}" "cnxtf:${CNXTF}" "fvit:${FVIT}" "regy:${REGY}" \
    --weights 0.15 0.15 0.233 0.233 0.234 \
    --confidence_weighting \
    --disagreement_mining \
    --soft_labels \
    --nonaves_perch_only \
    --percentile 95 --gamma 2.0 \
    --out "$P1" \
    > "${LOG}/gen_nc3_pseudo_r1.log" 2>&1

# ConvNeXt-Femto R2
log "════════ ConvNeXt-Femto R2 ════════"
cp configs/sed_ns_cnxtf_20s_r1.yaml configs/sed_ns_cnxtf_20s_r2.yaml
sed -i 's/r1/r2/g; s/round: 1/round: 2/' configs/sed_ns_cnxtf_20s_r2.yaml
full_round "configs/sed_ns_cnxtf_20s_r2.yaml" "cnxtf" 2 "$P1" "$CNXTF"
CNXTF=$(nc3_dir "cnxtf" 2)

# FastViT-T8 R2
log "════════ FastViT-T8 R2 ════════"
cp configs/sed_ns_fvit_20s_r1.yaml configs/sed_ns_fvit_20s_r2.yaml
sed -i 's/r1/r2/g; s/round: 1/round: 2/' configs/sed_ns_fvit_20s_r2.yaml
full_round "configs/sed_ns_fvit_20s_r2.yaml" "fvit" 2 "$P1" "$FVIT"
FVIT=$(nc3_dir "fvit" 2)

# RegNetY-008 R2
log "════════ RegNetY-008 R2 ════════"
cp configs/sed_ns_regy_20s_r1.yaml configs/sed_ns_regy_20s_r2.yaml
sed -i 's/r1/r2/g; s/round: 1/round: 2/' configs/sed_ns_regy_20s_r2.yaml
full_round "configs/sed_ns_regy_20s_r2.yaml" "regy" 2 "$P1" "$REGY"
REGY=$(nc3_dir "regy" 2)

# ═══════════════════════════════════════════════════════════════════
# PHASE 2+ — Full 3-way NC Co-evolution (no more B0/PVT)
# ═══════════════════════════════════════════════════════════════════

for GEN in 3 4; do
    log "═══════ PHASE 2 Gen ${GEN}: 3-way NC ═══════"
    P="pseudo_labels/nc3_r${GEN}.csv"
    gen_nc3_pseudo_3way "$P" "cnxtf:${CNXTF}" "fvit:${FVIT}" "regy:${REGY}"

    # ConvNeXt-Femto
    log "════════ ConvNeXt-Femto R${GEN} ════════"
    cp configs/sed_ns_cnxtf_20s_r1.yaml "configs/sed_ns_cnxtf_20s_r${GEN}.yaml"
    sed -i "s/r1/r${GEN}/g; s/round: 1/round: ${GEN}/" "configs/sed_ns_cnxtf_20s_r${GEN}.yaml"
    full_round "configs/sed_ns_cnxtf_20s_r${GEN}.yaml" "cnxtf" "$GEN" "$P" "$CNXTF"
    CNXTF=$(nc3_dir "cnxtf" "$GEN")

    # FastViT-T8
    log "════════ FastViT-T8 R${GEN} ════════"
    cp configs/sed_ns_fvit_20s_r1.yaml "configs/sed_ns_fvit_20s_r${GEN}.yaml"
    sed -i "s/r1/r${GEN}/g; s/round: 1/round: ${GEN}/" "configs/sed_ns_fvit_20s_r${GEN}.yaml"
    full_round "configs/sed_ns_fvit_20s_r${GEN}.yaml" "fvit" "$GEN" "$P" "$FVIT"
    FVIT=$(nc3_dir "fvit" "$GEN")

    # RegNetY-008
    log "════════ RegNetY-008 R${GEN} ════════"
    cp configs/sed_ns_regy_20s_r1.yaml "configs/sed_ns_regy_20s_r${GEN}.yaml"
    sed -i "s/r1/r${GEN}/g; s/round: 1/round: ${GEN}/" "configs/sed_ns_regy_20s_r${GEN}.yaml"
    full_round "configs/sed_ns_regy_20s_r${GEN}.yaml" "regy" "$GEN" "$P" "$REGY"
    REGY=$(nc3_dir "regy" "$GEN")
done

log "═══════ NC3 PIPELINE COMPLETE ═══════"
