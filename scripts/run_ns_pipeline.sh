#!/usr/bin/env bash
# ============================================================================
# Iterative Noisy Student Pipeline — BirdCLEF 2026
#
# 4 rounds × 5 folds × 2 models (SED + SSM) on GPU1
# One SED fold + one SSM fold at a time (sequential on GPU1)
#
# Schedule:
#   Round k:
#     1. gen_pseudo_ns.py  → pseudo_labels/ns_rK.csv
#     2. For fold 0..4:
#        a. train_sed_ns.py  --config sed_ns_b0_rK --fold f  (GPU1)
#        b. train_ssm_ns.py  --config proto_ssm_ns_rK --fold f  (GPU1)
#     3. infer_all_soundscapes for SED (after all folds done)
#     4. gen_pseudo_ns.py with SED+SSM → ns_rK.csv for next round
#
# Usage:
#   bash scripts/run_ns_pipeline.sh 2>&1 | tee outputs/ns_pipeline.log
#   # Or single round: ROUND=1 bash scripts/run_ns_pipeline.sh
# ============================================================================

set -euo pipefail
DEVICE="cuda:1"
N_FOLDS=5
N_ROUNDS=4
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Round 0: already done — pseudo_labels/ns_r0.csv exists
START_ROUND="${ROUND:-1}"

for ROUND in $(seq "$START_ROUND" "$N_ROUNDS"); do
    PREV_ROUND=$((ROUND - 1))
    PSEUDO_IN="pseudo_labels/ns_r${PREV_ROUND}.csv"
    PSEUDO_OUT="pseudo_labels/ns_r${ROUND}.csv"
    SED_DIR="outputs/sed-ns-b0-r${ROUND}"
    SSM_DIR="outputs/ssm-ns-b0-r${ROUND}"
    SED_CFG="configs/sed_ns_b0_r${ROUND}.yaml"
    SSM_CFG="configs/ssm_ns_b0_r${ROUND}.yaml"

    log "========================================================"
    log "  ROUND ${ROUND}/${N_ROUNDS}"
    log "  Pseudo in : ${PSEUDO_IN}"
    log "========================================================"

    if [ ! -f "$PSEUDO_IN" ]; then
        log "ERROR: ${PSEUDO_IN} not found. Run previous round first."
        exit 1
    fi

    mkdir -p "$SED_DIR" "$SSM_DIR"

    # ── Train 5 folds (SED then SSM for each fold) ──────────────────────────
    for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        log "--- Round ${ROUND}, Fold ${FOLD}: SED ---"
        SED_LOG="${LOG_DIR}/sed_ns_r${ROUND}_fold${FOLD}.log"
        python3 train_sed_ns.py \
            --config "$SED_CFG" \
            --fold   "$FOLD" \
            --device "$DEVICE" \
            2>&1 | tee "$SED_LOG"

        log "--- Round ${ROUND}, Fold ${FOLD}: SSM ---"
        SSM_LOG="${LOG_DIR}/ssm_ns_r${ROUND}_fold${FOLD}.log"
        python3 train_ssm_ns.py \
            --config "$SSM_CFG" \
            --fold   "$FOLD" \
            --device "$DEVICE" \
            2>&1 | tee "$SSM_LOG"

        log "  Fold ${FOLD} complete."
    done

    # ── Save SED OOF + run all-soundscape inference ──────────────────────────
    log "--- Round ${ROUND}: SED all-soundscape inference ---"
    INF_LOG="${LOG_DIR}/sed_ns_r${ROUND}_infer.log"
    python3 train_sed_ns.py \
        --config       "$SED_CFG" \
        --infer_all_ss \
        --device       "$DEVICE" \
        2>&1 | tee "$INF_LOG" || log "WARNING: SED inference failed (check $INF_LOG)"

    # ── Generate next-round pseudo labels ─────────────────────────────────────
    if [ "$ROUND" -lt "$N_ROUNDS" ]; then
        log "--- Round ${ROUND}: generating pseudo labels for Round $((ROUND+1)) ---"
        PSEUDO_NEXT="pseudo_labels/ns_r${ROUND}.csv"
        python3 scripts/gen_pseudo_ns.py \
            --round      "$ROUND" \
            --perch_csv  outputs/perch_teacher_all_ss.csv \
            --sed_dir    "$SED_DIR" \
            --ssm_dir    "$SSM_DIR" \
            --out        "$PSEUDO_NEXT" \
            2>&1 | tee "${LOG_DIR}/gen_pseudo_ns_r${ROUND}.log"
        log "  Pseudo labels saved: ${PSEUDO_NEXT}"
    fi

    log "  Round ${ROUND} COMPLETE."
done

log "========================================================"
log "  Noisy Student Pipeline FINISHED (${N_ROUNDS} rounds)"
log "========================================================"
