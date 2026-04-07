#!/usr/bin/env bash
# Threshold search for R2 pseudo labels.
# Tests stricter thresholds (high → low) until R2 fold0 beats R1 fold0.
# Uses early_stop=3 for fast iteration (~10 min per fold).
# Once a winning threshold is found, launches full chain (R2 fold1-4, R3, R4).
#
# Usage:
#   nohup bash scripts/search_r2_threshold.sh > outputs/logs/search_r2_threshold.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"

R1_BEST_AUC="0.9433"          # R1 fold0 best ss_auc to beat
CORR_NPZ="outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz"
ORIG_NPZ="outputs/sed-ns-b0-20s-r1/all_ss_probs.npz"
BACKUP_NPZ="outputs/sed-ns-b0-20s-r1/all_ss_probs_original.npz"
TEACHER_CSV="outputs/perch_teacher_aug_all_ss.csv"
CORRECTOR_ALPHA="0.40"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SEARCH] $*" >&2; }
mkdir -p "$LOG" pseudo_labels checkpoints

# ── helpers ───────────────────────────────────────────────────────────────────

get_best_auc() {
    # Extract best AUC from fold0 log
    local logfile="$1"
    grep "New best AUC=" "$logfile" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1
}

beats_r1() {
    local auc="$1"
    python3 -c "import sys; sys.exit(0 if float('$auc') >= float('$R1_BEST_AUC') else 1)"
}

gen_pseudo_with_thr() {
    local THR="$1"
    local PSEUDO_OUT="pseudo_labels/sed_20s_r1_thr${THR}.csv"

    log "Generating R1 pseudo labels with threshold_pct=${THR} ..."

    # Swap corrected probs in if available
    local SWAPPED=0
    if [ -f "$CORR_NPZ" ]; then
        cp "$ORIG_NPZ" "$BACKUP_NPZ"
        cp "$CORR_NPZ" "$ORIG_NPZ"
        SWAPPED=1
    fi

    local PERCH_ARG=""
    [ -f "$TEACHER_CSV" ] && PERCH_ARG="--perch_csv ${TEACHER_CSV}"

    python3 scripts/gen_pseudo_ns.py \
        --round      1 \
        --clip_sec   20 \
        --sed_dir    "outputs/sed-ns-b0-20s-r1" \
        --perch_w    0.50 \
        --sed_w      0.50 \
        --percentile "$THR" \
        $PERCH_ARG \
        --out        "$PSEUDO_OUT" \
        > "${LOG}/gen_pseudo_thr${THR}.log" 2>&1

    if [ "$SWAPPED" -eq 1 ] && [ -f "$BACKUP_NPZ" ]; then
        cp "$BACKUP_NPZ" "$ORIG_NPZ"
        rm "$BACKUP_NPZ"
    fi

    echo "$PSEUDO_OUT"
}

run_r2_fold0() {
    local PSEUDO_CSV="$1" THR="$2"
    local FOLD0_LOG="${LOG}/sed_ns_20s_r2_thr${THR}_fold0.log"
    local FOLD0_CKPT="outputs/sed-ns-b0-20s-r2/fold0_best.pt"

    # Always retrain fold0 (delete old ckpt if exists)
    rm -f "$FOLD0_CKPT"
    mkdir -p outputs/sed-ns-b0-20s-r2

    # Update R2 config to use this pseudo CSV
    sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO_CSV}|" configs/sed_ns_b0_20s_r2.yaml

    log "Training R2 fold0 with thr=${THR} (early_stop=3) ..."
    python3 train_sed_ns.py \
        --config configs/sed_ns_b0_20s_r2.yaml \
        --fold   0 \
        --device "$DEVICE" \
        > "$FOLD0_LOG" 2>&1

    local AUC
    AUC=$(get_best_auc "$FOLD0_LOG")
    log "  thr=${THR} → fold0 AUC=${AUC}  (target≥${R1_BEST_AUC})"
    echo "$AUC"
}

# ── Main search ───────────────────────────────────────────────────────────────

log "Starting threshold search (R1 fold0 target=${R1_BEST_AUC})"
log "Thresholds to try: 99 98 97 96 95 94 93"

BEST_AUC="0"
BEST_THR=""
declare -A RESULTS

for THR in 99 98 97 96 95 94 93; do
    log "══════════ Threshold=${THR} ══════════"

    PSEUDO_CSV=$(gen_pseudo_with_thr "$THR")
    AUC=$(run_r2_fold0 "$PSEUDO_CSV" "$THR")
    RESULTS[$THR]="$AUC"

    # Track best overall
    if python3 -c "exit(0 if float('${AUC:-0}') > float('${BEST_AUC}') else 1)" 2>/dev/null; then
        BEST_AUC="$AUC"
        BEST_THR="$THR"
    fi

    log "  Results so far:"
    for t in "${!RESULTS[@]}"; do
        log "    thr=${t} → ${RESULTS[$t]}"
    done

    if beats_r1 "$AUC"; then
        log "✓ R2 BEATS R1 at threshold=${THR}! AUC=${AUC} ≥ ${R1_BEST_AUC}"
        BEST_THR="$THR"
        BEST_AUC="$AUC"
        break
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────

log ""
log "═══════════ SEARCH COMPLETE ═══════════"
log "Results:"
for t in 99 98 97 96 95 94 93; do
    [ -n "${RESULTS[$t]+x}" ] && log "  thr=${t} → AUC=${RESULTS[$t]}"
done
log "Best: threshold=${BEST_THR}  AUC=${BEST_AUC}"

if [ -z "$BEST_THR" ]; then
    log "ERROR: No threshold found. Exiting without continuing chain."
    exit 1
fi

# Use best threshold even if it didn't beat R1 (pick whichever was highest)
if ! beats_r1 "$BEST_AUC"; then
    log "WARNING: Best AUC=${BEST_AUC} did not reach R1 target ${R1_BEST_AUC}."
    log "Continuing with best found threshold=${BEST_THR}."
fi

WINNING_CSV="pseudo_labels/sed_20s_r1_thr${BEST_THR}.csv"
sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${WINNING_CSV}|" configs/sed_ns_b0_20s_r2.yaml
log "R2 config updated → pseudo_labels_csv: ${WINNING_CSV}"

# ── Continue full chain from R2 fold1 onwards ─────────────────────────────────

log ""
log "Launching full chain from R2 fold1 → R4 ..."
nohup bash scripts/auto_sed_ns_20s_full.sh >> "${LOG}/auto_sed_ns_20s_full.log" 2>&1 &
log "auto_sed chain launched (PID=$!)"
log "Monitor: tail -f ${LOG}/auto_sed_ns_20s_full.log"
