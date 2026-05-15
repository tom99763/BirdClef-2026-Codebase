#!/bin/bash
# Tucker SED NS chain — correct NS design:
#   - GT labels for 66 labeled soundscapes are NEVER touched
#   - Pseudo labels applied to UNLABELED soundscapes only (10,592 files, cached subset)
#   - BASE (R0) = teacher for pseudo labels; student updates each round
#
# One-time setup (auto-runs if not exists):
#   python scripts/cache_unlabeled_ss.py --n 2000  (cached to birdclef-2026/unlabeled_ss_cache/)
#
# Pseudo label blend (BASE as fixed teacher):
#   R1: base_w=0.50  student_w=0.50
#   R2: base_w=0.30  student_w=0.70
#   R3+:base_w=0.05  student_w=0.95
#
# Usage:
#   GPU=1 START_ROUND=1 END_ROUND=8 \
#     nohup bash scripts/auto_tucker_ns.sh > outputs/logs/tucker_ns_b0.log 2>&1 &

set -euo pipefail

PYTHON=/home/lab/miniconda3/envs/tom/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/outputs/logs"
mkdir -p "$LOG_DIR" "$ROOT/pseudo_labels"

GPU=${GPU:-0}
START_ROUND=${START_ROUND:-1}
END_ROUND=${END_ROUND:-8}
BACKBONE=${BACKBONE:-tf_efficientnet_b0.ns_jft_in1k}
ARCH=${ARCH:-b0}
WORKERS=${WORKERS:-4}
N_UNLABELED=${N_UNLABELED:-2000}   # unlabeled soundscapes to cache

BASE_DIR="$ROOT/outputs/tucker-sed-b0"   # R0 checkpoint dir
UNLABELED_CACHE="$ROOT/birdclef-2026/unlabeled_ss_cache"
UNLABELED_META="$UNLABELED_CACHE/unlabeled_ss_cache_meta.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [tucker-ns-${ARCH}-GPU${GPU}] $*"; }

ns_dir()     { echo "$ROOT/outputs/tucker-ns-${ARCH}-r${1}"; }
pseudo_csv() { echo "$ROOT/pseudo_labels/tucker_ns_${ARCH}_unlabeled_r${1}.csv"; }

# ── Step 0: one-time cache of unlabeled soundscapes ──────────────────────────
if [ ! -f "$UNLABELED_META" ]; then
    log "Caching ${N_UNLABELED} unlabeled soundscapes → $UNLABELED_CACHE"
    $PYTHON "$ROOT/scripts/cache_unlabeled_ss.py" \
        --n "$N_UNLABELED" \
        --out "$UNLABELED_CACHE" \
        2>&1 | tee -a "$LOG_DIR/tucker_ns_${ARCH}_cache.log"
else
    N_CACHED=$(wc -l < "$UNLABELED_META")
    log "Unlabeled cache exists: $((N_CACHED-1)) windows in $UNLABELED_CACHE"
fi

# ── BASE pseudo labels for unlabeled soundscapes (run once) ──────────────────
BASE_PSEUDO="$(pseudo_csv 0)"
if [ ! -f "$BASE_PSEUDO" ]; then
    log "Running BASE inference on unlabeled soundscapes → $BASE_PSEUDO"
    $PYTHON "$ROOT/scripts/infer_tucker_ns_ss.py" \
        --ckpt_dir "$BASE_DIR" \
        --out "$BASE_PSEUDO" \
        --gpu "$GPU" \
        --backbone "$BACKBONE" \
        --cache_dir "$UNLABELED_CACHE" \
        --cache_meta "$UNLABELED_META" \
        2>&1 | tee -a "$LOG_DIR/tucker_ns_${ARCH}_infer_r0.log"
else
    log "BASE unlabeled pseudo labels already exist: $BASE_PSEUDO"
fi

blend_pseudo() {
    local ROUND=$1
    local OUT_CSV=$(pseudo_csv "$ROUND")
    local PREV_CSV=$(pseudo_csv "$((ROUND - 1))")  # previous student predictions

    if   [ "$ROUND" -eq 1 ]; then BASE_W=0.50; STUDENT_W=0.50
    elif [ "$ROUND" -eq 2 ]; then BASE_W=0.30; STUDENT_W=0.70
    else                           BASE_W=0.05; STUDENT_W=0.95
    fi

    log "Blending pseudo labels R${ROUND}: base_w=${BASE_W} student_w=${STUDENT_W}"
    $PYTHON - <<EOF
import pandas as pd, numpy as np
base = pd.read_csv("$BASE_PSEUDO")
prev = pd.read_csv("$PREV_CSV")
key = ["filename", "start_sec"]
merged = base.merge(prev, on=key, suffixes=("_base","_prev"))
species = [c for c in base.columns if c not in key]
out = merged[key].copy()
for s in species:
    out[s] = ${BASE_W} * merged[s + "_base"] + ${STUDENT_W} * merged[s + "_prev"]
out.to_csv("$OUT_CSV", index=False)
print(f"Blended pseudo labels saved: {len(out)} rows → $OUT_CSV")
EOF
}

# ── NS rounds ─────────────────────────────────────────────────────────────────
for ROUND in $(seq "$START_ROUND" "$END_ROUND"); do
    OUT_DIR=$(ns_dir "$ROUND")
    PSEUDO_CSV=$(pseudo_csv "$ROUND")

    log "===== ROUND $ROUND ====="

    if [ "$ROUND" -eq 1 ]; then
        PREV_DIR="$BASE_DIR"
    else
        PREV_DIR=$(ns_dir "$((ROUND - 1))")
    fi

    # Generate pseudo labels
    if [ ! -f "$PSEUDO_CSV" ]; then
        if [ "$ROUND" -eq 1 ]; then
            log "R1: using BASE unlabeled pseudo labels directly"
            cp "$BASE_PSEUDO" "$PSEUDO_CSV"
        else
            blend_pseudo "$ROUND"
        fi
    else
        log "Pseudo CSV already exists: $PSEUDO_CSV"
    fi

    # Train
    mkdir -p "$OUT_DIR"
    log "Training R${ROUND} → $OUT_DIR"
    PYTHONUNBUFFERED=1 $PYTHON "$ROOT/train_tucker_sed.py" \
        --backbone "$BACKBONE" \
        --folds 0,1,2,3,4 \
        --gpu "$GPU" \
        --num_workers "$WORKERS" \
        --epochs 25 \
        --patience 3 \
        --init_ckpt "$PREV_DIR" \
        --ema_decay 0.99 \
        --pseudo_unlabeled_sc_csv "$PSEUDO_CSV" \
        --unlabeled_sc_cache_dir "$UNLABELED_CACHE" \
        --out "$OUT_DIR" \
        2>&1 | tee "$LOG_DIR/tucker_ns_${ARCH}_r${ROUND}.log"

    # If any fold checkpoint is missing (e.g., interrupted), fall back to prev round's checkpoint
    for k in 0 1 2 3 4; do
        CKPT="$OUT_DIR/fold${k}_best_ns22.pt"
        PREV_CKPT="$PREV_DIR/fold${k}_best_ns22.pt"
        if [ ! -f "$CKPT" ] && [ -f "$PREV_CKPT" ]; then
            log "WARN: fold${k} checkpoint missing, copying from prev round as fallback"
            cp "$PREV_CKPT" "$CKPT"
        fi
    done

    # Infer student on unlabeled soundscapes for next round's pseudo labels
    if [ "$ROUND" -lt "$END_ROUND" ]; then
        NEXT_CSV=$(pseudo_csv "$ROUND")
        log "Running R${ROUND} student inference → unlabeled pseudo for R$((ROUND+1))"
        $PYTHON "$ROOT/scripts/infer_tucker_ns_ss.py" \
            --ckpt_dir "$OUT_DIR" \
            --out "$NEXT_CSV" \
            --gpu "$GPU" \
            --backbone "$BACKBONE" \
            --cache_dir "$UNLABELED_CACHE" \
            --cache_meta "$UNLABELED_META" \
            2>&1 | tee -a "$LOG_DIR/tucker_ns_${ARCH}_infer_r${ROUND}.log"
    fi

    log "Round $ROUND done."
done

log "Tucker NS chain complete (R${START_ROUND}→R${END_ROUND})."
