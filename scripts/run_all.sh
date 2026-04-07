#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Master Orchestrator (Dual-GPU Parallel)
#
# Both GPUs stay busy at all times:
#
#   GPU_A (stream_a): Phase1 → v6(30ep) → Optuna → ASL(30ep) → SoftSec(30ep)
#   GPU_B (stream_b): V2-S(30ep) → 10s-evals → CutMix(30ep)
#
# Each training block is followed immediately by holdout eval + soup + soup-holdout.
#
# Usage:
#   bash scripts/run_all.sh [GPU_A] [GPU_B]
#   bash scripts/run_all.sh 1 0   # default
# ============================================================

set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase

GPU_A=${1:-1}
GPU_B=${2:-0}
LOG="outputs/run_all.log"
mkdir -p outputs submissions/weights

log() { echo "[$(date '+%H:%M:%S')][MASTER] $*" | tee -a "$LOG"; }

LOCK="/tmp/birdclef_run_all.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
    echo "ABORT: already running (PID=$(cat $LOCK))"; exit 1
fi
echo $$ > "$LOCK"
trap "rm -f $LOCK /tmp/birdclef_stream_a_done /tmp/birdclef_stream_b_done" EXIT
rm -f /tmp/birdclef_stream_a_done /tmp/birdclef_stream_b_done

log "============================================================"
log " BirdCLEF 2026 Dual-GPU Orchestrator  PID=$$"
log " GPU_A=$GPU_A | GPU_B=$GPU_B"
log ""
log " GPU_A: Phase1 → v6(CE) → Optuna → ASL → SoftSec"
log " GPU_B: V2-S → 10s-evals → CutMix"
log "============================================================"

# Launch both streams in parallel
bash scripts/stream_a.sh "$GPU_A" &
PID_A=$!
log "Stream A launched (PID=$PID_A)"

bash scripts/stream_b.sh "$GPU_B" &
PID_B=$!
log "Stream B launched (PID=$PID_B)"

# Wait for both streams
wait $PID_A && log "Stream A finished OK" || log "Stream A exited with error"
wait $PID_B && log "Stream B finished OK" || log "Stream B exited with error"

# ── Final results table ────────────────────────────────────────────────────────
log ""
log "============================================================"
log " ALL EXPERIMENTS COMPLETE — Holdout AUC Ranking"
python3 -c "
import json, glob
results = []
for p in sorted(glob.glob('outputs/*/holdout_eval*.json')):
    try:
        d = json.load(open(p))
        name = d.get('run_name', p)
        auc = d.get('holdout_auc') or d.get('holdout_auc_10s')
        if auc: results.append((name, float(auc)))
    except: pass
print(f'  {\"Model\":<55} Holdout AUC')
print(f'  {\"-\"*66}')
for name, auc in sorted(results, key=lambda x: x[1], reverse=True):
    print(f'  {name:<55} {auc:.4f}')
" 2>&1 | tee -a "$LOG"
log "============================================================"
