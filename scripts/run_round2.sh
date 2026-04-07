#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Round 2 Orchestrator (Dual-GPU Parallel)
#
# Run AFTER run_all.sh completes.
# All experiments are B0-only.
#
#   GPU_C (stream_c): v12-bce(30ep) → v13-asl-cutmix(30ep)
#   GPU_D (stream_d): v14-50ep(50ep) → v15-no-sec(30ep) → v16-rating3(30ep)
#
# GPU_C total: ~20h  |  GPU_D total: ~33h  (v14 is the bottleneck)
#
# Usage:
#   bash scripts/run_round2.sh [GPU_C] [GPU_D]
#   bash scripts/run_round2.sh 1 0   # default
# ============================================================

set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase

GPU_C=${1:-1}
GPU_D=${2:-0}
LOG="outputs/run_round2.log"
mkdir -p outputs submissions/weights

log() { echo "[$(date '+%H:%M:%S')][MASTER] $*" | tee -a "$LOG"; }

LOCK="/tmp/birdclef_round2.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
    echo "ABORT: already running (PID=$(cat $LOCK))"; exit 1
fi
echo $$ > "$LOCK"
trap "rm -f $LOCK /tmp/birdclef_stream_c_done /tmp/birdclef_stream_d_done" EXIT
rm -f /tmp/birdclef_stream_c_done /tmp/birdclef_stream_d_done

log "============================================================"
log " BirdCLEF 2026 Round 2  PID=$$"
log " GPU_C=$GPU_C | GPU_D=$GPU_D"
log ""
log " GPU_C: v12-bce(30ep) → v13-asl-cutmix(30ep)           ~20h"
log " GPU_D: v14-50ep(50ep) → v15-no-sec(30ep) → v16-rating3(30ep)  ~33h"
log "============================================================"

bash scripts/stream_c.sh "$GPU_C" &
PID_C=$!
log "Stream C launched (PID=$PID_C)"

bash scripts/stream_d.sh "$GPU_D" &
PID_D=$!
log "Stream D launched (PID=$PID_D)"

wait $PID_C && log "Stream C finished OK" || log "Stream C exited with error"
wait $PID_D && log "Stream D finished OK" || log "Stream D exited with error"

# ── Final results table ────────────────────────────────────────────────────────
log ""
log "============================================================"
log " ROUND 2 COMPLETE — Holdout AUC Ranking"
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
