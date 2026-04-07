#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
LOG="outputs/logs"
mkdir -p "$LOG"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STACKER-SS] $*" | tee -a "$LOG/stacker_ss_chain.log"; }

# GPU conflict guard
while pgrep -f "python3.*train_stacker" > /dev/null; do
    log "Waiting for previous stacker training to finish..."
    sleep 60
done

log "=== Starting stacker-ss chain (9 architectures + pseudo labels) ==="
python3 scripts/train_stacker_v3_ss.py > "$LOG/stacker_ss_train.log" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    log "=== stacker-ss chain DONE (exit 0) ==="
else
    log "=== stacker-ss chain FAILED (exit $EXIT_CODE) ==="
    exit $EXIT_CODE
fi
