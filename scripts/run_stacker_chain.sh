#!/usr/bin/env bash
# run_stacker_chain.sh — Run the full stacker-v3 chain (9 architectures).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

LOG="outputs/logs"
mkdir -p "$LOG"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STACKER-V3] $*" | tee -a "$LOG/stacker_v3_chain.log"
}

# Wait for GPU1 to be free (other stacker jobs)
while pgrep -f "train_stacker" > /dev/null; do
    log "Waiting for previous stacker training to finish..."
    sleep 60
done

log "=== Starting stacker-v3 chain (9 architectures) ==="
log "  Feature layout: perch_raw | perch_prior_fused | mlp_probe | proto_ssm | sed_csebbs (1170 dim)"
log "  Architectures : LGBM, XGB, MLP, BiGRU, TCN, Transformer, SSM, FT-Transformer, CNN1D"
log "  GPU           : CUDA_VISIBLE_DEVICES=1"

python3 scripts/train_stacker_v3.py > "$LOG/stacker_v3_train.log" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    log "=== stacker-v3 chain DONE (exit 0) ==="
else
    log "=== stacker-v3 chain FAILED (exit $EXIT_CODE) — check $LOG/stacker_v3_train.log ==="
    exit $EXIT_CODE
fi
