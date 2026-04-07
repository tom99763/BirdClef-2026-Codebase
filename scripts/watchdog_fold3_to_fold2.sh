#!/bin/bash
# ============================================================
# Watchdog: fold3 → fold2 handover
# Monitors fold3 (PID 3681155). When it exits, launches fold2
# on GPU1 (CUDA_VISIBLE_DEVICES=1).
# Run in: tmux birdclef-gpu0:2 or birdclef-gpu1:2
# ============================================================
cd /home/lab/BirdClef-2026-Codebase

FOLD3_PID=3681155
LOG_PREFIX="[watchdog-fold3→fold2]"

log() { echo "$LOG_PREFIX $(date '+%H:%M:%S') $*"; }

log "Started. Monitoring fold3 PID=$FOLD3_PID ..."

while kill -0 $FOLD3_PID 2>/dev/null; do
    # Print heartbeat with latest epoch info
    RESULT=$(python3 -c "
import json
try:
    d = json.load(open('outputs/sed-b0-4fold-v30-fold3/result.json'))
    h = d.get('epoch_history', [])
    best = d.get('best_val_roc_auc', '?')
    ep = len(h)
    fin = d.get('finished', False)
    print(f'ep={ep}, best={best:.4f}, finished={fin}')
except Exception as e:
    print(f'(error reading result.json: {e})')
" 2>/dev/null)
    log "fold3 running — $RESULT"
    sleep 300  # check every 5 min
done

log "fold3 PID=$FOLD3_PID has exited."

# Verify fold3 actually finished cleanly
FINISHED=$(python3 -c "
import json, os
p = 'outputs/sed-b0-4fold-v30-fold3/result.json'
if os.path.exists(p):
    d = json.load(open(p))
    print(d.get('finished', False))
else:
    print(False)
" 2>/dev/null)

log "fold3 finished=$FINISHED"

# Check RAM before launching fold2
FREE_GB=$(free -g | awk '/Mem:/{print $7}')
log "Available RAM: ${FREE_GB} GB"

if [ "$FREE_GB" -lt 18 ]; then
    log "WARNING: Only ${FREE_GB} GB free — waiting 60s for RAM to free up..."
    sleep 60
    FREE_GB=$(free -g | awk '/Mem:/{print $7}')
    log "RAM after wait: ${FREE_GB} GB"
fi

log "Launching fold2 on GPU1 (CUDA_VISIBLE_DEVICES=1)..."
CUDA_VISIBLE_DEVICES=1 nohup python train_sed.py \
    --config configs/sed_b0_4fold_v30_fold2.yaml \
    --resume checkpoints/sed-b0-4fold-v30-fold2/soup_ep007_sed.pt \
    --extra_epochs 27 \
    >> outputs/sed-b0-4fold-v30-fold2/train.log 2>&1 &

FOLD2_PID=$!
log "fold2 launched with PID=$FOLD2_PID"
log "Tail logs: tail -f outputs/sed-b0-4fold-v30-fold2/train.log"
