#!/bin/bash
# Parallel soundscape 4-fold training on GPU1
# fold0+fold1 run simultaneously, then fold2+fold3
cd /home/lab/BirdClef-2026-Codebase
GPU=1

report() {
    local name=$1
    local out_dir="outputs/$name"
    local best=$(python3 -c "
import json, os
p = '$out_dir/result.json'
if os.path.exists(p):
    d = json.load(open(p))
    h = d.get('epoch_history', [])
    best = max((e.get('val_roc_auc',0) for e in h), default=0)
    ep = next((e['epoch'] for e in h if e.get('val_roc_auc',0)==best), 0)
    print(f'{best:.4f}@ep{ep}')
else:
    print('no result')
" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] $name DONE — best=$best"
}

echo "[$(date '+%H:%M:%S')] Starting soundscape 4-fold on GPU$GPU (2 parallel)"

# Pair 1: fold0 + fold1
echo "[$(date '+%H:%M:%S')] ===== Launching sed-ss-fold0 on GPU$GPU ====="
mkdir -p outputs/sed-ss-fold0
CUDA_VISIBLE_DEVICES=$GPU python train_sed.py --config configs/sed_ss_fold0.yaml > outputs/sed-ss-fold0/train.log 2>&1 &
PID0=$!

echo "[$(date '+%H:%M:%S')] ===== Launching sed-ss-fold1 on GPU$GPU ====="
mkdir -p outputs/sed-ss-fold1
CUDA_VISIBLE_DEVICES=$GPU python train_sed.py --config configs/sed_ss_fold1.yaml > outputs/sed-ss-fold1/train.log 2>&1 &
PID1=$!

echo "[$(date '+%H:%M:%S')] Pair1 PIDs: $PID0 $PID1 — waiting..."
wait $PID0; report sed-ss-fold0
wait $PID1; report sed-ss-fold1

# Pair 2: fold2 + fold3
echo "[$(date '+%H:%M:%S')] ===== Launching sed-ss-fold2 on GPU$GPU ====="
mkdir -p outputs/sed-ss-fold2
CUDA_VISIBLE_DEVICES=$GPU python train_sed.py --config configs/sed_ss_fold2.yaml > outputs/sed-ss-fold2/train.log 2>&1 &
PID2=$!

echo "[$(date '+%H:%M:%S')] ===== Launching sed-ss-fold3 on GPU$GPU ====="
mkdir -p outputs/sed-ss-fold3
CUDA_VISIBLE_DEVICES=$GPU python train_sed.py --config configs/sed_ss_fold3.yaml > outputs/sed-ss-fold3/train.log 2>&1 &
PID3=$!

echo "[$(date '+%H:%M:%S')] Pair2 PIDs: $PID2 $PID3 — waiting..."
wait $PID2; report sed-ss-fold2
wait $PID3; report sed-ss-fold3

echo "[$(date '+%H:%M:%S')] All 4 folds complete."
