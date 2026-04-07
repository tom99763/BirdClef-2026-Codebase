#!/bin/bash
# v5-style soundscape 4-fold training on GPU1
# Replicates best_sed_b0_v5.pt hyperparameters with proper k-fold splits
# Pair1 (fold0+fold1) parallel, then Pair2 (fold2+fold3)
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

echo "[$(date '+%H:%M:%S')] Starting v5-style 4-fold on GPU$GPU"

# Pair 1: fold0 + fold1
echo "[$(date '+%H:%M:%S')] ===== Launching sed-v5-fold0 on GPU$GPU ====="
mkdir -p outputs/sed-v5-fold0
CUDA_VISIBLE_DEVICES=$GPU python train_sed.py --config configs/sed_v5_fold0.yaml \
    > outputs/sed-v5-fold0/train.log 2>&1 &
PID0=$!

echo "[$(date '+%H:%M:%S')] ===== Launching sed-v5-fold1 on GPU$GPU ====="
mkdir -p outputs/sed-v5-fold1
CUDA_VISIBLE_DEVICES=$GPU python train_sed.py --config configs/sed_v5_fold1.yaml \
    > outputs/sed-v5-fold1/train.log 2>&1 &
PID1=$!

echo "[$(date '+%H:%M:%S')] Pair1 PIDs: $PID0 $PID1 — waiting..."
wait $PID0; report sed-v5-fold0
wait $PID1; report sed-v5-fold1

# Pair 2: fold2 + fold3
echo "[$(date '+%H:%M:%S')] ===== Launching sed-v5-fold2 on GPU$GPU ====="
mkdir -p outputs/sed-v5-fold2
CUDA_VISIBLE_DEVICES=$GPU python train_sed.py --config configs/sed_v5_fold2.yaml \
    > outputs/sed-v5-fold2/train.log 2>&1 &
PID2=$!

echo "[$(date '+%H:%M:%S')] ===== Launching sed-v5-fold3 on GPU$GPU ====="
mkdir -p outputs/sed-v5-fold3
CUDA_VISIBLE_DEVICES=$GPU python train_sed.py --config configs/sed_v5_fold3.yaml \
    > outputs/sed-v5-fold3/train.log 2>&1 &
PID3=$!

echo "[$(date '+%H:%M:%S')] Pair2 PIDs: $PID2 $PID3 — waiting..."
wait $PID2; report sed-v5-fold2
wait $PID3; report sed-v5-fold3

echo "[$(date '+%H:%M:%S')] v5-style 4-fold complete."
python3 -c "
import json, os
results = []
for k in range(4):
    p = f'outputs/sed-v5-fold{k}/result.json'
    if os.path.exists(p):
        d = json.load(open(p))
        h = d.get('epoch_history', [])
        best = max((e.get('val_roc_auc',0) for e in h), default=0)
        results.append(best)
        print(f'  sed-v5-fold{k}: best={best:.4f}')
if results:
    print(f'  Average best AUC: {sum(results)/len(results):.4f}')
"
