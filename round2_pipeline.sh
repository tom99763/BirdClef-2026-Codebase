#!/usr/bin/env bash
# round2_pipeline.sh — Round 2 full pipeline
# 1. Wait for pseudo label generation (PID=$1)
# 2. Extract nohuman label features for R2 cache (copy R1 + add R2 pseudo)
# 3. Train nohuman-label-head-r2 (GPU 0) + nohuman-label-pseudo-r2 (GPU 1) in parallel
# 4. Evaluate both runs

set -e
cd "$(dirname "$0")"

PID_PSEUDO=$1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: wait for pseudo generation ───────────────────────────────────────
log "Waiting for Round 2 pseudo label generation (PID=$PID_PSEUDO)..."
while kill -0 $PID_PSEUDO 2>/dev/null; do
    N=$(grep -c "^BC2026" outputs/pseudo_generate_r2.log 2>/dev/null || echo 0)
    log "  Progress: ~$N soundscapes processed"
    sleep 120
done
log "Pseudo generation done."

# Validate
if [ ! -f pseudo_labels/round2_pseudo.csv ]; then
    log "ERROR: pseudo_labels/round2_pseudo.csv not found!"
    exit 1
fi
NROWS=$(wc -l < pseudo_labels/round2_pseudo.csv)
log "Round 2 pseudo labels: $NROWS rows"

# ── Step 2: Build R2 cache ───────────────────────────────────────────────────
# Copy R1 cache (train + soundscape splits) then add new pseudo features
log "Building R2 cache from R1 cache..."
mkdir -p outputs/embeddings_cache_nohuman_label_r2

# Copy train + soundscape splits from R1
python3 -c "
import pandas as pd, shutil, os

r1_dir = 'outputs/embeddings_cache_nohuman_label'
r2_dir = 'outputs/embeddings_cache_nohuman_label_r2'
manifest = pd.read_csv(f'{r1_dir}/manifest.csv')

# Copy non-pseudo rows and their files
keep = manifest[manifest['split'] != 'pseudo'].copy()
path_col = 'npy_path' if 'npy_path' in manifest.columns else 'features_path'
for _, row in keep.iterrows():
    src = row[path_col]
    dst = src.replace(r1_dir, r2_dir)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

# Write partial manifest (will be extended with new pseudo)
keep.to_csv(f'{r2_dir}/manifest.csv', index=False)
print(f'Copied {len(keep)} rows to R2 cache (train+soundscape)')
"

# Extract new pseudo features
log "Extracting Round 2 pseudo label features..."
CUDA_VISIBLE_DEVICES=0 python extract_pseudo_label_features.py \
    --pseudo_csv pseudo_labels/round2_pseudo.csv \
    --cache_dir outputs/embeddings_cache_nohuman_label_r2 \
    --gpu 0 \
    >> outputs/extract_r2_pseudo.log 2>&1
log "R2 pseudo features extracted."

# ── Step 3: Train both R2 models in parallel ─────────────────────────────────
log "Starting nohuman-label-head-r2 (GPU 0) and nohuman-label-pseudo-r2 (GPU 1)..."

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --config configs/exp_nohuman_label_head_r2.yaml \
    > outputs/nohuman-label-head-r2.log 2>&1 &
PID_A=$!

CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --config configs/exp_nohuman_label_pseudo_r2.yaml \
    > outputs/nohuman-label-pseudo-r2.log 2>&1 &
PID_B=$!

log "nohuman-label-head-r2 PID=$PID_A | nohuman-label-pseudo-r2 PID=$PID_B"

# Monitor
while kill -0 $PID_A 2>/dev/null || kill -0 $PID_B 2>/dev/null; do
    A=$(grep "Epoch " outputs/nohuman-label-head-r2.log 2>/dev/null | grep -oP "Epoch +\K[0-9]+/[0-9]+" | tail -1)
    B=$(grep "Epoch " outputs/nohuman-label-pseudo-r2.log 2>/dev/null | grep -oP "Epoch +\K[0-9]+/[0-9]+" | tail -1)
    BA=$(grep "best=" outputs/nohuman-label-head-r2.log 2>/dev/null | grep -oP "best=\K[0-9.]+" | tail -1)
    BB=$(grep "best=" outputs/nohuman-label-pseudo-r2.log 2>/dev/null | grep -oP "best=\K[0-9.]+" | tail -1)
    log "head-r2: $A (best=$BA) | pseudo-r2: $B (best=$BB)"
    sleep 120
done

log "Round 2 training complete. Evaluating..."

# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python evaluate_final.py \
    --runs nohuman-label-head-r2 nohuman-label-pseudo-r2 \
    --gpu 0 \
    >> outputs/evaluate_r2_final.log 2>&1

log "Round 2 evaluation done."
grep -E "Official|Score|ROC-AUC" outputs/evaluate_r2_final.log | tail -10
