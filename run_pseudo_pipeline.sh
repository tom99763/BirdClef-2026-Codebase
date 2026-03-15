#!/bin/bash
# Pseudo-label pipeline (Round 1)
# Run after best-derived-v2 training completes.
#
# Step 1: Generate pseudo-labels using best-derived-v2 checkpoint (GPU 0)
# Step 2: Extract Perch embeddings for pseudo-labeled segments (GPU 1)
# Step 3: Train pseudo-r1 experiment (GPU 0)

set -e

CHECKPOINT="checkpoints/best-derived-v2/best_head"
PSEUDO_CSV="pseudo_labels/round1_pseudo.csv"
PSEUDO_CACHE="outputs/embeddings_cache_pseudo"
CONFIG="configs/default.yaml"

mkdir -p pseudo_labels

echo "=== Step 1: Generate pseudo-labels ==="
CUDA_VISIBLE_DEVICES=0 python3 pseudo_label.py generate \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --output $PSEUDO_CSV \
    --threshold 0.45 \
    --power 2.0

echo "Pseudo-labels generated: $(wc -l < $PSEUDO_CSV) rows"

echo "=== Step 2: Extract pseudo embeddings ==="
CUDA_VISIBLE_DEVICES=0 python3 extract_pseudo_embeddings.py \
    --config $CONFIG \
    --pseudo_csv $PSEUDO_CSV \
    --cache_dir $PSEUDO_CACHE \
    --batch_size 16

echo "=== Step 3: Train pseudo-r1 ==="
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --config configs/exp_pseudo_r1.yaml \
    data.pseudo_manifest_csv=$PSEUDO_CACHE/manifest.csv \
    > outputs/pseudo-r1.log 2>&1

echo "=== Pseudo-label pipeline complete ==="
