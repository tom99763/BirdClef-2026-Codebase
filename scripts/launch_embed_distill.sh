#!/bin/bash
# Wait for mel cache manifest to appear, then launch embedding distillation on GPU0
cd /home/lab/BirdClef-2026-Codebase

echo "[$(date)] Waiting for mel cache to complete (outputs/mel_cache/manifest.csv)..."
while [ ! -f outputs/mel_cache/manifest.csv ]; do
    DONE=$(ls outputs/mel_cache/train/ 2>/dev/null | wc -l)
    echo "[$(date)] mel_cache/train: ${DONE}/85536 files cached..."
    sleep 60
done

echo "[$(date)] Mel cache ready! Launching embed distillation on GPU0..."
CUDA_VISIBLE_DEVICES=0 python3 train_embed_distill.py \
    --config configs/embed_distill_b0_v1.yaml \
    --gpu 0 \
    2>&1 | tee outputs/embed_distill.log

echo "[$(date)] Embed distillation complete. Running convergence check..."
bash scripts/post_embed_distill.sh 2>&1 | tee -a outputs/embed_distill.log
