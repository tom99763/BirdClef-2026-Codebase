#!/usr/bin/env bash
# BirdCLEF 2025 Top-10 Improvements Pipeline
#
# Implements the multi-round noisy student approach from BirdCLEF 2025 1st place
# combined with FocalLoss, class weighting, TTA, and Model Soup.
#
# Usage:
#   bash scripts/run_birdclef25_pipeline.sh
#
# Estimated improvement over baseline: +5-10% padded cMAP
# (Based on BirdCLEF 2025: 0.872 baseline → 0.933 with 4 pseudo-label rounds)

set -e

echo "============================================================"
echo "  BirdCLEF 2025 Top-10 Improvements Pipeline"
echo "============================================================"

# ── Step 1: Base training with BirdCLEF 2025 improvements ────────────────────
# FocalLoss + sqrt class weighting + time masking + AdamW
echo ""
echo "[Step 1] Training with BirdCLEF 2025 improvements..."
python train.py --config configs/birdclef25_improvements.yaml \
    experiment.name=birdclef25-base-v1

# ── Step 2: Pseudo-label Round 1 ─────────────────────────────────────────────
# Generate soft pseudo-labels with PowerTransform (1st place key technique)
echo ""
echo "[Step 2] Generating pseudo-labels (Round 1, power=2.0)..."
python pseudo_label.py generate \
    --config configs/birdclef25_improvements.yaml \
    --checkpoint checkpoints/birdclef25-base-v1/best_head \
    --soundscapes_dir birdclef-2026/train_soundscapes \
    --output pseudo_labels/round1_pseudo.csv \
    --threshold 0.5 \
    --power 2.0

# ── Step 3: Retrain with pseudo-labels (Round 1) ─────────────────────────────
echo ""
echo "[Step 3] Training with Round 1 pseudo-labels..."
python train.py --config configs/pseudo_label_round1.yaml \
    experiment.name=pseudo-label-round1

# ── Step 4: Pseudo-label Round 2 ─────────────────────────────────────────────
echo ""
echo "[Step 4] Generating pseudo-labels (Round 2, power=2.0)..."
python pseudo_label.py generate \
    --config configs/pseudo_label_round1.yaml \
    --checkpoint checkpoints/pseudo-label-round1/best_head \
    --soundscapes_dir birdclef-2026/train_soundscapes \
    --output pseudo_labels/round2_pseudo.csv \
    --threshold 0.5 \
    --power 2.0

# ── Step 5: Retrain with Round 2 pseudo-labels ───────────────────────────────
echo ""
echo "[Step 5] Training with Round 2 pseudo-labels..."
python train.py --config configs/pseudo_label_round1.yaml \
    experiment.name=pseudo-label-round2

# ── Step 6: Model Soup (average best checkpoints) ────────────────────────────
# BirdCLEF 2025 3rd place: checkpoint weight averaging
echo ""
echo "[Step 6] Model Soup: averaging top checkpoints..."
mkdir -p checkpoints/soup
python -m src.utils.model_soup \
    --checkpoints \
        checkpoints/birdclef25-base-v1/best_head \
        checkpoints/pseudo-label-round1/best_head \
        checkpoints/pseudo-label-round2/best_head \
    --output checkpoints/soup/best_head \
    --config configs/birdclef25_improvements.yaml

# ── Step 7: Final inference with TTA ─────────────────────────────────────────
# BirdCLEF 2025 2nd place: temporal shift TTA (+0.012 AUC)
echo ""
echo "[Step 7] Final inference with TTA (temporal shifts)..."
python inference.py \
    --config configs/birdclef25_improvements.yaml \
    --checkpoint checkpoints/soup/best_head \
    --output submission_birdclef25.csv \
    --tta

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Final submission: submission_birdclef25.csv"
echo "  Checkpoints used: birdclef25-base-v1, pseudo-r1, pseudo-r2 → soup"
echo "============================================================"
