#!/usr/bin/env bash
# =============================================================================
# Experiment: Multi-round Pseudo-labeling  (BirdCLEF 2025 1st place technique)
#
# The highest-ROI technique from BirdCLEF 2025:
#   1st place: 4 rounds → 0.872 → 0.933 (+6.1% cMAP)
#   2nd place: 2 rounds → 0.87  → 0.91  (+4.0% cMAP)
#
# This script runs 2 rounds of pseudo-labeling (production-ready default).
# Extend to 4 rounds by appending rounds 3 & 4 following the same pattern.
#
# Pipeline:
#   [Stage 0] Train base model with BirdCLEF25 improvements
#   [Round 1] Generate pseudo-labels → retrain
#   [Round 2] Generate pseudo-labels → retrain
#   [Final]   Model Soup + TTA inference → submission
#
# Prerequisites:
#   - Trained base checkpoint at checkpoints/birdclef25-base/best_head
#     (run scripts/run_birdclef25_base.sh or reuse an existing run)
#
# Output:
#   - checkpoints/pseudo-r1/best_head
#   - checkpoints/pseudo-r2/best_head
#   - pseudo_labels/round1_pseudo.csv
#   - pseudo_labels/round2_pseudo.csv
#   - submission_pseudo_r2.csv
# =============================================================================
set -e
cd "$(dirname "$0")/.."

PSEUDO_DIR="pseudo_labels"
mkdir -p "$PSEUDO_DIR"

echo "============================================================"
echo "  BirdCLEF 2025 Pseudo-labeling Pipeline (2 rounds)"
echo "============================================================"

# ── Stage 0: Base training ────────────────────────────────────────────────────
# Skip if checkpoint already exists (so you can resume from here)
BASE_CKPT="checkpoints/birdclef25-base/best_head.index"
if [ -f "$BASE_CKPT" ]; then
    echo ""
    echo "[Stage 0] Base checkpoint found — skipping base training."
    echo "  $BASE_CKPT"
else
    echo ""
    echo "[Stage 0] Training base model (FocalLoss + class weights + time masking)..."
    python train.py \
        --config configs/birdclef25_improvements.yaml \
        experiment.name="birdclef25-base"
fi

# ── Round 1: Generate pseudo-labels ──────────────────────────────────────────
echo ""
echo "[Round 1 — Generate] Generating pseudo-labels with PowerTransform (power=2.0)..."
python pseudo_label.py generate \
    --config configs/birdclef25_improvements.yaml \
    --checkpoint checkpoints/birdclef25-base/best_head \
    --soundscapes_dir birdclef-2026/train_soundscapes \
    --output "${PSEUDO_DIR}/round1_pseudo.csv" \
    --threshold 0.5 \
    --power 2.0

echo ""
echo "[Round 1 — Train] Retraining with Round 1 pseudo-labels..."
python train.py \
    --config configs/pseudo_label_round1.yaml \
    experiment.name="pseudo-r1"

# ── Round 2: Generate pseudo-labels ──────────────────────────────────────────
echo ""
echo "[Round 2 — Generate] Generating pseudo-labels from Round 1 model..."
python pseudo_label.py generate \
    --config configs/pseudo_label_round1.yaml \
    --checkpoint checkpoints/pseudo-r1/best_head \
    --soundscapes_dir birdclef-2026/train_soundscapes \
    --output "${PSEUDO_DIR}/round2_pseudo.csv" \
    --threshold 0.5 \
    --power 2.0

echo ""
echo "[Round 2 — Train] Retraining with Round 2 pseudo-labels..."
python train.py \
    --config configs/pseudo_label_round1.yaml \
    experiment.name="pseudo-r2"

# ── Model Soup: average all checkpoints ──────────────────────────────────────
echo ""
echo "[Model Soup] Averaging checkpoints from all rounds..."
mkdir -p checkpoints/soup-pseudo
python -m src.utils.model_soup \
    --checkpoints \
        checkpoints/birdclef25-base/best_head \
        checkpoints/pseudo-r1/best_head \
        checkpoints/pseudo-r2/best_head \
    --output checkpoints/soup-pseudo/best_head \
    --config configs/birdclef25_improvements.yaml

# ── Final inference with TTA ──────────────────────────────────────────────────
echo ""
echo "[Inference] Generating submission with TTA (temporal shifts)..."
python inference.py \
    --config configs/birdclef25_improvements.yaml \
    --checkpoint checkpoints/soup-pseudo/best_head \
    --output submission_pseudo_r2.csv \
    --tta

echo ""
echo "============================================================"
echo "  Pseudo-labeling experiment complete!"
echo ""
echo "  Checkpoints:"
echo "    Stage 0 : checkpoints/birdclef25-base/best_head"
echo "    Round 1 : checkpoints/pseudo-r1/best_head"
echo "    Round 2 : checkpoints/pseudo-r2/best_head"
echo "    Soup    : checkpoints/soup-pseudo/best_head"
echo ""
echo "  Submission: submission_pseudo_r2.csv"
echo ""
echo "  Compare all runs:"
echo "    python analyze_results.py"
echo "============================================================"
