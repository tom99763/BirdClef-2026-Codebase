#!/usr/bin/env bash
# =============================================================================
# Run: BirdCLEF 2025 Improvements — Base Training
#
# Trains with all BirdCLEF 2025 static improvements enabled:
#   - FocalLoss (2nd & 5th place)
#   - Sqrt inverse-frequency class weighting (2nd place)
#   - Time masking / SpecAugment (universal across top teams)
#   - AdamW + cosine annealing + warmup
#   - Mixup + label smoothing (existing)
#
# This is the starting point before pseudo-labeling rounds.
# Run this first, then run exp_pseudo_label.sh to stack pseudo-label rounds.
#
# Usage:
#   bash scripts/run_birdclef25_base.sh
# =============================================================================
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  BirdCLEF 2025 Base Training"
echo "  Config: configs/birdclef25_improvements.yaml"
echo "============================================================"

python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="birdclef25-base"

echo ""
echo "Base training complete."
echo "Best checkpoint: checkpoints/birdclef25-base/best_head"
echo ""
echo "Next steps:"
echo "  1. Generate pseudo-labels and retrain:"
echo "       bash scripts/exp_pseudo_label.sh"
echo "  2. Or run inference directly:"
echo "       python inference.py --config configs/birdclef25_improvements.yaml \\"
echo "           --checkpoint checkpoints/birdclef25-base/best_head --tta"
