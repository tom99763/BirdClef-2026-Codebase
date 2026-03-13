#!/usr/bin/env bash
# =============================================================================
# Experiment: FocalLoss vs BCE  (BirdCLEF 2025 2nd & 5th place technique)
#
# FocalLoss reduces the loss contribution from easy (high-confidence) negatives,
# forcing the model to focus on hard/rare class examples.
#
# FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
#   gamma=0  → standard BCE
#   gamma=2  → standard focal (top-solutions default)
#
# Runs:
#   1. bce_baseline          — standard BCE (reference)
#   2. focal_gamma1          — gamma=1.0 (light focusing)
#   3. focal_gamma2          — gamma=2.0 (BirdCLEF25 2nd place default)
#   4. focal_gamma3          — gamma=3.0 (aggressive focusing)
#   5. focal_alpha_neg1      — gamma=2, alpha=-1 (no alpha weighting)
# =============================================================================
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  BirdCLEF 2025 Experiment: FocalLoss vs BCE"
echo "============================================================"

# 1. BCE baseline (reference for this sweep)
echo ""
echo "[1/5] BCE baseline..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="focal_bce_baseline" \
    training.loss="bce"

# 2. Focal gamma=1
echo ""
echo "[2/5] FocalLoss gamma=1.0..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="focal_gamma1" \
    training.loss="focal" \
    training.focal_gamma=1.0

# 3. Focal gamma=2 (default, 2nd place)
echo ""
echo "[3/5] FocalLoss gamma=2.0 (BirdCLEF25 2nd place)..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="focal_gamma2" \
    training.loss="focal" \
    training.focal_gamma=2.0

# 4. Focal gamma=3
echo ""
echo "[4/5] FocalLoss gamma=3.0..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="focal_gamma3" \
    training.loss="focal" \
    training.focal_gamma=3.0

# 5. Focal gamma=2, no alpha weighting
echo ""
echo "[5/5] FocalLoss gamma=2.0, alpha=-1 (no alpha weighting)..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="focal_gamma2_no_alpha" \
    training.loss="focal" \
    training.focal_gamma=2.0 \
    training.focal_alpha=-1.0

echo ""
echo "FocalLoss sweep complete. Compare with:"
echo "  python analyze_results.py"
