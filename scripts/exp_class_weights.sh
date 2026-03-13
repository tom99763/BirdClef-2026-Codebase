#!/usr/bin/env bash
# =============================================================================
# Experiment: Class Frequency Weighting  (BirdCLEF 2025 2nd place technique)
#
# Rare species are underrepresented in training data. Weighted sampling ensures
# rare species recordings are seen more often per epoch.
#
# weight = 1 / freq^power
#   none   → uniform sampling (baseline)
#   sqrt   → weight ∝ 1/sqrt(freq)  [2nd place BirdCLEF25]
#   linear → weight ∝ 1/freq        (aggressive, may hurt common species)
#
# Runs:
#   1. class_weight_none    — uniform sampling (reference)
#   2. class_weight_sqrt    — sqrt inverse frequency (2nd place)
#   3. class_weight_linear  — linear inverse frequency
# =============================================================================
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  BirdCLEF 2025 Experiment: Class Frequency Weighting"
echo "============================================================"

# 1. Uniform (reference)
echo ""
echo "[1/3] No class weighting (uniform sampling)..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="class_weight_none" \
    training.class_weight_mode="none"

# 2. Sqrt inverse frequency (BirdCLEF25 2nd place)
echo ""
echo "[2/3] Sqrt inverse frequency weighting (BirdCLEF25 2nd place)..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="class_weight_sqrt" \
    training.class_weight_mode="sqrt"

# 3. Linear inverse frequency
echo ""
echo "[3/3] Linear inverse frequency weighting..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="class_weight_linear" \
    training.class_weight_mode="linear"

echo ""
echo "Class weighting sweep complete. Compare with:"
echo "  python analyze_results.py"
