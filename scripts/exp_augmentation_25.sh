#!/usr/bin/env bash
# =============================================================================
# Experiment: BirdCLEF 2025 Augmentation Techniques
#
# Tests augmentation techniques used by top BirdCLEF 2025 teams:
#   - Time masking (SpecAugment on waveform) — universal across top solutions
#   - Combined: time masking + existing noise/gain
#
# Background noise injection is excluded here as it requires a noise_dir.
# To test background noise, set data.noise_dir and add augmentation.background_noise=true.
#
# Runs:
#   1. aug_baseline         — existing aug only (noise + gain + mixup)
#   2. aug_time_mask        — + time masking (2 masks, 10% ratio)
#   3. aug_time_mask_heavy  — + time masking (3 masks, 15% ratio)
#   4. aug_time_mask_light  — + time masking (1 mask, 5% ratio)
# =============================================================================
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  BirdCLEF 2025 Experiment: Augmentation Techniques"
echo "============================================================"

# 1. Baseline augmentation (no new techniques)
echo ""
echo "[1/4] Baseline augmentation (noise + gain + mixup)..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="aug_baseline_25" \
    augmentation.time_masking=false

# 2. + Time masking (default: 2 masks, 10%)
echo ""
echo "[2/4] + Time masking (2 masks, 10% ratio)..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="aug_time_mask_2x10" \
    augmentation.time_masking=true \
    augmentation.time_mask_n=2 \
    augmentation.time_mask_ratio=0.1

# 3. + Time masking (heavy: 3 masks, 15%)
echo ""
echo "[3/4] + Time masking heavy (3 masks, 15% ratio)..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="aug_time_mask_3x15" \
    augmentation.time_masking=true \
    augmentation.time_mask_n=3 \
    augmentation.time_mask_ratio=0.15

# 4. + Time masking (light: 1 mask, 5%)
echo ""
echo "[4/4] + Time masking light (1 mask, 5% ratio)..."
python train.py \
    --config configs/birdclef25_improvements.yaml \
    experiment.name="aug_time_mask_1x5" \
    augmentation.time_masking=true \
    augmentation.time_mask_n=1 \
    augmentation.time_mask_ratio=0.05

echo ""
echo "Augmentation sweep complete. Compare with:"
echo "  python analyze_results.py"
echo ""
echo "To test background noise injection, run:"
echo "  python train.py --config configs/birdclef25_improvements.yaml \\"
echo "      experiment.name=aug_bg_noise \\"
echo "      data.noise_dir=/path/to/noise_files \\"
echo "      augmentation.background_noise=true"
