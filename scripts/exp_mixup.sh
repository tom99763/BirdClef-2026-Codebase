#!/usr/bin/env bash
# =============================================================================
# Experiment: Mixup Alpha
#
# Hypothesis: Mixing raw waveforms from two different recordings forces the
# model to be more calibrated and reduces overfitting, especially useful
# because many species have very few training samples.
#
# Mixup alpha=0 is effectively disabled.
# Higher alpha → stronger mixing (labels are no longer one-hot).
#
# Runs: alpha ∈ {0 (off), 0.2, 0.4, 0.6}
# =============================================================================
set -e
cd "$(dirname "$0")/.."

ALPHAS=("0.0" "0.2" "0.4" "0.6")

for ALPHA in "${ALPHAS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Mixup sweep — mixup_alpha=${ALPHA}"
    echo "============================================================"

    python train.py \
        --config configs/default.yaml \
        experiment.name="mixup_alpha_${ALPHA}" \
        training.mixup_alpha="${ALPHA}"
done

echo ""
echo "Mixup sweep complete."
