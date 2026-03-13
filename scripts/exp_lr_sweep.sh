#!/usr/bin/env bash
# =============================================================================
# Experiment: Learning Rate Sweep
#
# Hypothesis: The default lr=1e-3 may be too aggressive or too conservative
# for the MLP head on top of Perch embeddings.  Cosine annealing makes the
# peak (initial) LR the most critical hyperparameter.
#
# Runs: lr ∈ {5e-4, 1e-3 (baseline), 3e-3, 5e-3}
# =============================================================================
set -e
cd "$(dirname "$0")/.."

LRS=("5e-4" "1e-3" "3e-3" "5e-3")

for LR in "${LRS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  LR sweep — learning_rate=${LR}"
    echo "============================================================"

    python train.py \
        --config configs/default.yaml \
        experiment.name="lr_sweep_${LR}" \
        training.learning_rate="${LR}"
done

echo ""
echo "LR sweep complete.  Compare val/padded_cmap across runs in WandB."
