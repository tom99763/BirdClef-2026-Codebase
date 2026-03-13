#!/usr/bin/env bash
# =============================================================================
# Strong Run — best settings from all sweep experiments
#
# Fill in the values below after reviewing WandB results from the sweeps.
# Placeholders are set to reasonable priors based on BirdClef literature.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  UPDATE THESE after running the sweep experiments:                      │
# │    BEST_LR          ← best from exp_lr_sweep.sh                        │
# │    BEST_MIXUP       ← best from exp_mixup.sh                           │
# │    BEST_DIM         ← best from exp_architecture.sh                    │
# │    BEST_DROPOUT     ← best from exp_architecture.sh                    │
# │    BEST_RATING      ← best from exp_data_quality.sh                    │
# │    BEST_SEC_LABELS  ← best from exp_data_quality.sh                    │
# │    BEST_CLIPS       ← best from exp_clips_per_file.sh                  │
# └─────────────────────────────────────────────────────────────────────────┘
# =============================================================================
set -e
cd "$(dirname "$0")/.."

# ── Best hyperparameters (update after sweeps) ────────────────────────────────
BEST_LR="1e-3"
BEST_MIXUP="0.4"
BEST_DIM="512"
BEST_DROPOUT="0.3"
BEST_RATING="3.0"
BEST_SEC_LABELS="true"
BEST_CLIPS="5"

echo ""
echo "============================================================"
echo "  Strong run"
echo "  lr=${BEST_LR}  mixup=${BEST_MIXUP}  dim=${BEST_DIM}"
echo "  dropout=${BEST_DROPOUT}  min_rating=${BEST_RATING}"
echo "  secondary=${BEST_SEC_LABELS}  clips=${BEST_CLIPS}"
echo "============================================================"

python train.py \
    --config configs/default.yaml \
    experiment.name="strong_run_v1" \
    training.learning_rate="${BEST_LR}" \
    training.mixup_alpha="${BEST_MIXUP}" \
    model.hidden_dim="${BEST_DIM}" \
    model.dropout="${BEST_DROPOUT}" \
    data.min_rating="${BEST_RATING}" \
    data.use_secondary_labels="${BEST_SEC_LABELS}" \
    audio.n_clips_per_file="${BEST_CLIPS}" \
    training.epochs="80"          # longer for the final run

echo ""
echo "Strong run complete."
