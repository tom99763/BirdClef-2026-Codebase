#!/usr/bin/env bash
# =============================================================================
# Experiment: Data Quality Filters
#
# Hypothesis 1 (rating filter):
#   train.csv has a 'rating' column (0–5).  Recordings with low ratings may
#   be mislabelled or have poor SNR, adding noisy supervision signal.
#   Filtering them out could give a cleaner training distribution — but at the
#   cost of fewer examples for rare species.
#
# Hypothesis 2 (secondary labels):
#   Each recording has primary_label (definite) and secondary_labels (audible
#   but less certain species).  Including them adds weak multi-label signal;
#   excluding them keeps labels clean but ignores real co-occurrences.
#
# Hypothesis 3 (soundscape data in training):
#   Train soundscapes come with segment-level labels and closely match the
#   test distribution (Pantanal soundscapes).  Adding them to training could
#   improve domain alignment.
#
# Runs:
#   Rating filter:        no filter (baseline) | min_rating=3 | min_rating=4
#   Secondary labels:     with (baseline) | without
#   Soundscapes in train: off (baseline) | on
# =============================================================================
set -e
cd "$(dirname "$0")/.."

# ── Rating filter sweep ───────────────────────────────────────────────────────
RATINGS=("0.0" "3.0" "4.0")
for RATING in "${RATINGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Data quality — min_rating=${RATING}"
    echo "============================================================"

    python train.py \
        --config configs/default.yaml \
        experiment.name="dq_rating${RATING}" \
        data.min_rating="${RATING}"
done

# ── Secondary labels ablation ────────────────────────────────────────────────
for USE_SEC in "true" "false"; do
    echo ""
    echo "============================================================"
    echo "  Data quality — use_secondary_labels=${USE_SEC}"
    echo "============================================================"

    python train.py \
        --config configs/default.yaml \
        experiment.name="dq_secondary_${USE_SEC}" \
        data.use_secondary_labels="${USE_SEC}"
done

# ── Soundscape data in training ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Data quality — use_soundscapes_in_train=true"
echo "============================================================"

python train.py \
    --config configs/default.yaml \
    experiment.name="dq_with_soundscapes" \
    training.use_soundscapes_in_train="true"

echo ""
echo "Data quality experiment complete."
