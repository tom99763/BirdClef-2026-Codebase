#!/usr/bin/env bash
# =============================================================================
# Experiment: Number of Random Clips per Recording
#
# Hypothesis: Each training recording can be sampled multiple times with
# different random start positions, creating a cheap form of data augmentation.
# More clips per file means more gradient updates per epoch but also
# proportionally more compute per epoch.
#
# n_clips=1 : each file contributes one clip → smallest dataset, least compute
# n_clips=3 : default
# n_clips=5 : larger effective dataset, more augmentation diversity
# n_clips=8 : aggressively samples long recordings
#
# Note: for short clips (<5 s) all crops are identical (zero-padded), so the
# real benefit only applies to recordings longer than ~10 s.
#
# Runs: n_clips ∈ {1, 3, 5, 8}
# =============================================================================
set -e
cd "$(dirname "$0")/.."

CLIPS=("1" "3" "5" "8")

for N in "${CLIPS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Clips per file sweep — n_clips_per_file=${N}"
    echo "============================================================"

    python train.py \
        --config configs/default.yaml \
        experiment.name="clips_n${N}" \
        audio.n_clips_per_file="${N}"
done

echo ""
echo "Clips-per-file sweep complete."
