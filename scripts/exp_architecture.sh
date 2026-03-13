#!/usr/bin/env bash
# =============================================================================
# Experiment: Classification Head Architecture
#
# Hypothesis: Perch embeddings are high-quality and compact.  A larger head
# may not help (or may overfit), while more dropout can improve calibration.
#
# We sweep hidden_dim × dropout independently.
#
# hidden_dim: how much capacity the head has to remap Perch's embedding space
# dropout:    regularisation — important since some species have <10 examples
#
# Runs:
#   hidden_dim ∈ {256, 512 (baseline), 1024}  (dropout fixed at 0.3)
#   dropout    ∈ {0.1, 0.3 (baseline), 0.5}   (hidden_dim fixed at 512)
# =============================================================================
set -e
cd "$(dirname "$0")/.."

# ── Hidden dim sweep (dropout fixed) ────────────────────────────────────────
DIMS=("256" "512" "1024")
for DIM in "${DIMS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Arch sweep — hidden_dim=${DIM}  dropout=0.3"
    echo "============================================================"

    python train.py \
        --config configs/default.yaml \
        experiment.name="arch_dim${DIM}_dp0.3" \
        model.hidden_dim="${DIM}" \
        model.dropout="0.3"
done

# ── Dropout sweep (hidden_dim fixed at 512) ──────────────────────────────────
DROPOUTS=("0.1" "0.3" "0.5")
for DP in "${DROPOUTS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Arch sweep — hidden_dim=512  dropout=${DP}"
    echo "============================================================"

    python train.py \
        --config configs/default.yaml \
        experiment.name="arch_dim512_dp${DP}" \
        model.hidden_dim="512" \
        model.dropout="${DP}"
done

echo ""
echo "Architecture sweep complete."
