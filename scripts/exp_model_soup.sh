#!/usr/bin/env bash
# =============================================================================
# Experiment: Model Soup — Checkpoint Weight Averaging (BirdCLEF 2025 3rd place)
#
# Instead of picking the single best checkpoint, average the weights of
# multiple checkpoints. This:
#   - Reduces variance without increasing inference cost
#   - Often beats the best single checkpoint by 0.3–1% cMAP
#   - Is free: no extra training needed
#
# Reference: "Model soups: averaging weights of multiple fine-tuned models
# improves accuracy without increasing inference time" (Wortsman et al. 2022)
#
# This script trains 3 independent runs with different seeds, then soups them.
# Alternatively, soup checkpoints from different experiments (e.g., pseudo rounds).
#
# Output:
#   - checkpoints/soup-3seed/best_head    (3-seed average)
#   - submission_soup.csv
# =============================================================================
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  BirdCLEF 2025 Experiment: Model Soup"
echo "============================================================"

# ── Option A: Train 3 seeds and soup ─────────────────────────────────────────
echo ""
echo "[Option A] Training 3 independent seeds..."

for SEED in 42 123 777; do
    echo ""
    echo "  Seed ${SEED}..."
    python train.py \
        --config configs/birdclef25_improvements.yaml \
        experiment.name="soup_seed${SEED}" \
        experiment.seed="${SEED}"
done

echo ""
echo "[Model Soup A] Averaging 3-seed checkpoints..."
mkdir -p checkpoints/soup-3seed
python -m src.utils.model_soup \
    --checkpoints \
        checkpoints/soup_seed42/best_head \
        checkpoints/soup_seed123/best_head \
        checkpoints/soup_seed777/best_head \
    --output checkpoints/soup-3seed/best_head \
    --config configs/birdclef25_improvements.yaml

echo ""
echo "[Inference A] Submission from 3-seed soup..."
python inference.py \
    --config configs/birdclef25_improvements.yaml \
    --checkpoint checkpoints/soup-3seed/best_head \
    --output submission_soup_3seed.csv \
    --tta

# ── Option B: Soup pseudo-label rounds (if they exist) ───────────────────────
CKPTS=""
for CKPT_PATH in \
    checkpoints/birdclef25-base/best_head \
    checkpoints/pseudo-r1/best_head \
    checkpoints/pseudo-r2/best_head; do
    if [ -f "${CKPT_PATH}.index" ]; then
        CKPTS="$CKPTS $CKPT_PATH"
    fi
done

if [ -n "$CKPTS" ]; then
    echo ""
    echo "[Option B] Souping pseudo-label checkpoints:${CKPTS}"
    mkdir -p checkpoints/soup-pseudo-rounds
    python -m src.utils.model_soup \
        --checkpoints $CKPTS \
        --output checkpoints/soup-pseudo-rounds/best_head \
        --config configs/birdclef25_improvements.yaml

    python inference.py \
        --config configs/birdclef25_improvements.yaml \
        --checkpoint checkpoints/soup-pseudo-rounds/best_head \
        --output submission_soup_pseudo.csv \
        --tta
    echo "  submission_soup_pseudo.csv written."
else
    echo ""
    echo "[Option B] No pseudo-label checkpoints found — skipping."
    echo "  Run scripts/exp_pseudo_label.sh first to generate them."
fi

echo ""
echo "============================================================"
echo "  Model Soup experiment complete!"
echo ""
echo "  Submissions:"
echo "    3-seed soup : submission_soup_3seed.csv"
echo "    Pseudo soup : submission_soup_pseudo.csv (if available)"
echo "============================================================"
