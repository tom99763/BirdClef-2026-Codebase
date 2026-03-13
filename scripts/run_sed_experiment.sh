#!/usr/bin/env bash
# =============================================================================
# Run: SED Model Experiment — Full Pipeline
#
# Trains the best-configuration SED model (EfficientNetV2-S + attention pooling,
# BirdCLEF 2025 2nd & 5th place architecture) and generates a submission.
#
# Pipeline:
#   1. Train SED model (EfficientNetV2-S, FocalLoss, sqrt class weights)
#   2. Inference with TTA → submission_sed.csv
#   3. Ensemble with Perch submission (if available) → submission_ensemble.csv
#
# Prerequisites:
#   pip install torch timm torchaudio
#
# Usage:
#   bash scripts/run_sed_experiment.sh
#   bash scripts/run_sed_experiment.sh --skip-train   # inference only
# =============================================================================
set -e
cd "$(dirname "$0")/.."

SKIP_TRAIN=false
for arg in "$@"; do
    [ "$arg" = "--skip-train" ] && SKIP_TRAIN=true
done

echo "============================================================"
echo "  BirdClef 2026 — SED Model Experiment"
echo "  Backbone: EfficientNetV2-S (in21k)"
echo "  Config  : configs/sed_default.yaml"
echo "============================================================"

# ── Step 1: Sanity check ─────────────────────────────────────────────────────
echo ""
echo "[Step 1] Running debug sanity check (3 epochs, 200 files)..."
python train_sed.py --config configs/sed_debug.yaml
echo "  Debug run OK."

# ── Step 2: Full SED training ─────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[Step 2] Training full SED model (50 epochs)..."
    python train_sed.py \
        --config configs/sed_default.yaml \
        experiment.name="sed-efficientnetv2s"
else
    echo ""
    echo "[Step 2] Skipping training (--skip-train)."
fi

# ── Step 3: Inference with TTA ────────────────────────────────────────────────
echo ""
echo "[Step 3] SED inference with TTA..."
python inference_sed.py \
    --config configs/sed_default.yaml \
    --checkpoint checkpoints/sed-efficientnetv2s/best_sed \
    --output submission_sed.csv \
    --tta

# ── Step 4: Ensemble with Perch submission (optional) ─────────────────────────
PERCH_SUBMISSION=""
for candidate in submission_birdclef25.csv submission_final.csv submission.csv; do
    if [ -f "$candidate" ]; then
        PERCH_SUBMISSION="$candidate"
        break
    fi
done

if [ -n "$PERCH_SUBMISSION" ]; then
    echo ""
    echo "[Step 4] Ensembling SED + Perch predictions..."
    echo "  Perch submission: ${PERCH_SUBMISSION}"
    python inference_sed.py \
        --config configs/sed_default.yaml \
        --checkpoint checkpoints/sed-efficientnetv2s/best_sed \
        --output submission_ensemble.csv \
        --tta \
        --ensemble_with "${PERCH_SUBMISSION}"
    echo "  Ensemble submission: submission_ensemble.csv"
else
    echo ""
    echo "[Step 4] No Perch submission found — skipping ensemble."
    echo "  Run the Perch pipeline first to generate one, then ensemble with:"
    echo "    python inference_sed.py --config configs/sed_default.yaml \\"
    echo "        --checkpoint checkpoints/sed-efficientnetv2s/best_sed \\"
    echo "        --tta --ensemble_with submission_perch.csv \\"
    echo "        --output submission_ensemble.csv"
fi

echo ""
echo "============================================================"
echo "  SED experiment complete!"
echo ""
echo "  SED submission     : submission_sed.csv"
if [ -f "submission_ensemble.csv" ]; then
echo "  Ensemble submission: submission_ensemble.csv"
fi
echo ""
echo "  Compare with Perch results:"
echo "    python analyze_results.py"
echo "============================================================"
