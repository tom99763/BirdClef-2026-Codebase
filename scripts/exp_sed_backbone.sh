#!/usr/bin/env bash
# =============================================================================
# Experiment: SED Backbone Comparison  (BirdCLEF 2025 5th place architecture)
#
# BirdCLEF 2025 5th place used 13 SED models across 4 backbone families:
#   4× EfficientNetV2-S
#   3× EfficientNetV2-B3
#   4× EfficientNet-B3-NS  (noisy-student pretrained)
#   2× EfficientNet-B0-NS
#
# This sweep evaluates each backbone on the same config to find the best
# single-model architecture before building an ensemble.
#
# Runs:
#   1. sed_b0_ns   — EfficientNet-B0-NS  (fast, lightweight)
#   2. sed_b3_ns   — EfficientNet-B3-NS  (5th place favourite)
#   3. sed_v2_b3   — EfficientNetV2-B3   (5th place)
#   4. sed_v2_s    — EfficientNetV2-S    (2nd & 5th place best single)
#   5. sed_nfnet   — ECA-NFNet-L0        (2nd place)
#
# Usage:
#   bash scripts/exp_sed_backbone.sh
# =============================================================================
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  BirdCLEF 2025 Experiment: SED Backbone Comparison"
echo "============================================================"

declare -A BACKBONES=(
    ["sed_b0_ns"]="tf_efficientnet_b0_ns"
    ["sed_b3_ns"]="tf_efficientnet_b3_ns"
    ["sed_v2_b3"]="tf_efficientnetv2_b3"
    ["sed_v2_s"]="tf_efficientnetv2_s_in21k"
    ["sed_nfnet"]="eca_nfnet_l0"
)

for RUN_NAME in sed_b0_ns sed_b3_ns sed_v2_b3 sed_v2_s sed_nfnet; do
    BACKBONE="${BACKBONES[$RUN_NAME]}"
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Backbone: ${BACKBONE}  →  run: ${RUN_NAME}"
    echo "──────────────────────────────────────────────────────────"

    python train_sed.py \
        --config configs/sed_default.yaml \
        experiment.name="${RUN_NAME}" \
        model.backbone="${BACKBONE}"
done

echo ""
echo "============================================================"
echo "  SED Backbone sweep complete!"
echo ""
echo "  Compare results:"
echo "    python analyze_results.py"
echo ""
echo "  To build an ensemble from the top-N backbones, run"
echo "  inference_sed.py for each and average the CSVs."
echo "============================================================"
