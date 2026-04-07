#!/usr/bin/env bash
# =============================================================================
# Phase 2+ Launch Script — BirdCLEF 2026
# Launches after v21/v22 (GPU0) and v23/v24 (GPU1) complete.
#
# GPU0 chain:  embed-distill-b0-v5 → sed-b0-v25-embed-head → sed-v2s-v2-asl
# GPU1 chain:  sed-b0-v26-asl-npcen → sed-b0-v27-soft-boost → sed-b0-v28-final-combo
#
# Usage:
#   bash scripts/phase2_launch.sh          # launch both GPU chains
#   bash scripts/phase2_launch.sh gpu0     # GPU0 only
#   bash scripts/phase2_launch.sh gpu1     # GPU1 only
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_TARGET="${1:-both}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
run_sed() {
    local gpu="$1" config="$2" log="$3"
    shift 3
    local extra="$*"
    echo "[chain] Starting: $config on GPU$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python train_sed.py \
        --config "$config" $extra \
        2>&1 | tee "$log"
    echo "[chain] Finished: $config"
    # Auto-eval after each SED experiment
    local exp_name
    exp_name=$(python3 -c "import yaml; d=yaml.safe_load(open('$config')); print(d['experiment']['name'])")
    local ckpt="checkpoints/${exp_name}/best_sed.pt"
    if [[ -f "$ckpt" ]]; then
        echo "[chain] Running holdout eval: $exp_name"
        CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_sed_holdout.py \
            --checkpoint "$ckpt" \
            --config "$config" \
            --run_name "$exp_name" \
            --gpu "$gpu" \
            2>&1 | tee "${LOG_DIR}/holdout_${exp_name}.log"
        python scripts/update_exp_results.py --run_name "$exp_name" 2>/dev/null || true
    fi
}

run_embed_distill() {
    local gpu="$1" config="$2" log="$3"
    echo "[chain] Starting embed distill: $config on GPU$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python train_embed_distill.py \
        --config "$config" \
        2>&1 | tee "$log"
    echo "[chain] Finished embed distill: $config"
}

# ── GPU 0 chain ───────────────────────────────────────────────────────────────
gpu0_chain() {
    echo "[GPU0] Phase 2 chain starting..."

    # Step 1: Improved embedding distillation
    run_embed_distill 0 \
        configs/embed_distill_b0_v5.yaml \
        "${LOG_DIR}/embed_distill_b0_v5.log"

    # Step 2: SED head-only from embed-distill-b0-v5 backbone
    EDCKPT="checkpoints/embed-distill-b0-v5/best_backbone.pt"
    if [[ -f "$EDCKPT" ]]; then
        EXTRA="--pretrained_backbone $EDCKPT"
    else
        echo "[GPU0] WARNING: embed-distill checkpoint not found, using ImageNet init"
        EXTRA=""
    fi
    run_sed 0 \
        configs/sed_b0_v25_embed_head.yaml \
        "${LOG_DIR}/sed_b0_v25_embed_head.log" \
        $EXTRA

    # Step 3: EfficientNetV2-S with ASL — the 2025 winner backbone
    run_sed 0 \
        configs/sed_v2s_v2_asl.yaml \
        "${LOG_DIR}/sed_v2s_v2_asl.log"

    echo "[GPU0] Phase 2 chain complete!"
}

# ── GPU 1 chain ───────────────────────────────────────────────────────────────
gpu1_chain() {
    echo "[GPU1] Phase 2 chain starting..."

    # Step 1: CRITICAL ABLATION — v23 without PCEN
    run_sed 1 \
        configs/sed_b0_v26_asl_npcen.yaml \
        "${LOG_DIR}/sed_b0_v26_asl_npcen.log"

    # Step 2: Soft KD with 10x oversample
    run_sed 1 \
        configs/sed_b0_v27_soft_boost.yaml \
        "${LOG_DIR}/sed_b0_v27_soft_boost.log"

    # Step 3: Final best-of-all combo
    run_sed 1 \
        configs/sed_b0_v28_final_combo.yaml \
        "${LOG_DIR}/sed_b0_v28_final_combo.log"

    echo "[GPU1] Phase 2 chain complete!"
}

# ── Launch ────────────────────────────────────────────────────────────────────
case "$GPU_TARGET" in
    gpu0) gpu0_chain ;;
    gpu1) gpu1_chain ;;
    both)
        gpu0_chain &
        GPU0_PID=$!
        gpu1_chain &
        GPU1_PID=$!
        wait $GPU0_PID
        wait $GPU1_PID
        echo "[phase2] Both GPU chains complete!"
        ;;
    *)
        echo "Usage: $0 [gpu0|gpu1|both]"
        exit 1
        ;;
esac
