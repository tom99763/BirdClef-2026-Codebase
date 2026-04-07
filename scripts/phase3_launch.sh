#!/usr/bin/env bash
# =============================================================================
# Phase 3 Launch Script — BirdCLEF 2026
# Starts after Phase 2 (v28 on GPU0, v27 on GPU1) completes.
#
# GPU0 chain: sed-b3-v1-asl → (eval) → sed-b3-v1-fold0 → sed-b3-v1-fold2
# GPU1 chain: sed-b0-v30-bgnoise → (eval) → sed-b3-v1-fold1 → sed-b3-v1-fold3
#
# Trigger: run after phase2_gpu0_corrected.sh and phase2_gpu1_corrected.sh finish
# Usage:
#   bash /tmp/phase3_launch.sh          # both GPUs
#   bash /tmp/phase3_launch.sh gpu0     # GPU0 only
#   bash /tmp/phase3_launch.sh gpu1     # GPU1 only
# =============================================================================
set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

GPU_TARGET="${1:-both}"

run_sed_eval() {
    local gpu="$1" config="$2" log="$3"
    local exp_name
    exp_name=$(python3 -c "import yaml; d=yaml.safe_load(open('$config')); print(d['experiment']['name'])")
    local ckpt="checkpoints/${exp_name}/best_sed.pt"
    local soup_ckpt="checkpoints/${exp_name}/soup_sed.pt"

    echo "[phase3] Training: $exp_name on GPU$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python train_sed.py --config "$config" 2>&1 | tee "$log"
    echo "[phase3] Training done: $exp_name"

    if [[ -f "$ckpt" ]]; then
        echo "[phase3] Holdout eval: $exp_name"
        CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_sed_holdout.py \
            --checkpoint "$ckpt" --config "$config" \
            --run_name "$exp_name" --gpu "$gpu" \
            2>&1 | tee "${LOG_DIR}/holdout_${exp_name}.log"
        python scripts/update_exp_results.py --run_name "$exp_name" 2>/dev/null || true

        python scripts/model_soup.py \
            --checkpoint_dir "checkpoints/${exp_name}" \
            --output_path "$soup_ckpt" \
            2>&1 | tee "${LOG_DIR}/soup_${exp_name}.log" || echo "[soup] skipped"

        if [[ -f "$soup_ckpt" ]]; then
            CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_sed_holdout.py \
                --checkpoint "$soup_ckpt" --config "$config" \
                --run_name "${exp_name}-soup" --gpu "$gpu" \
                2>&1 | tee "${LOG_DIR}/holdout_${exp_name}_soup.log"
            python scripts/update_exp_results.py --run_name "${exp_name}-soup" 2>/dev/null || true
        fi
    fi
}

gpu0_chain() {
    echo "[GPU0] Phase 3 chain starting..."

    # Step 1: B3-NS single model test (the 2025 dominant backbone)
    run_sed_eval 0 \
        configs/sed_b3_v1_asl.yaml \
        "${LOG_DIR}/sed_b3_v1_asl.log"

    # Step 2: B3 4-fold fold0
    run_sed_eval 0 \
        configs/sed_b3_v1_fold0.yaml \
        "${LOG_DIR}/sed_b3_v1_fold0.log"

    # Step 3: B3 4-fold fold2
    run_sed_eval 0 \
        configs/sed_b3_v1_fold2.yaml \
        "${LOG_DIR}/sed_b3_v1_fold2.log"

    echo "[GPU0] Phase 3 complete!"
}

gpu1_chain() {
    echo "[GPU1] Phase 3 chain starting..."

    # Step 1: B0 + background noise (isolate bg-noise contribution)
    run_sed_eval 1 \
        configs/sed_b0_v30_bgnoise.yaml \
        "${LOG_DIR}/sed_b0_v30_bgnoise.log"

    # Step 2: B3 4-fold fold1
    run_sed_eval 1 \
        configs/sed_b3_v1_fold1.yaml \
        "${LOG_DIR}/sed_b3_v1_fold1.log"

    # Step 3: B3 4-fold fold3
    run_sed_eval 1 \
        configs/sed_b3_v1_fold3.yaml \
        "${LOG_DIR}/sed_b3_v1_fold3.log"

    echo "[GPU1] Phase 3 complete!"
}

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
        echo "[phase3] Both GPU chains complete!"
        ;;
    *)
        echo "Usage: $0 [gpu0|gpu1|both]"
        exit 1
        ;;
esac
