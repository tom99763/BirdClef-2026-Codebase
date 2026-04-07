#!/bin/bash
# Sequential B0 embedding distillation experiments on GPU1
# Order: b0-v2 (in_chans=3) → b0-v3 (InfoNCE) → b0-v4 (heavy aug)
#
# Usage: bash scripts/launch_embed_distill_experiments.sh 2>&1 | tee outputs/embed_distill_experiments.log

cd /home/lab/BirdClef-2026-Codebase

EXPERIMENTS=(
    "configs/embed_distill_b0_v2.yaml"
    "configs/embed_distill_b0_v3.yaml"
    "configs/embed_distill_b0_v4.yaml"
)

for CONFIG in "${EXPERIMENTS[@]}"; do
    RUN=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG')); print(d['run_name'])")
    echo "[$(date)] =============================="
    echo "[$(date)] Starting: $RUN"
    echo "[$(date)] Config:   $CONFIG"
    echo "[$(date)] =============================="

    CUDA_VISIBLE_DEVICES=1 python3 train_embed_distill.py \
        --config "$CONFIG" \
        --gpu 1 \
        2>&1 | tee "outputs/${RUN}.log"

    echo "[$(date)] Finished: $RUN"
    VAL=$(python3 -c "import json; d=json.load(open('outputs/${RUN}/result.json')); print(f\"{d['best_val_cos']:.4f}\")" 2>/dev/null || echo "?")
    echo "[$(date)] best_val_cos=${VAL}  backbone → checkpoints/${RUN}/best_backbone.pt"
    echo ""
done

echo "[$(date)] All B0 embedding distillation experiments complete."
echo "Summary:"
for CONFIG in "${EXPERIMENTS[@]}"; do
    RUN=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG')); print(d['run_name'])")
    VAL=$(python3 -c "import json; d=json.load(open('outputs/${RUN}/result.json')); print(f\"{d['best_val_cos']:.4f}\")" 2>/dev/null || echo "?")
    echo "  ${RUN}: best_val_cos=${VAL}"
done
