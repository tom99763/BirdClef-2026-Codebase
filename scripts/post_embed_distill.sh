#!/bin/bash
# Post-embed-distill decision script.
# Reads result.json, checks if val_cos is still improving.
# If still improving (delta > 0.003 over last 5 ep): extend 20 more epochs.
# If converged: print backbone path and exit (caller will launch SED v15).

cd /home/lab/BirdClef-2026-Codebase
RESULT="outputs/embed-distill-b0-v1/result.json"
CONFIG="configs/embed_distill_b0_v1.yaml"
LOG="outputs/embed_distill.log"

echo "[$(date)] post_embed_distill: evaluating convergence..."

DECISION=$(python3 - <<'EOF'
import json

with open("outputs/embed-distill-b0-v1/result.json") as f:
    d = json.load(f)

hist = d.get("epoch_history", [])
if len(hist) < 5:
    print("extend")  # too few epochs to judge
else:
    last5 = hist[-5:]
    first_cos = last5[0]["val_cos"]
    last_cos  = last5[-1]["val_cos"]
    delta = last_cos - first_cos
    best  = d.get("best_val_cos", 0)
    print(f"[convergence check] last5 delta={delta:.4f}  best={best:.4f}", flush=True)
    # Extend if still gaining > 0.003 over last 5 epochs
    if delta > 0.003:
        print("extend")
    else:
        print("done")
EOF
)

echo "[$(date)] decision output: $DECISION"

if echo "$DECISION" | grep -q "extend"; then
    echo "[$(date)] Still improving — extending 20 more epochs..."
    CUDA_VISIBLE_DEVICES=0 python3 train_embed_distill.py \
        --config "$CONFIG" \
        --gpu 0 \
        --extra_epochs 20 \
        2>&1 | tee -a "$LOG"
    # Re-run decision after extension
    bash scripts/post_embed_distill.sh
else
    echo "[$(date)] Converged. backbone ready at checkpoints/embed-distill-b0-v1/best_backbone.pt"
    echo "[$(date)] val_cos=$(python3 -c "import json; d=json.load(open('outputs/embed-distill-b0-v1/result.json')); print(f\"{d['best_val_cos']:.4f}\")")"
    echo "[$(date)] Ready to launch SED v15 with --pretrained_backbone"
fi
