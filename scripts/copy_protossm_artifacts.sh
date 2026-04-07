#!/usr/bin/env bash
# 等待 proto_ssm_v18 訓練完成，自動複製 artifacts 到 weights/protossm/
set -euo pipefail
LOG="/home/lab/BirdClef-2026-Codebase/outputs/logs/proto_ssm_v18_rerun.log"
SRC="/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/new direction/pretrained"
DST="/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/new direction/weights/protossm"

echo "[$(date)] Waiting for ProtoSSM V18 rerun to finish..."
while pgrep -f "train_proto_ssm_v18.py" > /dev/null 2>&1; do
    sleep 60
done

# Check finished successfully
if grep -q "All V18 artifacts saved" "$LOG"; then
    echo "[$(date)] Training complete. Copying artifacts..."
    cp "$SRC/proto_ssm_v18.pt"           "$DST/"
    cp "$SRC/residual_ssm_v18.pt"        "$DST/"
    cp "$SRC/prior_tables_v18.pkl"       "$DST/"
    cp "$SRC/sklearn_v18.pkl"            "$DST/"
    cp "$SRC/thresholds_v18.npy"         "$DST/"
    cp "$SRC/artifacts_manifest_v18.json" "$DST/"
    echo "[$(date)] Done. Artifacts copied to weights/protossm/"
    # Print final AUC
    grep "Mean OOF AUC" "$LOG" | tail -1
else
    echo "[$(date)] WARNING: Training may have failed. Check $LOG"
fi
