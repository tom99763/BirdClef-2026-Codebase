#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/lab/miniconda3/envs/tom/bin/python3
WEIGHT_DIR="/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/new direction/weights/sed"
LOG="outputs/logs/export_pvt_r8.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [EXPORT-PVT-R8] $*"; }

cd /home/lab/BirdClef-2026-Codebase
mkdir -p outputs/logs

log "Waiting for PVT R8 fold4 checkpoint..."
while [ ! -f "outputs/sed-ns-pvt-20s-r8/fold4_best.pt" ]; do
    sleep 60
done

# Wait until fold4 training is fully done (pipeline log shows fold4: done)
log "Checkpoint found. Waiting for pipeline to mark fold4 done..."
while ! grep -q "PVT-R8 fold4: done" outputs/logs/auto_sed_ns_pvt_20s_r5r8.log 2>/dev/null; do
    sleep 60
done

log "PVT R8 fold4 done. Starting ONNX export..."

for F in 0 1 2 3 4; do
    PT="outputs/sed-ns-pvt-20s-r8/fold${F}_best.pt"
    OUT="${WEIGHT_DIR}/sed_ns_pvt_r8_fold${F}.onnx"
    if [ -f "$OUT" ]; then
        log "fold${F}: already exists, skipping"
        continue
    fi
    log "Exporting fold${F}..."
    $PYTHON scripts/export_sed_to_onnx.py \
        --pt       "$PT" \
        --out      "$OUT" \
        --backbone pvt_v2_b0 \
        --fp32 \
        --verify
    log "fold${F}: done → $OUT"
done

log "PVT R8 ONNX export complete (all 5 folds)"
