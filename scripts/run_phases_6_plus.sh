#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Phases 6, 7, 8, 9 Orchestrator
#
# Runs after Phase 5 (Optuna ensemble) completes.
#
#   Phase 6 : Train EfficientNetV2-S (sed_v2s_v1) on GPU0
#   Phase 7 : 10-second context window eval on soup/best checkpoints
#   Phase 8 : ASL loss experiment  (sed_b0_v9_asl)    on GPU1
#   Phase 9 : CutMix experiment    (sed_b0_v10_cutmix) on GPU1 (after v9)
#   Phase 10: Soft secondary labels (sed_b0_v11_soft_sec) on GPU1 (after v10)
#
# NOTE: No pseudo-labeling — BirdCLEF 2026 train_soundscapes are fully labeled.
#
# Usage:
#   bash scripts/run_phases_6_plus.sh [GPU0] [GPU1]
#   bash scripts/run_phases_6_plus.sh 0 1      # default
# ============================================================

set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase

GPU0=${1:-0}
GPU1=${2:-1}
LOG="outputs/phases_6_plus.log"
mkdir -p outputs

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Guard: abort if another instance is already running
LOCK="/tmp/birdclef_phases_6_plus.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
    echo "ABORT: another instance already running (PID=$(cat $LOCK)). Exiting."
    exit 1
fi
echo $$ > "$LOCK"
trap "rm -f $LOCK" EXIT

log "=========================================="
log " Phases 6-10 Orchestrator  PID=$$"
log " GPU0=$GPU0 (V2-S)  GPU1=$GPU1 (ASL / CutMix / SoftSec)"
log "=========================================="

# ── Phase 6: Train EfficientNetV2-S (background on GPU0) ──────────────────────
log ""
log "=========================================="
log "Phase 6: Train EfficientNetV2-S (sed_v2s_v1)"
log "  Backbone: tf_efficientnetv2_s.in21k_ft_in1k"
log "  GPU: $GPU0  epochs=30  batch=16"
log "=========================================="

CUDA_VISIBLE_DEVICES=$GPU0 python3 train_sed.py \
    --config configs/sed_v2s_v1.yaml \
    2>&1 | tee -a "$LOG" &
V2S_PID=$!
log "  V2-S training launched (PID=$V2S_PID)"

# ── Phase 7: 10-second context window eval ────────────────────────────────────
log ""
log "=========================================="
log "Phase 7: 10-second Context Window Eval"
log "  Evaluates all available soup/best checkpoints"
log "  2024 BirdCLEF 1st place technique: +0.015 LB"
log "=========================================="

for model_cfg in "sed-b0-v5 configs/sed_b0_v5.yaml" \
                 "sed-b0-v6 configs/sed_b0_v6.yaml" \
                 "sed-b2-v1 configs/sed_b2_v1.yaml"; do
    run_name=$(echo $model_cfg | cut -d' ' -f1)
    run_config=$(echo $model_cfg | cut -d' ' -f2)

    if [ -f "checkpoints/$run_name/soup_sed.pt" ]; then
        ckpt="checkpoints/$run_name/soup_sed.pt"
        tag="${run_name}-soup-10s"
    elif [ -f "checkpoints/$run_name/best_sed.pt" ]; then
        ckpt="checkpoints/$run_name/best_sed.pt"
        tag="${run_name}-best-10s"
    else
        log "  $run_name: no checkpoint found — skipping"
        continue
    fi

    log "  10s eval: $tag"
    CUDA_VISIBLE_DEVICES=$GPU1 python3 scripts/eval_10s_inference.py \
        --checkpoint "$ckpt" \
        --config "$run_config" \
        --run_name "$tag" \
        2>&1 | tee -a "$LOG"

    if [ -f "outputs/$tag/holdout_eval_10s.json" ]; then
        python3 -c "
import json
d = json.load(open('outputs/$tag/holdout_eval_10s.json'))
print(f'  {d[\"run_name\"]:<48} holdout_auc_10s={d[\"holdout_auc_10s\"]}')
" 2>&1 | tee -a "$LOG"
    fi
done

# ── Phase 8: ASL Loss experiment ──────────────────────────────────────────────
log ""
log "=========================================="
log "Phase 8: ASL Loss (sed_b0_v9_asl)"
log "  Asymmetric Loss: gamma_neg=4, gamma_pos=0, clip=0.05"
log "  BirdCLEF 2025 top-3 technique for multi-label"
log "  GPU: $GPU1  epochs=30"
log "=========================================="

CUDA_VISIBLE_DEVICES=$GPU1 python3 train_sed.py \
    --config configs/sed_b0_v9_asl.yaml \
    2>&1 | tee -a "$LOG"
log "  sed_b0_v9_asl training complete"

# Holdout eval
if [ -f "checkpoints/sed-b0-v9-asl/best_sed.pt" ]; then
    CUDA_VISIBLE_DEVICES=$GPU1 python3 scripts/eval_sed_holdout.py \
        --checkpoint checkpoints/sed-b0-v9-asl/best_sed.pt \
        --config configs/sed_b0_v9_asl.yaml \
        --run_name sed-b0-v9-asl \
        2>&1 | tee -a "$LOG"
fi

# Model soup
if ls checkpoints/sed-b0-v9-asl/soup_ep*.pt 1>/dev/null 2>&1; then
    python3 scripts/model_soup.py --run sed-b0-v9-asl --config configs/sed_b0_v9_asl.yaml \
        2>&1 | tee -a "$LOG"
    if [ -f "checkpoints/sed-b0-v9-asl/soup_sed.pt" ]; then
        cp checkpoints/sed-b0-v9-asl/soup_sed.pt submissions/weights/soup_sed_b0_v9_asl.pt
        CUDA_VISIBLE_DEVICES=$GPU1 python3 scripts/eval_sed_holdout.py \
            --checkpoint checkpoints/sed-b0-v9-asl/soup_sed.pt \
            --config configs/sed_b0_v9_asl.yaml \
            --run_name sed-b0-v9-asl-soup \
            2>&1 | tee -a "$LOG"
    fi
fi

# ── Phase 9: CutMix experiment ────────────────────────────────────────────────
log ""
log "=========================================="
log "Phase 9: CutMix augmentation (sed_b0_v10_cutmix)"
log "  CE loss + CutMix(alpha=1.0), no Mixup"
log "  BirdCLEF 2025 multiple teams"
log "  GPU: $GPU1  epochs=30"
log "=========================================="

CUDA_VISIBLE_DEVICES=$GPU1 python3 train_sed.py \
    --config configs/sed_b0_v10_cutmix.yaml \
    2>&1 | tee -a "$LOG"
log "  sed_b0_v10_cutmix training complete"

if [ -f "checkpoints/sed-b0-v10-cutmix/best_sed.pt" ]; then
    CUDA_VISIBLE_DEVICES=$GPU1 python3 scripts/eval_sed_holdout.py \
        --checkpoint checkpoints/sed-b0-v10-cutmix/best_sed.pt \
        --config configs/sed_b0_v10_cutmix.yaml \
        --run_name sed-b0-v10-cutmix \
        2>&1 | tee -a "$LOG"
fi

if ls checkpoints/sed-b0-v10-cutmix/soup_ep*.pt 1>/dev/null 2>&1; then
    python3 scripts/model_soup.py --run sed-b0-v10-cutmix --config configs/sed_b0_v10_cutmix.yaml \
        2>&1 | tee -a "$LOG"
    if [ -f "checkpoints/sed-b0-v10-cutmix/soup_sed.pt" ]; then
        cp checkpoints/sed-b0-v10-cutmix/soup_sed.pt submissions/weights/soup_sed_b0_v10_cutmix.pt
        CUDA_VISIBLE_DEVICES=$GPU1 python3 scripts/eval_sed_holdout.py \
            --checkpoint checkpoints/sed-b0-v10-cutmix/soup_sed.pt \
            --config configs/sed_b0_v10_cutmix.yaml \
            --run_name sed-b0-v10-cutmix-soup \
            2>&1 | tee -a "$LOG"
    fi
fi

# ── Phase 10: Soft secondary labels ───────────────────────────────────────────
log ""
log "=========================================="
log "Phase 10: Soft Secondary Labels (sed_b0_v11_soft_sec)"
log "  secondary_label_weight=0.3 vs 1.0 default"
log "  BirdCLEF 2025 top teams: reduces noise from uncertain labels"
log "  GPU: $GPU1  epochs=30"
log "=========================================="

CUDA_VISIBLE_DEVICES=$GPU1 python3 train_sed.py \
    --config configs/sed_b0_v11_soft_sec.yaml \
    2>&1 | tee -a "$LOG"
log "  sed_b0_v11_soft_sec training complete"

if [ -f "checkpoints/sed-b0-v11-soft-sec/best_sed.pt" ]; then
    CUDA_VISIBLE_DEVICES=$GPU1 python3 scripts/eval_sed_holdout.py \
        --checkpoint checkpoints/sed-b0-v11-soft-sec/best_sed.pt \
        --config configs/sed_b0_v11_soft_sec.yaml \
        --run_name sed-b0-v11-soft-sec \
        2>&1 | tee -a "$LOG"
fi

if ls checkpoints/sed-b0-v11-soft-sec/soup_ep*.pt 1>/dev/null 2>&1; then
    python3 scripts/model_soup.py --run sed-b0-v11-soft-sec --config configs/sed_b0_v11_soft_sec.yaml \
        2>&1 | tee -a "$LOG"
    if [ -f "checkpoints/sed-b0-v11-soft-sec/soup_sed.pt" ]; then
        cp checkpoints/sed-b0-v11-soft-sec/soup_sed.pt submissions/weights/soup_sed_b0_v11_soft_sec.pt
        CUDA_VISIBLE_DEVICES=$GPU1 python3 scripts/eval_sed_holdout.py \
            --checkpoint checkpoints/sed-b0-v11-soft-sec/soup_sed.pt \
            --config configs/sed_b0_v11_soft_sec.yaml \
            --run_name sed-b0-v11-soft-sec-soup \
            2>&1 | tee -a "$LOG"
    fi
fi

# ── Wait for Phase 6 (V2-S) to finish ─────────────────────────────────────────
log ""
log "Waiting for Phase 6 (V2-S, PID=$V2S_PID) …"
wait $V2S_PID && log "  V2-S done (exit 0)" || log "  V2-S exited with error"

if [ -f "checkpoints/sed-v2s-v1/best_sed.pt" ]; then
    log "=== sed-v2s-v1 Holdout Eval ==="
    CUDA_VISIBLE_DEVICES=$GPU0 python3 scripts/eval_sed_holdout.py \
        --checkpoint checkpoints/sed-v2s-v1/best_sed.pt \
        --config configs/sed_v2s_v1.yaml \
        --run_name sed-v2s-v1 \
        2>&1 | tee -a "$LOG"

    if ls checkpoints/sed-v2s-v1/soup_ep*.pt 1>/dev/null 2>&1; then
        python3 scripts/model_soup.py --run sed-v2s-v1 --config configs/sed_v2s_v1.yaml \
            2>&1 | tee -a "$LOG"
        if [ -f "checkpoints/sed-v2s-v1/soup_sed.pt" ]; then
            cp checkpoints/sed-v2s-v1/soup_sed.pt submissions/weights/soup_sed_v2s_v1.pt
            CUDA_VISIBLE_DEVICES=$GPU0 python3 scripts/eval_sed_holdout.py \
                --checkpoint checkpoints/sed-v2s-v1/soup_sed.pt \
                --config configs/sed_v2s_v1.yaml \
                --run_name sed-v2s-v1-soup \
                2>&1 | tee -a "$LOG"

            log "=== sed-v2s-v1-soup 10s Eval ==="
            CUDA_VISIBLE_DEVICES=$GPU0 python3 scripts/eval_10s_inference.py \
                --checkpoint checkpoints/sed-v2s-v1/soup_sed.pt \
                --config configs/sed_v2s_v1.yaml \
                --run_name sed-v2s-v1-soup-10s \
                2>&1 | tee -a "$LOG"
        fi
    fi
fi

# ── Final summary ──────────────────────────────────────────────────────────────
log ""
log "=========================================="
log " Phases 6-10 Complete!"
log " Summary of all holdout results:"
python3 -c "
import json, glob
results = []
for p in sorted(glob.glob('outputs/*/holdout_eval*.json')):
    try:
        d = json.load(open(p))
        name = d.get('run_name', p)
        auc  = d.get('holdout_auc') or d.get('holdout_auc_10s')
        if auc: results.append((name, float(auc)))
    except: pass
for name, auc in sorted(results, key=lambda x: x[1], reverse=True):
    print(f'  {name:<55} {auc:.4f}')
" 2>&1 | tee -a "$LOG"
log "=========================================="
