#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Perch-as-Teacher Knowledge Distillation
#
# Strategy:
#   GPU stream  : distill-b0-v1 (uses existing round5_pseudo, 1176 clips)
#   CPU (bg)    : extract Perch predictions for all 10658 soundscape files
#   After both  : distill-b0-v2-full (uses 256K clips from full extraction)
#
# Usage:
#   bash scripts/run_distill.sh [GPU]
#   bash scripts/run_distill.sh 0
# ============================================================

set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase

GPU=${1:-0}
LOG="outputs/run_distill.log"
mkdir -p outputs

log() { echo "[$(date '+%H:%M:%S')][DISTILL] $*" | tee -a "$LOG"; }

log "============================================================"
log " BirdCLEF 2026 Perch-as-Teacher Distillation  GPU=$GPU"
log " v1: round5_pseudo (1176 clips, fast start)"
log " v2: all 10658 soundscapes (after CPU extraction)"
log "============================================================"

# Step 1: Start CPU extraction of all 10658 soundscape files (background)
log ""
log "=== Starting Perch teacher extraction (CPU background) ==="
if [ -f "outputs/perch_teacher_all_ss.csv" ]; then
    existing=$(wc -l < "outputs/perch_teacher_all_ss.csv")
    log "Existing teacher CSV: $existing rows — resuming from checkpoint"
fi
nohup python3 scripts/extract_perch_teacher_all_ss.py \
    --output outputs/perch_teacher_all_ss.csv \
    2>&1 | tee -a "$LOG" &
EXTRACT_PID=$!
log "Teacher extraction launched (CPU PID=$EXTRACT_PID)"

# Step 2: Train distill-b0-v1 on GPU immediately
log ""
log "=== Training distill-b0-v1 (existing round5_pseudo, GPU=$GPU) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_distill.py \
    --config configs/distill_b0_v1.yaml \
    2>&1 | tee -a "$LOG"
log "distill-b0-v1 training done"

# Model soup for v1
if ls "checkpoints/distill-b0-v1/soup_ep"*.pt 1>/dev/null 2>&1; then
    log "Model Soup -> distill-b0-v1"
    python3 scripts/model_soup.py --run "distill-b0-v1" \
        --config "configs/distill_b0_v1.yaml" \
        2>&1 | tee -a "$LOG"
fi

# Holdout eval for v1
log "Holdout eval -> distill-b0-v1"
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout.py \
    --checkpoint "checkpoints/distill-b0-v1/best_sed.pt" \
    --config "configs/distill_b0_v1.yaml" --run_name "distill-b0-v1" \
    2>&1 | tee -a "$LOG"

python3 -c "
import json, os
for p in ['outputs/distill-b0-v1/sed_holdout_eval.json']:
    if os.path.exists(p):
        d = json.load(open(p))
        auc = d.get('holdout_auc','N/A')
        flag = ' *** BEATS v5 ***' if isinstance(auc, float) and auc > 0.9192 else ''
        print(f'  >>> distill-b0-v1  holdout_auc={auc}  (v5 baseline 0.9192){flag}')
" 2>&1 | tee -a "$LOG"

# TTA for v1
log "TTA eval -> distill-b0-v1"
for mode in hop shift both; do
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout_tta.py \
        --checkpoint "checkpoints/distill-b0-v1/best_sed.pt" \
        --config "configs/distill_b0_v1.yaml" --run_name "distill-b0-v1" \
        --tta_mode "$mode" \
        2>&1 | tee -a "$LOG"
done

# Step 3: Wait for Perch extraction to finish, then train v2
log ""
log "=== Waiting for teacher extraction (PID=$EXTRACT_PID) ==="
while kill -0 $EXTRACT_PID 2>/dev/null; do sleep 120; done
log "Teacher extraction done."

if [ -f "outputs/perch_teacher_all_ss.csv" ]; then
    n_rows=$(wc -l < "outputs/perch_teacher_all_ss.csv")
    log "Teacher CSV: $n_rows rows (segments)"
    log ""
    log "=== Training distill-b0-v2-full (all 10658 soundscapes, GPU=$GPU) ==="
    CUDA_VISIBLE_DEVICES=$GPU python3 train_distill.py \
        --config configs/distill_b0_v2_full.yaml \
        2>&1 | tee -a "$LOG"
    log "distill-b0-v2-full training done"

    # Soup + holdout for v2
    if ls "checkpoints/distill-b0-v2-full/soup_ep"*.pt 1>/dev/null 2>&1; then
        log "Model Soup -> distill-b0-v2-full"
        python3 scripts/model_soup.py --run "distill-b0-v2-full" \
            --config "configs/distill_b0_v2_full.yaml" \
            2>&1 | tee -a "$LOG"
    fi

    log "Holdout eval -> distill-b0-v2-full"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout.py \
        --checkpoint "checkpoints/distill-b0-v2-full/best_sed.pt" \
        --config "configs/distill_b0_v2_full.yaml" --run_name "distill-b0-v2-full" \
        2>&1 | tee -a "$LOG"

    python3 -c "
import json, os
for name in ['distill-b0-v1', 'distill-b0-v2-full']:
    p = f'outputs/{name}/sed_holdout_eval.json'
    if os.path.exists(p):
        d = json.load(open(p))
        auc = d.get('holdout_auc','N/A')
        flag = ' *** BEATS v5 ***' if isinstance(auc, float) and auc > 0.9192 else ''
        print(f'  >>> {name}  holdout_auc={auc}{flag}')
" 2>&1 | tee -a "$LOG"

    for mode in hop shift both; do
        CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout_tta.py \
            --checkpoint "checkpoints/distill-b0-v2-full/best_sed.pt" \
            --config "configs/distill_b0_v2_full.yaml" --run_name "distill-b0-v2-full" \
            --tta_mode "$mode" \
            2>&1 | tee -a "$LOG"
    done
fi

# Final summary
log ""
log "============================================================"
log " Distillation Complete — Holdout AUC"
python3 - 2>&1 | tee -a "$LOG" << 'PYEOF'
import json, glob
results = []
for p in sorted(glob.glob('outputs/distill*/sed_holdout_eval.json')):
    try:
        d = json.load(open(p))
        auc = d.get('holdout_auc')
        name = d.get('run_name', p)
        tta = d.get('tta_mode','none')
        if auc: results.append((float(auc), name, tta))
    except: pass
results.sort(reverse=True)
print(f"  {'Run':<50} {'AUC':>8}  TTA")
for auc, name, tta in results[:10]:
    flag = ' *** BEATS v5 ***' if auc > 0.9192 else ''
    print(f'  {name:<50} {auc:.4f}   {tta}{flag}')
# Compare v1 vs v2
v1 = next((a for a,n,t in results if 'v1' in n and t=='none'), None)
v2 = next((a for a,n,t in results if 'v2-full' in n and t=='none'), None)
if v1 and v2: print(f'  v2-full vs v1: {v2-v1:+.4f} (more unlabeled data effect)')
PYEOF
log "============================================================"
