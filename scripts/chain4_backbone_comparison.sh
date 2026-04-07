#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Chain 4: Backbone Comparison (GPU0)
#
# Pipeline:
#   Wait v15 → distill B2 → SED-head B2
#              → distill B4 → SED-head B4
#              → distill ConvNeXt-Tiny → SED-head ConvNeXt
#
# Each backbone: embed-distill (frozen backbone training) →
#   freeze backbone → train SED head only →
#   holdout eval → model soup → soup holdout eval
#
# Usage:
#   bash scripts/chain4_backbone_comparison.sh 2>&1 | tee outputs/chain4.log
# ============================================================

set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase

GPU=0
LOG="outputs/chain4.log"
mkdir -p outputs submissions/weights

log() { echo "[$(date '+%H:%M:%S')][CHAIN4] $*" | tee -a "$LOG"; }

log "============================================================"
log " BirdCLEF 2026 Chain 4 — Backbone Comparison  PID=$$"
log " GPU=$GPU"
log " Pipeline: v15 → distill(B2/B4/ConvNeXt) → SED-head each"
log "============================================================"

# ── Wait for v15 to finish ──────────────────────────────────────────────────
log "Waiting for sed-b0-v15-no-sec to finish..."
while true; do
    if [ -f "outputs/sed-b0-v15-no-sec/result.json" ]; then
        finished=$(python3 -c "import json; d=json.load(open('outputs/sed-b0-v15-no-sec/result.json')); print(d.get('finished',False))" 2>/dev/null || echo "False")
        if [ "$finished" = "True" ]; then
            log "v15 finished!"
            break
        fi
        ep=$(python3 -c "import json; d=json.load(open('outputs/sed-b0-v15-no-sec/result.json')); h=d.get('epoch_history',[]); print(h[-1]['epoch'] if h else '?')" 2>/dev/null || echo "?")
        auc=$(python3 -c "import json; d=json.load(open('outputs/sed-b0-v15-no-sec/result.json')); print(f\"{d.get('best_val_roc_auc',0):.4f}\")" 2>/dev/null || echo "?")
        log "  v15 still running: ep=$ep best_val=$auc"
    else
        log "  v15 not started yet"
    fi
    sleep 120
done

# ── v15 soup + holdout ──────────────────────────────────────────────────────
log "Running v15 soup + holdout eval..."
if ls "checkpoints/sed-b0-v15-no-sec/soup_ep"*.pt 1>/dev/null 2>&1; then
    python3 scripts/model_soup.py --run "sed-b0-v15-no-sec" --config "configs/sed_b0_v15_no_sec.yaml" 2>&1 | tee -a "$LOG"
fi
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout.py \
    --checkpoint "checkpoints/sed-b0-v15-no-sec/best_sed.pt" \
    --config "configs/sed_b0_v15_no_sec.yaml" \
    --run_name "sed-b0-v15-no-sec" 2>&1 | tee -a "$LOG"
if [ -f "checkpoints/sed-b0-v15-no-sec/soup_sed.pt" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout.py \
        --checkpoint "checkpoints/sed-b0-v15-no-sec/soup_sed.pt" \
        --config "configs/sed_b0_v15_no_sec.yaml" \
        --run_name "sed-b0-v15-no-sec-soup" 2>&1 | tee -a "$LOG"
fi

# ── Helper: distill → SED head → holdout → soup ────────────────────────────
distill_and_train() {
    local distill_cfg=$1
    local distill_run=$2
    local sed_cfg=$3
    local sed_run=$4

    log "=============================="
    log "Embed-distill: $distill_run"
    log "=============================="
    CUDA_VISIBLE_DEVICES=$GPU python3 train_embed_distill.py \
        --config "$distill_cfg" --gpu $GPU 2>&1 | tee -a "$LOG"

    BACKBONE="checkpoints/$distill_run/best_backbone.pt"
    if [ ! -f "$BACKBONE" ]; then
        log "ERROR: backbone not found: $BACKBONE — skipping $sed_run"
        return 1
    fi
    val_cos=$(python3 -c "import json; d=json.load(open('outputs/$distill_run/result.json')); print(f\"{d.get('best_val_cos',0):.4f}\")" 2>/dev/null || echo "?")
    log "  $distill_run complete: best_val_cos=$val_cos"

    log "=============================="
    log "SED head-only: $sed_run"
    log "=============================="
    CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py \
        --config "$sed_cfg" \
        --gpu $GPU \
        --pretrained_backbone "$BACKBONE" \
        2>&1 | tee -a "$LOG"
    log "$sed_run training done"

    # Holdout eval
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout.py \
        --checkpoint "checkpoints/$sed_run/best_sed.pt" \
        --config "$sed_cfg" --run_name "$sed_run" 2>&1 | tee -a "$LOG"

    # Soup
    if ls "checkpoints/$sed_run/soup_ep"*.pt 1>/dev/null 2>&1; then
        python3 scripts/model_soup.py --run "$sed_run" --config "$sed_cfg" 2>&1 | tee -a "$LOG"
        if [ -f "checkpoints/$sed_run/soup_sed.pt" ]; then
            CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout.py \
                --checkpoint "checkpoints/$sed_run/soup_sed.pt" \
                --config "$sed_cfg" --run_name "${sed_run}-soup" 2>&1 | tee -a "$LOG"
        fi
    fi

    python3 -c "
import json, os
for name in ['$sed_run', '${sed_run}-soup']:
    p = f'outputs/{name}/sed_holdout_eval.json'
    if os.path.exists(p):
        d = json.load(open(p))
        auc = d.get('holdout_auc','N/A')
        flag = ' *** BEATS 0.9532! ***' if isinstance(auc,float) and auc > 0.9532 else ''
        print(f'  >>> {name}  holdout_auc={auc}{flag}')
" 2>&1 | tee -a "$LOG"
}

# ── B2 ──────────────────────────────────────────────────────────────────────
distill_and_train \
    "configs/embed_distill_b2_v1.yaml"  "embed-distill-b2-v1" \
    "configs/sed_b2_v1_distill_head.yaml" "sed-b2-v1-distill-head"

# ── B4 ──────────────────────────────────────────────────────────────────────
distill_and_train \
    "configs/embed_distill_b4_v1.yaml"  "embed-distill-b4-v1" \
    "configs/sed_b4_v1_distill_head.yaml" "sed-b4-v1-distill-head"

# ── ConvNeXt-Tiny ────────────────────────────────────────────────────────────
distill_and_train \
    "configs/embed_distill_convnext_v1.yaml"  "embed-distill-convnext-v1" \
    "configs/sed_convnext_v1_distill_head.yaml" "sed-convnext-v1-distill-head"

# ── ConvNeXt-Small ───────────────────────────────────────────────────────────
distill_and_train \
    "configs/embed_distill_convnext_small_v1.yaml"  "embed-distill-convnext-small-v1" \
    "configs/sed_convnext_small_v1_distill_head.yaml" "sed-convnext-small-v1-distill-head"

# ── Final ranking ─────────────────────────────────────────────────────────
log ""
log "============================================================"
log " CHAIN 4 COMPLETE — Backbone Comparison Holdout AUC"
python3 -c "
import json, glob
results = []
for p in sorted(glob.glob('outputs/*/sed_holdout_eval.json')):
    try:
        d = json.load(open(p))
        name = d.get('run_name', p.split('/')[1])
        auc  = d.get('holdout_auc')
        if auc: results.append((float(auc), name))
    except: pass
print(f'  {\"Model\":<55} Holdout AUC  Beat 0.9532?')
print(f'  {\"-\"*75}')
for auc, name in sorted(results, reverse=True):
    flag = ' *** BEATS 0.9532 ***' if auc > 0.9532 else ''
    print(f'  {name:<55} {auc:.4f}{flag}')
" 2>&1 | tee -a "$LOG"
log "============================================================"
