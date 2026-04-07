#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Geographic Filtering Experiments
#
# Applies post-processing geographic masks to existing SED
# checkpoints (no retraining needed).
#
# Three mask modes:
#   ss_hard    : soundscape species -> 1.0, others -> 0.0
#   ss_soft    : soundscape species -> 1.0, others -> 0.1
#   sa_weighted: pred *= max(in_soundscape, sa_fraction)
#
# Usage:
#   bash scripts/run_geo.sh [GPU]
# ============================================================

set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase

GPU=${1:-0}
LOG="outputs/run_geo.log"
mkdir -p outputs

log() { echo "[$(date '+%H:%M:%S')][GEO] $*" | tee -a "$LOG"; }

log "============================================================"
log " BirdCLEF 2026 Geo-Filtering  GPU=$GPU"
log " Modes: ss_hard / ss_soft(0.1) / sa_weighted"
log "============================================================"

# Build geo mask
log "Building geographic species mask ..."
python3 scripts/build_geo_mask.py 2>&1 | tee -a "$LOG"
log "Geo mask built -> outputs/geo_mask.csv"

# Helper: eval all mask modes for one checkpoint
eval_geo() {
    local name=$1 ckpt=$2 cfg=$3
    log ""
    log "--- Geo eval: $name ---"

    if [ ! -f "outputs/$name/sed_holdout_eval.json" ]; then
        log "  baseline (no mask) ..."
        CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout.py \
            --checkpoint "$ckpt" --config "$cfg" --run_name "$name" \
            2>&1 | tee -a "$LOG"
    else
        log "  baseline already exists, skipping"
    fi

    log "  ss_hard (soundscape-only) ..."
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_geo_holdout.py \
        --checkpoint "$ckpt" --config "$cfg" --run_name "$name" \
        --mask_mode ss_hard \
        2>&1 | tee -a "$LOG"

    log "  ss_soft (factor=0.1) ..."
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_geo_holdout.py \
        --checkpoint "$ckpt" --config "$cfg" --run_name "$name" \
        --mask_mode ss_soft --soft_factor 0.1 \
        2>&1 | tee -a "$LOG"

    log "  sa_weighted (continuous SA fraction) ..."
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_geo_holdout.py \
        --checkpoint "$ckpt" --config "$cfg" --run_name "$name" \
        --mask_mode sa_weighted \
        2>&1 | tee -a "$LOG"
}

# Eval existing best checkpoints
log ""
log "=== Existing checkpoints ==="

if [ -f "checkpoints/sed-b0-v5/best_sed.pt" ]; then
    eval_geo "sed-b0-v5" "checkpoints/sed-b0-v5/best_sed.pt" "configs/sed_b0_v5.yaml"
fi

if [ -f "checkpoints/sed-b0-v9-asl/best_sed.pt" ]; then
    eval_geo "sed-b0-v9-asl" "checkpoints/sed-b0-v9-asl/best_sed.pt" "configs/sed_b0_v9_asl.yaml"
fi

# Eval SED_P checkpoints
log ""
log "=== SED_P checkpoints ==="

for run_name in sedp-b0-v1 sedp-b0-v2-fusion sedp-b0-v3-abl-no-pcen; do
    cfg_map=""
    case "$run_name" in
        sedp-b0-v1)             cfg_map="configs/sedp_b0_v1.yaml" ;;
        sedp-b0-v2-fusion)      cfg_map="configs/sedp_b0_v2_fusion.yaml" ;;
        sedp-b0-v3-abl-no-pcen) cfg_map="configs/sedp_b0_v3_abl_no_pcen.yaml" ;;
    esac

    best_ckpt="checkpoints/$run_name/best_sed.pt"
    soup_ckpt="checkpoints/$run_name/soup_sed.pt"

    if [ -f "$best_ckpt" ]; then
        eval_geo "$run_name" "$best_ckpt" "$cfg_map"
    fi
    if [ -f "$soup_ckpt" ]; then
        eval_geo "${run_name}-soup" "$soup_ckpt" "$cfg_map"
    fi
done

# Final comparison table
log ""
log "============================================================"
log " Geo-Filtering Results — Holdout AUC Comparison"
python3 - 2>&1 | tee -a "$LOG" << 'PYEOF'
import json, glob, os
results = []
for p in sorted(glob.glob('outputs/*/sed_holdout_eval.json')):
    try:
        d = json.load(open(p))
        auc = d.get('holdout_auc')
        name = d.get('run_name', p)
        mode = d.get('mask_mode', 'none')
        if auc: results.append((float(auc), name, mode))
    except: pass
results.sort(key=lambda x: x[0], reverse=True)
print(f"\n  {'Run':<58} {'AUC':>8}  Mask")
print(f"  {'-'*80}")
for auc, name, mode in results[:30]:
    flag = ' *** BEATS v5 ***' if auc > 0.9192 else ''
    print(f'  {name:<58} {auc:.4f}   {mode}{flag}')
baseline = {d.get('base_run', d.get('run_name','')): d.get('holdout_auc')
            for p in glob.glob('outputs/*/sed_holdout_eval.json')
            for d in [json.load(open(p))]
            if d.get('mask_mode','none') == 'none' and d.get('holdout_auc')}
print(f"\n  Geo mask delta vs baseline (holdout AUC):")
for p in sorted(glob.glob('outputs/*/sed_holdout_eval.json')):
    try:
        d = json.load(open(p))
        if d.get('mask_mode','none') == 'none': continue
        base = d.get('base_run','')
        base_auc = baseline.get(base)
        if base_auc and d.get('holdout_auc'):
            delta = d['holdout_auc'] - base_auc
            sign = '+' if delta >= 0 else ''
            print(f"    {d['run_name']:<60} {sign}{delta:+.4f}")
    except: pass
PYEOF
log "============================================================"
