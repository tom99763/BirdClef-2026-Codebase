#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Phases 1, 4, 5 Orchestrator
#
# Training (Phase 2+3) is already running externally.
# This script handles everything else:
#
#   Phase 1 : Inference enhancements eval (runs immediately on sed-b0-v5)
#   Wait    : Poll until BOTH sed-b0-v6 AND sed-b2-v1 finish
#   Holdout : Standalone holdout eval for best checkpoints
#   Phase 4 : Model Soup (weight averaging)
#   Soup eval: Holdout eval for soup checkpoints
#   Phase 5 : Optuna ensemble weight optimization
#
# Usage:
#   bash scripts/run_phases_1_4_5.sh [GPU1] [GPU0]
#   bash scripts/run_phases_1_4_5.sh 1 0      # default
# ============================================================

set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase

GPU1=${1:-1}
GPU0=${2:-0}
LOG="outputs/phases_1_4_5.log"
mkdir -p outputs

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Guard: abort if another instance of this script is already running
LOCK="/tmp/birdclef_phases_1_4_5.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
    echo "ABORT: another instance already running (PID=$(cat $LOCK)). Exiting."
    exit 1
fi
echo $$ > "$LOCK"
trap "rm -f $LOCK" EXIT

log "=========================================="
log " Phases 1+4+5 Orchestrator  PID=$$"
log " GPU1=$GPU1 (phases 1,4,5)  GPU0=$GPU0 (holdout eval)"
log "=========================================="

# ── Phase 1: Inference enhancements (runs NOW on sed-b0-v5) ───────────────────
log ""
log "=========================================="
log "Phase 1: Inference Enhancements (sed-b0-v5)"
log "=========================================="

python3 scripts/phase1_inference_eval.py \
    --checkpoint checkpoints/sed-b0-v5/best_sed.pt \
    --gpu "$GPU1" \
    2>&1 | tee -a "$LOG"

if [ -f "outputs/phase1_inference_eval.json" ]; then
    python3 -c "
import json
d = json.load(open('outputs/phase1_inference_eval.json'))
print(f'  Best variant : {d[\"best_variant\"]}')
print(f'  Best AUC     : {d[\"best_auc\"]:.4f}')
print(f'  Gain vs base : {d[\"best_gain\"]:+.4f}')
" 2>&1 | tee -a "$LOG"
fi

# ── Wait for sed-b0-v6 to finish ──────────────────────────────────────────────
log ""
log "Waiting for sed-b0-v6 (GPU$GPU1) to finish …"
while true; do
    finished=$(python3 -c "
import json, os
p = 'outputs/sed-b0-v6/result.json'
if not os.path.isfile(p): print('False'); exit()
d = json.load(open(p))
print(d.get('finished', False))
" 2>/dev/null || echo "False")
    if [ "$finished" = "True" ]; then
        log "  sed-b0-v6 finished!"
        break
    fi
    ep=$(python3 -c "import json,os; p='outputs/sed-b0-v6/result.json'; d=json.load(open(p)) if os.path.isfile(p) else {}; print(d.get('total_epochs_run',0))" 2>/dev/null || echo "?")
    log "  sed-b0-v6: ep=$ep — still training …"
    sleep 120
done

python3 -c "
import json
d = json.load(open('outputs/sed-b0-v6/result.json'))
h = d['epoch_history']
print(f'  Best SS val AUC : {d[\"best_val_roc_auc\"]:.4f} @ep{d[\"best_epoch\"]}')
print(f'  Total epochs    : {d[\"total_epochs_run\"]}')
print('  Last 5 epochs:')
for e in h[-5:]:
    print(f'    ep{e[\"epoch\"]:3d}  val={e[\"val_roc_auc\"]:.4f}  loss={e[\"train_loss\"]:.5f}')
" 2>&1 | tee -a "$LOG"

cp checkpoints/sed-b0-v6/best_sed.pt submissions/weights/best_sed_b0_v6.pt
log "  Copied → submissions/weights/best_sed_b0_v6.pt"

log ""
log "=== sed-b0-v6 Standalone Holdout Eval ==="
python3 scripts/eval_sed_holdout.py \
    --checkpoint checkpoints/sed-b0-v6/best_sed.pt \
    --config configs/sed_b0_v6.yaml \
    --run_name sed-b0-v6 \
    --gpu "$GPU1" \
    2>&1 | tee -a "$LOG"

# ── Wait for sed-b2-v1 to finish ──────────────────────────────────────────────
log ""
log "Waiting for sed-b2-v1 (GPU$GPU0) to finish …"
while true; do
    finished=$(python3 -c "
import json, os
p = 'outputs/sed-b2-v1/result.json'
if not os.path.isfile(p): print('False'); exit()
d = json.load(open(p))
print(d.get('finished', False))
" 2>/dev/null || echo "False")
    if [ "$finished" = "True" ]; then
        log "  sed-b2-v1 finished!"
        break
    fi
    ep=$(python3 -c "import json,os; p='outputs/sed-b2-v1/result.json'; d=json.load(open(p)) if os.path.isfile(p) else {}; print(d.get('total_epochs_run',0))" 2>/dev/null || echo "?")
    log "  sed-b2-v1: ep=$ep — still training …"
    sleep 120
done

python3 -c "
import json
d = json.load(open('outputs/sed-b2-v1/result.json'))
h = d['epoch_history']
print(f'  Best SS val AUC : {d[\"best_val_roc_auc\"]:.4f} @ep{d[\"best_epoch\"]}')
print(f'  Total epochs    : {d[\"total_epochs_run\"]}')
print('  Last 5 epochs:')
for e in h[-5:]:
    print(f'    ep{e[\"epoch\"]:3d}  val={e[\"val_roc_auc\"]:.4f}  loss={e[\"train_loss\"]:.5f}')
" 2>&1 | tee -a "$LOG"

cp checkpoints/sed-b2-v1/best_sed.pt submissions/weights/best_sed_b2_v1.pt 2>/dev/null || true
log "  Copied → submissions/weights/best_sed_b2_v1.pt"

log ""
log "=== sed-b2-v1 Standalone Holdout Eval ==="
python3 scripts/eval_sed_holdout.py \
    --checkpoint checkpoints/sed-b2-v1/best_sed.pt \
    --config configs/sed_b2_v1.yaml \
    --run_name sed-b2-v1 \
    --gpu "$GPU1" \
    2>&1 | tee -a "$LOG"

# ── Phase 4: Model Soup ────────────────────────────────────────────────────────
log ""
log "=========================================="
log "Phase 4: Model Soup (checkpoint weight averaging)"
log "=========================================="

for run_cfg in "sed-b0-v6 configs/sed_b0_v6.yaml" "sed-b2-v1 configs/sed_b2_v1.yaml"; do
    run_name=$(echo $run_cfg | cut -d' ' -f1)
    run_config=$(echo $run_cfg | cut -d' ' -f2)
    if ls checkpoints/$run_name/soup_ep*.pt 1>/dev/null 2>&1; then
        log "  Running Model Soup for $run_name …"
        python3 scripts/model_soup.py \
            --run "$run_name" \
            --config "$run_config" \
            2>&1 | tee -a "$LOG"
        if [ -f "checkpoints/$run_name/soup_sed.pt" ]; then
            cp "checkpoints/$run_name/soup_sed.pt" "submissions/weights/soup_${run_name}.pt"
            log "  Copied → submissions/weights/soup_${run_name}.pt"
        fi
    else
        log "  No soup_ep*.pt for $run_name — skipping soup"
    fi
done

# Soup holdout evals
log ""
log "=== Soup Holdout Evals ==="
for run_cfg in "sed-b0-v6 configs/sed_b0_v6.yaml" "sed-b2-v1 configs/sed_b2_v1.yaml"; do
    run_name=$(echo $run_cfg | cut -d' ' -f1)
    run_config=$(echo $run_cfg | cut -d' ' -f2)
    if [ -f "checkpoints/$run_name/soup_sed.pt" ]; then
        python3 scripts/eval_sed_holdout.py \
            --checkpoint "checkpoints/$run_name/soup_sed.pt" \
            --config "$run_config" \
            --run_name "${run_name}-soup" \
            --gpu "$GPU1" \
            2>&1 | tee -a "$LOG"
    fi
done

# ── Phase 5: Optuna ensemble weight optimization ───────────────────────────────
log ""
log "=========================================="
log "Phase 5: Optuna Ensemble Weight Optimization"
log "  Trials: 200  GPU: $GPU1"
log "=========================================="

# Prefer soup over best for each SED model
SED_CKPTS=""
SED_CFGS=""
declare -A MODEL_CFGS=(
    ["sed-b0-v5"]="configs/sed_b0_v5.yaml"
    ["sed-b0-v6"]="configs/sed_b0_v6.yaml"
    ["sed-b2-v1"]="configs/sed_b2_v1.yaml"
)
for model in "sed-b0-v5" "sed-b0-v6" "sed-b2-v1"; do
    cfg="${MODEL_CFGS[$model]}"
    if [ -f "checkpoints/$model/soup_sed.pt" ]; then
        ckpt="checkpoints/$model/soup_sed.pt"
        log "  $model: soup"
    elif [ -f "checkpoints/$model/best_sed.pt" ]; then
        ckpt="checkpoints/$model/best_sed.pt"
        log "  $model: best"
    else
        log "  $model: not found — skip"
        continue
    fi
    SED_CKPTS="$SED_CKPTS $ckpt"
    SED_CFGS="$SED_CFGS $cfg"
done

python3 scripts/optimize_ensemble.py \
    --gpu "$GPU1" \
    --sed_checkpoints $SED_CKPTS \
    --sed_configs $SED_CFGS \
    --n_trials 200 \
    --output outputs/ensemble_weights_optuna.json \
    2>&1 | tee -a "$LOG" || log "  (Phase 5 failed — check optuna / holdout data)"

if [ -f "outputs/ensemble_weights_optuna.json" ]; then
    python3 -c "
import json
d = json.load(open('outputs/ensemble_weights_optuna.json'))
print(f'  Best ensemble AUC : {d[\"best_auc\"]:.4f}')
print(f'  Equal-weight AUC  : {d[\"equal_w_auc\"]:.4f}')
print(f'  Gain              : {d[\"gain\"]:+.4f}')
print('  Optimal weights:')
for name, w in d['best_weights'].items():
    print(f'    {name:<42} {w:.4f}')
" 2>&1 | tee -a "$LOG"
fi

# ── Hand off to Phases 6+ ──────────────────────────────────────────────────────
log ""
log "=========================================="
log "Handing off to Phases 6+ (V2-S, 10s eval, pseudo-label retrain)"
log "=========================================="

bash scripts/run_phases_6_plus.sh "$GPU0" "$GPU1" 2>&1 | tee -a "$LOG" \
    || log "  (Phases 6+ failed — check $LOG)"

# ── Done ───────────────────────────────────────────────────────────────────────
log ""
log "=========================================="
log " All phases complete!"
log "  Phase 1 : inference enhancements → outputs/phase1_inference_eval.json"
log "  Phase 4 : Model Soup → submissions/weights/soup_*.pt"
log "  Phase 5 : Optuna weights → outputs/ensemble_weights_optuna.json"
log "  Phase 6 : V2-S training → checkpoints/sed-v2s-v1/"
log "  Phase 7 : 10s eval → outputs/*-10s/"
log "  Phase 8 : Pseudo retrain → checkpoints/sed-b0-v8-pseudo/"
log "  Log     : $LOG"
log "=========================================="
