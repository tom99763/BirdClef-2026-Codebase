#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Full SED Improvement Plan Orchestrator
#
# Runs all phases automatically after sed-b0-v5 completes:
#
#   Phase 0: Wait for sed-b0-v5 + resume + ensemble v3 eval
#   Phase 1: Inference enhancements (TTA + TopN + Smoothing)
#   Phase 2: Train sed-b0-v6  (CE + freq_masking) on GPU1
#   Phase 3: Train sed-b2-v1  (B2 backbone)       on GPU0  [parallel with Phase 2]
#   Phase 4: SED pseudo-labeling + train sed-b0-v6-pseudo
#   Phase 5: Optuna ensemble weight optimization
#   Report : HTML update after each phase
#
# Usage:
#   bash scripts/run_improvement_plan.sh [GPU1] [GPU0]
#   bash scripts/run_improvement_plan.sh 1 0          # default
# ============================================================

set -e
cd /home/lab/BirdClef-2026-Codebase

GPU1=${1:-1}   # GPU for sequential phases (Phase 1, 2, 4, 5)
GPU0=${2:-0}   # GPU for Phase 3 (parallel with Phase 2)

LOG="outputs/improvement_plan.log"
SED_JSON="outputs/sed-b0-v5/result.json"
ENSEMBLE_V3_LOG="outputs/ensemble_v3_holdout_eval.log"

mkdir -p outputs pseudo_labels submissions/weights

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=========================================="
log " SED Full Improvement Plan Orchestrator"
log " GPU1=$GPU1 (phases 1,2,4,5)  GPU0=$GPU0 (phase 3)"
log "=========================================="

# ── Phase 0: Wait for ensemble_v3_holdout_eval.log ────────────────────────────
log ""
log "Phase 0: Waiting for sed-b0-v5 + resume + ensemble v3 eval …"
log "  (polling $ENSEMBLE_V3_LOG every 5 min)"

while true; do
    if [ -f "$ENSEMBLE_V3_LOG" ]; then
        log "  ensemble_v3_holdout_eval.log found — all prior work complete!"
        break
    fi
    if [ -f "$SED_JSON" ]; then
        finished=$(python3 -c "import json; d=json.load(open('$SED_JSON')); print(d.get('finished', False))" 2>/dev/null || echo "False")
        total_ep=$(python3 -c "import json; d=json.load(open('$SED_JSON')); print(d.get('total_epochs_run', 0))" 2>/dev/null || echo "0")
        log "  sed-b0-v5: finished=$finished  ep=$total_ep — waiting …"
    fi
    sleep 300
done

log ""
log "=== sed-b0-v5 Final Result ==="
python3 -c "
import json
d = json.load(open('$SED_JSON'))
h = d['epoch_history']
print(f'  Best SS val AUC : {d[\"best_val_roc_auc\"]:.4f} @ep{d[\"best_epoch\"]}')
print(f'  Total epochs    : {d[\"total_epochs_run\"]}')
print('  Last 5 epochs:')
for e in h[-5:]:
    print(f'    ep{e[\"epoch\"]:3d}  val={e[\"val_roc_auc\"]:.4f}  loss={e[\"train_loss\"]:.5f}')
" 2>&1 | tee -a "$LOG"

python3 scripts/update_plan_report.py 2>&1 | tee -a "$LOG"

# ── Phase 1: Inference enhancements ───────────────────────────────────────────
log ""
log "=========================================="
log "Phase 1: Inference Enhancements"
log "  Variants: TTA + TopN + Smoothing + combinations"
log "  Checkpoint: checkpoints/sed-b0-v5/best_sed.pt  GPU=$GPU1"
log "=========================================="

python3 scripts/phase1_inference_eval.py \
    --checkpoint checkpoints/sed-b0-v5/best_sed.pt \
    --gpu "$GPU1" \
    2>&1 | tee -a "$LOG"

log ""
log "Phase 1 complete. Updating HTML …"
python3 scripts/update_plan_report.py 2>&1 | tee -a "$LOG"

python3 -c "
import json
d = json.load(open('outputs/phase1_inference_eval.json'))
print(f'  Best variant : {d[\"best_variant\"]}')
print(f'  Best AUC     : {d[\"best_auc\"]:.4f}')
print(f'  Gain vs base : {d[\"best_gain\"]:+.4f}')
" 2>&1 | tee -a "$LOG"

# ── Phase 2 + 3: Train in parallel ────────────────────────────────────────────
log ""
log "=========================================="
log "Phase 2: Train sed-b0-v6 (CE + freq_masking, GPU$GPU1)"
log "Phase 3: Train sed-b2-v1 (B2 backbone,       GPU$GPU0) [parallel]"
log "=========================================="

# Phase 3: sed-b2-v1 on GPU0 — start in background
log "  Starting sed-b2-v1 on GPU$GPU0 in background …"
(
    python3 train_sed.py \
        --config configs/sed_b2_v1.yaml \
        --gpu "$GPU0" \
        2>&1 | tee outputs/sed-b2-v1.log

    # Convergence check
    STILL_RISING_B2=$(python3 -c "
import json, os
path = 'outputs/sed-b2-v1/result.json'
if not os.path.isfile(path):
    print('no'); exit()
d = json.load(open(path))
h = d['epoch_history']
best_ep  = d['best_epoch']
total_ep = d['total_epochs_run']
if len(h) < 3:
    print('no')
else:
    last3 = [e['val_roc_auc'] for e in h[-3:]]
    print('yes' if (last3[-1] > last3[0] or best_ep >= total_ep - 2) else 'no')
" 2>/dev/null || echo "no")

    if [ "$STILL_RISING_B2" = "yes" ]; then
        echo "[$(date '+%H:%M:%S')] sed-b2-v1 still converging — resuming +20ep on GPU$GPU0" | tee -a "$LOG"
        python3 train_sed.py \
            --config configs/sed_b2_v1.yaml \
            --resume checkpoints/sed-b2-v1/best_sed.pt \
            --extra_epochs 20 \
            --gpu "$GPU0" \
            2>&1 | tee -a outputs/sed-b2-v1.log
    fi

    cp checkpoints/sed-b2-v1/best_sed.pt submissions/weights/best_sed_b2_v1.pt 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] sed-b2-v1 done" | tee -a "$LOG"
) &
PHASE3_PID=$!
log "  sed-b2-v1 background PID=$PHASE3_PID"

# Phase 2: sed-b0-v6 on GPU1 — run in foreground
log "  Training sed-b0-v6 on GPU$GPU1 …"
python3 train_sed.py \
    --config configs/sed_b0_v6.yaml \
    --gpu "$GPU1" \
    2>&1 | tee outputs/sed-b0-v6.log | tee -a "$LOG"

log ""
log "sed-b0-v6 training complete. Checking convergence …"

STILL_RISING_V6=$(python3 -c "
import json, os
path = 'outputs/sed-b0-v6/result.json'
if not os.path.isfile(path):
    print('no'); exit()
d = json.load(open(path))
h = d['epoch_history']
best_ep  = d['best_epoch']
total_ep = d['total_epochs_run']
if len(h) < 3:
    print('no')
else:
    last3 = [e['val_roc_auc'] for e in h[-3:]]
    print('yes' if (last3[-1] > last3[0] or best_ep >= total_ep - 2) else 'no')
" 2>/dev/null || echo "no")

if [ "$STILL_RISING_V6" = "yes" ]; then
    log "  sed-b0-v6 still converging — resuming +20 epochs on GPU$GPU1"
    python3 train_sed.py \
        --config configs/sed_b0_v6.yaml \
        --resume checkpoints/sed-b0-v6/best_sed.pt \
        --extra_epochs 20 \
        --gpu "$GPU1" \
        2>&1 | tee -a outputs/sed-b0-v6.log | tee -a "$LOG"
else
    log "  sed-b0-v6 converged. No resume needed."
fi

cp checkpoints/sed-b0-v6/best_sed.pt submissions/weights/best_sed_b0_v6.pt
log "  Copied → submissions/weights/best_sed_b0_v6.pt"

log ""
log "=== sed-b0-v6 Final Result ==="
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

log ""
log "=== sed-b0-v6 Standalone Holdout Eval ==="
python3 scripts/eval_sed_holdout.py \
    --checkpoint checkpoints/sed-b0-v6/best_sed.pt \
    --config configs/sed_b0_v6.yaml \
    --run_name sed-b0-v6 \
    --gpu "$GPU1" \
    2>&1 | tee -a "$LOG"

log ""
log "Phase 2 complete. Updating HTML …"
python3 scripts/update_plan_report.py 2>&1 | tee -a "$LOG"

# Wait for Phase 3 to finish before Phase 4
log ""
log "Waiting for Phase 3 (sed-b2-v1, PID=$PHASE3_PID) to complete …"
wait $PHASE3_PID || log "  (sed-b2-v1 exited with non-zero — check outputs/sed-b2-v1.log)"

log ""
log "=== sed-b2-v1 Final Result ==="
python3 -c "
import json, os
path = 'outputs/sed-b2-v1/result.json'
if not os.path.isfile(path):
    print('  result.json not found')
else:
    d = json.load(open(path))
    h = d['epoch_history']
    print(f'  Best SS val AUC : {d[\"best_val_roc_auc\"]:.4f} @ep{d[\"best_epoch\"]}')
    print(f'  Total epochs    : {d[\"total_epochs_run\"]}')
    print('  Last 5 epochs:')
    for e in h[-5:]:
        print(f'    ep{e[\"epoch\"]:3d}  val={e[\"val_roc_auc\"]:.4f}  loss={e[\"train_loss\"]:.5f}')
" 2>&1 | tee -a "$LOG"

python3 scripts/update_plan_report.py 2>&1 | tee -a "$LOG"

log ""
log "=== sed-b2-v1 Standalone Holdout Eval ==="
python3 scripts/eval_sed_holdout.py \
    --checkpoint checkpoints/sed-b2-v1/best_sed.pt \
    --config configs/sed_b2_v1.yaml \
    --run_name sed-b2-v1 \
    --gpu "$GPU1" \
    2>&1 | tee -a "$LOG" || log "  (sed-b2-v1 holdout eval failed — checkpoint may not exist)"

# ── Phase 4: Model Soup (checkpoint weight averaging) ─────────────────────────
# BirdCLEF 2026 has real train_soundscapes_labels.csv (1478 labeled clips) —
# unlike 2025 which had no soundscape labels. Pseudo-labeling adds noise on top
# of already-real labels. Model Soup gives free improvement with no extra training.
log ""
log "=========================================="
log "Phase 4: Model Soup (checkpoint weight averaging)"
log "  Averages top-3 epoch checkpoints per SED model"
log "  BirdCLEF 2025 1st place: ~+0.003 AUC, zero extra training cost"
log "  Note: skipping pseudo-label retrain (2026 has real soundscape labels)"
log "=========================================="

for soup_run in "sed-b0-v6 configs/sed_b0_v6.yaml" "sed-b2-v1 configs/sed_b2_v1.yaml"; do
    run_name=$(echo $soup_run | cut -d' ' -f1)
    run_cfg=$(echo $soup_run | cut -d' ' -f2)
    if ls checkpoints/$run_name/soup_ep*.pt 1>/dev/null 2>&1; then
        log "  Running Model Soup for $run_name …"
        python3 scripts/model_soup.py \
            --run "$run_name" \
            --config "$run_cfg" \
            2>&1 | tee -a "$LOG"
        if [ -f "checkpoints/$run_name/soup_sed.pt" ]; then
            cp "checkpoints/$run_name/soup_sed.pt" "submissions/weights/soup_${run_name}.pt"
            log "  Copied → submissions/weights/soup_${run_name}.pt"
        fi
    else
        log "  No soup_ep*.pt found for $run_name — using best_sed.pt only"
    fi
done

log ""
log "=== Model Soup Holdout Evals ==="
for soup_run_cfg in "sed-b0-v6 configs/sed_b0_v6.yaml" "sed-b2-v1 configs/sed_b2_v1.yaml"; do
    soup_run=$(echo $soup_run_cfg | cut -d' ' -f1)
    soup_cfg=$(echo $soup_run_cfg | cut -d' ' -f2)
    if [ -f "checkpoints/$soup_run/soup_sed.pt" ]; then
        python3 scripts/eval_sed_holdout.py \
            --checkpoint "checkpoints/$soup_run/soup_sed.pt" \
            --config "$soup_cfg" \
            --run_name "${soup_run}-soup" \
            --gpu "$GPU1" \
            2>&1 | tee -a "$LOG"
    fi
done

log ""
log "Phase 4 complete. Updating HTML …"
python3 scripts/update_plan_report.py 2>&1 | tee -a "$LOG"

# ── Phase 5: Optuna ensemble weight optimization ──────────────────────────────
log ""
log "=========================================="
log "Phase 5: Optuna Ensemble Weight Optimization"
log "  Models: Perch×3 + SED checkpoints (best + soup variants)"
log "  Trials: 200  GPU: $GPU1"
log "=========================================="

# Build SED checkpoint list — prefer soup over best where available
SED_CKPTS=""
SED_CFGS=""
declare -A MODEL_CFGS=(
    ["sed-b0-v5"]="configs/sed_b0_v5.yaml"
    ["sed-b0-v6"]="configs/sed_b0_v6.yaml"
    ["sed-b2-v1"]="configs/sed_b2_v1.yaml"
)

for ckpt_name in "sed-b0-v5" "sed-b0-v6" "sed-b2-v1"; do
    cfg="${MODEL_CFGS[$ckpt_name]}"
    # Prefer soup checkpoint over best
    if [ -f "checkpoints/$ckpt_name/soup_sed.pt" ]; then
        ckpt="checkpoints/$ckpt_name/soup_sed.pt"
        log "  $ckpt_name: using soup checkpoint"
    elif [ -f "checkpoints/$ckpt_name/best_sed.pt" ]; then
        ckpt="checkpoints/$ckpt_name/best_sed.pt"
        log "  $ckpt_name: using best checkpoint"
    else
        log "  $ckpt_name: no checkpoint found — skipping"
        continue
    fi
    SED_CKPTS="$SED_CKPTS $ckpt"
    SED_CFGS="$SED_CFGS $cfg"
done

log "  SED checkpoints: $SED_CKPTS"

python3 scripts/optimize_ensemble.py \
    --gpu "$GPU1" \
    --sed_checkpoints $SED_CKPTS \
    --sed_configs $SED_CFGS \
    --n_trials 200 \
    --output outputs/ensemble_weights_optuna.json \
    2>&1 | tee -a "$LOG" || log "  (Phase 5 failed — check for optuna or holdout data issues)"

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

python3 scripts/update_plan_report.py 2>&1 | tee -a "$LOG"

# ── Done ──────────────────────────────────────────────────────────────────────
log ""
log "=========================================="
log " All Phases Complete!"
log "  Phase 1: TTA + TopN + Smoothing inference"
log "  Phase 2: sed-b0-v6 (CE loss + freq_masking)"
log "  Phase 3: sed-b2-v1 (EfficientNet-B2 backbone)"
log "  Phase 4: Model Soup (top-3 checkpoint averaging)"
log "  Phase 5: Optuna ensemble weights (Perch×3 + SED soup)"
log "  HTML  : reports/sed_improvement_plan.html"
log "  Log   : $LOG"
log "=========================================="
log ""
log "Checkpoints in submissions/weights/:"
ls -lh submissions/weights/*.pt 2>/dev/null | tee -a "$LOG" || log "  (none found)"
