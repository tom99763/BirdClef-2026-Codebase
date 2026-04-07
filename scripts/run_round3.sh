#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Round 3 (All based on v5 dual-loss formula)
#
# Key insight: v5's holdout 0.9192 comes from frame_loss_weight=0.5
# All experiments here use BCE + dual loss (clip=0.5 + frame=0.5)
# Target: beat holdout AUC > 0.9193 to qualify for submission
#
# GPU_E: v17-dual30 → v19-dual-freqmask → v21-dual-rating3
# GPU_F: v18-dual-ss10 → v20-dual-mixup08 → v22-dual-noclipmix
#
# GPU_E total: ~33h  |  GPU_F total: ~33h
#
# Usage:
#   bash scripts/run_round3.sh [GPU_E] [GPU_F]
#   bash scripts/run_round3.sh 1 0   # default
# ============================================================

set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase

GPU_E=${1:-1}
GPU_F=${2:-0}
LOG="outputs/run_round3.log"
mkdir -p outputs submissions/weights

log() { echo "[$(date '+%H:%M:%S')][MASTER] $*" | tee -a "$LOG"; }

LOCK="/tmp/birdclef_round3.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
    echo "ABORT: already running (PID=$(cat $LOCK))"; exit 1
fi
echo $$ > "$LOCK"
trap "rm -f $LOCK" EXIT

log "============================================================"
log " BirdCLEF 2026 Round 3 — Dual-Loss Series  PID=$$"
log " GPU_E=$GPU_E | GPU_F=$GPU_F"
log ""
log " GPU_E: v17-dual30 → v19-dual-freqmask → v21-dual-rating3"
log " GPU_F: v18-dual-ss10 → v20-dual-mixup08 → v22-dual-noclipmix"
log " Submit threshold: holdout AUC > 0.9193 (v5 benchmark)"
log "============================================================"

holdout_eval() {
    local name=$1 cfg=$2 ckpt=$3 gpu=$4
    log "Holdout eval → $name"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval_sed_holdout.py \
        --checkpoint "$ckpt" --config "$cfg" --run_name "$name" \
        2>&1 | tee -a "$LOG"
    python3 -c "
import json, os
p = 'outputs/$name/sed_holdout_eval.json'
if not os.path.exists(p): p = 'outputs/$name/holdout_eval.json'
if os.path.exists(p):
    d = json.load(open(p))
    auc = d.get('holdout_auc', 'N/A')
    print(f'  >>> $name  holdout_auc={auc}  (target: >0.9193)')
    if isinstance(auc, float) and auc > 0.9193:
        print(f'  *** BEATS V5 BENCHMARK! Consider adding to ensemble. ***')
" 2>&1 | tee -a "$LOG"
}

soup_and_eval() {
    local name=$1 cfg=$2 gpu=$3
    if ls "checkpoints/$name/soup_ep"*.pt 1>/dev/null 2>&1; then
        log "Model Soup → $name"
        python3 scripts/model_soup.py --run "$name" --config "$cfg" 2>&1 | tee -a "$LOG"
        if [ -f "checkpoints/$name/soup_sed.pt" ]; then
            cp "checkpoints/$name/soup_sed.pt" "submissions/weights/soup_${name}.pt"
            holdout_eval "${name}-soup" "$cfg" "checkpoints/$name/soup_sed.pt" "$gpu"
        fi
    fi
}

train_eval() {
    local name=$1 cfg=$2 gpu=$3
    log "Training → $name"
    CUDA_VISIBLE_DEVICES=$gpu python3 train_sed.py --config "$cfg" \
        2>&1 | tee -a "$LOG"
    log "$name done"
    holdout_eval "$name" "$cfg" "checkpoints/$name/best_sed.pt" "$gpu"
    soup_and_eval "$name" "$cfg" "$gpu"
}

# ── Stream E: v17 → v19 → v21 ──────────────────────────────────────────────
(
    train_eval "sed-b0-v17-dual30"      "configs/sed_b0_v17_dual30.yaml"      "$GPU_E"
    train_eval "sed-b0-v19-dual-freqmask" "configs/sed_b0_v19_dual_freqmask.yaml" "$GPU_E"
    train_eval "sed-b0-v21-dual-rating3"  "configs/sed_b0_v21_dual_rating3.yaml"  "$GPU_E"
    log "[GPU_E] Stream E complete"
) &
PID_E=$!
log "Stream E launched (PID=$PID_E)"

# ── Stream F: v18 → v20 → v22 ──────────────────────────────────────────────
(
    train_eval "sed-b0-v18-dual-ss10"     "configs/sed_b0_v18_dual_ss10.yaml"     "$GPU_F"
    train_eval "sed-b0-v20-dual-mixup08"  "configs/sed_b0_v20_dual_mixup08.yaml"  "$GPU_F"
    train_eval "sed-b0-v22-dual-noclipmix" "configs/sed_b0_v22_dual_noclipmix.yaml" "$GPU_F"
    log "[GPU_F] Stream F complete"
) &
PID_F=$!
log "Stream F launched (PID=$PID_F)"

wait $PID_E && log "Stream E finished OK" || log "Stream E exited with error"
wait $PID_F && log "Stream F finished OK" || log "Stream F exited with error"

# ── Final ranking ────────────────────────────────────────────────────────────
log ""
log "============================================================"
log " ROUND 3 COMPLETE — Holdout AUC Ranking (target >0.9193)"
python3 -c "
import json, glob
results = []
for p in sorted(glob.glob('outputs/*/sed_holdout_eval.json') + glob.glob('outputs/*/holdout_eval.json')):
    try:
        d = json.load(open(p))
        name = d.get('run_name', p)
        auc  = d.get('holdout_auc') or d.get('holdout_auc_10s')
        if auc: results.append((name, float(auc)))
    except: pass
print(f'  {\"Model\":<55} Holdout AUC  Beat v5?')
print(f'  {\"-\"*75}')
for name, auc in sorted(results, key=lambda x: x[1], reverse=True):
    flag = ' *** BEATS V5 ***' if auc > 0.9193 else ''
    print(f'  {name:<55} {auc:.4f}{flag}')
" 2>&1 | tee -a "$LOG"
log "============================================================"
