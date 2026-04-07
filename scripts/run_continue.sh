#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Continue from stopped training
#
# 1. Holdout eval + soup for v6 and v2s (already trained)
# 2. Stream A (GPU_A): Optuna → ASL → SoftSec
# 3. Stream B (GPU_B): 10s-eval → CutMix
# Both streams run in parallel.
#
# Usage:
#   bash scripts/run_continue.sh [GPU_A] [GPU_B]
#   bash scripts/run_continue.sh 1 0   # default
# ============================================================

set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase

GPU_A=${1:-1}
GPU_B=${2:-0}
LOG="outputs/run_all.log"
mkdir -p outputs submissions/weights

log() { echo "[$(date '+%H:%M:%S')][MASTER] $*" | tee -a "$LOG"; }

log "============================================================"
log " BirdCLEF 2026 Continue  PID=$$"
log " GPU_A=$GPU_A | GPU_B=$GPU_B"
log " Step 1: Holdout + soup for v6 and v2s"
log " Step 2: Optuna(A) | 10s-eval(B)"
log " Step 3: ASL(A) | CutMix(B)"
log " Step 4: SoftSec(A)"
log "============================================================"

holdout_eval() {
    local name=$1 cfg=$2 ckpt=$3 gpu=$4
    log "Holdout eval → $name"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval_sed_holdout.py \
        --checkpoint "$ckpt" --config "$cfg" --run_name "$name" \
        2>&1 | tee -a "$LOG"
    python3 -c "
import json, os
p = 'outputs/$name/holdout_eval.json'
if os.path.exists(p):
    d = json.load(open(p))
    print(f'  >>> $name  holdout_auc={d.get(\"holdout_auc\",\"N/A\")}')
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

# ── Step 1: Holdout + soup for v6 (GPU_A) and v2s (GPU_B) in parallel ─────────
log "=== Step 1: Holdout + Soup for v6 and v2s ==="

(
    holdout_eval "sed-b0-v6" "configs/sed_b0_v6.yaml" \
        "checkpoints/sed-b0-v6/best_sed.pt" "$GPU_A"
    soup_and_eval "sed-b0-v6" "configs/sed_b0_v6.yaml" "$GPU_A"
) &
PID_V6_EVAL=$!

(
    holdout_eval "sed-v2s-v1" "configs/sed_v2s_v1.yaml" \
        "checkpoints/sed-v2s-v1/best_sed.pt" "$GPU_B"
    soup_and_eval "sed-v2s-v1" "configs/sed_v2s_v1.yaml" "$GPU_B"
) &
PID_V2S_EVAL=$!

wait $PID_V6_EVAL && log "v6 eval done" || log "v6 eval error"
wait $PID_V2S_EVAL && log "v2s eval done" || log "v2s eval error"

# Print first holdout results
log "=== First Holdout Results ==="
python3 -c "
import json, glob
for p in sorted(glob.glob('outputs/*/holdout_eval*.json')):
    try:
        d = json.load(open(p))
        name = d.get('run_name', p)
        auc = d.get('holdout_auc') or d.get('holdout_auc_10s')
        if auc: print(f'  {name:<48} {float(auc):.4f}')
    except: pass
" 2>&1 | tee -a "$LOG"

# ── Step 2+3+4: Stream A and B in parallel ─────────────────────────────────────
log "=== Step 2: Launching Stream A and B in parallel ==="

# Stream A: Optuna → ASL → SoftSec
(
    # Optuna ensemble
    log "[GPU_A] Phase 3: Optuna Ensemble (200 trials)"
    SED_CKPTS="" SED_CFGS=""
    for mc in "sed-b0-v5 configs/sed_b0_v5.yaml" "sed-b0-v6 configs/sed_b0_v6.yaml"; do
        m=$(echo $mc | cut -d' ' -f1); c=$(echo $mc | cut -d' ' -f2)
        if [ -f "checkpoints/$m/soup_sed.pt" ]; then
            SED_CKPTS="$SED_CKPTS checkpoints/$m/soup_sed.pt"; SED_CFGS="$SED_CFGS $c"
        elif [ -f "checkpoints/$m/best_sed.pt" ]; then
            SED_CKPTS="$SED_CKPTS checkpoints/$m/best_sed.pt"; SED_CFGS="$SED_CFGS $c"
        fi
    done
    python3 scripts/optimize_ensemble.py --gpu "$GPU_A" \
        --sed_checkpoints $SED_CKPTS --sed_configs $SED_CFGS \
        --n_trials 200 --output outputs/ensemble_weights_optuna.json \
        2>&1 | tee -a "$LOG" || log "[GPU_A] Optuna failed (non-fatal)"

    # ASL
    log "[GPU_A] Phase 6: ASL Loss (sed-b0-v9-asl, 30ep)"
    CUDA_VISIBLE_DEVICES=$GPU_A python3 train_sed.py --config configs/sed_b0_v9_asl.yaml \
        2>&1 | tee -a "$LOG"
    holdout_eval "sed-b0-v9-asl" "configs/sed_b0_v9_asl.yaml" \
        "checkpoints/sed-b0-v9-asl/best_sed.pt" "$GPU_A"
    soup_and_eval "sed-b0-v9-asl" "configs/sed_b0_v9_asl.yaml" "$GPU_A"

    # SoftSec
    log "[GPU_A] Phase 8: Soft Secondary Labels (sed-b0-v11-soft-sec, 30ep)"
    CUDA_VISIBLE_DEVICES=$GPU_A python3 train_sed.py --config configs/sed_b0_v11_soft_sec.yaml \
        2>&1 | tee -a "$LOG"
    holdout_eval "sed-b0-v11-soft-sec" "configs/sed_b0_v11_soft_sec.yaml" \
        "checkpoints/sed-b0-v11-soft-sec/best_sed.pt" "$GPU_A"
    soup_and_eval "sed-b0-v11-soft-sec" "configs/sed_b0_v11_soft_sec.yaml" "$GPU_A"
    log "[GPU_A] Stream A complete"
) &
PID_A=$!

# Stream B: 10s-eval → CutMix
(
    # 10s evals
    log "[GPU_B] Phase 5: 10s Context Window Eval"
    for mc in "sed-b0-v5 configs/sed_b0_v5.yaml" "sed-b0-v6 configs/sed_b0_v6.yaml" \
              "sed-v2s-v1 configs/sed_v2s_v1.yaml"; do
        m=$(echo $mc | cut -d' ' -f1); c=$(echo $mc | cut -d' ' -f2)
        if [ -f "checkpoints/$m/soup_sed.pt" ]; then
            ckpt="checkpoints/$m/soup_sed.pt"; tag="${m}-soup-10s"
        elif [ -f "checkpoints/$m/best_sed.pt" ]; then
            ckpt="checkpoints/$m/best_sed.pt"; tag="${m}-best-10s"
        else continue; fi
        log "[GPU_B] 10s eval: $tag"
        CUDA_VISIBLE_DEVICES=$GPU_B python3 scripts/eval_10s_inference.py \
            --checkpoint "$ckpt" --config "$c" --run_name "$tag" \
            2>&1 | tee -a "$LOG"
        python3 -c "
import json, os
p = 'outputs/$tag/holdout_eval_10s.json'
if os.path.exists(p):
    d = json.load(open(p))
    print(f'  >>> $tag  holdout_auc_10s={d.get(\"holdout_auc_10s\",\"N/A\")}')
" 2>&1 | tee -a "$LOG"
    done

    # CutMix
    log "[GPU_B] Phase 7: CutMix (sed-b0-v10-cutmix, 30ep)"
    CUDA_VISIBLE_DEVICES=$GPU_B python3 train_sed.py --config configs/sed_b0_v10_cutmix.yaml \
        2>&1 | tee -a "$LOG"
    holdout_eval "sed-b0-v10-cutmix" "configs/sed_b0_v10_cutmix.yaml" \
        "checkpoints/sed-b0-v10-cutmix/best_sed.pt" "$GPU_B"
    soup_and_eval "sed-b0-v10-cutmix" "configs/sed_b0_v10_cutmix.yaml" "$GPU_B"
    log "[GPU_B] Stream B complete"
) &
PID_B=$!

wait $PID_A && log "Stream A finished OK" || log "Stream A exited with error"
wait $PID_B && log "Stream B finished OK" || log "Stream B exited with error"

# ── Final results table ────────────────────────────────────────────────────────
log ""
log "============================================================"
log " ALL EXPERIMENTS COMPLETE — Holdout AUC Ranking"
python3 -c "
import json, glob
results = []
for p in sorted(glob.glob('outputs/*/holdout_eval*.json')):
    try:
        d = json.load(open(p))
        name = d.get('run_name', p)
        auc = d.get('holdout_auc') or d.get('holdout_auc_10s')
        if auc: results.append((name, float(auc)))
    except: pass
print(f'  {\"Model\":<55} Holdout AUC')
print(f'  {\"-\"*66}')
for name, auc in sorted(results, key=lambda x: x[1], reverse=True):
    print(f'  {name:<55} {auc:.4f}')
" 2>&1 | tee -a "$LOG"
log "============================================================"
