#!/bin/bash
# GPU_A stream: Phase1 → v6 → Optuna → ASL → SoftSec
# Each training immediately followed by holdout eval + soup + soup-holdout eval
set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase
GPU=${1:-1}
LOG="outputs/run_all.log"

log() { echo "[$(date '+%H:%M:%S')][GPU_A] $*" | tee -a "$LOG"; }

holdout_eval() {
    local name=$1 cfg=$2 ckpt=$3
    log "Holdout eval → $name"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout.py \
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
    local name=$1 cfg=$2
    if ls "checkpoints/$name/soup_ep"*.pt 1>/dev/null 2>&1; then
        log "Model Soup → $name"
        python3 scripts/model_soup.py --run "$name" --config "$cfg" 2>&1 | tee -a "$LOG"
        if [ -f "checkpoints/$name/soup_sed.pt" ]; then
            cp "checkpoints/$name/soup_sed.pt" "submissions/weights/soup_${name}.pt"
            holdout_eval "${name}-soup" "$cfg" "checkpoints/$name/soup_sed.pt"
        fi
    fi
}

# ── Phase 1: Inference enhancements ──────────────────────────────────────────
log "=== Phase 1: Inference Enhancements ==="
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/phase1_inference_eval.py \
    --checkpoint checkpoints/sed-b0-v5/best_sed.pt --gpu "$GPU" \
    2>&1 | tee -a "$LOG"
python3 -c "
import json
d = json.load(open('outputs/phase1_inference_eval.json'))
print(f'  Best: {d[\"best_variant\"]}  AUC={d[\"best_auc\"]:.4f}  gain={d[\"best_gain\"]:+.4f}')
" 2>&1 | tee -a "$LOG"

# ── Phase 2: sed-b0-v6 ───────────────────────────────────────────────────────
log "=== Phase 2: Train sed-b0-v6 (CE loss, 30ep) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py --config configs/sed_b0_v6.yaml \
    2>&1 | tee -a "$LOG"
log "sed-b0-v6 done"
cp checkpoints/sed-b0-v6/best_sed.pt submissions/weights/best_sed_b0_v6.pt
holdout_eval "sed-b0-v6" "configs/sed_b0_v6.yaml" "checkpoints/sed-b0-v6/best_sed.pt"
soup_and_eval "sed-b0-v6" "configs/sed_b0_v6.yaml"

# ── Phase 3: Optuna ensemble ─────────────────────────────────────────────────
log "=== Phase 3: Optuna Ensemble (v5 + v6, 200 trials) ==="
SED_CKPTS="" SED_CFGS=""
for mc in "sed-b0-v5 configs/sed_b0_v5.yaml" "sed-b0-v6 configs/sed_b0_v6.yaml"; do
    m=$(echo $mc | cut -d' ' -f1); c=$(echo $mc | cut -d' ' -f2)
    if [ -f "checkpoints/$m/soup_sed.pt" ]; then
        SED_CKPTS="$SED_CKPTS checkpoints/$m/soup_sed.pt"; SED_CFGS="$SED_CFGS $c"
        log "  $m: soup"
    elif [ -f "checkpoints/$m/best_sed.pt" ]; then
        SED_CKPTS="$SED_CKPTS checkpoints/$m/best_sed.pt"; SED_CFGS="$SED_CFGS $c"
        log "  $m: best"
    fi
done
python3 scripts/optimize_ensemble.py --gpu "$GPU" \
    --sed_checkpoints $SED_CKPTS --sed_configs $SED_CFGS \
    --n_trials 200 --output outputs/ensemble_weights_optuna.json \
    2>&1 | tee -a "$LOG" || log "Optuna failed"
python3 -c "
import json, os
p = 'outputs/ensemble_weights_optuna.json'
if os.path.exists(p):
    d = json.load(open(p))
    print(f'  Ensemble AUC={d[\"best_auc\"]:.4f}  equal-w={d[\"equal_w_auc\"]:.4f}  gain={d[\"gain\"]:+.4f}')
" 2>&1 | tee -a "$LOG"

# ── Phase 6: ASL loss ────────────────────────────────────────────────────────
log "=== Phase 6: ASL Loss (sed-b0-v9-asl, 30ep) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py --config configs/sed_b0_v9_asl.yaml \
    2>&1 | tee -a "$LOG"
log "sed-b0-v9-asl done"
holdout_eval "sed-b0-v9-asl" "configs/sed_b0_v9_asl.yaml" \
    "checkpoints/sed-b0-v9-asl/best_sed.pt"
soup_and_eval "sed-b0-v9-asl" "configs/sed_b0_v9_asl.yaml"

# ── Phase 8: Soft secondary labels ───────────────────────────────────────────
log "=== Phase 8: Soft Secondary Labels (sed-b0-v11-soft-sec, 30ep) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py --config configs/sed_b0_v11_soft_sec.yaml \
    2>&1 | tee -a "$LOG"
log "sed-b0-v11-soft-sec done"
holdout_eval "sed-b0-v11-soft-sec" "configs/sed_b0_v11_soft_sec.yaml" \
    "checkpoints/sed-b0-v11-soft-sec/best_sed.pt"
soup_and_eval "sed-b0-v11-soft-sec" "configs/sed_b0_v11_soft_sec.yaml"

log "=== Stream A complete ==="
echo "STREAM_A_DONE" > /tmp/birdclef_stream_a_done
