#!/bin/bash
# GPU_C stream (Round 2): BCE → ASL+CutMix
# Run after round 1 (run_all.sh) completes
set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase
GPU=${1:-1}
LOG="outputs/run_round2.log"

log() { echo "[$(date '+%H:%M:%S')][GPU_C] $*" | tee -a "$LOG"; }

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

# ── v12: BCE loss ──────────────────────────────────────────────────────────────
log "=== v12: BCE Loss (sed-b0-v12-bce, 30ep) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py --config configs/sed_b0_v12_bce.yaml \
    2>&1 | tee -a "$LOG"
log "sed-b0-v12-bce done"
holdout_eval "sed-b0-v12-bce" "configs/sed_b0_v12_bce.yaml" \
    "checkpoints/sed-b0-v12-bce/best_sed.pt"
soup_and_eval "sed-b0-v12-bce" "configs/sed_b0_v12_bce.yaml"

# ── v13: ASL + CutMix combo ────────────────────────────────────────────────────
log "=== v13: ASL + CutMix (sed-b0-v13-asl-cutmix, 30ep) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py --config configs/sed_b0_v13_asl_cutmix.yaml \
    2>&1 | tee -a "$LOG"
log "sed-b0-v13-asl-cutmix done"
holdout_eval "sed-b0-v13-asl-cutmix" "configs/sed_b0_v13_asl_cutmix.yaml" \
    "checkpoints/sed-b0-v13-asl-cutmix/best_sed.pt"
soup_and_eval "sed-b0-v13-asl-cutmix" "configs/sed_b0_v13_asl_cutmix.yaml"

log "=== Stream C complete ==="
echo "STREAM_C_DONE" > /tmp/birdclef_stream_c_done
