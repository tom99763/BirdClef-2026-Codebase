#!/bin/bash
# GPU_D stream (Round 2): 50ep → no-secondary → rating-filter
# Run after round 1 (run_all.sh) completes
set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase
GPU=${1:-0}
LOG="outputs/run_round2.log"

log() { echo "[$(date '+%H:%M:%S')][GPU_D] $*" | tee -a "$LOG"; }

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

# ── v14: 50-epoch CE ───────────────────────────────────────────────────────────
log "=== v14: Longer Training (sed-b0-v14-50ep, 50ep) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py --config configs/sed_b0_v14_50ep.yaml \
    2>&1 | tee -a "$LOG"
log "sed-b0-v14-50ep done"
holdout_eval "sed-b0-v14-50ep" "configs/sed_b0_v14_50ep.yaml" \
    "checkpoints/sed-b0-v14-50ep/best_sed.pt"
soup_and_eval "sed-b0-v14-50ep" "configs/sed_b0_v14_50ep.yaml"

# ── v15: No secondary labels ───────────────────────────────────────────────────
log "=== v15: No Secondary Labels (sed-b0-v15-no-sec, 30ep) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py --config configs/sed_b0_v15_no_sec.yaml \
    2>&1 | tee -a "$LOG"
log "sed-b0-v15-no-sec done"
holdout_eval "sed-b0-v15-no-sec" "configs/sed_b0_v15_no_sec.yaml" \
    "checkpoints/sed-b0-v15-no-sec/best_sed.pt"
soup_and_eval "sed-b0-v15-no-sec" "configs/sed_b0_v15_no_sec.yaml"

# ── v16: Quality filter (rating>=3) ───────────────────────────────────────────
log "=== v16: Rating Filter (sed-b0-v16-rating3, 30ep) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py --config configs/sed_b0_v16_rating3.yaml \
    2>&1 | tee -a "$LOG"
log "sed-b0-v16-rating3 done"
holdout_eval "sed-b0-v16-rating3" "configs/sed_b0_v16_rating3.yaml" \
    "checkpoints/sed-b0-v16-rating3/best_sed.pt"
soup_and_eval "sed-b0-v16-rating3" "configs/sed_b0_v16_rating3.yaml"

log "=== Stream D complete ==="
echo "STREAM_D_DONE" > /tmp/birdclef_stream_d_done
