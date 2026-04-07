#!/bin/bash
# GPU_B stream: V2-S → 10s-evals → CutMix
# Runs in parallel with stream_a.sh
set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase
GPU=${1:-0}
LOG="outputs/run_all.log"

log() { echo "[$(date '+%H:%M:%S')][GPU_B] $*" | tee -a "$LOG"; }

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

eval_10s() {
    local name=$1 cfg=$2 ckpt=$3 tag=$4
    log "10s eval → $tag"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_10s_inference.py \
        --checkpoint "$ckpt" --config "$cfg" --run_name "$tag" \
        2>&1 | tee -a "$LOG"
    python3 -c "
import json, os
p = 'outputs/$tag/holdout_eval_10s.json'
if os.path.exists(p):
    d = json.load(open(p))
    print(f'  >>> $tag  holdout_auc_10s={d.get(\"holdout_auc_10s\",\"N/A\")}')
" 2>&1 | tee -a "$LOG"
}

# ── Phase 4: EfficientNetV2-S ─────────────────────────────────────────────────
log "=== Phase 4: Train sed-v2s-v1 (V2-S, 30ep) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py --config configs/sed_v2s_v1.yaml \
    2>&1 | tee -a "$LOG"
log "sed-v2s-v1 done"
holdout_eval "sed-v2s-v1" "configs/sed_v2s_v1.yaml" \
    "checkpoints/sed-v2s-v1/best_sed.pt"
soup_and_eval "sed-v2s-v1" "configs/sed_v2s_v1.yaml"
if [ -f "checkpoints/sed-v2s-v1/soup_sed.pt" ]; then
    eval_10s "sed-v2s-v1" "configs/sed_v2s_v1.yaml" \
        "checkpoints/sed-v2s-v1/soup_sed.pt" "sed-v2s-v1-soup-10s"
fi

# ── Phase 5: 10s evals on v5/v6 (wait for v6-soup from stream_a) ─────────────
log "=== Phase 5: 10s Context Window Eval (v5/v6) ==="
log "  Waiting for sed-b0-v6 soup (stream_a) …"
for i in $(seq 1 180); do
    if [ -f "checkpoints/sed-b0-v6/soup_sed.pt" ]; then break; fi
    sleep 60
done
if [ -f "checkpoints/sed-b0-v6/soup_sed.pt" ]; then
    eval_10s "sed-b0-v6" "configs/sed_b0_v6.yaml" \
        "checkpoints/sed-b0-v6/soup_sed.pt" "sed-b0-v6-soup-10s"
fi
# v5 best (no soup from this run)
if [ -f "checkpoints/sed-b0-v5/best_sed.pt" ]; then
    eval_10s "sed-b0-v5" "configs/sed_b0_v5.yaml" \
        "checkpoints/sed-b0-v5/best_sed.pt" "sed-b0-v5-best-10s"
fi

# ── Phase 7: CutMix ──────────────────────────────────────────────────────────
log "=== Phase 7: CutMix (sed-b0-v10-cutmix, 30ep) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 train_sed.py --config configs/sed_b0_v10_cutmix.yaml \
    2>&1 | tee -a "$LOG"
log "sed-b0-v10-cutmix done"
holdout_eval "sed-b0-v10-cutmix" "configs/sed_b0_v10_cutmix.yaml" \
    "checkpoints/sed-b0-v10-cutmix/best_sed.pt"
soup_and_eval "sed-b0-v10-cutmix" "configs/sed_b0_v10_cutmix.yaml"

log "=== Stream B complete ==="
echo "STREAM_B_DONE" > /tmp/birdclef_stream_b_done
