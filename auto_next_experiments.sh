#!/bin/bash
# Auto-orchestrator: waits for current experiments to finish, evaluates, then
# launches pseudo-label pipeline + next experiments.
#
# Usage: bash auto_next_experiments.sh &

set -e

LOG_DIR="outputs"

echo "[auto] Waiting for no-human-voice to complete …"
while kill -0 $(pgrep -f "exp_no_human_voice") 2>/dev/null; do sleep 30; done
echo "[auto] no-human-voice done."

echo "[auto] Waiting for bigger-head to complete …"
while kill -0 $(pgrep -f "exp_bigger_head") 2>/dev/null; do sleep 30; done
echo "[auto] bigger-head done."

echo "[auto] Running evaluate_final on new experiments …"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_final.py \
    --runs no-human-voice bigger-head \
    > outputs/evaluate_new.log 2>&1 &
wait

echo "[auto] Starting pseudo-label pipeline on GPU 0 …"
bash run_pseudo_pipeline.sh > outputs/pseudo_pipeline.log 2>&1 &

# Check if nohuman results are good → also try nohuman+bigger
echo "[auto] Starting soundscape-heavy oversample experiment on GPU 1 …"
CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --config configs/exp_soundscape_heavy.yaml \
    > outputs/soundscape-heavy.log 2>&1

echo "[auto] All post-experiments launched."
