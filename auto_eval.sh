#!/usr/bin/env bash
# Wait for a training run to finish, then run evaluate_final.py
# Usage: bash auto_eval.sh <run_name> <pid> [gpu]

RUN=$1
PID=$2
GPU=${3:-0}

echo "[auto_eval] Watching $RUN (PID=$PID) on GPU $GPU"
while kill -0 $PID 2>/dev/null; do
    sleep 60
done
echo "[auto_eval] Training finished for $RUN. Starting evaluation..."

CUDA_VISIBLE_DEVICES=$GPU python evaluate_final.py \
    --runs "$RUN" \
    --gpu "$GPU" \
    >> "outputs/evaluate_${RUN}.log" 2>&1

echo "[auto_eval] Evaluation done for $RUN."
cat "outputs/evaluate_${RUN}.log" | grep -E "Official|Score"
