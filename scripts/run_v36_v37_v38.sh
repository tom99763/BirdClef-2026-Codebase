#!/bin/bash
# Sequential experiment runner: v36 → v37 → v38 on GPU1
# All experiments run on CUDA_VISIBLE_DEVICES=1

set -e
cd /home/lab/BirdClef-2026-Codebase
GPU=1
LOG_DIR=outputs
CKPT_DIR=checkpoints

run_exp() {
    local config=$1
    local name=$(python3 -c "import yaml; c=yaml.safe_load(open('$config')); print(c['experiment']['name'])")
    local out_dir="$LOG_DIR/$name"
    mkdir -p "$out_dir"

    echo "[$(date '+%H:%M:%S')] ===== Launching $name on GPU$GPU ====="
    CUDA_VISIBLE_DEVICES=$GPU nohup python train_sed.py --config "$config" \
        > "$out_dir/train.log" 2>&1
    # wait for completion (nohup without & — blocks until done)
    local exit_code=$?

    # Extract best AUC from result.json
    local best=$(python3 -c "
import json, os
p = '$out_dir/result.json'
if os.path.exists(p):
    d = json.load(open(p))
    h = d.get('epoch_history', [])
    best = max((e.get('val_roc_auc',0) for e in h), default=0)
    ep = next((e['epoch'] for e in h if e.get('val_roc_auc',0)==best), 0)
    print(f'{best:.4f}@ep{ep}')
else:
    print('no result')
" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] $name DONE — best=$best"
}

echo "[$(date '+%H:%M:%S')] Starting experiment chain: v36 → v37 → v38 on GPU$GPU"

run_exp configs/sed_b0_v36_ce_plain.yaml
run_exp configs/sed_b0_v37_ce_pseudo.yaml
run_exp configs/sed_b0_v38_ce_sec03.yaml

echo "[$(date '+%H:%M:%S')] All experiments complete."
