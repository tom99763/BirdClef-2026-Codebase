#!/usr/bin/env bash
# =============================================================================
# Auto-eval: Run holdout evaluation on a completed experiment and update Excel.
#
# Usage:
#   bash scripts/auto_eval_update.sh <exp_name> <config_path> <gpu_id>
#
# Example:
#   bash scripts/auto_eval_update.sh sed-b0-v26-asl-npcen configs/sed_b0_v26_asl_npcen.yaml 1
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

EXP_NAME="${1:?Usage: $0 <exp_name> <config> <gpu>}"
CONFIG="${2:?}"
GPU="${3:-0}"

CKPT="checkpoints/${EXP_NAME}/best_sed.pt"
SOUP_CKPT="checkpoints/${EXP_NAME}/soup_sed.pt"
LOG="logs/holdout_${EXP_NAME}.log"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

echo "================================================================"
echo "  Auto-eval: $EXP_NAME"
echo "  Config:    $CONFIG"
echo "  GPU:       $GPU"
echo "================================================================"

# 1. Holdout eval on best checkpoint
echo "[eval] Running holdout eval (best checkpoint)..."
CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval_sed_holdout.py \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --run_name "${EXP_NAME}" \
    --gpu "$GPU" \
    2>&1 | tee "$LOG"

# 2. Build model soup from top-k checkpoints
echo "[soup] Building model soup..."
SOUP_LOG="logs/soup_${EXP_NAME}.log"
python scripts/model_soup.py \
    --checkpoint_dir "checkpoints/${EXP_NAME}" \
    --output_path "$SOUP_CKPT" \
    2>&1 | tee "$SOUP_LOG" || echo "[soup] model_soup.py failed — skipping"

# 3. Holdout eval on soup
if [[ -f "$SOUP_CKPT" ]]; then
    echo "[eval] Running holdout eval (soup checkpoint)..."
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/eval_sed_holdout.py \
        --checkpoint "$SOUP_CKPT" \
        --config "$CONFIG" \
        --run_name "${EXP_NAME}-soup" \
        --gpu "$GPU" \
        2>&1 | tee "logs/holdout_${EXP_NAME}_soup.log"
fi

# 4. Update Excel results
echo "[excel] Updating exp_results.xlsx..."
python3 << 'PYEOF'
import sys, os, json, glob
import pandas as pd
from datetime import datetime

exp_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('EXP_NAME', '')

def get_holdout_auc(run_name):
    """Read holdout AUC from eval output JSON."""
    pattern = f"outputs/{run_name}/holdout_results.json"
    matches = glob.glob(pattern)
    if not matches:
        pattern = f"outputs/*/holdout_results.json"
        matches = [f for f in glob.glob(pattern) if run_name in f]
    if matches:
        with open(matches[0]) as f:
            d = json.load(f)
        return d.get('holdout_roc_auc', d.get('macro_roc_auc'))
    return None

def get_val_auc(run_name):
    """Read best val AUC from training results JSON."""
    pattern = f"outputs/{run_name}/results.json"
    matches = glob.glob(pattern)
    if matches:
        with open(matches[0]) as f:
            d = json.load(f)
        return d.get('best_val_roc_auc', d.get('val_roc_auc'))
    return None

try:
    df = pd.read_excel('reports/exp_results.xlsx')
    exp_name_env = os.environ.get('EXP_NAME', '')

    for name in [exp_name_env, exp_name_env + '-soup']:
        mask = df['Experiment Name'].str.strip() == name
        if not mask.any():
            continue
        idx = df[mask].index[0]

        holdout = get_holdout_auc(name)
        val = get_val_auc(name.replace('-soup', ''))

        if holdout is not None:
            df.loc[idx, 'Holdout AUC'] = round(holdout, 6)
            df.loc[idx, 'Status'] = 'done'
            print(f"  {name}: holdout_auc={holdout:.4f}")
        if val is not None:
            df.loc[idx, 'Val Metric'] = round(val, 6)
        df.loc[idx, 'Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')

        # Flag if exceeds threshold
        if holdout and holdout > 0.9193:
            print(f"\n  🎯 {name} holdout {holdout:.4f} > 0.9193 threshold! Consider new ensemble submission.")

    df.to_excel('reports/exp_results.xlsx', index=False)
    print("  Excel updated.")
except Exception as e:
    print(f"  Excel update failed: {e}")
PYEOF
EXP_NAME="$EXP_NAME" python3 /dev/stdin << 'PYEOF'
import sys, os
# already ran inline above
PYEOF

echo "================================================================"
echo "  Done: $EXP_NAME"
echo "================================================================"
