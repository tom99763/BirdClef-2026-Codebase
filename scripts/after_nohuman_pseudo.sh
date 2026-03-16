#!/bin/bash
# Auto-trigger: runs after nohuman-label-pseudo finishes
# 1. Waits for training to complete (finished=true in result.json)
# 2. Runs holdout evaluation
# 3. Starts nohuman-label-soundscape-train on GPU1

set -e
cd /home/lab/BirdClef-2026-Codebase

RESULT_JSON="outputs/nohuman-label-pseudo/result.json"
LOG="outputs/nohuman-label-pseudo.log"

echo "[watcher] Waiting for nohuman-label-pseudo to finish..."
while true; do
    finished=$(python3 -c "import json; d=json.load(open('$RESULT_JSON')); print(d.get('finished', False))" 2>/dev/null)
    if [ "$finished" = "True" ]; then
        echo "[watcher] nohuman-label-pseudo finished!"
        break
    fi
    sleep 30
done

# Print final result
echo ""
echo "=== nohuman-label-pseudo FINAL RESULT ==="
python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
h = d['epoch_history']
print(f'  Best val_roc_auc : {d[\"best_val_roc_auc\"]:.4f}  (ep{d[\"best_epoch\"]})')
print(f'  Total epochs     : {d[\"total_epochs_run\"]}')
print(f'  Total time       : {d[\"total_time_s\"]/60:.1f} min' if d['total_time_s'] else '  Total time: N/A')
print(f'  Last 5 epochs    :')
for e in h[-5:]:
    print(f'    ep{e[\"epoch\"]:3d}  val={e[\"val_roc_auc\"]:.4f}')
"

# Holdout evaluation
echo ""
echo "=== Running evaluate_holdout.py ==="
python3 evaluate_holdout.py --runs nohuman-label-pseudo 2>&1 | tee outputs/nohuman-label-pseudo/holdout_eval.log

# Start next experiment
echo ""
echo "=== Starting nohuman-label-soundscape-train on GPU1 ==="
nohup python3 train.py \
    --config configs/exp_nohuman_label_soundscape_train.yaml \
    --gpu 1 \
    > outputs/nohuman-label-soundscape-train.log 2>&1 &
echo "[watcher] nohuman-label-soundscape-train started (PID $!)"
echo "[watcher] Log: outputs/nohuman-label-soundscape-train.log"
