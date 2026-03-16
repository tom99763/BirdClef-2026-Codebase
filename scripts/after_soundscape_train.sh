#!/bin/bash
# Auto-trigger: runs after nohuman-label-soundscape-train finishes
# 1. Waits for training to complete (finished=true in result.json)
# 2. Runs holdout eval for soundscape-train alone
# 3. Runs ensemble holdout eval (pseudo + soundscape-train, raw + PP)

set -e
cd /home/lab/BirdClef-2026-Codebase

RESULT_JSON="outputs/nohuman-label-soundscape-train/result.json"

echo "[watcher] Waiting for nohuman-label-soundscape-train to finish..."
while true; do
    finished=$(python3 -c "import json; d=json.load(open('$RESULT_JSON')); print(d.get('finished', False))" 2>/dev/null)
    if [ "$finished" = "True" ]; then
        echo "[watcher] nohuman-label-soundscape-train finished!"
        break
    fi
    sleep 60
done

echo ""
echo "=== nohuman-label-soundscape-train FINAL RESULT ==="
python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
h = d['epoch_history']
print(f'  Best holdout val : {d[\"best_val_roc_auc\"]:.4f}  (ep{d[\"best_epoch\"]})')
print(f'  Total epochs     : {d[\"total_epochs_run\"]}')
t = d['total_time_s']
print(f'  Total time       : {t/60:.1f} min' if t else '  Total time: N/A')
print('  Last 5 epochs:')
for e in h[-5:]:
    print(f'    ep{e[\"epoch\"]:3d}  val={e[\"val_roc_auc\"]:.4f}')
"

echo ""
echo "=== Holdout eval: nohuman-label-soundscape-train (solo) ==="
python3 evaluate_holdout.py \
    --runs nohuman-label-soundscape-train \
    2>&1 | tee outputs/nohuman-label-soundscape-train/holdout_eval.log

echo ""
echo "=== Ensemble holdout eval: pseudo + soundscape-train (raw + PP) ==="
python3 evaluate_ensemble_holdout.py \
    --gpu 1 \
    2>&1 | tee outputs/ensemble_holdout_eval.log

echo ""
echo "[watcher] All evaluations complete."
echo "[watcher] Results saved to:"
echo "  outputs/nohuman-label-soundscape-train/holdout_eval.log"
echo "  outputs/ensemble_holdout_eval.log"
