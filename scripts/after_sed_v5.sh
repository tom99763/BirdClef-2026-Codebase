#!/bin/bash
# Runs after sed-b0-v5 finishes:
# 1. Copies best_sed.pt to submissions/weights
# 2. Runs 4-model full ensemble holdout eval (v3)

set -e
cd /home/lab/BirdClef-2026-Codebase

SED_JSON="outputs/sed-b0-v5/result.json"

echo "[watcher] Waiting for sed-b0-v5 to finish..."
while true; do
    finished=$(python3 -c "import json; d=json.load(open('$SED_JSON')); print(d.get('finished', False))" 2>/dev/null)
    if [ "$finished" = "True" ]; then
        echo "[watcher] sed-b0-v5 finished!"
        break
    fi
    sleep 300
done

echo ""
echo "=== sed-b0-v5 FINAL RESULT ==="
python3 -c "
import json
d = json.load(open('$SED_JSON'))
h = d['epoch_history']
print(f'  Best val         : {d[\"best_val_roc_auc\"]:.4f}  (ep{d[\"best_epoch\"]})')
print(f'  Total epochs     : {d[\"total_epochs_run\"]}')
t = d['total_time_s']
print(f'  Total time       : {t/60:.1f} min' if t else '  Total time: N/A')
print('  Last 5 epochs:')
for e in h[-5:]:
    print(f'    ep{e[\"epoch\"]:3d}  val={e[\"val_roc_auc\"]:.4f}')
"

echo ""
echo "=== Copying best checkpoint ==="
cp checkpoints/sed-b0-v5/best_sed.pt submissions/weights/best_sed_b0_v5.pt
echo "  Copied → submissions/weights/best_sed_b0_v5.pt"

echo ""
echo "=== 4-model full ensemble holdout eval (Perch×3 + sed-b0-v5) ==="
python3 evaluate_ensemble_v3_holdout.py \
    2>&1 | tee outputs/ensemble_v3_holdout_eval.log

echo ""
echo "[watcher] Done."
echo "  outputs/ensemble_v3_holdout_eval.log"
