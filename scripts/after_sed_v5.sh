#!/bin/bash
# Runs after sed-b0-v5 finishes:
# 1. Prints final result summary
# 2. Checks convergence — if val still rising, resumes for +20 epochs
# 3. Copies best_sed.pt to submissions/weights
# 4. Runs 4-model full ensemble holdout eval (v3)

set -e
cd /home/lab/BirdClef-2026-Codebase

SED_JSON="outputs/sed-b0-v5/result.json"
GPU=${1:-1}   # default GPU 1

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

# ── Convergence check: resume if val is still rising ──────────────────────────
STILL_RISING=$(python3 -c "
import json
d = json.load(open('$SED_JSON'))
h = d['epoch_history']
best_ep = d['best_epoch']
total_ep = d['total_epochs_run']
# Still rising if best is in last 3 epochs AND last-3 trend is positive
if len(h) < 3:
    print('no')
else:
    last3 = [e['val_roc_auc'] for e in h[-3:]]
    trend_up = last3[-1] > last3[0]
    best_recent = best_ep >= total_ep - 2
    print('yes' if (trend_up or best_recent) else 'no')
")

if [ "$STILL_RISING" = "yes" ]; then
    echo ""
    echo "=== Val AUC still rising — resuming for +20 epochs ==="
    python3 train_sed.py \
        --config configs/sed_b0_v5.yaml \
        --resume checkpoints/sed-b0-v5/best_sed.pt \
        --extra_epochs 20 \
        --gpu $GPU \
        2>&1 | tee -a outputs/sed-b0-v5.log

    echo ""
    echo "=== sed-b0-v5 RESUMED RESULT ==="
    python3 -c "
import json
d = json.load(open('$SED_JSON'))
h = d['epoch_history']
print(f'  Best val         : {d[\"best_val_roc_auc\"]:.4f}  (ep{d[\"best_epoch\"]})')
print(f'  Total epochs     : {d[\"total_epochs_run\"]}')
print('  Last 5 epochs:')
for e in h[-5:]:
    print(f'    ep{e[\"epoch\"]:3d}  val={e[\"val_roc_auc\"]:.4f}')
"
else
    echo "  Val converged (best not in last 3 epochs). Skipping resume."
fi

echo ""
echo "=== Copying best checkpoint ==="
cp checkpoints/sed-b0-v5/best_sed.pt submissions/weights/best_sed_b0_v5.pt
echo "  Copied → submissions/weights/best_sed_b0_v5.pt"

echo ""
echo "=== SED standalone holdout eval (sed-b0-v5) ==="
python3 scripts/eval_sed_holdout.py \
    --checkpoint checkpoints/sed-b0-v5/best_sed.pt \
    --config configs/sed_b0_v5.yaml \
    --run_name sed-b0-v5 \
    --gpu $GPU \
    2>&1 | tee outputs/sed-b0-v5-holdout.log

echo ""
echo "=== 4-model full ensemble holdout eval (Perch×3 + sed-b0-v5) ==="
python3 evaluate_ensemble_v3_holdout.py \
    2>&1 | tee outputs/ensemble_v3_holdout_eval.log

echo ""
echo "[watcher] Done."
echo "  outputs/ensemble_v3_holdout_eval.log"
