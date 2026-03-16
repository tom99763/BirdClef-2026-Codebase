#!/bin/bash
# Auto-trigger: runs after nohuman-embedding-soundscape finishes
# 1. Converts embedding-head to TFLite
# 2. Runs solo holdout eval
# 3. Runs 3-model Perch ensemble holdout eval (v2)
# 4. Runs 4-model full ensemble holdout eval (v3, waits for sed-b0-v5 if needed)

set -e
cd /home/lab/BirdClef-2026-Codebase

RESULT_JSON="outputs/nohuman-embedding-soundscape/result.json"

echo "[watcher] Waiting for nohuman-embedding-soundscape to finish..."
while true; do
    finished=$(python3 -c "import json; d=json.load(open('$RESULT_JSON')); print(d.get('finished', False))" 2>/dev/null)
    if [ "$finished" = "True" ]; then
        echo "[watcher] nohuman-embedding-soundscape finished!"
        break
    fi
    sleep 60
done

echo ""
echo "=== nohuman-embedding-soundscape FINAL RESULT ==="
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
echo "=== Converting embedding-head to TFLite ==="
python3 convert_embedding_head_tflite.py \
    --run nohuman-embedding-soundscape \
    2>&1 | tee outputs/nohuman-embedding-soundscape/tflite_convert.log

echo ""
echo "=== Solo holdout eval: nohuman-embedding-soundscape ==="
python3 evaluate_holdout.py \
    --runs nohuman-embedding-soundscape \
    2>&1 | tee outputs/nohuman-embedding-soundscape/holdout_eval.log

echo ""
echo "=== 3-model Perch ensemble holdout eval (v2) ==="
python3 evaluate_ensemble_v2_holdout.py \
    2>&1 | tee outputs/ensemble_v2_holdout_eval.log

echo ""
echo "=== Waiting for sed-b0-v5 to finish (for 4-model ensemble) ==="
SED_JSON="outputs/sed-b0-v5/result.json"
while true; do
    sed_done=$(python3 -c "import json; d=json.load(open('$SED_JSON')); print(d.get('finished', False))" 2>/dev/null)
    if [ "$sed_done" = "True" ]; then
        echo "[watcher] sed-b0-v5 finished!"
        break
    fi
    echo "[watcher] sed-b0-v5 still running... sleeping 5 min"
    sleep 300
done

echo ""
echo "=== Copying sed-b0-v5 best checkpoint to submissions/weights ==="
cp checkpoints/sed-b0-v5/best_sed.pt submissions/weights/best_sed_b0_v5.pt
echo "  Copied → submissions/weights/best_sed_b0_v5.pt"

echo ""
echo "=== 4-model full ensemble holdout eval (v3: Perch×3 + SED) ==="
python3 evaluate_ensemble_v3_holdout.py \
    2>&1 | tee outputs/ensemble_v3_holdout_eval.log

echo ""
echo "[watcher] All done."
echo "  outputs/nohuman-embedding-soundscape/holdout_eval.log"
echo "  outputs/ensemble_v2_holdout_eval.log"
echo "  outputs/ensemble_v3_holdout_eval.log"
