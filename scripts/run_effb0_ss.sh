#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_effb0_ss.sh — EfficientNet-B0 SS baseline (fold 0, patience=7)
#
# Base: effb0-ss-v1 (鏡像 hgnet-ss-v1，只換 backbone)
# Monitor: tail -f outputs/logs/effb0_ss_v1_train.log
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
FOLD=0

mkdir -p "$LOG" weights/hgnet reports

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [EFFB0-SS] $*" | tee -a "$LOG/effb0_ss.log"; }

log "=== Starting effb0-ss-v1 (fold $FOLD) ==="
python3 train_hgnet.py \
    --config configs/effb0_ss_v1.yaml \
    --device "$DEVICE" \
    --fold "$FOLD" \
    > "$LOG/effb0_ss_v1_train.log" 2>&1
log "effb0-ss-v1 fold $FOLD done"

# ── Summary ───────────────────────────────────────────────────────────────────
python3 - <<'PYEOF' | tee -a "$LOG/effb0_ss.log"
import json
from pathlib import Path

experiments = [
    ("effb0-ss-v1", "outputs/effb0-ss-v1/result.json"),
]

print("\n" + "="*55)
print("  EfficientNet-B0 SS — fold 0 summary")
print("="*55)
print(f"  {'Experiment':<25} {'Fold0 AUC':>10}  {'Pass?':>6}")
print("-"*55)
for name, rjson in experiments:
    if Path(rjson).exists():
        try:
            d = json.load(open(rjson))
            fold0 = next((f for f in d.get('folds', []) if f['fold'] == 0), None)
            if fold0:
                auc = fold0['best_auc']
                flag = "✓" if auc >= 0.9193 else "✗"
                print(f"  {name:<25} {auc:>10.4f}  {flag:>6}")
            else:
                print(f"  {name:<25} {'(no fold0)':>10}")
        except Exception:
            print(f"  {name:<25} {'(error)':>10}")
    else:
        print(f"  {name:<25} {'(not done)':>10}")
print("="*55)
PYEOF

log "=== run_effb0_ss.sh DONE ==="
