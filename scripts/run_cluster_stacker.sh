#!/usr/bin/env bash
# ── run_cluster_stacker.sh ────────────────────────────────────────────────────
# 排隊等待 stacker-ss 訓練完成後執行 cluster stacker 實驗
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
LOG="outputs/logs"
mkdir -p "$LOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [CLUSTER-STACKER] $*" | tee -a "$LOG/cluster_stacker.log"; }

log "=== Waiting for any running stacker to finish ==="
while pgrep -f "python3.*train_stacker" > /dev/null; do
    sleep 60
done
log "=== No stacker running — starting cluster stacker ==="

python3 scripts/train_cluster_stacker.py \
    > "$LOG/cluster_stacker_train.log" 2>&1

log "=== Cluster stacker DONE ==="

# Summary
python3 - <<'PYEOF' | tee -a "$LOG/cluster_stacker.log"
import json
from pathlib import Path
import pandas as pd

excel = Path("birdclef-2026/notebook resource/current_subs 2/stacker_weights/stacker_results_cluster.xlsx")
if excel.exists():
    df = pd.read_excel(excel)
    print("\n" + "="*60)
    print("  Cluster Stacker Results")
    print("="*60)
    for _, row in df.sort_values("oof_auc", ascending=False).iterrows():
        print(f"  {row['exp']:<45} {row['oof_auc']:.4f}")
    print("="*60)
else:
    print("(results not found)")
PYEOF
