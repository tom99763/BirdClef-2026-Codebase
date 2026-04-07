#!/usr/bin/env bash
# ── run_perch_ft_chain.sh ─────────────────────────────────────────────────────
# Multi-round Perch adapter fine-tuning chain: R0 → R1 → R2 → R3
# Each round depends on the previous round's checkpoint.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR" weights

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [PERCH-FT-CHAIN] $*" | tee -a "$LOG_DIR/perch_ft_chain.log"; }

# ── Round 0: Labeled soundscape supervised warm-up ───────────────────────────
log "=== Round 0: Supervised warm-up (labeled SS) ==="
python3 scripts/train_perch_ft.py --config configs/perch_ft_r0.yaml \
    2>&1 | tee "$LOG_DIR/perch_ft_r0_train.log"

if [ ! -f "weights/perch_adapter_r0.pt" ]; then
    log "ERROR: R0 checkpoint not found, aborting chain."
    exit 1
fi
log "=== Round 0 DONE ==="

# ── Round 1: + Prototype-filtered train audio ────────────────────────────────
log "=== Round 1: Prototype-filtered train audio ==="
python3 scripts/train_perch_ft.py --config configs/perch_ft_r1.yaml \
    2>&1 | tee "$LOG_DIR/perch_ft_r1_train.log"

if [ ! -f "weights/perch_adapter_r1.pt" ]; then
    log "ERROR: R1 checkpoint not found, aborting chain."
    exit 1
fi
log "=== Round 1 DONE ==="

# ── Round 2: Domain alignment ────────────────────────────────────────────────
log "=== Round 2: Domain alignment (MMD) ==="
python3 scripts/train_perch_ft.py --config configs/perch_ft_r2.yaml \
    2>&1 | tee "$LOG_DIR/perch_ft_r2_train.log"

if [ ! -f "weights/perch_adapter_r2.pt" ]; then
    log "ERROR: R2 checkpoint not found, aborting chain."
    exit 1
fi
log "=== Round 2 DONE ==="

# ── Round 3: Mean Teacher SSL ────────────────────────────────────────────────
log "=== Round 3: Mean Teacher SSL (unlabeled soundscapes) ==="
python3 scripts/train_perch_ft.py --config configs/perch_ft_r3.yaml \
    2>&1 | tee "$LOG_DIR/perch_ft_r3_train.log"

log "=== Round 3 DONE ==="

# ── Summary ──────────────────────────────────────────────────────────────────
log "=== Perch FT Chain COMPLETE ==="
python3 - <<'PYEOF' | tee -a "$LOG_DIR/perch_ft_chain.log"
import torch
from pathlib import Path

print("\n" + "="*60)
print("  Perch Adapter Fine-Tuning Chain — Results Summary")
print("="*60)
for rnd in range(4):
    ckpt_path = Path(f"weights/perch_adapter_r{rnd}.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        print(f"  R{rnd}: {ckpt_path} ({ckpt_path.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  R{rnd}: NOT FOUND")
print("="*60)
PYEOF
