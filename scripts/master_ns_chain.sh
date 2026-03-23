#!/usr/bin/env bash
# Master script: wait for Perch head training → regenerate teacher → start SED+SSM chains
#
# Usage:
#   nohup bash scripts/master_ns_chain.sh > outputs/logs/master_ns_chain.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
mkdir -p "$LOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [MASTER] $*"; }

# ── 1. Wait for Perch head training to finish ─────────────────────────────────
log "Waiting for Perch head training (train.py) to finish..."
while pgrep -f "train.py --config configs/exp_nohuman_label_soundscape_train.yaml" > /dev/null 2>&1; do
    sleep 60
done
log "Perch head training complete."

# ── 2. Re-extract Perch teacher predictions with new head ─────────────────────
log "Extracting Perch teacher predictions → outputs/perch_teacher_all_ss.csv"
python3 scripts/extract_perch_teacher_all_ss.py \
    --output outputs/perch_teacher_all_ss.csv \
    > "${LOG}/extract_perch_teacher.log" 2>&1
log "Teacher extraction done."

# ── 3. Regenerate Round-0 pseudo labels ──────────────────────────────────────
log "Generating pseudo_labels/ns_r0.csv from new teacher..."
python3 scripts/gen_pseudo_ns.py \
    --round    0 \
    --perch_csv outputs/perch_teacher_all_ss.csv \
    --perch_w  1.0 \
    --out      pseudo_labels/ns_r0.csv \
    > "${LOG}/gen_pseudo_ns_r0.log" 2>&1
log "ns_r0.csv ready."

# ── 4. Launch SED and SSM chains in parallel ─────────────────────────────────
log "Launching SED-10s NS chain..."
nohup bash scripts/auto_sed_ns_10s_full.sh \
    > "${LOG}/auto_sed_ns_10s_full.log" 2>&1 &
SED_PID=$!
log "SED chain PID: ${SED_PID}"

log "Launching SSM-10s NS chain..."
nohup bash scripts/auto_ssm_ns_10s_full.sh \
    > "${LOG}/auto_ssm_ns_10s_full.log" 2>&1 &
SSM_PID=$!
log "SSM chain PID: ${SSM_PID}"

# ── 5. Wait for both chains ───────────────────────────────────────────────────
log "Waiting for SED chain (PID ${SED_PID})..."
wait $SED_PID && log "SED chain DONE" || log "SED chain FAILED (exit $?)"

log "Waiting for SSM chain (PID ${SSM_PID})..."
wait $SSM_PID && log "SSM chain DONE" || log "SSM chain FAILED (exit $?)"

log "════════════════════════════════════════"
log "  MASTER NS CHAIN COMPLETE"
log "════════════════════════════════════════"
