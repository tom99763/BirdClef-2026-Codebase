#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Watchdog: ensure GPU jobs are running
#
# Safe to call multiple times — checks before starting.
# Only counts MAIN training processes (not DataLoader workers).
# Logs to outputs/watchdog.log
#
# GPU0: sed-b0-v15-no-sec (chain4 waits for it)
# GPU1: embed-distill-b0-v4 (chain3 waits for it)
# ============================================================

cd /home/lab/BirdClef-2026-Codebase
LOG="outputs/watchdog.log"
mkdir -p outputs

log() { echo "[$(date '+%H:%M:%S')][WATCHDOG] $*" | tee -a "$LOG"; }

# ── Helper: get PIDs of MAIN processes only (not DataLoader workers) ──────────
# Workers are children of the main training process (same pgrep pattern but
# their PPID is also in the match set). We exclude those.
main_pids() {
    local pattern="$1"
    local all_pids
    all_pids=$(pgrep -f "$pattern" 2>/dev/null) || true
    [ -z "$all_pids" ] && return
    local pid_set=" $all_pids "
    for pid in $all_pids; do
        ppid=$(awk '/^PPid:/{print $2}' /proc/$pid/status 2>/dev/null) || continue
        # If parent is also a match, this is a worker — skip
        if [[ "$pid_set" == *" $ppid "* ]]; then
            continue
        fi
        echo $pid
    done
}

count_main_procs() {
    local n
    n=$(main_pids "$1" | wc -l)
    echo $n
}

log "=== Watchdog check ==="

# ── GPU0: v15 SED training ─────────────────────────────────────────────────
V15_DONE=$(python3 -c "
import json, os
p = 'outputs/sed-b0-v15-no-sec/result.json'
print('True' if os.path.exists(p) and json.load(open(p)).get('finished') else 'False')
" 2>/dev/null || echo "False")

if [ "$V15_DONE" = "True" ]; then
    log "v15 already finished — skipping"
else
    N=$(count_main_procs 'train_sed.py.*sed_b0_v15')
    if [ "$N" -gt 1 ]; then
        log "WARNING: $N v15 main processes — killing duplicates"
        KEEP=$(main_pids 'train_sed.py.*sed_b0_v15' | sort -n | head -1)
        main_pids 'train_sed.py.*sed_b0_v15' | sort -n | tail -n +2 | xargs kill -9 2>/dev/null
        log "  Kept PID=$KEEP"
    elif [ "$N" -eq 1 ]; then
        log "v15 running OK (PID=$(main_pids 'train_sed.py.*sed_b0_v15'))"
    else
        log "v15 not running — starting on GPU0"
        bash -c 'CUDA_VISIBLE_DEVICES=0 python3 train_sed.py \
            --config configs/sed_b0_v15_no_sec.yaml \
            --gpu 0 \
            --pretrained_backbone checkpoints/embed-distill-b0-v1/best_backbone.pt \
            >> outputs/v15_restart.log 2>&1' &
        log "v15 start issued (will reparent to session)"
    fi
fi

# ── GPU1: embed-distill-b0-v4 ─────────────────────────────────────────────
V4_DONE=$(python3 -c "
import json, os
p = 'outputs/embed-distill-b0-v4/result.json'
print('True' if os.path.exists(p) and json.load(open(p)).get('finished') else 'False')
" 2>/dev/null || echo "False")

if [ "$V4_DONE" = "True" ]; then
    log "embed-distill-b0-v4 already finished — skipping"
else
    N=$(count_main_procs 'train_embed_distill.*b0_v4')
    if [ "$N" -gt 1 ]; then
        log "WARNING: $N v4 main processes — killing duplicates"
        KEEP=$(main_pids 'train_embed_distill.*b0_v4' | sort -n | head -1)
        main_pids 'train_embed_distill.*b0_v4' | sort -n | tail -n +2 | xargs kill -9 2>/dev/null
        log "  Kept PID=$KEEP"
    elif [ "$N" -eq 1 ]; then
        log "embed-distill-b0-v4 running OK (PID=$(main_pids 'train_embed_distill.*b0_v4'))"
    else
        log "embed-distill-b0-v4 not running — starting on GPU1"
        bash -c 'CUDA_VISIBLE_DEVICES=1 python3 train_embed_distill.py \
            --config configs/embed_distill_b0_v4.yaml \
            --gpu 1 \
            >> outputs/v4_distill_restart.log 2>&1' &
        log "embed-distill-b0-v4 start issued"
    fi
fi

# ── Chain scripts health check ────────────────────────────────────────────
for CHAIN in chain3_distill_sed chain4_backbone_comparison; do
    if pgrep -f "$CHAIN" > /dev/null; then
        log "$CHAIN running OK"
    else
        log "WARNING: $CHAIN not running!"
    fi
done

log "=== Watchdog done ==="
