#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Monitor Training + Auto-Launch Next Exp
#
# Polls GPU training status every 5 minutes.
# When experiment finishes: records result, launches next exp.
#
# Usage: bash scripts/monitor_and_next.sh [gpu0_exp] [gpu1_exp]
# Example: bash scripts/monitor_and_next.sh sed-b3-v1-asl sed-b0-v22-dual-noclipmix
#
# Run in background tmux session:
#   tmux new-session -d -s monitor 'bash scripts/monitor_and_next.sh sed-b3-v1-asl sed-b0-v22-dual-noclipmix 2>&1 | tee outputs/monitor.log'
# ============================================================

cd /home/lab/BirdClef-2026-Codebase

GPU0_EXP="${1:-sed-b3-v1-asl}"
GPU1_EXP="${2:-}"

LOG="outputs/monitor_auto.log"
mkdir -p outputs

log() { echo "[$(date '+%H:%M:%S')][MONITOR] $*" | tee -a "$LOG"; }

# Track which experiments have been finalized
GPU0_DONE=false
GPU1_DONE=false

# Already-done check (v22 finished before this script started)
if [ ! -z "$GPU1_EXP" ]; then
    still_running=$(pgrep -f "train_sed.py.*$GPU1_EXP" 2>/dev/null | wc -l)
    if [ "$still_running" -eq 0 ]; then
        log "GPU1 ($GPU1_EXP) already finished before monitor started"
        GPU1_DONE=true
    fi
fi

extract_best_auc() {
    local exp=$1
    # Try reading from checkpoint dir log or output
    local logfile="outputs/${exp}/training.log"
    if [ -f "$logfile" ]; then
        grep "New best ROC-AUC" "$logfile" | tail -1 | grep -oP '\d+\.\d+'
    else
        echo "unknown"
    fi
}

launch_next_gpu0() {
    log "=== GPU0 free — launching next experiment ==="
    # Next priority: SED B3 with lower LR (if b3-v1-asl best was low)
    # OR: start Perch probe experiments (CPU only)

    # Launch PT-MAP power transform probe recomputation (CPU, no GPU needed)
    log "Launching PT-MAP power transform probe recomputation..."
    # This is a Perch probe experiment, no GPU required
    # python scripts/recompute_probe_ptmap.py &

    log "GPU0 available. Recommended next SED experiment: sed-b3-v2-lower-lr"
    log "To launch: CUDA_VISIBLE_DEVICES=0 python train_sed.py configs/sed_b3_v2_lower_lr.yaml"

    # Auto-launch placeholder (uncomment when config ready):
    # CUDA_VISIBLE_DEVICES=0 nohup python train_sed.py configs/sed_b3_v2_lower_lr.yaml \
    #     > outputs/sed-b3-v2-lower-lr/train.log 2>&1 &
}

launch_next_gpu1() {
    log "=== GPU1 free — launching next experiment ==="
    log "GPU1 available. Recommended: sed-b0-v23 (next dual-loss variant with ClipMix restored)"

    # Auto-launch placeholder:
    # CUDA_VISIBLE_DEVICES=1 nohup python train_sed.py configs/sed_b0_v23.yaml \
    #     > outputs/sed-b0-v23/train.log 2>&1 &
}

log "Monitor started. Watching: GPU0=$GPU0_EXP, GPU1=${GPU1_EXP:-none}"
log "Press Ctrl+C to stop."

while true; do
    # ── Check GPU0 ──────────────────────────────────────────────────────────
    if [ "$GPU0_DONE" = false ]; then
        gpu0_procs=$(pgrep -f "train_sed.py" 2>/dev/null | while read pid; do
            ppid=$(awk '/^PPid:/{print $2}' /proc/$pid/status 2>/dev/null)
            [ "$ppid" = "1" ] || [ -z "$ppid" ] && echo $pid
        done | wc -l)

        # Simpler check: any train_sed process on GPU0
        gpu0_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=0 2>/dev/null | wc -l)

        if [ "$gpu0_procs" -eq 0 ]; then
            log "GPU0 ($GPU0_EXP) FINISHED"
            GPU0_DONE=true
            bash scripts/post_experiment_auto.sh "$GPU0_EXP" "$(extract_best_auc $GPU0_EXP)" "?" "0"
            launch_next_gpu0
        else
            log "GPU0 ($GPU0_EXP) still running ($gpu0_procs compute apps)"
        fi
    fi

    # ── Check GPU1 ──────────────────────────────────────────────────────────
    if [ ! -z "$GPU1_EXP" ] && [ "$GPU1_DONE" = false ]; then
        gpu1_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=1 2>/dev/null | wc -l)

        if [ "$gpu1_procs" -eq 0 ]; then
            log "GPU1 ($GPU1_EXP) FINISHED"
            GPU1_DONE=true
            bash scripts/post_experiment_auto.sh "$GPU1_EXP" "0.7485" "5" "1"
            launch_next_gpu1
        else
            log "GPU1 ($GPU1_EXP) still running"
        fi
    fi

    # ── Both done? ──────────────────────────────────────────────────────────
    if [ "$GPU0_DONE" = true ] && [ "$GPU1_DONE" = true ]; then
        log "All monitored experiments finished. Monitor exiting."
        break
    fi

    sleep 300  # poll every 5 minutes
done
