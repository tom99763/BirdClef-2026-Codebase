#!/usr/bin/env bash
# Noisy Classmate Watchdog — monitors pipeline health and auto-recovers
# Checks every 5 minutes:
#   1. Is the main pipeline process alive?
#   2. Is the training log being updated? (stall detection)
#   3. Are there error messages in logs?
# If stall detected (no log update for 20 min), kills stuck process and restarts pipeline.
#
# Usage:
#   nohup bash scripts/watchdog_nc.sh > outputs/logs/watchdog_nc.log 2>&1 &

PIPELINE_SCRIPT="scripts/auto_nc_dual_gpu.sh"
PIPELINE_LOG="outputs/logs/auto_nc_dual_gpu.log"
WATCHDOG_LOG="outputs/logs/watchdog_nc.log"
CHECK_INTERVAL=300      # 5 minutes
STALL_THRESHOLD=1200    # 20 minutes without log update = stall

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] $*"; }

log "Watchdog started. Check interval=${CHECK_INTERVAL}s, stall threshold=${STALL_THRESHOLD}s"

while true; do
    NOW=$(date +%s)

    # ── Check 1: Pipeline process alive? ─────────────────────────────────
    if ! pgrep -f "auto_nc_dual_gpu.sh" > /dev/null 2>&1; then
        log "ALERT: Pipeline process NOT running! Restarting..."
        nohup bash "$PIPELINE_SCRIPT" > "$PIPELINE_LOG" 2>&1 &
        NEW_PID=$!
        log "Restarted pipeline with PID $NEW_PID"
        sleep 30
        continue
    fi

    # ── Check 2: Find active training log and check for stall ────────────
    ACTIVE_LOG=""
    for pattern in "sed_ns_pvt_r*_fold*.log" "sed_ns_b0_r*_fold*.log" "sed_ns_pvt_r*_infer.log" "sed_ns_b0_r*_infer.log" "sed_corrector_*.log"; do
        LATEST=$(ls -t outputs/logs/$pattern 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            MOD_TIME=$(stat -c %Y "$LATEST" 2>/dev/null)
            if [ -n "$MOD_TIME" ]; then
                AGE=$((NOW - MOD_TIME))
                if [ $AGE -lt $STALL_THRESHOLD ]; then
                    ACTIVE_LOG="$LATEST"
                    break
                fi
            fi
        fi
    done

    if [ -n "$ACTIVE_LOG" ]; then
        MOD_TIME=$(stat -c %Y "$ACTIVE_LOG" 2>/dev/null)
        AGE=$((NOW - MOD_TIME))

        if [ $AGE -gt $STALL_THRESHOLD ]; then
            log "STALL DETECTED: $ACTIVE_LOG not updated for ${AGE}s (threshold=${STALL_THRESHOLD}s)"

            # Check for errors in the stalled log
            ERRORS=$(grep -i "error\|exception\|traceback\|killed\|OOM\|CUDA out of memory" "$ACTIVE_LOG" 2>/dev/null | tail -3)
            if [ -n "$ERRORS" ]; then
                log "Errors found in $ACTIVE_LOG:"
                echo "$ERRORS" | while read line; do log "  $line"; done
            fi

            # Kill stuck training processes
            log "Killing stuck training processes..."
            pkill -f "train_sed_ns" 2>/dev/null
            pkill -f "train_sed_residual_corrector" 2>/dev/null
            pkill -f "auto_nc_dual_gpu" 2>/dev/null
            sleep 10

            # Restart pipeline
            log "Restarting pipeline..."
            nohup bash "$PIPELINE_SCRIPT" > "$PIPELINE_LOG" 2>&1 &
            NEW_PID=$!
            log "Restarted pipeline with PID $NEW_PID"
            sleep 60
            continue
        else
            # Normal operation - periodic status
            LAST_LINE=$(tail -1 "$ACTIVE_LOG" 2>/dev/null | head -c 200)
            log "OK: $ACTIVE_LOG (${AGE}s ago) | $LAST_LINE"
        fi
    else
        # No recent log found — check pipeline log
        PIPE_AGE=0
        if [ -f "$PIPELINE_LOG" ]; then
            PIPE_MOD=$(stat -c %Y "$PIPELINE_LOG" 2>/dev/null)
            PIPE_AGE=$((NOW - PIPE_MOD))
        fi

        if [ $PIPE_AGE -gt $STALL_THRESHOLD ]; then
            log "STALL DETECTED: No active training log and pipeline log stale (${PIPE_AGE}s)"
            pkill -f "auto_nc_dual_gpu" 2>/dev/null
            sleep 5
            nohup bash "$PIPELINE_SCRIPT" > "$PIPELINE_LOG" 2>&1 &
            log "Restarted pipeline with PID $!"
            sleep 60
            continue
        else
            log "OK: Pipeline running, between steps (pipeline log ${PIPE_AGE}s ago)"
        fi
    fi

    # ── Check 3: GPU health ──────────────────────────────────────────────
    GPU1_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
    GPU1_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')

    # If GPU1 has 0% util and 0 memory for extended period during training, something is wrong
    if [ -n "$GPU1_UTIL" ] && [ "$GPU1_UTIL" -eq 0 ] && [ "$GPU1_MEM" -lt 100 ]; then
        # Check if we're supposed to be training
        if pgrep -f "train_sed_ns" > /dev/null 2>&1; then
            log "WARNING: GPU1 idle (${GPU1_UTIL}%, ${GPU1_MEM}MB) but training process exists"
        fi
    fi

    # ── Check 4: Duplicate processes ─────────────────────────────────────
    N_PIPELINES=$(pgrep -f "auto_nc_dual_gpu.sh" | wc -l)
    if [ "$N_PIPELINES" -gt 1 ]; then
        log "WARNING: $N_PIPELINES pipeline processes detected! Killing extras..."
        # Keep the newest one
        PIDS=$(pgrep -f "auto_nc_dual_gpu.sh" | sort -n)
        KEEP=$(echo "$PIDS" | tail -1)
        echo "$PIDS" | head -n -1 | while read pid; do
            log "  Killing duplicate PID $pid (keeping $KEEP)"
            kill "$pid" 2>/dev/null
        done
    fi

    sleep $CHECK_INTERVAL
done
