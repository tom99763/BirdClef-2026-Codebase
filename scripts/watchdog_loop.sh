#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Watchdog Loop
# Runs watchdog.sh every 15 minutes to ensure GPU jobs alive.
# Self-terminates once both v15 and v4 are finished.
# ============================================================
cd /home/lab/BirdClef-2026-Codebase
LOG="outputs/watchdog.log"

log() { echo "[$(date '+%H:%M:%S')][WATCHDOG-LOOP] $*" | tee -a "$LOG"; }
log "Watchdog loop started (PID=$$). Checking every 15 min."

while true; do
    # Check if both jobs are done
    V15_DONE=$(python3 -c "
import json, os
p = 'outputs/sed-b0-v15-no-sec/result.json'
print('True' if os.path.exists(p) and json.load(open(p)).get('finished') else 'False')
" 2>/dev/null || echo "False")
    V4_DONE=$(python3 -c "
import json, os
p = 'outputs/embed-distill-b0-v4/result.json'
print('True' if os.path.exists(p) and json.load(open(p)).get('finished') else 'False')
" 2>/dev/null || echo "False")

    if [ "$V15_DONE" = "True" ] && [ "$V4_DONE" = "True" ]; then
        log "Both v15 and v4-distill finished. Watchdog loop exiting."
        break
    fi

    # Run detached launcher (uses start_new_session, survives bash exits)
    python3 scripts/launch_detached.py 2>&1 | tee -a "$LOG"

    sleep 900  # 15 minutes
done
