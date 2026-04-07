#!/bin/bash
# Watchdog: monitor v24/v27/v28 → auto-launch v30/v31 when GPUs free
# v30 (B0 multi-pseudo) → GPU1 when v24 finishes
# v31 (V2S grand combo) → GPU1 when v27 finishes (after v30 OKed)
#
# Usage: bash scripts/watchdog_v30_v31.sh
# Run in birdclef-gpu1:0 (idle pane)

set -e
cd /home/lab/BirdClef-2026-Codebase

LOG_PREFIX="[watchdog-v30v31]"

log() { echo "$LOG_PREFIX $(date '+%H:%M:%S') $*"; }

check_finished() {
    local name=$1
    local ckpt_dir="checkpoints/$name"
    # Training is done if best_sed.pt exists and no train process uses it
    if [ -d "$ckpt_dir" ] && [ -f "$ckpt_dir/best_sed.pt" ]; then
        # Check if this experiment is still running in any tmux pane
        # If the process is gone, training is done
        local running=$(tmux capture-pane -t "birdclef-gpu1:1" -p -S -5 2>/dev/null | grep -c "train_sed.py\|Epoch.*$name\|$name" || true)
        echo "$running"
    else
        echo "0"
    fi
}

# ── Phase 1: Wait for v24 to finish, then launch v30 ─────────────────────────
log "Monitoring v24 (sed-b0-v24-soft-pseudo) on GPU1:1..."
V30_LAUNCHED=false

while true; do
    # Check if v24 pane shows early stop or training complete
    PANE1=$(tmux capture-pane -t "birdclef-gpu1:1" -p -S -8 2>/dev/null | tail -6)

    if echo "$PANE1" | grep -qE "Early stopping|SED training complete"; then
        log "✓ v24 finished! Launching v30 (B0 multi-pseudo) on GPU1:1..."
        sleep 10  # brief pause to let GPU memory free
        tmux send-keys -t "birdclef-gpu1:1" \
            "CUDA_VISIBLE_DEVICES=1 python train_sed.py --config configs/sed_b0_v30_multipseu.yaml --gpu 1 2>&1 | tee outputs/sed-b0-v30-multipseu.log" \
            Enter
        V30_LAUNCHED=true
        log "v30 launched on GPU1:1"
        break
    fi

    # Status heartbeat every 10 minutes
    V24_LAST=$(echo "$PANE1" | grep "Epoch\|val_roc" | tail -1)
    log "v24 status: $V24_LAST"
    sleep 600
done

# ── Phase 2: Wait for v27 to finish, then launch v31 ─────────────────────────
log "Monitoring v27 (sed-b0-v27-soft-boost) on GPU1:2..."

while true; do
    PANE2=$(tmux capture-pane -t "birdclef-gpu1:2" -p -S -8 2>/dev/null | tail -6)

    if echo "$PANE2" | grep -qE "Early stopping|SED training complete"; then
        log "✓ v27 finished! Launching v31 (V2S grand combo) on GPU1:2..."
        sleep 10
        tmux send-keys -t "birdclef-gpu1:2" \
            "CUDA_VISIBLE_DEVICES=1 python train_sed.py --config configs/sed_v2s_v4_grand_combo.yaml --gpu 1 2>&1 | tee outputs/sed-v2s-v4-grand-combo.log" \
            Enter
        log "v31 launched on GPU1:2"
        break
    fi

    V27_LAST=$(echo "$PANE2" | grep "Epoch\|val_roc" | tail -1)
    log "v27 status: $V27_LAST | v30 launched=$V30_LAUNCHED"
    sleep 600
done

log "All experiments queued. Monitor with tmux."
