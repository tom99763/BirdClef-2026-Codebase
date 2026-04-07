#!/usr/bin/env bash
# Launch SED + ProtoSSM training in parallel on GPU1.
#
# Schedule:
#   1. Extract Perch embeddings for ProtoSSM (blocking, ~5 min on CPU)
#   2. Launch SED (all 5 folds, sequential) in background → LOG_SED
#   3. Launch ProtoSSM (all 5 folds, sequential) in background → LOG_PROTO
#   Both run simultaneously; ProtoSSM is tiny so GPU contention is minimal.
#
# Usage:
#   bash scripts/launch_experiments.sh
#   bash scripts/launch_experiments.sh --dry-run   # print commands only

set -euo pipefail
# Do NOT set CUDA_VISIBLE_DEVICES — train scripts use --device cuda:1 directly.
# Setting CUDA_VISIBLE_DEVICES=1 remaps GPU1→cuda:0, conflicting with cuda:1 in configs.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR" outputs/sed-b0-v1 outputs/proto-ssm-v1

LOG_SED="$LOG_DIR/sed_b0_v1.log"
LOG_PROTO="$LOG_DIR/proto_ssm_v1.log"
LOG_EXTRACT="$LOG_DIR/extract_embeddings.log"

run() {
    echo "[CMD] $*"
    [[ $DRY -eq 0 ]] && "$@"
}

echo "============================================================"
echo "  BirdCLEF-2026 Experiment Launcher"
echo "  $(date)"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ── Step 1: Extract Perch embeddings (prerequisite for ProtoSSM) ──────────────
NPZ="outputs/perch_labeled_ss.npz"
if [[ -f "$NPZ" ]]; then
    echo "[skip] $NPZ already exists"
else
    echo ""
    echo "[1/3] Extracting Perch embeddings for ProtoSSM …"
    if [[ $DRY -eq 0 ]]; then
        python3 scripts/extract_ss_labeled_embeddings.py 2>&1 | tee "$LOG_EXTRACT"
    else
        echo "[CMD] python3 scripts/extract_ss_labeled_embeddings.py > $LOG_EXTRACT"
    fi
    echo "[1/3] Extraction done."
fi

# ── Step 2: Launch SED (all folds) in background ─────────────────────────────
echo ""
echo "[2/3] Launching SED (sed-b0-v1, all 5 folds) → $LOG_SED"
if [[ $DRY -eq 0 ]]; then
    python3 train_sed.py \
        --config configs/sed_b0_v1.yaml \
        2>&1 | tee "$LOG_SED" &
    SED_PID=$!
    echo "      SED PID=$SED_PID"
else
    echo "[CMD] python3 train_sed.py --config configs/sed_b0_v1.yaml > $LOG_SED &"
    SED_PID=0
fi

# ── Step 3: Launch ProtoSSM (all folds) in background ────────────────────────
echo ""
echo "[3/3] Launching ProtoSSM (proto-ssm-v1, all 5 folds) → $LOG_PROTO"
if [[ $DRY -eq 0 ]]; then
    python3 train_proto_ssm.py \
        --config configs/proto_ssm_v1.yaml \
        2>&1 | tee "$LOG_PROTO" &
    PROTO_PID=$!
    echo "      ProtoSSM PID=$PROTO_PID"
else
    echo "[CMD] python3 train_proto_ssm.py --config configs/proto_ssm_v1.yaml > $LOG_PROTO &"
    PROTO_PID=0
fi

# ── Save PID file for monitoring ──────────────────────────────────────────────
if [[ $DRY -eq 0 ]]; then
    echo "{\"sed_pid\": $SED_PID, \"proto_pid\": $PROTO_PID, \"started\": \"$(date -Iseconds)\"}" \
        > "$LOG_DIR/pids.json"
fi

echo ""
echo "============================================================"
echo "  Both experiments launched."
echo "  Monitor with:"
echo "    python3 scripts/monitor_experiments.py --excel"
echo "  Logs:"
echo "    tail -f $LOG_SED"
echo "    tail -f $LOG_PROTO"
echo "============================================================"

# Wait for both to finish
if [[ $DRY -eq 0 ]]; then
    wait $SED_PID   && echo "SED  finished."   || echo "SED  exited with error $?"
    wait $PROTO_PID && echo "Proto finished."  || echo "Proto exited with error $?"
    echo ""
    echo "All experiments done at $(date)"
    python3 scripts/monitor_experiments.py --excel
fi
