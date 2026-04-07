#!/bin/bash
# ============================================================
# BirdCLEF 2026 — TTA Holdout Evaluation
#
# TTA strategy (BirdCLEF 2025 2nd-place):
#   - mode=hop   : dense 2.5s-hop sampling (2× clips, fast)
#   - mode=shift : ±2.5s shift per clip position (3× preds avg)
#   - mode=both  : hop + shift (6× coverage, best quality)
#
# Evaluates current best checkpoints with all 3 TTA modes.
# Compares against baseline (no TTA) from sed_holdout_eval.json.
#
# Waits for new SED experiments to finish first, then evaluates:
#   1. sed-b0-v5/best_sed.pt  (current best: holdout 0.9192)
#   2. sed-b0-v9-asl (when available, may beat v5)
#   3. sed-b0-v11-soft-sec (when available)
#   4. Any new model with holdout_auc > 0.9193
#
# Usage:
#   bash scripts/run_tta_eval.sh [GPU] [WAIT_FOR_EXPERIMENTS]
#   bash scripts/run_tta_eval.sh 0 1    # GPU=0, wait=yes (default)
#   bash scripts/run_tta_eval.sh 0 0    # GPU=0, run immediately
# ============================================================

set -euo pipefail
cd /home/lab/BirdClef-2026-Codebase

GPU=${1:-0}
WAIT_FOR_EXP=${2:-1}
LOG="outputs/run_tta_eval.log"
CONTINUE_PID=2346973   # run_continue.sh master PID
mkdir -p outputs

log() { echo "[$(date '+%H:%M:%S')][TTA] $*" | tee -a "$LOG"; }

log "============================================================"
log " BirdCLEF 2026 TTA Holdout Evaluation  GPU=$GPU"
log " TTA: 2.5s temporal shifts (2025 2nd-place technique)"
log "============================================================"

# ── Optionally wait for run_continue.sh experiments to finish ────────────────
if [ "$WAIT_FOR_EXP" = "1" ]; then
    log "Waiting for run_continue.sh (PID=$CONTINUE_PID) to finish..."
    while kill -0 $CONTINUE_PID 2>/dev/null; do
        sleep 60
    done
    log "Experiments finished. Starting TTA evaluation."
fi

# ── Helper: run all TTA modes for a single checkpoint ────────────────────────
run_tta_modes() {
    local name=$1 cfg=$2 ckpt=$3
    log ""
    log "--- $name ---"
    log "  checkpoint: $ckpt"

    if [ ! -f "$ckpt" ]; then
        log "  SKIP: checkpoint not found"
        return
    fi

    # baseline (no TTA) — skip if already exists
    if [ -f "outputs/${name}/sed_holdout_eval.json" ]; then
        local base_auc
        base_auc=$(python3 -c "import json; d=json.load(open('outputs/${name}/sed_holdout_eval.json')); print(d.get('holdout_auc','N/A'))")
        log "  baseline (no TTA): $base_auc  (already computed)"
    else
        log "  Running baseline (mode=none)..."
        CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout_tta.py \
            --checkpoint "$ckpt" --config "$cfg" \
            --run_name "$name" --tta_mode none \
            2>&1 | tee -a "$LOG"
    fi

    # hop TTA (fast, ~2× clips)
    log "  Running TTA mode=hop (2.5s hop, dense)..."
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout_tta.py \
        --checkpoint "$ckpt" --config "$cfg" \
        --run_name "$name" --tta_mode hop \
        2>&1 | tee -a "$LOG"

    # shift TTA (±2.5s, 3 preds per clip)
    log "  Running TTA mode=shift (±2.5s shift)..."
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout_tta.py \
        --checkpoint "$ckpt" --config "$cfg" \
        --run_name "$name" --tta_mode shift \
        2>&1 | tee -a "$LOG"

    # both (best quality)
    log "  Running TTA mode=both (hop + shift)..."
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout_tta.py \
        --checkpoint "$ckpt" --config "$cfg" \
        --run_name "$name" --tta_mode both \
        2>&1 | tee -a "$LOG"
}

# ── v5: current best (holdout 0.9192) ────────────────────────────────────────
run_tta_modes "sed-b0-v5" \
    "configs/sed_b0_v5.yaml" \
    "checkpoints/sed-b0-v5/best_sed.pt"

# ── v9-asl: if finished ───────────────────────────────────────────────────────
if [ -f "checkpoints/sed-b0-v9-asl/best_sed.pt" ]; then
    run_tta_modes "sed-b0-v9-asl" \
        "configs/sed_b0_v9_asl.yaml" \
        "checkpoints/sed-b0-v9-asl/best_sed.pt"
fi

# ── v11-soft-sec: if finished ─────────────────────────────────────────────────
if [ -f "checkpoints/sed-b0-v11-soft-sec/best_sed.pt" ]; then
    run_tta_modes "sed-b0-v11-soft-sec" \
        "configs/sed_b0_v11_soft_sec.yaml" \
        "checkpoints/sed-b0-v11-soft-sec/best_sed.pt"
fi

# ── Any Round 2/3 models with holdout > 0.9193 ───────────────────────────────
for name in sed-b0-v12-bce sed-b0-v13-asl-cutmix sed-b0-v14-50ep \
            sed-b0-v15-no-sec sed-b0-v16-rating3 \
            sed-b0-v17-dual30 sed-b0-v18-dual-ss10 sed-b0-v19-dual-freqmask \
            sed-b0-v20-dual-mixup08 sed-b0-v21-dual-rating3 sed-b0-v22-dual-noclipmix; do
    json="outputs/${name}/sed_holdout_eval.json"
    if [ -f "$json" ]; then
        auc=$(python3 -c "import json; d=json.load(open('$json')); print(d.get('holdout_auc', 0))" 2>/dev/null || echo 0)
        above=$(python3 -c "print(1 if float('$auc') > 0.9193 else 0)" 2>/dev/null || echo 0)
        if [ "$above" = "1" ]; then
            # convert "sed-b0-v17-dual30" → "sed_b0_v17_dual30" for config filename
        cfg_base="${name/sed-b0-/sed_b0_}"
        cfg_base="${cfg_base//-/_}"
        cfg=$(ls "configs/${cfg_base}.yaml" 2>/dev/null || echo "")
            ckpt="checkpoints/$name/best_sed.pt"
            # Try soup too
            soup="checkpoints/$name/soup_sed.pt"
            [ -n "$cfg" ] && run_tta_modes "$name" "$cfg" "$ckpt"
            if [ -f "$soup" ]; then
                [ -n "$cfg" ] && run_tta_modes "${name}-soup" "$cfg" "$soup"
            fi
        fi
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
log ""
log "============================================================"
log " TTA Evaluation Complete — Results Summary"
log " (baseline: v5 holdout_auc=0.9192)"
python3 - 2>&1 | tee -a "$LOG" << 'PYEOF'
import json, glob

results = []
for p in sorted(glob.glob("outputs/*/sed_holdout_eval.json")):
    try:
        d = json.load(open(p))
        name = d.get("run_name", p)
        auc  = d.get("holdout_auc")
        tta  = d.get("tta_mode", "none")
        if auc:
            results.append((name, float(auc), tta))
    except:
        pass

results.sort(key=lambda x: x[1], reverse=True)
print(f"\n  {'Run':<55} {'AUC':>8}  {'TTA':<8}  Beat v5?")
print(f"  {'-'*80}")
for name, auc, tta in results:
    flag = " *** BEATS V5 ***" if auc > 0.9192 else ""
    print(f"  {name:<55} {auc:.4f}   {tta:<8}{flag}")
PYEOF
log "============================================================"
