#!/usr/bin/env bash
# Launch SED improvement experiments v31-v36 sequentially on GPU1.
#
# Experiment matrix:
#   v31: BCE, lr=1e-3, cosine, no warmup             (LR schedule ablation)
#   v32: BCE, lr=5e-4, cosine, no warmup             (LR ablation vs v5)
#   v33: BCE, lr=1e-3, warm_restarts T0=10           (scheduler ablation)
#   v34: FocalBCE gamma=2.0, alpha=0.75, lr=1e-3    (focal baseline)
#   v35: FocalBCE gamma=3.0, alpha=0.75, lr=1e-3    (focal + higher gamma)
#   v36: BCEPosWeight sqrt, lr=1e-3                  (per-class weighting)
#
# All experiments: no warmup, dual loss, soundscape_val_frac=1.0, GPU1

set -e
export CUDA_VISIBLE_DEVICES=1

LOG_DIR="outputs"
mkdir -p "$LOG_DIR"

run_exp() {
    local config="$1"
    local name="$2"
    local logfile="$LOG_DIR/${name}.log"
    echo "=============================="
    echo "Starting: $name"
    echo "Config:   $config"
    echo "Log:      $logfile"
    echo "=============================="
    python3 train_sed.py --config "$config" 2>&1 | tee "$logfile"
    echo "Done: $name"
}

# Helper: exit 0 if experiment result.json has finished=true, else exit 1
_is_done() { python3 -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if d.get('finished') else 1)" 2>/dev/null; }

# Pair 1: Warm restarts + focal baseline
run_exp configs/sed_b0_v33_warmrestart.yaml  sed_b0_v33_warmrestart
_is_done outputs/sed-b0-v34-focal-g2/result.json    && echo "SKIP v34: already finished" || run_exp configs/sed_b0_v34_focal_g2.yaml sed_b0_v34_focal_g2

# Pair 3: Focal high-gamma + per-class weighting
_is_done outputs/sed-b0-v35-focal-g3/result.json    && echo "SKIP v35: already finished" || run_exp configs/sed_b0_v35_focal_g3.yaml sed_b0_v35_focal_g3
_is_done outputs/sed-b0-v36-pos-weight/result.json  && echo "SKIP v36: already finished" || run_exp configs/sed_b0_v36_pos_weight.yaml sed_b0_v36_pos_weight

echo ""
echo "=============================="
echo "All v31-v36 experiments done."
echo "=============================="

# ── Regularization ablations (added 2026-03-22) ──────────────────────────────
# v37: full regularization (dropout=0.3, wd=0.01, ls=0.1) + BCE + WarmRestart
# v38: full regularization + focal gamma=2.0
# Motivation: v33 shows overfitting (val peaks ep3=0.7625, drops ep4-5)
echo ""
echo "=============================="
echo "Starting regularization ablations v37-v38"
echo "=============================="

_is_done outputs/sed-b0-v37-reg/result.json       && echo "SKIP v37: already finished" || run_exp configs/sed_b0_v37_reg.yaml sed_b0_v37_reg
_is_done outputs/sed-b0-v38-reg-focal/result.json && echo "SKIP v38: already finished" || run_exp configs/sed_b0_v38_reg_focal.yaml sed_b0_v38_reg_focal

echo ""
echo "=============================="
echo "All v37-v38 regularization experiments done."
echo "=============================="
