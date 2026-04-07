#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# auto_hgnet_ss.sh — HGNet SS ablation study (fold 0 only, patience=7)
#
# Base: hgnet-ss-v1 (train_audio + soundscapes GT, AUC=0.9238)
#
# 13 experiments:
#   ss-v1          : baseline SS GT (DONE)
#   ss-asl         : ASL loss
#   ss-focal       : Focal BCE + SpecAugment
#   ss-sumix       : SumixFreq
#   ss-cutmix      : CutMix + GainNorm
#   ss-swa         : SWA + LabelSmooth
#   ss-combo       : ASL + SumixFreq + SpecAugment + GainNorm + SWA
#   ss-warmrestart : CosineAnnealingWarmRestarts T0=5
#   ss-asl-sumix   : ASL + SumixFreq (ablation)
#   ss-higher-ss   : ss_gt_weight=0.5 (more soundscape data)
#   ss-strong-aug  : maximum augmentation pressure
#   ss-llrd        : LLRD backbone_lr*0.1
#   ss-asl-warmrestart: ASL + WarmRestart
#
# Results: reports/hgnet_experiments.xlsx
# Monitor: tail -f outputs/logs/auto_hgnet_ss.log
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
FOLD=0

mkdir -p "$LOG" sed_improved reports

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AUTO-HGNET-SS] $*" | tee -a "$LOG/auto_hgnet_ss.log"; }

run_exp() {
    local config="$1"
    local tag="$2"
    local result_json="$3"

    if [ -f "$result_json" ]; then
        fold_done=$(python3 -c "
import json
try:
    d = json.load(open('$result_json'))
    folds = [f['fold'] for f in d.get('folds', [])]
    print('yes' if $FOLD in folds else 'no')
except: print('no')
" 2>/dev/null)
        if [ "$fold_done" = "yes" ]; then
            log "SKIP $tag — fold $FOLD already done (result exists)"
            return
        fi
    fi

    log "=== Starting $tag (fold $FOLD) ==="
    python3 train_hgnet.py \
        --config "$config" \
        --device "$DEVICE" \
        --fold "$FOLD" \
        > "$LOG/hgnet_ss_${tag}_train.log" 2>&1
    log "$tag fold $FOLD done"
}

# ── Experiments (ss-v1 already done → auto-skipped) ──────────────────────────
run_exp "configs/hgnet_ss_v1.yaml"            "ss_v1"            "outputs/hgnet-ss-v1/result.json"
run_exp "configs/hgnet_ss_asl.yaml"           "ss_asl"           "outputs/hgnet-ss-asl/result.json"
run_exp "configs/hgnet_ss_focal.yaml"         "ss_focal"         "outputs/hgnet-ss-focal/result.json"
run_exp "configs/hgnet_ss_sumix.yaml"         "ss_sumix"         "outputs/hgnet-ss-sumix/result.json"
run_exp "configs/hgnet_ss_cutmix.yaml"        "ss_cutmix"        "outputs/hgnet-ss-cutmix/result.json"
run_exp "configs/hgnet_ss_swa.yaml"           "ss_swa"           "outputs/hgnet-ss-swa/result.json"
run_exp "configs/hgnet_ss_warmrestart.yaml"   "ss_warmrestart"   "outputs/hgnet-ss-warmrestart/result.json"
run_exp "configs/hgnet_ss_asl_sumix.yaml"     "ss_asl_sumix"     "outputs/hgnet-ss-asl-sumix/result.json"
run_exp "configs/hgnet_ss_higher_ss.yaml"     "ss_higher_ss"     "outputs/hgnet-ss-higher-ss/result.json"
run_exp "configs/hgnet_ss_llrd.yaml"          "ss_llrd"          "outputs/hgnet-ss-llrd/result.json"
run_exp "configs/hgnet_ss_strong_aug.yaml"    "ss_strong_aug"    "outputs/hgnet-ss-strong-aug/result.json"
run_exp "configs/hgnet_ss_asl_warmrestart.yaml" "ss_asl_warmrestart" "outputs/hgnet-ss-asl-warmrestart/result.json"
run_exp "configs/hgnet_ss_combo.yaml"         "ss_combo"         "outputs/hgnet-ss-combo/result.json"

# ── Summary ───────────────────────────────────────────────────────────────────
python3 - <<'PYEOF' | tee -a "$LOG/auto_hgnet_ss.log"
import json
from pathlib import Path

experiments = [
    ("hgnet-ss-v1",             "outputs/hgnet-ss-v1/result.json"),
    ("hgnet-ss-asl",            "outputs/hgnet-ss-asl/result.json"),
    ("hgnet-ss-focal",          "outputs/hgnet-ss-focal/result.json"),
    ("hgnet-ss-sumix",          "outputs/hgnet-ss-sumix/result.json"),
    ("hgnet-ss-cutmix",         "outputs/hgnet-ss-cutmix/result.json"),
    ("hgnet-ss-swa",            "outputs/hgnet-ss-swa/result.json"),
    ("hgnet-ss-warmrestart",    "outputs/hgnet-ss-warmrestart/result.json"),
    ("hgnet-ss-asl-sumix",      "outputs/hgnet-ss-asl-sumix/result.json"),
    ("hgnet-ss-higher-ss",      "outputs/hgnet-ss-higher-ss/result.json"),
    ("hgnet-ss-llrd",           "outputs/hgnet-ss-llrd/result.json"),
    ("hgnet-ss-strong-aug",     "outputs/hgnet-ss-strong-aug/result.json"),
    ("hgnet-ss-asl-warmrestart","outputs/hgnet-ss-asl-warmrestart/result.json"),
    ("hgnet-ss-combo",          "outputs/hgnet-ss-combo/result.json"),
]

print("\n" + "="*65)
print("  HGNet SS Ablation — fold 0 final summary")
print("="*65)
print(f"  {'Experiment':<30} {'Fold0 AUC':>10}  {'Pass?':>6}")
print("-"*65)
best_exp, best_auc = "", 0.0
for name, rjson in experiments:
    if Path(rjson).exists():
        try:
            d = json.load(open(rjson))
            fold0 = next((f for f in d.get('folds', []) if f['fold'] == 0), None)
            if fold0:
                auc = fold0['best_auc']
                flag = "✓" if auc >= 0.9193 else "✗"
                print(f"  {name:<30} {auc:>10.4f}  {flag:>6}")
                if auc > best_auc:
                    best_auc, best_exp = auc, name
            else:
                print(f"  {name:<30} {'(no fold0)':>10}")
        except Exception:
            print(f"  {name:<30} {'(error)':>10}")
    else:
        print(f"  {name:<30} {'(not done)':>10}")
print("-"*65)
print(f"  Best: {best_exp}  AUC={best_auc:.4f}")
print("="*65)
PYEOF

log "=== auto_hgnet_ss.sh ALL DONE ==="
