#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# auto_hgnet.sh — HGNetV2-B0 autonomous training pipeline
#
# Strategy:
#   v1: Faithful notebook reproduction (train_audio, uniform LR)
#       → expect AUC ≈ notebook baseline
#   v2: Semi-supervised with SED pseudo labels
#       → init from best v1 fold + soundscape pseudo labels
#
# Monitor:
#   tail -f outputs/logs/auto_hgnet_main.log
#   tail -f outputs/logs/hgnet_v1_train.log
#   tail -f outputs/logs/hgnet_v2_train.log
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
PASS_THRESHOLD=0.9193

mkdir -p "$LOG" sed_improved

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AUTO-HGNET] $*" | tee -a "$LOG/auto_hgnet_main.log"; }

# ── Helper: get best AUC from result.json ─────────────────────────────────────
get_auc() {
    python3 -c "
import json, sys
try:
    d = json.load(open('$1'))
    aucs = [f['best_auc'] for f in d.get('folds', [])]
    print(f'{max(aucs):.4f}' if aucs else '0')
except: print('0')
" 2>/dev/null
}

# ── Helper: get path to best fold checkpoint ──────────────────────────────────
get_best_ckpt() {
    python3 -c "
import json
from pathlib import Path
try:
    d    = json.load(open('$1'))
    best = max(d['folds'], key=lambda x: x['best_auc'])
    p    = Path('$2') / f\"fold{best['fold']}_best.pt\"
    print(str(p))
except: print('')
" 2>/dev/null
}

# ── Helper: copy good folds to sed_improved ───────────────────────────────────
copy_to_sed_improved() {
    local result_json="$1"; local exp_dir="$2"; local exp_tag="$3"
    python3 - <<PYEOF
import json, shutil
from pathlib import Path
d   = json.load(open('$result_json'))
thr = $PASS_THRESHOLD
copied = 0
for f in d.get('folds', []):
    src = Path('$exp_dir') / f"fold{f['fold']}_best.pt"
    if src.exists() and f['best_auc'] >= thr:
        dst = Path('sed_improved') / f"${exp_tag}_fold{f['fold']}_auc{f['best_auc']:.4f}.pt"
        shutil.copy2(str(src), str(dst))
        print(f"Copied: {dst.name}")
        copied += 1
print(f"Total {copied} folds copied (threshold={thr})")
PYEOF
}

# ── Helper: update YAML field ──────────────────────────────────────────────────
update_yaml() {
    local yaml_file="$1"; local key="$2"; local value="$3"
    python3 -c "
import yaml, sys
with open('$yaml_file') as f:
    cfg = yaml.safe_load(f)
# Support nested key like 'model.init_ckpt'
keys = '$key'.split('.')
d = cfg
for k in keys[:-1]:
    d = d[k]
d[keys[-1]] = '$value'
with open('$yaml_file', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f'Updated $yaml_file: $key = $value')
"
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: hgnet-v1 (train_audio, exact notebook settings)
# ══════════════════════════════════════════════════════════════════════════════
V1_RESULT="outputs/hgnet-v1/result.json"
if [ ! -f "$V1_RESULT" ]; then
    log "=== Starting hgnet-v1 (exact notebook reproduction) ==="
    python3 train_hgnet.py \
        --config configs/hgnet_v1.yaml \
        --device "$DEVICE" \
        > "$LOG/hgnet_v1_train.log" 2>&1
fi
V1_AUC=$(get_auc "$V1_RESULT")
log "v1 complete — best fold AUC = $V1_AUC"
copy_to_sed_improved "$V1_RESULT" "outputs/hgnet-v1" "hgnet_v1"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: hgnet-v2 (semi-supervised, init from best v1 fold)
# ══════════════════════════════════════════════════════════════════════════════
V2_RESULT="outputs/hgnet-v2/result.json"
if [ ! -f "$V2_RESULT" ]; then
    # Resolve best v1 checkpoint
    V1_BEST_CKPT=$(get_best_ckpt "$V1_RESULT" "outputs/hgnet-v1")
    if [ -z "$V1_BEST_CKPT" ]; then
        log "ERROR: Could not find best v1 checkpoint — aborting v2"
        exit 1
    fi
    log "Best v1 checkpoint: $V1_BEST_CKPT"

    # Choose pseudo label CSV: prefer round5_pseudo (SED predictions), fallback to sed_20s_r1
    if [ -f "pseudo_labels/round5_pseudo.csv" ]; then
        PSEUDO_CSV="pseudo_labels/round5_pseudo.csv"
    elif [ -f "pseudo_labels/sed_20s_r1.csv" ]; then
        PSEUDO_CSV="pseudo_labels/sed_20s_r1.csv"
    else
        log "WARNING: No pseudo label CSV found — v2 will run without soundscape pseudo labels"
        PSEUDO_CSV=""
    fi

    # Update v2 config with best v1 ckpt and pseudo CSV
    cp configs/hgnet_v2.yaml configs/hgnet_v2.yaml.bak
    python3 - <<PYEOF
import yaml
with open('configs/hgnet_v2.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['data']['init_ckpt'] = '$V1_BEST_CKPT'
if '$PSEUDO_CSV':
    cfg['data']['perch_ss_csv'] = '$PSEUDO_CSV'
with open('configs/hgnet_v2.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f"Updated configs/hgnet_v2.yaml: data.init_ckpt={cfg['data']['init_ckpt']}")
print(f"  perch_ss_csv={cfg['data'].get('perch_ss_csv','')}")
PYEOF

    log "=== Starting hgnet-v2 (semi-supervised with pseudo labels) ==="
    python3 train_hgnet.py \
        --config configs/hgnet_v2.yaml \
        --device "$DEVICE" \
        > "$LOG/hgnet_v2_train.log" 2>&1
fi
V2_AUC=$(get_auc "$V2_RESULT")
log "v2 complete — best fold AUC = $V2_AUC"
copy_to_sed_improved "$V2_RESULT" "outputs/hgnet-v2" "hgnet_v2"

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
python3 - <<PYEOF | tee -a "$LOG/auto_hgnet_main.log"
import json
from pathlib import Path

summary = {"v1_best_auc": 0.0, "v2_best_auc": 0.0, "sed_improved": []}
for version in ["v1", "v2"]:
    rj = Path(f"outputs/hgnet-{version}/result.json")
    if rj.exists():
        d = json.load(open(rj))
        aucs = [f['best_auc'] for f in d.get('folds', [])]
        summary[f"{version}_best_auc"] = max(aucs) if aucs else 0.0

summary["sed_improved"] = [p.name for p in sorted(Path("sed_improved").glob("hgnet_*.pt"))]

print("\n=== HGNet Pipeline Summary ===")
print(f"  v1 best AUC: {summary['v1_best_auc']:.4f}")
print(f"  v2 best AUC: {summary['v2_best_auc']:.4f}")
print(f"  sed_improved files: {len(summary['sed_improved'])}")
for f in summary['sed_improved']:
    print(f"    {f}")

with open("outputs/auto_hgnet_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved: outputs/auto_hgnet_summary.json")
PYEOF

log "=== auto_hgnet.sh COMPLETE ==="
