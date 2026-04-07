#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# auto_finetune.sh — 4-day autonomous fine-tuning pipeline
#
# Strategy: Fine-tune competitor SED (AUC=0.9478) progressively
#
#   v1: train_audio only, low LR (baseline — expect 0.93+)
#   v2: train_audio + Perch SS pseudo labels (expect 0.94+)
#   [gen pseudo]: use best v2 fold to generate updated pseudo on soundscapes
#   v3: train_audio + updated pseudo, init from best v2 (expect 0.94-0.95)
#   [gen pseudo]: use best v3 fold to generate round2 pseudo
#   v4: another noisy student round, init from best v3
#   [soup]: model soup of all folds >= 0.9193
#   [summary]: write auto_finetune_summary.json
#
# Monitor:
#   tail -f outputs/logs/auto_finetune_main.log
#   tail -f outputs/logs/finetune_v1_train.log
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
SUMMARY="outputs/auto_finetune_summary.json"
PASS_THRESHOLD=0.9193

mkdir -p "$LOG" sed_improved

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AUTO-FT] $*" | tee -a "$LOG/auto_finetune_main.log"; }

# ── Helper: export newly added .pt files in sed_improved/ to ONNX ─────────────
export_onnx_for_tag() {
    local tag="$1"   # e.g. ftcomp_v1
    log "Exporting ${tag} checkpoints to ONNX (3-ch input, output=probs) …"
    python3 - <<PYEOF >> "$LOG/auto_finetune_onnx.log" 2>&1
import torch, torch.nn as nn, sys
from pathlib import Path
sys.path.insert(0, '.')
from train_finetune_competitor import SEDModel, SR

class SEDModelONNX(nn.Module):
    """Wrapper: 3-ch mel input → sigmoid probs. Matches notebook MelTransform output."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):           # x: (B, 3, n_mels, T)
        feat = self.base.backbone(x)
        feat = self.base.gem_pool(feat)
        return self.base.head(feat)['clipwise_prob']   # 'probs' output

n_mels = 224
T      = int(5 * SR / 512) + 1    # 313
dummy  = torch.zeros(1, 3, n_mels, T)   # 3-channel, matches notebook MelTransform

pt_files = sorted(Path('sed_improved').glob('${tag}*.pt'))
print(f'[${tag}] Converting {len(pt_files)} .pt files ...')
for pt_path in pt_files:
    onnx_path = pt_path.with_suffix('.onnx')
    try:
        ck    = torch.load(str(pt_path), map_location='cpu', weights_only=False)
        state = ck.get('state_dict', ck.get('model_state_dict', ck))
        base  = SEDModel()
        base.load_state_dict(state, strict=False)
        base.eval()
        model = SEDModelONNX(base)
        model.eval()
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names  = ['mel'],
            output_names = ['probs'],
            dynamic_axes = {'mel':   {0: 'batch', 3: 'time'},
                            'probs': {0: 'batch'}},
            opset_version = 17,
        )
        print(f'  OK: {onnx_path.name} ({onnx_path.stat().st_size/1e6:.1f} MB)')
    except Exception as e:
        print(f'  FAIL {pt_path.name}: {e}')
PYEOF
    log "ONNX export done for ${tag}"
}

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

# ── Helper: get path to best fold checkpoint ─────────────────────────────────
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
for f in d.get('folds', []):
    src = Path('$exp_dir') / f"fold{f['fold']}_best.pt"
    if src.exists() and f['best_auc'] >= thr:
        dst = Path('sed_improved') / f"${exp_tag}_fold{f['fold']}_auc{f['best_auc']:.4f}.pt"
        shutil.copy2(str(src), str(dst))
        print(f"Copied: {dst.name}")
PYEOF
}

# ── Helper: generate pseudo labels on soundscapes using a checkpoint ──────────
gen_pseudo_from_ckpt() {
    local ckpt="$1"; local out_csv="$2"
    log "Generating pseudo labels from $ckpt → $out_csv"
    python3 - <<PYEOF > "$LOG/gen_pseudo_$(basename $out_csv .csv).log" 2>&1
import os, sys, torch, numpy as np, soundfile as sf, librosa, pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '.')
from train_finetune_competitor import SEDModel, MelTransform, SR, NUM_CLASSES

device = torch.device('cuda:0')
tax    = pd.read_csv('birdclef-2026/taxonomy.csv')
species = tax['primary_label'].tolist()

model = SEDModel().to(device)
ckpt  = torch.load('$ckpt', map_location='cpu', weights_only=False)
state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
model.load_state_dict(state, strict=False)
model.eval()

mel_tf = MelTransform().to(device)
STRIDE = SR * 5

ss_dir = Path('birdclef-2026/train_soundscapes')
rows   = []
for ogg in tqdm(sorted(ss_dir.glob('*.ogg')), desc='SS pseudo'):
    try:
        audio, sr = sf.read(str(ogg), dtype='float32', always_2d=False)
        if audio.ndim == 2: audio = audio.mean(axis=1)
        if sr != SR: audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    except: continue
    n = len(audio) // STRIDE
    for i in range(n):
        clip = audio[i*STRIDE:(i+1)*STRIDE]
        clip = np.pad(clip, (0, max(0, SR*5 - len(clip))))[:SR*5]
        m = np.abs(clip).max()
        if m > 1e-8: clip = clip / m
        wav = torch.from_numpy(clip[None]).to(device)
        with torch.no_grad():
            mel  = mel_tf(wav)
            prob = torch.sigmoid(model(mel)['clipwise_logit']).cpu().numpy()[0]
        rid = f'{ogg.stem}_{(i+1)*5}'
        rows.append([rid] + list(prob))

df = pd.DataFrame(rows, columns=['row_id'] + species)
df.to_csv('$out_csv', index=False)
print(f'Saved {len(df)} rows → $out_csv')
PYEOF
    log "Done: $out_csv ($(wc -l < $out_csv) rows)"
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: finetune-comp-v1 (train_audio only, baseline)
# ══════════════════════════════════════════════════════════════════════════════
V1_RESULT="outputs/finetune-comp-v1/result.json"
if [ ! -f "$V1_RESULT" ]; then
    log "=== Starting finetune-comp-v1 (train_audio + hard labels) ==="
    python3 train_finetune_competitor.py \
        --config configs/finetune_comp_v1.yaml \
        --device "$DEVICE" \
        > "$LOG/finetune_v1_train.log" 2>&1
fi
V1_AUC=$(get_auc "$V1_RESULT")
log "v1 complete — best fold AUC = $V1_AUC"
copy_to_sed_improved "$V1_RESULT" "outputs/finetune-comp-v1" "ftcomp_v1"
export_onnx_for_tag "ftcomp_v1"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: finetune-comp-v2 (+ Perch SS pseudo labels)
# ══════════════════════════════════════════════════════════════════════════════
V2_RESULT="outputs/finetune-comp-v2/result.json"
if [ ! -f "$V2_RESULT" ]; then
    log "=== Starting finetune-comp-v2 (+ Perch SS pseudo, ss_weight=0.3) ==="
    python3 train_finetune_competitor.py \
        --config configs/finetune_comp_v2.yaml \
        --device "$DEVICE" \
        > "$LOG/finetune_v2_train.log" 2>&1
fi
V2_AUC=$(get_auc "$V2_RESULT")
log "v2 complete — best fold AUC = $V2_AUC"
copy_to_sed_improved "$V2_RESULT" "outputs/finetune-comp-v2" "ftcomp_v2"
export_onnx_for_tag "ftcomp_v2"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Generate updated pseudo labels from best v2 fold
# ══════════════════════════════════════════════════════════════════════════════
V2_BEST_CKPT=$(get_best_ckpt "$V2_RESULT" "outputs/finetune-comp-v2")
V3_PSEUDO="pseudo_labels/finetune_v2_ss_pseudo.csv"
if [ ! -f "$V3_PSEUDO" ] && [ -n "$V2_BEST_CKPT" ] && [ -f "$V2_BEST_CKPT" ]; then
    log "=== Generating v3 pseudo labels from $V2_BEST_CKPT ==="
    gen_pseudo_from_ckpt "$V2_BEST_CKPT" "$V3_PSEUDO"
else
    log "Using existing pseudo: $V3_PSEUDO (or v2 ckpt not found)"
    # Fall back to Perch pseudo if v2 not available
    [ ! -f "$V3_PSEUDO" ] && V3_PSEUDO="pseudo_labels/ns_r0_perch_aug.csv"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: finetune-comp-v3 (noisy student round 1, init from v2 best)
# ══════════════════════════════════════════════════════════════════════════════
V3_RESULT="outputs/finetune-comp-v3/result.json"
if [ ! -f "$V3_RESULT" ]; then
    log "=== Starting finetune-comp-v3 (noisy student r1, init from v2 best) ==="
    # Update config: competitor_ckpt → v2 best, perch_ss_csv → v3 pseudo
    python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('configs/finetune_comp_v3.yaml'))
cfg['data']['competitor_ckpt'] = '$V2_BEST_CKPT'
cfg['data']['perch_ss_csv']    = '$V3_PSEUDO'
yaml.dump(cfg, open('configs/finetune_comp_v3_run.yaml', 'w'), default_flow_style=False)
print('Config written: configs/finetune_comp_v3_run.yaml')
"
    python3 train_finetune_competitor.py \
        --config configs/finetune_comp_v3_run.yaml \
        --device "$DEVICE" \
        > "$LOG/finetune_v3_train.log" 2>&1
fi
V3_AUC=$(get_auc "$V3_RESULT")
log "v3 complete — best fold AUC = $V3_AUC"
copy_to_sed_improved "$V3_RESULT" "outputs/finetune-comp-v3" "ftcomp_v3"
export_onnx_for_tag "ftcomp_v3"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Generate updated pseudo labels from best v3 fold
# ══════════════════════════════════════════════════════════════════════════════
V3_BEST_CKPT=$(get_best_ckpt "$V3_RESULT" "outputs/finetune-comp-v3")
V4_PSEUDO="pseudo_labels/finetune_v3_ss_pseudo.csv"
if [ ! -f "$V4_PSEUDO" ] && [ -n "$V3_BEST_CKPT" ] && [ -f "$V3_BEST_CKPT" ]; then
    log "=== Generating v4 pseudo labels from $V3_BEST_CKPT ==="
    gen_pseudo_from_ckpt "$V3_BEST_CKPT" "$V4_PSEUDO"
else
    log "Using existing pseudo: $V4_PSEUDO"
    [ ! -f "$V4_PSEUDO" ] && V4_PSEUDO="$V3_PSEUDO"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: finetune-comp-v4 (noisy student round 2)
# ══════════════════════════════════════════════════════════════════════════════
V4_RESULT="outputs/finetune-comp-v4/result.json"
if [ ! -f "$V4_RESULT" ]; then
    log "=== Starting finetune-comp-v4 (noisy student r2, init from v3 best) ==="
    python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/finetune_comp_v4.yaml'))
cfg['data']['competitor_ckpt'] = '$V3_BEST_CKPT'
cfg['data']['perch_ss_csv']    = '$V4_PSEUDO'
yaml.dump(cfg, open('configs/finetune_comp_v4_run.yaml', 'w'), default_flow_style=False)
"
    python3 train_finetune_competitor.py \
        --config configs/finetune_comp_v4_run.yaml \
        --device "$DEVICE" \
        > "$LOG/finetune_v4_train.log" 2>&1
fi
V4_AUC=$(get_auc "$V4_RESULT")
log "v4 complete — best fold AUC = $V4_AUC"
copy_to_sed_improved "$V4_RESULT" "outputs/finetune-comp-v4" "ftcomp_v4"
export_onnx_for_tag "ftcomp_v4"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Model soup from all finetune folds >= threshold
# ══════════════════════════════════════════════════════════════════════════════
log "=== Building model soup from all fine-tuned folds ==="
python3 - <<'PYEOF' > "$LOG/auto_finetune_soup.log" 2>&1
import torch, json
from pathlib import Path

soup_files = sorted(Path('sed_improved').glob('ftcomp_v*_fold*_auc*.pt'))
print(f'Souping {len(soup_files)} checkpoints: {[f.name for f in soup_files]}')

if len(soup_files) >= 2:
    states = []
    for f in soup_files:
        ck = torch.load(str(f), map_location='cpu', weights_only=False)
        sd = ck.get('state_dict', ck.get('model_state_dict', ck))
        states.append(sd)
    avg = {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}
    torch.save({'model_state_dict': avg, 'soup_sources': [f.name for f in soup_files]},
               'sed_improved/ftcomp_soup_all.pt')
    print('Saved → sed_improved/ftcomp_soup_all.pt')
else:
    print('Not enough checkpoints for soup')
PYEOF
log "Soup done"
export_onnx_for_tag "ftcomp_soup"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Summary
# ══════════════════════════════════════════════════════════════════════════════
python3 - > "$SUMMARY" <<PYEOF
import json
from pathlib import Path

results = {}
for v, path in [('v1', 'outputs/finetune-comp-v1/result.json'),
                ('v2', 'outputs/finetune-comp-v2/result.json'),
                ('v3', 'outputs/finetune-comp-v3/result.json'),
                ('v4', 'outputs/finetune-comp-v4/result.json')]:
    try:    results[v] = json.load(open(path))
    except: results[v] = {'status': 'not_found'}

results['sed_improved_pt']   = [f.name for f in sorted(Path('sed_improved').glob('ftcomp_*.pt'))]
results['sed_improved_onnx'] = [f.name for f in sorted(Path('sed_improved').glob('*.onnx'))]
print(json.dumps(results, indent=2))
PYEOF

log "══════════════════════════════════════════"
log "AUTO FINETUNE PIPELINE COMPLETE"
log "  v1 AUC: $V1_AUC"
log "  v2 AUC: $V2_AUC"
log "  v3 AUC: $V3_AUC"
log "  v4 AUC: $V4_AUC"
log "  sed_improved/: $(ls sed_improved/ftcomp_*.pt 2>/dev/null | wc -l) .pt, $(ls sed_improved/*.onnx 2>/dev/null | wc -l) .onnx files"
log "  Summary: $SUMMARY"
log "══════════════════════════════════════════"
