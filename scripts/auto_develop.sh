#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# auto_develop.sh — Autonomous SED improvement pipeline (4-day run)
#
# Experiment chain:
#   distill-competitor-b0-v1  (already running in background)
#   → (if AUC > 0.9193) copy to sed_improved/
#   → distill-competitor-b0-v2  (with soundscape KD branch)
#   → distill-competitor-b0-v3  (temperature scaling + LLRD)
#   → model soup across all good folds
#   → eval summary
#
# Usage (run AFTER v1 finishes, OR launch standalone to chain from the start):
#   nohup bash scripts/auto_develop.sh > outputs/logs/auto_develop.log 2>&1 &
#
# Monitor:
#   tail -f outputs/logs/auto_develop.log
#   cat outputs/auto_develop_summary.json
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
LOG="outputs/logs"
SUMMARY="outputs/auto_develop_summary.json"
PASS_THRESHOLD=0.9193   # soundscape AUC threshold to copy to sed_improved/

mkdir -p "$LOG" outputs/competitor_pseudo sed_improved

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AUTO-DEV] $*" | tee -a "$LOG/auto_develop_main.log"; }

# ── Helper: get best_auc from result.json ─────────────────────────────────────
get_auc() {
    local result_json="$1"
    python3 -c "
import json, sys
try:
    d = json.load(open('$result_json'))
    aucs = [f['best_auc'] for f in d.get('folds', [])]
    print(f'{max(aucs):.4f}' if aucs else d.get('mean_fold_auc', '0'))
except: print('0')
" 2>/dev/null
}

# ── Helper: wait for a running job to finish ──────────────────────────────────
wait_for_npz() {
    local npz_path="$1"
    local desc="$2"
    log "Waiting for $desc …"
    while [ ! -f "$npz_path" ]; do
        sleep 60
    done
    log "$desc ready: $npz_path"
}

wait_for_result() {
    local result_json="$1"
    local desc="$2"
    log "Waiting for $desc to finish …"
    while [ ! -f "$result_json" ]; do
        sleep 120
    done
    # Wait until "finished" or no process writing to log
    sleep 60
    log "$desc finished: auc=$(get_auc $result_json)"
}

# ── Helper: copy best folds to sed_improved ───────────────────────────────────
copy_to_sed_improved() {
    local exp_dir="$1"
    local exp_name="$2"
    local result_json="${exp_dir}/result.json"

    python3 - <<PYEOF
import json, shutil, os
from pathlib import Path

result = json.load(open('$result_json'))
exp_dir = Path('$exp_dir')
out_dir = Path('sed_improved')
exp_name = '$exp_name'
threshold = $PASS_THRESHOLD

copied = []
for fold_info in result.get('folds', []):
    fold = fold_info['fold']
    auc  = fold_info['best_auc']
    src  = exp_dir / f'fold{fold}_best.pt'
    if src.exists() and auc >= threshold:
        dst = out_dir / f'{exp_name}_fold{fold}_auc{auc:.4f}.pt'
        shutil.copy2(str(src), str(dst))
        copied.append(f'  fold{fold} auc={auc:.4f} → {dst.name}')
        print(f'Copied: {dst}')

if not copied:
    # Copy best fold regardless of threshold for analysis
    best = max(result.get('folds', []), key=lambda x: x['best_auc'], default=None)
    if best:
        src = exp_dir / f"fold{best['fold']}_best.pt"
        dst = out_dir / f"{exp_name}_fold{best['fold']}_auc{best['best_auc']:.4f}_BELOW_THR.pt"
        if src.exists():
            shutil.copy2(str(src), str(dst))
            print(f'Below threshold but copied best: {dst}')
print('Done')
PYEOF
}

# ── Step 0: Ensure competitor pseudo labels exist (wait if needed) ─────────────
PSEUDO_NPZ="outputs/competitor_pseudo/train_audio_probs.npz"
wait_for_npz "$PSEUDO_NPZ" "competitor train_audio pseudo labels"

# ── Step 1: Wait for distill-competitor-b0-v1 ─────────────────────────────────
V1_RESULT="outputs/distill-competitor-b0-v1/result.json"
if [ ! -f "$V1_RESULT" ]; then
    log "v1 not yet started — launching …"
    python3 train_distill_competitor.py \
        --config configs/distill_competitor_b0_v1.yaml \
        --device "$DEVICE" \
        > "${LOG}/distill_competitor_v1_train.log" 2>&1
else
    wait_for_result "$V1_RESULT" "distill-competitor-b0-v1"
fi

V1_AUC=$(get_auc "$V1_RESULT")
log "v1 complete — best fold AUC = $V1_AUC"
copy_to_sed_improved "outputs/distill-competitor-b0-v1" "distill_v1"

# ── Step 2: Generate soundscape pseudo labels for v2 ──────────────────────────
SS_NPZ="outputs/competitor_pseudo/soundscape_probs.npz"
if [ ! -f "$SS_NPZ" ]; then
    log "Generating competitor pseudo labels on soundscapes …"
    python3 - <<'PYEOF' > "${LOG}/gen_competitor_ss_pseudo.log" 2>&1
"""Generate competitor SED predictions on all train_soundscapes."""
import os, sys, torch, numpy as np, soundfile as sf, librosa
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, '.')
import pandas as pd

from scripts.gen_competitor_pseudo import SEDModel, MelTransform, CLIP_SAMPLES, SR

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
taxonomy = pd.read_csv('birdclef-2026/taxonomy.csv')
num_classes = len(taxonomy)

model = SEDModel(num_classes=num_classes).to(device)
ckpt  = torch.load(
    'birdclef-2026/notebook resource/current_subs/weights/competitor_sed_fold0.pt',
    map_location='cpu', weights_only=False
)
model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
model.eval()
mel_tf = MelTransform().to(device)

ss_dir  = Path('birdclef-2026/train_soundscapes')
ogg_files = sorted(ss_dir.glob('*.ogg'))
print(f'Soundscapes: {len(ogg_files)}')

STRIDE = SR * 5
out_rids, out_probs = [], []
for ogg in tqdm(ogg_files, desc='SS inference'):
    try:
        audio, sr = sf.read(str(ogg), dtype='float32', always_2d=False)
        if audio.ndim == 2: audio = audio.mean(axis=1)
        if sr != SR: audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    except: continue
    n = len(audio) // STRIDE
    for i in range(n):
        clip = audio[i*STRIDE:(i+1)*STRIDE]
        if len(clip) < CLIP_SAMPLES: clip = np.pad(clip,(0,CLIP_SAMPLES-len(clip)))
        m = np.abs(clip).max()
        if m > 1e-8: clip = clip / m
        wav  = torch.from_numpy(clip[None]).to(device)
        with torch.no_grad():
            mel  = mel_tf(wav)
            prob = model(mel).cpu().numpy()[0]
        out_rids.append(f'{ogg.stem}_{(i+1)*5}')
        out_probs.append(prob)

probs_arr = np.stack(out_probs, axis=0).astype(np.float32)
np.savez_compressed('outputs/competitor_pseudo/soundscape_probs.npz',
                    row_ids=np.array(out_rids), probs=probs_arr)
print(f'Saved {len(out_rids)} rows → outputs/competitor_pseudo/soundscape_probs.npz')
PYEOF
    log "Soundscape pseudo labels generated: $SS_NPZ"
fi

# ── Step 3: Train distill-competitor-b0-v2 ────────────────────────────────────
V2_RESULT="outputs/distill-competitor-b0-v2/result.json"
if [ ! -f "$V2_RESULT" ]; then
    log "Starting distill-competitor-b0-v2 (with soundscape KD) …"
    python3 train_distill_competitor.py \
        --config configs/distill_competitor_b0_v2.yaml \
        --device "$DEVICE" \
        > "${LOG}/distill_competitor_v2_train.log" 2>&1
fi
V2_AUC=$(get_auc "$V2_RESULT")
log "v2 complete — best fold AUC = $V2_AUC"
copy_to_sed_improved "outputs/distill-competitor-b0-v2" "distill_v2"

# ── Step 4: Train distill-competitor-b0-v3 ────────────────────────────────────
V3_RESULT="outputs/distill-competitor-b0-v3/result.json"
if [ ! -f "$V3_RESULT" ]; then
    log "Starting distill-competitor-b0-v3 (temperature KD + LLRD) …"
    python3 train_distill_competitor.py \
        --config configs/distill_competitor_b0_v3.yaml \
        --device "$DEVICE" \
        > "${LOG}/distill_competitor_v3_train.log" 2>&1
fi
V3_AUC=$(get_auc "$V3_RESULT")
log "v3 complete — best fold AUC = $V3_AUC"
copy_to_sed_improved "outputs/distill-competitor-b0-v3" "distill_v3"

# ── Step 5: Mean Teacher Semi-Supervised Learning ─────────────────────────────
MT_RESULT="outputs/ssl-mean-teacher-b0-v1/result.json"
if [ ! -f "$MT_RESULT" ]; then
    log "Starting ssl-mean-teacher-b0-v1 (EMA teacher, labeled+soundscape SSL) …"
    python3 train_ssl_mean_teacher.py \
        --config configs/ssl_mean_teacher_b0_v1.yaml \
        --device "$DEVICE" \
        > "${LOG}/ssl_mean_teacher_v1_train.log" 2>&1
fi
MT_AUC=$(get_auc "$MT_RESULT")
log "Mean Teacher complete — best fold AUC = $MT_AUC"
copy_to_sed_improved "outputs/ssl-mean-teacher-b0-v1" "ssl_mt_v1"

# ── Step 6: Model soup ─────────────────────────────────────────────────────────
log "Building model soup from all distilled folds …"
python3 - <<'PYEOF' > "${LOG}/auto_develop_soup.log" 2>&1
import torch, os
from pathlib import Path

soup_files = sorted(Path('sed_improved').glob('distill_v*_fold*_auc*.pt'))
soup_files = [f for f in soup_files if 'BELOW_THR' not in f.name]
print(f'Souping {len(soup_files)} checkpoints: {[f.name for f in soup_files]}')

if len(soup_files) < 2:
    print('Not enough checkpoints for soup'); exit(0)

states = []
for f in soup_files:
    ckpt = torch.load(str(f), map_location='cpu', weights_only=False)
    sd   = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    states.append(sd)

# Average all state dicts
avg_state = {k: sum(s[k].float() for s in states) / len(states)
             for k in states[0].keys()}

torch.save({'model_state_dict': avg_state,
            'soup_sources': [f.name for f in soup_files]},
           'sed_improved/distill_soup_all.pt')
print('Saved → sed_improved/distill_soup_all.pt')
PYEOF
log "Soup done → sed_improved/distill_soup_all.pt"

# ── Step 7: Summary ───────────────────────────────────────────────────────────
python3 - > "$SUMMARY" <<PYEOF
import json
from pathlib import Path

summary = {
    'distill_v1': {},
    'distill_v2': {},
    'distill_v3': {},
    'sed_improved_files': [],
}

for v, path in [('distill_v1','outputs/distill-competitor-b0-v1/result.json'),
                ('distill_v2','outputs/distill-competitor-b0-v2/result.json'),
                ('distill_v3','outputs/distill-competitor-b0-v3/result.json'),
                ('ssl_mt_v1', 'outputs/ssl-mean-teacher-b0-v1/result.json')]:
    try:
        d = json.load(open(path))
        summary[v] = d
    except:
        summary[v] = {'status': 'not_found'}

summary['sed_improved_files'] = [f.name for f in sorted(Path('sed_improved').glob('*.pt'))]
print(json.dumps(summary, indent=2))
PYEOF

log "══════════════════════════════════════════"
log "AUTO-DEV PIPELINE COMPLETE"
log "Summary: $SUMMARY"
log "sed_improved/: $(ls sed_improved/*.pt 2>/dev/null | wc -l) .pt files"
log "══════════════════════════════════════════"
