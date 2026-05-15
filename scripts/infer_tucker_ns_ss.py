#!/usr/bin/env python3
"""5s sliding-window PyTorch inference on cached soundscapes — matches bc2026-distilled-sed.ipynb.

Works for both labeled and unlabeled soundscape caches.
No post-processing (no Gaussian smoothing).
Output: filename, start_sec, <234 species...>  (soft probs, 5-fold mean sigmoid)
"""
import os, sys, argparse, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from train_tucker_sed import BirdSEDModel, MelSpecTransform, BACKBONE_NAME, DROP_PATH_RATE, SR

ROOT      = Path("/home/lab/BirdClef-2026-Codebase")
COMP_DIR  = ROOT / "birdclef-2026"
CLIP_5S   = 5 * SR   # 160000 samples per 5s window

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_dir",   required=True, help="dir with fold{k}_best_ns22.pt")
ap.add_argument("--out",        required=True, help="output CSV path")
ap.add_argument("--gpu",        type=int, default=0)
ap.add_argument("--backbone",   type=str, default=None)
ap.add_argument("--batch",      type=int, default=32, help="5s chunks per forward pass")
ap.add_argument("--cache_dir",  type=str, default=None,
                help="soundscape cache dir (default: birdclef-2026/waveform_cache)")
ap.add_argument("--cache_meta", type=str, default=None,
                help="cache meta CSV (default: <cache_dir>/soundscape_cache_meta.csv)")
args = ap.parse_args()

DEFAULT_CACHE = ROOT / "birdclef-2026/waveform_cache"
CACHE_DIR  = Path(args.cache_dir)  if args.cache_dir  else DEFAULT_CACHE
META_CSV   = Path(args.cache_meta) if args.cache_meta else CACHE_DIR / "soundscape_cache_meta.csv"

device   = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
backbone = args.backbone or BACKBONE_NAME

sample_sub     = pd.read_csv(COMP_DIR / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
NUM_CLASSES    = len(PRIMARY_LABELS)
print(f"Species: {NUM_CLASSES}  device: {device}  cache: {CACHE_DIR}")

# Load fold checkpoints
ckpt_dir   = Path(args.ckpt_dir)
ckpt_files = sorted(ckpt_dir.glob("fold*_best_ns22.pt"))
assert len(ckpt_files) > 0, f"No fold*_best_ns22.pt found in {ckpt_dir}"
print(f"Loading {len(ckpt_files)} folds from {ckpt_dir}")

mel_tf = MelSpecTransform().to(device)

def load_model(ckpt_path):
    m = BirdSEDModel(backbone_name=backbone, drop_path_rate=DROP_PATH_RATE).to(device)
    raw   = torch.load(ckpt_path, map_location=device)
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    m.load_state_dict(state, strict=False)
    m.eval()
    return m

@torch.no_grad()
def infer_batch(model, chunks_np):
    """chunks_np: (B, CLIP_5S) float32 numpy → (B, NUM_CLASSES) blend probs (Tucker-exact).
    Tucker notebook: 0.5*sigmoid(clip) + 0.5*sigmoid(fmax), then average probs across folds."""
    wav_t = torch.from_numpy(chunks_np).to(device).unsqueeze(1)  # (B, 1, samples)
    mel   = mel_tf(wav_t)
    for i in range(mel.size(0)):
        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
    clip_logits, framewise = model(mel, return_framewise=True)
    fmax_logits  = framewise.max(dim=1).values
    p_clip  = torch.sigmoid(clip_logits).float().cpu().numpy()
    p_fmax  = torch.sigmoid(fmax_logits).float().cpu().numpy()
    p_blend = 0.5 * p_clip + 0.5 * p_fmax
    return np.nan_to_num(p_blend, nan=0.0)

# Load soundscape cache meta — rows are (filename, start_sec) 5s windows
sc_meta = pd.read_csv(META_CSV)
sc_meta["start_sec"] = sc_meta["start_sec"].astype(int)
print(f"Soundscape windows: {len(sc_meta)}")

# Extract 5s chunks from cached 60s waveforms
sc_files = {}   # filename → float32 numpy array (full 60s waveform)
for fn in sc_meta["filename"].unique():
    rows = sc_meta[sc_meta["filename"] == fn]
    cf   = CACHE_DIR / rows.iloc[0]["cache_file"]
    if cf.exists():
        wav = torch.load(cf, map_location="cpu", weights_only=True).float().div(32767.0).numpy()
        sc_files[fn] = wav

chunks, valid_idx = [], []
for i, row in sc_meta.iterrows():
    fn  = row["filename"]
    if fn not in sc_files:
        continue
    wav         = sc_files[fn]
    start_s     = int(row["start_sec"])
    start_samp  = start_s * SR
    end_samp    = start_samp + CLIP_5S
    if end_samp <= len(wav):
        chunk = wav[start_samp:end_samp]
    else:
        chunk = np.zeros(CLIP_5S, dtype=np.float32)
        available = len(wav) - start_samp
        if available > 0:
            chunk[:available] = wav[start_samp:]
    chunks.append(chunk.astype(np.float32))
    valid_idx.append(i)

print(f"Prepared {len(chunks)}/{len(sc_meta)} 5s chunks")

# 5-fold ensemble: average blended probs across folds (Tucker-exact)
all_probs = np.zeros((len(chunks), NUM_CLASSES), dtype=np.float32)
chunks_arr = np.stack(chunks)   # (N, CLIP_5S)

for ckpt_path in ckpt_files:
    print(f"  Fold: {ckpt_path.name}")
    model = load_model(ckpt_path)
    fold_probs = np.zeros_like(all_probs)
    for start in range(0, len(chunks), args.batch):
        batch = chunks_arr[start:start + args.batch]
        fold_probs[start:start + len(batch)] = infer_batch(model, batch)
    all_probs += fold_probs
    del model; torch.cuda.empty_cache(); gc.collect()

all_probs /= len(ckpt_files)
all_probs = np.nan_to_num(all_probs, nan=0.0)
print(f"Prob stats: min={all_probs.min():.4f} max={all_probs.max():.4f} "
      f"nan={np.isnan(all_probs).sum()}")

# Build output — one row per (filename, start_sec), matching sc_cache_meta
rows = []
for j, sc_i in enumerate(valid_idx):
    row = sc_meta.iloc[sc_i]
    rec = {"filename": row["filename"], "start_sec": int(row["start_sec"])}
    for k, lbl in enumerate(PRIMARY_LABELS):
        rec[lbl] = float(all_probs[j, k])
    rows.append(rec)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"Saved {len(rows)} rows → {out_path}")
