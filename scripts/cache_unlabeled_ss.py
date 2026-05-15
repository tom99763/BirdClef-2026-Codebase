#!/usr/bin/env python3
"""One-time: sample N unlabeled soundscapes and cache as int16 .pt tensors.

Excludes the 66 GT-labeled soundscape files from selection.
Output: birdclef-2026/unlabeled_ss_cache/<file>.pt  +  unlabeled_ss_cache_meta.csv

Usage:
    python scripts/cache_unlabeled_ss.py --n 2000
"""
import sys, random, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import soundfile as sf

ROOT     = Path("/home/lab/BirdClef-2026-Codebase")
COMP_DIR = ROOT / "birdclef-2026"
SS_DIR   = COMP_DIR / "train_soundscapes"
SR       = 32000

ap = argparse.ArgumentParser()
ap.add_argument("--n",    type=int, default=2000, help="Number of unlabeled files to cache")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out",  type=str, default=str(COMP_DIR / "unlabeled_ss_cache"))
args = ap.parse_args()

CACHE_OUT = Path(args.out)
CACHE_OUT.mkdir(parents=True, exist_ok=True)

labeled_files = set(pd.read_csv(COMP_DIR / "train_soundscapes_labels.csv")["filename"].unique())

all_ss       = sorted(SS_DIR.glob("*.ogg"))
unlabeled_ss = [f for f in all_ss if f.name not in labeled_files]
print(f"Total soundscapes: {len(all_ss)}  Labeled: {len(labeled_files)}  Unlabeled: {len(unlabeled_ss)}")

random.seed(args.seed)
selected = sorted(random.sample(unlabeled_ss, min(args.n, len(unlabeled_ss))))
print(f"Caching {len(selected)} files → {CACHE_OUT}")

rows = []
for idx, f in enumerate(selected):
    cache_file = CACHE_OUT / (f.stem + ".pt")
    if not cache_file.exists():
        wav, sr = sf.read(f, dtype="float32")
        assert sr == SR, f"SR mismatch: {sr}"
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_i16 = np.clip(wav * 32767, -32768, 32767).astype(np.int16)
        torch.save(torch.from_numpy(wav_i16), cache_file)

    n_windows = 60 // 5
    for w in range(n_windows):
        rows.append({"filename": f.name, "start_sec": w * 5, "cache_file": cache_file.name})

    if (idx + 1) % 200 == 0:
        print(f"  {idx+1}/{len(selected)}")

meta_path = CACHE_OUT / "unlabeled_ss_cache_meta.csv"
pd.DataFrame(rows).to_csv(meta_path, index=False)
print(f"Done. {len(selected)} files, {len(rows)} windows → {meta_path}")
