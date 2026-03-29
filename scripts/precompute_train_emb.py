#!/usr/bin/env python3
"""
precompute_train_emb.py
────────────────────────────────────────────────────────────────────────────────
Run all 35549 train_audio clips through the frozen Perch backbone once.
Saves embeddings + labels to outputs/perch_train_emb.npz

Usage:
  /home/lab/miniconda3/envs/perch_ft/bin/python scripts/precompute_train_emb.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, '/home/lab/BirdClef-2026-Codebase')

import pickle, time
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial

import jax, jax.numpy as jnp
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from chirp.models import efficientnet, frontend as frontend_
from flax import linen as nn
import soundfile as sf
import librosa

ROOT        = Path('birdclef-2026')
OUT         = Path('outputs/perch_train_emb.npz')
WEIGHTS     = Path('weights/perch_jax_backbone/perch_backbone_params.pkl')
SAMPLE_RATE = 32000
N_SAMPLES   = 160000  # 5s @ 32kHz
BATCH_SIZE  = 64

print(f'JAX devices: {jax.devices()}')

# ── Load backbone ──────────────────────────────────────────────────────────────
print('Loading backbone...')
with open(WEIGHTS, 'rb') as f:
    backbone_vars = pickle.load(f)

class FixedEM(nn.Module):
    frontend: frontend_.MelSpectrogram = frontend_.MelSpectrogram(
        features=128, stride=320, kernel_size=640, sample_rate=32000,
        freq_range=(60, 16000), power=1.0,
        scaling_config=frontend_.LogScalingConfig(), nfft=1024)
    backbone: nn.Module = efficientnet.EfficientNet(
        efficientnet.EfficientNetModel.B3, include_top=False)

    @nn.compact
    def __call__(self, x, train=False):
        s = self.frontend._magnitude_scale(self.frontend(x))
        return jnp.mean(self.backbone(jnp.expand_dims(s, -1), train=train), axis=(-2, -3))

backbone = FixedEM()

@jax.jit
def embed(x):
    return backbone.apply(backbone_vars, x, train=False, mutable=False)

# Warmup JIT
print('JIT warmup...')
_ = embed(jnp.zeros((2, N_SAMPLES)))
print('  done')

# ── Load taxonomy ──────────────────────────────────────────────────────────────
taxonomy = pd.read_csv(ROOT / 'taxonomy.csv')
SPECIES  = taxonomy['primary_label'].tolist()
N_CLS    = len(SPECIES)
sp2idx   = {s: i for i, s in enumerate(SPECIES)}

# ── Load train.csv ─────────────────────────────────────────────────────────────
train_df = pd.read_csv(ROOT / 'train.csv')
n_total  = len(train_df)
print(f'Total train clips: {n_total}')

# ── Audio loader ───────────────────────────────────────────────────────────────
def load_clip(path, n_samples=N_SAMPLES):
    try:
        audio, sr = sf.read(str(path), dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        if len(audio) < n_samples:
            audio = np.tile(audio, int(np.ceil(n_samples / len(audio))))
        return audio[:n_samples].astype(np.float32)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)

# ── Precompute ─────────────────────────────────────────────────────────────────
all_emb    = np.zeros((n_total, 1536), dtype=np.float32)
all_labels = np.zeros((n_total, N_CLS), dtype=np.float32)
filenames  = []

t0 = time.time()
for batch_start in range(0, n_total, BATCH_SIZE):
    batch_rows = train_df.iloc[batch_start:batch_start + BATCH_SIZE]
    audios = []
    for _, row in batch_rows.iterrows():
        path = ROOT / 'train_audio' / row['filename']
        audios.append(load_clip(path))
        sp = str(row['primary_label'])
        i = batch_start + len(audios) - 1
        if sp in sp2idx:
            all_labels[i, sp2idx[sp]] = 1.0
        filenames.append(row['filename'])

    batch_audio = jnp.array(np.stack(audios))
    batch_emb   = embed(batch_audio)
    end = min(batch_start + BATCH_SIZE, n_total)
    all_emb[batch_start:end] = np.array(batch_emb)

    if (batch_start // BATCH_SIZE + 1) % 20 == 0:
        done = batch_start + len(audios)
        elapsed = time.time() - t0
        eta = elapsed / done * (n_total - done)
        print(f'  {done}/{n_total}  ({elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m)', flush=True)

elapsed = time.time() - t0
print(f'\nDone! {n_total} clips in {elapsed/60:.1f} min')

np.savez(OUT,
         emb=all_emb,
         labels=all_labels,
         filenames=np.array(filenames))
print(f'Saved → {OUT}  ({all_emb.shape})')
