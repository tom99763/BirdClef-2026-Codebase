#!/usr/bin/env python3
"""
train_perch_lora.py
────────────────────────────────────────────────────────────────────────────────
Perch Backbone Fine-tuning with Embedding-Space LoRA Adapter
GPU1 only (CUDA_VISIBLE_DEVICES=1)

Architecture:
  Frozen Perch EfficientNet-B3 backbone → 1536-dim embedding
  → Low-rank adapter: emb' = emb + (emb @ A @ B) * (alpha/rank)
  → LayerNorm → Dropout → 234-class linear head

Training:
  Data: BirdCLEF 2026 train.csv (35549 clips, 234 classes)
  Loss: Focal BCE (gamma=2.0, pos_weight=10.0)
  Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  Schedule: Cosine decay with 500-step warmup
  Epochs: 30

Eval: Pre-computed Perch embeddings → adapter → head (fast, no I/O)
      Metric: macro ROC-AUC on 75 valid soundscape classes
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU1 busy with SED NS training
import sys
sys.path.insert(0, '/home/lab/BirdClef-2026-Codebase')

import pickle, time, json
import numpy as np
import pandas as pd
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import soundfile as sf
import librosa
from sklearn.metrics import roc_auc_score
from functools import partial

# ── Config ────────────────────────────────────────────────────────────────────

ROOT        = Path('birdclef-2026')
WEIGHTS_DIR = Path('weights/perch_jax_backbone')
OUT_DIR     = Path('outputs/perch-lora-ft')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 32000
CLIP_SAMPLES = 160000  # 5 seconds at 32kHz
BATCH_SIZE  = 32
N_EPOCHS    = 30
WARMUP_STEPS = 500
LR          = 1e-3
WD          = 1e-4
RANK        = 8
ALPHA       = 16.0
DROPOUT     = 0.2
GAMMA_FOCAL = 2.0
POS_WEIGHT  = 10.0
SEED        = 42

print(f'JAX devices: {jax.devices()}')
print(f'Output dir: {OUT_DIR}')

# ── Load backbone ─────────────────────────────────────────────────────────────

print('\n[1] Loading Perch backbone...')
with open(WEIGHTS_DIR / 'perch_backbone_params.pkl', 'rb') as f:
    backbone_vars = pickle.load(f)  # {'params': ..., 'batch_stats': ...}

# Rebuild FixedEM model for backbone inference
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from chirp.models import efficientnet, frontend as frontend_
from flax import linen as flax_nn

class FixedEM(flax_nn.Module):
    frontend: frontend_.MelSpectrogram = frontend_.MelSpectrogram(
        features=128, stride=320, kernel_size=640, sample_rate=32000,
        freq_range=(60, 16000), power=1.0,
        scaling_config=frontend_.LogScalingConfig(), nfft=1024)
    backbone: flax_nn.Module = efficientnet.EfficientNet(
        efficientnet.EfficientNetModel.B3, include_top=False)

    @flax_nn.compact
    def __call__(self, x, train=False):
        s = self.frontend._magnitude_scale(self.frontend(x))
        feat = self.backbone(jnp.expand_dims(s, -1), train=train)
        return jnp.mean(feat, axis=(-2, -3))  # (B, 1536)

backbone_model = FixedEM()
print(f'    Backbone vars: params={len(jax.tree_util.tree_leaves(backbone_vars["params"]))} leaves')

# JIT-compiled backbone inference (frozen)
@partial(jax.jit, static_argnames=['train'])
def backbone_forward(x, train=False):
    return backbone_model.apply(backbone_vars, x, train=train, mutable=False)

# Verify backbone
dummy = jnp.zeros((2, CLIP_SAMPLES))
emb_test = backbone_forward(dummy)
print(f'    Backbone output: {emb_test.shape}, mean={float(emb_test.mean()):.4f}')

# ── Head model ────────────────────────────────────────────────────────────────

taxonomy = pd.read_csv(ROOT / 'taxonomy.csv')
N_CLASSES = len(taxonomy)
SPECIES   = taxonomy['primary_label'].tolist()
sp2idx    = {s: i for i, s in enumerate(SPECIES)}
print(f'\n[2] Classes: {N_CLASSES}')

class LoRAHead(nn.Module):
    """Low-rank embedding adapter + classification head."""
    n_classes: int = 234
    rank: int = 8
    alpha: float = 16.0
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, emb, train=False):
        # LoRA in embedding space: emb + emb @ A @ B * scale
        scale = self.alpha / self.rank
        A = self.param('lora_A', nn.initializers.normal(0.01), (emb.shape[-1], self.rank))
        B = self.param('lora_B', nn.initializers.zeros, (self.rank, emb.shape[-1]))
        x = emb + scale * (emb @ A @ B)
        # Normalize
        x = nn.LayerNorm()(x)
        # Dropout
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        # Head
        logits = nn.Dense(self.n_classes,
                          kernel_init=nn.initializers.normal(0.02))(x)
        return logits

head_model = LoRAHead(n_classes=N_CLASSES, rank=RANK, alpha=ALPHA, dropout_rate=DROPOUT)
rng = jax.random.PRNGKey(SEED)
dummy_emb = jnp.zeros((2, 1536))
head_vars = head_model.init({'params': rng, 'dropout': jax.random.PRNGKey(1)},
                            dummy_emb, train=True)
head_params = head_vars['params']

n_head_params = sum(v.size for v in jax.tree_util.tree_leaves(head_params))
lora_a = head_params['lora_A']
lora_b = head_params['lora_B']
print(f'    Head params: {n_head_params:,}')
print(f'    LoRA A: {lora_a.shape}, B: {lora_b.shape}  (rank={RANK})')

# ── Loss ──────────────────────────────────────────────────────────────────────

def focal_bce(logits, labels, gamma=2.0, pos_weight=10.0):
    """Focal BCE loss."""
    prob   = jax.nn.sigmoid(logits)
    bce_p  = -labels * jnp.log(prob + 1e-7)
    bce_n  = -(1 - labels) * jnp.log(1 - prob + 1e-7)
    fl_p   = ((1 - prob) ** gamma) * bce_p * pos_weight
    fl_n   = (prob ** gamma) * bce_n
    return jnp.mean(fl_p + fl_n)

# ── Optimizer ─────────────────────────────────────────────────────────────────

def make_optimizer(n_train_steps):
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LR,
        warmup_steps=WARMUP_STEPS,
        decay_steps=n_train_steps,
        end_value=LR * 0.01,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(sched, weight_decay=WD),
    )

# ── Data loading ──────────────────────────────────────────────────────────────

def load_audio_clip(path: Path, target_sr=32000, n_samples=160000) -> np.ndarray:
    """Load audio, resample to target_sr, return fixed-length clip."""
    try:
        audio, sr = sf.read(str(path), dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        # Pad or trim to n_samples
        if len(audio) < n_samples:
            n_repeats = int(np.ceil(n_samples / len(audio)))
            audio = np.tile(audio, n_repeats)
        audio = audio[:n_samples]
        return audio.astype(np.float32)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)

def build_dataset(df: pd.DataFrame):
    """Build list of (audio_path, label_vector) from dataframe."""
    samples = []
    for _, row in df.iterrows():
        path = ROOT / 'train_audio' / row['filename']
        label = np.zeros(N_CLASSES, dtype=np.float32)
        sp = str(row['primary_label'])
        if sp in sp2idx:
            label[sp2idx[sp]] = 1.0
        samples.append((path, label))
    return samples

def batch_iter(samples, batch_size, shuffle=True, rng_seed=0):
    """Iterate batches. Loads audio on-the-fly."""
    indices = np.arange(len(samples))
    if shuffle:
        rng_np = np.random.RandomState(rng_seed)
        rng_np.shuffle(indices)
    for start in range(0, len(indices) - batch_size + 1, batch_size):
        batch_idx = indices[start:start + batch_size]
        audios = np.stack([load_audio_clip(samples[i][0]) for i in batch_idx])
        labels = np.stack([samples[i][1] for i in batch_idx])
        yield audios, labels

# ── Train/Eval step ───────────────────────────────────────────────────────────

@jax.jit
def train_step(head_params, opt_state, emb, labels, rng):
    def loss_fn(params):
        logits = head_model.apply(
            {'params': params}, emb, train=True,
            rngs={'dropout': rng}
        )
        return focal_bce(logits, labels)

    loss, grads = jax.value_and_grad(loss_fn)(head_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, head_params)
    new_params = optax.apply_updates(head_params, updates)
    return new_params, new_opt_state, loss

@jax.jit
def eval_step(head_params, emb):
    return head_model.apply({'params': head_params}, emb, train=False)

# ── Evaluation on labeled soundscapes ─────────────────────────────────────────

print('\n[3] Loading labeled soundscapes for eval...')
ss_data    = np.load('outputs/perch_labeled_ss.npz')
ss_emb     = jnp.array(ss_data['emb'],    dtype=jnp.float32)   # (739, 1536)
ss_labels  = ss_data['labels'].astype(np.float32)               # (739, 234)
valid_cls  = ss_labels.sum(0) > 0                               # 75 valid classes
print(f'    Windows: {ss_emb.shape[0]}, valid classes: {valid_cls.sum()}')

def eval_auc(head_params):
    logits = eval_step(head_params, ss_emb)
    probs  = jax.nn.sigmoid(logits)
    probs_np = np.array(probs)
    return roc_auc_score(ss_labels[:, valid_cls], probs_np[:, valid_cls], average='macro')

# ── Load train data ───────────────────────────────────────────────────────────

print('\n[4] Building train dataset...')
train_df = pd.read_csv(ROOT / 'train.csv')
train_samples = build_dataset(train_df)
n_train = len(train_samples)
n_steps_per_epoch = n_train // BATCH_SIZE
n_total_steps = N_EPOCHS * n_steps_per_epoch
print(f'    Train samples: {n_train}')
print(f'    Steps per epoch: {n_steps_per_epoch}, total: {n_total_steps}')

# ── Init optimizer ────────────────────────────────────────────────────────────

optimizer = make_optimizer(n_total_steps)
opt_state = optimizer.init(head_params)

# Baseline eval (head with random weights, frozen backbone)
print('\n[5] Baseline evaluation (random head)...')
auc_init = eval_auc(head_params)
print(f'    Baseline AUC = {auc_init:.4f}')

# ── Training loop ─────────────────────────────────────────────────────────────

print('\n[6] Training...')
best_auc   = 0.0
best_params = head_params
history    = []
rng_key    = jax.random.PRNGKey(SEED)

for epoch in range(1, N_EPOCHS + 1):
    ep_start = time.time()
    losses   = []

    for step, (audios, labels) in enumerate(
            batch_iter(train_samples, BATCH_SIZE, shuffle=True, rng_seed=epoch)):
        # Backbone forward (frozen, JIT)
        emb_batch = backbone_forward(jnp.array(audios))

        # Head forward + backward
        rng_key, step_rng = jax.random.split(rng_key)
        head_params, opt_state, loss = train_step(
            head_params, opt_state,
            emb_batch, jnp.array(labels),
            step_rng
        )
        losses.append(float(loss))

        if (step + 1) % 50 == 0:
            print(f'  Ep{epoch:02d} step {step+1}/{n_steps_per_epoch} '
                  f'loss={np.mean(losses[-50:]):.4f}', flush=True)

    # Epoch eval
    auc = eval_auc(head_params)
    ep_time = time.time() - ep_start
    mean_loss = np.mean(losses)
    print(f'Epoch {epoch:02d}/{N_EPOCHS}  loss={mean_loss:.4f}  '
          f'AUC={auc:.4f}  ({ep_time:.0f}s)', flush=True)

    history.append({'epoch': epoch, 'loss': float(mean_loss), 'auc': float(auc)})

    if auc > best_auc:
        best_auc = auc
        best_params = jax.tree_util.tree_map(lambda x: np.array(x), head_params)
        with open(OUT_DIR / 'best_head_params.pkl', 'wb') as f:
            pickle.dump(best_params, f)
        print(f'  ★ New best AUC = {best_auc:.4f}  (saved)', flush=True)

    # Save history
    with open(OUT_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

print(f'\n✅ Done! Best AUC = {best_auc:.4f}')
print(f'   Head params saved → {OUT_DIR}/best_head_params.pkl')
print(f'   History → {OUT_DIR}/history.json')
