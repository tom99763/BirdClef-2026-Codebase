#!/usr/bin/env python3
"""
train_perch_mlp_head.py
────────────────────────────────────────────────────────────────────────────────
Train a 2-layer MLP head on pre-computed Perch embeddings.
Uses all 35549 train_audio clips (vs 739 soundscape windows before).

Architecture:
  Pre-computed 1536-dim embedding
  → Linear(1536→512) → GELU → Dropout(0.3)
  → Linear(512→234)

Eval: Labeled soundscapes (739 windows, macro AUC, 75 valid classes)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys, json, pickle, time
sys.path.insert(0, '/home/lab/BirdClef-2026-Codebase')

import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT_DIR  = Path('outputs/perch-mlp-head')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPOCHS    = 60
BATCH_SIZE  = 512
LR          = 3e-4
WD          = 1e-3
DROPOUT     = 0.3
HIDDEN      = 512
SEED        = 42

print(f'JAX: {jax.__version__}  devices: {jax.devices()}')

# ── Load pre-computed embeddings ───────────────────────────────────────────────
print('\n[1] Loading pre-computed train embeddings...')
d = np.load('outputs/perch_train_emb.npz')
train_emb    = d['emb'].astype(np.float32)       # (35549, 1536)
train_labels = d['labels'].astype(np.float32)    # (35549, 234)
print(f'    Train: {train_emb.shape}  pos_rate={train_labels.mean():.4f}')

print('\n[2] Loading labeled soundscape eval...')
ss = np.load('outputs/perch_labeled_ss.npz')
ss_emb    = jnp.array(ss['emb'], dtype=jnp.float32)    # (739, 1536)
ss_labels = ss['labels'].astype(np.float32)             # (739, 234)
valid_cls  = ss_labels.sum(0) > 0                        # 75 valid
print(f'    Eval windows: {ss_emb.shape[0]}, valid classes: {valid_cls.sum()}')

# ── Model ─────────────────────────────────────────────────────────────────────
class MLPHead(nn.Module):
    n_classes: int = 234
    hidden:    int = 512
    dropout:   float = 0.3

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Dense(self.hidden)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.n_classes)(x)

model = MLPHead(n_classes=234, hidden=HIDDEN, dropout=DROPOUT)
rng   = jax.random.PRNGKey(SEED)
params = model.init(
    {'params': rng, 'dropout': jax.random.PRNGKey(1)},
    jnp.zeros((2, 1536)), train=True
)['params']

n_params = sum(v.size for v in jax.tree_util.tree_leaves(params))
print(f'\n[3] Model: {n_params:,} params  (Linear 1536→{HIDDEN}→234)')

# ── Loss ──────────────────────────────────────────────────────────────────────
def focal_bce(logits, labels, gamma=2.0, pos_weight=8.0):
    p   = jax.nn.sigmoid(logits)
    bce = -(labels * jnp.log(p + 1e-7) * pos_weight * ((1-p)**gamma)
           + (1-labels) * jnp.log(1-p + 1e-7) * (p**gamma))
    return bce.mean()

# ── Optimizer ─────────────────────────────────────────────────────────────────
n_steps = N_EPOCHS * (len(train_emb) // BATCH_SIZE)
sched = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=LR,
    warmup_steps=200, decay_steps=n_steps, end_value=LR * 0.01)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(sched, weight_decay=WD))
opt_state = optimizer.init(params)

# ── Train / Eval steps ────────────────────────────────────────────────────────
@jax.jit
def train_step(params, opt_state, emb, labels, rng):
    def loss_fn(p):
        logits = model.apply({'params': p}, emb, train=True, rngs={'dropout': rng})
        return focal_bce(logits, labels)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_opt, loss

@jax.jit
def eval_logits(params, emb):
    return model.apply({'params': params}, emb, train=False)

def eval_auc(params):
    logits = eval_logits(params, ss_emb)
    probs  = np.array(jax.nn.sigmoid(logits))
    return roc_auc_score(ss_labels[:, valid_cls], probs[:, valid_cls], average='macro')

# ── Baseline ──────────────────────────────────────────────────────────────────
print(f'\n[4] Baseline (random params): AUC = {eval_auc(params):.4f}')

# ── Training ──────────────────────────────────────────────────────────────────
print(f'\n[5] Training {N_EPOCHS} epochs, {len(train_emb)//BATCH_SIZE} steps/epoch...')

rng_key  = jax.random.PRNGKey(SEED + 1)
best_auc = 0.0
best_p   = params
history  = []

for epoch in range(1, N_EPOCHS + 1):
    t0     = time.time()
    perm   = np.random.RandomState(epoch).permutation(len(train_emb))
    losses = []

    for start in range(0, len(perm) - BATCH_SIZE + 1, BATCH_SIZE):
        idx    = perm[start:start + BATCH_SIZE]
        emb_b  = jnp.array(train_emb[idx])
        lab_b  = jnp.array(train_labels[idx])
        rng_key, step_rng = jax.random.split(rng_key)
        params, opt_state, loss = train_step(params, opt_state, emb_b, lab_b, step_rng)
        losses.append(float(loss))

    auc    = eval_auc(params)
    t_ep   = time.time() - t0
    m_loss = float(np.mean(losses))
    print(f'Epoch {epoch:02d}/{N_EPOCHS}  loss={m_loss:.4f}  AUC={auc:.4f}  ({t_ep:.1f}s)', flush=True)
    history.append({'epoch': epoch, 'loss': m_loss, 'auc': auc})

    if auc > best_auc:
        best_auc = auc
        best_p   = jax.tree_util.tree_map(lambda x: np.array(x), params)
        with open(OUT_DIR / 'best_params.pkl', 'wb') as f:
            pickle.dump(best_p, f)
        print(f'  ★ New best AUC = {best_auc:.4f}', flush=True)

    with open(OUT_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

print(f'\n✅ Done! Best AUC = {best_auc:.4f}')
print(f'   Saved → {OUT_DIR}/best_params.pkl')
