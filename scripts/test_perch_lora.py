#!/usr/bin/env python3
"""Quick test of backbone + LoRA head."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU1 busy with SED NS training
import sys
sys.path.insert(0, '/home/lab/BirdClef-2026-Codebase')

import pickle
import jax, jax.numpy as jnp
print(f'JAX: {jax.__version__}', flush=True)

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from chirp.models import efficientnet, frontend as frontend_
from flax import linen as nn
import numpy as np

# Load backbone
with open('weights/perch_jax_backbone/perch_backbone_params.pkl', 'rb') as f:
    bv = pickle.load(f)
print(f'Backbone loaded, keys: {list(bv.keys())}', flush=True)

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
emb = backbone.apply(bv, jnp.zeros((2, 160000)), train=False, mutable=False)
print(f'Backbone OK: {emb.shape} mean={float(emb.mean()):.4f}', flush=True)

# Test LoRAHead
class LoRAHead(nn.Module):
    n_classes: int = 234
    rank: int = 8
    alpha: float = 16.0
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, emb, train=False):
        scale = self.alpha / self.rank
        A = self.param('lora_A', nn.initializers.normal(0.01), (emb.shape[-1], self.rank))
        B = self.param('lora_B', nn.initializers.zeros, (self.rank, emb.shape[-1]))
        x = emb + scale * (emb @ A @ B)
        x = nn.LayerNorm()(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        logits = nn.Dense(self.n_classes)(x)
        return logits

head = LoRAHead()
rng = jax.random.PRNGKey(42)
hvars = head.init({'params': rng, 'dropout': jax.random.PRNGKey(1)}, emb, train=True)
hp = hvars['params']
n = sum(v.size for v in jax.tree_util.tree_leaves(hp))
print(f'Head params: {n:,}', flush=True)
print(f'  lora_A: {hp["lora_A"].shape}  lora_B: {hp["lora_B"].shape}', flush=True)

logits = head.apply({'params': hp}, emb, train=False)
print(f'Head output: {logits.shape}', flush=True)
print('✅ All tests passed!', flush=True)
