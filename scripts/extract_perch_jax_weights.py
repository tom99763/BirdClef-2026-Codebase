"""
extract_perch_jax_weights.py
────────────────────────────────────────────────────────────────────────────
從 TF SavedModel 的 variables/ checkpoint 提取 Perch v2 權重，
還原為 JAX/Flax 可用的 parameter tree，存成 orbax checkpoint。

原理：jax2tf 將 JAX params flatten 為 _tf_var_leaves/0, /1, /2... 的順序。
反向操作：初始化 Flax 模型 → 取得 tree structure → 依序填入 TF 變數值。
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 使用 GPU1

import sys
sys.path.insert(0, '/home/lab/BirdClef-2026-Codebase')

import math
import numpy as np
from pathlib import Path

# ── Step 1: 載入 TF variables ─────────────────────────────────────────────────
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # CPU only for TF

TF_CKPT = Path('models/bird-vocalization-classifier-tensorflow2-perch_v2-v2/variables/variables')

print('Loading TF checkpoint...')
ckpt = tf.train.load_checkpoint(str(TF_CKPT))
var_map = ckpt.get_variable_to_shape_map()

# 取出 _tf_var_leaves 數量
n_leaves = sum(1 for k in var_map if '_tf_var_leaves/' in k)
print(f'Found {n_leaves} tf_var_leaves (expect ~497)')

# 載入所有 leaves 按順序
tf_leaves = []
for i in range(n_leaves):
    key = f'_tf_var_leaves/{i}/.ATTRIBUTES/VARIABLE_VALUE'
    val = ckpt.get_tensor(key)
    tf_leaves.append(np.array(val))

print(f'Loaded {len(tf_leaves)} leaves')
total_params = sum(v.size for v in tf_leaves)
print(f'Total parameters: {total_params:,}')

# ── Step 2: 初始化 Flax EmbeddingModel ────────────────────────────────────────
print('\nInitializing Flax EmbeddingModel...')
import jax
import jax.numpy as jnp
from chirp.models import perch_2, efficientnet, frontend as frontend_

# 和 TF SavedModel 的 input: (None, 160000) float32 相同
SAMPLE_RATE = 32000
WINDOW_SEC  = 5.0
N_SAMPLES   = int(SAMPLE_RATE * WINDOW_SEC)  # 160000

from chirp.models import frontend as frontend_
from flax import linen as nn
import math

# perch_2.EmbeddingModel 缺少 magnitude_scaling 定義，修補之
class FixedEmbeddingModel(perch_2.EmbeddingModel):
    """Patched EmbeddingModel with magnitude_scaling using frontend._magnitude_scale."""

    @nn.compact
    def __call__(self, inputs, train: bool, sow: bool = True):
        unscaled_spec = self.frontend(inputs)
        # 直接使用 frontend 的 _magnitude_scale 方法
        spectrogram = self.frontend._magnitude_scale(unscaled_spec)

        spatial_embedding = self.backbone(
            jnp.expand_dims(spectrogram, axis=-1), train=train
        )
        avg_embedding = jnp.mean(spatial_embedding, axis=(-2, -3))
        if sow:
            self.sow('intermediates', 'spectrogram', spectrogram)
            self.sow('intermediates', 'embedding', avg_embedding)
            self.sow('intermediates', 'spatial_embedding', spatial_embedding)
        return spatial_embedding

model = FixedEmbeddingModel()

# 用小 batch 初始化
rng = jax.random.PRNGKey(0)
dummy = jnp.zeros((1, N_SAMPLES), dtype=jnp.float32)

print('  Running model.init()...')
variables = model.init(rng, dummy, train=False)
params = variables['params']

# 統計 Flax 的 tree leaves
flax_leaves = jax.tree_util.tree_leaves(params)
print(f'  Flax leaves: {len(flax_leaves)}')
print(f'  Flax total params: {sum(v.size for v in flax_leaves):,}')

# ── Step 3: 驗證 shape 一致性 ─────────────────────────────────────────────────
print('\nVerifying shape alignment...')
if len(tf_leaves) != len(flax_leaves):
    print(f'  WARNING: leaf count mismatch: TF={len(tf_leaves)}, Flax={len(flax_leaves)}')
    # 顯示差異
    for i, (tf_v, fl_v) in enumerate(zip(tf_leaves[:20], flax_leaves[:20])):
        match = '✓' if tf_v.shape == fl_v.shape else '✗'
        print(f'  [{i:3d}] TF:{tf_v.shape} Flax:{fl_v.shape} {match}')
else:
    mismatches = [(i, tf_v.shape, fl_v.shape)
                  for i, (tf_v, fl_v) in enumerate(zip(tf_leaves, flax_leaves))
                  if tf_v.shape != fl_v.shape]
    if mismatches:
        print(f'  Shape mismatches: {len(mismatches)}')
        for i, ts, fs in mismatches[:10]:
            print(f'    [{i}] TF:{ts} vs Flax:{fs}')
    else:
        print(f'  All {len(tf_leaves)} shapes match!')

# ── Step 4: 重建 Flax params ──────────────────────────────────────────────────
print('\nRebuilding Flax params from TF weights...')

# 用 TF 權重替換 Flax 初始化的值
tf_arr = [jnp.array(v) for v in tf_leaves]
new_params = jax.tree_util.tree_unflatten(
    jax.tree_util.tree_structure(params),
    tf_arr[:len(flax_leaves)]
)

# ── Step 5: 快速驗證（前向傳播） ──────────────────────────────────────────────
print('\nVerifying forward pass with loaded weights...')

# 也需要 batch_stats（如果有 BatchNorm）
if 'batch_stats' in variables:
    other_vars = {'batch_stats': variables['batch_stats']}
else:
    other_vars = {}

test_audio = jnp.zeros((1, N_SAMPLES), dtype=jnp.float32)
out = model.apply(
    {'params': new_params, **other_vars},
    test_audio,
    train=False,
    mutable=False,
)
print(f'  Output shape: {out.shape}')  # 應為 (1, H, W, C)

# 均值取 embedding
emb = jnp.mean(out, axis=(-2, -3))
print(f'  Embedding shape: {emb.shape}')  # 應為 (1, 1536)

# ── Step 6: 存成 orbax checkpoint ────────────────────────────────────────────
print('\nSaving orbax checkpoint...')
import orbax.checkpoint as ocp

save_dir = Path('weights/perch_jax_ckpt')
save_dir.mkdir(parents=True, exist_ok=True)

checkpointer = ocp.StandardCheckpointer()
checkpointer.save(
    str(save_dir),
    args=ocp.args.StandardSave({'params': new_params, **other_vars})
)
print(f'Saved → {save_dir}')
print('\n✅ Perch v2 JAX checkpoint extracted successfully!')
print('   Now you can fine-tune end-to-end with chirp/train/classifier.py')
