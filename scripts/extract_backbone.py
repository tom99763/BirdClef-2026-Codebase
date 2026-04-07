import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax, jax.numpy as jnp
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from chirp.models import efficientnet, frontend as frontend_
from flax import linen as nn
from pathlib import Path
import numpy as np

class FixedEM(nn.Module):
    frontend: frontend_.MelSpectrogram = frontend_.MelSpectrogram(
        features=128, stride=320, kernel_size=640, sample_rate=32000,
        freq_range=(60,16000), power=1.0,
        scaling_config=frontend_.LogScalingConfig(), nfft=1024)
    backbone: nn.Module = efficientnet.EfficientNet(
        efficientnet.EfficientNetModel.B3, include_top=False)
    @nn.compact
    def __call__(self, x, train, sow=True):
        s = self.frontend._magnitude_scale(self.frontend(x))
        return jnp.mean(self.backbone(jnp.expand_dims(s,-1), train=train), axis=(-2,-3))

print('[1] init Flax model', flush=True)
model = FixedEM()
v0 = model.init(jax.random.PRNGKey(0), jnp.zeros((1,160000)), train=False)
flat = jax.tree_util.tree_leaves(v0)
tree = jax.tree_util.tree_structure(v0)
print(f'    leaves={len(flat)}  params={sum(x.size for x in flat):,}', flush=True)

print('[2] load TF checkpoint', flush=True)
ckpt = tf.train.load_checkpoint(
    'models/bird-vocalization-classifier-tensorflow2-perch_v2-v2/variables/variables')
vm = ckpt.get_variable_to_shape_map()
bk = sorted([(int(k.split('/')[1]), k) for k,s in vm.items()
              if '_tf_var_leaves/' in k and tuple(s) != (14795,1536,4)])
print(f'    TF backbone keys={len(bk)}', flush=True)

print('[3] shape check', flush=True)
ok = sum(1 for i,(_, k) in enumerate(bk[:len(flat)]) if flat[i].shape == tuple(vm[k]))
print(f'    shape matches: {ok}/{min(len(flat),len(bk))}', flush=True)

print('[4] load TF values', flush=True)
new_flat = [jnp.array(ckpt.get_tensor(k)) for _, k in bk[:len(flat)]]
new_v = jax.tree_util.tree_unflatten(tree, new_flat)

print('[5] forward pass', flush=True)
out = model.apply(new_v, jnp.zeros((1,160000)), train=False, mutable=False)
print(f'    shape={out.shape} mean={float(out.mean()):.4f} std={float(out.std()):.4f}', flush=True)

print('[6] save orbax checkpoint', flush=True)
p = Path('weights/perch_jax_backbone')
p.mkdir(parents=True, exist_ok=True)
import pickle
with open(p / 'perch_backbone_params.pkl', 'wb') as f:
    pickle.dump(jax.tree_util.tree_map(lambda x: np.array(x), new_v), f)
print(f'    params saved as pickle')
print(f'    Saved → {p}', flush=True)
print('✅ Done!', flush=True)
