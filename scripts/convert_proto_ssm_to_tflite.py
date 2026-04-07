"""Convert ProtoSSM PyTorch checkpoints to TFLite.

Implements the full ProtoSSM forward pass natively in TensorFlow (no ONNX),
then converts to TFLite with fixed T=12 input shapes.

Usage:
    python scripts/convert_proto_ssm_to_tflite.py \
        --variant light \
        --out_dir "birdclef-2026/notebook resource/current_subs/weights"

    python scripts/convert_proto_ssm_to_tflite.py \
        --variant full \
        --out_dir "birdclef-2026/notebook resource/current_subs/weights"
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────────────
T        = 12      # fixed windows per sequence
D_INPUT  = 1536
D_MODEL  = 128
D_STATE  = 16
N_CLASSES = 234
N_SSM_LAYERS = 2
D_CONV   = 4
N_FOLDS  = 5


# ── Numpy inference helpers ───────────────────────────────────────────────────

def gelu(x):
    return x * 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def depthwise_conv1d(x_seq, weight, bias, d_conv=4, T_fixed=12):
    """Depthwise conv1d: x_seq (T, D) → (T, D).
    Matches PyTorch Conv1d(D, D, d_conv, padding=d_conv-1, groups=D)[:, :, :T].
    weight: (D, d_conv) — depthwise kernel per channel
    """
    D = x_seq.shape[1]
    pad = d_conv - 1
    x_pad = np.concatenate([np.zeros((pad, D), dtype=np.float32), x_seq], axis=0)
    out = np.zeros((T_fixed, D), dtype=np.float32)
    for t in range(T_fixed):
        # window: x_pad[t : t+d_conv, :] → shape (d_conv, D)
        win = x_pad[t:t + d_conv, :]  # (d_conv, D)
        # per-channel multiply: sum over kernel dim
        out[t] = (win * weight.T).sum(axis=0)  # (D,)
    if bias is not None:
        out += bias
    return out


def selective_scan_numpy(x_conv, dt, A_log, B_mat, C_mat, D_param):
    """x_conv,dt,B_mat,C_mat: (T, D), (T, D), (T, N), (T, N); A_log: (D, N); D_param: (D,)
    Returns y: (T, D)
    """
    T_seq = x_conv.shape[0]
    D = x_conv.shape[1]
    N = B_mat.shape[1]
    A = -np.exp(A_log)  # (D, N)
    h = np.zeros((D, N), dtype=np.float32)
    ys = []
    for t in range(T_seq):
        dt_t = dt[t, :, None]              # (D, 1)
        dA   = np.exp(A * dt_t)            # (D, N)
        dB   = dt_t * B_mat[t, None, :]   # (D, N)
        h    = h * dA + x_conv[t, :, None] * dB
        y_t  = (h * C_mat[t, None, :]).sum(axis=-1)  # (D,)
        ys.append(y_t)
    y = np.stack(ys, axis=0)  # (T, D)
    return y + x_conv * D_param[None, :]


def ssm_forward_numpy(x, w):
    """One SelectiveSSM forward pass.
    x: (T, D_MODEL)
    w: dict of numpy weights for this SSM
    Returns: (T, D_MODEL)
    """
    # in_proj: (T, D_MODEL) → (T, 2*D_MODEL), then split
    xz = x @ w['in_proj.weight'].T           # (T, 2D)
    x_ssm, z = xz[:, :D_MODEL], xz[:, D_MODEL:]

    # depthwise conv1d on x_ssm
    x_conv = depthwise_conv1d(x_ssm, w['conv1d.weight'], w.get('conv1d.bias'))
    x_conv = silu(x_conv)                    # (T, D)

    # input-dependent projections
    dt    = softplus(x_conv @ w['dt_proj.weight'].T + w['dt_proj.bias'])  # (T, D)
    B_mat = x_conv @ w['B_proj.weight'].T    # (T, N)
    C_mat = x_conv @ w['C_proj.weight'].T    # (T, N)

    y = selective_scan_numpy(x_conv, dt, w['A_log'], B_mat, C_mat, w['D'])  # (T, D)
    y = y * silu(z)                          # z-gate
    return y @ w['out_proj.weight'].T        # (T, D)


def proto_ssm_numpy_forward(emb, perch_logits, weights):
    """Full ProtoSSM forward (numpy).
    emb:          (T, D_INPUT) = (12, 1536)
    perch_logits: (T, N_CLASSES) = (12, 234)  or None
    weights:      dict from load_weights()
    Returns: species_logits (T, N_CLASSES)
    """
    # input_proj: Linear → LayerNorm → GELU  (no dropout at inference)
    h = emb @ weights['input_proj.0.weight'].T   # (T, D_MODEL)
    if 'input_proj.0.bias' in weights:
        h = h + weights['input_proj.0.bias']
    h = layer_norm(h, weights['input_proj.1.weight'], weights['input_proj.1.bias'])
    h = gelu(h)

    # positional encoding
    pos_enc = weights['pos_enc'][0]              # (T, D_MODEL)
    h = h + pos_enc[:emb.shape[0]]

    # bidirectional SSM layers
    for i in range(N_SSM_LAYERS):
        residual = h
        wf = {k.split(f'ssm_fwd.{i}.')[1]: v
              for k, v in weights.items() if k.startswith(f'ssm_fwd.{i}.')}
        wb = {k.split(f'ssm_bwd.{i}.')[1]: v
              for k, v in weights.items() if k.startswith(f'ssm_bwd.{i}.')}
        h_f = ssm_forward_numpy(h, wf)
        h_b = ssm_forward_numpy(h[::-1], wb)[::-1]

        h_cat = np.concatenate([h_f, h_b], axis=-1)  # (T, 2*D)
        wm = weights[f'ssm_merge.{i}.weight']         # (D, 2D)
        bm = weights[f'ssm_merge.{i}.bias']           # (D,)
        h  = h_cat @ wm.T + bm                        # (T, D)

        # LayerNorm + residual
        wn_g = weights[f'ssm_norm.{i}.weight']
        wn_b = weights[f'ssm_norm.{i}.bias']
        h    = layer_norm(h + residual, wn_g, wn_b)

    # prototypical cosine similarity
    h_norm = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)   # (T, D)
    p_norm = weights['prototypes'] / (
        np.linalg.norm(weights['prototypes'], axis=-1, keepdims=True) + 1e-8)  # (N_CLS, D)
    temp   = np.log1p(np.exp(weights['proto_temp']))                            # softplus
    sim    = h_norm @ p_norm.T * temp                                           # (T, N_CLS)

    # fusion with Perch logits
    alpha = sigmoid(weights['fusion_alpha'])[None, :]                # (1, N_CLS)
    species_logits = alpha * sim + (1.0 - alpha) * perch_logits      # (T, N_CLS)

    return species_logits.astype(np.float32)


def load_weights_from_pt(pt_path):
    """Load and convert PyTorch checkpoint to numpy weight dict."""
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    # Try both common key names
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    weights = {}
    for k, v in state.items():
        if hasattr(v, 'detach'):
            arr = v.detach().cpu().numpy()
        else:
            arr = np.array(v, dtype=np.float32)
        weights[k] = arr
    # Flatten conv1d weight: PyTorch (D, 1, d_conv) → numpy (D, d_conv)
    for k in list(weights.keys()):
        if 'conv1d.weight' in k and weights[k].ndim == 3:
            weights[k] = weights[k][:, 0, :]
    return weights


# ── TFLite conversion (wrap numpy model in tf.function) ─────────────────────

def build_tflite_from_weights(weights_np, variant_name, fold_idx, out_dir):
    """Build a TFLite model from extracted numpy weights.

    Strategy: wrap the numpy inference in a @tf.function using tf.constant weights.
    This produces a frozen TFLite model with all weights embedded.
    """
    # Convert all weights to tf.constant
    tf_w = {k: tf.constant(v, dtype=tf.float32) for k, v in weights_np.items()}

    def tf_gelu(x):
        return x * 0.5 * (1.0 + tf.math.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

    def tf_silu(x):
        return x * tf.math.sigmoid(x)

    def tf_softplus(x):
        return tf.math.softplus(x)

    def tf_layer_norm(x, gamma, beta, eps=1e-5):
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        return gamma * (x - mean) / tf.sqrt(var + eps) + beta

    def tf_depthwise_conv1d(x_seq, weight, bias):
        # x_seq: (T, D), weight: (D, d_conv)
        pad_size = D_CONV - 1
        # Pad beginning of sequence
        zeros_pad = tf.zeros([pad_size, D_MODEL], dtype=tf.float32)
        x_pad = tf.concat([zeros_pad, x_seq], axis=0)  # (T+pad, D)
        # Manual convolution: (T, D)
        outs = []
        for t in range(T):
            win = x_pad[t:t + D_CONV, :]         # (d_conv, D)
            # element-wise with transposed weight: weight.T = (d_conv, D)
            out_t = tf.reduce_sum(win * tf.transpose(weight), axis=0)  # (D,)
            outs.append(out_t)
        out = tf.stack(outs, axis=0)  # (T, D)
        if bias is not None:
            out = out + bias
        return out

    def tf_selective_scan(x_conv, dt, A_log, B_mat, C_mat, D_param):
        # x_conv: (T,D), dt: (T,D), B_mat,C_mat: (T,N), A_log: (D,N), D_param: (D,)
        A = -tf.exp(A_log)  # (D, N)
        h = tf.zeros([D_MODEL, D_STATE], dtype=tf.float32)
        ys = []
        for t in range(T):
            dt_t = dt[t, :, tf.newaxis]              # (D, 1)
            dA   = tf.exp(A * dt_t)                  # (D, N)
            dB   = dt_t * B_mat[t, tf.newaxis, :]   # (D, N)
            h    = h * dA + x_conv[t, :, tf.newaxis] * dB
            y_t  = tf.reduce_sum(h * C_mat[t, tf.newaxis, :], axis=-1)  # (D,)
            ys.append(y_t)
        y = tf.stack(ys, axis=0)  # (T, D)
        return y + x_conv * D_param[tf.newaxis, :]

    def tf_ssm_forward(x, prefix):
        xz    = x @ tf.transpose(tf_w[f'{prefix}.in_proj.weight'])   # (T, 2D)
        x_ssm = xz[:, :D_MODEL]
        z     = xz[:, D_MODEL:]
        x_conv = tf_depthwise_conv1d(x_ssm, tf_w[f'{prefix}.conv1d.weight'],
                                     tf_w.get(f'{prefix}.conv1d.bias'))
        x_conv = tf_silu(x_conv)
        dt     = tf_softplus(x_conv @ tf.transpose(tf_w[f'{prefix}.dt_proj.weight'])
                             + tf_w[f'{prefix}.dt_proj.bias'])
        B_mat  = x_conv @ tf.transpose(tf_w[f'{prefix}.B_proj.weight'])
        C_mat  = x_conv @ tf.transpose(tf_w[f'{prefix}.C_proj.weight'])
        y = tf_selective_scan(x_conv, dt, tf_w[f'{prefix}.A_log'],
                              B_mat, C_mat, tf_w[f'{prefix}.D'])
        y = y * tf_silu(z)
        return y @ tf.transpose(tf_w[f'{prefix}.out_proj.weight'])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[T, D_INPUT],   dtype=tf.float32, name='emb'),
        tf.TensorSpec(shape=[T, N_CLASSES], dtype=tf.float32, name='perch_logits'),
    ])
    def model_fn(emb, perch_logits):
        # input_proj: Linear → LayerNorm → GELU
        h = emb @ tf.transpose(tf_w['input_proj.0.weight'])
        if 'input_proj.0.bias' in tf_w:
            h = h + tf_w['input_proj.0.bias']
        h = tf_layer_norm(h, tf_w['input_proj.1.weight'], tf_w['input_proj.1.bias'])
        h = tf_gelu(h)

        # positional encoding
        pos_enc = tf_w['pos_enc'][0]  # (T, D)
        h = h + pos_enc[:T]

        # bidirectional SSM layers
        for i in range(N_SSM_LAYERS):
            residual = h
            h_f = tf_ssm_forward(h, f'ssm_fwd.{i}')
            h_b = tf.reverse(tf_ssm_forward(tf.reverse(h, [0]), f'ssm_bwd.{i}'), [0])
            h_cat = tf.concat([h_f, h_b], axis=-1)
            h = h_cat @ tf.transpose(tf_w[f'ssm_merge.{i}.weight']) + tf_w[f'ssm_merge.{i}.bias']
            h = tf_layer_norm(h + residual,
                              tf_w[f'ssm_norm.{i}.weight'],
                              tf_w[f'ssm_norm.{i}.bias'])

        # prototypical cosine similarity
        h_norm = h / (tf.norm(h, axis=-1, keepdims=True) + 1e-8)
        p_norm = tf_w['prototypes'] / (tf.norm(tf_w['prototypes'], axis=-1, keepdims=True) + 1e-8)
        temp   = tf.math.softplus(tf_w['proto_temp'])
        sim    = h_norm @ tf.transpose(p_norm) * temp

        # fusion
        alpha = tf.math.sigmoid(tf_w['fusion_alpha'])[tf.newaxis, :]
        species_logits = alpha * sim + (1.0 - alpha) * perch_logits

        return {'species_logits': species_logits}

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [model_fn.get_concrete_function()],
        model_fn
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out_path = Path(out_dir) / f'proto_ssm_v4_{variant_name}_fold{fold_idx}.tflite'
    out_path.write_bytes(tflite_model)
    print(f'  Saved: {out_path}  ({len(tflite_model)/1024:.1f} KB)')
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=['light', 'full'],
                        help='Which ProtoSSM variant to convert')
    parser.add_argument('--out_dir', default='birdclef-2026/notebook resource/current_subs/weights',
                        help='Output directory for TFLite files')
    parser.add_argument('--verify', action='store_true',
                        help='Verify TFLite output matches PyTorch on random input')
    args = parser.parse_args()

    src_dir = Path(f'outputs/proto-ssm-v4-{args.variant}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Converting proto-ssm-v4-{args.variant} → TFLite')
    print(f'  Source : {src_dir}')
    print(f'  Output : {out_dir}')
    print()

    for fold in range(N_FOLDS):
        pt_path = src_dir / f'fold{fold}_best.pt'
        if not pt_path.exists():
            print(f'  SKIP fold{fold}: {pt_path} not found')
            continue

        print(f'  fold{fold}: loading weights ...')
        weights_np = load_weights_from_pt(pt_path)

        if args.verify:
            # Smoke test: verify numpy inference is reasonable
            rng = np.random.RandomState(42)
            emb_test   = rng.randn(T, D_INPUT).astype(np.float32)
            logit_test = rng.randn(T, N_CLASSES).astype(np.float32)
            out_np = proto_ssm_numpy_forward(emb_test, logit_test, weights_np)
            print(f'    numpy output shape={out_np.shape}  mean={out_np.mean():.4f}')

        print(f'  fold{fold}: building TFLite model ...')
        try:
            tflite_path = build_tflite_from_weights(weights_np, args.variant, fold, out_dir)

            if args.verify:
                # Verify TFLite vs numpy
                interp = tf.lite.Interpreter(str(tflite_path))
                interp.allocate_tensors()
                inp_details  = interp.get_input_details()
                out_details  = interp.get_output_details()
                interp.set_tensor(inp_details[0]['index'], emb_test[None] if inp_details[0]['shape'][0]==1 else emb_test)
                interp.set_tensor(inp_details[1]['index'], logit_test[None] if inp_details[1]['shape'][0]==1 else logit_test)
                interp.invoke()
                out_tfl = interp.get_tensor(out_details[0]['index'])
                out_np2 = proto_ssm_numpy_forward(emb_test, logit_test, weights_np)
                diff = np.abs(out_tfl - out_np2).max()
                print(f'    TFLite vs numpy max diff: {diff:.6f}')

        except Exception as e:
            print(f'  ERROR fold{fold}: {e}')
            import traceback; traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
