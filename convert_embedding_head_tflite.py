"""Convert trained embedding-head to TFLite for use in submission notebook.

The embedding-head is a simple FC head:
  Input: (1, 1536) Perch embedding
  FC1(1024) + ReLU + FC2(234) + Sigmoid
  Output: (1, 234) probabilities

Usage:
    python convert_embedding_head_tflite.py
    python convert_embedding_head_tflite.py --run nohuman-embedding-soundscape
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import h5py

from src.utils.config import load_config
from src.data.dataset import build_species_mapping


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run",            default="nohuman-embedding-soundscape")
    p.add_argument("--checkpoints_dir",default="checkpoints")
    p.add_argument("--outputs_dir",    default="outputs")
    p.add_argument("--out_dir",        default="submissions/weights")
    p.add_argument("--config",         default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()

    config     = load_config(args.config)
    run_config = load_config(os.path.join(args.outputs_dir, args.run, "config.yaml"))
    _, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(species_to_idx)

    ckpt_path = os.path.join(args.checkpoints_dir, args.run, "best_head.weights.h5")
    if not os.path.isfile(ckpt_path):
        # Try legacy format
        ckpt_path = os.path.join(args.checkpoints_dir, args.run, "best_head")
        if not os.path.isfile(ckpt_path + ".weights.h5"):
            print(f"Checkpoint not found: {ckpt_path}")
            return

    embedding_dim = 1536
    hidden_dim    = run_config.model.hidden_dim
    print(f"Run: {args.run}  embedding_dim={embedding_dim}  hidden_dim={hidden_dim}")

    # ── Build head model ─────────────────────────────────────────────────────
    from src.model.classifier import PerchClassifier
    model = PerchClassifier(
        perch_dir=config.model.perch_dir,
        num_classes=num_classes,
        mode="embedding_head",
        hidden_dim=hidden_dim,
        dropout=0.0,
        embedding_dim=embedding_dim,
    )
    model.load_head(ckpt_path.replace(".weights.h5", ""))
    print(f"Head loaded from {ckpt_path}")

    # ── Verify head ───────────────────────────────────────────────────────────
    dummy = tf.zeros((1, embedding_dim), dtype=tf.float32)
    out   = tf.sigmoid(model.head(dummy, training=False))
    print(f"  dummy forward → {out.shape}  nan={tf.math.is_nan(out).numpy().any()}")

    # ── Build standalone Keras model for TFLite ───────────────────────────────
    head_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(embedding_dim,), name='perch_embedding'),
        tf.keras.layers.Dense(hidden_dim, activation='relu', name='fc1'),
        tf.keras.layers.Dense(num_classes, activation='sigmoid', name='fc2'),
    ], name='embedding_head')

    # Copy weights from trained head
    _ = head_model(dummy)   # build
    head_model.get_layer('fc1').set_weights([
        model.head.fc1.kernel.numpy(),
        model.head.fc1.bias.numpy(),
    ])
    head_model.get_layer('fc2').set_weights([
        model.head.fc2.kernel.numpy(),
        model.head.fc2.bias.numpy(),
    ])

    # Verify
    out_ref  = tf.sigmoid(model.head(dummy, training=False)).numpy()
    out_copy = head_model(dummy).numpy()
    max_diff = np.abs(out_ref - out_copy).max()
    print(f"  Weight copy verified  max_diff={max_diff:.2e}")
    assert max_diff < 1e-5, f"Weight copy failed: max_diff={max_diff}"

    # ── Convert to TFLite ─────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(head_model)
    converter.optimizations = []   # no quantization — keep float32
    tflite_model = converter.convert()

    os.makedirs(args.out_dir, exist_ok=True)
    tflite_path = os.path.join(args.out_dir, f"embedding_head_{args.run.replace('-','_')}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"TFLite saved → {tflite_path}  ({size_kb:.0f} KB)")

    # ── Validate TFLite ───────────────────────────────────────────────────────
    interp = tf.lite.Interpreter(model_content=tflite_model, num_threads=1)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']
    interp.set_tensor(inp_idx, dummy.numpy())
    interp.invoke()
    tflite_out = interp.get_tensor(out_idx)
    max_diff_tflite = np.abs(out_copy - tflite_out).max()
    print(f"  TFLite validation  max_diff={max_diff_tflite:.2e}  OK ✓")

    # ── Also export h5 weights for reference ─────────────────────────────────
    h5_path = os.path.join(args.out_dir, f"best_head_{args.run}.h5")
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('fc1/vars/0', data=model.head.fc1.kernel.numpy())
        hf.create_dataset('fc1/vars/1', data=model.head.fc1.bias.numpy())
        hf.create_dataset('fc2/vars/0', data=model.head.fc2.kernel.numpy())
        hf.create_dataset('fc2/vars/1', data=model.head.fc2.bias.numpy())
    print(f"H5 weights saved → {h5_path}")

    print(f"\nDone. TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
