"""BirdClef 2026 — Pseudo-labeling Pipeline

Multi-round semi-supervised training inspired by BirdCLEF 2025 top solutions:
  - 1st place: 4 rounds, PowerTransform sharpening → 0.872 → 0.933
  - 2nd place: 2 rounds, F2-score thresholding → 0.87 → 0.91
  - 4th place: 2 rounds from 10-model ensemble

Workflow:
  1. Run inference on unlabeled soundscapes (test or extra recordings)
  2. Apply PowerTransform to sharpen confident predictions
  3. Filter pseudo-labels by confidence threshold
  4. Write a pseudo_labels.csv compatible with ClipDataset
  5. Retrain using original data + pseudo-labeled data

Usage (Round 1 — generate pseudo-labels from a trained checkpoint):
    python pseudo_label.py generate \
        --config configs/default.yaml \
        --checkpoint checkpoints/my_run/best_head \
        --soundscapes_dir birdclef-2026/train_soundscapes \
        --output pseudo_labels/round1_pseudo.csv \
        --threshold 0.5 \
        --power 2.0

Usage (Round 2 — train with pseudo-labels then generate next round):
    python train.py --config configs/pseudo_label.yaml

    python pseudo_label.py generate \
        --config configs/pseudo_label.yaml \
        --checkpoint checkpoints/pseudo-r1/best_head \
        --output pseudo_labels/round2_pseudo.csv
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils.config import load_config
from src.utils.audio import load_audio
from src.data.dataset import build_species_mapping
from src.model.losses import power_transform


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_soundscape(
    filepath: str,
    model,
    sample_rate: int,
    clip_duration: int,
    batch_size: int,
) -> tuple:
    """
    Split a soundscape into 5-second clips and return (row_ids, probabilities).
    """
    ss_id = re.sub(r"\.ogg$", "", os.path.basename(filepath), flags=re.IGNORECASE)
    audio = load_audio(filepath, sample_rate)
    if audio is None:
        return [], np.empty((0,))

    clip_length = clip_duration * sample_rate
    n_segments = len(audio) // clip_length
    if n_segments == 0:
        return [], np.empty((0,))

    # Optionally add TTA: shifted clips (2nd place: 2.5-second shifts)
    clips_normal = np.stack([
        audio[i * clip_length: (i + 1) * clip_length]
        for i in range(n_segments)
    ])

    half = clip_length // 2
    audio_shifted = audio[half:]
    n_shifted = len(audio_shifted) // clip_length
    if n_shifted > 0:
        clips_shifted = np.stack([
            audio_shifted[i * clip_length: (i + 1) * clip_length]
            for i in range(n_shifted)
        ])
    else:
        clips_shifted = None

    row_ids = [f"{ss_id}_{(i + 1) * clip_duration}" for i in range(n_segments)]

    def _infer(clips):
        preds = []
        for start in range(0, len(clips), batch_size):
            batch = tf.constant(clips[start: start + batch_size], dtype=tf.float32)
            logits = model(batch, training=False)
            preds.append(tf.sigmoid(logits).numpy())
        return np.concatenate(preds, axis=0)

    probs = _infer(clips_normal)

    if clips_shifted is not None:
        probs_shifted = _infer(clips_shifted)
        # Align shifted predictions to the normal segments by interleaving
        # (simple approach: average with nearest shifted pred if available)
        n_use = min(len(probs), len(probs_shifted))
        probs[:n_use] = 0.5 * probs[:n_use] + 0.5 * probs_shifted[:n_use]

    return row_ids, probs


# ── Generate command ──────────────────────────────────────────────────────────

def cmd_generate(args):
    config = load_config(args.config)
    target_species, species_to_idx = build_species_mapping(
        config.data.sample_submission_csv
    )
    num_classes = len(target_species)

    from src.model.classifier import PerchClassifier
    print("Loading model …")
    model = PerchClassifier(
        perch_dir=config.model.perch_dir,
        num_classes=num_classes,
        mode=config.model.mode,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        taxonomy_csv=getattr(config.data, 'taxonomy_csv', None),
        sample_submission_csv=getattr(config.data, 'sample_submission_csv', None),
    )
    model.load_head(args.checkpoint)

    soundscapes_dir = args.soundscapes_dir or config.data.train_soundscapes_dir
    ogg_files = sorted(glob.glob(os.path.join(soundscapes_dir, "*.ogg")))
    print(f"Soundscapes: {len(ogg_files)}")

    batch_size = args.batch_size or config.training.batch_size * 2
    sample_rate = config.audio.sample_rate
    clip_duration = config.audio.clip_duration

    all_row_ids, all_probs = [], []
    for fp in ogg_files:
        print(f"  {os.path.basename(fp)}")
        row_ids, probs = predict_soundscape(fp, model, sample_rate, clip_duration, batch_size)
        if len(row_ids) > 0:
            all_row_ids.extend(row_ids)
            all_probs.append(probs)

    if not all_probs:
        print("ERROR: No predictions generated.")
        return

    probs_all = np.concatenate(all_probs, axis=0)   # (N, num_classes)
    print(f"\nRaw predictions: {probs_all.shape}")

    # ── PowerTransform (BirdCLEF 2025 1st place) ─────────────────────────────
    if args.power > 1.0:
        print(f"Applying PowerTransform (power={args.power}) …")
        probs_all = power_transform(probs_all, power=args.power)

    # ── Confidence threshold filtering ────────────────────────────────────────
    # Keep only segments where at least one class exceeds threshold
    max_conf = probs_all.max(axis=1)
    keep_mask = max_conf >= args.threshold
    n_kept = keep_mask.sum()
    print(f"Segments kept (max_prob >= {args.threshold}): {n_kept} / {len(all_row_ids)}")

    if n_kept == 0:
        print("WARNING: No segments passed the threshold. Try lowering --threshold.")
        return

    kept_row_ids = [r for r, m in zip(all_row_ids, keep_mask) if m]
    kept_probs = probs_all[keep_mask]

    # ── Build pseudo_labels.csv ───────────────────────────────────────────────
    # Format: row_id, primary_label (top class), soft_labels (semicolon sep)
    # Also write soft prob columns so ClipDataset can use them as soft targets.
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # Build a DataFrame with row_id + soft probability per class
    df_soft = pd.DataFrame(kept_probs, columns=target_species)
    df_soft.insert(0, "row_id", kept_row_ids)

    # Derive primary_label (argmax) and hard labels above a binary threshold
    top_idx = kept_probs.argmax(axis=1)
    df_soft["primary_label"] = [target_species[i] for i in top_idx]
    # Secondary labels: all classes > 0.5 (excluding primary)
    def _secondary(row_probs, primary):
        secs = [target_species[i] for i, p in enumerate(row_probs) if p >= 0.5
                and target_species[i] != primary]
        return ";".join(secs)
    df_soft["secondary_labels"] = [
        _secondary(kept_probs[i], df_soft["primary_label"].iloc[i])
        for i in range(len(kept_probs))
    ]

    df_soft.to_csv(args.output, index=False)
    print(f"\nPseudo-labels saved → {args.output}")
    print(f"  Rows: {len(df_soft)}")
    print(f"  Species coverage: {(kept_probs >= 0.5).any(axis=0).sum()} / {num_classes}")
    print("\nTop-5 most common pseudo-labeled species:")
    print(df_soft["primary_label"].value_counts().head())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="BirdClef 2026 Pseudo-labeling")
    sub = parser.add_subparsers(dest="cmd")

    gen = sub.add_parser("generate", help="Generate pseudo-labels from a checkpoint")
    gen.add_argument("--config", default="configs/default.yaml")
    gen.add_argument("--checkpoint", required=True,
                     help="Path to saved head weights")
    gen.add_argument("--soundscapes_dir", default=None,
                     help="Directory with .ogg soundscape files to pseudo-label")
    gen.add_argument("--output", default="pseudo_labels/round1_pseudo.csv",
                     help="Output CSV path")
    gen.add_argument("--threshold", type=float, default=0.5,
                     help="Min confidence for a segment to be included")
    gen.add_argument("--power", type=float, default=2.0,
                     help="PowerTransform exponent (1.0 = disabled, 2.0 = 1st place default)")
    gen.add_argument("--batch_size", type=int, default=None)
    gen.add_argument("--gpu", default=None,
                     help="CUDA_VISIBLE_DEVICES (e.g. 0, 1, 0,1)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.cmd == "generate":
        cmd_generate(args)
    else:
        print("Usage: python pseudo_label.py generate --help")
