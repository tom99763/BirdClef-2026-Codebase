"""BirdClef 2026 — Inference / Submission Generation

Processes all soundscape files (test or train) and writes submission.csv.

Usage:
    # Local validation (uses train_soundscapes from config)
    python inference.py --config configs/default.yaml --checkpoint checkpoints/my_run/best_head

    # Kaggle test (auto-detects test_soundscapes if populated)
    python inference.py --config configs/default.yaml --checkpoint checkpoints/my_run/best_head

    # Override soundscapes directory explicitly
    python inference.py --config configs/default.yaml \
        --checkpoint checkpoints/my_run/best_head \
        --soundscapes_dir /path/to/test_soundscapes
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


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="BirdClef 2026 Inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved head weights (no extension needed)")
    parser.add_argument("--soundscapes_dir", default=None,
                        help="Override the soundscapes directory")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size for inference")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--tta", action="store_true",
                        help="Enable TTA with 2.5-second temporal shifts (BirdCLEF 2025 2nd place)")
    parser.add_argument("--gpu", default=None,
                        help="CUDA_VISIBLE_DEVICES (e.g. 0, 1, 0,1)")
    return parser.parse_args()


# ── Per-soundscape inference ─────────────────────────────────────────────────

def _batched_infer(model, clips: np.ndarray, batch_size: int) -> np.ndarray:
    """Run batched inference and return sigmoid probabilities."""
    all_preds = []
    for start in range(0, len(clips), batch_size):
        batch = tf.constant(clips[start: start + batch_size], dtype=tf.float32)
        logits = model(batch, training=False)
        all_preds.append(tf.sigmoid(logits).numpy())
    return np.concatenate(all_preds, axis=0)


def process_soundscape(
    filepath: str,
    model,
    sample_rate: int,
    clip_duration: int,
    batch_size: int,
    tta: bool = False,
) -> tuple:
    """
    Split a soundscape into consecutive 5-second clips and run inference.

    Args:
        tta: If True, also predict half-shifted clips and average (BirdCLEF 2025
             2nd place technique: +0.012 AUC via 2.5-second temporal shifts).

    Returns:
        row_ids    : List of strings like "BC2026_Train_0001_xxx_5", "_10", …
        predictions: np.ndarray of shape (n_segments, n_classes)
    """
    ss_id = re.sub(r"\.ogg$", "", os.path.basename(filepath), flags=re.IGNORECASE)

    audio = load_audio(filepath, sample_rate)
    if audio is None:
        return [], np.empty((0,))

    clip_length = clip_duration * sample_rate
    n_segments = len(audio) // clip_length
    if n_segments == 0:
        return [], np.empty((0,))

    clips = np.stack([
        audio[i * clip_length: (i + 1) * clip_length]
        for i in range(n_segments)
    ])
    row_ids = [f"{ss_id}_{(i + 1) * clip_duration}" for i in range(n_segments)]

    preds = _batched_infer(model, clips, batch_size)

    if tta:
        # 2.5-second shifted clips (half-window TTA from BirdCLEF 2025 2nd place)
        half = clip_length // 2
        audio_shifted = audio[half:]
        n_shifted = len(audio_shifted) // clip_length
        if n_shifted > 0:
            clips_shifted = np.stack([
                audio_shifted[i * clip_length: (i + 1) * clip_length]
                for i in range(n_shifted)
            ])
            preds_shifted = _batched_infer(model, clips_shifted, batch_size)
            n_use = min(len(preds), len(preds_shifted))
            preds[:n_use] = 0.5 * preds[:n_use] + 0.5 * preds_shifted[:n_use]

    return row_ids, preds


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = load_config(args.config)

    # ── Decide soundscapes directory ──────────────────────────────────────────
    if args.soundscapes_dir:
        soundscapes_dir = args.soundscapes_dir
    else:
        # On Kaggle the test_soundscapes folder is populated at evaluation time
        data_root = os.path.dirname(config.data.train_soundscapes_dir)
        test_dir = os.path.join(data_root, "test_soundscapes")
        if os.path.isdir(test_dir) and glob.glob(os.path.join(test_dir, "*.ogg")):
            soundscapes_dir = test_dir
            print(f"Using test soundscapes: {soundscapes_dir}")
        else:
            soundscapes_dir = config.data.train_soundscapes_dir
            print(f"Using train soundscapes: {soundscapes_dir}")

    ogg_files = sorted(glob.glob(os.path.join(soundscapes_dir, "*.ogg")))
    if args.max_files:
        ogg_files = ogg_files[: args.max_files]
    print(f"Soundscapes to process: {len(ogg_files)}")

    # ── Species mapping ───────────────────────────────────────────────────────
    target_species, species_to_idx = build_species_mapping(
        config.data.sample_submission_csv
    )
    num_classes = len(target_species)

    # ── Model ─────────────────────────────────────────────────────────────────
    from src.model.classifier import PerchClassifier
    print("\nLoading model …")
    model = PerchClassifier(
        perch_dir=config.model.perch_dir,
        num_classes=num_classes,
        mode=config.model.mode,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    )
    model.load_head(args.checkpoint)

    infer_batch_size = args.batch_size or config.training.batch_size * 2

    # ── Run inference ─────────────────────────────────────────────────────────
    all_row_ids, all_preds = [], []

    for filepath in ogg_files:
        print(f"  {os.path.basename(filepath)}")
        row_ids, preds = process_soundscape(
            filepath=filepath,
            model=model,
            sample_rate=config.audio.sample_rate,
            clip_duration=config.audio.clip_duration,
            batch_size=infer_batch_size,
            tta=args.tta,
        )
        if len(row_ids) > 0:
            all_row_ids.extend(row_ids)
            all_preds.append(preds)

    if not all_preds:
        print("ERROR: No predictions generated.")
        return

    # ── Write submission ──────────────────────────────────────────────────────
    predictions = np.concatenate(all_preds, axis=0)
    submission = pd.DataFrame(predictions, columns=target_species)
    submission.insert(0, "row_id", all_row_ids)
    submission.to_csv(args.output, index=False)

    print(f"\nSubmission saved → {args.output}  ({submission.shape[0]} rows × {num_classes} species)")
    print(submission.head())


if __name__ == "__main__":
    main()
