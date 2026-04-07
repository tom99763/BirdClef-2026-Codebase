"""Extract Perch teacher predictions for ALL train_soundscapes files.

BirdCLEF 2026 has 10,658 unlabeled soundscape .ogg files in train_soundscapes/.
This script runs the trained Perch head on all of them and saves 234-dim soft
probability predictions for each 5-second window — building the "teacher database"
for knowledge distillation.

These Pantanal-domain predictions transfer Perch's calibrated knowledge of
co-occurring species to the EfficientNet-B0 student SED model.

Usage:
    python scripts/extract_perch_teacher_all_ss.py [--output PATH] [--batch_size N]
    python scripts/extract_perch_teacher_all_ss.py --output outputs/perch_teacher_all_ss.csv
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.audio import load_audio
from src.data.dataset import build_species_mapping

CONFIG       = "configs/exp_nohuman_label_soundscape_train.yaml"
CHECKPOINT   = "checkpoints/nohuman-label-soundscape-train/best_head.weights.h5"
SS_DIR       = "birdclef-2026/train_soundscapes"
OUTPUT_CSV   = "outputs/perch_teacher_all_ss.csv"
BATCH_SIZE   = 32
SR           = 32_000
CLIP_DUR     = 5
SAVE_EVERY   = 500   # save checkpoint every N files


def predict_file(filepath, model, clip_samples, batch_size):
    """Returns (row_ids, probs_array) for all clips in the file."""
    ss_id = re.sub(r"\.ogg$", "", os.path.basename(filepath), flags=re.IGNORECASE)
    audio = load_audio(filepath, SR)
    if audio is None or len(audio) < clip_samples:
        return [], None

    n_segs = len(audio) // clip_samples
    clips  = np.stack([audio[i * clip_samples: (i + 1) * clip_samples]
                       for i in range(n_segs)], axis=0)

    preds = []
    for start in range(0, len(clips), batch_size):
        batch = tf.constant(clips[start: start + batch_size], dtype=tf.float32)
        logits = model(batch, training=False)
        preds.append(tf.sigmoid(logits).numpy())
    probs = np.concatenate(preds, axis=0)

    # row_id = {ss_id}_{end_second}  (consistent with pseudo_label.py)
    row_ids = [f"{ss_id}_{(i + 1) * CLIP_DUR}" for i in range(n_segs)]
    return row_ids, probs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",     default=OUTPUT_CSV)
    p.add_argument("--config",     default=CONFIG)
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--ss_dir",     default=SS_DIR)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--limit",      type=int, default=None,
                   help="Limit number of files (for testing)")
    return p.parse_args()


def main():
    args  = parse_args()
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    config = load_config(args.config)
    target_species, _ = build_species_mapping(config.data.sample_submission_csv)
    num_classes        = len(target_species)
    clip_samples       = SR * CLIP_DUR

    # ── Load Perch + trained head ─────────────────────────────────────────────
    from src.model.classifier import PerchClassifier
    print(f"Loading Perch model ({config.model.mode}) + head …")
    model = PerchClassifier(
        perch_dir             = config.model.perch_dir,
        num_classes           = num_classes,
        mode                  = config.model.mode,
        hidden_dim            = config.model.hidden_dim,
        dropout               = config.model.dropout,
        taxonomy_csv          = config.data.taxonomy_csv,
        sample_submission_csv = config.data.sample_submission_csv,
    )
    model.load_head(args.checkpoint)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Classes: {num_classes}")

    # ── Find all soundscape files ─────────────────────────────────────────────
    ogg_files = sorted(glob.glob(os.path.join(args.ss_dir, "*.ogg")))
    if args.limit:
        ogg_files = ogg_files[:args.limit]
    n_files = len(ogg_files)
    print(f"\nSoundscape files: {n_files}")

    # Resume if partial output exists
    done_ids = set()
    if os.path.exists(args.output):
        try:
            existing = pd.read_csv(args.output, usecols=["row_id"])
            done_ids = set(existing["row_id"].astype(str))
            print(f"Resuming — already processed {len(done_ids)} rows")
        except Exception:
            pass

    # ── Inference ─────────────────────────────────────────────────────────────
    cols  = ["row_id"] + [str(sp) for sp in target_species]
    chunk_rows = []
    t0    = time.time()

    for fi, filepath in enumerate(ogg_files):
        row_ids, probs = predict_file(filepath, model, clip_samples, args.batch_size)
        if row_ids is None or len(row_ids) == 0:
            continue

        for rid, prob in zip(row_ids, probs):
            if rid in done_ids:
                continue
            row = [rid] + list(prob)
            chunk_rows.append(row)

        # Periodic save
        if (fi + 1) % SAVE_EVERY == 0 or fi == n_files - 1:
            if chunk_rows:
                df_chunk = pd.DataFrame(chunk_rows, columns=cols)
                mode   = "a" if os.path.exists(args.output) else "w"
                header = not os.path.exists(args.output)
                df_chunk.to_csv(args.output, mode=mode, header=header, index=False)
                chunk_rows = []
            elapsed = time.time() - t0
            rate    = (fi + 1) / elapsed
            eta     = (n_files - fi - 1) / rate if rate > 0 else 0
            print(f"  [{fi+1}/{n_files}]  {rate:.1f} files/s  ETA: {eta/60:.0f} min")

    # Final chunk
    if chunk_rows:
        df_chunk = pd.DataFrame(chunk_rows, columns=cols)
        mode   = "a" if os.path.exists(args.output) else "w"
        header = not os.path.exists(args.output)
        df_chunk.to_csv(args.output, mode=mode, header=header, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = pd.read_csv(args.output, usecols=["row_id"])
    total_rows = len(total)
    elapsed = time.time() - t0
    print(f"\nDone! {total_rows} segments from {n_files} files → {args.output}")
    print(f"Time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
