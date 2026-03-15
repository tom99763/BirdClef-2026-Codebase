"""Extract Perch embeddings for pseudo-labeled soundscape segments.

Reads pseudo_labels.csv (from pseudo_label.py generate), extracts Perch
embeddings for each high-confidence segment, and saves them to a cache
directory compatible with CachedEmbeddingDataset (split="pseudo").

Usage:
    python extract_pseudo_embeddings.py \
        --config configs/default.yaml \
        --pseudo_csv pseudo_labels/round1_pseudo.csv \
        --cache_dir outputs/embeddings_cache_pseudo \
        --gpu 0
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.audio import load_audio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--pseudo_csv", required=True,
                   help="pseudo_labels.csv from pseudo_label.py generate")
    p.add_argument("--cache_dir", default="outputs/embeddings_cache_pseudo")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu", default=None)
    return p.parse_args()


def find_embedding_key(perch_model) -> str:
    sig = perch_model.signatures["serving_default"]
    dummy = tf.zeros((1, 32000 * 5), dtype=tf.float32)
    outputs = sig(inputs=dummy)
    for key in ("embedding", "embeddings", "label", "logits"):
        if key in outputs:
            return key
    return next(iter(outputs.keys()))


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)
    sample_rate = config.audio.sample_rate
    clip_duration = config.audio.clip_duration
    clip_length = clip_duration * sample_rate

    # Load pseudo labels
    df = pd.read_csv(args.pseudo_csv)
    print(f"Pseudo-labeled segments: {len(df)}")

    # Load Perch
    print(f"Loading Perch from: {config.model.perch_dir}")
    perch = tf.saved_model.load(config.model.perch_dir)
    key = find_embedding_key(perch)
    sig = perch.signatures["serving_default"]
    print(f"Embedding key='{key}'")

    os.makedirs(os.path.join(args.cache_dir, "pseudo"), exist_ok=True)
    manifest_rows = []

    audio_buf, meta_buf = [], []

    def flush():
        if not audio_buf:
            return
        batch = tf.constant(np.stack(audio_buf), dtype=tf.float32)
        embs = sig(inputs=batch)[key].numpy()
        for (filename, start_sec, label), emb in zip(meta_buf, embs):
            safe_name = filename.replace("/", "__").replace("\\", "__")
            npy_name = f"{safe_name}_s{start_sec}.npy"
            npy_path = os.path.join(args.cache_dir, "pseudo", npy_name)
            np.save(npy_path, emb)
            manifest_rows.append({
                "npy_path": npy_path,
                "source_file": filename,
                "clip_idx": start_sec,
                "label": label,
                "split": "pseudo",
            })
        audio_buf.clear()
        meta_buf.clear()

    # Group by file to avoid re-loading audio
    current_file, current_audio = None, None

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_id = str(row["row_id"])
        # Parse row_id: basename_endsec → filename, start_sec
        parts = row_id.rsplit("_", 1)
        if len(parts) != 2:
            continue
        basename, end_str = parts
        try:
            end_sec = int(end_str)
        except ValueError:
            continue
        start_sec = max(0, end_sec - clip_duration)
        filename = basename + ".ogg"

        if filename != current_file:
            flush()
            current_file = filename
            filepath = os.path.join(config.data.train_soundscapes_dir, filename)
            current_audio = load_audio(filepath, sample_rate)

        if current_audio is None:
            continue

        start_sample = int(start_sec * sample_rate)
        clip = current_audio[start_sample: start_sample + clip_length]
        if len(clip) < clip_length:
            clip = np.pad(clip, (0, clip_length - len(clip)))

        audio_buf.append(clip)
        meta_buf.append((filename, start_sec, str(row.get("primary_label", "unknown"))))

        if len(audio_buf) >= args.batch_size:
            flush()

    flush()

    manifest_path = os.path.join(args.cache_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nDone. {len(manifest_rows)} pseudo embeddings saved.")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
