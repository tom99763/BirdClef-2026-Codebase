"""BirdClef 2026 — Offline Perch Embedding Extraction

Pre-computes Perch embeddings for all training clips and saves them to disk.
This allows training the classification head without re-running the heavy Perch
backbone every epoch.

Outputs:
  <cache_dir>/train/     — embeddings for clips from train_audio/
  <cache_dir>/soundscape/ — embeddings for labeled soundscape segments
  <cache_dir>/manifest.csv — maps each .npy file to its label + split

Usage:
    python extract_embeddings.py --config configs/default.yaml
    python extract_embeddings.py --config configs/default.yaml --split train --n_clips 5
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.audio import load_audio, random_crop, parse_time_str
from src.data.dataset import build_species_mapping


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Perch embeddings")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "soundscapes", "all"],
                        default="all")
    parser.add_argument("--n_clips", type=int, default=None,
                        help="Clips per file (overrides config audio.n_clips_per_file)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", default=None,
                        help="CUDA_VISIBLE_DEVICES (e.g. 0, 1, 0,1)")
    return parser.parse_args()


# ── Embedding extraction ──────────────────────────────────────────────────────

def find_embedding_key(perch_model) -> str:
    sig = perch_model.signatures["serving_default"]
    dummy = tf.zeros((1, 32000 * 5), dtype=tf.float32)
    outputs = sig(inputs=dummy)
    for key in ("embedding", "embeddings", "label", "logits"):
        if key in outputs:
            return key
    return next(iter(outputs.keys()))


def extract(perch_model, clips: np.ndarray, key: str) -> np.ndarray:
    """Run Perch on a batch of clips and return embeddings."""
    batch = tf.constant(clips, dtype=tf.float32)
    outputs = perch_model.signatures["serving_default"](inputs=batch)
    return outputs[key].numpy()


def flush(
    perch_model,
    key: str,
    audio_buf: list,
    meta_buf: list,
    out_dir: str,
    manifest_rows: list,
):
    """Extract embeddings for buffered clips, save .npy files, update manifest."""
    if not audio_buf:
        return
    embeddings = extract(perch_model, np.stack(audio_buf), key)
    for (filename, clip_idx, label, split), emb in zip(meta_buf, embeddings):
        safe_name = filename.replace("/", "__").replace("\\", "__")
        npy_name = f"{safe_name}_c{clip_idx}.npy"
        npy_path = os.path.join(out_dir, split, npy_name)
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, emb)
        manifest_rows.append({
            "npy_path": npy_path,
            "source_file": filename,
            "clip_idx": clip_idx,
            "label": label,
            "split": split,
        })
    audio_buf.clear()
    meta_buf.clear()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = load_config(args.config)

    sample_rate = config.audio.sample_rate
    clip_length = config.audio.clip_duration * sample_rate
    n_clips = args.n_clips or config.audio.n_clips_per_file
    cache_dir = config.cache.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Load Perch
    print(f"Loading Perch from: {config.model.perch_dir}")
    perch = tf.saved_model.load(config.model.perch_dir)
    key = find_embedding_key(perch)
    emb_dim = int(perch.signatures["serving_default"](
        inputs=tf.zeros((1, clip_length), tf.float32))[key].shape[-1])
    print(f"Embedding key='{key}'  dim={emb_dim}")

    manifest_rows: list = []
    audio_buf: list = []
    meta_buf: list = []
    bs = args.batch_size

    # ── train_audio ───────────────────────────────────────────────────────────
    if args.split in ("train", "all"):
        print("\nExtracting train_audio embeddings …")
        train_df = pd.read_csv(config.data.train_csv)

        for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
            filepath = os.path.join(config.data.train_audio_dir, str(row["filename"]))
            audio = load_audio(filepath, sample_rate)
            if audio is None:
                continue
            for clip_idx in range(n_clips):
                clip = random_crop(audio, clip_length)
                audio_buf.append(clip)
                meta_buf.append((str(row["filename"]), clip_idx,
                                 str(row["primary_label"]), "train"))
                if len(audio_buf) >= bs:
                    flush(perch, key, audio_buf, meta_buf, cache_dir, manifest_rows)

        flush(perch, key, audio_buf, meta_buf, cache_dir, manifest_rows)

    # ── soundscapes ───────────────────────────────────────────────────────────
    if args.split in ("soundscapes", "all"):
        print("\nExtracting soundscape embeddings …")
        labels_df = pd.read_csv(config.data.soundscapes_labels_csv)

        current_file, current_audio = None, None
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
            filename = str(row["filename"])
            if filename != current_file:
                current_file = filename
                filepath = os.path.join(config.data.train_soundscapes_dir, filename)
                current_audio = load_audio(filepath, sample_rate)

            if current_audio is None:
                continue

            start_sec = parse_time_str(str(row["start"]))
            start_sample = int(start_sec * sample_rate)
            clip = current_audio[start_sample : start_sample + clip_length]
            if len(clip) < clip_length:
                clip = np.pad(clip, (0, clip_length - len(clip)))

            audio_buf.append(clip)
            meta_buf.append((filename, int(start_sec),
                              str(row["primary_label"]), "soundscape"))
            if len(audio_buf) >= bs:
                flush(perch, key, audio_buf, meta_buf, cache_dir, manifest_rows)

        flush(perch, key, audio_buf, meta_buf, cache_dir, manifest_rows)

    # ── Save manifest ─────────────────────────────────────────────────────────
    manifest_path = os.path.join(cache_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nDone. {len(manifest_rows)} embeddings saved.")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
