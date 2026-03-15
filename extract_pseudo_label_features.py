"""Extract Perch LABEL features for pseudo-labeled soundscape segments.

Reads pseudo_labels/round1_pseudo.csv, extracts 5-sec clips per row_id,
runs Perch to get 234-dim label features, saves .npy files.

Usage:
    python extract_pseudo_label_features.py --gpu 0
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.audio import load_audio
from src.data.dataset import build_species_mapping


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--pseudo_csv", default="pseudo_labels/round1_pseudo.csv")
    p.add_argument("--cache_dir", default="outputs/embeddings_cache_perch_label")
    p.add_argument("--gpu", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def parse_row_id(row_id: str):
    """Parse '{basename}_{end_sec}' → (basename, end_sec)."""
    parts = row_id.rsplit("_", 1)
    return parts[0], int(parts[1])


def get_label_indices(perch_dir, taxonomy_csv, target_species):
    labels_csv = os.path.join(perch_dir, "assets", "labels.csv")
    bc_labels = pd.read_csv(labels_csv)
    bc_labels = (bc_labels.reset_index()
                 .rename({"inat2024_fsd50k": "scientific_name", "index": "bc_index"}, axis=1)
                 .set_index("scientific_name"))
    n_perch = len(bc_labels)

    taxonomy = pd.read_csv(taxonomy_csv)
    mapping = taxonomy.join(bc_labels, on="scientific_name", how="left")
    mapping["bc_index"] = mapping["bc_index"].fillna(n_perch).astype(int)
    mapping = mapping[["primary_label", "bc_index"]].set_index("primary_label")

    indices = [int(mapping.loc[pl][0]) if pl in mapping.index else n_perch
               for pl in target_species]
    covered = sum(1 for i in indices if i < n_perch)
    print(f"Label coverage: {covered}/{len(target_species)} species")
    return indices, n_perch


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)
    sr = config.audio.sample_rate
    clip_length = config.audio.clip_duration * sr

    target_species, _ = build_species_mapping(config.data.sample_submission_csv)
    label_indices, n_perch = get_label_indices(
        config.model.perch_dir, config.data.taxonomy_csv, target_species
    )

    print(f"Loading Perch from: {config.model.perch_dir}")
    perch = tf.saved_model.load(config.model.perch_dir)

    pseudo_df = pd.read_csv(args.pseudo_csv)
    print(f"Pseudo segments: {len(pseudo_df)}")

    out_dir = os.path.join(args.cache_dir, "pseudo")
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = os.path.join(args.cache_dir, "manifest.csv")
    if os.path.isfile(manifest_path):
        existing = pd.read_csv(manifest_path)
        existing_paths = set(existing["npy_path"].tolist())
    else:
        existing = pd.DataFrame()
        existing_paths = set()

    new_rows = []
    audio_buf, meta_buf = [], []

    def flush():
        if not audio_buf:
            return
        batch = tf.constant(np.stack(audio_buf), dtype=tf.float32)
        out = perch.signatures["serving_default"](inputs=batch)
        label = tf.pad(out["label"], [[0, 0], [0, 1]])
        features = tf.gather(label, label_indices, axis=1).numpy()
        for (npy_path, src, primary_label), feat in zip(meta_buf, features):
            np.save(npy_path, feat)
            new_rows.append({
                "npy_path": npy_path,
                "source_file": src,
                "clip_idx": 0,
                "label": primary_label,
                "split": "pseudo",
            })
        audio_buf.clear()
        meta_buf.clear()

    cache = {}  # filename → audio

    for _, row in tqdm(pseudo_df.iterrows(), total=len(pseudo_df)):
        row_id = str(row["row_id"])
        primary_label = str(row.get("primary_label", "unknown"))

        basename, end_sec = parse_row_id(row_id)
        filename = basename + ".ogg"
        start_sec = end_sec - config.audio.clip_duration

        npy_name = f"{row_id}.npy"
        npy_path = os.path.join(out_dir, npy_name)
        if npy_path in existing_paths:
            continue

        if filename not in cache:
            filepath = os.path.join(config.data.train_soundscapes_dir, filename)
            cache = {filename: load_audio(filepath, sr)}

        audio = cache.get(filename)
        if audio is None:
            continue

        start_sample = max(0, int(start_sec * sr))
        clip = audio[start_sample: start_sample + clip_length]
        if len(clip) < clip_length:
            clip = np.pad(clip, (0, clip_length - len(clip)))

        audio_buf.append(clip)
        meta_buf.append((npy_path, filename, primary_label))

        if len(audio_buf) >= args.batch_size:
            flush()

    flush()

    # Append to manifest
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates("npy_path")
        combined.to_csv(manifest_path, index=False)
        print(f"\nDone. Added {len(new_rows)} pseudo label features → {manifest_path}")
        print(f"Total in manifest: {len(combined)}")
    else:
        print("No new features extracted (all already cached).")


if __name__ == "__main__":
    main()
