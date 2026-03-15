"""Extract Perch LABEL features (not embeddings) for training a calibration head.

Instead of the 1536-dim embedding, this script extracts Perch's raw `label`
output (logits over ~10k species) indexed to our 234 target species.

These features preserve Perch's pre-trained species knowledge and allow
training a small calibration head that improves on Perch's zero-shot predictions.

Usage:
    python extract_perch_labels.py --config configs/default.yaml
    python extract_perch_labels.py --config configs/default.yaml --split train --gpu 0
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["train", "soundscapes", "all"], default="all")
    p.add_argument("--n_clips", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu", default=None)
    p.add_argument("--cache_dir", default=None,
                   help="Override cache dir (default: config.cache.cache_dir + '_perch_label')")
    return p.parse_args()


def get_label_indices(perch_dir: str, taxonomy_csv: str, target_species: list) -> list:
    """Map target species → Perch label indices via scientific names."""
    labels_csv = os.path.join(perch_dir, "assets", "labels.csv")
    bc_labels = pd.read_csv(labels_csv)
    bc_labels = (bc_labels.reset_index()
                 .rename({"inat2024_fsd50k": "scientific_name", "index": "bc_index"}, axis=1)
                 .set_index("scientific_name"))

    taxonomy = pd.read_csv(taxonomy_csv)
    mapping = taxonomy.join(bc_labels, on="scientific_name", how="left")
    mapping["bc_index"] = mapping["bc_index"].fillna(len(bc_labels)).astype(int)
    mapping = mapping[["primary_label", "bc_index"]].set_index("primary_label")

    n_perch = len(bc_labels)
    indices = [int(mapping.loc[pl][0]) if pl in mapping.index else n_perch
               for pl in target_species]
    covered = sum(1 for i in indices if i < n_perch)
    print(f"Perch label coverage: {covered}/{len(target_species)} species")
    return indices, n_perch


def extract_label_features(perch_model, clips: np.ndarray, label_indices: list, n_perch: int) -> np.ndarray:
    """Run Perch, extract label output, index to target species."""
    batch = tf.constant(clips, dtype=tf.float32)
    out = perch_model.signatures["serving_default"](inputs=batch)
    # Pad so OOV index (n_perch) → 0
    label = tf.pad(out["label"], [[0, 0], [0, 1]])
    features = tf.gather(label, label_indices, axis=1).numpy()
    return features  # (N, 234)


def flush(perch_model, label_indices, n_perch, audio_buf, meta_buf, out_dir, manifest_rows):
    if not audio_buf:
        return
    features = extract_label_features(perch_model, np.stack(audio_buf), label_indices, n_perch)
    for (filename, clip_idx, label, split), feat in zip(meta_buf, features):
        safe_name = filename.replace("/", "__").replace("\\", "__")
        npy_name = f"{safe_name}_c{clip_idx}.npy"
        npy_path = os.path.join(out_dir, split, npy_name)
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, feat)
        manifest_rows.append({
            "npy_path": npy_path,
            "source_file": filename,
            "clip_idx": clip_idx,
            "label": label,
            "split": split,
        })
    audio_buf.clear()
    meta_buf.clear()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)
    sample_rate = config.audio.sample_rate
    clip_length = config.audio.clip_duration * sample_rate
    n_clips = args.n_clips or config.audio.n_clips_per_file

    cache_dir = args.cache_dir or (config.cache.cache_dir.rstrip("/") + "_perch_label")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache dir: {cache_dir}")

    target_species, _ = build_species_mapping(config.data.sample_submission_csv)

    print(f"Loading Perch from: {config.model.perch_dir}")
    perch = tf.saved_model.load(config.model.perch_dir)

    label_indices, n_perch = get_label_indices(
        config.model.perch_dir, config.data.taxonomy_csv, target_species
    )

    manifest_path = os.path.join(cache_dir, "manifest.csv")
    if os.path.isfile(manifest_path):
        manifest_rows = pd.read_csv(manifest_path).to_dict("records")
        print(f"Resuming: {len(manifest_rows)} existing entries")
    else:
        manifest_rows = []

    audio_buf, meta_buf = [], []
    bs = args.batch_size

    # ── train_audio ──────────────────────────────────────────────────────────
    if args.split in ("train", "all"):
        print("\nExtracting train_audio label features …")
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
                    flush(perch, label_indices, n_perch, audio_buf, meta_buf, cache_dir, manifest_rows)
        flush(perch, label_indices, n_perch, audio_buf, meta_buf, cache_dir, manifest_rows)

    # ── soundscapes ──────────────────────────────────────────────────────────
    if args.split in ("soundscapes", "all"):
        print("\nExtracting soundscape label features …")
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
            clip = current_audio[start_sample: start_sample + clip_length]
            if len(clip) < clip_length:
                clip = np.pad(clip, (0, clip_length - len(clip)))
            audio_buf.append(clip)
            meta_buf.append((filename, int(start_sec), str(row["primary_label"]), "soundscape"))
            if len(audio_buf) >= bs:
                flush(perch, label_indices, n_perch, audio_buf, meta_buf, cache_dir, manifest_rows)
        flush(perch, label_indices, n_perch, audio_buf, meta_buf, cache_dir, manifest_rows)

    df = pd.DataFrame(manifest_rows).drop_duplicates(subset="npy_path")
    df.to_csv(manifest_path, index=False)
    print(f"\nDone. {len(df)} label features saved → {manifest_path}")


if __name__ == "__main__":
    main()
