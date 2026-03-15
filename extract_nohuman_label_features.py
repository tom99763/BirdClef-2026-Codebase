"""Extract Perch LABEL features from human-voice-filtered audio.

Combines Silero VAD human speech removal with Perch label feature extraction.
Saves 234-dim .npy files to outputs/embeddings_cache_nohuman_label/.

Usage:
    python extract_nohuman_label_features.py --config configs/default.yaml --gpu 0
    python extract_nohuman_label_features.py --split train --gpu 0
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
from src.audio.human_filter import SpeechFilter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["train", "soundscapes", "all"], default="all")
    p.add_argument("--n_clips", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu", default=None)
    p.add_argument("--vad_threshold", type=float, default=0.4)
    p.add_argument("--cache_dir", default="outputs/embeddings_cache_nohuman_label")
    return p.parse_args()


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
    print(f"Perch label coverage: {covered}/{len(target_species)} species")
    return indices, n_perch


def extract_label_features(perch_model, clips, label_indices, n_perch):
    batch = tf.constant(clips, dtype=tf.float32)
    out = perch_model.signatures["serving_default"](inputs=batch)
    label = tf.pad(out["label"], [[0, 0], [0, 1]])
    return tf.gather(label, label_indices, axis=1).numpy()


def flush(perch, label_indices, n_perch, audio_buf, meta_buf, out_dir, manifest_rows):
    if not audio_buf:
        return
    features = extract_label_features(perch, np.stack(audio_buf), label_indices, n_perch)
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
    sr = config.audio.sample_rate
    clip_length = config.audio.clip_duration * sr
    n_clips = args.n_clips or config.audio.n_clips_per_file

    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"Cache dir: {args.cache_dir}")

    target_species, _ = build_species_mapping(config.data.sample_submission_csv)
    label_indices, n_perch = get_label_indices(
        config.model.perch_dir, config.data.taxonomy_csv, target_species
    )

    print(f"Loading Perch from: {config.model.perch_dir}")
    perch = tf.saved_model.load(config.model.perch_dir)

    print(f"Loading SpeechFilter (VAD threshold={args.vad_threshold}) …")
    voice_filter = SpeechFilter(threshold=args.vad_threshold)

    manifest_path = os.path.join(args.cache_dir, "manifest.csv")
    if os.path.isfile(manifest_path):
        manifest_rows = pd.read_csv(manifest_path).to_dict("records")
        existing_paths = {r["npy_path"] for r in manifest_rows}
        print(f"Resuming: {len(manifest_rows)} existing entries")
    else:
        manifest_rows = []
        existing_paths = set()

    audio_buf, meta_buf = [], []
    bs = args.batch_size

    def maybe_filter(audio):
        start, end = voice_filter.find_clean_window(audio, sr=sr)
        return audio[start:end] if (end - start) > clip_length // 2 else audio

    # ── train_audio ──────────────────────────────────────────────────────────
    if args.split in ("train", "all"):
        print("\nExtracting train_audio label features (nohuman) …")
        train_df = pd.read_csv(config.data.train_csv)
        for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
            filename = str(row["filename"])
            safe_name = filename.replace("/", "__").replace("\\", "__")
            # Check if any clip for this file already exists
            if any(f"{safe_name}_c0.npy" in p for p in existing_paths):
                continue
            filepath = os.path.join(config.data.train_audio_dir, filename)
            audio = load_audio(filepath, sr)
            if audio is None:
                continue
            audio = maybe_filter(audio)
            if len(audio) < clip_length:
                audio = np.pad(audio, (0, clip_length - len(audio)))
            for clip_idx in range(n_clips):
                clip = random_crop(audio, clip_length)
                audio_buf.append(clip)
                meta_buf.append((filename, clip_idx, str(row["primary_label"]), "train"))
                if len(audio_buf) >= bs:
                    flush(perch, label_indices, n_perch, audio_buf, meta_buf,
                          args.cache_dir, manifest_rows)
        flush(perch, label_indices, n_perch, audio_buf, meta_buf,
              args.cache_dir, manifest_rows)

    # ── soundscapes ──────────────────────────────────────────────────────────
    if args.split in ("soundscapes", "all"):
        print("\nExtracting soundscape label features (nohuman) …")
        labels_df = pd.read_csv(config.data.soundscapes_labels_csv)
        current_file, current_audio = None, None
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
            filename = str(row["filename"])
            if filename != current_file:
                current_file = filename
                filepath = os.path.join(config.data.train_soundscapes_dir, filename)
                raw = load_audio(filepath, sr)
                current_audio = maybe_filter(raw) if raw is not None else None
            if current_audio is None:
                continue
            start_sec = parse_time_str(str(row["start"]))
            start_sample = int(start_sec * sr)
            clip = current_audio[start_sample: start_sample + clip_length]
            if len(clip) < clip_length:
                clip = np.pad(clip, (0, clip_length - len(clip)))
            audio_buf.append(clip)
            meta_buf.append((filename, int(start_sec), str(row["primary_label"]), "soundscape"))
            if len(audio_buf) >= bs:
                flush(perch, label_indices, n_perch, audio_buf, meta_buf,
                      args.cache_dir, manifest_rows)
        flush(perch, label_indices, n_perch, audio_buf, meta_buf,
              args.cache_dir, manifest_rows)

    df = pd.DataFrame(manifest_rows).drop_duplicates(subset="npy_path")
    df.to_csv(manifest_path, index=False)
    print(f"\nDone. {len(df)} nohuman label features saved → {manifest_path}")


if __name__ == "__main__":
    main()
