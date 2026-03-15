"""
Final evaluation using the official BirdCLEF 2026 competition metric.

For every experiment checkpoint found in checkpoints/, this script:
  1. Runs inference on train_soundscapes (or a specified soundscapes directory)
  2. Normalises predictions to row-sum = 1  (required by the official metric)
  3. Builds ground-truth from train_soundscapes_labels.csv
  4. Scores with the official padded cMAP via src/metrics/kaggle_metric.score()
  5. Writes the score to outputs/<run_name>/kaggle_score.json
  6. Prints a ranked summary table

Usage:
    python evaluate_final.py
    python evaluate_final.py --config configs/default.yaml
    python evaluate_final.py --soundscapes_dir birdclef-2026/train_soundscapes
    python evaluate_final.py --runs baseline birdclef25-base   # specific runs only
    python evaluate_final.py --gpu 0
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils.config import load_config
from src.utils.audio import load_audio
from src.data.dataset import build_species_mapping
from src.metrics.kaggle_metric import score as kaggle_score


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Final evaluation — official BirdCLEF metric")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--soundscapes_dir", default=None,
                   help="Override soundscapes directory (default: config value)")
    p.add_argument("--labels_csv", default=None,
                   help="Override ground-truth labels CSV")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--runs", nargs="*", default=None,
                   help="Evaluate only these run names (default: all)")
    p.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES")
    return p.parse_args()


# ── Ground-truth builder ──────────────────────────────────────────────────────

def _end_to_seconds(end_str: str) -> int:
    """Convert HH:MM:SS to total seconds."""
    h, m, s = end_str.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def build_ground_truth(
    labels_csv: str,
    target_species: list,
) -> pd.DataFrame:
    """Parse train_soundscapes_labels.csv → binary solution DataFrame.

    Row IDs are built to match inference.py output:
        <filename_without_ext>_<end_seconds>
    e.g. BC2026_Train_0039_S22_20211231_201500_5
    """
    df = pd.read_csv(labels_csv)
    species_set = set(target_species)

    rows = []
    for _, row in df.iterrows():
        fname = re.sub(r"\.ogg$", "", row["filename"], flags=re.IGNORECASE)
        end_sec = _end_to_seconds(str(row["end"]))
        row_id = f"{fname}_{end_sec}"

        label_vec = np.zeros(len(target_species), dtype=np.float32)
        for code in str(row["primary_label"]).split(";"):
            code = code.strip()
            if code in species_set:
                label_vec[target_species.index(code)] = 1.0

        rows.append([row_id] + label_vec.tolist())

    solution = pd.DataFrame(rows, columns=["row_id"] + target_species)
    # Aggregate duplicate row_ids (same time window, multiple label rows) by taking max
    solution = solution.groupby("row_id", sort=False).max().reset_index()
    return solution


# ── Inference helpers ─────────────────────────────────────────────────────────

def _batched_infer(model, clips: np.ndarray, batch_size: int) -> np.ndarray:
    all_preds = []
    for start in range(0, len(clips), batch_size):
        batch = tf.constant(clips[start: start + batch_size], dtype=tf.float32)
        logits = model(batch, training=False)
        all_preds.append(tf.sigmoid(logits).numpy())
    return np.concatenate(all_preds, axis=0)


def run_inference(
    model,
    ogg_files: list,
    sample_rate: int,
    clip_duration: int,
    batch_size: int,
) -> pd.DataFrame:
    """Run inference on all soundscape files, return raw (un-normalised) preds."""
    all_row_ids, all_preds = [], []
    for filepath in ogg_files:
        audio = load_audio(filepath, sample_rate)
        if audio is None:
            continue
        clip_length = clip_duration * sample_rate
        n_segments = len(audio) // clip_length
        if n_segments == 0:
            continue

        clips = np.stack([
            audio[i * clip_length: (i + 1) * clip_length]
            for i in range(n_segments)
        ])
        ss_id = re.sub(r"\.ogg$", "", os.path.basename(filepath), flags=re.IGNORECASE)
        row_ids = [f"{ss_id}_{(i + 1) * clip_duration}" for i in range(n_segments)]

        preds = _batched_infer(model, clips, batch_size)
        all_row_ids.extend(row_ids)
        all_preds.append(preds)

    if not all_preds:
        return pd.DataFrame()
    predictions = np.concatenate(all_preds, axis=0)
    return predictions, all_row_ids



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)

    soundscapes_dir = args.soundscapes_dir or config.data.train_soundscapes_dir
    labels_csv = args.labels_csv or config.data.soundscapes_labels_csv

    # Only evaluate on files that have ground-truth labels (avoids scanning all 10k+ files)
    labels_df = pd.read_csv(labels_csv)
    labelled_filenames = set(labels_df["filename"].unique())
    all_ogg_files = sorted(glob.glob(os.path.join(soundscapes_dir, "*.ogg")))
    ogg_files = [f for f in all_ogg_files if os.path.basename(f) in labelled_filenames]
    if not ogg_files:
        print(f"ERROR: No labelled .ogg files found in {soundscapes_dir}")
        return
    print(f"Soundscapes: {len(ogg_files)} labelled files (of {len(all_ogg_files)} total) in {soundscapes_dir}")

    target_species, species_to_idx = build_species_mapping(
        config.data.sample_submission_csv
    )
    num_classes = len(target_species)
    print(f"Species: {num_classes}")

    # Build ground truth once
    print(f"Building ground truth from {labels_csv} …")
    solution = build_ground_truth(labels_csv, target_species)
    print(f"  {len(solution)} labelled segments")

    # Discover run checkpoints
    if args.runs:
        run_names = args.runs
    else:
        run_names = sorted([
            os.path.basename(d)
            for d in glob.glob(os.path.join(args.checkpoints_dir, "*"))
            if os.path.isdir(d)
        ])
    print(f"\nRuns to evaluate: {run_names}\n")

    results = []

    for run_name in run_names:
        ckpt_path = os.path.join(args.checkpoints_dir, run_name, "best_head")
        if not (
            os.path.isfile(ckpt_path + ".weights.h5")
            or os.path.isfile(ckpt_path)
        ):
            print(f"[{run_name}] checkpoint not found, skipping.")
            continue

        print(f"[{run_name}] Loading model …")
        # Use run-specific config if available (handles varying hidden_dim etc.)
        run_config_path = os.path.join(args.outputs_dir, run_name, "config.yaml")
        run_config = load_config(run_config_path) if os.path.isfile(run_config_path) else config

        from src.model.classifier import PerchClassifier
        model = PerchClassifier(
            perch_dir=config.model.perch_dir,
            num_classes=num_classes,
            mode=run_config.model.mode,
            hidden_dim=run_config.model.hidden_dim,
            dropout=run_config.model.dropout,
            # label_head mode needs taxonomy mapping to extract Perch species features
            taxonomy_csv=config.data.get("taxonomy_csv", None),
            sample_submission_csv=config.data.get("sample_submission_csv", None),
        )
        model.load_head(ckpt_path)

        print(f"[{run_name}] Running inference on {len(ogg_files)} soundscapes …")
        result = run_inference(
            model=model,
            ogg_files=ogg_files,
            sample_rate=config.audio.sample_rate,
            clip_duration=config.audio.clip_duration,
            batch_size=args.batch_size,
        )
        if isinstance(result, pd.DataFrame):
            print(f"[{run_name}] No predictions generated, skipping.")
            continue
        predictions_raw, row_ids = result

        submission = pd.DataFrame(predictions_raw, columns=target_species)
        submission.insert(0, "row_id", row_ids)

        # Save submission CSV
        sub_dir = os.path.join(args.outputs_dir, run_name)
        os.makedirs(sub_dir, exist_ok=True)
        sub_path = os.path.join(sub_dir, "submission_train_soundscapes.csv")
        submission.to_csv(sub_path, index=False)

        # Score — align rows before passing to scorer (scorer does del in-place)
        sol_aligned = solution.copy()
        sub_aligned = submission.copy()
        common = sol_aligned["row_id"].isin(sub_aligned["row_id"])
        sol_aligned = sol_aligned[common].reset_index(drop=True)
        sub_aligned = sub_aligned[
            sub_aligned["row_id"].isin(sol_aligned["row_id"])
        ].reset_index(drop=True)

        try:
            roc_auc = kaggle_score(sol_aligned, sub_aligned, row_id_column_name="row_id")
        except Exception as e:
            print(f"[{run_name}] Scoring error: {e}")
            roc_auc = None

        score_data = {
            "run_name": run_name,
            "kaggle_roc_auc": roc_auc,
            "n_soundscape_segments": len(row_ids),
        }

        # Merge with existing result.json if present
        result_path = os.path.join(sub_dir, "result.json")
        if os.path.isfile(result_path):
            with open(result_path) as f:
                existing = json.load(f)
            existing["kaggle_roc_auc"] = roc_auc
            with open(result_path, "w") as f:
                json.dump(existing, f, indent=2)
        else:
            with open(os.path.join(sub_dir, "kaggle_score.json"), "w") as f:
                json.dump(score_data, f, indent=2)

        print(f"[{run_name}] Official ROC-AUC = {roc_auc:.4f}" if roc_auc is not None
              else f"[{run_name}] Score unavailable")
        results.append(score_data)

        # Free model memory between runs
        del model
        tf.keras.backend.clear_session()

    # ── Ranked summary ────────────────────────────────────────────────────────
    if results:
        print("\n" + "=" * 55)
        print(f"  {'Run':<30} {'Official cMAP':>13}")
        print("=" * 55)
        for r in sorted(results,
                        key=lambda x: x["kaggle_roc_auc"] or 0,
                        reverse=True):
            cmap_str = f"{r['kaggle_roc_auc']:.4f}" if r["kaggle_roc_auc"] is not None else "  N/A"
            print(f"  {r['run_name']:<30} {cmap_str:>13}")
        print("=" * 55)
        print(f"\nScores written to outputs/<run>/result.json  (key: kaggle_roc_auc)")
        print("Metric: macro-averaged ROC-AUC over species with ≥1 positive label (official BirdCLEF 2026)")


if __name__ == "__main__":
    main()
