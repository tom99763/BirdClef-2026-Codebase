"""Evaluate Perch label-head models on the held-out train_audio validation set.

Uses configs/holdout_val_files.csv (7,037 files, 206 species, 20% stratified split).
Embeddings are read directly from the pre-computed nohuman cache — no GPU needed.

This metric is NOT leaked (holdout files were excluded from all training runs)
and covers 206/234 species vs the 66-soundscape set's 75 species.

Usage:
    python evaluate_holdout.py
    python evaluate_holdout.py --runs nohuman-label-pseudo nohuman-os3-pseudo-r5
    python evaluate_holdout.py --mode label_head   # for label-head models
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils.config import load_config
from src.data.dataset import build_species_mapping
from src.metrics.kaggle_metric import score as kaggle_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout_csv",    default="configs/holdout_val_files.csv")
    p.add_argument("--config",         default="configs/default.yaml")
    p.add_argument("--checkpoints_dir",default="checkpoints")
    p.add_argument("--outputs_dir",    default="outputs")
    p.add_argument("--runs",           nargs="*", default=None)
    p.add_argument("--gpu",            default=None)
    return p.parse_args()


def load_embeddings_for_files(
    file_list: list,
    manifest_csv: str,
    label_col: str = "label",
) -> tuple:
    """Load cached embeddings for a list of source_files.

    Returns (embeddings, source_files_found, labels_per_clip).
    """
    manifest = pd.read_csv(manifest_csv)
    # Keep only files in our holdout list
    manifest = manifest[manifest["source_file"].isin(set(file_list))]

    embeddings, primary_labels = [], []
    for _, row in manifest.iterrows():
        npy_path = row["npy_path"]
        if not os.path.isfile(npy_path):
            continue
        emb = np.load(npy_path)           # (1536,) or (D,)
        embeddings.append(emb)
        primary_labels.append(str(row[label_col]))

    if not embeddings:
        return None, None, None
    return (
        np.stack(embeddings).astype(np.float32),
        np.array(primary_labels),
    )


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)
    target_species, species_to_idx = build_species_mapping(
        config.data.sample_submission_csv
    )
    num_classes = len(target_species)
    print(f"Species: {num_classes}")

    # ── Load holdout file list ────────────────────────────────────────────────
    holdout = pd.read_csv(args.holdout_csv)
    print(f"Holdout files: {len(holdout)}  species: {holdout['primary_label'].nunique()}")

    # ── Build ground-truth labels (load both caches lazily) ──────────────────
    holdout_files = set(holdout["filename"].unique())
    file_to_label = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))

    def _load_holdout_X_y(cache_name: str):
        """Load embeddings from the given cache for holdout files."""
        mcsv = f"outputs/{cache_name}/manifest.csv"
        if not os.path.isfile(mcsv):
            print(f"  Cache manifest not found: {mcsv}")
            return None, None, None
        mf = pd.read_csv(mcsv)
        mf = mf[mf["source_file"].isin(holdout_files) & (mf["split"] == "holdout")].copy()
        mf["primary_label"] = mf["source_file"].map(file_to_label)
        mf = mf.dropna(subset=["primary_label"])
        print(f"  [{cache_name}] {len(mf)} clips from {mf['source_file'].nunique()} files")
        embs, labs = [], []
        for _, row in mf.iterrows():
            npy = row["npy_path"]
            if not os.path.isfile(npy):
                continue
            embs.append(np.load(npy))
            labs.append(str(row["primary_label"]))
        if not embs:
            return None, None, None
        X_ = np.stack(embs).astype(np.float32)
        y_ = np.zeros((len(labs), num_classes), dtype=np.float32)
        for i, sp in enumerate(labs):
            if sp in species_to_idx:
                y_[i, species_to_idx[sp]] = 1.0
        pos = np.where(y_.sum(0) > 0)[0]
        print(f"    shape={X_.shape}  species_with_pos={len(pos)}/234")
        return X_, y_, pos

    # Cache both modes so we don't re-load for multiple runs of the same mode
    _cache = {}

    def get_Xy(mode: str):
        cache_name = "embeddings_cache_nohuman_label" if mode == "label_head" \
                     else "embeddings_cache_nohuman"
        if cache_name not in _cache:
            print(f"Loading holdout embeddings from {cache_name} …")
            _cache[cache_name] = _load_holdout_X_y(cache_name)
        return _cache[cache_name]

    # ── Discover runs ─────────────────────────────────────────────────────────
    if args.runs:
        run_names = args.runs
    else:
        run_names = sorted([
            os.path.basename(d)
            for d in glob.glob(os.path.join(args.checkpoints_dir, "*"))
            if os.path.isdir(d)
        ])

    results = []
    print(f"\nRuns to evaluate: {run_names}\n")

    for run_name in run_names:
        ckpt_path = os.path.join(args.checkpoints_dir, run_name, "best_head")
        if not (
            os.path.isfile(ckpt_path + ".weights.h5") or
            os.path.isfile(ckpt_path)
        ):
            print(f"[{run_name}] checkpoint not found, skipping.")
            continue

        run_config_path = os.path.join(args.outputs_dir, run_name, "config.yaml")
        run_config = (
            load_config(run_config_path)
            if os.path.isfile(run_config_path)
            else config
        )

        mode = run_config.model.mode
        if mode not in ("label_head", "embedding_head"):
            print(f"[{run_name}] mode={mode} — skipping (only label/embedding head supported)")
            continue

        print(f"[{run_name}] mode={mode}  loading …")

        X, y, species_with_pos = get_Xy(mode)
        if X is None:
            print(f"[{run_name}] No embeddings found for mode={mode}, skipping.")
            continue

        emb_dim = X.shape[1]

        from src.model.classifier import PerchClassifier
        model = PerchClassifier(
            perch_dir=config.model.perch_dir,
            num_classes=num_classes,
            mode=mode,
            hidden_dim=run_config.model.hidden_dim,
            dropout=run_config.model.dropout,
            embedding_dim=emb_dim,  # cache mode: skip Perch backbone
        )
        model.load_head(ckpt_path)

        # Run head on pre-extracted embeddings
        batch_size = 512
        preds_list = []
        for start in range(0, len(X), batch_size):
            batch = tf.constant(X[start: start + batch_size])
            logits = model.head(batch, training=False)
            preds_list.append(tf.sigmoid(logits).numpy())
        preds = np.concatenate(preds_list, axis=0)

        # Compute macro ROC-AUC only over species with positives in holdout
        from sklearn.metrics import roc_auc_score
        try:
            roc = roc_auc_score(
                y[:, species_with_pos],
                preds[:, species_with_pos],
                average="macro",
            )
        except Exception as e:
            print(f"  Scoring error: {e}")
            roc = None

        print(f"[{run_name}] holdout ROC-AUC = {roc:.4f}" if roc else f"[{run_name}] N/A")
        results.append({"run": run_name, "holdout_roc_auc": roc, "mode": mode})

        # Save to result.json
        result_path = os.path.join(args.outputs_dir, run_name, "result.json")
        if os.path.isfile(result_path):
            with open(result_path) as f:
                existing = json.load(f)
            existing["holdout_roc_auc"] = roc
            with open(result_path, "w") as f:
                json.dump(existing, f, indent=2)

        del model
        tf.keras.backend.clear_session()

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        print("\n" + "=" * 55)
        print(f"  {'Run':<35} {'Holdout ROC-AUC':>15}")
        print("=" * 55)
        for r in sorted(results, key=lambda x: x["holdout_roc_auc"] or 0, reverse=True):
            s = f"{r['holdout_roc_auc']:.4f}" if r["holdout_roc_auc"] else "  N/A"
            print(f"  {r['run']:<35} {s:>15}")
        print("=" * 55)
        print("\nThis metric uses 7,037 holdout files (206/234 species), NO data leak.")
        print("It should rank models better than the 66-soundscape metric.")


if __name__ == "__main__":
    main()
