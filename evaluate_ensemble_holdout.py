"""Ensemble holdout evaluation: nohuman-label-pseudo + nohuman-label-soundscape-train.

Evaluates 4 combinations:
  1. nohuman-label-pseudo          (raw)
  2. nohuman-label-soundscape-train (raw)
  3. ensemble average              (raw)
  4. ensemble average              (+PP: threshold 0.02)

Note: slide_max PP is not applied here because holdout clips are individual
recordings with no temporal ordering — it only makes sense for soundscapes.

Usage:
    python evaluate_ensemble_holdout.py
    python evaluate_ensemble_holdout.py --gpu 1
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from src.utils.config import load_config
from src.data.dataset import build_species_mapping
from src.model.classifier import PerchClassifier


RUNS = [
    "nohuman-label-pseudo",
    "nohuman-label-soundscape-train",
]
CACHE_NAME  = "embeddings_cache_nohuman_label"
HOLDOUT_CSV = "configs/holdout_val_files.csv"
CONFIG      = "configs/default.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",            default=None)
    p.add_argument("--holdout_csv",    default=HOLDOUT_CSV)
    p.add_argument("--config",         default=CONFIG)
    p.add_argument("--checkpoints_dir",default="checkpoints")
    p.add_argument("--outputs_dir",    default="outputs")
    return p.parse_args()


def load_holdout_embeddings(holdout_csv, cache_name, species_to_idx, num_classes):
    """Load holdout embeddings and ground-truth labels."""
    holdout = pd.read_csv(holdout_csv)
    holdout_files = set(holdout["filename"].unique())
    file_to_label = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))

    mcsv = f"outputs/{cache_name}/manifest.csv"
    mf   = pd.read_csv(mcsv)
    mf   = mf[mf["source_file"].isin(holdout_files) & (mf["split"] == "holdout")].copy()
    mf["primary_label"] = mf["source_file"].map(file_to_label)
    mf   = mf.dropna(subset=["primary_label"])
    print(f"  [{cache_name}] {len(mf)} clips from {mf['source_file'].nunique()} files")

    embs, labs = [], []
    for _, row in mf.iterrows():
        if not os.path.isfile(row["npy_path"]):
            continue
        embs.append(np.load(row["npy_path"]))
        labs.append(str(row["primary_label"]))

    X = np.stack(embs).astype(np.float32)
    y = np.zeros((len(labs), num_classes), dtype=np.float32)
    for i, sp in enumerate(labs):
        if sp in species_to_idx:
            y[i, species_to_idx[sp]] = 1.0

    species_with_pos = np.where(y.sum(0) > 0)[0]
    print(f"  shape={X.shape}  species_with_pos={len(species_with_pos)}/234")
    return X, y, species_with_pos


def predict(model, X, batch_size=512):
    """Run model.head on pre-extracted embeddings, return sigmoid probs."""
    preds = []
    for start in range(0, len(X), batch_size):
        batch  = tf.constant(X[start: start + batch_size])
        logits = model.head(batch, training=False)
        out    = logits[0] if isinstance(logits, tuple) else logits
        preds.append(tf.sigmoid(out).numpy())
    return np.concatenate(preds, axis=0)


def roc_auc(y, preds, species_with_pos):
    try:
        return roc_auc_score(
            y[:, species_with_pos],
            preds[:, species_with_pos],
            average="macro",
        )
    except Exception as e:
        print(f"  Scoring error: {e}")
        return None


def pp_threshold(preds, threshold=0.02):
    """Zero out predictions below threshold."""
    out = preds.copy()
    out[out < threshold] = 0.0
    return out


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)
    target_species, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(target_species)
    print(f"Species: {num_classes}")

    # ── Load holdout data ─────────────────────────────────────────────────────
    print(f"\nLoading holdout embeddings …")
    X, y, species_with_pos = load_holdout_embeddings(
        args.holdout_csv, CACHE_NAME, species_to_idx, num_classes
    )

    # ── Predict with each run ─────────────────────────────────────────────────
    all_preds = {}
    for run_name in RUNS:
        ckpt_path     = os.path.join(args.checkpoints_dir, run_name, "best_head")
        run_cfg_path  = os.path.join(args.outputs_dir, run_name, "config.yaml")
        run_config    = load_config(run_cfg_path) if os.path.isfile(run_cfg_path) else config

        if not (os.path.isfile(ckpt_path + ".weights.h5") or os.path.isfile(ckpt_path)):
            print(f"\n[{run_name}] checkpoint not found — skipping")
            continue

        print(f"\n[{run_name}] loading …")
        model = PerchClassifier(
            perch_dir=config.model.perch_dir,
            num_classes=num_classes,
            mode="label_head",
            hidden_dim=run_config.model.hidden_dim,
            dropout=run_config.model.dropout,
            embedding_dim=X.shape[1],
        )
        model.load_head(ckpt_path)
        preds = predict(model, X)
        all_preds[run_name] = preds
        print(f"  [{run_name}] done  pred_range=[{preds.min():.4f}, {preds.max():.4f}]")

        del model
        tf.keras.backend.clear_session()

    if len(all_preds) < 2:
        print("\nNeed both checkpoints for ensemble — aborting.")
        return

    # ── Ensemble ──────────────────────────────────────────────────────────────
    preds_a = all_preds[RUNS[0]]
    preds_b = all_preds[RUNS[1]]
    preds_ens = (preds_a + preds_b) / 2.0
    preds_ens_pp = pp_threshold(preds_ens, threshold=0.02)

    # ── Score all 4 combinations ──────────────────────────────────────────────
    results = [
        (RUNS[0],                    roc_auc(y, preds_a,      species_with_pos), "raw"),
        (RUNS[1],                    roc_auc(y, preds_b,      species_with_pos), "raw"),
        ("ensemble (avg)",           roc_auc(y, preds_ens,    species_with_pos), "raw"),
        ("ensemble (avg) + PP",      roc_auc(y, preds_ens_pp, species_with_pos), "+threshold(0.02)"),
    ]

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {'Model':<38}  {'Holdout ROC-AUC':>14}  PP")
    print(f"{'='*65}")
    for name, score, pp in results:
        s = f"{score:.4f}" if score else "  N/A"
        marker = " ★" if score and score == max(r[1] for r in results if r[1]) else ""
        print(f"  {name:<38}  {s:>14}  {pp}{marker}")
    print(f"{'='*65}")
    print(f"\nBaseline (nohuman-label-pseudo, confirmed LB=0.849+PP): holdout=0.9453")
    print(f"Holdout ↔ LB gap ≈ 0.096  (individual recordings vs soundscape domain)")
    print(f"\nAll {len(y)} holdout clips, {len(species_with_pos)}/234 species with positives.")


if __name__ == "__main__":
    main()
