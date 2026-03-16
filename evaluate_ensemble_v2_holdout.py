"""Ensemble holdout eval: label-head (pseudo) + label-head (soundscape) + embedding-head.

Evaluates all combinations:
  1. label-pseudo           (raw)
  2. label-soundscape-train (raw)
  3. embedding-soundscape   (raw)
  4. ensemble avg (1+2)
  5. ensemble avg (1+2+3)
  6. ensemble avg (1+2+3) + PP threshold(0.02)

Usage:
    python evaluate_ensemble_v2_holdout.py
    python evaluate_ensemble_v2_holdout.py --gpu 0
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
    ("nohuman-label-pseudo",          "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-label-soundscape-train", "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-embedding-soundscape",   "embedding_head", "embeddings_cache_nohuman"),
]
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
    holdout = pd.read_csv(holdout_csv)
    holdout_files = set(holdout["filename"].unique())
    file_to_label = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))

    mcsv = f"outputs/{cache_name}/manifest.csv"
    mf   = pd.read_csv(mcsv)
    mf   = mf[mf["source_file"].isin(holdout_files) & (mf["split"] == "holdout")].copy()
    mf["primary_label"] = mf["source_file"].map(file_to_label)
    mf   = mf.dropna(subset=["primary_label"])
    print(f"  [{cache_name}] {len(mf)} clips  files={mf['source_file'].nunique()}")

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
    print(f"Species: {num_classes}\n")

    all_preds = {}
    y_ref, species_with_pos_ref = None, None

    # ── Load each model ───────────────────────────────────────────────────────
    for run_name, mode, cache_name in RUNS:
        ckpt_path    = os.path.join(args.checkpoints_dir, run_name, "best_head")
        run_cfg_path = os.path.join(args.outputs_dir, run_name, "config.yaml")

        if not (os.path.isfile(ckpt_path + ".weights.h5") or os.path.isfile(ckpt_path)):
            print(f"[{run_name}] checkpoint not found — skipping")
            continue

        print(f"\n[{run_name}] mode={mode}  loading embeddings …")
        X, y, species_with_pos = load_holdout_embeddings(
            args.holdout_csv, cache_name, species_to_idx, num_classes
        )
        if X is None:
            print(f"  No embeddings — skipping")
            continue

        if y_ref is None:
            y_ref = y
            species_with_pos_ref = species_with_pos

        run_config = load_config(run_cfg_path) if os.path.isfile(run_cfg_path) else config
        emb_dim    = X.shape[1]

        print(f"[{run_name}] loading model …")
        model = PerchClassifier(
            perch_dir=config.model.perch_dir,
            num_classes=num_classes,
            mode=mode,
            hidden_dim=run_config.model.hidden_dim,
            dropout=0.0,
            embedding_dim=emb_dim,
        )
        model.load_head(ckpt_path)
        preds = predict(model, X)
        all_preds[run_name] = preds
        print(f"  pred_range=[{preds.min():.4f}, {preds.max():.4f}]")

        del model
        tf.keras.backend.clear_session()

    if not all_preds:
        print("No models loaded — aborting.")
        return

    # ── Ensemble combinations ─────────────────────────────────────────────────
    label_pseudo = all_preds.get("nohuman-label-pseudo")
    label_ss     = all_preds.get("nohuman-label-soundscape-train")
    emb_ss       = all_preds.get("nohuman-embedding-soundscape")

    results = []

    if label_pseudo is not None:
        results.append(("label-pseudo",          label_pseudo, "raw"))
    if label_ss is not None:
        results.append(("label-soundscape-train", label_ss,    "raw"))
    if emb_ss is not None:
        results.append(("embedding-soundscape",   emb_ss,      "raw"))

    # 2-model ensemble
    if label_pseudo is not None and label_ss is not None:
        ens_2 = (label_pseudo + label_ss) / 2.0
        results.append(("ensemble(label×2)",      ens_2, "raw"))
        results.append(("ensemble(label×2)+PP",   pp_threshold(ens_2), "+PP(0.02)"))

    # 3-model ensemble
    if label_pseudo is not None and label_ss is not None and emb_ss is not None:
        ens_3 = (label_pseudo + label_ss + emb_ss) / 3.0
        results.append(("ensemble(label×2+emb)",  ens_3, "raw"))
        results.append(("ensemble(label×2+emb)+PP", pp_threshold(ens_3), "+PP(0.02)"))

    # ── Score ─────────────────────────────────────────────────────────────────
    scored = []
    for name, preds, pp in results:
        score = roc_auc(y_ref, preds, species_with_pos_ref)
        scored.append((name, score, pp))

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'Model':<40}  {'Holdout AUC':>11}  PP")
    print(f"{'='*70}")
    best_score = max(s for _, s, _ in scored if s)
    for name, score, pp in scored:
        s      = f"{score:.4f}" if score else "  N/A"
        marker = " ★" if score and score == best_score else ""
        print(f"  {name:<40}  {s:>11}  {pp}{marker}")
    print(f"{'='*70}")
    print(f"\nBaseline (nohuman-label-pseudo, LB=0.849+PP):  holdout=0.9453")
    print(f"Prev best ensemble (label×2):                  holdout=0.9595")
    print(f"Competitor SED (best_fold0.pt, LB=0.862):      holdout=0.9883")
    print(f"\n{len(y_ref)} holdout clips  {len(species_with_pos_ref)}/234 species with positives")

    # ── Save to log ───────────────────────────────────────────────────────────
    log_path = "outputs/ensemble_v2_holdout_eval.log"
    with open(log_path, "w") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  {'Model':<40}  {'Holdout AUC':>11}  PP\n")
        f.write(f"{'='*70}\n")
        for name, score, pp in scored:
            s = f"{score:.4f}" if score else "  N/A"
            marker = " ★" if score and score == best_score else ""
            f.write(f"  {name:<40}  {s:>11}  {pp}{marker}\n")
        f.write(f"{'='*70}\n")
    print(f"\nResults saved → {log_path}")


if __name__ == "__main__":
    main()
