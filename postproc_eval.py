"""
Post-Processing Evaluation — BirdCLEF 2026
===========================================
Standalone script; does NOT modify any existing file.

Steps
-----
1. Run inference on the 66 labelled validation soundscapes and cache raw
   predictions as .npy (skip if cache already exists, use --force to re-run).
2. Grid-search a large library of post-processing techniques.
3. Print a ranked table and write outputs/postproc_results.json.

Usage
-----
    python postproc_eval.py --runs nohuman-label-head perch-label-head
    python postproc_eval.py --runs nohuman-label-head --force   # re-run inference
    python postproc_eval.py --gpu 0
"""

import argparse
import glob
import json
import os
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d, median_filter

# ── project helpers (read-only imports) ──────────────────────────────────────
from src.utils.config import load_config
from src.utils.audio import load_audio
from src.data.dataset import build_species_mapping
from src.metrics.kaggle_metric import score as kaggle_score


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",          default="configs/default.yaml")
    p.add_argument("--runs",  nargs="+",
                   default=["nohuman-label-head", "perch-label-head",
                             "nohuman-label-pseudo", "label-head-pseudo"])
    p.add_argument("--cache_dir",       default="outputs/postproc_cache")
    p.add_argument("--results_path",    default="outputs/postproc_results.json")
    p.add_argument("--batch_size",      type=int, default=512)
    p.add_argument("--gpu",             default=None)
    p.add_argument("--force",           action="store_true",
                   help="Re-run inference even if cache exists")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth (same logic as evaluate_final.py, read-only)
# ─────────────────────────────────────────────────────────────────────────────

def build_ground_truth(labels_csv: str, target_species: list) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    species_set = set(target_species)
    rows = []
    for _, row in df.iterrows():
        fname   = re.sub(r"\.ogg$", "", row["filename"], flags=re.IGNORECASE)
        h, m, s = str(row["end"]).strip().split(":")
        end_sec = int(h) * 3600 + int(m) * 60 + int(s)
        row_id  = f"{fname}_{end_sec}"
        vec     = np.zeros(len(target_species), dtype=np.float32)
        for code in str(row["primary_label"]).split(";"):
            code = code.strip()
            if code in species_set:
                vec[target_species.index(code)] = 1.0
        rows.append([row_id] + vec.tolist())
    sol = pd.DataFrame(rows, columns=["row_id"] + target_species)
    return sol.groupby("row_id", sort=False).max().reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Inference + cache
# ─────────────────────────────────────────────────────────────────────────────

def run_and_cache(run_name, config, ogg_files, target_species, batch_size,
                  cache_dir, force):
    """Return (preds np.ndarray, row_ids list).  Uses cache when available."""
    os.makedirs(cache_dir, exist_ok=True)
    preds_path   = os.path.join(cache_dir, f"{run_name}_preds.npy")
    rowids_path  = os.path.join(cache_dir, f"{run_name}_rowids.json")

    if not force and os.path.isfile(preds_path) and os.path.isfile(rowids_path):
        print(f"  [cache] Loading cached predictions for {run_name}")
        preds   = np.load(preds_path)
        row_ids = json.load(open(rowids_path))
        return preds, row_ids

    # ── load model ────────────────────────────────────────────────────────────
    from src.model.classifier import PerchClassifier

    ckpt_path    = f"checkpoints/{run_name}/best_head"
    run_cfg_path = f"outputs/{run_name}/config.yaml"
    run_config   = load_config(run_cfg_path) if os.path.isfile(run_cfg_path) else config

    model = PerchClassifier(
        perch_dir=config.model.perch_dir,
        num_classes=len(target_species),
        mode=run_config.model.mode,
        hidden_dim=run_config.model.hidden_dim,
        dropout=run_config.model.dropout,
        taxonomy_csv=config.data.get("taxonomy_csv", None),
        sample_submission_csv=config.data.get("sample_submission_csv", None),
    )
    model.load_head(ckpt_path)

    # ── inference ─────────────────────────────────────────────────────────────
    sr          = config.audio.sample_rate
    clip_len    = config.audio.clip_duration * sr
    all_preds, all_row_ids = [], []

    for filepath in ogg_files:
        audio = load_audio(filepath, sr)
        if audio is None:
            continue
        n_segs = len(audio) // clip_len
        if n_segs == 0:
            continue
        clips  = np.stack([audio[i * clip_len:(i + 1) * clip_len] for i in range(n_segs)])
        ss_id  = re.sub(r"\.ogg$", "", os.path.basename(filepath), flags=re.IGNORECASE)
        row_ids_file = [f"{ss_id}_{(i + 1) * config.audio.clip_duration}" for i in range(n_segs)]

        chunk_preds = []
        for start in range(0, len(clips), batch_size):
            batch   = tf.constant(clips[start:start + batch_size], dtype=tf.float32)
            logits  = model(batch, training=False)
            chunk_preds.append(tf.sigmoid(logits).numpy())
        all_preds.append(np.concatenate(chunk_preds, axis=0))
        all_row_ids.extend(row_ids_file)

    del model
    tf.keras.backend.clear_session()

    preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    np.save(preds_path, preds)
    json.dump(all_row_ids, open(rowids_path, "w"))
    print(f"  [cache] Saved {preds.shape} predictions for {run_name}")
    return preds, all_row_ids


# ─────────────────────────────────────────────────────────────────────────────
# Build soundscape groups (stem → sorted list of row indices)
# ─────────────────────────────────────────────────────────────────────────────

def build_sc_groups(row_ids):
    groups = {}
    for i, rid in enumerate(row_ids):
        stem = rid.rsplit("_", 1)[0]
        groups.setdefault(stem, []).append(i)
    for stem in groups:
        groups[stem].sort(key=lambda i: int(row_ids[i].rsplit("_", 1)[1]))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing functions
# ─────────────────────────────────────────────────────────────────────────────

def pp_sliding_max(preds, sc_groups, w_local, window):
    """Blend each chunk with species-wise max over ±(window//2) neighbourhood."""
    out = preds.copy()
    for idxs in sc_groups.values():
        chunk = preds[idxs]
        T = len(idxs)
        for t in range(T):
            lo = max(0, t - window // 2)
            hi = min(T, t + window // 2 + 1)
            local_max = chunk[lo:hi].max(axis=0)
            out[idxs[t]] = w_local * chunk[t] + (1.0 - w_local) * local_max
    return out


def pp_file_max(preds, sc_groups, w_local):
    """Blend each chunk with the species-wise max over the entire soundscape."""
    out = preds.copy()
    for idxs in sc_groups.values():
        chunk    = preds[idxs]
        file_max = chunk.max(axis=0, keepdims=True)
        out[idxs] = w_local * chunk + (1.0 - w_local) * file_max
    return out


def pp_file_mean(preds, sc_groups, w_local):
    """Blend each chunk with the species-wise mean over the entire soundscape."""
    out = preds.copy()
    for idxs in sc_groups.values():
        chunk     = preds[idxs]
        file_mean = chunk.mean(axis=0, keepdims=True)
        out[idxs] = w_local * chunk + (1.0 - w_local) * file_mean
    return out


def pp_gaussian(preds, sc_groups, sigma):
    """Gaussian smoothing along the time axis within each soundscape."""
    out = preds.copy()
    for idxs in sc_groups.values():
        if len(idxs) > 1:
            out[idxs] = gaussian_filter1d(preds[idxs], sigma=sigma, axis=0)
    return np.clip(out, 0.0, 1.0)


def pp_median(preds, sc_groups, size):
    """Median filter along the time axis within each soundscape."""
    out = preds.copy()
    for idxs in sc_groups.values():
        if len(idxs) > 1:
            out[idxs] = median_filter(preds[idxs], size=(size, 1), mode="nearest")
    return np.clip(out, 0.0, 1.0)


def pp_topk_mask(preds, k):
    """Zero out all but top-k species per chunk (keeps only high-confidence predictions)."""
    out = np.zeros_like(preds)
    top_k_idx = np.argpartition(preds, -k, axis=1)[:, -k:]
    for row_i, cols in enumerate(top_k_idx):
        out[row_i, cols] = preds[row_i, cols]
    return out


def pp_power(preds, gamma):
    """Apply power transform p^gamma (sharpens or softens predictions)."""
    return np.clip(preds ** gamma, 0.0, 1.0)


def pp_percentile_clip(preds, sc_groups, lo_pct, hi_pct):
    """Per-soundscape percentile clipping — suppress extreme values."""
    out = preds.copy()
    for idxs in sc_groups.values():
        chunk = preds[idxs]
        lo = np.percentile(chunk, lo_pct)
        hi = np.percentile(chunk, hi_pct)
        if hi > lo:
            out[idxs] = np.clip((chunk - lo) / (hi - lo), 0.0, 1.0)
    return out


def pp_temporal_max_pool(preds, sc_groups):
    """Replace each chunk with the max prediction across all chunks in the file."""
    out = preds.copy()
    for idxs in sc_groups.values():
        file_max = preds[idxs].max(axis=0)
        out[idxs] = file_max
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helper
# ─────────────────────────────────────────────────────────────────────────────

def score_predictions(preds, row_ids, solution, target_species):
    submission = pd.DataFrame(preds, columns=target_species)
    submission.insert(0, "row_id", row_ids)
    sol = solution.copy()
    common = sol["row_id"].isin(submission["row_id"])
    sol = sol[common].reset_index(drop=True)
    sub = submission[submission["row_id"].isin(sol["row_id"])].reset_index(drop=True)
    try:
        return kaggle_score(sol, sub, row_id_column_name="row_id")
    except Exception as e:
        print(f"    Scoring error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing experiments
# ─────────────────────────────────────────────────────────────────────────────

def build_experiment_grid(preds, row_ids, sc_groups):
    """
    Returns list of (label, processed_preds) for every PP config to try.
    """
    experiments = []

    # 0. Baseline (no PP)
    experiments.append(("baseline (no PP)", preds))

    # ── Single techniques ─────────────────────────────────────────────────────

    # Sliding-window max blend — vary weight and window size
    for w, win in product([0.5, 0.6, 0.7, 0.8], [3, 5, 7, 9]):
        label = f"slide_max(w={w}, win={win})"
        experiments.append((label, pp_sliding_max(preds, sc_groups, w, win)))

    # File-level max blend — vary weight
    for w in [0.75, 0.80, 0.85, 0.88, 0.90, 0.93, 0.95]:
        label = f"file_max(w={w})"
        experiments.append((label, pp_file_max(preds, sc_groups, w)))

    # File-level mean blend
    for w in [0.80, 0.85, 0.90, 0.95]:
        label = f"file_mean(w={w})"
        experiments.append((label, pp_file_mean(preds, sc_groups, w)))

    # Gaussian smoothing
    for sigma in [0.5, 0.75, 1.0, 1.5, 2.0]:
        label = f"gaussian(sigma={sigma})"
        experiments.append((label, pp_gaussian(preds, sc_groups, sigma)))

    # Median filter
    for size in [3, 5]:
        label = f"median(size={size})"
        experiments.append((label, pp_median(preds, sc_groups, size)))

    # Power transform
    for gamma in [0.5, 0.7, 1.5, 2.0]:
        label = f"power(gamma={gamma})"
        experiments.append((label, pp_power(preds, gamma)))

    # Full temporal max pooling
    experiments.append(("temporal_max_pool", pp_temporal_max_pool(preds, sc_groups)))

    # ── Combined pipelines ────────────────────────────────────────────────────

    # postproc_v1 reference
    pp_v1 = pp_sliding_max(preds, sc_groups, 0.60, 7)
    pp_v1 = pp_file_max(pp_v1, sc_groups, 0.88)
    pp_v1 = pp_gaussian(pp_v1, sc_groups, 0.75)
    experiments.append(("postproc_v1 (slide+file_max+gauss)", pp_v1))

    # Slide + Gaussian (skip file_max)
    for w, win, sigma in [(0.6, 7, 0.75), (0.7, 5, 1.0), (0.6, 5, 0.75)]:
        pp = pp_sliding_max(preds, sc_groups, w, win)
        pp = pp_gaussian(pp, sc_groups, sigma)
        experiments.append((f"slide+gauss(w={w},win={win},σ={sigma})", pp))

    # File_max + Gaussian
    for w_fm, sigma in [(0.88, 0.75), (0.90, 1.0), (0.85, 0.75)]:
        pp = pp_file_max(preds, sc_groups, w_fm)
        pp = pp_gaussian(pp, sc_groups, sigma)
        experiments.append((f"file_max+gauss(w={w_fm},σ={sigma})", pp))

    # Slide + File_max (skip Gaussian)
    for w_s, win, w_fm in [(0.60, 7, 0.88), (0.70, 7, 0.88), (0.60, 5, 0.90)]:
        pp = pp_sliding_max(preds, sc_groups, w_s, win)
        pp = pp_file_max(pp, sc_groups, w_fm)
        experiments.append((f"slide+file_max(ws={w_s},win={win},wfm={w_fm})", pp))

    # Full 3-stage with varied params
    for w_s, win, w_fm, sigma in [
        (0.60, 7, 0.88, 0.75),
        (0.60, 5, 0.88, 0.75),
        (0.70, 7, 0.88, 0.75),
        (0.60, 7, 0.90, 0.75),
        (0.60, 7, 0.88, 1.00),
        (0.70, 5, 0.90, 1.00),
        (0.65, 7, 0.88, 0.75),
    ]:
        pp = pp_sliding_max(preds, sc_groups, w_s, win)
        pp = pp_file_max(pp, sc_groups, w_fm)
        pp = pp_gaussian(pp, sc_groups, sigma)
        experiments.append((
            f"3stage(ws={w_s},win={win},wfm={w_fm},σ={sigma})", pp
        ))

    # Power + 3-stage
    for gamma in [0.7, 1.5]:
        pp = pp_power(preds, gamma)
        pp = pp_sliding_max(pp, sc_groups, 0.60, 7)
        pp = pp_file_max(pp, sc_groups, 0.88)
        pp = pp_gaussian(pp, sc_groups, 0.75)
        experiments.append((f"power({gamma})+3stage", pp))

    # File_max + Gaussian + Slide (reversed order)
    pp = pp_file_max(preds, sc_groups, 0.88)
    pp = pp_gaussian(pp, sc_groups, 0.75)
    pp = pp_sliding_max(pp, sc_groups, 0.60, 7)
    experiments.append(("file_max+gauss+slide (reversed)", pp))

    return experiments


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)

    # ── Soundscapes + ground truth ────────────────────────────────────────────
    labels_csv = config.data.soundscapes_labels_csv
    sc_dir     = config.data.train_soundscapes_dir
    labels_df  = pd.read_csv(labels_csv)
    labelled   = set(labels_df["filename"].unique())
    ogg_files  = sorted(
        f for f in glob.glob(os.path.join(sc_dir, "*.ogg"))
        if os.path.basename(f) in labelled
    )
    print(f"Labelled soundscapes: {len(ogg_files)}")

    target_species, _ = build_species_mapping(config.data.sample_submission_csv)
    solution = build_ground_truth(labels_csv, target_species)
    print(f"Ground-truth segments: {len(solution)}")

    # ── Process each run ──────────────────────────────────────────────────────
    all_results = {}

    for run_name in args.runs:
        ckpt_path = f"checkpoints/{run_name}/best_head.weights.h5"
        if not os.path.isfile(ckpt_path):
            print(f"\n[{run_name}] checkpoint not found, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  Run: {run_name}")
        print(f"{'='*60}")

        preds, row_ids = run_and_cache(
            run_name, config, ogg_files, target_species,
            args.batch_size, args.cache_dir, args.force
        )
        sc_groups = build_sc_groups(row_ids)
        print(f"  Predictions: {preds.shape}  |  Soundscapes: {len(sc_groups)}")

        # Baseline score
        baseline_auc = score_predictions(preds, row_ids, solution, target_species)
        print(f"  Baseline ROC-AUC: {baseline_auc:.4f}")

        # Build experiment grid
        print(f"  Running post-processing grid...")
        experiments = build_experiment_grid(preds, row_ids, sc_groups)
        print(f"  {len(experiments)} configurations to evaluate")

        run_results = []
        for label, pp_preds in experiments:
            auc = score_predictions(pp_preds, row_ids, solution, target_species)
            if auc is not None:
                delta = auc - baseline_auc
                run_results.append({"label": label, "roc_auc": auc, "delta": delta})

        # Sort by ROC-AUC descending
        run_results.sort(key=lambda x: x["roc_auc"], reverse=True)
        all_results[run_name] = {
            "baseline": baseline_auc,
            "results": run_results,
        }

        # Print top-20
        print(f"\n  {'Config':<48} {'ROC-AUC':>9} {'Delta':>8}")
        print(f"  {'-'*48} {'-'*9} {'-'*8}")
        for r in run_results[:20]:
            sign = "+" if r["delta"] >= 0 else ""
            print(f"  {r['label']:<48} {r['roc_auc']:.4f}    {sign}{r['delta']:.4f}")
        if len(run_results) > 20:
            print(f"  ... ({len(run_results) - 20} more configs)")

        # Best config
        best = run_results[0]
        print(f"\n  ★ Best: {best['label']}")
        print(f"    ROC-AUC {best['roc_auc']:.4f} (baseline {baseline_auc:.4f}, "
              f"delta {best['delta']:+.4f})")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.results_path}")

    # ── Cross-run summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY — best post-processing per run")
    print(f"{'='*70}")
    print(f"  {'Run':<35} {'Baseline':>9} {'Best PP':>9} {'Delta':>8} {'Config'}")
    print(f"  {'-'*35} {'-'*9} {'-'*9} {'-'*8}")
    for run_name, data in all_results.items():
        if not data["results"]:
            continue
        best = data["results"][0]
        print(f"  {run_name:<35} {data['baseline']:.4f}    {best['roc_auc']:.4f}   "
              f"{best['delta']:+.4f}  {best['label']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
