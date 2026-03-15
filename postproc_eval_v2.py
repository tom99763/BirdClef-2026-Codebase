"""
Post-Processing Evaluation v2 — BirdCLEF 2026
==============================================
Adds BirdCLEF 2025 top-solution techniques on top of v1's grid.

New techniques (BirdCLEF 2025 top solutions):
  TTA ±2.5s  — 2nd place, largest single improvement (+0.012 on their data)
               Predict each 5s window also at −2.5s and +2.5s offsets; average.
  Power boost — 3rd place: boost top-N confident predictions, suppress rest.
  VLOM avg   — 2nd place: (geometric_mean + RMS) / 2 for ensemble aggregation.
  Quantile-Mix — 38th place: α·raw + (1−α)·rank_normalised.
  Amplitude norm — 2nd place: librosa.util.normalize per clip before inference.
  Max-pool blend — 2nd place: element-wise max across overlapping segments.

Also re-runs the best configs from v1 for comparison.

Usage
-----
    python postproc_eval_v2.py --runs nohuman-label-head --gpu 0
    python postproc_eval_v2.py --runs nohuman-label-head --tta --gpu 0   # includes TTA inference
    python postproc_eval_v2.py --runs nohuman-label-head perch-label-head --gpu 0
"""

import argparse, glob, json, os, re, sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d, median_filter

from src.utils.config import load_config
from src.utils.audio import load_audio
from src.data.dataset import build_species_mapping
from src.metrics.kaggle_metric import score as kaggle_score


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/default.yaml")
    p.add_argument("--runs", nargs="+",
                   default=["nohuman-label-head", "perch-label-head"])
    p.add_argument("--cache_dir",    default="outputs/postproc_cache")
    p.add_argument("--results_path", default="outputs/postproc_results_v2.json")
    p.add_argument("--batch_size",   type=int, default=512)
    p.add_argument("--gpu",          default=None)
    p.add_argument("--force",        action="store_true",
                   help="Re-run baseline inference even if cache exists")
    p.add_argument("--tta",          action="store_true",
                   help="Also run TTA (±2.5s shift) inference — takes ~3× longer")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth
# ─────────────────────────────────────────────────────────────────────────────

def build_ground_truth(labels_csv, target_species):
    df = pd.read_csv(labels_csv)
    sp_set = set(target_species)
    rows = []
    for _, row in df.iterrows():
        fname = re.sub(r"\.ogg$", "", row["filename"], flags=re.IGNORECASE)
        h, m, s = str(row["end"]).strip().split(":")
        end_sec = int(h)*3600 + int(m)*60 + int(s)
        vec = np.zeros(len(target_species), dtype=np.float32)
        for code in str(row["primary_label"]).split(";"):
            code = code.strip()
            if code in sp_set:
                vec[target_species.index(code)] = 1.0
        rows.append([f"{fname}_{end_sec}"] + vec.tolist())
    sol = pd.DataFrame(rows, columns=["row_id"] + target_species)
    return sol.groupby("row_id", sort=False).max().reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_clips(model, clips, batch_size):
    preds = []
    for i in range(0, len(clips), batch_size):
        batch = tf.constant(clips[i:i+batch_size], dtype=tf.float32)
        preds.append(tf.sigmoid(model(batch, training=False)).numpy())
    return np.concatenate(preds, axis=0)


def run_inference(model, ogg_files, sr, clip_dur, batch_size,
                  amp_normalize=False, offsets_sec=(0,)):
    """
    Run inference with optional amplitude normalisation and multiple time offsets.

    offsets_sec: tuple of start offsets (in seconds) relative to the normal clip start.
                 e.g. (0,) = baseline;  (0, -2.5, 2.5) = TTA
    """
    clip_len   = clip_dur * sr
    half_clip  = clip_len // 2

    all_preds, all_row_ids = [], []

    for filepath in ogg_files:
        audio = load_audio(filepath, sr)
        if audio is None:
            continue
        n_segs = len(audio) // clip_len
        if n_segs == 0:
            continue

        ss_id    = re.sub(r"\.ogg$", "", os.path.basename(filepath), flags=re.IGNORECASE)
        row_ids  = [f"{ss_id}_{(i+1)*clip_dur}" for i in range(n_segs)]
        seg_preds = []   # (n_offsets, n_segs, n_classes)

        for off_sec in offsets_sec:
            off_samp = int(off_sec * sr)
            clips = []
            for i in range(n_segs):
                start = i * clip_len + off_samp
                end   = start + clip_len
                if start < 0:
                    clip = audio[:clip_len]
                    clip = np.concatenate([np.zeros(-start, dtype=np.float32), clip])[:clip_len]
                elif end > len(audio):
                    clip = audio[max(0, len(audio)-clip_len):]
                    clip = np.pad(clip, (0, clip_len - len(clip)))
                else:
                    clip = audio[start:end]

                if amp_normalize:
                    mx = np.abs(clip).max()
                    if mx > 1e-6:
                        clip = clip / mx

                clips.append(clip)

            clips_arr = np.stack(clips, axis=0)
            seg_preds.append(_infer_clips(model, clips_arr, batch_size))

        # Aggregate across offsets
        if len(seg_preds) == 1:
            file_preds = seg_preds[0]
        else:
            # VLOM (geometric mean + RMS) / 2  — 2nd place technique
            stacked = np.stack(seg_preds, axis=0)      # (K, T, C)
            geo = np.exp(np.mean(np.log(stacked + 1e-8), axis=0))
            rms = np.sqrt(np.mean(stacked**2, axis=0))
            file_preds = (geo + rms) / 2

        all_preds.append(file_preds)
        all_row_ids.extend(row_ids)

    return np.concatenate(all_preds, axis=0).astype(np.float32), all_row_ids


def load_or_run(run_name, config, ogg_files, target_species, batch_size,
                cache_dir, force, suffix="baseline"):
    os.makedirs(cache_dir, exist_ok=True)
    preds_path  = os.path.join(cache_dir, f"{run_name}_{suffix}_preds.npy")
    rowids_path = os.path.join(cache_dir, f"{run_name}_{suffix}_rowids.json")

    if not force and os.path.isfile(preds_path) and os.path.isfile(rowids_path):
        print(f"  [cache] {run_name} / {suffix}")
        return np.load(preds_path), json.load(open(rowids_path))

    from src.model.classifier import PerchClassifier
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
    model.load_head(f"checkpoints/{run_name}/best_head")

    amp_norm = "ampnorm" in suffix
    offsets  = (0, -2.5, 2.5) if "tta" in suffix else (0,)
    print(f"  [infer] {run_name} / {suffix}  amp_norm={amp_norm}  offsets={offsets}")

    preds, row_ids = run_inference(
        model, ogg_files,
        sr=config.audio.sample_rate,
        clip_dur=config.audio.clip_duration,
        batch_size=batch_size,
        amp_normalize=amp_norm,
        offsets_sec=offsets,
    )
    del model
    tf.keras.backend.clear_session()

    np.save(preds_path, preds)
    json.dump(row_ids, open(rowids_path, "w"))
    print(f"  [cache] saved {preds.shape}")
    return preds, row_ids


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing library
# ─────────────────────────────────────────────────────────────────────────────

def build_sc_groups(row_ids):
    groups = {}
    for i, rid in enumerate(row_ids):
        stem = rid.rsplit("_", 1)[0]
        groups.setdefault(stem, []).append(i)
    for stem in groups:
        groups[stem].sort(key=lambda i: int(row_ids[i].rsplit("_", 1)[1]))
    return groups


# ── v1 best techniques ────────────────────────────────────────────────────────

def pp_sliding_max(preds, sc_groups, w, window):
    out = preds.copy()
    for idxs in sc_groups.values():
        chunk = preds[idxs]; T = len(idxs)
        for t in range(T):
            lo, hi = max(0, t-window//2), min(T, t+window//2+1)
            out[idxs[t]] = w*chunk[t] + (1-w)*chunk[lo:hi].max(axis=0)
    return out

def pp_file_max(preds, sc_groups, w):
    out = preds.copy()
    for idxs in sc_groups.values():
        chunk = preds[idxs]
        out[idxs] = w*chunk + (1-w)*chunk.max(axis=0, keepdims=True)
    return out

def pp_gaussian(preds, sc_groups, sigma):
    out = preds.copy()
    for idxs in sc_groups.values():
        if len(idxs) > 1:
            out[idxs] = gaussian_filter1d(preds[idxs], sigma=sigma, axis=0)
    return np.clip(out, 0.0, 1.0)

def pp_median(preds, sc_groups, size):
    out = preds.copy()
    for idxs in sc_groups.values():
        if len(idxs) > 1:
            out[idxs] = median_filter(preds[idxs], size=(size, 1), mode="nearest")
    return np.clip(out, 0.0, 1.0)


# ── NEW: BirdCLEF 2025 techniques ────────────────────────────────────────────

def pp_power_boost(preds, top_n, boost_gamma, suppress_gamma):
    """
    3rd place: boost top-N confident predictions per chunk, suppress rest.
    boost_gamma   < 1 → amplifies   (e.g. 0.5: p^0.5 > p for p<1)
    suppress_gamma > 1 → suppresses (e.g. 2.0: p^2 < p for p<1)
    """
    out = np.zeros_like(preds)
    for i in range(len(preds)):
        row = preds[i]
        top_idx = np.argpartition(row, -top_n)[-top_n:]
        mask = np.zeros(len(row), dtype=bool)
        mask[top_idx] = True
        out[i,  mask] = np.clip(row[ mask] ** boost_gamma,    0, 1)
        out[i, ~mask] = np.clip(row[~mask] ** suppress_gamma, 0, 1)
    return out

def pp_quantile_mix(preds, sc_groups, alpha=0.5):
    """
    38th place: blend raw scores with rank-normalised scores.
    α=1 → pure raw;  α=0 → pure rank.
    """
    out = preds.copy()
    for idxs in sc_groups.values():
        chunk = preds[idxs]; T, C = chunk.shape
        rank_norm = np.zeros_like(chunk)
        for c in range(C):
            ranks = np.argsort(np.argsort(chunk[:, c]))
            rank_norm[:, c] = ranks / max(T-1, 1)
        out[idxs] = alpha*chunk + (1-alpha)*rank_norm
    return np.clip(out, 0.0, 1.0)

def pp_file_max_pool(preds, sc_groups):
    """2nd place: replace each chunk with the file-level max (extreme version of file_max w=0)."""
    out = preds.copy()
    for idxs in sc_groups.values():
        file_max = preds[idxs].max(axis=0)
        out[idxs] = file_max
    return out

def pp_sigmoid_temperature(preds, temperature):
    """Calibrate by rescaling logits via temperature before sigmoid."""
    logits = np.log(np.clip(preds, 1e-7, 1-1e-7) / (1 - np.clip(preds, 1e-7, 1-1e-7)))
    return 1.0 / (1.0 + np.exp(-logits / temperature))

def pp_threshold_zero(preds, threshold):
    """Zero out predictions below threshold (hard suppression of noise floor)."""
    out = preds.copy()
    out[out < threshold] = 0.0
    return out

def pp_blend_vlom(preds_a, preds_b, alpha=0.5):
    """
    VLOM blend between two prediction sets.
    Computes (geo_mean + rms) / 2 as 2nd place did for their ensemble.
    """
    stacked = np.stack([preds_a, preds_b], axis=0)
    geo = np.exp(np.mean(np.log(stacked + 1e-8), axis=0))
    rms = np.sqrt(np.mean(stacked**2, axis=0))
    vlom = (geo + rms) / 2
    return alpha * vlom + (1-alpha) * (preds_a + preds_b) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helper
# ─────────────────────────────────────────────────────────────────────────────

def score_preds(preds, row_ids, solution, target_species):
    sub = pd.DataFrame(preds, columns=target_species)
    sub.insert(0, "row_id", row_ids)
    sol = solution.copy()
    common = sol["row_id"].isin(sub["row_id"])
    sol = sol[common].reset_index(drop=True)
    sub = sub[sub["row_id"].isin(sol["row_id"])].reset_index(drop=True)
    try:
        return kaggle_score(sol, sub, row_id_column_name="row_id")
    except Exception as e:
        print(f"    scoring error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Experiment grid builder
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(preds_base, row_ids, sc_groups, preds_tta=None, preds_ampnorm=None):
    """
    Returns list of (label, processed_preds).
    preds_tta     : predictions from TTA inference (if available)
    preds_ampnorm : predictions from amp-normalised inference (if available)
    """
    exps = []

    # ── Baseline ─────────────────────────────────────────────────────────────
    exps.append(("baseline", preds_base))

    # ── v1 best (for reference) ───────────────────────────────────────────────
    exps.append(("v1_best: slide_max(w=0.5,win=9)",
                 pp_sliding_max(preds_base, sc_groups, 0.5, 9)))
    exps.append(("v1_best: median(size=3)",
                 pp_median(preds_base, sc_groups, 3)))

    # ── TTA (2nd place) ───────────────────────────────────────────────────────
    if preds_tta is not None:
        exps.append(("TTA ±2.5s (plain avg)",
                     (preds_base + preds_tta) / 2))
        # TTA + VLOM
        exps.append(("TTA ±2.5s (VLOM)",
                     pp_blend_vlom(preds_base, preds_tta, alpha=0.5)))
        # TTA then slide_max
        tta_avg = (preds_base + preds_tta) / 2
        exps.append(("TTA + slide_max(w=0.5,win=9)",
                     pp_sliding_max(tta_avg, sc_groups, 0.5, 9)))
        exps.append(("TTA + median(size=3)",
                     pp_median(tta_avg, sc_groups, 3)))
        exps.append(("TTA + file_max(w=0.88)",
                     pp_file_max(tta_avg, sc_groups, 0.88)))
        pp_tta_3s = pp_sliding_max(tta_avg, sc_groups, 0.5, 9)
        pp_tta_3s = pp_file_max(pp_tta_3s, sc_groups, 0.88)
        pp_tta_3s = pp_gaussian(pp_tta_3s, sc_groups, 0.75)
        exps.append(("TTA + 3stage(slide+file_max+gauss)", pp_tta_3s))

    # ── Amplitude normalisation (2nd place) ───────────────────────────────────
    if preds_ampnorm is not None:
        exps.append(("amp_norm inference (plain)", preds_ampnorm))
        exps.append(("amp_norm + slide_max(w=0.5,win=9)",
                     pp_sliding_max(preds_ampnorm, sc_groups, 0.5, 9)))
        if preds_tta is not None:
            exps.append(("amp_norm + TTA (plain avg)",
                         (preds_ampnorm + preds_tta) / 2))

    # ── Power boost (3rd place) ───────────────────────────────────────────────
    for top_n, bg, sg in [
        (5,  0.7, 2.0),
        (10, 0.7, 2.0),
        (5,  0.5, 2.0),
        (10, 0.5, 2.0),
        (5,  0.7, 3.0),
        (3,  0.5, 2.0),
    ]:
        exps.append((f"power_boost(top={top_n},bg={bg},sg={sg})",
                     pp_power_boost(preds_base, top_n, bg, sg)))

    # Power boost then slide_max
    for top_n, bg, sg in [(5, 0.7, 2.0), (10, 0.5, 2.0)]:
        pp = pp_power_boost(preds_base, top_n, bg, sg)
        exps.append((f"power_boost({top_n},{bg},{sg})+slide_max(0.5,9)",
                     pp_sliding_max(pp, sc_groups, 0.5, 9)))

    # ── Quantile-Mix (38th place) ─────────────────────────────────────────────
    for alpha in [0.3, 0.5, 0.7, 0.8, 0.9]:
        exps.append((f"quantile_mix(α={alpha})",
                     pp_quantile_mix(preds_base, sc_groups, alpha)))
    # Quantile-Mix then slide_max
    for alpha in [0.5, 0.7]:
        pp = pp_quantile_mix(preds_base, sc_groups, alpha)
        exps.append((f"quantile_mix({alpha})+slide_max(0.5,9)",
                     pp_sliding_max(pp, sc_groups, 0.5, 9)))

    # ── Temperature calibration ───────────────────────────────────────────────
    for T in [0.5, 0.7, 0.8, 1.2, 1.5, 2.0]:
        exps.append((f"temperature(T={T})",
                     pp_sigmoid_temperature(preds_base, T)))
    # Temperature then slide_max
    for T in [0.7, 0.8]:
        pp = pp_sigmoid_temperature(preds_base, T)
        exps.append((f"temp({T})+slide_max(0.5,9)",
                     pp_sliding_max(pp, sc_groups, 0.5, 9)))

    # ── Threshold zeroing ─────────────────────────────────────────────────────
    for thr in [0.01, 0.02, 0.05, 0.1]:
        exps.append((f"threshold_zero({thr})",
                     pp_threshold_zero(preds_base, thr)))
    # Threshold then slide_max
    pp = pp_threshold_zero(preds_base, 0.02)
    exps.append(("threshold(0.02)+slide_max(0.5,9)",
                 pp_sliding_max(pp, sc_groups, 0.5, 9)))

    # ── File-level max pool (extreme blend) ───────────────────────────────────
    exps.append(("file_max_pool_extreme",
                 pp_file_max_pool(preds_base, sc_groups)))
    exps.append(("file_max(w=0.5)+slide_max(0.5,9)",
                 pp_sliding_max(pp_file_max(preds_base, sc_groups, 0.5), sc_groups, 0.5, 9)))

    # ── Compound pipelines ────────────────────────────────────────────────────
    # Best v1 + new techniques
    p_slide = pp_sliding_max(preds_base, sc_groups, 0.5, 9)

    for alpha in [0.5, 0.7]:
        exps.append((f"slide_max(0.5,9)+quantile_mix({alpha})",
                     pp_quantile_mix(p_slide, sc_groups, alpha)))

    for T in [0.7, 0.8]:
        exps.append((f"slide_max(0.5,9)+temp({T})",
                     pp_sigmoid_temperature(p_slide, T)))

    for size in [3]:
        exps.append((f"slide_max(0.5,9)+median({size})",
                     pp_median(p_slide, sc_groups, size)))

    for top_n, bg, sg in [(5, 0.7, 2.0), (10, 0.5, 2.0)]:
        exps.append((f"slide_max(0.5,9)+power_boost({top_n},{bg},{sg})",
                     pp_power_boost(p_slide, top_n, bg, sg)))

    # 3-way compound
    for alpha, T in [(0.7, 0.8), (0.5, 0.7)]:
        pp3 = pp_sliding_max(preds_base, sc_groups, 0.5, 9)
        pp3 = pp_quantile_mix(pp3, sc_groups, alpha)
        pp3 = pp_sigmoid_temperature(pp3, T)
        exps.append((f"slide+quantile({alpha})+temp({T})", pp3))

    return exps


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)

    # Soundscapes
    labels_csv = config.data.soundscapes_labels_csv
    sc_dir     = config.data.train_soundscapes_dir
    labels_df  = pd.read_csv(labels_csv)
    labelled   = set(labels_df["filename"].unique())
    ogg_files  = sorted(f for f in glob.glob(os.path.join(sc_dir, "*.ogg"))
                        if os.path.basename(f) in labelled)
    print(f"Labelled soundscapes: {len(ogg_files)}")

    target_species, _ = build_species_mapping(config.data.sample_submission_csv)
    solution = build_ground_truth(labels_csv, target_species)
    print(f"Ground-truth segments: {len(solution)}\n")

    all_results = {}

    for run_name in args.runs:
        if not os.path.isfile(f"checkpoints/{run_name}/best_head.weights.h5"):
            print(f"[{run_name}] checkpoint not found, skipping.")
            continue

        print(f"\n{'='*65}")
        print(f"  Run: {run_name}")
        print(f"{'='*65}")

        # ── baseline inference ─────────────────────────────────────────────
        preds_base, row_ids = load_or_run(
            run_name, config, ogg_files, target_species,
            args.batch_size, args.cache_dir, args.force, suffix="baseline")

        sc_groups = build_sc_groups(row_ids)
        baseline_auc = score_preds(preds_base, row_ids, solution, target_species)
        print(f"  Baseline ROC-AUC : {baseline_auc:.4f}")

        # ── TTA inference (optional) ───────────────────────────────────────
        preds_tta = None
        if args.tta:
            preds_tta, _ = load_or_run(
                run_name, config, ogg_files, target_species,
                args.batch_size, args.cache_dir, args.force, suffix="tta")
            tta_auc = score_preds(preds_tta, row_ids, solution, target_species)
            print(f"  TTA-only ROC-AUC : {tta_auc:.4f}")

        # ── Amplitude-norm inference ───────────────────────────────────────
        preds_ampnorm = None
        if args.tta:   # run alongside TTA since we have the model loaded anyway
            preds_ampnorm, _ = load_or_run(
                run_name, config, ogg_files, target_species,
                args.batch_size, args.cache_dir, args.force, suffix="ampnorm")
            an_auc = score_preds(preds_ampnorm, row_ids, solution, target_species)
            print(f"  AmpNorm ROC-AUC  : {an_auc:.4f}")

        # ── Grid search ───────────────────────────────────────────────────
        experiments = build_grid(preds_base, row_ids, sc_groups, preds_tta, preds_ampnorm)
        print(f"\n  {len(experiments)} configurations to evaluate…")

        run_results = []
        for label, pp_preds in experiments:
            auc = score_preds(pp_preds, row_ids, solution, target_species)
            if auc is not None:
                run_results.append({"label": label, "roc_auc": auc,
                                    "delta": auc - baseline_auc})

        run_results.sort(key=lambda x: x["roc_auc"], reverse=True)
        all_results[run_name] = {"baseline": baseline_auc, "results": run_results}

        # Print top-25
        print(f"\n  {'Config':<52} {'ROC-AUC':>9} {'Delta':>8}")
        print(f"  {'-'*52} {'-'*9} {'-'*8}")
        for r in run_results[:25]:
            sign = "+" if r["delta"] >= 0 else ""
            print(f"  {r['label']:<52} {r['roc_auc']:.4f}   {sign}{r['delta']:.4f}")

        best = run_results[0]
        print(f"\n  ★ Best: {best['label']}")
        print(f"    {best['roc_auc']:.4f} (baseline {baseline_auc:.4f}, delta {best['delta']:+.4f})")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {args.results_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Run':<35} {'Baseline':>9} {'Best':>9} {'Delta':>8}  Config")
    print(f"  {'-'*35} {'-'*9} {'-'*9} {'-'*8}")
    for rn, data in all_results.items():
        if not data["results"]:
            continue
        best = data["results"][0]
        print(f"  {rn:<35} {data['baseline']:.4f}    {best['roc_auc']:.4f}   "
              f"{best['delta']:+.4f}  {best['label']}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
