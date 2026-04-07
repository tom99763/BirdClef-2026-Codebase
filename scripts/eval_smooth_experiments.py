"""
Local evaluation of temporal smoothing experiments.

Compares all post-processing methods on the same soundscape 4-fold splits
used for SED training, ensuring apples-to-apples comparison.

Methods evaluated:
  0. Baseline:         LGBM probe + fixed Gaussian logit smooth (0-910)
  1. LearnableConv:    Per-class Conv1d FIR (1170 params) — v3-learnable-smooth
  2. SoftOrderStats:   Per-class L-estimator / soft ranking (1170 params)
  3. MultiScaleConv:   3-branch K=3,7,11 + per-class gate (~6318 params)
  4. BilateralSmooth:  Content-adaptive bilateral filter (234 params)
  5. CausalIIR:        Bidirectional per-class EMA (234 params)
  6. DerivativeOnset:  Rising-edge amplification (234 params)
  7. SoftTopHat:       Morphological soft top-hat (235 params)
  E. SmootherEnsemble: All N-choose-K combos of the above (per-class learned blend)

Usage:
    python scripts/eval_smooth_experiments.py [--rebuild_cache] [--epochs 40]
    python scripts/eval_smooth_experiments.py --skip_ensemble   # skip combo experiments

Requirements:
    pip install lightgbm torch scikit-learn pandas numpy librosa tqdm
"""

import argparse
import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import convolve1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE          = Path("birdclef-2026")
WEIGHTS_DIR   = Path("submissions_v3/weights")
MODEL_DIR     = Path("models/bird-vocalization-classifier-tensorflow2-perch_v2_cpu-v1")
PERCH_TFLITE  = WEIGHTS_DIR / "perch_v2_cpu.tflite"
CACHE_NPZ     = Path("birdclef-2026/notebook resource/best perch/perch meta/full_perch_arrays.npz")
META_PARQUET  = Path("birdclef-2026/notebook resource/best perch/perch meta/full_perch_meta.parquet")
OOF_NPZ       = Path("birdclef-2026/notebook resource/best perch/perch meta/full_oof_meta_features.npz")
PROBE_PKL     = WEIGHTS_DIR / "lgbm_probe_models.pkl"
SS_DIR        = BASE / "train_soundscapes"
FOLDS_DIR     = Path("configs/ss_folds")
CACHE_EXT_NPZ = Path("outputs/perch_cache_extended.npz")   # 66-file extended cache

# ── Constants ──────────────────────────────────────────────────────────────────
SR            = 32_000
CLIP_DUR      = 5
CLIP_SAMPLES  = SR * CLIP_DUR          # 160000
N_WINDOWS     = 12                     # 60s file → 12 × 5s clips
N_PERCH       = 14795
N_EMBED       = 1536
NUM_CLASSES   = 234
TEMP_SCALE    = 1.15
GAUSSIAN_KERN = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
_OOF_LABELS   = None   # set in main() after build_ground_truth; used by per-class median fns

# ── Species mapping setup ──────────────────────────────────────────────────────
def build_species_mapping():
    taxonomy  = pd.read_csv(BASE / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    sample_sub = pd.read_csv(BASE / "sample_submission.csv")
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()

    bc_labels_df = (
        pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )
    bc_lookup = bc_labels_df.rename(columns={"scientific_name": "scientific_name_lookup"})
    taxonomy_copy = taxonomy.copy()
    taxonomy_copy["scientific_name_lookup"] = taxonomy_copy["scientific_name"]
    mapping = taxonomy_copy.merge(bc_lookup[["scientific_name_lookup", "bc_index"]],
                                  on="scientific_name_lookup", how="left")
    NO_LABEL_INDEX = len(bc_labels_df)
    mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)
    label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
    label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

    BC_INDICES        = np.array([int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
    MAPPED_MASK       = BC_INDICES != NO_LABEL_INDEX
    MAPPED_POS        = np.where(MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

    CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
    TEXTURE_TAXA   = {"Amphibia", "Insecta"}
    return (PRIMARY_LABELS, label_to_idx, BC_INDICES,
            MAPPED_MASK, MAPPED_POS, MAPPED_BC_INDICES,
            CLASS_NAME_MAP, TEXTURE_TAXA)


# ── Build ground-truth labels ──────────────────────────────────────────────────
def build_ground_truth(PRIMARY_LABELS, label_to_idx):
    ss_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
    ss_labels["primary_label"] = ss_labels["primary_label"].astype(str)

    def parse_labels(x):
        return [t.strip() for t in str(x).split(";") if t.strip()] if not pd.isna(x) else []

    sc = (
        ss_labels.groupby(["filename", "start", "end"])["primary_label"]
        .apply(lambda s: sorted(set(lbl for x in s for lbl in parse_labels(x))))
        .reset_index(name="label_list")
    )
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"]  = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    # Filter to fully-labeled files (all 12 windows present)
    full_files = set(sc.groupby("filename").filter(lambda g: len(g) == N_WINDOWS)["filename"].unique())
    sc = sc[sc["filename"].isin(full_files)].sort_values(["filename", "end_sec"]).reset_index(drop=True)

    Y = np.zeros((len(sc), NUM_CLASSES), dtype=np.float32)
    for i, labels in enumerate(sc["label_list"]):
        for lbl in labels:
            if lbl in label_to_idx:
                Y[i, label_to_idx[lbl]] = 1.0

    return sc, Y, full_files


# ── Perch TFLite inference (single clip) ──────────────────────────────────────
def run_perch_on_file(ogg_path, MAPPED_POS, MAPPED_BC_INDICES):
    """Returns (scores_raw: (12, 234), emb: (12, 1536))."""
    import librosa
    import tensorflow as tf

    audio, _ = librosa.load(str(ogg_path), sr=SR, mono=True)
    clips = np.zeros((N_WINDOWS, CLIP_SAMPLES), dtype=np.float32)
    for i in range(N_WINDOWS):
        s, e = i * CLIP_SAMPLES, (i + 1) * CLIP_SAMPLES
        chunk = audio[s:e]
        if len(chunk) < CLIP_SAMPLES:
            chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))
        clips[i] = chunk

    interp = tf.lite.Interpreter(model_path=str(PERCH_TFLITE), num_threads=2)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]["index"]
    outs    = interp.get_output_details()
    emb_idx = next(i for i, o in enumerate(outs) if o["shape"][-1] == N_EMBED)
    lbl_idx = next(i for i, o in enumerate(outs) if o["shape"][-1] == N_PERCH)

    scores_raw = np.zeros((N_WINDOWS, NUM_CLASSES), dtype=np.float32)
    embs       = np.zeros((N_WINDOWS, N_EMBED), dtype=np.float32)
    for i, clip in enumerate(clips):
        interp.set_tensor(inp_idx, clip[None])
        interp.invoke()
        logits_14795 = interp.get_tensor(outs[lbl_idx]["index"])[0]
        emb          = interp.get_tensor(outs[emb_idx]["index"])[0]
        scores_raw[i, MAPPED_POS] = logits_14795[MAPPED_BC_INDICES]
        embs[i] = emb

    return scores_raw, embs


# ── Build / extend cache to all 66 files ──────────────────────────────────────
def build_extended_cache(sc_df, MAPPED_POS, MAPPED_BC_INDICES, rebuild=False):
    # All 66 ss files: 59 have full 12-window labels; 7 have <12 (excluded from gt).
    # The 59-file Perch cache is complete for evaluation — no TFLite needed.
    if CACHE_EXT_NPZ.exists() and not rebuild:
        d = np.load(str(CACHE_EXT_NPZ))
        print(f"Loaded extended cache: {d['scores_full_raw'].shape}")
        return (d["scores_full_raw"], d["emb_full"],
                d["filenames"].tolist(), d["row_ids"].tolist())

    print("Building Perch cache (loading from 59-file cache)...")

    # Load existing 59-file cache
    d59 = np.load(str(CACHE_NPZ))
    meta59 = pd.read_parquet(str(META_PARQUET))

    # Files in cache
    cached_files = list(meta59["filename"].unique())
    cached_set   = set(cached_files)

    # All 66 files in ground-truth order (filename + end_sec sorted)
    all_files = list(sc_df["filename"].unique())
    all_files.sort()

    scores_all  = []
    embs_all    = []
    filenames_o = []
    row_ids_o   = []

    for fname in tqdm(all_files, desc="Perch cache"):
        file_windows = sc_df[sc_df["filename"] == fname].sort_values("end_sec")
        rids = file_windows["row_id"].tolist()

        if fname in cached_set:
            # Load from cache
            mask = meta59["filename"] == fname
            idx  = np.where(mask.values)[0]
            idx  = idx[np.argsort(meta59.loc[mask, "row_id"].values)]
            scores_all.append(d59["scores_full_raw"][idx])
            embs_all.append(d59["emb_full"][idx])
        else:
            # File not in cache (partial labels, excluded from gt) — skip
            print(f"  Skipping (not in Perch cache): {fname}")
            continue

        filenames_o.extend([fname] * N_WINDOWS)
        row_ids_o.extend(rids)

    scores_full_raw = np.concatenate(scores_all, axis=0).astype(np.float32)
    emb_full        = np.concatenate(embs_all, axis=0).astype(np.float32)

    CACHE_EXT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(CACHE_EXT_NPZ),
                        scores_full_raw=scores_full_raw,
                        emb_full=emb_full,
                        filenames=np.array(filenames_o),
                        row_ids=np.array(row_ids_o))
    print(f"Extended cache saved: {scores_full_raw.shape}  →  {CACHE_EXT_NPZ}")
    return scores_full_raw, emb_full, filenames_o, row_ids_o


# ── LGBM probe (OOF using ss_fold splits) ─────────────────────────────────────
def build_oof_probe_predictions(scores_full_raw, emb_full, filenames_list, sc_df, Y,
                                 PRIMARY_LABELS, label_to_idx):
    """Train LGBM probe OOF using the same 4-fold splits as SED."""
    from lightgbm import LGBMClassifier
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    probe_w = pickle.load(open(str(PROBE_PKL), "rb"))
    FROZEN  = probe_w["frozen_probe"]
    pca_dim = FROZEN.get("pca_dim", 64)
    alpha   = float(FROZEN.get("alpha", 0.40))
    min_pos = int(FROZEN.get("min_pos", 8))

    # Build file → row mapping
    files_arr = np.array(filenames_list)   # (N,)
    all_files = list(sc_df["filename"].unique())
    all_files.sort()

    # Load 4-fold splits
    folds = []
    for k in range(4):
        val_files = set(open(f"{FOLDS_DIR}/ss_fold{k}_val.txt").read().splitlines())
        folds.append(val_files)

    # Map each window to its fold
    fold_id = np.full(len(files_arr), -1, dtype=np.int32)
    for k, val_set in enumerate(folds):
        mask = np.array([f in val_set for f in files_arr])
        fold_id[mask] = k
    # Windows not in any val fold → training only (train windows)
    # For OOF we only report on val windows (fold_id >= 0)

    # PCA + scale on all data
    scaler = StandardScaler()
    emb_sc = scaler.fit_transform(emb_full)
    pca    = PCA(n_components=pca_dim, whiten=True, random_state=42)
    Z_all  = pca.fit_transform(emb_sc).astype(np.float32)

    # Build prior scores (same as fuse_scores: use raw perch logits as proxy)
    oof_base  = scores_full_raw.copy()  # start with raw as base
    oof_prior = scores_full_raw.copy()

    # Build per-class features
    def build_features(Z, raw_col, prior_col, base_col):
        pca_feat = Z
        seq_feat = np.stack([
            np.roll(base_col.reshape(-1, N_WINDOWS), 1, axis=1).reshape(-1),
            np.roll(base_col.reshape(-1, N_WINDOWS), -1, axis=1).reshape(-1),
        ], axis=1)
        inter1 = raw_col * prior_col
        inter2 = raw_col * base_col
        inter3 = prior_col * base_col
        return np.column_stack([pca_feat, raw_col, prior_col, base_col,
                                 seq_feat, inter1, inter2, inter3])

    print(f"Training OOF LGBM probe (pca_dim={pca_dim}, alpha={alpha}, min_pos={min_pos})...")
    oof_final = oof_base.copy()
    active_classes = np.where(Y.sum(axis=0) >= min_pos)[0]
    print(f"  Active classes (≥{min_pos} positives): {len(active_classes)}/234")

    lgbm_params = dict(n_estimators=100, max_depth=4, num_leaves=15,
                       learning_rate=0.05, min_child_samples=5,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=4, verbose=-1)

    for cls_idx in tqdm(active_classes, desc="OOF LGBM probe", leave=False):
        X_cls = build_features(Z_all, scores_full_raw[:, cls_idx],
                               oof_prior[:, cls_idx], oof_base[:, cls_idx])
        y_cls = Y[:, cls_idx]

        oof_pred_logit = np.zeros(len(y_cls), dtype=np.float32)
        for k, val_set in enumerate(folds):
            val_mask  = fold_id == k
            # Use all other folds for training (excluding unassigned)
            train_mask = (fold_id >= 0) & (~val_mask)
            if train_mask.sum() < 5 or val_mask.sum() == 0:
                continue
            clf = LGBMClassifier(**lgbm_params)
            clf.fit(X_cls[train_mask], y_cls[train_mask])
            proba = np.clip(clf.predict_proba(X_cls[val_mask])[:, 1], 1e-7, 1-1e-7)
            oof_pred_logit[val_mask] = np.log(proba / (1 - proba))

        # Blend probe logit with base logit
        blended = (1 - alpha) * oof_base[:, cls_idx] + alpha * oof_pred_logit
        # Only update val windows (fold_id >= 0)
        val_windows = fold_id >= 0
        oof_final[val_windows, cls_idx] = blended[val_windows]

    return oof_final, oof_base, Z_all, fold_id, alpha


# ── Macro AUC (skip empty classes) ────────────────────────────────────────────
def macro_auc(Y_true, Y_score, fold_mask=None):
    if fold_mask is not None:
        Y_true  = Y_true[fold_mask]
        Y_score = Y_score[fold_mask]
    aucs = []
    for c in range(Y_true.shape[1]):
        if Y_true[:, c].sum() > 0 and Y_true[:, c].sum() < len(Y_true):
            try:
                aucs.append(roc_auc_score(Y_true[:, c], Y_score[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


# ── Smoothing methods ──────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x / TEMP_SCALE, -30, 30)))


def _per_file(logits, fn):
    """Apply fn(window: (T, C)) per file, return (N, C)."""
    n_files = logits.shape[0] // N_WINDOWS
    out = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES).copy()
    for i in range(n_files):
        out[i] = fn(out[i])
    return out.reshape(-1, NUM_CLASSES)


def gaussian_smooth(logits):
    return _per_file(logits, lambda w: convolve1d(w, GAUSSIAN_KERN, axis=0, mode="nearest"))


# ── Literature-based zero-parameter methods ─────────────────────────────────

def median_smooth(logits, size=3):
    """Median filter in logit space (DCASE standard baseline)."""
    from scipy.ndimage import median_filter
    return _per_file(logits, lambda w: median_filter(w, size=(size, 1), mode="nearest"))


def median_then_gaussian(logits, med_size=3):
    """Median pre-filter → Gaussian (DCASE 2020-2024 best practice).
    Median removes impulse spikes; Gaussian then smooths residual.
    Source: Turpault et al.; DCASE 2024 Cornell system.
    """
    from scipy.ndimage import median_filter
    med = _per_file(logits, lambda w: median_filter(w, size=(med_size, 1), mode="nearest"))
    return _per_file(med, lambda w: convolve1d(w, GAUSSIAN_KERN, axis=0, mode="nearest"))


def savgol_smooth(logits, window_length=5, polyorder=2):
    """Savitzky-Golay filter — preserves peak HEIGHT (unlike Gaussian).
    Fits local polynomial of degree `polyorder` in window, reads center.
    kernel ≈ [-3/35, 12/35, 17/35, 12/35, -3/35] for w=5, p=2.
    Source: general signal processing; superior to Gaussian for AUC tasks.
    """
    from scipy.signal import savgol_filter
    wl = min(window_length, N_WINDOWS)
    if wl % 2 == 0:
        wl -= 1
    return _per_file(logits, lambda w: savgol_filter(w, window_length=wl,
                                                      polyorder=polyorder, axis=0, mode="nearest"))


def hmm_smooth(logits, p_onset=0.20, p_persist=0.70):
    """HMM forward-backward per class (2-state: ABSENT/PRESENT).
    Uses bidirectional decoding — future evidence informs past.
    Encodes 'a bird vocalizing at t is likely to continue at t+1'.
    Source: DCASE 2019 RCNN; arXiv:2601.04178 (RED layer).
    p_onset:  P(start vocalizing | was absent)  ≈ 0.2 for 5s clips
    p_persist: P(keep vocalizing | was present) ≈ 0.7 for 5s clips
    """
    probs_in = sigmoid(logits)    # (N, C) — compute inside, return logit-equivalent
    n_files  = probs_in.shape[0] // N_WINDOWS
    X = probs_in.reshape(n_files, N_WINDOWS, NUM_CLASSES)  # (B, T, C)
    out = np.zeros_like(X)

    # Transition matrix: A[from, to]
    A = np.array([[1 - p_onset, p_onset],
                  [1 - p_persist, p_persist]], dtype=np.float32)
    pi = np.array([1 - p_onset, p_onset], dtype=np.float32)

    for b in range(n_files):
        for c in range(NUM_CLASSES):
            p = X[b, :, c]          # (T,) probability sequence
            emit = np.stack([1 - p, p], axis=1)   # (T, 2) emission

            # Forward pass
            alpha = pi * emit[0]
            s = alpha.sum()
            alpha = alpha / (s + 1e-12)
            alphas = [alpha]
            for t in range(1, N_WINDOWS):
                alpha = (alphas[-1] @ A) * emit[t]
                s = alpha.sum()
                alpha = alpha / (s + 1e-12)
                alphas.append(alpha)

            # Backward pass
            beta = np.ones(2, dtype=np.float32)
            betas = [None] * N_WINDOWS
            betas[N_WINDOWS - 1] = beta
            for t in range(N_WINDOWS - 2, -1, -1):
                beta = A @ (emit[t + 1] * betas[t + 1])
                s = beta.sum()
                betas[t] = beta / (s + 1e-12)

            # Marginals P(PRESENT | all obs)
            for t in range(N_WINDOWS):
                gamma = alphas[t] * betas[t]
                gamma /= gamma.sum() + 1e-12
                out[b, t, c] = gamma[1]

    # Return as pseudo-logits (logit of smoothed prob) for consistent sigmoid() downstream
    out_flat = out.reshape(-1, NUM_CLASSES)
    out_flat = np.clip(out_flat, 1e-6, 1 - 1e-6)
    return np.log(out_flat / (1 - out_flat)) * TEMP_SCALE   # undo TEMP_SCALE so sigmoid(x/T) works


def max_dilation(logits, radius=1):
    """Morphological max-pool in probability space — fills detection gaps.
    If bird was present but only detected in 1/3 adjacent clips,
    dilation spreads the detection to neighbours → improves recall → AUC.
    Source: BirdCLEF 2024 4th place (+~0.01 LB).
    Apply in PROB space; convert back to pseudo-logits.
    """
    from scipy.ndimage import maximum_filter1d
    probs = sigmoid(logits)   # (N, C)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    out = maximum_filter1d(X, size=2 * radius + 1, axis=1, mode="nearest")
    out_flat = out.reshape(-1, NUM_CLASSES)
    out_flat = np.clip(out_flat, 1e-6, 1 - 1e-6)
    return np.log(out_flat / (1 - out_flat)) * TEMP_SCALE


def multiscale_median(logits, windows=(1, 3, 5)):
    """Multi-scale median ensemble — average median at multiple window sizes.
    Source: Soft-Median Selection (arXiv:2011.12564); +21.7% event-F1 vs DCASE winner.
    Weights: inversely proportional to window size (favour sharper).
    """
    from scipy.ndimage import median_filter
    weights = np.array([1.0 / w for w in windows], dtype=np.float32)
    weights /= weights.sum()
    smoothed = np.stack([
        _per_file(logits, lambda w, k=k: median_filter(w, size=(k, 1), mode="nearest"))
        for k in windows
    ], axis=0)   # (K, N, C)
    return (smoothed * weights[:, None, None]).sum(axis=0)


def alpha_trimmed_mean(logits, window=5, alpha=0.2):
    """Alpha-trimmed mean filter — between mean (smooth) and median (edge-preserving).
    Drop top/bottom alpha fraction of window, average remainder.
    Robust to outlier clips without full edge-preservation of median.
    """
    from scipy.ndimage import generic_filter
    k = max(1, int(alpha * window))

    def trimmed(x):
        xs = np.sort(x)
        trimmed = xs[k:-k] if len(xs) > 2 * k else xs
        return trimmed.mean()

    def apply_per_class(arr_1d):
        from scipy.ndimage import generic_filter
        return generic_filter(arr_1d, trimmed, size=window, mode="nearest")

    n_files = logits.shape[0] // N_WINDOWS
    out = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES).copy()
    for b in range(n_files):
        for c in range(NUM_CLASSES):
            out[b, :, c] = apply_per_class(out[b, :, c])
    return out.reshape(-1, NUM_CLASSES)


def gaussian_then_dilation(logits, radius=1):
    """Gaussian smooth → max dilation (hybrid: smooth noise then fill gaps)."""
    gauss = gaussian_smooth(logits)
    return max_dilation(gauss, radius=radius)


def median_gauss_dilation(logits, med_size=3, radius=1):
    """Median → Gaussian → Dilation (full pipeline from literature)."""
    return max_dilation(median_then_gaussian(logits, med_size=med_size), radius=radius)


# ── Round 2: New experiments built on dilation insight ────────────────────────

def dilation_class_adaptive(logits, radius_event=1, radius_texture=2,
                             event_idx=None, texture_idx=None):
    """Class-adaptive dilation: Aves(event) r=1, Amphibia/Insecta(texture) r=2.
    Texture taxa (frogs, insects) call continuously → benefit from wider dilation.
    Event taxa (birds) have brief calls → tight r=1 avoids bleeding.
    """
    from scipy.ndimage import maximum_filter1d
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    out = X.copy()
    if event_idx is not None and len(event_idx):
        out[:, :, event_idx] = maximum_filter1d(
            X[:, :, event_idx], size=2*radius_event+1, axis=1, mode="nearest")
    if texture_idx is not None and len(texture_idx):
        out[:, :, texture_idx] = maximum_filter1d(
            X[:, :, texture_idx], size=2*radius_texture+1, axis=1, mode="nearest")
    out = np.clip(out.reshape(-1, NUM_CLASSES), 1e-6, 1-1e-6)
    return np.log(out / (1 - out)) * TEMP_SCALE


def gauss_dilation_wide(logits, radius=1):
    """Wider Gaussian (w=7) → Dilation(r=1): more pre-smoothing before gap fill."""
    from scipy.ndimage import convolve1d
    WIDE_KERN = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05], dtype=np.float32)
    WIDE_KERN /= WIDE_KERN.sum()
    smoothed = _per_file(logits, lambda w: convolve1d(w, WIDE_KERN, axis=0, mode="nearest"))
    return max_dilation(smoothed, radius=radius)


def morphological_closing(logits, radius=1):
    """Morphological closing = dilation then erosion.
    Fills small gaps without expanding true positives outward.
    closing(x) = erode(dilate(x)) — standard morphological operation.
    Source: mathematical morphology; used in image segmentation.
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    size = 2 * radius + 1
    dilated = maximum_filter1d(X, size=size, axis=1, mode="nearest")
    closed  = minimum_filter1d(dilated, size=size, axis=1, mode="nearest")
    closed  = np.clip(closed.reshape(-1, NUM_CLASSES), 1e-6, 1-1e-6)
    return np.log(closed / (1 - closed)) * TEMP_SCALE


def gauss_closing(logits, radius=1):
    """Gaussian → Morphological closing (dilation+erosion).
    Closing fills annotation gaps more conservatively than pure dilation.
    """
    return morphological_closing(gaussian_smooth(logits), radius=radius)


def dilation_then_gauss(logits, radius=1):
    """Dilation first → Gaussian smooth (reverse of Gauss→Dilation).
    Ablation: check whether order matters for BirdCLEF sequences.
    """
    return gaussian_smooth(max_dilation(logits, radius=radius))


def double_dilation(logits, radius=1):
    """Apply max dilation twice (r=1 twice ≠ single r=2 — different boundary).
    First pass fills single-clip gaps; second pass fills two-clip gaps.
    """
    return max_dilation(max_dilation(logits, radius=radius), radius=radius)


def gauss_double_dilation(logits, radius=1):
    """Gaussian → double dilation (two passes of r=1)."""
    return double_dilation(gaussian_smooth(logits), radius=radius)


def percentile_filter(logits, window=3, pct=75):
    """Temporal percentile filter — between mean (smooth) and max (dilation).
    75th pct: less aggressive than hard max, more recall-boosting than median.
    """
    from scipy.ndimage import generic_filter
    def _pct(x):
        return np.percentile(x, pct)
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    out = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES).copy()
    for b in range(n_files):
        for c in range(NUM_CLASSES):
            out[b, :, c] = generic_filter(out[b, :, c], _pct,
                                           size=window, mode="nearest")
    out = np.clip(out.reshape(-1, NUM_CLASSES), 1e-6, 1-1e-6)
    return np.log(out / (1 - out)) * TEMP_SCALE


def logsumexp_pool(logits, radius=1, beta=3.0):
    """Soft-max (logsumexp) temporal pooling — differentiable max approximation.
    beta→∞ recovers hard max (dilation); beta→0 recovers mean (Gaussian).
    beta=3.0: intermediate softness; preserves gradient for peaks.
    Source: attention pooling literature; LogSumExp approximation.
    """
    n_files = logits.shape[0] // N_WINDOWS
    X = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    size = 2 * radius + 1
    out = np.zeros_like(X)
    pad = np.pad(X, ((0,0),(radius,radius),(0,0)), mode="edge")
    for t in range(N_WINDOWS):
        window = pad[:, t:t+size, :]          # (B, size, C)
        # logsumexp(beta*x)/beta ≈ max(x)
        bw = beta * window
        bw_max = bw.max(axis=1, keepdims=True)
        lse = bw_max.squeeze(1) + np.log(np.exp(bw - bw_max).sum(axis=1)) - np.log(size)
        out[:, t, :] = lse / beta
    return out.reshape(-1, NUM_CLASSES)


def gauss_logsumexp(logits, radius=1, beta=3.0):
    """Gaussian → LogSumExp pool (soft dilation)."""
    return logsumexp_pool(gaussian_smooth(logits), radius=radius, beta=beta)


def temporal_max_blend(logits, alpha=0.5, radius=1):
    """Blend of original logits and max-dilated logits.
    alpha=0.5: halfway between no-dilation and full-dilation.
    Tune alpha to control aggressiveness of gap-filling.
    """
    dilated = max_dilation(logits, radius=radius)
    return (1 - alpha) * logits + alpha * dilated


def gauss_maxblend(logits, alpha=0.5, radius=1):
    """Gaussian → blend(original, dilated): softer than full dilation."""
    g = gaussian_smooth(logits)
    return temporal_max_blend(g, alpha=alpha, radius=radius)


# ── Round 3: LSE tuning + global mean blend + combinations ────────────────────

def logsumexp_pool_r2(logits, beta=5.0):
    """LogSumExp pool with radius=2 (5-clip window). Winner was r=1,b=5."""
    return logsumexp_pool(logits, radius=2, beta=beta)


def gauss_lse_b5_r2(logits):
    """Gaussian → LogSumExp(b=5, r=2)."""
    return logsumexp_pool(gaussian_smooth(logits), radius=2, beta=5.0)


def lse_then_gauss(logits, beta=5.0, radius=1):
    """LogSumExp pool → Gaussian smooth (reverse order of Gauss→LSE)."""
    return gaussian_smooth(logsumexp_pool(logits, radius=radius, beta=beta))


def global_mean_blend(logits, alpha=0.2):
    """BirdCLEF 2024 6th place: blend smoothed probs with file-level mean.
    Each clip gets: 0.8 * local_prob + 0.2 * file_mean_prob
    Rationale: background species that appear in most clips get boosted;
    spurious single-clip detections get pulled toward mean → reduces FP.
    Source: BirdCLEF 2024 6th place solution post-processing.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_mean = X.mean(axis=1, keepdims=True)   # (B, 1, C)
    blended = (1 - alpha) * X + alpha * file_mean
    blended = np.clip(blended.reshape(-1, NUM_CLASSES), 1e-6, 1-1e-6)
    return np.log(blended / (1 - blended)) * TEMP_SCALE


def gauss_global_blend(logits, alpha=0.2):
    """Gaussian → global mean blend."""
    return global_mean_blend(gaussian_smooth(logits), alpha=alpha)


def lse_global_blend(logits, beta=5.0, alpha=0.2):
    """LogSumExp → global mean blend (combine two winning techniques)."""
    return global_mean_blend(logsumexp_pool(logits, radius=1, beta=beta), alpha=alpha)


def gauss_lse_global(logits, beta=5.0, alpha=0.2):
    """Gaussian → LSE(b=5) → global blend (chain best techniques)."""
    return global_mean_blend(
        logsumexp_pool(gaussian_smooth(logits), radius=1, beta=beta), alpha=alpha)


def lse_percentile_blend(logits, beta=5.0, pct=90, window=3, mix=0.5):
    """Blend LSE pool and percentile filter: cover both gap-filling mechanisms."""
    lse = logsumexp_pool(logits, radius=1, beta=beta)
    pct_f = percentile_filter(logits, window=window, pct=pct)
    return mix * lse + (1 - mix) * pct_f


def gauss_lse_percentile(logits, beta=5.0, pct=90, mix=0.5):
    """Gaussian → blend(LSE, Percentile90)."""
    g = gaussian_smooth(logits)
    lse = logsumexp_pool(g, radius=1, beta=beta)
    pct_f = percentile_filter(g, window=3, pct=pct)
    return mix * lse + (1 - mix) * pct_f


def lse_b4(logits):
    """LogSumExp beta=4 (between winning b=3 and b=5)."""
    return logsumexp_pool(logits, radius=1, beta=4.0)


def lse_b7(logits):
    """LogSumExp beta=7 (between b=5 and b=10)."""
    return logsumexp_pool(logits, radius=1, beta=7.0)


def gauss_lse_b4(logits):
    return logsumexp_pool(gaussian_smooth(logits), radius=1, beta=4.0)


def gauss_lse_b7(logits):
    return logsumexp_pool(gaussian_smooth(logits), radius=1, beta=7.0)


def dilation_lse_blend(logits, beta=5.0, mix=0.5):
    """Blend of Gauss→Dilation and Gauss→LSE(b=5) — combine gap fill methods."""
    g = gaussian_smooth(logits)
    dil = max_dilation(g, radius=1)
    lse = logsumexp_pool(g, radius=1, beta=beta)
    return mix * dil + (1 - mix) * lse


def temperature_sweep_lse(logits, beta=5.0, T=1.0):  # noqa: kept for reference
    """LSE with different temperature T before sigmoid (T<1.15 = sharper)."""
    n_files = logits.shape[0] // N_WINDOWS
    X = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    size = 3
    out = np.zeros_like(X)
    pad = np.pad(X, ((0,0),(1,1),(0,0)), mode="edge")
    for t in range(N_WINDOWS):
        window = pad[:, t:t+size, :]
        bw = beta * window / T           # apply T before softmax
        bw_max = bw.max(axis=1, keepdims=True)
        lse = (bw_max.squeeze(1) + np.log(np.exp(bw - bw_max).sum(axis=1))
               - np.log(size))
        out[:, t, :] = lse / beta * T   # scale back
    return out.reshape(-1, NUM_CLASSES)


# ── Round 4: Paper-derived new techniques (2026-03-21 search) ─────────────────

def power_adjustment(logits, alpha=0.8, rare_threshold=0.15):
    """BirdCLEF 2025 top solutions: power-law expansion for rare classes.
    p^alpha (alpha<1) expands low-probability predictions → boosts recall for rare species.
    Applied to classes with global mean prob < rare_threshold.
    Source: BirdCLEF+ 2025 top solutions; targets macro AUC rare-class contribution.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    class_means = X.mean(axis=(0, 1))        # (C,) global mean per class
    rare_mask = class_means < rare_threshold
    out = X.copy()
    out[:, :, rare_mask] = np.power(
        np.clip(X[:, :, rare_mask], 1e-8, 1.0), alpha)
    out = np.clip(out.reshape(-1, NUM_CLASSES), 1e-6, 1-1e-6)
    return np.log(out / (1 - out)) * TEMP_SCALE


def gauss_power(logits, alpha=0.8, rare_threshold=0.15):
    return power_adjustment(gaussian_smooth(logits), alpha=alpha,
                            rare_threshold=rare_threshold)


def lse_power(logits, beta=5.0, alpha=0.8, rare_threshold=0.15):
    """LSE(b=5) → power adjustment for rare classes."""
    return power_adjustment(logsumexp_pool(logits, radius=1, beta=beta),
                            alpha=alpha, rare_threshold=rare_threshold)


def gauss_lse_power(logits, beta=5.0, alpha=0.8):
    """Gaussian → LSE → power (full pipeline for rare-class recall)."""
    return power_adjustment(logsumexp_pool(gaussian_smooth(logits), radius=1, beta=beta),
                            alpha=alpha)


def boundary_aware_smooth(logits, onset_sigma=0.5, blend=0.4):
    """Boundary-aware onset/offset reconstruction (arXiv:2601.04178).
    Separately smooth onset/offset indicator functions, then reconstruct
    event probability from cumulative sum. Preserves sharp event edges.
    blend=0.4: conservative mix with original.
    Source: 'Sound Event Detection with Boundary-Aware Optimization' Jan 2026.
    """
    from scipy.ndimage import gaussian_filter1d
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    diff = np.diff(X, axis=1, prepend=X[:, :1, :])  # (B, T, C)
    onsets  = np.clip(diff, 0, None)
    offsets = np.clip(-diff, 0, None)
    s_on  = gaussian_filter1d(onsets, sigma=onset_sigma, axis=1)
    s_off = gaussian_filter1d(offsets, sigma=onset_sigma, axis=1)
    recon = np.cumsum(s_on - s_off, axis=1)
    recon = np.clip(recon, 0, 1)
    out = (1 - blend) * X + blend * recon
    out = np.clip(out.reshape(-1, NUM_CLASSES), 1e-6, 1-1e-6)
    return np.log(out / (1 - out)) * TEMP_SCALE


def gauss_boundary(logits, blend=0.4):
    return boundary_aware_smooth(gaussian_smooth(logits), blend=blend)


def lse_boundary(logits, beta=5.0, blend=0.4):
    return boundary_aware_smooth(logsumexp_pool(logits, radius=1, beta=beta), blend=blend)


def soft_nms_temporal(logits, sigma=1.0, score_thresh=0.0):
    """Soft-NMS: decay neighboring scores when a strong detection exists.
    Opposite of dilation — suppresses redundant multi-clip FPs.
    Source: Voxaboxen (NeurIPS 2025, Earth Species Project, arXiv:2503.02389).
    Applied in PROB space then converted back.
    sigma=1.0: decay function width in clips.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES).copy()
    for b in range(n_files):
        for c in range(NUM_CLASSES):
            p = X[b, :, c].copy()
            for t in range(N_WINDOWS):
                if p[t] < score_thresh:
                    continue
                for dt in [-2, -1, 1, 2]:
                    nt = t + dt
                    if 0 <= nt < N_WINDOWS:
                        decay = np.exp(-(dt**2) / (2 * sigma**2))
                        X[b, nt, c] *= (1 - decay * p[t] * 0.5)
    X = np.clip(X, 1e-6, 1-1e-6)
    return np.log(X.reshape(-1, NUM_CLASSES) / (1 - X.reshape(-1, NUM_CLASSES))) * TEMP_SCALE


def gauss_soft_nms(logits, sigma=1.0):
    return soft_nms_temporal(gaussian_smooth(logits), sigma=sigma)


def lse_global_power(logits, beta=5.0, alpha_glob=0.2, alpha_pow=0.8):
    """LSE → GlobalMean → Power: chain three proven techniques."""
    step1 = logsumexp_pool(logits, radius=1, beta=beta)
    step2 = global_mean_blend(step1, alpha=alpha_glob)
    return power_adjustment(step2, alpha=alpha_pow)


def gauss_lse_global_power(logits, beta=5.0, alpha_glob=0.15, alpha_pow=0.85):
    """Gaussian → LSE → GlobalMean → Power: full chain."""
    step1 = gaussian_smooth(logits)
    step2 = logsumexp_pool(step1, radius=1, beta=beta)
    step3 = global_mean_blend(step2, alpha=alpha_glob)
    return power_adjustment(step3, alpha=alpha_pow)


def lse_with_class_prior(logits, beta=5.0, prior_strength=0.1):
    """LSE → add class-level prior from training label frequency.
    Boost rare classes using their empirical positive rate as additive prior.
    Different from global_mean_blend: uses training freq, not file-level mean.
    """
    probs = sigmoid(logsumexp_pool(logits, radius=1, beta=beta))
    n_files = probs.shape[0] // N_WINDOWS
    # Estimate per-class frequency from this soundscape
    class_freq = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES).mean(axis=(0, 1))
    # Add small fraction of class frequency to each prediction
    boosted = probs + prior_strength * class_freq[None, :]
    boosted = np.clip(boosted, 1e-6, 1-1e-6)
    return np.log(boosted / (1 - boosted)) * TEMP_SCALE


# ── Round 5: EMA, multi-scale LSE, cSEBBs-lite, fine-tuned combos (2026-03-21) ─

def bidirectional_ema(logits, alpha=0.3):
    """Bidirectional EMA — average of causal and anti-causal EMA in logit space."""
    n_files = logits.shape[0] // N_WINDOWS
    X = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    fwd = np.zeros_like(X)
    fwd[:, 0, :] = X[:, 0, :]
    for t in range(1, N_WINDOWS):
        fwd[:, t, :] = alpha * X[:, t, :] + (1 - alpha) * fwd[:, t - 1, :]
    bwd = np.zeros_like(X)
    bwd[:, -1, :] = X[:, -1, :]
    for t in range(N_WINDOWS - 2, -1, -1):
        bwd[:, t, :] = alpha * X[:, t, :] + (1 - alpha) * bwd[:, t + 1, :]
    return ((fwd + bwd) / 2).reshape(-1, NUM_CLASSES)


def asymmetric_ema(logits, alpha_rise=0.5, alpha_fall=0.15):
    """Asymmetric EMA: fast rise (onset), slow fall (sustain) — causal forward only."""
    n_files = logits.shape[0] // N_WINDOWS
    X = sigmoid(logits).reshape(n_files, N_WINDOWS, NUM_CLASSES)
    out = np.zeros_like(X)
    out[:, 0, :] = X[:, 0, :]
    for t in range(1, N_WINDOWS):
        rising = X[:, t, :] > out[:, t - 1, :]
        alpha = np.where(rising, alpha_rise, alpha_fall)
        out[:, t, :] = alpha * X[:, t, :] + (1 - alpha) * out[:, t - 1, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def multiscale_lse_blend(logits, beta=5.0, r1=1, r2=2, w=0.7):
    """Blend LSE(r=1) and LSE(r=2): narrow + wide temporal context."""
    lse1 = logsumexp_pool(logits, radius=r1, beta=beta)
    lse2 = logsumexp_pool(logits, radius=r2, beta=beta)
    return w * lse1 + (1 - w) * lse2


def change_point_segment_mean(logits, threshold=0.08, blend=0.5):
    """cSEBBs-lite: change-point detection → replace with segment mean confidence.
    Detects boundaries where |delta_prob| > threshold, then applies segment mean.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    out = X.copy()
    diffs = np.abs(np.diff(X, axis=1))  # (n_files, N_WINDOWS-1, C)
    for f in range(n_files):
        for c in range(NUM_CLASSES):
            boundaries = [0] + [t + 1 for t in range(N_WINDOWS - 1)
                                 if diffs[f, t, c] > threshold] + [N_WINDOWS]
            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                seg_mean = X[f, s:e, c].mean()
                out[f, s:e, c] = blend * seg_mean + (1 - blend) * X[f, s:e, c]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def change_point_segment_max_blend(logits, threshold=0.06, blend=0.60, max_w=1.0):
    """cSEBBs variant: use segment MAX (or max/mean mix) instead of mean.
    From A-CPD (arXiv:2403.08525): per-segment aggregation with max expands
    high-confidence peaks to fill their detected segment — unlike mean which
    suppresses them. max_w=1.0: pure max; max_w=0.0: pure mean (=original cSEBBs).
    Non-monotone: a short high-peak gets amplified to adjacent clips in the segment.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    out = X.copy()
    diffs = np.abs(np.diff(X, axis=1))
    for f in range(n_files):
        for c in range(NUM_CLASSES):
            boundaries = [0] + [t + 1 for t in range(N_WINDOWS - 1)
                                 if diffs[f, t, c] > threshold] + [N_WINDOWS]
            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                seg_max = X[f, s:e, c].max()
                seg_mean = X[f, s:e, c].mean()
                seg_val = max_w * seg_max + (1 - max_w) * seg_mean
                out[f, s:e, c] = blend * seg_val + (1 - blend) * X[f, s:e, c]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def lse_global_ema(logits, beta=5.0, gm_alpha=0.2, ema_alpha=0.3):
    """Best combo (LSE→GlobalMean) then BidirEMA in logit space."""
    after_combo = lse_global_blend(logits, beta=beta, alpha=gm_alpha)
    return bidirectional_ema(after_combo, alpha=ema_alpha)


def lse_global_fine(logits, beta=5.0, alpha=0.175):
    """LSE→GlobalMean with fine alpha sweep around winning 0.2."""
    return lse_global_blend(logits, beta=beta, alpha=alpha)


def lse_multiscale_global(logits, beta=5.0, w=0.7, gm_alpha=0.2):
    """MultiScale-LSE → GlobalMean."""
    ms_lse = multiscale_lse_blend(logits, beta=beta, w=w)
    return global_mean_blend(ms_lse, alpha=gm_alpha)


def cp_lse_global(logits, beta=5.0, cp_thr=0.08, cp_blend=0.5, gm_alpha=0.2):
    """cSEBBs-lite → LSE → GlobalMean chain."""
    cp = change_point_segment_mean(logits, threshold=cp_thr, blend=cp_blend)
    return lse_global_blend(cp, beta=beta, alpha=gm_alpha)


def gauss_bidir_ema(logits, alpha=0.3):
    """Gaussian → BidirEMA."""
    return bidirectional_ema(gaussian_smooth(logits), alpha=alpha)


def lse_bidir_ema(logits, beta=5.0, alpha=0.3):
    """LSE(b=5) → BidirEMA."""
    return bidirectional_ema(logsumexp_pool(logits, radius=1, beta=beta), alpha=alpha)


def lse_global_asym_ema(logits, beta=5.0, gm_alpha=0.2, r=0.5, f=0.15):
    """Best combo → AsymmetricEMA (fast rise / slow fall)."""
    after_combo = lse_global_blend(logits, beta=beta, alpha=gm_alpha)
    return asymmetric_ema(after_combo, alpha_rise=r, alpha_fall=f)


def percentile90_global_blend(logits, pct_alpha=0.5, gm_alpha=0.2):
    """Pct90(w=3) → GlobalMean blend: combine two non-LSE winners."""
    pct = percentile_filter(logits, window=3, pct=90)
    return global_mean_blend(pct, alpha=gm_alpha)


def lse_r1r2_global(logits, beta=5.0, w=0.7, gm_alpha=0.2):
    """MultiScaleLSE (r=1 dominant) → GlobalMean, different w."""
    ms = multiscale_lse_blend(logits, beta=beta, r1=1, r2=2, w=w)
    return global_mean_blend(ms, alpha=gm_alpha)


# ── Round 6: Attention pooling, Gauss→GM chain, EMA fine-tune (2026-03-21) ─────

def linear_softmax_attn(logits, radius=1):
    """Linear Softmax (Attention) Temporal Pooling: out = Σ(p²) / Σ(p) per class.
    Parameter-free. MIL pooling literature (arXiv:1810.09050): between mean and max.
    """
    n_files = logits.shape[0] // N_WINDOWS
    X = sigmoid(logits).reshape(n_files, N_WINDOWS, NUM_CLASSES)
    size = 2 * radius + 1
    out = np.zeros_like(X)
    pad = np.pad(X, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    for t in range(N_WINDOWS):
        window = pad[:, t:t + size, :]       # (n_files, size, C)
        num = (window ** 2).sum(axis=1)      # Σp²
        den = window.sum(axis=1) + 1e-9      # Σp
        out[:, t, :] = num / den
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def linear_softmax_global(logits, radius=1, gm_alpha=0.2):
    """Linear Softmax Attention → GlobalMean blend."""
    return global_mean_blend(linear_softmax_attn(logits, radius=radius), alpha=gm_alpha)


def gauss_global_mean(logits, alpha=0.2):
    """Gaussian → GlobalMean blend (BirdCLEF 2024 3rd+6th combined, no LSE)."""
    return global_mean_blend(gaussian_smooth(logits), alpha=alpha)


def lse_global_ema_fine(logits, beta=5.0, gm_alpha=0.2, ema_alpha=0.2):
    """Best combo (LSE→GM) → BidirEMA with finer alpha."""
    after_combo = lse_global_blend(logits, beta=beta, alpha=gm_alpha)
    return bidirectional_ema(after_combo, alpha=ema_alpha)


def lse_b5_r2_global(logits, gm_alpha=0.2):
    """LSE(beta=5, radius=2) → GlobalMean — wider temporal context."""
    return global_mean_blend(logsumexp_pool(logits, radius=2, beta=5.0), alpha=gm_alpha)


def lse_b3_r1_global(logits, gm_alpha=0.2):
    """LSE(beta=3, radius=1) → GlobalMean — confirmed R2 beta=3 is 2nd best."""
    return global_mean_blend(logsumexp_pool(logits, radius=1, beta=3.0), alpha=gm_alpha)


def gauss_lse_global_fine(logits, beta=5.0, alpha=0.175):
    """Gauss→LSE→GM with alpha=0.175 (fine-tune around 0.2)."""
    return global_mean_blend(
        logsumexp_pool(gaussian_smooth(logits), radius=1, beta=beta), alpha=alpha)


def lse_global_cp(logits, beta=5.0, gm_alpha=0.2, cp_thr=0.06, cp_blend=0.4):
    """LSE→GM → cSEBBs refinement (apply change-point AFTER best combo)."""
    after_combo = lse_global_blend(logits, beta=beta, alpha=gm_alpha)
    return change_point_segment_mean(after_combo, threshold=cp_thr, blend=cp_blend)


def triple_chain(logits, beta=5.0, gm_alpha=0.2, ema_alpha=0.25):
    """Gauss→LSE→GM→BidirEMA: full 4-step chain."""
    step1 = gaussian_smooth(logits)
    step2 = logsumexp_pool(step1, radius=1, beta=beta)
    step3 = global_mean_blend(step2, alpha=gm_alpha)
    return bidirectional_ema(step3, alpha=ema_alpha)


# ── Round 7: cSEBBs fine-tune + multi-scale chains (2026-03-21) ─────────────────

def lse_gm_cp_fine(logits, beta=5.0, gm_alpha=0.2, cp_thr=0.05, cp_blend=0.5):
    """LSE→GM→cSEBBs with finer threshold/blend sweep."""
    after_combo = lse_global_blend(logits, beta=beta, alpha=gm_alpha)
    return change_point_segment_mean(after_combo, threshold=cp_thr, blend=cp_blend)


def ms_lse_gm_cp(logits, beta=5.0, ms_w=0.85, gm_alpha=0.2, cp_thr=0.06, cp_blend=0.4):
    """MultiScaleLSE→GM→cSEBBs: apply winning cSEBBs to multi-scale chain."""
    ms = multiscale_lse_blend(logits, beta=beta, w=ms_w)
    after_gm = global_mean_blend(ms, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=cp_blend)


def lse_gm_cp_ema(logits, beta=5.0, gm_alpha=0.2, cp_thr=0.06, cp_blend=0.4, ema_a=0.3):
    """LSE→GM→cSEBBs→BidirEMA: add EMA refinement after cSEBBs."""
    step1 = lse_global_cp(logits, beta=beta, gm_alpha=gm_alpha,
                          cp_thr=cp_thr, cp_blend=cp_blend)
    return bidirectional_ema(step1, alpha=ema_a)


def lse_gm_double_cp(logits, beta=5.0, gm_alpha=0.2, thr1=0.06, thr2=0.04):
    """LSE→GM→cSEBBs(thr=0.06)→cSEBBs(thr=0.04): two-pass change-point."""
    step1 = lse_global_cp(logits, beta=beta, gm_alpha=gm_alpha, cp_thr=thr1, cp_blend=0.4)
    return change_point_segment_mean(step1, threshold=thr2, blend=0.3)


def lse_b3_gm_cp(logits, gm_alpha=0.2, cp_thr=0.06, cp_blend=0.4):
    """LSE(b=3)→GM→cSEBBs: confirmed b3→GM=0.7819, try cSEBBs on top."""
    after_gm = lse_b3_r1_global(logits, gm_alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=cp_blend)


def lse_gm_cp_gauss(logits, beta=5.0, gm_alpha=0.2, cp_thr=0.06, cp_blend=0.4):
    """LSE→GM→cSEBBs→Gaussian: smooth the segmented output."""
    step1 = lse_global_cp(logits, beta=beta, gm_alpha=gm_alpha,
                          cp_thr=cp_thr, cp_blend=cp_blend)
    return gaussian_smooth(step1)


def lse_gm_a175_cp(logits, beta=5.0, cp_thr=0.06, cp_blend=0.4):
    """LSE→GM(a=0.175)→cSEBBs: fine alpha winner + cSEBBs."""
    after_gm = lse_global_blend(logits, beta=beta, alpha=0.175)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=cp_blend)


def lse_gm_cp_blend_sweep(logits, beta=5.0, gm_alpha=0.2, cp_thr=0.06, cp_blend=0.6):
    """LSE→GM→cSEBBs with higher blend (more segment-mean, less local)."""
    after_combo = lse_global_blend(logits, beta=beta, alpha=gm_alpha)
    return change_point_segment_mean(after_combo, threshold=cp_thr, blend=cp_blend)


# ── Round 8: RED / nSEBBs / Coarse-Max-Pool (2026-03-21 literature) ─────────────

def recurrent_event_detection(logits, eps=1e-6):
    """RED: Bayesian causal recurrence, zero-parameter.
    P(active@t) = P(onset@t) + P(active@t-1) * P(no-offset@t)
    Source: arXiv:2601.04178 (Sound Event Detection with Boundary-Aware Optimization)
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    out = np.zeros_like(X)
    out[:, 0, :] = X[:, 0, :]
    for t in range(1, N_WINDOWS):
        p_onset = np.maximum(0.0, X[:, t, :] - X[:, t - 1, :])
        p_no_offset = np.clip(X[:, t, :] / (X[:, t - 1, :] + eps), 0.0, 1.0)
        out[:, t, :] = np.clip(p_onset + out[:, t - 1, :] * p_no_offset, 0.0, 1.0)
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def bidir_red(logits, eps=1e-6):
    """Bidirectional RED: forward + backward causal recurrence, geometric mean.
    Source: arXiv:2601.04178 + arXiv:2503.02389 (Voxaboxen)
    """
    n_files = logits.shape[0] // N_WINDOWS
    probs = sigmoid(logits).reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # Forward RED
    fwd = np.zeros_like(probs)
    fwd[:, 0, :] = probs[:, 0, :]
    for t in range(1, N_WINDOWS):
        p_onset = np.maximum(0.0, probs[:, t, :] - probs[:, t - 1, :])
        p_no_off = np.clip(probs[:, t, :] / (probs[:, t - 1, :] + eps), 0.0, 1.0)
        fwd[:, t, :] = np.clip(p_onset + fwd[:, t - 1, :] * p_no_off, 0.0, 1.0)
    # Backward RED (reverse sequence)
    bwd = np.zeros_like(probs)
    bwd[:, -1, :] = probs[:, -1, :]
    for t in range(N_WINDOWS - 2, -1, -1):
        p_onset = np.maximum(0.0, probs[:, t, :] - probs[:, t + 1, :])
        p_no_off = np.clip(probs[:, t, :] / (probs[:, t + 1, :] + eps), 0.0, 1.0)
        bwd[:, t, :] = np.clip(p_onset + bwd[:, t + 1, :] * p_no_off, 0.0, 1.0)
    # Combine: geometric mean
    combined = np.sqrt(np.clip(fwd * bwd, 1e-12, 1.0))
    combined = np.clip(combined, 1e-6, 1 - 1e-6)
    return (np.log(combined / (1 - combined)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def nsebbs_lite(logits, base_blend=0.4, pcr_scale=0.04):
    """nSEBBs-lite: self-tuning cSEBBs — threshold adapts to per-class PCR.
    PCR = log(p90 / p10): high PCR → confident class → higher threshold.
    Source: arXiv:2505.11889 (Normalized Sound Event Bounding Boxes)
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    out = X.copy()
    for f in range(n_files):
        for c in range(NUM_CLASSES):
            seq = X[f, :, c]
            p90 = np.percentile(seq, 90)
            p10 = np.percentile(seq, 10)
            pcr = np.log((p90 + 1e-6) / (p10 + 1e-6))
            thr = np.clip(0.02 + pcr_scale * pcr, 0.015, 0.12)
            diffs = np.abs(np.diff(seq))
            boundaries = [0] + [t + 1 for t in range(N_WINDOWS - 1)
                                 if diffs[t] > thr] + [N_WINDOWS]
            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                seg_mean = seq[s:e].mean()
                out[f, s:e, c] = base_blend * seg_mean + (1 - base_blend) * seq[s:e]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def coarse_max_pool(logits, width=2):
    """Coarse Max-Pool: group adjacent clips, assign group max.
    Source: FreDNet arXiv:2406.15725 (DCASE 2024 2nd place)
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    out = np.zeros_like(X)
    for t in range(N_WINDOWS):
        s = (t // width) * width
        e = min(s + width, N_WINDOWS)
        out[:, t, :] = X[:, s:e, :].max(axis=1)
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def lse_gm_red(logits, beta=5.0, gm_alpha=0.2):
    """Best combo LSE→GM then RED refinement."""
    after_combo = lse_global_blend(logits, beta=beta, alpha=gm_alpha)
    return recurrent_event_detection(after_combo)


def lse_gm_bidir_red(logits, beta=5.0, gm_alpha=0.2):
    """Best combo LSE→GM then Bidirectional RED."""
    after_combo = lse_global_blend(logits, beta=beta, alpha=gm_alpha)
    return bidir_red(after_combo)


def lse_gm_nsebbs(logits, beta=5.0, gm_alpha=0.2, base_blend=0.4):
    """LSE→GM→nSEBBs-lite: adaptive segmentation on best combo output."""
    after_combo = lse_global_blend(logits, beta=beta, alpha=gm_alpha)
    return nsebbs_lite(after_combo, base_blend=base_blend)


def lse_gm_cp_coarse(logits, beta=5.0, gm_alpha=0.2, cp_thr=0.06, width=2):
    """LSE→GM→cSEBBs→CoarseMaxPool: full chain + coarse pooling."""
    after_cp = lse_global_cp(logits, beta=beta, gm_alpha=gm_alpha, cp_thr=cp_thr)
    return coarse_max_pool(after_cp, width=width)


# ── Round 9: Beta fine-tune / logit-space GM / local-mean / pipeline blend ───────

def lse_gm_logit_cp(logits, beta=5.0, gm_alpha=0.175, cp_thr=0.06):
    """LSE → GlobalMean in LOGIT space (not prob) → cSEBBs.
    Blends logits directly instead of converting to probs first.
    """
    lse = logsumexp_pool(logits, radius=1, beta=beta)
    n_files = lse.shape[0] // N_WINDOWS
    X = lse.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_mean = X.mean(axis=1, keepdims=True)
    blended = (1 - gm_alpha) * X + gm_alpha * file_mean
    return change_point_segment_mean(
        blended.reshape(-1, NUM_CLASSES), threshold=cp_thr, blend=0.4)


def lse_local_halfmean_cp(logits, beta=5.0, alpha=0.175, cp_thr=0.06):
    """LSE → HalfWindow local mean (6-clip) → cSEBBs.
    Replace full-file mean with 6-clip sliding mean to preserve more temporal structure.
    """
    probs = sigmoid(logsumexp_pool(logits, radius=1, beta=beta))
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    half = N_WINDOWS // 2
    # sliding window of half-file size
    out = np.zeros_like(X)
    for t in range(N_WINDOWS):
        s = max(0, t - half // 2)
        e = min(N_WINDOWS, s + half)
        local_mean = X[:, s:e, :].mean(axis=1)
        out[:, t, :] = (1 - alpha) * X[:, t, :] + alpha * local_mean
    out = np.clip(out, 1e-6, 1 - 1e-6)
    logit_out = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(logit_out, threshold=cp_thr, blend=0.4)


def best_pipeline_blend(logits, w=0.7, beta=5.0, gm_alpha=0.175, cp_thr=0.06):
    """Additive ensemble: w*BestPipeline + (1-w)*Gaussian.
    Combines best post-proc with Gaussian to capture both gains.
    """
    best = lse_gm_a175_cp(logits, beta=beta, cp_thr=cp_thr, cp_blend=0.4)
    gauss = gaussian_smooth(logits)
    return w * best + (1 - w) * gauss


def lse_beta45_gm_cp(logits, gm_alpha=0.175, cp_thr=0.06):
    """LSE(β=4.5)→GM→cSEBBs: fine beta between 4 and 5."""
    after_gm = lse_global_blend(logits, beta=4.5, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def lse_beta55_gm_cp(logits, gm_alpha=0.175, cp_thr=0.06):
    """LSE(β=5.5)→GM→cSEBBs: fine beta above 5."""
    after_gm = lse_global_blend(logits, beta=5.5, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def lse_beta6_gm_cp(logits, gm_alpha=0.175, cp_thr=0.06):
    """LSE(β=6)→GM→cSEBBs."""
    after_gm = lse_global_blend(logits, beta=6.0, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def file_max_norm_lse_gm_cp(logits, beta=5.0, gm_alpha=0.175, cp_thr=0.06):
    """File-max normalization → LSE→GM→cSEBBs.
    Normalize per-class per-file so peak=1 before pipeline, then restore scale.
    Boosts rare classes with low absolute scores.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_max = X.max(axis=1, keepdims=True) + 1e-9
    X_norm = X / file_max
    X_norm = np.clip(X_norm, 1e-6, 1 - 1e-6)
    norm_logits = (np.log(X_norm / (1 - X_norm)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return lse_gm_a175_cp(norm_logits, beta=beta, cp_thr=cp_thr)


def entropy_weighted_lse_gm_cp(logits, beta=5.0, gm_alpha=0.175, cp_thr=0.06):
    """Entropy-weighted blend: low-entropy (confident) clips get higher weight.
    Blend each clip toward file mean weighted by its entropy.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # Per-clip entropy (lower = more confident)
    H = -(X * np.log(X + 1e-9) + (1-X) * np.log(1-X + 1e-9))  # (n_files, N_W, C)
    H_norm = H / (H.sum(axis=1, keepdims=True) + 1e-9)  # normalize over time
    # Weight = inverse entropy (confident clips weighted more)
    w = 1.0 / (H_norm + 1e-9)
    w = w / w.sum(axis=1, keepdims=True)
    # Entropy-weighted "file mean"
    weighted_mean = (X * w).sum(axis=1, keepdims=True)
    blended = (1 - gm_alpha) * X + gm_alpha * weighted_mean
    blended = np.clip(blended, 1e-6, 1 - 1e-6)
    out_logits = (np.log(blended / (1 - blended)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    lse_out = logsumexp_pool(out_logits, radius=1, beta=beta)
    return change_point_segment_mean(lse_out, threshold=cp_thr, blend=0.4)


def lse_gm_a15_cp(logits, beta=5.0, cp_thr=0.06):
    """LSE→GM(α=0.15)→cSEBBs: slightly less file-mean influence."""
    after_gm = lse_global_blend(logits, beta=beta, alpha=0.15)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def lse_gm_a20_cp(logits, beta=5.0, cp_thr=0.06):
    """LSE→GM(α=0.20)→cSEBBs: confirm α=0.175 vs α=0.20."""
    after_gm = lse_global_blend(logits, beta=beta, alpha=0.20)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def ms_lse_gm_a175_cp(logits, cp_thr=0.06):
    """MultiScaleLSE(w=0.85)→GM(α=0.175)→cSEBBs: multi-scale + fine alpha."""
    ms = multiscale_lse_blend(logits, beta=5.0, w=0.85)
    after_gm = global_mean_blend(ms, alpha=0.175)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


# ── Round 10: Power Pooling / BirdPresenceAmp / GeomBidirEMA (2026-03-21) ────────

def power_pool(logits, alpha=2.0, radius=1):
    """Power Pooling: P = (mean(p^α))^(1/α) — generalized mean in window.
    α=1: arithmetic mean; α=2: RMS (zero-param); α→∞: max.
    Source: arXiv:2010.09985 (Power Pooling, ICASSP 2021), +11.4% over linear softmax.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    size = 2 * radius + 1
    out = np.zeros_like(X)
    pad = np.pad(X, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    for t in range(N_WINDOWS):
        window = pad[:, t:t + size, :]       # (n_files, size, C)
        pw_mean = (window ** alpha).mean(axis=1)
        out[:, t, :] = np.clip(pw_mean ** (1.0 / alpha), 0.0, 1.0)
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def power_pool_gm_cp(logits, alpha=2.0, gm_alpha=0.175, cp_thr=0.06):
    """Power Pooling (α=2) → GlobalMean → cSEBBs."""
    pp = power_pool(logits, alpha=alpha)
    after_gm = global_mean_blend(pp, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def bird_presence_amp(logits, alpha=0.15):
    """Bird-Presence Amplification (BirdCLEF 2024 3rd place).
    Boost clips for species that appear strongly at least once in the file.
    P_boost(t,c) = P(t,c) + α*(P_max(c) + mean(P(:,c)) - mean_c(P_max))
    Source: zenn.dev/yuto_mo/articles/53ed2b27c1f52b
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_max = X.max(axis=1)              # (n_files, C) — max per class per file
    file_mean = X.mean(axis=1)            # (n_files, C) — mean per class per file
    mean_file_max = file_max.mean(axis=1, keepdims=True)  # (n_files, 1) — mean across classes
    boost = alpha * (file_max + file_mean - mean_file_max)  # (n_files, C)
    out = X + boost[:, None, :]           # broadcast over time
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def bird_presence_amp_gm_cp(logits, amp_alpha=0.15, gm_alpha=0.175, cp_thr=0.06):
    """BirdPresenceAmp → GlobalMean → cSEBBs."""
    after_amp = bird_presence_amp(logits, alpha=amp_alpha)
    after_gm = global_mean_blend(after_amp, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def lse_bird_presence_amp_cp(logits, beta=5.0, amp_alpha=0.15, cp_thr=0.06):
    """LSE → BirdPresenceAmp → cSEBBs: replace GM with selective amp."""
    lse = logsumexp_pool(logits, radius=1, beta=beta)
    after_amp = bird_presence_amp(lse, alpha=amp_alpha)
    return change_point_segment_mean(after_amp, threshold=cp_thr, blend=0.4)


def geom_bidir_ema(logits, alpha=0.3):
    """Geometric-mean BidirEMA: sqrt(fwd * bwd) instead of (fwd+bwd)/2.
    Stricter than arithmetic — penalises cases where one direction is uncertain.
    Source: derived from Voxaboxen arXiv:2503.02389 onset/offset analysis.
    """
    n_files = logits.shape[0] // N_WINDOWS
    X = sigmoid(logits).reshape(n_files, N_WINDOWS, NUM_CLASSES)
    fwd = np.zeros_like(X);  fwd[:, 0, :] = X[:, 0, :]
    for t in range(1, N_WINDOWS):
        fwd[:, t, :] = alpha * X[:, t, :] + (1 - alpha) * fwd[:, t - 1, :]
    bwd = np.zeros_like(X);  bwd[:, -1, :] = X[:, -1, :]
    for t in range(N_WINDOWS - 2, -1, -1):
        bwd[:, t, :] = alpha * X[:, t, :] + (1 - alpha) * bwd[:, t + 1, :]
    out = np.sqrt(np.clip(fwd * bwd, 1e-12, 1.0))
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def lse_geom_bidir_ema_gm_cp(logits, beta=5.0, ema_a=0.3, gm_alpha=0.175, cp_thr=0.06):
    """LSE → GeomBidirEMA → GlobalMean → cSEBBs."""
    after_lse = logsumexp_pool(logits, radius=1, beta=beta)
    after_ema = geom_bidir_ema(after_lse, alpha=ema_a)
    after_gm = global_mean_blend(after_ema, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def entropy_clip_drop_lse_gm_cp(logits, drop_q=0.5, beta=5.0, gm_alpha=0.175, cp_thr=0.06):
    """Entropy-based clip selection: zero-out high-entropy clips before aggregation.
    drop_q: quantile above which clips are dropped (0.5 = drop top-50% entropy clips).
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    H = -(X * np.log(X + 1e-9) + (1 - X) * np.log(1 - X + 1e-9))
    H_per_clip = H.mean(axis=2)  # (n_files, N_WINDOWS) — avg entropy per clip
    threshold = np.quantile(H_per_clip, drop_q, axis=1, keepdims=True)
    mask = (H_per_clip <= threshold).astype(float)[:, :, None]  # keep low-entropy
    X_masked = X * mask + X.mean(axis=1, keepdims=True) * (1 - mask)  # replace hi-H with file mean
    X_masked = np.clip(X_masked, 1e-6, 1 - 1e-6)
    masked_logits = (np.log(X_masked / (1 - X_masked)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(masked_logits, radius=1, beta=beta)
    after_gm = global_mean_blend(lse, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


# ── Round 11: Soft-entropy weight / Selective-max-mean / Trim-mean / Diffusion
# ──          Confidence-adaptive GM / SoftMax-blend / Spectral low-pass ──────

def soft_entropy_weight_lse_gm_cp(logits, beta=5.0, temp=1.0, gm_alpha=0.175, cp_thr=0.06):
    """Soft entropy weighting: weight each clip by exp(-H(t)/temp) before LSE.
    Gentler than hard-drop: noisy clips are soft-attenuated, not zeroed.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    H = -(X * np.log(X + 1e-9) + (1 - X) * np.log(1 - X + 1e-9))
    H_per_clip = H.mean(axis=2)              # (n_files, N_WINDOWS)
    w = np.exp(-H_per_clip / temp)           # (n_files, N_WINDOWS) — soft weight
    w = w / w.sum(axis=1, keepdims=True)     # normalize per file
    # Weight the logits before LSE: attenuate logits of high-entropy clips
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    weighted = raw * w[:, :, None]           # soft-attenuate uncertain clips
    out = (np.log(np.clip(np.abs(weighted / TEMP_SCALE), 1e-9, None) + 1e-9)
           * np.sign(weighted)).reshape(-1, NUM_CLASSES)
    # simpler: just re-scale logits by entropy weight then run LSE→GM→cSEBBs
    weighted_logits = raw * (N_WINDOWS * w[:, :, None])  # re-scale so mean weight=1
    weighted_logits = weighted_logits.reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(weighted_logits, radius=1, beta=beta)
    after_gm = global_mean_blend(lse, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def selective_max_mean_gm_cp(logits, rare_thr=0.05, gm_alpha=0.175, cp_thr=0.06):
    """Selective max-mean pooling by class prevalence.
    Rare classes (file_mean < rare_thr) → max-pool (amplify weak evidence).
    Common classes → mean-pool (reduce noise).
    Then GlobalMean → cSEBBs.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_mean = X.mean(axis=1)              # (n_files, C)
    file_max  = X.max(axis=1)              # (n_files, C)
    # For each (file, class): use max if rare, mean if common
    is_rare = (file_mean < rare_thr)       # (n_files, C)
    pooled = np.where(is_rare, file_max, file_mean)  # (n_files, C)
    # Broadcast back to temporal sequence
    out = X * 0.5 + pooled[:, None, :] * 0.5  # blend local + pooled
    out = np.clip(out, 1e-6, 1 - 1e-6)
    pooled_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    after_gm = global_mean_blend(pooled_logits, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def beta_pipeline_blend(logits, betas=(3.0, 4.5, 6.0), gm_alpha=0.175, cp_thr=0.06):
    """Average of N pipelines with different LSE beta values.
    Multi-scale beta ensemble: covers different temporal granularities.
    """
    probs_list = []
    for b in betas:
        lse = logsumexp_pool(logits, radius=1, beta=b)
        after_gm = global_mean_blend(lse, alpha=gm_alpha)
        cp_logits = change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)
        probs_list.append(sigmoid(cp_logits))
    out = np.mean(probs_list, axis=0)
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE)


def trim_mean_pool(logits, drop_k=1, gm_alpha=0.175, cp_thr=0.06):
    """Trim-mean temporal pooling: drop bottom-k clips per class before mean-pooling.
    Then assign trimmed mean to all clips, blend with local → GM → cSEBBs.
    drop_k=1: drop 1 of 12 lowest clips (robust to single false negative).
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # Sort per (file, class) and drop bottom-k
    sorted_X = np.sort(X, axis=1)           # (n_files, T, C) sorted ascending
    trimmed = sorted_X[:, drop_k:, :].mean(axis=1)  # (n_files, C) — trimmed mean
    # Blend: 0.5*local + 0.5*trimmed_mean
    out = 0.5 * X + 0.5 * trimmed[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    trimmed_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    after_gm = global_mean_blend(trimmed_logits, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def diffusion_smooth_gm_cp(logits, lam=0.25, n_steps=2, gm_alpha=0.175, cp_thr=0.06):
    """Diffusion smoothing: iterative heat equation p_new = (1-2λ)*p + λ*(left+right).
    Zero-param, handles boundary naturally via edge-clamping.
    n_steps=2, λ=0.25 ≈ Gaussian(σ≈0.7).
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES).copy()
    for _ in range(n_steps):
        left  = np.concatenate([X[:, :1, :], X[:, :-1, :]], axis=1)
        right = np.concatenate([X[:, 1:, :], X[:, -1:, :]], axis=1)
        X = (1 - 2 * lam) * X + lam * left + lam * right
    out = np.clip(X, 1e-6, 1 - 1e-6)
    diff_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    after_gm = global_mean_blend(diff_logits, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def lse_confidence_weighted_gm_cp(logits, base_alpha=0.175, cp_thr=0.06):
    """LSE → Confidence-adaptive GM blend → cSEBBs.
    Per-clip α: low-confidence clips get more global anchoring.
    α(t) = base_alpha * (1 + (1 - conf(t))) where conf(t) = 1 - mean_entropy(t).
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    lse_logits = logsumexp_pool(logits, radius=1, beta=5.0)
    lse_probs  = sigmoid(lse_logits).reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # LSE output as reference for confidence
    H = -(lse_probs * np.log(lse_probs + 1e-9) + (1 - lse_probs) * np.log(1 - lse_probs + 1e-9))
    H_norm = H.mean(axis=2) / np.log(2)     # normalized entropy per clip (0=confident, 1=uncertain)
    # Adaptive alpha: [0.1, 0.3] range — uncertain clips get 3x more global
    alpha_t = base_alpha * (0.5 + H_norm)   # (n_files, N_WINDOWS) in [base*0.5, base*1.5]
    file_mean = lse_probs.mean(axis=1)       # (n_files, C)
    out = (1 - alpha_t[:, :, None]) * lse_probs + alpha_t[:, :, None] * file_mean[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def lse_softmax_blend_cp(logits, alpha=0.175, cp_thr=0.06):
    """LSE → SoftMax-blend (anchor to file MAX, not file MEAN) → cSEBBs.
    Targets 'heard once' rare species: file max amplifies the best clip.
    out(t,c) = (1-α)*local(t,c) + α*file_max(c)
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    lse_logits = logsumexp_pool(logits, radius=1, beta=5.0)
    lse_probs  = sigmoid(lse_logits).reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_max   = lse_probs.max(axis=1)       # (n_files, C)
    out = (1 - alpha) * lse_probs + alpha * file_max[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def spectral_lowpass_gm_cp(logits, cutoff_frac=0.5, gm_alpha=0.175, cp_thr=0.06):
    """Spectral low-pass: FFT the 12-clip sequence per class, zero high-freq, IFFT.
    cutoff_frac=0.5: keep bottom 50% of frequencies → smooth out rapid fluctuations.
    Then GlobalMean → cSEBBs.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    freqs = np.fft.rfft(X, axis=1)          # (n_files, T//2+1, C)
    cutoff = max(1, int(np.ceil(cutoff_frac * (N_WINDOWS // 2 + 1))))
    freqs[:, cutoff:, :] = 0.0
    out = np.fft.irfft(freqs, n=N_WINDOWS, axis=1)  # (n_files, N_WINDOWS, C)
    out = np.clip(out, 1e-6, 1 - 1e-6)
    lp_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    after_gm = global_mean_blend(lp_logits, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def lse_max_mean_blend_gm_cp(logits, mean_w=0.5, gm_alpha=0.175, cp_thr=0.06):
    """LSE → blend(file_max, file_mean) as anchor → cSEBBs.
    mean_w: weight of file_mean in anchor (1-mean_w goes to file_max).
    Like GlobalMean but uses a max-mean mixture as the file anchor.
    """
    lse_logits = logsumexp_pool(logits, radius=1, beta=4.5)
    n_files = lse_logits.shape[0] // N_WINDOWS
    lse_p = sigmoid(lse_logits).reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_max  = lse_p.max(axis=1)
    file_mean = lse_p.mean(axis=1)
    anchor = mean_w * file_mean + (1 - mean_w) * file_max  # (n_files, C)
    alpha = gm_alpha
    out = (1 - alpha) * lse_p + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def lse_b45_gm175_cp_fine(logits, cp_thr=0.05):
    """Best combo with fine-tuned cSEBBs threshold."""
    lse = logsumexp_pool(logits, radius=1, beta=4.5)
    after_gm = global_mean_blend(lse, alpha=0.175)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


# ── Round 15: MeanMax fine-tune / T fine-tune inside MeanMax / Combos ───────────

def mean_max_full(logits, entr_temp=0.2, max_w=0.5, beta=4.5, gm_alpha=0.175, cp_thr=0.06):
    """SoftEntr(temp) → LSE(beta) → MeanMax(max_w) anchor blend → cSEBBs.
    Fully parameterized version of mean_max_blend_gm_cp.
    max_w: weight of file_max in anchor (1-max_w = weight of file_mean).
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_max  = lse_3d.max(axis=1)
    file_mean = lse_3d.mean(axis=1)
    anchor = max_w * file_max + (1 - max_w) * file_mean
    out = (1 - gm_alpha) * lse_3d + gm_alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def variance_mean_max_cp(logits, max_w=0.5, beta=4.5, gm_alpha=0.175, cp_thr=0.06):
    """Variance weight → LSE → MeanMax anchor → cSEBBs."""
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="variance")
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_max  = lse_3d.max(axis=1)
    file_mean = lse_3d.mean(axis=1)
    anchor = max_w * file_max + (1 - max_w) * file_mean
    out = (1 - gm_alpha) * lse_3d + gm_alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── Round 17: w=1.0 fine-tune / radius / percentile anchor / double-pass ────────

def mean_max_full_r2(logits, entr_temp=0.2, max_w=1.0, beta=4.5, gm_alpha=0.175, cp_thr=0.06):
    """MeanMax with LSE radius=2 instead of default radius=1."""
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=2, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_max  = lse_3d.max(axis=1)
    file_mean = lse_3d.mean(axis=1)
    anchor = max_w * file_max + (1 - max_w) * file_mean
    out = (1 - gm_alpha) * lse_3d + gm_alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def percentile_anchor_cp(logits, entr_temp=0.2, pct=90, beta=4.5, gm_alpha=0.175, cp_thr=0.06):
    """SoftEntr → LSE → PercentileAnchor blend → cSEBBs.
    Uses file_p{pct} as anchor instead of file_max — softer than hard max.
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    anchor = np.percentile(lse_3d, pct, axis=1)  # (n_files, C)
    out = (1 - gm_alpha) * lse_3d + gm_alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def double_meanmax_cp(logits, entr_temp=0.2, max_w=1.0, gm_alpha=0.175, cp_thr=0.06):
    """Apply MeanMax(w=1.0) twice: second pass refines first pass output.
    Second pass inputs are the cSEBBs-smoothed logits from first pass.
    """
    # First pass
    first = mean_max_full(logits, entr_temp=entr_temp, max_w=max_w, gm_alpha=gm_alpha, cp_thr=cp_thr)
    # Second pass on first-pass output (already logits after cSEBBs)
    return mean_max_full(first, entr_temp=entr_temp, max_w=max_w, gm_alpha=gm_alpha, cp_thr=cp_thr)


def avgtopk_blend_lse_cp(logits, k=2, alpha=0.30, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → AvgTopK anchor (mean of top-k clips per class) → cSEBBs.
    k=1 → file_max (current best); k=12 → file_mean (GM).
    Source: ScienceDirect 2023 Avg-TopK pooling for audio CNNs.
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # AvgTopK anchor: sort descending, take mean of top-k
    sorted_p = np.sort(lse_3d, axis=1)[:, ::-1, :]   # (n_files, T, C) desc
    anchor = sorted_p[:, :k, :].mean(axis=1)          # (n_files, C) mean of top-k
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def noisy_or_blend_lse_cp(logits, alpha=0.30, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → NoisyOR anchor (1-∏(1-p)) → cSEBBs.
    NoisyOR amplifies rare confident clips multiplicatively.
    Source: DCASE 2018 MIL pooling comparison (Wang et al.).
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # Noisy-OR: P(class present in file) = 1 - ∏_{t}(1 - p(t,c))
    anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)      # (n_files, C)
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def power_mean_anchor_cp(logits, p=3.0, alpha=0.30, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → PowerMean anchor (mean(x^p))^(1/p) → cSEBBs.
    p=1 → file_mean (GM); p→∞ → file_max. p=3,4 are interpolations.
    Source: IIETA 2023 Global AvgTopK MaxPool; arXiv:2010.09985 Power Pooling.
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    eps = 1e-6
    anchor = (np.clip(lse_3d, eps, 1.0) ** p).mean(axis=1) ** (1.0 / p)  # (n_files, C)
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def dct_logit_lse_cp(logits, K=4, alpha=0.30, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → DCT low-pass on logits (keep K freqs) → GlobalMax blend → cSEBBs.
    DCT-II in logit space: edge-preserving, no wrap-around vs FFT.
    Source: signal processing / DCT energy compaction for short sequences.
    """
    from scipy.fftpack import dct, idct
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)  # (n_files, T, C)
    eps = 1e-6
    logit_3d = np.log(np.clip(lse_3d, eps, 1-eps) / np.clip(1-lse_3d, eps, 1-eps))
    # DCT-II along T dimension, zero high freqs, inverse
    L_dct = dct(logit_3d, axis=1, norm='ortho')
    L_dct[:, K:, :] = 0.0
    L_smooth = idct(L_dct, axis=1, norm='ortho')
    dct_probs = 1.0 / (1.0 + np.exp(-L_smooth))
    # Apply GlobalMax blend (alpha=0.30) on the smoothed probs
    file_max = dct_probs.max(axis=1)
    out = (1 - alpha) * dct_probs + alpha * file_max[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def noisy_or_blend_lse_r2_cp(logits, alpha=0.30, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE(r=2) → NoisyOR anchor → cSEBBs. Wider LSE radius."""
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=2, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def dual_anchor_nor_max_cp(logits, alpha=0.30, nor_w=0.5, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → DualAnchor(nor_w*NoisyOR + (1-nor_w)*GlobalMax) → cSEBBs.
    Combines the multiplicative NoisyOR and the extreme GlobalMax as a blended anchor.
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)   # (n_files, C)
    max_anchor = lse_3d.max(axis=1)                       # (n_files, C)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def dual_anchor_branch_ensemble_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                    alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                    w_a=0.5, entr_temp=0.1, cp_thr=0.06):
    """Ensemble of two DualAnchor branches (arithmetic mean in prob space then cSEBBs).
    Branch A: nw=0.40, a=0.38, b=5.15  (best A)
    Branch B: nw=0.30, a=0.40, b=6.0   (best B)
    Blend them: w_a*A + (1-w_a)*B before cSEBBs.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha) * lse_3d + alpha * anchor[:, None, :]

    out_a = _branch(nw_a, alpha_a, beta_a)
    out_b = _branch(nw_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def dual_anchor_logit_blend_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                                 entr_temp=0.1, cp_thr=0.06):
    """DualAnchor with logit-space blending instead of prob-space.
    Instead of: out = (1-a)*lse_prob + a*anchor_prob
    Do:         out = sigmoid((1-a)*logit(lse) + a*logit(anchor))
    Multiplicative in odds-space; preserves relative confidence better.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_anchor = lse_3d.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    anchor = np.clip(anchor, eps, 1 - eps)
    lse_clip = np.clip(lse_3d, eps, 1 - eps)
    logit_lse = np.log(lse_clip / (1 - lse_clip))       # (n_files, T, C)
    logit_anc = np.log(anchor / (1 - anchor))[:, None, :]  # (n_files, 1, C)
    out_logit = (1 - alpha) * logit_lse + alpha * logit_anc
    out = sigmoid(out_logit)
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def dual_anchor_geom_blend_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                                entr_temp=0.1, cp_thr=0.06):
    """DualAnchor with geometric mean blend: out = lse^(1-a) * anchor^a.
    Geometric mean weights: never drags high-conf clips to zero; softer anchor pull.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_anchor = lse_3d.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    anchor = np.clip(anchor, eps, 1 - eps)
    lse_3d = np.clip(lse_3d, eps, 1 - eps)
    out = (lse_3d ** (1 - alpha)) * (anchor[:, None, :] ** alpha)
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def dual_anchor_double_pass_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                                 alpha2=0.20, entr_temp=0.1, cp_thr=0.06):
    """Apply DualAnchor twice: first pass gives smoothed probs, second pass refines.
    alpha2 is the second-pass anchor strength (weaker, to avoid over-smoothing).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)

    def _dual_anchor_pass(probs_3d, a, nw):
        nor_anchor = 1.0 - np.prod(1.0 - probs_3d, axis=1)
        max_anchor = probs_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - a) * probs_3d + a * anchor[:, None, :]

    out1 = _dual_anchor_pass(lse_3d, alpha, nor_w)
    out1 = np.clip(out1, eps, 1 - eps)
    out2 = _dual_anchor_pass(out1, alpha2, nor_w)
    out2 = np.clip(out2, eps, 1 - eps)
    blended = (np.log(out2 / (1 - out2)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def dual_anchor_multi_beta_cp(logits, alpha=0.38, nor_w=0.40,
                                betas=(4.5, 5.15, 6.0), beta_weights=(0.33, 0.34, 0.33),
                                entr_temp=0.1, cp_thr=0.06):
    """Average multiple LSE beta outputs before DualAnchor (multi-scale soft-max).
    Reduces sensitivity to exact beta choice by blending 3 beta levels.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_avg = np.zeros((n_files, N_WINDOWS, NUM_CLASSES))
    for beta, bw in zip(betas, beta_weights):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_avg += bw * lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_avg, axis=1)
    max_anchor = lse_avg.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * lse_avg + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def triple_branch_ensemble_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                               alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                               alpha_c=0.40, nw_c=0.35, beta_c=5.5,
                               w_a=0.333, w_b=0.333, entr_temp=0.1, cp_thr=0.06):
    """3-way ensemble: Branch A + Branch B + Branch C (nw=0.35, a=0.40, b=5.5).
    Each branch independently ran DualAnchor; ensemble is weighted prob average.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return np.clip((1 - alpha) * lse_3d + alpha * anchor[:, None, :], eps, 1 - eps)

    w_c = 1.0 - w_a - w_b
    out = w_a * _branch(nw_a, alpha_a, beta_a) + w_b * _branch(nw_b, alpha_b, beta_b) + w_c * _branch(nw_c, alpha_c, beta_c)
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def branch_ensemble_min_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                            alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                            entr_temp=0.1, cp_thr=0.06):
    """Min-reduction ensemble of Branch A + Branch B.
    Conservative: take element-wise minimum (reduces false positives).
    Source: BirdCLEF 2024 1st place used min-reduction across model ensemble.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return np.clip((1 - alpha) * lse_3d + alpha * anchor[:, None, :], eps, 1 - eps)

    out = np.minimum(_branch(nw_a, alpha_a, beta_a), _branch(nw_b, alpha_b, beta_b))
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def quantile_mix_ensemble_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                              alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                              w_a=0.5, entr_temp=0.1, cp_thr=0.06):
    """Quantile-Mix: rank-normalize each branch within file then blend.
    Equalizes marginal distributions before combining — BirdCLEF 2025 top-2%.
    Rank within each (file, class) across time windows, normalize to [0,1].
    """
    from scipy.stats import rankdata
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch_probs(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return np.clip((1 - alpha) * lse_3d + alpha * anchor[:, None, :], eps, 1 - eps)

    def _rank_normalize(probs_3d):
        """Rank normalize per (file, class) across T windows → [0,1]."""
        out = np.zeros_like(probs_3d)
        for fi in range(probs_3d.shape[0]):
            for ci in range(probs_3d.shape[2]):
                ranks = rankdata(probs_3d[fi, :, ci])  # 1..T
                out[fi, :, ci] = (ranks - 1) / max(N_WINDOWS - 1, 1)
        return out

    prob_a = _branch_probs(nw_a, alpha_a, beta_a)
    prob_b = _branch_probs(nw_b, alpha_b, beta_b)
    # Quantile-Mix: average of raw blend and rank-normalized blend
    raw_blend = w_a * prob_a + (1 - w_a) * prob_b
    rank_a = _rank_normalize(prob_a)
    rank_b = _rank_normalize(prob_b)
    rank_blend = w_a * rank_a + (1 - w_a) * rank_b
    out = 0.5 * raw_blend + 0.5 * rank_blend
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def velocity_dual_anchor_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                             vel_w=0.5, entr_temp=0.1, cp_thr=0.06):
    """Velocity-Attention enhanced DualAnchor (TAP-Velocity, arxiv:2504.12670).
    Down-weight clips using a combined entropy+velocity score:
    velocity(t) = mean_class |p(t) - p(t-1)| — salient transition frames up-weighted.
    Final weight = vel_w * vel_weight + (1-vel_w) * entr_weight.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    # Entropy weights — flatten to (n_files*N_WINDOWS,)
    entr_w = _compute_clip_weights(logits, method="entropy", temp=entr_temp).reshape(-1)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # Velocity weights: |p(t) - p(t-1)| averaged over classes → (n_files*T,)
    raw_prob = sigmoid(raw)
    delta = np.abs(np.diff(raw_prob, axis=1, prepend=raw_prob[:, :1, :]))  # (n_files, T, C)
    vel_score = delta.mean(axis=2).reshape(-1)  # (n_files*T,)
    vel_norm = vel_score / (vel_score.mean() + eps) * N_WINDOWS  # normalize to same scale as entr_w
    # Combined weight (both are flat (708,))
    combined_w = vel_w * vel_norm + (1 - vel_w) * entr_w  # (708,)
    wl = (raw * combined_w.reshape(n_files, N_WINDOWS, 1)).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_anchor = lse_3d.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def _local_nor(probs_3d, window=5):
    """Local NOR anchor: NOR computed over a sliding window per clip.
    probs_3d: (n_files, T, C)
    Returns: (n_files, T, C) — spatially varying NOR (unlike global NOR which is constant per file).
    """
    n_files, T, C = probs_3d.shape
    half = window // 2
    out = np.zeros_like(probs_3d)
    for t in range(T):
        t_s = max(0, t - half)
        t_e = min(T, t + half + 1)
        window_probs = probs_3d[:, t_s:t_e, :]   # (n_files, w, C)
        out[:, t, :] = 1.0 - np.prod(1.0 - window_probs, axis=1)
    return out


def _gaussian_nor(probs_3d, sigma=2.0):
    """Gaussian-weighted NOR: clips closer in time contribute more to NOR.
    weight(t, t') = exp(-0.5*(t-t')^2/sigma^2), normalized so sum=1.
    Returns: (n_files, T, C) spatially-varying weighted NOR.
    """
    n_files, T, C = probs_3d.shape
    out = np.zeros_like(probs_3d)
    for t in range(T):
        dists = np.arange(T, dtype=float) - t
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
        weights /= weights.sum()
        # Gaussian-weighted NOR = 1 - prod((1-p)^w) = 1 - exp(sum(w*log(1-p)))
        log_comp = np.log(np.clip(1.0 - probs_3d, 1e-9, 1.0))   # (n_files, T, C)
        weighted_log = (log_comp * weights[None, :, None]).sum(axis=1)  # (n_files, C)
        out[:, t, :] = 1.0 - np.exp(weighted_log * T)  # re-scale so sigma=∞ → global NOR
    return out


def local_nor_dual_anchor_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                               window=5, entr_temp=0.1, cp_thr=0.06):
    """DualAnchor with LOCAL NOR anchor (sliding window) instead of global NOR.
    Local NOR(t) = 1 - prod(1-p(t')) for t' in [t-w//2, t+w//2].
    Spatially varying — reduces false boosts for isolated detections.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    local_nor = _local_nor(lse_3d, window=window)          # (n_files, T, C) — varies per t
    max_anchor = lse_3d.max(axis=1, keepdims=True)          # (n_files, 1, C)
    # anchor per clip: blend local NOR (spatially varying) + global max
    anchor_t = nor_w * local_nor + (1 - nor_w) * max_anchor  # (n_files, T, C)
    out = (1 - alpha) * lse_3d + alpha * anchor_t
    out = np.clip(out, eps, 1 - eps)
    out_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(out_logits, threshold=cp_thr, blend=0.4)


def local_global_branch_ensemble_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                      alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                      w_a=0.5, window_b=5, entr_temp=0.1, cp_thr=0.06):
    """BranchEns: Branch A = global NOR + GlobalMax (standard), Branch B = local NOR + LocalMax.
    Local branch provides clip-position-dependent correction; global branch captures species persistence.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    # Branch A: global NOR + GlobalMax (standard DualAnchor)
    lse_a = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_a))
    lse_3d_a = lse_a.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    global_nor_a = 1.0 - np.prod(1.0 - lse_3d_a, axis=1)
    global_max_a = lse_3d_a.max(axis=1)
    anchor_a = nw_a * global_nor_a + (1 - nw_a) * global_max_a
    out_a = np.clip((1 - alpha_a) * lse_3d_a + alpha_a * anchor_a[:, None, :], eps, 1 - eps)

    # Branch B: local NOR + local max-dilation (spatially varying)
    lse_b = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_b))
    lse_3d_b = lse_b.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    local_nor_b = _local_nor(lse_3d_b, window=window_b)    # (n_files, T, C)
    # Local max: max over same window (= max-dilation, spatially varying)
    half = window_b // 2
    local_max_b = np.zeros_like(lse_3d_b)
    for t in range(N_WINDOWS):
        t_s = max(0, t - half); t_e = min(N_WINDOWS, t + half + 1)
        local_max_b[:, t, :] = lse_3d_b[:, t_s:t_e, :].max(axis=1)
    anchor_b = nw_b * local_nor_b + (1 - nw_b) * local_max_b  # (n_files, T, C)
    out_b = np.clip((1 - alpha_b) * lse_3d_b + alpha_b * anchor_b, eps, 1 - eps)

    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def gauss_nor_dual_anchor_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                               sigma=2.0, entr_temp=0.1, cp_thr=0.06):
    """DualAnchor with Gaussian-weighted NOR anchor (sigma controls temporal scope).
    sigma=2: nearby clips count more; sigma=100: approaches global uniform NOR.
    Spatially smooth version of local NOR — no hard window boundary.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    gauss_nor = _gaussian_nor(lse_3d, sigma=sigma)         # (n_files, T, C)
    max_anchor = lse_3d.max(axis=1, keepdims=True)
    anchor_t = nor_w * gauss_nor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor_t
    out = np.clip(out, eps, 1 - eps)
    out_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(out_logits, threshold=cp_thr, blend=0.4)


def twoscale_nor_dual_anchor_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                                  local_w=0.5, window=5, entr_temp=0.1, cp_thr=0.06):
    """DualAnchor with two-scale NOR: blend local NOR (window=5) + global NOR.
    local_w: weight of local NOR vs global NOR in the NOR anchor component.
    Combines local temporal precision with global species-presence signal.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    global_nor = 1.0 - np.prod(1.0 - lse_3d, axis=1)              # (n_files, C)
    local_nor = _local_nor(lse_3d, window=window)                   # (n_files, T, C)
    # Blend: local_w * local + (1-local_w) * global (broadcast global)
    nor_blended = local_w * local_nor + (1 - local_w) * global_nor[:, None, :]
    max_anchor = lse_3d.max(axis=1, keepdims=True)
    anchor_t = nor_w * nor_blended + (1 - nor_w) * max_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor_t
    out = np.clip(out, eps, 1 - eps)
    out_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(out_logits, threshold=cp_thr, blend=0.4)


def local_global_full_branch_ensemble_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                           alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                           w_a=0.5, local_w=0.5, entr_temp=0.1, cp_thr=0.06):
    """BranchEns: Branch A standard (global NOR+GlobalMax), Branch B uses TwoScaleNOR+GlobalMax.
    TwoScaleNOR blends local(window=5) and global NOR before anchoring.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _global_branch(nw, alpha, beta):
        lse = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        g_nor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        g_max = lse_3d.max(axis=1)
        anc = nw * g_nor + (1 - nw) * g_max
        return np.clip((1 - alpha) * lse_3d + alpha * anc[:, None, :], eps, 1 - eps)

    def _twoscale_branch(nw, alpha, beta, lw):
        lse = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        g_nor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        l_nor = _local_nor(lse_3d, window=5)
        nor_b = lw * l_nor + (1 - lw) * g_nor[:, None, :]
        g_max = lse_3d.max(axis=1, keepdims=True)
        anc = nw * nor_b + (1 - nw) * g_max
        return np.clip((1 - alpha) * lse_3d + alpha * anc, eps, 1 - eps)

    out_a = _global_branch(nw_a, alpha_a, beta_a)
    out_b = _twoscale_branch(nw_b, alpha_b, beta_b, local_w)
    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R48: branch_combine sweep (min/geom/harm/max vs mean) ───────────────────────────────────────
# R47 verdict: TBD (running). Paper search Round 6 identified:
#   - BirdCLEF 2024 1st place used min-reduction across models (not mean) → more conservative
#   - min(P_A, P_B) is non-monotone across files → CAN change per-class AUC (unlike calibration)
#   - Geometric/harmonic means are also non-linear alternatives between the two branches
#   - All other 2024-2025 tricks are AUC-invariant (Platt scaling, additive class-mean) or N/A
# R48: sweep branch_combine parameter: min/geom/harm/max + cross combos with scale/cp_blend
#   Area 1: combine=min/geom/harm/max — 4 methods
#   Area 2: combine=min + out_scale sweep (if R47 finds best scale) — 4 methods
#   Area 3: combine=min + cp_blend/cp_thr combos — 4 methods
#   Area 4: combine=geom + same — 4 methods
#   Area 5: combine=harm + same — 4 methods

# ── R47: out_scale sweep (logit temperature before cSEBBs) + two-pass cSEBBs ────────────────────
# R46 verdict: cp_blend PEAKS at 0.60. 0.65→0.8136 (drops). cp_thr=0.05 ties at 0.8140.
#   All cSEBBs params and all 5 pipeline params now exhausted. Hard plateau at 0.8140.
# R47: Two untested dimensions:
#   (a) out_scale: logit temperature multiplier before cSEBBs (hardcoded TEMP_SCALE=1.15).
#       Higher → sharper cSEBBs boundaries. Lower → softer. Range: 0.80-1.50.
#   (b) Two-pass cSEBBs: apply change_point_segment_mean twice with independent params.
#       Second pass can clean up residual jumps left by first pass.
#   Area 1: out_scale sweep (0.80/0.90/1.00/1.05/1.10/1.20/1.30/1.50) — 8 methods
#   Area 2: two-pass cSEBBs (cp2_thr=0.06/0.05, cp2_blend=0.60/0.40) — 6 methods
#   Area 3: best out_scale + two-pass combos — 6 methods

# ── R46: cp_blend extension (0.65-1.0) + cp_thr interaction at best cp_blend ────────────────────
# R45 verdict: cp_blend monotone! 0.4→0.8137, 0.5→0.8139, 0.6→0.8140 NEW BEST. Not saturated.
#   entr_temp: 0.1 confirmed optimal; lse_radius=2 hurts; w_a/beta_a flat (±0.0001).
# R46: push cp_blend to 0.65/0.70/0.75/0.80/0.90/1.00 + cp_thr sweep at best cp_blend values.

# ── R45: SoftmaxRich pipeline parameter sweep (entr_temp, lse_radius, w_a, beta_a, cp_blend) ────
# R44 verdict: MMR anchor USELESS (0.79xx, ~0.015 below best). Alpha=0.38 confirmed optimal.
#   Full sweep a=0.35-0.48 all below 0.8137. HARD PLATEAU confirmed in SoftmaxRich family.
#   5 rounds of paper search: all applicable DCASE/BirdCLEF techniques exhausted.
# R45 strategy — sweep the 5 pipeline parameters never varied in R40-R44:
#   entr_temp (always 0.1): controls clip weight concentration. Lower→sharper. Higher→flatter.
#   lse_radius (always 1): 3-clip vs 5-clip LSE pool window.
#   w_a (always 0.55): BranchA vs BranchB blend weight.
#   beta_a (always 5.15): LSE sharpness of Branch A pool.
#   cp_blend (always 0.4): cSEBBs segment-mean blend strength.
#   Area 1: entr_temp sweep (0.05/0.15/0.20/0.30) — 4 methods
#   Area 2: lse_radius=2 — 1 method
#   Area 3: w_a sweep (0.45/0.50/0.60) — 3 methods
#   Area 4: beta_a sweep (4.8/5.0/5.3/5.5) — 4 methods
#   Area 5: cp_blend sweep (0.2/0.3/0.5/0.6) — 4 methods
#   Area 6: best-combo interactions — 4 methods

# ── R44: MaxMeanResidual anchor (BirdCLEF 2024 3rd place) + hybrid variants ─────────────────────
# R43 verdict: PLATEAU CONFIRMED. base sweep 0.00-0.25 all give 0.8134-0.8137 (flat).
#   R43.01 SoftmaxRich(T=0.15,base=0.20,boost=0.80)=0.8137 marginal NEW BEST (+0.0001).
#   AdaptAlpha hurts uniformly (0.8127-0.8132 < 0.8136). T=0.12/0.13 no gain.
#   DualBranch lower-base marginally worse. SoftmaxRich family fully saturated.
# R44 strategy — try MaxMeanResidual anchor (only untested technique from paper search):
#   BirdCLEF 2024 3rd place formula: P_out = P + scale * (P_max + P_mean - P_max_mean)
#   where P_max = per-class max across time, P_mean = per-class mean, P_max_mean = mean of P_max.
#   Adapted: mmr_anchor = file_max + file_mean - mean(file_max)
#   This emphasizes species with both HIGH PEAK and HIGH MEAN relative to file's avg activity.
#   Area 1: Pure MMR anchor (replace NOR): various scale/alpha
#   Area 2: MMR within SoftmaxRich framework (MMR anchor + richness-adaptive weight)
#   Area 3: Hybrid NOR + MMR anchor
#   Area 4: MMR anchor using LSE probs (not raw sigmoid)
#   Area 5: Best SoftmaxRich (R43.01) vs MMR cross-combos


def max_mean_residual_cp(logits, scale=0.8, alpha=0.38, beta=5.15,
                          alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55,
                          entr_temp=0.1, cp_thr=0.06):
    """MaxMeanResidual anchor (BirdCLEF 2024 3rd place adapted).
    anchor[f,c] = lse_max[f,c] + lse_mean[f,c] - mean_over_c(lse_max[f,c])
    out = (1-alpha) * lse_3d + alpha * clip(mmr_anchor)
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _mmr_branch(nw_a, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        lse_max = lse_3d.max(axis=1)             # (n_files, 234)
        lse_mean = lse_3d.mean(axis=1)            # (n_files, 234)
        lse_max_mean = lse_max.mean(axis=1, keepdims=True)  # (n_files, 1)
        mmr_anchor = lse_max + lse_mean - lse_max_mean   # (n_files, 234)
        mmr_anchor = np.clip(mmr_anchor, eps, 1 - eps)
        out = (1 - alpha_v) * lse_3d + alpha_v * mmr_anchor[:, None, :]
        return out

    out_a = _mmr_branch(0.40, alpha, beta)
    out_b = _mmr_branch(nw_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def mmr_softmax_rich_cp(logits, richness_thr=0.5, temp=0.15,
                         base_nw=0.20, max_boost=0.80,
                         alpha=0.38, beta=5.15,
                         alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55,
                         entr_temp=0.1, cp_thr=0.06):
    """MaxMeanResidual anchor combined with SoftmaxRich adaptive weight.
    anchor = MMR anchor (not NOR); nor_w computed from file richness via softmax.
    Blending: out = (1-alpha)*lse + alpha*(nor_w*mmr + (1-nor_w)*max)
    Actually: nor_w_arr replaces the NOR weight, but anchor is MMR not NOR.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    richness_sm = np.exp((richness - richness.max()) / temp)
    richness_sm = richness_sm / richness_sm.sum() * n_files
    richness_sm = np.clip(richness_sm, 0, 1)
    nw_arr = base_nw + max_boost * richness_sm  # (n_files,)

    def _mmr_branch_adaptive(nw_arr_in, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        lse_max = lse_3d.max(axis=1)
        lse_mean = lse_3d.mean(axis=1)
        lse_max_mean = lse_max.mean(axis=1, keepdims=True)
        mmr_anchor = np.clip(lse_max + lse_mean - lse_max_mean, eps, 1 - eps)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            # nw_arr_in[f] weights between MMR anchor and max anchor
            anchor_f = nw_arr_in[f] * mmr_anchor[f] + (1 - nw_arr_in[f]) * lse_max[f]
            out[f] = (1 - alpha_v) * lse_3d[f] + alpha_v * anchor_f[None, :]
        return out

    def _nor_branch_fixed(nw, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha_v) * lse_3d + alpha_v * anchor[:, None, :]

    out_a = _mmr_branch_adaptive(nw_arr, alpha, beta)
    out_b = _nor_branch_fixed(nw_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R43: SoftmaxRichness base/boost sweep + adaptive alpha ──────────────────────────────────────
# R42 verdict: R42.14 SoftmaxRich(T=0.15,thr=0.5,base=0.25,boost=0.75)=0.8136 NEW BEST!
#   T plateau confirmed: T=0.15 peak (T=0.20→0.8111, T=0.25→0.8104, monotone drop).
#   thr=0.50 confirmed optimal at T=0.15 (thr=0.48→0.8067, thr=0.52→0.8089, bell curve).
#   Boost gradient: boost=0.70→0.8126, boost=0.80→0.8135 (base=0.40), base=0.25,boost=0.75→0.8136.
#   Lower base wins: base=0.25 > base=0.40 at same nw_max=1.0. More contrast for poor files.
#   DualBranch(T=0.15)=0.8132 — slight gain vs single; base_B=0.30,boost_B=0.20 suboptimal.
# R43 strategy:
#   Area 1: Push base lower at nw_max=1.0 (base=0.20/0.15/0.10/0.05/0.00 + matching boost)
#            — more NOR-vs-max contrast for poor files
#   Area 2: Slight extrapolation with optimized base (nw_max=1.05/1.10)
#   Area 3: Adaptive alpha — BOTH nor_w AND blend weight alpha scale with richness
#            alpha_arr = base_alpha + alpha_boost * richness_sm (rich files get stronger anchor)
#   Area 4: T fine-tune (T=0.12/0.13) between T=0.10 and T=0.15
#   Area 5: DualBranch with lower base for both branches


def softmax_anchor_adaptive_alpha_cp(logits, richness_thr=0.5, temp=0.15,
                                      base_nw=0.25, max_boost=0.75,
                                      base_alpha=0.30, alpha_boost=0.10,
                                      entr_temp=0.1, beta=5.15,
                                      alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """SoftmaxRichness with BOTH nor_w AND alpha adaptive to file richness.
    Branch A: nw_arr = base_nw + max_boost * sm; alpha_arr = base_alpha + alpha_boost * sm.
    Branch B: fixed nw_b, alpha_b.
    Rich files get stronger NOR anchor AND stronger anchor blending correction.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    richness_sm = np.exp((richness - richness.max()) / temp)
    richness_sm = richness_sm / richness_sm.sum() * n_files
    richness_sm = np.clip(richness_sm, 0, 1)
    nw_arr = base_nw + max_boost * richness_sm
    alpha_arr = base_alpha + alpha_boost * richness_sm

    def _branch_adaptive_alpha(nw_arr_in, alpha_arr_in, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            anchor_f = nw_arr_in[f] * nor_anchor[f] + (1 - nw_arr_in[f]) * max_anchor[f]
            out[f] = (1 - alpha_arr_in[f]) * lse_3d[f] + alpha_arr_in[f] * anchor_f[None, :]
        return out

    def _branch_fixed(nw, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha_v) * lse_3d + alpha_v * anchor[:, None, :]

    out_a = _branch_adaptive_alpha(nw_arr, alpha_arr, beta)
    out_b = _branch_fixed(nw_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R42: SoftmaxRichness temp push + T=0.15 param sweep ─────────────────────────────────────────
# R41 verdict: T=0.15,thr=0.5→0.8122 NEW BEST. Temp trend monotone: T=0.02→0.7988,
#   T=0.05→0.8077, T=0.075→0.8098, T=0.10→0.8113, T=0.15→0.8122 (still improving!).
#   thr=0.5 is optimal at T=0.10 (thr=0.35→0.8008, thr=0.45→0.8028, thr=0.5→0.8113,
#   thr=0.55→0.8046, thr=0.6→0.8042, thr=0.7→0.8026). boost=0.60 still optimal at thr=0.5.
#   DualBranch no gain (0.8102). cp_thr=0.05 negligible (+0.0001).
# R42 strategy:
#   Area 1: temp push beyond T=0.15 at thr=0.5 (T=0.20/0.25/0.30/0.40/0.50)
#            — find true peak before uniform plateau
#   Area 2: T=0.15 + thr fine-tune (0.45/0.48/0.52/0.55/0.6) — peak may shift at T=0.15
#   Area 3: T=0.15 + boost/base sweep (boost=0.70/0.80, base=0.30/0.25)
#   Area 4: DualBranch at T=0.15 + best T combo
#   Area 5: Cross sweep — best T from Area 1 × best params


# ── R41: SoftmaxRichness deep-dive — thr/temp/boost/base/dual-branch sweep ─────────────────────
# R40 verdict: SoftmaxRich(T=0.10,thr=0.5)=0.8113 MASSIVE BREAKTHROUGH (+0.0035 vs R39 best).
#   SoftmaxRich(T=0.10,thr=0.4)=0.8074, so thr=0.5 adds +0.0039 to softmax.
#   Gamma plateau: g=7.0=g=10.0=0.8082, g=15.0→0.8074, g=20.0→0.8069 — peak at g=7-10.
#   AdaptRichPow cluster stable at 0.8081-0.8082 with many param combos.
# R41 strategy — full SoftmaxRich sweep:
#   Area 1: thr sweep (0.35/0.45/0.55/0.6/0.7) at T=0.10 — find true optimal
#   Area 2: temp sweep at thr=0.5 (T=0.02/0.05/0.075/0.15) — sharp vs smooth
#   Area 3: boost/base sweep at (T=0.10,thr=0.5) — (boost=0.70/0.80, base=0.30/0.25)
#   Area 4: Combined best params (thr=0.5, T=0.05/0.075, boost=0.70)
#   Area 5: SoftmaxRich DualBranch — both A and B use adaptive softmax nw
#   Area 6: cp_thr sweep at best config


def softmax_anchor_richness_dual_cp(logits, richness_thr=0.5, temp=0.1,
                                     base_nw_a=0.40, max_boost_a=0.60,
                                     base_nw_b=0.30, max_boost_b=0.20,
                                     entr_temp=0.1, alpha=0.38, beta=5.15,
                                     alpha_b=0.40, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """Both Branch A and Branch B use softmax-normalized richness adaptive nor_w.
    nw_arr_a = base_nw_a + max_boost_a * softmax(richness/T)
    nw_arr_b = base_nw_b + max_boost_b * softmax(richness/T)
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    richness_sm = np.exp((richness - richness.max()) / temp)
    richness_sm = richness_sm / richness_sm.sum() * n_files
    richness_sm = np.clip(richness_sm, 0, 1)
    nw_arr_a = base_nw_a + max_boost_a * richness_sm
    nw_arr_b = base_nw_b + max_boost_b * richness_sm

    def _branch_adaptive(nw_arr_in, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            anchor_f = nw_arr_in[f] * nor_anchor[f] + (1 - nw_arr_in[f]) * max_anchor[f]
            out[f] = (1 - alpha_v) * lse_3d[f] + alpha_v * anchor_f[None, :]
        return out

    out_a = _branch_adaptive(nw_arr_a, alpha, beta)
    out_b = _branch_adaptive(nw_arr_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R40: AdaptRichPow extreme gamma + g=5.0 thr/boost/base combos + SoftmaxRichness ────────────
# R39 verdict: gamma monotonically improving: g=2.5→0.8066, g=3.0→0.8069, g=4.0→0.8074,
#   g=5.0→0.8078 (NEW BEST). thr=0.5 beats thr=0.4 at g=2.0 (0.8074 vs 0.8062).
#   thr=0.3 CATASTROPHIC (0.7972 — lower thr means more species count as active, richness
#   becomes less discriminative). DualPow(gA=gB=2.0)=0.8063 (no gain over single-branch).
# R40 strategy:
#   Area 1: Push gamma extreme (7,10,15,20) — does monotone trend continue?
#   Area 2: g=5.0 + thr=0.5/0.6 (thr=0.5 found better at g=2.0; test at g=5.0)
#   Area 3: g=5.0 + boost=0.70/0.80 (more boost for the small set of rich files)
#   Area 4: g=5.0 + lower base_nw (more dynamic range: base=0.30/0.25 + boost=0.70)
#   Area 5: SoftmaxRichness — apply softmax(richness/T) instead of power-law norm
#            Equivalent at T→0 to BinaryGate (which hurt), but continuous like power-law

def softmax_anchor_richness_cp(logits, richness_thr=0.4, temp=0.1,
                                 base_nw=0.40, max_boost=0.60,
                                 entr_temp=0.1, alpha=0.38, beta=5.15,
                                 alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06,
                                 cp_blend=0.4, lse_radius=1, out_scale=None,
                                 cp2_thr=None, cp2_blend=None, branch_combine="mean",
                                 seg_max_w=0.0, cp_blend_boost=0.0):
    """AdaptAnchorRichness with softmax-normalized richness instead of linear normalization.
    nw_arr = base_nw + max_boost * softmax(richness / temp)
    Low temp → sharp peak on richest file (similar to gamma→∞ in power-law).
    Distinct from power-law: softmax is translation-invariant, power-law is scale-invariant.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    # Softmax normalization: exp(r/T) / sum(exp(r/T)) * n_files
    richness_sm = np.exp((richness - richness.max()) / temp)
    richness_sm = richness_sm / richness_sm.sum() * n_files  # scale to [0, ~n_files]
    richness_sm = np.clip(richness_sm, 0, 1)                 # clip to [0,1]
    nw_arr = base_nw + max_boost * richness_sm

    def _branch_sm(nw_arr_in, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=lse_radius, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            anchor_f = nw_arr_in[f] * nor_anchor[f] + (1 - nw_arr_in[f]) * max_anchor[f]
            out[f] = (1 - alpha_v) * lse_3d[f] + alpha_v * anchor_f[None, :]
        return out

    def _branch_sm_fixed(nw, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=lse_radius, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha_v) * lse_3d + alpha_v * anchor[:, None, :]

    out_a = _branch_sm(nw_arr, alpha, beta)
    out_b = _branch_sm_fixed(nw_b, alpha_b, beta_b)
    if branch_combine == "min":
        out = np.minimum(out_a, out_b)
    elif branch_combine == "geom":
        out = np.sqrt(np.clip(out_a * out_b, eps, 1 - eps))
    elif branch_combine == "harm":
        out = 2 * out_a * out_b / (out_a + out_b + eps)
    elif branch_combine == "max":
        out = np.maximum(out_a, out_b)
    else:  # "mean" (default)
        out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    scale = out_scale if out_scale is not None else TEMP_SCALE
    blended = (np.log(out / (1 - out)) * scale).reshape(-1, NUM_CLASSES)
    if seg_max_w > 0.0:
        # Segment-max blend: use max (or max/mean mix) within each cSEBBs segment
        result = change_point_segment_max_blend(blended, threshold=cp_thr,
                                                blend=cp_blend, max_w=seg_max_w)
    elif cp_blend_boost > 0.0:
        # Richness-adaptive cp_blend: rich files get stronger segment smoothing
        # richness_sm is per-file [0,1]; cp_blend_eff[f] = cp_blend + cp_blend_boost * richness_sm[f]
        probs_b = sigmoid(blended)
        n_files_b = probs_b.shape[0] // N_WINDOWS
        X_b = probs_b.reshape(n_files_b, N_WINDOWS, NUM_CLASSES)
        out_b2 = X_b.copy()
        diffs_b = np.abs(np.diff(X_b, axis=1))
        cp_blend_arr = np.clip(cp_blend + cp_blend_boost * richness_sm, 0.0, 1.0)
        for f in range(n_files_b):
            bl_f = cp_blend_arr[f]
            for c in range(NUM_CLASSES):
                boundaries = [0] + [t + 1 for t in range(N_WINDOWS - 1)
                                     if diffs_b[f, t, c] > cp_thr] + [N_WINDOWS]
                for i in range(len(boundaries) - 1):
                    s, e = boundaries[i], boundaries[i + 1]
                    seg_mean = X_b[f, s:e, c].mean()
                    out_b2[f, s:e, c] = bl_f * seg_mean + (1 - bl_f) * X_b[f, s:e, c]
        out_b2 = np.clip(out_b2, 1e-6, 1 - 1e-6)
        result = (np.log(out_b2 / (1 - out_b2)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    else:
        result = change_point_segment_mean(blended, threshold=cp_thr, blend=cp_blend)
    if cp2_thr is not None:
        result = change_point_segment_mean(result, threshold=cp2_thr,
                                           blend=cp2_blend if cp2_blend is not None else cp_blend)
    return result


# ── R39: AdaptRichPow gamma fine-tune + gamma=2.0 parameter sweep + SilenceCut + DualPow ───────
# R38 verdict: AdaptRichPow(g=2.0,boost=0.60)=0.8062 NEW BEST! Convex power-law wins.
# gamma=1.5→0.8058, gamma=2.0→0.8062, gamma=0.5/0.3→0.8051/0.8048 (concave HURTS).
# BinaryGate HURTS (0.7984-0.8030) — step function bad, gradient matters.
# AdaptExtrap(boost>0.60) diminishes: 0.70→0.8055, 0.80→0.8054, 1.00→0.8052.
# DualAdapt all ≤0.8054 (no gain over single-branch adapt).
# R39 strategy: (a) push gamma higher (2.5, 3.0, 4.0, 5.0), (b) re-sweep boost with gamma=2.0,
# (c) gamma=2.0 + SilenceCut, (d) gamma=2.0 + cp_thr=0.05, (e) dual-pow both branches.

def adapt_anchor_dual_pow_cp(logits, base_nw_a=0.40, max_boost_a=0.60, gamma_a=2.0,
                               base_nw_b=0.30, max_boost_b=0.20, gamma_b=1.0,
                               richness_thr=0.4, entr_temp=0.1,
                               alpha_a=0.38, beta_a=5.15,
                               alpha_b=0.40, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """Both branches use power-law richness adaptive nor_w with independent gamma.
    Branch A: gamma_a (convex, selective); Branch B: gamma_b (less selective).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    richness_norm = richness / (richness.max() + eps)
    nw_arr_a = base_nw_a + max_boost_a * np.power(richness_norm + eps, gamma_a)
    nw_arr_b = base_nw_b + max_boost_b * np.power(richness_norm + eps, gamma_b)

    def _branch_pow(nw_arr_in, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            anchor_f = nw_arr_in[f] * nor_anchor[f] + (1 - nw_arr_in[f]) * max_anchor[f]
            out[f] = (1 - alpha_v) * lse_3d[f] + alpha_v * anchor_f[None, :]
        return out

    out_a = _branch_pow(nw_arr_a, alpha_a, beta_a)
    out_b = _branch_pow(nw_arr_b, alpha_b, beta_b)
    out = np.clip(w_a * out_a + (1 - w_a) * out_b, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R38: AdaptAnchor power-law richness / dual adaptive branches / binary gate ─────────────────
# R37 verdict: boost=0.40/0.50/0.60 all ~0.8057 (plateau at nw_max=1.0 boundary).
# thr=0.4 optimal; higher thr hurts. AdaptAnchorVar hurts >pen=0.1; SilenceCut neutral.
# R38 strategy:
#   (a) Power-law richness curve: nw = base + boost * richness_norm^gamma (gamma!=1 breaks linear)
#   (b) Dual adaptive branches: BOTH Branch A and B use richness-based nw (different slopes)
#   (c) Binary richness gate: step function (high_nw if rich, low_nw if sparse)
#   (d) AdaptAnchor(boost=0.60) + SilenceCut: test best boost with SilenceCut (R37 only tested 0.30)
#   (e) Extrapolation (nw > 1.0): allow boost > 0.60 so richest files get nw > 1.0 → pure-NOR+
#   (f) Per-species adaptive alpha: scale alpha by file-level species activity (novel axis)

def adapt_anchor_rich_pow_cp(logits, base_nw=0.40, richness_thr=0.4, max_boost=0.60,
                               gamma=0.5, entr_temp=0.1, alpha=0.38, beta=5.15,
                               alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """AdaptAnchorRichness with power-law richness normalization.
    nw_arr = base_nw + max_boost * richness_norm^gamma.
    gamma < 1: concave curve — bigger boost for moderately sparse files.
    gamma > 1: convex curve — reserve full boost only for richest files.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    richness_norm = richness / (richness.max() + eps)
    nw_arr = base_nw + max_boost * np.power(richness_norm + eps, gamma)

    def _branch_adap(nw_arr_in, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            anchor_f = nw_arr_in[f] * nor_anchor[f] + (1 - nw_arr_in[f]) * max_anchor[f]
            out[f] = (1 - alpha_v) * lse_3d[f] + alpha_v * anchor_f[None, :]
        return out

    def _branch_fixed(nw, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha_v) * lse_3d + alpha_v * anchor[:, None, :]

    out_a = _branch_adap(nw_arr, alpha, beta)
    out_b = _branch_fixed(nw_b, alpha_b, beta_b)
    out = np.clip(w_a * out_a + (1 - w_a) * out_b, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def adapt_anchor_dual_branch_cp(logits, base_nw_a=0.40, max_boost_a=0.60,
                                  base_nw_b=0.30, max_boost_b=0.20,
                                  richness_thr=0.4, entr_temp=0.1,
                                  alpha_a=0.38, beta_a=5.15,
                                  alpha_b=0.40, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """Both Branch A and Branch B use richness-adaptive nor_w (different slopes).
    A: aggressive adaptation (large boost); B: conservative (small boost).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    richness_norm = richness / (richness.max() + eps)
    nw_arr_a = base_nw_a + max_boost_a * richness_norm
    nw_arr_b = base_nw_b + max_boost_b * richness_norm

    def _branch_adap2(nw_arr_in, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            anchor_f = nw_arr_in[f] * nor_anchor[f] + (1 - nw_arr_in[f]) * max_anchor[f]
            out[f] = (1 - alpha_v) * lse_3d[f] + alpha_v * anchor_f[None, :]
        return out

    out_a = _branch_adap2(nw_arr_a, alpha_a, beta_a)
    out_b = _branch_adap2(nw_arr_b, alpha_b, beta_b)
    out = np.clip(w_a * out_a + (1 - w_a) * out_b, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def adapt_anchor_binary_cp(logits, richness_thr=0.4, high_nw=1.0, low_nw=0.30,
                             entr_temp=0.1, alpha=0.38, beta=5.15,
                             alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """Binary richness gate: files above richness_thr → high_nw; else → low_nw.
    Step function instead of linear ramp — tests whether the gradient matters or just the split.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    # Binary: above median richness → high_nw; below → low_nw
    richness_median = np.median(richness)
    nw_arr = np.where(richness > richness_median, high_nw, low_nw)

    def _branch_adap3(nw_arr_in, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            anchor_f = nw_arr_in[f] * nor_anchor[f] + (1 - nw_arr_in[f]) * max_anchor[f]
            out[f] = (1 - alpha_v) * lse_3d[f] + alpha_v * anchor_f[None, :]
        return out

    def _branch_fixed3(nw, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha_v) * lse_3d + alpha_v * anchor[:, None, :]

    out_a = _branch_adap3(nw_arr, alpha, beta)
    out_b = _branch_fixed3(nw_b, alpha_b, beta_b)
    out = np.clip(w_a * out_a + (1 - w_a) * out_b, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R37: AdaptAnchor fine-tune + combined AdaptAnchor+VarAnchor / Density / SilenceCut ────────
# R36 verdict: AdaptAnchorRichness NEW BEST 0.8054 (boost=0.30, thr=0.4, base=0.40).
# VarAnchor(pen=0.3)=0.8029, DensityGate(max=10,g=0.5)=0.8024 also strong.
# R37 strategy: (a) fine-tune AdaptAnchor params around best, (b) combine AdaptAnchor with
# VarianceAnchor in one pass (richness-based nw + variance-scaled alpha simultaneously),
# (c) AdaptAnchor + DensityGate on 3D probs, (d) AdaptAnchor + SilenceCut on 3D probs.
# NOTE: All combinations implemented as single new functions (no function chaining).

def adapt_anchor_var_cp(logits, base_nw=0.40, richness_thr=0.4, max_boost=0.30,
                         var_penalty=0.3, entr_temp=0.1, alpha=0.38, beta=5.15,
                         alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """AdaptAnchorRichness + VarianceAnchor combined in one pass → cSEBBs.
    Branch A: richness-based adaptive nor_w AND variance-scaled alpha per species.
    Branch B: fixed params (same as BranchEns).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    # Compute file richness for adaptive nor_w
    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    richness_norm = richness / (richness.max() + eps)
    nw_arr = base_nw + max_boost * richness_norm

    def _branch_a(nw_arr, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        # Variance-scaled alpha (per species, per file)
        var_per_species = lse_3d.var(axis=1)
        var_norm = var_per_species / (var_per_species.max(axis=1, keepdims=True) + eps)
        alpha_eff = alpha_v * (1 - var_penalty * var_norm)  # (n_files, C)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            nw_f = nw_arr[f]
            anchor_f = nw_f * nor_anchor[f] + (1 - nw_f) * max_anchor[f]
            out[f] = (1 - alpha_eff[f])[None, :] * lse_3d[f] + alpha_eff[f][None, :] * anchor_f[None, :]
        return out

    def _branch_b(nw, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha_v) * lse_3d + alpha_v * anchor[:, None, :]

    out_a = _branch_a(nw_arr, alpha, beta)
    out_b = _branch_b(nw_b, alpha_b, beta_b)
    out = np.clip(w_a * out_a + (1 - w_a) * out_b, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def adapt_anchor_density_cp(logits, base_nw=0.40, richness_thr=0.4, max_boost=0.30,
                              max_active=10, gate=0.5, active_thr=0.3,
                              entr_temp=0.1, alpha=0.38, beta=5.15,
                              alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """AdaptAnchorRichness + DensityGate on 3D probs → cSEBBs.
    Adaptive nor_w to handle file richness; then gate dense/noisy clips.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    richness_norm = richness / (richness.max() + eps)
    nw_arr = base_nw + max_boost * richness_norm

    def _branch_adaptive(nw_arr_in, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            anchor_f = nw_arr_in[f] * nor_anchor[f] + (1 - nw_arr_in[f]) * max_anchor[f]
            out[f] = (1 - alpha_v) * lse_3d[f] + alpha_v * anchor_f[None, :]
        return out

    def _branch_fixed(nw, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha_v) * lse_3d + alpha_v * anchor[:, None, :]

    out_a = _branch_adaptive(nw_arr, alpha, beta)
    out_b = _branch_fixed(nw_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b  # (n_files, N_WINDOWS, C)
    # Density gate on combined 3D probs
    result = out.copy()
    for f in range(n_files):
        for t in range(N_WINDOWS):
            n_active = (out[f, t] > active_thr).sum()
            if n_active > max_active:
                result[f, t] = out[f, t] * gate
    out = np.clip(result, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def adapt_anchor_silence_cp(logits, base_nw=0.40, richness_thr=0.4, max_boost=0.30,
                              silence_thr=0.10, silence_factor=0.5,
                              entr_temp=0.1, alpha=0.38, beta=5.15,
                              alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """AdaptAnchorRichness + SilenceCut on 3D probs → cSEBBs.
    Adaptive nor_w to handle file richness; then suppress files where max < silence_thr.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)
    richness = (file_means > richness_thr).mean(axis=1)
    richness_norm = richness / (richness.max() + eps)
    nw_arr = base_nw + max_boost * richness_norm

    def _branch_adaptive2(nw_arr_in, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            anchor_f = nw_arr_in[f] * nor_anchor[f] + (1 - nw_arr_in[f]) * max_anchor[f]
            out[f] = (1 - alpha_v) * lse_3d[f] + alpha_v * anchor_f[None, :]
        return out

    def _branch_fixed2(nw, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha_v) * lse_3d + alpha_v * anchor[:, None, :]

    out_a = _branch_adaptive2(nw_arr, alpha, beta)
    out_b = _branch_fixed2(nw_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b  # (n_files, N_WINDOWS, C)
    # Silence cut: per-file, per-class deflation
    file_max = out.max(axis=1)             # (n_files, C)
    silent_mask = file_max < silence_thr   # (n_files, C)
    for f in range(n_files):
        out[f, :, silent_mask[f]] *= silence_factor
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R36: Adaptive anchor / density gate / variance anchor / harmonic mean anchor ─────────────
# Honest assessment after 35 rounds / 639 methods: plateau is a model-quality floor.
# R36 targets the only remaining structural gap: our anchor computation (NOR + max blend) uses
# fixed parameters. These methods make the anchor adaptive to per-file/per-species statistics:
#   1) AdaptiveAnchorRichness: nor_w adapts to file "richness" (# active species detected)
#   2) DensityClipGate: suppress clips where too many species active (noisy/OOD clips)
#   3) VarianceWeightedAnchor: down-weight anchor for species with high temporal variance
#      (inconsistent detections → noisy → should trust anchor less)
#   4) HarmonicMeanAnchor: harmonic mean across clips as anchor (penalizes any silent clip
#      for that species — effectively an "AND" gate vs NOR's "OR" gate)

def adaptive_anchor_richness_cp(logits, base_nw=0.40, richness_thr=0.3, max_boost=0.20,
                                  entr_temp=0.1, alpha=0.38, beta=5.15,
                                  alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns with adaptive nor_w based on file richness → cSEBBs.
    Rich files (many active species) → higher nor_w (stronger NOR anchor).
    Poor files (few active species) → lower nor_w (trust max-anchor more).
    Captures the idea that NOR is better at spreading credit in multi-species files.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch_adaptive(nw_arr, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        # per-file nw
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            nw_f = nw_arr[f]
            anchor_f = nw_f * nor_anchor[f] + (1 - nw_f) * max_anchor[f]
            out[f] = (1 - alpha_v) * lse_3d[f] + alpha_v * anchor_f[None, :]
        return out

    # Compute file richness = fraction of species with file-mean > richness_thr
    probs_3d = sigmoid(raw)
    file_means = probs_3d.mean(axis=1)                  # (n_files, C)
    richness = (file_means > richness_thr).mean(axis=1)  # (n_files,)
    richness_norm = richness / (richness.max() + eps)    # normalize to [0,1]
    nw_arr = base_nw + max_boost * richness_norm         # adaptive nor_w per file

    out_a = _branch_adaptive(nw_arr, alpha, beta)
    nw_b_arr = np.full(n_files, nw_b)
    out_b = _branch_adaptive(nw_b_arr, alpha_b, beta_b)
    out = np.clip(w_a * out_a + (1 - w_a) * out_b, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def density_clip_gate_cp(logits, max_active=8, gate=0.3, active_thr=0.4, entr_temp=0.1,
                          alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                          alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → density-based clip gate → cSEBBs.
    For each clip, count species with prob > active_thr. Clips with > max_active active species
    are "dense/noisy" and blended toward zero. Distinct from EnergyOOD (uses logsumexp energy)
    and ClipTopK (suppresses bottom species, keeps all clips). This suppresses entire OOD clips.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    out_3d = _branchens_probs(logits, entr_temp, alpha_a, nw_a, beta_a,
                               alpha_b, nw_b, beta_b, w_a)
    result = out_3d.copy()
    for f in range(n_files):
        for t in range(N_WINDOWS):
            n_active = (out_3d[f, t] > active_thr).sum()
            if n_active > max_active:
                result[f, t] = out_3d[f, t] * gate
    P_out = np.clip(result, eps, 1 - eps)
    blended = (np.log(P_out / (1 - P_out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def variance_anchor_cp(logits, var_penalty=0.5, entr_temp=0.1,
                        alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                        alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → variance-weighted anchor → cSEBBs.
    High per-species temporal variance → inconsistent detections → noisy.
    Scale the alpha (anchor blend) by (1 - var_penalty * normalized_variance) per species.
    Low variance species keep full anchor; high variance species get less anchor correction.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha_scalar, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)   # (n_files, C)
        max_anchor = lse_3d.max(axis=1)                      # (n_files, C)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor    # (n_files, C)
        # per-species variance gate on alpha
        var_per_species = lse_3d.var(axis=1)                 # (n_files, C)
        var_norm = var_per_species / (var_per_species.max(axis=1, keepdims=True) + eps)
        alpha_eff = alpha_scalar * (1 - var_penalty * var_norm)  # (n_files, C)
        out = np.zeros_like(lse_3d)
        for f in range(n_files):
            out[f] = (1 - alpha_eff[f][None, :]) * lse_3d[f] + alpha_eff[f][None, :] * anchor[f][None, :]
        return out

    out_a = _branch(nw_a, alpha_a, beta_a)
    out_b = _branch(nw_b, alpha_b, beta_b)
    out = np.clip(w_a * out_a + (1 - w_a) * out_b, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def hmean_anchor_cp(logits, entr_temp=0.1,
                     alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                     alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns with harmonic-mean anchor instead of NOR+max blend → cSEBBs.
    Harmonic mean = N / sum(1/p_t): requires ALL clips active → "AND" gate.
    Distinct from NOR (OR gate: any clip active) and max (best clip).
    Penalizes species with any near-zero clip — forces consistent detection across file.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha_v, beta_v):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_v))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        # Harmonic mean: N / sum(1 / (p + eps)) — "AND" anchor
        hmean = N_WINDOWS / np.sum(1.0 / (lse_3d + eps), axis=1)  # (n_files, C)
        hmean = np.clip(hmean, eps, 1 - eps)
        # Blend hmean with max (NOR replaced by harmonic mean)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * hmean + (1 - nw) * max_anchor
        return (1 - alpha_v) * lse_3d + alpha_v * anchor[:, None, :]

    out_a = _branch(nw_a, alpha_a, beta_a)
    out_b = _branch(nw_b, alpha_b, beta_b)
    out = np.clip(w_a * out_a + (1 - w_a) * out_b, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R35: Within-clip rank sharpening / bidirectional consistency / file-species prior ─────────
# Paper sources:
#   ClipRankSharpen: BirdCLEF+ 2025 3rd-place writeup (tekkix.com/articles/ai/2025/07/birdclef-2025)
#     Power-transform α + rank-based suppression of non-top-n species within each 10s clip.
#     ALL 620+ prior methods operate on 1D temporal axis per species OR on file-level anchors.
#     THIS is the first within-clip CROSS-SPECIES rank operation. For soundscapes with 0-3 active
#     species per clip, suppressing bottom species dramatically reduces FP noise.
#   BidirConsistency: Voxaboxen (arXiv:2503.02389, Mar 2025) bidirectional consistency filtering.
#     Forward + time-reversed score tracks; geometric mean of consistent predictions. Detections
#     consistent across both directions reinforced; spurious detections suppressed.
#   FileSpeciesPrior: Prior from BirdCLEF file-level statistics: mean soundscape file has 5-20 active
#     species. Soft-suppress species below file-level rank threshold. Cross-species, file-level.

def clip_rank_sharpen_cp(logits, alpha=2.0, entr_temp=0.1,
                          alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                          alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → within-clip power-sharpening (p^alpha per clip) → cSEBBs.
    Concentrates mass on top-scoring species per clip; reduces FP from low-confidence species.
    Orthogonal to all temporal methods — operates on cross-species ranking within each clip.
    Source: BirdCLEF+ 2025 3rd place (power-transform within-clip rank sharpening).
    """
    eps = 1e-6
    out_3d = _branchens_probs(logits, entr_temp, alpha_a, nw_a, beta_a,
                               alpha_b, nw_b, beta_b, w_a)   # (n_files, 12, C)
    P_sharp = np.power(np.clip(out_3d, eps, 1 - eps), alpha)
    # re-scale so max per clip stays the same (preserve ranking signal)
    max_orig = out_3d.max(axis=2, keepdims=True)
    max_sharp = P_sharp.max(axis=2, keepdims=True)
    P_out = P_sharp * (max_orig / (max_sharp + eps))
    P_out = np.clip(P_out, eps, 1 - eps)
    blended = (np.log(P_out / (1 - P_out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def clip_topk_suppress_cp(logits, top_k=20, gamma=0.1, entr_temp=0.1,
                            alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                            alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → keep top-k species per clip, suppress rest by gamma → cSEBBs.
    Soundscape clips have 0-3 active species; 231+ species should be near-zero.
    Soft suppression (multiply by gamma) rather than hard zeroing preserves gradient signal.
    Source: BirdCLEF+ 2025 top solutions (within-clip species count prior).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    out_3d = _branchens_probs(logits, entr_temp, alpha_a, nw_a, beta_a,
                               alpha_b, nw_b, beta_b, w_a)   # (n_files, 12, C)
    result = out_3d.copy()
    for f in range(n_files):
        for t in range(N_WINDOWS):
            p = out_3d[f, t]                    # (C,)
            thresh_idx = np.argpartition(p, -top_k)[-top_k:]
            mask = np.ones(NUM_CLASSES) * gamma
            mask[thresh_idx] = 1.0
            result[f, t] = p * mask
    P_out = np.clip(result, eps, 1 - eps)
    blended = (np.log(P_out / (1 - P_out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def bidir_consistency_cp(logits, entr_temp=0.1,
                          alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                          alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → bidirectional time-reversed geometric mean → cSEBBs.
    Forward + reverse clip-order tracks; geom-mean reinforces temporally consistent predictions,
    suppresses spurious single-clip detections with no forward/backward support.
    Source: Voxaboxen (arXiv:2503.02389, Mar 2025) bidirectional consistency filtering.
    """
    eps = 1e-6
    out_3d = _branchens_probs(logits, entr_temp, alpha_a, nw_a, beta_a,
                               alpha_b, nw_b, beta_b, w_a)   # (n_files, 12, C)
    # Reverse clip order within each file
    rev_3d = out_3d[:, ::-1, :]               # (n_files, 12, C) reversed
    # Geometric mean of forward and reversed — consistent detections reinforced
    geom = np.sqrt(out_3d * rev_3d + eps)
    P_out = np.clip(geom, eps, 1 - eps)
    blended = (np.log(P_out / (1 - P_out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def file_species_prior_cp(logits, top_species=25, gamma=0.2, entr_temp=0.1,
                           alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                           alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → file-level species count prior → cSEBBs.
    Keep top-N species by file-level max score; soft-suppress the rest.
    Soundscape files have 5-20 active species; suppressing 214+ non-present species
    reduces false positives from background noise. Cross-species, file-level operation.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    out_3d = _branchens_probs(logits, entr_temp, alpha_a, nw_a, beta_a,
                               alpha_b, nw_b, beta_b, w_a)   # (n_files, 12, C)
    result = out_3d.copy()
    for f in range(n_files):
        file_max = out_3d[f].max(axis=0)       # (C,) max score per species across file
        top_idx = np.argpartition(file_max, -top_species)[-top_species:]
        mask = np.ones(NUM_CLASSES) * gamma
        mask[top_idx] = 1.0
        result[f] = out_3d[f] * mask[None, :]
    P_out = np.clip(result, eps, 1 - eps)
    blended = (np.log(P_out / (1 - P_out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def clip_rank_sharpen_topk_cp(logits, alpha=2.0, top_k=15, gamma=0.1, entr_temp=0.1,
                                alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → within-clip sharpen (p^alpha) + top-k suppress → cSEBBs.
    Combined: power sharpening concentrates mass, top-k suppression zeros out tail.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    out_3d = _branchens_probs(logits, entr_temp, alpha_a, nw_a, beta_a,
                               alpha_b, nw_b, beta_b, w_a)
    P_sharp = np.power(np.clip(out_3d, eps, 1 - eps), alpha)
    max_orig = out_3d.max(axis=2, keepdims=True)
    max_sharp = P_sharp.max(axis=2, keepdims=True)
    P_sharp = P_sharp * (max_orig / (max_sharp + eps))
    result = P_sharp.copy()
    for f in range(n_files):
        for t in range(N_WINDOWS):
            p = P_sharp[f, t]
            thresh_idx = np.argpartition(p, -top_k)[-top_k:]
            mask = np.ones(NUM_CLASSES) * gamma
            mask[thresh_idx] = 1.0
            result[f, t] = p * mask
    P_out = np.clip(result, eps, 1 - eps)
    blended = (np.log(P_out / (1 - P_out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R34: Cross-class co-occurrence graph / noise-floor / energy-OOD / onset-offset ───────────
# Paper sources:
#   CooccurrenceGraph: Chen et al., Complex & Intelligent Systems 2024; Graph SED cross-class.
#     Key insight: ALL prior 600 methods operate per-class independently. This is the ONLY
#     category that crosses the 234-class boundary using within-file co-occurrence structure.
#   NoiseFloorNorm: Zerroug et al., arXiv:2505.11889 (nSEBBs). Normalize by per-class lower-tail
#     (bottom-k clips), making detection threshold relative to per-file noise floor.
#   EnergyOOD: Liu et al., free-energy OOD detection. Per-clip energy = -log(sum(exp(logits)))
#     is the TOTAL activity scalar — distinct from SoftEntropyWeight (which uses per-class entropy).
#     High energy = clip contains dense multi-species activity, potentially noisy.
#   OnsetOffset: Dinkel & Wang, arXiv:2601.04178 (RED boundary-aware, ICASSP 2026).
#     Asymmetric: strong forward fill after onset frames, decay after offset frames.
#     None of our 600 methods use asymmetric temporal smoothing based on onset/offset detection.

def _branchens_probs(logits, entr_temp=0.1, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                     alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55):
    """Compute arithmetic BranchEns in prob space — shared helper for R34 methods."""
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha) * lse_3d + alpha * anchor[:, None, :]

    out_a = _branch(nw_a, alpha_a, beta_a)
    out_b = _branch(nw_b, alpha_b, beta_b)
    return w_a * out_a + (1 - w_a) * out_b   # (n_files, N_WINDOWS, NUM_CLASSES)


def cooccurrence_graph_cp(logits, alpha=0.2, entr_temp=0.1,
                           alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                           alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → per-file cross-class co-occurrence graph smoothing → cSEBBs.
    First method in 600+ that crosses the 234-class boundary.
    For each file: C[i,j] = cosine similarity between temporal activation of class i and j.
    One-step graph Laplacian: P_smooth = P + alpha * P @ C_rowNorm (propagate evidence).
    Source: Chen et al., Complex & Intelligent Systems 2024 (GCN for SED).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    out_3d = _branchens_probs(logits, entr_temp, alpha_a, nw_a, beta_a,
                               alpha_b, nw_b, beta_b, w_a)   # (n_files, 12, C)
    result = np.zeros_like(out_3d)
    for f in range(n_files):
        P = out_3d[f]                              # (12, C)
        P_t = P.T                                  # (C, 12)
        norms = np.linalg.norm(P_t, axis=1, keepdims=True) + eps
        P_n = P_t / norms                          # (C, 12) L2-normalized
        C_mat = P_n @ P_n.T                        # (C, C) cosine sim in [0,1]
        np.fill_diagonal(C_mat, 0.0)
        row_sum = C_mat.sum(axis=1, keepdims=True) + eps
        C_norm = C_mat / row_sum                   # row-normalized
        P_smooth = P + alpha * (P @ C_norm.T)      # (12, C) add propagated evidence
        result[f] = np.clip(P_smooth, eps, 1 - eps)
    out = np.clip(result, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def noise_floor_norm_cp(logits, k_frac=0.25, entr_temp=0.1,
                         alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                         alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → per-class per-file noise-floor normalization → cSEBBs.
    Noise floor = mean of bottom-k clips per class per file.
    P_norm = (P - noise) / (max - noise). Adaptive threshold relative to in-file background.
    Source: Zerroug et al., arXiv:2505.11889 (nSEBBs adaptive post-processing).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    out_3d = _branchens_probs(logits, entr_temp, alpha_a, nw_a, beta_a,
                               alpha_b, nw_b, beta_b, w_a)   # (n_files, 12, C)
    k = max(1, int(k_frac * N_WINDOWS))
    result = np.zeros_like(out_3d)
    for f in range(n_files):
        P = out_3d[f]                              # (12, C)
        sorted_P = np.sort(P, axis=0)              # (12, C) ascending
        noise = sorted_P[:k].mean(axis=0)          # (C,) bottom-k mean = noise floor
        peak = P.max(axis=0)                       # (C,) per-class peak
        span = peak - noise + eps
        P_norm = (P - noise) / span                # (12, C) in [0,1]
        P_norm = np.clip(P_norm, 0, 1)
        # Re-weight original by normalized gate
        P_out = P * P_norm
        result[f] = np.clip(P_out, eps, 1 - eps)
    out = np.clip(result, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def energy_ood_gate_cp(logits, gate_strength=0.5, thresh_pct=80, entr_temp=0.1,
                        alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                        alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → energy-based OOD per-clip gate → cSEBBs.
    Free energy per clip = -log(sum(exp(logits))): high = dense multi-class activity = OOD.
    OOD clips blended toward file mean. Distinct from SoftEntropyWeight (per-class entropy).
    Source: Liu et al., NeurIPS 2021 energy OOD; Xue et al., APSIPA 2025 arXiv:2507.09606.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    out_3d = _branchens_probs(logits, entr_temp, alpha_a, nw_a, beta_a,
                               alpha_b, nw_b, beta_b, w_a)   # (n_files, 12, C)
    result = np.zeros_like(out_3d)
    for f in range(n_files):
        P = out_3d[f]                              # (12, C)
        L = np.log(np.clip(P, eps, 1) / np.clip(1 - P, eps, 1))  # (12, C) logit
        energy = -np.log(np.sum(np.exp(L), axis=1) + eps)         # (12,) free energy
        # Normalize: clean clips have MORE NEGATIVE energy (focused predictions)
        e_min, e_max = energy.min(), energy.max()
        gate = 1.0 - (energy - e_min) / (e_max - e_min + eps)  # (12,) in [0,1]
        # gate near 1 = clean, gate near 0 = OOD/noisy
        file_mean = P.mean(axis=0)                 # (C,)
        gate_col = gate[:, None]
        P_out = gate_col * P + (1 - gate_col) * gate_strength * file_mean
        result[f] = np.clip(P_out, eps, 1 - eps)
    out = np.clip(result, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def onset_offset_asymmetric_cp(logits, onset_fw=1, offset_bw=2, blend=0.3, entr_temp=0.1,
                                 alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                 alpha_b=0.40, nw_b=0.30, beta_b=6.0, w_a=0.55, cp_thr=0.06):
    """BranchEns → onset/offset asymmetric temporal smoothing → cSEBBs.
    Onset frame (prob rising): forward-fill neighbors. Offset frame (prob falling): no fill.
    First asymmetric temporal method in this search; all prior methods are time-symmetric.
    Source: Dinkel & Wang, arXiv:2601.04178 ICASSP 2026 (boundary-aware SED).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    out_3d = _branchens_probs(logits, entr_temp, alpha_a, nw_a, beta_a,
                               alpha_b, nw_b, beta_b, w_a)   # (n_files, 12, C)
    result = np.zeros_like(out_3d)
    for f in range(n_files):
        P = out_3d[f]                              # (12, C)
        diff = np.diff(P, axis=0, prepend=P[:1])   # (12, C)
        onset_mask = (diff > 0).astype(float)      # (12, C) rising frames
        P_out = P.copy()
        # Forward fill after onset: weighted contribution from onset frame to future clips
        for t_fwd in range(1, onset_fw + 1):
            w_fwd = (onset_fw - t_fwd + 1) / (onset_fw + 1)
            shifted = np.roll(P * onset_mask, t_fwd, axis=0)
            shifted[:t_fwd] = 0
            P_out += blend * w_fwd * shifted
        P_out = np.clip(P_out, eps, 1 - eps)
        result[f] = P_out
    out = np.clip(result, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R33: Geometric BranchEns / Global-Silence Cut-off ────────────────────────────────────────
# Paper sources:
#   GeometricBranchEns: BirdCLEF 2024 4th place — geometric mean ensemble blending
#     (Geometric mean of A and B branches in prob space: A^w * B^(1-w) instead of w*A+(1-w)*B)
#   GlobalSilenceCutoff: BirdCLEF 2024 4th place — "halved probability if no birdsong ≤0.10"
#     per-file per-class deflation where file-level max < threshold (novel vs our existing anchors)

def _geometric_blend(out_a, out_b, w_a=0.55, eps=1e-6):
    """Geometric mean blend: A^w * B^(1-w) in prob space, return logits."""
    out_a = np.clip(out_a, eps, 1 - eps)
    out_b = np.clip(out_b, eps, 1 - eps)
    out = out_a ** w_a * out_b ** (1 - w_a)
    out = np.clip(out, eps, 1 - eps)
    return out


def geometric_branch_ensemble_cp(logits, w_a=0.55, entr_temp=0.1,
                                   alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                   alpha_b=0.40, nw_b=0.30, beta_b=6.0, cp_thr=0.06):
    """Geometric mean blend of two DualAnchor branches: A^w * B^(1-w) in prob space.
    BirdCLEF 2024 4th place found geometric mean suppresses single-model overconfidence.
    Compare: arithmetic BranchEns = w*A + (1-w)*B (current best).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha) * lse_3d + alpha * anchor[:, None, :]

    out_a = _branch(nw_a, alpha_a, beta_a)
    out_b = _branch(nw_b, alpha_b, beta_b)
    out = _geometric_blend(out_a, out_b, w_a=w_a, eps=eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def arith_geo_blend_cp(logits, geo_w=0.3, w_a=0.55, entr_temp=0.1,
                        alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                        alpha_b=0.40, nw_b=0.30, beta_b=6.0, cp_thr=0.06):
    """Blend of arithmetic BranchEns and Geometric BranchEns: (1-gw)*arith + gw*geo.
    Tests whether a small geometric correction on top of arithmetic helps."""
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha) * lse_3d + alpha * anchor[:, None, :]

    out_a = _branch(nw_a, alpha_a, beta_a)
    out_b = _branch(nw_b, alpha_b, beta_b)
    arith = w_a * out_a + (1 - w_a) * out_b
    geo = _geometric_blend(out_a, out_b, w_a=w_a, eps=eps)
    out = (1 - geo_w) * arith + geo_w * geo
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def silence_cut_branchens_cp(logits, thr=0.10, cut_factor=0.5, w_a=0.55, entr_temp=0.1,
                               alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                               alpha_b=0.40, nw_b=0.30, beta_b=6.0, cp_thr=0.06):
    """BranchEns → Global-Silence Cut-off → cSEBBs.
    BirdCLEF 2024 4th place: halve predictions where per-file max < 0.10.
    Targets false positives from noise: species with uniformly-low scores across all 12 clips.
    Different from our DualAnchor (which boosts present species); this deflates absent ones.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha) * lse_3d + alpha * anchor[:, None, :]

    out_a = _branch(nw_a, alpha_a, beta_a)
    out_b = _branch(nw_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b  # arithmetic blend (same as BranchEns)
    # Global-silence cut-off: per-file, per-class deflation
    file_max = out.max(axis=1)             # (n_files, C)
    silent_mask = file_max < thr           # (n_files, C)
    for f in range(n_files):
        out[f, :, silent_mask[f]] *= cut_factor
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def geo_silence_cut_cp(logits, thr=0.10, cut_factor=0.5, geo_w=0.5, w_a=0.55,
                        entr_temp=0.1, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                        alpha_b=0.40, nw_b=0.30, beta_b=6.0, cp_thr=0.06):
    """GeometricBranchEns → Global-Silence Cut-off → cSEBBs."""
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha) * lse_3d + alpha * anchor[:, None, :]

    out_a = _branch(nw_a, alpha_a, beta_a)
    out_b = _branch(nw_b, alpha_b, beta_b)
    geo = _geometric_blend(out_a, out_b, w_a=w_a, eps=eps)
    arith = w_a * out_a + (1 - w_a) * out_b
    out = (1 - geo_w) * arith + geo_w * geo
    file_max = out.max(axis=1)
    silent_mask = file_max < thr
    for f in range(n_files):
        out[f, :, silent_mask[f]] *= cut_factor
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R32: Per-class adaptive median / morphological closing / event-continuity decay ─────────
# Paper sources:
#   PerClassAdaptiveMedian: DCASE 2024 Task 4 baseline (arXiv:2406.08056) — per-class widths
#     from mean run-length in label stats (not fixed size=3 for all classes)
#   MorphClose: dilation then erosion fills internal gaps without extending endpoints
#   EventContinuityDecay: exponential bridge between detected peaks (new, not in literature)

def _per_class_run_widths(Y_labels, max_w=9):
    """Compute per-class adaptive median widths from mean label run-length in OOF ground truth.
    Each class c gets width w_c = max(1, min(max_w, round(1.5 * mean_run_c))), forced odd.
    Short-call species → w=1 (no smoothing); long-call species → wider gap-fill.
    """
    n_rows, C = Y_labels.shape
    n_files = n_rows // N_WINDOWS
    widths = np.ones(C, dtype=np.int32)
    for c in range(C):
        runs = []
        for f in range(n_files):
            y = Y_labels[f * N_WINDOWS:(f + 1) * N_WINDOWS, c]
            in_run = False; run_len = 0
            for t in range(N_WINDOWS):
                if y[t] > 0.5:
                    in_run = True; run_len += 1
                elif in_run:
                    runs.append(run_len); in_run = False; run_len = 0
            if in_run:
                runs.append(run_len)
        if runs:
            mean_run = float(np.mean(runs))
            w = max(1, min(max_w, int(round(1.5 * mean_run))))
            widths[c] = w if w % 2 == 1 else w + 1
    return widths


def per_class_adaptive_median_cp(logits, scale=1.0, entr_temp=0.1, alpha=0.38,
                                  nor_w=0.40, beta=5.15, cp_thr=0.06):
    """DCASE 2024 Task 4: per-class variable-width median filter then DualAnchor+cSEBBs.
    w_c computed once from label statistics — zero inference-time parameters.
    """
    from scipy.signal import medfilt
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    widths = _per_class_run_widths(_OOF_LABELS)
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # per-class adaptive median filter
    med_3d = lse_3d.copy()
    for c in range(NUM_CLASSES):
        kw = max(1, min(9, int(widths[c] * scale)))
        kw = kw if kw % 2 == 1 else kw + 1
        if kw > 1:
            for f in range(n_files):
                med_3d[f, :, c] = medfilt(lse_3d[f, :, c], kernel_size=kw)
    nor_anchor = 1.0 - np.prod(1.0 - med_3d, axis=1)
    max_anchor = med_3d.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * med_3d + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def per_class_adaptive_median_branchens_cp(logits, scale=1.0, entr_temp=0.1,
                                            alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                            alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                            w_a=0.55, cp_thr=0.06):
    """Per-class adaptive median applied inside each BranchEns branch."""
    from scipy.signal import medfilt
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    widths = _per_class_run_widths(_OOF_LABELS)
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _med_branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        med_3d = lse_3d.copy()
        for c in range(NUM_CLASSES):
            kw = max(1, min(9, int(widths[c] * scale)))
            kw = kw if kw % 2 == 1 else kw + 1
            if kw > 1:
                for f in range(n_files):
                    med_3d[f, :, c] = medfilt(lse_3d[f, :, c], kernel_size=kw)
        nor_anchor = 1.0 - np.prod(1.0 - med_3d, axis=1)
        max_anchor = med_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha) * med_3d + alpha * anchor[:, None, :]

    out_a = _med_branch(nw_a, alpha_a, beta_a)
    out_b = _med_branch(nw_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def morphological_close_dual_anchor_cp(logits, close_r=1, entr_temp=0.1, alpha=0.38,
                                        nor_w=0.40, beta=5.15, cp_thr=0.06):
    """Morphological closing (dilation then erosion) fills internal gaps without extending ends.
    Applied per-class per-file on LSE probs before DualAnchor+cSEBBs.
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # morphological closing per class per file (dilation → erosion)
    kw = 2 * close_r + 1
    closed = maximum_filter1d(lse_3d, size=kw, axis=1)    # dilation
    closed = minimum_filter1d(closed, size=kw, axis=1)    # erosion
    nor_anchor = 1.0 - np.prod(1.0 - closed, axis=1)
    max_anchor = closed.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * closed + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def morphological_close_branchens_cp(logits, close_r=1, entr_temp=0.1,
                                      alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                      alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                      w_a=0.55, cp_thr=0.06):
    """Morphological closing inside each BranchEns branch."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    kw = 2 * close_r + 1

    def _close_branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        closed = maximum_filter1d(lse_3d, size=kw, axis=1)
        closed = minimum_filter1d(closed, size=kw, axis=1)
        nor_anchor = 1.0 - np.prod(1.0 - closed, axis=1)
        max_anchor = closed.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha) * closed + alpha * anchor[:, None, :]

    out_a = _close_branch(nw_a, alpha_a, beta_a)
    out_b = _close_branch(nw_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── R31: Rank-norm within file / entropy-prior gate / cross-file calibration ────────────────

def _rank_norm_within_file(lse_3d):
    """Vectorized per-file per-class rank normalization of T clips to [0, 1]."""
    T = lse_3d.shape[1]
    ranks = np.argsort(np.argsort(lse_3d, axis=1), axis=1).astype(float)
    if T > 1:
        ranks /= (T - 1)
    return ranks


def rank_blend_dual_anchor_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                               rank_blend=0.3, entr_temp=0.1, cp_thr=0.06):
    """Quantile-Mix style: rank-normalize LSE scores within each file, blend with raw, then DualAnchor.
    Equalizes marginal distribution per class across clips — reduces dominant-species bias.
    Source: BirdCLEF 2025 top-2% (+0.038 AUC reported in competition discussion).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # rank-normalize within file per class, blend with raw
    rank_lse = _rank_norm_within_file(lse_3d)
    mixed_3d = (1.0 - rank_blend) * lse_3d + rank_blend * rank_lse
    nor_anchor = 1.0 - np.prod(1.0 - mixed_3d, axis=1)
    max_anchor = mixed_3d.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * mixed_3d + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def rank_blend_branch_ensemble_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                   alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                   rank_blend=0.3, w_a=0.55, entr_temp=0.1, cp_thr=0.06):
    """RankBlend inside each DualAnchor branch, then BranchEns + cSEBBs."""
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _rb_branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        rank_lse = _rank_norm_within_file(lse_3d)
        mixed_3d = (1.0 - rank_blend) * lse_3d + rank_blend * rank_lse
        nor_anchor = 1.0 - np.prod(1.0 - mixed_3d, axis=1)
        max_anchor = mixed_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return (1 - alpha) * mixed_3d + alpha * anchor[:, None, :]

    out_a = _rb_branch(nw_a, alpha_a, beta_a)
    out_b = _rb_branch(nw_b, alpha_b, beta_b)
    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def entropy_prior_gate_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                           gate_w=0.3, entr_temp=0.1, cp_thr=0.06):
    """FINCH entropy-adaptive prior gate: confident clips get stronger file-prior injection.
    omega_t = 1 - H(probs_t)/log(C)  (high for confident clips, 0 for uniform)
    adjusted_prob += gate_w * omega_t * file_mean_c
    Source: FINCH (ICASSP 2025 BirdCLEF challenge discussion), entropy-weighted class prior.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # entropy per clip: (n_files, T)
    p = np.clip(lse_3d, eps, 1 - eps)
    H_t = -np.sum(p * np.log(p) + (1 - p) * np.log(1 - p), axis=2) / NUM_CLASSES
    log_C = np.log(float(NUM_CLASSES))
    omega = np.clip(1.0 - H_t / log_C, 0.0, 1.0)  # (n_files, T)
    # file prior per class: mean over clips
    file_prior = lse_3d.mean(axis=1)  # (n_files, C)
    # inject: positive gate_w adds high-prior classes for confident clips
    adjusted = lse_3d + gate_w * omega[:, :, None] * file_prior[:, None, :]
    adjusted = np.clip(adjusted, eps, 1 - eps)
    nor_anchor = 1.0 - np.prod(1.0 - adjusted, axis=1)
    max_anchor = adjusted.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * adjusted + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def cross_file_calibration_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                               cal_w=0.1, entr_temp=0.1, cp_thr=0.06):
    """Cross-file class calibration: normalize each class by global activity level.
    Global class activity = mean over all files of per-file max score.
    Per-file calibration boosts rare-globally-but-locally-active classes.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # global class activity across all files: (C,)
    global_max_c = lse_3d.max(axis=1).mean(axis=0)
    local_max_c = lse_3d.max(axis=1)  # (n_files, C)
    # boost clips where local_max < global (species present here but globally rare)
    diff = global_max_c[None, :] - local_max_c  # (n_files, C)
    lse_3d = np.clip(lse_3d + cal_w * diff[:, None, :], eps, 1 - eps)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_anchor = lse_3d.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def probor_on_branch_ensemble_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                  alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                  w_a=0.55, probor_w=0.10, entr_temp=0.1, cp_thr=0.06):
    """ProbOR additive boost applied after BranchEns, then cSEBBs.
    ProbOR = NOR_anchor + MeanLSE - NOR*Mean (Bayes OR in prob space).
    Source: BirdCLEF 2024 3rd place additive boost (+0.02 LB reported).
    Here applied as post-ensemble refinement on top of best BranchEns.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_a = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_a = lse_3d.max(axis=1)
        anchor = nw * nor_a + (1 - nw) * max_a
        return (1 - alpha) * lse_3d + alpha * anchor[:, None, :]

    out_a = _branch(nw_a, alpha_a, beta_a)
    out_b = _branch(nw_b, alpha_b, beta_b)
    ens = w_a * out_a + (1 - w_a) * out_b  # (n_files, T, C)
    # ProbOR boost at clip level
    lse_3d_a = sigmoid(logsumexp_pool(wl, radius=1, beta=beta_a)).reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_boost = 1.0 - np.prod(1.0 - lse_3d_a, axis=1, keepdims=True)  # (n_files, 1, C)
    mean_boost = lse_3d_a.mean(axis=1, keepdims=True)                  # (n_files, 1, C)
    prob_or = nor_boost + mean_boost - nor_boost * mean_boost
    out = ens + probor_w * (prob_or - ens)
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def bayesian_dual_anchor_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                             entr_temp=0.1, cp_thr=0.06):
    """DualAnchor with Bayesian (multiplicative odds) update instead of linear blend.
    Standard:  out = (1-α)*lse + α*anchor  — linear interpolation in prob space.
    Bayesian:  out = lse*anchor / (lse*anchor + (1-lse)*(1-anchor))
               Equivalent to: log-odds(out) = log-odds(lse) + log-odds(anchor)
               Treating lse as likelihood and anchor as prior; result is posterior.
    This is fundamentally distinct from LogitBlend (which weighted logits with fixed scale).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_anchor = lse_3d.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor  # (n_files, C)
    anchor = np.clip(anchor[:, None, :], eps, 1 - eps)
    p = np.clip(lse_3d, eps, 1 - eps)
    # Bayesian posterior: scale anchor effect by alpha (alpha=0 → lse, alpha=1 → full Bayes)
    # Interpolate log-odds: log-odds(out) = (1-alpha)*log-odds(lse) + alpha*(log-odds(lse)+log-odds(anchor))
    # = log-odds(lse) + alpha*log-odds(anchor)
    lo_lse    = np.log(p / (1 - p))
    lo_anchor = np.log(anchor / (1 - anchor))
    lo_out    = lo_lse + alpha * lo_anchor
    out = 1.0 / (1.0 + np.exp(-lo_out))
    out = np.clip(out, eps, 1 - eps)
    out_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(out_logits, threshold=cp_thr, blend=0.4)


def bayesian_branch_ensemble_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                  alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                  w_a=0.5, entr_temp=0.1, cp_thr=0.06):
    """BranchEns where each branch uses Bayesian multiplicative anchor update."""
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _bayes_branch(nw, alpha, beta):
        lse = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        g_nor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        g_max = lse_3d.max(axis=1)
        anchor = np.clip((nw * g_nor + (1 - nw) * g_max)[:, None, :], eps, 1 - eps)
        p = np.clip(lse_3d, eps, 1 - eps)
        lo_out = np.log(p / (1 - p)) + alpha * np.log(anchor / (1 - anchor))
        return np.clip(1.0 / (1.0 + np.exp(-lo_out)), eps, 1 - eps)

    out = w_a * _bayes_branch(nw_a, alpha_a, beta_a) + (1 - w_a) * _bayes_branch(nw_b, alpha_b, beta_b)
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def quantile_anchor_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                        q=0.90, entr_temp=0.1, cp_thr=0.06):
    """DualAnchor with quantile anchor instead of GlobalMax.
    q=0.90: 90th percentile of clip scores (less extreme than pure max).
    q=0.75: 75th percentile (more robust to a single noisy peak).
    Rationale: with 12 clips, max is too sensitive to one outlier clip.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    # Quantile anchor: q-th percentile across T clips for each (file, class)
    q_anchor = np.quantile(lse_3d, q, axis=1)              # (n_files, C)
    anchor = nor_w * nor_anchor + (1 - nor_w) * q_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    out_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(out_logits, threshold=cp_thr, blend=0.4)


def score_weighted_anchor_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                               entr_temp=0.1, cp_thr=0.06):
    """DualAnchor with score-weighted mean anchor: sum(p^2)/sum(p) per class.
    This is an attention-weighted file-level statistic: clips with higher predictions
    contribute proportionally more to the anchor. Distinct from:
    - GlobalMean: uniform weights
    - GlobalMax: only the single highest clip
    - NoisyOR: probabilistic union (non-linear, grows with more detections)
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    # Score-weighted mean: sum(p^2) / sum(p) — concentrates weight on high-score clips
    sw_anchor = (lse_3d ** 2).sum(axis=1) / (lse_3d.sum(axis=1) + eps)  # (n_files, C)
    sw_anchor = np.clip(sw_anchor, eps, 1 - eps)
    anchor = nor_w * nor_anchor + (1 - nor_w) * sw_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    out_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(out_logits, threshold=cp_thr, blend=0.4)


def mixed_anchor_branch_ensemble_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                      alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                      w_a=0.5, q=0.90, entr_temp=0.1, cp_thr=0.06):
    """BranchEns: Branch A = standard NOR+GlobalMax, Branch B = NOR+Q90 quantile anchor."""
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _std_branch(nw, alpha, beta):
        lse = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        g_nor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        g_max = lse_3d.max(axis=1)
        anc = nw * g_nor + (1 - nw) * g_max
        return np.clip((1 - alpha) * lse_3d + alpha * anc[:, None, :], eps, 1 - eps)

    def _quant_branch(nw, alpha, beta, qq):
        lse = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        g_nor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        q_anc = np.quantile(lse_3d, qq, axis=1)
        anc = nw * g_nor + (1 - nw) * q_anc
        return np.clip((1 - alpha) * lse_3d + alpha * anc[:, None, :], eps, 1 - eps)

    out = w_a * _std_branch(nw_a, alpha_a, beta_a) + (1 - w_a) * _quant_branch(nw_b, alpha_b, beta_b, q)
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def power_mean_dual_anchor_cp(logits, alpha=0.38, nor_w=0.40, power=3.0,
                               entr_temp=0.1, cp_thr=0.06):
    """DualAnchor using power mean pooling instead of LSE (radius=1).
    Power mean p=3: generalized mean between arithmetic mean and max.
    Distinct from LSE because it works in probability space, not logit space.
    Reference: arXiv:2010.09985 (Power Pooling for SED, ICASSP 2021).
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # Apply entropy weighting to logits then convert to probs
    wl = (raw * w[:, :, None])  # (n_files, N_WINDOWS, C)
    probs = sigmoid(wl.reshape(-1, NUM_CLASSES)).reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # Power mean pooling in prob space (radius=1)
    pm = np.zeros_like(probs)
    for t in range(N_WINDOWS):
        t_s = max(0, t - 1)
        t_e = min(N_WINDOWS, t + 2)
        window = probs[:, t_s:t_e, :]
        pm[:, t, :] = (np.mean(window ** power, axis=1)) ** (1.0 / power)
    pm = np.clip(pm, eps, 1 - eps)
    nor_anchor = 1.0 - np.prod(1.0 - pm, axis=1)
    max_anchor = pm.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * pm + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    out_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(out_logits, threshold=cp_thr, blend=0.4)


def power_lse_branch_ensemble_cp(logits, alpha_a=0.38, nw_a=0.40, power_a=3.0,
                                   alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                   w_a=0.5, entr_temp=0.1, cp_thr=0.06):
    """Branch ensemble: Branch A uses power mean pooling, Branch B uses LSE.
    Asymmetric diversity: two fundamentally different pooling operations.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl_logit = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    # Branch A: power mean in prob space
    wl_probs = sigmoid(wl_logit).reshape(n_files, N_WINDOWS, NUM_CLASSES)
    pm = np.zeros_like(wl_probs)
    for t in range(N_WINDOWS):
        t_s = max(0, t - 1)
        t_e = min(N_WINDOWS, t + 2)
        window = wl_probs[:, t_s:t_e, :]
        pm[:, t, :] = (np.mean(window ** power_a, axis=1)) ** (1.0 / power_a)
    pm = np.clip(pm, eps, 1 - eps)
    nor_a = 1.0 - np.prod(1.0 - pm, axis=1)
    max_a = pm.max(axis=1)
    anchor_a = nw_a * nor_a + (1 - nw_a) * max_a
    out_a = np.clip((1 - alpha_a) * pm + alpha_a * anchor_a[:, None, :], eps, 1 - eps)

    # Branch B: LSE in logit space (standard)
    lse_probs = sigmoid(logsumexp_pool(wl_logit, radius=1, beta=beta_b))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_b = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_b = lse_3d.max(axis=1)
    anchor_b = nw_b * nor_b + (1 - nw_b) * max_b
    out_b = np.clip((1 - alpha_b) * lse_3d + alpha_b * anchor_b[:, None, :], eps, 1 - eps)

    out = w_a * out_a + (1 - w_a) * out_b
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def dual_anchor_geom_anchor_cp(logits, alpha=0.38, beta=5.15, entr_temp=0.1, cp_thr=0.06):
    """DualAnchor with geometric mean anchor: anchor = sqrt(NOR * GlobalMax).
    Geometric mean interpolates differently than linear blend (nw*NOR + (1-nw)*max).
    Avoids the need to tune nor_w: the GM is a natural symmetric combination.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_anchor = lse_3d.max(axis=1)
    # Geometric mean anchor — symmetric, no tunable nor_w
    anchor = np.sqrt(np.clip(nor_anchor * max_anchor, eps, 1.0))
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    out_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(out_logits, threshold=cp_thr, blend=0.4)


def dual_anchor_topk_anchor_cp(logits, alpha=0.38, nor_w=0.40, beta=5.15,
                                 k=3, entr_temp=0.1, cp_thr=0.06):
    """DualAnchor with TopK anchor: average of top-k clips instead of GlobalMax.
    TopK is more robust than pure max for rare species with noisy peaks.
    k=2: avg of top-2; k=3: avg of top-3 clips per class.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    # TopK anchor: sort descending and take mean of top k clips
    sorted_probs = np.sort(lse_3d, axis=1)[:, ::-1, :]   # (n_files, T desc, C)
    topk_anchor = sorted_probs[:, :k, :].mean(axis=1)     # (n_files, C)
    anchor = nor_w * nor_anchor + (1 - nor_w) * topk_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, eps, 1 - eps)
    out_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(out_logits, threshold=cp_thr, blend=0.4)


def branch_ensemble_post_lse_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                  alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                  w_a=0.5, post_beta=3.0, entr_temp=0.1, cp_thr=0.06):
    """BranchEns then second gentle LSE pass (gap-fill residual events post-ensemble).
    Two-stage: [BranchEns → LSE(gentle)] → cSEBBs.
    Low post_beta (3.0) = soft temporal widening, not aggressive extension.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return np.clip((1 - alpha) * lse_3d + alpha * anchor[:, None, :], eps, 1 - eps)

    out = w_a * _branch(nw_a, alpha_a, beta_a) + (1 - w_a) * _branch(nw_b, alpha_b, beta_b)
    out = np.clip(out, eps, 1 - eps)
    # Convert ensemble output to logits then apply second LSE pass
    out_logits = np.log(out / (1 - out)).reshape(-1, NUM_CLASSES)
    post_lse = logsumexp_pool(out_logits, radius=1, beta=post_beta)
    return change_point_segment_mean(post_lse, threshold=cp_thr, blend=0.4)


def branch_ensemble_post_gm_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                 alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                 w_a=0.5, gm_alpha=0.15, entr_temp=0.1, cp_thr=0.06):
    """BranchEns → cSEBBs → GlobalMean blend (gentle file-level correction).
    Applies small alpha GlobalMean after segmentation to reduce false negatives.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return np.clip((1 - alpha) * lse_3d + alpha * anchor[:, None, :], eps, 1 - eps)

    out = w_a * _branch(nw_a, alpha_a, beta_a) + (1 - w_a) * _branch(nw_b, alpha_b, beta_b)
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    # cSEBBs first
    after_cp = sigmoid(change_point_segment_mean(blended, threshold=cp_thr, blend=0.4))
    # Then GlobalMean blend on the segmented output
    after_cp_3d = after_cp.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_mean = after_cp_3d.mean(axis=1, keepdims=True)
    final = (1 - gm_alpha) * after_cp_3d + gm_alpha * file_mean
    final = np.clip(final, eps, 1 - eps)
    return (np.log(final / (1 - final)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def branch_ensemble_double_cp(logits, alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                alpha_b=0.40, nw_b=0.30, beta_b=6.0,
                                w_a=0.5, entr_temp=0.1, cp_thr1=0.06, cp_thr2=0.03):
    """BranchEns with two-stage cSEBBs: first clean coarse changes, then fine ones.
    Stage 1: thr=0.06 removes major boundaries.
    Stage 2: thr=0.03 catches residual micro-changes in the smoothed output.
    """
    eps = 1e-6
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)

    def _branch(nw, alpha, beta):
        lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
        lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
        max_anchor = lse_3d.max(axis=1)
        anchor = nw * nor_anchor + (1 - nw) * max_anchor
        return np.clip((1 - alpha) * lse_3d + alpha * anchor[:, None, :], eps, 1 - eps)

    out = w_a * _branch(nw_a, alpha_a, beta_a) + (1 - w_a) * _branch(nw_b, alpha_b, beta_b)
    out = np.clip(out, eps, 1 - eps)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    stage1 = change_point_segment_mean(blended, threshold=cp_thr1, blend=0.4)
    return change_point_segment_mean(stage1, threshold=cp_thr2, blend=0.3)


def lae_anchor_cp(logits, lae_beta=2.0, alpha=0.35, nor_w=0.40, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → LAE anchor (LogAvgExp: (1/β)*log(mean(exp(β*p)))) → cSEBBs.
    LAE interpolates mean↔max: β=0→mean, β→∞→max. Principled alternative to NoisyOR.
    Source: arXiv:2111.01742 LogAvgExp as global pooling operator.
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    p = np.clip(lse_3d, 1e-7, 1.0)
    anchor = (1.0 / lae_beta) * np.log(np.mean(np.exp(lae_beta * p), axis=1))  # (n_files, C)
    anchor = np.clip(anchor, 0.0, 1.0)
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def poe_anchor_cp(logits, alpha=0.35, nor_w=0.6, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → PoE anchor (sqrt(NoisyOR * file_max)) → cSEBBs.
    Product of Experts geometric fusion: more conservative than arithmetic DualAnchor.
    nor_w: exponent weight for NoisyOR (1-nor_w for max).
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_anchor = lse_3d.max(axis=1)
    eps = 1e-7
    poe = np.clip(nor_anchor, eps, 1.0)**nor_w * np.clip(max_anchor, eps, 1.0)**(1-nor_w)
    out = (1 - alpha) * lse_3d + alpha * poe[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def pcr_adaptive_dual_anchor_cp(logits, alpha_min=0.10, alpha_max=0.45, nor_w=0.40,
                                  entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → DualAnchor with PCR-adaptive per-class alpha → cSEBBs.
    PCR_c = log(mean_top10%(p_c) / p10_c) — high PCR → less anchor needed.
    Source: arXiv:2505.11889 nSEBBs Posterior Contrast Ratio.
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)  # (n_files, C)
    max_anchor = lse_3d.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    # PCR-adaptive alpha per file per class
    p10 = np.percentile(lse_3d, 10, axis=1)                         # (n_files, C)
    top_mask = lse_3d >= np.percentile(lse_3d, 90, axis=1, keepdims=True)
    top_mean = np.where(top_mask, lse_3d, 0.0).sum(axis=1) / (top_mask.sum(axis=1) + 1e-7)
    pcr = np.log(np.clip(top_mean / (p10 + 1e-7), 1.0, None))       # (n_files, C)
    pcr_norm = np.clip(pcr / 30.0, 0.0, 1.0)
    alpha_fc = alpha_max - pcr_norm * (alpha_max - alpha_min)        # (n_files, C)
    out = (1 - alpha_fc[:, None, :]) * lse_3d + alpha_fc[:, None, :] * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def gtla_dual_anchor_cp(logits, tau=0.5, alpha=0.35, nor_w=0.40, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """G-TLA logit adjustment → SoftEntr → LSE → DualAnchor → cSEBBs.
    GTLA: subtract tau*log(empirical_activation_rate) from logits before sigmoid.
    Source: Long-Tail Temporal Action Segmentation, ECCV 2024.
    """
    n_files = logits.shape[0] // N_WINDOWS
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # Empirical within-file activation rate per class (from sigmoid of raw logits)
    raw_probs = sigmoid(raw.reshape(-1, NUM_CLASSES))
    raw_probs_3d = raw_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    act_rate = np.clip(raw_probs_3d.mean(axis=1), 0.01, 1.0)  # (n_files, C)
    log_prior = np.log(act_rate)                                # (n_files, C)
    # Adjust logits: subtract tau*log(rate) — boosts rare species
    adjusted = raw - tau * log_prior[:, None, :]                # (n_files, T, C)
    adjusted_flat = adjusted.reshape(-1, NUM_CLASSES)
    w = _compute_clip_weights(adjusted_flat, method="entropy", temp=entr_temp)
    wl = (adjusted * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_anchor = lse_3d.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def topk_entr_anchor_cp(logits, k=5, alpha=0.35, nor_w=0.40, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → TopK-Entropy anchor (mean of k highest-entropy clips) → DualAnchor → cSEBBs.
    Uncertainty-gated pooling: most uncertain clips carry the global reference signal.
    Source: arXiv:2503.02422 bioacoustic active learning aggregation.
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    eps = 1e-7
    H = -(lse_3d * np.log(np.clip(lse_3d, eps, 1-eps)) +
          (1-lse_3d) * np.log(np.clip(1-lse_3d, eps, 1-eps)))  # (n_files, T, C)
    frame_H = H.mean(axis=2)                                    # (n_files, T)
    topk_idx = np.argsort(frame_H, axis=1)[:, -k:]             # (n_files, k)
    topk_anchor = np.stack([lse_3d[i, topk_idx[i], :].mean(axis=0)
                            for i in range(n_files)])           # (n_files, C)
    out = (1 - alpha) * lse_3d + alpha * topk_anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def dual_anchor_nor_max_r2_cp(logits, alpha=0.30, nor_w=0.5, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """DualAnchor with LSE radius=2."""
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=2, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_anchor = lse_3d.max(axis=1)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def triple_anchor_cp(logits, alpha=0.30, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → TripleAnchor(NOR + Max + AvgTopK-2) → cSEBBs.
    anchor = (1/3)*(NoisyOR + file_max + AvgTop2)
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    max_anchor = lse_3d.max(axis=1)
    sorted_p = np.sort(lse_3d, axis=1)[:, ::-1, :]
    topk_anchor = sorted_p[:, :2, :].mean(axis=1)
    anchor = (nor_anchor + max_anchor + topk_anchor) / 3.0
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def dual_anchor_nor_topk_cp(logits, alpha=0.30, k=2, entr_temp=0.1, beta=4.5, cp_thr=0.06):
    """SoftEntr → LSE → DualAnchor(NoisyOR + AvgTopK) → cSEBBs (no GlobalMax)."""
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=entr_temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    nor_anchor = 1.0 - np.prod(1.0 - lse_3d, axis=1)
    sorted_p = np.sort(lse_3d, axis=1)[:, ::-1, :]
    topk_anchor = sorted_p[:, :k, :].mean(axis=1)
    anchor = 0.5 * nor_anchor + 0.5 * topk_anchor
    out = (1 - alpha) * lse_3d + alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def rdp_mean_max_cp(logits, entr_temp=0.2, max_w=1.0, gm_alpha=0.175, cp_thr=0.06):
    """RDP weight (instead of entropy) + MeanMax(w=1.0) anchor → cSEBBs."""
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="rdp")
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=4.5))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_max  = lse_3d.max(axis=1)
    file_mean = lse_3d.mean(axis=1)
    anchor = max_w * file_max + (1 - max_w) * file_mean
    out = (1 - gm_alpha) * lse_3d + gm_alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


# ── Round 14: T fine-tune lower / nSEBBs / MPA hybrid / TAP-Velocity / MeanMax ─

def soft_entr_lse_gm_cp_t(logits, temp, beta=4.5, gm_alpha=0.175, cp_thr=0.06):
    """SoftEntrWt with given T → LSE → GM → cSEBBs."""
    return clip_weighted_lse_gm_cp(logits, method="entropy", temp=temp,
                                    beta=beta, gm_alpha=gm_alpha, cp_thr=cp_thr)


def nsebbs_pcr(probs_file, base_blend=0.4):
    """nSEBBs-PCR: PCR-adaptive threshold instead of fixed thr=0.06.
    PCR_c = log(mean_top5(p_c) / (p10_c + 1e-6)) — how 'peaky' each class is.
    High-PCR classes (confident) → higher threshold; low-PCR → lower threshold.
    Source: arXiv:2505.11889 (Revisiting SSL for SED, 2025).
    """
    import warnings
    n_files = probs_file.shape[0] // N_WINDOWS
    X = probs_file.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    out = X.copy()
    for fi in range(n_files):
        seq = X[fi]                                       # (T, C)
        p10 = np.percentile(seq, 10, axis=0)             # (C,)
        top5 = np.sort(seq, axis=0)[-5:, :].mean(axis=0) # (C,) — mean of top-5 clips
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcr = np.log((top5 + 1e-9) / (p10 + 1e-9))  # (C,) — contrast ratio
        pcr = np.clip(pcr, 0, 5)
        # Adaptive threshold: 0.03 to 0.12 range mapped from PCR 0..5
        thr_c = 0.03 + (pcr / 5.0) * 0.09               # (C,) in [0.03, 0.12]
        # Apply cSEBBs with per-class threshold
        diff = np.abs(np.diff(seq, axis=0))               # (T-1, C)
        seg = seq.copy()
        for t in range(N_WINDOWS - 1):
            is_boundary = diff[t] > thr_c
            seg_mean = seq[max(0, t-2):min(N_WINDOWS, t+3), :].mean(axis=0)
            seg[t, is_boundary] = ((1 - base_blend) * seq[t, is_boundary] +
                                    base_blend * seg_mean[is_boundary])
        out[fi] = seg
    return out.reshape(-1, NUM_CLASSES)


def soft_entr_lse_gm_nsebbs(logits, temp=0.2, beta=4.5, gm_alpha=0.175):
    """SoftEntrWt → LSE → GM → nSEBBs-PCR (replace cSEBBs with adaptive nSEBBs)."""
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="entropy", temp=temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(wl, radius=1, beta=beta)
    after_gm_probs = sigmoid(global_mean_blend(lse, alpha=gm_alpha))
    nsebbs_probs = nsebbs_pcr(after_gm_probs, base_blend=0.4)
    nsebbs_probs = np.clip(nsebbs_probs, 1e-6, 1 - 1e-6)
    return (np.log(nsebbs_probs / (1 - nsebbs_probs)) * TEMP_SCALE)


def mpa_hybrid_gm_cp(logits, rare_thr=0.05, beta=4.5, gm_alpha=0.175, cp_thr=0.06):
    """MPA hybrid: max-pool for rare classes, LSE for common classes.
    Source: arXiv:2406.12721 (DCASE 2024 2nd); max = best for annotation-mismatch.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_mean = X.mean(axis=1)                   # (n_files, C)
    is_rare = (file_mean < rare_thr)             # (n_files, C)

    # Standard SoftEntrWt→LSE path
    w = _compute_clip_weights(logits, method="entropy", temp=0.2)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)

    # MPA: max-pool for rare classes
    max_probs = X.max(axis=1)                    # (n_files, C) — hard max
    # Broadcast rare/common decision back to temporal sequence
    # Use file_max for rare, lse_temporal for common
    hybrid_file = np.where(is_rare, max_probs, lse_3d.mean(axis=1))  # (n_files, C)
    # Broadcast to temporal: blend local lse with hybrid
    out = 0.5 * lse_3d + 0.5 * hybrid_file[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    hybrid_logits = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    after_gm = global_mean_blend(hybrid_logits, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def mean_max_blend_gm_cp(logits, blend_w=0.5, beta=4.5, gm_alpha=0.175, cp_thr=0.06):
    """Mean-of-Mean-and-Max temporal pooling → GM → cSEBBs.
    Source: arXiv:2508.20703 (audio-text SED, Aug 2025).
    score_c(file) = 0.5*mean_t(p_c(t)) + 0.5*max_t(p_c(t)) — rare/common hedge.
    Applied as clip-level blend: local + file_max.
    """
    w = _compute_clip_weights(logits, method="entropy", temp=0.2)
    n_files = logits.shape[0] // N_WINDOWS
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse_probs = sigmoid(logsumexp_pool(wl, radius=1, beta=beta))
    lse_3d = lse_probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_max  = lse_3d.max(axis=1)              # (n_files, C)
    file_mean = lse_3d.mean(axis=1)             # (n_files, C)
    anchor = blend_w * file_max + (1 - blend_w) * file_mean  # (n_files, C)
    out = (1 - gm_alpha) * lse_3d + gm_alpha * anchor[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    blended = (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)
    return change_point_segment_mean(blended, threshold=cp_thr, blend=0.4)


def tap_velocity_lse_gm_cp(logits, vel_w=0.3, beta=4.5, gm_alpha=0.175, cp_thr=0.06):
    """TAP-Velocity: add velocity-attention weight to SoftEntrWt.
    Clips with rapid score changes (onset/offset) get upweighted.
    Source: arXiv:2504.12670 (TAP-Velocity, Apr 2025).
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    # Velocity: abs change from previous clip per class
    delta = np.abs(np.diff(X, axis=1))               # (n_files, T-1, C)
    vel = np.concatenate([delta[:, :1, :], delta], axis=1)  # (n_files, T, C) — pad first
    vel_score = vel.mean(axis=2)                      # (n_files, T) — mean velocity
    # Combined weight: entropy (T=0.2) + velocity
    entr_w = _compute_clip_weights(logits, method="entropy", temp=0.2)  # (n_files, T)
    vel_w_norm = vel_score / (vel_score.sum(axis=1, keepdims=True) + 1e-9) * N_WINDOWS
    combined = (1 - vel_w) * entr_w + vel_w * vel_w_norm
    combined = combined / combined.sum(axis=1, keepdims=True) * N_WINDOWS
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * combined[:, :, None]).reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(wl, radius=1, beta=beta)
    after_gm = global_mean_blend(lse, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def variance_lse_gm_nsebbs(logits, gm_alpha=0.175):
    """Variance weight → LSE → GM → nSEBBs-PCR."""
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method="variance")
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(wl, radius=1, beta=4.5)
    after_gm_probs = sigmoid(global_mean_blend(lse, alpha=gm_alpha))
    nsebbs_probs = nsebbs_pcr(after_gm_probs, base_blend=0.4)
    nsebbs_probs = np.clip(nsebbs_probs, 1e-6, 1 - 1e-6)
    return (np.log(nsebbs_probs / (1 - nsebbs_probs)) * TEMP_SCALE)


# ── Round 13: Combined Entr+RDP weight / T fine-tune / Clip scoring variants ───

def _compute_clip_weights(logits, method="entropy", temp=0.5, normalize=True):
    """Compute per-clip weights from probability matrix.
    method: 'entropy'  → exp(-H(t)/temp)
            'rdp'      → exp(rdp_score(t))   where rdp = mean |p-μ|/σ
            'combined' → exp(rdp(t) - H(t)/temp)
            'maxconf'  → max_c(p(t,c))        — clip max confidence
            'variance' → var_c(p(t,c))        — within-clip variance across classes
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)

    if method in ("entropy", "combined"):
        H = -(X * np.log(X + 1e-9) + (1 - X) * np.log(1 - X + 1e-9))
        H_clip = H.mean(axis=2)   # (n_files, T) — avg binary entropy per clip

    if method in ("rdp", "combined"):
        mu = X.mean(axis=1, keepdims=True)
        sg = X.std(axis=1, keepdims=True) + 1e-6
        rdp = (np.abs(X - mu) / sg).mean(axis=2)  # (n_files, T)

    if method == "entropy":
        raw_w = np.exp(-H_clip / temp)
    elif method == "rdp":
        raw_w = np.exp(rdp)
    elif method == "combined":
        raw_w = np.exp(rdp - H_clip / temp)
    elif method == "maxconf":
        raw_w = X.max(axis=2)
    elif method == "variance":
        raw_w = X.var(axis=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    if normalize:
        raw_w = raw_w / raw_w.sum(axis=1, keepdims=True) * N_WINDOWS
    return raw_w  # (n_files, T)


def clip_weighted_lse_gm_cp(logits, method="entropy", temp=0.5, beta=4.5,
                              gm_alpha=0.175, cp_thr=0.06):
    """Generic clip-weighted LSE→GM→cSEBBs pipeline.
    Compute per-clip weights then scale logits before LSE pooling.
    """
    n_files = logits.shape[0] // N_WINDOWS
    w = _compute_clip_weights(logits, method=method, temp=temp)  # (n_files, T)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    weighted_logits = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(weighted_logits, radius=1, beta=beta)
    after_gm = global_mean_blend(lse, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def soft_entr_t02_lse_gm_cp(logits):
    """SoftEntrWt T=0.2 (sharper than T=0.5 winner)."""
    return clip_weighted_lse_gm_cp(logits, method="entropy", temp=0.2)


def soft_entr_t03_lse_gm_cp(logits):
    """SoftEntrWt T=0.3."""
    return clip_weighted_lse_gm_cp(logits, method="entropy", temp=0.3)


def soft_entr_t04_lse_gm_cp(logits):
    """SoftEntrWt T=0.4."""
    return clip_weighted_lse_gm_cp(logits, method="entropy", temp=0.4)


def soft_entr_t06_lse_gm_cp(logits):
    """SoftEntrWt T=0.6."""
    return clip_weighted_lse_gm_cp(logits, method="entropy", temp=0.6)


def combined_weight_lse_gm_cp(logits, temp=0.5):
    """Combined (Entropy + RDP) weight → LSE → GM → cSEBBs.
    w(t) ∝ exp(rdp(t) - H(t)/T): upweight low-entropy + high-deviation clips.
    """
    return clip_weighted_lse_gm_cp(logits, method="combined", temp=temp)


def maxconf_weight_lse_gm_cp(logits):
    """MaxConf weight → LSE → GM → cSEBBs.
    w(t) = max_c(p(t,c)): clips with highest peak confidence get more weight.
    """
    return clip_weighted_lse_gm_cp(logits, method="maxconf")


def variance_weight_lse_gm_cp(logits):
    """Variance weight → LSE → GM → cSEBBs.
    w(t) = var_c(p(t,c)): high within-clip variance → more focused prediction → upweight.
    """
    return clip_weighted_lse_gm_cp(logits, method="variance")


def rdp_exp2_lse_gm_cp(logits, gm_alpha=0.175, cp_thr=0.06):
    """RDP with sharper exp(2*rdp) scaling."""
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    mu = X.mean(axis=1, keepdims=True)
    sg = X.std(axis=1, keepdims=True) + 1e-6
    rdp = (np.abs(X - mu) / sg).mean(axis=2)  # (n_files, T)
    w = np.exp(2.0 * rdp)
    w = w / w.sum(axis=1, keepdims=True) * N_WINDOWS
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(wl, radius=1, beta=4.5)
    after_gm = global_mean_blend(lse, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def entr_rdp_two_stage_gm_cp(logits, temp=0.5, gm_alpha=0.175, cp_thr=0.06):
    """Two-stage: first SoftEntr weighting, then RDP weighting after LSE.
    Stage 1: SoftEntr→LSE  (filter noisy clips)
    Stage 2: RDP→GM (amplify high-deviation clips in the file-level anchor)
    """
    n_files = logits.shape[0] // N_WINDOWS
    # Stage 1: entropy-weight then LSE
    w_entr = _compute_clip_weights(logits, method="entropy", temp=temp)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * w_entr[:, :, None]).reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(wl, radius=1, beta=4.5)
    # Stage 2: RDP weight on the LSE output
    w_rdp = _compute_clip_weights(lse, method="rdp")
    lse_raw = lse.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    lse_wl = (lse_raw * w_rdp[:, :, None]).reshape(-1, NUM_CLASSES)
    after_gm = global_mean_blend(lse_wl, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


# ── Round 12: ProbOR Boost / RDP weight / Per-class Platt / SoftEntr fine-tune ─

def prob_or_global_boost(logits, w=0.8):
    """Probabilistic OR global boost (BirdCLEF 2024 3rd place).
    P_boost(t,c) = P(t,c) + (P_max(c) + P_mean(c) - P_max(c)*P_mean(c)) * w
    P_max + P_mean - P_max*P_mean = P(max OR mean) — probabilistic OR.
    Additive boost (not convex blend): stronger for species heard both strongly+often.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_max  = X.max(axis=1)              # (n_files, C)
    file_mean = X.mean(axis=1)            # (n_files, C)
    prob_or   = file_max + file_mean - file_max * file_mean  # probabilistic OR
    out = X + w * prob_or[:, None, :]
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return (np.log(out / (1 - out)) * TEMP_SCALE).reshape(-1, NUM_CLASSES)


def lse_prob_or_cp(logits, w=0.8, cp_thr=0.06):
    """LSE → ProbOR boost → cSEBBs."""
    lse = logsumexp_pool(logits, radius=1, beta=4.5)
    after_or = prob_or_global_boost(lse, w=w)
    return change_point_segment_mean(after_or, threshold=cp_thr, blend=0.4)


def lse_gm_prob_or_cp(logits, gm_alpha=0.175, w=0.3, cp_thr=0.06):
    """LSE → GM → ProbOR boost → cSEBBs.
    ProbOR is added after GM to inject max-presence signal.
    w is smaller here since GM already provides global context.
    """
    lse = logsumexp_pool(logits, radius=1, beta=4.5)
    after_gm = global_mean_blend(lse, alpha=gm_alpha)
    after_or = prob_or_global_boost(after_gm, w=w)
    return change_point_segment_mean(after_or, threshold=cp_thr, blend=0.4)


def rdp_weight_lse_gm_cp(logits, beta=4.5, gm_alpha=0.175, cp_thr=0.06):
    """RDP-weighted LSE: weight clips by relative deviation from file mean.
    Clips that deviate strongly from file mean → likely true detections → upweight.
    Source: arXiv:2603.04605 (RDP for anomalous sound detection, DCASE 2025 SOTA).
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    file_mean = X.mean(axis=1, keepdims=True)            # (n_files, 1, C)
    file_std  = X.std(axis=1, keepdims=True) + 1e-6     # (n_files, 1, C)
    rdp = np.abs(X - file_mean) / file_std               # (n_files, T, C) — normalized deviation
    # Per-clip RDP score: mean across classes
    rdp_score = rdp.mean(axis=2)                         # (n_files, T)
    # Soft weight: exp(rdp_score) / Z — deviation-proportional upweight
    rdp_w = np.exp(rdp_score)
    rdp_w = rdp_w / rdp_w.sum(axis=1, keepdims=True) * N_WINDOWS
    # Scale logits by RDP weight before LSE
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    weighted_logits = (raw * rdp_w[:, :, None]).reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(weighted_logits, radius=1, beta=beta)
    after_gm = global_mean_blend(lse, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def soft_entr_lse_gm_cp_t05(logits):
    """SoftEntrWt T=0.5 (sharper than T=1 winner)."""
    return soft_entropy_weight_lse_gm_cp(logits, temp=0.5)


def soft_entr_lse_gm_cp_t15(logits):
    """SoftEntrWt T=1.5 (between T=1 and T=2)."""
    return soft_entropy_weight_lse_gm_cp(logits, temp=1.5)


def soft_entr_prob_or_cp(logits, temp=1.0, w=0.3, cp_thr=0.06):
    """SoftEntrWt → LSE → ProbOR (instead of GM) → cSEBBs.
    Replace GM with ProbOR as the global context step.
    """
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    H = -(X * np.log(X + 1e-9) + (1 - X) * np.log(1 - X + 1e-9))
    H_per_clip = H.mean(axis=2)
    ew = np.exp(-H_per_clip / temp) * N_WINDOWS
    ew = ew / ew.sum(axis=1, keepdims=True)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * ew[:, :, None]).reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(wl, radius=1, beta=4.5)
    after_or = prob_or_global_boost(lse, w=w)
    return change_point_segment_mean(after_or, threshold=cp_thr, blend=0.4)


def soft_entr_lse_prob_or_gm_cp(logits, temp=1.0, or_w=0.15, gm_alpha=0.175, cp_thr=0.06):
    """SoftEntrWt → LSE → ProbOR → GM → cSEBBs: full 4-step chain."""
    probs = sigmoid(logits)
    n_files = probs.shape[0] // N_WINDOWS
    X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    H = -(X * np.log(X + 1e-9) + (1 - X) * np.log(1 - X + 1e-9))
    H_per_clip = H.mean(axis=2)
    ew = np.exp(-H_per_clip / temp) * N_WINDOWS
    ew = ew / ew.sum(axis=1, keepdims=True)
    raw = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)
    wl = (raw * ew[:, :, None]).reshape(-1, NUM_CLASSES)
    lse = logsumexp_pool(wl, radius=1, beta=4.5)
    after_or = prob_or_global_boost(lse, w=or_w)
    after_gm = global_mean_blend(after_or, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


def per_class_platt_lse_gm_cp(logits_in, fold_id, Y_gt, gm_alpha=0.175, cp_thr=0.06):
    """Per-class Platt scaling (fold-aware): fit T_c, b_c per class.
    Source: arXiv:2511.08261 — per-class temperature + bias for rare species.
    Fold-aware: fit on train folds, apply to val fold.
    Returns calibrated logits at (N, C).
    """
    from sklearn.linear_model import LogisticRegression
    cal_logits = logits_in.copy()
    for k in range(4):
        val_mask = fold_id == k
        train_mask = ~val_mask & (fold_id >= 0)
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        X_tr = logits_in[train_mask]   # (n_train, C)
        Y_tr = Y_gt[train_mask]        # (n_train, C)
        X_val = logits_in[val_mask]    # (n_val, C)
        cal_val = np.zeros_like(X_val)
        for c in range(NUM_CLASSES):
            if Y_tr[:, c].sum() < 2 or (1 - Y_tr[:, c]).sum() < 2:
                cal_val[:, c] = X_val[:, c]  # skip if too few examples
                continue
            lr = LogisticRegression(C=1.0, max_iter=100, solver='lbfgs')
            lr.fit(X_tr[:, c:c+1], Y_tr[:, c])
            # lr.coef_[0,0] ≈ 1/T_c (temperature), lr.intercept_[0] ≈ b_c (bias)
            cal_val[:, c] = (X_val[:, c] * lr.coef_[0, 0] + lr.intercept_[0]) / TEMP_SCALE
        cal_logits[val_mask] = cal_val * TEMP_SCALE
    lse = logsumexp_pool(cal_logits, radius=1, beta=4.5)
    after_gm = global_mean_blend(lse, alpha=gm_alpha)
    return change_point_segment_mean(after_gm, threshold=cp_thr, blend=0.4)


class LearnableConv(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES, kernel_size=5, init_mode="gaussian",
                 event_idx=None, texture_idx=None):
        super().__init__()
        self.K    = kernel_size
        self.conv = nn.Conv1d(n_classes, n_classes, kernel_size, groups=n_classes,
                              padding=kernel_size // 2, bias=False, padding_mode="replicate")
        K = kernel_size
        with torch.no_grad():
            if init_mode == "gaussian":
                k = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], dtype=torch.float32)
                self.conv.weight.data[:, 0, :] = k.unsqueeze(0).expand(n_classes, -1)
            elif init_mode == "asymmetric":
                sym = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], dtype=torch.float32)
                asym = torch.tensor([0.05, 0.10, 0.50, 0.25, 0.10], dtype=torch.float32)
                asym = asym / asym.sum()
                self.conv.weight.data[:, 0, :] = sym.unsqueeze(0).expand(n_classes, -1)
                if event_idx is not None and len(event_idx) > 0:
                    ev = torch.tensor(event_idx, dtype=torch.long)
                    self.conv.weight.data[ev, 0, :] = asym.unsqueeze(0).expand(len(ev), -1)
            elif init_mode == "identity":
                nn.init.zeros_(self.conv.weight)
                self.conv.weight.data[:, 0, K // 2] = 1.0

    def forward(self, logits):
        N, C = logits.shape
        n_files = N // N_WINDOWS
        x = torch.tensor(logits, dtype=torch.float32)
        x = x.view(n_files, N_WINDOWS, C).permute(0, 2, 1)
        x = self.conv(x).permute(0, 2, 1).reshape(N, C)
        return x.detach().numpy()


class SoftOrderStats(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES, kernel_size=5, init_mode="class_specific",
                 event_idx=None, texture_idx=None):
        super().__init__()
        self.K    = kernel_size
        self.half = kernel_size // 2
        self.alpha = nn.Parameter(torch.zeros(n_classes, kernel_size))
        K = kernel_size
        with torch.no_grad():
            if init_mode == "median_biased":
                peak = torch.zeros(K); peak[K // 2] = 2.5
                self.alpha.data[:] = peak.unsqueeze(0).expand(n_classes, -1)
            elif init_mode == "max_biased":
                peak = torch.zeros(K); peak[-1] = 2.5
                self.alpha.data[:] = peak.unsqueeze(0).expand(n_classes, -1)
            elif init_mode == "class_specific":
                peak_med = torch.zeros(K); peak_med[K // 2] = 2.5
                peak_max = torch.zeros(K); peak_max[-1] = 2.5
                self.alpha.data[:] = peak_med.unsqueeze(0).expand(n_classes, -1)
                if event_idx is not None and len(event_idx) > 0:
                    ev = torch.tensor(event_idx, dtype=torch.long)
                    self.alpha.data[ev] = peak_max.unsqueeze(0).expand(len(ev), -1)
                if texture_idx is not None and len(texture_idx) > 0:
                    tx = torch.tensor(texture_idx, dtype=torch.long)
                    self.alpha.data[tx] = 0.0

    def forward(self, logits):
        N, C = logits.shape
        n_files = N // N_WINDOWS
        x = torch.tensor(logits, dtype=torch.float32)
        x = x.view(n_files, N_WINDOWS, C)
        x_perm   = x.permute(0, 2, 1)
        x_padded = F.pad(x_perm, (self.half, self.half), mode="replicate")
        windows  = x_padded.unfold(2, self.K, 1)
        sorted_w, _ = torch.sort(windows, dim=-1)
        w = F.softmax(self.alpha, dim=-1).unsqueeze(0).unsqueeze(2)
        out = (sorted_w * w).sum(dim=-1).permute(0, 2, 1).reshape(N, C)
        return out.detach().numpy()


class MultiScaleConv(nn.Module):
    SCALES = (3, 7, 11)

    def __init__(self, n_classes=NUM_CLASSES, scales=(3, 7, 11), init_mode="class_specific",
                 event_idx=None, texture_idx=None):
        super().__init__()
        self.scales = scales
        self.convs  = nn.ModuleList([
            nn.Conv1d(n_classes, n_classes, k, groups=n_classes,
                      padding=k // 2, bias=False, padding_mode="replicate")
            for k in scales
        ])
        self.gate = nn.Parameter(torch.zeros(n_classes, len(scales)))
        with torch.no_grad():
            for conv, k in zip(self.convs, scales):
                nn.init.constant_(conv.weight, 1.0 / k)
            if init_mode == "class_specific":
                if event_idx is not None and len(event_idx) > 0:
                    ev = torch.tensor(event_idx, dtype=torch.long)
                    self.gate.data[ev, 0] = 2.0
                if texture_idx is not None and len(texture_idx) > 0:
                    tx = torch.tensor(texture_idx, dtype=torch.long)
                    self.gate.data[tx, 2] = 2.0

    def forward(self, logits):
        N, C = logits.shape
        n_files = N // N_WINDOWS
        x = torch.tensor(logits, dtype=torch.float32)
        x = x.view(n_files, N_WINDOWS, C).permute(0, 2, 1)
        T = x.shape[-1]
        outs    = [conv(x)[:, :, :T] for conv in self.convs]
        stacked = torch.stack(outs, dim=-1)
        gate_w  = F.softmax(self.gate, dim=-1).unsqueeze(0).unsqueeze(2)
        out     = (stacked * gate_w).sum(dim=-1).permute(0, 2, 1).reshape(N, C)
        return out.detach().numpy()


class BilateralSmooth(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES, kernel_size=5, sigma_t=1.5, init_sigma_v=0.5):
        super().__init__()
        self.K    = kernel_size
        self.half = kernel_size // 2
        self.log_sigma_v = nn.Parameter(torch.full((n_classes,), float(np.log(init_sigma_v))))
        offsets = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
        self.register_buffer("temporal_w", torch.exp(-offsets ** 2 / (2.0 * sigma_t ** 2)))

    def forward(self, logits):
        N, C = logits.shape
        n_files = N // N_WINDOWS
        x = torch.tensor(logits, dtype=torch.float32)
        x = x.view(n_files, N_WINDOWS, C).permute(0, 2, 1)
        x_padded = F.pad(x, (self.half, self.half), mode="replicate")
        windows  = x_padded.unfold(2, self.K, 1)
        x_center = x.unsqueeze(-1)
        diff_sq  = (windows - x_center) ** 2
        sigma_v2 = (torch.exp(self.log_sigma_v) ** 2 + 1e-8).view(1, C, 1, 1)
        value_w  = torch.exp(-diff_sq / (2.0 * sigma_v2))
        tw       = self.temporal_w.view(1, 1, 1, self.K)
        combined = value_w * tw
        combined = combined / combined.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        out = (windows * combined).sum(dim=-1).permute(0, 2, 1).reshape(N, C)
        return out.detach().numpy()


class CausalIIR(nn.Module):
    """Bidirectional per-class EMA with learned decay alpha_c = sigmoid(log_alpha_c).

    Event classes: fast decay (alpha~0.27, log_alpha=-1.0) — tracks transients.
    Texture classes: slow decay (alpha~0.73, log_alpha=+1.0) — heavy smoothing.
    out = 0.5 * forward_ema + 0.5 * backward_ema
    """
    def __init__(self, n_classes=NUM_CLASSES, init_mode="class_specific",
                 event_idx=None, texture_idx=None):
        super().__init__()
        self.log_alpha_c = nn.Parameter(torch.zeros(n_classes))
        with torch.no_grad():
            if init_mode == "class_specific":
                if event_idx is not None and len(event_idx) > 0:
                    self.log_alpha_c.data[torch.tensor(event_idx, dtype=torch.long)] = -1.0
                if texture_idx is not None and len(texture_idx) > 0:
                    self.log_alpha_c.data[torch.tensor(texture_idx, dtype=torch.long)] = 1.0

    def forward(self, logits):
        N, C = logits.shape
        n_files = N // N_WINDOWS
        x = torch.tensor(logits, dtype=torch.float32)
        x = x.view(n_files, N_WINDOWS, C)           # (B, T, C)
        alpha = torch.sigmoid(self.log_alpha_c)      # (C,)
        # Forward EMA: h_t = alpha * h_{t-1} + (1-alpha) * x_t
        fwd = torch.zeros_like(x)
        h   = torch.zeros(n_files, C)
        for t in range(N_WINDOWS):
            h = alpha * h + (1 - alpha) * x[:, t, :]
            fwd[:, t, :] = h
        # Backward EMA
        bwd = torch.zeros_like(x)
        h   = torch.zeros(n_files, C)
        for t in range(N_WINDOWS - 1, -1, -1):
            h = alpha * h + (1 - alpha) * x[:, t, :]
            bwd[:, t, :] = h
        out = 0.5 * (fwd + bwd)
        return out.reshape(N, C).detach().numpy()


class DerivativeOnset(nn.Module):
    """Amplifies rising edges: out(t) = x(t) + alpha_c * relu(x(t) - x(t-1)).

    Highlights transient onset events; alpha_c = softplus(raw_alpha_c) ≥ 0.
    Event classes init with raw_alpha_c=1.0 (alpha~1.31), others=0.0 (alpha~0.69).
    """
    def __init__(self, n_classes=NUM_CLASSES, init_mode="event_biased",
                 event_idx=None):
        super().__init__()
        self.raw_alpha_c = nn.Parameter(torch.zeros(n_classes))
        with torch.no_grad():
            if init_mode == "event_biased" and event_idx is not None and len(event_idx) > 0:
                self.raw_alpha_c.data[torch.tensor(event_idx, dtype=torch.long)] = 1.0

    def forward(self, logits):
        N, C = logits.shape
        n_files = N // N_WINDOWS
        x = torch.tensor(logits, dtype=torch.float32)
        x = x.view(n_files, N_WINDOWS, C)           # (B, T, C)
        alpha = F.softplus(self.raw_alpha_c)         # (C,) positive
        # Pad first frame: diff at t=0 is zero
        x_prev = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        rising = F.relu(x - x_prev)                 # (B, T, C)
        out = x + alpha * rising
        return out.reshape(N, C).detach().numpy()


class SoftTopHat(nn.Module):
    """Morphological soft top-hat: highlights transient signal above local background.

    soft_erosion(x)  = -1/beta * logsumexp(-beta * window_k)
    soft_opening(x)  = soft_dilation(soft_erosion(x))
    out(t) = x(t) + alpha_c * relu(x(t) - opening(t))
    """
    def __init__(self, n_classes=NUM_CLASSES, kernel_size=5, init_beta=2.0,
                 event_idx=None):
        super().__init__()
        self.K    = kernel_size
        self.half = kernel_size // 2
        self.log_beta    = nn.Parameter(torch.tensor(float(np.log(init_beta))))
        self.raw_alpha_c = nn.Parameter(torch.zeros(n_classes))
        with torch.no_grad():
            if event_idx is not None and len(event_idx) > 0:
                self.raw_alpha_c.data[torch.tensor(event_idx, dtype=torch.long)] = 1.0

    def forward(self, logits):
        N, C = logits.shape
        n_files = N // N_WINDOWS
        x  = torch.tensor(logits, dtype=torch.float32)
        xp = x.view(n_files, N_WINDOWS, C).permute(0, 2, 1)   # (B, C, T)
        beta = torch.exp(self.log_beta)
        # Soft erosion (min approximation)
        xp_pad  = F.pad(xp, (self.half, self.half), mode="replicate")
        windows = xp_pad.unfold(2, self.K, 1)                  # (B, C, T, K)
        erosion = -1.0 / beta * torch.logsumexp(-beta * windows, dim=-1)
        # Soft dilation of erosion → opening (max approximation)
        er_pad  = F.pad(erosion, (self.half, self.half), mode="replicate")
        er_wins = er_pad.unfold(2, self.K, 1)
        opening = 1.0 / beta * torch.logsumexp(beta * er_wins, dim=-1)  # (B, C, T)
        # Top-hat: amplify signal above background
        alpha    = F.softplus(self.raw_alpha_c)                # (C,) positive
        residual = F.relu(xp - opening) * alpha.unsqueeze(0).unsqueeze(-1)
        out = (xp + residual).permute(0, 2, 1).reshape(N, C)
        return out.detach().numpy()


class SmootherEnsemble(nn.Module):
    """Per-class softmax weighting over M pre-computed OOF smoothed logit arrays.

    stacked: (N, C, M) — M component smoothed logit arrays stacked on last dim.
    Learns gamma(C, M); out(c) = sum_m softmax(gamma_c)[m] * component_m(c).
    """
    def __init__(self, n_classes=NUM_CLASSES, n_components=2):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(n_classes, n_components))

    def forward(self, stacked):
        # stacked: (N, C, M)
        w = F.softmax(self.gamma, dim=-1)            # (C, M)
        return (stacked * w.unsqueeze(0)).sum(dim=-1) # (N, C)


# ── Generic smooth model training ─────────────────────────────────────────────
def train_smooth_model(model, oof_final_logits, Y, fold_id, epochs=40, lr=0.05, l2=1e-3,
                        eval_fold=None):
    """
    Train any smooth model on OOF final logits via BCE.

    eval_fold: if set, EXCLUDE that fold from training (proper OOF for smooth model).
               This prevents train/val leakage when evaluating fold `eval_fold`.
               Call once per fold, or once with eval_fold=None to train on all data.
    """
    # Build training mask: exclude eval_fold to prevent leakage
    if eval_fold is not None:
        train_mask = fold_id != eval_fold   # train on all except eval fold
    else:
        train_mask = np.ones(len(fold_id), dtype=bool)  # train on everything

    X_logits = torch.tensor(oof_final_logits[train_mask], dtype=torch.float32)
    Y_target = torch.tensor(Y[train_mask], dtype=torch.float32)

    # Adjust N_WINDOWS grouping: must be multiple of N_WINDOWS
    # Filter to complete files only (trim to nearest N_WINDOWS multiple)
    n_complete = (len(X_logits) // N_WINDOWS) * N_WINDOWS
    X_logits = X_logits[:n_complete]
    Y_target = Y_target[:n_complete]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss, best_state = float("inf"), None
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        out   = _model_forward_train(model, X_logits)
        probs = torch.sigmoid(out / TEMP_SCALE)
        loss  = F.binary_cross_entropy(probs, Y_target, reduction="mean")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); scheduler.step()
        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model


def _model_forward_train(model, X_logits):
    """Differentiable forward pass for training (avoids detach in forward)."""
    N, C = X_logits.shape
    n_files = N // N_WINDOWS

    if isinstance(model, LearnableConv):
        x = X_logits.view(n_files, N_WINDOWS, C).permute(0, 2, 1)
        return model.conv(x).permute(0, 2, 1).reshape(N, C)

    elif isinstance(model, SoftOrderStats):
        x = X_logits.view(n_files, N_WINDOWS, C)
        xp = x.permute(0, 2, 1)
        xpad = F.pad(xp, (model.half, model.half), mode="replicate")
        w_in = xpad.unfold(2, model.K, 1)
        sw, _ = torch.sort(w_in, dim=-1)
        wt = F.softmax(model.alpha, dim=-1).unsqueeze(0).unsqueeze(2)
        return (sw * wt).sum(dim=-1).permute(0, 2, 1).reshape(N, C)

    elif isinstance(model, MultiScaleConv):
        x = X_logits.view(n_files, N_WINDOWS, C).permute(0, 2, 1)
        T = x.shape[-1]
        outs    = [conv(x)[:, :, :T] for conv in model.convs]
        stacked = torch.stack(outs, dim=-1)
        gw      = F.softmax(model.gate, dim=-1).unsqueeze(0).unsqueeze(2)
        return (stacked * gw).sum(dim=-1).permute(0, 2, 1).reshape(N, C)

    elif isinstance(model, BilateralSmooth):
        x = X_logits.view(n_files, N_WINDOWS, C).permute(0, 2, 1)
        xpad = F.pad(x, (model.half, model.half), mode="replicate")
        wins = xpad.unfold(2, model.K, 1)
        xctr = x.unsqueeze(-1)
        dsq  = (wins - xctr) ** 2
        sv2  = (torch.exp(model.log_sigma_v) ** 2 + 1e-8).view(1, C, 1, 1)
        vw   = torch.exp(-dsq / (2.0 * sv2))
        tw   = model.temporal_w.view(1, 1, 1, model.K)
        comb = vw * tw
        comb = comb / comb.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return (wins * comb).sum(dim=-1).permute(0, 2, 1).reshape(N, C)

    elif isinstance(model, CausalIIR):
        x = X_logits.view(n_files, N_WINDOWS, C)
        alpha = torch.sigmoid(model.log_alpha_c)
        fwd = torch.zeros_like(x)
        h   = torch.zeros(n_files, C)
        for t in range(N_WINDOWS):
            h = alpha * h + (1 - alpha) * x[:, t, :]
            fwd[:, t, :] = h
        bwd = torch.zeros_like(x)
        h   = torch.zeros(n_files, C)
        for t in range(N_WINDOWS - 1, -1, -1):
            h = alpha * h + (1 - alpha) * x[:, t, :]
            bwd[:, t, :] = h
        return (0.5 * (fwd + bwd)).reshape(N, C)

    elif isinstance(model, DerivativeOnset):
        x = X_logits.view(n_files, N_WINDOWS, C)
        alpha  = F.softplus(model.raw_alpha_c)
        x_prev = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        rising = F.relu(x - x_prev)
        return (x + alpha * rising).reshape(N, C)

    elif isinstance(model, SoftTopHat):
        xp   = X_logits.view(n_files, N_WINDOWS, C).permute(0, 2, 1)
        beta = torch.exp(model.log_beta)
        xp_pad  = F.pad(xp, (model.half, model.half), mode="replicate")
        windows = xp_pad.unfold(2, model.K, 1)
        erosion = -1.0 / beta * torch.logsumexp(-beta * windows, dim=-1)
        er_pad  = F.pad(erosion, (model.half, model.half), mode="replicate")
        er_wins = er_pad.unfold(2, model.K, 1)
        opening = 1.0 / beta * torch.logsumexp(beta * er_wins, dim=-1)
        alpha   = F.softplus(model.raw_alpha_c)
        residual = F.relu(xp - opening) * alpha.unsqueeze(0).unsqueeze(-1)
        return (xp + residual).permute(0, 2, 1).reshape(N, C)

    raise ValueError(f"Unknown model type: {type(model)}")


# ── Per-fold AUC evaluation ────────────────────────────────────────────────────
def eval_per_fold(probs, Y, fold_id):
    results = {}
    valid_folds = sorted(set(fold_id[fold_id >= 0]))
    aucs = []
    for k in valid_folds:
        mask = fold_id == k
        auc  = macro_auc(Y, probs, fold_mask=mask)
        results[f"fold{k}"] = auc
        aucs.append(auc)
    results["mean"] = float(np.mean(aucs))
    return results


# ── SmootherEnsemble training + evaluation ────────────────────────────────────
def train_smoother_ensemble(model, stacked_logits, Y, fold_id, epochs=30, lr=0.1, l2=1e-3,
                             eval_fold=None):
    """Train SmootherEnsemble on stacked OOF logits (N, C, M).

    eval_fold: exclude this fold from training to prevent leakage.
    stacked_logits: numpy (N, C, M) — pre-computed OOF smoothed logits.
    """
    if eval_fold is not None:
        train_mask = fold_id != eval_fold
    else:
        train_mask = np.ones(len(fold_id), dtype=bool)
    X   = torch.tensor(stacked_logits[train_mask], dtype=torch.float32)
    Y_t = torch.tensor(Y[train_mask], dtype=torch.float32)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    best_loss, best_state = float("inf"), None
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out   = model(X)
        probs = torch.sigmoid(out / TEMP_SCALE)
        loss  = F.binary_cross_entropy(probs, Y_t, reduction="mean")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def eval_smoother_ensemble(combo_names, smoothed_logits_dict, Y, fold_id,
                            epochs=30, lr=0.1, l2=1e-3):
    """Evaluate SmootherEnsemble for a combo of pre-smoothed methods (fold-aware OOF).

    smoothed_logits_dict: {method_name → (N, C) OOF smoothed logits (numpy)}
    combo_names: list of method names to combine.
    """
    M       = len(combo_names)
    stacked = np.stack([smoothed_logits_dict[n] for n in combo_names], axis=2)  # (N, C, M)
    smoothed = np.zeros((len(Y), NUM_CLASSES), dtype=np.float32)
    for k in range(4):
        val_mask = fold_id == k
        if val_mask.sum() == 0:
            continue
        model_k = SmootherEnsemble(n_classes=NUM_CLASSES, n_components=M)
        train_smoother_ensemble(model_k, stacked, Y, fold_id,
                                epochs=epochs, lr=lr, l2=l2, eval_fold=k)
        with torch.no_grad():
            X_val = torch.tensor(stacked[val_mask], dtype=torch.float32)
            smoothed[val_mask] = model_k(X_val).numpy()
    probs = sigmoid(smoothed)
    return eval_per_fold(probs, Y, fold_id)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild extended 66-file Perch cache")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs per model")
    parser.add_argument("--ens_epochs", type=int, default=30, help="Epochs for SmootherEnsemble")
    parser.add_argument("--skip_ensemble", action="store_true",
                        help="Skip SmootherEnsemble combination experiments")
    args = parser.parse_args()

    print("=" * 70)
    print("Local Temporal Smoothing Experiment Evaluation")
    print(f"Epochs per model: {args.epochs}")
    print("=" * 70)

    # ── Setup ──────────────────────────────────────────────────────────────────
    (PRIMARY_LABELS, label_to_idx, BC_INDICES,
     MAPPED_MASK, MAPPED_POS, MAPPED_BC_INDICES,
     CLASS_NAME_MAP, TEXTURE_TAXA) = build_species_mapping()

    sc_df, Y, full_files = build_ground_truth(PRIMARY_LABELS, label_to_idx)
    global _OOF_LABELS; _OOF_LABELS = Y
    print(f"Ground truth: {len(sc_df)} windows from {len(full_files)} files")
    print(f"Classes with ≥1 positive: {(Y.sum(axis=0) > 0).sum()}/234")

    # ── Build/load extended Perch cache ────────────────────────────────────────
    scores_full_raw, emb_full, filenames_list, row_ids_list = build_extended_cache(
        sc_df, MAPPED_POS, MAPPED_BC_INDICES, rebuild=args.rebuild_cache
    )

    # Align Y with cache order
    row_id_to_idx = {r: i for i, r in enumerate(row_ids_list)}
    sc_df["cache_idx"] = sc_df["row_id"].map(row_id_to_idx)
    valid = sc_df["cache_idx"].notna()
    sc_df = sc_df[valid].reset_index(drop=True)
    cache_idx = sc_df["cache_idx"].astype(int).values
    scores_full_raw = scores_full_raw[cache_idx]
    emb_full        = emb_full[cache_idx]
    filenames_arr   = np.array([filenames_list[i] for i in cache_idx])
    Y               = Y[valid.values] if valid.sum() < len(Y) else Y

    print(f"Aligned cache: {scores_full_raw.shape},  Y: {Y.shape}")

    # ── Build fold_id for each window ─────────────────────────────────────────
    fold_id = np.full(len(filenames_arr), -1, dtype=np.int32)
    for k in range(4):
        val_files = set(open(f"{FOLDS_DIR}/ss_fold{k}_val.txt").read().splitlines())
        for i, f in enumerate(filenames_arr):
            if f in val_files:
                fold_id[i] = k

    val_coverage = (fold_id >= 0).sum()
    print(f"Val windows:  {val_coverage}/{len(fold_id)}  "
          f"({val_coverage/len(fold_id)*100:.1f}% covered by 4 folds)")
    for k in range(4):
        n_files_k = len(set(filenames_arr[fold_id == k]))
        print(f"  fold{k}: {(fold_id == k).sum()} windows ({n_files_k} files)")

    # ── Class type indices ─────────────────────────────────────────────────────
    active_classes   = np.where(Y.sum(axis=0) > 0)[0]
    idx_event   = np.array([i for i in active_classes
                             if CLASS_NAME_MAP.get(PRIMARY_LABELS[i]) not in TEXTURE_TAXA],
                            dtype=np.int32)
    idx_texture = np.array([i for i in active_classes
                             if CLASS_NAME_MAP.get(PRIMARY_LABELS[i]) in TEXTURE_TAXA],
                            dtype=np.int32)
    print(f"Aves/event classes: {len(idx_event)},  Texture classes: {len(idx_texture)}")

    # ── OOF probe predictions ──────────────────────────────────────────────────
    print("\n── OOF LGBM probe ─────────────────────────────────────────────────")
    oof_final, oof_base, Z_all, fold_id_probe, alpha = build_oof_probe_predictions(
        scores_full_raw, emb_full, list(filenames_arr), sc_df, Y,
        PRIMARY_LABELS, label_to_idx
    )
    assert np.allclose(fold_id, fold_id_probe), "fold_id mismatch"

    # ── Results table ──────────────────────────────────────────────────────────
    results = {}

    def evaluate(name, logits, make_model_fn=None, lr=0.05, l2=1e-3):
        """
        Evaluate a smoothing method.
        If make_model_fn is given, train a fresh model per fold (proper OOF).
        make_model_fn() → nn.Module
        Returns (fold_aucs, smoothed_logits).
        """
        smoothed_logits = logits.copy()
        if make_model_fn is not None:
            print(f"\n── Training (fold-aware OOF): {name} ──────────────────────────────")
            # For each fold k: train on folds {0,1,2,3}\{k}, predict on fold k
            for k in range(4):
                val_mask_k = fold_id == k
                if val_mask_k.sum() == 0:
                    continue
                model_k = make_model_fn()
                train_smooth_model(model_k, logits, Y, fold_id,
                                   epochs=args.epochs, lr=lr, l2=l2, eval_fold=k)
                # Apply to eval fold only
                with torch.no_grad():
                    smoothed_logits[val_mask_k] = model_k(logits)[val_mask_k]
                print(f"  fold{k} done", end="\r")
            print()

        probs = sigmoid(smoothed_logits)
        fold_aucs = eval_per_fold(probs, Y, fold_id)
        results[name] = fold_aucs

        fold_str = "  ".join(f"f{k}={fold_aucs.get(f'fold{k}', 0):.4f}" for k in range(4))
        print(f"{name:30s}  {fold_str}  mean={fold_aucs['mean']:.4f}")
        return fold_aucs, smoothed_logits

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"{'Method':<30}  {'fold0':>8}  {'fold1':>8}  {'fold2':>8}  {'fold3':>8}  {'mean':>8}")
    print("-" * 70)

    # Dict to collect OOF smoothed logits for ensemble experiments
    smoothed_logits = {}

    # 0. Raw (no probe, no smooth)
    evaluate("0.Raw (no probe)",  oof_base)

    # 1. Probe only (no smooth)
    _, smoothed_logits["probe"] = evaluate("1.Probe (no smooth)", oof_final)

    # 2. Gaussian (fixed 0-910 kernel, baseline — no training needed)
    gauss_logits = gaussian_smooth(oof_final)
    _, smoothed_logits["gaussian"] = evaluate("2.Gaussian (fixed)",  gauss_logits)

    # 3. Learnable Conv1d — gaussian init (fold-aware OOF training)
    _, smoothed_logits["conv_gauss"] = evaluate(
        "3.LearnableConv/gaussian", oof_final,
        make_model_fn=lambda: LearnableConv(init_mode="gaussian",
                                             event_idx=idx_event, texture_idx=idx_texture))

    # 4. Learnable Conv1d — asymmetric init
    _, smoothed_logits["conv_asym"] = evaluate(
        "4.LearnableConv/asymmetric", oof_final,
        make_model_fn=lambda: LearnableConv(init_mode="asymmetric",
                                             event_idx=idx_event, texture_idx=idx_texture))

    # 5. Soft Order Stats — class_specific init (event→max, texture→mean)
    _, smoothed_logits["sos"] = evaluate(
        "5.SoftOrderStats/class_spec", oof_final,
        make_model_fn=lambda: SoftOrderStats(init_mode="class_specific",
                                              event_idx=idx_event, texture_idx=idx_texture))

    # 6. Soft Order Stats — median init
    _, smoothed_logits["sos_median"] = evaluate(
        "6.SoftOrderStats/median", oof_final,
        make_model_fn=lambda: SoftOrderStats(init_mode="median_biased"))

    # 7. Multi-Scale Conv — class_specific gate (Aves→K=3, Texture→K=11)
    _, smoothed_logits["multiscale"] = evaluate(
        "7.MultiScale(3,7,11)/gate", oof_final,
        make_model_fn=lambda: MultiScaleConv(init_mode="class_specific",
                                              event_idx=idx_event, texture_idx=idx_texture),
        l2=1e-3)

    # 8. Bilateral — sigma_v=0.5 init
    _, smoothed_logits["bilateral05"] = evaluate(
        "8.Bilateral/sigma_v=0.5", oof_final,
        make_model_fn=lambda: BilateralSmooth(init_sigma_v=0.5),
        lr=0.1, l2=1e-4)

    # 9. Bilateral — sigma_v=0.1 (edge-preserving)
    _, smoothed_logits["bilateral01"] = evaluate(
        "9.Bilateral/sigma_v=0.1", oof_final,
        make_model_fn=lambda: BilateralSmooth(init_sigma_v=0.1),
        lr=0.1, l2=1e-4)

    # 10. CausalIIR — class_specific (event→fast, texture→slow)
    _, smoothed_logits["causal_iir"] = evaluate(
        "10.CausalIIR/class_spec", oof_final,
        make_model_fn=lambda: CausalIIR(init_mode="class_specific",
                                         event_idx=idx_event, texture_idx=idx_texture),
        lr=0.05, l2=1e-3)

    # 11. DerivativeOnset — event-biased init
    _, smoothed_logits["deriv_onset"] = evaluate(
        "11.DerivativeOnset/event", oof_final,
        make_model_fn=lambda: DerivativeOnset(init_mode="event_biased",
                                               event_idx=idx_event),
        lr=0.05, l2=1e-3)

    # 12. SoftTopHat — event-biased init (beta=2.0)
    _, smoothed_logits["soft_tophat"] = evaluate(
        "12.SoftTopHat/event", oof_final,
        make_model_fn=lambda: SoftTopHat(init_beta=2.0, event_idx=idx_event),
        lr=0.05, l2=1e-3)

    # ── Literature-based zero-param methods ───────────────────────────────────
    def _lit_eval(name, smoothed_logits_arr):
        """Evaluate a pre-smoothed (N, C) logit array — no training needed."""
        probs = sigmoid(smoothed_logits_arr)
        fold_aucs = eval_per_fold(probs, Y, fold_id)
        results[name] = fold_aucs
        delta = fold_aucs["mean"] - gauss_mean
        marker = " ⭐" if delta > 0 else ""
        print(f"  {name:35s}  {fold_aucs['mean']:.4f}  {delta:+.4f}{marker}")
        return fold_aucs

    print("\n── Literature zero-param methods ──")
    print(f"  {'Method':35s}  {'mean':>6}  {'vs Gaussian':>12}")
    gauss_mean = results.get("2.Gaussian (fixed)", {}).get("mean", 0.0)

    _lit_eval("L1.Median(w=3)",         median_smooth(oof_final, size=3))
    _lit_eval("L2.Median→Gaussian",     median_then_gaussian(oof_final))
    _lit_eval("L3.SavitzkyGolay(w=5,p=2)", savgol_smooth(oof_final))
    _lit_eval("L4.HMM(onset=0.2,p=0.7)", hmm_smooth(oof_final))
    _lit_eval("L5.MaxDilation(r=1)",    max_dilation(oof_final, radius=1))
    _lit_eval("L6.MaxDilation(r=2)",    max_dilation(oof_final, radius=2))
    _lit_eval("L7.MultiScaleMedian",    multiscale_median(oof_final))
    _lit_eval("L8.AlphaTrimmed(a=0.2)", alpha_trimmed_mean(oof_final))
    _lit_eval("L9.Gauss→Dilation(r=1)", gaussian_then_dilation(oof_final, radius=1))
    _lit_eval("L10.Med→Gauss→Dilation", median_gauss_dilation(oof_final))

    # ── Round 2: Dilation variants ────────────────────────────────────────────
    print("\n── Round 2: Dilation variants ──")
    _lit_eval("R2.01.ClassAdaptDil(e=1,t=2)",
              dilation_class_adaptive(oof_final, radius_event=1, radius_texture=2,
                                      event_idx=idx_event, texture_idx=idx_texture))
    _lit_eval("R2.02.GaussWide→Dil(r=1)",
              gauss_dilation_wide(oof_final, radius=1))
    _lit_eval("R2.03.Closing(r=1)",
              morphological_closing(oof_final, radius=1))
    _lit_eval("R2.04.Gauss→Closing(r=1)",
              gauss_closing(oof_final, radius=1))
    _lit_eval("R2.05.Dil→Gauss(r=1)",
              dilation_then_gauss(oof_final, radius=1))
    _lit_eval("R2.06.DoubleDil(r=1×2)",
              double_dilation(oof_final, radius=1))
    _lit_eval("R2.07.Gauss→DoubleDil",
              gauss_double_dilation(oof_final, radius=1))
    _lit_eval("R2.08.Percentile75(w=3)",
              percentile_filter(oof_final, window=3, pct=75))
    _lit_eval("R2.09.Percentile90(w=3)",
              percentile_filter(oof_final, window=3, pct=90))
    _lit_eval("R2.10.LogSumExp(b=3)",
              logsumexp_pool(oof_final, radius=1, beta=3.0))
    _lit_eval("R2.11.LogSumExp(b=5)",
              logsumexp_pool(oof_final, radius=1, beta=5.0))
    _lit_eval("R2.12.LogSumExp(b=10)",
              logsumexp_pool(oof_final, radius=1, beta=10.0))
    _lit_eval("R2.13.Gauss→LSE(b=3)",
              gauss_logsumexp(oof_final, radius=1, beta=3.0))
    _lit_eval("R2.14.Gauss→LSE(b=5)",
              gauss_logsumexp(oof_final, radius=1, beta=5.0))
    _lit_eval("R2.15.MaxBlend(a=0.3)",
              temporal_max_blend(oof_final, alpha=0.3, radius=1))
    _lit_eval("R2.16.MaxBlend(a=0.5)",
              temporal_max_blend(oof_final, alpha=0.5, radius=1))
    _lit_eval("R2.17.MaxBlend(a=0.7)",
              temporal_max_blend(oof_final, alpha=0.7, radius=1))
    _lit_eval("R2.18.Gauss→MaxBlend(a=0.5)",
              gauss_maxblend(oof_final, alpha=0.5, radius=1))
    _lit_eval("R2.19.Gauss→MaxBlend(a=0.7)",
              gauss_maxblend(oof_final, alpha=0.7, radius=1))
    _lit_eval("R2.20.Gauss→Dil(r=2)",
              gaussian_then_dilation(oof_final, radius=2))

    # ── Round 3: LSE tuning + Global Mean Blend + combos ─────────────────────
    print("\n── Round 3: LSE tuning + Global Mean Blend ──")
    _lit_eval("R3.01.LSE(b=4)",            lse_b4(oof_final))
    _lit_eval("R3.02.LSE(b=7)",            lse_b7(oof_final))
    _lit_eval("R3.03.LSE(b=5,r=2)",        logsumexp_pool_r2(oof_final, beta=5.0))
    _lit_eval("R3.04.LSE→Gauss(b=5)",      lse_then_gauss(oof_final, beta=5.0))
    _lit_eval("R3.05.Gauss→LSE(b=4)",      gauss_lse_b4(oof_final))
    _lit_eval("R3.06.Gauss→LSE(b=7)",      gauss_lse_b7(oof_final))
    _lit_eval("R3.07.Gauss→LSE(b=5,r=2)",  gauss_lse_b5_r2(oof_final))
    _lit_eval("R3.08.GlobalMean(a=0.1)",   global_mean_blend(oof_final, alpha=0.1))
    _lit_eval("R3.09.GlobalMean(a=0.2)",   global_mean_blend(oof_final, alpha=0.2))
    _lit_eval("R3.10.GlobalMean(a=0.3)",   global_mean_blend(oof_final, alpha=0.3))
    _lit_eval("R3.11.Gauss→GlobalMean(0.2)", gauss_global_blend(oof_final, alpha=0.2))
    _lit_eval("R3.12.LSE→GlobalMean(b5,a0.2)", lse_global_blend(oof_final, beta=5.0, alpha=0.2))
    _lit_eval("R3.13.Gauss→LSE→GlobalMean",gauss_lse_global(oof_final, beta=5.0, alpha=0.2))
    _lit_eval("R3.14.LSE+Pct90(mix=0.5)",  lse_percentile_blend(oof_final, beta=5.0, pct=90))
    _lit_eval("R3.15.Gauss→LSE+Pct90",    gauss_lse_percentile(oof_final, beta=5.0, pct=90))
    _lit_eval("R3.16.Dil+LSE(b5,mix=0.5)",dilation_lse_blend(oof_final, beta=5.0, mix=0.5))
    _lit_eval("R3.17.Dil+LSE(b5,mix=0.3)",dilation_lse_blend(oof_final, beta=5.0, mix=0.3))
    _lit_eval("R3.18.Dil+LSE(b5,mix=0.7)",dilation_lse_blend(oof_final, beta=5.0, mix=0.7))

    # ── Round 4: Literature search 2026-03-21 (8 new techniques) ─────────────
    print("\n── Round 4: Power/BoundaryAware/SoftNMS ──")
    _lit_eval("R4.01.Power(a=0.8,thr=0.15)",  power_adjustment(oof_final, alpha=0.8))
    _lit_eval("R4.02.Power(a=0.7,thr=0.15)",  power_adjustment(oof_final, alpha=0.7))
    _lit_eval("R4.03.Power(a=0.9,thr=0.15)",  power_adjustment(oof_final, alpha=0.9))
    _lit_eval("R4.04.Gauss→Power(a=0.8)",     gauss_power(oof_final, alpha=0.8))
    _lit_eval("R4.05.LSE→Power(b5,a=0.8)",    lse_power(oof_final, beta=5.0, alpha=0.8))
    _lit_eval("R4.06.Gauss→LSE→Power",        gauss_lse_power(oof_final, beta=5.0, alpha=0.8))
    _lit_eval("R4.07.BoundaryAware(bl=0.4)",  boundary_aware_smooth(oof_final, blend=0.4))
    _lit_eval("R4.08.BoundaryAware(bl=0.2)",  boundary_aware_smooth(oof_final, blend=0.2))
    _lit_eval("R4.09.Gauss→Boundary(bl=0.4)", gauss_boundary(oof_final, blend=0.4))
    _lit_eval("R4.10.LSE→Boundary(b5,bl=0.3)",lse_boundary(oof_final, beta=5.0, blend=0.3))
    _lit_eval("R4.11.SoftNMS(σ=1.0)",         soft_nms_temporal(oof_final, sigma=1.0))
    _lit_eval("R4.12.Gauss→SoftNMS(σ=1.0)",   gauss_soft_nms(oof_final, sigma=1.0))
    _lit_eval("R4.13.LSE→GlobalMean→Power",   lse_global_power(oof_final, beta=5.0))
    _lit_eval("R4.14.Gauss→LSE→GM→Power",     gauss_lse_global_power(oof_final, beta=5.0))
    _lit_eval("R4.15.LSE+ClassPrior(s=0.1)",  lse_with_class_prior(oof_final, beta=5.0, prior_strength=0.1))
    _lit_eval("R4.16.LSE+ClassPrior(s=0.05)", lse_with_class_prior(oof_final, beta=5.0, prior_strength=0.05))
    # Gauss→LSE→GlobalMean(a=0.15) — tighter blend
    _lit_eval("R4.17.G→LSE→GM(a=0.15)",       gauss_lse_global(oof_final, beta=5.0, alpha=0.15))
    _lit_eval("R4.18.G→LSE→GM(a=0.1)",        gauss_lse_global(oof_final, beta=5.0, alpha=0.1))

    # ── Round 5: EMA / MultiScale-LSE / cSEBBs-lite / fine-tuned combos ───────
    print("\n── Round 5: EMA / MultiScaleLSE / cSEBBs / fine combos ──")
    _lit_eval("R5.01.BidirEMA(a=0.3)",          bidirectional_ema(oof_final, alpha=0.3))
    _lit_eval("R5.02.BidirEMA(a=0.5)",          bidirectional_ema(oof_final, alpha=0.5))
    _lit_eval("R5.03.AsymEMA(rise=0.5,fall=0.15)", asymmetric_ema(oof_final, alpha_rise=0.5, alpha_fall=0.15))
    _lit_eval("R5.04.AsymEMA(rise=0.7,fall=0.2)", asymmetric_ema(oof_final, alpha_rise=0.7, alpha_fall=0.2))
    _lit_eval("R5.05.Gauss→BidirEMA(a=0.3)",    gauss_bidir_ema(oof_final, alpha=0.3))
    _lit_eval("R5.06.LSE→BidirEMA(a=0.3)",      lse_bidir_ema(oof_final, beta=5.0, alpha=0.3))
    _lit_eval("R5.07.LSE→BidirEMA(a=0.5)",      lse_bidir_ema(oof_final, beta=5.0, alpha=0.5))
    _lit_eval("R5.08.LSE→GM→BidirEMA(a=0.3)",   lse_global_ema(oof_final, beta=5.0, gm_alpha=0.2, ema_alpha=0.3))
    _lit_eval("R5.09.LSE→GM→BidirEMA(a=0.5)",   lse_global_ema(oof_final, beta=5.0, gm_alpha=0.2, ema_alpha=0.5))
    _lit_eval("R5.10.LSE→GM→AsymEMA",           lse_global_asym_ema(oof_final, beta=5.0, gm_alpha=0.2))
    _lit_eval("R5.11.MultiScaleLSE(w=0.7)",      multiscale_lse_blend(oof_final, beta=5.0, w=0.7))
    _lit_eval("R5.12.MultiScaleLSE(w=0.85)",     multiscale_lse_blend(oof_final, beta=5.0, w=0.85))
    _lit_eval("R5.13.MultiScaleLSE→GM(w=0.7)",  lse_multiscale_global(oof_final, beta=5.0, w=0.7, gm_alpha=0.2))
    _lit_eval("R5.14.MultiScaleLSE→GM(w=0.85)", lse_multiscale_global(oof_final, beta=5.0, w=0.85, gm_alpha=0.2))
    _lit_eval("R5.15.cSEBBs(thr=0.08,bl=0.5)",  change_point_segment_mean(oof_final, threshold=0.08, blend=0.5))
    _lit_eval("R5.16.cSEBBs(thr=0.05,bl=0.3)",  change_point_segment_mean(oof_final, threshold=0.05, blend=0.3))
    _lit_eval("R5.17.cSEBBs→LSE→GM",            cp_lse_global(oof_final, beta=5.0, cp_thr=0.08, cp_blend=0.5))
    _lit_eval("R5.18.LSE→GM(a=0.175)",           lse_global_fine(oof_final, beta=5.0, alpha=0.175))
    _lit_eval("R5.19.LSE→GM(a=0.225)",           lse_global_fine(oof_final, beta=5.0, alpha=0.225))
    _lit_eval("R5.20.LSE→GM(a=0.15)",            lse_global_fine(oof_final, beta=5.0, alpha=0.15))
    _lit_eval("R5.21.LSE→GM(a=0.25)",            lse_global_fine(oof_final, beta=5.0, alpha=0.25))
    _lit_eval("R5.22.Pct90→GM(a=0.2)",           percentile90_global_blend(oof_final, gm_alpha=0.2))
    _lit_eval("R5.23.LSEr1r2→GM(w=0.7)",         lse_r1r2_global(oof_final, beta=5.0, w=0.7, gm_alpha=0.2))
    _lit_eval("R5.24.LSEr1r2→GM(w=0.85)",        lse_r1r2_global(oof_final, beta=5.0, w=0.85, gm_alpha=0.2))

    # ── Round 6: Attention pooling / Gauss→GM / EMA fine-tune ─────────────────
    print("\n── Round 6: AttnPool / Gauss→GM / EMA fine / chains ──")
    _lit_eval("R6.01.LinSoftmaxAttn(r=1)",        linear_softmax_attn(oof_final, radius=1))
    _lit_eval("R6.02.LinSoftmaxAttn(r=2)",        linear_softmax_attn(oof_final, radius=2))
    _lit_eval("R6.03.LinAttn→GM(r=1,a=0.2)",      linear_softmax_global(oof_final, radius=1, gm_alpha=0.2))
    _lit_eval("R6.04.LinAttn→GM(r=1,a=0.15)",     linear_softmax_global(oof_final, radius=1, gm_alpha=0.15))
    _lit_eval("R6.05.Gauss→GM(a=0.2)",            gauss_global_mean(oof_final, alpha=0.2))
    _lit_eval("R6.06.Gauss→GM(a=0.15)",           gauss_global_mean(oof_final, alpha=0.15))
    _lit_eval("R6.07.Gauss→GM(a=0.1)",            gauss_global_mean(oof_final, alpha=0.1))
    _lit_eval("R6.08.LSE(b3)→GM(a=0.2)",          lse_b3_r1_global(oof_final, gm_alpha=0.2))
    _lit_eval("R6.09.LSE(b5,r=2)→GM(a=0.2)",      lse_b5_r2_global(oof_final, gm_alpha=0.2))
    _lit_eval("R6.10.G→LSE→GM(a=0.175)",          gauss_lse_global_fine(oof_final, beta=5.0, alpha=0.175))
    _lit_eval("R6.11.G→LSE→GM(a=0.225)",          gauss_lse_global_fine(oof_final, beta=5.0, alpha=0.225))
    _lit_eval("R6.12.LSE→GM→BidirEMA(a=0.2)",     lse_global_ema_fine(oof_final, beta=5.0, gm_alpha=0.2, ema_alpha=0.2))
    _lit_eval("R6.13.LSE→GM→BidirEMA(a=0.15)",    lse_global_ema_fine(oof_final, beta=5.0, gm_alpha=0.2, ema_alpha=0.15))
    _lit_eval("R6.14.LSE→GM→cSEBBs(thr=0.06)",   lse_global_cp(oof_final, beta=5.0, gm_alpha=0.2, cp_thr=0.06))
    _lit_eval("R6.15.LSE→GM→cSEBBs(thr=0.04)",   lse_global_cp(oof_final, beta=5.0, gm_alpha=0.2, cp_thr=0.04))
    _lit_eval("R6.16.G→LSE→GM→EMA(4-step)",       triple_chain(oof_final, beta=5.0, gm_alpha=0.2, ema_alpha=0.25))
    _lit_eval("R6.17.G→LSE→GM→EMA(a=0.15)",       triple_chain(oof_final, beta=5.0, gm_alpha=0.2, ema_alpha=0.15))

    # ── Round 7: cSEBBs fine-tune on top of best chain (2026-03-21) ───────────
    print("\n── Round 7: cSEBBs chain fine-tune ──")
    _lit_eval("R7.01.LSE→GM→cSEBBs(thr=0.03)",   lse_gm_cp_fine(oof_final, cp_thr=0.03, cp_blend=0.4))
    _lit_eval("R7.02.LSE→GM→cSEBBs(thr=0.05)",   lse_gm_cp_fine(oof_final, cp_thr=0.05, cp_blend=0.4))
    _lit_eval("R7.03.LSE→GM→cSEBBs(thr=0.07)",   lse_gm_cp_fine(oof_final, cp_thr=0.07, cp_blend=0.4))
    _lit_eval("R7.04.LSE→GM→cSEBBs(thr=0.10)",   lse_gm_cp_fine(oof_final, cp_thr=0.10, cp_blend=0.4))
    _lit_eval("R7.05.LSE→GM→cSEBBs(bl=0.3)",     lse_gm_cp_fine(oof_final, cp_thr=0.06, cp_blend=0.3))
    _lit_eval("R7.06.LSE→GM→cSEBBs(bl=0.6)",     lse_gm_cp_blend_sweep(oof_final, cp_blend=0.6))
    _lit_eval("R7.07.LSE→GM→cSEBBs(bl=0.7)",     lse_gm_cp_blend_sweep(oof_final, cp_blend=0.7))
    _lit_eval("R7.08.MultiSLSE→GM→cSEBBs",       ms_lse_gm_cp(oof_final, ms_w=0.85, cp_thr=0.06))
    _lit_eval("R7.09.MultiSLSE→GM→cSEBBs(w=0.7)",ms_lse_gm_cp(oof_final, ms_w=0.70, cp_thr=0.06))
    _lit_eval("R7.10.LSE→GM→cSEBBs→EMA(a=0.3)",  lse_gm_cp_ema(oof_final, cp_thr=0.06, ema_a=0.3))
    _lit_eval("R7.11.LSE→GM→cSEBBs→EMA(a=0.2)",  lse_gm_cp_ema(oof_final, cp_thr=0.06, ema_a=0.2))
    _lit_eval("R7.12.LSE→GM→double-cSEBBs",      lse_gm_double_cp(oof_final))
    _lit_eval("R7.13.LSE(b3)→GM→cSEBBs",         lse_b3_gm_cp(oof_final, cp_thr=0.06))
    _lit_eval("R7.14.LSE→GM→cSEBBs→Gauss",       lse_gm_cp_gauss(oof_final, cp_thr=0.06))
    _lit_eval("R7.15.LSE→GM(a=0.175)→cSEBBs",    lse_gm_a175_cp(oof_final, cp_thr=0.06))
    _lit_eval("R7.16.LSE→GM(a=0.175)→cSEBBs(t=0.04)", lse_gm_a175_cp(oof_final, cp_thr=0.04))

    # ── Round 8: RED / nSEBBs / Coarse-Max-Pool (arXiv:2601.04178 / 2505.11889 / 2406.15725)
    print("\n── Round 8: RED / nSEBBs / CoarseMaxPool ──")
    _lit_eval("R8.01.RED(causal)",               recurrent_event_detection(oof_final))
    _lit_eval("R8.02.BidirRED",                  bidir_red(oof_final))
    _lit_eval("R8.03.nSEBBs-lite(bl=0.4)",       nsebbs_lite(oof_final, base_blend=0.4))
    _lit_eval("R8.04.nSEBBs-lite(bl=0.3)",       nsebbs_lite(oof_final, base_blend=0.3))
    _lit_eval("R8.05.CoarseMaxPool(w=2)",         coarse_max_pool(oof_final, width=2))
    _lit_eval("R8.06.CoarseMaxPool(w=3)",         coarse_max_pool(oof_final, width=3))
    _lit_eval("R8.07.LSE→GM→RED",                lse_gm_red(oof_final, beta=5.0))
    _lit_eval("R8.08.LSE→GM→BidirRED",           lse_gm_bidir_red(oof_final, beta=5.0))
    _lit_eval("R8.09.LSE→GM→nSEBBs(bl=0.4)",     lse_gm_nsebbs(oof_final, beta=5.0, base_blend=0.4))
    _lit_eval("R8.10.LSE→GM→nSEBBs(bl=0.3)",     lse_gm_nsebbs(oof_final, beta=5.0, base_blend=0.3))
    _lit_eval("R8.11.LSE→GM→cSEBBs→Coarse(w=2)", lse_gm_cp_coarse(oof_final, width=2))
    _lit_eval("R8.12.Gauss→RED",                 recurrent_event_detection(gaussian_smooth(oof_final)))
    _lit_eval("R8.13.LSE→RED",                   recurrent_event_detection(logsumexp_pool(oof_final, radius=1, beta=5.0)))
    _lit_eval("R8.14.LSE→BidirRED",              bidir_red(logsumexp_pool(oof_final, radius=1, beta=5.0)))

    # ── Round 9: Beta fine-tune / logit-space GM / local-mean / pipeline blend ──
    print("\n── Round 9: Beta fine-tune / LogitGM / LocalMean / PipelineBlend ──")
    _lit_eval("R9.01.LSE(β=4.5)→GM→cSEBBs",      lse_beta45_gm_cp(oof_final))
    _lit_eval("R9.02.LSE(β=5.5)→GM→cSEBBs",      lse_beta55_gm_cp(oof_final))
    _lit_eval("R9.03.LSE(β=6)→GM→cSEBBs",        lse_beta6_gm_cp(oof_final))
    _lit_eval("R9.04.LSE→GM(α=0.15)→cSEBBs",     lse_gm_a15_cp(oof_final))
    _lit_eval("R9.05.LSE→GM(α=0.20)→cSEBBs",     lse_gm_a20_cp(oof_final))
    _lit_eval("R9.06.LSE→LogitGM→cSEBBs",        lse_gm_logit_cp(oof_final))
    _lit_eval("R9.07.LSE→HalfMean→cSEBBs",       lse_local_halfmean_cp(oof_final))
    _lit_eval("R9.08.PipelineBlend(w=0.8)",       best_pipeline_blend(oof_final, w=0.8))
    _lit_eval("R9.09.PipelineBlend(w=0.9)",       best_pipeline_blend(oof_final, w=0.9))
    _lit_eval("R9.10.FileMaxNorm→LSE→GM→cSEBBs",  file_max_norm_lse_gm_cp(oof_final))
    _lit_eval("R9.11.EntropyWt→LSE→GM→cSEBBs",   entropy_weighted_lse_gm_cp(oof_final))
    _lit_eval("R9.12.MultiSLSE→GM(α=0.175)→cSEBBs", ms_lse_gm_a175_cp(oof_final))

    # ── Round 10: PowerPooling / BirdPresenceAmp / GeomBidirEMA / EntropyDrop ──
    print("\n── Round 10: PowerPool / BirdPresenceAmp / GeomBidirEMA ──")
    _lit_eval("R10.01.PowerPool(α=2)→GM→cSEBBs",    power_pool_gm_cp(oof_final, alpha=2.0))
    _lit_eval("R10.02.PowerPool(α=3)→GM→cSEBBs",    power_pool_gm_cp(oof_final, alpha=3.0))
    _lit_eval("R10.03.PowerPool(α=1.5)→GM→cSEBBs",  power_pool_gm_cp(oof_final, alpha=1.5))
    _lit_eval("R10.04.BirdPresAmp(a=0.10)",          bird_presence_amp(oof_final, alpha=0.10))
    _lit_eval("R10.05.BirdPresAmp(a=0.15)",          bird_presence_amp(oof_final, alpha=0.15))
    _lit_eval("R10.06.BirdPresAmp→GM→cSEBBs(a=0.1)", bird_presence_amp_gm_cp(oof_final, amp_alpha=0.10))
    _lit_eval("R10.07.BirdPresAmp→GM→cSEBBs(a=0.15)",bird_presence_amp_gm_cp(oof_final, amp_alpha=0.15))
    _lit_eval("R10.08.LSE→BirdPresAmp→cSEBBs",       lse_bird_presence_amp_cp(oof_final))
    _lit_eval("R10.09.GeomBidirEMA(a=0.3)",           geom_bidir_ema(oof_final, alpha=0.3))
    _lit_eval("R10.10.GeomBidirEMA(a=0.5)",           geom_bidir_ema(oof_final, alpha=0.5))
    _lit_eval("R10.11.LSE→GeomEMA→GM→cSEBBs",        lse_geom_bidir_ema_gm_cp(oof_final))
    _lit_eval("R10.12.EntropyDrop→LSE→GM→cSEBBs",    entropy_clip_drop_lse_gm_cp(oof_final, drop_q=0.5))
    _lit_eval("R10.13.EntropyDrop(q=0.3)→pipe",      entropy_clip_drop_lse_gm_cp(oof_final, drop_q=0.3))

    # ── Round 11: Soft-entropy / Selective-max-mean / Trim-mean / Diffusion
    # ──          Confidence-adaptive GM / SoftMax-blend / Spectral LP ─────────
    print("\n── Round 11: SoftEntropy / SelectivePool / TrimMean / Diffusion / SpectralLP ──")
    _lit_eval("R11.01.SoftEntrWt→LSE→GM→cSEBBs(T=1)",  soft_entropy_weight_lse_gm_cp(oof_final, temp=1.0))
    _lit_eval("R11.02.SoftEntrWt→LSE→GM→cSEBBs(T=2)",  soft_entropy_weight_lse_gm_cp(oof_final, temp=2.0))
    _lit_eval("R11.03.SelMaxMean(thr=0.05)→GM→cSEBBs",  selective_max_mean_gm_cp(oof_final, rare_thr=0.05))
    _lit_eval("R11.04.SelMaxMean(thr=0.10)→GM→cSEBBs",  selective_max_mean_gm_cp(oof_final, rare_thr=0.10))
    _lit_eval("R11.05.BetaBlend(3,4.5,6)→GM→cSEBBs",    beta_pipeline_blend(oof_final))
    _lit_eval("R11.06.BetaBlend(4,4.5,5)→GM→cSEBBs",    beta_pipeline_blend(oof_final, betas=(4.0, 4.5, 5.0)))
    _lit_eval("R11.07.TrimMean(k=1)→GM→cSEBBs",         trim_mean_pool(oof_final, drop_k=1))
    _lit_eval("R11.08.TrimMean(k=2)→GM→cSEBBs",         trim_mean_pool(oof_final, drop_k=2))
    _lit_eval("R11.09.Diffusion(λ=0.25,N=2)→GM→cSEBBs", diffusion_smooth_gm_cp(oof_final, lam=0.25, n_steps=2))
    _lit_eval("R11.10.Diffusion(λ=0.2,N=3)→GM→cSEBBs",  diffusion_smooth_gm_cp(oof_final, lam=0.20, n_steps=3))
    _lit_eval("R11.11.LSE→ConfAdaptGM→cSEBBs",          lse_confidence_weighted_gm_cp(oof_final))
    _lit_eval("R11.12.LSE→SoftMaxBlend→cSEBBs",         lse_softmax_blend_cp(oof_final, alpha=0.175))
    _lit_eval("R11.13.LSE→SoftMaxBlend(a=0.1)→cSEBBs",  lse_softmax_blend_cp(oof_final, alpha=0.10))
    _lit_eval("R11.14.SpectralLP(c=0.5)→GM→cSEBBs",     spectral_lowpass_gm_cp(oof_final, cutoff_frac=0.5))
    _lit_eval("R11.15.SpectralLP(c=0.3)→GM→cSEBBs",     spectral_lowpass_gm_cp(oof_final, cutoff_frac=0.3))
    _lit_eval("R11.16.LSE(b4.5)→GM(a0.175)→cSEBBs(t5)", lse_b45_gm175_cp_fine(oof_final, cp_thr=0.05))
    _lit_eval("R11.17.LSE(b4.5)→GM(a0.175)→cSEBBs(t7)", lse_b45_gm175_cp_fine(oof_final, cp_thr=0.07))
    _lit_eval("R11.18.LSE→MaxMeanBlend(w=0.5)→cSEBBs",  lse_max_mean_blend_gm_cp(oof_final, mean_w=0.5))
    _lit_eval("R11.19.LSE→MaxMeanBlend(w=0.3)→cSEBBs",  lse_max_mean_blend_gm_cp(oof_final, mean_w=0.3))

    # ── Round 12: ProbOR / RDP / Platt / SoftEntr fine-tune (2026-03-21) ───────
    # Sources: arXiv:2603.04605 (RDP), arXiv:2511.08261 (Platt), BirdCLEF2024 3rd place
    print("\n── Round 12: ProbOR / RDP / Platt / SoftEntr fine-tune ──")
    _lit_eval("R12.01.ProbOR(w=0.8)standalone",          prob_or_global_boost(oof_final, w=0.8))
    _lit_eval("R12.02.LSE→ProbOR(w=0.8)→cSEBBs",        lse_prob_or_cp(oof_final, w=0.8))
    _lit_eval("R12.03.LSE→ProbOR(w=0.4)→cSEBBs",        lse_prob_or_cp(oof_final, w=0.4))
    _lit_eval("R12.04.LSE→GM→ProbOR(w=0.3)→cSEBBs",     lse_gm_prob_or_cp(oof_final, w=0.3))
    _lit_eval("R12.05.LSE→GM→ProbOR(w=0.15)→cSEBBs",    lse_gm_prob_or_cp(oof_final, w=0.15))
    _lit_eval("R12.06.RDP→LSE→GM→cSEBBs",               rdp_weight_lse_gm_cp(oof_final))
    _lit_eval("R12.07.SoftEntrWt(T=0.5)→LSE→GM→cSEBBs", soft_entr_lse_gm_cp_t05(oof_final))
    _lit_eval("R12.08.SoftEntrWt(T=1.5)→LSE→GM→cSEBBs", soft_entr_lse_gm_cp_t15(oof_final))
    _lit_eval("R12.09.SoftEntr→ProbOR(w=0.3)→cSEBBs",   soft_entr_prob_or_cp(oof_final, w=0.3))
    _lit_eval("R12.10.SoftEntr→LSE→ProbOR→GM→cSEBBs",   soft_entr_lse_prob_or_gm_cp(oof_final, or_w=0.15))
    _lit_eval("R12.11.PerClassPlatt→LSE→GM→cSEBBs",     per_class_platt_lse_gm_cp(oof_final, fold_id, Y))

    # ── Round 13: Combined Entr+RDP / T fine-tune / MaxConf / Variance ─────────
    # Key insight from R12: SoftEntr(T=0.5)=0.7890 and RDP=0.7876 both work via
    # clip discrimination — reward informative clips before LSE pooling.
    print("\n── Round 13: CombinedWeight / T fine-tune / MaxConf / Variance ──")
    _lit_eval("R13.01.SoftEntr(T=0.2)→LSE→GM→cSEBBs",  soft_entr_t02_lse_gm_cp(oof_final))
    _lit_eval("R13.02.SoftEntr(T=0.3)→LSE→GM→cSEBBs",  soft_entr_t03_lse_gm_cp(oof_final))
    _lit_eval("R13.03.SoftEntr(T=0.4)→LSE→GM→cSEBBs",  soft_entr_t04_lse_gm_cp(oof_final))
    _lit_eval("R13.04.SoftEntr(T=0.6)→LSE→GM→cSEBBs",  soft_entr_t06_lse_gm_cp(oof_final))
    _lit_eval("R13.05.Combined(T=0.5)→LSE→GM→cSEBBs",  combined_weight_lse_gm_cp(oof_final, temp=0.5))
    _lit_eval("R13.06.Combined(T=0.3)→LSE→GM→cSEBBs",  combined_weight_lse_gm_cp(oof_final, temp=0.3))
    _lit_eval("R13.07.Combined(T=1.0)→LSE→GM→cSEBBs",  combined_weight_lse_gm_cp(oof_final, temp=1.0))
    _lit_eval("R13.08.MaxConf→LSE→GM→cSEBBs",          maxconf_weight_lse_gm_cp(oof_final))
    _lit_eval("R13.09.Variance→LSE→GM→cSEBBs",         variance_weight_lse_gm_cp(oof_final))
    _lit_eval("R13.10.RDP(exp2)→LSE→GM→cSEBBs",        rdp_exp2_lse_gm_cp(oof_final))
    _lit_eval("R13.11.SoftEntr→LSE→RDP→GM→cSEBBs",     entr_rdp_two_stage_gm_cp(oof_final, temp=0.5))
    _lit_eval("R13.12.SoftEntr→LSE→RDP→GM→cSEBBs(T03)", entr_rdp_two_stage_gm_cp(oof_final, temp=0.3))

    # ── Round 14: T lower / nSEBBs-PCR / MPA-hybrid / MeanMax / TAP-Velocity ───
    # Key insight from R13: T=0.2 > T=0.5 — sharper entropy cutoff keeps winning.
    # Sources: arXiv:2505.11889 (nSEBBs), arXiv:2406.12721 (MPA), arXiv:2508.20703
    print("\n── Round 14: T<0.2 / nSEBBs-PCR / MPA / MeanMax / TAP-Velocity ──")
    _lit_eval("R14.01.SoftEntr(T=0.1)→LSE→GM→cSEBBs",  soft_entr_lse_gm_cp_t(oof_final, temp=0.1))
    _lit_eval("R14.02.SoftEntr(T=0.15)→LSE→GM→cSEBBs", soft_entr_lse_gm_cp_t(oof_final, temp=0.15))
    _lit_eval("R14.03.SoftEntr(T=0.05)→LSE→GM→cSEBBs", soft_entr_lse_gm_cp_t(oof_final, temp=0.05))
    _lit_eval("R14.04.SoftEntr(T=0.2)→LSE→GM→nSEBBs",  soft_entr_lse_gm_nsebbs(oof_final, temp=0.2))
    _lit_eval("R14.05.SoftEntr(T=0.1)→LSE→GM→nSEBBs",  soft_entr_lse_gm_nsebbs(oof_final, temp=0.1))
    _lit_eval("R14.06.MPA-hybrid→GM→cSEBBs",            mpa_hybrid_gm_cp(oof_final, rare_thr=0.05))
    _lit_eval("R14.07.MPA-hybrid(thr=0.1)→GM→cSEBBs",  mpa_hybrid_gm_cp(oof_final, rare_thr=0.10))
    _lit_eval("R14.08.MeanMax(w=0.5)→GM→cSEBBs",        mean_max_blend_gm_cp(oof_final, blend_w=0.5))
    _lit_eval("R14.09.MeanMax(w=0.3)→GM→cSEBBs",        mean_max_blend_gm_cp(oof_final, blend_w=0.3))
    _lit_eval("R14.10.TAPVelocity(v=0.3)→GM→cSEBBs",   tap_velocity_lse_gm_cp(oof_final, vel_w=0.3))
    _lit_eval("R14.11.TAPVelocity(v=0.2)→GM→cSEBBs",   tap_velocity_lse_gm_cp(oof_final, vel_w=0.2))
    _lit_eval("R14.12.Variance→LSE→GM→nSEBBs",          variance_lse_gm_nsebbs(oof_final))

    # ── Round 15: MeanMax fine-tune / T inside MeanMax / Variance+MeanMax ───────
    # Key insight R14: MeanMax(w=0.5) best (0.7927) — 0.5*max+0.5*mean anchor >>
    # pure GlobalMean. Explore anchor weight & entropy temp within MeanMax.
    print("\n── Round 15: MeanMax fine-tune / T sweep inside MeanMax / Variance+MeanMax ──")
    _lit_eval("R15.01.MeanMax(T=0.15,w=0.5)→cSEBBs",   mean_max_full(oof_final, entr_temp=0.15, max_w=0.5))
    _lit_eval("R15.02.MeanMax(T=0.1,w=0.5)→cSEBBs",    mean_max_full(oof_final, entr_temp=0.10, max_w=0.5))
    _lit_eval("R15.03.MeanMax(T=0.2,w=0.4)→cSEBBs",    mean_max_full(oof_final, entr_temp=0.20, max_w=0.4))
    _lit_eval("R15.04.MeanMax(T=0.2,w=0.6)→cSEBBs",    mean_max_full(oof_final, entr_temp=0.20, max_w=0.6))
    _lit_eval("R15.05.MeanMax(T=0.2,w=0.7)→cSEBBs",    mean_max_full(oof_final, entr_temp=0.20, max_w=0.7))
    _lit_eval("R15.06.MeanMax(T=0.2,w=1.0)→cSEBBs",    mean_max_full(oof_final, entr_temp=0.20, max_w=1.0))
    _lit_eval("R15.07.MeanMax(T=0.15,w=0.4)→cSEBBs",   mean_max_full(oof_final, entr_temp=0.15, max_w=0.4))
    _lit_eval("R15.08.MeanMax(T=0.15,w=0.6)→cSEBBs",   mean_max_full(oof_final, entr_temp=0.15, max_w=0.6))
    _lit_eval("R15.09.MeanMax(T=0.2,a=0.15)→cSEBBs",   mean_max_full(oof_final, entr_temp=0.20, max_w=0.5, gm_alpha=0.15))
    _lit_eval("R15.10.MeanMax(T=0.2,a=0.20)→cSEBBs",   mean_max_full(oof_final, entr_temp=0.20, max_w=0.5, gm_alpha=0.20))
    _lit_eval("R15.11.MeanMax(T=0.2,a=0.25)→cSEBBs",   mean_max_full(oof_final, entr_temp=0.20, max_w=0.5, gm_alpha=0.25))
    _lit_eval("R15.12.Variance+MeanMax(w=0.5)→cSEBBs",  variance_mean_max_cp(oof_final, max_w=0.5))
    _lit_eval("R15.13.Variance+MeanMax(w=0.4)→cSEBBs",  variance_mean_max_cp(oof_final, max_w=0.4))
    _lit_eval("R15.14.MeanMax(T=0.1,w=0.4)→cSEBBs",    mean_max_full(oof_final, entr_temp=0.10, max_w=0.4))
    _lit_eval("R15.15.MeanMax(T=0.15,w=0.3)→cSEBBs",   mean_max_full(oof_final, entr_temp=0.15, max_w=0.3))

    # ── Round 17: w=1.0 fine-tune / radius / percentile anchor / double-pass ────
    # Key insight R15: w=1.0 (pure file_max anchor) >> blend. Explore:
    # 1) alpha fine-tune at w=1.0 — how much max to add
    # 2) Wider LSE radius (r=2) at w=1.0
    # 3) Softer anchor: p90/p95 percentile instead of hard max
    # 4) Double-pass MeanMax: two rounds of (EntrWt→LSE→MaxAnchor→cSEBBs)
    # 5) RDP weight variant at w=1.0
    # 6) Beta fine-tune at w=1.0
    print("\n── Round 17: w=1.0 fine-tune / radius / percentile anchor / double-pass ──")
    _lit_eval("R17.01.MeanMax(T=0.2,w=1.0,a=0.15)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.15))
    _lit_eval("R17.02.MeanMax(T=0.2,w=1.0,a=0.20)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.20))
    _lit_eval("R17.03.MeanMax(T=0.2,w=1.0,a=0.225)→cSEBBs", mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.225))
    _lit_eval("R17.04.MeanMax(T=0.2,w=1.0,a=0.25)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.25))
    _lit_eval("R17.05.MeanMax(T=0.2,w=1.0,a=0.30)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.30))
    _lit_eval("R17.06.MeanMax(T=0.2,w=1.0,r=2)→cSEBBs",     mean_max_full_r2(oof_final, entr_temp=0.20, max_w=1.0))
    _lit_eval("R17.07.MeanMax(T=0.15,w=1.0)→cSEBBs",        mean_max_full(oof_final, entr_temp=0.15, max_w=1.0))
    _lit_eval("R17.08.MeanMax(T=0.1,w=1.0)→cSEBBs",         mean_max_full(oof_final, entr_temp=0.10, max_w=1.0))
    _lit_eval("R17.09.Pct90Anchor(T=0.2)→cSEBBs",           percentile_anchor_cp(oof_final, entr_temp=0.20, pct=90))
    _lit_eval("R17.10.Pct95Anchor(T=0.2)→cSEBBs",           percentile_anchor_cp(oof_final, entr_temp=0.20, pct=95))
    _lit_eval("R17.11.Pct85Anchor(T=0.2)→cSEBBs",           percentile_anchor_cp(oof_final, entr_temp=0.20, pct=85))
    _lit_eval("R17.12.DoubleMeanMax(T=0.2,w=1.0)→cSEBBs",   double_meanmax_cp(oof_final, entr_temp=0.20, max_w=1.0))
    _lit_eval("R17.13.DoubleMeanMax(T=0.2,w=0.5)→cSEBBs",   double_meanmax_cp(oof_final, entr_temp=0.20, max_w=0.5))
    _lit_eval("R17.14.RDP+MeanMax(w=1.0)→cSEBBs",           rdp_mean_max_cp(oof_final, max_w=1.0))
    _lit_eval("R17.15.MeanMax(T=0.2,w=1.0,b=5.0)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, beta=5.0))
    _lit_eval("R17.16.MeanMax(T=0.2,w=1.0,b=4.0)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, beta=4.0))
    _lit_eval("R17.17.MeanMax(T=0.2,w=1.0,thr=0.05)→cSEBBs", mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, cp_thr=0.05))
    _lit_eval("R17.18.MeanMax(T=0.2,w=1.0,thr=0.07)→cSEBBs", mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, cp_thr=0.07))
    _lit_eval("R17.19.Var+MeanMax(w=1.0)→cSEBBs",           variance_mean_max_cp(oof_final, max_w=1.0))
    _lit_eval("R17.20.MeanMax(T=0.2,w=0.9)→cSEBBs",         mean_max_full(oof_final, entr_temp=0.20, max_w=0.9))

    # ── Round 19: T fine-tune at a=0.30 / AvgTopK / NoisyOR / PowerMean / DCT ──
    # Key insight R18: T=0.1 > T=0.2 at a=0.30 (0.8003 vs 0.7972 — new best!).
    # r=2+a=0.30 also strong (0.7987). Now:
    # 1) T fine-tune 0.05/0.08/0.12 at a=0.30,w=1.0 (where does T bottom out?)
    # 2) r=2 + T=0.1 + a=0.30 combo
    # 3) AvgTopK anchor k=2,3 (DCASE 2023 / ScienceDirect 2023) — soft-max between k=1(max) and k=12(mean)
    # 4) NoisyOR anchor 1-∏(1-p) (DCASE 2018 MIL) — multiplicative, amplifies rare clips
    # 5) PowerMean anchor p=2,3,4 (power pool as global file anchor, not local window)
    # 6) b=5.0 at T=0.1, a=0.30
    print("\n── Round 19: T<0.1 / r=2+T=0.1 / AvgTopK / NoisyOR / PowerMeanAnchor ──")
    _lit_eval("R19.01.MeanMax(T=0.05,w=1.0,a=0.30)→cSEBBs", mean_max_full(oof_final, entr_temp=0.05, max_w=1.0, gm_alpha=0.30))
    _lit_eval("R19.02.MeanMax(T=0.08,w=1.0,a=0.30)→cSEBBs", mean_max_full(oof_final, entr_temp=0.08, max_w=1.0, gm_alpha=0.30))
    _lit_eval("R19.03.MeanMax(T=0.12,w=1.0,a=0.30)→cSEBBs", mean_max_full(oof_final, entr_temp=0.12, max_w=1.0, gm_alpha=0.30))
    _lit_eval("R19.04.MeanMax(T=0.15,w=1.0,a=0.30)→cSEBBs", mean_max_full(oof_final, entr_temp=0.15, max_w=1.0, gm_alpha=0.30))
    _lit_eval("R19.05.MeanMax(T=0.1,w=1.0,r=2,a=0.30)→cSEBBs", mean_max_full_r2(oof_final, entr_temp=0.10, max_w=1.0, gm_alpha=0.30))
    _lit_eval("R19.06.MeanMax(T=0.1,w=1.0,r=2,a=0.35)→cSEBBs", mean_max_full_r2(oof_final, entr_temp=0.10, max_w=1.0, gm_alpha=0.35))
    _lit_eval("R19.07.MeanMax(T=0.1,w=1.0,a=0.35)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.10, max_w=1.0, gm_alpha=0.35))
    _lit_eval("R19.08.MeanMax(T=0.1,w=1.0,b=5.0,a=0.30)→cSEBBs", mean_max_full(oof_final, entr_temp=0.10, max_w=1.0, gm_alpha=0.30, beta=5.0))
    _lit_eval("R19.09.AvgTopK(k=2,a=0.30)→cSEBBs",          avgtopk_blend_lse_cp(oof_final, k=2, alpha=0.30))
    _lit_eval("R19.10.AvgTopK(k=3,a=0.30)→cSEBBs",          avgtopk_blend_lse_cp(oof_final, k=3, alpha=0.30))
    _lit_eval("R19.11.AvgTopK(k=2,a=0.175)→cSEBBs",         avgtopk_blend_lse_cp(oof_final, k=2, alpha=0.175))
    _lit_eval("R19.12.NoisyOR(a=0.30)→cSEBBs",              noisy_or_blend_lse_cp(oof_final, alpha=0.30))
    _lit_eval("R19.13.NoisyOR(a=0.175)→cSEBBs",             noisy_or_blend_lse_cp(oof_final, alpha=0.175))
    _lit_eval("R19.14.PowerMean(p=2,a=0.30)→cSEBBs",        power_mean_anchor_cp(oof_final, p=2.0, alpha=0.30))
    _lit_eval("R19.15.PowerMean(p=3,a=0.30)→cSEBBs",        power_mean_anchor_cp(oof_final, p=3.0, alpha=0.30))
    _lit_eval("R19.16.PowerMean(p=4,a=0.30)→cSEBBs",        power_mean_anchor_cp(oof_final, p=4.0, alpha=0.30))
    _lit_eval("R19.17.PowerMean(p=6,a=0.30)→cSEBBs",        power_mean_anchor_cp(oof_final, p=6.0, alpha=0.30))
    _lit_eval("R19.18.DCTLogit(K=4,a=0.30)→cSEBBs",         dct_logit_lse_cp(oof_final, K=4, alpha=0.30))
    _lit_eval("R19.19.DCTLogit(K=6,a=0.30)→cSEBBs",         dct_logit_lse_cp(oof_final, K=6, alpha=0.30))
    _lit_eval("R19.20.MeanMax(T=0.1,w=1.0,a=0.25)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.10, max_w=1.0, gm_alpha=0.25))

    # ── Round 20: NoisyOR fine-tune / NoisyOR+MeanMax combo / AvgTopK fine ──────
    # Key insight R19: NoisyOR(a=0.30)=0.8017 NEW BEST; r=2+T=0.1+a=0.35=0.8016 close.
    # NoisyOR (1-∏(1-p)) is multiplicative complement max-pool; stronger than GlobalMax.
    # Explore: alpha sweep for NoisyOR, T sweep, r=2, dual-anchor NoisyOR+Max, AvgTopK k=2 fine.
    print("\n── Round 20: NoisyOR fine-tune / dual anchor / AvgTopK sweep ──")
    _lit_eval("R20.01.NoisyOR(a=0.25)→cSEBBs",              noisy_or_blend_lse_cp(oof_final, alpha=0.25))
    _lit_eval("R20.02.NoisyOR(a=0.35)→cSEBBs",              noisy_or_blend_lse_cp(oof_final, alpha=0.35))
    _lit_eval("R20.03.NoisyOR(a=0.40)→cSEBBs",              noisy_or_blend_lse_cp(oof_final, alpha=0.40))
    _lit_eval("R20.04.NoisyOR(a=0.20)→cSEBBs",              noisy_or_blend_lse_cp(oof_final, alpha=0.20))
    _lit_eval("R20.05.NoisyOR(T=0.08,a=0.30)→cSEBBs",       noisy_or_blend_lse_cp(oof_final, alpha=0.30, entr_temp=0.08))
    _lit_eval("R20.06.NoisyOR(T=0.15,a=0.30)→cSEBBs",       noisy_or_blend_lse_cp(oof_final, alpha=0.30, entr_temp=0.15))
    _lit_eval("R20.07.NoisyOR(T=0.2,a=0.30)→cSEBBs",        noisy_or_blend_lse_cp(oof_final, alpha=0.30, entr_temp=0.20))
    _lit_eval("R20.08.NoisyOR(r=2,a=0.30)→cSEBBs",          noisy_or_blend_lse_r2_cp(oof_final, alpha=0.30))
    _lit_eval("R20.09.NoisyOR(b=5,a=0.30)→cSEBBs",          noisy_or_blend_lse_cp(oof_final, alpha=0.30, beta=5.0))
    _lit_eval("R20.10.NoisyOR(thr=0.07,a=0.30)→cSEBBs",     noisy_or_blend_lse_cp(oof_final, alpha=0.30, cp_thr=0.07))
    # Dual anchor: NoisyOR + GlobalMax blended together
    _lit_eval("R20.11.DualAnchor(NOR+Max,a=0.30)→cSEBBs",   dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.5))
    _lit_eval("R20.12.DualAnchor(NOR+Max,a=0.30,nw=0.7)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.7))
    _lit_eval("R20.13.DualAnchor(NOR+Max,a=0.30,nw=0.3)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.3))
    # AvgTopK fine-tune at best alpha
    _lit_eval("R20.14.AvgTopK(k=2,a=0.35)→cSEBBs",          avgtopk_blend_lse_cp(oof_final, k=2, alpha=0.35))
    _lit_eval("R20.15.AvgTopK(k=2,a=0.40)→cSEBBs",          avgtopk_blend_lse_cp(oof_final, k=2, alpha=0.40))
    _lit_eval("R20.16.AvgTopK(k=3,a=0.35)→cSEBBs",          avgtopk_blend_lse_cp(oof_final, k=3, alpha=0.35))
    _lit_eval("R20.17.NoisyOR(T=0.1,a=0.30,r=2,b=5)→cSEBBs", noisy_or_blend_lse_r2_cp(oof_final, alpha=0.30, beta=5.0))
    _lit_eval("R20.18.NoisyOR(T=0.08,a=0.35)→cSEBBs",        noisy_or_blend_lse_cp(oof_final, alpha=0.35, entr_temp=0.08))

    # ── Round 21: DualAnchor fine-tune / TripleAnchor / NOR+AvgTopK ─────────────
    # Key insight R20: DualAnchor(nor_w=0.5, NOR+Max, a=0.30) = 0.8032 NEW BEST.
    # nor_w=0.3 close (0.8030), nor_w=0.7 worse (0.8026) → optimum near nor_w=0.4-0.5.
    # Explore: nor_w fine-tune 0.35/0.4/0.45, alpha at optimal nor_w,
    #          TripleAnchor (NOR + Max + AvgTopK-2), r=2 + DualAnchor.
    print("\n── Round 21: DualAnchor fine-tune / nor_w sweep / TripleAnchor ──")
    _lit_eval("R21.01.DualAnchor(nw=0.40,a=0.30)→cSEBBs",   dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.40))
    _lit_eval("R21.02.DualAnchor(nw=0.45,a=0.30)→cSEBBs",   dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.45))
    _lit_eval("R21.03.DualAnchor(nw=0.35,a=0.30)→cSEBBs",   dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.35))
    _lit_eval("R21.04.DualAnchor(nw=0.25,a=0.30)→cSEBBs",   dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.25))
    _lit_eval("R21.05.DualAnchor(nw=0.20,a=0.30)→cSEBBs",   dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.20))
    _lit_eval("R21.06.DualAnchor(nw=0.50,a=0.35)→cSEBBs",   dual_anchor_nor_max_cp(oof_final, alpha=0.35, nor_w=0.50))
    _lit_eval("R21.07.DualAnchor(nw=0.50,a=0.25)→cSEBBs",   dual_anchor_nor_max_cp(oof_final, alpha=0.25, nor_w=0.50))
    _lit_eval("R21.08.DualAnchor(nw=0.50,a=0.40)→cSEBBs",   dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.50))
    _lit_eval("R21.09.DualAnchor(nw=0.50,T=0.08,a=0.30)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.50, entr_temp=0.08))
    _lit_eval("R21.10.DualAnchor(nw=0.50,r=2,a=0.30)→cSEBBs", dual_anchor_nor_max_r2_cp(oof_final, alpha=0.30, nor_w=0.50))
    _lit_eval("R21.11.DualAnchor(nw=0.40,r=2,a=0.30)→cSEBBs", dual_anchor_nor_max_r2_cp(oof_final, alpha=0.30, nor_w=0.40))
    _lit_eval("R21.12.DualAnchor(nw=0.50,b=5,a=0.30)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.50, beta=5.0))
    # Triple anchor: NOR + GlobalMax + AvgTopK(k=2)
    _lit_eval("R21.13.TripleAnchor(NOR+Max+ATK,a=0.30)→cSEBBs", triple_anchor_cp(oof_final, alpha=0.30))
    _lit_eval("R21.14.TripleAnchor(a=0.35)→cSEBBs",           triple_anchor_cp(oof_final, alpha=0.35))
    # NoisyOR + AvgTopK dual (no GlobalMax)
    _lit_eval("R21.15.NOR+AvgTopK(k=2,a=0.30)→cSEBBs",       dual_anchor_nor_topk_cp(oof_final, alpha=0.30, k=2))
    _lit_eval("R21.16.DualAnchor(nw=0.40,a=0.35)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.35, nor_w=0.40))
    _lit_eval("R21.17.DualAnchor(nw=0.30,a=0.35)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.35, nor_w=0.30))
    _lit_eval("R21.18.DualAnchor(nw=0.50,thr=0.07,a=0.30)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.30, nor_w=0.50, cp_thr=0.07))

    # ── Round 22: New paper methods + DualAnchor fine-tune at a=0.35,nw=0.40 ────
    # Key insight R21: nw=0.40,a=0.35 = 0.8036 NEW BEST.
    # Alpha still rising (0.30→0.35); nw optimum ~0.35-0.45.
    # New methods from literature: LAE anchor (arXiv:2111.01742),
    #   PCR-adaptive alpha (arXiv:2505.11889), PoE geometric fusion,
    #   G-TLA logit adjustment (ECCV 2024), TopK-Entropy anchor (arXiv:2503.02422).
    print("\n── Round 22: LAE / PCR-adaptive / PoE / G-TLA / TopK-Entropy / fine-tune ──")
    # DualAnchor fine-tune around new optimum
    _lit_eval("R22.01.DualAnchor(nw=0.40,a=0.38)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40))
    _lit_eval("R22.02.DualAnchor(nw=0.40,a=0.40)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.40))
    _lit_eval("R22.03.DualAnchor(nw=0.38,a=0.35)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.35, nor_w=0.38))
    _lit_eval("R22.04.DualAnchor(nw=0.42,a=0.35)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.35, nor_w=0.42))
    _lit_eval("R22.05.DualAnchor(nw=0.35,a=0.35)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.35, nor_w=0.35))
    _lit_eval("R22.06.DualAnchor(nw=0.40,a=0.45)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.45, nor_w=0.40))
    # LAE anchor: LogAvgExp β sweep as file anchor (arXiv:2111.01742)
    _lit_eval("R22.07.LAE(β=1.0,a=0.35)→cSEBBs",             lae_anchor_cp(oof_final, lae_beta=1.0, alpha=0.35))
    _lit_eval("R22.08.LAE(β=2.0,a=0.35)→cSEBBs",             lae_anchor_cp(oof_final, lae_beta=2.0, alpha=0.35))
    _lit_eval("R22.09.LAE(β=3.0,a=0.35)→cSEBBs",             lae_anchor_cp(oof_final, lae_beta=3.0, alpha=0.35))
    _lit_eval("R22.10.LAE(β=0.5,a=0.35)→cSEBBs",             lae_anchor_cp(oof_final, lae_beta=0.5, alpha=0.35))
    # PoE anchor: geometric mean of NoisyOR + GlobalMax
    _lit_eval("R22.11.PoE(nw=0.5,a=0.35)→cSEBBs",            poe_anchor_cp(oof_final, alpha=0.35, nor_w=0.5))
    _lit_eval("R22.12.PoE(nw=0.6,a=0.35)→cSEBBs",            poe_anchor_cp(oof_final, alpha=0.35, nor_w=0.6))
    _lit_eval("R22.13.PoE(nw=0.4,a=0.35)→cSEBBs",            poe_anchor_cp(oof_final, alpha=0.35, nor_w=0.4))
    # PCR-adaptive alpha (nSEBBs 2025): per-class α based on posterior contrast
    _lit_eval("R22.14.PCR-Adaptive(nw=0.40,amax=0.45)→cSEBBs", pcr_adaptive_dual_anchor_cp(oof_final, alpha_min=0.10, alpha_max=0.45, nor_w=0.40))
    _lit_eval("R22.15.PCR-Adaptive(nw=0.40,amax=0.50)→cSEBBs", pcr_adaptive_dual_anchor_cp(oof_final, alpha_min=0.15, alpha_max=0.50, nor_w=0.40))
    # G-TLA: logit adjustment before sigmoid (ECCV 2024)
    _lit_eval("R22.16.GTLA(τ=0.3,nw=0.40,a=0.35)→cSEBBs",    gtla_dual_anchor_cp(oof_final, tau=0.3, alpha=0.35, nor_w=0.40))
    _lit_eval("R22.17.GTLA(τ=0.5,nw=0.40,a=0.35)→cSEBBs",    gtla_dual_anchor_cp(oof_final, tau=0.5, alpha=0.35, nor_w=0.40))
    _lit_eval("R22.18.GTLA(τ=1.0,nw=0.40,a=0.35)→cSEBBs",    gtla_dual_anchor_cp(oof_final, tau=1.0, alpha=0.35, nor_w=0.40))
    # TopK-Entropy anchor: mean of k highest-entropy clips as anchor
    _lit_eval("R22.19.TopKEntr(k=5,a=0.35)→cSEBBs",           topk_entr_anchor_cp(oof_final, k=5, alpha=0.35))
    _lit_eval("R22.20.TopKEntr(k=3,a=0.35)→cSEBBs",           topk_entr_anchor_cp(oof_final, k=3, alpha=0.35))

    # ── Round 23: alpha converge at nw=0.40 / PCR-Adaptive fine / PoE + r=2 ────
    # Key insight R22: alpha plateauing at 0.38-0.40 (nw=0.40, a=0.38 & a=0.40 tied at 0.8038).
    # PCR-Adaptive works (0.8035) — try with a_max=0.55-0.60 for more range.
    # LAE/G-TLA/TopK-Entropy confirmed weak. Focus: alpha 0.39-0.43 at nw=0.40,
    # broader PCR range, and nw fine at a=0.40.
    print("\n── Round 23: alpha 0.38-0.43 / nw fine at a=0.40 / PCR range ──")
    _lit_eval("R23.01.DualAnchor(nw=0.40,a=0.39)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.39, nor_w=0.40))
    _lit_eval("R23.02.DualAnchor(nw=0.40,a=0.41)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.41, nor_w=0.40))
    _lit_eval("R23.03.DualAnchor(nw=0.40,a=0.42)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.42, nor_w=0.40))
    _lit_eval("R23.04.DualAnchor(nw=0.40,a=0.43)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.43, nor_w=0.40))
    _lit_eval("R23.05.DualAnchor(nw=0.40,a=0.50)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.50, nor_w=0.40))
    _lit_eval("R23.06.DualAnchor(nw=0.38,a=0.40)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.38))
    _lit_eval("R23.07.DualAnchor(nw=0.42,a=0.40)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.42))
    _lit_eval("R23.08.DualAnchor(nw=0.35,a=0.40)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.35))
    _lit_eval("R23.09.DualAnchor(nw=0.45,a=0.40)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.45))
    _lit_eval("R23.10.DualAnchor(nw=0.30,a=0.40)→cSEBBs",    dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30))
    _lit_eval("R23.11.PCR-Adaptive(nw=0.40,amax=0.55)→cSEBBs", pcr_adaptive_dual_anchor_cp(oof_final, alpha_min=0.15, alpha_max=0.55, nor_w=0.40))
    _lit_eval("R23.12.PCR-Adaptive(nw=0.40,amax=0.60)→cSEBBs", pcr_adaptive_dual_anchor_cp(oof_final, alpha_min=0.20, alpha_max=0.60, nor_w=0.40))
    _lit_eval("R23.13.PCR-Adaptive(nw=0.38,amax=0.50)→cSEBBs", pcr_adaptive_dual_anchor_cp(oof_final, alpha_min=0.15, alpha_max=0.50, nor_w=0.38))
    _lit_eval("R23.14.DualAnchor(nw=0.40,a=0.38,r=2)→cSEBBs", dual_anchor_nor_max_r2_cp(oof_final, alpha=0.38, nor_w=0.40))
    _lit_eval("R23.15.DualAnchor(nw=0.40,a=0.40,r=2)→cSEBBs", dual_anchor_nor_max_r2_cp(oof_final, alpha=0.40, nor_w=0.40))
    _lit_eval("R23.16.DualAnchor(nw=0.40,a=0.38,b=5)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.0))
    _lit_eval("R23.17.DualAnchor(nw=0.40,a=0.40,thr=0.07)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.40, cp_thr=0.07))
    _lit_eval("R23.18.DualAnchor(nw=0.40,a=0.38,T=0.08)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, entr_temp=0.08))

    # ── Round 18: a=0.30 combos / r=2+a / a=0.35-0.45 sweep / T at w=1.0 ──────
    # Key insight R17: a=0.30 >> a=0.175 at w=1.0; r=2 ties a=0.30 — combine them!
    # Alpha trend monotonically increases up to 0.30 — likely continues to ~0.35.
    # Also try r=2+a=0.30, r=2+a=0.25, and higher alpha 0.35-0.45.
    print("\n── Round 18: alpha>0.30 sweep / r=2+alpha combos / cSEBBs fine-tune ──")
    _lit_eval("R18.01.MeanMax(T=0.2,w=1.0,a=0.35)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.35))
    _lit_eval("R18.02.MeanMax(T=0.2,w=1.0,a=0.40)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.40))
    _lit_eval("R18.03.MeanMax(T=0.2,w=1.0,a=0.45)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.45))
    _lit_eval("R18.04.MeanMax(T=0.2,w=1.0,a=0.50)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.50))
    _lit_eval("R18.05.MeanMax(T=0.2,w=1.0,r=2,a=0.30)→cSEBBs", mean_max_full_r2(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.30))
    _lit_eval("R18.06.MeanMax(T=0.2,w=1.0,r=2,a=0.25)→cSEBBs", mean_max_full_r2(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.25))
    _lit_eval("R18.07.MeanMax(T=0.2,w=1.0,r=2,a=0.35)→cSEBBs", mean_max_full_r2(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.35))
    _lit_eval("R18.08.MeanMax(T=0.2,w=1.0,r=2,a=0.20)→cSEBBs", mean_max_full_r2(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.20))
    _lit_eval("R18.09.MeanMax(T=0.15,w=1.0,a=0.30)→cSEBBs", mean_max_full(oof_final, entr_temp=0.15, max_w=1.0, gm_alpha=0.30))
    _lit_eval("R18.10.MeanMax(T=0.1,w=1.0,a=0.30)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.10, max_w=1.0, gm_alpha=0.30))
    _lit_eval("R18.11.MeanMax(T=0.2,w=1.0,a=0.30,thr=0.07)→cSEBBs", mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.30, cp_thr=0.07))
    _lit_eval("R18.12.MeanMax(T=0.2,w=1.0,a=0.30,thr=0.05)→cSEBBs", mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.30, cp_thr=0.05))
    _lit_eval("R18.13.MeanMax(T=0.2,w=1.0,a=0.30,b=5.0)→cSEBBs", mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.30, beta=5.0))
    _lit_eval("R18.14.MeanMax(T=0.2,w=1.0,a=0.30,b=4.0)→cSEBBs", mean_max_full(oof_final, entr_temp=0.20, max_w=1.0, gm_alpha=0.30, beta=4.0))
    _lit_eval("R18.15.MeanMax(T=0.2,w=0.9,a=0.30)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=0.9, gm_alpha=0.30))
    _lit_eval("R18.16.MeanMax(T=0.2,w=0.8,a=0.30)→cSEBBs",  mean_max_full(oof_final, entr_temp=0.20, max_w=0.8, gm_alpha=0.30))

    # ── Round 24: beta fine-tune / nw=0.30 alpha sweep / r=2+beta / thr/T combos ──
    # Key insights R23:
    #   - b=5.0 > b=4.5 at (nw=0.40, a=0.38) → sweep b=4.7-6.0
    #   - nw=0.30 + a=0.40 = 0.8042 (ties best) → higher alpha (0.42-0.48)
    #   - nw=0.35 + a=0.40 = 0.8039 → try a=0.42 there too
    #   - r=2 underperformed alone at a=0.38/0.40 — try r=2 + b=5
    print("\n── Round 24: beta sweep / nw=0.30 alpha / r=2+beta / thr / T combos ──")
    # Branch A: beta fine-tune at (nw=0.40, a=0.38) — current best branch
    _lit_eval("R24.01.DualAnchor(nw=0.40,a=0.38,b=4.7)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=4.7))
    _lit_eval("R24.02.DualAnchor(nw=0.40,a=0.38,b=4.9)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=4.9))
    _lit_eval("R24.03.DualAnchor(nw=0.40,a=0.38,b=5.2)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.2))
    _lit_eval("R24.04.DualAnchor(nw=0.40,a=0.38,b=5.5)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.5))
    _lit_eval("R24.05.DualAnchor(nw=0.40,a=0.38,b=6.0)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=6.0))
    # Branch B: beta=5 at nw=0.30 branch (also 0.8042 at b=4.5)
    _lit_eval("R24.06.DualAnchor(nw=0.30,a=0.40,b=5.0)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, beta=5.0))
    _lit_eval("R24.07.DualAnchor(nw=0.30,a=0.40,b=5.5)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, beta=5.5))
    # nw=0.30 alpha fine-tune: does higher alpha help? (nw=0.30 peaked at a=0.40 in R23)
    _lit_eval("R24.08.DualAnchor(nw=0.30,a=0.42)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.42, nor_w=0.30))
    _lit_eval("R24.09.DualAnchor(nw=0.30,a=0.44)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.44, nor_w=0.30))
    _lit_eval("R24.10.DualAnchor(nw=0.30,a=0.46)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.46, nor_w=0.30))
    # nw=0.30, b=5, alpha sweep
    _lit_eval("R24.11.DualAnchor(nw=0.30,a=0.42,b=5)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.42, nor_w=0.30, beta=5.0))
    _lit_eval("R24.12.DualAnchor(nw=0.30,a=0.44,b=5)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.44, nor_w=0.30, beta=5.0))
    # nw=0.35 alpha fine-tune (0.8039 at a=0.40)
    _lit_eval("R24.13.DualAnchor(nw=0.35,a=0.42)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.42, nor_w=0.35))
    _lit_eval("R24.14.DualAnchor(nw=0.35,a=0.42,b=5)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.42, nor_w=0.35, beta=5.0))
    # r=2 + beta=5: r=2 hurt alone but may interact with b=5
    _lit_eval("R24.15.DualAnchor(nw=0.40,a=0.38,r=2,b=5)→cSEBBs", dual_anchor_nor_max_r2_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.0))
    _lit_eval("R24.16.DualAnchor(nw=0.30,a=0.40,r=2,b=5)→cSEBBs", dual_anchor_nor_max_r2_cp(oof_final, alpha=0.40, nor_w=0.30, beta=5.0))
    # cp_thr fine-tune at best (nw=0.40, a=0.38, b=5)
    _lit_eval("R24.17.DualAnchor(nw=0.40,a=0.38,b=5,thr=0.04)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.0, cp_thr=0.04))
    _lit_eval("R24.18.DualAnchor(nw=0.40,a=0.38,b=5,thr=0.05)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.0, cp_thr=0.05))
    _lit_eval("R24.19.DualAnchor(nw=0.40,a=0.38,b=5,thr=0.07)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.0, cp_thr=0.07))
    _lit_eval("R24.20.DualAnchor(nw=0.40,a=0.38,b=5,thr=0.08)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.0, cp_thr=0.08))
    # T fine-tune at nw=0.30, a=0.40 (T=0.1 is default)
    _lit_eval("R24.21.DualAnchor(nw=0.30,a=0.40,T=0.08)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, entr_temp=0.08))
    _lit_eval("R24.22.DualAnchor(nw=0.30,a=0.40,T=0.12)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, entr_temp=0.12))
    _lit_eval("R24.23.DualAnchor(nw=0.30,a=0.40,T=0.15)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, entr_temp=0.15))
    # nw=0.40, a=0.38, b=5 + T variants
    _lit_eval("R24.24.DualAnchor(nw=0.40,a=0.38,b=5,T=0.08)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.0, entr_temp=0.08))
    _lit_eval("R24.25.DualAnchor(nw=0.40,a=0.38,b=5,T=0.12)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.0, entr_temp=0.12))

    # ── Round 25: beta fine-tune both branches / nw=0.35 beta sweep / alpha cross ──
    # Key insights R24:
    #   Branch A (nw=0.40, a=0.38): b=4.9(0.8043)<b=5.2(0.8044)>b=5.5(0.8042) — peak ~5.2
    #   Branch B (nw=0.30, a=0.40): b=5.0(0.8040)<b=5.5(0.8044) — still rising!
    #   nw=0.35, a=0.40 = 0.8039@b=4.5 — unexplored with higher beta
    #   thr=0.06 confirmed optimal; r=2 consistently hurts; T=0.10 optimal
    print("\n── Round 25: beta fine-tune both branches / nw=0.35 sweep / cross combos ──")
    # Branch A fine-tune: b=5.1-5.3 at (nw=0.40, a=0.38)
    _lit_eval("R25.01.DualAnchor(nw=0.40,a=0.38,b=5.1)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.1))
    _lit_eval("R25.02.DualAnchor(nw=0.40,a=0.38,b=5.15)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15))
    _lit_eval("R25.03.DualAnchor(nw=0.40,a=0.38,b=5.25)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.25))
    _lit_eval("R25.04.DualAnchor(nw=0.40,a=0.38,b=5.3)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.3))
    # Branch B beta push: b=6.0-7.0 at (nw=0.30, a=0.40) — still rising at b=5.5
    _lit_eval("R25.05.DualAnchor(nw=0.30,a=0.40,b=6.0)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0))
    _lit_eval("R25.06.DualAnchor(nw=0.30,a=0.40,b=6.5)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.5))
    _lit_eval("R25.07.DualAnchor(nw=0.30,a=0.40,b=7.0)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, beta=7.0))
    _lit_eval("R25.08.DualAnchor(nw=0.30,a=0.40,b=5.8)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, beta=5.8))
    # nw=0.35 beta sweep (peaked at b=4.5/0.8039 before)
    _lit_eval("R25.09.DualAnchor(nw=0.35,a=0.40,b=5.0)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.35, beta=5.0))
    _lit_eval("R25.10.DualAnchor(nw=0.35,a=0.40,b=5.2)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.35, beta=5.2))
    _lit_eval("R25.11.DualAnchor(nw=0.35,a=0.40,b=5.5)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.35, beta=5.5))
    _lit_eval("R25.12.DualAnchor(nw=0.35,a=0.42,b=5.2)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.42, nor_w=0.35, beta=5.2))
    # nw=0.40 alpha cross: a=0.39/0.40 with b=5.2 (best A was a=0.38)
    _lit_eval("R25.13.DualAnchor(nw=0.40,a=0.39,b=5.2)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.39, nor_w=0.40, beta=5.2))
    _lit_eval("R25.14.DualAnchor(nw=0.40,a=0.40,b=5.2)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.40, beta=5.2))
    _lit_eval("R25.15.DualAnchor(nw=0.40,a=0.37,b=5.2)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.37, nor_w=0.40, beta=5.2))
    # nw=0.25 — even lower NOR weight, compensate with higher beta
    _lit_eval("R25.16.DualAnchor(nw=0.25,a=0.40,b=5.5)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.25, beta=5.5))
    _lit_eval("R25.17.DualAnchor(nw=0.25,a=0.40,b=6.0)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.25, beta=6.0))
    # Best B + thr combo (thr=0.05 gave 0.8039 at b=5.0 — try at b=5.5)
    _lit_eval("R25.18.DualAnchor(nw=0.30,a=0.40,b=5.5,thr=0.05)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, beta=5.5, cp_thr=0.05))
    # b=5.2 cross at nw=0.30 (best A beta at B branch)
    _lit_eval("R25.19.DualAnchor(nw=0.30,a=0.40,b=5.2)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.30, beta=5.2))
    # nw=0.42 beta sweep
    _lit_eval("R25.20.DualAnchor(nw=0.42,a=0.40,b=5.2)→cSEBBs", dual_anchor_nor_max_cp(oof_final, alpha=0.40, nor_w=0.42, beta=5.2))

    # ── Round 26: structural breakthrough attempts — plateau at 0.8044 ──────────
    # R25 verdict: 5+ configs all = 0.8044 regardless of nw/beta/branch.
    # Trying fundamentally different blend architectures to break through.
    # Branch A best: nw=0.40, a=0.38, b=5.15
    # Branch B best: nw=0.30, a=0.40, b=6.0
    print("\n── Round 26: ensemble / logit-blend / geom-blend / double-pass / multi-beta ──")
    # Ensemble of Branch A + Branch B (50/50 average in prob space)
    _lit_eval("R26.01.BranchEns(A+B,w=0.5)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.5))
    _lit_eval("R26.02.BranchEns(A+B,w=0.6)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.6))
    _lit_eval("R26.03.BranchEns(A+B,w=0.4)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.4))
    # Logit-space anchor blending (multiplicative in odds space)
    _lit_eval("R26.04.LogitBlend(nw=0.40,a=0.38,b=5.15)→cSEBBs",
              dual_anchor_logit_blend_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15))
    _lit_eval("R26.05.LogitBlend(nw=0.30,a=0.40,b=6.0)→cSEBBs",
              dual_anchor_logit_blend_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0))
    _lit_eval("R26.06.LogitBlend(nw=0.40,a=0.50,b=5.15)→cSEBBs",
              dual_anchor_logit_blend_cp(oof_final, alpha=0.50, nor_w=0.40, beta=5.15))
    _lit_eval("R26.07.LogitBlend(nw=0.30,a=0.50,b=6.0)→cSEBBs",
              dual_anchor_logit_blend_cp(oof_final, alpha=0.50, nor_w=0.30, beta=6.0))
    # Geometric mean blend (lse^(1-a) * anchor^a)
    _lit_eval("R26.08.GeomBlend(nw=0.40,a=0.38,b=5.15)→cSEBBs",
              dual_anchor_geom_blend_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15))
    _lit_eval("R26.09.GeomBlend(nw=0.30,a=0.40,b=6.0)→cSEBBs",
              dual_anchor_geom_blend_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0))
    _lit_eval("R26.10.GeomBlend(nw=0.40,a=0.50,b=5.15)→cSEBBs",
              dual_anchor_geom_blend_cp(oof_final, alpha=0.50, nor_w=0.40, beta=5.15))
    # Double-pass DualAnchor (apply twice with weaker second pass)
    _lit_eval("R26.11.DoublePass(a1=0.38,a2=0.15,nw=0.40,b=5.15)→cSEBBs",
              dual_anchor_double_pass_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, alpha2=0.15))
    _lit_eval("R26.12.DoublePass(a1=0.38,a2=0.20,nw=0.40,b=5.15)→cSEBBs",
              dual_anchor_double_pass_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, alpha2=0.20))
    _lit_eval("R26.13.DoublePass(a1=0.38,a2=0.10,nw=0.40,b=5.15)→cSEBBs",
              dual_anchor_double_pass_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, alpha2=0.10))
    # Multi-beta blend (average 3 beta levels before DualAnchor)
    _lit_eval("R26.14.MultiBeta(b=[4.5,5.15,6.0],nw=0.40,a=0.38)→cSEBBs",
              dual_anchor_multi_beta_cp(oof_final, alpha=0.38, nor_w=0.40,
                                        betas=(4.5, 5.15, 6.0), beta_weights=(0.33, 0.34, 0.33)))
    _lit_eval("R26.15.MultiBeta(b=[5.0,5.15,5.3],nw=0.40,a=0.38)→cSEBBs",
              dual_anchor_multi_beta_cp(oof_final, alpha=0.38, nor_w=0.40,
                                        betas=(5.0, 5.15, 5.3), beta_weights=(0.33, 0.34, 0.33)))
    _lit_eval("R26.16.MultiBeta(b=[5.5,6.0,6.5],nw=0.30,a=0.40)→cSEBBs",
              dual_anchor_multi_beta_cp(oof_final, alpha=0.40, nor_w=0.30,
                                        betas=(5.5, 6.0, 6.5), beta_weights=(0.33, 0.34, 0.33)))
    # Branch ensemble with logit averaging (instead of prob averaging)
    _lit_eval("R26.17.BranchEnsLogit(A+B,w=0.5)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.5,
                                              beta_a=5.15, beta_b=6.5))
    # Higher alpha logit blend (logit-space may support higher alpha without ceiling)
    _lit_eval("R26.18.LogitBlend(nw=0.40,a=0.60,b=5.15)→cSEBBs",
              dual_anchor_logit_blend_cp(oof_final, alpha=0.60, nor_w=0.40, beta=5.15))
    _lit_eval("R26.19.LogitBlend(nw=0.35,a=0.50,b=5.5)→cSEBBs",
              dual_anchor_logit_blend_cp(oof_final, alpha=0.50, nor_w=0.35, beta=5.5))
    _lit_eval("R26.20.MultiBeta(b=[4.5,5.15,6.0],nw=0.30,a=0.40)→cSEBBs",
              dual_anchor_multi_beta_cp(oof_final, alpha=0.40, nor_w=0.30,
                                        betas=(4.5, 5.15, 6.0), beta_weights=(0.33, 0.34, 0.33)))

    # ── Round 27: branch ensemble fine-tune / 3-way / quantile-mix / velocity ──
    # R26 verdict: BranchEns(w=0.5)=0.8045 NEW BEST; LogitBlend/GeomBlend HURT;
    #              DoublePass hurts (-0.0016); MultiBeta neutral (0.8043-0.8044).
    # New papers: Quantile-Mix (BirdCLEF 2025 top-2%, +0.038), Min-reduction
    #             (BirdCLEF 2024 1st), Velocity-Attention (arxiv:2504.12670).
    print("\n── Round 27: ensemble fine-tune / 3-way / quantile-mix / velocity-attn ──")
    # Branch A: nw=0.40,a=0.38,b=5.15  Branch B: nw=0.30,a=0.40,b=6.0
    # Fine-tune ensemble weight (w=0.5 was best, try nearby)
    _lit_eval("R27.01.BranchEns(A+B,w=0.45)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.45))
    _lit_eval("R27.02.BranchEns(A+B,w=0.48)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.48))
    _lit_eval("R27.03.BranchEns(A+B,w=0.52)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.52))
    _lit_eval("R27.04.BranchEns(A+B,w=0.55)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.55))
    # Try different beta pairs in the ensemble
    _lit_eval("R27.05.BranchEns(b5.1+b6.5,w=0.5)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.5, beta_a=5.1, beta_b=6.5))
    _lit_eval("R27.06.BranchEns(b5.2+b5.8,w=0.5)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.5, beta_a=5.2, beta_b=5.8))
    _lit_eval("R27.07.BranchEns(b5.15+b7.0,w=0.5)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.5, beta_a=5.15, beta_b=7.0))
    # 3-way ensemble: A + B + C(nw=0.35,a=0.40,b=5.5)
    _lit_eval("R27.08.TripleEns(A+B+C,eq)→cSEBBs",
              triple_branch_ensemble_cp(oof_final, w_a=0.333, w_b=0.333))
    _lit_eval("R27.09.TripleEns(A+B+C,w=0.4/0.4/0.2)→cSEBBs",
              triple_branch_ensemble_cp(oof_final, w_a=0.40, w_b=0.40))
    _lit_eval("R27.10.TripleEns(A+B+C,w=0.35/0.35/0.3)→cSEBBs",
              triple_branch_ensemble_cp(oof_final, w_a=0.35, w_b=0.35))
    # Min-reduction ensemble (BirdCLEF 2024 1st: conservative anti-FP)
    _lit_eval("R27.11.MinEns(A+B)→cSEBBs",
              branch_ensemble_min_cp(oof_final))
    # NOTE: QuantileMix SKIPPED — rank-norm destroys info with only 12 windows (tested, gave 0.681)
    # Velocity-Attention enhanced DualAnchor (TAP, arxiv:2504.12670) — fixed broadcast
    _lit_eval("R27.12.VelAttn(vw=0.3,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              velocity_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, vel_w=0.3))
    _lit_eval("R27.13.VelAttn(vw=0.5,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              velocity_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, vel_w=0.5))
    _lit_eval("R27.14.VelAttn(vw=0.7,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              velocity_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, vel_w=0.7))
    # VelAttn on Branch B
    _lit_eval("R27.15.VelAttn(vw=0.5,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              velocity_dual_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0, vel_w=0.5))
    # BranchEns with cSEBBs thr fine-tune
    _lit_eval("R27.16.BranchEns(w=0.5,thr=0.05)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.5, cp_thr=0.05))
    _lit_eval("R27.17.BranchEns(w=0.5,thr=0.07)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.5, cp_thr=0.07))
    # BranchEns with slightly different betas (fine-tune)
    _lit_eval("R27.18.BranchEns(A+B,w=0.5,bA=5.175)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.5, beta_a=5.175, beta_b=6.0))
    _lit_eval("R27.19.BranchEns(A+B,w=0.5,bB=6.25)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.5, beta_a=5.15, beta_b=6.25))
    # VelAttn + BranchEns: velocity weighting then ensemble
    _lit_eval("R27.20.VelEns(vw=0.3,A+B,w=0.5)→cSEBBs",
              dual_anchor_branch_ensemble_cp(oof_final, w_a=0.5,
                                              alpha_a=0.38, nw_a=0.40, beta_a=5.15,
                                              alpha_b=0.40, nw_b=0.30, beta_b=6.0))

    # ── Round 28: power mean / geom anchor / topk anchor / post-refinement ──────
    # R27 verdict: VelAttn CATASTROPHIC (0.72-0.74), MinEns=0.8044, TripleEns neutral.
    #              Plateau at 0.8045 with 18 methods. Fold 0 outlier (0.6913).
    # R28: genuinely different pooling/anchor/post-processing operations.
    # Strategy: asymmetric branches, new anchor aggregation, post-ensemble refinement.
    print("\n── Round 28: power-mean / geom-anchor / topk / post-LSE / double-cSEBBs ──")
    # Area 1: Power mean pooling (prob-space, radius=1) — asymmetric to LSE
    _lit_eval("R28.01.PowerMean(p=3,nw=0.40,a=0.38)→cSEBBs",
              power_mean_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, power=3.0))
    _lit_eval("R28.02.PowerMean(p=4,nw=0.40,a=0.38)→cSEBBs",
              power_mean_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, power=4.0))
    _lit_eval("R28.03.PowerMean(p=5,nw=0.30,a=0.40)→cSEBBs",
              power_mean_dual_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, power=5.0))
    # Area 2: Asymmetric branch ensemble (PowerMean A + LSE B)
    _lit_eval("R28.04.PowerLSEEns(p=3+b=6.0,w=0.5)→cSEBBs",
              power_lse_branch_ensemble_cp(oof_final, power_a=3.0, beta_b=6.0, w_a=0.5))
    _lit_eval("R28.05.PowerLSEEns(p=4+b=6.0,w=0.5)→cSEBBs",
              power_lse_branch_ensemble_cp(oof_final, power_a=4.0, beta_b=6.0, w_a=0.5))
    _lit_eval("R28.06.PowerLSEEns(p=3+b=5.15,w=0.5)→cSEBBs",
              power_lse_branch_ensemble_cp(oof_final, power_a=3.0, beta_b=5.15, w_a=0.5,
                                            alpha_b=0.38, nw_b=0.40))
    # Area 3: Geometric mean anchor — sqrt(NOR * GlobalMax), no tunable nor_w
    _lit_eval("R28.07.GeomAnchor(a=0.38,b=5.15)→cSEBBs",
              dual_anchor_geom_anchor_cp(oof_final, alpha=0.38, beta=5.15))
    _lit_eval("R28.08.GeomAnchor(a=0.40,b=6.0)→cSEBBs",
              dual_anchor_geom_anchor_cp(oof_final, alpha=0.40, beta=6.0))
    _lit_eval("R28.09.GeomAnchor(a=0.45,b=5.15)→cSEBBs",
              dual_anchor_geom_anchor_cp(oof_final, alpha=0.45, beta=5.15))
    # Area 4: TopK anchor — mean of top-k clips instead of GlobalMax
    _lit_eval("R28.10.TopK(k=2,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              dual_anchor_topk_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, k=2))
    _lit_eval("R28.11.TopK(k=3,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              dual_anchor_topk_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, k=3))
    _lit_eval("R28.12.TopK(k=2,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              dual_anchor_topk_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0, k=2))
    # Area 5: Post-ensemble LSE refinement (second gentle smoothing pass)
    _lit_eval("R28.13.BranchEns→LSE(b=2.0)→cSEBBs",
              branch_ensemble_post_lse_cp(oof_final, post_beta=2.0))
    _lit_eval("R28.14.BranchEns→LSE(b=3.0)→cSEBBs",
              branch_ensemble_post_lse_cp(oof_final, post_beta=3.0))
    _lit_eval("R28.15.BranchEns→LSE(b=4.5)→cSEBBs",
              branch_ensemble_post_lse_cp(oof_final, post_beta=4.5))
    # Area 6: Post-cSEBBs GlobalMean blend (gentle file-level correction)
    _lit_eval("R28.16.BranchEns→cSEBBs→GM(a=0.10)→logit",
              branch_ensemble_post_gm_cp(oof_final, gm_alpha=0.10))
    _lit_eval("R28.17.BranchEns→cSEBBs→GM(a=0.15)→logit",
              branch_ensemble_post_gm_cp(oof_final, gm_alpha=0.15))
    _lit_eval("R28.18.BranchEns→cSEBBs→GM(a=0.20)→logit",
              branch_ensemble_post_gm_cp(oof_final, gm_alpha=0.20))
    # Area 7: Double-stage cSEBBs (coarse then fine)
    _lit_eval("R28.19.BranchEns→cP(0.06)→cP(0.03)→logit",
              branch_ensemble_double_cp(oof_final, cp_thr1=0.06, cp_thr2=0.03))
    _lit_eval("R28.20.BranchEns→cP(0.06)→cP(0.04)→logit",
              branch_ensemble_double_cp(oof_final, cp_thr1=0.06, cp_thr2=0.04))

    # ── Round 33: Geometric BranchEns / Global-Silence Cut-off ─────────────────────────────────
    # R32 verdict: PerClassAdaptiveMedian best=0.7931, MorphClose below that. Plateau at 0.8045.
    # R33 strategy: BirdCLEF 2024 competition solutions (4th place, zenn.dev/yuto_mo):
    #   1) Geometric Mean BranchEns: A^w * B^(1-w) instead of w*A+(1-w)*B — suppresses
    #      single-branch overconfidence; different from our GeomAnchor (R28) which blends LSE+anchor
    #   2) Global-Silence Cut-off: deflate classes where per-file max < threshold
    #      (BirdCLEF24 4th: "halved prob if no birdsong ≤0.10") — distinct from DualAnchor which
    #      boosts present classes; this deflates absent classes to reduce false positives
    #   3) Combinations: Geo+SilenceCut, ArithGeoBlend+SilenceCut
    print("\n── Round 33: GeometricBranchEns / GlobalSilenceCutoff ──")
    # Area 1: Geometric Mean BranchEns (A^w * B^(1-w) in prob space)
    _lit_eval("R33.01.GeoBranchEns(w=0.55)→cSEBBs",
              geometric_branch_ensemble_cp(oof_final, w_a=0.55))
    _lit_eval("R33.02.GeoBranchEns(w=0.50)→cSEBBs",
              geometric_branch_ensemble_cp(oof_final, w_a=0.50))
    _lit_eval("R33.03.GeoBranchEns(w=0.60)→cSEBBs",
              geometric_branch_ensemble_cp(oof_final, w_a=0.60))
    _lit_eval("R33.04.GeoBranchEns(w=0.55,b_a=5.5,b_b=6.5)→cSEBBs",
              geometric_branch_ensemble_cp(oof_final, w_a=0.55, beta_a=5.5, beta_b=6.5))
    # Area 1b: ArithGeo blend (partial geometric correction on arithmetic BranchEns)
    _lit_eval("R33.05.ArithGeo(gw=0.2,w=0.55)→cSEBBs",
              arith_geo_blend_cp(oof_final, geo_w=0.2, w_a=0.55))
    _lit_eval("R33.06.ArithGeo(gw=0.3,w=0.55)→cSEBBs",
              arith_geo_blend_cp(oof_final, geo_w=0.3, w_a=0.55))
    _lit_eval("R33.07.ArithGeo(gw=0.5,w=0.55)→cSEBBs",
              arith_geo_blend_cp(oof_final, geo_w=0.5, w_a=0.55))
    # Area 2: Global-Silence Cut-off (per-file deflation where max < thr)
    _lit_eval("R33.08.BranchEns→SilenceCut(thr=0.05,f=0.5)→cSEBBs",
              silence_cut_branchens_cp(oof_final, thr=0.05, cut_factor=0.5))
    _lit_eval("R33.09.BranchEns→SilenceCut(thr=0.10,f=0.5)→cSEBBs",
              silence_cut_branchens_cp(oof_final, thr=0.10, cut_factor=0.5))
    _lit_eval("R33.10.BranchEns→SilenceCut(thr=0.15,f=0.5)→cSEBBs",
              silence_cut_branchens_cp(oof_final, thr=0.15, cut_factor=0.5))
    _lit_eval("R33.11.BranchEns→SilenceCut(thr=0.10,f=0.25)→cSEBBs",
              silence_cut_branchens_cp(oof_final, thr=0.10, cut_factor=0.25))
    _lit_eval("R33.12.BranchEns→SilenceCut(thr=0.10,f=0.0)→cSEBBs",
              silence_cut_branchens_cp(oof_final, thr=0.10, cut_factor=0.0))
    _lit_eval("R33.13.BranchEns→SilenceCut(thr=0.05,f=0.25)→cSEBBs",
              silence_cut_branchens_cp(oof_final, thr=0.05, cut_factor=0.25))
    # Area 3: Geometric + SilenceCut
    _lit_eval("R33.14.GeoBranchEns→SilenceCut(thr=0.10,f=0.5)→cSEBBs",
              geo_silence_cut_cp(oof_final, thr=0.10, cut_factor=0.5, geo_w=1.0, w_a=0.55))
    _lit_eval("R33.15.ArithGeo(gw=0.3)→SilenceCut(thr=0.10,f=0.5)→cSEBBs",
              geo_silence_cut_cp(oof_final, thr=0.10, cut_factor=0.5, geo_w=0.3, w_a=0.55))
    _lit_eval("R33.16.GeoBranchEns→SilenceCut(thr=0.05,f=0.5)→cSEBBs",
              geo_silence_cut_cp(oof_final, thr=0.05, cut_factor=0.5, geo_w=1.0, w_a=0.55))
    _lit_eval("R33.17.BranchEns→SilenceCut(thr=0.08,f=0.5)→cSEBBs",
              silence_cut_branchens_cp(oof_final, thr=0.08, cut_factor=0.5))
    _lit_eval("R33.18.BranchEns→SilenceCut(thr=0.20,f=0.5)→cSEBBs",
              silence_cut_branchens_cp(oof_final, thr=0.20, cut_factor=0.5))
    _lit_eval("R33.19.GeoBranchEns→SilenceCut(thr=0.10,f=0.25)→cSEBBs",
              geo_silence_cut_cp(oof_final, thr=0.10, cut_factor=0.25, geo_w=1.0, w_a=0.55))
    _lit_eval("R33.20.ArithGeo(gw=0.1,w=0.55)→cSEBBs",
              arith_geo_blend_cp(oof_final, geo_w=0.1, w_a=0.55))

    # ── Round 40: Extreme gamma / g=5.0 thr+boost combos / SoftmaxRichness ─────────────────────────
    # R39 verdict: gamma monotone: g=2.5→0.8066, g=3.0→0.8069, g=4.0→0.8074, g=5.0→0.8078 BEST.
    # thr=0.5 better than 0.4 at g=2.0 (0.8074 vs 0.8062). thr=0.3 CATASTROPHIC (0.7972).
    # DualPow no gain. SilenceCut neutral with power-law. cp_thr=0.05 marginal.
    # R40 strategy:
    #   Area 1: Gamma extreme (7, 10, 15, 20) — limit of power-law = argmax selector
    #   Area 2: g=5.0 + thr=0.5/0.6 (thr=0.5 found better at g=2.0)
    #   Area 3: g=5.0 + boost=0.70/0.80 (more boost for the few rich files)
    #   Area 4: g=5.0 + lower base_nw (base=0.30, boost=0.70) for more dynamic range
    #   Area 5: SoftmaxRichness (different from power-law: translation-invariant)
    print("\n── Round 40: ExtremeGamma / g=5.0 thr+boost / g=5+base sweep / SoftmaxRichness ──")
    # Area 1: Push gamma extreme
    _lit_eval("R40.01.AdaptRichPow(g=7.0,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=7.0))
    _lit_eval("R40.02.AdaptRichPow(g=10.0,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=10.0))
    _lit_eval("R40.03.AdaptRichPow(g=15.0,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=15.0))
    _lit_eval("R40.04.AdaptRichPow(g=20.0,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=20.0))
    # Area 2: g=5.0 + thr sweep (thr=0.5 was better at g=2.0)
    _lit_eval("R40.05.AdaptRichPow(g=5.0,thr=0.5)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.5, max_boost=0.60, gamma=5.0))
    _lit_eval("R40.06.AdaptRichPow(g=5.0,thr=0.6)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.6, max_boost=0.60, gamma=5.0))
    _lit_eval("R40.07.AdaptRichPow(g=7.0,thr=0.5)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.5, max_boost=0.60, gamma=7.0))
    _lit_eval("R40.08.AdaptRichPow(g=10.0,thr=0.5)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.5, max_boost=0.60, gamma=10.0))
    # Area 3: g=5.0 + boost sweep
    _lit_eval("R40.09.AdaptRichPow(g=5.0,boost=0.70)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.70, gamma=5.0))
    _lit_eval("R40.10.AdaptRichPow(g=5.0,boost=0.80)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.80, gamma=5.0))
    _lit_eval("R40.11.AdaptRichPow(g=7.0,boost=0.70)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.70, gamma=7.0))
    # Area 4: Lower base_nw for more dynamic range
    _lit_eval("R40.12.AdaptRichPow(g=5.0,base=0.30,boost=0.70)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.30, richness_thr=0.4, max_boost=0.70, gamma=5.0))
    _lit_eval("R40.13.AdaptRichPow(g=5.0,base=0.25,boost=0.75)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.25, richness_thr=0.4, max_boost=0.75, gamma=5.0))
    _lit_eval("R40.14.AdaptRichPow(g=7.0,base=0.30,boost=0.70)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.30, richness_thr=0.4, max_boost=0.70, gamma=7.0))
    # Area 5: SoftmaxRichness (translation-invariant vs power-law scale-invariant)
    _lit_eval("R40.15.SoftmaxRich(T=0.10,thr=0.4)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.4, temp=0.10))
    _lit_eval("R40.16.SoftmaxRich(T=0.05,thr=0.4)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.4, temp=0.05))
    _lit_eval("R40.17.SoftmaxRich(T=0.20,thr=0.4)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.4, temp=0.20))
    _lit_eval("R40.18.SoftmaxRich(T=0.10,thr=0.5)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.10))
    # Best combos
    _lit_eval("R40.19.AdaptRichPow(g=5.0,thr=0.5,boost=0.70)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.5, max_boost=0.70, gamma=5.0))
    _lit_eval("R40.20.AdaptRichPow(g=7.0,thr=0.5,boost=0.70)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.5, max_boost=0.70, gamma=7.0))

    # ── Round 41: SoftmaxRichness deep-dive — thr/temp/boost/base/dual-branch sweep ─────────────
    # R40 verdict: SoftmaxRich(T=0.10,thr=0.5)=0.8113 MASSIVE BREAKTHROUGH!
    #   SoftmaxRich(T=0.10,thr=0.4)=0.8074 → thr=0.5 adds +0.0039 to softmax.
    #   T=0.05,thr=0.4→0.8059; T=0.20,thr=0.4→0.8067 → T=0.10 sweet-spot at thr=0.4.
    #   AdaptRichPow plateau: g=7.0≈g=10.0=0.8082, g=15.0→0.8074, g=20.0→0.8069.
    # R41 strategy: full SoftmaxRich parameter sweep
    print("\n── Round 41: SoftmaxRichness deep-dive sweep ──")
    # Area 1: thr sweep at T=0.10 — find true optimal threshold
    _lit_eval("R41.01.SoftmaxRich(T=0.10,thr=0.35)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.35, temp=0.10))
    _lit_eval("R41.02.SoftmaxRich(T=0.10,thr=0.45)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.45, temp=0.10))
    _lit_eval("R41.03.SoftmaxRich(T=0.10,thr=0.55)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.55, temp=0.10))
    _lit_eval("R41.04.SoftmaxRich(T=0.10,thr=0.6)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.60, temp=0.10))
    _lit_eval("R41.05.SoftmaxRich(T=0.10,thr=0.7)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.70, temp=0.10))
    # Area 2: temp sweep at thr=0.5 — sharp vs smooth
    _lit_eval("R41.06.SoftmaxRich(T=0.02,thr=0.5)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.02))
    _lit_eval("R41.07.SoftmaxRich(T=0.05,thr=0.5)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.05))
    _lit_eval("R41.08.SoftmaxRich(T=0.075,thr=0.5)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.075))
    _lit_eval("R41.09.SoftmaxRich(T=0.15,thr=0.5)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15))
    # Area 3: boost/base sweep at (T=0.10, thr=0.5)
    _lit_eval("R41.10.SoftmaxRich(T=0.10,thr=0.5,boost=0.70)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.10, max_boost=0.70))
    _lit_eval("R41.11.SoftmaxRich(T=0.10,thr=0.5,boost=0.80)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.10, max_boost=0.80))
    _lit_eval("R41.12.SoftmaxRich(T=0.10,thr=0.5,base=0.30,boost=0.70)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.10, base_nw=0.30, max_boost=0.70))
    _lit_eval("R41.13.SoftmaxRich(T=0.10,thr=0.5,base=0.25,boost=0.75)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.10, base_nw=0.25, max_boost=0.75))
    # Area 4: combined best params — lower T with boost at thr=0.5
    _lit_eval("R41.14.SoftmaxRich(T=0.05,thr=0.5,boost=0.70)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.05, max_boost=0.70))
    _lit_eval("R41.15.SoftmaxRich(T=0.075,thr=0.5,boost=0.70)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.075, max_boost=0.70))
    _lit_eval("R41.16.SoftmaxRich(T=0.05,thr=0.6)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.6, temp=0.05))
    # Area 5: SoftmaxRich DualBranch — both A and B use adaptive softmax nw
    _lit_eval("R41.17.SoftmaxRichDual(T=0.10,thr=0.5)→cSEBBs",
              softmax_anchor_richness_dual_cp(oof_final, richness_thr=0.5, temp=0.10))
    _lit_eval("R41.18.SoftmaxRichDual(T=0.05,thr=0.5)→cSEBBs",
              softmax_anchor_richness_dual_cp(oof_final, richness_thr=0.5, temp=0.05))
    # Area 6: cp_thr sweep at best config (T=0.10, thr=0.5)
    _lit_eval("R41.19.SoftmaxRich(T=0.10,thr=0.5,cp=0.05)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.10, cp_thr=0.05))
    _lit_eval("R41.20.SoftmaxRich(T=0.10,thr=0.5,cp=0.08)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.10, cp_thr=0.08))

    # ── Round 42: SoftmaxRichness temp push + T=0.15 param sweep ─────────────────────────────────
    # R41 verdict: T=0.15,thr=0.5→0.8122 NEW BEST. Temp trend monotone at thr=0.5.
    #   thr=0.5 is optimal at T=0.10 (bell curve: thr=0.35→0.8008...peak at 0.5...thr=0.7→0.8026).
    #   boost=0.60 optimal; DualBranch no gain; cp_thr=0.05 negligible.
    print("\n── Round 42: SoftmaxRichness temp push + T=0.15 param sweep ──")
    # Area 1: temp push beyond T=0.15 — find plateau
    _lit_eval("R42.01.SoftmaxRich(T=0.20,thr=0.5)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.20))
    _lit_eval("R42.02.SoftmaxRich(T=0.25,thr=0.5)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.25))
    _lit_eval("R42.03.SoftmaxRich(T=0.30,thr=0.5)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.30))
    _lit_eval("R42.04.SoftmaxRich(T=0.40,thr=0.5)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.40))
    _lit_eval("R42.05.SoftmaxRich(T=0.50,thr=0.5)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.50))
    # Area 2: T=0.15 + thr fine-tune — peak may shift at warmer T
    _lit_eval("R42.06.SoftmaxRich(T=0.15,thr=0.45)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.45, temp=0.15))
    _lit_eval("R42.07.SoftmaxRich(T=0.15,thr=0.48)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.48, temp=0.15))
    _lit_eval("R42.08.SoftmaxRich(T=0.15,thr=0.52)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.52, temp=0.15))
    _lit_eval("R42.09.SoftmaxRich(T=0.15,thr=0.55)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.55, temp=0.15))
    _lit_eval("R42.10.SoftmaxRich(T=0.15,thr=0.6)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.60, temp=0.15))
    # Area 3: T=0.15 + boost/base sweep
    _lit_eval("R42.11.SoftmaxRich(T=0.15,thr=0.5,boost=0.70)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, max_boost=0.70))
    _lit_eval("R42.12.SoftmaxRich(T=0.15,thr=0.5,boost=0.80)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, max_boost=0.80))
    _lit_eval("R42.13.SoftmaxRich(T=0.15,thr=0.5,base=0.30,boost=0.70)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.30, max_boost=0.70))
    _lit_eval("R42.14.SoftmaxRich(T=0.15,thr=0.5,base=0.25,boost=0.75)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.25, max_boost=0.75))
    # Area 4: DualBranch at T=0.15
    _lit_eval("R42.15.SoftmaxRichDual(T=0.15,thr=0.5)→cSEBBs",
              softmax_anchor_richness_dual_cp(oof_final, richness_thr=0.5, temp=0.15))
    _lit_eval("R42.16.SoftmaxRichDual(T=0.20,thr=0.5)→cSEBBs",
              softmax_anchor_richness_dual_cp(oof_final, richness_thr=0.5, temp=0.20))
    # Area 5: Cross sweep — T=0.20 × thr/boost
    _lit_eval("R42.17.SoftmaxRich(T=0.20,thr=0.48)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.48, temp=0.20))
    _lit_eval("R42.18.SoftmaxRich(T=0.20,thr=0.5,boost=0.70)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.20, max_boost=0.70))
    _lit_eval("R42.19.SoftmaxRich(T=0.25,thr=0.48)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.48, temp=0.25))
    _lit_eval("R42.20.SoftmaxRich(T=0.15,thr=0.5,cp=0.05)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, cp_thr=0.05))

    # ── Round 43: SoftmaxRich base sweep + adaptive alpha ────────────────────────────────────────
    # R42 verdict: R42.14 SoftmaxRich(T=0.15,thr=0.5,base=0.25,boost=0.75)=0.8136 NEW BEST!
    #   T=0.15 is confirmed peak. thr=0.50 confirmed optimal. Lower base > higher base.
    #   Boost trend: base=0.40,boost=0.80(nw_max=1.20)=0.8135 ≈ base=0.25,boost=0.75(nw_max=1.0)=0.8136.
    print("\n── Round 43: SoftmaxRich base sweep + adaptive alpha ──")
    # Area 1: Lower base at nw_max=1.0 — maximum NOR/max contrast for poor files
    _lit_eval("R43.01.SoftmaxRich(T=0.15,base=0.20,boost=0.80)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80))
    _lit_eval("R43.02.SoftmaxRich(T=0.15,base=0.15,boost=0.85)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.15, max_boost=0.85))
    _lit_eval("R43.03.SoftmaxRich(T=0.15,base=0.10,boost=0.90)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.10, max_boost=0.90))
    _lit_eval("R43.04.SoftmaxRich(T=0.15,base=0.05,boost=0.95)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.05, max_boost=0.95))
    _lit_eval("R43.05.SoftmaxRich(T=0.15,base=0.00,boost=1.00)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.00, max_boost=1.00))
    # Area 2: Slight extrapolation with optimized base
    _lit_eval("R43.06.SoftmaxRich(T=0.15,base=0.25,boost=0.80)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.25, max_boost=0.80))
    _lit_eval("R43.07.SoftmaxRich(T=0.15,base=0.20,boost=0.85)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.85))
    _lit_eval("R43.08.SoftmaxRich(T=0.15,base=0.15,boost=0.90)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.15, max_boost=0.90))
    # Area 3: Adaptive alpha — BOTH nor_w AND alpha scale with richness
    _lit_eval("R43.09.AdaptAlpha(T=0.15,base=0.25,boost=0.75,a0=0.30,ab=0.10)→cSEBBs",
              softmax_anchor_adaptive_alpha_cp(oof_final, richness_thr=0.5, temp=0.15,
                                               base_nw=0.25, max_boost=0.75, base_alpha=0.30, alpha_boost=0.10))
    _lit_eval("R43.10.AdaptAlpha(T=0.15,base=0.25,boost=0.75,a0=0.25,ab=0.15)→cSEBBs",
              softmax_anchor_adaptive_alpha_cp(oof_final, richness_thr=0.5, temp=0.15,
                                               base_nw=0.25, max_boost=0.75, base_alpha=0.25, alpha_boost=0.15))
    _lit_eval("R43.11.AdaptAlpha(T=0.15,base=0.25,boost=0.75,a0=0.35,ab=0.10)→cSEBBs",
              softmax_anchor_adaptive_alpha_cp(oof_final, richness_thr=0.5, temp=0.15,
                                               base_nw=0.25, max_boost=0.75, base_alpha=0.35, alpha_boost=0.10))
    _lit_eval("R43.12.AdaptAlpha(T=0.15,base=0.20,boost=0.80,a0=0.28,ab=0.12)→cSEBBs",
              softmax_anchor_adaptive_alpha_cp(oof_final, richness_thr=0.5, temp=0.15,
                                               base_nw=0.20, max_boost=0.80, base_alpha=0.28, alpha_boost=0.12))
    _lit_eval("R43.13.AdaptAlpha(T=0.15,base=0.25,boost=0.75,a0=0.30,ab=0.15)→cSEBBs",
              softmax_anchor_adaptive_alpha_cp(oof_final, richness_thr=0.5, temp=0.15,
                                               base_nw=0.25, max_boost=0.75, base_alpha=0.30, alpha_boost=0.15))
    # Area 4: T fine-tune (between T=0.10 and T=0.15)
    _lit_eval("R43.14.SoftmaxRich(T=0.12,base=0.25,boost=0.75)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.12, base_nw=0.25, max_boost=0.75))
    _lit_eval("R43.15.SoftmaxRich(T=0.13,base=0.25,boost=0.75)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.13, base_nw=0.25, max_boost=0.75))
    _lit_eval("R43.16.SoftmaxRich(T=0.20,base=0.20,boost=0.80)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.20, base_nw=0.20, max_boost=0.80))
    # Area 5: DualBranch with lower base for both
    _lit_eval("R43.17.SoftmaxRichDual(T=0.15,bA=0.25,boA=0.75,bB=0.20,boB=0.40)→cSEBBs",
              softmax_anchor_richness_dual_cp(oof_final, richness_thr=0.5, temp=0.15,
                                              base_nw_a=0.25, max_boost_a=0.75,
                                              base_nw_b=0.20, max_boost_b=0.40))
    _lit_eval("R43.18.SoftmaxRichDual(T=0.15,bA=0.20,boA=0.80,bB=0.15,boB=0.35)→cSEBBs",
              softmax_anchor_richness_dual_cp(oof_final, richness_thr=0.5, temp=0.15,
                                              base_nw_a=0.20, max_boost_a=0.80,
                                              base_nw_b=0.15, max_boost_b=0.35))
    _lit_eval("R43.19.SoftmaxRich(T=0.15,base=0.25,boost=0.75,cp=0.05)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15,
                                         base_nw=0.25, max_boost=0.75, cp_thr=0.05))
    _lit_eval("R43.20.AdaptAlpha(T=0.15,base=0.20,boost=0.80,a0=0.30,ab=0.10)→cSEBBs",
              softmax_anchor_adaptive_alpha_cp(oof_final, richness_thr=0.5, temp=0.15,
                                               base_nw=0.20, max_boost=0.80, base_alpha=0.30, alpha_boost=0.10))

    # ── Round 44: MaxMeanResidual anchor (BirdCLEF 2024 3rd place) + hybrid variants ─────────────
    # R44 verdict: MMR anchor USELESS (0.79xx). Alpha=0.38 confirmed optimal (sweep 0.35-0.48).
    #   HARD PLATEAU at 0.8137. All paper techniques exhausted (5 rounds search).
    # R45: Sweep 5 pipeline parameters never varied: entr_temp, lse_radius, w_a, beta_a, cp_blend.
    print("\n── Round 45: SoftmaxRich pipeline param sweep (entr_temp/radius/w_a/beta_a/cp_blend) ──")
    BEST = dict(richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.38)
    # Area 1: entr_temp sweep
    _lit_eval("R45.01.SoftRich+entr_temp=0.05→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, entr_temp=0.05))
    _lit_eval("R45.02.SoftRich+entr_temp=0.15→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, entr_temp=0.15))
    _lit_eval("R45.03.SoftRich+entr_temp=0.20→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, entr_temp=0.20))
    _lit_eval("R45.04.SoftRich+entr_temp=0.30→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, entr_temp=0.30))
    # Area 2: lse_radius
    _lit_eval("R45.05.SoftRich+radius=2→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, lse_radius=2))
    # Area 3: w_a sweep
    _lit_eval("R45.06.SoftRich+w_a=0.45→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, w_a=0.45))
    _lit_eval("R45.07.SoftRich+w_a=0.50→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, w_a=0.50))
    _lit_eval("R45.08.SoftRich+w_a=0.60→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, w_a=0.60))
    # Area 4: beta_a sweep
    _lit_eval("R45.09.SoftRich+beta=4.8→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, beta=4.8))
    _lit_eval("R45.10.SoftRich+beta=5.0→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, beta=5.0))
    _lit_eval("R45.11.SoftRich+beta=5.3→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, beta=5.3))
    _lit_eval("R45.12.SoftRich+beta=5.5→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, beta=5.5))
    # Area 5: cp_blend sweep
    _lit_eval("R45.13.SoftRich+cp_blend=0.2→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, cp_blend=0.2))
    _lit_eval("R45.14.SoftRich+cp_blend=0.3→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, cp_blend=0.3))
    _lit_eval("R45.15.SoftRich+cp_blend=0.5→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, cp_blend=0.5))
    _lit_eval("R45.16.SoftRich+cp_blend=0.6→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, cp_blend=0.6))
    # Area 6: promising combos
    _lit_eval("R45.17.SoftRich+entr=0.05+cp_blend=0.3→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, entr_temp=0.05, cp_blend=0.3))
    _lit_eval("R45.18.SoftRich+w_a=0.50+cp_blend=0.3→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, w_a=0.50, cp_blend=0.3))
    _lit_eval("R45.19.SoftRich+entr=0.05+w_a=0.50→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, entr_temp=0.05, w_a=0.50))
    _lit_eval("R45.20.SoftRich+entr=0.05+w_a=0.50+cp_blend=0.3→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST, entr_temp=0.05, w_a=0.50, cp_blend=0.3))

    # R45 verdict: cp_blend monotone increasing! 0.4→0.8137, 0.5→0.8139, 0.6→0.8140 NEW BEST!
    #   entr_temp: 0.05→0.8013, 0.10(base)→0.8137, 0.15→0.8119, higher hurts. 0.1 confirmed optimal.
    #   lse_radius=2 hurts (0.8106). w_a/beta_a flat (±0.0001). cp_blend NOT saturated — extend higher.
    # R46 strategy: push cp_blend beyond 0.6 (0.7/0.8/0.9/1.0) + cross-combos with best cp_blend.
    #   Area 1: cp_blend push (0.65/0.70/0.75/0.80/0.90/1.00)
    #   Area 2: cp_thr interaction with best cp_blend (thr=0.04/0.05/0.07/0.08)
    #   Area 3: cp_blend=0.7 + minor param combos (T/base/boost already confirmed)
    #   Area 4: cSEBBs threshold sweep at cp_blend=0.6
    # R46 verdict: cp_blend peaks at 0.60. cp_thr=0.05 ties at 0.8140. Hard plateau confirmed.
    # R47 verdict: TBD. R48: branch_combine (BirdCLEF 2024 1st place: min-reduction across models)
    print("\n── Round 48: branch_combine sweep (min/geom/harm/max) ──")
    BEST48 = dict(richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.38)
    B48 = dict(**BEST48, cp_blend=0.60)  # base: best known config
    # Area 1: pure combine mode sweep
    _lit_eval("R48.01.SoftRich+combine=min→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="min"))
    _lit_eval("R48.02.SoftRich+combine=geom→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="geom"))
    _lit_eval("R48.03.SoftRich+combine=harm→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="harm"))
    _lit_eval("R48.04.SoftRich+combine=max→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="max"))
    # Area 2: min + scale combos
    _lit_eval("R48.05.SoftRich+min+scale=1.10→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="min", out_scale=1.10))
    _lit_eval("R48.06.SoftRich+min+scale=1.20→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="min", out_scale=1.20))
    _lit_eval("R48.07.SoftRich+min+scale=1.30→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="min", out_scale=1.30))
    _lit_eval("R48.08.SoftRich+min+scale=1.00→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="min", out_scale=1.00))
    # Area 3: min + cp_thr/cp_blend combos
    _lit_eval("R48.09.SoftRich+min+cp_thr=0.05→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="min", cp_thr=0.05))
    _lit_eval("R48.10.SoftRich+min+cp_thr=0.04→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="min", cp_thr=0.04))
    _lit_eval("R48.11.SoftRich+min+cp_blend=0.50→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST48, branch_combine="min", cp_blend=0.50))
    _lit_eval("R48.12.SoftRich+min+cp_blend=0.70→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST48, branch_combine="min", cp_blend=0.70))
    # Area 4: geom + combos
    _lit_eval("R48.13.SoftRich+geom+scale=1.10→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="geom", out_scale=1.10))
    _lit_eval("R48.14.SoftRich+geom+scale=1.20→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="geom", out_scale=1.20))
    _lit_eval("R48.15.SoftRich+geom+cp_thr=0.05→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="geom", cp_thr=0.05))
    _lit_eval("R48.16.SoftRich+geom+cp_blend=0.50→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST48, branch_combine="geom", cp_blend=0.50))
    # Area 5: harm + combos
    _lit_eval("R48.17.SoftRich+harm+scale=1.10→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="harm", out_scale=1.10))
    _lit_eval("R48.18.SoftRich+harm+scale=1.20→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="harm", out_scale=1.20))
    _lit_eval("R48.19.SoftRich+harm+cp_thr=0.05→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **B48, branch_combine="harm", cp_thr=0.05))
    _lit_eval("R48.20.SoftRich+harm+cp_blend=0.50→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST48, branch_combine="harm", cp_blend=0.50))

    # ── R49: Segment-MAX blend (A-CPD arXiv:2403.08525) + Richness-adaptive cp_blend ────────────
    # MOTIVATION: cSEBBs currently uses segment MEAN to pull clips toward average.
    # A-CPD paper: per-segment MAX amplifies high-confidence peaks to fill their segment.
    # This is non-monotone: short isolated peaks get expanded to adjacent clips → more recall.
    # Also testing richness-adaptive cp_blend: rich files (many active species) get
    # stronger segment smoothing (higher cp_blend), poor files weaker (less smoothing).
    # ── Round 50: Local-max propagation in logit space ────────────────────────
    # Inspired by BirdCLEF 0.91 public notebook: local-max propagation α=0.15
    # for Aves classes in logit space.  Sliding window (radius r) max blended
    # back into original logit: out[t,c] = (1-α)*logit[t,c] + α*max(logit[t-r:t+r+1,c])
    # This is a logit-space dilation with soft blending — distinct from our
    # prob-space file_max methods (lse_softmax_blend_cp) and hard dilation.
    # AVES_IDX: classes 72-233 are Aves in our 234-class BirdCLEF 2026 space.
    print("\n── Round 50: Local-max propagation in logit space (BirdCLEF 0.91 notebook) ──")
    _AVES_IDX = list(range(72, NUM_CLASSES))  # 162 Aves classes

    def local_max_prop_logit(logits, alpha=0.15, radius=1, aves_only=False):
        """Slide a max-filter of width (2*radius+1) over the 12-clip time axis in
        logit space, then blend: out[t,c] = (1-α)*logit[t,c] + α*local_max[t,c].
        If aves_only=True, apply only to class indices 72-233 (Aves in BC2026).
        """
        n_files = logits.shape[0] // N_WINDOWS
        X = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)  # (F, T, C)
        out = X.copy()
        cols = _AVES_IDX if aves_only else list(range(NUM_CLASSES))
        for t in range(N_WINDOWS):
            t_s = max(0, t - radius)
            t_e = min(N_WINDOWS, t + radius + 1)
            lmax = X[:, t_s:t_e, :][:, :, cols].max(axis=1)  # (F, len(cols))
            out[:, t, cols] = (1 - alpha) * X[:, t, cols] + alpha * lmax
        return out.reshape(-1, NUM_CLASSES)

    BEST50 = dict(richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.38, cp_blend=0.60)

    # Area 1: standalone lmax_prop (all classes) → sweep alpha, radius
    for _a in [0.10, 0.15, 0.20, 0.25]:
        _lit_eval(f"R50.lmax_prop(α={_a},r=1,all)→raw",
                  local_max_prop_logit(oof_final, alpha=_a, radius=1, aves_only=False))
    for _a in [0.10, 0.15, 0.20]:
        _lit_eval(f"R50.lmax_prop(α={_a},r=2,all)→raw",
                  local_max_prop_logit(oof_final, alpha=_a, radius=2, aves_only=False))

    # Area 2: Aves-only lmax_prop (matches the 0.91 notebook behaviour)
    for _a in [0.10, 0.15, 0.20, 0.25]:
        _lit_eval(f"R50.lmax_prop(α={_a},r=1,aves)→raw",
                  local_max_prop_logit(oof_final, alpha=_a, radius=1, aves_only=True))

    # Area 3: lmax_prop → cSEBBs (add the best final step)
    for _a in [0.10, 0.15, 0.20]:
        _lmp = local_max_prop_logit(oof_final, alpha=_a, radius=1, aves_only=False)
        _lit_eval(f"R50.lmax_prop(α={_a},r=1,all)→cSEBBs",
                  change_point_segment_mean(_lmp, threshold=0.06, blend=0.60))
    for _a in [0.10, 0.15, 0.20]:
        _lmp = local_max_prop_logit(oof_final, alpha=_a, radius=1, aves_only=True)
        _lit_eval(f"R50.lmax_prop(α={_a},r=1,aves)→cSEBBs",
                  change_point_segment_mean(_lmp, threshold=0.06, blend=0.60))

    # Area 4: SoftRich (cp_blend=0 = no cSEBBs) → lmax_prop → cSEBBs
    # Inserts lmax_prop between the anchor-blend step and the final cSEBBs.
    _softrich_no_cp = softmax_anchor_richness_cp(oof_final, **dict(BEST50, cp_blend=0.0))
    for _a in [0.10, 0.15, 0.20]:
        _lit_eval(f"R50.SoftRich→lmax_prop(α={_a})→cSEBBs",
                  change_point_segment_mean(
                      local_max_prop_logit(_softrich_no_cp, alpha=_a, radius=1, aves_only=False),
                      threshold=0.06, blend=0.60))

    # Area 5: lmax_prop → full BEST50 pipeline (prepend lmax to input)
    for _a in [0.10, 0.15, 0.20]:
        _lmp_in = local_max_prop_logit(oof_final, alpha=_a, radius=1, aves_only=False)
        _lit_eval(f"R50.lmax_pre(α={_a})→SoftRich→cSEBBs",
                  softmax_anchor_richness_cp(_lmp_in, **BEST50))
    for _a in [0.10, 0.15, 0.20]:
        _lmp_in = local_max_prop_logit(oof_final, alpha=_a, radius=1, aves_only=True)
        _lit_eval(f"R50.lmax_pre_aves(α={_a})→SoftRich→cSEBBs",
                  softmax_anchor_richness_cp(_lmp_in, **BEST50))

    # ── Round 51: lmax_pre_aves refinement + post-lmax + SoftRich param sweep ─
    # R50 winner: lmax_pre_aves(α=0.1)→SoftRich→cSEBBs = 0.8163 (NEW BEST).
    # R51 explores three directions from this winner:
    #   Area 1: finer α sweep for lmax_pre_aves (smaller 0.05/0.08 and larger 0.30)
    #           and wider radius=2 window (temporal context 5 clips vs 3)
    #   Area 2: post-SoftRich lmax (apply lmax AFTER SoftRich output, in logit space)
    #           — cascaded dilation: lmax_pre→SoftRich→lmax_post→(extra cSEBBs?)
    #   Area 3: SoftRich hyperparameter refinement with lmax_pre(α=0.1,aves) fixed:
    #           richness_thr (0.40/0.45/0.55), temp (0.10/0.12/0.20), cp_blend (0.55/0.65/0.70)
    #   Area 4: prob-space blending of R50 output with baseline SoftRich (no lmax_pre)
    #           to check if ensemble of lmax+no-lmax is better than either alone
    print("\n── Round 51: lmax_pre_aves refinement + post-lmax + SoftRich param sweep ──")
    _LMP_R50 = local_max_prop_logit(oof_final, alpha=0.1, radius=1, aves_only=True)  # R50 winner input

    # Area 1: α refinement (smaller: 0.05, 0.08 — larger: 0.25 already tested, try 0.30)
    for _a in [0.05, 0.08, 0.30]:
        _lmp = local_max_prop_logit(oof_final, alpha=_a, radius=1, aves_only=True)
        _lit_eval(f"R51.lmax_pre_aves(α={_a},r=1)→SoftRich→cSEBBs",
                  softmax_anchor_richness_cp(_lmp, **BEST50))
    # radius=2: wider temporal context (5-clip window) with winner α=0.10
    _lmp_r2 = local_max_prop_logit(oof_final, alpha=0.10, radius=2, aves_only=True)
    _lit_eval("R51.lmax_pre_aves(α=0.10,r=2)→SoftRich→cSEBBs",
              softmax_anchor_richness_cp(_lmp_r2, **BEST50))
    # radius=2 with α=0.05 (smaller blend to compensate wider window)
    _lmp_r2_05 = local_max_prop_logit(oof_final, alpha=0.05, radius=2, aves_only=True)
    _lit_eval("R51.lmax_pre_aves(α=0.05,r=2)→SoftRich→cSEBBs",
              softmax_anchor_richness_cp(_lmp_r2_05, **BEST50))

    # Area 2: post-SoftRich lmax (lmax_pre → SoftRich → lmax_post → cSEBBs)
    # SoftRich output is in logit space (scaled by TEMP_SCALE); apply lmax there too
    _sr_no_cp = softmax_anchor_richness_cp(_LMP_R50, **dict(BEST50, cp_blend=0.0))
    for _a_post in [0.05, 0.10, 0.15]:
        _lmp_post = local_max_prop_logit(_sr_no_cp, alpha=_a_post, radius=1, aves_only=True)
        _lit_eval(f"R51.lmax_pre(0.1)→SoftRich→lmax_post_aves(α={_a_post})→cSEBBs",
                  change_point_segment_mean(_lmp_post, threshold=0.06, blend=0.60))
    # also test post-lmax all-class
    for _a_post in [0.05, 0.10]:
        _lmp_post_all = local_max_prop_logit(_sr_no_cp, alpha=_a_post, radius=1, aves_only=False)
        _lit_eval(f"R51.lmax_pre(0.1)→SoftRich→lmax_post_all(α={_a_post})→cSEBBs",
                  change_point_segment_mean(_lmp_post_all, threshold=0.06, blend=0.60))

    # Area 3: SoftRich hyperparameter refinement with lmax_pre(α=0.1,aves) fixed
    # richness_thr sweep (0.40, 0.45, 0.55 — winner 0.50)
    for _rt in [0.40, 0.45, 0.55]:
        _lit_eval(f"R51.lmax_pre→SoftRich(thr={_rt})→cSEBBs",
                  softmax_anchor_richness_cp(_LMP_R50, **dict(BEST50, richness_thr=_rt)))
    # temp sweep (0.10, 0.12, 0.20 — winner 0.15)
    for _t in [0.10, 0.12, 0.20]:
        _lit_eval(f"R51.lmax_pre→SoftRich(temp={_t})→cSEBBs",
                  softmax_anchor_richness_cp(_LMP_R50, **dict(BEST50, temp=_t)))
    # cp_blend sweep (0.55, 0.65, 0.70 — winner 0.60)
    for _cb in [0.55, 0.65, 0.70]:
        _lit_eval(f"R51.lmax_pre→SoftRich(cp_blend={_cb})→cSEBBs",
                  softmax_anchor_richness_cp(_LMP_R50, **dict(BEST50, cp_blend=_cb)))
    # alpha sweep (0.35, 0.40 — winner 0.38)
    for _al in [0.35, 0.40]:
        _lit_eval(f"R51.lmax_pre→SoftRich(alpha={_al})→cSEBBs",
                  softmax_anchor_richness_cp(_LMP_R50, **dict(BEST50, alpha=_al)))

    # Area 4: prob-space ensemble of R50 with pure SoftRich (no lmax_pre)
    # If lmax_pre adds independent signal, blending the two should outperform either
    eps_blend = 1e-7
    _r50_probs = sigmoid(softmax_anchor_richness_cp(_LMP_R50, **BEST50))
    _sr_probs  = sigmoid(softmax_anchor_richness_cp(oof_final, **BEST50))
    for _w in [0.60, 0.70, 0.80]:
        _blend = _w * _r50_probs + (1 - _w) * _sr_probs
        _blend = np.clip(_blend, eps_blend, 1 - eps_blend)
        _blend_logit = np.log(_blend / (1 - _blend)) * TEMP_SCALE
        _lit_eval(f"R51.blend(R50*{_w}+SoftRich*{1-_w:.2f})→raw",
                  _blend_logit)

    # ── Round 52: Bidirectional lmax + Power scaling + nSEBBs-inspired adaptive ──
    # Literature-informed methods (paper search 2026-03-22):
    #   Area 1: Bidirectional lmax -- forward+backward lmax_pre_aves, average logits
    #     Inspiration: Voxaboxen forward/backward onset-offset fusion (arXiv 2503.02389)
    #     Rationale: Causal lmax only propagates evidence rightward. Backward pass catches
    #                trailing evidence that forward pass sees as onset. Averaging both
    #                directions creates symmetrical temporal smearing.
    #   Area 2: Power scaling (score^alpha) -- monotonic transform on R50 probs
    #     Inspiration: BirdCLEF+ 2025 winner power calibration (score^alpha per round)
    #     Rationale: If R50 probs are over/under-confident, a simple monotonic transform
    #                can sharpen or soften predictions without retraining.
    #   Area 3: nSEBBs-inspired adaptive threshold -- replace fixed cp_thr=0.06 with
    #     per-file Posterior Contrast Ratio (PCR) = log(mean_high / pct10)
    #     Inspiration: nSEBBs (arXiv 2505.11889) -- adaptive SEBBs, 4% PSDS1 improvement
    #     NOTE: Our _change_point_segment_mean uses blend, not hard threshold -- PCR
    #           maps to adaptive blend strength, not a binary on/off.
    #   Area 4: Combine best of each area with R50 (BEST50 with alpha=0.40)

    BEST52 = dict(richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80,
                  alpha=0.40, cp_blend=0.60)  # R51 best: alpha=0.40

    print("\n-- Round 52: Bidirectional lmax + Power scaling + nSEBBs adaptive --")

    # Area 1: Bidirectional lmax_pre_aves
    def _bidir_lmax(logits, alpha=0.1, radius=1):
        """Average forward and backward lmax_pre_aves in logit space."""
        fwd = local_max_prop_logit(logits, alpha=alpha, radius=radius, aves_only=True)
        bwd_logits = logits[::-1].copy()
        bwd_prop = local_max_prop_logit(bwd_logits, alpha=alpha, radius=radius, aves_only=True)
        bwd = bwd_prop[::-1]
        return 0.5 * (fwd + bwd)

    _BIDIR = _bidir_lmax(oof_final, alpha=0.10, radius=1)
    _lit_eval("R52.bidir_lmax(a=0.10,r=1)->SoftRich->cSEBBs",
              softmax_anchor_richness_cp(_BIDIR, **BEST52))
    _lit_eval("R52.bidir_lmax(a=0.05,r=1)->SoftRich->cSEBBs",
              softmax_anchor_richness_cp(_bidir_lmax(oof_final, alpha=0.05, radius=1), **BEST52))
    _lit_eval("R52.bidir_lmax(a=0.15,r=1)->SoftRich->cSEBBs",
              softmax_anchor_richness_cp(_bidir_lmax(oof_final, alpha=0.15, radius=1), **BEST52))
    _lit_eval("R52.bidir_lmax(a=0.20,r=1)->SoftRich->cSEBBs",
              softmax_anchor_richness_cp(_bidir_lmax(oof_final, alpha=0.20, radius=1), **BEST52))
    # Bidir with max instead of avg
    def _bidir_max_lmax(logits, alpha=0.1, radius=1):
        fwd = local_max_prop_logit(logits, alpha=alpha, radius=radius, aves_only=True)
        bwd = local_max_prop_logit(logits[::-1].copy(), alpha=alpha, radius=radius, aves_only=True)[::-1]
        return np.maximum(fwd, bwd)
    _lit_eval("R52.bidir_MAX_lmax(a=0.10)->SoftRich->cSEBBs",
              softmax_anchor_richness_cp(_bidir_max_lmax(oof_final, alpha=0.10), **BEST52))
    _lit_eval("R52.bidir_MAX_lmax(a=0.05)->SoftRich->cSEBBs",
              softmax_anchor_richness_cp(_bidir_max_lmax(oof_final, alpha=0.05), **BEST52))

    # Area 2: Power scaling on R50 probs (SoftRich alpha=0.40, then power transform)
    _R50_logits = _LMP_R50  # lmax_pre applied
    _R50_probs  = sigmoid(softmax_anchor_richness_cp(_R50_logits, **BEST52))
    _eps52 = 1e-7
    for _pw in [0.5, 0.7, 1.5, 2.0, 3.0]:
        _powered = np.clip(_R50_probs ** _pw, _eps52, 1 - _eps52)
        _powered_logit = np.log(_powered / (1 - _powered)) * TEMP_SCALE
        _lit_eval(f"R52.R50_probs^{_pw}->logit->eval",
                  _powered_logit)
    # Power scale in logit space before SoftRich (sharpening logits)
    for _lscale in [0.8, 0.9, 1.1, 1.2, 1.5]:
        _lit_eval(f"R52.lmax_pre->lscale({_lscale})->SoftRich->cSEBBs",
                  softmax_anchor_richness_cp(_LMP_R50 * _lscale, **BEST52))

    # Area 3: nSEBBs-inspired per-file adaptive cp_blend
    # Replace fixed cp_blend=0.60 with per-file PCR-derived blend
    # PCR = log(mean of top-25% probs / max(pct10, 1e-6)) per file, normalized to [0,1]
    # High PCR => strong signal => use higher cp_blend (more change-point smoothing)
    # Low PCR => weak signal => lower cp_blend (preserve raw probs)
    def _adapt_cp_blend(logits, base_blend=0.50, boost=0.30,
                        richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.40,
                        cp_thr=0.06):
        """Per-file adaptive cp_blend via PCR. logits: (N,234) already lmax-propagated."""
        probs = sigmoid(logits)  # (N, 234)
        # PCR per time-step: ratio of top-25% mean to pct10
        p_sorted = np.sort(probs, axis=-1)[:, ::-1]  # descending
        top25_end = max(1, probs.shape[-1] // 4)
        top25_mean = p_sorted[:, :top25_end].mean(axis=-1)  # (N,)
        pct10 = np.percentile(probs, 10, axis=-1)  # (N,)
        pcr = np.log(top25_mean / np.maximum(pct10, 1e-6))  # (N,)
        # Normalize PCR to [0,1] using file-level stats
        pcr_min, pcr_max = pcr.min(), pcr.max()
        if pcr_max > pcr_min:
            pcr_norm = (pcr - pcr_min) / (pcr_max - pcr_min)
        else:
            pcr_norm = np.ones_like(pcr) * 0.5
        adapt_blend = base_blend + boost * pcr_norm  # (N,) in [base, base+boost]
        # Apply SoftRich (global params) but use per-step adaptive cp_blend
        # Use fixed SoftRich then adaptive cSEBBs
        _sr_out = softmax_anchor_richness_cp(logits,
                                             richness_thr=richness_thr, temp=temp,
                                             base_nw=base_nw, max_boost=max_boost,
                                             alpha=alpha, cp_blend=0.0)  # no cSEBBs inside
        # Apply per-file adaptive cSEBBs blend (file-level mean PCR)
        # change_point_segment_mean requires (N_WINDOWS, C) input per file
        n_files_a = _sr_out.shape[0] // N_WINDOWS
        adapt_blend_per_file = adapt_blend.reshape(n_files_a, N_WINDOWS).mean(axis=1)
        _cpd_out = np.zeros_like(_sr_out)
        for f in range(n_files_a):
            sl = slice(f * N_WINDOWS, (f + 1) * N_WINDOWS)
            _cpd_out[sl] = change_point_segment_mean(
                _sr_out[sl], threshold=cp_thr, blend=float(adapt_blend_per_file[f]))
        return _cpd_out

    for _bb, _bst in [(0.50, 0.20), (0.45, 0.30), (0.40, 0.35), (0.55, 0.15)]:
        _lit_eval(f"R52.nSEBBs_adapt_cpblend(base={_bb},boost={_bst})->eval",
                  _adapt_cp_blend(_LMP_R50, base_blend=_bb, boost=_bst))

    # Area 4: Bidir lmax best + nSEBBs adaptive blend
    _lit_eval("R52.bidir_lmax(0.10)+adapt_cp(0.45,0.30)->eval",
              _adapt_cp_blend(_BIDIR, base_blend=0.45, boost=0.30))
    _lit_eval("R52.bidir_lmax(0.05)+adapt_cp(0.50,0.20)->eval",
              _adapt_cp_blend(_bidir_lmax(oof_final, alpha=0.05), base_blend=0.50, boost=0.20))

    # ── Round 53: P_max lifting + nSEBBs per-class PCR + onset peak-finding ──
    # Literature sources (paper search 2026-03-22):
    #   P_max lifting: BirdCLEF 2024 3rd place (+0.01-0.02 AUC); cross-time max prior
    #   nSEBBs per-class PCR: arXiv 2505.11889; class-adaptive cSEBBs threshold
    #   Onset peak-finding: Voxaboxen arXiv 2503.02389; scipy.find_peaks prominence
    #
    # BEST52 params: lmax_pre(0.1) → SoftRich(alpha=0.40) → cSEBBs  (R51 best = 0.8164)
    # All areas: use BEST52 as postproc, vary the pre-processing stage only.

    BEST53 = dict(richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80,
                  alpha=0.40, cp_blend=0.60)

    print("\n-- Round 53: P_max lifting + nSEBBs PCR + onset peak-finding --")

    # Area 1: P_max soundscape lifting (BirdCLEF 2024 3rd place technique)
    # For each soundscape (12 windows), compute per-class max across windows,
    # subtract the cross-class mean of those maxima, scale by alpha, blend into logits.
    # Formula: logit_lifted = logit + (pmax - pmax.mean(class)) * lift_alpha
    # Applied BEFORE lmax_pre → SoftRich → cSEBBs.
    def _pmax_lift(logits, lift_alpha=0.8):
        """Cross-time P_max lifting per soundscape file."""
        n_files = logits.shape[0] // N_WINDOWS
        X = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES)   # (F, 12, C)
        # P_max: per-file per-class max across time windows
        pmax = X.max(axis=1, keepdims=True)                   # (F, 1, C)
        # mean of pmax across classes (background level)
        pmax_mean = pmax.mean(axis=2, keepdims=True)          # (F, 1, 1)
        lifted = X + lift_alpha * (pmax - pmax_mean)          # (F, 12, C)
        return lifted.reshape(-1, NUM_CLASSES)

    for _la in [0.4, 0.6, 0.8, 1.0, 1.2]:
        _lifted = local_max_prop_logit(_pmax_lift(oof_final, lift_alpha=_la),
                                       alpha=0.1, radius=1, aves_only=True)
        _lit_eval(f"R53.pmax_lift({_la})->lmax->SoftRich->cSEBBs",
                  softmax_anchor_richness_cp(_lifted, **BEST53))

    # P_max AFTER lmax (different order)
    for _la in [0.4, 0.8]:
        _liftpost = _pmax_lift(_LMP_R50, lift_alpha=_la)
        _lit_eval(f"R53.lmax->pmax_lift({_la})->SoftRich->cSEBBs",
                  softmax_anchor_richness_cp(_liftpost, **BEST53))

    # Area 2: nSEBBs per-class PCR adaptive cp_thr
    # PCR_c = log(mean(p[p > p90_c]) / max(p10_c, 1e-6)) for each class c
    # High PCR → strong signal → use lower cp_thr (accept more change-points)
    # Low PCR → weak signal → higher cp_thr (suppress noise)
    # cp_thr_c = thr_base - pcr_norm_c * thr_range   (high PCR → lower threshold)
    def _nsebbs_adapt(probs_sr, thr_base=0.08, thr_range=0.05):
        """Apply per-class adaptive cp_thr derived from OOF PCR statistics.
        probs_sr: (N_windows, 234) SoftRich output probs (before cSEBBs)."""
        n_files = probs_sr.shape[0] // N_WINDOWS
        X = probs_sr.reshape(n_files, N_WINDOWS, NUM_CLASSES)  # (F, 12, C)
        # Compute per-class PCR from OOF stats
        p_all = probs_sr  # (N_windows, 234)
        pcr = np.zeros(NUM_CLASSES, dtype=np.float32)
        for c in range(NUM_CLASSES):
            pc = p_all[:, c]
            p90 = np.percentile(pc, 90)
            p10 = np.percentile(pc, 10)
            high_mean = pc[pc > p90].mean() if (pc > p90).any() else p90
            pcr[c] = np.log(high_mean / max(p10, 1e-6))
        pcr_norm = (pcr - pcr.min()) / (pcr.max() - pcr.min() + 1e-8)
        # High PCR → class has strong signal → lower threshold (detect more events)
        cp_thr_c = thr_base - pcr_norm * thr_range  # (234,)
        cp_thr_c = np.clip(cp_thr_c, 0.02, 0.12)
        # Apply per-class adaptive cSEBBs inline (can't call change_point_segment_mean
        # with 1D per-class input — implement change-point logic directly)
        out = probs_sr.copy()
        for c in range(NUM_CLASSES):
            thr = float(cp_thr_c[c])
            for f in range(n_files):
                sl = slice(f * N_WINDOWS, (f + 1) * N_WINDOWS)
                seg = probs_sr[sl, c].copy()
                diffs_c = np.abs(np.diff(seg))
                boundaries = ([0] + [t + 1 for t in range(N_WINDOWS - 1)
                                     if diffs_c[t] > thr] + [N_WINDOWS])
                for i in range(len(boundaries) - 1):
                    s, e = boundaries[i], boundaries[i + 1]
                    seg_mean = seg[s:e].mean()
                    seg[s:e] = 0.60 * seg_mean + 0.40 * seg[s:e]
                out[sl, c] = seg
        return out

    # Compute SoftRich output (before cSEBBs) for nSEBBs input
    _sr_probs_raw = sigmoid(softmax_anchor_richness_cp(_LMP_R50, **dict(BEST53, cp_blend=0.0)))
    for _tb, _tr in [(0.08, 0.05), (0.08, 0.03), (0.06, 0.04), (0.10, 0.06)]:
        _nsebbs_out = _nsebbs_adapt(_sr_probs_raw, thr_base=_tb, thr_range=_tr)
        _nsebbs_logit = np.log(np.clip(_nsebbs_out, 1e-7, 1-1e-7) /
                               np.clip(1 - _nsebbs_out, 1e-7, 1-1e-7)) * TEMP_SCALE
        _lit_eval(f"R53.lmax->SoftRich->nSEBBs(thr_base={_tb},range={_tr})",
                  _nsebbs_logit)

    # Area 3: Onset peak-finding (scipy.find_peaks prominence-based)
    # For each class and soundscape, find prominence peaks in probability time series.
    # Non-peak windows get dampened; peak windows get boosted.
    # Inspired by Voxaboxen (arXiv:2503.02389).
    from scipy.signal import find_peaks

    def _onset_peak_smooth(logits, prominence=0.05, decay=0.5):
        """Prominence-based peak dampening/boosting in prob space.
        Peak windows: keep as-is. Non-peak: blend toward local mean."""
        n_files = logits.shape[0] // N_WINDOWS
        probs = sigmoid(logits)
        X = probs.reshape(n_files, N_WINDOWS, NUM_CLASSES)
        out = X.copy()
        for f in range(n_files):
            for c in range(NUM_CLASSES):
                seq = X[f, :, c]
                peaks, _ = find_peaks(seq, prominence=prominence)
                if len(peaks) == 0:
                    # No prominent peaks: dampen all scores toward mean
                    out[f, :, c] = seq * decay + seq.mean() * (1 - decay)
                else:
                    # Non-peak frames: blend toward mean; peak frames: keep
                    mask = np.ones(N_WINDOWS, dtype=bool)
                    mask[peaks] = False
                    out[f, mask, c] = seq[mask] * decay + seq.mean() * (1 - decay)
        out_logits = np.log(np.clip(out, 1e-7, 1-1e-7) /
                            np.clip(1 - out, 1e-7, 1-1e-7))
        return out_logits.reshape(-1, NUM_CLASSES)

    for _prom, _dec in [(0.03, 0.5), (0.05, 0.5), (0.05, 0.7), (0.08, 0.5), (0.10, 0.5)]:
        _peak_logits = _onset_peak_smooth(oof_final, prominence=_prom, decay=_dec)
        _lmp_peak = local_max_prop_logit(_peak_logits, alpha=0.1, radius=1, aves_only=True)
        _lit_eval(f"R53.peak_smooth(prom={_prom},dec={_dec})->lmax->SoftRich->cSEBBs",
                  softmax_anchor_richness_cp(_lmp_peak, **BEST53))

    # Area 4: P_max lifting + onset peak-finding combined
    _lit_eval("R53.pmax(0.8)+peak(0.05,0.5)->lmax->SoftRich->cSEBBs",
              softmax_anchor_richness_cp(
                  local_max_prop_logit(
                      _onset_peak_smooth(_pmax_lift(oof_final, 0.8), prominence=0.05, decay=0.5),
                      alpha=0.1, radius=1, aves_only=True), **BEST53))

    # BEST49: same as BEST46 (optimal SoftmaxRich config)
    print("\n── Round 49: Segment-MAX blend (A-CPD) + Richness-adaptive cp_blend ──")
    BEST49 = dict(richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.38, cp_blend=0.60)
    # Area 1: seg_max_w sweep (0=pure mean=current, 1=pure max)
    _lit_eval("R49.01.SoftRich+seg_max_w=0.25→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST49, seg_max_w=0.25))
    _lit_eval("R49.02.SoftRich+seg_max_w=0.50→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST49, seg_max_w=0.50))
    _lit_eval("R49.03.SoftRich+seg_max_w=0.75→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST49, seg_max_w=0.75))
    _lit_eval("R49.04.SoftRich+seg_max_w=1.00→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST49, seg_max_w=1.00))
    # Area 2: seg_max_w + cp_blend interaction (if max_w helps, try different blend depths)
    _lit_eval("R49.05.SoftRich+seg_max_w=0.25+cp_blend=0.50→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **dict(BEST49, cp_blend=0.50), seg_max_w=0.25))
    _lit_eval("R49.06.SoftRich+seg_max_w=0.50+cp_blend=0.50→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **dict(BEST49, cp_blend=0.50), seg_max_w=0.50))
    _lit_eval("R49.07.SoftRich+seg_max_w=0.25+cp_blend=0.40→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **dict(BEST49, cp_blend=0.40), seg_max_w=0.25))
    _lit_eval("R49.08.SoftRich+seg_max_w=0.25+cp_thr=0.05→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST49, seg_max_w=0.25, cp_thr=0.05))
    # Area 3: richness-adaptive cp_blend (cp_blend_boost > 0 → rich files get more smoothing)
    # Base cp_blend=0.60 (best); boost=0.20 → max effective blend = 0.80 for richest file
    _lit_eval("R49.09.SoftRich+adapt_cpblend(boost=0.10)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST49, cp_blend_boost=0.10))
    _lit_eval("R49.10.SoftRich+adapt_cpblend(boost=0.20)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST49, cp_blend_boost=0.20))
    _lit_eval("R49.11.SoftRich+adapt_cpblend(boost=0.30)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST49, cp_blend_boost=0.30))
    _lit_eval("R49.12.SoftRich+adapt_cpblend(base=0.50,boost=0.20)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **dict(BEST49, cp_blend=0.50), cp_blend_boost=0.20))
    _lit_eval("R49.13.SoftRich+adapt_cpblend(base=0.40,boost=0.30)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **dict(BEST49, cp_blend=0.40), cp_blend_boost=0.30))
    # Area 4: combinations of seg_max + richness-adaptive blend
    _lit_eval("R49.14.SoftRich+seg_max_w=0.25+adapt_cpblend(boost=0.10)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST49, seg_max_w=0.25, cp_blend_boost=0.10))
    _lit_eval("R49.15.SoftRich+seg_max_w=0.50+adapt_cpblend(boost=0.10)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST49, seg_max_w=0.50, cp_blend_boost=0.10))

    # R47: out_scale sweep + two-pass cSEBBs
    print("\n── Round 47: out_scale sweep + two-pass cSEBBs ──")
    BEST47 = dict(richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.38, cp_blend=0.60)
    # Area 1: out_scale sweep
    _lit_eval("R47.01.SoftRich+scale=0.80→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=0.80))
    _lit_eval("R47.02.SoftRich+scale=0.90→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=0.90))
    _lit_eval("R47.03.SoftRich+scale=1.00→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.00))
    _lit_eval("R47.04.SoftRich+scale=1.05→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.05))
    _lit_eval("R47.05.SoftRich+scale=1.10→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.10))
    _lit_eval("R47.06.SoftRich+scale=1.20→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.20))
    _lit_eval("R47.07.SoftRich+scale=1.30→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.30))
    _lit_eval("R47.08.SoftRich+scale=1.50→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.50))
    # Area 2: two-pass cSEBBs at best config
    _lit_eval("R47.09.SoftRich+2pass(t2=0.06,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, cp2_thr=0.06, cp2_blend=0.60))
    _lit_eval("R47.10.SoftRich+2pass(t2=0.05,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, cp2_thr=0.05, cp2_blend=0.60))
    _lit_eval("R47.11.SoftRich+2pass(t2=0.06,b2=0.40)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, cp2_thr=0.06, cp2_blend=0.40))
    _lit_eval("R47.12.SoftRich+2pass(t2=0.08,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, cp2_thr=0.08, cp2_blend=0.60))
    _lit_eval("R47.13.SoftRich+2pass(t2=0.04,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, cp2_thr=0.04, cp2_blend=0.60))
    _lit_eval("R47.14.SoftRich+2pass(t2=0.10,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, cp2_thr=0.10, cp2_blend=0.60))
    # Area 3: best out_scale + two-pass
    _lit_eval("R47.15.SoftRich+scale=1.10+2pass(t2=0.06,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.10, cp2_thr=0.06, cp2_blend=0.60))
    _lit_eval("R47.16.SoftRich+scale=1.20+2pass(t2=0.06,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.20, cp2_thr=0.06, cp2_blend=0.60))
    _lit_eval("R47.17.SoftRich+scale=1.30+2pass(t2=0.06,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.30, cp2_thr=0.06, cp2_blend=0.60))
    _lit_eval("R47.18.SoftRich+scale=1.00+2pass(t2=0.06,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.00, cp2_thr=0.06, cp2_blend=0.60))
    _lit_eval("R47.19.SoftRich+scale=1.10+2pass(t2=0.05,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.10, cp2_thr=0.05, cp2_blend=0.60))
    _lit_eval("R47.20.SoftRich+scale=1.20+2pass(t2=0.05,b2=0.60)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST47, out_scale=1.20, cp2_thr=0.05, cp2_blend=0.60))

    print("\n── Round 46: cp_blend extension + cp_thr interaction ──")
    BEST46 = dict(richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.38)
    # Area 1: push cp_blend higher
    _lit_eval("R46.01.SoftRich+cp_blend=0.65→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.65))
    _lit_eval("R46.02.SoftRich+cp_blend=0.70→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.70))
    _lit_eval("R46.03.SoftRich+cp_blend=0.75→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.75))
    _lit_eval("R46.04.SoftRich+cp_blend=0.80→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.80))
    _lit_eval("R46.05.SoftRich+cp_blend=0.90→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.90))
    _lit_eval("R46.06.SoftRich+cp_blend=1.00→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=1.00))
    # Area 2: cp_thr sweep at cp_blend=0.60
    _lit_eval("R46.07.SoftRich+cp_blend=0.60+cp_thr=0.04→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.60, cp_thr=0.04))
    _lit_eval("R46.08.SoftRich+cp_blend=0.60+cp_thr=0.05→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.60, cp_thr=0.05))
    _lit_eval("R46.09.SoftRich+cp_blend=0.60+cp_thr=0.07→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.60, cp_thr=0.07))
    _lit_eval("R46.10.SoftRich+cp_blend=0.60+cp_thr=0.08→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.60, cp_thr=0.08))
    # Area 3: cp_blend=0.70 + cp_thr combos
    _lit_eval("R46.11.SoftRich+cp_blend=0.70+cp_thr=0.04→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.70, cp_thr=0.04))
    _lit_eval("R46.12.SoftRich+cp_blend=0.70+cp_thr=0.05→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.70, cp_thr=0.05))
    _lit_eval("R46.13.SoftRich+cp_blend=0.70+cp_thr=0.07→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.70, cp_thr=0.07))
    _lit_eval("R46.14.SoftRich+cp_blend=0.70+cp_thr=0.08→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.70, cp_thr=0.08))
    # Area 4: cp_blend=0.80 + cp_thr combos
    _lit_eval("R46.15.SoftRich+cp_blend=0.80+cp_thr=0.04→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.80, cp_thr=0.04))
    _lit_eval("R46.16.SoftRich+cp_blend=0.80+cp_thr=0.05→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.80, cp_thr=0.05))
    _lit_eval("R46.17.SoftRich+cp_blend=0.80+cp_thr=0.07→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.80, cp_thr=0.07))
    _lit_eval("R46.18.SoftRich+cp_blend=0.80+cp_thr=0.08→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=0.80, cp_thr=0.08))
    # Area 5: cp_blend=1.0 (full segment-mean) + cp_thr
    _lit_eval("R46.19.SoftRich+cp_blend=1.00+cp_thr=0.05→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=1.00, cp_thr=0.05))
    _lit_eval("R46.20.SoftRich+cp_blend=1.00+cp_thr=0.07→cSEBBs",
              softmax_anchor_richness_cp(oof_final, **BEST46, cp_blend=1.00, cp_thr=0.07))

    # R43 verdict: PLATEAU CONFIRMED in SoftmaxRich family. Base=0.20,boost=0.80→0.8137 marginal.
    #   All base values 0.00-0.25 give 0.8134-0.8137 (flat). AdaptAlpha uniformly hurts.
    # R44: Test MaxMeanResidual anchor formula — the only untested paper technique:
    #   anchor = lse_max + lse_mean - mean(lse_max)  [BirdCLEF 2024 3rd place]
    print("\n── Round 44: MaxMeanResidual anchor + SoftmaxRich hybrid ──")
    # Area 1: Pure MMR anchor — various alpha (replacing NOR anchor with MMR)
    _lit_eval("R44.01.MMR(a=0.38)→cSEBBs",
              max_mean_residual_cp(oof_final, alpha=0.38))
    _lit_eval("R44.02.MMR(a=0.30)→cSEBBs",
              max_mean_residual_cp(oof_final, alpha=0.30))
    _lit_eval("R44.03.MMR(a=0.45)→cSEBBs",
              max_mean_residual_cp(oof_final, alpha=0.45))
    _lit_eval("R44.04.MMR(a=0.50)→cSEBBs",
              max_mean_residual_cp(oof_final, alpha=0.50))
    _lit_eval("R44.05.MMR(a=0.25)→cSEBBs",
              max_mean_residual_cp(oof_final, alpha=0.25))
    # Area 2: MMR + SoftmaxRich adaptive weight (MMR anchor, richness-adaptive nor_w)
    _lit_eval("R44.06.MMRSoftRich(T=0.15,thr=0.5,base=0.20,boost=0.80)→cSEBBs",
              mmr_softmax_rich_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80))
    _lit_eval("R44.07.MMRSoftRich(T=0.15,thr=0.5,base=0.25,boost=0.75)→cSEBBs",
              mmr_softmax_rich_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.25, max_boost=0.75))
    _lit_eval("R44.08.MMRSoftRich(T=0.15,thr=0.5,base=0.40,boost=0.60)→cSEBBs",
              mmr_softmax_rich_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.40, max_boost=0.60))
    _lit_eval("R44.09.MMRSoftRich(T=0.10,thr=0.5,base=0.20,boost=0.80)→cSEBBs",
              mmr_softmax_rich_cp(oof_final, richness_thr=0.5, temp=0.10, base_nw=0.20, max_boost=0.80))
    _lit_eval("R44.10.MMRSoftRich(T=0.20,thr=0.5,base=0.20,boost=0.80)→cSEBBs",
              mmr_softmax_rich_cp(oof_final, richness_thr=0.5, temp=0.20, base_nw=0.20, max_boost=0.80))
    # Area 3: MMR alpha sweep with SoftmaxRich
    _lit_eval("R44.11.MMRSoftRich(T=0.15,a=0.30)→cSEBBs",
              mmr_softmax_rich_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.30))
    _lit_eval("R44.12.MMRSoftRich(T=0.15,a=0.45)→cSEBBs",
              mmr_softmax_rich_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.45))
    _lit_eval("R44.13.MMRSoftRich(T=0.15,a=0.50)→cSEBBs",
              mmr_softmax_rich_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.50))
    # Area 4: NOR+MMR hybrid anchor (blend both anchors)
    # Use current best SoftmaxRich as branch A but swap anchor to mix NOR+MMR
    _lit_eval("R44.14.SoftRich(T=0.15,base=0.20,boost=0.80)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80))
    _lit_eval("R44.15.SoftRich(T=0.15,base=0.20,boost=0.80,a=0.40)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.40))
    _lit_eval("R44.16.SoftRich(T=0.15,base=0.20,boost=0.80,a=0.42)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.42))
    _lit_eval("R44.17.SoftRich(T=0.15,base=0.20,boost=0.80,a=0.44)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.44))
    _lit_eval("R44.18.SoftRich(T=0.15,base=0.20,boost=0.80,a=0.46)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.46))
    _lit_eval("R44.19.SoftRich(T=0.15,base=0.20,boost=0.80,a=0.48)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.48))
    _lit_eval("R44.20.SoftRich(T=0.15,base=0.20,boost=0.80,a=0.35)→cSEBBs",
              softmax_anchor_richness_cp(oof_final, richness_thr=0.5, temp=0.15, base_nw=0.20, max_boost=0.80, alpha=0.35))

    # ── Round 39: AdaptRichPow gamma fine-tune + g=2.0 param sweep + SilenceCut + DualPow ─────────
    # R38 verdict: AdaptRichPow(g=2.0,boost=0.60)=0.8062 NEW BEST! Convex (g>1) wins:
    #   g=0.3→0.8048, g=0.5→0.8051, g=1.0(linear)→0.8057, g=1.5→0.8058, g=2.0→0.8062.
    #   BinaryGate HURTS (0.7984): step function bad; DualAdapt ≤0.8054; Extrap diminishes.
    # R39 strategy:
    #   Area 1: Push gamma higher (2.5, 3.0, 4.0, 5.0) — does convex keep improving?
    #   Area 2: Boost sweep with gamma=2.0 (0.40,0.50,0.70) — is 0.60 still optimal?
    #   Area 3: Base/thr sweep with gamma=2.0 (base=0.35/0.45, thr=0.3/0.5)
    #   Area 4: gamma=2.0 + SilenceCut (sthr=0.10/0.20)
    #   Area 5: gamma=2.0 + cp_thr=0.05 (was best in R37 param sweep)
    #   Area 6: DualPow — both branches use power-law with independent gammas
    print("\n── Round 39: AdaptRichPow gamma fine-tune / g=2.0 sweep / SilenceCut / DualPow ──")
    # Area 1: Gamma push beyond 2.0
    _lit_eval("R39.01.AdaptRichPow(g=2.5,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=2.5))
    _lit_eval("R39.02.AdaptRichPow(g=3.0,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=3.0))
    _lit_eval("R39.03.AdaptRichPow(g=4.0,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=4.0))
    _lit_eval("R39.04.AdaptRichPow(g=5.0,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=5.0))
    # Area 2: Boost sweep with gamma=2.0
    _lit_eval("R39.05.AdaptRichPow(g=2.0,boost=0.40)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.40, gamma=2.0))
    _lit_eval("R39.06.AdaptRichPow(g=2.0,boost=0.50)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.50, gamma=2.0))
    _lit_eval("R39.07.AdaptRichPow(g=2.0,boost=0.70)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.70, gamma=2.0))
    _lit_eval("R39.08.AdaptRichPow(g=2.0,boost=0.80)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.80, gamma=2.0))
    # Area 3: base_nw and thr sweep with gamma=2.0
    _lit_eval("R39.09.AdaptRichPow(g=2.0,base=0.35)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.35, richness_thr=0.4, max_boost=0.60, gamma=2.0))
    _lit_eval("R39.10.AdaptRichPow(g=2.0,base=0.45)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.45, richness_thr=0.4, max_boost=0.60, gamma=2.0))
    _lit_eval("R39.11.AdaptRichPow(g=2.0,thr=0.3)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.3, max_boost=0.60, gamma=2.0))
    _lit_eval("R39.12.AdaptRichPow(g=2.0,thr=0.5)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.5, max_boost=0.60, gamma=2.0))
    # Area 4: gamma=2.0 + SilenceCut
    _lit_eval("R39.13.AdaptPowSilence(g=2.0,sthr=0.10,sf=0.5)→cSEBBs",
              adapt_anchor_silence_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60,
                                       silence_thr=0.10, silence_factor=0.5))
    _lit_eval("R39.14.AdaptPowSilence(g=2.0,sthr=0.20,sf=0.5)→cSEBBs",
              adapt_anchor_silence_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60,
                                       silence_thr=0.20, silence_factor=0.5))
    # Area 5: gamma=2.0 + cp_thr=0.05
    _lit_eval("R39.15.AdaptRichPow(g=2.0,cp=0.05)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60,
                                        gamma=2.0, cp_thr=0.05))
    _lit_eval("R39.16.AdaptRichPow(g=2.0,cp=0.04)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60,
                                        gamma=2.0, cp_thr=0.04))
    # Area 6: DualPow — both branches with power-law gamma
    _lit_eval("R39.17.DualPow(gA=2.0,gB=1.0)→cSEBBs",
              adapt_anchor_dual_pow_cp(oof_final, base_nw_a=0.40, max_boost_a=0.60, gamma_a=2.0,
                                        base_nw_b=0.30, max_boost_b=0.20, gamma_b=1.0))
    _lit_eval("R39.18.DualPow(gA=2.0,gB=2.0)→cSEBBs",
              adapt_anchor_dual_pow_cp(oof_final, base_nw_a=0.40, max_boost_a=0.60, gamma_a=2.0,
                                        base_nw_b=0.30, max_boost_b=0.20, gamma_b=2.0))
    _lit_eval("R39.19.AdaptRichPow(g=3.0,boost=0.70)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.70, gamma=3.0))
    _lit_eval("R39.20.AdaptRichPow(g=2.0,g2.5,boost=0.70)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.70, gamma=2.5))

    # ── Round 38: AdaptAnchor power-law / dual adaptive / binary gate / best+SilenceCut ──────────
    # R37 verdict: boost=0.40/0.50/0.60 plateau at ~0.8057 (nw_max=1.0 boundary).
    # thr=0.4 optimal. AdaptAnchorSilence neutral at boost=0.30; untested at boost=0.60.
    # AdaptAnchorVar hurts at pen>0.1. DensityGate combo underwhelms.
    # R38 strategy:
    #   Area 1: Power-law richness (gamma<1 concave, gamma>1 convex) — breaks linear assumption
    #   Area 2: Dual adaptive branches (both A and B get richness-based nw)
    #   Area 3: Binary richness gate (step function vs linear ramp)
    #   Area 4: AdaptAnchor(boost=0.60) + SilenceCut (untested combination)
    #   Area 5: Extrapolation nw>1.0 (boost=0.70,0.80 with base=0.40)
    print("\n── Round 38: AdaptRichPow / DualAdaptBranch / BinaryGate / AdaptSilence(boost=0.60) ──")
    # Area 1: Power-law richness curve (gamma sweep with boost=0.60, thr=0.4)
    _lit_eval("R38.01.AdaptRichPow(g=0.5,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=0.5))
    _lit_eval("R38.02.AdaptRichPow(g=0.3,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=0.3))
    _lit_eval("R38.03.AdaptRichPow(g=2.0,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=2.0))
    _lit_eval("R38.04.AdaptRichPow(g=1.5,boost=0.60)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60, gamma=1.5))
    _lit_eval("R38.05.AdaptRichPow(g=0.5,boost=0.40)→cSEBBs",
              adapt_anchor_rich_pow_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.40, gamma=0.5))
    # Area 2: Dual adaptive branches
    _lit_eval("R38.06.DualAdapt(Ba=0.60,Bb=0.20)→cSEBBs",
              adapt_anchor_dual_branch_cp(oof_final, base_nw_a=0.40, max_boost_a=0.60,
                                           base_nw_b=0.30, max_boost_b=0.20))
    _lit_eval("R38.07.DualAdapt(Ba=0.60,Bb=0.30)→cSEBBs",
              adapt_anchor_dual_branch_cp(oof_final, base_nw_a=0.40, max_boost_a=0.60,
                                           base_nw_b=0.30, max_boost_b=0.30))
    _lit_eval("R38.08.DualAdapt(Ba=0.60,Bb=0.10)→cSEBBs",
              adapt_anchor_dual_branch_cp(oof_final, base_nw_a=0.40, max_boost_a=0.60,
                                           base_nw_b=0.30, max_boost_b=0.10))
    _lit_eval("R38.09.DualAdapt(Ba=0.40,Bb=0.40)→cSEBBs",
              adapt_anchor_dual_branch_cp(oof_final, base_nw_a=0.40, max_boost_a=0.40,
                                           base_nw_b=0.30, max_boost_b=0.40))
    # Area 3: Binary richness gate (step function)
    _lit_eval("R38.10.BinaryGate(hi=1.0,lo=0.30)→cSEBBs",
              adapt_anchor_binary_cp(oof_final, richness_thr=0.4, high_nw=1.0, low_nw=0.30))
    _lit_eval("R38.11.BinaryGate(hi=0.80,lo=0.40)→cSEBBs",
              adapt_anchor_binary_cp(oof_final, richness_thr=0.4, high_nw=0.80, low_nw=0.40))
    _lit_eval("R38.12.BinaryGate(hi=1.0,lo=0.40)→cSEBBs",
              adapt_anchor_binary_cp(oof_final, richness_thr=0.4, high_nw=1.0, low_nw=0.40))
    _lit_eval("R38.13.BinaryGate(hi=0.90,lo=0.30)→cSEBBs",
              adapt_anchor_binary_cp(oof_final, richness_thr=0.4, high_nw=0.90, low_nw=0.30))
    # Area 4: AdaptAnchor(boost=0.60) + SilenceCut (R37 only tested boost=0.30 with SilenceCut)
    _lit_eval("R38.14.AdaptSilence(boost=0.60,sthr=0.10,sf=0.5)→cSEBBs",
              adapt_anchor_silence_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60,
                                       silence_thr=0.10, silence_factor=0.5))
    _lit_eval("R38.15.AdaptSilence(boost=0.60,sthr=0.20,sf=0.5)→cSEBBs",
              adapt_anchor_silence_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60,
                                       silence_thr=0.20, silence_factor=0.5))
    _lit_eval("R38.16.AdaptSilence(boost=0.60,sthr=0.05,sf=0.5)→cSEBBs",
              adapt_anchor_silence_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60,
                                       silence_thr=0.05, silence_factor=0.5))
    # Area 5: Extrapolation nw>1.0 (boost > 0.60 means richest files get nw > 1.0)
    _lit_eval("R38.17.AdaptExtrap(boost=0.70)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.70))
    _lit_eval("R38.18.AdaptExtrap(boost=0.80)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.80))
    _lit_eval("R38.19.AdaptExtrap(boost=1.00)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=1.00))
    _lit_eval("R38.20.DualAdapt(Ba=0.60,Bb=0.60,base_b=0.10)→cSEBBs",
              adapt_anchor_dual_branch_cp(oof_final, base_nw_a=0.40, max_boost_a=0.60,
                                           base_nw_b=0.10, max_boost_b=0.60))

    # ── Round 37: AdaptAnchor fine-tune + AdaptAnchor+VarAnchor / Density / SilenceCut combos ─────
    # R36 verdict: AdaptAnchorRichness(boost=0.30,thr=0.4)=0.8054 NEW BEST (plateau broken!).
    # VarAnchor(pen=0.3)=0.8029, DensityGate(max=10,g=0.5)=0.8024. HMeanAnchor≤0.8001.
    # R37 strategy:
    #   (a) Fine-tune AdaptAnchor: thr in {0.5,0.6,0.7}, boost in {0.40,0.50}, base in {0.35,0.45},
    #       cp_thr in {0.04,0.05,0.08}
    #   (b) Combine AdaptAnchor + VarAnchor in one pass: richness-based nw + variance-scaled alpha
    #   (c) AdaptAnchor + DensityGate on 3D probs (OOD clip suppression after adaptive anchor)
    #   (d) AdaptAnchor + SilenceCut on 3D probs (file-level silence suppression)
    print("\n── Round 37: AdaptAnchor fine-tune / AdaptAnchorVar / AdaptAnchorDensity / AdaptAnchorSilence ──")
    # Area 1: richness_thr sweep (best: 0.4)
    _lit_eval("R37.01.AdaptAnchor(boost=0.30,thr=0.5)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.5, max_boost=0.30))
    _lit_eval("R37.02.AdaptAnchor(boost=0.30,thr=0.6)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.6, max_boost=0.30))
    _lit_eval("R37.03.AdaptAnchor(boost=0.30,thr=0.7)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.7, max_boost=0.30))
    # Area 2: max_boost sweep (best: 0.30)
    _lit_eval("R37.04.AdaptAnchor(boost=0.40,thr=0.4)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.40))
    _lit_eval("R37.05.AdaptAnchor(boost=0.50,thr=0.4)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.50))
    _lit_eval("R37.06.AdaptAnchor(boost=0.60,thr=0.4)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.60))
    # Area 3: base_nw sweep (best: 0.40)
    _lit_eval("R37.07.AdaptAnchor(base=0.35,boost=0.30,thr=0.4)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.35, richness_thr=0.4, max_boost=0.30))
    _lit_eval("R37.08.AdaptAnchor(base=0.45,boost=0.30,thr=0.4)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.45, richness_thr=0.4, max_boost=0.30))
    # Area 4: cp_thr sweep with best params
    _lit_eval("R37.09.AdaptAnchor(cp_thr=0.04)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30, cp_thr=0.04))
    _lit_eval("R37.10.AdaptAnchor(cp_thr=0.05)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30, cp_thr=0.05))
    _lit_eval("R37.11.AdaptAnchor(cp_thr=0.08)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30, cp_thr=0.08))
    # Area 5: AdaptAnchor + VarAnchor combined (richness-based nw + variance-scaled alpha)
    _lit_eval("R37.12.AdaptAnchorVar(pen=0.3)→cSEBBs",
              adapt_anchor_var_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30, var_penalty=0.3))
    _lit_eval("R37.13.AdaptAnchorVar(pen=0.5)→cSEBBs",
              adapt_anchor_var_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30, var_penalty=0.5))
    _lit_eval("R37.14.AdaptAnchorVar(pen=0.2,thr=0.5)→cSEBBs",
              adapt_anchor_var_cp(oof_final, base_nw=0.40, richness_thr=0.5, max_boost=0.30, var_penalty=0.2))
    # Area 6: AdaptAnchor + DensityGate on 3D probs
    _lit_eval("R37.15.AdaptAnchorDensity(max=10,g=0.5,athr=0.3)→cSEBBs",
              adapt_anchor_density_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30,
                                       max_active=10, gate=0.5, active_thr=0.3))
    _lit_eval("R37.16.AdaptAnchorDensity(max=8,g=0.3,athr=0.4)→cSEBBs",
              adapt_anchor_density_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30,
                                       max_active=8, gate=0.3, active_thr=0.4))
    # Area 7: AdaptAnchor + SilenceCut on 3D probs
    _lit_eval("R37.17.AdaptAnchorSilence(sthr=0.10,sf=0.5)→cSEBBs",
              adapt_anchor_silence_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30,
                                       silence_thr=0.10, silence_factor=0.5))
    _lit_eval("R37.18.AdaptAnchorSilence(sthr=0.20,sf=0.5)→cSEBBs",
              adapt_anchor_silence_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30,
                                       silence_thr=0.20, silence_factor=0.5))
    _lit_eval("R37.19.AdaptAnchorSilence(sthr=0.10,sf=0.25)→cSEBBs",
              adapt_anchor_silence_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30,
                                       silence_thr=0.10, silence_factor=0.25))
    _lit_eval("R37.20.AdaptAnchorVar(pen=0.1,thr=0.4)→cSEBBs",
              adapt_anchor_var_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30, var_penalty=0.1))

    # ── Round 36: Adaptive anchor / density gate / variance anchor / harmonic mean anchor ─────────
    # R35 verdict: ClipTopK(k=30)=0.8024, FileSpeciesPrior(n=30)=0.8020, ClipSharpen≤0.7969,
    # BidirConsistency=0.7790. Within-clip cross-species ops confirmed below plateau 0.8045.
    # R36 strategy: adapt the ANCHOR computation (currently fixed nor_w, max_w, alpha) to per-file
    # or per-species statistics. All 35 prior rounds use FIXED anchor params for ALL files/species.
    # The anchor is the strongest lever (BranchEns is best method), so making it adaptive is key:
    #   1) AdaptiveAnchorRichness: nor_w scales with file richness (# active species)
    #   2) DensityClipGate: gate entire OOD clips by active-species count
    #   3) VarianceAnchor: per-species alpha scales inversely with temporal variance
    #   4) HarmonicMeanAnchor: harmonic mean ("AND" gate) replaces NOR ("OR" gate) as anchor
    # Honest caveat (agent analysis): isotonic/rank normalization cannot help AUC mathematically.
    # These 4 methods are the only remaining candidates with non-trivial theoretical basis.
    print("\n── Round 36: AdaptiveAnchor / DensityClipGate / VarianceAnchor / HMeanAnchor ──")
    # Area 1: File-richness adaptive nor_w
    _lit_eval("R36.01.AdaptAnchor(base=0.40,boost=0.20,thr=0.3)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.3, max_boost=0.20))
    _lit_eval("R36.02.AdaptAnchor(base=0.40,boost=0.10,thr=0.3)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.3, max_boost=0.10))
    _lit_eval("R36.03.AdaptAnchor(base=0.35,boost=0.20,thr=0.2)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.35, richness_thr=0.2, max_boost=0.20))
    _lit_eval("R36.04.AdaptAnchor(base=0.40,boost=0.30,thr=0.4)→cSEBBs",
              adaptive_anchor_richness_cp(oof_final, base_nw=0.40, richness_thr=0.4, max_boost=0.30))
    # Area 2: Density clip gate (OOD clip suppression by active-species count)
    _lit_eval("R36.05.DensityGate(max=8,g=0.3,thr=0.4)→cSEBBs",
              density_clip_gate_cp(oof_final, max_active=8, gate=0.3, active_thr=0.4))
    _lit_eval("R36.06.DensityGate(max=5,g=0.3,thr=0.4)→cSEBBs",
              density_clip_gate_cp(oof_final, max_active=5, gate=0.3, active_thr=0.4))
    _lit_eval("R36.07.DensityGate(max=10,g=0.5,thr=0.3)→cSEBBs",
              density_clip_gate_cp(oof_final, max_active=10, gate=0.5, active_thr=0.3))
    _lit_eval("R36.08.DensityGate(max=8,g=0.1,thr=0.4)→cSEBBs",
              density_clip_gate_cp(oof_final, max_active=8, gate=0.1, active_thr=0.4))
    # Area 3: Variance-weighted anchor
    _lit_eval("R36.09.VarAnchor(pen=0.5)→cSEBBs",
              variance_anchor_cp(oof_final, var_penalty=0.5))
    _lit_eval("R36.10.VarAnchor(pen=0.3)→cSEBBs",
              variance_anchor_cp(oof_final, var_penalty=0.3))
    _lit_eval("R36.11.VarAnchor(pen=0.8)→cSEBBs",
              variance_anchor_cp(oof_final, var_penalty=0.8))
    # Area 4: Harmonic mean anchor
    _lit_eval("R36.12.HMeanAnchor(nw=0.40,a=0.38,b=5.15)→cSEBBs",
              hmean_anchor_cp(oof_final))
    _lit_eval("R36.13.HMeanAnchor(nw=0.30,a=0.40,b=6.0)→cSEBBs",
              hmean_anchor_cp(oof_final, nw_a=0.30, alpha_a=0.40, beta_a=6.0))
    _lit_eval("R36.14.HMeanAnchor(nw=0.50,a=0.35,b=5.15)→cSEBBs",
              hmean_anchor_cp(oof_final, nw_a=0.50, alpha_a=0.35, beta_a=5.15))
    # Area 5: Additional sweeps
    _lit_eval("R36.15.DensityGate(max=15,g=0.3,thr=0.4)→cSEBBs",
              density_clip_gate_cp(oof_final, max_active=15, gate=0.3, active_thr=0.4))
    _lit_eval("R36.16.HMeanAnchor(nw=0.40,a=0.50,b=5.15)→cSEBBs",
              hmean_anchor_cp(oof_final, nw_a=0.40, alpha_a=0.50, beta_a=5.15))

    # ── Round 35: Within-clip rank sharpening / bidirectional consistency / file-species prior ────
    # R34 verdict: NoiseFloor CATASTROPHIC (0.69), CooccGraph best=0.7857 (α=0.10), EnergyOOD=0.7934,
    # OnsetOffset=0.7988. Plateau 0.8045 holds after 620+ methods.
    # R35 strategy: FIRST within-clip CROSS-SPECIES operations — all 620 prior methods are either:
    #   a) per-species temporal (along 12-clip axis), or b) file-level scalar anchors.
    #   R35 exploits the 234-species axis within a single clip:
    #   1) ClipRankSharpen (BirdCLEF+ 2025 3rd): p^alpha per clip — concentrates mass on top species
    #   2) ClipTopKSuppress: keep top-k species per clip, suppress rest by gamma
    #   3) BidirConsistency (Voxaboxen arXiv:2503.02389): geom-mean of forward+reversed clip order
    #   4) FileSpeciesPrior: file-level top-N species mask (soundscapes: ~5-20 active/file)
    #   5) Combined ClipSharpen+TopK — jointly sharpened and sparsified
    print("\n── Round 35: ClipRankSharpen / ClipTopKSuppress / BidirConsistency / FileSpeciesPrior ──")
    # Area 1: Clip-level power sharpening
    _lit_eval("R35.01.ClipSharpen(a=1.5)→cSEBBs",
              clip_rank_sharpen_cp(oof_final, alpha=1.5))
    _lit_eval("R35.02.ClipSharpen(a=2.0)→cSEBBs",
              clip_rank_sharpen_cp(oof_final, alpha=2.0))
    _lit_eval("R35.03.ClipSharpen(a=3.0)→cSEBBs",
              clip_rank_sharpen_cp(oof_final, alpha=3.0))
    _lit_eval("R35.04.ClipSharpen(a=1.2)→cSEBBs",
              clip_rank_sharpen_cp(oof_final, alpha=1.2))
    # Area 2: Clip-level top-k suppress
    _lit_eval("R35.05.ClipTopK(k=30,g=0.1)→cSEBBs",
              clip_topk_suppress_cp(oof_final, top_k=30, gamma=0.1))
    _lit_eval("R35.06.ClipTopK(k=20,g=0.1)→cSEBBs",
              clip_topk_suppress_cp(oof_final, top_k=20, gamma=0.1))
    _lit_eval("R35.07.ClipTopK(k=10,g=0.1)→cSEBBs",
              clip_topk_suppress_cp(oof_final, top_k=10, gamma=0.1))
    _lit_eval("R35.08.ClipTopK(k=20,g=0.3)→cSEBBs",
              clip_topk_suppress_cp(oof_final, top_k=20, gamma=0.3))
    # Area 3: Bidirectional time-reversed consistency
    _lit_eval("R35.09.BidirConsist→cSEBBs",
              bidir_consistency_cp(oof_final))
    _lit_eval("R35.10.BidirConsist(b_b=5.5,nw_b=0.35)→cSEBBs",
              bidir_consistency_cp(oof_final, beta_b=5.5, nw_b=0.35))
    # Area 4: File-level species count prior
    _lit_eval("R35.11.FileSpeciesPrior(n=30,g=0.2)→cSEBBs",
              file_species_prior_cp(oof_final, top_species=30, gamma=0.2))
    _lit_eval("R35.12.FileSpeciesPrior(n=20,g=0.2)→cSEBBs",
              file_species_prior_cp(oof_final, top_species=20, gamma=0.2))
    _lit_eval("R35.13.FileSpeciesPrior(n=40,g=0.1)→cSEBBs",
              file_species_prior_cp(oof_final, top_species=40, gamma=0.1))
    _lit_eval("R35.14.FileSpeciesPrior(n=25,g=0.3)→cSEBBs",
              file_species_prior_cp(oof_final, top_species=25, gamma=0.3))
    # Area 5: Combined clip sharpen + top-k suppress
    _lit_eval("R35.15.ClipSharpTopK(a=2.0,k=20,g=0.1)→cSEBBs",
              clip_rank_sharpen_topk_cp(oof_final, alpha=2.0, top_k=20, gamma=0.1))
    _lit_eval("R35.16.ClipSharpTopK(a=1.5,k=30,g=0.1)→cSEBBs",
              clip_rank_sharpen_topk_cp(oof_final, alpha=1.5, top_k=30, gamma=0.1))
    _lit_eval("R35.17.ClipSharpTopK(a=2.0,k=15,g=0.05)→cSEBBs",
              clip_rank_sharpen_topk_cp(oof_final, alpha=2.0, top_k=15, gamma=0.05))
    # Area 6: Additional sweeps
    _lit_eval("R35.18.ClipTopK(k=50,g=0.1)→cSEBBs",
              clip_topk_suppress_cp(oof_final, top_k=50, gamma=0.1))
    _lit_eval("R35.19.FileSpeciesPrior(n=15,g=0.2)→cSEBBs",
              file_species_prior_cp(oof_final, top_species=15, gamma=0.2))
    _lit_eval("R35.20.ClipSharpTopK(a=3.0,k=20,g=0.1)→cSEBBs",
              clip_rank_sharpen_topk_cp(oof_final, alpha=3.0, top_k=20, gamma=0.1))

    # ── Round 34: Cross-class co-occurrence graph / noise-floor norm / energy-OOD / onset-offset ──
    # R33 verdict: GeoBranchEns=0.8045 (same as arith), SilenceCut(thr=0.20)=0.8045. No breakthrough.
    # Plateau confirmed at 0.8045 after 600+ methods. Fold-0 OOF AUC≈0.691 is the model-quality floor.
    # R34 strategy: four fundamentally different directions from literature:
    #   1) Co-occurrence Graph (Chen et al. CIS 2024): cross-class cosine-sim graph smooth — first method
    #      to cross the 234-class boundary; all prior methods operate per-class independently
    #   2) Noise-floor normalization (nSEBBs, Zerroug et al. arXiv:2505.11889): subtract per-file
    #      noise floor (mean of bottom-k clips) then rescale — removes file-level DC bias
    #   3) Energy OOD gate (Liu et al. NeurIPS 2021 + Xue et al. APSIPA 2025 arXiv:2507.09606):
    #      detect OOD clips via -logsumexp energy; blend toward file mean for OOD clips
    #   4) Onset/Offset asymmetric smooth (Dinkel & Wang arXiv:2601.04178, ICASSP 2026):
    #      forward-fill after onset, backward-fill before offset — first asymmetric method in 600+
    print("\n── Round 34: CooccurrenceGraph / NoiseFloorNorm / EnergyOOD / OnsetOffset ──")
    # Area 1: Co-occurrence graph smooth (alpha = graph blend weight)
    _lit_eval("R34.01.CooccGraph(a=0.10)→cSEBBs",
              cooccurrence_graph_cp(oof_final, alpha=0.10))
    _lit_eval("R34.02.CooccGraph(a=0.20)→cSEBBs",
              cooccurrence_graph_cp(oof_final, alpha=0.20))
    _lit_eval("R34.03.CooccGraph(a=0.30)→cSEBBs",
              cooccurrence_graph_cp(oof_final, alpha=0.30))
    _lit_eval("R34.04.CooccGraph(a=0.50)→cSEBBs",
              cooccurrence_graph_cp(oof_final, alpha=0.50))
    _lit_eval("R34.05.CooccGraph(a=0.20,b_b=5.5,nw_b=0.35)→cSEBBs",
              cooccurrence_graph_cp(oof_final, alpha=0.20, beta_b=5.5, nw_b=0.35))
    # Area 2: Noise-floor normalization (k_frac = fraction of clips used for noise estimate)
    _lit_eval("R34.06.NoiseFloor(k=0.17)→cSEBBs",
              noise_floor_norm_cp(oof_final, k_frac=0.17))
    _lit_eval("R34.07.NoiseFloor(k=0.25)→cSEBBs",
              noise_floor_norm_cp(oof_final, k_frac=0.25))
    _lit_eval("R34.08.NoiseFloor(k=0.33)→cSEBBs",
              noise_floor_norm_cp(oof_final, k_frac=0.33))
    _lit_eval("R34.09.NoiseFloor(k=0.10)→cSEBBs",
              noise_floor_norm_cp(oof_final, k_frac=0.10))
    # Area 3: Energy OOD gate (gate_strength = how much to pull OOD clips toward file mean)
    _lit_eval("R34.10.EnergyOOD(gs=0.30,tp=80)→cSEBBs",
              energy_ood_gate_cp(oof_final, gate_strength=0.30, thresh_pct=80))
    _lit_eval("R34.11.EnergyOOD(gs=0.50,tp=80)→cSEBBs",
              energy_ood_gate_cp(oof_final, gate_strength=0.50, thresh_pct=80))
    _lit_eval("R34.12.EnergyOOD(gs=0.80,tp=80)→cSEBBs",
              energy_ood_gate_cp(oof_final, gate_strength=0.80, thresh_pct=80))
    _lit_eval("R34.13.EnergyOOD(gs=0.50,tp=90)→cSEBBs",
              energy_ood_gate_cp(oof_final, gate_strength=0.50, thresh_pct=90))
    # Area 4: Onset/offset asymmetric temporal smooth
    _lit_eval("R34.14.OnsetOffset(fw=1,bw=2,bl=0.20)→cSEBBs",
              onset_offset_asymmetric_cp(oof_final, onset_fw=1, offset_bw=2, blend=0.20))
    _lit_eval("R34.15.OnsetOffset(fw=1,bw=2,bl=0.30)→cSEBBs",
              onset_offset_asymmetric_cp(oof_final, onset_fw=1, offset_bw=2, blend=0.30))
    _lit_eval("R34.16.OnsetOffset(fw=2,bw=3,bl=0.20)→cSEBBs",
              onset_offset_asymmetric_cp(oof_final, onset_fw=2, offset_bw=3, blend=0.20))
    _lit_eval("R34.17.OnsetOffset(fw=2,bw=3,bl=0.30)→cSEBBs",
              onset_offset_asymmetric_cp(oof_final, onset_fw=2, offset_bw=3, blend=0.30))
    # Area 5: Additional parameter sweeps
    _lit_eval("R34.18.EnergyOOD(gs=0.50,tp=70)→cSEBBs",
              energy_ood_gate_cp(oof_final, gate_strength=0.50, thresh_pct=70))
    _lit_eval("R34.19.NoiseFloor(k=0.25,b_b=5.5,nw_b=0.35)→cSEBBs",
              noise_floor_norm_cp(oof_final, k_frac=0.25, beta_b=5.5, nw_b=0.35))
    _lit_eval("R34.20.OnsetOffset(fw=1,bw=1,bl=0.30)→cSEBBs",
              onset_offset_asymmetric_cp(oof_final, onset_fw=1, offset_bw=1, blend=0.30))

    # ── Round 32: Per-class adaptive median / morphological closing / event decay ──────────────
    # R31 verdict: RankBlend catastrophic (0.70-0.74), EntrPriorGate/CrossFileCalib/ProbOR all < 0.8045.
    # Plateau confirmed at 0.8045 after 539 methods.
    # R32 strategy: two DCASE 2024-sourced techniques genuinely not yet tested:
    #   1) Per-class adaptive median (DCASE 2024 Task 4 baseline, arXiv:2406.08056)
    #      w_c from per-class label run-length stats — long-call species get wider window
    #   2) Morphological closing (dilation then erosion) fills internal gaps without extending ends
    #      scipy.ndimage maximum_filter1d → minimum_filter1d per class
    #   Both combined with BranchEns (our best anchor method) via +cSEBBs
    print("\n── Round 32: PerClassAdaptiveMedian / MorphClose / combos ──")
    # Area 1: Per-class adaptive median (width from label run-length stats)
    _lit_eval("R32.01.PerClsMed(s=1.0,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              per_class_adaptive_median_cp(oof_final, scale=1.0, alpha=0.38, nor_w=0.40, beta=5.15))
    _lit_eval("R32.02.PerClsMed(s=0.5,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              per_class_adaptive_median_cp(oof_final, scale=0.5, alpha=0.38, nor_w=0.40, beta=5.15))
    _lit_eval("R32.03.PerClsMed(s=0.75,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              per_class_adaptive_median_cp(oof_final, scale=0.75, alpha=0.38, nor_w=0.40, beta=5.15))
    _lit_eval("R32.04.PerClsMed(s=1.0,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              per_class_adaptive_median_cp(oof_final, scale=1.0, alpha=0.40, nor_w=0.30, beta=6.0))
    _lit_eval("R32.05.PerClsMed(s=0.5,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              per_class_adaptive_median_cp(oof_final, scale=0.5, alpha=0.40, nor_w=0.30, beta=6.0))
    # Area 1b: Per-class adaptive median inside BranchEns
    _lit_eval("R32.06.PerClsMedBranchEns(s=1.0,w=0.55)→cSEBBs",
              per_class_adaptive_median_branchens_cp(oof_final, scale=1.0, w_a=0.55))
    _lit_eval("R32.07.PerClsMedBranchEns(s=0.5,w=0.55)→cSEBBs",
              per_class_adaptive_median_branchens_cp(oof_final, scale=0.5, w_a=0.55))
    _lit_eval("R32.08.PerClsMedBranchEns(s=0.75,w=0.55)→cSEBBs",
              per_class_adaptive_median_branchens_cp(oof_final, scale=0.75, w_a=0.55))
    _lit_eval("R32.09.PerClsMedBranchEns(s=1.0,w=0.50)→cSEBBs",
              per_class_adaptive_median_branchens_cp(oof_final, scale=1.0, w_a=0.50))
    _lit_eval("R32.10.PerClsMedBranchEns(s=0.5,w=0.50)→cSEBBs",
              per_class_adaptive_median_branchens_cp(oof_final, scale=0.5, w_a=0.50))
    # Area 2: Morphological closing (dilation→erosion, fills internal gaps)
    _lit_eval("R32.11.MorphClose(r=1,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              morphological_close_dual_anchor_cp(oof_final, close_r=1, alpha=0.38, nor_w=0.40, beta=5.15))
    _lit_eval("R32.12.MorphClose(r=2,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              morphological_close_dual_anchor_cp(oof_final, close_r=2, alpha=0.38, nor_w=0.40, beta=5.15))
    _lit_eval("R32.13.MorphClose(r=1,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              morphological_close_dual_anchor_cp(oof_final, close_r=1, alpha=0.40, nor_w=0.30, beta=6.0))
    # Area 2b: Morphological closing inside BranchEns
    _lit_eval("R32.14.MorphCloseBranchEns(r=1,w=0.55)→cSEBBs",
              morphological_close_branchens_cp(oof_final, close_r=1, w_a=0.55))
    _lit_eval("R32.15.MorphCloseBranchEns(r=2,w=0.55)→cSEBBs",
              morphological_close_branchens_cp(oof_final, close_r=2, w_a=0.55))
    _lit_eval("R32.16.MorphCloseBranchEns(r=1,w=0.50)→cSEBBs",
              morphological_close_branchens_cp(oof_final, close_r=1, w_a=0.50))
    _lit_eval("R32.17.MorphCloseBranchEns(r=2,w=0.50)→cSEBBs",
              morphological_close_branchens_cp(oof_final, close_r=2, w_a=0.50))
    # Area 3: Opening (erosion→dilation, removes isolated spikes — inverse of closing)
    _lit_eval("R32.18.MorphOpen(r=1,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              morphological_close_dual_anchor_cp(oof_final, close_r=1, alpha=0.38, nor_w=0.40, beta=5.15))
    _lit_eval("R32.19.PerClsMed+MorphClose(s=0.5,r=1,w=0.55)→cSEBBs",
              per_class_adaptive_median_branchens_cp(oof_final, scale=0.5, w_a=0.55))
    _lit_eval("R32.20.PerClsMed+MorphClose(s=1.0,r=1,w=0.55)→cSEBBs",
              per_class_adaptive_median_branchens_cp(oof_final, scale=1.0, w_a=0.55))

    # ── Round 31: Rank-norm within file / entropy-prior gate / cross-file calib / ProbOR+BranchEns ──
    # R30 verdict: Bayesian 0.7930, Q90 0.8034-0.8039, MixedQ90Ens 0.8039. All below 0.8045.
    # Plateau confirmed at 0.8045 after 519 methods. Post-processing ceiling likely reached.
    # R31 strategy: 3 genuinely novel ideas from BirdCLEF competition literature:
    #   1) Rank-norm within file (Quantile-Mix, BirdCLEF 2025 top-2%): equalize marginal dist
    #   2) Entropy-prior gate (FINCH-style): inject file prior proportional to clip confidence
    #   3) Cross-file class calibration: normalize by global class activity
    #   4) ProbOR additive boost on BranchEns (BirdCLEF 2024 3rd)
    print("\n── Round 31: Rank-norm / entropy-prior gate / cross-file calib / ProbOR+Ens ──")
    # Area 1: RankBlend (rank-normalize within file, blend with raw before DualAnchor)
    _lit_eval("R31.01.RankBlend(rb=0.1,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              rank_blend_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, rank_blend=0.1))
    _lit_eval("R31.02.RankBlend(rb=0.2,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              rank_blend_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, rank_blend=0.2))
    _lit_eval("R31.03.RankBlend(rb=0.3,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              rank_blend_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, rank_blend=0.3))
    _lit_eval("R31.04.RankBlend(rb=0.5,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              rank_blend_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, rank_blend=0.5))
    _lit_eval("R31.05.RankBlend(rb=0.3,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              rank_blend_dual_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0, rank_blend=0.3))
    # Area 1b: RankBlend inside BranchEns branches
    _lit_eval("R31.06.RankBranchEns(rb=0.1,w=0.55)→cSEBBs",
              rank_blend_branch_ensemble_cp(oof_final, rank_blend=0.1, w_a=0.55))
    _lit_eval("R31.07.RankBranchEns(rb=0.2,w=0.55)→cSEBBs",
              rank_blend_branch_ensemble_cp(oof_final, rank_blend=0.2, w_a=0.55))
    _lit_eval("R31.08.RankBranchEns(rb=0.3,w=0.55)→cSEBBs",
              rank_blend_branch_ensemble_cp(oof_final, rank_blend=0.3, w_a=0.55))
    _lit_eval("R31.09.RankBranchEns(rb=0.1,w=0.50)→cSEBBs",
              rank_blend_branch_ensemble_cp(oof_final, rank_blend=0.1, w_a=0.50))
    _lit_eval("R31.10.RankBranchEns(rb=0.2,w=0.60)→cSEBBs",
              rank_blend_branch_ensemble_cp(oof_final, rank_blend=0.2, w_a=0.60))
    # Area 2: Entropy-prior gate (confident clips inject file-level class prior)
    _lit_eval("R31.11.EntrPrior(gw=0.1,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              entropy_prior_gate_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, gate_w=0.1))
    _lit_eval("R31.12.EntrPrior(gw=0.2,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              entropy_prior_gate_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, gate_w=0.2))
    _lit_eval("R31.13.EntrPrior(gw=0.3,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              entropy_prior_gate_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, gate_w=0.3))
    _lit_eval("R31.14.EntrPrior(gw=0.5,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              entropy_prior_gate_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, gate_w=0.5))
    _lit_eval("R31.15.EntrPrior(gw=0.2,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              entropy_prior_gate_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0, gate_w=0.2))
    # Area 3: Cross-file class calibration (boost rare-globally classes per-file)
    _lit_eval("R31.16.CrossFileCal(cw=0.05,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              cross_file_calibration_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, cal_w=0.05))
    _lit_eval("R31.17.CrossFileCal(cw=0.1,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              cross_file_calibration_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, cal_w=0.1))
    _lit_eval("R31.18.CrossFileCal(cw=0.2,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              cross_file_calibration_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, cal_w=0.2))
    # Area 4: ProbOR additive boost after BranchEns (BirdCLEF 2024 3rd place technique)
    _lit_eval("R31.19.ProbOR+BranchEns(pw=0.05,w=0.55)→cSEBBs",
              probor_on_branch_ensemble_cp(oof_final, w_a=0.55, probor_w=0.05))
    _lit_eval("R31.20.ProbOR+BranchEns(pw=0.10,w=0.55)→cSEBBs",
              probor_on_branch_ensemble_cp(oof_final, w_a=0.55, probor_w=0.10))

    # ── Round 30: Bayesian update / quantile anchor / score-weighted mean ─────────
    # R29 verdict: LocalNOR HURTS (-0.009), GaussNOR weak (0.8020), LocalGlobalEns weak (0.7993).
    #              Global NOR/max is essential — spatially local anchors cannot match it.
    # R30 strategy: new mathematical update rules for anchor blending (not anchor TYPE).
    #   1) Bayesian multiplicative update (log-odds addition vs. linear prob blend)
    #   2) Quantile anchor Q90/Q75 (vs GlobalMax): robust to one outlier clip
    #   3) Score-weighted mean (sum(p^2)/sum(p)): attention-concentrated file stat
    print("\n── Round 30: Bayesian update / Q90 anchor / score-weighted mean ──")
    # Area 1: Bayesian (log-odds additive) anchor update
    _lit_eval("R30.01.Bayes(a=0.20,nw=0.40,b=5.15)→cSEBBs",
              bayesian_dual_anchor_cp(oof_final, alpha=0.20, nor_w=0.40, beta=5.15))
    _lit_eval("R30.02.Bayes(a=0.30,nw=0.40,b=5.15)→cSEBBs",
              bayesian_dual_anchor_cp(oof_final, alpha=0.30, nor_w=0.40, beta=5.15))
    _lit_eval("R30.03.Bayes(a=0.40,nw=0.40,b=5.15)→cSEBBs",
              bayesian_dual_anchor_cp(oof_final, alpha=0.40, nor_w=0.40, beta=5.15))
    _lit_eval("R30.04.Bayes(a=0.30,nw=0.30,b=6.0)→cSEBBs",
              bayesian_dual_anchor_cp(oof_final, alpha=0.30, nor_w=0.30, beta=6.0))
    _lit_eval("R30.05.Bayes(a=0.40,nw=0.30,b=6.0)→cSEBBs",
              bayesian_dual_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0))
    # Bayesian BranchEns: both branches use log-odds anchor update
    _lit_eval("R30.06.BayesEns(A+B,w=0.5)→cSEBBs",
              bayesian_branch_ensemble_cp(oof_final, w_a=0.5))
    _lit_eval("R30.07.BayesEns(A+B,w=0.55)→cSEBBs",
              bayesian_branch_ensemble_cp(oof_final, w_a=0.55))
    # Area 2: Quantile anchor Q90 (more robust than pure GlobalMax with 12 clips)
    _lit_eval("R30.08.Q90(nw=0.40,a=0.38,b=5.15)→cSEBBs",
              quantile_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, q=0.90))
    _lit_eval("R30.09.Q75(nw=0.40,a=0.38,b=5.15)→cSEBBs",
              quantile_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, q=0.75))
    _lit_eval("R30.10.Q90(nw=0.30,a=0.40,b=6.0)→cSEBBs",
              quantile_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0, q=0.90))
    _lit_eval("R30.11.Q83(nw=0.40,a=0.38,b=5.15)→cSEBBs",
              quantile_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, q=0.833))
    # Area 3: Score-weighted mean anchor sum(p^2)/sum(p)
    _lit_eval("R30.12.ScoreWt(nw=0.40,a=0.38,b=5.15)→cSEBBs",
              score_weighted_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15))
    _lit_eval("R30.13.ScoreWt(nw=0.30,a=0.40,b=6.0)→cSEBBs",
              score_weighted_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0))
    _lit_eval("R30.14.ScoreWt(nw=0.40,a=0.45,b=5.15)→cSEBBs",
              score_weighted_anchor_cp(oof_final, alpha=0.45, nor_w=0.40, beta=5.15))
    # Area 4: Mixed BranchEns — Branch A=standard, Branch B=Q90 anchor
    _lit_eval("R30.15.MixedEns(A+Q90B,w=0.5)→cSEBBs",
              mixed_anchor_branch_ensemble_cp(oof_final, w_a=0.5, q=0.90))
    _lit_eval("R30.16.MixedEns(A+Q75B,w=0.5)→cSEBBs",
              mixed_anchor_branch_ensemble_cp(oof_final, w_a=0.5, q=0.75))
    _lit_eval("R30.17.MixedEns(A+Q90B,w=0.6)→cSEBBs",
              mixed_anchor_branch_ensemble_cp(oof_final, w_a=0.6, q=0.90))
    # Area 5: Cross-concept combos — Bayesian with different configs
    _lit_eval("R30.18.Bayes(a=0.50,nw=0.40,b=5.15)→cSEBBs",
              bayesian_dual_anchor_cp(oof_final, alpha=0.50, nor_w=0.40, beta=5.15))
    _lit_eval("R30.19.Bayes(a=0.20,nw=0.30,b=6.0)→cSEBBs",
              bayesian_dual_anchor_cp(oof_final, alpha=0.20, nor_w=0.30, beta=6.0))
    _lit_eval("R30.20.ScoreWt(nw=0.40,a=0.38,b=6.0)→cSEBBs",
              score_weighted_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=6.0))

    # ── Round 29: local NOR / Gaussian-weighted NOR / two-scale / local-branch ──
    # R28 verdict: GeomAnchor=0.8038, TopK=0.8036, PostLSE=0.8010, DoublecP=0.8035.
    #              PowerMean much worse. Plateau at 0.8045 holds after 479 methods.
    # KEY INSIGHT: Fold 0 is stuck at 0.691; Fold 2 is ALREADY at ceiling (0.867 w/ Gaussian!).
    # R29 strategy: spatially LOCAL anchors — anchor varies per clip position.
    # Global NOR/max is constant across file; local NOR adapts to temporal context.
    print("\n── Round 29: local NOR / Gauss-NOR / two-scale / local-branch ensemble ──")
    # Area 1: Local NOR anchor (window over neighbouring clips only)
    _lit_eval("R29.01.LocalNOR(w=5,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              local_nor_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, window=5))
    _lit_eval("R29.02.LocalNOR(w=7,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              local_nor_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, window=7))
    _lit_eval("R29.03.LocalNOR(w=9,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              local_nor_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, window=9))
    _lit_eval("R29.04.LocalNOR(w=5,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              local_nor_dual_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0, window=5))
    _lit_eval("R29.05.LocalNOR(w=7,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              local_nor_dual_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0, window=7))
    # Area 2: Gaussian-weighted NOR (smooth version of local NOR)
    _lit_eval("R29.06.GaussNOR(σ=1.5,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              gauss_nor_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, sigma=1.5))
    _lit_eval("R29.07.GaussNOR(σ=2.0,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              gauss_nor_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, sigma=2.0))
    _lit_eval("R29.08.GaussNOR(σ=3.0,nw=0.40,a=0.38,b=5.15)→cSEBBs",
              gauss_nor_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15, sigma=3.0))
    _lit_eval("R29.09.GaussNOR(σ=2.0,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              gauss_nor_dual_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0, sigma=2.0))
    # Area 3: Two-scale NOR (blend of local + global NOR)
    _lit_eval("R29.10.TwoScaleNOR(lw=0.5,w=5,nw=0.40,a=0.38)→cSEBBs",
              twoscale_nor_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15,
                                          local_w=0.5, window=5))
    _lit_eval("R29.11.TwoScaleNOR(lw=0.3,w=5,nw=0.40,a=0.38)→cSEBBs",
              twoscale_nor_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15,
                                          local_w=0.3, window=5))
    _lit_eval("R29.12.TwoScaleNOR(lw=0.7,w=5,nw=0.40,a=0.38)→cSEBBs",
              twoscale_nor_dual_anchor_cp(oof_final, alpha=0.38, nor_w=0.40, beta=5.15,
                                          local_w=0.7, window=5))
    _lit_eval("R29.13.TwoScaleNOR(lw=0.5,w=5,nw=0.30,a=0.40,b=6.0)→cSEBBs",
              twoscale_nor_dual_anchor_cp(oof_final, alpha=0.40, nor_w=0.30, beta=6.0,
                                          local_w=0.5, window=5))
    # Area 4: Local-Global BranchEns (A=global, B=local NOR+LocalMax)
    _lit_eval("R29.14.LocalGlobalEns(A+B,w=0.5)→cSEBBs",
              local_global_branch_ensemble_cp(oof_final, w_a=0.5, window_b=5))
    _lit_eval("R29.15.LocalGlobalEns(A+B,w=0.6)→cSEBBs",
              local_global_branch_ensemble_cp(oof_final, w_a=0.6, window_b=5))
    _lit_eval("R29.16.LocalGlobalEns(A+B,w=0.4)→cSEBBs",
              local_global_branch_ensemble_cp(oof_final, w_a=0.4, window_b=5))
    # Area 5: TwoScaleNOR BranchEns (A=global, B=TwoScaleNOR)
    _lit_eval("R29.17.TwoScaleEns(A+Btwo,lw=0.5,w=0.5)→cSEBBs",
              local_global_full_branch_ensemble_cp(oof_final, w_a=0.5, local_w=0.5))
    _lit_eval("R29.18.TwoScaleEns(A+Btwo,lw=0.3,w=0.5)→cSEBBs",
              local_global_full_branch_ensemble_cp(oof_final, w_a=0.5, local_w=0.3))
    _lit_eval("R29.19.TwoScaleEns(A+Btwo,lw=0.7,w=0.5)→cSEBBs",
              local_global_full_branch_ensemble_cp(oof_final, w_a=0.5, local_w=0.7))
    # Area 6: LocalNOR BranchEns (both branches local NOR, different window)
    _lit_eval("R29.20.LocalGlobalEns(w7_b,w=0.5)→cSEBBs",
              local_global_branch_ensemble_cp(oof_final, w_a=0.5, window_b=7))

    # ── Individual summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Individual Methods (sorted by mean AUC)")
    print("=" * 70)
    ranked_indiv = sorted(results.items(), key=lambda x: -x[1]["mean"])
    gauss_mean   = results.get("2.Gaussian (fixed)", {}).get("mean", 0.0)
    for rank, (name, r) in enumerate(ranked_indiv, 1):
        delta  = r["mean"] - gauss_mean
        marker = " ⭐" if delta > 0 else ""
        print(f"  #{rank:2d}  {name:35s}  mean={r['mean']:.4f}  vs Gaussian {delta:+.4f}{marker}")

    # ── SmootherEnsemble combinations ─────────────────────────────────────────
    if not args.skip_ensemble:
        # Only use learnable methods (exclude raw probe/gaussian — already input to all)
        ENS_METHODS = [
            ("conv",     "conv_gauss"),
            ("sos",      "sos"),
            ("multiscl", "multiscale"),
            ("bilateral","bilateral05"),
            ("iir",      "causal_iir"),
            ("onset",    "deriv_onset"),
            ("tophat",   "soft_tophat"),
        ]
        from itertools import combinations as _combos
        ens_results = {}
        n_total = sum(
            len(list(_combos(ENS_METHODS, k)))
            for k in range(2, len(ENS_METHODS) + 1)
        )
        exp_num = 0
        print(f"\n── SmootherEnsemble: {n_total} combinations ({'skip with --skip_ensemble'}) ──")
        print(f"{'Ensemble':42s}  {'f0':>6}  {'f1':>6}  {'f2':>6}  {'f3':>6}  {'mean':>7}")
        print("-" * 80)
        for k in range(2, len(ENS_METHODS) + 1):
            for combo in _combos(ENS_METHODS, k):
                labels    = [c[0] for c in combo]
                keys      = [c[1] for c in combo]
                ens_name  = f"E{exp_num:02d}.{'+'  .join(labels)}"
                exp_num  += 1
                fold_aucs = eval_smoother_ensemble(
                    keys, smoothed_logits, Y, fold_id,
                    epochs=args.ens_epochs, lr=0.1, l2=1e-3,
                )
                results[ens_name] = fold_aucs
                ens_results[ens_name] = fold_aucs
                delta  = fold_aucs["mean"] - gauss_mean
                marker = " ⭐" if delta > 0 else ""
                f_str  = "  ".join(f"{fold_aucs.get(f'fold{j}', 0):.4f}" for j in range(4))
                print(f"  {ens_name:40s}  {f_str}  mean={fold_aucs['mean']:.4f}  {delta:+.4f}{marker}")

        # Best ensemble
        if ens_results:
            best_ens_name, best_ens = max(ens_results.items(), key=lambda x: x[1]["mean"])
            print(f"\nBest ensemble: {best_ens_name} (mean={best_ens['mean']:.4f})")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Final Summary (all methods, sorted by mean AUC)")
    print("=" * 70)
    ranked = sorted(results.items(), key=lambda x: -x[1]["mean"])
    for rank, (name, r) in enumerate(ranked, 1):
        delta  = r["mean"] - gauss_mean
        marker = " ⭐" if delta > 0 else ""
        print(f"  #{rank:3d}  {name:42s}  mean={r['mean']:.4f}  {delta:+.4f}{marker}")

    # ── Save results ───────────────────────────────────────────────────────────
    import json
    out_path = Path("outputs/smooth_experiments_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"epochs": args.epochs, "ens_epochs": args.ens_epochs,
                   "results": results}, f, indent=2)
    print(f"\nResults saved → {out_path}")

    best_name, best_r = ranked[0]
    print(f"\nBest overall: {best_name} (mean AUC={best_r['mean']:.4f})")


if __name__ == "__main__":
    main()
