"""
train_stacker_v3_ss.py — Semi-supervised stacking ensemble meta-learner for BirdCLEF 2026.

Extends train_stacker_v3.py by adding ~127,896 pseudo-labelled rows from all 10,658
train soundscapes.  No new inference is run — existing CSV files serve as feature proxies.

Feature layout (identical to v3, 5 × 234 = 1170 dims):
  perch_raw | perch_prior_fused | mlp_probe | proto_ssm | sed_csebbs

Pseudo feature sources:
  PSEUDO_PERCH   → ns_r0_perch_aug.csv    (8,354 files, Perch probs)
  PSEUDO_PROTO   → ns_r0_protossm.csv     (~8,537 files, ProtoSSM probs)
  PSEUDO_SED_ENS → finetune_v3_ss_pseudo.csv (10,658 files, SED ensemble probs)

Architectures (9 total, same as v3):
  1. LGBM          (context window, 3510 dim)
  2. XGBoost       (context window, 3510 dim)
  3. MLP           (context window, reshaped to (3, 5, 234))
  4. BiGRU         (sequence, (12, 1170))
  5. TCN           (sequence, (12, 1170))
  6. Transformer   (sequence, (12, 1170))
  7. SSM           (sequence, (12, 1170))
  8. FT-Transformer (sequence, (12, 1170))
  9. CNN1D         (sequence, (12, 1170))

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_stacker_v3_ss.py

Output dir: birdclef-2026/notebook resource/current_subs 2/stacker_weights/
All artifacts use the _ss suffix (e.g. stacker_lgbm_ss.pkl).
"""

import os
import gc
import json
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import onnxruntime as ort

from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from xgboost import XGBClassifier

import torchaudio
import torchaudio.transforms as T

import wandb
import openpyxl

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/lab/BirdClef-2026-Codebase")
NB_DIR      = BASE_DIR / "birdclef-2026" / "notebook resource" / "current_subs 2"
PERCH_META  = NB_DIR / "perch meta"
WEIGHTS_DIR = NB_DIR / "weights"
OUT_DIR     = NB_DIR / "stacker_weights"
OUTPUTS     = BASE_DIR / "outputs"

# Pseudo label CSV paths
PSEUDO_PERCH   = BASE_DIR / "pseudo_labels" / "ns_r0_perch_aug.csv"
PSEUDO_PROTO   = BASE_DIR / "pseudo_labels" / "ns_r0_protossm.csv"
PSEUDO_SED_ENS = BASE_DIR / "pseudo_labels" / "finetune_v3_ss_pseudo.csv"

# Intermediate cache for pseudo features
CACHE_PSEUDO = OUTPUTS / "stacker_pseudo_features.npz"

# Normalization stats saved by v3 (do NOT recompute — reuse same stats)
NORM_STATS_PATH = OUT_DIR / "stacker_norm_v3.npz"

# SED feature cache from v3 (for labeled 59 files)
CACHE_SED = OUTPUTS / "stacker_train_sed_csebbs_v3.npy"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
SEED         = 42
SR           = 32_000
N_WIN        = 12           # 12 × 5 s = 60 s
WIN_SAMPLES  = SR * 5
N_CLASSES    = 234
N_MODELS     = 5            # perch_raw | perch_prior_fused | mlp_probe | proto_ssm | sed_csebbs
EPS          = 1e-6
FEAT_DIM     = N_MODELS * N_CLASSES   # 1170

# Multi-chunk context window
CONTEXT_K    = 1
CONTEXT_SIZE = 2 * CONTEXT_K + 1   # 3
CTX_FEAT_DIM = CONTEXT_SIZE * FEAT_DIM   # 3510

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] device={DEVICE}  out={OUT_DIR}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Primary label order (234 classes) ───────────────────────────────────────
_sample_sub     = pd.read_csv(BASE_DIR / "birdclef-2026" / "sample_submission.csv", nrows=0)
PRIMARY_LABELS  = _sample_sub.columns[1:].tolist()   # 234 labels
assert len(PRIMARY_LABELS) == N_CLASSES, f"Expected {N_CLASSES} labels, got {len(PRIMARY_LABELS)}"
print(f"[config] PRIMARY_LABELS count={len(PRIMARY_LABELS)}")

# ─── W&B init ──────────────────────────────────────────────────────────────────
wandb.init(
    project="birdclef-2026",
    name="stacker-v3-ss",
    config={
        "n_models"       : N_MODELS,
        "n_classes"      : N_CLASSES,
        "n_windows"      : N_WIN,
        "context_k"      : CONTEXT_K,
        "context_size"   : CONTEXT_SIZE,
        "feat_dim"       : FEAT_DIM,
        "seed"           : SEED,
        "mode"           : "semi-supervised",
        "pseudo_csv"     : str(PSEUDO_SED_ENS),
        "feature_layout" : ["perch_raw", "perch_prior_fused", "mlp_probe", "proto_ssm", "sed_csebbs"],
        "architectures"  : ["lgbm", "xgb", "mlp", "bigru", "tcn", "transformer", "ssm",
                            "ft_transformer", "cnn1d"],
    },
    tags=["stacker", "meta-learner", "v3-ss", "semi-supervised"],
)
print("[wandb] run initialized")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def safe_logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clip-safe logit transform."""
    p = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro AUC, skipping classes with no positive samples."""
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            try:
                aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def build_context_features(X_win: np.ndarray,
                           filenames: np.ndarray,
                           unique_files_list: list,
                           context_k: int = 1) -> np.ndarray:
    """
    Build context-windowed features.
    For each window t, concat [t-k, ..., t, ..., t+k]. Pad edges with zeros.

    Args:
        X_win           : (N, feat_dim)
        filenames       : (N,)
        unique_files_list: list of unique filenames
        context_k       : int
    Returns:
        X_ctx : (N, (2*context_k+1)*feat_dim)
    """
    N, F = X_win.shape
    ctx_size = 2 * context_k + 1
    X_ctx = np.zeros((N, ctx_size * F), dtype=np.float32)
    for fname in unique_files_list:
        mask = filenames == fname
        rows = np.where(mask)[0]
        Tf = len(rows)
        for local_t, row_idx in enumerate(rows):
            parts = []
            for offset in range(-context_k, context_k + 1):
                t_nb = local_t + offset
                if 0 <= t_nb < Tf:
                    parts.append(X_win[rows[t_nb]])
                else:
                    parts.append(np.zeros(F, dtype=np.float32))
            X_ctx[row_idx] = np.concatenate(parts)
    return X_ctx


def win_to_file_seq(X_w: np.ndarray, files: list, filenames: np.ndarray) -> np.ndarray:
    """(N_rows, F) → (n_files, N_WIN, F)."""
    n_files = len(files)
    F = X_w.shape[1]
    out = np.zeros((n_files, N_WIN, F), dtype=np.float32)
    for fi, fname in enumerate(files):
        mask = filenames == fname
        rows = np.where(mask)[0]
        if len(rows) != N_WIN:
            # Pad / truncate gracefully
            take = min(len(rows), N_WIN)
            out[fi, :take] = X_w[rows[:take]]
        else:
            out[fi] = X_w[rows]
    return out


def win_to_file_labels(Y_w: np.ndarray, files: list, filenames: np.ndarray) -> np.ndarray:
    """(N_rows, C) → (n_files, N_WIN, C)."""
    n_files = len(files)
    out = np.zeros((n_files, N_WIN, N_CLASSES), dtype=np.float32)
    for fi, fname in enumerate(files):
        mask = filenames == fname
        rows = np.where(mask)[0]
        take = min(len(rows), N_WIN)
        out[fi, :take] = Y_w[rows[:take]]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# [1/8] LOAD LABELED FEATURES (59 soundscapes, 708 rows) — same as v3
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/8] Loading labeled features (708 rows) …")

# --- Meta
meta = pd.read_parquet(PERCH_META / "full_perch_meta.parquet")   # (708, ≥4)
assert len(meta) == 708, f"Expected 708 rows, got {len(meta)}"
filenames_708 = meta["filename"].values       # (708,)
row_ids_708   = meta["row_id"].values         # (708,)

unique_files = list(dict.fromkeys(filenames_708))   # 59 unique, order-preserved
assert len(unique_files) == 59, f"Expected 59 unique files, got {len(unique_files)}"
file_to_idx  = {f: i for i, f in enumerate(unique_files)}
labeled_set  = set(unique_files)

# 1. perch_raw
perch_raw_data  = np.load(PERCH_META / "full_perch_arrays.npz")
perch_raw       = perch_raw_data["scores_full_raw"].astype(np.float32)   # (708, 234)
perch_raw_logit = safe_logit(perch_raw)
print(f"  perch_raw   : {perch_raw.shape}")

# 2. perch_prior_fused (already logit)
oof_data    = np.load(PERCH_META / "full_oof_meta_features.npz")
perch_prior = oof_data["oof_base"].astype(np.float32)   # (708, 234)
fold_id_lab = oof_data["fold_id"].astype(np.int32)       # (708,)  values 1–5
print(f"  perch_prior : {perch_prior.shape}  folds={np.unique(fold_id_lab)}")

# 3. mlp_probe
mlp_probe_path = OUTPUTS / "mlp_probe_oof.npy"
if mlp_probe_path.exists():
    mlp_probe = np.load(mlp_probe_path).astype(np.float32)
    print(f"  mlp_probe   : {mlp_probe.shape}  (mlp_probe_oof.npy)")
else:
    if "oof_prior" in oof_data:
        mlp_probe = oof_data["oof_prior"].astype(np.float32)
        print(f"  mlp_probe   : {mlp_probe.shape}  (fallback: oof_prior)")
    else:
        mlp_probe = perch_prior.copy()
        print(f"  mlp_probe   : {mlp_probe.shape}  (fallback: perch_prior)")

# 4. proto_ssm (file-level → broadcast to 708 windows)
proto_preds_59 = np.load(OUTPUTS / "proto_ssm_oof_preds.npy").astype(np.float32)
proto_files_59 = np.load(OUTPUTS / "proto_ssm_oof_file_list.npy", allow_pickle=True)

proto_logit_708 = np.zeros((708, N_CLASSES), dtype=np.float32)
for win_i, fname in enumerate(filenames_708):
    mask = proto_files_59 == fname
    if mask.any():
        fi = int(np.where(mask)[0][0])
        proto_logit_708[win_i] = proto_preds_59[fi]
print(f"  proto_ssm   : {proto_logit_708.shape}")

# 5. sed_csebbs
assert CACHE_SED.exists(), f"SED cache not found: {CACHE_SED}  Run train_stacker_v3.py first."
sed_csebbs_probs = np.load(CACHE_SED)
assert sed_csebbs_probs.shape == (708, N_CLASSES), f"Bad SED cache shape: {sed_csebbs_probs.shape}"
sed_csebbs_logit = safe_logit(sed_csebbs_probs)
print(f"  sed_csebbs  : {sed_csebbs_logit.shape}")

# Ground-truth labels
label_data    = np.load(OUTPUTS / "perch_labeled_ss.npz", allow_pickle=True)
label_y_raw   = label_data["labels"].astype(np.float32)
label_row_ids = label_data["row_ids"]

rid_to_label = dict(zip(label_row_ids, range(len(label_row_ids))))
Y_lab = np.zeros((708, N_CLASSES), dtype=np.float32)
missing_lab = 0
for i, rid in enumerate(row_ids_708):
    if rid in rid_to_label:
        Y_lab[i] = label_y_raw[rid_to_label[rid]]
    else:
        missing_lab += 1
print(f"  labels      : {Y_lab.shape}  missing={missing_lab}  pos_rate={Y_lab.mean():.4f}")

# Build labeled feature matrix X_lab (708, 1170)
mlp_rng = (mlp_probe.min(), mlp_probe.max())
if mlp_rng[0] >= -0.1 and mlp_rng[1] <= 1.1:
    mlp_probe_logit = safe_logit(mlp_probe)
    print(f"  mlp_probe   : prob range → logit converted")
else:
    mlp_probe_logit = mlp_probe.astype(np.float32)
    print(f"  mlp_probe   : already logit space")

X_lab = np.concatenate([
    perch_raw_logit,    # (708, 234)
    perch_prior,        # (708, 234)
    mlp_probe_logit,    # (708, 234)
    proto_logit_708,    # (708, 234)
    sed_csebbs_logit,   # (708, 234)
], axis=1).astype(np.float32)
assert X_lab.shape == (708, FEAT_DIM), f"X_lab shape mismatch: {X_lab.shape}"
print(f"  X_lab       : {X_lab.shape}")

# Load saved normalisation stats from v3 (do NOT recompute)
assert NORM_STATS_PATH.exists(), (
    f"Norm stats not found: {NORM_STATS_PATH}\n"
    "Run train_stacker_v3.py first to generate stacker_norm_v3.npz."
)
_norm = np.load(NORM_STATS_PATH)
X_mean = _norm["mean"].astype(np.float32)   # (1, 1170)
X_std  = _norm["std"].astype(np.float32)    # (1, 1170)
X_std[X_std < 1e-8] = 1.0
print(f"  norm stats  : loaded from {NORM_STATS_PATH}")

X_lab_norm = (X_lab - X_mean) / X_std   # (708, 1170)

# fold_id per window (labels 1–5)
# fold_id per unique file
file_fold_lab = np.array(
    [fold_id_lab[np.where(filenames_708 == f)[0][0]] for f in unique_files],
    dtype=np.int32,
)  # (59,)


# ═══════════════════════════════════════════════════════════════════════════════
# [2/8] LOAD PSEUDO LABEL CSVs
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2/8] Loading pseudo label CSVs …")


def load_pseudo_csv(csv_path: Path, n_classes: int = N_CLASSES,
                    label_cols: list = None) -> dict:
    """
    Load a pseudo label CSV.
    Returns dict: filename (str) → np.ndarray (n_wins, n_classes) float32.

    Row-id format: BC2026_Train_XXXX_SXX_YYYYMMDD_HHMMSS_<sec>
    Filename = all parts except last underscore-field + '.ogg'
    """
    if label_cols is None:
        label_cols = PRIMARY_LABELS

    print(f"  reading {csv_path.name} …", end=" ", flush=True)
    df = pd.read_csv(csv_path)
    print(f"{len(df)} rows", flush=True)

    # Parse filename
    df["_filename"] = df["row_id"].apply(
        lambda x: "_".join(x.split("_")[:-1]) + ".ogg"
    )

    # Align columns to PRIMARY_LABELS; fill missing with 0.5 (neutral)
    csv_cols = [c for c in df.columns if c in set(label_cols)]
    missing_cols = [c for c in label_cols if c not in set(df.columns)]
    if missing_cols:
        print(f"    WARNING: {len(missing_cols)} label cols missing in {csv_path.name}; filling 0.5")
        for mc in missing_cols:
            df[mc] = 0.5

    result = {}
    for fname, grp in df.groupby("_filename"):
        grp = grp.sort_values("row_id")   # sort by start-sec (lex sort works since sec is last field)
        arr = grp[label_cols].values.astype(np.float32)   # (n_wins, 234)
        result[fname] = arr

    return result


perch_dict  = load_pseudo_csv(PSEUDO_PERCH)
proto_dict  = load_pseudo_csv(PSEUDO_PROTO)
sed_dict    = load_pseudo_csv(PSEUDO_SED_ENS)

print(f"  perch_dict  : {len(perch_dict)} files")
print(f"  proto_dict  : {len(proto_dict)} files")
print(f"  sed_dict    : {len(sed_dict)} files  (authoritative)")


# ═══════════════════════════════════════════════════════════════════════════════
# [3/8] BUILD PSEUDO FEATURE MATRIX  — cache to CACHE_PSEUDO
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3/8] Building pseudo feature matrix …")

if CACHE_PSEUDO.exists():
    print(f"  [cache] loading pseudo features from {CACHE_PSEUDO} …")
    _cache = np.load(CACHE_PSEUDO, allow_pickle=True)
    X_pseudo_raw   = _cache["X_pseudo_raw"]    # (N_pseudo_rows, 1170)
    Y_pseudo       = _cache["Y_pseudo"]        # (N_pseudo_rows, 234)
    pseudo_filenames = _cache["pseudo_filenames"]  # (N_pseudo_rows,)
    print(f"  X_pseudo_raw : {X_pseudo_raw.shape}")
    print(f"  Y_pseudo     : {Y_pseudo.shape}")
else:
    # All files in PSEUDO_SED_ENS are the authoritative set (covers all 10,658 files)
    all_pseudo_files = sorted(sed_dict.keys())

    # Exclude the 59 labeled soundscapes to avoid duplication
    pseudo_files_filtered = [f for f in all_pseudo_files if f not in labeled_set]
    print(f"  all pseudo files   : {len(all_pseudo_files)}")
    print(f"  after dedup        : {len(pseudo_files_filtered)}")

    n_pseudo_files = len(pseudo_files_filtered)
    n_pseudo_rows  = n_pseudo_files * N_WIN   # each file has exactly 12 windows

    X_pseudo_raw   = np.zeros((n_pseudo_rows, FEAT_DIM), dtype=np.float32)
    Y_pseudo       = np.zeros((n_pseudo_rows, N_CLASSES), dtype=np.float32)
    pseudo_filenames_list = []

    for fi, fname in enumerate(tqdm(pseudo_files_filtered, desc="Building pseudo X/Y")):
        row_start = fi * N_WIN
        row_end   = row_start + N_WIN

        # ── Y_pseudo: SED ensemble as soft targets ──
        sed_arr = sed_dict[fname]   # (n_wins, 234); n_wins may be < 12
        take    = min(len(sed_arr), N_WIN)
        Y_pseudo[row_start:row_start + take] = sed_arr[:take]

        # ── perch features (slots 0, 1, 2 in feature layout) ──
        if fname in perch_dict:
            parr  = perch_dict[fname]
            ptake = min(len(parr), N_WIN)
            perch_logit_f = safe_logit(parr[:ptake])   # (ptake, 234)
            X_pseudo_raw[row_start:row_start + ptake, 0:N_CLASSES]             = perch_logit_f
            X_pseudo_raw[row_start:row_start + ptake, N_CLASSES:2*N_CLASSES]   = perch_logit_f
            X_pseudo_raw[row_start:row_start + ptake, 2*N_CLASSES:3*N_CLASSES] = perch_logit_f
        # else: stays 0 (neutral logit)

        # ── proto_ssm features (slot 3) ──
        if fname in proto_dict:
            parr  = proto_dict[fname]
            ptake = min(len(parr), N_WIN)
            proto_logit_f = safe_logit(parr[:ptake])   # (ptake, 234)
            X_pseudo_raw[row_start:row_start + ptake, 3*N_CLASSES:4*N_CLASSES] = proto_logit_f
        # else: stays 0

        # ── sed_csebbs features (slot 4) — always available ──
        sed_logit_f = safe_logit(sed_arr[:take])
        X_pseudo_raw[row_start:row_start + take, 4*N_CLASSES:5*N_CLASSES] = sed_logit_f

        pseudo_filenames_list.extend([fname] * N_WIN)

    pseudo_filenames = np.array(pseudo_filenames_list, dtype=object)

    print(f"  X_pseudo_raw : {X_pseudo_raw.shape}")
    print(f"  Y_pseudo     : {Y_pseudo.shape}  pos_rate={Y_pseudo.mean():.4f}")

    # Cache
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PSEUDO,
        X_pseudo_raw=X_pseudo_raw,
        Y_pseudo=Y_pseudo,
        pseudo_filenames=pseudo_filenames,
    )
    print(f"  cached → {CACHE_PSEUDO}")

# Normalise pseudo features using the SAME stats as labeled data
X_pseudo_norm = (X_pseudo_raw - X_mean) / X_std   # (N_pseudo_rows, 1170)
print(f"  X_pseudo_norm: {X_pseudo_norm.shape}")

# Unique pseudo files and their fold assignment (deterministic by index)
pseudo_unique_files = list(dict.fromkeys(pseudo_filenames))
n_pseudo_files = len(pseudo_unique_files)
pseudo_file_fold = np.array([(i % 5) + 1 for i in range(n_pseudo_files)], dtype=np.int32)  # 1–5

pseudo_file_to_fold = {f: pseudo_file_fold[i] for i, f in enumerate(pseudo_unique_files)}
fold_id_pseudo = np.array([pseudo_file_to_fold[f] for f in pseudo_filenames], dtype=np.int32)

print(f"  pseudo unique files: {n_pseudo_files}")
print(f"  pseudo fold dist: {dict(zip(*np.unique(pseudo_file_fold, return_counts=True)))}")


# ═══════════════════════════════════════════════════════════════════════════════
# [4/8] COMBINE LABELED + PSEUDO DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/8] Combining labeled + pseudo data …")

# Window-level combined arrays
X_all_norm   = np.vstack([X_lab_norm,  X_pseudo_norm]).astype(np.float32)
Y_all        = np.vstack([Y_lab,       Y_pseudo]).astype(np.float32)
filenames_all = np.concatenate([filenames_708, pseudo_filenames])
fold_id_all   = np.concatenate([fold_id_lab,  fold_id_pseudo]).astype(np.int32)

N_all = len(X_all_norm)
print(f"  X_all_norm   : {X_all_norm.shape}")
print(f"  Y_all        : {Y_all.shape}  pos_rate={Y_all.mean():.4f}")
print(f"  fold_id_all  : {dict(zip(*np.unique(fold_id_all, return_counts=True)))}")

# File-level combined arrays (for sequence models)
all_unique_files   = unique_files + pseudo_unique_files          # labeled first, then pseudo
file_fold_all      = np.concatenate([file_fold_lab, pseudo_file_fold]).astype(np.int32)  # (N_files,)

X_all_seq = win_to_file_seq(X_all_norm, all_unique_files, filenames_all)       # (N_files, 12, 1170)
Y_all_seq = win_to_file_labels(Y_all, all_unique_files, filenames_all)          # (N_files, 12, 234)

n_all_files = len(all_unique_files)
print(f"  X_all_seq    : {X_all_seq.shape}")
print(f"  Y_all_seq    : {Y_all_seq.shape}")
print(f"  file_fold_all: {dict(zip(*np.unique(file_fold_all, return_counts=True)))}")

# CV evaluation only on labeled data (708 rows / 59 files) since GT is only there
# Pseudo data: all folds 1–5 (train+val); but AUC logged only on labeled val fold
# We define a helper that tracks labeled-only val AUC while training on all data.

# Index arrays for labeled rows within X_all_norm
N_labeled_rows   = 708
N_labeled_files  = 59

# For sequence models: labeled files are indices 0..58, pseudo start at 59
labeled_file_idx = np.arange(N_labeled_files)


# ═══════════════════════════════════════════════════════════════════════════════
# CV EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def cv_eval_flat_ss(name: str, pred_fn) -> tuple:
    """
    CV for flat (window-level) models using all data.
    - Training fold: all rows where fold_id != fv (labeled + pseudo)
    - Validation: only labeled rows where fold_id == fv (GT labels)
    Returns: (mean_val_auc_on_labeled, oof_on_labeled_708)
    """
    oof_labeled = np.zeros((N_labeled_rows, N_CLASSES), dtype=np.float32)
    fold_aucs   = []

    for fv in np.unique(fold_id_all):
        tr_mask = fold_id_all != fv                    # train: all data except val fold
        # val mask: only labeled rows in this fold
        va_lab_mask = (fold_id_all[:N_labeled_rows] == fv)   # (708,) bool

        if va_lab_mask.sum() == 0:
            continue

        preds_tr = pred_fn(X_all_norm[tr_mask], Y_all[tr_mask],
                           X_all_norm[:N_labeled_rows][va_lab_mask])
        oof_labeled[va_lab_mask] = preds_tr

        fa = macro_auc(Y_all[:N_labeled_rows][va_lab_mask], preds_tr)
        fold_aucs.append(fa)
        wandb.log({"arch": name, "fold": int(fv), "fold_val_auc_labeled": fa})

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
    print(f"  [{name}] fold aucs (labeled val): {[f'{a:.4f}' for a in fold_aucs]}  → mean={mean_auc:.4f}")
    wandb.log({f"{name}_oof_auc_labeled": mean_auc})
    return mean_auc, oof_labeled


def cv_eval_seq_ss(name: str, model_cls, **train_kwargs) -> tuple:
    """
    CV for sequence models (file-level).
    - Training: all files except val fold
    - Validation AUC: only labeled files in val fold
    Returns: (mean_val_auc_on_labeled, oof_window_labeled_708)
    """
    # We accumulate OOF predictions for labeled files only (indexed 0..58)
    oof_file_labeled = np.zeros((N_labeled_files, N_WIN, N_CLASSES), dtype=np.float32)
    fold_aucs        = []

    for fv in tqdm(np.unique(file_fold_all), desc=f"{name} CV"):
        tr_mask_file = file_fold_all != fv
        va_mask_file = file_fold_all == fv

        # Among validation files, pick only labeled ones (indices 0..58)
        va_labeled_file_idx = np.where(
            (file_fold_all == fv) & (np.arange(n_all_files) < N_labeled_files)
        )[0]

        if len(va_labeled_file_idx) == 0:
            continue

        model = train_seq_model(
            model_cls(),
            X_all_seq[tr_mask_file], Y_all_seq[tr_mask_file],
            X_all_seq[va_labeled_file_idx], Y_all_seq[va_labeled_file_idx],
            **train_kwargs,
        )
        model.eval()
        with torch.no_grad():
            preds = model(
                torch.from_numpy(X_all_seq[va_labeled_file_idx]).float().to(DEVICE)
            ).cpu().numpy()   # (n_va_lab, 12, 234)

        # Store in labeled OOF (va_labeled_file_idx are within 0..58)
        for out_pos, fi in enumerate(va_labeled_file_idx):
            oof_file_labeled[fi] = preds[out_pos]

        yva_flat = Y_all_seq[va_labeled_file_idx].reshape(-1, N_CLASSES)
        pva_flat = preds.reshape(-1, N_CLASSES)
        fa       = macro_auc(yva_flat, pva_flat)
        fold_aucs.append(fa)
        wandb.log({"arch": name, "fold": int(fv), "fold_val_auc_labeled": fa})

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
    print(f"  [{name}] fold aucs (labeled val): {[f'{a:.4f}' for a in fold_aucs]}  → mean={mean_auc:.4f}")
    wandb.log({f"{name}_oof_auc_labeled": mean_auc})

    # Flatten oof_file_labeled → (708, 234) in window order
    oof_win = np.zeros((N_labeled_rows, N_CLASSES), dtype=np.float32)
    for fi, fname in enumerate(unique_files):   # labeled files only
        mask = filenames_708 == fname
        rows = np.where(mask)[0]
        oof_win[rows] = oof_file_labeled[fi]

    return mean_auc, oof_win


# ═══════════════════════════════════════════════════════════════════════════════
# [5/8] CV EVALUATION — ALL 9 ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5/8] Cross-validation (9 architectures, semi-supervised) …")
oof_aucs = {}

# ─── Baseline: equal-weight average of 5 model logits (on labeled only) ───────
def baseline_avg_logit(X_w: np.ndarray) -> np.ndarray:
    parts = [X_w[:, i * N_CLASSES:(i + 1) * N_CLASSES] for i in range(N_MODELS)]
    return np.mean(parts, axis=0)

def baseline_fn(X_tr, Y_tr, X_va):
    return baseline_avg_logit(X_va)

base_auc, _ = cv_eval_flat_ss("baseline", baseline_fn)
oof_aucs["baseline"] = base_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 1: LGBM — per-class LightGBM, context features (3510 dim)
# ─────────────────────────────────────────────────────────────────────────────

def lgbm_fn(X_ctx_tr: np.ndarray, Y_tr: np.ndarray, X_ctx_va: np.ndarray) -> np.ndarray:
    preds = np.zeros((len(X_ctx_va), N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        if Y_tr[:, c].sum() == 0:
            continue
        clf = lgb.LGBMClassifier(
            n_estimators=20,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            n_jobs=8,
            verbose=-1,
        )
        y_bin = (Y_tr[:, c] > 0.5).astype(np.int32)
        if y_bin.sum() == 0:
            continue
        clf.fit(X_ctx_tr, y_bin)
        prob = clf.predict_proba(X_ctx_va)[:, 1].astype(np.float32)
        preds[:, c] = safe_logit(prob)
    return preds


def cv_eval_lgbm_ss() -> tuple:
    oof_labeled = np.zeros((N_labeled_rows, N_CLASSES), dtype=np.float32)
    fold_aucs_l = []

    # Use labeled-only for LGBM CV: pseudo labels are too large for per-class tree CV
    fold_id_lab = fold_id_all[:N_labeled_rows]
    X_lab_norm  = X_all_norm[:N_labeled_rows]
    Y_lab       = Y_all[:N_labeled_rows]
    fn_lab      = filenames_all[:N_labeled_rows]

    for fv in tqdm(np.unique(fold_id_lab), desc="LGBM CV"):
        tr_lab_m  = fold_id_lab != fv
        va_lab_m  = fold_id_lab == fv

        if va_lab_m.sum() == 0:
            continue

        fn_tr     = fn_lab[tr_lab_m]
        fn_va     = fn_lab[va_lab_m]
        uf_tr     = list(dict.fromkeys(fn_tr))
        uf_va     = list(dict.fromkeys(fn_va))

        X_ctx_tr  = build_context_features(X_lab_norm[tr_lab_m], fn_tr, uf_tr, CONTEXT_K)
        X_ctx_va  = build_context_features(X_lab_norm[va_lab_m], fn_va, uf_va, CONTEXT_K)

        preds = lgbm_fn(X_ctx_tr, Y_lab[tr_lab_m], X_ctx_va)
        oof_labeled[va_lab_m] = preds

        fa = macro_auc(Y_all[:N_labeled_rows][va_lab_m], preds)
        fold_aucs_l.append(fa)
        wandb.log({"arch": "lgbm", "fold": int(fv), "fold_val_auc_labeled": fa})

    mean_auc = float(np.mean(fold_aucs_l)) if fold_aucs_l else 0.0
    print(f"  [lgbm] fold aucs: {[f'{a:.4f}' for a in fold_aucs_l]}  → mean={mean_auc:.4f}")
    wandb.log({"lgbm_oof_auc_labeled": mean_auc})
    return mean_auc, oof_labeled


lgbm_auc, lgbm_oof = cv_eval_lgbm_ss()
oof_aucs["lgbm"] = lgbm_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 2: XGBoost — per-class XGBClassifier, context features (3510 dim)
# ─────────────────────────────────────────────────────────────────────────────

def xgb_fn(X_ctx_tr: np.ndarray, Y_tr: np.ndarray, X_ctx_va: np.ndarray) -> np.ndarray:
    preds = np.zeros((len(X_ctx_va), N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        if Y_tr[:, c].sum() == 0:
            continue
        clf = XGBClassifier(
            n_estimators=20,
            learning_rate=0.05,
            max_depth=3,
            tree_method="hist",
            device="cuda",
            random_state=SEED,
            eval_metric="logloss",
            verbosity=0,
        )
        y_bin = (Y_tr[:, c] > 0.5).astype(np.int32)
        if y_bin.sum() == 0:
            continue
        clf.fit(X_ctx_tr, y_bin)
        prob = clf.predict_proba(X_ctx_va)[:, 1].astype(np.float32)
        preds[:, c] = safe_logit(prob)
    return preds


def cv_eval_xgb_ss() -> tuple:
    oof_labeled = np.zeros((N_labeled_rows, N_CLASSES), dtype=np.float32)
    fold_aucs_x = []

    # Use labeled-only for XGB CV: pseudo labels are too large for per-class tree CV
    fold_id_lab = fold_id_all[:N_labeled_rows]
    X_lab_norm  = X_all_norm[:N_labeled_rows]
    Y_lab       = Y_all[:N_labeled_rows]
    fn_lab      = filenames_all[:N_labeled_rows]

    for fv in tqdm(np.unique(fold_id_lab), desc="XGB CV"):
        tr_lab_m  = fold_id_lab != fv
        va_lab_m  = fold_id_lab == fv

        if va_lab_m.sum() == 0:
            continue

        fn_tr    = fn_lab[tr_lab_m]
        fn_va    = fn_lab[va_lab_m]
        uf_tr    = list(dict.fromkeys(fn_tr))
        uf_va    = list(dict.fromkeys(fn_va))

        X_ctx_tr = build_context_features(X_lab_norm[tr_lab_m], fn_tr, uf_tr, CONTEXT_K)
        X_ctx_va = build_context_features(X_lab_norm[va_lab_m], fn_va, uf_va, CONTEXT_K)

        preds = xgb_fn(X_ctx_tr, Y_lab[tr_lab_m], X_ctx_va)
        oof_labeled[va_lab_m] = preds

        fa = macro_auc(Y_all[:N_labeled_rows][va_lab_m], preds)
        fold_aucs_x.append(fa)
        wandb.log({"arch": "xgb", "fold": int(fv), "fold_val_auc_labeled": fa})

    mean_auc = float(np.mean(fold_aucs_x)) if fold_aucs_x else 0.0
    print(f"  [xgb] fold aucs: {[f'{a:.4f}' for a in fold_aucs_x]}  → mean={mean_auc:.4f}")
    wandb.log({"xgb_oof_auc_labeled": mean_auc})
    return mean_auc, oof_labeled


xgb_auc, xgb_oof = cv_eval_xgb_ss()
oof_aucs["xgb"] = xgb_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 3: MLP — per-class shared MLP with context window
# Input: (B, context_size=3, N_MODELS=5, N_CLASSES=234)
# ─────────────────────────────────────────────────────────────────────────────

class StackerMLP(nn.Module):
    def __init__(self, n_models: int = N_MODELS, n_classes: int = N_CLASSES,
                 context_size: int = CONTEXT_SIZE, hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.n_models     = n_models
        self.n_classes    = n_classes
        self.context_size = context_size
        in_dim = context_size * n_models   # 3*5=15
        self.fc1     = nn.Linear(in_dim, hidden)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, context_size, N_MODELS, N_CLASSES) → (B, N_CLASSES)"""
        B, S, M, C = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, C, S * M)   # (B, C, 15)
        x = x.reshape(B * C, S * M)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)         # (B*C, 1)
        return x.reshape(B, C)  # (B, C)


def X_ctx_to_mlp_input(X_ctx: np.ndarray) -> torch.Tensor:
    """(B, 3510) → (B, 3, 5, 234)"""
    B = X_ctx.shape[0]
    t = torch.from_numpy(X_ctx).float()
    t = t.reshape(B, CONTEXT_SIZE, FEAT_DIM)
    t = t.reshape(B, CONTEXT_SIZE, N_MODELS, N_CLASSES)
    return t


def train_mlp(X_ctx_tr: np.ndarray, Y_tr: np.ndarray,
              X_ctx_va: np.ndarray, Y_va: np.ndarray,
              epochs: int = 100, patience: int = 15,
              lr: float = 1e-3, wd: float = 1e-3) -> StackerMLP:
    model = StackerMLP().to(DEVICE)
    pos_rate = Y_tr.mean(axis=0)
    pos_w    = np.clip((1.0 - pos_rate) / np.maximum(pos_rate, 1e-6), 0, 20).astype(np.float32)
    pw_tensor = torch.from_numpy(pos_w).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optim_    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=epochs)

    Xtr_t = X_ctx_to_mlp_input(X_ctx_tr).to(DEVICE)
    Ytr_t = torch.from_numpy(Y_tr).float().to(DEVICE)
    Xva_t = X_ctx_to_mlp_input(X_ctx_va).to(DEVICE)

    ds     = TensorDataset(Xtr_t, Ytr_t)
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)

    best_auc   = 0.0
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optim_.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim_.step()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                va_preds = model(Xva_t).cpu().numpy()
            auc = macro_auc(Y_va, va_preds)
            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def cv_eval_mlp_ss() -> tuple:
    oof_labeled = np.zeros((N_labeled_rows, N_CLASSES), dtype=np.float32)
    fold_aucs_m = []

    for fv in tqdm(np.unique(fold_id_all), desc="MLP CV"):
        tr_mask  = fold_id_all != fv
        va_lab_m = (fold_id_all[:N_labeled_rows] == fv)

        if va_lab_m.sum() == 0:
            continue

        fn_tr    = filenames_all[tr_mask]
        fn_va    = filenames_all[:N_labeled_rows][va_lab_m]
        uf_tr    = list(dict.fromkeys(fn_tr))
        uf_va    = list(dict.fromkeys(fn_va))

        X_ctx_tr = build_context_features(X_all_norm[tr_mask], fn_tr, uf_tr, CONTEXT_K)
        X_ctx_va = build_context_features(X_all_norm[:N_labeled_rows][va_lab_m], fn_va, uf_va, CONTEXT_K)

        Y_tr_fold = Y_all[tr_mask]
        Y_va_fold = Y_all[:N_labeled_rows][va_lab_m]

        model = train_mlp(X_ctx_tr, Y_tr_fold, X_ctx_va, Y_va_fold)
        model.eval()
        with torch.no_grad():
            preds = model(X_ctx_to_mlp_input(X_ctx_va).to(DEVICE)).cpu().numpy()

        oof_labeled[va_lab_m] = preds
        fa = macro_auc(Y_va_fold, preds)
        fold_aucs_m.append(fa)
        wandb.log({"arch": "mlp", "fold": int(fv), "fold_val_auc_labeled": fa})

    mean_auc = float(np.mean(fold_aucs_m)) if fold_aucs_m else 0.0
    print(f"  [mlp] fold aucs: {[f'{a:.4f}' for a in fold_aucs_m]}  → mean={mean_auc:.4f}")
    wandb.log({"mlp_oof_auc_labeled": mean_auc})
    return mean_auc, oof_labeled


mlp_auc, mlp_oof = cv_eval_mlp_ss()
oof_aucs["mlp"] = mlp_auc


# ─────────────────────────────────────────────────────────────────────────────
# Generic sequence model trainer (shared)
# ─────────────────────────────────────────────────────────────────────────────

def train_seq_model(model: nn.Module,
                    X_tr_seq: np.ndarray, Y_tr_seq: np.ndarray,
                    X_va_seq: np.ndarray, Y_va_seq: np.ndarray,
                    epochs: int = 150, patience: int = 15,
                    lr: float = 5e-4, wd: float = 1e-3,
                    batch_size: int = 16) -> nn.Module:
    model = model.to(DEVICE)
    pos_rate  = Y_tr_seq.mean(axis=(0, 1))
    pos_w     = np.clip((1.0 - pos_rate) / np.maximum(pos_rate, 1e-6), 0, 20).astype(np.float32)
    pw_tensor = torch.from_numpy(pos_w).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optim_    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=epochs)

    Xtr = torch.from_numpy(X_tr_seq).float()
    Ytr = torch.from_numpy(Y_tr_seq).float()
    Xva = torch.from_numpy(X_va_seq).float().to(DEVICE)
    Yva_np = Y_va_seq

    ds     = TensorDataset(Xtr, Ytr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_auc   = 0.0
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim_.zero_grad()
            pred = model(xb)   # (B, T, 234)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                va_pred = model(Xva).cpu().numpy()   # (n_va, T, 234)
            va_flat  = va_pred.reshape(-1, N_CLASSES)
            yva_flat = Yva_np.reshape(-1, N_CLASSES)
            auc = macro_auc(yva_flat, va_flat)
            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Arch 4: BiGRU
# ─────────────────────────────────────────────────────────────────────────────

class StackerBiGRU(nn.Module):
    """File-level BiGRU stacker. Input: (B, T=12, 1170) → Output: (B, T, 234)"""
    def __init__(self, in_features: int = FEAT_DIM, hidden: int = 128,
                 n_layers: int = 2, n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj  = nn.Linear(in_features, hidden)
        self.gru      = nn.GRU(hidden, hidden // 2, num_layers=n_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x, _ = self.gru(x)
        return self.out_proj(x)


bigru_auc, bigru_oof = cv_eval_seq_ss(
    "bigru", StackerBiGRU,
    epochs=150, patience=15, lr=5e-4, wd=1e-3, batch_size=32
)
oof_aucs["bigru"] = bigru_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 5: TCN
# ─────────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self._pad     = pad
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.norm1    = nn.LayerNorm(out_ch)
        self.norm2    = nn.LayerNorm(out_ch)
        self.drop     = nn.Dropout(dropout)
        self.residual = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        T_  = x.shape[1]
        h   = x.transpose(1, 2)
        h   = self.conv1(h)[..., :T_]
        h   = h.transpose(1, 2)
        h   = self.drop(F.gelu(self.norm1(h)))
        h   = h.transpose(1, 2)
        h   = self.conv2(h)[..., :T_]
        h   = h.transpose(1, 2)
        h   = self.drop(F.gelu(self.norm2(h)))
        return h + res


class StackerTCN(nn.Module):
    def __init__(self, in_features: int = FEAT_DIM, channels: int = 128,
                 n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj    = nn.Linear(in_features, channels)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(channels, channels, kernel_size=3, dilation=d, dropout=dropout)
            for d in [1, 2, 4]
        ])
        self.out_proj = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.tcn_blocks:
            x = block(x)
        return self.out_proj(x)


tcn_auc, tcn_oof = cv_eval_seq_ss(
    "tcn", StackerTCN,
    epochs=150, patience=15, lr=5e-4, wd=1e-3, batch_size=32
)
oof_aucs["tcn"] = tcn_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 6: Transformer
# ─────────────────────────────────────────────────────────────────────────────

class StackerTransformer(nn.Module):
    def __init__(self, in_features: int = FEAT_DIM, d_model: int = 128,
                 nhead: int = 4, dim_ff: int = 256, n_layers: int = 2,
                 n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj  = nn.Linear(in_features, d_model)
        self.pos_emb  = nn.Parameter(torch.zeros(1, N_WIN, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x) + self.pos_emb
        x = self.encoder(x)
        return self.out_proj(x)


tfm_auc, tfm_oof = cv_eval_seq_ss(
    "transformer", StackerTransformer,
    epochs=150, patience=15, lr=5e-4, wd=1e-3, batch_size=32
)
oof_aucs["transformer"] = tfm_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 7: SSM (Mamba-style selective SSM)
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveSSMLayer(nn.Module):
    def __init__(self, d_model: int = 128, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, 2 * d_model)
        self.conv1d   = nn.Conv1d(d_model, d_model, kernel_size=3,
                                   padding=1, groups=d_model)
        self.x_proj   = nn.Linear(d_model, d_state * 2 + 1)
        self.dt_proj  = nn.Linear(1, d_model)
        self.A_log    = nn.Parameter(torch.randn(d_model, d_state))
        self.D        = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

    def ssm_scan(self, x: torch.Tensor, B_p: torch.Tensor,
                 C_p: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        batch, T, d = x.shape
        A = -torch.exp(self.A_log.float())
        h = torch.zeros(batch, d, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = F.softplus(dt[:, t, :])
            dA   = torch.exp(dt_t.unsqueeze(-1) * A)
            dB   = dt_t.unsqueeze(-1) * B_p[:, t, :].unsqueeze(1)
            h    = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y    = (h * C_p[:, t, :].unsqueeze(1)).sum(-1)
            ys.append(y)
        return torch.stack(ys, dim=1)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_in = self.norm(x_in)
        xz   = self.in_proj(x_in)
        x, z = xz.chunk(2, dim=-1)
        x    = x.transpose(1, 2)
        x    = self.conv1d(x).transpose(1, 2)
        params = self.x_proj(x)
        B_p  = params[..., :self.d_state]
        C_p  = params[..., self.d_state:2 * self.d_state]
        dt   = self.dt_proj(params[..., -1:])
        y    = self.ssm_scan(x, B_p, C_p, dt)
        y    = y * F.silu(z)
        y    = self.out_proj(y)
        return x_in + self.drop(y)


class StackerSSM(nn.Module):
    def __init__(self, in_features: int = FEAT_DIM, d_model: int = 128,
                 d_state: int = 16, n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj  = nn.Linear(in_features, d_model)
        self.ssm1     = SelectiveSSMLayer(d_model, d_state, dropout)
        self.ssm2     = SelectiveSSMLayer(d_model, d_state, dropout)
        self.out_proj = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.ssm1(x)
        x = self.ssm2(x)
        return self.out_proj(x)


ssm_auc, ssm_oof = cv_eval_seq_ss(
    "ssm", StackerSSM,
    epochs=150, patience=15, lr=5e-4, wd=1e-3, batch_size=32
)
oof_aucs["ssm"] = ssm_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 8: FT-Transformer
# ─────────────────────────────────────────────────────────────────────────────

class StackerFTTransformer(nn.Module):
    def __init__(self, n_models: int = N_MODELS, n_classes: int = N_CLASSES,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_models  = n_models
        self.n_classes = n_classes
        self.d_model   = d_model
        self.tokenizer = nn.Linear(n_classes, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 2, dropout, batch_first=True, activation="gelu"
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, n_layers)
        self.out_proj = nn.Linear(d_model * n_models, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x       = x.view(B * T, self.n_models, self.n_classes)
        tokens  = self.tokenizer(x)
        out     = self.encoder(tokens)
        out     = out.reshape(B * T, -1)
        out     = self.out_proj(out)
        return out.view(B, T, self.n_classes)


ft_tfm_auc, ft_tfm_oof = cv_eval_seq_ss(
    "ft_transformer", StackerFTTransformer,
    epochs=150, patience=15, lr=3e-4, wd=1e-3, batch_size=32
)
oof_aucs["ft_transformer"] = ft_tfm_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 9: CNN1D
# ─────────────────────────────────────────────────────────────────────────────

class StackerCNN1D(nn.Module):
    def __init__(self, in_features: int = FEAT_DIM, channels: int = 128,
                 n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj  = nn.Linear(in_features, channels)
        self.conv1    = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2    = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm     = nn.LayerNorm(channels)
        self.drop     = nn.Dropout(dropout)
        self.out_proj = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.in_proj(x)           # (B, T, 128)
        h   = res.transpose(1, 2)       # (B, 128, T)
        h   = F.gelu(self.conv1(h) + self.conv2(h))
        h   = h.transpose(1, 2)         # (B, T, 128)
        h   = self.drop(self.norm(h))
        return self.out_proj(res + h)


cnn1d_auc, cnn1d_oof = cv_eval_seq_ss(
    "cnn1d", StackerCNN1D,
    epochs=150, patience=15, lr=5e-4, wd=1e-3, batch_size=32
)
oof_aucs["cnn1d"] = cnn1d_auc


# ═══════════════════════════════════════════════════════════════════════════════
# [6/8] FINAL FIT ON ALL DATA (labeled + pseudo)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[6/8] Final fit on all data (labeled + pseudo) …")

def ss_name(arch: str, ext: str) -> str:
    """Return artifact filename with OOF AUC score, e.g. stacker_mlp_ss_auc0.9594.onnx"""
    auc = oof_aucs.get(arch, 0.0)
    return f"stacker_{arch}_ss_auc{auc:.4f}.{ext}"

best_arch = max(
    [k for k in oof_aucs if k != "baseline"],
    key=lambda k: oof_aucs[k],
)
print(f"  best architecture: {best_arch}  (labeled val AUC={oof_aucs[best_arch]:.4f})")

# Build full context features for LGBM/XGB (labeled-only) and MLP/neural (all)
# LGBM/XGB full fit uses labeled-only to avoid 127K×3510 matrix being too slow
X_ctx_lab = build_context_features(
    X_all_norm[:N_labeled_rows],
    filenames_all[:N_labeled_rows],
    list(dict.fromkeys(filenames_all[:N_labeled_rows])),
    CONTEXT_K,
)
print(f"  X_ctx_lab (labeled only): {X_ctx_lab.shape}")
X_ctx_all = build_context_features(X_all_norm, filenames_all, all_unique_files, CONTEXT_K)
print(f"  X_ctx_all (all):          {X_ctx_all.shape}")


# ── LGBM (full fit, labeled-only) ─────────────────────────────────────────────
print("  Fitting LGBM (full, labeled-only) …")
lgbm_models = []
Y_lab_full = Y_all[:N_labeled_rows]
for c in tqdm(range(N_CLASSES), desc="LGBM full fit"):
    clf = lgb.LGBMClassifier(
        n_estimators=20,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        n_jobs=8,
        verbose=-1,
    )
    y_bin = (Y_lab_full[:, c] > 0.5).astype(np.int32)
    if y_bin.sum() > 0:
        clf.fit(X_ctx_lab, y_bin)
    lgbm_models.append(clf)
with open(OUT_DIR / ss_name("lgbm", "pkl"), "wb") as f:
    pickle.dump(lgbm_models, f)
print(f"  LGBM saved → {OUT_DIR / 'stacker_lgbm_ss.pkl'}")


# ── XGBoost (full fit) ────────────────────────────────────────────────────────
print("  Fitting XGBoost (full) …")
xgb_models = []
for c in tqdm(range(N_CLASSES), desc="XGB full fit"):
    clf = XGBClassifier(
        n_estimators=20,
        learning_rate=0.05,
        max_depth=3,
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        eval_metric="logloss",
        verbosity=0,
    )
    y_bin = (Y_lab_full[:, c] > 0.5).astype(np.int32)
    if y_bin.sum() > 0:
        clf.fit(X_ctx_lab, y_bin)
    xgb_models.append(clf)
with open(OUT_DIR / ss_name("xgb", "pkl"), "wb") as f:
    pickle.dump(xgb_models, f)
print(f"  XGB saved → {OUT_DIR / 'stacker_xgb_ss.pkl'}")


# ── MLP (full fit + ONNX export) ─────────────────────────────────────────────
print("  Fitting MLP (full) …")
mlp_model = train_mlp(X_ctx_all, Y_all, X_ctx_all, Y_all, epochs=100, patience=100)
mlp_model.eval()
torch.save(mlp_model.state_dict(), OUT_DIR / ss_name("mlp", "pt"))

dummy_mlp = torch.zeros(1, CONTEXT_SIZE, N_MODELS, N_CLASSES)
torch.onnx.export(
    mlp_model.cpu(), dummy_mlp,
    str(OUT_DIR / ss_name("mlp", "onnx")),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=14,
)
print(f"  MLP saved → {OUT_DIR / 'stacker_mlp_ss.onnx'}")
mlp_model = mlp_model.to(DEVICE)


# ── BiGRU (full fit + ONNX export) ────────────────────────────────────────────
print("  Fitting BiGRU (full) …")
bigru_model = train_seq_model(
    StackerBiGRU(), X_all_seq, Y_all_seq, X_all_seq, Y_all_seq,
    epochs=150, patience=150, lr=5e-4, wd=1e-3, batch_size=32
)
bigru_model.eval()
torch.save(bigru_model.state_dict(), OUT_DIR / ss_name("bigru", "pt"))
dummy_seq = torch.zeros(1, N_WIN, FEAT_DIM)
torch.onnx.export(
    bigru_model.cpu(), dummy_seq,
    str(OUT_DIR / ss_name("bigru", "onnx")),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  BiGRU saved → {OUT_DIR / 'stacker_bigru_ss.onnx'}")
bigru_model = bigru_model.to(DEVICE)


# ── TCN (full fit + ONNX export) ─────────────────────────────────────────────
print("  Fitting TCN (full) …")
tcn_model = train_seq_model(
    StackerTCN(), X_all_seq, Y_all_seq, X_all_seq, Y_all_seq,
    epochs=150, patience=150, lr=5e-4, wd=1e-3, batch_size=32
)
tcn_model.eval()
torch.save(tcn_model.state_dict(), OUT_DIR / ss_name("tcn", "pt"))
torch.onnx.export(
    tcn_model.cpu(), dummy_seq,
    str(OUT_DIR / ss_name("tcn", "onnx")),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  TCN saved → {OUT_DIR / 'stacker_tcn_ss.onnx'}")
tcn_model = tcn_model.to(DEVICE)


# ── Transformer (full fit + ONNX export) ─────────────────────────────────────
print("  Fitting Transformer (full) …")
tfm_model = train_seq_model(
    StackerTransformer(), X_all_seq, Y_all_seq, X_all_seq, Y_all_seq,
    epochs=150, patience=150, lr=5e-4, wd=1e-3, batch_size=32
)
tfm_model.eval()
torch.save(tfm_model.state_dict(), OUT_DIR / ss_name("transformer", "pt"))
torch.onnx.export(
    tfm_model.cpu(), dummy_seq,
    str(OUT_DIR / ss_name("transformer", "onnx")),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  Transformer saved → {OUT_DIR / 'stacker_transformer_ss.onnx'}")
tfm_model = tfm_model.to(DEVICE)


# ── SSM (full fit + ONNX export) ─────────────────────────────────────────────
print("  Fitting SSM (full) …")
ssm_model = train_seq_model(
    StackerSSM(), X_all_seq, Y_all_seq, X_all_seq, Y_all_seq,
    epochs=150, patience=150, lr=5e-4, wd=1e-3, batch_size=32
)
ssm_model.eval()
torch.save(ssm_model.state_dict(), OUT_DIR / ss_name("ssm", "pt"))
torch.onnx.export(
    ssm_model.cpu(), dummy_seq,
    str(OUT_DIR / ss_name("ssm", "onnx")),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  SSM saved → {OUT_DIR / 'stacker_ssm_ss.onnx'}")
ssm_model = ssm_model.to(DEVICE)


# ── FT-Transformer (full fit + ONNX export) ──────────────────────────────────
print("  Fitting FT-Transformer (full) …")
ft_tfm_model = train_seq_model(
    StackerFTTransformer(), X_all_seq, Y_all_seq, X_all_seq, Y_all_seq,
    epochs=150, patience=150, lr=3e-4, wd=1e-3, batch_size=32
)
ft_tfm_model.eval()
torch.save(ft_tfm_model.state_dict(), OUT_DIR / ss_name("ft_transformer", "pt"))
torch.onnx.export(
    ft_tfm_model.cpu(), dummy_seq,
    str(OUT_DIR / ss_name("ft_transformer", "onnx")),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  FT-Transformer saved → {OUT_DIR / 'stacker_ft_transformer_ss.onnx'}")
ft_tfm_model = ft_tfm_model.to(DEVICE)


# ── CNN1D (full fit + ONNX export) ────────────────────────────────────────────
print("  Fitting CNN1D (full) …")
cnn1d_model = train_seq_model(
    StackerCNN1D(), X_all_seq, Y_all_seq, X_all_seq, Y_all_seq,
    epochs=150, patience=150, lr=5e-4, wd=1e-3, batch_size=32
)
cnn1d_model.eval()
torch.save(cnn1d_model.state_dict(), OUT_DIR / ss_name("cnn1d", "pt"))
torch.onnx.export(
    cnn1d_model.cpu(), dummy_seq,
    str(OUT_DIR / ss_name("cnn1d", "onnx")),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  CNN1D saved → {OUT_DIR / 'stacker_cnn1d_ss.onnx'}")


# ═══════════════════════════════════════════════════════════════════════════════
# [7/8] SAVE META JSON
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[7/8] Saving meta JSON …")
meta_dict = {
    "best_arch"         : best_arch,
    "oof_aucs_labeled"  : {k: round(v, 6) for k, v in oof_aucs.items()},
    "n_models"          : N_MODELS,
    "n_classes"         : N_CLASSES,
    "n_windows"         : N_WIN,
    "context_k"         : CONTEXT_K,
    "context_size"      : CONTEXT_SIZE,
    "feature_layout"    : ["perch_raw", "perch_prior_fused", "mlp_probe", "proto_ssm", "sed_csebbs"],
    "feature_dim"       : FEAT_DIM,
    "context_feat_dim"  : CTX_FEAT_DIM,
    "temperature"       : 1.5,
    "architectures"     : ["lgbm", "xgb", "mlp", "bigru", "tcn", "transformer",
                           "ssm", "ft_transformer", "cnn1d"],
    "mode"              : "semi-supervised",
    "n_labeled_files"   : N_labeled_files,
    "n_pseudo_files"    : n_pseudo_files,
    "n_total_files"     : n_all_files,
    "pseudo_csv"        : str(PSEUDO_SED_ENS),
    "norm_stats_from"   : str(NORM_STATS_PATH),
    "trained_date"      : time.strftime("%Y-%m-%d"),
}
with open(OUT_DIR / "stacker_meta_ss.json", "w") as f:
    json.dump(meta_dict, f, indent=2)
print(f"  meta saved → {OUT_DIR / 'stacker_meta_ss.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# [8/8] W&B + EXCEL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  OOF AUC SUMMARY (validated on labeled 59 soundscapes)")
print("=" * 65)
print(f"  {'Model':<20} {'OOF macro AUC':>14}")
print("-" * 45)
arch_list = ["baseline", "lgbm", "xgb", "mlp", "bigru", "tcn",
             "transformer", "ssm", "ft_transformer", "cnn1d"]
for k in arch_list:
    marker  = " << best" if k == best_arch else ""
    auc_val = oof_aucs.get(k, float("nan"))
    print(f"  {k:<20} {auc_val:>14.4f}{marker}")
print("=" * 65)
print(f"\n[done]  All artifacts saved to: {OUT_DIR}")

# ─── Excel export ──────────────────────────────────────────────────────────────
rows = []
for k in arch_list:
    rows.append({
        "arch"                 : k,
        "oof_macro_auc_labeled": round(oof_aucs.get(k, float("nan")), 6),
        "best"                 : k == best_arch,
        "trained_date"         : time.strftime("%Y-%m-%d"),
        "mode"                 : "semi-supervised",
        "n_pseudo_files"       : n_pseudo_files,
    })
df_results = pd.DataFrame(rows)
excel_path = OUT_DIR / "stacker_results_ss.xlsx"

if excel_path.exists():
    with pd.ExcelWriter(str(excel_path), engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        df_results.to_excel(writer, sheet_name="OOF_AUC_SS", index=False)
else:
    df_results.to_excel(str(excel_path), index=False, sheet_name="OOF_AUC_SS")
print(f"  Excel saved → {excel_path}")

# ─── W&B final summary + finish ───────────────────────────────────────────────
wandb.log({
    "summary/best_arch"         : best_arch,
    "summary/best_oof_auc"      : oof_aucs[best_arch],
    "summary/n_pseudo_files"    : n_pseudo_files,
    "summary/n_total_files"     : n_all_files,
    **{f"summary/oof_{k}": oof_aucs.get(k, float("nan")) for k in arch_list},
})
wandb.finish()
print("[wandb] run finished")
