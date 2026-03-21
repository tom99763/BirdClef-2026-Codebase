"""
Local evaluation of temporal smoothing experiments.

Compares all 4 post-processing methods on the same soundscape 4-fold splits
used for SED training, ensuring apples-to-apples comparison.

Methods evaluated:
  0. Baseline:         LGBM probe + fixed Gaussian logit smooth (0-910)
  1. LearnableConv:    Per-class Conv1d FIR (1170 params) — v3-learnable-smooth
  2. SoftOrderStats:   Per-class L-estimator / soft ranking (1170 params)
  3. MultiScaleConv:   3-branch K=3,7,11 + per-class gate (~6318 params)
  4. BilateralSmooth:  Content-adaptive bilateral filter (234 params)

Usage:
    python scripts/eval_smooth_experiments.py [--rebuild_cache]

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


def gaussian_smooth(logits):
    n_files = logits.shape[0] // N_WINDOWS
    sm = logits.reshape(n_files, N_WINDOWS, NUM_CLASSES).copy()
    for i in range(n_files):
        sm[i] = convolve1d(sm[i], GAUSSIAN_KERN, axis=0, mode="nearest")
    return sm.reshape(-1, NUM_CLASSES)


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


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild extended 66-file Perch cache")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs per model")
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
        """
        if make_model_fn is not None:
            print(f"\n── Training (fold-aware OOF): {name} ──────────────────────────────")
            # For each fold k: train on folds {0,1,2,3}\{k}, predict on fold k
            smoothed = logits.copy()
            for k in range(4):
                val_mask_k = fold_id == k
                if val_mask_k.sum() == 0:
                    continue
                model_k = make_model_fn()
                train_smooth_model(model_k, logits, Y, fold_id,
                                   epochs=args.epochs, lr=lr, l2=l2, eval_fold=k)
                # Apply to eval fold only
                with torch.no_grad():
                    smoothed[val_mask_k] = model_k(logits)[val_mask_k]
                print(f"  fold{k} done", end="\r")
            logits = smoothed
            print()

        probs = sigmoid(logits)
        fold_aucs = eval_per_fold(probs, Y, fold_id)
        results[name] = fold_aucs

        fold_str = "  ".join(f"f{k}={fold_aucs.get(f'fold{k}', 0):.4f}" for k in range(4))
        print(f"{name:30s}  {fold_str}  mean={fold_aucs['mean']:.4f}")
        return fold_aucs

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"{'Method':<30}  {'fold0':>8}  {'fold1':>8}  {'fold2':>8}  {'fold3':>8}  {'mean':>8}")
    print("-" * 70)

    # 0. Raw (no probe, no smooth)
    evaluate("0.Raw (no probe)",  oof_base)

    # 1. Probe only (no smooth)
    evaluate("1.Probe (no smooth)", oof_final)

    # 2. Gaussian (fixed 0-910 kernel, baseline — no training needed)
    gauss_logits = gaussian_smooth(oof_final)
    evaluate("2.Gaussian (fixed)",  gauss_logits)

    # 3. Learnable Conv1d — gaussian init (fold-aware OOF training)
    evaluate("3.LearnableConv/gaussian", oof_final,
             make_model_fn=lambda: LearnableConv(init_mode="gaussian",
                                                  event_idx=idx_event, texture_idx=idx_texture))

    # 4. Learnable Conv1d — asymmetric init
    evaluate("4.LearnableConv/asymmetric", oof_final,
             make_model_fn=lambda: LearnableConv(init_mode="asymmetric",
                                                  event_idx=idx_event, texture_idx=idx_texture))

    # 5. Soft Order Stats — class_specific init (event→max, texture→mean)
    evaluate("5.SoftOrderStats/class_spec", oof_final,
             make_model_fn=lambda: SoftOrderStats(init_mode="class_specific",
                                                   event_idx=idx_event, texture_idx=idx_texture))

    # 6. Soft Order Stats — median init
    evaluate("6.SoftOrderStats/median", oof_final,
             make_model_fn=lambda: SoftOrderStats(init_mode="median_biased"))

    # 7. Multi-Scale Conv — class_specific gate (Aves→K=3, Texture→K=11)
    evaluate("7.MultiScale(3,7,11)/gate", oof_final,
             make_model_fn=lambda: MultiScaleConv(init_mode="class_specific",
                                                   event_idx=idx_event, texture_idx=idx_texture),
             l2=1e-3)

    # 8. Bilateral — sigma_v=0.5 init
    evaluate("8.Bilateral/sigma_v=0.5", oof_final,
             make_model_fn=lambda: BilateralSmooth(init_sigma_v=0.5),
             lr=0.1, l2=1e-4)

    # 9. Bilateral — sigma_v=0.1 (edge-preserving)
    evaluate("9.Bilateral/sigma_v=0.1", oof_final,
             make_model_fn=lambda: BilateralSmooth(init_sigma_v=0.1),
             lr=0.1, l2=1e-4)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary (sorted by mean AUC)  [fold-aware OOF training, no train/val leak]")
    print("=" * 70)
    ranked = sorted(results.items(), key=lambda x: -x[1]["mean"])
    for rank, (name, r) in enumerate(ranked, 1):
        delta = r["mean"] - results["2.Gaussian (fixed)"]["mean"]
        marker = " ⭐" if r["mean"] > results["2.Gaussian (fixed)"]["mean"] else ""
        print(f"  #{rank:2d}  {name:35s}  mean={r['mean']:.4f}  vs Gaussian {delta:+.4f}{marker}")

    # ── Save results ───────────────────────────────────────────────────────────
    import json
    out_path = Path("outputs/smooth_experiments_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"epochs": args.epochs, "results": results}, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Print diagnostics for top model ───────────────────────────────────────
    best_name, best_r = ranked[0]
    print(f"\nBest method: {best_name} (mean AUC={best_r['mean']:.4f})")


if __name__ == "__main__":
    main()
