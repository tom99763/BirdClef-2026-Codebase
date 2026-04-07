#!/usr/bin/env python3
"""
run_probe_experiments.py
========================
Systematically evaluates multiple Perch probe configurations.
GroupKFold-5 OOF AUC for each method. Writes results to Excel.

Experiments:
  baseline   – current pipeline (PCA64 + LogReg + LP++ init)
  P1         – ABT(10) + CL2N + kNN-15
  P2         – baseline + LaplacianShot post-processing
  P3         – Perch(1536) + SED-B0(1280) concat fusion  [needs sed_emb.npy]
  A1_fecam   – FeCAM: Mahalanobis NCM with Ledoit-Wolf
  A2_cl2n    – CL2N + kNN-15 (no ABT)
  A3_abt     – ABT(10) + PCA(64) + LogReg (compare rogue-dim removal)
  B1_umap    – UMAP(64) + CL2N + kNN-15
  B2_sed     – same as P3
  C1_mlp     – trainable MLP adapter 1536→512→128, then kNN
  D1_mreach  – Mutual Reachability CL2N+kNN (HDBSCAN distance, noise-robust)
  D2_mst     – MST label propagation via mutual reachability
  D3_hdbscan – HDBSCAN sub-prototype NCM (per-class acoustic sub-clusters)
  D5_lp      – Label Propagation on support graph → refined labels → D1 retrieval
  D6_transductive – Transductive LP: per-file joint graph (support + query windows)
  D7_denseprot    – Density-weighted prototype NCM (inverse core-dist weights)
  D8_cross_attn   – Temperature-scaled softmax kNN cross-attention (T=10)
  D9_gcn          – 2-layer trained GCN on support set (GPU-accelerated, ASL loss)

Usage:
  python scripts/run_probe_experiments.py
  python scripts/run_probe_experiments.py --experiments P1 P2 A1_fecam
  python scripts/run_probe_experiments.py --sed_emb outputs/sed_probe_emb.npy
"""
import argparse
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent

# ── Paths ──────────────────────────────────────────────────────────────────────
META_PATH  = ROOT / 'submissions_v2/few_shot/full_perch_meta.parquet'
NPZ_PATH   = ROOT / 'submissions_v2/few_shot/full_perch_arrays.npz'
DATA_DIR   = ROOT / 'birdclef-2026'
EXCEL_PATH = ROOT / 'reports/probe_experiments.xlsx'
N_WINDOWS  = 12


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    meta_full = pd.read_parquet(META_PATH)
    arr = np.load(NPZ_PATH)
    emb_full        = arr['emb_full'].astype(np.float32)          # (708, 1536)
    scores_full_raw = arr['scores_full_raw'].astype(np.float32)   # (708, 234)

    sample_sub    = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

    sc_labels = pd.read_csv(DATA_DIR / 'train_soundscapes_labels.csv')
    sc_labels['primary_label'] = sc_labels['primary_label'].astype(str)

    def parse_labels(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(';') if t.strip()]

    sc_clean = (
        sc_labels.groupby(['filename', 'start', 'end'])['primary_label']
        .apply(lambda s: sorted(set(lbl for x in s for lbl in parse_labels(x))))
        .reset_index(name='label_list')
    )
    sc_clean['end_sec'] = pd.to_timedelta(sc_clean['end']).dt.total_seconds().astype(int)
    sc_clean['row_id']  = (sc_clean['filename'].str.replace('.ogg', '', regex=False)
                           + '_' + sc_clean['end_sec'].astype(str))

    Y_SC = np.zeros((len(sc_clean), len(PRIMARY_LABELS)), dtype=np.uint8)
    for i, labels in enumerate(sc_clean['label_list']):
        idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
        if idxs:
            Y_SC[i, idxs] = 1

    rid_to_row = {rid: i for i, rid in enumerate(sc_clean['row_id'])}
    aligned    = [rid_to_row[rid] for rid in meta_full['row_id']]
    Y_FULL     = Y_SC[aligned]

    # Apply PT-MAP power transform (beta=0.5) — same as production probe_cache.pkl
    # sign(x)*|x|^0.5 compresses heavy-tail Perch dims → better geometry
    PTMAP_BETA = 0.5
    emb_full = np.sign(emb_full) * np.abs(emb_full) ** PTMAP_BETA
    print(f'Support clips: {len(emb_full)}  Classes with positives: {(Y_FULL.sum(0)>0).sum()}/234')
    print(f'PT-MAP beta={PTMAP_BETA} applied. emb range: {emb_full.min():.4f} to {emb_full.max():.4f}')
    return meta_full, emb_full, scores_full_raw, Y_FULL, PRIMARY_LABELS


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)

def cl2n(x, mean_vec=None):
    if mean_vec is None:
        mean_vec = x.mean(axis=0)
    centered = x - mean_vec
    return l2_normalize(centered), mean_vec

def all_but_top(x, n_top=10):
    """Remove top n_top PCA directions (rogue/environment dims)."""
    mu = x.mean(axis=0)
    X  = x - mu
    pca_top = PCA(n_components=n_top, random_state=42)
    pca_top.fit(X)
    proj = X @ pca_top.components_.T @ pca_top.components_
    return (X - proj) + mu

def tukey(x, lam=0.5):
    return np.sign(x) * np.abs(x) ** lam


# ══════════════════════════════════════════════════════════════════════════════
# Classifiers
# ══════════════════════════════════════════════════════════════════════════════

def predict_knn(Z_tr, Y_tr, Z_va, k=15):
    """Per-class binary kNN. Returns (N_va, 234) proba."""
    n_cls = Y_tr.shape[1]
    preds = np.zeros((len(Z_va), n_cls), dtype=np.float32)
    sim   = Z_va @ Z_tr.T   # cosine sim if L2-normalized (N_va, N_tr)
    for c in range(n_cls):
        pos_count = Y_tr[:, c].sum()
        if pos_count == 0:
            continue
        top_k_idx = np.argsort(sim, axis=1)[:, -k:]
        for i in range(len(Z_va)):
            neighbors_y = Y_tr[top_k_idx[i], c]
            preds[i, c] = neighbors_y.mean()
    return preds


def predict_fecam(Z_tr, Y_tr, Z_va, min_pos=2):
    """FeCAM: Mahalanobis NCM with Ledoit-Wolf per-class covariance."""
    n_cls = Y_tr.shape[1]
    preds = np.zeros((len(Z_va), n_cls), dtype=np.float32)
    Z_tr_t = tukey(Z_tr)
    Z_va_t = tukey(Z_va)
    for c in range(n_cls):
        mask = Y_tr[:, c].astype(bool)
        if mask.sum() < min_pos:
            if mask.sum() == 1:
                # Euclidean fallback for singletons
                proto = Z_tr_t[mask].mean(axis=0)
                preds[:, c] = -np.sum((Z_va_t - proto) ** 2, axis=1)
            continue
        X_c = Z_tr_t[mask]
        mu_c = X_c.mean(axis=0)
        lw = LedoitWolf().fit(X_c)
        cov = lw.covariance_
        std = np.sqrt(np.diag(cov))
        corr = cov / (np.outer(std, std) + 1e-8)
        inv_corr = np.linalg.pinv(corr)
        diff = Z_va_t - mu_c
        preds[:, c] = -np.einsum('nd,dd,nd->n', diff, inv_corr, diff)
    # Convert Mahalanobis scores to [0,1] via per-class sigmoid normalization
    for c in range(n_cls):
        col = preds[:, c]
        col_range = col.max() - col.min()
        if col_range > 0:
            preds[:, c] = (col - col.min()) / col_range
    return preds


def predict_logreg(Z_tr, Y_tr, Z_va, C=0.5, min_pos=8):
    """Per-class LogReg (LP++ style: proto-initialized)."""
    n_cls = Y_tr.shape[1]
    preds = np.zeros((len(Z_va), n_cls), dtype=np.float32)
    pos_counts = Y_tr.sum(axis=0)
    for c in range(n_cls):
        n_pos = pos_counts[c]
        if n_pos == 0:
            continue
        y = Y_tr[:, c]
        if y.sum() == len(y):
            preds[:, c] = 1.0
            continue
        if n_pos < min_pos:
            # Prototype fallback
            proto = Z_tr[y.astype(bool)].mean(axis=0)
            proto /= np.linalg.norm(proto) + 1e-8
            preds[:, c] = np.clip(Z_va @ proto, 0, 1)
            continue
        clf = LogisticRegression(C=C, max_iter=300, solver='lbfgs', class_weight='balanced')
        clf.fit(Z_tr, y)
        preds[:, c] = clf.predict_proba(Z_va)[:, 1]
    return preds


def predict_mlp(emb_tr, Y_tr, emb_va, hidden=512, out_dim=128, epochs=200, lr=1e-3):
    """Trainable MLP adapter: 1536→512→128 → kNN-15."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(emb_tr.shape[1], hidden), nn.LayerNorm(hidden),
                nn.GELU(), nn.Dropout(0.3),
                nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim),
            )
        def forward(self, x):
            return F.normalize(self.net(x), dim=-1)

    X_tr = torch.tensor(emb_tr, dtype=torch.float32)
    Y_t  = torch.tensor(Y_tr, dtype=torch.float32)
    model = Adapter()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        Z = model(X_tr)
        loss = F.binary_cross_entropy_with_logits(Z @ Z.T, Y_t @ Y_t.T / Y_t.sum(1).clamp(1)[:, None])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        Z_tr_mlp = model(X_tr).numpy()
        Z_va_mlp = model(torch.tensor(emb_va, dtype=torch.float32)).numpy()
    return predict_knn(Z_tr_mlp, Y_tr, Z_va_mlp, k=15)


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing: LaplacianShot
# ══════════════════════════════════════════════════════════════════════════════

def laplacian_shot(preds_file, emb_file, lam=0.5, n_iter=20, k=5):
    """
    preds_file: (12, C) initial predictions for one soundscape file
    emb_file:   (12, D) L2-normalized embeddings for same file
    Returns:    (12, C) refined predictions
    """
    Q = len(preds_file)
    emb = emb_file / (np.linalg.norm(emb_file, axis=1, keepdims=True) + 1e-8)
    sim = emb @ emb.T  # (Q, Q)
    np.fill_diagonal(sim, -np.inf)
    W   = np.zeros_like(sim)
    for i in range(Q):
        top_k = np.argsort(sim[i])[-k:]
        W[i, top_k] = np.clip(sim[i, top_k], 0, None)
    W = np.maximum(W, W.T)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(W.sum(1)) + 1e-8))
    L = np.eye(Q) - D_inv_sqrt @ W @ D_inv_sqrt

    Y = preds_file.copy()
    for _ in range(n_iter):
        grad_Y = lam * (L @ Y)
        Y = np.clip(preds_file - grad_Y, 0, 1)
    return Y


def apply_laplacian_shot_oof(va_preds, va_embs, va_filenames, lam=0.5, n_iter=20):
    """Apply LaplacianShot per soundscape file across the val fold."""
    result = va_preds.copy()
    for fn in np.unique(va_filenames):
        mask = va_filenames == fn
        if mask.sum() < 2:
            continue
        result[mask] = laplacian_shot(va_preds[mask], va_embs[mask], lam=lam, n_iter=n_iter)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# OOF evaluation engine
# ══════════════════════════════════════════════════════════════════════════════

def oof_auc(meta_full, emb_full, Y_FULL, probe_fn, n_splits=5):
    """
    probe_fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va) → (N_va, 234) predictions
    Returns macro OOF AUC (skip empty classes in each val fold).
    """
    gkf    = GroupKFold(n_splits=n_splits)
    groups = meta_full['filename'].to_numpy()
    oof_preds = np.zeros_like(Y_FULL, dtype=np.float32)

    for fold_i, (tr_idx, va_idx) in enumerate(
            tqdm(list(gkf.split(emb_full, groups=groups)), desc='OOF folds')):
        tr_idx = np.sort(tr_idx)
        va_idx = np.sort(va_idx)
        emb_tr = emb_full[tr_idx]
        emb_va = emb_full[va_idx]
        Y_tr   = Y_FULL[tr_idx]
        meta_tr = meta_full.iloc[tr_idx].reset_index(drop=True)
        meta_va = meta_full.iloc[va_idx].reset_index(drop=True)
        oof_preds[va_idx] = probe_fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va)

    # Macro AUC (skip classes with no positives in full set)
    keep = Y_FULL.sum(axis=0) > 0
    auc  = roc_auc_score(Y_FULL[:, keep], oof_preds[:, keep], average='macro')
    return auc, oof_preds


# ══════════════════════════════════════════════════════════════════════════════
# Experiment definitions
# ══════════════════════════════════════════════════════════════════════════════

def make_baseline_fn():
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        emb_tr_l2 = l2_normalize(emb_tr)
        mean_vec  = emb_tr_l2.mean(axis=0)
        emb_tr_c  = emb_tr_l2 - mean_vec
        emb_va_l2 = l2_normalize(emb_va)
        emb_va_c  = emb_va_l2 - mean_vec
        pca = PCA(n_components=64, whiten=True, random_state=42)
        Z_tr = pca.fit_transform(emb_tr_c).astype(np.float32)
        Z_va = pca.transform(emb_va_c).astype(np.float32)
        return predict_logreg(Z_tr, Y_tr, Z_va, C=0.5, min_pos=8)
    return fn


def make_p1_abt_cl2n_knn_fn():
    """Priority 1: ABT(10) + CL2N + kNN-15."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        # ABT on train
        emb_tr_abt = all_but_top(emb_tr, n_top=10)
        # Fit ABT transform from train, apply to val
        mu_tr = emb_tr.mean(axis=0)
        X_tr  = emb_tr - mu_tr
        X_va  = emb_va - mu_tr
        pca_top = PCA(n_components=10, random_state=42)
        pca_top.fit(X_tr)
        proj_tr = X_tr @ pca_top.components_.T @ pca_top.components_
        proj_va = X_va @ pca_top.components_.T @ pca_top.components_
        emb_tr_abt = (X_tr - proj_tr) + mu_tr
        emb_va_abt = (X_va - proj_va) + mu_tr
        # CL2N
        Z_tr, mean_vec = cl2n(emb_tr_abt)
        centered_va    = emb_va_abt - mean_vec
        Z_va           = l2_normalize(centered_va)
        return predict_knn(Z_tr, Y_tr, Z_va, k=15)
    return fn


def make_p2_laplacianshot_fn():
    """Priority 2: PCA64+LogReg baseline + LaplacianShot post-processing."""
    baseline_fn = make_baseline_fn()
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        base_preds = baseline_fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va)
        va_emb_l2  = l2_normalize(emb_va)
        va_fnames  = meta_va['filename'].to_numpy()
        return apply_laplacian_shot_oof(base_preds, va_emb_l2, va_fnames, lam=0.4, n_iter=20)
    return fn


def make_a1_fecam_fn():
    """Method A1: FeCAM - Mahalanobis NCM."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        emb_tr_l2 = l2_normalize(emb_tr)
        mean_vec  = emb_tr_l2.mean(axis=0)
        emb_tr_c  = emb_tr_l2 - mean_vec
        emb_va_l2 = l2_normalize(emb_va)
        emb_va_c  = emb_va_l2 - mean_vec
        # PCA(64) for tractable Mahalanobis inversion
        pca = PCA(n_components=64, whiten=False, random_state=42)
        Z_tr = pca.fit_transform(emb_tr_c).astype(np.float32)
        Z_va = pca.transform(emb_va_c).astype(np.float32)
        return predict_fecam(Z_tr, Y_tr, Z_va, min_pos=2)
    return fn


def make_a2_cl2n_knn_fn():
    """Method A2: CL2N + kNN-15 (no ABT, raw 1536-dim)."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        return predict_knn(Z_tr, Y_tr, Z_va, k=15)
    return fn


def make_a3_abt_pca_logreg_fn():
    """Method A3: ABT(10) + PCA(64) + LogReg (rogue dim removal + standard probe)."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        mu_tr = emb_tr.mean(axis=0)
        X_tr  = emb_tr - mu_tr
        X_va  = emb_va - mu_tr
        pca_top = PCA(n_components=10, random_state=42)
        pca_top.fit(X_tr)
        proj_tr   = X_tr @ pca_top.components_.T @ pca_top.components_
        proj_va   = X_va @ pca_top.components_.T @ pca_top.components_
        emb_tr_c  = X_tr - proj_tr
        emb_va_c  = X_va - proj_va
        emb_tr_l2 = l2_normalize(emb_tr_c)
        mean_vec  = emb_tr_l2.mean(axis=0)
        emb_tr_cp = emb_tr_l2 - mean_vec
        emb_va_l2 = l2_normalize(emb_va_c)
        emb_va_cp = emb_va_l2 - mean_vec
        pca = PCA(n_components=64, whiten=True, random_state=42)
        Z_tr = pca.fit_transform(emb_tr_cp).astype(np.float32)
        Z_va = pca.transform(emb_va_cp).astype(np.float32)
        return predict_logreg(Z_tr, Y_tr, Z_va, C=0.5, min_pos=8)
    return fn


def make_b1_umap_fn():
    """Method B1: UMAP(64) + CL2N + kNN-15."""
    try:
        import umap
    except ImportError:
        print('  SKIP B1_umap: umap-learn not installed. pip install umap-learn')
        return None

    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        reducer = umap.UMAP(n_components=64, n_neighbors=15, min_dist=0.1,
                             metric='cosine', random_state=42, n_jobs=4)
        Z_tr_raw = reducer.fit_transform(emb_tr)
        Z_va_raw = reducer.transform(emb_va)
        Z_tr, mean_vec = cl2n(Z_tr_raw.astype(np.float32))
        Z_va = l2_normalize((Z_va_raw.astype(np.float32)) - mean_vec)
        return predict_knn(Z_tr, Y_tr, Z_va, k=15)
    return fn


def make_p3_sed_fusion_fn(sed_emb_full):
    """Priority 3 / B2: Perch(1536) + SED(1280) concat, PCA(128) + kNN-15."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        # emb_tr and emb_va are Perch embs; we pass indices via meta
        # Actually we need to track indices — handled via meta_tr/meta_va
        # Use filename+row_id alignment from outer scope (sed_emb_full pre-aligned)
        # Simple approach: pass sed_emb slices in same order as emb_tr/emb_va
        # (guaranteed by GroupKFold over aligned arrays)
        raise RuntimeError('Use make_p3_sed_fusion_fn_aligned instead')
    return fn


def make_p3_sed_fusion_fn_aligned(sed_emb_full):
    """P3/B2 with pre-aligned SED embeddings (same clip order as Perch)."""
    def fn(emb_tr_perch, Y_tr, emb_va_perch, meta_tr, meta_va, idx_tr, idx_va):
        sed_tr = sed_emb_full[idx_tr]
        sed_va = sed_emb_full[idx_va]
        # L2-normalize each modality separately
        perch_tr = l2_normalize(emb_tr_perch)
        perch_va = l2_normalize(emb_va_perch)
        sed_tr_n = l2_normalize(sed_tr)
        sed_va_n = l2_normalize(sed_va)
        # Concat
        joint_tr = np.hstack([perch_tr, sed_tr_n])   # (N, 2816)
        joint_va = np.hstack([perch_va, sed_va_n])
        # PCA(128) on joint
        scaler = StandardScaler()
        joint_tr_s = scaler.fit_transform(joint_tr)
        joint_va_s = scaler.transform(joint_va)
        pca = PCA(n_components=128, whiten=True, random_state=42)
        Z_tr = pca.fit_transform(joint_tr_s).astype(np.float32)
        Z_va = pca.transform(joint_va_s).astype(np.float32)
        Z_tr_cl2n, mean_vec = cl2n(Z_tr)
        Z_va_cl2n = l2_normalize(Z_va - mean_vec)
        return predict_knn(Z_tr_cl2n, Y_tr, Z_va_cl2n, k=15)
    return fn


def make_c1_mlp_fn():
    """Method C1: Trainable MLP adapter 1536→512→128, then kNN-15."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        return predict_mlp(emb_tr, Y_tr, emb_va, hidden=512, out_dim=128, epochs=150)
    return fn


# ══════════════════════════════════════════════════════════════════════════════
# HDBSCAN-inspired experiments: D1, D2, D3
# ══════════════════════════════════════════════════════════════════════════════

def make_d1_mreach_knn_fn(k_core=10, k_nn=15):
    """D1: Mutual Reachability CL2N+kNN.
    Replaces cosine distance with HDBSCAN mutual reachability distance.
    Penalizes isolated/noisy support points via core distance.
    d_mreach(q,i) = max(core_dist_k(i), cosine_dist(q,i))
    Retrieval: lowest d_mreach → highest trustworthiness-weighted similarity.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        # CL2N preprocessing (same as A2)
        Z_tr, mean_vec = cl2n(emb_tr)   # (N_tr, D), L2-normed
        Z_va = l2_normalize(emb_va - mean_vec)  # (N_va, D)

        # Pairwise cosine distances within support
        sim_tr   = Z_tr @ Z_tr.T                        # (N_tr, N_tr)
        dist_tr  = np.clip(1.0 - sim_tr, 0, 2)
        np.fill_diagonal(dist_tr, np.inf)
        # core_dist[i] = distance to k_core-th nearest support neighbor
        sorted_d = np.sort(dist_tr, axis=1)
        core_dist = sorted_d[:, min(k_core - 1, sorted_d.shape[1] - 1)]  # (N_tr,)
        np.fill_diagonal(dist_tr, 0)

        # Query-support cosine distances
        sim_va   = Z_va @ Z_tr.T                        # (N_va, N_tr)
        dist_va  = np.clip(1.0 - sim_va, 0, 2)         # (N_va, N_tr)

        # Mutual reachability: max(core_dist(support_i), dist(q, i))
        # core_dist broadcast: (1, N_tr)
        d_mreach = np.maximum(core_dist[None, :], dist_va)  # (N_va, N_tr)

        # kNN by lowest mutual reachability distance → weighted vote
        n_cls = Y_tr.shape[1]
        preds = np.zeros((len(Z_va), n_cls), dtype=np.float32)
        for i in range(len(Z_va)):
            nn_idx = np.argpartition(d_mreach[i], k_nn)[:k_nn]
            # Weight by inverse mreach distance (density-aware)
            w = 1.0 / (d_mreach[i, nn_idx] + 1e-6)
            w = w / w.sum()
            preds[i] = (Y_tr[nn_idx].astype(np.float32) * w[:, None]).sum(0)
        return preds
    return fn


def make_d2_mst_knn_fn(k_core=10, k_nn=15):
    """D2: MST-based label propagation via mutual reachability.
    Builds MST over CL2N support embeddings using mutual reachability distances.
    For each query, finds k nearest support nodes then uses MST shortest-path
    distances as propagation weights (captures density-connected structure).
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr = len(Z_tr)

        # Build pairwise mutual reachability distance on support
        sim_tr   = Z_tr @ Z_tr.T
        dist_tr  = np.clip(1.0 - sim_tr, 0, 2).astype(np.float64)
        np.fill_diagonal(dist_tr, np.inf)
        sorted_d  = np.sort(dist_tr, axis=1)
        core_dist = sorted_d[:, min(k_core - 1, sorted_d.shape[1] - 1)]
        np.fill_diagonal(dist_tr, 0)
        mreach = np.maximum(np.maximum(core_dist[:, None], core_dist[None, :]),
                            dist_tr)  # (N_tr, N_tr)

        # Compute MST of support mutual reachability graph
        tree = minimum_spanning_tree(csr_matrix(mreach))
        tree_sym = tree + tree.T   # symmetric

        # Shortest-path distance along MST (captures density-connected topology)
        sp_dist = shortest_path(tree_sym, method='D', directed=False)  # (N_tr, N_tr)

        # For each query: find k nearest supports by cosine, then re-weight
        # by MST shortest-path distance (penalizes paths through sparse regions)
        sim_va   = Z_va @ Z_tr.T                # (N_va, N_tr)
        dist_va  = np.clip(1.0 - sim_va, 0, 2)

        n_cls = Y_tr.shape[1]
        preds = np.zeros((len(Z_va), n_cls), dtype=np.float32)

        for i in range(len(Z_va)):
            # Anchor: single nearest support node
            anchor = int(np.argmin(dist_va[i]))
            # MST-path distance from anchor to all other support nodes
            mst_dists = sp_dist[anchor]           # (N_tr,)
            # Combined score: weight by both cosine sim and MST proximity
            score = sim_va[i] / (1.0 + mst_dists)  # high sim + MST-close = best
            nn_idx = np.argpartition(-score, k_nn)[:k_nn]
            w = score[nn_idx]; w = np.clip(w, 0, None)
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones(k_nn) / k_nn
            preds[i] = (Y_tr[nn_idx].astype(np.float32) * w[:, None]).sum(0)
        return preds
    return fn


def make_d3_hdbscan_proto_fn(min_cluster_size=2, k_nn=15):
    """D3: HDBSCAN sub-prototype NCM.
    For each class, runs HDBSCAN to find acoustic sub-clusters (call types).
    Uses per-cluster prototypes instead of a single mean.
    Query similarity = max cosine sim to any prototype of the class.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        try:
            import hdbscan as hdbscan_lib
        except ImportError:
            print('  SKIP D3: hdbscan not installed')
            return np.zeros((len(emb_va), Y_tr.shape[1]), dtype=np.float32)

        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        n_cls = Y_tr.shape[1]
        preds = np.zeros((len(Z_va), n_cls), dtype=np.float32)

        for c in range(n_cls):
            mask = Y_tr[:, c].astype(bool)
            n_pos = mask.sum()
            if n_pos == 0:
                continue
            pos_embs = Z_tr[mask]   # (n_pos, D), already L2-normed

            if n_pos < 4:
                # Too few for clustering → single prototype
                proto = l2_normalize(pos_embs.mean(0, keepdims=True))
                preds[:, c] = np.clip(Z_va @ proto[0], 0, 1)
                continue

            # HDBSCAN clustering on positive embeddings
            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                metric='euclidean',
            )
            labels = clusterer.fit_predict(pos_embs)
            valid_cls = [l for l in np.unique(labels) if l >= 0]

            if not valid_cls:
                # All noise → single prototype
                proto = l2_normalize(pos_embs.mean(0, keepdims=True))
                preds[:, c] = np.clip(Z_va @ proto[0], 0, 1)
                continue

            # Build per-cluster prototypes + noise singleton prototypes
            protos = []
            for lbl in valid_cls:
                cluster_embs = pos_embs[labels == lbl]
                p = cluster_embs.mean(0)
                p = p / (np.linalg.norm(p) + 1e-10)
                protos.append(p)
            # Noise points (-1): treat as individual prototypes
            noise_embs = pos_embs[labels == -1]
            for ne in noise_embs:
                protos.append(ne / (np.linalg.norm(ne) + 1e-10))

            protos = np.stack(protos)   # (n_protos, D)

            # Similarity to closest prototype (soft-max over prototypes)
            sim_to_protos = Z_va @ protos.T   # (N_va, n_protos)
            # Use max similarity as the score for this class
            preds[:, c] = sim_to_protos.max(axis=1).clip(0, 1)

        return preds
    return fn


def make_d5_lp_fn(k_graph=10, alpha=0.5, n_iter=20, k_core=10, k_nn=15):
    """D5: Label Propagation on support k-NN graph → refined labels → D1 retrieval."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        # CL2N preprocessing
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)

        # ── Build support k-NN graph ──────────────────────────────────────────
        sim_tr = Z_tr @ Z_tr.T                         # (N_tr, N_tr)
        np.fill_diagonal(sim_tr, -np.inf)
        W = np.zeros_like(sim_tr)
        for i in range(len(Z_tr)):
            top_k = np.argpartition(sim_tr[i], -k_graph)[-k_graph:]
            W[i, top_k] = np.clip(sim_tr[i, top_k], 0, None)
        W = np.maximum(W, W.T)                         # symmetrize
        row_sum = W.sum(1, keepdims=True)
        S = W / (row_sum + 1e-8)                       # row-normalized

        # ── Iterative LP: F ← (1-α)·Y + α·S·F ───────────────────────────────
        F = Y_tr.astype(np.float32).copy()
        Y_float = Y_tr.astype(np.float32)
        for _ in range(n_iter):
            F = (1.0 - alpha) * Y_float + alpha * (S @ F)

        # ── D1 mutual reachability with refined labels ─────────────────────────
        sim_tr2 = Z_tr @ Z_tr.T
        np.fill_diagonal(sim_tr2, -np.inf)
        dist_tr = np.clip(1.0 - sim_tr2, 0, 2).astype(np.float32)
        sorted_d = np.sort(dist_tr, axis=1)
        core_dist = sorted_d[:, min(k_core - 1, sorted_d.shape[1] - 1)]
        np.fill_diagonal(dist_tr, 0)

        dist_va = np.clip(1.0 - (Z_va @ Z_tr.T), 0, 2).astype(np.float32)
        d_mreach = np.maximum(core_dist[None, :], dist_va)

        preds = np.zeros((len(Z_va), Y_tr.shape[1]), dtype=np.float32)
        for i in range(len(Z_va)):
            nn_idx = np.argpartition(d_mreach[i], k_nn)[:k_nn]
            w = 1.0 / (d_mreach[i, nn_idx] + 1e-6)
            w = w / w.sum()
            preds[i] = (F[nn_idx] * w[:, None]).sum(0)
        return preds
    return fn


def make_d6_transductive_fn(k_graph=8, alpha=0.5, n_iter=30, k_nn=15):
    """D6: Transductive LP — per-file joint graph (support + query windows)."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        # CL2N preprocessing
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        NUM_CLASSES = Y_tr.shape[1]
        N_tr = len(Z_tr)

        preds = np.zeros((len(Z_va), NUM_CLASSES), dtype=np.float32)

        # Process each soundscape file separately
        filenames = meta_va['filename'].to_numpy()
        for fn_name in np.unique(filenames):
            qmask = filenames == fn_name
            q_idx = np.where(qmask)[0]
            N_q = len(q_idx)
            if N_q == 0:
                continue

            # Build joint embedding: support + query
            Z_joint = np.concatenate([Z_tr, Z_va[q_idx]], axis=0)  # (N_tr+N_q, D)
            N_joint = len(Z_joint)

            # Build k-NN graph over joint embeddings
            sim_joint = Z_joint @ Z_joint.T
            np.fill_diagonal(sim_joint, -np.inf)
            W = np.zeros((N_joint, N_joint), dtype=np.float32)
            for i in range(N_joint):
                top_k = np.argpartition(sim_joint[i], -k_graph)[-k_graph:]
                W[i, top_k] = np.clip(sim_joint[i, top_k], 0, None)
            W = np.maximum(W, W.T)
            row_sum = W.sum(1, keepdims=True)
            S = (W / (row_sum + 1e-8)).astype(np.float32)

            # Initialize labels: support=Y_tr (fixed), query=0
            F = np.zeros((N_joint, NUM_CLASSES), dtype=np.float32)
            F[:N_tr] = Y_tr.astype(np.float32)

            # Iterative LP with support clamped
            Y_clamped = F.copy()
            for _ in range(n_iter):
                F = alpha * (S @ F)
                F[:N_tr] = Y_clamped[:N_tr]          # clamp support labels
                F[N_tr:] = np.clip(F[N_tr:], 0, 1)

            preds[q_idx] = F[N_tr:]
        return preds
    return fn


def make_d7_denseprot_fn(k_core=10):
    """D7: Density-weighted prototype NCM (inverse core-dist weights)."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        # CL2N preprocessing
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        NUM_CLASSES = Y_tr.shape[1]

        # Compute core distances on support
        sim_tr = Z_tr @ Z_tr.T
        np.fill_diagonal(sim_tr, -np.inf)
        dist_tr = np.clip(1.0 - sim_tr, 0, 2).astype(np.float32)
        sorted_d = np.sort(dist_tr, axis=1)
        core_dist = sorted_d[:, min(k_core - 1, sorted_d.shape[1] - 1)]
        density = 1.0 / (core_dist + 1e-6)            # (N_tr,)

        # Per-class density-weighted prototype
        protos = np.zeros((NUM_CLASSES, Z_tr.shape[1]), dtype=np.float32)
        for c in range(NUM_CLASSES):
            pos_mask = Y_tr[:, c] > 0
            if not pos_mask.any():
                continue
            w = density[pos_mask]
            w = w / (w.sum() + 1e-8)
            protos[c] = (Z_tr[pos_mask] * w[:, None]).sum(0)
            norm = np.linalg.norm(protos[c])
            if norm > 1e-8:
                protos[c] /= norm

        preds = np.clip(Z_va @ protos.T, 0, 1)        # (N_va, C)
        return preds
    return fn


def make_d8_cross_attn_fn(k_nn=15, temperature=10.0):
    """D8: Temperature-scaled softmax kNN cross-attention."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        # CL2N preprocessing
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)

        sim = Z_va @ Z_tr.T                            # (N_va, N_tr)
        preds = np.zeros((len(Z_va), Y_tr.shape[1]), dtype=np.float32)

        for i in range(len(Z_va)):
            top_idx = np.argpartition(sim[i], -k_nn)[-k_nn:]
            sim_top = sim[i, top_idx]
            # Softmax with temperature
            sim_top -= sim_top.max()                   # numerical stability
            attn = np.exp(temperature * sim_top)
            attn /= (attn.sum() + 1e-8)
            preds[i] = (Y_tr[top_idx].astype(np.float32) * attn[:, None]).sum(0)
        return preds
    return fn


def _build_pyg_edge_index(Z, k, device, add_self_loops=True):
    """Build sparse COO edge_index for PyG from CL2N embeddings.
    Returns edge_index (2, E) on device.
    """
    import torch
    from torch_geometric.utils import to_undirected, add_self_loops as pyg_add_self_loops

    N = len(Z)
    sim = Z @ Z.T
    np.fill_diagonal(sim, -np.inf)

    rows, cols = [], []
    for i in range(N):
        top_k = np.argpartition(sim[i], -k)[-k:]
        rows.extend([i] * k)
        cols.extend(top_k.tolist())

    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    edge_index = to_undirected(edge_index, num_nodes=N)
    if add_self_loops:
        edge_index, _ = pyg_add_self_loops(edge_index, num_nodes=N)
    return edge_index


def _asl_loss(logits, targets, gamma_neg=4, clip=0.05):
    import torch
    xs_pos = torch.sigmoid(logits)
    xs_neg = (1.0 - xs_pos + clip).clamp(max=1)
    loss = targets * torch.log(xs_pos.clamp(min=1e-8)) \
         + (1 - targets) * torch.log(xs_neg.clamp(min=1e-8)) * xs_neg.pow(gamma_neg)
    return -loss.mean()


def make_d9_gcn_fn(hidden_dim=256, k_graph=10, epochs=300, lr=3e-4, dropout=0.3):
    """D9: Transductive 2-layer PyG GCNConv on joint support+query graph (ASL loss).

    Key fix vs old D9: the full joint graph (support + query) is built ONCE and
    used for both training and inference — no distribution shift.
    Only support nodes contribute to the loss (query labels masked out).
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CL2N preprocessing
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = Z_tr.shape[1]

        # ── Build JOINT graph (support + query) on CPU, move to device ────────
        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)          # (N, D)
        edge_index = _build_pyg_edge_index(Z_joint, k_graph, device)

        X = torch.tensor(Z_joint, dtype=torch.float32, device=device)

        # Label tensor: support rows = Y_tr, query rows = 0
        Y_full = torch.zeros(N_tr + N_va, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        # ── PyG GCN model ─────────────────────────────────────────────────────
        class GCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, NUM_CLASSES)
                self.bn    = nn.BatchNorm1d(hidden_dim)
                self.drop  = nn.Dropout(dropout)

            def forward(self, x, edge_index):
                h = torch.relu(self.bn(self.conv1(x, edge_index)))
                h = self.drop(h)
                return self.conv2(h, edge_index)

        model = GCN().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        # ── Transductive training: loss only on support nodes ─────────────────
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = model(X, edge_index)
            loss = _asl_loss(logits[train_mask], Y_full[train_mask])
            loss.backward()
            opt.step()
            scheduler.step()

        # ── Read off query node predictions ───────────────────────────────────
        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, edge_index)).cpu().numpy()

        return out[N_tr:].astype(np.float32)
    return fn


def make_d10_gat_fn(hidden_dim=256, heads=4, k_graph=10, epochs=300, lr=1e-3, dropout=0.3):
    """D10: Transductive 2-layer PyG GATConv (multi-head attention, ASL loss).

    Paper basis: Graph-Based Audio Classification (Sensors 2024) shows
    GAT > GCN > GraphSAGE for ecoacoustic classification with pre-trained embeddings.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GATConv

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CL2N preprocessing
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = Z_tr.shape[1]

        # ── Joint graph (support + query) ─────────────────────────────────────
        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)
        edge_index = _build_pyg_edge_index(Z_joint, k_graph, device)
        X = torch.tensor(Z_joint, dtype=torch.float32, device=device)

        Y_full = torch.zeros(N_tr + N_va, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        # ── PyG GAT model ─────────────────────────────────────────────────────
        class GAT(nn.Module):
            def __init__(self):
                super().__init__()
                # Layer 1: multi-head attention, concat → hidden_dim*heads
                self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
                # Layer 2: single head → NUM_CLASSES
                self.conv2 = GATConv(hidden_dim * heads, NUM_CLASSES, heads=1,
                                     concat=False, dropout=dropout)
                self.bn = nn.BatchNorm1d(hidden_dim * heads)

            def forward(self, x, edge_index):
                h = torch.relu(self.bn(self.conv1(x, edge_index)))
                return self.conv2(h, edge_index)

        model = GAT().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = model(X, edge_index)
            loss = _asl_loss(logits[train_mask], Y_full[train_mask])
            loss.backward()
            opt.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, edge_index)).cpu().numpy()

        return out[N_tr:].astype(np.float32)
    return fn


def _build_pyg_edge_index_with_attr(Z, k, device):
    """Build sparse COO edge_index + cosine-sim edge_attr for PyG.
    Returns (edge_index [2,E], edge_attr [E,1]) both on device.
    """
    import torch
    from torch_geometric.utils import to_undirected, add_self_loops as pyg_add_sl

    N = len(Z)
    sim = Z @ Z.T
    np.fill_diagonal(sim, -np.inf)

    rows, cols, attrs = [], [], []
    for i in range(N):
        top_k = np.argpartition(sim[i], -k)[-k:]
        for j in top_k:
            rows.append(i); cols.append(int(j))
            attrs.append(float(np.clip(sim[i, j], 0, 1)))

    ei = torch.tensor([rows, cols], dtype=torch.long)
    ea = torch.tensor(attrs, dtype=torch.float32).unsqueeze(1)

    # Symmetrize: average attr of (i,j) and (j,i)
    ei_T = ei.flip(0)
    ei_all = torch.cat([ei, ei_T], dim=1)
    ea_all = torch.cat([ea, ea], dim=0)

    # Deduplicate & average (sparse scatter)
    idx = ei_all[0] * N + ei_all[1]
    sort_order = torch.argsort(idx)
    idx_s = idx[sort_order]; ea_s = ea_all[sort_order]
    unique_idx, inverse = torch.unique(idx_s, return_inverse=True)
    ea_mean = torch.zeros(len(unique_idx), 1).scatter_reduce_(
        0, inverse.unsqueeze(1), ea_s, reduce='mean', include_self=False)
    row_u = (unique_idx // N).long()
    col_u = (unique_idx % N).long()
    ei_final = torch.stack([row_u, col_u], dim=0)

    # Self-loops with attr=1.0
    sl_idx = torch.arange(N, dtype=torch.long)
    sl_ei  = torch.stack([sl_idx, sl_idx], dim=0)
    sl_ea  = torch.ones(N, 1)
    ei_final = torch.cat([ei_final, sl_ei], dim=1).to(device)
    ea_final = torch.cat([ea_mean, sl_ea], dim=0).to(device)
    return ei_final, ea_final


def _enrich_node_features(Z_tr, Y_tr, Z_va, pca_dim=64, add_density=True, k_core=10):
    """Enrich CL2N node features with PCA projection + density scalar.
    Returns (X_tr_enriched, X_va_enriched) as float32 arrays.
    """
    pca = PCA(n_components=pca_dim, whiten=False, random_state=42)
    Z_pca_tr = pca.fit_transform(Z_tr).astype(np.float32)
    Z_pca_va = pca.transform(Z_va).astype(np.float32)

    if add_density:
        sim_tr = Z_tr @ Z_tr.T
        np.fill_diagonal(sim_tr, -np.inf)
        dist_tr = np.clip(1.0 - sim_tr, 0, 2)
        sorted_d = np.sort(dist_tr, axis=1)
        core_dist_tr = sorted_d[:, min(k_core - 1, sorted_d.shape[1] - 1)]
        density_tr = (1.0 / (core_dist_tr + 1e-6)).reshape(-1, 1).astype(np.float32)
        density_tr /= (density_tr.max() + 1e-8)

        # For val: compute density relative to train graph
        dist_va = np.clip(1.0 - (Z_va @ Z_tr.T), 0, 2)
        sorted_va = np.sort(dist_va, axis=1)
        core_dist_va = sorted_va[:, min(k_core - 1, sorted_va.shape[1] - 1)]
        density_va = (1.0 / (core_dist_va + 1e-6)).reshape(-1, 1).astype(np.float32)
        density_va /= (density_tr.max() + 1e-8)  # normalize by train scale

        X_tr = np.concatenate([Z_tr, Z_pca_tr, density_tr], axis=1)
        X_va = np.concatenate([Z_va, Z_pca_va, density_va], axis=1)
    else:
        X_tr = np.concatenate([Z_tr, Z_pca_tr], axis=1)
        X_va = np.concatenate([Z_va, Z_pca_va], axis=1)

    return X_tr.astype(np.float32), X_va.astype(np.float32)


def make_d11_sage_fn(hidden_dim=256, k_graph=10, epochs=300, lr=1e-3, dropout=0.3):
    """D11: Transductive 2-layer PyG GraphSAGE (mean aggregation, ASL loss)."""
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import SAGEConv

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = Z_tr.shape[1]

        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)
        edge_index = _build_pyg_edge_index(Z_joint, k_graph, device)
        X = torch.tensor(Z_joint, dtype=torch.float32, device=device)

        Y_full = torch.zeros(N_tr + N_va, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        class SAGE(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv(in_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, NUM_CLASSES)
                self.bn   = nn.BatchNorm1d(hidden_dim)
                self.drop = nn.Dropout(dropout)

            def forward(self, x, edge_index):
                h = torch.relu(self.bn(self.conv1(x, edge_index)))
                h = self.drop(h)
                return self.conv2(h, edge_index)

        model = SAGE().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = model(X, edge_index)
            loss = _asl_loss(logits[train_mask], Y_full[train_mask])
            loss.backward()
            opt.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, edge_index)).cpu().numpy()

        return out[N_tr:].astype(np.float32)
    return fn


# ── D12: APPNP ───────────────────────────────────────────────────────────────
def make_d12_appnp_fn(hidden_dim=512, k_hops=10, alpha=0.15, epochs=400,
                      lr=1e-3, dropout=0.4, k_graph=10):
    """D12: APPNP — MLP feature transform + PageRank-style propagation.

    Decouples learning (MLP) from propagation (fixed K-step personalized PageRank).
    Provably avoids oversmoothing. Equivalent to infinite-depth GCN on small graphs.
    Best choice when graph is small and labels are sparse.
    Paper: Klicpera et al., ICLR 2019 "Predict then Propagate".
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch, torch.nn as nn
        from torch_geometric.nn import APPNP

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = Z_tr.shape[1]

        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)
        edge_index = _build_pyg_edge_index(Z_joint, k_graph, device)
        X = torch.tensor(Z_joint, dtype=torch.float32, device=device)
        Y_full = torch.zeros(N_tr + N_va, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        class APPNPNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(in_dim, hidden_dim)
                self.lin2 = nn.Linear(hidden_dim, NUM_CLASSES)
                self.bn   = nn.BatchNorm1d(hidden_dim)
                self.drop = nn.Dropout(dropout)
                self.prop = APPNP(K=k_hops, alpha=alpha)

            def forward(self, x, edge_index):
                h = self.drop(torch.relu(self.bn(self.lin1(x))))
                h = self.lin2(h)
                return self.prop(h, edge_index)   # propagate logits

        model = APPNPNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X, edge_index)
            loss = _asl_loss(out[train_mask], Y_full[train_mask])
            loss.backward(); opt.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, edge_index)).cpu().numpy()
        return out[N_tr:].astype(np.float32)
    return fn


# ── D13: TransformerConv with cosine-sim edge features ───────────────────────
def make_d13_transformer_conv_fn(hidden_dim=256, heads=4, k_graph=10,
                                  epochs=300, lr=1e-3, dropout=0.3):
    """D13: TransformerConv with edge_attr = cosine similarity between clips.

    Unlike GAT (which computes attention from node features alone),
    TransformerConv conditions attention on edge features:
      a_ij = softmax( (W_Q x_i · W_K x_j) / √d + W_E e_ij )
    Edge feature = cosine sim → explicitly leverages acoustic proximity.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch, torch.nn as nn
        from torch_geometric.nn import TransformerConv

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = Z_tr.shape[1]

        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)
        edge_index, edge_attr = _build_pyg_edge_index_with_attr(Z_joint, k_graph, device)
        X = torch.tensor(Z_joint, dtype=torch.float32, device=device)
        Y_full = torch.zeros(N_tr + N_va, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        class TConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = TransformerConv(in_dim, hidden_dim, heads=heads,
                                             edge_dim=1, dropout=dropout, concat=True)
                self.conv2 = TransformerConv(hidden_dim * heads, NUM_CLASSES,
                                             heads=1, edge_dim=1, dropout=dropout, concat=False)
                self.bn = nn.BatchNorm1d(hidden_dim * heads)

            def forward(self, x, ei, ea):
                h = torch.relu(self.bn(self.conv1(x, ei, ea)))
                return self.conv2(h, ei, ea)

        model = TConvNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X, edge_index, edge_attr)
            loss = _asl_loss(out[train_mask], Y_full[train_mask])
            loss.backward(); opt.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, edge_index, edge_attr)).cpu().numpy()
        return out[N_tr:].astype(np.float32)
    return fn


# ── D14: GIN — maximum expressivity ──────────────────────────────────────────
def make_d14_gin_fn(hidden_dim=256, k_graph=10, epochs=300, lr=1e-3, dropout=0.3):
    """D14: GINConv (Graph Isomorphism Network) — WL-test upper bound expressivity.

    h_v = MLP((1+ε)·h_v + Σ_{u∈N(v)} h_u)
    Most expressive GNN for distinguishing non-isomorphic graph structures.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch, torch.nn as nn
        from torch_geometric.nn import GINConv

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = Z_tr.shape[1]

        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)
        edge_index = _build_pyg_edge_index(Z_joint, k_graph, device)
        X = torch.tensor(Z_joint, dtype=torch.float32, device=device)
        Y_full = torch.zeros(N_tr + N_va, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        class GINNet(nn.Module):
            def __init__(self):
                super().__init__()
                mlp1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                     nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
                mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                     nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, NUM_CLASSES))
                self.conv1 = GINConv(mlp1, train_eps=True)
                self.conv2 = GINConv(mlp2, train_eps=True)

            def forward(self, x, ei):
                return self.conv2(torch.relu(self.conv1(x, ei)), ei)

        model = GINNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X, edge_index)
            loss = _asl_loss(out[train_mask], Y_full[train_mask])
            loss.backward(); opt.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, edge_index)).cpu().numpy()
        return out[N_tr:].astype(np.float32)
    return fn


# ── D15: Enriched node features + GAT ────────────────────────────────────────
def make_d15_enriched_gat_fn(pca_dim=64, heads=4, hidden_dim=256,
                               k_graph=10, epochs=300, lr=1e-3, dropout=0.3):
    """D15: GATConv with enriched node features: CL2N(1536) + PCA(64) + density(1).

    Hypothesis: giving GNN richer initial node representations (=1601-dim) allows
    attention to specialize more effectively than using raw 1536-dim alone.
    Density (1/core_dist) = local cluster tightness — a node in a dense cluster
    should have its label estimates propagated more confidently.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch, torch.nn as nn
        from torch_geometric.nn import GATConv

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)

        # Enrich features: CL2N + PCA + density
        X_tr_rich, X_va_rich = _enrich_node_features(Z_tr, Y_tr, Z_va, pca_dim=pca_dim)
        N_tr, N_va = len(X_tr_rich), len(X_va_rich)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = X_tr_rich.shape[1]

        # Build graph from CL2N (not enriched) for topology
        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)
        edge_index = _build_pyg_edge_index(Z_joint, k_graph, device)

        X_joint = np.concatenate([X_tr_rich, X_va_rich], axis=0)
        X = torch.tensor(X_joint, dtype=torch.float32, device=device)
        Y_full = torch.zeros(N_tr + N_va, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        class EnrichedGAT(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
                self.conv2 = GATConv(hidden_dim * heads, NUM_CLASSES, heads=1,
                                     concat=False, dropout=dropout)
                self.bn = nn.BatchNorm1d(hidden_dim * heads)

            def forward(self, x, ei):
                h = torch.relu(self.bn(self.conv1(x, ei)))
                return self.conv2(h, ei)

        model = EnrichedGAT().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X, edge_index)
            loss = _asl_loss(out[train_mask], Y_full[train_mask])
            loss.backward(); opt.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, edge_index)).cpu().numpy()
        return out[N_tr:].astype(np.float32)
    return fn


# ── D16: LPA → APPNP (LP-refined labels as extra node feature) ───────────────
def make_d16_lpa_appnp_fn(k_graph=10, lp_alpha=0.5, lp_iter=20, hidden_dim=512,
                            appnp_k=10, appnp_alpha=0.15, epochs=400, lr=1e-3):
    """D16: LabelPropagation (PyG) pre-softens labels → APPNP uses them as extra feature.

    Idea: LP gives a "soft prior" over labels for each support clip (denoise labels).
    We concatenate LP-soft-labels to Perch features as additional input to APPNP.
    Query clips get LP-soft-labels initialized to 0 (unknown), then propagated.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch, torch.nn as nn
        from torch_geometric.nn import APPNP, LabelPropagation

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]

        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)
        edge_index = _build_pyg_edge_index(Z_joint, k_graph, device)

        # ── Step 1: Label Propagation to get soft label priors ────────────────
        lp_model = LabelPropagation(num_layers=lp_iter, alpha=lp_alpha)
        # Y_joint: support = hard labels, query = 0 (unlabeled mask)
        Y_joint_np = np.zeros((N_tr + N_va, NUM_CLASSES), dtype=np.float32)
        Y_joint_np[:N_tr] = Y_tr.astype(np.float32)
        Y_joint_t = torch.tensor(Y_joint_np, device=device)
        mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        mask[:N_tr] = True

        with torch.no_grad():
            lp_out = lp_model(Y_joint_t, edge_index, mask=mask).cpu().numpy()  # (N, C)

        # ── Step 2: Concatenate LP soft labels to Perch features ──────────────
        X_joint_raw = np.concatenate([Z_joint, lp_out], axis=1).astype(np.float32)
        in_dim = X_joint_raw.shape[1]

        X = torch.tensor(X_joint_raw, dtype=torch.float32, device=device)
        Y_full = torch.zeros(N_tr + N_va, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        # ── Step 3: APPNP on enriched features ────────────────────────────────
        class LPA_APPNP(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(in_dim, hidden_dim)
                self.lin2 = nn.Linear(hidden_dim, NUM_CLASSES)
                self.bn   = nn.BatchNorm1d(hidden_dim)
                self.drop = nn.Dropout(0.4)
                self.prop = APPNP(K=appnp_k, alpha=appnp_alpha)

            def forward(self, x, ei):
                h = self.drop(torch.relu(self.bn(self.lin1(x))))
                return self.prop(self.lin2(h), ei)

        model = LPA_APPNP().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X, edge_index)
            loss = _asl_loss(out[train_mask], Y_full[train_mask])
            loss.backward(); opt.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, edge_index)).cpu().numpy()
        return out[N_tr:].astype(np.float32)
    return fn


# ── D17: Heterogeneous Graph — clip nodes + species nodes ────────────────────
def make_d17_hetero_fn(hidden_dim=256, heads=2, k_graph=10, epochs=300,
                        lr=1e-3, dropout=0.3):
    """D17: Heterogeneous bipartite graph with clip nodes + species prototype nodes.

    Graph structure:
      - clip nodes   (N_tr+N_va, 1536-dim Perch)  ← support + query
      - species nodes (234, 1536-dim) = per-class average of positive support clips
      Edges:
        clip→clip   : kNN cosine similarity
        clip→species: Y_tr labels (hard, support only); query uses model predictions
        species→clip: transpose of above

    Uses PyG HeteroConv with SAGEConv per edge type.
    Species nodes aggregate from their positive clips → query clips then
    aggregate from species nodes → effectively multi-label prototype retrieval
    with learned, graph-contextualized representations.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch, torch.nn as nn
        from torch_geometric.nn import SAGEConv, HeteroConv

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = Z_tr.shape[1]

        # ── Species prototype nodes: avg of positive support clips ────────────
        species_emb = np.zeros((NUM_CLASSES, in_dim), dtype=np.float32)
        for c in range(NUM_CLASSES):
            pos = Y_tr[:, c] > 0
            if pos.any():
                proto = Z_tr[pos].mean(0)
                norm = np.linalg.norm(proto)
                species_emb[c] = proto / (norm + 1e-8)
            # else: zero vector = unknown species

        # ── Build hetero data on CPU, move to device ──────────────────────────
        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)
        N_clips = N_tr + N_va

        # clip→clip kNN
        sim_cc = Z_joint @ Z_joint.T
        np.fill_diagonal(sim_cc, -np.inf)
        rows_cc, cols_cc = [], []
        for i in range(N_clips):
            top_k = np.argpartition(sim_cc[i], -k_graph)[-k_graph:]
            rows_cc.extend([i] * k_graph); cols_cc.extend(top_k.tolist())
        ei_cc = torch.tensor([rows_cc, cols_cc], dtype=torch.long, device=device)

        # clip→species: support clips only (from Y_tr)
        rows_cs, cols_cs = [], []
        for i in range(N_tr):
            for c in np.where(Y_tr[i] > 0)[0]:
                rows_cs.append(i); cols_cs.append(int(c))
        if not rows_cs:
            rows_cs, cols_cs = [0], [0]   # degenerate guard
        ei_cs = torch.tensor([rows_cs, cols_cs], dtype=torch.long, device=device)
        ei_sc = ei_cs.flip(0)  # species→clip

        # Node features
        X_clip    = torch.tensor(Z_joint,   dtype=torch.float32, device=device)
        X_species = torch.tensor(species_emb, dtype=torch.float32, device=device)

        # Labels
        Y_clip = torch.zeros(N_clips, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_clip[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_clips, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        class HeteroGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = HeteroConv({
                    ('clip', 'cc', 'clip'):    SAGEConv(in_dim, hidden_dim),
                    ('clip', 'cs', 'species'): SAGEConv((in_dim, in_dim), hidden_dim),
                    ('species', 'sc', 'clip'): SAGEConv((in_dim, in_dim), hidden_dim),
                }, aggr='sum')
                self.conv2 = HeteroConv({
                    ('clip', 'cc', 'clip'):    SAGEConv(hidden_dim, NUM_CLASSES),
                    ('species', 'sc', 'clip'): SAGEConv((hidden_dim, hidden_dim), NUM_CLASSES),
                }, aggr='sum')
                self.bn = nn.BatchNorm1d(hidden_dim)

            def forward(self, x_c, x_s, ei_cc, ei_cs, ei_sc):
                x_dict = {'clip': x_c, 'species': x_s}
                ei_dict = {
                    ('clip','cc','clip'):    ei_cc,
                    ('clip','cs','species'): ei_cs,
                    ('species','sc','clip'): ei_sc,
                }
                out1 = self.conv1(x_dict, ei_dict)
                h_clip = torch.relu(self.bn(out1['clip']))
                h_spec = torch.relu(out1['species'])

                x_dict2 = {'clip': h_clip, 'species': h_spec}
                ei_dict2 = {
                    ('clip','cc','clip'):    ei_cc,
                    ('species','sc','clip'): ei_sc,
                }
                out2 = self.conv2(x_dict2, ei_dict2)
                return out2['clip']   # (N_clips, NUM_CLASSES)

        model = HeteroGNN().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X_clip, X_species, ei_cc, ei_cs, ei_sc)
            loss = _asl_loss(out[train_mask], Y_clip[train_mask])
            loss.backward(); opt.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X_clip, X_species, ei_cc, ei_cs, ei_sc)).cpu().numpy()
        return out[N_tr:].astype(np.float32)
    return fn


# ── D18: APPNP + enriched node features ──────────────────────────────────────
def make_d18_appnp_enriched_fn(pca_dim=64, hidden_dim=512, k_hops=10, alpha=0.15,
                                 k_graph=10, epochs=400, lr=1e-3, dropout=0.4):
    """D18: APPNP with enriched node features (CL2N + PCA64 + density).

    Combines the stability of APPNP (D12) with enriched node features (D15).
    PCA64 adds compressed global structure; density adds local cluster tightness.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch, torch.nn as nn
        from torch_geometric.nn import APPNP

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)

        X_tr_r, X_va_r = _enrich_node_features(Z_tr, Y_tr, Z_va, pca_dim=pca_dim)
        N_tr, N_va = len(X_tr_r), len(X_va_r)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = X_tr_r.shape[1]

        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)
        edge_index = _build_pyg_edge_index(Z_joint, k_graph, device)

        X_joint = np.concatenate([X_tr_r, X_va_r], axis=0)
        X = torch.tensor(X_joint, dtype=torch.float32, device=device)
        Y_full = torch.zeros(N_tr + N_va, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        class EnrichedAPPNP(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(in_dim, hidden_dim)
                self.lin2 = nn.Linear(hidden_dim, NUM_CLASSES)
                self.bn   = nn.BatchNorm1d(hidden_dim)
                self.drop = nn.Dropout(dropout)
                self.prop = APPNP(K=k_hops, alpha=alpha)

            def forward(self, x, ei):
                h = self.drop(torch.relu(self.bn(self.lin1(x))))
                return self.prop(self.lin2(h), ei)

        model = EnrichedAPPNP().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X, edge_index)
            loss = _asl_loss(out[train_mask], Y_full[train_mask])
            loss.backward(); opt.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, edge_index)).cpu().numpy()
        return out[N_tr:].astype(np.float32)
    return fn


# ── D19: PyG LabelPropagation → D8 cross-attention ───────────────────────────
def make_d19_lpa_d8_fn(k_graph=10, lp_alpha=0.5, lp_iter=20, temperature=10.0, k_nn=15):
    """D19: PyG LabelPropagation to refine support labels → D8 cross-attention retrieval.

    LP denoises support labels (multi-label noise is common in soundscape data).
    D8 cross-attention (best individual probe, 0.696) then retrieves using refined labels.
    This is a parameter-free combination of two proven methods.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch
        from torch_geometric.nn import LabelPropagation

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]

        # Build support-only graph for LP
        edge_index = _build_pyg_edge_index(Z_tr, k_graph, device)
        Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        mask_all = torch.ones(N_tr, dtype=torch.bool, device=device)

        lp = LabelPropagation(num_layers=lp_iter, alpha=lp_alpha)
        with torch.no_grad():
            Y_refined = lp(Y_tr_t, edge_index, mask=mask_all).cpu().numpy()

        # D8 cross-attention with refined labels
        sim = Z_va @ Z_tr.T   # (N_va, N_tr)
        preds = np.zeros((N_va, NUM_CLASSES), dtype=np.float32)
        for i in range(N_va):
            top_idx = np.argpartition(sim[i], -k_nn)[-k_nn:]
            sim_top = sim[i, top_idx]
            sim_top -= sim_top.max()
            attn = np.exp(temperature * sim_top)
            attn /= (attn.sum() + 1e-8)
            preds[i] = (Y_refined[top_idx] * attn[:, None]).sum(0)
        return preds
    return fn


# ── D20: APPNP + cosine-edge warmstart ───────────────────────────────────────
def make_d20_appnp_edge_fn(hidden_dim=512, k_hops=15, alpha=0.1, k_graph=15,
                             epochs=400, lr=1e-3, dropout=0.35):
    """D20: APPNP with edge-weighted propagation (cosine-sim as diffusion weights).

    Standard APPNP uses unweighted graph Ã. Here we use cosine-sim as edge weights
    in the personalized PageRank diffusion step: stronger neighbors contribute more.
    Implemented via manual weighted propagation replacing APPNP's fixed Ã.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch, torch.nn as nn

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = Z_tr.shape[1]
        N = N_tr + N_va

        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)

        # Build cosine-weighted adjacency (sparse → dense for propagation)
        sim = Z_joint @ Z_joint.T
        np.fill_diagonal(sim, -np.inf)
        W = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            top_k = np.argpartition(sim[i], -k_graph)[-k_graph:]
            W[i, top_k] = np.clip(sim[i, top_k], 0, 1)
        W = np.maximum(W, W.T)
        W += np.eye(N, dtype=np.float32)
        D_inv = np.diag(1.0 / (W.sum(1) + 1e-8))
        S = torch.tensor(D_inv @ W, dtype=torch.float32, device=device)  # row-normalized

        X = torch.tensor(Z_joint, dtype=torch.float32, device=device)
        Y_full = torch.zeros(N, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        class WeightedAPPNP(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(in_dim, hidden_dim)
                self.lin2 = nn.Linear(hidden_dim, NUM_CLASSES)
                self.bn   = nn.BatchNorm1d(hidden_dim)
                self.drop = nn.Dropout(dropout)

            def forward(self, x, S, alpha, K):
                h = self.drop(torch.relu(self.bn(self.lin1(x))))
                Z0 = self.lin2(h)   # initial predictions
                Z  = Z0
                for _ in range(K):
                    Z = (1 - alpha) * (S @ Z) + alpha * Z0
                return Z

        model = WeightedAPPNP().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X, S, alpha, k_hops)
            loss = _asl_loss(out[train_mask], Y_full[train_mask])
            loss.backward(); opt.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, S, alpha, k_hops)).cpu().numpy()
        return out[N_tr:].astype(np.float32)
    return fn


# ── D21: Nearest-Positive Retrieval (GraFPrint max-aggregation inspired) ─────
def make_d21_max_sim_fn():
    """D21: Nearest-Positive Retrieval — for each class, score = max cosine sim
    to any positive support clip of that class.

    Inspired by GraFPrint's max-relative aggregation: instead of averaging
    neighbor features (mean aggr), take the MAX. In retrieval terms, this means
    "how similar is the query to the BEST matching positive example of each class?"
    Particularly effective for rare species with only 1-2 support clips.

    Paper: GraFPrint (ICASSP 2025) — max-relative conv captures extreme deviations.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        NUM_CLASSES = Y_tr.shape[1]

        sim = Z_va @ Z_tr.T   # (N_va, N_tr)
        preds = np.zeros((len(Z_va), NUM_CLASSES), dtype=np.float32)

        for c in range(NUM_CLASSES):
            pos_idx = np.where(Y_tr[:, c] > 0)[0]
            if len(pos_idx) == 0:
                continue
            # Max similarity to any positive support clip
            preds[:, c] = sim[:, pos_idx].max(axis=1).clip(0, 1)

        return preds
    return fn


def make_d21b_softtop3_fn(top_k=3, temperature=8.0):
    """D21b: Soft-max over top-K positive clips per class (blend of D21 and D8).

    Rather than hard max (D21) or all-neighbor mean (D8), take softmax-weighted
    average over the top-3 most similar POSITIVE clips for each class.
    This is robust to a single noisy positive example corrupting the score.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        NUM_CLASSES = Y_tr.shape[1]

        sim = Z_va @ Z_tr.T   # (N_va, N_tr)
        preds = np.zeros((len(Z_va), NUM_CLASSES), dtype=np.float32)

        for c in range(NUM_CLASSES):
            pos_idx = np.where(Y_tr[:, c] > 0)[0]
            if len(pos_idx) == 0:
                continue
            sim_pos = sim[:, pos_idx]   # (N_va, n_pos)

            if sim_pos.shape[1] <= top_k:
                # Fewer positives than top_k: soft-max over all
                sim_top = sim_pos
            else:
                # Select top-K by sim for each query
                top_idx = np.argpartition(sim_pos, -top_k, axis=1)[:, -top_k:]
                sim_top = np.take_along_axis(sim_pos, top_idx, axis=1)

            sim_top2 = sim_top - sim_top.max(axis=1, keepdims=True)  # stability
            attn = np.exp(temperature * sim_top2)
            attn /= attn.sum(axis=1, keepdims=True) + 1e-8
            # preds[:,c] = weighted sum of 1 (positive label) × attn = sum(attn)
            # since all selected are positives, weighted score = Σ attn_j * 1
            preds[:, c] = attn.sum(axis=1).clip(0, 1)   # normalized [0,1] by construction

        return preds
    return fn


# ── D22: ATGNN-inspired Label Co-occurrence Graph Post-processing ─────────────
def make_d22_label_cooccur_fn(base_probe_fn, alpha=0.3, n_iter=3):
    """D22: ATGNN Label-Label Graph (LLG) post-processing on D8 predictions.

    From ATGNN (IEEE Signal Processing Letters 2024): the Label-Label Graph (LLG)
    alone adds +0.4 mAP on FSD50K. Build a species co-occurrence adjacency matrix
    from the support labels, then propagate predictions through it.

    Species that co-occur in the same soundscape clips should share prediction signal:
    if query looks like species A, and A always co-occurs with B in training data,
    boost B's prediction too.

    Process:
      1. Build C[a,b] = count clips where both species a and b are positive
      2. Normalize row-wise → S_label (234×234 row-stochastic)
      3. Get initial predictions P from base_probe_fn (e.g. D8)
      4. Refine: P_t = (1-α)·P + α·P·S_label  (propagate in label space)
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        # ── Step 1: Get base predictions (D8) ────────────────────────────────
        P = base_probe_fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va)

        # ── Step 2: Build species co-occurrence matrix from support ───────────
        Y_float = Y_tr.astype(np.float32)
        C = Y_float.T @ Y_float   # (C, C)  raw co-occurrence counts
        np.fill_diagonal(C, 0)    # remove self-loops

        row_sum = C.sum(1, keepdims=True)
        S_label = (C / (row_sum + 1e-8)).astype(np.float32)  # row-normalized

        # ── Step 3: Propagate predictions through label graph ─────────────────
        P0 = P.copy()
        for _ in range(n_iter):
            P = (1.0 - alpha) * P0 + alpha * (P @ S_label)

        return np.clip(P, 0, 1)
    return fn


# ── D23: Ensemble D8 + D1 (best two individual probes) ───────────────────────
def make_d23_d8d1_ensemble_fn(w_d8=0.5, w_d1=0.5):
    """D23: Simple ensemble of D8_cross_attn (0.696) and D1_mreach (0.695).

    Both methods are complementary:
    - D8 uses temperature-scaled softmax attention (sharp, focuses on best K neighbors)
    - D1 uses mutual reachability distance (density-aware, noise-robust)
    Averaging should smooth errors of each.
    """
    d8_fn = make_d8_cross_attn_fn()
    d1_fn = make_d1_mreach_knn_fn()

    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        p_d8 = d8_fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va)
        p_d1 = d1_fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va)
        return w_d8 * p_d8 + w_d1 * p_d1
    return fn


# ── D24: GraFPrint-style Max-Relative kNN (semi-parametric, no training) ──────
def make_d24_grafprint_knn_fn(k_graph=15, k_nn=15, temperature=10.0):
    """D24: GraFPrint max-relative contrast signature for kNN retrieval.

    GraFPrint uses h_i = max_{j∈N(i)} relu(x_j - x_i) as a "local contrast"
    feature. Here we apply this to compute a CONTRAST SIGNATURE for each clip:
      contrast_i = max_{j∈topK_train} relu(Z_tr[j] - Z_tr[i])   (D,)

    Then at inference: the query's contrast vs each support clip's contrast
    gives a richer similarity measure than raw cosine alone.

    Final retrieval: cosine(contrast_query, contrast_support) with temperature softmax.
    No learnable parameters — purely geometric.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr = len(Z_tr)

        # ── Compute contrast signatures for support clips ─────────────────────
        sim_tr = Z_tr @ Z_tr.T
        np.fill_diagonal(sim_tr, -np.inf)
        contrast_tr = np.zeros_like(Z_tr)
        for i in range(N_tr):
            top_k = np.argpartition(sim_tr[i], -k_graph)[-k_graph:]
            diffs = np.maximum(Z_tr[top_k] - Z_tr[i], 0)   # relu(x_j - x_i)
            contrast_tr[i] = diffs.max(axis=0)               # max over neighbors

        # L2-normalize contrast signatures
        norm = np.linalg.norm(contrast_tr, axis=1, keepdims=True)
        contrast_tr_n = contrast_tr / (norm + 1e-8)

        # ── Compute contrast signatures for query clips ───────────────────────
        sim_qtr = Z_va @ Z_tr.T   # (N_va, N_tr) — use train as context graph
        contrast_va = np.zeros_like(Z_va)
        for i in range(len(Z_va)):
            top_k = np.argpartition(sim_qtr[i], -k_graph)[-k_graph:]
            diffs = np.maximum(Z_tr[top_k] - Z_va[i], 0)
            contrast_va[i] = diffs.max(axis=0)

        norm_va = np.linalg.norm(contrast_va, axis=1, keepdims=True)
        contrast_va_n = contrast_va / (norm_va + 1e-8)

        # ── Temperature-scaled softmax retrieval in contrast space ────────────
        sim_contrast = contrast_va_n @ contrast_tr_n.T   # (N_va, N_tr)
        preds = np.zeros((len(Z_va), Y_tr.shape[1]), dtype=np.float32)
        for i in range(len(Z_va)):
            top_idx = np.argpartition(sim_contrast[i], -k_nn)[-k_nn:]
            sim_top = sim_contrast[i, top_idx]
            sim_top -= sim_top.max()
            attn = np.exp(temperature * sim_top)
            attn /= (attn.sum() + 1e-8)
            preds[i] = (Y_tr[top_idx].astype(np.float32) * attn[:, None]).sum(0)

        return preds
    return fn


# ── D25: GATv2Conv — fixes static attention problem of GATv1 ─────────────────
def make_d25_gatv2_fn(hidden_dim=256, heads=4, k_graph=10, epochs=200,
                       lr=5e-4, dropout=0.5):
    """D25: GATv2Conv (Brody et al. 2022) — dynamic attention vs GATv1's static.

    GATv1 computes attention as: e_ij = a^T · [W·x_i || W·x_j]
    → This is STATIC: attention score doesn't depend on the query context.
    GATv2 fixes this: e_ij = a^T · LeakyReLU(W·[x_i || x_j])
    → Dynamic: full expressive power, captures asymmetric relations.

    Paper: 'How Attentive are Graph Attention Networks?' (ICLR 2022).
    Expected to outperform D10_gat (0.570) significantly.
    Sensors 2024 shows GAT dominates on ecoacoustic; GATv2 should be better still.
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch, torch.nn as nn
        from torch_geometric.nn import GATv2Conv

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        N_tr, N_va = len(Z_tr), len(Z_va)
        NUM_CLASSES = Y_tr.shape[1]
        in_dim = Z_tr.shape[1]

        Z_joint = np.concatenate([Z_tr, Z_va], axis=0)
        edge_index = _build_pyg_edge_index(Z_joint, k_graph, device)
        X = torch.tensor(Z_joint, dtype=torch.float32, device=device)
        Y_full = torch.zeros(N_tr + N_va, NUM_CLASSES, dtype=torch.float32, device=device)
        Y_full[:N_tr] = torch.tensor(Y_tr, dtype=torch.float32, device=device)
        train_mask = torch.zeros(N_tr + N_va, dtype=torch.bool, device=device)
        train_mask[:N_tr] = True

        class GATv2Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads,
                                       dropout=dropout, concat=True)
                self.conv2 = GATv2Conv(hidden_dim * heads, NUM_CLASSES, heads=1,
                                       dropout=dropout, concat=False)
                self.bn = nn.BatchNorm1d(hidden_dim * heads)

            def forward(self, x, ei):
                h = torch.relu(self.bn(self.conv1(x, ei)))
                return self.conv2(h, ei)

        model = GATv2Net().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X, edge_index)
            loss = _asl_loss(out[train_mask], Y_full[train_mask])
            loss.backward(); opt.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(X, edge_index)).cpu().numpy()
        return out[N_tr:].astype(np.float32)
    return fn


# ── D26: Few-Shot Episode GNN — attentional support selection (Interspeech 2019)
def make_d26_episode_attn_fn(k_nn=15, temperature=10.0, entropy_weight=True):
    """D26: Few-Shot Attentional GNN (Zhang et al. Interspeech 2019).

    Original paper builds a episode graph (support+query) with pairwise similarity
    edges and uses attentional selection over support examples per query.
    Key insight: not all support examples are equally useful for a query — entropy
    of attention weights indicates confidence.

    Simplified parameter-free version:
    1. For each query, compute attention over ALL support clips (full D8 style)
    2. Weight by inverse entropy of per-class attention → high-entropy = uncertain
    3. Entropy-weighted class scores: classes where attention is concentrated (low
       entropy) get higher weight than diffuse attention
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        NUM_CLASSES = Y_tr.shape[1]

        sim = Z_va @ Z_tr.T   # (N_va, N_tr)
        preds = np.zeros((len(Z_va), NUM_CLASSES), dtype=np.float32)

        for i in range(len(Z_va)):
            top_idx = np.argpartition(sim[i], -k_nn)[-k_nn:]
            sim_top = sim[i, top_idx]
            sim_top2 = sim_top - sim_top.max()
            attn = np.exp(temperature * sim_top2)
            attn /= (attn.sum() + 1e-8)   # (k_nn,)

            # Base prediction (same as D8)
            base_pred = (Y_tr[top_idx].astype(np.float32) * attn[:, None]).sum(0)

            if entropy_weight:
                # Per-class entropy: for class c, attn conditioned on Y_tr[:,c]>0
                # Low entropy → attention concentrated on few clips → confident
                conf = np.ones(NUM_CLASSES, dtype=np.float32)
                for c in range(NUM_CLASSES):
                    pos_in_top = Y_tr[top_idx, c] > 0
                    if pos_in_top.any():
                        a_c = attn[pos_in_top]
                        a_c = a_c / (a_c.sum() + 1e-8)
                        entropy = -(a_c * np.log(a_c + 1e-8)).sum()
                        conf[c] = np.exp(-entropy)   # high entropy → low conf
                preds[i] = base_pred * conf
            else:
                preds[i] = base_pred

        return np.clip(preds, 0, 1)
    return fn


# ── D27-D30: Deep GNN + JK + Regularization ──────────────────────────────────

def _make_deep_gnn_fn(arch='sage', jk_mode='lstm', hidden_dim=512, num_layers=4,
                      k_graph=15, epochs=600, lr=5e-4, dropout=0.5,
                      weight_decay=1e-3, drop_edge=0.3, use_pairnorm=False,
                      patience=50):
    """Generic deep GNN factory: SAGE/GCN/GATv2 × N layers + JK + DropEdge.

    Key improvements over D9-D11:
    - Depth: 4 layers (D11 had 2) → larger receptive field
    - JumpingKnowledge: adaptively blends per-layer representations (Xu ICML 2018)
      - 'lstm': LSTM over layers → learns which hop distance matters per node
      - 'max' : element-wise max → takes best signal across any layer
      - 'cat' : concat all layers → most expressive, most params
    - DropEdge (p=0.3): randomly removes 30% of edges each fwd pass
      → stronger graph structure regularization than dropout on nodes alone
    - PairNorm: centers+normalizes node repr after each layer
      → prevents over-smoothing in deeper networks
    - hidden_dim=512 (2x D11's 256)
    - weight_decay=1e-3 (10x D11's 1e-4)
    - dropout=0.5 (vs 0.3)
    - k_graph=15 (larger neighborhood)
    - Early stopping (patience=50 on training loss)
    """
    def fn(emb_tr, Y_tr, emb_va, meta_tr, meta_va):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv, GCNConv, GATv2Conv
        from torch_geometric.nn import JumpingKnowledge, PairNorm
        from torch_geometric.utils import dropout_edge

        NUM_CLASSES = Y_tr.shape[1]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        Z_tr, mean_vec = cl2n(emb_tr)
        Z_va = l2_normalize(emb_va - mean_vec)
        Z_all = np.concatenate([Z_tr, Z_va], axis=0)
        n_tr = len(Z_tr)

        edge_index_full = _build_pyg_edge_index(Z_all, k=k_graph, device=device)
        X = torch.tensor(Z_all, dtype=torch.float32, device=device)
        in_dim = X.shape[1]

        Y_np = np.zeros((len(Z_all), NUM_CLASSES), dtype=np.float32)
        Y_np[:n_tr] = Y_tr.astype(np.float32)
        Y_t = torch.tensor(Y_np, device=device)
        mask = torch.zeros(len(Z_all), dtype=torch.bool, device=device)
        mask[:n_tr] = True

        class DeepGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()
                self.bns   = nn.ModuleList()
                self.pns   = nn.ModuleList() if use_pairnorm else None

                for i in range(num_layers):
                    in_c = in_dim if i == 0 else hidden_dim
                    if arch == 'sage':
                        self.convs.append(SAGEConv(in_c, hidden_dim))
                    elif arch == 'gcn':
                        self.convs.append(GCNConv(in_c, hidden_dim))
                    elif arch == 'gatv2':
                        h = 4 if i < num_layers - 1 else 1
                        out_c = hidden_dim // h
                        self.convs.append(GATv2Conv(in_c, out_c, heads=h,
                                                     dropout=dropout, concat=True))
                        hidden_dim_effective = hidden_dim  # heads * out_c = hidden_dim
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
                    if use_pairnorm:
                        self.pns.append(PairNorm())

                # JK aggregation
                jk_channels = hidden_dim
                self.jk = JumpingKnowledge(jk_mode, jk_channels,
                                            num_layers=num_layers)
                jk_out = jk_channels if jk_mode in ('lstm', 'max') else jk_channels * num_layers
                self.drop = nn.Dropout(dropout)
                self.classifier = nn.Linear(jk_out, NUM_CLASSES)

            def forward(self, x, edge_index):
                xs = []
                for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                    # DropEdge: randomly drop edges each forward pass
                    ei, _ = dropout_edge(edge_index, p=drop_edge,
                                         training=self.training)
                    x = conv(x, ei)
                    x = bn(x)
                    if use_pairnorm:
                        x = self.pns[i](x)
                    x = F.relu(x)
                    x = self.drop(x)
                    xs.append(x)
                x = self.jk(xs)
                return self.classifier(x)

        model = DeepGNN().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_loss = float('inf')
        patience_cnt = 0
        best_state = None

        for ep in range(epochs):
            model.train()
            logits = model(X, edge_index_full)[mask]
            loss = _asl_loss(logits, Y_t[mask])
            opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()

            l = loss.item()
            if l < best_loss - 1e-4:
                best_loss = l
                patience_cnt = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            logits_all = model(X, edge_index_full)
        preds_va = torch.sigmoid(logits_all[n_tr:]).cpu().numpy()
        return np.clip(preds_va, 0, 1)
    return fn


def make_d27_sage_jk_lstm_fn():
    """D27: GraphSAGE × 4 + JK-LSTM + DropEdge(0.3) + dropout=0.5 + wd=1e-3 + ES.

    JK-LSTM learns a per-node weighting of 4 hop-distances via LSTM hidden state.
    Key advantage: each node can choose its own effective receptive field.
    """
    return _make_deep_gnn_fn(arch='sage', jk_mode='lstm', hidden_dim=512,
                              num_layers=4, k_graph=15, epochs=600, lr=5e-4,
                              dropout=0.5, weight_decay=1e-3, drop_edge=0.3,
                              use_pairnorm=False, patience=50)


def make_d28_sage_jk_max_fn():
    """D28: GraphSAGE × 4 + JK-max + DropEdge(0.3) + dropout=0.5 + wd=1e-3 + ES.

    JK-max takes element-wise maximum across all 4 layer representations.
    Parameter-free aggregation (unlike LSTM) — avoids adding GNN-specific params.
    """
    return _make_deep_gnn_fn(arch='sage', jk_mode='max', hidden_dim=512,
                              num_layers=4, k_graph=15, epochs=600, lr=5e-4,
                              dropout=0.5, weight_decay=1e-3, drop_edge=0.3,
                              use_pairnorm=False, patience=50)


def make_d29_gatv2_jk_pairnorm_fn():
    """D29: GATv2Conv × 3 + JK-max + PairNorm + DropEdge(0.3) + wd=1e-3 + ES.

    GATv2 fixes GATv1's static attention bug (attention recomputed per query node).
    PairNorm prevents over-smoothing in deeper stacks by centering+normalizing.
    JK-max captures different attention granularities per layer.
    """
    return _make_deep_gnn_fn(arch='gatv2', jk_mode='max', hidden_dim=512,
                              num_layers=3, k_graph=15, epochs=600, lr=5e-4,
                              dropout=0.5, weight_decay=1e-3, drop_edge=0.3,
                              use_pairnorm=True, patience=50)


def make_d30_gcn_jk_cat_pairnorm_fn():
    """D30: GCNConv × 4 + JK-cat + PairNorm + dropout=0.6 + wd=5e-3 + ES.

    JK-cat concatenates all 4 layer representations → most expressive JK variant.
    Strongest regularization (dropout=0.6, wd=5e-3) — GCN is simpler so needs more.
    PairNorm after each layer to prevent rank collapse.
    """
    return _make_deep_gnn_fn(arch='gcn', jk_mode='cat', hidden_dim=256,
                              num_layers=4, k_graph=15, epochs=600, lr=5e-4,
                              dropout=0.6, weight_decay=5e-3, drop_edge=0.3,
                              use_pairnorm=True, patience=50)


# ══════════════════════════════════════════════════════════════════════════════
# OOF runner with index-aware wrapper for SED fusion
# ══════════════════════════════════════════════════════════════════════════════

def oof_auc_with_indices(meta_full, emb_full, Y_FULL, probe_fn_with_idx, n_splits=5):
    """Variant for experiments that need fold indices (e.g. SED fusion)."""
    gkf    = GroupKFold(n_splits=n_splits)
    groups = meta_full['filename'].to_numpy()
    oof_preds = np.zeros_like(Y_FULL, dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(
            tqdm(list(gkf.split(emb_full, groups=groups)), desc='OOF folds')):
        tr_idx = np.sort(tr_idx); va_idx = np.sort(va_idx)
        meta_tr = meta_full.iloc[tr_idx].reset_index(drop=True)
        meta_va = meta_full.iloc[va_idx].reset_index(drop=True)
        oof_preds[va_idx] = probe_fn_with_idx(
            emb_full[tr_idx], Y_FULL[tr_idx],
            emb_full[va_idx], meta_tr, meta_va,
            tr_idx, va_idx
        )
    keep = Y_FULL.sum(axis=0) > 0
    auc  = roc_auc_score(Y_FULL[:, keep], oof_preds[:, keep], average='macro')
    return auc, oof_preds


# ══════════════════════════════════════════════════════════════════════════════
# Excel writer
# ══════════════════════════════════════════════════════════════════════════════

def write_result(name, description, oof_auc_val, runtime_min, notes=''):
    row = {
        'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M'),
        'experiment':  name,
        'description': description,
        'oof_auc':     round(oof_auc_val, 5),
        'runtime_min': round(runtime_min, 1),
        'notes':       notes,
    }
    EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if EXCEL_PATH.exists():
        try:
            df = pd.read_excel(EXCEL_PATH, sheet_name='probe_results')
        except Exception:
            df = pd.DataFrame(columns=row.keys())
    else:
        df = pd.DataFrame(columns=row.keys())
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values('oof_auc', ascending=False).reset_index(drop=True)

    if EXCEL_PATH.exists():
        with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl',
                            mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='probe_results', index=False)
    else:
        with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name='probe_results', index=False)
    print(f'  → Excel updated: {name} OOF_AUC={oof_auc_val:.5f}  ({runtime_min:.1f} min)')
    return df


def print_leaderboard(df):
    print('\n' + '='*62)
    print('  PROBE EXPERIMENT LEADERBOARD')
    print('='*62)
    print(f"  {'Rank':<5} {'Experiment':<25} {'OOF AUC':<12} {'Runtime'}")
    print('-'*62)
    for i, row in df.iterrows():
        print(f"  {i+1:<5} {row['experiment']:<25} {row['oof_auc']:.5f}      {row['runtime_min']:.1f} min")
    print('='*62 + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_REGISTRY = {
    'baseline':   ('PCA64 + LogReg (LP++ init) — current pipeline',
                   make_baseline_fn, False),
    'P1':         ('ABT(10) + CL2N + kNN-15 — remove rogue Perch dims',
                   make_p1_abt_cl2n_knn_fn, False),
    'P2':         ('PCA64+LogReg + LaplacianShot post-proc per file',
                   make_p2_laplacianshot_fn, False),
    'A1_fecam':   ('FeCAM: Mahalanobis NCM w/ Ledoit-Wolf on PCA64',
                   make_a1_fecam_fn, False),
    'A2_cl2n':    ('CL2N + kNN-15 on raw 1536-dim (no ABT)',
                   make_a2_cl2n_knn_fn, False),
    'A3_abt':     ('ABT(10) + PCA64 + LogReg (rogue-dim removal + standard probe)',
                   make_a3_abt_pca_logreg_fn, False),
    'B1_umap':    ('UMAP(64) + CL2N + kNN-15',
                   make_b1_umap_fn, False),
    'C1_mlp':     ('MLP adapter 1536→512→128 + kNN-15 (trainable)',
                   make_c1_mlp_fn, False),
    'P3':         ('Perch(1536)+SED-B0(1280) concat → PCA128 → CL2N → kNN-15',
                   lambda: None, True),
    'B2':         ('Same as P3 (alias)',
                   lambda: None, True),
    'D1_mreach':  ('Mutual Reachability CL2N+kNN-15 (HDBSCAN distance, noise-robust)',
                   make_d1_mreach_knn_fn, False),
    'D2_mst':     ('MST label propagation via mutual reachability (density-connected)',
                   make_d2_mst_knn_fn, False),
    'D3_hdbscan': ('HDBSCAN sub-prototype NCM (per-class acoustic sub-clusters)',
                   make_d3_hdbscan_proto_fn, False),
    'D5_lp':      ('LP on support k-NN graph → refined labels → D1 mutual reachability',
                   make_d5_lp_fn, False),
    'D6_transductive': ('Transductive LP: per-file joint support+query graph',
                   make_d6_transductive_fn, False),
    'D7_denseprot': ('Density-weighted prototype NCM (1/core_dist weights)',
                   make_d7_denseprot_fn, False),
    'D8_cross_attn': ('Temperature-scaled softmax kNN (T=10, top-15)',
                   make_d8_cross_attn_fn, False),
    'D9_gcn':     ('Transductive PyG GCNConv joint graph (ASL loss, GPU)',
                   make_d9_gcn_fn, False),
    'D10_gat':    ('Transductive PyG GATConv 4-head joint graph (ASL loss, GPU)',
                   make_d10_gat_fn, False),
    'D11_sage':   ('Transductive PyG GraphSAGE joint graph (ASL loss, GPU)',
                   make_d11_sage_fn, False),
    # ── D12-D20: Advanced PyG + Feature Engineering ──────────────────────────
    'D12_appnp':  ('APPNP K=10 α=0.15 — MLP + PageRank propagation (no oversmoothing)',
                   make_d12_appnp_fn, False),
    'D13_tconv':  ('TransformerConv 4-head + cosine-sim edge_attr (edge-conditioned attention)',
                   make_d13_transformer_conv_fn, False),
    'D14_gin':    ('GINConv (WL-test expressivity, ε trainable, 2-layer MLP aggr)',
                   make_d14_gin_fn, False),
    'D15_enrich_gat': ('GATConv + enriched nodes: CL2N(1536)+PCA(64)+density(1)',
                   make_d15_enriched_gat_fn, False),
    'D16_lpa_appnp': ('PyG LabelPropagation → LP-soft-labels concat to Perch → APPNP',
                   make_d16_lpa_appnp_fn, False),
    'D17_hetero': ('HeteroConv: clip+species nodes, clip→clip+clip↔species edges',
                   make_d17_hetero_fn, False),
    'D18_appnp_enrich': ('APPNP + enriched nodes: CL2N(1536)+PCA(64)+density(1)',
                   make_d18_appnp_enriched_fn, False),
    'D19_lpa_d8': ('PyG LabelPropagation (support) → refined labels → D8 cross-attn',
                   make_d19_lpa_d8_fn, False),
    'D20_appnp_edge': ('Weighted APPNP: cosine-sim edge weights in PageRank diffusion',
                   make_d20_appnp_edge_fn, False),
    # ── D21-D26: Literature-inspired experiments ──────────────────────────────
    'D21_max_sim':  ('GraFPrint-inspired: max cosine-sim to any positive support clip per class',
                   make_d21_max_sim_fn, False),
    'D21b_softtop3': ('Soft-max over top-3 positive clips per class (T=8, blend D21+D8)',
                   make_d21b_softtop3_fn, False),
    'D22_label_cooccur': ('ATGNN LLG: species co-occurrence graph post-processing on D8',
                   lambda: make_d22_label_cooccur_fn(make_d8_cross_attn_fn()), False),
    'D22b_label_cooccur_d1': ('ATGNN LLG post-processing on D1_mreach',
                   lambda: make_d22_label_cooccur_fn(make_d1_mreach_knn_fn()), False),
    'D23_d8d1':     ('Ensemble: D8_cross_attn (0.696) + D1_mreach (0.695), w=0.5/0.5',
                   make_d23_d8d1_ensemble_fn, False),
    'D24_grafprint': ('GraFPrint max-relative contrast signature → temperature kNN (no training)',
                   make_d24_grafprint_knn_fn, False),
    'D25_gatv2':    ('GATv2Conv 4-head transductive (dynamic attention, fixes GATv1 static)',
                   make_d25_gatv2_fn, False),
    'D26_episode':  ('Few-Shot Episode Attn GNN: entropy-weighted class confidence (Interspeech 2019)',
                   make_d26_episode_attn_fn, False),
    # ── D27-D30: Deep GNN + JK + DropEdge + PairNorm + early stopping ─────────
    'D27_sage_jk_lstm': ('SAGE×4 + JK-LSTM + DropEdge(0.3) + dropout=0.5 + wd=1e-3 + ES',
                   make_d27_sage_jk_lstm_fn, False),
    'D28_sage_jk_max':  ('SAGE×4 + JK-max  + DropEdge(0.3) + dropout=0.5 + wd=1e-3 + ES',
                   make_d28_sage_jk_max_fn, False),
    'D29_gatv2_jk':     ('GATv2×3 + JK-max + PairNorm + DropEdge(0.3) + wd=1e-3 + ES',
                   make_d29_gatv2_jk_pairnorm_fn, False),
    'D30_gcn_jk_cat':   ('GCN×4  + JK-cat + PairNorm + dropout=0.6 + wd=5e-3 + ES',
                   make_d30_gcn_jk_cat_pairnorm_fn, False),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+',
                        default=list(EXPERIMENT_REGISTRY.keys()),
                        help='Which experiments to run')
    parser.add_argument('--sed_emb', default='outputs/sed_probe_emb.npy',
                        help='Path to SED backbone embeddings (for P3/B2)')
    args = parser.parse_args()

    print(f'\n{"="*62}')
    print('  BirdCLEF 2026 — Perch Probe Experiment Runner')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'{"="*62}\n')

    meta_full, emb_full, scores_full_raw, Y_FULL, PRIMARY_LABELS = load_data()

    # Load SED embeddings if available
    sed_emb_full = None
    sed_emb_path = Path(args.sed_emb)
    if sed_emb_path.exists():
        sed_emb_full = np.load(sed_emb_path).astype(np.float32)
        print(f'SED embeddings loaded: {sed_emb_full.shape}')
    else:
        print(f'SED embeddings not found at {sed_emb_path} — P3/B2 will be skipped')

    results_df = None

    for exp_name in args.experiments:
        if exp_name not in EXPERIMENT_REGISTRY:
            print(f'SKIP unknown experiment: {exp_name}')
            continue

        desc, fn_factory, needs_idx = EXPERIMENT_REGISTRY[exp_name]

        # Skip SED-dependent experiments if embeddings not available
        if exp_name in ('P3', 'B2') and sed_emb_full is None:
            print(f'SKIP {exp_name}: SED embeddings not available')
            continue

        print(f'\n[{exp_name}] {desc}')
        t0 = time.time()

        try:
            if exp_name in ('P3', 'B2'):
                probe_fn = make_p3_sed_fusion_fn_aligned(sed_emb_full)
                auc, _ = oof_auc_with_indices(meta_full, emb_full, Y_FULL, probe_fn)
            else:
                fn = fn_factory()
                if fn is None:
                    print(f'  SKIP {exp_name}: factory returned None')
                    continue
                auc, _ = oof_auc(meta_full, emb_full, Y_FULL, fn)

        except Exception as e:
            print(f'  ERROR in {exp_name}: {e}')
            import traceback; traceback.print_exc()
            continue

        runtime_min = (time.time() - t0) / 60
        results_df  = write_result(exp_name, desc, auc, runtime_min)

    if results_df is not None:
        print_leaderboard(results_df)
        print(f'Results saved to: {EXCEL_PATH}')


if __name__ == '__main__':
    main()
