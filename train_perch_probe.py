"""
train_perch_probe.py — Trainable Perch Embedding Probe Head

Replaces sklearn LogReg probe with a gradient-descent trained MLP.
Also supports Prototype classifier (F) and Tip-Adapter cache (E).

Input  : pre-computed 1536-dim Perch embeddings (embeddings_cache_nohuman/)
PCA(k) : fitted on ALL 107k embeddings for stable decomposition
Head   : Linear or MLP trained with ASL loss on 739 labeled soundscape clips
Eval   : GroupKFold-5 OOF ROC-AUC (same protocol as LogReg probe)

New features (D/E/F):
  D: add_train_clips  — add 85k weakly-labeled train_audio clips to MLP training
  E: TipAdapter       — cache-based retrieval from support soundscape clips
  F: ProtoClassifier  — class prototype (mean embedding) distance score
     + per-class gamma learned on fold training data

Usage:
    python train_perch_probe.py --config configs/perch_probe_pca128_v1.yaml
    python train_perch_probe.py --config configs/perch_probe_v2_full.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax as scipy_softmax
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ── ASL Loss (same as SED training) ─────────────────────────────────────────

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        xs_pos = pred_sigmoid
        xs_neg = 1 - pred_sigmoid

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        lo_pos = target * torch.log(xs_pos.clamp(min=self.eps))
        lo_neg = (1 - target) * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * target
            pt1 = xs_neg * (1 - target)
            pt  = pt0 + pt1
            one_sided_w = torch.pow(1 - pt, self.gamma_neg) * (1 - target) + \
                          torch.pow(1 - pt, self.gamma_pos) * target
            lo_pos *= one_sided_w
            lo_neg *= one_sided_w

        loss = -(lo_pos + lo_neg)
        return loss.mean()


# ── MLP Probe ─────────────────────────────────────────────────────────────────

class ProbeNet(nn.Module):
    """MLP probe head on top of PCA-reduced Perch embeddings."""

    def __init__(self, pca_dim: int, hidden_dim: int, num_classes: int = 234,
                 dropout: float = 0.3, use_hidden: bool = True):
        super().__init__()
        if use_hidden and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(pca_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(pca_dim, num_classes)

    def forward(self, x):
        return self.net(x)


# ── (F) Prototype Classifier ─────────────────────────────────────────────────

class ProtoClassifier:
    """
    Prototype-based classifier (no parameters, computed per fold).

    For each class c:  proto[c] = mean(Z[Y[:, c] > 0.5])
    Score:  s[i, c] = -||Z[i] - proto[c]||^2  (higher = closer)

    Scores are per-class min-max normalised to [0,1] so they blend
    directly with MLP sigmoid probabilities.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.protos: np.ndarray = None       # (C, D)
        self.has_proto: np.ndarray = None    # (C,) bool

    def fit(self, Z: np.ndarray, Y: np.ndarray):
        D = Z.shape[1]
        self.protos = np.zeros((self.num_classes, D), dtype=np.float32)
        self.has_proto = np.zeros(self.num_classes, dtype=bool)
        for c in range(self.num_classes):
            mask = Y[:, c] > 0.5
            if mask.sum() > 0:
                self.protos[c] = Z[mask].mean(0)
                self.has_proto[c] = True

    def predict_probs(self, Z: np.ndarray) -> np.ndarray:
        """Return per-class probability-like scores in [0, 1]. Shape (N, C)."""
        # Squared L2 distance to each prototype: (N, C)
        diff = Z[:, None, :] - self.protos[None, :, :]   # (N, C, D)
        dists = (diff ** 2).sum(-1)                        # (N, C)
        scores = -dists                                    # higher = closer

        # Per-class min-max normalisation → [0, 1]
        s_min = scores.min(0, keepdims=True)
        s_max = scores.max(0, keepdims=True)
        probs = (scores - s_min) / (s_max - s_min + 1e-8)

        # Classes with no prototype → 0
        probs[:, ~self.has_proto] = 0.0
        return probs.astype(np.float32)


# ── (E) Tip-Adapter ───────────────────────────────────────────────────────────

class TipAdapter:
    """
    Cache-based retrieval adapter (Tip-Adapter, Zhang et al. 2022).

    keys   = L2-normalised support embeddings  (N_sup, D)
    values = multi-hot labels                  (N_sup, C)

    Inference:
        sim[i, j] = cosine_sim(Z_query[i], keys[j])
        attn[i]   = softmax(sim[i] * temperature)
        pred[i]   = attn[i] @ values          ← soft label aggregation in [0, 1]
    """

    def __init__(self, temperature: float = 10.0):
        self.temperature = temperature
        self.keys: np.ndarray = None    # (N_sup, D) L2-normed
        self.values: np.ndarray = None  # (N_sup, C)

    def fit(self, Z_support: np.ndarray, Y_support: np.ndarray):
        norms = np.linalg.norm(Z_support, axis=1, keepdims=True).clip(min=1e-8)
        self.keys = (Z_support / norms).astype(np.float32)
        self.values = Y_support.astype(np.float32)

    def predict_probs(self, Z_query: np.ndarray) -> np.ndarray:
        """Return soft predictions in [0, 1]. Shape (N, C)."""
        norms = np.linalg.norm(Z_query, axis=1, keepdims=True).clip(min=1e-8)
        Z_norm = (Z_query / norms).astype(np.float32)
        sim = Z_norm @ self.keys.T                          # (N, N_sup)
        attn = np.exp(sim * self.temperature)
        attn = attn / attn.sum(1, keepdims=True).clip(min=1e-8)  # (N, N_sup)
        return (attn @ self.values).astype(np.float32)      # (N, C)


# ── All-But-Top preprocessing ────────────────────────────────────────────────

def all_but_top(embs: np.ndarray, n_top: int = 10) -> np.ndarray:
    """
    Remove top-n_top principal components from embeddings (Mu & Viswanath, ICLR 2018).
    Removes 'rogue dimensions' (recording-environment variance) before PCA.
    Returns cleaned embeddings with same shape as input.
    """
    mu = embs.mean(axis=0)
    X = embs - mu
    pca_top = PCA(n_components=n_top)
    pca_top.fit(X)
    proj = X @ pca_top.components_.T @ pca_top.components_
    return (X - proj) + mu


# ── FeCAM: Mahalanobis NCM with Ledoit-Wolf + Tukey (Goswami et al., NeurIPS 2023) ──

class FeCAMClassifier:
    """
    Feature Covariance Aware Metric (FeCAM).
    Replaces Euclidean/cosine prototype distance with per-class Mahalanobis distance.

    Three stabilization tricks:
      1. Tukey transform: x → sign(x)*|x|^0.5  (Gaussianizes skewed dims)
      2. Ledoit-Wolf shrinkage: regularized covariance (handles small-N per class)
      3. Correlation normalization: normalize diagonals before inversion
    """

    def __init__(self, num_classes: int, lam: float = 0.5):
        self.num_classes = num_classes
        self.lam = lam
        self.prototypes: np.ndarray = None   # (C, D) — Tukey-transformed means
        self.inv_covs: list = None           # list of (D, D) inverse correlation matrices
        self.has_proto: np.ndarray = None    # (C,) bool

    @staticmethod
    def _tukey(X: np.ndarray, lam: float = 0.5) -> np.ndarray:
        return np.sign(X) * np.abs(X) ** lam

    def fit(self, Z: np.ndarray, Y: np.ndarray):
        """Z: (N, D), Y: (N, C) multi-hot."""
        D = Z.shape[1]
        self.prototypes = np.zeros((self.num_classes, D), dtype=np.float32)
        self.inv_covs = [None] * self.num_classes
        self.has_proto = np.zeros(self.num_classes, dtype=bool)

        # Shared covariance fallback (for classes with very few samples)
        Z_t = self._tukey(Z)
        try:
            lw_global = LedoitWolf().fit(Z_t)
            std_g = np.sqrt(np.diag(lw_global.covariance_))
            corr_g = lw_global.covariance_ / (np.outer(std_g, std_g) + 1e-8)
            shared_inv_cov = np.linalg.pinv(corr_g).astype(np.float32)
        except Exception:
            shared_inv_cov = np.eye(D, dtype=np.float32)

        for c in range(self.num_classes):
            mask = Y[:, c] > 0.5
            n_pos = mask.sum()
            if n_pos == 0:
                continue
            X_c = self._tukey(Z[mask])
            self.prototypes[c] = X_c.mean(axis=0)
            self.has_proto[c] = True

            if n_pos >= max(D // 4, 5):   # enough samples for per-class cov
                try:
                    lw = LedoitWolf().fit(X_c)
                    std = np.sqrt(np.diag(lw.covariance_))
                    corr = lw.covariance_ / (np.outer(std, std) + 1e-8)
                    self.inv_covs[c] = np.linalg.pinv(corr).astype(np.float32)
                except Exception:
                    self.inv_covs[c] = shared_inv_cov
            else:
                self.inv_covs[c] = shared_inv_cov   # fallback to global

    def predict_probs(self, Z: np.ndarray) -> np.ndarray:
        """Return per-class scores (higher = more similar). Shape (N, C)."""
        Z_t = self._tukey(Z).astype(np.float32)
        N, C = len(Z), self.num_classes
        scores = np.full((N, C), -1e9, dtype=np.float32)

        for c in range(C):
            if not self.has_proto[c]:
                continue
            diff = Z_t - self.prototypes[c][None, :]   # (N, D)
            # Mahalanobis: diag of (N, D) @ (D, D) @ (D, N) — computed row-wise
            maha = np.einsum('nd,dd,nd->n', diff, self.inv_covs[c], diff)
            scores[:, c] = -maha   # higher = closer

        # Per-class min-max normalization to [0, 1]
        s_min = scores.min(0, keepdims=True)
        s_max = scores.max(0, keepdims=True)
        probs = (scores - s_min) / (s_max - s_min + 1e-8)
        probs[:, ~self.has_proto] = 0.0
        return probs.astype(np.float32)


# ── LaplacianShot transductive inference ─────────────────────────────────────

def laplacianshot_inference(
    query_embs: np.ndarray,      # (Q, D) — already PCA-transformed, L2-normalized
    prototypes: np.ndarray,      # (C, D) — L2-normalized class centroids
    k: int = 7,
    lam: float = 0.5,
    n_iter: int = 20,
) -> np.ndarray:
    """
    Transductive batch inference using Laplacian regularization (Ziko et al., ICML 2020).
    Exploits mutual similarity among ALL query clips simultaneously.
    Best for soundscape batch inference (all 739 clips available at once).

    Returns (Q, C) soft label predictions.
    """
    Q = len(query_embs)

    # Normalize for cosine similarity
    norms = np.linalg.norm(query_embs, axis=1, keepdims=True).clip(min=1e-8)
    Z = query_embs / norms
    proto_norms = np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-8)
    P = prototypes / proto_norms

    # Unary: cosine distance to each prototype
    unary = 1.0 - Z @ P.T    # (Q, C) in [0, 2]; 0 = identical

    # Build kNN affinity graph over query embeddings
    sim = Z @ Z.T             # (Q, Q) cosine similarity
    np.fill_diagonal(sim, -np.inf)
    W = np.zeros((Q, Q), dtype=np.float32)
    for i in range(Q):
        top_k_idx = np.argsort(sim[i])[-k:]
        vals = sim[i][top_k_idx]
        W[i, top_k_idx] = np.clip(vals, 0, None)
    W = np.maximum(W, W.T)    # symmetrize

    # Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    D_diag = W.sum(1)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(D_diag) + 1e-8))
    L = np.eye(Q, dtype=np.float32) - D_inv_sqrt @ W @ D_inv_sqrt

    # Iterative bound optimizer
    Y = scipy_softmax(-unary, axis=1)
    for _ in range(n_iter):
        grad = unary + lam * (L @ Y)
        Y = scipy_softmax(-grad, axis=1)

    return Y.astype(np.float32)


# ── (F) Per-class blend weight (learnable gamma) ─────────────────────────────

def fit_per_class_gamma(
    mlp_probs:   np.ndarray,   # (N, C)
    proto_probs: np.ndarray,   # (N, C)
    tip_probs:   np.ndarray,   # (N, C)   (may be zeros if not used)
    Y:           np.ndarray,   # (N, C)
    grid:        tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> np.ndarray:
    """
    For each class c, find the scalar gamma_c in `grid` that maximises
    AUC for:  final[:,c] = gamma_c*mlp + delta_c*proto + (1-gamma_c-delta_c)*tip

    Simplified: sweep gamma ∈ [0,1] for (mlp vs proto+tip average).
    Returns gamma array of shape (C,).
    """
    C = Y.shape[1]
    gammas = np.zeros(C, dtype=np.float32)
    other_probs = 0.5 * proto_probs + 0.5 * tip_probs   # average of non-MLP sources

    for c in range(C):
        y_c = Y[:, c]
        if y_c.sum() == 0:
            gammas[c] = 0.5   # no positives → default
            continue
        best_auc_c = -1.0
        best_g = 0.5
        for g in grid:
            blend = g * mlp_probs[:, c] + (1.0 - g) * other_probs[:, c]
            try:
                a = roc_auc_score(y_c, blend)
            except Exception:
                a = -1.0
            if a > best_auc_c:
                best_auc_c = a
                best_g = g
        gammas[c] = best_g
    return gammas


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    def __init__(self, Z: np.ndarray, Y: np.ndarray):
        self.Z = torch.from_numpy(Z.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, i):
        return self.Z[i], self.Y[i]


# ── Helpers ───────────────────────────────────────────────────────────────────

def macro_auc(y_true, y_score):
    keep = y_true.sum(0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro')


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for Z, Y in loader:
        Z, Y = Z.to(device), Y.to(device)
        optimizer.zero_grad()
        logits = model(Z)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(Z)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits = []
    for Z, _ in loader:
        Z = Z.to(device)
        all_logits.append(model(Z).cpu())
    return torch.cat(all_logits, 0).numpy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg['experiment']['name']
    seed = cfg['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # WandB
    use_wandb = cfg.get('wandb', {}).get('enabled', False)
    if use_wandb:
        wandb.init(
            project=cfg['wandb'].get('project', 'birdclef-2026'),
            name=exp_name,
            config=cfg,
            tags=cfg['wandb'].get('tags', []),
        )

    # ── Load manifest ─────────────────────────────────────────────────────────
    base_dir = Path(cfg['data']['base_dir'])
    manifest_path = Path(cfg['data']['manifest_path'])
    manifest = pd.read_csv(manifest_path)
    print(f'Manifest: {len(manifest)} rows, splits: {manifest["split"].value_counts().to_dict()}')

    # ── Load taxonomy → primary_label → class index ───────────────────────────
    sample_sub = pd.read_csv(base_dir / 'sample_submission.csv')
    primary_labels = sample_sub.columns[1:].tolist()
    label2idx = {lb: i for i, lb in enumerate(primary_labels)}
    NUM_CLASSES = len(primary_labels)
    print(f'Species: {NUM_CLASSES}')

    # ── Load ALL embeddings for PCA fitting ───────────────────────────────────
    pca_splits = set(cfg['data'].get('pca_fit_splits', ['train', 'soundscape']))
    pca_mask = manifest['split'].isin(pca_splits)
    pca_rows = manifest[pca_mask].reset_index(drop=True)
    print(f'Loading {len(pca_rows)} embeddings for PCA fitting...')

    def load_emb_batch(rows, desc='Loading'):
        embs = []
        for _, row in tqdm(rows.iterrows(), total=len(rows), desc=desc):
            e = np.load(row['npy_path'])
            embs.append(e)
        return np.stack(embs, 0).astype(np.float32)

    E_pca = load_emb_batch(pca_rows, 'PCA embs')
    print(f'PCA input: {E_pca.shape}')

    # ── (ABT) All-But-Top preprocessing ──────────────────────────────────────
    abt_n_top = cfg['model'].get('abt_n_top', 0)
    if abt_n_top > 0:
        print(f'Applying All-But-Top({abt_n_top}): removing top {abt_n_top} rogue PCA directions...')
        E_pca = all_but_top(E_pca, n_top=abt_n_top)
        print(f'ABT done. Shape: {E_pca.shape}')

    # ── Fit StandardScaler + PCA ──────────────────────────────────────────────
    pca_dim = cfg['model']['pca_dim']
    print(f'Fitting StandardScaler + PCA({pca_dim})...')
    scaler = StandardScaler()
    E_pca_scaled = scaler.fit_transform(E_pca)

    n_comp = min(pca_dim, E_pca_scaled.shape[0] - 1, E_pca_scaled.shape[1])
    pca = PCA(n_components=n_comp, whiten=cfg['model'].get('pca_whiten', True))
    pca.fit(E_pca_scaled)
    print(f'PCA({n_comp}) explained variance: {pca.explained_variance_ratio_.sum():.4f}')

    del E_pca, E_pca_scaled
    import gc; gc.collect()

    # ── Load labeled soundscape embeddings + labels ───────────────────────────
    ss_mask = manifest['split'] == 'soundscape'
    ss_rows = manifest[ss_mask].reset_index(drop=True)
    print(f'Labeled soundscape clips: {len(ss_rows)}')

    E_ss = load_emb_batch(ss_rows, 'SS embs')

    # Build multi-hot label matrix
    Y_ss = np.zeros((len(ss_rows), NUM_CLASSES), dtype=np.float32)
    for i, row in ss_rows.iterrows():
        raw = str(row.get('label', ''))
        for sp in raw.split(';'):
            sp = sp.strip()
            if sp in label2idx:
                Y_ss[i, label2idx[sp]] = 1.0

    print(f'Y_ss positive rate: {Y_ss.mean():.4f}  (should be ~0.02–0.05)')

    # PCA transform soundscape embeddings
    E_ss_scaled = scaler.transform(E_ss)
    Z_ss = pca.transform(E_ss_scaled).astype(np.float32)
    del E_ss, E_ss_scaled

    # Group by file for GroupKFold
    groups = ss_rows['source_file'].values

    # ── (D) Optional: add train_audio clips as additional MLP training data ───
    # Note: train clips are used ONLY for MLP training, NOT for proto/tip keys.
    # Proto and TipAdapter always use soundscape support clips only (multi-hot labels).
    add_train = cfg['data'].get('add_train_clips', False)
    ss_repeat = max(1, int(cfg['data'].get('ss_repeat', 1)))  # SFDA: oversample soundscape clips N times
    Z_train_extra = None
    Y_train_extra = None
    if add_train:
        tr_mask = manifest['split'] == 'train'
        tr_rows = manifest[tr_mask].reset_index(drop=True)
        print(f'Loading {len(tr_rows)} train clips for extra MLP supervision...')
        E_tr = load_emb_batch(tr_rows, 'Train embs')
        E_tr_scaled = scaler.transform(E_tr)
        Z_tr_extra = pca.transform(E_tr_scaled).astype(np.float32)
        del E_tr, E_tr_scaled

        Y_tr_extra = np.zeros((len(tr_rows), NUM_CLASSES), dtype=np.float32)
        for i, row in tr_rows.iterrows():
            sp = str(row.get('label', '')).strip()
            if sp in label2idx:
                Y_tr_extra[i, label2idx[sp]] = 1.0

        Z_train_extra = Z_tr_extra
        Y_train_extra = Y_tr_extra
        print(f'Train extra: {Z_tr_extra.shape}')

    # ── Feature flags ─────────────────────────────────────────────────────────
    use_proto       = cfg['model'].get('use_proto', False)
    use_tip         = cfg['model'].get('use_tip', False)
    use_fecam       = cfg['model'].get('use_fecam', False)    # FeCAM Mahalanobis NCM
    use_laplacian   = cfg['model'].get('use_laplacian', False) # LaplacianShot transductive
    laplacian_k     = cfg['model'].get('laplacian_k', 7)
    laplacian_lam   = cfg['model'].get('laplacian_lam', 0.5)
    use_gamma       = cfg['model'].get('use_gamma', False)
    tip_temperature = cfg['model'].get('tip_temperature', 10.0)
    blend_mlp       = cfg['model'].get('blend_mlp', 1.0)
    blend_proto     = cfg['model'].get('blend_proto', 0.0)
    blend_tip       = cfg['model'].get('blend_tip', 0.0)
    blend_fecam     = cfg['model'].get('blend_fecam', 0.0)

    print(f'Features: proto={use_proto}  tip={use_tip}  fecam={use_fecam}  laplacian={use_laplacian}  gamma={use_gamma}')
    print(f'Blend weights — mlp={blend_mlp}  proto={blend_proto}  tip={blend_tip}  fecam={blend_fecam}')

    # ── Training hyperparams ──────────────────────────────────────────────────
    n_splits     = cfg['training'].get('n_folds', 5)
    hidden_dim   = cfg['model'].get('hidden_dim', 256)
    use_hidden   = cfg['model'].get('use_hidden', True)
    dropout      = cfg['model'].get('dropout', 0.3)
    epochs       = cfg['training']['epochs']
    batch_size   = cfg['training']['batch_size']
    lr           = cfg['training']['learning_rate']
    wd           = cfg['training'].get('weight_decay', 1e-4)
    patience     = cfg['training'].get('patience', 15)
    gamma_neg    = cfg['training'].get('asl_gamma_neg', 4.0)
    gamma_pos    = cfg['training'].get('asl_gamma_pos', 0.0)
    asl_clip     = cfg['training'].get('asl_clip', 0.05)

    criterion = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=asl_clip)

    # ── GroupKFold cross-validation ───────────────────────────────────────────
    gkf = GroupKFold(n_splits=n_splits)
    oof_logits_mlp  = np.zeros_like(Y_ss)   # MLP-only logits
    oof_probs_blend = np.zeros_like(Y_ss)   # blended final probs
    best_auc_per_fold = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(Z_ss, groups=groups), 1):
        print(f'\n{"="*50}  Fold {fold}/{n_splits}  {"="*50}')

        # Soundscape train/val splits
        Z_tr_ss, Y_tr_ss = Z_ss[tr_idx], Y_ss[tr_idx]
        Z_va,    Y_va    = Z_ss[va_idx],  Y_ss[va_idx]

        # MLP training set: soundscape (optionally repeated) + optional train_audio extras
        # ss_repeat > 1 is an SFDA trick: oversample target-domain labeled clips to increase
        # their gradient signal relative to the large train_audio set.
        Z_tr_mlp = np.tile(Z_tr_ss, (ss_repeat, 1)) if ss_repeat > 1 else Z_tr_ss
        Y_tr_mlp = np.tile(Y_tr_ss, (ss_repeat, 1)) if ss_repeat > 1 else Y_tr_ss
        if Z_train_extra is not None:
            Z_tr_mlp = np.concatenate([Z_tr_mlp, Z_train_extra], 0)
            Y_tr_mlp = np.concatenate([Y_tr_mlp, Y_train_extra], 0)

        tr_ds = EmbeddingDataset(Z_tr_mlp, Y_tr_mlp)
        va_ds = EmbeddingDataset(Z_va,     Y_va)
        tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # ── Train MLP ─────────────────────────────────────────────────────────
        model = ProbeNet(pca_dim=n_comp, hidden_dim=hidden_dim,
                         num_classes=NUM_CLASSES, dropout=dropout,
                         use_hidden=use_hidden).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc_fold  = 0.0
        best_logits    = np.zeros((len(va_idx), NUM_CLASSES), dtype=np.float32)
        no_imp = 0

        for ep in range(1, epochs + 1):
            tr_loss = train_one_epoch(model, tr_ld, optimizer, criterion, device)
            scheduler.step()

            logits = predict(model, va_ld, device)
            probs  = torch.sigmoid(torch.from_numpy(logits)).numpy()
            auc    = macro_auc(Y_va, probs)

            if auc > best_auc_fold:
                best_auc_fold = auc
                no_imp = 0
                best_logits = logits.copy()
            else:
                no_imp += 1

            if ep % 10 == 0 or ep == epochs or no_imp == patience:
                print(f'  Fold {fold} ep{ep:3d}: loss={tr_loss:.4f}  val_auc={auc:.4f}  best={best_auc_fold:.4f}')

            if use_wandb:
                wandb.log({f'fold{fold}/val_auc': auc, f'fold{fold}/loss': tr_loss, 'epoch': ep})

            if no_imp >= patience:
                print(f'  Early stop at ep{ep}')
                break

        oof_logits_mlp[va_idx] = best_logits
        best_auc_per_fold.append(best_auc_fold)
        print(f'Fold {fold} MLP AUC: {best_auc_fold:.4f}')

        # ── (E) TipAdapter — fit on fold's soundscape train clips ─────────────
        tip_probs_va = np.zeros((len(va_idx), NUM_CLASSES), dtype=np.float32)
        if use_tip:
            tip = TipAdapter(temperature=tip_temperature)
            tip.fit(Z_tr_ss, Y_tr_ss)
            tip_probs_va = tip.predict_probs(Z_va)
            tip_auc = macro_auc(Y_va, tip_probs_va)
            print(f'  Fold {fold} TipAdapter AUC: {tip_auc:.4f}')

        # ── (F) ProtoClassifier — fit on fold's soundscape train clips ─────────
        proto_probs_va = np.zeros((len(va_idx), NUM_CLASSES), dtype=np.float32)
        if use_proto:
            proto = ProtoClassifier(NUM_CLASSES)
            proto.fit(Z_tr_ss, Y_tr_ss)
            proto_probs_va = proto.predict_probs(Z_va)
            proto_auc = macro_auc(Y_va, proto_probs_va)
            print(f'  Fold {fold} Proto AUC:       {proto_auc:.4f}')

        # ── FeCAM: Mahalanobis NCM ─────────────────────────────────────────────
        fecam_probs_va = np.zeros((len(va_idx), NUM_CLASSES), dtype=np.float32)
        if use_fecam:
            fecam = FeCAMClassifier(NUM_CLASSES)
            fecam.fit(Z_tr_ss, Y_tr_ss)
            fecam_probs_va = fecam.predict_probs(Z_va)
            fecam_auc = macro_auc(Y_va, fecam_probs_va)
            print(f'  Fold {fold} FeCAM AUC:       {fecam_auc:.4f}')

        # ── Blend ─────────────────────────────────────────────────────────────
        mlp_probs_va = torch.sigmoid(torch.from_numpy(best_logits)).numpy()

        # ── LaplacianShot: transductive batch refinement ───────────────────────
        if use_laplacian:
            # Build prototypes from fold training soundscape clips
            proto_means = np.zeros((NUM_CLASSES, n_comp), dtype=np.float32)
            has_proto_lp = np.zeros(NUM_CLASSES, dtype=bool)
            for c in range(NUM_CLASSES):
                mask_c = Y_tr_ss[:, c] > 0.5
                if mask_c.sum() > 0:
                    proto_means[c] = Z_tr_ss[mask_c].mean(0)
                    has_proto_lp[c] = True
            # Only use classes with prototypes
            lp_scores = laplacianshot_inference(
                Z_va, proto_means, k=laplacian_k, lam=laplacian_lam
            )
            lp_auc = macro_auc(Y_va, lp_scores)
            print(f'  Fold {fold} LaplacianShot AUC: {lp_auc:.4f}')
            # Blend MLP + LaplacianShot (LaplacianShot replaces proto in blend)
            mlp_probs_va = 0.5 * mlp_probs_va + 0.5 * lp_scores

        if use_gamma and (use_proto or use_tip or use_fecam):
            all_other = np.zeros_like(proto_probs_va)
            n_sources = 0
            if use_proto:
                all_other += proto_probs_va; n_sources += 1
            if use_tip:
                all_other += tip_probs_va; n_sources += 1
            if use_fecam:
                all_other += fecam_probs_va; n_sources += 1
            all_other /= max(n_sources, 1)
            gammas = fit_per_class_gamma(mlp_probs_va, all_other,
                                         np.zeros_like(all_other), Y_va)
            blended = gammas[None, :] * mlp_probs_va + (1.0 - gammas[None, :]) * all_other
            print(f'  Fold {fold} gamma mean={gammas.mean():.3f}  min={gammas.min():.3f}  max={gammas.max():.3f}')
        else:
            total_w = blend_mlp + blend_proto + blend_tip + blend_fecam
            blended = (blend_mlp   * mlp_probs_va
                     + blend_proto  * proto_probs_va
                     + blend_tip    * tip_probs_va
                     + blend_fecam  * fecam_probs_va) / max(total_w, 1e-8)

        blend_auc = macro_auc(Y_va, blended)
        print(f'  Fold {fold} Blended AUC:      {blend_auc:.4f}')

        oof_probs_blend[va_idx] = blended

    # ── OOF evaluation ────────────────────────────────────────────────────────
    oof_probs_mlp = torch.sigmoid(torch.from_numpy(oof_logits_mlp)).numpy()
    oof_auc_mlp   = macro_auc(Y_ss, oof_probs_mlp)
    oof_auc_blend = macro_auc(Y_ss, oof_probs_blend)

    print(f'\n{"="*60}')
    print(f'OOF AUC (MLP only):  {oof_auc_mlp:.4f}')
    print(f'OOF AUC (blended):   {oof_auc_blend:.4f}')
    print(f'Per-fold MLP:        {[f"{a:.4f}" for a in best_auc_per_fold]}')
    print(f'{"="*60}')

    oof_auc = oof_auc_blend   # primary metric

    if use_wandb:
        wandb.log({'oof_auc': oof_auc, 'oof_auc_mlp': oof_auc_mlp, 'oof_auc_blend': oof_auc_blend})

    # ── Train final model on ALL soundscape data ──────────────────────────────
    print('\nTraining final model on ALL soundscape data...')
    Z_all = Z_ss
    Y_all = Y_ss
    if Z_train_extra is not None:
        Z_all = np.concatenate([Z_all, Z_train_extra], 0)
        Y_all = np.concatenate([Y_all, Y_train_extra], 0)

    all_ds = EmbeddingDataset(Z_all, Y_all)
    all_ld = DataLoader(all_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    final_model = ProbeNet(pca_dim=n_comp, hidden_dim=hidden_dim,
                           num_classes=NUM_CLASSES, dropout=dropout,
                           use_hidden=use_hidden).to(device)
    final_opt   = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=wd)
    final_sched = torch.optim.lr_scheduler.CosineAnnealingLR(final_opt, T_max=epochs)

    for ep in range(1, epochs + 1):
        loss = train_one_epoch(final_model, all_ld, final_opt, criterion, device)
        final_sched.step()
        if ep % 20 == 0:
            print(f'  Final ep{ep}: loss={loss:.4f}')

    # ── Build final TipAdapter + ProtoClassifier on ALL soundscape data ────────
    final_tip   = None
    final_proto = None

    if use_tip:
        final_tip = TipAdapter(temperature=tip_temperature)
        final_tip.fit(Z_ss, Y_ss)
        print('Final TipAdapter fitted on all 739 soundscape clips.')

    if use_proto:
        final_proto = ProtoClassifier(NUM_CLASSES)
        final_proto.fit(Z_ss, Y_ss)
        print('Final ProtoClassifier fitted on all 739 soundscape clips.')

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path(cfg['output']['dir']) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / 'pca_params.npz',
             pca_components=pca.components_,
             pca_mean=pca.mean_,
             pca_scale=np.sqrt(pca.explained_variance_) if pca.whiten else np.ones(n_comp),
             scaler_mean=scaler.mean_,
             scaler_scale=scaler.scale_,
             pca_dim=np.array([n_comp]),
             oof_auc=np.array([oof_auc]))

    save_dict = {
        'model_state_dict': final_model.state_dict(),
        'pca_dim':     n_comp,
        'hidden_dim':  hidden_dim,
        'num_classes': NUM_CLASSES,
        'oof_auc':     oof_auc,
        'oof_auc_mlp': oof_auc_mlp,
        'per_fold_auc': best_auc_per_fold,
        'blend_mlp':   blend_mlp,
        'blend_proto': blend_proto,
        'blend_tip':   blend_tip,
    }

    if final_tip is not None:
        save_dict['tip_keys']       = final_tip.keys
        save_dict['tip_values']     = final_tip.values
        save_dict['tip_temperature'] = final_tip.temperature

    if final_proto is not None:
        save_dict['proto_protos']   = final_proto.protos
        save_dict['proto_has_proto'] = final_proto.has_proto

    torch.save(save_dict, out_dir / 'probe_head.pt')

    result = {
        'experiment':   exp_name,
        'pca_dim':      n_comp,
        'hidden_dim':   hidden_dim,
        'use_proto':    use_proto,
        'use_tip':      use_tip,
        'use_gamma':    use_gamma,
        'add_train':    add_train,
        'oof_auc':      float(oof_auc),
        'oof_auc_mlp':  float(oof_auc_mlp),
        'oof_auc_blend': float(oof_auc_blend),
        'per_fold_auc': [float(a) for a in best_auc_per_fold],
    }
    with open(out_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\nSaved to {out_dir}/')
    print(f'OOF AUC = {oof_auc:.4f}  (pca_dim={n_comp}, hidden={hidden_dim})')
    print(f'  MLP-only: {oof_auc_mlp:.4f}  |  blended: {oof_auc_blend:.4f}')

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
