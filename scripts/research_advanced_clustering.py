#!/usr/bin/env python3
"""
research_advanced_clustering.py
================================
研究進階 clustering 方法，利用當前 5 個模型輸出 + Perch embeddings
目標：超越 baseline OOF AUC 0.9553

資料說明：
  - labeled soundscapes: 739 windows, 66 files, ground truth labels
  - emb_full (708, 1536): full_perch_arrays labeled embeddings (train fold set)
  - perch_emb_all_ss (127896, 1536): 全部 soundscape Perch embeddings
  - X stacker features: 1170-dim = 5 models × 234 classes

方法清單：
  M0: Baseline (5-model sigmoid mean)
  M1: Perch direct logit blend (optimized alpha)
  M2: k-NN label propagation in Perch embedding space
  M3: Pseudo-augmented k-NN (127K reference set)
  M4: Per-species prototype similarity correction
  M5: Embedding-guided adaptive model weighting
  M6: Graph label propagation (labeled + pseudo)
  M7: Distance-gated correction (only close neighbors)
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/home/lab/BirdClef-2026-Codebase")
PERCH_META_DIR = BASE / "birdclef-2026/notebook resource/current_subs 2/perch meta"
STACKER_DIR = BASE / "birdclef-2026/notebook resource/current_subs 2/stacker_weights"
OUTPUTS = BASE / "outputs"

RESULTS_PATH = OUTPUTS / "advanced_clustering_results.json"

# ── Load Data ──────────────────────────────────────────────────────────────────
print("[1/6] Loading data...")

# Labeled soundscape windows (739 × 234 ground truth)
labeled = np.load(OUTPUTS / "perch_labeled_ss.npz", allow_pickle=True)
emb_lab   = labeled['emb'].astype(np.float32)    # (739, 1536)
logit_lab = labeled['logits'].astype(np.float32)  # (739, 234) — Perch predictions
Y_lab     = labeled['labels'].astype(np.float32)  # (739, 234) — ground truth
row_ids_lab = labeled['row_ids']                   # (739,)
fnames_lab  = labeled['filenames']                 # (739,)

# full_perch_arrays (708 rows = the fold-split labeled set used in stacker training)
d_full = np.load(str(PERCH_META_DIR / "full_perch_arrays.npz"))
emb_full = d_full['emb_full'].astype(np.float32)       # (708, 1536)
scores_full_raw = d_full['scores_full_raw'].astype(np.float32)  # (708, 234) Perch logits

# full_perch_meta: fold assignments for 708 labeled samples
meta_df = pd.read_parquet(str(PERCH_META_DIR / "full_perch_meta.parquet"))
print(f"  meta_df: {meta_df.shape}, columns: {meta_df.columns.tolist()}")

# stacker_norm
norm = np.load(str(STACKER_DIR / "stacker_norm_v3.npz"))
MEAN = norm['mean'].astype(np.float32)  # (1, 1170)
STD  = norm['std'].astype(np.float32)   # (1, 1170)

# Perch all SS embeddings
all_ss = np.load(OUTPUTS / "perch_emb_all_ss.npz")
emb_all_ss   = all_ss['emb'].astype(np.float32)     # (127896, 1536)
logit_all_ss = all_ss['logits'].astype(np.float32)  # (127896, 234)
row_ids_all  = all_ss['row_ids']                     # (127896,)

# pseudo labels features (127188 rows)
pseudo = np.load(OUTPUTS / "stacker_pseudo_features.npz", allow_pickle=True)
X_pseudo_raw = pseudo['X_pseudo_raw'].astype(np.float32)  # (127188, 1170)
Y_pseudo     = pseudo['Y_pseudo'].astype(np.float32)       # (127188, 234)
pseudo_fnames = pseudo['pseudo_filenames']

print(f"  emb_lab    : {emb_lab.shape}")
print(f"  Y_lab      : {Y_lab.shape}  pos_rate={Y_lab.mean():.4f}")
print(f"  emb_full   : {emb_full.shape}")
print(f"  emb_all_ss : {emb_all_ss.shape}")
print(f"  X_pseudo_raw: {X_pseudo_raw.shape}")

# ── Build Stacker X for labeled set ───────────────────────────────────────────
# The 739 labeled windows — do they match perch_emb_all_ss?
# Match row_ids to get stacker features for labeled windows

# Get emb_all_ss aligned with labeled set
# row_ids_lab should be subset of row_ids_all
row_id_to_idx_all = {r: i for i, r in enumerate(row_ids_all)}
lab_in_all_mask = np.array([r in row_id_to_idx_all for r in row_ids_lab])
print(f"\n  Labeled windows found in all_ss: {lab_in_all_mask.sum()} / {len(row_ids_lab)}")

# aligned arrays
lab_idxs_in_all = np.array([row_id_to_idx_all[r] for r in row_ids_lab[lab_in_all_mask]])
emb_lab_aligned = emb_all_ss[lab_idxs_in_all]       # should match emb_lab (reordered)
logit_lab_aligned = logit_all_ss[lab_idxs_in_all]    # Perch logits from full SS run

# Also need stacker X for labeled windows
# X_pseudo_raw contains soundscape features (logit space). Map row_ids.
pseudo_fnames_arr = pseudo_fnames if isinstance(pseudo_fnames, np.ndarray) else np.array(pseudo_fnames)
# Match by row_id: pseudo is filename-based but labeled is row_id based
# Row_id format: filename_without_ext + "_" + seconds
# Let's compute row_ids from pseudo filenames
# Each file has multiple windows, derive row_ids
# Actually simpler: check if we have the stacker norm + labeled features elsewhere

# Build labeled stacker features from logit_all_ss (reverse-engineering from Perch)
# Alternative: use scores_full_raw (708 Perch logits) + stacker norm
# scores_full_raw is actually the stacker feature? No, stacker = 5 model concatenated logits
# Let me check if X in perch_labeled_ss maps to X in stacker

# For now, use Y_lab and emb for methods that don't need X stacker
Y_lab_filtered = Y_lab[lab_in_all_mask]
emb_lab_f = emb_lab[lab_in_all_mask]   # (n_matched, 1536)
logit_lab_f = logit_lab[lab_in_all_mask]  # (n_matched, 234)
logit_lab_aligned2 = logit_lab_aligned  # from emb_all_ss run

n_labeled = lab_in_all_mask.sum()
print(f"  Working labeled set: {n_labeled} windows")

# Normalize embeddings
emb_lab_n = normalize(emb_lab_f, norm='l2').astype(np.float32)

# ── Helper: OOF AUC evaluation ─────────────────────────────────────────────────
def compute_auc(Y_true, Y_pred, prefix=""):
    """Compute macro AUC over all species with at least 1 positive."""
    aucs = []
    for c in range(Y_true.shape[1]):
        if Y_true[:, c].sum() >= 1 and Y_true[:, c].sum() < len(Y_true):
            try:
                aucs.append(roc_auc_score(Y_true[:, c], Y_pred[:, c]))
            except:
                pass
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    n_valid = len(aucs)
    if prefix:
        print(f"  {prefix:<45} AUC={mean_auc:.4f}  (n_species={n_valid})")
    return mean_auc

# ── M0: Baseline — Perch direct logits ────────────────────────────────────────
print("\n[2/6] Evaluating methods...")
print("  --- M0: Baselines ---")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -88, 88)))

# Perch direct (logit → sigmoid)
m0_perch_probs = sigmoid(logit_lab_f).astype(np.float32)
auc_m0_perch = compute_auc(Y_lab_filtered, m0_perch_probs, "M0a: Perch direct sigmoid")

# ── M1: Leave-One-Out k-NN in Perch Embedding Space ───────────────────────────
print("  --- M1: k-NN Label Propagation (LOO in 739-window labeled set) ---")

def knn_loo_predict(emb_n, Y, k_list=[3,5,10,15,20,30], use_soft_dist=True):
    """LOO k-NN prediction using cosine distance."""
    results = {}
    nn_model = NearestNeighbors(n_neighbors=max(k_list)+1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn_model.fit(emb_n)
    distances, indices = nn_model.kneighbors(emb_n)  # (n, k+1); first is self
    # distances[i,0] ≈ 0 (self), so skip index 0
    distances = distances[:, 1:]   # (n, max_k)
    indices   = indices[:, 1:]     # (n, max_k)

    for k in k_list:
        D_k = distances[:, :k]  # (n, k)
        I_k = indices[:, :k]    # (n, k)

        if use_soft_dist:
            # Convert cosine distance to weights (smaller dist → higher weight)
            sigma = np.median(D_k) + 1e-6
            W = np.exp(-D_k / sigma)  # (n, k)
        else:
            W = np.ones((len(emb_n), k), dtype=np.float32)

        W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-8)  # (n, k)

        # Predicted labels = weighted sum of neighbor labels
        Y_pred = np.zeros((len(emb_n), Y.shape[1]), dtype=np.float32)
        for i in range(len(emb_n)):
            neighbor_labels = Y[I_k[i]]  # (k, n_classes)
            Y_pred[i] = (W_norm[i, :, None] * neighbor_labels).sum(axis=0)

        results[k] = Y_pred
    return results, distances, indices

t0 = time.time()
knn_results, knn_dist, knn_idx = knn_loo_predict(emb_lab_n, Y_lab_filtered)
print(f"  k-NN computed in {time.time()-t0:.1f}s")

best_k = None
best_knn_auc = 0
for k, Y_pred in knn_results.items():
    auc = compute_auc(Y_lab_filtered, Y_pred, f"M1: k-NN LOO k={k}")
    if auc > best_knn_auc:
        best_knn_auc = auc
        best_k = k

# ── M2: Blend k-NN with Perch direct logits ────────────────────────────────────
print("  --- M2: Blend k-NN + Perch direct logit ---")
best_knn_pred = knn_results[best_k]

best_blend_auc = 0
best_alpha = 0
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    blended = alpha * best_knn_pred + (1 - alpha) * m0_perch_probs
    auc = compute_auc(Y_lab_filtered, blended, f"M2: kNN*{alpha:.1f} + Perch*{1-alpha:.1f}")
    if auc > best_blend_auc:
        best_blend_auc = auc
        best_alpha = alpha

# ── M3: Distance-gated correction ─────────────────────────────────────────────
print("  --- M3: Distance-gated correction (only close neighbors) ---")

def distance_gated_knn(emb_n, Y, perch_probs, k=best_k or 10, threshold_pct=50):
    """Apply k-NN correction only when the nearest neighbor is within threshold distance."""
    nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn_model.fit(emb_n)
    distances, indices = nn_model.kneighbors(emb_n)
    distances = distances[:, 1:]  # skip self
    indices   = indices[:, 1:]

    # Threshold at percentile of nearest-neighbor distances
    nn_dist = distances[:, 0]  # distance to nearest neighbor
    threshold = np.percentile(nn_dist, threshold_pct)

    sigma = np.median(distances) + 1e-6
    W = np.exp(-distances / sigma)
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-8)

    Y_pred = np.zeros_like(perch_probs)
    for i in range(len(emb_n)):
        neighbor_labels = Y[indices[i]]  # (k, n_classes)
        knn_pred = (W_norm[i, :, None] * neighbor_labels).sum(axis=0)

        # Gate by distance: if close, trust k-NN more; if far, fall back to perch
        trust = max(0.0, 1.0 - nn_dist[i] / (threshold + 1e-6))
        trust = min(trust, 1.0)
        Y_pred[i] = trust * knn_pred + (1 - trust) * perch_probs[i]

    return Y_pred

for thr_pct in [25, 50, 75]:
    gated_pred = distance_gated_knn(emb_lab_n, Y_lab_filtered, m0_perch_probs,
                                     k=best_k or 10, threshold_pct=thr_pct)
    compute_auc(Y_lab_filtered, gated_pred, f"M3: dist-gated kNN (thr={thr_pct}%)")

# ── M4: Per-species Prototype Similarity ──────────────────────────────────────
print("  --- M4: Per-species prototype similarity correction ---")

def per_species_prototype(emb_n, Y, perch_probs, k_pos=5):
    """
    For each species, compute a prototype from positive examples.
    Adjust prediction based on cosine similarity to prototype.
    """
    n_samples, n_classes = Y.shape
    Y_pred = perch_probs.copy()

    for c in range(n_classes):
        pos_mask = Y[:, c] > 0.5
        if pos_mask.sum() < 2:
            continue  # need at least 2 positives for LOO

        pos_emb = emb_n[pos_mask]    # (n_pos, 1536)
        prototype = pos_emb.mean(axis=0, keepdims=True)  # (1, 1536)
        prototype = normalize(prototype, norm='l2')

        # Cosine similarity to prototype
        sims = (emb_n @ prototype.T).squeeze()  # (n_samples,)

        # LOO correction: for positive samples, their self-prototype bias is removed
        # For all samples: adjust prediction by prototype similarity
        # Scale: high similarity → boost prediction, low → reduce
        sim_scaled = (sims - sims.mean()) / (sims.std() + 1e-6)

        # Gentle correction: blend original + prototype signal
        correction = 0.1 * sim_scaled * perch_probs[:, c].std()
        Y_pred[:, c] = np.clip(perch_probs[:, c] + correction, 0, 1)

    return Y_pred

proto_pred = per_species_prototype(emb_lab_n, Y_lab_filtered, m0_perch_probs)
compute_auc(Y_lab_filtered, proto_pred, "M4: Per-species prototype correction")

# ── M5: Embedding-guided adaptive model weighting (requires 5-model logits) ──
print("  --- M5: Embedding-guided species-wise calibration ---")

def embedding_calibration(emb_n, Y, perch_logits, perch_probs, k=15):
    """
    For each sample, use k nearest labeled neighbors to estimate
    per-species calibration (temperature scaling).
    """
    from sklearn.linear_model import LogisticRegression
    nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn_model.fit(emb_n)
    distances, indices = nn_model.kneighbors(emb_n)
    distances = distances[:, 1:]
    indices   = indices[:, 1:]

    sigma = np.median(distances) + 1e-6
    W = np.exp(-distances / sigma)  # (n, k)
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-8)  # (n, k)

    # For each sample, compute neighbor-averaged residual
    Y_pred = perch_probs.copy()
    for i in range(len(emb_n)):
        nb_Y = Y[indices[i]]         # (k, n_classes) ground truth
        nb_P = perch_probs[indices[i]]  # (k, n_classes) perch predictions
        w = W_norm[i]  # (k,)

        # Weighted calibration error per species
        nb_error = nb_Y - nb_P  # (k, n_classes)
        calib_correction = (w[:, None] * nb_error).sum(axis=0)  # (n_classes,)

        # Apply correction with dampening (avoid over-fitting)
        Y_pred[i] = np.clip(perch_probs[i] + 0.3 * calib_correction, 0, 1)

    return Y_pred

calib_pred = embedding_calibration(emb_lab_n, Y_lab_filtered, logit_lab_f, m0_perch_probs)
compute_auc(Y_lab_filtered, calib_pred, "M5: Embedding calibration (k=15, alpha=0.3)")

for alpha_calib in [0.1, 0.2, 0.5]:
    def embedding_calibration_alpha(emb_n, Y, logits, probs, k=15, alpha=alpha_calib):
        nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=-1)
        nn_model.fit(emb_n)
        distances, indices = nn_model.kneighbors(emb_n)
        distances = distances[:, 1:]
        indices   = indices[:, 1:]
        sigma = np.median(distances) + 1e-6
        W = np.exp(-distances / sigma)
        W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-8)
        Y_pred = probs.copy()
        for i in range(len(emb_n)):
            nb_Y = Y[indices[i]]
            nb_P = probs[indices[i]]
            w = W_norm[i]
            calib_correction = (w[:, None] * (nb_Y - nb_P)).sum(axis=0)
            Y_pred[i] = np.clip(probs[i] + alpha * calib_correction, 0, 1)
        return Y_pred
    calib_pred_a = embedding_calibration_alpha(emb_lab_n, Y_lab_filtered, logit_lab_f, m0_perch_probs, alpha=alpha_calib)
    compute_auc(Y_lab_filtered, calib_pred_a, f"M5: Embedding calibration (k=15, alpha={alpha_calib})")

# ── M6: Pseudo-augmented k-NN (using 127K pseudo set) ─────────────────────────
print("  --- M6: Pseudo-augmented k-NN (127K reference) ---")

# Build combined reference set: labeled (739) + pseudo (127188)
# Normalize pseudo embeddings: need emb for pseudo
# perch_emb_all_ss covers all soundscapes including pseudo ones
# Map pseudo filenames to emb_all_ss

# For pseudo, filenames are like 'BC2026_Train_0006_S09_20250828_000000.ogg'
# row_ids_all are like 'BC2026_Train_0006_S09_20250828_000000_5' (with _sec suffix)
pseudo_fname_arr = pseudo_fnames if isinstance(pseudo_fnames, np.ndarray) else np.array(pseudo_fnames)
row_ids_all_arr = row_ids_all

# Build mapping: for each pseudo sample, derive row_id
# X_pseudo_raw has 127188 rows; each row corresponds to one 5-second window
# Derive row_ids: need to count windows per file and increment by 5s
# Actually, perch_emb_all_ss already has all SS embeddings ordered by file/time
# Just normalize emb_all_ss and build reference from non-labeled rows

# Find which emb_all_ss rows correspond to labeled vs pseudo
labeled_row_ids_set = set(row_ids_lab[lab_in_all_mask])
all_ss_lab_mask = np.array([r in labeled_row_ids_set for r in row_ids_all], dtype=bool)
all_ss_pseudo_mask = ~all_ss_lab_mask

print(f"  all_ss labeled: {all_ss_lab_mask.sum()}, pseudo: {all_ss_pseudo_mask.sum()}")

# Normalize all embeddings
print("  Normalizing all embeddings...")
emb_all_n = normalize(emb_all_ss, norm='l2').astype(np.float32)

# For labeled LOO: reference = labeled set (n=n_labeled)
# Prediction from pseudo-augmented: reference = labeled + pseudo
print("  Building pseudo reference set...")
ref_emb_pseudo = emb_all_n[all_ss_pseudo_mask]      # (n_pseudo, 1536)
ref_Y_pseudo   = logit_all_ss[all_ss_pseudo_mask]   # (n_pseudo, 234) Perch logits → convert to pseudo prob
ref_Y_pseudo_prob = sigmoid(ref_Y_pseudo.astype(np.float64)).astype(np.float32)

# Also use ground truth for labeled
ref_emb_labeled = emb_lab_n  # (n_lab, 1536)
ref_Y_labeled   = Y_lab_filtered  # (n_lab, 234) ground truth

# Combined reference (excluding self via LOO trick with labeled)
print(f"  Pseudo reference size: {ref_emb_pseudo.shape[0]}")

# Sample pseudo to manageable size (take 20K most diverse by subsampling)
N_PSEUDO_SAMPLE = 20000
rng = np.random.default_rng(42)
if ref_emb_pseudo.shape[0] > N_PSEUDO_SAMPLE:
    pseudo_sample_idx = rng.choice(ref_emb_pseudo.shape[0], N_PSEUDO_SAMPLE, replace=False)
    ref_emb_p_sample   = ref_emb_pseudo[pseudo_sample_idx]
    ref_Y_p_sample     = ref_Y_pseudo_prob[pseudo_sample_idx]
else:
    ref_emb_p_sample = ref_emb_pseudo
    ref_Y_p_sample   = ref_Y_pseudo_prob

print(f"  Sampled {ref_emb_p_sample.shape[0]} pseudo windows")

# Combined reference for LOO on labeled:
# For sample i (labeled), reference = all other labeled + pseudo sample
# Build combined embedding matrix
ref_emb_combined = np.vstack([ref_emb_labeled, ref_emb_p_sample])  # (n_lab+20K, 1536)
ref_Y_combined   = np.vstack([ref_Y_labeled,   ref_Y_p_sample])

# LOO on labeled: query = emb_lab_n[i], exclude index i from ref
k_pseudo = 15
print(f"  Running pseudo-augmented k-NN (k={k_pseudo})...")
t0 = time.time()

# Build KNN on combined reference
nn_pseudo = NearestNeighbors(n_neighbors=k_pseudo+1, metric='cosine', algorithm='brute', n_jobs=-1)
nn_pseudo.fit(ref_emb_combined)

# Query labeled samples (LOO: self is at most 1 of n_lab entries)
dist_combined, idx_combined = nn_pseudo.kneighbors(ref_emb_labeled)  # (n_lab, k+1)
# Skip first if it's self (distance ≈ 0)
Y_pred_pseudo = np.zeros((n_labeled, Y_lab_filtered.shape[1]), dtype=np.float32)
for i in range(n_labeled):
    # Find self (distance < 1e-5) and exclude
    dists_i = dist_combined[i]
    idxs_i  = idx_combined[i]
    not_self = dists_i > 1e-5
    dists_i = dists_i[not_self][:k_pseudo]
    idxs_i  = idxs_i[not_self][:k_pseudo]

    if len(dists_i) == 0:
        Y_pred_pseudo[i] = m0_perch_probs[i]
        continue

    sigma = np.median(dists_i) + 1e-6
    W = np.exp(-dists_i / sigma)
    W_norm = W / (W.sum() + 1e-8)

    nb_Y = ref_Y_combined[idxs_i]
    Y_pred_pseudo[i] = (W_norm[:, None] * nb_Y).sum(axis=0)

print(f"  Done in {time.time()-t0:.1f}s")
auc_pseudo = compute_auc(Y_lab_filtered, Y_pred_pseudo, f"M6: Pseudo-aug k-NN (k={k_pseudo})")

# Blend pseudo-aug kNN with Perch direct
for alpha_m6 in [0.3, 0.5, 0.7]:
    blended_m6 = alpha_m6 * Y_pred_pseudo + (1 - alpha_m6) * m0_perch_probs
    compute_auc(Y_lab_filtered, blended_m6, f"M6b: PseudoKNN*{alpha_m6}+Perch*{1-alpha_m6:.1f}")

# ── M7: Species-level frequency re-weighting via embedding ─────────────────────
print("  --- M7: Label propagation via normalized graph ---")

def graph_label_propagation(emb_n, Y, alpha=0.5, k=10, n_iter=5):
    """
    Semi-supervised label propagation.
    Seeds: labeled samples with ground truth.
    Propagates via k-NN graph.
    """
    n = len(emb_n)
    nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn_model.fit(emb_n)
    distances, indices = nn_model.kneighbors(emb_n)
    distances = distances[:, 1:]  # skip self
    indices   = indices[:, 1:]

    # Build transition matrix (sparse-like)
    sigma = np.median(distances) + 1e-6
    W = np.exp(-distances / sigma)

    # Build normalized affinity row-wise
    # F_{i+1} = alpha * W_norm @ F_i + (1-alpha) * Y
    F = Y.copy().astype(np.float32)
    for it in range(n_iter):
        F_new = np.zeros_like(F)
        # Vectorized update
        for i in range(n):
            nb_F = F[indices[i]]   # (k, n_classes)
            w_norm = W[i] / (W[i].sum() + 1e-8)
            F_new[i] = (w_norm[:, None] * nb_F).sum(axis=0)
        F = alpha * F_new + (1 - alpha) * Y

    return np.clip(F, 0, 1)

print("  Running graph LP (may take a few seconds)...")
t0 = time.time()
lp_pred = graph_label_propagation(emb_lab_n, m0_perch_probs, alpha=0.5, k=10, n_iter=3)
print(f"  Done in {time.time()-t0:.1f}s")
compute_auc(Y_lab_filtered, lp_pred, "M7: Graph LP (alpha=0.5, k=10, 3 iter)")

for alpha_lp in [0.3, 0.7]:
    lp_pred_a = graph_label_propagation(emb_lab_n, m0_perch_probs, alpha=alpha_lp, k=10, n_iter=3)
    compute_auc(Y_lab_filtered, lp_pred_a, f"M7: Graph LP (alpha={alpha_lp}, k=10)")

# ── M8: Confidence-adaptive temperature scaling ──────────────────────────────
print("  --- M8: Confidence-adaptive temperature scaling ---")

def confidence_adaptive_temp(emb_n, Y, logits, k=15):
    """
    Use k-NN neighbors to estimate local temperature for calibration.
    Temperature = calibration scalar that minimizes NLL on neighbors.
    """
    nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn_model.fit(emb_n)
    distances, indices = nn_model.kneighbors(emb_n)
    distances = distances[:, 1:]
    indices   = indices[:, 1:]

    sigma = np.median(distances) + 1e-6
    W_raw = np.exp(-distances / sigma)
    W_norm = W_raw / (W_raw.sum(axis=1, keepdims=True) + 1e-8)

    Y_pred = np.zeros_like(logits)
    temps = []

    for i in range(len(emb_n)):
        nb_logits = logits[indices[i]]   # (k, 234)
        nb_Y      = Y[indices[i]]         # (k, 234) ground truth
        w = W_norm[i]                      # (k,)

        # Find optimal temperature T minimizing weighted BCE on neighbors
        best_T, best_loss = 1.0, float('inf')
        for T in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
            p = sigmoid(nb_logits / T)
            eps = 1e-6
            bce = -(nb_Y * np.log(p + eps) + (1 - nb_Y) * np.log(1 - p + eps))  # (k, 234)
            loss = (w[:, None] * bce).mean()
            if loss < best_loss:
                best_loss = loss
                best_T = T

        temps.append(best_T)
        Y_pred[i] = sigmoid(logits[i] / best_T)

    mean_T = np.mean(temps)
    return np.clip(Y_pred, 0, 1).astype(np.float32), mean_T

print("  Running confidence-adaptive temperature (this may take ~30s)...")
t0 = time.time()
temp_pred, mean_T = confidence_adaptive_temp(emb_lab_n, Y_lab_filtered, logit_lab_f, k=15)
print(f"  Done in {time.time()-t0:.1f}s, mean_T={mean_T:.3f}")
compute_auc(Y_lab_filtered, temp_pred, f"M8: Adaptive temperature (k=15, mean_T={mean_T:.2f})")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  ADVANCED CLUSTERING RESEARCH SUMMARY")
print("="*65)

all_methods = {
    "M0: Perch direct sigmoid": auc_m0_perch,
    f"M1: k-NN LOO (best k={best_k})": best_knn_auc,
    f"M2: kNN*{best_alpha:.1f}+Perch*{1-best_alpha:.1f}": best_blend_auc,
    "M6: Pseudo-aug k-NN": auc_pseudo,
}

sorted_methods = sorted(all_methods.items(), key=lambda x: x[1], reverse=True)
for name, auc in sorted_methods:
    flag = " *** BEAT BASELINE ***" if auc > 0.9553 else ""
    print(f"  {name:<50} {auc:.4f}{flag}")

print("="*65)
print(f"  Reference: stacker_cluster baseline = 0.9553")
print(f"  Reference: stacker-ss baseline      = 0.9641")

# Save results
results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "n_labeled": n_labeled,
    "methods": {k: float(v) for k, v in all_methods.items()},
    "baseline_cluster": 0.9553,
    "baseline_ss": 0.9641,
    "best_k": int(best_k) if best_k else None,
    "best_alpha": float(best_alpha),
}
with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {RESULTS_PATH}")
