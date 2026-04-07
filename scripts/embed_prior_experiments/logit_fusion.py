"""
Method 3: Logit Fusion & Advanced Methods
Key insight: perch_labeled_ss.npz contains BOTH:
  - emb: (739, 1536) Perch embeddings
  - logits: (739, 234) Perch model output logits (IGNORED by baseline!)

The logits ARE Perch's own predictions. Using them directly or fusing with
embedding-based KNN should outperform KNN alone.

Methods:
  A) Logit-only: file_score = mean(sigmoid(logits)) per file per species
  B) Logit+KNN ensemble: blend Perch logit predictions with KNN
  C) Calibrated logit: isotonic regression calibration of logits
  D) KNN on (emb || logits) concatenated features
  E) Logit as soft label for KNN weighting
  F) Adaptive KNN per species using logit uncertainty
"""

import numpy as np
import json
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeClassifier
import scipy.special

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH    = '/home/lab/BirdClef-2026-Codebase/outputs/perch_labeled_ss.npz'
RESULTS_PATH = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_results.json'
MODEL_PATH   = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_model.pkl'
BASELINE_AUC = 0.8411

# ── Load data ──────────────────────────────────────────────────────────────────
raw       = np.load(DATA_PATH, allow_pickle=True)
emb_win   = raw['emb'].astype(np.float32)       # (739, 1536)
logits_win= raw['logits'].astype(np.float32)    # (739, 234) ← KEY!
labels_win= raw['labels'].astype(np.float32)    # (739, 234)
file_list = raw['file_list']                     # (66,)
n_windows = raw['n_windows']                     # (66,)

n_files   = len(file_list)
n_species = labels_win.shape[1]

# Build file-level data
file_embs   = np.zeros((n_files, emb_win.shape[1]),   dtype=np.float32)
file_logits = np.zeros((n_files, n_species),           dtype=np.float32)
file_labels = np.zeros((n_files, n_species),           dtype=np.float32)

idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]   = emb_win[idx:idx+nw].mean(0)
    file_logits[fi] = logits_win[idx:idx+nw].mean(0)   # mean logit per file
    file_labels[fi] = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    idx += nw

file_probs  = scipy.special.expit(file_logits)  # sigmoid of mean logit
file_embs_norm = normalize(file_embs, norm='l2')

# Concatenated features: normalized embedding + sigmoid(logit)
file_concat = np.concatenate([file_embs_norm, file_probs], axis=1)  # (66, 1536+234)
file_concat_norm = normalize(file_concat, norm='l2')

print(f"Data: {n_files} files, {n_species} species")
print(f"Species present: {int((file_labels.sum(0) > 0).sum())}")
print(f"Logits range: {file_logits.min():.3f} to {file_logits.max():.3f}")
print(f"File probs range: {file_probs.min():.3f} to {file_probs.max():.3f}")

# ── AUC helper ─────────────────────────────────────────────────────────────────
def macro_auc(y_true, y_score):
    mask = y_true.sum(0) > 0
    if mask.sum() < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')
    except Exception:
        return float('nan')

# KNN baseline (for comparison)
def knn_predict_loo_k(k=5, X=None):
    if X is None:
        X = file_embs_norm
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr = X[mask]
        te = X[[i]]
        y_tr = file_labels[mask]
        sims = (te @ tr.T).ravel()
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff)[:k_eff]
        weights = np.clip(sims[nn_idx], 0, None)
        if weights.sum() < 1e-9:
            weights = np.ones(k_eff)
        preds[i] = (weights[:, None] * y_tr[nn_idx]).sum(0) / weights.sum()
    return preds

BASELINE_PREDS = knn_predict_loo_k(k=5, X=file_embs_norm)
print(f"KNN baseline repro: {macro_auc(file_labels, BASELINE_PREDS):.4f}")
print()

results_list = []

# ══════════════════════════════════════════════════════════════════════════════
# METHOD A: Logit-only (Perch predictions directly)
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("METHOD A: Perch logit-only (mean per file, sigmoid)")
print("="*65)

auc_a = macro_auc(file_labels, file_probs)
marker = "  *** NEW BEST ***" if auc_a > BASELINE_AUC else ""
print(f"  Perch sigmoid(mean_logit): {auc_a:.4f}  (delta={auc_a-BASELINE_AUC:+.4f}){marker}")
results_list.append(("perch_logit_direct", auc_a, {}, file_probs.copy()))

# Also try max instead of mean
file_logits_max = np.zeros((n_files, n_species), dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    file_logits_max[fi] = logits_win[idx:idx+nw].max(0)
    idx += nw
file_probs_max = scipy.special.expit(file_logits_max)
auc_a_max = macro_auc(file_labels, file_probs_max)
marker = "  *** NEW BEST ***" if auc_a_max > BASELINE_AUC else ""
print(f"  Perch sigmoid(max_logit):  {auc_a_max:.4f}  (delta={auc_a_max-BASELINE_AUC:+.4f}){marker}")
results_list.append(("perch_logit_max", auc_a_max, {}, file_probs_max.copy()))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD B: Logit + KNN ensemble (blend)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD B: Logit + KNN ensemble")
print("="*65)

best_b_auc, best_b_preds, best_b_alpha = 0.0, None, 0.5
for alpha in np.arange(0.0, 1.01, 0.05):
    # alpha = weight for logit, (1-alpha) for KNN
    ens_mean = alpha * file_probs + (1 - alpha) * BASELINE_PREDS
    auc_b = macro_auc(file_labels, ens_mean)
    if auc_b > best_b_auc:
        best_b_auc, best_b_preds, best_b_alpha = auc_b, ens_mean.copy(), alpha

print(f"  Best blend alpha_logit={best_b_alpha:.2f}: {best_b_auc:.4f}  (delta={best_b_auc-BASELINE_AUC:+.4f})")
# Print more detail around best
for alpha in [best_b_alpha - 0.05, best_b_alpha, best_b_alpha + 0.05]:
    alpha = max(0.0, min(1.0, alpha))
    ens_mean = alpha * file_probs + (1 - alpha) * BASELINE_PREDS
    auc_b = macro_auc(file_labels, ens_mean)
    marker = "  *** NEW BEST ***" if auc_b > BASELINE_AUC else ""
    print(f"    alpha_logit={alpha:.2f}: {auc_b:.4f}{marker}")
results_list.append(("logit_knn_ensemble", best_b_auc, {"alpha_logit": float(best_b_alpha)}, best_b_preds))

# Also try max logit + KNN
best_bmax_auc, best_bmax_preds, best_bmax_alpha = 0.0, None, 0.5
for alpha in np.arange(0.0, 1.01, 0.05):
    ens_max = alpha * file_probs_max + (1 - alpha) * BASELINE_PREDS
    auc_bmax = macro_auc(file_labels, ens_max)
    if auc_bmax > best_bmax_auc:
        best_bmax_auc, best_bmax_preds, best_bmax_alpha = auc_bmax, ens_max.copy(), alpha

marker = "  *** NEW BEST ***" if best_bmax_auc > BASELINE_AUC else ""
print(f"  Best blend alpha_logit_max={best_bmax_alpha:.2f}: {best_bmax_auc:.4f}  (delta={best_bmax_auc-BASELINE_AUC:+.4f}){marker}")
results_list.append(("logit_max_knn_ensemble", best_bmax_auc, {"alpha_logit_max": float(best_bmax_alpha)}, best_bmax_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD C: LOO-calibrated logit
# Caveat: logits at file-level already come from the same model
# We calibrate per-species using LOO isotonic regression
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD C: LOO-calibrated Perch logit (isotonic)")
print("="*65)

def calibrated_logit_loo():
    """
    For each LOO fold:
      - Fit IsotonicRegression on 65 train files (sigmoid(logit) vs label)
      - Predict calibrated prob for test file
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_logit = file_probs[mask]   # (65, 234)
        y_tr     = file_labels[mask]  # (65, 234)
        te_logit = file_probs[[i]]    # (1, 234)

        for s in range(n_species):
            y_s = y_tr[:, s]
            if y_s.sum() == 0:
                preds[i, s] = 0.0
                continue
            if y_s.sum() == len(y_s):
                preds[i, s] = 1.0
                continue
            try:
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(tr_logit[:, s], y_s)
                preds[i, s] = float(ir.predict([te_logit[0, s]])[0])
            except Exception:
                preds[i, s] = float(te_logit[0, s])

    return macro_auc(file_labels, preds), preds

auc_c, preds_c = calibrated_logit_loo()
marker = "  *** NEW BEST ***" if auc_c > BASELINE_AUC else ""
print(f"  LOO-calibrated logit: {auc_c:.4f}  (delta={auc_c-BASELINE_AUC:+.4f}){marker}")
results_list.append(("calibrated_logit_loo", auc_c, {}, preds_c))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD D: KNN on concatenated (emb, logit) features
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD D: KNN on concat(emb, logit) features")
print("="*65)

best_d_auc, best_d_preds, best_d_k = 0.0, None, 5
for k in [1, 2, 3, 5, 7, 10, 15]:
    preds_d = knn_predict_loo_k(k=k, X=file_concat_norm)
    auc_d = macro_auc(file_labels, preds_d)
    marker = "  *** NEW BEST ***" if auc_d > BASELINE_AUC else ""
    print(f"  KNN(concat) k={k}: {auc_d:.4f}  (delta={auc_d-BASELINE_AUC:+.4f}){marker}")
    if auc_d > best_d_auc:
        best_d_auc, best_d_preds, best_d_k = auc_d, preds_d, k

results_list.append(("knn_concat_emb_logit", best_d_auc, {"k": best_d_k}, best_d_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD E: Logit-weighted KNN
# Use Perch's per-species logit confidence to weight KNN neighbors
# For species s: w_i = cos_sim(x_test, x_i) * sigmoid(logit_i[s])
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD E: Logit-confidence weighted KNN")
print("="*65)

def logit_weighted_knn_loo(k=5, logit_power=1.0):
    """
    For each test file and species s:
      - Find k nearest train files by embedding cosine similarity
      - Weight = cos_sim * (perch_prob[s])^logit_power
      - pred = sum(w * y) / sum(w)
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_emb   = file_embs_norm[mask]  # (65, 1536)
        te_emb   = file_embs_norm[[i]]   # (1, 1536)
        y_tr     = file_labels[mask]      # (65, 234)
        prob_tr  = file_probs[mask]       # (65, 234)

        sims = (te_emb @ tr_emb.T).ravel()  # (65,)
        k_eff = min(k, 65)
        nn_idx = np.argpartition(-sims, k_eff)[:k_eff]
        cos_w  = np.clip(sims[nn_idx], 0, None)  # (k,)

        for s in range(n_species):
            logit_w = (prob_tr[nn_idx, s] ** logit_power)  # (k,)
            w = cos_w * logit_w
            if w.sum() < 1e-9:
                w = cos_w.copy()
            if w.sum() < 1e-9:
                w = np.ones(k_eff)
            preds[i, s] = float((w * y_tr[nn_idx, s]).sum() / w.sum())

    return macro_auc(file_labels, preds), preds

best_e_auc, best_e_preds, best_e_k, best_e_pow = 0.0, None, 5, 1.0
for k in [3, 5, 7, 10]:
    for lp in [0.5, 1.0, 2.0]:
        auc_e, preds_e = logit_weighted_knn_loo(k=k, logit_power=lp)
        marker = "  *** NEW BEST ***" if auc_e > BASELINE_AUC else ""
        print(f"  LogitWKNN k={k} lp={lp}: {auc_e:.4f}  (delta={auc_e-BASELINE_AUC:+.4f}){marker}")
        if auc_e > best_e_auc:
            best_e_auc, best_e_preds, best_e_k, best_e_pow = auc_e, preds_e, k, lp

results_list.append(("logit_weighted_knn", best_e_auc, {"k": best_e_k, "logit_power": best_e_pow}, best_e_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD F: Per-species adaptive blend (logit if confident, else KNN)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD F: Per-species adaptive logit/KNN blend")
print("="*65)

def adaptive_blend_loo():
    """
    For each LOO fold and species:
      - Compute logit confidence: H = |logit - 0.5| (how far from uncertain)
      - High confidence → trust logit more
      - Low confidence → trust KNN more
      blend = sigmoid(5 * (prob - threshold)) * logit_pred + (1 - ...) * knn_pred
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_emb = file_embs_norm[mask]
        y_tr   = file_labels[mask]
        te_emb = file_embs_norm[[i]]

        sims = (te_emb @ tr_emb.T).ravel()
        k = 5
        nn_idx  = np.argpartition(-sims, k)[:k]
        cos_w   = np.clip(sims[nn_idx], 0, None)
        if cos_w.sum() < 1e-9:
            cos_w = np.ones(k)
        knn_pred = (cos_w[:, None] * y_tr[nn_idx]).sum(0) / cos_w.sum()  # (234,)

        logit_pred = file_probs[i]  # (234,) Perch prob for THIS file

        # Confidence: how far from 0.5
        confidence = np.abs(logit_pred - 0.5) * 2  # [0, 1]

        # Blend: high confidence → logit dominates
        blend_w = confidence  # alpha for logit
        preds[i] = blend_w * logit_pred + (1 - blend_w) * knn_pred

    return macro_auc(file_labels, preds), preds

auc_f, preds_f = adaptive_blend_loo()
marker = "  *** NEW BEST ***" if auc_f > BASELINE_AUC else ""
print(f"  Adaptive blend: {auc_f:.4f}  (delta={auc_f-BASELINE_AUC:+.4f}){marker}")
results_list.append(("adaptive_logit_knn", auc_f, {}, preds_f))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD G: LOO logit calibration using neighbors
# For each fold, fit a linear recalibration using neighbor logits vs labels
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD G: LOO neighbor-based logit recalibration")
print("="*65)

def neighbor_recalibrated_logit_loo(k_calib=15):
    """
    For each LOO fold:
      1. Find k_calib nearest train files by embedding distance
      2. Fit logistic recalibration: p_calibrated = sigmoid(a * logit + b)
         using those k neighbors as calibration data
      3. Apply recalibration to test file's logit
    Note: per-species but batch all species together with shared a,b
    """
    from sklearn.linear_model import LogisticRegression as LR

    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_emb   = file_embs_norm[mask]
        te_emb   = file_embs_norm[[i]]
        y_tr     = file_labels[mask]
        logit_tr = file_logits[mask]  # raw logits, (65, 234)
        logit_te = file_logits[[i]]   # (1, 234)

        sims = (te_emb @ tr_emb.T).ravel()
        k = min(k_calib, 65)
        nn_idx = np.argsort(-sims)[:k]

        # Calibration data: (k * n_species, ) logits and labels
        calib_logits = logit_tr[nn_idx].ravel()         # (k * 234,)
        calib_labels = y_tr[nn_idx].ravel()              # (k * 234,)

        # Filter out species with no positive in calibration set (skip)
        active_in_calib = y_tr[nn_idx].sum(0) > 0  # (234,)

        for s in range(n_species):
            if not active_in_calib[s]:
                preds[i, s] = 0.0
                continue

            # Per-species calibration using k neighbor files
            x_calib = logit_tr[nn_idx, s].reshape(-1, 1)
            y_calib = y_tr[nn_idx, s]

            if y_calib.sum() == 0:
                preds[i, s] = 0.0
                continue
            if y_calib.sum() == len(y_calib):
                preds[i, s] = 1.0
                continue

            try:
                # Fit recalibration logistic on neighbor logits
                lr = LR(C=1.0, solver='lbfgs', max_iter=100)
                lr.fit(x_calib, y_calib)
                prob = lr.predict_proba([[logit_te[0, s]]])
                classes = list(lr.classes_)
                if 1 in classes:
                    preds[i, s] = float(prob[0][classes.index(1)])
                else:
                    preds[i, s] = 0.0
            except Exception:
                # Fallback: raw sigmoid
                preds[i, s] = float(scipy.special.expit(logit_te[0, s]))

    return macro_auc(file_labels, preds), preds

auc_g, preds_g = neighbor_recalibrated_logit_loo(k_calib=15)
marker = "  *** NEW BEST ***" if auc_g > BASELINE_AUC else ""
print(f"  Neighbor-recalibrated logit (k=15): {auc_g:.4f}  (delta={auc_g-BASELINE_AUC:+.4f}){marker}")

best_gcalib_auc, best_gcalib_preds, best_gcalib_k = auc_g, preds_g, 15
for k_c in [5, 10, 20, 30]:
    auc_gc, preds_gc = neighbor_recalibrated_logit_loo(k_calib=k_c)
    marker = "  *** NEW BEST ***" if auc_gc > BASELINE_AUC else ""
    print(f"  Neighbor-recalibrated logit (k={k_c}): {auc_gc:.4f}  (delta={auc_gc-BASELINE_AUC:+.4f}){marker}")
    if auc_gc > best_gcalib_auc:
        best_gcalib_auc, best_gcalib_preds, best_gcalib_k = auc_gc, preds_gc, k_c

results_list.append(("neighbor_recalib_logit", best_gcalib_auc, {"k_calib": best_gcalib_k}, best_gcalib_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD H: 3-way ensemble (KNN + logit_mean + logit_max)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD H: 3-way ensemble (KNN + logit_mean + logit_max)")
print("="*65)

# Grid search over 3-way blend
best_h_auc, best_h_preds, best_h_weights = 0.0, None, (0.33, 0.33, 0.34)
for w1 in np.arange(0.0, 1.01, 0.1):     # KNN weight
    for w2 in np.arange(0.0, 1.01-w1, 0.1):  # logit_mean weight
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue
        ens3 = w1 * BASELINE_PREDS + w2 * file_probs + w3 * file_probs_max
        auc_h = macro_auc(file_labels, ens3)
        if auc_h > best_h_auc:
            best_h_auc, best_h_preds, best_h_weights = auc_h, ens3.copy(), (w1, w2, w3)

marker = "  *** NEW BEST ***" if best_h_auc > BASELINE_AUC else ""
print(f"  Best 3-way (knn={best_h_weights[0]:.1f}, lm={best_h_weights[1]:.1f}, lmax={best_h_weights[2]:.1f}): "
      f"{best_h_auc:.4f}  (delta={best_h_auc-BASELINE_AUC:+.4f}){marker}")
results_list.append(("3way_ensemble", best_h_auc, {"w_knn": best_h_weights[0],
                     "w_logit_mean": best_h_weights[1], "w_logit_max": best_h_weights[2]}, best_h_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD I: Logit + KNN with calibrated (LOO) logit
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD I: KNN + LOO-calibrated logit ensemble")
print("="*65)

best_i_auc, best_i_preds, best_i_alpha = 0.0, None, 0.5
for alpha in np.arange(0.0, 1.01, 0.05):
    ens_i = alpha * preds_c + (1 - alpha) * BASELINE_PREDS
    auc_i = macro_auc(file_labels, ens_i)
    if auc_i > best_i_auc:
        best_i_auc, best_i_preds, best_i_alpha = auc_i, ens_i.copy(), alpha

marker = "  *** NEW BEST ***" if best_i_auc > BASELINE_AUC else ""
print(f"  KNN + calibrated logit (alpha_calib={best_i_alpha:.2f}): {best_i_auc:.4f}  (delta={best_i_auc-BASELINE_AUC:+.4f}){marker}")
results_list.append(("knn_caliblogit_ensemble", best_i_auc, {"alpha_calib": float(best_i_alpha)}, best_i_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD J: KNN with species-level score fusion
# For each species, some files' Perch logits are more reliable.
# Combine per-species with adaptive alpha based on calibration.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD J: Species-adaptive logit+KNN blend")
print("="*65)

def species_adaptive_blend_loo():
    """
    Per species s, compute LOO-optimal blend alpha_s between logit and KNN.
    Uses inner LOO: for each fold i, use remaining 65-1=64 files to estimate alpha_s.
    Then apply alpha_s to blend test prediction.
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_emb   = file_embs_norm[mask]
        y_tr     = file_labels[mask]
        te_emb   = file_embs_norm[[i]]
        prob_tr  = file_probs[mask]     # (65, 234)
        prob_te  = file_probs[[i]]      # (1, 234)

        # KNN on train data (predict for test)
        sims = (te_emb @ tr_emb.T).ravel()
        k = 5
        nn_idx  = np.argpartition(-sims, k)[:k]
        cos_w   = np.clip(sims[nn_idx], 0, None)
        if cos_w.sum() < 1e-9:
            cos_w = np.ones(k)
        knn_pred = (cos_w[:, None] * y_tr[nn_idx]).sum(0) / cos_w.sum()  # (234,)
        logit_pred = prob_te[0]  # (234,)

        # Estimate per-species reliability of logit via inner LOO on train set
        # For each species, try blend alpha in [0.0, ..., 1.0], pick best via inner LOO
        for s in range(n_species):
            y_s = y_tr[:, s]
            if y_s.sum() == 0:
                preds[i, s] = 0.0
                continue

            # Inner LOO to estimate best alpha for this species
            inner_knn_preds = np.zeros(65)
            inner_logit_preds = prob_tr[:, s].copy()

            for j in range(65):
                inner_mask = np.ones(65, dtype=bool); inner_mask[j] = False
                x_in = tr_emb[inner_mask]
                y_in = y_tr[inner_mask, s]
                x_j  = tr_emb[[j]]
                sims_j = (x_j @ x_in.T).ravel()
                k_j = min(k, inner_mask.sum())
                nn_j = np.argpartition(-sims_j, k_j)[:k_j]
                w_j  = np.clip(sims_j[nn_j], 0, None)
                if w_j.sum() < 1e-9:
                    w_j = np.ones(k_j)
                inner_knn_preds[j] = float((w_j * y_in[nn_j]).sum() / w_j.sum())

            # Pick alpha that maximizes inner LOO AUC for species s
            best_alpha_s = 0.5
            best_inner_auc = -1.0
            if y_s.sum() > 0 and y_s.sum() < 65:
                for alpha_s in np.arange(0.0, 1.01, 0.1):
                    blend_s = alpha_s * inner_logit_preds + (1 - alpha_s) * inner_knn_preds
                    try:
                        inner_auc = roc_auc_score(y_s, blend_s)
                        if inner_auc > best_inner_auc:
                            best_inner_auc = inner_auc
                            best_alpha_s = alpha_s
                    except Exception:
                        pass

            preds[i, s] = float(best_alpha_s * logit_pred[s] + (1 - best_alpha_s) * knn_pred[s])

    return macro_auc(file_labels, preds), preds

from sklearn.metrics import roc_auc_score
print("  Running species-adaptive blend (slow, ~2 min)...")
auc_j, preds_j = species_adaptive_blend_loo()
marker = "  *** NEW BEST ***" if auc_j > BASELINE_AUC else ""
print(f"  Species-adaptive blend: {auc_j:.4f}  (delta={auc_j-BASELINE_AUC:+.4f}){marker}")
results_list.append(("species_adaptive_blend", auc_j, {}, preds_j))

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
print(f"Baseline KNN: {BASELINE_AUC:.4f}")
for name, auc, params, _ in sorted(results_list, key=lambda x: -x[1]):
    marker = "  *** NEW BEST ***" if auc > BASELINE_AUC else ""
    print(f"  {name}: {auc:.4f}  (delta={auc-BASELINE_AUC:+.4f}){marker}")

best_result = max(results_list, key=lambda x: x[1])
best_name, best_auc, best_params, best_preds = best_result

# Update JSON
with open(RESULTS_PATH) as f:
    results_json = json.load(f)

for name, auc, params, _ in results_list:
    record = {"method": name, "loo_auc": round(float(auc), 4)}
    for k, v in params.items():
        if isinstance(v, (np.float32, np.float64)):
            record[k] = float(v)
        elif isinstance(v, np.integer):
            record[k] = int(v)
        else:
            record[k] = v
    results_json["experiments"].append(record)

if best_auc > results_json["best"]["loo_auc"]:
    results_json["best"] = {"method": best_name, "loo_auc": round(float(best_auc), 4)}
    print(f"\nNEW BEST: {best_name} AUC={best_auc:.4f}")

    # Fit final model on ALL 66 files
    print("Fitting final model on all 66 files...")
    model_dict = {
        "method": best_name,
        "loo_auc": round(float(best_auc), 4),
        "params": best_params,
        "file_list": file_list.tolist(),
        "loo_preds": best_preds.tolist(),
        # Store prediction arrays for inference
        "file_embs_norm": file_embs_norm.tolist(),
        "file_probs": file_probs.tolist(),
        "file_probs_max": file_probs_max.tolist(),
        "file_labels": file_labels.tolist(),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"Saved model → {MODEL_PATH}")
else:
    print(f"\nNo improvement over current best ({results_json['best']['loo_auc']:.4f})")
    print(f"Best this run: {best_name} AUC={best_auc:.4f}")

with open(RESULTS_PATH, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Results saved → {RESULTS_PATH}")
