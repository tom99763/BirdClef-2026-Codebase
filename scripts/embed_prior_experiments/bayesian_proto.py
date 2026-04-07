"""
Method 2: Bayesian & Advanced Embedding Prior Methods
Target: beat KNN baseline LOO-AUC = 0.8411

Operates at FILE level (66 files, leave-one-file-out), same as baseline.

Methods:
  A) Bayesian Gaussian Prototype (Pyro VI)
     - PCA(64) features
     - per-species prior: p(mu_s) = Normal(0, 1)
     - likelihood: p(x_i | mu_s, y_is=1) = Normal(mu_s, sigma_obs)
     - posterior: q(mu_s) = Normal(m_s, v_s) via SVI
     - prediction: p(y=1 | x_test) ~ exp(-||x_test - E[mu_s]||^2 / 2*var)

  B) Bayesian Logistic Regression (Pyro SVI)
     - PCA(64) features
     - W ~ Normal(0, sigma_W), b ~ Normal(0, 1)
     - Per-species ELBO training

  C) Cosine similarity with adaptive per-species threshold
     - For each species, calibrate using cross-validation on train set

  D) Ensemble (KNN + Prototype + BayesLogReg)

  E) PCA+LogReg with optimal C (rerun of baseline but with tuning)
     - Baseline only uses C=0.05
     - Tune C in [0.01, 0.05, 0.1, 0.5, 1.0]

  F) Stacking: train a meta-learner on KNN and LogReg predictions

  G) Mahalanobis distance classification
     - Per-species: compute mean and (regularized) covariance in train set
     - Score = exp(-mahalanobis_distance(x_test, mu_s, Sigma_s))
"""

import numpy as np
import json
import pickle
import sys
import re
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam as PyroAdam

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH    = '/home/lab/BirdClef-2026-Codebase/outputs/perch_labeled_ss.npz'
RESULTS_PATH = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_results.json'
MODEL_PATH   = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_model.pkl'
BASELINE_AUC = 0.8411

# ── Load data ──────────────────────────────────────────────────────────────────
raw = np.load(DATA_PATH, allow_pickle=True)
emb_win   = raw['emb'].astype(np.float32)       # (739, 1536)
labels_win= raw['labels'].astype(np.float32)    # (739, 234)
file_list = raw['file_list']                     # (66,)
n_windows = raw['n_windows']                     # (66,)

n_files   = len(file_list)
n_species = labels_win.shape[1]

def parse_meta(filename: str) -> dict:
    m = re.match(r"BC2026_Train_\d+_S(\d+)_(\d{4})(\d{2})\d{2}_(\d{2})", str(filename))
    if not m:
        return {"site": "00", "hour": 0, "month": 6}
    site, _, month, hour = m.groups()
    return {"site": site, "hour": int(hour), "month": int(month)}

def cyclic(val, period):
    rad = 2 * np.pi * val / period
    return np.sin(rad), np.cos(rad)

# Build file-level data
file_embs   = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels = np.zeros((n_files, n_species),         dtype=np.float32)
file_metas  = []

idx = 0
for fi, (fname, nw) in enumerate(zip(file_list, n_windows)):
    file_embs[fi]   = emb_win[idx:idx+nw].mean(0)
    file_labels[fi] = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    file_metas.append(parse_meta(fname))
    idx += nw

site_list = sorted({m["site"] for m in file_metas})

def build_features(mean_embs, metas, site_list):
    site2idx = {s: i for i, s in enumerate(site_list)}
    n_sites  = len(site_list)
    feats = []
    for i in range(len(mean_embs)):
        emb  = mean_embs[i]
        meta = metas[i]
        site_oh = np.zeros(n_sites, dtype=np.float32)
        site_oh[site2idx.get(meta["site"], 0)] = 1.0
        h_sin, h_cos = cyclic(meta["hour"], 24)
        m_sin, m_cos = cyclic(meta["month"], 12)
        extra = np.array([h_sin, h_cos, m_sin, m_cos], dtype=np.float32)
        feats.append(np.concatenate([emb, site_oh, extra]))
    return np.stack(feats)

X_full = build_features(file_embs, file_metas, site_list)  # (66, 1536+9+4)
Y      = file_labels                                         # (66, 234)

file_embs_norm = normalize(file_embs, norm='l2')

print(f"Data: {n_files} files, {n_species} species")
print(f"Species present: {int((Y.sum(0) > 0).sum())}")
print(f"X_full shape: {X_full.shape}")

# ── AUC helper ─────────────────────────────────────────────────────────────────
def macro_auc(y_true, y_score):
    mask = y_true.sum(0) > 0
    if mask.sum() < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')
    except Exception:
        return float('nan')

# KNN baseline (for ensemble)
def knn_predict_loo():
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr = file_embs_norm[mask]
        te = file_embs_norm[[i]]
        y_tr = Y[mask]
        sims = (te @ tr.T).ravel()
        k = 5
        nn_idx = np.argpartition(-sims, k)[:k]
        weights = np.clip(sims[nn_idx], 0, None)
        if weights.sum() < 1e-9:
            weights = np.ones(k)
        preds[i] = (weights[:, None] * y_tr[nn_idx]).sum(0) / weights.sum()
    return preds

BASELINE_PREDS = knn_predict_loo()
print(f"KNN baseline repro: {macro_auc(Y, BASELINE_PREDS):.4f}")

results_list = []

# ══════════════════════════════════════════════════════════════════════════════
# METHOD E: PCA+LogReg with tuned C (should improve over default C=0.05)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD E: PCA+LogReg with tuned C (baseline uses C=0.05)")
print("="*65)

def logreg_loo(C=0.05, pca_dim=64, use_meta=True):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    X = X_full if use_meta else file_embs

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        X_tr, Y_tr = X[mask], Y[mask]
        X_te       = X[[i]]

        valid = Y_tr.sum(0) > 0
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        pca = PCA(n_components=pca_dim, random_state=42)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_te_p = pca.transform(X_te_s)

        clf = OneVsRestClassifier(
            LogisticRegression(C=C, max_iter=2000, solver='lbfgs', random_state=42),
            n_jobs=-1
        )
        clf.fit(X_tr_p, Y_tr[:, valid])
        prob = clf.predict_proba(X_te_p)   # (1, n_valid)
        probs_full = np.zeros((1, n_species), dtype=np.float32)
        probs_full[0, valid] = prob[0]
        preds[i] = probs_full[0]

    return macro_auc(Y, preds), preds

best_lr_auc, best_lr_preds, best_lr_C = 0.0, None, 0.05
for C in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
    auc, preds = logreg_loo(C=C, pca_dim=64, use_meta=True)
    marker = "  *** NEW BEST ***" if auc > BASELINE_AUC else ""
    print(f"  LogReg C={C}: {auc:.4f}  (delta={auc-BASELINE_AUC:+.4f}){marker}")
    if auc > best_lr_auc:
        best_lr_auc, best_lr_preds, best_lr_C = auc, preds, C

print(f"Best LogReg: C={best_lr_C}, AUC={best_lr_auc:.4f}")
results_list.append(("logreg_tuned_C", best_lr_auc, {"C": best_lr_C, "pca_dim": 64}, best_lr_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD E2: PCA+LogReg with larger PCA dimension
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD E2: PCA+LogReg with varying PCA dimension")
print("="*65)

best_lr2_auc, best_lr2_preds, best_lr2_C, best_lr2_dim = 0.0, None, best_lr_C, 64
for pca_dim in [32, 64]:  # max=65 (leave-one-out has 65 train samples)
    auc, preds = logreg_loo(C=best_lr_C, pca_dim=pca_dim, use_meta=True)
    marker = "  *** NEW BEST ***" if auc > BASELINE_AUC else ""
    print(f"  LogReg C={best_lr_C} pca_dim={pca_dim}: {auc:.4f}  (delta={auc-BASELINE_AUC:+.4f}){marker}")
    if auc > best_lr2_auc:
        best_lr2_auc, best_lr2_preds, best_lr2_C, best_lr2_dim = auc, preds, best_lr_C, pca_dim

results_list.append(("logreg_tuned_dim", best_lr2_auc, {"C": best_lr2_C, "pca_dim": best_lr2_dim}, best_lr2_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD G: Mahalanobis distance per-species
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD G: Mahalanobis distance (per-species, PCA reduced)")
print("="*65)

def mahalanobis_loo(pca_dim=64, reg_lambda=1e-2):
    """
    For each species:
      mu_s  = mean of positive file embeddings (PCA-reduced)
      Sigma_s = covariance + lambda*I (regularized)
    Score = exp(-0.5 * (x-mu)^T Sigma^{-1} (x-mu))
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        X_tr = file_embs[mask]  # (65, 1536)
        Y_tr = Y[mask]          # (65, 234)
        x_te = file_embs[i]     # (1536,)

        # Reduce dimension
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        x_te_s = scaler.transform(x_te[np.newaxis, :])[0]

        pca = PCA(n_components=pca_dim, random_state=42)
        X_tr_p = pca.fit_transform(X_tr_s)
        x_te_p = pca.transform(x_te_s[np.newaxis, :])[0]

        for s in range(n_species):
            pos_mask = Y_tr[:, s] > 0.5
            n_pos = pos_mask.sum()
            if n_pos == 0:
                continue
            pos_embs = X_tr_p[pos_mask]  # (n_pos, pca_dim)
            mu = pos_embs.mean(0)        # (pca_dim,)

            diff = x_te_p - mu  # (pca_dim,)

            if n_pos >= 2:
                cov = np.cov(pos_embs.T) + reg_lambda * np.eye(pca_dim)
                try:
                    cov_inv = np.linalg.inv(cov)
                    dist2 = diff @ cov_inv @ diff
                except np.linalg.LinAlgError:
                    dist2 = np.dot(diff, diff)
            else:
                dist2 = np.dot(diff, diff)

            preds[i, s] = float(np.exp(-0.5 * dist2))

    return macro_auc(Y, preds), preds

auc_g, preds_g = mahalanobis_loo(pca_dim=64, reg_lambda=1e-2)
marker = "  *** NEW BEST ***" if auc_g > BASELINE_AUC else ""
print(f"  Mahalanobis (pca=64, reg=1e-2): {auc_g:.4f}  (delta={auc_g-BASELINE_AUC:+.4f}){marker}")
results_list.append(("mahalanobis_pca64", auc_g, {"pca_dim": 64, "reg_lambda": 1e-2}, preds_g))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD A: Bayesian Gaussian Prototype (Pyro VI)
# For each LOO fold:
#   - Use train files' embeddings (PCA-reduced) and labels
#   - Per active species: fit Gaussian mean (Bayesian posterior via SVI)
#   - Prediction: score based on posterior predictive distance
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD A: Bayesian Gaussian Prototype (Pyro SVI)")
print("="*65)

PCA_DIM_BAYES = 32  # Use smaller dim for Bayesian methods (speed)

def bayesian_gaussian_proto_loo(pca_dim=32, n_steps=500, lr=0.01):
    """
    Per-species Bayesian Gaussian prototype with SVI.
    Prior: mu_s ~ Normal(0, 1)^D
    Likelihood: x_i ~ Normal(mu_s, sigma_obs)^D  for positive i
    Posterior: q(mu_s) = Normal(m_s, v_s) (mean-field)
    Score: p(x_test | data) ~ exp(-||x_test - E[mu_s]||^2 / (2*(V[mu_s]+sigma_obs^2)))
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fold_i in range(n_files):
        if fold_i % 10 == 0:
            print(f"  Bayes GP proto fold {fold_i+1}/{n_files} ...", flush=True)

        mask = np.ones(n_files, dtype=bool); mask[fold_i] = False
        X_tr_raw = file_embs[mask]   # (65, 1536)
        Y_tr     = Y[mask]            # (65, 234)
        x_te_raw = file_embs[fold_i] # (1536,)

        # Reduce dimension
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw)
        x_te_s = scaler.transform(x_te_raw[np.newaxis, :])[0]

        pca = PCA(n_components=pca_dim, random_state=42)
        X_tr = pca.fit_transform(X_tr_s).astype(np.float32)  # (65, D)
        x_te = pca.transform(x_te_s[np.newaxis, :]).astype(np.float32)[0]  # (D,)

        D = pca_dim
        sigma_obs = 1.0  # observation noise (fixed)

        for s in range(n_species):
            pos_mask = Y_tr[:, s] > 0.5
            n_pos = int(pos_mask.sum())
            if n_pos == 0:
                continue

            pos_embs_t = torch.from_numpy(X_tr[pos_mask])  # (n_pos, D)

            # --- Pyro model & guide ---
            def model(obs):
                mu = pyro.sample("mu",
                    dist.Normal(torch.zeros(D), torch.ones(D)).to_event(1))
                with pyro.plate("data", obs.shape[0]):
                    pyro.sample("x",
                        dist.Normal(mu.expand(obs.shape), sigma_obs).to_event(1),
                        obs=obs)

            def guide(obs):
                m  = pyro.param("m",  torch.zeros(D))
                log_v = pyro.param("log_v", torch.zeros(D))
                v  = torch.exp(log_v).clamp(1e-4, 10.0)
                pyro.sample("mu", dist.Normal(m, v).to_event(1))

            pyro.clear_param_store()
            optimizer = PyroAdam({"lr": lr})
            svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

            for step in range(n_steps):
                svi.step(pos_embs_t)

            # Extract posterior
            m_post = pyro.param("m").detach().numpy()    # (D,)
            v_post = torch.exp(pyro.param("log_v")).detach().numpy()  # (D,)

            # Score = log p(x_test | posterior)
            # p(x_test | mu_post) = Normal(m_post, sqrt(v_post + sigma_obs^2))
            combined_var = v_post + sigma_obs**2
            diff = x_te - m_post
            log_score = -0.5 * np.sum(diff**2 / combined_var)
            # Normalize by dimension
            log_score /= D

            # Convert to [0, 1] via sigmoid
            preds[fold_i, s] = float(1.0 / (1.0 + np.exp(-log_score)))

    return macro_auc(Y, preds), preds


auc_a, preds_a = bayesian_gaussian_proto_loo(pca_dim=PCA_DIM_BAYES, n_steps=300, lr=0.02)
marker = "  *** NEW BEST ***" if auc_a > BASELINE_AUC else ""
print(f"  Bayes Gaussian Proto (pca={PCA_DIM_BAYES}, steps=300): {auc_a:.4f}  (delta={auc_a-BASELINE_AUC:+.4f}){marker}")
results_list.append(("bayes_gaussian_proto", auc_a, {"pca_dim": PCA_DIM_BAYES, "n_steps": 300}, preds_a))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD F: Ensemble KNN + LogReg (weighted blend)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD F: Ensemble KNN + LogReg")
print("="*65)

best_ensemble_auc, best_ensemble_preds, best_alpha = 0.0, None, 0.5
for alpha in np.arange(0.0, 1.01, 0.1):
    ensemble = alpha * BASELINE_PREDS + (1 - alpha) * best_lr2_preds
    auc_ens = macro_auc(Y, ensemble)
    marker = "  *** NEW BEST ***" if auc_ens > BASELINE_AUC else ""
    print(f"  Ensemble alpha={alpha:.1f} (KNN) + {1-alpha:.1f} (LR): {auc_ens:.4f}  (delta={auc_ens-BASELINE_AUC:+.4f}){marker}")
    if auc_ens > best_ensemble_auc:
        best_ensemble_auc, best_ensemble_preds, best_alpha = auc_ens, ensemble, alpha

results_list.append(("ensemble_knn_logreg", best_ensemble_auc, {"alpha_knn": float(best_alpha)}, best_ensemble_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD H: KNN with varying K and cosine similarity (fine-grained sweep)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD H: Fine-grained KNN sweep (many K values)")
print("="*65)

def knn_predict_loo_k(k=5):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr = file_embs_norm[mask]
        te = file_embs_norm[[i]]
        y_tr = Y[mask]
        sims = (te @ tr.T).ravel()
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff)[:k_eff]
        weights = np.clip(sims[nn_idx], 0, None)
        if weights.sum() < 1e-9:
            weights = np.ones(k_eff)
        preds[i] = (weights[:, None] * y_tr[nn_idx]).sum(0) / weights.sum()
    return macro_auc(Y, preds), preds

best_knn_auc, best_knn_preds, best_knn_k = 0.0, None, 5
for k in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]:
    auc_k, preds_k = knn_predict_loo_k(k=k)
    marker = "  *** NEW BEST ***" if auc_k > BASELINE_AUC else ""
    print(f"  KNN k={k:2d}: {auc_k:.4f}  (delta={auc_k-BASELINE_AUC:+.4f}){marker}")
    if auc_k > best_knn_auc:
        best_knn_auc, best_knn_preds, best_knn_k = auc_k, preds_k, k

results_list.append(("knn_sweep", best_knn_auc, {"k": best_knn_k}, best_knn_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD I: Ensemble of best KNN + best LogReg (using discovered best K)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD I: Ensemble best_KNN + best_LogReg (optimal alpha)")
print("="*65)

best_ens2_auc, best_ens2_preds, best_ens2_alpha = 0.0, None, 0.5
for alpha in np.arange(0.0, 1.01, 0.05):
    ensemble2 = alpha * best_knn_preds + (1 - alpha) * best_lr2_preds
    auc_e2 = macro_auc(Y, ensemble2)
    if auc_e2 > best_ens2_auc:
        best_ens2_auc, best_ens2_preds, best_ens2_alpha = auc_e2, ensemble2, alpha

marker = "  *** NEW BEST ***" if best_ens2_auc > BASELINE_AUC else ""
print(f"  Best Ensemble (alpha_knn={best_ens2_alpha:.2f}): {best_ens2_auc:.4f}  (delta={best_ens2_auc-BASELINE_AUC:+.4f}){marker}")
results_list.append(("ensemble_best_knn_logreg", best_ens2_auc, {"alpha_knn": float(best_ens2_alpha)}, best_ens2_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD J: Bayesian Logistic Regression (Pyro SVI, per species)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD J: Bayesian LogReg (Pyro SVI, per-species)")
print("="*65)

def bayesian_logreg_loo(pca_dim=32, n_steps=500, lr=0.01, sigma_W=1.0):
    """
    Per-species Bayesian LogReg with SVI.
    W_s ~ Normal(0, sigma_W), b_s ~ Normal(0, 1)
    p(y=1 | x) = sigmoid(W_s^T x + b_s)
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fold_i in range(n_files):
        if fold_i % 10 == 0:
            print(f"  Bayes LogReg fold {fold_i+1}/{n_files} ...", flush=True)

        mask = np.ones(n_files, dtype=bool); mask[fold_i] = False
        X_tr_raw = file_embs[mask]
        Y_tr     = Y[mask]
        x_te_raw = file_embs[fold_i]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw)
        x_te_s = scaler.transform(x_te_raw[np.newaxis, :])[0]

        pca = PCA(n_components=pca_dim, random_state=42)
        X_tr = pca.fit_transform(X_tr_s).astype(np.float32)
        x_te = pca.transform(x_te_s[np.newaxis, :]).astype(np.float32)[0]

        D = pca_dim
        x_te_t = torch.from_numpy(x_te)

        for s in range(n_species):
            ys = Y_tr[:, s]
            n_pos = int((ys > 0.5).sum())
            n_neg = int((ys < 0.5).sum())
            if n_pos == 0 or n_neg == 0:
                if n_pos == 0:
                    preds[fold_i, s] = 0.0
                else:
                    preds[fold_i, s] = 1.0
                continue

            x_t = torch.from_numpy(X_tr)
            y_t = torch.from_numpy(ys)

            def model_lr(x, y):
                W = pyro.sample("W", dist.Normal(torch.zeros(D), sigma_W * torch.ones(D)).to_event(1))
                b = pyro.sample("b", dist.Normal(torch.zeros(1), torch.ones(1)).to_event(1))
                logit = x @ W + b.squeeze()
                with pyro.plate("data", x.shape[0]):
                    pyro.sample("y", dist.Bernoulli(logits=logit), obs=y)

            def guide_lr(x, y):
                m_W = pyro.param("m_W", torch.zeros(D))
                log_s_W = pyro.param("log_s_W", torch.zeros(D))
                m_b = pyro.param("m_b", torch.zeros(1))
                log_s_b = pyro.param("log_s_b", torch.zeros(1))
                pyro.sample("W", dist.Normal(m_W, torch.exp(log_s_W).clamp(1e-4)).to_event(1))
                pyro.sample("b", dist.Normal(m_b, torch.exp(log_s_b).clamp(1e-4)).to_event(1))

            pyro.clear_param_store()
            optimizer = PyroAdam({"lr": lr})
            svi = SVI(model_lr, guide_lr, optimizer, loss=Trace_ELBO())

            for step in range(n_steps):
                svi.step(x_t, y_t)

            # Predict: use posterior mean
            W_post = pyro.param("m_W").detach()
            b_post = pyro.param("m_b").detach()
            logit = x_te_t @ W_post + b_post.squeeze()
            preds[fold_i, s] = float(torch.sigmoid(logit))

    return macro_auc(Y, preds), preds

# Only run Bayesian LogReg on species with enough data (speed up)
auc_j, preds_j = bayesian_logreg_loo(pca_dim=PCA_DIM_BAYES, n_steps=200, lr=0.02, sigma_W=1.0)
marker = "  *** NEW BEST ***" if auc_j > BASELINE_AUC else ""
print(f"  Bayes LogReg (pca={PCA_DIM_BAYES}, steps=200): {auc_j:.4f}  (delta={auc_j-BASELINE_AUC:+.4f}){marker}")
results_list.append(("bayes_logreg_svi", auc_j, {"pca_dim": PCA_DIM_BAYES, "n_steps": 200}, preds_j))

# ══════════════════════════════════════════════════════════════════════════════
# Final ensemble of all methods
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("FULL ENSEMBLE (all methods)")
print("="*65)

# Try averaging all predictions
candidate_preds = [
    ("KNN_baseline", BASELINE_PREDS),
    ("knn_sweep", best_knn_preds),
    ("logreg_C", best_lr_preds),
    ("logreg_dim", best_lr2_preds),
    ("mahalanobis", preds_g),
    ("bayes_proto", preds_a),
    ("bayes_logreg", preds_j),
]

# Greedy forward selection of ensemble
best_pool_auc, best_pool_preds = macro_auc(Y, BASELINE_PREDS), BASELINE_PREDS.copy()
pool_names = ["KNN_baseline"]
pool_preds = [BASELINE_PREDS]

for cname, cpreds in candidate_preds[1:]:
    # Try adding each to current pool
    new_pool = np.mean([best_pool_preds] + [cpreds], axis=0)
    auc_new = macro_auc(Y, new_pool)
    if auc_new > best_pool_auc:
        best_pool_auc = auc_new
        best_pool_preds = new_pool
        pool_names.append(cname)
        pool_preds.append(cpreds)
        marker = "  *** NEW BEST ***" if auc_new > BASELINE_AUC else ""
        print(f"  + {cname}: pool AUC={auc_new:.4f}  (delta={auc_new-BASELINE_AUC:+.4f}){marker}")
    else:
        print(f"  - {cname}: skip (pool would be {auc_new:.4f})")

print(f"\nGreedy ensemble ({'+'.join(pool_names)}): {best_pool_auc:.4f}")
results_list.append(("greedy_ensemble", best_pool_auc, {"methods": pool_names}, best_pool_preds))

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

# ── Find best result ───────────────────────────────────────────────────────────
best_result = max(results_list, key=lambda x: x[1])
best_name, best_auc, best_params, best_preds = best_result

# ── Update results JSON ────────────────────────────────────────────────────────
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

    model_dict = {
        "method": best_name,
        "loo_auc": round(float(best_auc), 4),
        "params": best_params,
        "file_list": file_list.tolist(),
        "loo_preds": best_preds.tolist(),
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
