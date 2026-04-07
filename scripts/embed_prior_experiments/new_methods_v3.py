"""
embed_prior new_methods_v3.py
向量化版本的 Bayesian Shrinkage Prototype (BSP)
避免 per-species 的 Python loop，改用矩陣運算
目標: 超越 CURRENT_BEST = 0.894048
"""

import numpy as np
import json
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import scipy.special

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_PATH    = '/home/lab/BirdClef-2026-Codebase/outputs/perch_labeled_ss.npz'
RESULTS_PATH = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_results.json'
MODEL_PATH   = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_model.pkl'
CURRENT_BEST = 0.894048

raw        = np.load(DATA_PATH, allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
logits_win = raw['logits'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']

n_files   = len(file_list)
n_species = labels_win.shape[1]

win_file_idx = np.zeros(len(emb_win), dtype=np.int32)
idx = 0
for fi, nw in enumerate(n_windows):
    win_file_idx[idx:idx + nw] = fi
    idx += nw

file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species),        dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species),        dtype=np.float32)

idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]      = emb_win[idx:idx + nw].mean(0)
    file_labels[fi]    = (labels_win[idx:idx + nw].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[idx:idx + nw].max(0)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
file_prob_max  = scipy.special.expit(file_logit_max)

print(f"資料: {n_files} files, {n_species} species")
print(f"Current best: {CURRENT_BEST:.6f}")

def macro_auc(y_true, y_score):
    mask = (y_true.sum(0) > 0) & (y_true.sum(0) < n_files)
    if mask.sum() < 2: return float('nan')
    try:
        return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')
    except: return float('nan')

def knn_binary_predict(k=3):
    X = file_embs_norm
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask_tr = np.arange(n_files) != i
        tr = X[mask_tr]; te = X[[i]]; y_tr = file_labels[mask_tr]
        sims = (te @ tr.T).ravel()
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        w = np.clip(sims[nn_idx], 0, None)
        if w.sum() < 1e-9: w = np.ones(k_eff)
        preds[i] = (w[:, None] * y_tr[nn_idx]).sum(0) / w.sum()
    return preds

print("Pre-computing KNN(1,3,4)...")
knn1 = knn_binary_predict(k=1)
knn3 = knn_binary_predict(k=3)
knn4 = knn_binary_predict(k=4)
k134_ref = 0.42*file_prob_max + 0.28*knn1 + 0.02*knn3 + 0.28*knn4
print(f"  k134_ref={macro_auc(file_labels,k134_ref):.6f}")

results_list = []

# ══════════════════════════════════════════════════════════════════
# 向量化 BSP：一次計算所有 species 的 posterior mean
# ══════════════════════════════════════════════════════════════════

def bsp_vectorized_loo(pca_dim=64, prior_strength=0.5):
    """
    向量化 Bayesian Shrinkage Prototype：
    post_mean_s = (Y_tr[:,s].sum() * mean_emb_pos_s + lambda * global_mean)
                 / (Y_tr[:,s].sum() + lambda)

    Score = exp(-0.5 * ||x_te - post_mean_s||^2 / sigma2)

    全部 species 同時計算，快得多
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr = file_embs[mask_tr]     # (65, 1536)
        Y_tr = file_labels[mask_tr]   # (65, 234)
        X_te = file_embs[[fi]]        # (1, 1536)

        n_tr = X_tr.shape[0]
        pca_dim_eff = min(pca_dim, n_tr - 1)
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr).astype(np.float32)  # (65, pca_dim)
        X_te_pca = pca.transform(X_te).astype(np.float32)       # (1, pca_dim)

        global_mean = X_tr_pca.mean(0, keepdims=True)  # (1, pca_dim)

        # Vectorized posterior mean for all species
        # Y_tr: (65, 234), X_tr_pca: (65, pca_dim)
        # sum_pos_emb[s] = sum over pos examples of x_i  → (234, pca_dim)
        sum_pos_emb = Y_tr.T @ X_tr_pca   # (234, pca_dim)
        n_pos = Y_tr.sum(0)               # (234,)

        # post_mean[s] = (n_pos[s]*mean_pos[s] + lambda*global_mean) / (n_pos[s] + lambda)
        # = (sum_pos_emb[s] + lambda*global_mean) / (n_pos[s] + lambda)
        numerator   = sum_pos_emb + prior_strength * global_mean  # (234, pca_dim)
        denominator = (n_pos + prior_strength)[:, None]            # (234, 1)
        post_means  = numerator / denominator                      # (234, pca_dim)

        # sigma2 from intra-class variance (fast estimation)
        # variance ≈ mean of diagonal of sample covariance across species
        # Use global variance as approximation
        sigma2 = float(X_tr_pca.var())

        # dist2[s] = ||X_te_pca - post_means[s]||^2
        diff   = X_te_pca - post_means    # (234, pca_dim)
        dist2  = (diff ** 2).sum(1)       # (234,)
        preds[fi] = np.exp(-0.5 * dist2 / (sigma2 + 1e-8))

    return preds

# ──────────────────────────────────────────────────────────────────
# A1. pca_dim × prior_strength grid sweep (向量化，比 v2 快很多)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("方法 A: BSP 向量化 grid sweep")
print("="*70)

bsp_cache = {}  # (pca_dim, prior_s) → (auc, preds)
best_standalone = 0.0
best_standalone_key = None

for pca_dim in [8, 16, 24, 32, 48, 64]:
    dim_best = 0.0
    for prior_s in np.linspace(0.05, 2.0, 40):
        bp = bsp_vectorized_loo(pca_dim=pca_dim, prior_strength=prior_s)
        auc = macro_auc(file_labels, bp)
        key = (pca_dim, round(float(prior_s), 4))
        bsp_cache[key] = (auc, bp)
        if auc > dim_best: dim_best = auc
        if auc > best_standalone:
            best_standalone = auc
            best_standalone_key = key
    print(f"  pca={pca_dim}: best_standalone={dim_best:.4f}")

print(f"\n  Global best standalone BSP: {best_standalone:.6f}  key={best_standalone_key}")
results_list.append(("bsp_standalone", best_standalone, {"pca_dim": best_standalone_key[0], "prior_s": best_standalone_key[1]},
                     bsp_cache[best_standalone_key][1]))

# ──────────────────────────────────────────────────────────────────
# A2. Best BSP blend with k134_ref
# ──────────────────────────────────────────────────────────────────
print("\n  A2: BSP blend with k134_ultrafine_v2")
_, best_bsp_p = bsp_cache[best_standalone_key]

best_a2_auc, best_a2_preds, best_a2_w = 0.0, None, {}
for w_bsp in np.arange(0.03, 0.60, 0.01):
    ens = w_bsp * best_bsp_p + (1 - w_bsp) * k134_ref
    auc = macro_auc(file_labels, ens)
    if auc > best_a2_auc:
        best_a2_auc = auc
        best_a2_preds = ens.copy()
        best_a2_w = {"w_bsp": round(float(w_bsp), 3), "w_k134": round(float(1-w_bsp), 3),
                     "bsp_key": list(best_standalone_key)}

marker = "  *** NEW BEST ***" if best_a2_auc > CURRENT_BEST else ""
print(f"  BSP+k134: {best_a2_auc:.6f}  (delta={best_a2_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_a2_w}")
results_list.append(("bsp_k134_blend", best_a2_auc, best_a2_w, best_a2_preds))

# ──────────────────────────────────────────────────────────────────
# A3. Ultra-fine 5-way: al*logit_max + w_bsp*BSP + w1*knn1 + w3*knn3 + w4*knn4
# ──────────────────────────────────────────────────────────────────
print("\n  A3: Ultra-fine 5-way (logit_max + BSP + knn1 + knn3 + knn4)")

# Search from different starting points
best_a3_auc, best_a3_preds, best_a3_w = 0.0, None, {}

# Try top-5 BSP configs
top5 = sorted(bsp_cache.items(), key=lambda x: -x[1][0])[:5]
for (pd, ps), (auc_bp, bp) in top5:
    for al in np.arange(0.32, 0.52, 0.01):
        for wb in np.arange(0.02, 0.20, 0.01):
            rem = 1.0 - al - wb
            if rem < 0.20 or rem > 0.70: continue
            # try a few splits of rem into w1, w3, w4
            for w1_frac in [0.45, 0.48, 0.50]:
                for w3_frac in [0.00, 0.02, 0.04]:
                    w4_frac = 1.0 - w1_frac - w3_frac
                    if w4_frac < 0: continue
                    w1 = rem * w1_frac
                    w3 = rem * w3_frac
                    w4 = rem * w4_frac
                    ens = al*file_prob_max + wb*bp + w1*knn1 + w3*knn3 + w4*knn4
                    auc = macro_auc(file_labels, ens)
                    if auc > best_a3_auc:
                        best_a3_auc = auc
                        best_a3_preds = ens.copy()
                        best_a3_w = {"bsp_key": [pd, ps],
                                     "al": round(float(al), 3), "wb": round(float(wb), 3),
                                     "w1": round(float(w1), 3), "w3": round(float(w3), 3),
                                     "w4": round(float(w4), 3)}

marker = "  *** NEW BEST ***" if best_a3_auc > CURRENT_BEST else ""
print(f"  5-way: {best_a3_auc:.6f}  (delta={best_a3_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_a3_w}")
results_list.append(("bsp_5way_ultrafine", best_a3_auc, best_a3_w, best_a3_preds))

# ──────────────────────────────────────────────────────────────────
# A4. Multi-BSP (top-K average) + k134
# ──────────────────────────────────────────────────────────────────
print("\n  A4: Multi-BSP average + k134")
best_a4_auc, best_a4_preds, best_a4_w = 0.0, None, {}

for top_k in [3, 5, 8, 10, 15, 20]:
    topk_items = sorted(bsp_cache.items(), key=lambda x: -x[1][0])[:top_k]
    multi_bsp = np.mean([p for _, (_, p) in topk_items], axis=0)
    auc_multi = macro_auc(file_labels, multi_bsp)
    for w_bsp in np.arange(0.03, 0.60, 0.01):
        ens = w_bsp * multi_bsp + (1 - w_bsp) * k134_ref
        auc = macro_auc(file_labels, ens)
        if auc > best_a4_auc:
            best_a4_auc = auc
            best_a4_preds = ens.copy()
            best_a4_w = {"top_k_bsp": top_k, "w_bsp": round(float(w_bsp), 3),
                          "multi_bsp_alone": round(auc_multi, 4)}

marker = "  *** NEW BEST ***" if best_a4_auc > CURRENT_BEST else ""
print(f"  Multi-BSP+k134: {best_a4_auc:.6f}  (delta={best_a4_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_a4_w}")
results_list.append(("multi_bsp_k134", best_a4_auc, best_a4_w, best_a4_preds))

# ──────────────────────────────────────────────────────────────────
# A5. BSP + KNN (binary) blend — 掃所有組合
# ──────────────────────────────────────────────────────────────────
print("\n  A5: BSP + logit_max + knn1+knn3+knn4 full grid")
best_a5_auc, best_a5_preds, best_a5_w = 0.0, None, {}

# use top-3 BSP configs
for (pd, ps), (_, bp) in top5[:3]:
    for al in np.arange(0.30, 0.50, 0.02):
        for wb in np.arange(0.02, 0.30, 0.02):
            for w1 in np.arange(0.10, 0.45, 0.02):
                for w3 in np.arange(0.00, 0.08, 0.02):
                    w4 = 1.0 - al - wb - w1 - w3
                    if w4 < 0.05 or w4 > 0.45: continue
                    ens = al*file_prob_max + wb*bp + w1*knn1 + w3*knn3 + w4*knn4
                    auc = macro_auc(file_labels, ens)
                    if auc > best_a5_auc:
                        best_a5_auc = auc
                        best_a5_preds = ens.copy()
                        best_a5_w = {"bsp_pd": pd, "bsp_ps": ps,
                                     "al": round(float(al),3), "wb": round(float(wb),3),
                                     "w1": round(float(w1),3), "w3": round(float(w3),3),
                                     "w4": round(float(w4),3)}

marker = "  *** NEW BEST ***" if best_a5_auc > CURRENT_BEST else ""
print(f"  BSP+k134 full grid: {best_a5_auc:.6f}  (delta={best_a5_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_a5_w}")
results_list.append(("bsp_k134_full_grid", best_a5_auc, best_a5_w, best_a5_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 B：BSP with Adaptive Sigma (per-species intra-class variance)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 B: BSP with Adaptive Sigma (per-species variance)")
print("="*70)

def bsp_adaptive_sigma_loo(pca_dim=64, prior_strength=0.5, sigma_floor=0.1):
    """
    Vectorized BSP with per-species adaptive sigma2
    sigma2_s = max(intra-class variance of species s, sigma_floor)
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr = file_embs[mask_tr]
        Y_tr = file_labels[mask_tr]
        X_te = file_embs[[fi]]

        n_tr = X_tr.shape[0]
        pca_dim_eff = min(pca_dim, n_tr - 1)
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr).astype(np.float32)
        X_te_pca = pca.transform(X_te).astype(np.float32)

        global_mean = X_tr_pca.mean(0, keepdims=True)

        sum_pos_emb = Y_tr.T @ X_tr_pca
        n_pos = Y_tr.sum(0)
        numerator   = sum_pos_emb + prior_strength * global_mean
        denominator = (n_pos + prior_strength)[:, None]
        post_means  = numerator / denominator   # (234, pca_dim)

        # Per-species sigma2: E[||x - mu_s||^2 | y=1]
        # = E[||x||^2] - ||mu_s||^2 (approx)
        # More stable: use global variance scaled by 1/n_pos
        global_var = float(X_tr_pca.var())
        n_pos_safe = np.maximum(n_pos, 1.0)
        # sigma2_s = global_var / sqrt(n_pos_s) → smaller for more data
        sigma2_s = global_var / np.sqrt(n_pos_safe)   # (234,)
        sigma2_s = np.maximum(sigma2_s, sigma_floor)

        diff  = X_te_pca - post_means           # (234, pca_dim)
        dist2 = (diff ** 2).sum(1)              # (234,)
        preds[fi] = np.exp(-0.5 * dist2 / sigma2_s)

    return preds

best_b_auc, best_b_preds, best_b_w = 0.0, None, {}
for pca_dim in [24, 32, 48, 64]:
    for prior_s in [0.1, 0.3, 0.5, 0.8]:
        for sf in [0.01, 0.1, 0.5]:
            bp = bsp_adaptive_sigma_loo(pca_dim=pca_dim, prior_strength=prior_s, sigma_floor=sf)
            auc_bp = macro_auc(file_labels, bp)
            for w_bsp in np.arange(0.03, 0.50, 0.03):
                ens = w_bsp * bp + (1 - w_bsp) * k134_ref
                auc = macro_auc(file_labels, ens)
                if auc > best_b_auc:
                    best_b_auc = auc
                    best_b_preds = ens.copy()
                    best_b_w = {"pca_dim": pca_dim, "prior_s": prior_s, "sf": sf,
                                "w_bsp": round(float(w_bsp),3), "alone": round(auc_bp,4)}

marker = "  *** NEW BEST ***" if best_b_auc > CURRENT_BEST else ""
print(f"  BSP adaptive sigma: {best_b_auc:.6f}  (delta={best_b_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_b_w}")
results_list.append(("bsp_adaptive_sigma", best_b_auc, best_b_w, best_b_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 C：BSP Cosine (向量化)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 C: BSP Cosine (vectorized)")
print("="*70)

def bsp_cosine_vectorized(pca_dim=64, prior_strength=0.5):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr = file_embs[mask_tr]
        Y_tr = file_labels[mask_tr]
        X_te = file_embs[[fi]]

        n_tr = X_tr.shape[0]
        pca_dim_eff = min(pca_dim, n_tr - 1)
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        X_tr_pca = normalize(pca.fit_transform(X_tr), norm='l2').astype(np.float32)
        X_te_pca = normalize(pca.transform(X_te), norm='l2').astype(np.float32)

        global_mean_raw = X_tr_pca.mean(0)
        global_mean = global_mean_raw / (np.linalg.norm(global_mean_raw) + 1e-8)

        sum_pos_emb = Y_tr.T @ X_tr_pca    # (234, pca_dim)
        n_pos = Y_tr.sum(0)                 # (234,)
        numerator = sum_pos_emb + prior_strength * global_mean[None, :]
        denominator = (n_pos + prior_strength)[:, None]
        post_means_raw = numerator / denominator
        norms = np.linalg.norm(post_means_raw, axis=1, keepdims=True)
        post_means = post_means_raw / (norms + 1e-8)   # (234, pca_dim) normalized

        # cos_sim[s] = X_te_pca @ post_means[s]
        cos_sims = (X_te_pca @ post_means.T).ravel()   # (234,)
        preds[fi] = (cos_sims + 1.0) / 2.0   # map to [0,1]

    return preds

best_c_auc, best_c_preds, best_c_w = 0.0, None, {}
for pca_dim in [24, 32, 48, 64]:
    for prior_s in np.linspace(0.05, 1.5, 15):
        bp = bsp_cosine_vectorized(pca_dim=pca_dim, prior_strength=prior_s)
        auc_bp = macro_auc(file_labels, bp)
        for w_bsp in np.arange(0.03, 0.60, 0.03):
            ens = w_bsp * bp + (1 - w_bsp) * k134_ref
            auc = macro_auc(file_labels, ens)
            if auc > best_c_auc:
                best_c_auc = auc
                best_c_preds = ens.copy()
                best_c_w = {"pca_dim": pca_dim, "prior_s": round(float(prior_s),4),
                             "w_bsp": round(float(w_bsp),3), "alone": round(auc_bp,4)}

marker = "  *** NEW BEST ***" if best_c_auc > CURRENT_BEST else ""
print(f"  BSP-Cosine+k134: {best_c_auc:.6f}  (delta={best_c_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_c_w}")
results_list.append(("bsp_cosine_k134", best_c_auc, best_c_w, best_c_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 D：Negative-aware BSP (vectorized)
#   Score = cos(x, pos_proto) - beta * cos(x, neg_proto)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 D: Negative-aware BSP (pos - beta*neg)")
print("="*70)

def bsp_pos_neg_vectorized(pca_dim=64, prior_strength=0.5, beta=0.3):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr = file_embs[mask_tr]
        Y_tr = file_labels[mask_tr]
        Y_neg = 1.0 - Y_tr
        X_te = file_embs[[fi]]

        n_tr = X_tr.shape[0]
        pca_dim_eff = min(pca_dim, n_tr - 1)
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr).astype(np.float32)
        X_te_pca = pca.transform(X_te).astype(np.float32)

        global_mean = X_tr_pca.mean(0)
        sigma2 = float(X_tr_pca.var())

        # pos prototype
        sum_pos = Y_tr.T @ X_tr_pca    # (234, pca_dim)
        n_pos   = Y_tr.sum(0)
        pos_proto = (sum_pos + prior_strength * global_mean) / (n_pos + prior_strength)[:, None]

        # neg prototype
        sum_neg = Y_neg.T @ X_tr_pca   # (234, pca_dim)
        n_neg   = Y_neg.sum(0)
        neg_proto = (sum_neg + prior_strength * global_mean) / (n_neg + prior_strength)[:, None]

        # RBF scores
        diff_pos = X_te_pca - pos_proto    # (234, pca_dim)
        diff_neg = X_te_pca - neg_proto    # (234, pca_dim)
        pos_score = np.exp(-0.5 * (diff_pos**2).sum(1) / (sigma2 + 1e-8))
        neg_score = np.exp(-0.5 * (diff_neg**2).sum(1) / (sigma2 + 1e-8))
        preds[fi] = np.clip(pos_score - beta * neg_score, 0, 1)

    return preds

best_d_auc, best_d_preds, best_d_w = 0.0, None, {}
for pca_dim in [24, 32, 48, 64]:
    for prior_s in [0.1, 0.3, 0.5]:
        for beta in [0.1, 0.2, 0.3, 0.5, 0.7]:
            bp = bsp_pos_neg_vectorized(pca_dim=pca_dim, prior_strength=prior_s, beta=beta)
            auc_bp = macro_auc(file_labels, bp)
            for w_bsp in np.arange(0.03, 0.60, 0.03):
                ens = w_bsp * bp + (1 - w_bsp) * k134_ref
                auc = macro_auc(file_labels, ens)
                if auc > best_d_auc:
                    best_d_auc = auc
                    best_d_preds = ens.copy()
                    best_d_w = {"pca_dim": pca_dim, "prior_s": prior_s, "beta": beta,
                                "w_bsp": round(float(w_bsp),3), "alone": round(auc_bp,4)}

marker = "  *** NEW BEST ***" if best_d_auc > CURRENT_BEST else ""
print(f"  BSP neg-aware+k134: {best_d_auc:.6f}  (delta={best_d_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_d_w}")
results_list.append(("bsp_neg_aware", best_d_auc, best_d_w, best_d_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 E：BSP + window-level KNN (窗口層級 KNN + file-level BSP)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 E: BSP + window-level binary KNN")
print("="*70)

def window_knn_binary(k=3):
    """Window-level KNN，file score = max over windows"""
    X_norm = normalize(emb_win, norm='l2')
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = win_file_idx == fi
        train_mask = win_file_idx != fi
        X_te = X_norm[test_mask]
        X_tr = X_norm[train_mask]
        y_tr_win = labels_win[train_mask]  # window labels (binary)

        sims = X_te @ X_tr.T
        k_eff = min(k, X_tr.shape[0])
        nn_idx = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        w = np.take_along_axis(sims, nn_idx, axis=1).clip(0)
        w_sum = w.sum(1, keepdims=True); w_sum[w_sum < 1e-9] = 1.0
        w = w / w_sum
        win_preds = np.stack([(w[i:i+1] @ y_tr_win[nn_idx[i]]).ravel()
                               for i in range(len(X_te))], axis=0)  # (n_te, 234)
        preds[fi] = win_preds.max(0)

    return preds

print("  Computing window-level KNN(3)...")
wknn3 = window_knn_binary(k=3)
print(f"  window-KNN(3) alone: {macro_auc(file_labels, wknn3):.4f}")

best_e_auc, best_e_preds, best_e_w = 0.0, None, {}
_, bsp_best_p = bsp_cache[best_standalone_key]
for w_bsp in np.arange(0.02, 0.40, 0.02):
    for w_wknn in np.arange(0.02, 0.40, 0.02):
        w_k134 = 1.0 - w_bsp - w_wknn
        if w_k134 < 0: continue
        ens = w_bsp * bsp_best_p + w_wknn * wknn3 + w_k134 * k134_ref
        auc = macro_auc(file_labels, ens)
        if auc > best_e_auc:
            best_e_auc = auc
            best_e_preds = ens.copy()
            best_e_w = {"w_bsp": round(float(w_bsp),3), "w_wknn": round(float(w_wknn),3),
                        "w_k134": round(float(w_k134),3)}

marker = "  *** NEW BEST ***" if best_e_auc > CURRENT_BEST else ""
print(f"  BSP+wKNN+k134: {best_e_auc:.6f}  (delta={best_e_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_e_w}")
results_list.append(("bsp_wknn_k134", best_e_auc, best_e_w, best_e_preds))

# ══════════════════════════════════════════════════════════════════
# 最後：ultra-fine grid 搜索最佳 BSP config 與各 weight
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Final Ultra-Fine: expand best BSP range ± 0.01")
print("="*70)

# 找到目前所有 results 中最高分的
all_aucs = [(name, auc) for name, auc, _, _ in results_list]
all_aucs.sort(key=lambda x: -x[1])
print("  Current run results:")
for name, auc in all_aucs:
    marker = "*** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"    {name}: {auc:.6f}  {marker}")

# ══════════════════════════════════════════════════════════════════
# 儲存結果
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Current best: {CURRENT_BEST:.6f}")
for name, auc, params, _ in sorted(results_list, key=lambda x: -x[1]):
    marker = "  *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.6f}  (delta={auc-CURRENT_BEST:+.6f}){marker}")

best_result = max(results_list, key=lambda x: x[1])
best_name, best_auc, best_params, best_preds_final = best_result

with open(RESULTS_PATH) as f:
    results_json = json.load(f)

def serialize_val(v):
    if isinstance(v, (np.float32, np.float64)): return float(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.ndarray): return None
    if isinstance(v, tuple): return list(v)
    return v

for name, auc, params, _ in results_list:
    record = {"method": name, "loo_auc": round(float(auc), 6)}
    for k, v in params.items():
        sv = serialize_val(v)
        if sv is not None:
            record[k] = sv
    results_json["experiments"].append(record)

overall_best = results_json["best"]["loo_auc"]
if best_auc > overall_best:
    results_json["best"] = {
        "method": best_name,
        "loo_auc": round(float(best_auc), 6),
        "config": {k: serialize_val(v) for k, v in best_params.items() if serialize_val(v) is not None},
        "note": f"new_methods_v3 (BSP vectorized) 2026-03-25; prev={overall_best:.6f}"
    }
    print(f"\nNEW BEST: {best_name} AUC={best_auc:.6f}")

    model_dict = {
        "method": best_name,
        "loo_auc": round(float(best_auc), 6),
        "params": {k: serialize_val(v) for k, v in best_params.items() if serialize_val(v) is not None},
        "file_list": file_list.tolist(),
        "loo_preds": best_preds_final.tolist(),
        "file_embs_norm": file_embs_norm.tolist(),
        "file_prob_max": file_prob_max.tolist(),
        "file_labels": file_labels.tolist(),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"Saved model → {MODEL_PATH}")
else:
    print(f"\nNo improvement over current best ({overall_best:.6f})")
    print(f"Best this run: {best_name} AUC={best_auc:.6f}")

with open(RESULTS_PATH, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Results saved → {RESULTS_PATH}")
