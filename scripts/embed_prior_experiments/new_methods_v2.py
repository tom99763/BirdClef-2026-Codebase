"""
embed_prior new_methods_v2.py
接續 v1 的發現：
  - BSP(pca=64, prior=0.5) = 0.9045 (standalone!) — 最有潛力
  - 深入 BSP：更細的 prior_strength sweep + blend with k134
  - 方法 4: window-level KLT mean k134 style
  - 方法 5: KNN + KLT 6-way
  - 方法 6: BSP + k134_ultrafine_v2 blend (最有希望)

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

print(f"資料: {n_files} files, {n_species} species, {len(emb_win)} windows")
print(f"Current best: {CURRENT_BEST:.6f}")

def macro_auc(y_true, y_score):
    mask = (y_true.sum(0) > 0) & (y_true.sum(0) < n_files)
    if mask.sum() < 2: return float('nan')
    try:
        return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')
    except Exception: return float('nan')

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
print(f"  KNN1={macro_auc(file_labels,knn1):.4f}, KNN3={macro_auc(file_labels,knn3):.4f}, "
      f"KNN4={macro_auc(file_labels,knn4):.4f}")
print(f"  k134_ref={macro_auc(file_labels,k134_ref):.6f}")

results_list = []

# ══════════════════════════════════════════════════════════════════
# 方法 A：Bayesian Shrinkage Prototype (BSP) — 深入探索
#
# 核心：每個 species s 的 prototype = shrinkage toward global mean
#   post_mean_s = (n_pos * sample_mean_s + lambda * global_mean) / (n_pos + lambda)
# Score: RBF( x_test, post_mean_s )
#
# 探索：
#   A1. pca_dim ∈ {8,16,24,32,48,64}  × prior_strength ∈ linspace(0.05, 1.0, 20)
#   A2. sigma^2 adaptive (use intra-class variance from training data)
#   A3. Asymmetric: score = max(cos_pos_proto, -beta*cos_neg_proto)
#   A4. BSP blend with k134_ultrafine_v2
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 A: Bayesian Shrinkage Prototype — 深入探索")
print("="*70)

def bsp_loo(pca_dim=64, prior_strength=0.5, sigma2_scale=1.0):
    """
    LOO BSP with optional adaptive sigma2
    sigma2 = sigma2_scale * (mean intra-class variance in PCA space)
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr = file_embs[mask_tr]
        Y_tr = file_labels[mask_tr]
        X_te = file_embs[[fi]]

        n_tr = X_tr.shape[0]
        pca_dim_eff = min(pca_dim, n_tr - 1, X_tr.shape[1])
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        global_mean = X_tr_pca.mean(0)

        # Compute sigma2 from intra-class variance
        var_list = []
        for s in range(n_species):
            pos_idx = Y_tr[:, s] > 0.5
            if pos_idx.sum() >= 2:
                var_list.append(X_tr_pca[pos_idx].var(0).mean())
        sigma2 = float(np.median(var_list) if var_list else pca_dim_eff) * sigma2_scale

        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            pos_idx = Y_tr[:, s] > 0.5
            n_pos = pos_idx.sum()
            if n_pos == 0:
                post_mean = global_mean
            else:
                sample_mean = X_tr_pca[pos_idx].mean(0)
                post_mean = (n_pos * sample_mean + prior_strength * global_mean) / (n_pos + prior_strength)
            diff = X_te_pca[0] - post_mean
            dist2 = (diff ** 2).sum()
            scores[s] = np.exp(-0.5 * dist2 / (sigma2 + 1e-8))

        preds[fi] = scores
    return preds

# A1. pca_dim × prior_strength sweep
print("\n  A1: pca_dim × prior_strength grid sweep")
best_a1_alone, best_a1_config = 0.0, {}
bsp_results = {}  # (pca_dim, prior_s) → preds

for pca_dim in [8, 16, 24, 32, 48, 64]:
    for prior_s in np.linspace(0.05, 1.5, 30):
        bp = bsp_loo(pca_dim=pca_dim, prior_strength=prior_s)
        auc = macro_auc(file_labels, bp)
        bsp_results[(pca_dim, round(prior_s, 4))] = (auc, bp)
        if auc > best_a1_alone:
            best_a1_alone = auc
            best_a1_config = {"pca_dim": pca_dim, "prior_s": round(float(prior_s), 4)}

    # 顯示此 pca_dim 的最佳
    best_for_dim = max((v[0] for k, v in bsp_results.items() if k[0] == pca_dim))
    print(f"    pca={pca_dim}: best_alone={best_for_dim:.4f}")

print(f"  A1 best standalone: {best_a1_alone:.6f}  config={best_a1_config}")

# A2. 用最佳 standalone BSP blend with k134_ref
print("\n  A2: BSP blend with k134_ultrafine_v2")
_, best_bsp_preds = max(bsp_results.values(), key=lambda x: x[0])

best_a2_auc, best_a2_preds, best_a2_w = 0.0, None, {}
for w_bsp in np.arange(0.05, 0.55, 0.02):
    w_k134 = 1.0 - w_bsp
    ens = w_bsp * best_bsp_preds + w_k134 * k134_ref
    auc = macro_auc(file_labels, ens)
    if auc > best_a2_auc:
        best_a2_auc = auc
        best_a2_preds = ens.copy()
        best_a2_w = {"w_bsp": float(w_bsp), "w_k134": float(w_k134),
                     "bsp_config": best_a1_config}

marker = "  *** NEW BEST ***" if best_a2_auc > CURRENT_BEST else ""
print(f"  A2 BSP+k134 blend: {best_a2_auc:.6f}  (delta={best_a2_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_a2_w}")
results_list.append(("bsp_k134_blend", best_a2_auc, best_a2_w, best_a2_preds))

# A3. 更細：3-way BSP + logit_max + KNN
print("\n  A3: 3-way BSP + logit_max + KNN sweep")
best_a3_auc, best_a3_preds, best_a3_w = 0.0, None, {}

# 先找 top-5 BSP configs (by standalone AUC) 用來 blend
top5_bsp = sorted(bsp_results.items(), key=lambda x: -x[0][0] if False else -x[1][0])[:5]
print(f"  Top-5 BSP configs (standalone):")
for (pd, ps), (auc, _) in top5_bsp:
    print(f"    pca={pd}, prior={ps:.4f}: {auc:.4f}")

for (pd, ps), (auc_bp, bp) in top5_bsp:
    for al in np.arange(0.30, 0.55, 0.02):
        for w_bsp in np.arange(0.05, 0.50, 0.02):
            w_knn = 1.0 - al - w_bsp
            if w_knn < 0: continue
            ens = al*file_prob_max + w_bsp*bp + w_knn*(0.28*knn1 + 0.02*knn3 + 0.28*knn4)/0.58
            auc = macro_auc(file_labels, ens)
            if auc > best_a3_auc:
                best_a3_auc = auc
                best_a3_preds = ens.copy()
                best_a3_w = {"pca_dim": pd, "prior_s": ps, "al": float(al),
                              "w_bsp": float(w_bsp), "w_knn": float(1.0-al-w_bsp)}

marker = "  *** NEW BEST ***" if best_a3_auc > CURRENT_BEST else ""
print(f"  A3 3-way: {best_a3_auc:.6f}  (delta={best_a3_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_a3_w}")
results_list.append(("bsp_logitmax_knn_3way", best_a3_auc, best_a3_w, best_a3_preds))

# A4. 4-way: BSP + logit_max + KNN1 + KNN4
print("\n  A4: 4-way BSP + logit_max + KNN1 + KNN4")
best_a4_auc, best_a4_preds, best_a4_w = 0.0, None, {}

# 用 standalone best BSP
_, best_bsp_p = top5_bsp[0][1]
best_bsp_cfg = top5_bsp[0][0]

for al in np.arange(0.30, 0.55, 0.02):
    for w_bsp in np.arange(0.02, 0.25, 0.02):
        for w1 in np.arange(0.15, 0.40, 0.02):
            for w3 in np.arange(0.00, 0.08, 0.02):
                w4 = 1.0 - al - w_bsp - w1 - w3
                if w4 < 0: continue
                ens = al*file_prob_max + w_bsp*best_bsp_p + w1*knn1 + w3*knn3 + w4*knn4
                auc = macro_auc(file_labels, ens)
                if auc > best_a4_auc:
                    best_a4_auc = auc
                    best_a4_preds = ens.copy()
                    best_a4_w = {"bsp_cfg": best_bsp_cfg,
                                 "al": float(al), "w_bsp": float(w_bsp),
                                 "w1": float(w1), "w3": float(w3), "w4": float(w4)}

marker = "  *** NEW BEST ***" if best_a4_auc > CURRENT_BEST else ""
print(f"  A4 4-way: {best_a4_auc:.6f}  (delta={best_a4_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_a4_w}")
results_list.append(("bsp_4way", best_a4_auc, best_a4_w, best_a4_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 B：BSP with Negative Prototype (Repulsion)
#   Score_s = RBF(x_te, pos_proto_s) - beta * RBF(x_te, neg_proto_s)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 B: BSP with Negative Prototype Repulsion")
print("="*70)

def bsp_pos_neg_loo(pca_dim=64, prior_strength=0.5, beta=0.5):
    """
    Score_s = RBF(x, pos_proto_s) - beta * RBF(x, neg_proto_s)
    Clipped to [0, 1] range
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
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        global_mean = X_tr_pca.mean(0)
        sigma2 = float(X_tr_pca.var())

        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            pos_idx = Y_tr[:, s] > 0.5
            neg_idx = ~pos_idx
            n_pos = pos_idx.sum()
            n_neg = neg_idx.sum()

            if n_pos == 0:
                pos_proto = global_mean
            else:
                sm = X_tr_pca[pos_idx].mean(0)
                pos_proto = (n_pos * sm + prior_strength * global_mean) / (n_pos + prior_strength)

            if n_neg == 0:
                neg_proto = global_mean
            else:
                sm_neg = X_tr_pca[neg_idx].mean(0)
                neg_proto = (n_neg * sm_neg + prior_strength * global_mean) / (n_neg + prior_strength)

            x = X_te_pca[0]
            pos_score = np.exp(-0.5 * ((x - pos_proto)**2).sum() / (sigma2 + 1e-8))
            neg_score = np.exp(-0.5 * ((x - neg_proto)**2).sum() / (sigma2 + 1e-8))
            scores[s] = np.clip(pos_score - beta * neg_score, 0, 1)

        preds[fi] = scores
    return preds

best_b_auc, best_b_preds, best_b_w = 0.0, None, {}
for pca_dim in [24, 32, 48, 64]:
    for prior_s in [0.1, 0.3, 0.5]:
        for beta in [0.1, 0.2, 0.3, 0.5]:
            bp = bsp_pos_neg_loo(pca_dim=pca_dim, prior_strength=prior_s, beta=beta)
            auc_bp = macro_auc(file_labels, bp)
            # blend with k134
            for w_bsp in np.arange(0.05, 0.50, 0.05):
                ens = w_bsp * bp + (1 - w_bsp) * k134_ref
                auc = macro_auc(file_labels, ens)
                if auc > best_b_auc:
                    best_b_auc = auc
                    best_b_preds = ens.copy()
                    best_b_w = {"pca_dim": pca_dim, "prior_s": prior_s, "beta": beta,
                                "w_bsp": float(w_bsp), "alone_auc": round(auc_bp, 4)}

marker = "  *** NEW BEST ***" if best_b_auc > CURRENT_BEST else ""
print(f"  BSP pos-neg: {best_b_auc:.6f}  (delta={best_b_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_b_w}")
results_list.append(("bsp_pos_neg", best_b_auc, best_b_w, best_b_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 C：Fine-grained multi-BSP ensemble
#   BSP 在多個 (pca_dim, prior_s) 的平均
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 C: Multi-BSP ensemble")
print("="*70)

# 取 top-10 BSP configs 做 ensemble
top10_bsp = sorted(bsp_results.items(), key=lambda x: -x[1][0])[:10]
multi_bsp_preds = np.mean([p for _, (_, p) in top10_bsp], axis=0)
auc_multi = macro_auc(file_labels, multi_bsp_preds)
print(f"  Top-10 BSP average: {auc_multi:.4f}")

best_c_auc, best_c_preds, best_c_w = 0.0, None, {}
for w_bsp in np.arange(0.05, 0.60, 0.02):
    ens = w_bsp * multi_bsp_preds + (1 - w_bsp) * k134_ref
    auc = macro_auc(file_labels, ens)
    if auc > best_c_auc:
        best_c_auc = auc
        best_c_preds = ens.copy()
        best_c_w = {"w_bsp_ensemble": float(w_bsp), "n_bsp": 10}

marker = "  *** NEW BEST ***" if best_c_auc > CURRENT_BEST else ""
print(f"  Multi-BSP+k134: {best_c_auc:.6f}  (delta={best_c_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_c_w}")
results_list.append(("multi_bsp_k134", best_c_auc, best_c_w, best_c_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 D：BSP Cosine (用 cosine similarity 而非 RBF distance)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 D: BSP Cosine similarity")
print("="*70)

def bsp_cosine_loo(pca_dim=64, prior_strength=0.5):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr = file_embs[mask_tr]
        Y_tr = file_labels[mask_tr]
        X_te = file_embs[[fi]]

        n_tr = X_tr.shape[0]
        pca_dim_eff = min(pca_dim, n_tr - 1)
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        X_tr_pca = normalize(pca.fit_transform(X_tr), norm='l2')
        X_te_pca = normalize(pca.transform(X_te), norm='l2')

        global_mean = X_tr_pca.mean(0)
        global_mean /= (np.linalg.norm(global_mean) + 1e-8)

        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            pos_idx = Y_tr[:, s] > 0.5
            n_pos = pos_idx.sum()
            if n_pos == 0:
                post_mean = global_mean
            else:
                sm = X_tr_pca[pos_idx].mean(0)
                post_mean = (n_pos * sm + prior_strength * global_mean) / (n_pos + prior_strength)
            pm_norm = post_mean / (np.linalg.norm(post_mean) + 1e-8)
            cos_sim = float(X_te_pca[0] @ pm_norm)
            # map [-1,1] → [0,1]
            scores[s] = (cos_sim + 1.0) / 2.0

        preds[fi] = scores
    return preds

best_d_auc, best_d_preds, best_d_w = 0.0, None, {}
for pca_dim in [24, 32, 48, 64]:
    for prior_s in np.linspace(0.05, 1.0, 10):
        bp = bsp_cosine_loo(pca_dim=pca_dim, prior_strength=prior_s)
        auc_bp = macro_auc(file_labels, bp)
        for w_bsp in np.arange(0.05, 0.60, 0.05):
            ens = w_bsp * bp + (1 - w_bsp) * k134_ref
            auc = macro_auc(file_labels, ens)
            if auc > best_d_auc:
                best_d_auc = auc
                best_d_preds = ens.copy()
                best_d_w = {"pca_dim": pca_dim, "prior_s": round(float(prior_s), 4),
                             "w_bsp": float(w_bsp), "alone_auc": round(auc_bp, 4)}

marker = "  *** NEW BEST ***" if best_d_auc > CURRENT_BEST else ""
print(f"  BSP-Cosine+k134: {best_d_auc:.6f}  (delta={best_d_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_d_w}")
results_list.append(("bsp_cosine_k134", best_d_auc, best_d_w, best_d_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 E：Ultra-fine k134 + BSP 5-way
#   al*logit_max + w1*knn1 + w3*knn3 + w4*knn4 + wb*bsp
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 E: 5-way (k134 + BSP) ultra fine sweep")
print("="*70)

# 用 best standalone BSP
best_bsp_key = max(bsp_results, key=lambda k: bsp_results[k][0])
_, best_bsp_p5 = bsp_results[best_bsp_key]
print(f"  Using BSP key={best_bsp_key}, standalone AUC={bsp_results[best_bsp_key][0]:.4f}")

best_e_auc, best_e_preds, best_e_w = 0.0, None, {}
for al in np.arange(0.35, 0.50, 0.01):
    for wb in np.arange(0.02, 0.25, 0.01):
        for w1 in np.arange(0.15, 0.40, 0.01):
            for w3 in np.arange(0.00, 0.06, 0.01):
                w4 = 1.0 - al - wb - w1 - w3
                if w4 < 0.10 or w4 > 0.45: continue
                ens = al*file_prob_max + wb*best_bsp_p5 + w1*knn1 + w3*knn3 + w4*knn4
                auc = macro_auc(file_labels, ens)
                if auc > best_e_auc:
                    best_e_auc = auc
                    best_e_preds = ens.copy()
                    best_e_w = {"bsp_cfg": best_bsp_key,
                                "al": round(float(al), 3), "wb": round(float(wb), 3),
                                "w1": round(float(w1), 3), "w3": round(float(w3), 3),
                                "w4": round(float(w4), 3)}

marker = "  *** NEW BEST ***" if best_e_auc > CURRENT_BEST else ""
print(f"  5-way k134+BSP: {best_e_auc:.6f}  (delta={best_e_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_e_w}")
results_list.append(("k134_bsp_5way", best_e_auc, best_e_w, best_e_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 F：窗口層級 BSP (window-level prototype scoring)
#   對每個 test window，計算 prototype score，再 max over windows
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 F: Window-level BSP")
print("="*70)

def window_bsp_loo(pca_dim=64, prior_strength=0.5):
    """
    對每個 test window 計算 prototype score，file score = max over windows
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = win_file_idx == fi
        train_file_mask = np.arange(n_files) != fi

        # 用 train file embeddings 建 PCA + prototype
        X_tr_file = file_embs[train_file_mask]
        Y_tr_file = file_labels[train_file_mask]

        n_tr = X_tr_file.shape[0]
        pca_dim_eff = min(pca_dim, n_tr - 1)
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_file)
        global_mean = X_tr_pca.mean(0)

        # test windows
        X_te_wins = pca.transform(emb_win[test_mask])  # (n_te_wins, pca_dim_eff)
        var_list = []
        for s in range(n_species):
            pos_idx = Y_tr_file[:, s] > 0.5
            if pos_idx.sum() >= 2:
                var_list.append(X_tr_pca[pos_idx].var(0).mean())
        sigma2 = float(np.median(var_list) if var_list else pca_dim_eff)

        win_scores = np.zeros((X_te_wins.shape[0], n_species), dtype=np.float32)
        for s in range(n_species):
            pos_idx = Y_tr_file[:, s] > 0.5
            n_pos = pos_idx.sum()
            if n_pos == 0:
                post_mean = global_mean
            else:
                sm = X_tr_pca[pos_idx].mean(0)
                post_mean = (n_pos * sm + prior_strength * global_mean) / (n_pos + prior_strength)
            diff = X_te_wins - post_mean[None, :]   # (n_te_wins, pca_dim_eff)
            dist2 = (diff ** 2).sum(1)              # (n_te_wins,)
            win_scores[:, s] = np.exp(-0.5 * dist2 / (sigma2 + 1e-8))

        preds[fi] = win_scores.max(0)
    return preds

best_f_auc, best_f_preds, best_f_w = 0.0, None, {}
for pca_dim in [32, 48, 64]:
    for prior_s in [0.1, 0.3, 0.5]:
        bp = window_bsp_loo(pca_dim=pca_dim, prior_strength=prior_s)
        auc_bp = macro_auc(file_labels, bp)
        print(f"    win-BSP(pca={pca_dim}, prior={prior_s:.2f}): {auc_bp:.4f}")
        for w_bsp in np.arange(0.05, 0.60, 0.05):
            ens = w_bsp * bp + (1 - w_bsp) * k134_ref
            auc = macro_auc(file_labels, ens)
            if auc > best_f_auc:
                best_f_auc = auc
                best_f_preds = ens.copy()
                best_f_w = {"pca_dim": pca_dim, "prior_s": prior_s,
                             "w_bsp": float(w_bsp), "alone_auc": round(auc_bp, 4)}

marker = "  *** NEW BEST ***" if best_f_auc > CURRENT_BEST else ""
print(f"  win-BSP+k134: {best_f_auc:.6f}  (delta={best_f_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_f_w}")
results_list.append(("win_bsp_k134", best_f_auc, best_f_w, best_f_preds))

# ══════════════════════════════════════════════════════════════════
# 總結
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Current best: {CURRENT_BEST:.6f}")
for name, auc, params, _ in sorted(results_list, key=lambda x: -x[1]):
    marker = "  *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.6f}  (delta={auc-CURRENT_BEST:+.6f}){marker}")

if not results_list:
    print("No results.")
    import sys; sys.exit(0)

best_result = max(results_list, key=lambda x: x[1])
best_name, best_auc, best_params, best_preds = best_result

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
        "note": f"new_methods_v2 (BSP) 2026-03-25; prev={overall_best:.6f}"
    }
    print(f"\nNEW BEST: {best_name} AUC={best_auc:.6f}")

    model_dict = {
        "method": best_name,
        "loo_auc": round(float(best_auc), 6),
        "params": {k: serialize_val(v) for k, v in best_params.items() if serialize_val(v) is not None},
        "file_list": file_list.tolist(),
        "loo_preds": best_preds.tolist(),
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
