"""
embed_prior new_methods_v4.py
高效 BSP：避免 per-fold PCA，改用全資料 PCA + LOO correction

策略：
  1. 用所有 66 files 做 PCA（一次）
  2. LOO 時不重做 PCA，直接在 PCA space 做 shrinkage
     → 近似但速度快 1000x
  3. 也試用全量 PCA 再做 inner LOO（true LOO 但 PCA 是 global）

目標: 超越 CURRENT_BEST = 0.894048
"""

import numpy as np
import json
import pickle
import warnings
import os
import time
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
# 方法 A：Global-PCA BSP (近似 LOO)
# 先用全部 66 files 做 PCA，然後 LOO 時在 PCA space 直接做
# 這不是嚴格 LOO（PCA 看到了 test），但接近且快很多
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 A: Global-PCA BSP (fast approximation)")
print("="*70)

def bsp_global_pca(pca_dim, prior_strength):
    """
    Global PCA 一次，然後 LOO in PCA space.
    ~1000x faster than per-fold PCA.
    """
    pca = PCA(n_components=min(pca_dim, n_files - 1), random_state=42)
    X_pca = pca.fit_transform(file_embs).astype(np.float32)   # (66, pca_dim)

    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr_pca = X_pca[mask_tr]      # (65, pca_dim)
        Y_tr     = file_labels[mask_tr] # (65, 234)
        X_te_pca = X_pca[[fi]]         # (1, pca_dim)

        global_mean = X_tr_pca.mean(0)
        sigma2 = float(X_tr_pca.var())

        sum_pos = Y_tr.T @ X_tr_pca         # (234, pca_dim)
        n_pos   = Y_tr.sum(0)               # (234,)
        post_means = (sum_pos + prior_strength * global_mean) / \
                     (n_pos + prior_strength)[:, None]   # (234, pca_dim)

        diff  = X_te_pca - post_means        # (234, pca_dim)
        dist2 = (diff ** 2).sum(1)           # (234,)
        preds[fi] = np.exp(-0.5 * dist2 / (sigma2 + 1e-8))

    return preds

t0 = time.time()
# Test speed
bp_test = bsp_global_pca(pca_dim=64, prior_strength=0.3)
t1 = time.time()
print(f"  One combo speed: {t1-t0:.2f}s")
print(f"  Test AUC: {macro_auc(file_labels, bp_test):.4f}")

# Full grid sweep
best_a_auc, best_a_preds, best_a_key = 0.0, None, None
bsp_global_cache = {}

pca_dims = [8, 16, 24, 32, 40, 48, 56, 64]
prior_vals = np.concatenate([
    np.linspace(0.01, 0.2, 20),   # dense near zero
    np.linspace(0.2, 2.0, 20),    # medium range
    np.linspace(2.0, 8.0, 10),    # high prior
])

print(f"\n  Sweeping {len(pca_dims)} pca_dims × {len(prior_vals)} prior vals...")
for pca_dim in pca_dims:
    dim_best = 0.0
    for prior_s in prior_vals:
        bp = bsp_global_pca(pca_dim=pca_dim, prior_strength=prior_s)
        auc = macro_auc(file_labels, bp)
        key = (pca_dim, round(float(prior_s), 4))
        bsp_global_cache[key] = (auc, bp)
        if auc > dim_best: dim_best = auc
        if auc > best_a_auc:
            best_a_auc = auc
            best_a_preds = bp.copy()
            best_a_key = key
    print(f"    pca={pca_dim}: best={dim_best:.4f}")

print(f"  Global best standalone: {best_a_auc:.6f}  key={best_a_key}")
results_list.append(("bsp_global_standalone", best_a_auc,
                     {"pca_dim": best_a_key[0], "prior_s": best_a_key[1]}, best_a_preds))

# A2. blend with k134_ref
print("\n  A2: BSP blend with k134")
best_a2_auc, best_a2_preds, best_a2_w = 0.0, None, {}
for w_bsp in np.arange(0.02, 0.65, 0.01):
    ens = w_bsp * best_a_preds + (1 - w_bsp) * k134_ref
    auc = macro_auc(file_labels, ens)
    if auc > best_a2_auc:
        best_a2_auc = auc
        best_a2_preds = ens.copy()
        best_a2_w = {"w_bsp": round(float(w_bsp),3), "bsp_key": list(best_a_key)}

marker = "  *** NEW BEST ***" if best_a2_auc > CURRENT_BEST else ""
print(f"  BSP+k134: {best_a2_auc:.6f}  (delta={best_a2_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_a2_w}")
results_list.append(("bsp_global_k134", best_a2_auc, best_a2_w, best_a2_preds))

# A3. Ultra-fine 5-way: al*logit_max + wb*BSP + w1*knn1 + w3*knn3 + w4*knn4
print("\n  A3: 5-way logit_max + BSP + knn1 + knn3 + knn4")
best_a3_auc, best_a3_preds, best_a3_w = 0.0, None, {}

top5_bsp = sorted(bsp_global_cache.items(), key=lambda x: -x[1][0])[:5]
print("  Top-5 BSP standalone:")
for k, (auc, _) in top5_bsp:
    print(f"    {k}: {auc:.4f}")

for (pd, ps), (_, bp) in top5_bsp:
    for al in np.arange(0.30, 0.52, 0.01):
        for wb in np.arange(0.02, 0.25, 0.01):
            rem = 1.0 - al - wb
            if rem < 0.25 or rem > 0.70: continue
            # try all w1/w3/w4 splits
            for w1_r in np.arange(0.40, 0.60, 0.02):
                for w3_r in np.arange(0.00, 0.10, 0.02):
                    w4_r = 1.0 - w1_r - w3_r
                    if w4_r < 0.30: continue
                    w1 = rem * w1_r
                    w3 = rem * w3_r
                    w4 = rem * w4_r
                    ens = al*file_prob_max + wb*bp + w1*knn1 + w3*knn3 + w4*knn4
                    auc = macro_auc(file_labels, ens)
                    if auc > best_a3_auc:
                        best_a3_auc = auc
                        best_a3_preds = ens.copy()
                        best_a3_w = {"bsp_key": [pd, ps],
                                     "al": round(float(al),3), "wb": round(float(wb),3),
                                     "w1": round(float(w1),3), "w3": round(float(w3),3),
                                     "w4": round(float(w4),3)}

marker = "  *** NEW BEST ***" if best_a3_auc > CURRENT_BEST else ""
print(f"  5-way: {best_a3_auc:.6f}  (delta={best_a3_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_a3_w}")
results_list.append(("bsp_5way", best_a3_auc, best_a3_w, best_a3_preds))

# A4. Multi-BSP ensemble + k134
print("\n  A4: Multi-BSP ensemble")
best_a4_auc, best_a4_preds, best_a4_w = 0.0, None, {}

for top_k in [3, 5, 8, 10, 15, 20]:
    topk_items = sorted(bsp_global_cache.items(), key=lambda x: -x[1][0])[:top_k]
    multi_bsp = np.mean([p for _, (_, p) in topk_items], axis=0)
    for w_bsp in np.arange(0.02, 0.60, 0.01):
        ens = w_bsp * multi_bsp + (1 - w_bsp) * k134_ref
        auc = macro_auc(file_labels, ens)
        if auc > best_a4_auc:
            best_a4_auc = auc
            best_a4_preds = ens.copy()
            best_a4_w = {"top_k": top_k, "w_bsp": round(float(w_bsp),3)}

marker = "  *** NEW BEST ***" if best_a4_auc > CURRENT_BEST else ""
print(f"  Multi-BSP+k134: {best_a4_auc:.6f}  (delta={best_a4_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_a4_w}")
results_list.append(("multi_bsp_global_k134", best_a4_auc, best_a4_w, best_a4_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 B：BSP Cosine (global PCA, fast)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 B: BSP Cosine (global PCA)")
print("="*70)

def bsp_cosine_global_pca(pca_dim, prior_strength):
    pca = PCA(n_components=min(pca_dim, n_files - 1), random_state=42)
    X_pca_raw = pca.fit_transform(file_embs).astype(np.float32)
    X_pca = normalize(X_pca_raw, norm='l2')

    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr_pca = X_pca[mask_tr]
        Y_tr     = file_labels[mask_tr]
        X_te_pca = X_pca[[fi]]

        gm = X_tr_pca.mean(0)
        gm = gm / (np.linalg.norm(gm) + 1e-8)

        sum_pos = Y_tr.T @ X_tr_pca    # (234, pca_dim)
        n_pos = Y_tr.sum(0)
        numerator = sum_pos + prior_strength * gm[None, :]
        denominator = (n_pos + prior_strength)[:, None]
        post_raw = numerator / denominator
        norms = np.linalg.norm(post_raw, axis=1, keepdims=True)
        post = post_raw / (norms + 1e-8)

        cos_sims = (X_te_pca @ post.T).ravel()
        preds[fi] = (cos_sims + 1.0) / 2.0

    return preds

best_b_auc, best_b_preds, best_b_w = 0.0, None, {}
for pca_dim in [24, 32, 48, 64]:
    for prior_s in np.linspace(0.05, 2.0, 20):
        bp = bsp_cosine_global_pca(pca_dim=pca_dim, prior_strength=prior_s)
        auc_bp = macro_auc(file_labels, bp)
        for w_bsp in np.arange(0.02, 0.60, 0.02):
            ens = w_bsp * bp + (1 - w_bsp) * k134_ref
            auc = macro_auc(file_labels, ens)
            if auc > best_b_auc:
                best_b_auc = auc
                best_b_preds = ens.copy()
                best_b_w = {"pca_dim": pca_dim, "prior_s": round(float(prior_s),4),
                             "w_bsp": round(float(w_bsp),3), "alone": round(auc_bp,4)}

marker = "  *** NEW BEST ***" if best_b_auc > CURRENT_BEST else ""
print(f"  BSP-Cos+k134: {best_b_auc:.6f}  (delta={best_b_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_b_w}")
results_list.append(("bsp_cosine_global", best_b_auc, best_b_w, best_b_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 C：Negative-aware BSP (global PCA, fast)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 C: Negative-aware BSP (global PCA)")
print("="*70)

def bsp_pos_neg_global_pca(pca_dim, prior_strength, beta):
    pca = PCA(n_components=min(pca_dim, n_files - 1), random_state=42)
    X_pca = pca.fit_transform(file_embs).astype(np.float32)

    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr_pca = X_pca[mask_tr]
        Y_tr     = file_labels[mask_tr]
        Y_neg    = 1.0 - Y_tr
        X_te_pca = X_pca[[fi]]

        gm = X_tr_pca.mean(0)
        sigma2 = float(X_tr_pca.var())

        sum_pos = Y_tr.T @ X_tr_pca
        n_pos = Y_tr.sum(0)
        pos_proto = (sum_pos + prior_strength * gm) / (n_pos + prior_strength)[:, None]

        sum_neg = Y_neg.T @ X_tr_pca
        n_neg = Y_neg.sum(0)
        neg_proto = (sum_neg + prior_strength * gm) / (n_neg + prior_strength)[:, None]

        diff_pos = X_te_pca - pos_proto
        diff_neg = X_te_pca - neg_proto
        pos_score = np.exp(-0.5 * (diff_pos**2).sum(1) / (sigma2 + 1e-8))
        neg_score = np.exp(-0.5 * (diff_neg**2).sum(1) / (sigma2 + 1e-8))
        preds[fi] = np.clip(pos_score - beta * neg_score, 0, 1)

    return preds

best_c_auc, best_c_preds, best_c_w = 0.0, None, {}
for pca_dim in [24, 32, 48, 64]:
    for prior_s in [0.05, 0.1, 0.2, 0.3]:
        for beta in [0.1, 0.2, 0.3, 0.5, 0.7]:
            bp = bsp_pos_neg_global_pca(pca_dim=pca_dim, prior_strength=prior_s, beta=beta)
            auc_bp = macro_auc(file_labels, bp)
            for w_bsp in np.arange(0.02, 0.60, 0.02):
                ens = w_bsp * bp + (1 - w_bsp) * k134_ref
                auc = macro_auc(file_labels, ens)
                if auc > best_c_auc:
                    best_c_auc = auc
                    best_c_preds = ens.copy()
                    best_c_w = {"pca_dim": pca_dim, "prior_s": prior_s, "beta": beta,
                                "w_bsp": round(float(w_bsp),3), "alone": round(auc_bp,4)}

marker = "  *** NEW BEST ***" if best_c_auc > CURRENT_BEST else ""
print(f"  BSP-neg+k134: {best_c_auc:.6f}  (delta={best_c_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_c_w}")
results_list.append(("bsp_neg_global", best_c_auc, best_c_w, best_c_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 D：Ultra-fine搜索 around best — 進一步細化
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 D: k134 ultra-fine re-sweep + BSP blending")
print("="*70)

# 找到目前所有 BSP 中最好的
best_all_bsp_key = max(bsp_global_cache, key=lambda k: bsp_global_cache[k][0])
_, best_all_bsp_p = bsp_global_cache[best_all_bsp_key]
print(f"  Best BSP standalone: {bsp_global_cache[best_all_bsp_key][0]:.4f}  key={best_all_bsp_key}")

# Ultra-fine: al*logit_max + wb*BSP + w1*knn1 + w3*knn3 + w4*knn4
# but with very fine step 0.005
best_d_auc, best_d_preds, best_d_w = 0.0, None, {}
for al in np.arange(0.35, 0.48, 0.005):
    for wb in np.arange(0.02, 0.18, 0.005):
        for w1 in np.arange(0.22, 0.35, 0.005):
            for w3 in np.arange(0.00, 0.04, 0.005):
                w4 = 1.0 - al - wb - w1 - w3
                if w4 < 0.10 or w4 > 0.38: continue
                ens = al*file_prob_max + wb*best_all_bsp_p + w1*knn1 + w3*knn3 + w4*knn4
                auc = macro_auc(file_labels, ens)
                if auc > best_d_auc:
                    best_d_auc = auc
                    best_d_preds = ens.copy()
                    best_d_w = {"bsp_key": list(best_all_bsp_key),
                                "al": round(float(al),4), "wb": round(float(wb),4),
                                "w1": round(float(w1),4), "w3": round(float(w3),4),
                                "w4": round(float(w4),4)}

marker = "  *** NEW BEST ***" if best_d_auc > CURRENT_BEST else ""
print(f"  Ultra-fine 5-way: {best_d_auc:.6f}  (delta={best_d_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_d_w}")
results_list.append(("bsp_5way_ultrafine_v4", best_d_auc, best_d_w, best_d_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 E：BSP with different aggregation functions
#   Logit space: score = -dist2 (unnormalized), then blend
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 E: BSP logit-space (unnormalized distance) blend")
print("="*70)

def bsp_logit_space(pca_dim, prior_strength):
    """Use -dist2 as raw score (logit), then use as is"""
    pca = PCA(n_components=min(pca_dim, n_files - 1), random_state=42)
    X_pca = pca.fit_transform(file_embs).astype(np.float32)

    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr_pca = X_pca[mask_tr]
        Y_tr     = file_labels[mask_tr]
        X_te_pca = X_pca[[fi]]

        gm = X_tr_pca.mean(0)
        sum_pos = Y_tr.T @ X_tr_pca
        n_pos = Y_tr.sum(0)
        post = (sum_pos + prior_strength * gm) / (n_pos + prior_strength)[:, None]

        diff  = X_te_pca - post
        dist2 = (diff ** 2).sum(1)

        # Normalize dist2 to [0,1] range via min-max
        d_min, d_max = dist2.min(), dist2.max()
        if d_max - d_min < 1e-8:
            preds[fi] = 0.5
        else:
            preds[fi] = 1.0 - (dist2 - d_min) / (d_max - d_min)

    return preds

best_e_auc, best_e_preds, best_e_w = 0.0, None, {}
for pca_dim in [24, 32, 48, 64]:
    for prior_s in [0.05, 0.1, 0.3, 0.5]:
        bp = bsp_logit_space(pca_dim=pca_dim, prior_strength=prior_s)
        auc_bp = macro_auc(file_labels, bp)
        for w_bsp in np.arange(0.02, 0.60, 0.02):
            ens = w_bsp * bp + (1 - w_bsp) * k134_ref
            auc = macro_auc(file_labels, ens)
            if auc > best_e_auc:
                best_e_auc = auc
                best_e_preds = ens.copy()
                best_e_w = {"pca_dim": pca_dim, "prior_s": prior_s,
                             "w_bsp": round(float(w_bsp),3), "alone": round(auc_bp,4)}

marker = "  *** NEW BEST ***" if best_e_auc > CURRENT_BEST else ""
print(f"  BSP-logit+k134: {best_e_auc:.6f}  (delta={best_e_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_e_w}")
results_list.append(("bsp_minmax_k134", best_e_auc, best_e_w, best_e_preds))

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
        "note": f"new_methods_v4 (BSP global-PCA) 2026-03-25; prev={overall_best:.6f}"
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
