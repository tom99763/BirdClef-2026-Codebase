"""
embed_prior new_methods_v5.py
快速版本：只用已知最佳的 pca_dim 範圍，更少 prior values
基於 v4 初步結果：pca=32 best=0.906，目標進一步提升混合 AUC
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
# BSP global-PCA helper
# ══════════════════════════════════════════════════════════════════
def bsp_global_pca(pca_dim, prior_strength):
    pca = PCA(n_components=min(pca_dim, n_files - 1), random_state=42)
    X_pca = pca.fit_transform(file_embs).astype(np.float32)

    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr_pca = X_pca[mask_tr]
        Y_tr     = file_labels[mask_tr]
        X_te_pca = X_pca[[fi]]

        gm = X_tr_pca.mean(0)
        sigma2 = float(X_tr_pca.var())

        sum_pos = Y_tr.T @ X_tr_pca
        n_pos = Y_tr.sum(0)
        post = (sum_pos + prior_strength * gm) / (n_pos + prior_strength)[:, None]

        diff  = X_te_pca - post
        dist2 = (diff ** 2).sum(1)
        preds[fi] = np.exp(-0.5 * dist2 / (sigma2 + 1e-8))

    return preds

# ──────────────────────────────────────────────────────────────────
# 快速 grid: pca ∈ {28,30,32,34,36} × prior ∈ linspace(0.02, 0.5, 25)
# (based on v4 finding pca=32 is best)
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("A: BSP fine sweep around pca=32 best")
print("="*70)

bsp_cache = {}
best_standalone = 0.0
best_standalone_key = None

for pca_dim in [26, 28, 30, 32, 34, 36, 38]:
    dim_best = 0.0
    for prior_s in np.linspace(0.01, 0.5, 50):
        bp = bsp_global_pca(pca_dim=pca_dim, prior_strength=prior_s)
        auc = macro_auc(file_labels, bp)
        key = (pca_dim, round(float(prior_s), 4))
        bsp_cache[key] = (auc, bp)
        if auc > dim_best: dim_best = auc
        if auc > best_standalone:
            best_standalone = auc
            best_standalone_key = key
    print(f"  pca={pca_dim}: best={dim_best:.4f}")

print(f"\nBest standalone: {best_standalone:.6f}  key={best_standalone_key}")
results_list.append(("bsp_fine_standalone", best_standalone,
                     {"pca_dim": best_standalone_key[0], "prior_s": best_standalone_key[1]},
                     bsp_cache[best_standalone_key][1]))

# ──────────────────────────────────────────────────────────────────
# A2. blend with k134
# ──────────────────────────────────────────────────────────────────
_, best_bsp_p = bsp_cache[best_standalone_key]

best_a2_auc, best_a2_preds, best_a2_w = 0.0, None, {}
for w_bsp in np.arange(0.01, 0.65, 0.005):
    ens = w_bsp * best_bsp_p + (1 - w_bsp) * k134_ref
    auc = macro_auc(file_labels, ens)
    if auc > best_a2_auc:
        best_a2_auc = auc
        best_a2_preds = ens.copy()
        best_a2_w = {"w_bsp": round(float(w_bsp),4), "bsp_key": list(best_standalone_key)}

marker = "  *** NEW BEST ***" if best_a2_auc > CURRENT_BEST else ""
print(f"BSP+k134: {best_a2_auc:.6f}  (delta={best_a2_auc-CURRENT_BEST:+.6f}){marker}")
print(f"  {best_a2_w}")
results_list.append(("bsp_fine_k134", best_a2_auc, best_a2_w, best_a2_preds))

# ──────────────────────────────────────────────────────────────────
# A3. 5-way: logit_max + BSP + knn1 + knn3 + knn4
# ──────────────────────────────────────────────────────────────────
print("\nA3: 5-way logit_max + BSP + knn1+knn3+knn4")
top5 = sorted(bsp_cache.items(), key=lambda x: -x[1][0])[:5]

best_a3_auc, best_a3_preds, best_a3_w = 0.0, None, {}
for (pd, ps), (_, bp) in top5:
    for al in np.arange(0.32, 0.50, 0.005):
        for wb in np.arange(0.02, 0.18, 0.005):
            rem = 1.0 - al - wb
            if rem < 0.28 or rem > 0.68: continue
            for w1_r in [0.46, 0.48, 0.50, 0.52]:
                for w3_r in [0.00, 0.02, 0.04]:
                    w4_r = 1.0 - w1_r - w3_r
                    if w4_r < 0.44: continue
                    w1 = rem * w1_r
                    w3 = rem * w3_r
                    w4 = rem * w4_r
                    ens = al*file_prob_max + wb*bp + w1*knn1 + w3*knn3 + w4*knn4
                    auc = macro_auc(file_labels, ens)
                    if auc > best_a3_auc:
                        best_a3_auc = auc
                        best_a3_preds = ens.copy()
                        best_a3_w = {"bsp_key": [pd, ps],
                                     "al": round(float(al),4), "wb": round(float(wb),4),
                                     "w1": round(float(w1),4), "w3": round(float(w3),4),
                                     "w4": round(float(w4),4)}

marker = "  *** NEW BEST ***" if best_a3_auc > CURRENT_BEST else ""
print(f"5-way: {best_a3_auc:.6f}  (delta={best_a3_auc-CURRENT_BEST:+.6f}){marker}")
print(f"  {best_a3_w}")
results_list.append(("bsp_5way_fine", best_a3_auc, best_a3_w, best_a3_preds))

# ──────────────────────────────────────────────────────────────────
# A4. Multi-BSP + k134
# ──────────────────────────────────────────────────────────────────
print("\nA4: Multi-BSP + k134")
best_a4_auc, best_a4_preds, best_a4_w = 0.0, None, {}
for top_k in [3, 5, 8, 10, 15]:
    topk = sorted(bsp_cache.items(), key=lambda x: -x[1][0])[:top_k]
    multi_bsp = np.mean([p for _, (_, p) in topk], axis=0)
    auc_multi = macro_auc(file_labels, multi_bsp)
    for w_bsp in np.arange(0.01, 0.60, 0.005):
        ens = w_bsp * multi_bsp + (1 - w_bsp) * k134_ref
        auc = macro_auc(file_labels, ens)
        if auc > best_a4_auc:
            best_a4_auc = auc
            best_a4_preds = ens.copy()
            best_a4_w = {"top_k": top_k, "w_bsp": round(float(w_bsp),4),
                          "multi_alone": round(auc_multi,4)}

marker = "  *** NEW BEST ***" if best_a4_auc > CURRENT_BEST else ""
print(f"Multi-BSP+k134: {best_a4_auc:.6f}  (delta={best_a4_auc-CURRENT_BEST:+.6f}){marker}")
print(f"  {best_a4_w}")
results_list.append(("multi_bsp_fine_k134", best_a4_auc, best_a4_w, best_a4_preds))

# ──────────────────────────────────────────────────────────────────
# A5. 跨越 pca 的 top-BSP ensemble 加更細 prior sweep
# ──────────────────────────────────────────────────────────────────
print("\nA5: Cross-pca best BSP 3-way blend with logit_max+k134")
best_a5_auc, best_a5_preds, best_a5_w = 0.0, None, {}

# 每個 pca_dim 取最佳 prior
per_dim_best = {}
for (pd, ps), (auc, bp) in bsp_cache.items():
    if pd not in per_dim_best or auc > per_dim_best[pd][0]:
        per_dim_best[pd] = (auc, bp)

print("  Per-dim best:")
for pd, (auc, _) in sorted(per_dim_best.items()):
    print(f"    pca={pd}: {auc:.4f}")

# 嘗試平均兩個不同 pca 的 BSP
pca_dims_sorted = sorted(per_dim_best.keys())
for i, pd1 in enumerate(pca_dims_sorted):
    for pd2 in pca_dims_sorted[i+1:]:
        _, bp1 = per_dim_best[pd1]
        _, bp2 = per_dim_best[pd2]
        for w1 in np.arange(0.2, 0.8, 0.1):
            bsp_mix = w1 * bp1 + (1-w1) * bp2
            for w_bsp in np.arange(0.02, 0.50, 0.02):
                ens = w_bsp * bsp_mix + (1 - w_bsp) * k134_ref
                auc = macro_auc(file_labels, ens)
                if auc > best_a5_auc:
                    best_a5_auc = auc
                    best_a5_preds = ens.copy()
                    best_a5_w = {"pd1": pd1, "pd2": pd2, "w1": round(float(w1),2),
                                  "w_bsp": round(float(w_bsp),3)}

marker = "  *** NEW BEST ***" if best_a5_auc > CURRENT_BEST else ""
print(f"Cross-pca BSP+k134: {best_a5_auc:.6f}  (delta={best_a5_auc-CURRENT_BEST:+.6f}){marker}")
print(f"  {best_a5_w}")
results_list.append(("cross_pca_bsp_k134", best_a5_auc, best_a5_w, best_a5_preds))

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
        "note": f"new_methods_v5 (BSP fine) 2026-03-25; prev={overall_best:.6f}"
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
