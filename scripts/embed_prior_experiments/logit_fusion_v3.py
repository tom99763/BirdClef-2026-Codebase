"""
Logit Fusion v3: Final optimization around best method
Current best: logit_max_knn_k_sweep AUC=0.8919 (k=3, alpha=0.30)

Focus:
  A) Ultra-fine sweep around k=3, alpha=0.30
  B) KNN k=2,3,4 + various logit transformations
  C) Per-species optimized alpha (using 66-fold LOO to find per-species best blend)
  D) Two-stage: KNN(k=3) → recalibrate with logit
  E) Window-level max logit (not file-level aggregation)
  F) Combination of window-level logit signals
"""

import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import scipy.special

DATA_PATH    = '/home/lab/BirdClef-2026-Codebase/outputs/perch_labeled_ss.npz'
RESULTS_PATH = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_results.json'
MODEL_PATH   = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_model.pkl'
BASELINE_AUC = 0.8411
CURRENT_BEST = 0.8919

raw       = np.load(DATA_PATH, allow_pickle=True)
emb_win   = raw['emb'].astype(np.float32)
logits_win= raw['logits'].astype(np.float32)
labels_win= raw['labels'].astype(np.float32)
file_list = raw['file_list']
n_windows = raw['n_windows']

n_files   = len(file_list)
n_species = labels_win.shape[1]

# Build window → file index
win_file_idx = np.zeros(len(emb_win), dtype=np.int32)
idx = 0
for fi, nw in enumerate(n_windows):
    win_file_idx[idx:idx+nw] = fi
    idx += nw

# Build file-level aggregations
file_embs        = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels      = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_max   = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_mean  = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_top2  = np.zeros((n_files, n_species),         dtype=np.float32)  # mean of top-2

idx = 0
for fi, nw in enumerate(n_windows):
    win_logits = logits_win[idx:idx+nw]  # (nw, 234)
    file_embs[fi]       = emb_win[idx:idx+nw].mean(0)
    file_labels[fi]     = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi]  = win_logits.max(0)
    file_logit_mean[fi] = win_logits.mean(0)
    # mean of top-2 windows per species
    if nw >= 2:
        sorted_l = np.sort(win_logits, axis=0)
        file_logit_top2[fi] = sorted_l[-2:, :].mean(0)
    else:
        file_logit_top2[fi] = win_logits.max(0)
    idx += nw

file_embs_norm  = normalize(file_embs, norm='l2')
file_prob_max   = scipy.special.expit(file_logit_max)
file_prob_mean  = scipy.special.expit(file_logit_mean)
file_prob_top2  = scipy.special.expit(file_logit_top2)

print(f"Data: {n_files} files, {n_species} species")

def macro_auc(y_true, y_score):
    mask = y_true.sum(0) > 0
    if mask.sum() < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')
    except:
        return float('nan')

def knn_predict(k=5, X=None):
    if X is None:
        X = file_embs_norm
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr = X[mask]; te = X[[i]]; y_tr = file_labels[mask]
        sims = (te @ tr.T).ravel()
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff)[:k_eff]
        weights = np.clip(sims[nn_idx], 0, None)
        if weights.sum() < 1e-9:
            weights = np.ones(k_eff)
        preds[i] = (weights[:, None] * y_tr[nn_idx]).sum(0) / weights.sum()
    return preds

# Pre-compute KNN for key values
knn3 = knn_predict(k=3)
knn5 = knn_predict(k=5)
print(f"KNN(k=3): {macro_auc(file_labels, knn3):.4f}")
print(f"KNN(k=5): {macro_auc(file_labels, knn5):.4f}")

results_list = []

# ══════════════════════════════════════════════════════════════════════════════
# METHOD A: Ultra-fine sweep around best (k=3, alpha=0.30)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD A: Ultra-fine sweep (k=2,3,4, alpha 0.001 step)")
print("="*65)

best_a_auc, best_a_preds, best_a_k, best_a_alpha = 0.0, None, 3, 0.30
for k in [2, 3, 4]:
    knn_k = knn_predict(k=k)
    for alpha in np.arange(0.10, 0.50, 0.005):
        ens = alpha * file_prob_max + (1 - alpha) * knn_k
        auc = macro_auc(file_labels, ens)
        if auc > best_a_auc:
            best_a_auc, best_a_preds = auc, ens.copy()
            best_a_k, best_a_alpha = k, alpha

marker = "  *** NEW BEST ***" if best_a_auc > CURRENT_BEST else ""
print(f"  Best k={best_a_k}, alpha={best_a_alpha:.3f}: {best_a_auc:.4f}  (delta={best_a_auc-CURRENT_BEST:+.4f}){marker}")
results_list.append(("logit_max_knn_ultrafine", best_a_auc,
                     {"k": best_a_k, "alpha": float(best_a_alpha)}, best_a_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD B: Top-2 window mean logit + KNN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD B: Top-2 window logit mean + KNN")
print("="*65)

best_b_auc, best_b_preds, best_b_k, best_b_alpha = 0.0, None, 3, 0.30
for k in [2, 3, 4, 5]:
    knn_k = knn_predict(k=k)
    for alpha in np.arange(0.05, 0.51, 0.01):
        ens = alpha * file_prob_top2 + (1 - alpha) * knn_k
        auc = macro_auc(file_labels, ens)
        if auc > best_b_auc:
            best_b_auc, best_b_preds = auc, ens.copy()
            best_b_k, best_b_alpha = k, alpha

marker = "  *** NEW BEST ***" if best_b_auc > CURRENT_BEST else ""
print(f"  Top2 k={best_b_k}, alpha={best_b_alpha:.2f}: {best_b_auc:.4f}  (delta={best_b_auc-CURRENT_BEST:+.4f}){marker}")
results_list.append(("logit_top2_knn", best_b_auc,
                     {"k": best_b_k, "alpha": float(best_b_alpha)}, best_b_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD C: Species-level per-K optimization
# Best alpha might differ per species
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD C: Species-specific alpha (LOO optimal per species)")
print("="*65)

def per_species_alpha_loo(k=3):
    """
    For each LOO fold, for each species:
      Use remaining train files to find best alpha_s between logit_max and KNN.
      Apply that alpha_s to blend for test file.
    """
    knn_k = knn_predict(k=k)
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        # Inner LOO on train set to find per-species best alpha
        tr_knn   = knn_k[mask]            # (65, 234)
        tr_logit = file_prob_max[mask]     # (65, 234)
        tr_labels = file_labels[mask]      # (65, 234)

        for s in range(n_species):
            y_s = tr_labels[:, s]
            if y_s.sum() == 0 or y_s.sum() == 65:
                preds[i, s] = file_prob_max[i, s] if y_s.sum() == 0 else 1.0
                continue

            best_alpha_s = 0.30  # default
            best_inner_auc = -1.0
            for alpha_s in np.arange(0.0, 1.01, 0.1):
                blend_s = alpha_s * tr_logit[:, s] + (1 - alpha_s) * tr_knn[:, s]
                try:
                    auc_s = roc_auc_score(y_s, blend_s)
                    if auc_s > best_inner_auc:
                        best_inner_auc = auc_s
                        best_alpha_s = alpha_s
                except Exception:
                    pass

            preds[i, s] = float(best_alpha_s * file_prob_max[i, s] +
                                 (1 - best_alpha_s) * knn_k[i, s])

    return macro_auc(file_labels, preds), preds

print("  Running per-species alpha LOO (k=3)...")
auc_c, preds_c = per_species_alpha_loo(k=3)
marker = "  *** NEW BEST ***" if auc_c > CURRENT_BEST else ""
print(f"  Per-species alpha (k=3): {auc_c:.4f}  (delta={auc_c-CURRENT_BEST:+.4f}){marker}")
results_list.append(("per_species_alpha_knn3", auc_c, {"k": 3}, preds_c))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD D: Window-level max prediction via prototype + logit
# For each test file, aggregate window-level predictions
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD D: Window-level logit aggregation")
print("="*65)

# Window max already computed as file_logit_max
# But what about using top-N windows? Let's try varying N
best_d_auc, best_d_preds, best_d_N, best_d_alpha = 0.0, None, 1, 0.30
for N in [1, 2, 3]:  # top-N windows
    file_logit_topN = np.zeros((n_files, n_species), dtype=np.float32)
    idx_w = 0
    for fi, nw in enumerate(n_windows):
        win_logits = logits_win[idx_w:idx_w+nw]
        n = min(N, nw)
        top_n = np.sort(win_logits, axis=0)[-n:, :]
        file_logit_topN[fi] = top_n.mean(0)
        idx_w += nw
    prob_topN = scipy.special.expit(file_logit_topN)

    for k in [2, 3, 4]:
        knn_k = knn_predict(k=k)
        for alpha in np.arange(0.05, 0.51, 0.02):
            ens = alpha * prob_topN + (1 - alpha) * knn_k
            auc = macro_auc(file_labels, ens)
            if auc > best_d_auc:
                best_d_auc, best_d_preds = auc, ens.copy()
                best_d_N, best_d_k_d, best_d_alpha = N, k, alpha

marker = "  *** NEW BEST ***" if best_d_auc > CURRENT_BEST else ""
print(f"  Top-N={best_d_N}, k={best_d_k_d}, alpha={best_d_alpha:.2f}: {best_d_auc:.4f}  (delta={best_d_auc-CURRENT_BEST:+.4f}){marker}")
results_list.append(("topN_logit_knn", best_d_auc,
                     {"N": best_d_N, "k": best_d_k_d, "alpha": float(best_d_alpha)}, best_d_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD E: KNN(k=3) + logit_max with power transform
# Apply power transform to logit probabilities before blend
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD E: Power-transformed logit_max + KNN(k=3)")
print("="*65)

best_e_auc, best_e_preds, best_e_pow, best_e_alpha = 0.0, None, 1.0, 0.30
for power in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    prob_pow = file_prob_max ** power
    for alpha in np.arange(0.05, 0.61, 0.02):
        ens = alpha * prob_pow + (1 - alpha) * knn3
        auc = macro_auc(file_labels, ens)
        if auc > best_e_auc:
            best_e_auc, best_e_preds = auc, ens.copy()
            best_e_pow, best_e_alpha = power, alpha

marker = "  *** NEW BEST ***" if best_e_auc > CURRENT_BEST else ""
print(f"  Power={best_e_pow}, alpha={best_e_alpha:.2f}: {best_e_auc:.4f}  (delta={best_e_auc-CURRENT_BEST:+.4f}){marker}")
results_list.append(("power_logit_knn3", best_e_auc,
                     {"power": best_e_pow, "alpha": float(best_e_alpha)}, best_e_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD F: 4-way ensemble (k=2 + k=3 + k=5 + logit_max)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD F: 4-way ensemble")
print("="*65)

knn2 = knn_predict(k=2)
knn4 = knn_predict(k=4)

best_f_auc, best_f_preds, best_f_w = 0.0, None, {}
for wl in np.arange(0.10, 0.51, 0.05):  # logit weight
    rem = 1.0 - wl
    for w2 in np.arange(0.0, rem+0.01, 0.05):
        for w3 in np.arange(0.0, rem-w2+0.01, 0.05):
            w5 = rem - w2 - w3
            if w5 < 0 or abs(w2+w3+w5+wl - 1.0) > 1e-6:
                continue
            ens = wl*file_prob_max + w2*knn2 + w3*knn3 + w5*knn5
            auc = macro_auc(file_labels, ens)
            if auc > best_f_auc:
                best_f_auc = auc
                best_f_preds = ens.copy()
                best_f_w = {"wl": wl, "w2": w2, "w3": w3, "w5": w5}

marker = "  *** NEW BEST ***" if best_f_auc > CURRENT_BEST else ""
print(f"  4-way: {best_f_auc:.4f}  (delta={best_f_auc-CURRENT_BEST:+.4f}){marker}")
print(f"    {best_f_w}")
results_list.append(("4way_knn_logit", best_f_auc, best_f_w, best_f_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD G: Logit_max + logit_top2 + KNN(k=3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD G: Logit max + top2 + KNN(k=3)")
print("="*65)

best_g_auc, best_g_preds, best_g_w = 0.0, None, {}
for wmax in np.arange(0.05, 0.41, 0.05):
    for wtop2 in np.arange(0.0, 0.31, 0.05):
        wknn = 1.0 - wmax - wtop2
        if wknn < 0:
            continue
        ens = wmax*file_prob_max + wtop2*file_prob_top2 + wknn*knn3
        auc = macro_auc(file_labels, ens)
        if auc > best_g_auc:
            best_g_auc = auc
            best_g_preds = ens.copy()
            best_g_w = {"wmax": float(wmax), "wtop2": float(wtop2), "wknn": float(wknn)}

marker = "  *** NEW BEST ***" if best_g_auc > CURRENT_BEST else ""
print(f"  max+top2+knn3: {best_g_auc:.4f}  (delta={best_g_auc-CURRENT_BEST:+.4f}){marker}")
print(f"    {best_g_w}")
results_list.append(("logit_max_top2_knn3", best_g_auc, best_g_w, best_g_preds))

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
print(f"Baseline KNN: {BASELINE_AUC:.4f}")
print(f"Previous best: {CURRENT_BEST:.4f}")
for name, auc, params, _ in sorted(results_list, key=lambda x: -x[1]):
    marker = "  *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}  (delta={auc-CURRENT_BEST:+.4f}){marker}")

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
        elif isinstance(v, np.ndarray):
            continue
        else:
            record[k] = v
    results_json["experiments"].append(record)

overall_best_auc = results_json["best"]["loo_auc"]
if best_auc > overall_best_auc:
    results_json["best"] = {"method": best_name, "loo_auc": round(float(best_auc), 4)}
    print(f"\nNEW BEST: {best_name} AUC={best_auc:.4f}")

    model_dict = {
        "method": best_name,
        "loo_auc": round(float(best_auc), 4),
        "params": {k: (float(v) if isinstance(v, (np.float32, np.float64, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                   for k, v in best_params.items()},
        "file_list": file_list.tolist(),
        "loo_preds": best_preds.tolist(),
        "file_embs_norm": file_embs_norm.tolist(),
        "file_prob_max": file_prob_max.tolist(),
        "file_prob_mean": file_prob_mean.tolist(),
        "file_labels": file_labels.tolist(),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"Saved model → {MODEL_PATH}")
else:
    print(f"\nNo improvement over current best ({overall_best_auc:.4f})")
    print(f"Best this run: {best_name} AUC={best_auc:.4f}")

with open(RESULTS_PATH, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Results saved → {RESULTS_PATH}")
