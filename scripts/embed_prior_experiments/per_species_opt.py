"""
Optimizing per_species_alpha method (current best: 0.9026)
- Try different K for KNN
- Try finer alpha grid in per-species optimization
- Combine per-species alpha with 4-way ensemble
- Try per-species logit_max vs logit_mean vs logit_top2
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
CURRENT_BEST = 0.9026

raw       = np.load(DATA_PATH, allow_pickle=True)
emb_win   = raw['emb'].astype(np.float32)
logits_win= raw['logits'].astype(np.float32)
labels_win= raw['labels'].astype(np.float32)
file_list = raw['file_list']
n_windows = raw['n_windows']

n_files   = len(file_list)
n_species = labels_win.shape[1]

# Build file-level data
file_embs       = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels     = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_max  = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_mean = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_top2 = np.zeros((n_files, n_species),         dtype=np.float32)

idx = 0
for fi, nw in enumerate(n_windows):
    win_logits = logits_win[idx:idx+nw]
    file_embs[fi]       = emb_win[idx:idx+nw].mean(0)
    file_labels[fi]     = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi]  = win_logits.max(0)
    file_logit_mean[fi] = win_logits.mean(0)
    if nw >= 2:
        file_logit_top2[fi] = np.sort(win_logits, axis=0)[-2:].mean(0)
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

results_list = []

# ══════════════════════════════════════════════════════════════════════════════
# METHOD A: Per-species alpha with different K values
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD A: Per-species alpha, varying K")
print("="*65)

def per_species_alpha_loo(k=3, alpha_grid=None, logit_preds=None):
    """
    For each LOO fold and species:
      Inner LOO on 65 train files to find per-species best blend alpha.
    """
    if alpha_grid is None:
        alpha_grid = np.arange(0.0, 1.01, 0.1)
    if logit_preds is None:
        logit_preds = file_prob_max

    knn_all = knn_predict(k=k)
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_knn   = knn_all[mask]           # (65, 234)
        tr_logit = logit_preds[mask]       # (65, 234)
        tr_labels = file_labels[mask]      # (65, 234)

        for s in range(n_species):
            y_s = tr_labels[:, s]
            if y_s.sum() == 0:
                preds[i, s] = 0.0
                continue
            if y_s.sum() == 65:
                preds[i, s] = 1.0
                continue

            best_alpha_s = 0.30
            best_inner_auc = -1.0
            for alpha_s in alpha_grid:
                blend_s = alpha_s * tr_logit[:, s] + (1 - alpha_s) * tr_knn[:, s]
                try:
                    auc_s = roc_auc_score(y_s, blend_s)
                    if auc_s > best_inner_auc:
                        best_inner_auc = auc_s
                        best_alpha_s = alpha_s
                except Exception:
                    pass

            preds[i, s] = float(best_alpha_s * logit_preds[i, s] +
                                 (1 - best_alpha_s) * knn_all[i, s])

    return macro_auc(file_labels, preds), preds

# Try different K values
best_a_auc, best_a_preds, best_a_k = 0.0, None, 3
for k in [1, 2, 3, 4, 5, 7]:
    auc, preds = per_species_alpha_loo(k=k)
    marker = "  *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  Per-species alpha k={k}: {auc:.4f}  (delta={auc-CURRENT_BEST:+.4f}){marker}")
    if auc > best_a_auc:
        best_a_auc, best_a_preds, best_a_k = auc, preds, k

results_list.append(("per_species_alpha_k_sweep", best_a_auc, {"k": best_a_k}, best_a_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD B: Per-species alpha with finer grid
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD B: Per-species alpha with fine grid (0.05 step)")
print("="*65)

fine_grid = np.arange(0.0, 1.01, 0.05)
auc_b, preds_b = per_species_alpha_loo(k=best_a_k, alpha_grid=fine_grid)
marker = "  *** NEW BEST ***" if auc_b > CURRENT_BEST else ""
print(f"  Per-species alpha fine grid (k={best_a_k}): {auc_b:.4f}  (delta={auc_b-CURRENT_BEST:+.4f}){marker}")
results_list.append(("per_species_alpha_fine", auc_b, {"k": best_a_k, "alpha_step": 0.05}, preds_b))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD C: Per-species alpha using logit_mean instead of logit_max
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD C: Per-species alpha with logit_mean (k=3)")
print("="*65)

auc_c, preds_c = per_species_alpha_loo(k=3, logit_preds=file_prob_mean)
marker = "  *** NEW BEST ***" if auc_c > CURRENT_BEST else ""
print(f"  Per-species alpha logit_mean (k=3): {auc_c:.4f}  (delta={auc_c-CURRENT_BEST:+.4f}){marker}")
results_list.append(("per_species_alpha_mean", auc_c, {"k": 3, "logit": "mean"}, preds_c))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD D: Ensemble of per-species alpha (logit_max) + per-species alpha (logit_mean)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD D: Ensemble of per-species alpha variants")
print("="*65)

best_d_auc, best_d_preds, best_d_alpha = 0.0, None, 0.5
for alpha in np.arange(0.0, 1.01, 0.05):
    # Use best from method A and preds_c (logit_mean)
    ens = alpha * best_a_preds + (1 - alpha) * preds_c
    auc = macro_auc(file_labels, ens)
    if auc > best_d_auc:
        best_d_auc, best_d_preds, best_d_alpha = auc, ens.copy(), alpha

marker = "  *** NEW BEST ***" if best_d_auc > CURRENT_BEST else ""
print(f"  Ensemble max+mean per-species (alpha={best_d_alpha:.2f}): {best_d_auc:.4f}  (delta={best_d_auc-CURRENT_BEST:+.4f}){marker}")
results_list.append(("ensemble_ps_alpha_max_mean", best_d_auc, {"alpha_max": float(best_d_alpha)}, best_d_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD E: Per-species alpha but using both KNN neighbors and logit
# Also optimize which logit source per species (max vs mean)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD E: Per-species adaptive (best of logit_max, logit_mean, KNN)")
print("="*65)

def per_species_adaptive_loo(k=3):
    """
    For each species, also try using just logit_max or logit_mean without KNN blend.
    Pick whatever gives best inner-LOO AUC.
    """
    knn_all = knn_predict(k=k)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    alpha_grid = np.arange(0.0, 1.01, 0.1)

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_knn      = knn_all[mask]
        tr_logitmax = file_prob_max[mask]
        tr_logitmean= file_prob_mean[mask]
        tr_labels   = file_labels[mask]

        for s in range(n_species):
            y_s = tr_labels[:, s]
            if y_s.sum() == 0:
                preds[i, s] = 0.0
                continue
            if y_s.sum() == 65:
                preds[i, s] = 1.0
                continue

            best_alpha_s = 0.30
            best_use_max = True  # which logit source
            best_inner_auc = -1.0

            for use_max in [True, False]:
                tr_logit = tr_logitmax if use_max else tr_logitmean
                pred_te  = file_prob_max[i, s] if use_max else file_prob_mean[i, s]
                for alpha_s in alpha_grid:
                    blend_s = alpha_s * tr_logit[:, s] + (1 - alpha_s) * tr_knn[:, s]
                    try:
                        auc_s = roc_auc_score(y_s, blend_s)
                        if auc_s > best_inner_auc:
                            best_inner_auc = auc_s
                            best_alpha_s = alpha_s
                            best_use_max = use_max
                    except Exception:
                        pass

            logit_pred = file_prob_max[i, s] if best_use_max else file_prob_mean[i, s]
            preds[i, s] = float(best_alpha_s * logit_pred + (1 - best_alpha_s) * knn_all[i, s])

    return macro_auc(file_labels, preds), preds

auc_e, preds_e = per_species_adaptive_loo(k=3)
marker = "  *** NEW BEST ***" if auc_e > CURRENT_BEST else ""
print(f"  Per-species adaptive (max/mean, k=3): {auc_e:.4f}  (delta={auc_e-CURRENT_BEST:+.4f}){marker}")
results_list.append(("per_species_adaptive_logit", auc_e, {"k": 3}, preds_e))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD F: Per-species alpha + 4-way KNN ensemble
# Use per-species alpha but blend with multi-K KNN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD F: Per-species alpha on multi-K KNN")
print("="*65)

def per_species_multik_loo(k_list=[2, 3, 5]):
    """
    KNN prediction = mean of K=2, K=3, K=5 predictions.
    Then per-species optimal blend with logit_max.
    """
    knn_preds_list = [knn_predict(k=k) for k in k_list]
    knn_ensemble = np.mean(knn_preds_list, axis=0)

    preds = np.zeros((n_files, n_species), dtype=np.float32)
    alpha_grid = np.arange(0.0, 1.01, 0.1)

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_knn   = knn_ensemble[mask]
        tr_logit = file_prob_max[mask]
        tr_labels = file_labels[mask]

        for s in range(n_species):
            y_s = tr_labels[:, s]
            if y_s.sum() == 0:
                preds[i, s] = 0.0
                continue
            if y_s.sum() == 65:
                preds[i, s] = 1.0
                continue

            best_alpha_s = 0.30
            best_inner_auc = -1.0
            for alpha_s in alpha_grid:
                blend_s = alpha_s * tr_logit[:, s] + (1 - alpha_s) * tr_knn[:, s]
                try:
                    auc_s = roc_auc_score(y_s, blend_s)
                    if auc_s > best_inner_auc:
                        best_inner_auc = auc_s
                        best_alpha_s = alpha_s
                except Exception:
                    pass

            preds[i, s] = float(best_alpha_s * file_prob_max[i, s] +
                                 (1 - best_alpha_s) * knn_ensemble[i, s])

    return macro_auc(file_labels, preds), preds

auc_f, preds_f = per_species_multik_loo(k_list=[2, 3, 5])
marker = "  *** NEW BEST ***" if auc_f > CURRENT_BEST else ""
print(f"  Per-species alpha on multi-K KNN: {auc_f:.4f}  (delta={auc_f-CURRENT_BEST:+.4f}){marker}")
results_list.append(("per_species_multik_knn", auc_f, {"k_list": [2, 3, 5]}, preds_f))

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

    # Save final model
    model_dict = {
        "method": best_name,
        "loo_auc": round(float(best_auc), 4),
        "params": {k: (float(v) if isinstance(v, (np.float32, np.float64, float)) else
                       int(v) if isinstance(v, (np.integer, int)) else v)
                   for k, v in best_params.items() if not isinstance(v, np.ndarray)},
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
