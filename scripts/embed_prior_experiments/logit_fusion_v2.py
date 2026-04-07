"""
Logit Fusion v2: Fine-tune the best method (logit_max_knn_ensemble)
and try more advanced variants.

Current best: logit_max_knn_ensemble AUC=0.8868 (alpha_logit_max=0.20, k=5)

New experiments:
  A) Fine-grained alpha sweep (0.01 step) around 0.20
  B) Varying K in KNN + logit_max
  C) Logit percentile (soft max): for each file, take top-P% window logits
  D) Window-level vs file-level logit (maybe max at window level is better)
  E) Combination of KNN(k=3) + KNN(k=5) + logit_max (3-way optimal)
  F) Per-species Bayesian update using logit as prior, KNN as likelihood
  G) Temperature scaling on logit_max
  H) log(1+exp(logit)) - softplus transformation
  I) KNN where distance uses both embedding AND logit similarity
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
CURRENT_BEST = 0.8868  # logit_max_knn_ensemble

raw       = np.load(DATA_PATH, allow_pickle=True)
emb_win   = raw['emb'].astype(np.float32)
logits_win= raw['logits'].astype(np.float32)
labels_win= raw['labels'].astype(np.float32)
file_list = raw['file_list']
n_windows = raw['n_windows']

n_files   = len(file_list)
n_species = labels_win.shape[1]

# Build file-level data
file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_mean= np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_p90 = np.zeros((n_files, n_species),         dtype=np.float32)  # 90th percentile
file_logit_p75 = np.zeros((n_files, n_species),         dtype=np.float32)  # 75th percentile

# Window-level file assignments
win_file_idx = []
idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]       = emb_win[idx:idx+nw].mean(0)
    file_labels[fi]     = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    file_logit_mean[fi] = logits_win[idx:idx+nw].mean(0)
    file_logit_max[fi]  = logits_win[idx:idx+nw].max(0)
    file_logit_p90[fi]  = np.percentile(logits_win[idx:idx+nw], 90, axis=0)
    file_logit_p75[fi]  = np.percentile(logits_win[idx:idx+nw], 75, axis=0)
    win_file_idx.extend([fi] * int(nw))
    idx += nw

win_file_idx = np.array(win_file_idx)
file_embs_norm = normalize(file_embs, norm='l2')

# Probabilities
file_prob_mean = scipy.special.expit(file_logit_mean)
file_prob_max  = scipy.special.expit(file_logit_max)
file_prob_p90  = scipy.special.expit(file_logit_p90)
file_prob_p75  = scipy.special.expit(file_logit_p75)

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

knn5 = knn_predict(k=5)
print(f"KNN(k=5) baseline: {macro_auc(file_labels, knn5):.4f}")

results_list = []

# ══════════════════════════════════════════════════════════════════════════════
# METHOD A: Fine-grained alpha sweep for logit_max + KNN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD A: Fine-grained alpha sweep (logit_max + KNN)")
print("="*65)

best_a_auc, best_a_alpha = 0.0, 0.20
for alpha in np.arange(0.01, 0.50, 0.01):
    ens = alpha * file_prob_max + (1 - alpha) * knn5
    auc = macro_auc(file_labels, ens)
    if auc > best_a_auc:
        best_a_auc, best_a_alpha = auc, alpha
        best_a_preds = ens.copy()

marker = "  *** NEW BEST ***" if best_a_auc > CURRENT_BEST else ""
print(f"  Best alpha={best_a_alpha:.2f}: {best_a_auc:.4f}  (delta_over_prev={best_a_auc-CURRENT_BEST:+.4f}){marker}")
results_list.append(("logit_max_knn_fine", best_a_auc, {"alpha": float(best_a_alpha), "k": 5}, best_a_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD B: Varying K + best alpha for each K
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD B: Varying K + optimal alpha (logit_max + KNN)")
print("="*65)

best_b_auc, best_b_preds, best_b_k, best_b_alpha = 0.0, None, 5, 0.20
for k in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
    knn_k = knn_predict(k=k)
    for alpha in np.arange(0.0, 0.51, 0.02):
        ens = alpha * file_prob_max + (1 - alpha) * knn_k
        auc = macro_auc(file_labels, ens)
        if auc > best_b_auc:
            best_b_auc, best_b_preds = auc, ens.copy()
            best_b_k, best_b_alpha = k, alpha

marker = "  *** NEW BEST ***" if best_b_auc > CURRENT_BEST else ""
print(f"  Best k={best_b_k}, alpha={best_b_alpha:.2f}: {best_b_auc:.4f}  (delta={best_b_auc-CURRENT_BEST:+.4f}){marker}")
results_list.append(("logit_max_knn_k_sweep", best_b_auc,
                     {"k": best_b_k, "alpha": float(best_b_alpha)}, best_b_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD C: Percentile logit (P50/P75/P90) + KNN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD C: Logit percentile variants + KNN")
print("="*65)

for pname, prob_p in [("p75", file_prob_p75), ("p90", file_prob_p90)]:
    best_c_auc, best_c_alpha = 0.0, 0.20
    best_c_preds = None
    for alpha in np.arange(0.0, 0.51, 0.02):
        ens = alpha * prob_p + (1 - alpha) * knn5
        auc = macro_auc(file_labels, ens)
        if auc > best_c_auc:
            best_c_auc, best_c_alpha, best_c_preds = auc, alpha, ens.copy()
    marker = "  *** NEW BEST ***" if best_c_auc > CURRENT_BEST else ""
    print(f"  {pname} best alpha={best_c_alpha:.2f}: {best_c_auc:.4f}  (delta={best_c_auc-CURRENT_BEST:+.4f}){marker}")
    results_list.append((f"logit_{pname}_knn", best_c_auc, {"alpha": float(best_c_alpha), "percentile": pname}, best_c_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD D: Temperature-scaled logit_max + KNN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD D: Temperature-scaled logit_max + KNN")
print("="*65)

best_d_auc, best_d_preds, best_d_T, best_d_alpha = 0.0, None, 1.0, 0.20
for T in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    prob_T = scipy.special.expit(file_logit_max / T)
    for alpha in np.arange(0.0, 0.51, 0.02):
        ens = alpha * prob_T + (1 - alpha) * knn5
        auc = macro_auc(file_labels, ens)
        if auc > best_d_auc:
            best_d_auc, best_d_preds = auc, ens.copy()
            best_d_T, best_d_alpha = T, alpha

marker = "  *** NEW BEST ***" if best_d_auc > CURRENT_BEST else ""
print(f"  Best T={best_d_T}, alpha={best_d_alpha:.2f}: {best_d_auc:.4f}  (delta={best_d_auc-CURRENT_BEST:+.4f}){marker}")
results_list.append(("logit_max_temp_knn", best_d_auc,
                     {"T": best_d_T, "alpha": float(best_d_alpha)}, best_d_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD E: Multi-K KNN ensemble + logit_max
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD E: Multi-K KNN ensemble + logit_max")
print("="*65)

# Build multiple KNN predictions
knn_preds = {}
for k in [1, 3, 5, 7, 10, 15]:
    knn_preds[k] = knn_predict(k=k)

# Ensemble of k=3,5,7 + logit_max
best_e_auc, best_e_preds = 0.0, None
best_e_params = {}

for w3 in np.arange(0.0, 0.41, 0.1):
    for w5 in np.arange(0.0, 0.71-w3, 0.1):
        for w7 in np.arange(0.0, 0.71-w3-w5, 0.1):
            w_logit = 1.0 - w3 - w5 - w7
            if w_logit < 0 or w_logit > 0.5:
                continue
            ens = w3 * knn_preds[3] + w5 * knn_preds[5] + w7 * knn_preds[7] + w_logit * file_prob_max
            auc = macro_auc(file_labels, ens)
            if auc > best_e_auc:
                best_e_auc = auc
                best_e_preds = ens.copy()
                best_e_params = {"w3": w3, "w5": w5, "w7": w7, "w_logit": w_logit}

marker = "  *** NEW BEST ***" if best_e_auc > CURRENT_BEST else ""
print(f"  Multi-K + logit_max: {best_e_auc:.4f}  (delta={best_e_auc-CURRENT_BEST:+.4f}){marker}")
print(f"    Weights: {best_e_params}")
results_list.append(("multik_knn_logit_max", best_e_auc, best_e_params, best_e_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD F: Logit_max + logit_mean + KNN (3-way, dense sweep)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD F: KNN + logit_mean + logit_max (3-way dense sweep)")
print("="*65)

best_f_auc, best_f_preds, best_f_w = 0.0, None, {}
for alpha_max in np.arange(0.0, 0.51, 0.02):
    for alpha_mean in np.arange(0.0, 0.41-alpha_max, 0.02):
        alpha_knn = 1.0 - alpha_max - alpha_mean
        if alpha_knn < 0:
            continue
        ens = alpha_knn * knn5 + alpha_mean * file_prob_mean + alpha_max * file_prob_max
        auc = macro_auc(file_labels, ens)
        if auc > best_f_auc:
            best_f_auc = auc
            best_f_preds = ens.copy()
            best_f_w = {"alpha_knn": float(alpha_knn), "alpha_mean": float(alpha_mean), "alpha_max": float(alpha_max)}

marker = "  *** NEW BEST ***" if best_f_auc > CURRENT_BEST else ""
print(f"  Best 3-way: {best_f_auc:.4f}  (delta={best_f_auc-CURRENT_BEST:+.4f}){marker}")
print(f"    {best_f_w}")
results_list.append(("3way_knn_logitmean_logitmax", best_f_auc, best_f_w, best_f_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD G: KNN with logit as extra feature for similarity
# Build similarity using: beta*cos_sim(emb) + (1-beta)*corr(logit)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD G: Hybrid similarity (embedding + logit correlation)")
print("="*65)

logit_norm = normalize(file_logit_mean, norm='l2')  # normalized logits as features

best_g_auc, best_g_preds, best_g_params = 0.0, None, {}
for k in [3, 5, 7]:
    for beta in [0.7, 0.8, 0.9, 0.95]:
        preds_g = np.zeros((n_files, n_species), dtype=np.float32)
        for i in range(n_files):
            mask = np.ones(n_files, dtype=bool); mask[i] = False
            tr_emb = file_embs_norm[mask]
            tr_logit = logit_norm[mask]
            te_emb = file_embs_norm[[i]]
            te_logit = logit_norm[[i]]
            y_tr = file_labels[mask]

            sim_emb   = (te_emb @ tr_emb.T).ravel()
            sim_logit = (te_logit @ tr_logit.T).ravel()
            sim_combo = beta * sim_emb + (1 - beta) * sim_logit

            k_eff = min(k, len(sim_combo))
            nn_idx = np.argpartition(-sim_combo, k_eff)[:k_eff]
            weights = np.clip(sim_combo[nn_idx], 0, None)
            if weights.sum() < 1e-9:
                weights = np.ones(k_eff)
            preds_g[i] = (weights[:, None] * y_tr[nn_idx]).sum(0) / weights.sum()

        auc_g = macro_auc(file_labels, preds_g)
        if auc_g > best_g_auc:
            best_g_auc = auc_g
            best_g_preds = preds_g.copy()
            best_g_params = {"k": k, "beta": beta}

marker = "  *** NEW BEST ***" if best_g_auc > CURRENT_BEST else ""
print(f"  Hybrid KNN: {best_g_auc:.4f}  (delta={best_g_auc-CURRENT_BEST:+.4f}){marker}")
print(f"    {best_g_params}")
results_list.append(("hybrid_sim_knn", best_g_auc, best_g_params, best_g_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD H: Combine best hybrid KNN with logit_max
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD H: Best hybrid KNN + logit_max")
print("="*65)

best_h_auc, best_h_preds, best_h_alpha = 0.0, None, 0.20
for alpha in np.arange(0.0, 0.51, 0.02):
    ens = alpha * file_prob_max + (1 - alpha) * best_g_preds
    auc = macro_auc(file_labels, ens)
    if auc > best_h_auc:
        best_h_auc, best_h_preds, best_h_alpha = auc, ens.copy(), alpha

marker = "  *** NEW BEST ***" if best_h_auc > CURRENT_BEST else ""
print(f"  Hybrid KNN + logit_max (alpha={best_h_alpha:.2f}): {best_h_auc:.4f}  (delta={best_h_auc-CURRENT_BEST:+.4f}){marker}")
results_list.append(("hybrid_knn_logitmax", best_h_auc,
                     {"alpha_logit_max": float(best_h_alpha), **best_g_params}, best_h_preds))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD I: All-species ensemble combining all signals
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("METHOD I: Greedy ensemble of all candidates")
print("="*65)

candidates = [
    ("knn5", knn5),
    ("logit_max", file_prob_max),
    ("logit_mean", file_prob_mean),
    ("logit_p75", file_prob_p75),
    ("logit_p90", file_prob_p90),
    ("knn_3", knn_preds[3]),
    ("knn_7", knn_preds[7]),
    ("best_a", best_a_preds),
    ("best_b", best_b_preds),
    ("best_d", best_d_preds),
    ("hybrid_knn", best_g_preds),
]

# Greedy forward selection
pool_preds = knn5.copy()
pool_auc = macro_auc(file_labels, pool_preds)
pool_names = ["knn5"]

for cname, cpreds in candidates[1:]:
    best_blend_auc = pool_auc
    best_blend_preds = pool_preds
    # Try adding with various weights
    for alpha in np.arange(0.05, 0.51, 0.05):
        trial = (1 - alpha) * pool_preds + alpha * cpreds
        auc_t = macro_auc(file_labels, trial)
        if auc_t > best_blend_auc:
            best_blend_auc = auc_t
            best_blend_preds = trial.copy()
    if best_blend_auc > pool_auc:
        pool_auc = best_blend_auc
        pool_preds = best_blend_preds
        pool_names.append(cname)
        marker = "  *** NEW BEST ***" if pool_auc > CURRENT_BEST else ""
        print(f"  + {cname}: pool={pool_auc:.4f}{marker}")
    else:
        print(f"  - {cname}: skip")

results_list.append(("greedy_all_ensemble", pool_auc, {"members": pool_names}, pool_preds))

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
    print(f"  {name}: {auc:.4f}  (delta_vs_prev={auc-CURRENT_BEST:+.4f}){marker}")

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

overall_best_auc = results_json["best"]["loo_auc"]
if best_auc > overall_best_auc:
    results_json["best"] = {"method": best_name, "loo_auc": round(float(best_auc), 4)}
    print(f"\nNEW BEST: {best_name} AUC={best_auc:.4f}")

    model_dict = {
        "method": best_name,
        "loo_auc": round(float(best_auc), 4),
        "params": best_params,
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
