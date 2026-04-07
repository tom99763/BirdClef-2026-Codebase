"""
Fit ensemble (70% knn3+prob_max + 30% knn3+prob_mean) on all 66 files
and save embed_prior_blend.pkl for production use.
"""
import numpy as np, scipy.special, json, pickle, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
logits_win = raw['logits'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_mean= np.zeros((n_files, n_species),         dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    wl = logits_win[idx:idx+nw]
    file_embs[fi]       = emb_win[idx:idx+nw].mean(0)
    file_labels[fi]     = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi]  = wl.max(0)
    file_logit_mean[fi] = wl.mean(0)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
file_prob_max  = scipy.special.expit(file_logit_max)
file_prob_mean = scipy.special.expit(file_logit_mean)

# EXACT knn_predict (matches logit_fusion_v3.py)
def knn_predict(k=3, X=None):
    if X is None: X = file_embs_norm
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr = X[mask]; te = X[[i]]; y_tr = file_labels[mask]
        sims = (te @ tr.T).ravel()
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff)[:k_eff]
        weights = np.clip(sims[nn_idx], 0, None)
        if weights.sum() < 1e-9: weights = np.ones(k_eff)
        preds[i] = (weights[:, None] * y_tr[nn_idx]).sum(0) / weights.sum()
    return preds

knn3 = knn_predict(k=3)
print(f"KNN-3 LOO-AUC (raw): computed", flush=True)

# Find per-species alpha using ALL 66 files (production alpha, not LOO)
# For each species, find alpha maximizing AUC on all 66 files
ALPHA_GRID = np.arange(0.0, 1.01, 0.1)
N_TRAIN = n_files - 1  # 65

def per_species_alpha_loo(prob_feat, knn_k_preds):
    """LOO predictions + per-species alpha (matches logit_fusion_v3.py)."""
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    alphas = np.full(n_species, 0.30, dtype=np.float32)  # store avg alpha

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_knn    = knn_k_preds[mask]
        tr_logit  = prob_feat[mask]
        tr_labels = file_labels[mask]

        for s in range(n_species):
            y_s = tr_labels[:, s]
            if y_s.sum() == 0:
                preds[i, s] = prob_feat[i, s]; continue
            if y_s.sum() == N_TRAIN:
                preds[i, s] = 1.0; continue

            best_alpha_s, best_inner_auc = 0.30, -1.0
            for a in ALPHA_GRID:
                bl = a * tr_logit[:, s] + (1-a) * tr_knn[:, s]
                try:
                    v = roc_auc_score(y_s, bl)
                    if v > best_inner_auc:
                        best_inner_auc, best_alpha_s = v, a
                except: pass

            preds[i, s] = float(best_alpha_s * prob_feat[i, s] +
                                 (1 - best_alpha_s) * knn_k_preds[i, s])
            alphas[s] = best_alpha_s  # overwrite with last fold (approx)
    return preds, alphas

print("Computing loo preds for prob_max...", flush=True)
p_max, alphas_max = per_species_alpha_loo(file_prob_max, knn3)
print("Computing loo preds for prob_mean...", flush=True)
p_mean, alphas_mean = per_species_alpha_loo(file_prob_mean, knn3)

# Blend: 70% prob_max + 30% prob_mean
p_blend = 0.7 * p_max + 0.3 * p_mean

mask_auc = file_labels.sum(0) > 0
from sklearn.metrics import roc_auc_score
auc_max   = roc_auc_score(file_labels[:, mask_auc], p_max[:, mask_auc], average='macro')
auc_mean  = roc_auc_score(file_labels[:, mask_auc], p_mean[:, mask_auc], average='macro')
auc_blend = roc_auc_score(file_labels[:, mask_auc], p_blend[:, mask_auc], average='macro')
print(f"LOO AUC: prob_max={auc_max:.4f}, prob_mean={auc_mean:.4f}, blend(0.7/0.3)={auc_blend:.4f}", flush=True)

# Save pkl
ep = {
    'method': 'ens_k3pmx0.7_k3pmn0.3',
    'loo_auc': round(float(auc_blend), 6),
    'config': {'k': 3, 'weight_prob_max': 0.7, 'weight_prob_mean': 0.3},
    'file_list': file_list.tolist() if hasattr(file_list, 'tolist') else list(file_list),
    'file_embs_norm': file_embs_norm.tolist(),
    'file_labels': file_labels.tolist(),
    'file_prob_max': file_prob_max.tolist(),
    'file_prob_mean': file_prob_mean.tolist(),
    'alphas_prob_max': alphas_max.tolist(),
    'alphas_prob_mean': alphas_mean.tolist(),
    'loo_preds': p_blend.tolist(),
}

out_path = "outputs/embed_prior_blend.pkl"
with open(out_path, 'wb') as f:
    pickle.dump(ep, f)
print(f"Saved: {out_path}", flush=True)

# Also copy to current_subs/weights
import shutil
ws_path = "/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/current_subs/weights/embed_prior_blend.pkl"
shutil.copy(out_path, ws_path)
print(f"Copied to weights: {ws_path}", flush=True)
print("done", flush=True)
