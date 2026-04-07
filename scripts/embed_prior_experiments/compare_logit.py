"""Compare pure KNN vs KNN+logit at file level (correct LOO setup)."""
import numpy as np
import scipy.special
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import os
os.chdir("/home/lab/BirdClef-2026-Codebase")

raw = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
logits_win = raw['logits'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

# Build file-level aggregations
file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species), dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species), dtype=np.float32)

idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]      = emb_win[idx:idx+nw].mean(0)
    file_labels[fi]    = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[idx:idx+nw].max(0)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
file_prob_max  = scipy.special.expit(file_logit_max)

def macro_auc(y_true, y_score):
    mask = y_true.sum(0) > 0
    return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')

def knn_loo(k):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        sims = (file_embs_norm[[i]] @ file_embs_norm[mask].T).ravel()
        top  = np.argsort(-sims)[:k]
        w    = sims[top].clip(0); w = w / (w.sum() + 1e-8)
        preds[i] = (w[:, None] * file_labels[mask][top]).sum(0)
    return preds

print(f"Data: {n_files} files, {n_species} species", flush=True)

k1 = knn_loo(1); k3 = knn_loo(3); k4 = knn_loo(4); k5 = knn_loo(5)

print(f"1. KNN-5 baseline              : {macro_auc(file_labels, k5):.4f}", flush=True)
print(f"2. KNN k134 no-logit (norm)    : {macro_auc(file_labels, (0.17*k1+0.09*k3+0.38*k4)/0.64):.4f}", flush=True)
print(f"3. k134+logit (current)        : {macro_auc(file_labels, 0.36*file_prob_max+0.17*k1+0.09*k3+0.38*k4):.4f}", flush=True)
print(f"4. logit only (Perch direct)   : {macro_auc(file_labels, file_prob_max):.4f}", flush=True)
print("done", flush=True)
