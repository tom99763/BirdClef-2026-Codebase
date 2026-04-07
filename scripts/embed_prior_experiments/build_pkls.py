import numpy as np, pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import os
os.chdir("/home/lab/BirdClef-2026-Codebase")

raw = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_embs   = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels = np.zeros((n_files, n_species), dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]   = emb_win[idx:idx+nw].mean(0)
    file_labels[fi] = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    idx += nw
file_embs_norm = normalize(file_embs, norm='l2')

# cosine pkl (for k=4, multi-k135, multi-k134)
pkl_cosine = {
    "method": "cosine_knn",
    "file_embs_norm": file_embs_norm,
    "file_labels": file_labels,
    "file_list": list(file_list),
}
with open("outputs/embed_prior_cosine.pkl", "wb") as f:
    pickle.dump(pkl_cosine, f)
print("Saved embed_prior_cosine.pkl")

# mahal pkl
pca = PCA(n_components=32, random_state=42).fit(file_embs_norm)
X32 = pca.transform(file_embs_norm).astype(np.float32)
cov = np.cov(X32.T) + np.eye(32) * 1e-3
inv_cov = np.linalg.inv(cov).astype(np.float32)
pkl_mahal = {
    "method": "mahalanobis_knn_k5_dim32",
    "loo_auc": 0.8467,
    "file_embs_norm": file_embs_norm,
    "file_labels": file_labels,
    "file_list": list(file_list),
    "pca_components": pca.components_.astype(np.float32),
    "pca_mean": pca.mean_.astype(np.float32),
    "file_embs_pca": X32,
    "inv_cov": inv_cov,
}
with open("outputs/embed_prior_mahal.pkl", "wb") as f:
    pickle.dump(pkl_mahal, f)
print("Saved embed_prior_mahal.pkl")
print("done")
