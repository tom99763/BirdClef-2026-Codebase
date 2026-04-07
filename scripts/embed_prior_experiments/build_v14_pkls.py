"""
Build pkl files for v14-v32 notebooks.
Using the same X_combined_n space as embed_prior_attn.pkl but with k=3, k=4, k=5.
"""
import numpy as np, pickle, re, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# Load base pkl to reuse X_combined_n, pca params
with open("outputs/embed_prior_attn.pkl", "rb") as f:
    base_ep = pickle.load(f)

X_combined_n = base_ep['X_combined_n']
file_labels  = base_ep['file_labels']
file_list    = base_ep['file_list']

# Validate loo_auc for various k
from sklearn.metrics import roc_auc_score

n_files  = len(file_list)
n_species = file_labels.shape[1]

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

def attn_knn_loo(X, k=10, T=0.2):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr[top]]).sum(0)
    return preds

def save_pkl(path, k, T=0.2):
    y = attn_knn_loo(X_combined_n, k=k, T=T)
    auc = macro_auc(file_labels, y)
    ep = dict(base_ep)  # copy all fields
    ep['k'] = k
    ep['temperature'] = T
    ep['T'] = T
    ep['loo_auc'] = auc
    ep['method'] = f'attn_k{k}_T{T:.2f}_pca24_day'
    with open(path, 'wb') as f:
        pickle.dump(ep, f)
    print(f"  Saved {path}: k={k}, T={T:.2f}, LOO={auc:.4f}")
    return auc

print("Building k-variant pkls...")
save_pkl("outputs/embed_prior_attn_k3.pkl", k=3, T=0.2)
save_pkl("outputs/embed_prior_attn_k4.pkl", k=4, T=0.2)
save_pkl("outputs/embed_prior_attn_k5.pkl", k=5, T=0.2)
save_pkl("outputs/embed_prior_attn_k4_T018.pkl", k=4, T=0.18)

# Copy to current_subs/weights/
import shutil
WEIGHTS_DIR = "birdclef-2026/notebook resource/current_subs/weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
for fname in ["embed_prior_attn_k3.pkl", "embed_prior_attn_k4.pkl",
              "embed_prior_attn_k5.pkl", "embed_prior_attn_k4_T018.pkl"]:
    src = f"outputs/{fname}"
    dst = f"{WEIGHTS_DIR}/{fname}"
    shutil.copy2(src, dst)
    print(f"  Copied {fname} → {WEIGHTS_DIR}/")

print("\ndone")
