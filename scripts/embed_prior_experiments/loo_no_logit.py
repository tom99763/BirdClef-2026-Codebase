import numpy as np
from sklearn.metrics import roc_auc_score
import os

os.chdir("/home/lab/BirdClef-2026-Codebase")

data = np.load("outputs/perch_labeled_ss.npz")
embeddings = data["emb"]
labels     = data["labels"]
logits_all = data["logits"]
filenames  = data["filenames"]
unique_files = np.unique(filenames)
print(f"Loaded: emb{embeddings.shape}, {len(unique_files)} files", flush=True)

def knn(X_tr, Y_tr, X_te, k):
    tr_n = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-8)
    te_n = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-8)
    sims = te_n @ tr_n.T
    idx  = np.argsort(-sims, axis=1)[:, :k]
    w    = np.take_along_axis(sims, idx, axis=1).clip(0, 1)
    w    = w / (w.sum(1, keepdims=True) + 1e-8)
    return (w[:, :, None] * Y_tr[idx]).sum(1).astype(np.float32)

def run(name, fn):
    ss, tt = [], []
    for f in unique_files:
        tr = filenames != f
        te = filenames == f
        s = fn(embeddings[tr], labels[tr], embeddings[te], logits_all[te])
        ss.append(s.mean(0))
        tt.append(labels[te].max(0))
    S = np.array(ss); T = np.array(tt)
    aucs = [roc_auc_score(T[:, i], S[:, i]) for i in range(234) if T[:, i].sum() > 0]
    print(f"{name}: {np.mean(aucs):.4f}", flush=True)

run("1. KNN-5 baseline         ", lambda Xr, Yr, Xt, Lt: knn(Xr, Yr, Xt, 5))
run("2. KNN k134 no-logit norm ", lambda Xr, Yr, Xt, Lt: (0.17*knn(Xr,Yr,Xt,1)+0.09*knn(Xr,Yr,Xt,3)+0.38*knn(Xr,Yr,Xt,4))/0.64)
run("3. k134+logit (current)   ", lambda Xr, Yr, Xt, Lt: 0.36/(1+np.exp(-Lt.astype(np.float32)))+0.17*knn(Xr,Yr,Xt,1)+0.09*knn(Xr,Yr,Xt,3)+0.38*knn(Xr,Yr,Xt,4))
print("done", flush=True)
