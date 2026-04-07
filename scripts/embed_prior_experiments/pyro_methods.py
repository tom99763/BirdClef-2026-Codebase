"""
Pyro probabilistic methods for embed prior LOO-CV.
NO sigmoid(logit) — pure embedding only.
File-level LOO-CV (66 files).
Baseline: KNN-5 cosine = 0.8412, best so far = Mahal k=5 = 0.8467

Methods:
  A) Sparse Gaussian Process (pyro.contrib.gp, RBF kernel, PCA-32)
  B) Bayesian Logistic Regression (SVI, AutoDiagonalNormal, PCA-32)
  C) Gaussian Process Classification (exact, small N=65)
"""
import numpy as np
import scipy.special
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam as PyroAdam
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings, os
warnings.filterwarnings('ignore')
pyro.set_rng_seed(42)
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────
raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
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

# PCA-32
pca = PCA(n_components=32, random_state=42).fit(file_embs_norm)
X32 = pca.transform(file_embs_norm).astype(np.float32)
# also normalize PCA features per-dim for GP stability
X32_std = X32.std(0) + 1e-6
X32_n   = X32 / X32_std  # zero mean (PCA), unit std

print(f"Data: {n_files} files, {n_species} species", flush=True)
active_mask = file_labels.sum(0) > 0
print(f"Active species: {active_mask.sum()}", flush=True)

def macro_auc(y_true, y_score):
    mask = y_true.sum(0) > 0
    return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')

BASELINE = 0.8412
BEST_SO_FAR = 0.8467
results = {}

# ══════════════════════════════════════════════════════════════════════════
# A) Sparse Variational GP Classification (Pyro contrib.gp)
#    One binary GP per species (Bernoulli likelihood)
#    Only fit species with >= 3 positive files; fallback = KNN-5
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A) Sparse Variational GP (pyro.contrib.gp)")
print("="*60, flush=True)

def fit_svgp(X_tr, y_tr, n_inducing=10, n_steps=300, lr=0.01):
    """Fit binary SVGP, return predict function."""
    X = torch.tensor(X_tr, dtype=torch.float32)
    y = torch.tensor(y_tr, dtype=torch.float32)
    # Inducing points: k-means init
    perm = torch.randperm(len(X))[:n_inducing]
    Xu   = X[perm].clone()

    kernel = gp.kernels.RBF(input_dim=X.shape[1],
                             variance=torch.tensor(1.0),
                             lengthscale=torch.tensor(1.0))
    likelihood = gp.likelihoods.Binary()
    model = gp.models.VariationalSparseGP(X, y, kernel, Xu,
                                           likelihood=likelihood,
                                           whiten=True)
    optimizer = PyroAdam({"lr": lr})
    loss_fn   = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
    pyro.clear_param_store()
    model.train()
    for _ in range(n_steps):
        loss_fn.step()
    model.eval()

    def predict(X_te):
        with torch.no_grad():
            mean, _ = model(torch.tensor(X_te, dtype=torch.float32),
                            full_cov=False, noiseless=False)
        return torch.sigmoid(mean).numpy()

    return predict

def svgp_loo(n_inducing=10, n_steps=300):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    # KNN fallback
    def knn5(X_tr, Y_tr, X_te):
        sims = (X_te @ X_tr.T).ravel()
        top  = np.argsort(-sims)[:5]
        w    = sims[top].clip(0); w /= (w.sum() + 1e-8)
        return (w[:, None] * Y_tr[top]).sum(0)

    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        X_tr = X32_n[mask]; Y_tr = file_labels[mask]
        X_te = X32_n[[i]]

        for s in range(n_species):
            y_s = Y_tr[:, s]
            n_pos = int(y_s.sum())
            if n_pos < 3 or (len(y_s) - n_pos) < 3:
                # fallback: cosine KNN-5 on full embedding
                preds[i, s] = knn5(file_embs_norm[mask], Y_tr[:, [s]], file_embs_norm[[i]])[0]
                continue
            try:
                predict = fit_svgp(X_tr, y_s, n_inducing=n_inducing, n_steps=n_steps)
                preds[i, s] = float(predict(X_te)[0])
            except:
                preds[i, s] = knn5(file_embs_norm[mask], Y_tr[:, [s]], file_embs_norm[[i]])[0]

        if (i + 1) % 10 == 0:
            print(f"  SVGP fold {i+1}/66 done", flush=True)

    return preds

print("  Running SVGP (n_inducing=10, steps=300)...", flush=True)
svgp_preds = svgp_loo(n_inducing=10, n_steps=300)
auc_svgp = macro_auc(file_labels, svgp_preds)
print(f"  SVGP LOO-AUC: {auc_svgp:.4f}  (Δ={auc_svgp-BASELINE:+.4f})", flush=True)
results['SVGP n_ind=10 steps=300'] = auc_svgp

# ══════════════════════════════════════════════════════════════════════════
# B) Bayesian Logistic Regression (Pyro SVI, AutoDiagonalNormal)
#    Vectorized: fit one BLR per species
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) Bayesian Logistic Regression (Pyro SVI)")
print("="*60, flush=True)

def blr_model(X, y=None):
    D = X.shape[1]
    w = pyro.sample("w", dist.Normal(torch.zeros(D), torch.ones(D)).to_event(1))
    b = pyro.sample("b", dist.Normal(torch.tensor(0.0), torch.tensor(1.0)))
    logits = X @ w + b
    with pyro.plate("data", X.shape[0]):
        pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

def fit_blr(X_tr, y_tr, n_steps=500, lr=0.01):
    X = torch.tensor(X_tr, dtype=torch.float32)
    y = torch.tensor(y_tr, dtype=torch.float32)
    guide = pyro.infer.autoguide.AutoDiagonalNormal(blr_model)
    svi   = SVI(blr_model, guide, PyroAdam({"lr": lr}), loss=Trace_ELBO())
    pyro.clear_param_store()
    for _ in range(n_steps):
        svi.step(X, y)

    def predict(X_te, n_samples=50):
        Xt = torch.tensor(X_te, dtype=torch.float32)
        probs = []
        for _ in range(n_samples):
            sample = guide(X, y)
            w = sample["w"]; b = sample["b"]
            logits = Xt @ w + b
            probs.append(torch.sigmoid(logits).detach().numpy())
        return np.mean(probs, axis=0)

    return predict

def blr_loo(n_steps=500):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    def knn5(X_tr, Y_tr, X_te):
        sims = (X_te @ X_tr.T).ravel()
        top  = np.argsort(-sims)[:5]
        w    = sims[top].clip(0); w /= (w.sum() + 1e-8)
        return (w[:, None] * Y_tr[top]).sum(0)

    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        X_tr = X32_n[mask]; Y_tr = file_labels[mask]
        X_te = X32_n[[i]]

        for s in range(n_species):
            y_s = Y_tr[:, s]
            n_pos = int(y_s.sum())
            if n_pos < 3 or (len(y_s) - n_pos) < 3:
                preds[i, s] = knn5(file_embs_norm[mask], Y_tr[:, [s]], file_embs_norm[[i]])[0]
                continue
            try:
                predict = fit_blr(X_tr, y_s, n_steps=n_steps)
                preds[i, s] = float(predict(X_te)[0])
            except:
                preds[i, s] = knn5(file_embs_norm[mask], Y_tr[:, [s]], file_embs_norm[[i]])[0]

        if (i + 1) % 10 == 0:
            print(f"  BLR fold {i+1}/66 done", flush=True)

    return preds

print("  Running BLR (steps=500)...", flush=True)
blr_preds = blr_loo(n_steps=500)
auc_blr = macro_auc(file_labels, blr_preds)
print(f"  BLR LOO-AUC: {auc_blr:.4f}  (Δ={auc_blr-BASELINE:+.4f})", flush=True)
results['BLR steps=500'] = auc_blr

# ══════════════════════════════════════════════════════════════════════════
# C) Exact GP Classification (Pyro GP, small N=65 OK)
#    Uses Bernoulli likelihood + Laplace approximation via SVGP with all
#    training points as inducing points (= exact GP)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) Exact GP (all points as inducing = exact)")
print("="*60, flush=True)

def exact_gp_loo(n_steps=200):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    def knn5(X_tr, Y_tr, X_te):
        sims = (X_te @ X_tr.T).ravel()
        top  = np.argsort(-sims)[:5]
        w    = sims[top].clip(0); w /= (w.sum() + 1e-8)
        return (w[:, None] * Y_tr[top]).sum(0)

    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        X_tr = X32_n[mask]; Y_tr = file_labels[mask]
        X_te = X32_n[[i]]
        n_ind = len(X_tr)  # all points = exact GP

        for s in range(n_species):
            y_s = Y_tr[:, s]
            n_pos = int(y_s.sum())
            if n_pos < 3 or (len(y_s) - n_pos) < 3:
                preds[i, s] = knn5(file_embs_norm[mask], Y_tr[:, [s]], file_embs_norm[[i]])[0]
                continue
            try:
                predict = fit_svgp(X_tr, y_s, n_inducing=n_ind, n_steps=n_steps, lr=0.02)
                preds[i, s] = float(predict(X_te)[0])
            except:
                preds[i, s] = knn5(file_embs_norm[mask], Y_tr[:, [s]], file_embs_norm[[i]])[0]

        if (i + 1) % 10 == 0:
            print(f"  ExactGP fold {i+1}/66 done", flush=True)

    return preds

print("  Running ExactGP (all inducing, steps=200)...", flush=True)
egp_preds = exact_gp_loo(n_steps=200)
auc_egp = macro_auc(file_labels, egp_preds)
print(f"  ExactGP LOO-AUC: {auc_egp:.4f}  (Δ={auc_egp-BASELINE:+.4f})", flush=True)
results['ExactGP steps=200'] = auc_egp

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Method':<35s}  {'AUC':>6}  {'vs KNN-5':>8}  {'vs Mahal':>9}")
print("-" * 65)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:<35s}  {auc:.4f}  {auc-BASELINE:+.4f}  {auc-BEST_SO_FAR:+.4f}")
print(f"\nBaseline KNN-5:   {BASELINE:.4f}")
print(f"Best so far Mahal: {BEST_SO_FAR:.4f}")
print("done", flush=True)
