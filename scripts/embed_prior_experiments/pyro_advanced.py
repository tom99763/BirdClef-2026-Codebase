"""
Advanced Pyro methods for embed prior LOO-CV.
NO logit — pure embedding only. File-level LOO-CV (66 files).
Baselines: KNN-5=0.8412, Mahal-k5=0.8467

Methods:
  A) Hierarchical Bayesian Logistic Regression (shared prior across labels)
  B) Semi-Supervised VAE (latent embedding → label prediction)
  C) Deep Kernel Learning (neural warping + GP)
  D) Mixture of Experts (discrete latent + amortized inference)
"""
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.optim import Adam as PyroAdam
from pyro.infer.autoguide import AutoDiagonalNormal
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings, os
warnings.filterwarnings('ignore')
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

pca = PCA(n_components=32, random_state=42).fit(file_embs_norm)
X32 = pca.transform(file_embs_norm).astype(np.float32)
X32_std = X32.std(0) + 1e-6
X32_n   = (X32 / X32_std).astype(np.float32)

active_species = np.where(file_labels.sum(0) > 0)[0]  # species with >=1 positive file
print(f"Data: {n_files} files, {n_species} species ({len(active_species)} active)", flush=True)

BASELINE    = 0.8412
BEST_NO_LOG = 0.8467
results = {}

def macro_auc(y_true, y_score):
    mask = y_true.sum(0) > 0
    return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')

def knn5_fallback(X_tr_norm, Y_tr, X_te_norm):
    sims = (X_te_norm @ X_tr_norm.T).ravel()
    top  = np.argsort(-sims)[:5]
    w    = sims[top].clip(0); w /= (w.sum() + 1e-8)
    return (w[:, None] * Y_tr[top]).sum(0)

# ══════════════════════════════════════════════════════════════════════════
# A) Hierarchical Bayesian Logistic Regression
#    Joint model: all labels share global weight prior → better small-N
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A) Hierarchical Bayesian Logistic Regression")
print("="*60, flush=True)

def hblr_model(X, Y):
    """X: (N,D), Y: (N,S) — all species jointly."""
    D = X.shape[1]; S = Y.shape[1]
    # Global shrinkage prior
    tau  = pyro.sample("tau",  dist.HalfCauchy(torch.tensor(1.0)))
    with pyro.plate("species", S):
        # Per-species weights drawn from global prior
        lam  = pyro.sample("lam",  dist.HalfCauchy(torch.ones(S)))
        with pyro.plate("dims", D):
            W = pyro.sample("W", dist.Normal(
                torch.zeros(D, S), (tau * lam).unsqueeze(0).expand(D, S)
            ))
        b = pyro.sample("b", dist.Normal(torch.zeros(S), torch.ones(S)))
    logits = X @ W + b  # (N, S)
    with pyro.plate("data", X.shape[0]):
        pyro.sample("obs", dist.Bernoulli(logits=logits).to_event(1), obs=Y)

def fit_hblr(X_tr, Y_tr, n_steps=800, lr=0.02):
    X = torch.tensor(X_tr, dtype=torch.float32)
    Y = torch.tensor(Y_tr, dtype=torch.float32)
    guide = AutoDiagonalNormal(hblr_model)
    svi   = SVI(hblr_model, guide, PyroAdam({"lr": lr}), loss=Trace_ELBO())
    pyro.clear_param_store()
    for _ in range(n_steps):
        svi.step(X, Y)

    def predict(X_te, n_samp=30):
        Xt = torch.tensor(X_te, dtype=torch.float32)
        preds = []
        for _ in range(n_samp):
            with torch.no_grad():
                sample = guide(X, Y)
                W = sample["W"]; b = sample["b"]
                preds.append(torch.sigmoid(Xt @ W + b).numpy())
        return np.mean(preds, axis=0)  # (n_te, S)

    return predict

preds_hblr = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    mask = np.ones(n_files, bool); mask[i] = False
    X_tr = X32_n[mask]; Y_tr = file_labels[mask, :][:, active_species]
    X_te = X32_n[[i]]
    try:
        predict = fit_hblr(X_tr, Y_tr, n_steps=800)
        p = predict(X_te)  # (1, n_active)
        preds_hblr[i, active_species] = p[0]
        # fallback for inactive species
        fallback = knn5_fallback(file_embs_norm[mask], file_labels[mask], file_embs_norm[[i]])
        inactive = np.ones(n_species, bool); inactive[active_species] = False
        preds_hblr[i, inactive] = fallback[inactive]
    except Exception as e:
        preds_hblr[i] = knn5_fallback(file_embs_norm[mask], file_labels[mask], file_embs_norm[[i]])
    if (i+1) % 10 == 0:
        print(f"  HBLR fold {i+1}/66", flush=True)

auc_hblr = macro_auc(file_labels, preds_hblr)
print(f"  HBLR LOO-AUC: {auc_hblr:.4f}  (Δ vs KNN-5={auc_hblr-BASELINE:+.4f}, Δ vs Mahal={auc_hblr-BEST_NO_LOG:+.4f})", flush=True)
results['Hierarchical BLR'] = auc_hblr

# ══════════════════════════════════════════════════════════════════════════
# B) Semi-Supervised VAE
#    Encoder: emb → z (latent), Decoder: z → label probs
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) Semi-Supervised VAE")
print("="*60, flush=True)

class SSVAE(nn.Module):
    def __init__(self, emb_dim=32, latent_dim=16, n_labels=75):
        super().__init__()
        self.n_labels = n_labels
        self.encoder_shared = nn.Sequential(
            nn.Linear(emb_dim, 64), nn.ReLU()
        )
        self.enc_mu    = nn.Linear(64, latent_dim)
        self.enc_scale = nn.Sequential(nn.Linear(64, latent_dim), nn.Softplus())
        self.decoder   = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, n_labels)   # logits for each label
        )

    def model(self, X, Y):
        pyro.module("ssvae", self)
        N = X.shape[0]
        with pyro.plate("data", N):
            z  = pyro.sample("z", dist.Normal(
                torch.zeros(N, self.enc_mu.out_features),
                torch.ones(N, self.enc_mu.out_features)
            ).to_event(1))
            logits = self.decoder(z)  # (N, n_labels)
            pyro.sample("obs", dist.Bernoulli(logits=logits).to_event(1), obs=Y)

    def guide(self, X, Y):
        pyro.module("ssvae", self)
        N = X.shape[0]
        h  = self.encoder_shared(X)
        mu    = self.enc_mu(h)
        scale = self.enc_scale(h) + 1e-5
        with pyro.plate("data", N):
            pyro.sample("z", dist.Normal(mu, scale).to_event(1))

    def predict(self, X):
        h  = self.encoder_shared(X)
        mu = self.enc_mu(h)
        return torch.sigmoid(self.decoder(mu))  # deterministic predict

def fit_ssvae(X_tr, Y_tr, n_steps=1000, lr=0.005):
    X = torch.tensor(X_tr, dtype=torch.float32)
    Y = torch.tensor(Y_tr, dtype=torch.float32)
    model = SSVAE(emb_dim=X.shape[1], latent_dim=12, n_labels=Y.shape[1])
    svi = SVI(model.model, model.guide, PyroAdam({"lr": lr}), loss=Trace_ELBO())
    pyro.clear_param_store()
    for _ in range(n_steps):
        svi.step(X, Y)
    return model

preds_ssvae = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    mask = np.ones(n_files, bool); mask[i] = False
    X_tr = X32_n[mask]; Y_tr = file_labels[mask, :][:, active_species]
    X_te = X32_n[[i]]
    try:
        model = fit_ssvae(X_tr, Y_tr, n_steps=1000)
        with torch.no_grad():
            p = model.predict(torch.tensor(X_te)).numpy()
        preds_ssvae[i, active_species] = p[0]
        fallback = knn5_fallback(file_embs_norm[mask], file_labels[mask], file_embs_norm[[i]])
        inactive = np.ones(n_species, bool); inactive[active_species] = False
        preds_ssvae[i, inactive] = fallback[inactive]
    except Exception as e:
        preds_ssvae[i] = knn5_fallback(file_embs_norm[mask], file_labels[mask], file_embs_norm[[i]])
    if (i+1) % 10 == 0:
        print(f"  SS-VAE fold {i+1}/66", flush=True)

auc_ssvae = macro_auc(file_labels, preds_ssvae)
print(f"  SS-VAE LOO-AUC: {auc_ssvae:.4f}  (Δ vs KNN-5={auc_ssvae-BASELINE:+.4f}, Δ vs Mahal={auc_ssvae-BEST_NO_LOG:+.4f})", flush=True)
results['SS-VAE latent=12'] = auc_ssvae

# ══════════════════════════════════════════════════════════════════════════
# C) Deep Kernel Learning (neural warping + Sparse GP per species)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) Deep Kernel Learning (DKL)")
print("="*60, flush=True)

class DKLClassifier(nn.Module):
    def __init__(self, in_dim=32, warp_dim=8, n_inducing=12):
        super().__init__()
        self.warping = nn.Sequential(
            nn.Linear(in_dim, 32), nn.Tanh(),
            nn.Linear(32, warp_dim)
        )
        kernel = gp.kernels.RBF(
            input_dim=warp_dim,
            variance=torch.tensor(1.0),
            lengthscale=torch.tensor(1.0)
        )
        deep_kernel = gp.kernels.Warping(kernel, iwarping_fn=self.warping)
        self.gp = gp.models.VariationalSparseGP(
            X=torch.zeros(1, in_dim), y=torch.zeros(1),
            kernel=deep_kernel,
            Xu=torch.randn(n_inducing, warp_dim),
            likelihood=gp.likelihoods.Binary(),
            whiten=True
        )

    def forward(self, X_te):
        with torch.no_grad():
            mean, _ = self.gp(X_te, full_cov=False, noiseless=False)
        return torch.sigmoid(mean)

def fit_dkl(X_tr, y_tr, n_steps=400, lr=0.01):
    X = torch.tensor(X_tr, dtype=torch.float32)
    y = torch.tensor(y_tr, dtype=torch.float32)
    clf = DKLClassifier(in_dim=X.shape[1])
    clf.gp.set_data(X, y)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    clf.train()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = clf.gp.guide()  # ELBO
        loss.backward()
        optimizer.step()
    clf.eval()
    return clf

preds_dkl = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    mask = np.ones(n_files, bool); mask[i] = False
    X_tr_n  = X32_n[mask]; X_te_n = X32_n[[i]]
    Y_tr    = file_labels[mask]

    for s in active_species:
        y_s = Y_tr[:, s]
        if y_s.sum() < 3 or (len(y_s) - y_s.sum()) < 3:
            sims = (file_embs_norm[[i]] @ file_embs_norm[mask].T).ravel()
            top  = np.argsort(-sims)[:5]; w = sims[top].clip(0); w /= (w.sum()+1e-8)
            preds_dkl[i, s] = float((w * Y_tr[top, s]).sum())
            continue
        try:
            clf = fit_dkl(X_tr_n, y_s, n_steps=400)
            with torch.no_grad():
                preds_dkl[i, s] = float(clf(torch.tensor(X_te_n))[0])
        except:
            sims = (file_embs_norm[[i]] @ file_embs_norm[mask].T).ravel()
            top  = np.argsort(-sims)[:5]; w = sims[top].clip(0); w /= (w.sum()+1e-8)
            preds_dkl[i, s] = float((w * Y_tr[top, s]).sum())

    if (i+1) % 10 == 0:
        print(f"  DKL fold {i+1}/66", flush=True)

auc_dkl = macro_auc(file_labels, preds_dkl)
print(f"  DKL LOO-AUC: {auc_dkl:.4f}  (Δ vs KNN-5={auc_dkl-BASELINE:+.4f}, Δ vs Mahal={auc_dkl-BEST_NO_LOG:+.4f})", flush=True)
results['DKL warp=8 ind=12'] = auc_dkl

# ══════════════════════════════════════════════════════════════════════════
# D) Sparse Bayesian Regression (Horseshoe prior → feature selection)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("D) Sparse Bayesian Regression (Horseshoe prior)")
print("="*60, flush=True)

def horseshoe_model(X, Y):
    """Horseshoe prior for sparse feature selection. X:(N,D), Y:(N,S)."""
    D = X.shape[1]; S = Y.shape[1]
    tau   = pyro.sample("tau",   dist.HalfCauchy(torch.tensor(0.1)))
    with pyro.plate("species", S):
        with pyro.plate("dims", D):
            lam = pyro.sample("lam", dist.HalfCauchy(torch.ones(D, S)))
            W   = pyro.sample("W",   dist.Normal(
                torch.zeros(D, S), tau * lam
            ))
        b = pyro.sample("b", dist.Normal(torch.zeros(S), torch.ones(S)))
    logits = X @ W + b
    with pyro.plate("data", X.shape[0]):
        pyro.sample("obs", dist.Bernoulli(logits=logits).to_event(1), obs=Y)

def fit_horseshoe(X_tr, Y_tr, n_steps=600, lr=0.02):
    X = torch.tensor(X_tr, dtype=torch.float32)
    Y = torch.tensor(Y_tr, dtype=torch.float32)
    guide = AutoDiagonalNormal(horseshoe_model)
    svi   = SVI(horseshoe_model, guide, PyroAdam({"lr": lr}), loss=Trace_ELBO())
    pyro.clear_param_store()
    for _ in range(n_steps):
        svi.step(X, Y)

    def predict(X_te, n_samp=30):
        Xt = torch.tensor(X_te, dtype=torch.float32)
        preds = []
        for _ in range(n_samp):
            with torch.no_grad():
                s = guide(X, Y)
                preds.append(torch.sigmoid(Xt @ s["W"] + s["b"]).numpy())
        return np.mean(preds, axis=0)

    return predict

preds_hs = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    mask = np.ones(n_files, bool); mask[i] = False
    X_tr = X32_n[mask]; Y_tr = file_labels[mask, :][:, active_species]
    X_te = X32_n[[i]]
    try:
        predict = fit_horseshoe(X_tr, Y_tr, n_steps=600)
        p = predict(X_te)
        preds_hs[i, active_species] = p[0]
        fallback = knn5_fallback(file_embs_norm[mask], file_labels[mask], file_embs_norm[[i]])
        inactive = np.ones(n_species, bool); inactive[active_species] = False
        preds_hs[i, inactive] = fallback[inactive]
    except Exception as e:
        preds_hs[i] = knn5_fallback(file_embs_norm[mask], file_labels[mask], file_embs_norm[[i]])
    if (i+1) % 10 == 0:
        print(f"  Horseshoe fold {i+1}/66", flush=True)

auc_hs = macro_auc(file_labels, preds_hs)
print(f"  Horseshoe LOO-AUC: {auc_hs:.4f}  (Δ vs KNN-5={auc_hs-BASELINE:+.4f}, Δ vs Mahal={auc_hs-BEST_NO_LOG:+.4f})", flush=True)
results['Horseshoe BLR'] = auc_hs

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Method':<30s}  {'AUC':>6}  {'vs KNN-5':>8}  {'vs Mahal':>9}")
print("-" * 60)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:<30s}  {auc:.4f}  {auc-BASELINE:+.4f}  {auc-BEST_NO_LOG:+.4f}")
print(f"\nKNN-5 baseline: {BASELINE:.4f}")
print(f"Mahal best:     {BEST_NO_LOG:.4f}")
print("done", flush=True)
