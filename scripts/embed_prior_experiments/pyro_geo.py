"""
Geo-aware embed prior: use site/hour/month metadata as Bayesian features.
Goal: same as KNN (predict P(species|file context)), but leverage geography/time.
NO sigmoid(logit) — pure embedding + metadata only.
File-level LOO-CV (66 files).

Baselines:
  KNN-5 cosine (emb only):            0.8412
  Mahalanobis KNN k=5 PCA-32 (best):  0.8467

Feature sets compared:
  (a) KNN-5 cosine [baseline]
  (b) KNN-5 on combined [pca32 + site_oh(9) + sin/cos(hour) + sin/cos(month)]
  (c) BLR (Pyro SVI) on combined features
  (d) Hierarchical BLR: site random effects on embedding features
  (e) Site-conditioned KNN: use site as hard filter first, then emb similarity
  (f) KNN-5 emb + site prior blend
"""
import numpy as np
import re, os, warnings
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam as PyroAdam
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

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

print(f"Data: {n_files} files, {n_species} species", flush=True)
active_mask = file_labels.sum(0) > 0
print(f"Active species: {active_mask.sum()}", flush=True)

# ── Parse metadata from filenames ──────────────────────────────────────────
SITES = ['S03', 'S08', 'S09', 'S13', 'S15', 'S18', 'S19', 'S22', 'S23']
site2idx = {s: i for i, s in enumerate(SITES)}

file_sites  = np.zeros(n_files, dtype=np.int32)   # site index 0-8
file_hours  = np.zeros(n_files, dtype=np.float32)  # hour 0-23
file_months = np.zeros(n_files, dtype=np.float32)  # month 1-12

for fi, fname in enumerate(file_list):
    m = re.match(r'BC2026_Train_\d+_(S\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})', fname)
    if m:
        site, yr, mo, dy, hr, mn = m.groups()
        file_sites[fi]  = site2idx.get(site, 0)
        file_hours[fi]  = int(hr)
        file_months[fi] = int(mo)

# Site one-hot (9), hour sin/cos (2), month sin/cos (2)
site_oh    = np.eye(len(SITES), dtype=np.float32)[file_sites]               # (66, 9)
hour_enc   = np.stack([np.sin(2*np.pi*file_hours/24),
                        np.cos(2*np.pi*file_hours/24)], axis=1).astype(np.float32)  # (66, 2)
month_enc  = np.stack([np.sin(2*np.pi*(file_months-1)/12),
                        np.cos(2*np.pi*(file_months-1)/12)], axis=1).astype(np.float32)  # (66, 2)

# PCA-32 embeddings
pca     = PCA(n_components=32, random_state=42).fit(file_embs_norm)
X_pca   = pca.transform(file_embs_norm).astype(np.float32)
X_pca_s = X_pca / (X_pca.std(0) + 1e-6)   # standardized

# Combined feature: [pca32 + site_oh9 + hour2 + month2] = 45 dims
X_combined = np.concatenate([X_pca_s, site_oh, hour_enc, month_enc], axis=1)  # (66, 45)

print(f"Feature shapes: pca={X_pca_s.shape}, combined={X_combined.shape}", flush=True)
print(f"Site distribution: {dict(zip(SITES, site_oh.sum(0).astype(int)))}", flush=True)

BASELINE   = 0.8412
BEST_SO_FAR = 0.8467
results = {}

def macro_auc(y_true, y_score):
    mask = y_true.sum(0) > 0
    return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')

def cosine_knn(X_tr_norm, Y_tr, X_te_norm, k=5):
    sims = (X_te_norm @ X_tr_norm.T).ravel()
    top  = np.argsort(-sims)[:k]
    w    = sims[top].clip(0); w /= (w.sum() + 1e-8)
    return (w[:, None] * Y_tr[top]).sum(0)

# ══════════════════════════════════════════════════════════════════════════
# A) KNN-5 cosine baseline (emb only)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A) KNN-5 cosine baseline (emb only)")
print("="*60, flush=True)

preds_a = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    mask = np.ones(n_files, bool); mask[i] = False
    preds_a[i] = cosine_knn(file_embs_norm[mask], file_labels[mask],
                             file_embs_norm[[i]], k=5)
auc_a = macro_auc(file_labels, preds_a)
print(f"  KNN-5 cosine: {auc_a:.4f}  (Δ={auc_a-BASELINE:+.4f})")
results['A) KNN-5 cosine (emb)'] = auc_a

# ══════════════════════════════════════════════════════════════════════════
# B) KNN-5 on combined feature space [pca32 + site + hour + month]
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) KNN-5 cosine on combined [pca32+site+hour+month]")
print("="*60, flush=True)

X_comb_norm = normalize(X_combined, norm='l2')
preds_b = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    mask = np.ones(n_files, bool); mask[i] = False
    preds_b[i] = cosine_knn(X_comb_norm[mask], file_labels[mask],
                              X_comb_norm[[i]], k=5)
auc_b = macro_auc(file_labels, preds_b)
print(f"  KNN-5 combined: {auc_b:.4f}  (Δ={auc_b-BASELINE:+.4f})")
results['B) KNN-5 cosine (combined)'] = auc_b

# ══════════════════════════════════════════════════════════════════════════
# C) Site-conditioned KNN: prefer same-site files, then fall back
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) Site-conditioned KNN (same-site priority)")
print("="*60, flush=True)

def site_cond_knn(emb_tr, labels_tr, sites_tr, emb_te, site_te, k_site=3, k_global=5):
    """KNN within same site (if >= k_site available), else global."""
    same_site = (sites_tr == site_te)
    if same_site.sum() >= k_site:
        # Use same-site KNN
        emb_ss = emb_tr[same_site]
        lab_ss = labels_tr[same_site]
        emb_ss_n = normalize(emb_ss, norm='l2')
        return cosine_knn(emb_ss_n, lab_ss, normalize(emb_te, norm='l2'), k=min(k_site, same_site.sum()))
    else:
        return cosine_knn(normalize(emb_tr, norm='l2'), labels_tr,
                           normalize(emb_te, norm='l2'), k=k_global)

preds_c = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    mask = np.ones(n_files, bool); mask[i] = False
    preds_c[i] = site_cond_knn(
        file_embs[mask], file_labels[mask], file_sites[mask],
        file_embs[[i]], file_sites[i]
    )
auc_c = macro_auc(file_labels, preds_c)
print(f"  Site-cond KNN: {auc_c:.4f}  (Δ={auc_c-BASELINE:+.4f})")
results['C) Site-conditioned KNN'] = auc_c

# ══════════════════════════════════════════════════════════════════════════
# D) KNN-5 emb blend with site prior
#    site_prior[s] = P(species s | site), computed from training files
#    final = alpha * knn + (1-alpha) * site_prior
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("D) KNN-5 emb + site prior blend")
print("="*60, flush=True)

def site_prior_blend_loo(alpha=0.7):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        knn_pred = cosine_knn(file_embs_norm[mask], file_labels[mask],
                               file_embs_norm[[i]], k=5)
        # Site prior: mean labels of training files with same site
        same_site = mask & (file_sites == file_sites[i])
        if same_site.sum() > 0:
            site_p = file_labels[same_site].mean(0)
        else:
            site_p = file_labels[mask].mean(0)  # global fallback
        preds[i] = alpha * knn_pred + (1 - alpha) * site_p
    return preds

for alpha in [0.5, 0.7, 0.9]:
    p = site_prior_blend_loo(alpha=alpha)
    auc = macro_auc(file_labels, p)
    print(f"  alpha={alpha}: {auc:.4f}  (Δ={auc-BASELINE:+.4f})")
    results[f'D) KNN+site_prior alpha={alpha}'] = auc

# ══════════════════════════════════════════════════════════════════════════
# E) Bayesian Logistic Regression (Pyro SVI) on combined features
#    Feature: [pca32 + site_oh9 + hour2 + month2] = 45 dims
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("E) Bayesian LR (Pyro SVI) on combined [pca32+geo]")
print("="*60, flush=True)

def blr_model(X, y=None):
    D = X.shape[1]
    w = pyro.sample("w", dist.Normal(torch.zeros(D), torch.ones(D)).to_event(1))
    b = pyro.sample("b", dist.Normal(torch.tensor(0.0), torch.tensor(1.0)))
    logits = X @ w + b
    with pyro.plate("data", X.shape[0]):
        pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

def fit_blr(X_tr, y_tr, n_steps=300, lr=0.02):
    X = torch.tensor(X_tr, dtype=torch.float32)
    y = torch.tensor(y_tr, dtype=torch.float32)
    guide = pyro.infer.autoguide.AutoDiagonalNormal(blr_model)
    svi   = SVI(blr_model, guide, PyroAdam({"lr": lr}), loss=Trace_ELBO())
    pyro.clear_param_store()
    for _ in range(n_steps):
        svi.step(X, y)

    def predict(X_te, n_samples=30):
        Xt = torch.tensor(X_te, dtype=torch.float32)
        probs = []
        for _ in range(n_samples):
            sample = guide(X, y)
            w_s = sample["w"]; b_s = sample["b"]
            probs.append(torch.sigmoid(Xt @ w_s + b_s).detach().numpy())
        return np.mean(probs, axis=0)

    return predict

def blr_loo(X_feat, n_steps=300):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    X_feat_n = normalize(X_feat, norm='l2')

    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        X_tr = X_feat[mask]; Y_tr = file_labels[mask]
        X_te = X_feat[[i]]

        for s in range(n_species):
            y_s = Y_tr[:, s]
            n_pos = int(y_s.sum())
            if n_pos < 3 or (len(y_s) - n_pos) < 3:
                preds[i, s] = cosine_knn(file_embs_norm[mask], Y_tr[:, [s]],
                                          file_embs_norm[[i]], k=5)[0]
                continue
            try:
                predict = fit_blr(X_tr, y_s, n_steps=n_steps)
                preds[i, s] = float(predict(X_te)[0])
            except:
                preds[i, s] = cosine_knn(file_embs_norm[mask], Y_tr[:, [s]],
                                          file_embs_norm[[i]], k=5)[0]

        if (i + 1) % 10 == 0:
            print(f"  BLR fold {i+1}/66 done", flush=True)

    return preds

print("  Running BLR on combined features (steps=300)...", flush=True)
blr_preds = blr_loo(X_combined, n_steps=300)
auc_blr = macro_auc(file_labels, blr_preds)
print(f"  BLR combined: {auc_blr:.4f}  (Δ={auc_blr-BASELINE:+.4f})", flush=True)
results['E) BLR combined (pca32+geo)'] = auc_blr

# ══════════════════════════════════════════════════════════════════════════
# F) Hierarchical BLR: site random effects
#    w_s ~ N(mu_w, sigma_w)  per species per site random intercept
#    logit = X_emb @ w_global + b_site[site_id]
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("F) Hierarchical BLR: site random intercepts + pca32 features")
print("="*60, flush=True)

N_SITES = len(SITES)

def hier_blr_model(X_emb, site_ids, y=None):
    D = X_emb.shape[1]
    # Global embedding weights
    w = pyro.sample("w", dist.Normal(torch.zeros(D), torch.ones(D)).to_event(1))
    # Site random intercepts (hierarchical)
    sigma_b = pyro.sample("sigma_b", dist.HalfNormal(torch.tensor(1.0)))
    b_raw   = pyro.sample("b_raw", dist.Normal(torch.zeros(N_SITES), torch.ones(N_SITES)).to_event(1))
    b       = b_raw * sigma_b  # non-centered parameterization
    logits  = X_emb @ w + b[site_ids]
    with pyro.plate("data", X_emb.shape[0]):
        pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

def fit_hier_blr(X_tr, sites_tr, y_tr, n_steps=300, lr=0.02):
    X   = torch.tensor(X_tr, dtype=torch.float32)
    sid = torch.tensor(sites_tr, dtype=torch.long)
    y   = torch.tensor(y_tr, dtype=torch.float32)
    guide = pyro.infer.autoguide.AutoDiagonalNormal(hier_blr_model)
    svi   = SVI(hier_blr_model, guide, PyroAdam({"lr": lr}), loss=Trace_ELBO())
    pyro.clear_param_store()
    for _ in range(n_steps):
        svi.step(X, sid, y)

    def predict(X_te, site_te, n_samples=30):
        Xt  = torch.tensor(X_te, dtype=torch.float32)
        sid_te = torch.tensor([site_te], dtype=torch.long)
        probs = []
        for _ in range(n_samples):
            sample  = guide(X, sid, y)
            w_s     = sample["w"]
            sigma_b = sample["sigma_b"]
            b_raw   = sample["b_raw"]
            b       = b_raw * sigma_b
            probs.append(torch.sigmoid(Xt @ w_s + b[sid_te]).detach().numpy())
        return np.mean(probs, axis=0)

    return predict

def hier_blr_loo(n_steps=300):
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        X_tr = X_pca_s[mask]; Y_tr = file_labels[mask]
        X_te = X_pca_s[[i]]
        sites_tr = file_sites[mask]
        site_te  = file_sites[i]

        for s in range(n_species):
            y_s = Y_tr[:, s]
            n_pos = int(y_s.sum())
            if n_pos < 3 or (len(y_s) - n_pos) < 3:
                preds[i, s] = cosine_knn(file_embs_norm[mask], Y_tr[:, [s]],
                                          file_embs_norm[[i]], k=5)[0]
                continue
            try:
                predict = fit_hier_blr(X_tr, sites_tr, y_s, n_steps=n_steps)
                preds[i, s] = float(predict(X_te, site_te)[0])
            except:
                preds[i, s] = cosine_knn(file_embs_norm[mask], Y_tr[:, [s]],
                                          file_embs_norm[[i]], k=5)[0]

        if (i + 1) % 10 == 0:
            print(f"  HierBLR fold {i+1}/66 done", flush=True)

    return preds

print("  Running Hierarchical BLR (steps=300)...", flush=True)
hier_preds = hier_blr_loo(n_steps=300)
auc_hier = macro_auc(file_labels, hier_preds)
print(f"  HierBLR: {auc_hier:.4f}  (Δ={auc_hier-BASELINE:+.4f})", flush=True)
results['F) HierBLR site-intercepts + pca32'] = auc_hier

# ══════════════════════════════════════════════════════════════════════════
# G) Sklearn LR on combined features (fast, non-Bayesian reference)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("G) sklearn LR on combined features (non-Bayesian reference)")
print("="*60, flush=True)

def sklearn_lr_loo(X_feat):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        X_tr = X_feat[mask]; Y_tr = file_labels[mask]
        X_te = X_feat[[i]]
        for s in range(n_species):
            y_s = Y_tr[:, s]
            n_pos = int(y_s.sum())
            if n_pos < 3 or (len(y_s) - n_pos) < 3:
                preds[i, s] = cosine_knn(file_embs_norm[mask], Y_tr[:, [s]],
                                          file_embs_norm[[i]], k=5)[0]
                continue
            try:
                clf = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
                clf.fit(X_tr, y_s)
                preds[i, s] = clf.predict_proba(X_te)[0, 1]
            except:
                preds[i, s] = cosine_knn(file_embs_norm[mask], Y_tr[:, [s]],
                                          file_embs_norm[[i]], k=5)[0]
    return preds

lr_preds = sklearn_lr_loo(X_combined)
auc_lr = macro_auc(file_labels, lr_preds)
print(f"  sklearn LR combined: {auc_lr:.4f}  (Δ={auc_lr-BASELINE:+.4f})")
results['G) sklearn LR (combined)'] = auc_lr

# Only emb features
lr_emb_preds = sklearn_lr_loo(X_pca_s)
auc_lr_emb = macro_auc(file_labels, lr_emb_preds)
print(f"  sklearn LR pca32:    {auc_lr_emb:.4f}  (Δ={auc_lr_emb-BASELINE:+.4f})")
results['G) sklearn LR (pca32 only)'] = auc_lr_emb

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Method':<45s}  {'AUC':>6}  {'vs KNN-5':>8}  {'vs Mahal':>9}")
print("-" * 75)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:<45s}  {auc:.4f}  {auc-BASELINE:+.4f}  {auc-BEST_SO_FAR:+.4f}")
print(f"\nBaseline KNN-5 cosine:      {BASELINE:.4f}")
print(f"Best so far Mahal k=5 PCA:  {BEST_SO_FAR:.4f}")
print("done", flush=True)
