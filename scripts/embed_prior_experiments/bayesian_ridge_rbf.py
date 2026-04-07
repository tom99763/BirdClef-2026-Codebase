"""
新方法：Bayesian Ridge Regression + RBF Nystroem + LogReg
在 combined [pca32+geo] 空間（45 dims）
No logit — 純 embedding + geographic metadata
LOO-AUC baseline: Attn-KNN-12 combined = 0.8606
"""
import numpy as np, os, re, json, pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load & aggregate to file level ────────────────────────────────────
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

# ── Geographic metadata ───────────────────────────────────────────────
SITES = ['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx = {s: i for i, s in enumerate(SITES)}
file_sites  = np.zeros(n_files, dtype=np.int32)
file_hours  = np.zeros(n_files, dtype=np.float32)
file_months = np.zeros(n_files, dtype=np.float32)
for fi, fname in enumerate(file_list):
    m = re.match(r'BC2026_Train_\d+_(S\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})', fname)
    if m:
        site, yr, mo, dy, hr, mn = m.groups()
        file_sites[fi]  = site2idx.get(site, 0)
        file_hours[fi]  = int(hr)
        file_months[fi] = int(mo)

pca     = PCA(n_components=32, random_state=42).fit(file_embs_norm)
X_pca   = pca.transform(file_embs_norm).astype(np.float32)
pca_std = X_pca.std(0) + 1e-6
X_pca_s = X_pca / pca_std

site_oh   = np.eye(len(SITES), dtype=np.float32)[file_sites]
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24), np.cos(2*np.pi*file_hours/24)], axis=1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12), np.cos(2*np.pi*(file_months-1)/12)], axis=1).astype(np.float32)

X_combined   = np.concatenate([X_pca_s, site_oh, hour_enc, month_enc], axis=1).astype(np.float32)  # (66, 45)
X_combined_n = normalize(X_combined, norm='l2')

print(f"Data: {n_files} files, {n_species} species, feature_dim={X_combined.shape[1]}", flush=True)

BASELINE   = 0.8412   # KNN-5 cosine
BEST_NOLOGIT = 0.8606 # Attn-KNN-12 combined

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

def attn_knn_fallback(X_n, idx_te, idx_tr, k=12, T=0.2):
    """Fallback: Attention KNN on combined space (best so far)."""
    sims = (X_n[[idx_te]] @ X_n[idx_tr].T).ravel()
    top  = np.argsort(-sims)[:k]
    logits = sims[top] / T
    logits -= logits.max()
    w = np.exp(logits); w /= w.sum()
    return (w[:, None] * file_labels[idx_tr][top]).sum(0)

results = {}

# ══════════════════════════════════════════════════════════════════════
# A) Bayesian Ridge Regression (per-species, combined features)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A) Bayesian Ridge Regression (combined [pca32+geo])")
print("="*60, flush=True)

def bayesian_ridge_loo():
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    tr_indices = np.arange(n_files)

    for i in range(n_files):
        tr_idx = tr_indices[tr_indices != i]
        X_tr   = X_combined[tr_idx]
        X_te   = X_combined[[i]]

        for s in range(n_species):
            y_s   = file_labels[tr_idx, s]
            n_pos = int(y_s.sum())
            if n_pos < 2 or (len(y_s) - n_pos) < 2:
                preds[i, s] = attn_knn_fallback(X_combined_n, i, tr_idx)[s]
                continue
            try:
                clf = BayesianRidge(max_iter=300, tol=1e-3)
                clf.fit(X_tr, y_s)
                p = float(clf.predict(X_te)[0])
                preds[i, s] = np.clip(p, 0.0, 1.0)
            except Exception:
                preds[i, s] = attn_knn_fallback(X_combined_n, i, tr_idx)[s]

        if (i + 1) % 10 == 0:
            print(f"  BayesRidge fold {i+1}/66 done", flush=True)

    return preds

print("  Running BayesianRidge LOO-CV...", flush=True)
br_preds = bayesian_ridge_loo()
auc_br   = macro_auc(file_labels, br_preds)
print(f"  BayesianRidge AUC: {auc_br:.4f}  (Δ vs KNN baseline={auc_br-BASELINE:+.4f}, "
      f"Δ vs best nologit={auc_br-BEST_NOLOGIT:+.4f})", flush=True)
results['bayesian_ridge_combined'] = auc_br

# ══════════════════════════════════════════════════════════════════════
# B) RBF Nystroem + Logistic Regression (combined features)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) RBF Nystroem + LogReg (combined [pca32+geo])")
print("="*60, flush=True)

def rbf_logreg_loo(n_components=32, gamma=0.1):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    tr_indices = np.arange(n_files)

    for i in range(n_files):
        tr_idx = tr_indices[tr_indices != i]
        X_tr   = X_combined[tr_idx]
        X_te   = X_combined[[i]]
        n_comp = min(n_components, len(tr_idx) - 1)

        # Fit Nystroem on training data only
        nys  = Nystroem(kernel='rbf', gamma=gamma,
                        n_components=n_comp, random_state=42)
        X_tr_rbf = nys.fit_transform(X_tr)
        X_te_rbf = nys.transform(X_te)

        for s in range(n_species):
            y_s   = file_labels[tr_idx, s]
            n_pos = int(y_s.sum())
            if n_pos < 2 or (len(y_s) - n_pos) < 2:
                preds[i, s] = attn_knn_fallback(X_combined_n, i, tr_idx)[s]
                continue
            try:
                clf = LogisticRegression(C=1.0, max_iter=300, solver='lbfgs')
                clf.fit(X_tr_rbf, y_s)
                preds[i, s] = clf.predict_proba(X_te_rbf)[0, 1]
            except Exception:
                preds[i, s] = attn_knn_fallback(X_combined_n, i, tr_idx)[s]

        if (i + 1) % 10 == 0:
            print(f"  RBF-LR fold {i+1}/66 done", flush=True)

    return preds

print("  Running RBF Nystroem + LogReg (n_comp=32, gamma=0.1)...", flush=True)
rbf_preds = rbf_logreg_loo(n_components=32, gamma=0.1)
auc_rbf   = macro_auc(file_labels, rbf_preds)
print(f"  RBF-LR AUC: {auc_rbf:.4f}  (Δ vs KNN baseline={auc_rbf-BASELINE:+.4f}, "
      f"Δ vs best nologit={auc_rbf-BEST_NOLOGIT:+.4f})", flush=True)
results['rbf_nystroem_logreg'] = auc_rbf

# ── Sweep gamma for RBF ────────────────────────────────────────────────
print("\n  Sweeping gamma values...", flush=True)
for gamma in [0.01, 0.5, 1.0]:
    p = rbf_logreg_loo(n_components=32, gamma=gamma)
    a = macro_auc(file_labels, p)
    print(f"  RBF-LR gamma={gamma}: {a:.4f}  (Δ={a-BEST_NOLOGIT:+.4f})", flush=True)
    results[f'rbf_nystroem_g{gamma}'] = a

# ══════════════════════════════════════════════════════════════════════
# C) Combined: BayesRidge predictions as feature for Attn-KNN
#    (Stack: use BayesRidge per-species score + emb similarity)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) Stacking: blend BayesRidge + Attn-KNN")
print("="*60, flush=True)

# Use cached br_preds
knn_preds = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    tr_idx = np.arange(n_files)[np.arange(n_files) != i]
    knn_preds[i] = attn_knn_fallback(X_combined_n, i, tr_idx)

for alpha in [0.2, 0.3, 0.5]:
    blend = alpha * br_preds + (1 - alpha) * knn_preds
    a = macro_auc(file_labels, blend)
    print(f"  Blend alpha_BR={alpha}: {a:.4f}  (Δ={a-BEST_NOLOGIT:+.4f})", flush=True)
    results[f'blend_BR_KNN_a{alpha}'] = a

# ══════════════════════════════════════════════════════════════════════
# SUMMARY + update results.json
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Method':<35s}  {'AUC':>6}  {'vs best_nologit':>15}")
print("-"*62)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    marker = " ← NEW BEST" if auc > BEST_NOLOGIT else ""
    print(f"  {name:<35s}  {auc:.4f}  {auc-BEST_NOLOGIT:+.4f}{marker}")

print(f"\nBest nologit so far: Attn-KNN-12 combined = {BEST_NOLOGIT:.4f}")

# Update results.json
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

new_best = False
best_nologit = data.get('best_nologit', {}).get('loo_auc', BEST_NOLOGIT)

for name, auc in results.items():
    data['experiments'].append({'method': name, 'loo_auc': round(auc, 6),
                                  'features': 'combined_pca32_geo', 'note': 'no_logit'})
    if auc > best_nologit:
        best_nologit = auc
        data['best_nologit'] = {'method': name, 'loo_auc': round(auc, 6),
                                  'note': 'No logit; NEW BEST'}
        new_best = True

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

if new_best:
    print(f"\n新最佳（no-logit）: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")
else:
    print(f"\n未超越 best nologit ({best_nologit:.4f})，Attn-KNN-12 combined 仍是最佳。")
print("done", flush=True)
