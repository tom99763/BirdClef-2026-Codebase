"""
Comprehensive embed prior experiment: all methods vs KNN baseline.
NO sigmoid(logit) — pure embedding-based methods only.
File-level LOO-CV (66 files).
"""
import numpy as np
import scipy.special
import scipy.sparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KernelDensity, LocalOutlierFactor
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering
import os
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────
raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

# Build file-level aggregations
file_embs   = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels = np.zeros((n_files, n_species), dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]   = emb_win[idx:idx+nw].mean(0)
    file_labels[fi] = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')

# PCA variants for methods that need lower dim
pca32 = PCA(n_components=32, random_state=42).fit(file_embs_norm)
pca64 = PCA(n_components=min(64, n_files - 1), random_state=42).fit(file_embs_norm)
X32  = pca32.transform(file_embs_norm).astype(np.float32)
X64  = pca64.transform(file_embs_norm).astype(np.float32)

print(f"Data: {n_files} files, {n_species} species", flush=True)
active_species = (file_labels.sum(0) > 0).sum()
print(f"Active species (>=1 file): {active_species}", flush=True)

# ── AUC helper ─────────────────────────────────────────────────────────────
def macro_auc(y_true, y_score):
    mask = y_true.sum(0) > 0
    return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')

# ── LOO runner ─────────────────────────────────────────────────────────────
results = {}

def run_loo(name, score_fn, X=None):
    """score_fn(X_tr, Y_tr, X_te) → (1, n_species) scores"""
    if X is None:
        X = file_embs_norm
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        try:
            preds[i] = score_fn(X[mask], file_labels[mask], X[[i]])
        except Exception as e:
            preds[i] = 0.0
    try:
        auc = macro_auc(file_labels, preds)
    except:
        auc = float('nan')
    delta = auc - 0.8412
    marker = "  *** BEST ***" if auc > max(results.values(), default=0) else ""
    print(f"  {name:<45s}: {auc:.4f}  (Δ={delta:+.4f}){marker}", flush=True)
    results[name] = auc
    return preds

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BASELINE: Cosine KNN (various k)")
print("="*60, flush=True)

def cosine_knn(k):
    def fn(X_tr, Y_tr, X_te):
        sims = (X_te @ X_tr.T).ravel()
        top  = np.argsort(-sims)[:k]
        w    = sims[top].clip(0); w /= (w.sum() + 1e-8)
        return (w[:, None] * Y_tr[top]).sum(0)
    return fn

for k in [1, 2, 3, 4, 5, 7, 10]:
    run_loo(f"KNN cosine k={k}", cosine_knn(k))

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A) GMM METHODS (per-species, binary)")
print("="*60, flush=True)

def gmm_score(n_comp, cov_type, X_all):
    """Per-species GMM: P(species=1 | x) via density ratio."""
    def fn(X_tr, Y_tr, X_te):
        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            pos = Y_tr[:, s] > 0.5
            if pos.sum() < 2:
                scores[s] = 0.0
                continue
            neg = ~pos
            try:
                gmm_pos = GaussianMixture(n_components=min(n_comp, pos.sum()),
                                          covariance_type=cov_type, random_state=42)
                gmm_pos.fit(X_tr[pos])
                ll_pos = gmm_pos.score_samples(X_te)
                if neg.sum() >= 2:
                    gmm_neg = GaussianMixture(n_components=min(n_comp, neg.sum()),
                                              covariance_type=cov_type, random_state=42)
                    gmm_neg.fit(X_tr[neg])
                    ll_neg = gmm_neg.score_samples(X_te)
                    scores[s] = scipy.special.expit(ll_pos - ll_neg)
                else:
                    scores[s] = scipy.special.expit(ll_pos[0] / 10.0)
            except:
                scores[s] = 0.0
        return scores
    return fn

run_loo("GMM diag n=1 (=Gaussian per class)", gmm_score(1, 'diag', None), X64)
run_loo("GMM diag n=2",                       gmm_score(2, 'diag', None), X64)
run_loo("GMM spherical n=1",                  gmm_score(1, 'spherical', None), X64)
run_loo("GMM spherical n=2",                  gmm_score(2, 'spherical', None), X64)

# Bayesian GMM
def bayes_gmm_score(n_comp, cov_type, X_all):
    def fn(X_tr, Y_tr, X_te):
        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            pos = Y_tr[:, s] > 0.5
            if pos.sum() < 2:
                continue
            neg = ~pos
            try:
                gmm_pos = BayesianGaussianMixture(n_components=n_comp,
                                                   covariance_type=cov_type, random_state=42)
                gmm_pos.fit(X_tr[pos])
                ll_pos = gmm_pos.score_samples(X_te)
                if neg.sum() >= 2:
                    gmm_neg = BayesianGaussianMixture(n_components=n_comp,
                                                       covariance_type=cov_type, random_state=42)
                    gmm_neg.fit(X_tr[neg])
                    ll_neg = gmm_neg.score_samples(X_te)
                    scores[s] = scipy.special.expit(ll_pos - ll_neg)
                else:
                    scores[s] = scipy.special.expit(ll_pos[0] / 10.0)
            except:
                scores[s] = 0.0
        return scores
    return fn

run_loo("BayesianGMM diag n=3", bayes_gmm_score(3, 'diag', None), X64)
run_loo("BayesianGMM sph  n=3", bayes_gmm_score(3, 'spherical', None), X64)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) DISCRIMINANT ANALYSIS")
print("="*60, flush=True)

def lda_score():
    def fn(X_tr, Y_tr, X_te):
        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            y = Y_tr[:, s].astype(int)
            if y.sum() < 2 or (1-y).sum() < 2:
                continue
            try:
                clf = LinearDiscriminantAnalysis()
                clf.fit(X_tr, y)
                scores[s] = clf.predict_proba(X_te)[0, 1]
            except:
                pass
        return scores
    return fn

run_loo("LDA (per-species)", lda_score(), X64)

def qda_score():
    def fn(X_tr, Y_tr, X_te):
        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            y = Y_tr[:, s].astype(int)
            if y.sum() < 3 or (1-y).sum() < 3:
                continue
            try:
                clf = QuadraticDiscriminantAnalysis(reg_param=0.1)
                clf.fit(X_tr, y)
                scores[s] = clf.predict_proba(X_te)[0, 1]
            except:
                pass
        return scores
    return fn

run_loo("QDA reg=0.1 (per-species)", qda_score(), X32)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) KDE (Kernel Density Estimation per species)")
print("="*60, flush=True)

def kde_score(bw):
    def fn(X_tr, Y_tr, X_te):
        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            pos = Y_tr[:, s] > 0.5
            if pos.sum() < 2:
                continue
            neg = ~pos
            try:
                kde_pos = KernelDensity(bandwidth=bw, kernel='gaussian').fit(X_tr[pos])
                ll_pos = kde_pos.score_samples(X_te)
                if neg.sum() >= 2:
                    kde_neg = KernelDensity(bandwidth=bw, kernel='gaussian').fit(X_tr[neg])
                    ll_neg = kde_neg.score_samples(X_te)
                    scores[s] = scipy.special.expit(ll_pos - ll_neg)
                else:
                    scores[s] = scipy.special.expit(ll_pos[0])
            except:
                pass
        return scores
    return fn

for bw in [0.5, 1.0, 2.0]:
    run_loo(f"KDE bw={bw}", kde_score(bw), X32)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("D) MAHALANOBIS DISTANCE")
print("="*60, flush=True)

def mahal_knn(k, dim):
    def fn(X_tr, Y_tr, X_te):
        cov = np.cov(X_tr.T) + np.eye(X_tr.shape[1]) * 1e-3
        try:
            inv_cov = np.linalg.inv(cov)
        except:
            inv_cov = np.eye(X_tr.shape[1])
        diff = X_tr - X_te  # (65, dim)
        dists = np.sqrt(np.maximum((diff @ inv_cov * diff).sum(1), 0))
        top   = np.argsort(dists)[:k]
        w     = 1.0 / (dists[top] + 1e-8)
        w    /= w.sum()
        return (w[:, None] * Y_tr[top]).sum(0)
    return fn

for k in [3, 5]:
    run_loo(f"Mahalanobis KNN k={k} dim32", mahal_knn(k, 32), X32)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("E) GRAPH / SEMI-SUPERVISED METHODS")
print("="*60, flush=True)

def label_prop_score(kernel, n_nb, alpha=None):
    """Transductive: include test point as unlabeled node."""
    def fn(X_tr, Y_tr, X_te):
        scores = np.zeros(n_species, dtype=np.float32)
        X_all = np.vstack([X_tr, X_te])
        for s in range(n_species):
            y_tr = Y_tr[:, s].astype(int)
            if y_tr.sum() < 1:
                continue
            try:
                y_all = np.append(y_tr, -1)  # -1 = unlabeled for test
                if alpha is not None:
                    clf = LabelSpreading(kernel=kernel, n_neighbors=n_nb,
                                         alpha=alpha, max_iter=200)
                else:
                    clf = LabelPropagation(kernel=kernel, n_neighbors=n_nb,
                                           max_iter=200)
                clf.fit(X_all, y_all)
                scores[s] = clf.label_distributions_[-1, 1] if clf.classes_[1] == 1 else 0.0
            except:
                pass
        return scores
    return fn

run_loo("LabelProp kNN nb=5",          label_prop_score('knn', 5),     X32)
run_loo("LabelProp kNN nb=3",          label_prop_score('knn', 3),     X32)
run_loo("LabelSpreading kNN a=0.2",    label_prop_score('knn', 5, 0.2), X32)
run_loo("LabelSpreading rbf  a=0.2",   label_prop_score('rbf', 5, 0.2), X32)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("F) CLUSTERING + CLASSIFICATION")
print("="*60, flush=True)

def kmeans_proto_knn(n_clust_per_class, k_nn):
    """KMeans prototypes per class → KNN on centroids."""
    def fn(X_tr, Y_tr, X_te):
        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            pos = Y_tr[:, s] > 0.5
            if pos.sum() < 2:
                # fallback: mean prototype
                proto = X_tr[pos].mean(0, keepdims=True) if pos.sum() > 0 else X_tr.mean(0, keepdims=True)
                sims  = (X_te @ proto.T).ravel()
                scores[s] = float(sims[0].clip(0))
                continue
            try:
                nc = min(n_clust_per_class, pos.sum())
                km = KMeans(n_clusters=nc, random_state=42, n_init=3).fit(X_tr[pos])
                centroids = normalize(km.cluster_centers_, norm='l2')
                sims = (X_te @ centroids.T).ravel()
                top  = np.argsort(-sims)[:k_nn]
                w    = sims[top].clip(0); w /= (w.sum() + 1e-8)
                scores[s] = float(w.sum())  # positive evidence
            except:
                pass
        return scores
    return fn

run_loo("KMeans proto(2) → KNN k=3", kmeans_proto_knn(2, 3), X64)
run_loo("KMeans proto(3) → KNN k=3", kmeans_proto_knn(3, 3), X64)

def affprop_score():
    """Affinity Propagation exemplars → cosine KNN."""
    def fn(X_tr, Y_tr, X_te):
        scores = np.zeros(n_species, dtype=np.float32)
        try:
            ap = AffinityPropagation(random_state=42, max_iter=200, damping=0.7)
            ap.fit(X_tr)
            exemplars = X_tr[ap.cluster_centers_indices_]
            exem_norm = normalize(exemplars, norm='l2')
            sims      = (X_te @ exem_norm.T).ravel()
            exem_labs = Y_tr[ap.cluster_centers_indices_]
            k = min(5, len(exemplars))
            top = np.argsort(-sims)[:k]
            w   = sims[top].clip(0); w /= (w.sum() + 1e-8)
            scores = (w[:, None] * exem_labs[top]).sum(0)
        except:
            pass
        return scores
    return fn

run_loo("AffinityProp exemplars → KNN", affprop_score())

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("G) PROTOTYPE VARIANTS")
print("="*60, flush=True)

def proto_mean():
    """Simple mean prototype per species."""
    def fn(X_tr, Y_tr, X_te):
        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            pos = Y_tr[:, s] > 0.5
            if pos.sum() == 0:
                continue
            proto = normalize(X_tr[pos].mean(0, keepdims=True), norm='l2')
            scores[s] = float((X_te @ proto.T).clip(0))
        return scores
    return fn

def proto_weighted_mean(temp=10.0):
    """Attention-weighted prototype: weight each pos sample by similarity."""
    def fn(X_tr, Y_tr, X_te):
        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            pos = Y_tr[:, s] > 0.5
            if pos.sum() == 0:
                continue
            sims  = (X_te @ X_tr[pos].T).ravel()
            w     = scipy.special.softmax(sims * temp)
            proto = normalize((w[:, None] * X_tr[pos]).sum(0, keepdims=True), norm='l2')
            scores[s] = float((X_te @ proto.T).clip(0))
        return scores
    return fn

run_loo("Prototype mean cosine",       proto_mean())
run_loo("Prototype attention T=5",     proto_weighted_mean(5.0))
run_loo("Prototype attention T=20",    proto_weighted_mean(20.0))

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("H) MULTI-K KNN ENSEMBLE (no logit)")
print("="*60, flush=True)

def multik_ensemble(weights):
    """Weighted ensemble of KNN with different k values."""
    ks, ws = zip(*weights)
    def fn(X_tr, Y_tr, X_te):
        out = np.zeros(n_species, dtype=np.float32)
        for k, w in zip(ks, ws):
            sims = (X_te @ X_tr.T).ravel()
            top  = np.argsort(-sims)[:k]
            sw   = sims[top].clip(0); sw /= (sw.sum() + 1e-8)
            out += w * (sw[:, None] * Y_tr[top]).sum(0)
        return out
    return fn

run_loo("Multi-K KNN [1,3,5] equal",      multik_ensemble([(1,1/3),(3,1/3),(5,1/3)]))
run_loo("Multi-K KNN [1,3,4] k4-heavy",   multik_ensemble([(1,0.17/0.64),(3,0.09/0.64),(4,0.38/0.64)]))
run_loo("Multi-K KNN [1,2,3] equal",       multik_ensemble([(1,1/3),(2,1/3),(3,1/3)]))

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60, flush=True)

sorted_results = sorted(results.items(), key=lambda x: -x[1])
baseline = results.get("KNN cosine k=5", 0.8412)

print(f"\n{'Method':<45s}  {'AUC':>6}  {'vs KNN-5':>8}")
print("-" * 65)
for name, auc in sorted_results[:15]:
    delta = auc - baseline
    marker = " ← BEST" if name == sorted_results[0][0] else ""
    print(f"  {name:<45s}  {auc:.4f}  {delta:+.4f}{marker}")

print(f"\nKNN-5 baseline: {baseline:.4f}")
print(f"Best method:    {sorted_results[0][0]} = {sorted_results[0][1]:.4f}")
print("done", flush=True)
