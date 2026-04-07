"""
batch125: Embedding-based classifiers as embed prior
Methods: Mahalanobis KNN, GMM per species, Bayesian Ridge, RBF+LogReg, Attention-weighted KNN
All work directly on Perch embeddings [739,1536] → species probs [739,234]
"""
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── data ──────────────────────────────────────────────────────────────────────
data = np.load('outputs/perch_labeled_ss.npz', allow_pickle=True)
EMB      = data['emb'].astype(np.float32)        # [739, 1536]
LOGITS   = data['logits'].astype(np.float32)     # [739, 234]
LABELS   = data['labels'].astype(np.float32)     # [739, 234]
fnames   = data['filenames']
file_list = data['file_list']

# derive integer file_ids
file_ids = np.array([np.where(file_list == fn)[0][0] for fn in fnames])
n_files  = len(file_list)
N_SP     = LABELS.shape[1]
EPS      = 1e-9

# species that appear in at least one file
sp_present = (LABELS.max(0) > 0)  # [234]
print(f"[batch125] EMB={EMB.shape}, LABELS={LABELS.shape}, files={n_files}, species={sp_present.sum()}")

# L2-normalize embeddings
EMB_NORM = normalize(EMB, norm='l2')

# ── results store ──────────────────────────────────────────────────────────────
results_path = Path('outputs/embed_prior_results.json')
with open(results_path) as f:
    store = json.load(f)
tried = {e['method'] for e in store.get('experiments', [])}
best_loo = store['best']['loo_auc']
best_method = store['best']['method']
print(f"[batch125] Current best: {best_method} LOO={best_loo:.6f}")

def loo_auc(pred_probs):
    """File-level LOO-AUC from window-level pred_probs [739,234]."""
    auc_list = []
    for fi in range(n_files):
        mask = (file_ids == fi)
        file_score = pred_probs[mask].mean(0)   # [234]
        file_true  = LABELS[mask].max(0)         # [234]
        sp = sp_present & (file_true >= 0)       # only labeled species
        if sp.sum() < 2: continue
        try:
            auc_list.append(roc_auc_score(file_true[sp], file_score[sp]))
        except Exception:
            pass
    return float(np.mean(auc_list))

def save_result(method, score, config, note=''):
    delta = score - best_loo
    tag = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    r = {'method': method, 'loo_auc': score, 'config': config, 'note': note}
    store['experiments'].append(r)
    with open(results_path, 'w') as f:
        json.dump(store, f, indent=2)
    return delta

# ── helpers ───────────────────────────────────────────────────────────────────
def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(-1, keepdims=True) + EPS)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

# ═════════════════════════════════════════════════════════════════════════════
# M1: Mahalanobis-distance KNN
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M1] Mahalanobis KNN...")

def mahal_knn_loo(k=40, pca_dim=128, use_ledoit=True):
    """LOO-CV with Mahalanobis distance KNN."""
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]

        # PCA for covariance estimation
        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        # Fit covariance
        if use_ledoit:
            cov = LedoitWolf().fit(X_tr_pca)
        else:
            cov = EmpiricalCovariance().fit(X_tr_pca)

        # Mahalanobis distances: test vs train
        diff = X_te_pca[:, None, :] - X_tr_pca[None, :, :]  # [n_te, n_tr, d]
        VI   = cov.precision_
        mahal_sq = np.einsum('tid,dj,tij->ti', diff, VI, diff)  # [n_te, n_tr]

        # KNN labels
        nn_idx = np.argsort(mahal_sq, axis=1)[:, :k]           # [n_te, k]
        nn_labels = y_tr[nn_idx]                                 # [n_te, k, 234]
        # weight by inverse mahalanobis distance
        nn_dist = mahal_sq[np.arange(len(X_te))[:, None], nn_idx] + EPS
        weights = 1.0 / nn_dist                                  # [n_te, k]
        weights /= weights.sum(1, keepdims=True)
        pred[test_mask] = np.einsum('tk,tkc->tc', weights, nn_labels)

    return pred

# einsum for mahal is memory-heavy; use chunked version for safety
def mahal_knn_loo_v2(k=40, pca_dim=128):
    """Memory-efficient Mahalanobis KNN LOO-CV."""
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]
        n_tr = X_tr.shape[0]

        pca  = PCA(n_components=min(pca_dim, n_tr-1, X_tr.shape[1]))
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        cov = LedoitWolf().fit(X_tr_pca)
        VI  = cov.precision_  # [d, d]

        # Mahal: for each test point compute distances to all train
        # d^2(x,y) = (x-y)^T VI (x-y)
        # = x^T VI x - 2 x^T VI y + y^T VI y
        XV_tr = X_tr_pca @ VI  # [n_tr, d]
        diag_tr = (XV_tr * X_tr_pca).sum(1)  # [n_tr]

        mahal_sq = np.zeros((len(X_te), n_tr), dtype=np.float64)
        for ti, xte in enumerate(X_te_pca):
            xte_VI = xte @ VI
            d2 = diag_tr - 2 * (XV_tr * xte).sum(1) + (xte_VI * xte).sum()
            mahal_sq[ti] = d2

        nn_idx = np.argsort(mahal_sq, axis=1)[:, :k]
        nn_dist = mahal_sq[np.arange(len(X_te))[:, None], nn_idx].astype(np.float32) + EPS
        weights = 1.0 / nn_dist
        weights /= weights.sum(1, keepdims=True)
        nn_labels = y_tr[nn_idx]
        pred[test_mask] = np.einsum('tk,tkc->tc', weights, nn_labels)

    return pred

m1_configs = [
    {'k': 20, 'pca_dim': 64},
    {'k': 40, 'pca_dim': 128},
    {'k': 60, 'pca_dim': 64},
    {'k': 40, 'pca_dim': 64},
    {'k': 20, 'pca_dim': 128},
]

m1_best = 0.0
for cfg in m1_configs:
    k, pca_dim = cfg['k'], cfg['pca_dim']
    mname = f'mahal_k{k}_p{pca_dim}'
    if mname in tried:
        print(f'  {mname}: already tried, skip')
        continue
    pred = mahal_knn_loo_v2(**cfg)
    score = loo_auc(pred)
    delta = save_result(mname, score, cfg)
    tag = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    if score > best_loo:
        best_loo = score
        best_method = mname
        store['best'] = {'method': mname, 'loo_auc': score}
        with open(results_path, 'w') as f:
            json.dump(store, f, indent=2)
    if score > m1_best:
        m1_best = score

print(f"  M1 done, best={m1_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M2: Gaussian Mixture Model per species
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M2] GMM per species...")

def gmm_loo(pca_dim=32, n_components=2):
    """LOO-CV: fit GMM for each species (pos/neg), score = log_likelihood ratio."""
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        sp_scores = np.zeros((len(X_te), N_SP), dtype=np.float32)
        for sp in range(N_SP):
            pos_idx = np.where(y_tr[:, sp] > 0.5)[0]
            if len(pos_idx) < n_components:
                # fallback: cosine similarity to positive mean
                if len(pos_idx) == 0:
                    continue
                pos_mean = X_tr_pca[pos_idx].mean(0)
                sims = X_te_pca @ pos_mean / (np.linalg.norm(pos_mean) + EPS)
                sp_scores[:, sp] = sigmoid(sims * 3.0)
                continue
            pos_data = X_tr_pca[pos_idx]
            try:
                gmm_pos = GaussianMixture(n_components=min(n_components, len(pos_idx)),
                                          covariance_type='diag', max_iter=50,
                                          random_state=42)
                gmm_pos.fit(pos_data)
                ll_pos = gmm_pos.score_samples(X_te_pca)
            except Exception:
                continue
            # background: all training data
            try:
                gmm_bg = GaussianMixture(n_components=n_components,
                                         covariance_type='diag', max_iter=50,
                                         random_state=42)
                gmm_bg.fit(X_tr_pca)
                ll_bg = gmm_bg.score_samples(X_te_pca)
            except Exception:
                continue
            llr = ll_pos - ll_bg
            sp_scores[:, sp] = sigmoid(llr)

        pred[test_mask] = sp_scores

    return pred

m2_configs = [
    {'pca_dim': 32, 'n_components': 2},
    {'pca_dim': 64, 'n_components': 2},
    {'pca_dim': 32, 'n_components': 1},
]

m2_best = 0.0
for cfg in m2_configs:
    mname = f'gmm_p{cfg["pca_dim"]}_c{cfg["n_components"]}'
    if mname in tried:
        print(f'  {mname}: already tried, skip')
        continue
    pred = gmm_loo(**cfg)
    score = loo_auc(pred)
    delta = score - best_loo
    save_result(mname, score, cfg)
    tag = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    if score > best_loo:
        best_loo = score
        best_method = mname
        store['best'] = {'method': mname, 'loo_auc': score}
        with open(results_path, 'w') as f:
            json.dump(store, f, indent=2)
    if score > m2_best:
        m2_best = score

print(f"  M2 done, best={m2_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M3: Bayesian Ridge Regression per species
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M3] Bayesian Ridge per species...")

def bayesian_ridge_loo(pca_dim=64):
    """LOO-CV: BayesianRidge regressor per species on PCA-reduced embeddings."""
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        sp_scores = np.zeros((len(X_te), N_SP), dtype=np.float32)
        for sp in range(N_SP):
            y_sp = y_tr[:, sp]
            if y_sp.sum() < 1 or y_sp.mean() > 0.999:
                sp_scores[:, sp] = y_sp.mean()
                continue
            try:
                br = BayesianRidge(max_iter=100)
                br.fit(X_tr_pca, y_sp)
                p = br.predict(X_te_pca)
                sp_scores[:, sp] = np.clip(p, 0, 1)
            except Exception:
                sp_scores[:, sp] = y_sp.mean()

        pred[test_mask] = sp_scores

    return pred

m3_configs = [
    {'pca_dim': 64},
    {'pca_dim': 128},
    {'pca_dim': 32},
]

m3_best = 0.0
for cfg in m3_configs:
    mname = f'bayridge_p{cfg["pca_dim"]}'
    if mname in tried:
        print(f'  {mname}: already tried, skip')
        continue
    pred = bayesian_ridge_loo(**cfg)
    score = loo_auc(pred)
    delta = score - best_loo
    save_result(mname, score, cfg)
    tag = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    if score > best_loo:
        best_loo = score
        best_method = mname
        store['best'] = {'method': mname, 'loo_auc': score}
        with open(results_path, 'w') as f:
            json.dump(store, f, indent=2)
    if score > m3_best:
        m3_best = score

print(f"  M3 done, best={m3_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M4: RBF kernel approximation + LogReg
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M4] Nystroem RBF + LogReg...")

def rbf_logreg_loo(n_components=128, pca_dim=64, gamma=0.1, C=1.0):
    """LOO-CV: Nystroem RBF feature map + per-species logistic regression."""
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]
        n_tr = X_tr.shape[0]

        # PCA first to reduce dimensions
        pca = PCA(n_components=min(pca_dim, n_tr-1, X_tr.shape[1]))
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        # Nystroem RBF feature approximation
        nystr = Nystroem(kernel='rbf', gamma=gamma,
                         n_components=min(n_components, n_tr-1),
                         random_state=42)
        X_tr_rbf = nystr.fit_transform(X_tr_pca)
        X_te_rbf = nystr.transform(X_te_pca)

        sp_scores = np.zeros((len(X_te), N_SP), dtype=np.float32)
        for sp in range(N_SP):
            y_sp = y_tr[:, sp]
            if y_sp.sum() < 2 or (y_sp == 0).sum() < 2:
                sp_scores[:, sp] = y_sp.mean()
                continue
            try:
                lr = LogisticRegression(C=C, max_iter=200, solver='lbfgs',
                                        random_state=42)
                lr.fit(X_tr_rbf, (y_sp > 0.5).astype(int))
                p = lr.predict_proba(X_te_rbf)[:, 1]
                sp_scores[:, sp] = p
            except Exception:
                sp_scores[:, sp] = y_sp.mean()

        pred[test_mask] = sp_scores

    return pred

m4_configs = [
    {'n_components': 64,  'pca_dim': 32, 'gamma': 0.1,  'C': 1.0},
    {'n_components': 128, 'pca_dim': 64, 'gamma': 0.1,  'C': 1.0},
    {'n_components': 64,  'pca_dim': 32, 'gamma': 0.05, 'C': 1.0},
]

m4_best = 0.0
for cfg in m4_configs:
    mname = f'rbflog_nc{cfg["n_components"]}_p{cfg["pca_dim"]}_g{int(cfg["gamma"]*100):02d}'
    if mname in tried:
        print(f'  {mname}: already tried, skip')
        continue
    pred = rbf_logreg_loo(**cfg)
    score = loo_auc(pred)
    delta = score - best_loo
    save_result(mname, score, cfg)
    tag = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    if score > best_loo:
        best_loo = score
        best_method = mname
        store['best'] = {'method': mname, 'loo_auc': score}
        with open(results_path, 'w') as f:
            json.dump(store, f, indent=2)
    if score > m4_best:
        m4_best = score

print(f"  M4 done, best={m4_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M5: Attention-weighted KNN (logit-based attention)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M5] Attention-weighted KNN...")

def attn_knn_loo(k=40, pca_dim=128, temp=1.0):
    """
    LOO-CV: For each test window, find K nearest training neighbors by cosine.
    Compute attention weights = softmax(logit_similarity) across K neighbors.
    Final prediction = attention-weighted average of neighbor labels.
    logit_similarity = cos(logit_test, logit_train_k)
    """
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    LOGITS_NORM = normalize(LOGITS, norm='l2')

    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr   = EMB_NORM[train_mask]
        X_te   = EMB_NORM[test_mask]
        L_tr   = LOGITS_NORM[train_mask]
        L_te   = LOGITS_NORM[test_mask]
        y_tr   = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)
        X_tr_pca = normalize(X_tr_pca, norm='l2')
        X_te_pca = normalize(X_te_pca, norm='l2')

        # cosine similarity for KNN selection
        sim = X_te_pca @ X_tr_pca.T  # [n_te, n_tr]
        nn_idx = np.argsort(-sim, axis=1)[:, :k]  # [n_te, k]

        sp_scores = np.zeros((len(X_te), N_SP), dtype=np.float32)
        for ti in range(len(X_te)):
            k_idx = nn_idx[ti]
            # logit cosine similarity for attention
            l_sim = (L_te[ti] @ L_tr[k_idx].T)  # [k]
            attn = softmax(l_sim / temp)          # [k]
            # weighted average of neighbor labels
            sp_scores[ti] = attn @ y_tr[k_idx]   # [234]

        pred[test_mask] = sp_scores

    return pred

# also try: attention on embedding similarity itself (no logit re-weighting)
def emb_knn_loo(k=40, pca_dim=128, use_dist_weight=True):
    """Standard cosine KNN LOO-CV for comparison."""
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = normalize(pca.fit_transform(X_tr), norm='l2')
        X_te_pca = normalize(pca.transform(X_te), norm='l2')

        sim = X_te_pca @ X_tr_pca.T      # [n_te, n_tr]
        nn_idx = np.argsort(-sim, axis=1)[:, :k]
        nn_sim = sim[np.arange(len(X_te))[:, None], nn_idx]  # [n_te, k]

        if use_dist_weight:
            weights = softmax(nn_sim * 10.0)   # sharpen similarities
        else:
            weights = np.ones_like(nn_sim) / k

        nn_labels = y_tr[nn_idx]   # [n_te, k, 234]
        pred[test_mask] = np.einsum('tk,tkc->tc', weights, nn_labels)

    return pred

m5_configs = [
    {'name': 'attn_knn_k40_p128_t1',  'fn': attn_knn_loo,  'cfg': {'k': 40, 'pca_dim': 128, 'temp': 1.0}},
    {'name': 'attn_knn_k40_p128_t05', 'fn': attn_knn_loo,  'cfg': {'k': 40, 'pca_dim': 128, 'temp': 0.5}},
    {'name': 'attn_knn_k20_p64_t1',   'fn': attn_knn_loo,  'cfg': {'k': 20, 'pca_dim': 64,  'temp': 1.0}},
    {'name': 'emb_knn_k40_p128_sw',   'fn': emb_knn_loo,   'cfg': {'k': 40, 'pca_dim': 128, 'use_dist_weight': True}},
    {'name': 'emb_knn_k40_p128_uni',  'fn': emb_knn_loo,   'cfg': {'k': 40, 'pca_dim': 128, 'use_dist_weight': False}},
]

m5_best = 0.0
for item in m5_configs:
    mname = item['name']
    if mname in tried:
        print(f'  {mname}: already tried, skip')
        continue
    pred = item['fn'](**item['cfg'])
    score = loo_auc(pred)
    delta = score - best_loo
    save_result(mname, score, item['cfg'])
    tag = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    if score > best_loo:
        best_loo = score
        best_method = mname
        store['best'] = {'method': mname, 'loo_auc': score}
        with open(results_path, 'w') as f:
            json.dump(store, f, indent=2)
    if score > m5_best:
        m5_best = score

print(f"  M5 done, best={m5_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M6: Blend best embedding method with logits
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M6] Blend logits + best embedding method...")

def logit_blend_loo(logit_w=0.5, pca_dim=128, k=40):
    """Blend sigmoid(logits) with cosine KNN predictions."""
    logit_pred = sigmoid(LOGITS)
    knn_pred   = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)

    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = normalize(pca.fit_transform(X_tr), norm='l2')
        X_te_pca = normalize(pca.transform(X_te), norm='l2')

        sim = X_te_pca @ X_tr_pca.T
        nn_idx = np.argsort(-sim, axis=1)[:, :k]
        nn_sim = sim[np.arange(len(X_te))[:, None], nn_idx]
        weights = softmax(nn_sim * 10.0)
        knn_pred[test_mask] = np.einsum('tk,tkc->tc', weights, y_tr[nn_idx])

    return logit_w * logit_pred + (1 - logit_w) * knn_pred

m6_configs = [
    {'logit_w': 0.9, 'pca_dim': 128, 'k': 40},
    {'logit_w': 0.8, 'pca_dim': 128, 'k': 40},
    {'logit_w': 0.7, 'pca_dim': 128, 'k': 40},
    {'logit_w': 0.95, 'pca_dim': 128, 'k': 40},
]

m6_best = 0.0
for cfg in m6_configs:
    mname = f'logblend_lw{int(cfg["logit_w"]*100):02d}_k{cfg["k"]}_p{cfg["pca_dim"]}'
    if mname in tried:
        print(f'  {mname}: already tried, skip')
        continue
    pred = logit_blend_loo(**cfg)
    score = loo_auc(pred)
    delta = score - best_loo
    save_result(mname, score, cfg)
    tag = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    if score > best_loo:
        best_loo = score
        best_method = mname
        store['best'] = {'method': mname, 'loo_auc': score}
        with open(results_path, 'w') as f:
            json.dump(store, f, indent=2)
    if score > m6_best:
        m6_best = score

print(f"  M6 done, best={m6_best:.6f}")

# ── final summary ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"[batch125] SUMMARY")
print(f"  Global best: {store['best']['method']} LOO={store['best']['loo_auc']:.6f}")
print(f"  M1 Mahal KNN best:       {m1_best:.6f}")
print(f"  M2 GMM best:             {m2_best:.6f}")
print(f"  M3 Bayesian Ridge best:  {m3_best:.6f}")
print(f"  M4 RBF+LogReg best:      {m4_best:.6f}")
print(f"  M5 Attn/Emb KNN best:    {m5_best:.6f}")
print(f"  M6 Logit blend best:     {m6_best:.6f}")
