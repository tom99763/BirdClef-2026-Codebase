"""
Embed Prior Auto Loop: Spectral KNN + KDE + Nearest Centroid + LOF-weighted

Methods:
1. spectral_knn: Project files to spectral embedding of sim graph → KNN on spectral space
2. kde_per_species: KDE in PCA space per species → P(emb | species) ∝ KDE score
3. nearest_centroid: Per-class centroid in PCA space → cosine distance prediction
4. lof_weighted_knn: Down-weight KNN neighbors by their Local Outlier Factor (hub reduction)
5. harmonic_knn: Harmonic mean ensemble of geo_k5 + win_k1 logspace predictions

EP-only LOO-AUC target: beat interaction_knn (0.9199)
Full pipeline best: 0.9444 (used as JSON comparison)
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestCentroid
from sklearn.neighbors import KernelDensity
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win   = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))

file_labels   = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_embs_avg  = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_embs_avg[fi]  = emb_win[s:e].mean(0)

emb_win_norm   = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id    = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

EPS = 1e-7
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

# ── Load base PKL ──────────────────────────────────────────────────────────────
with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep_base = pickle.load(f)
X_ref = ep_base['X_combined_n'].astype(np.float32)   # (66, 39) PCA24 + geo
fl    = ep_base['file_labels'].astype(np.float32)

file_prob_max = sigmoid(file_logit_max)
base_logit = np.log(file_prob_max.clip(EPS)) - np.log((1-file_prob_max).clip(EPS))
sim_ref = X_ref @ X_ref.T

# ── PCA features ──────────────────────────────────────────────────────────────
emb_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)
pca32 = PCA(n_components=32, random_state=42).fit(emb_norm)
X_pca32 = pca32.transform(emb_norm).astype(np.float32)          # (66, 32)
X_pca32 = (X_pca32 - X_pca32.mean(0)) / X_pca32.std(0).clip(1e-8)

pca64 = PCA(n_components=64, random_state=42).fit(emb_norm)
X_pca64 = pca64.transform(emb_norm).astype(np.float32)          # (66, 64)
X_pca64 = (X_pca64 - X_pca64.mean(0)) / X_pca64.std(0).clip(1e-8)

T = 0.2
def softmax_knn(sims, fl_in, k=5):
    """Softmax-weighted KNN over all files (LOO applied before calling)."""
    top = np.argsort(-sims)[:k]
    ls = sims[top]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
    return (w[:,None] * fl_in[top]).sum(0)

def eval_logspace(y_ep, a, b):
    pred = sigmoid(a * base_logit + b * np.log(y_ep.clip(EPS)))
    if not np.isfinite(pred).all(): return 0.0
    return macro_auc(file_labels, pred)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Spectral KNN
# Compute spectral embedding of X_ref similarity graph → KNN in spectral space
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Method 1: Spectral KNN ===", flush=True)
from scipy.linalg import eigh

best1 = 0; best1_cfg = {}
# Build normalized Laplacian from X_ref similarities
for n_spec in [8, 16, 24]:
    sim_s = sim_ref.copy(); np.fill_diagonal(sim_s, 0)
    sim_s = np.maximum(sim_s, 0)
    D = sim_s.sum(1); D_inv_sqrt = 1.0 / np.sqrt(D.clip(1e-8))
    L_sym = np.eye(n_files) - (D_inv_sqrt[:,None] * sim_s * D_inv_sqrt[None,:])
    # Smallest n_spec eigenvectors (skip first trivial one)
    eigvals, eigvecs = eigh(L_sym, subset_by_index=[1, n_spec])
    X_spec = (eigvecs * D_inv_sqrt[:,None]).astype(np.float32)  # (66, n_spec-1)
    X_spec = normalize(X_spec, norm='l2')
    sim_spec = X_spec @ X_spec.T
    np.fill_diagonal(sim_spec, -np.inf)
    # LOO KNN
    for k in [3, 5, 7]:
        y1 = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j != i])
            sims_i = sim_spec[i, tr]
            y1[i] = softmax_knn(sims_i, fl[tr], k=k)
        for a in [0.7, 0.8, 0.9, 1.0]:
            for b in [0.8, 1.0, 1.2, 1.5]:
                auc = eval_logspace(y1, a, b)
                if auc > best1:
                    best1 = auc; best1_cfg = {'n_spec': n_spec, 'k': k, 'a': a, 'b': b}
    print(f"  n_spec={n_spec}: best so far={best1:.4f}", flush=True)
results['spectral_knn'] = (best1, best1_cfg)
print(f"  Best: {best1:.4f}  cfg={best1_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: KDE per species
# Fit KDE in PCA-32 space for each species (positive files vs all)
# Predict: log P(emb | species) − log P(emb | background)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: KDE per species ===", flush=True)
best2 = 0; best2_cfg = {}

for bw in [0.3, 0.5, 0.8, 1.2]:
    loo_preds_kde = np.zeros((n_files, n_species), np.float32)
    for fi_test in range(n_files):
        tr_mask = np.arange(n_files) != fi_test
        X_tr = X_pca32[tr_mask]
        X_te = X_pca32[[fi_test]]
        # Background KDE
        kde_bg = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde_bg.fit(X_tr)
        log_bg = kde_bg.score_samples(X_te)[0]  # scalar
        for si in range(n_species):
            pos_idx = np.where(file_labels[tr_mask, si] > 0.5)[0]
            if len(pos_idx) == 0:
                loo_preds_kde[fi_test, si] = sigmoid(file_logit_max[fi_test, si])
                continue
            kde_pos = KernelDensity(kernel='gaussian', bandwidth=bw)
            kde_pos.fit(X_tr[pos_idx])
            log_pos = kde_pos.score_samples(X_te)[0]
            loo_preds_kde[fi_test, si] = sigmoid(log_pos - log_bg)
    for a in [0.7, 0.8, 0.9, 1.0]:
        for b in [0.5, 0.8, 1.0, 1.2]:
            pred2 = sigmoid(a * base_logit + b * np.log(loo_preds_kde.clip(EPS)))
            if np.isfinite(pred2).all():
                auc2 = macro_auc(file_labels, pred2)
                if auc2 > best2:
                    best2 = auc2; best2_cfg = {'bw': bw, 'a': a, 'b': b}
    print(f"  bw={bw}: best so far={best2:.4f}", flush=True)
results['kde_per_species'] = (best2, best2_cfg)
print(f"  Best: {best2:.4f}  cfg={best2_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Nearest Centroid (LOO in PCA-64 space)
# Per-species centroid from positive files; predict = 1 / (1 + dist_to_centroid)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Nearest Centroid ===", flush=True)
best3 = 0; best3_cfg = {}

for pca_n, X_pca in [(32, X_pca32), (64, X_pca64)]:
    X_pn = normalize(X_pca, norm='l2').astype(np.float32)
    for scale in [1.0, 2.0, 5.0]:
        loo_nc = np.zeros((n_files, n_species), np.float32)
        for fi_test in range(n_files):
            tr_mask = np.arange(n_files) != fi_test
            X_tr = X_pn[tr_mask]
            X_te = X_pn[[fi_test]]
            for si in range(n_species):
                pos_idx = np.where(file_labels[tr_mask, si] > 0.5)[0]
                if len(pos_idx) == 0:
                    loo_nc[fi_test, si] = sigmoid(file_logit_max[fi_test, si])
                    continue
                centroid = X_tr[pos_idx].mean(0)
                dist = np.linalg.norm(X_te[0] - centroid)
                loo_nc[fi_test, si] = sigmoid(-scale * dist)
        for a in [0.7, 0.8, 0.9, 1.0]:
            for b in [0.5, 0.8, 1.0, 1.2]:
                pred3 = sigmoid(a * base_logit + b * np.log(loo_nc.clip(EPS)))
                if np.isfinite(pred3).all():
                    auc3 = macro_auc(file_labels, pred3)
                    if auc3 > best3:
                        best3 = auc3; best3_cfg = {'pca_n': pca_n, 'scale': scale, 'a': a, 'b': b}
    print(f"  pca={pca_n}: best so far={best3:.4f}", flush=True)
results['nearest_centroid'] = (best3, best3_cfg)
print(f"  Best: {best3:.4f}  cfg={best3_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: LOF-weighted KNN (Local Outlier Factor down-weights hubs)
# LOF > 1 means file is an outlier; weight = 1 / LOF(j)^gamma
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: LOF-weighted KNN ===", flush=True)
best4 = 0; best4_cfg = {}

# Compute LOF scores on full X_ref (slight leakage but standard)
for lof_k in [5, 10]:
    lof = LocalOutlierFactor(n_neighbors=lof_k, metric='cosine', novelty=False)
    lof.fit(X_ref)
    lof_scores = -lof.negative_outlier_factor_  # > 1 means outlier
    lof_scores = np.maximum(lof_scores, 1.0)    # floor at 1

    sc = sim_ref.copy(); np.fill_diagonal(sc, -np.inf)
    for gamma in [0.5, 1.0, 2.0]:
        for k in [3, 5, 7]:
            y4 = np.zeros((n_files, n_species), np.float32)
            for i in range(n_files):
                tr = np.array([j for j in range(n_files) if j != i])
                sims_i = sc[i, tr]
                top_i  = np.argsort(-sims_i)[:k]
                raw_w  = sims_i[top_i]
                lof_w  = lof_scores[tr[top_i]]
                adj_w  = raw_w / (lof_w**gamma)
                adj_w  = np.maximum(adj_w, 0)
                ws = adj_w.sum()
                adj_w  = adj_w / ws if ws > 1e-8 else np.ones(k)/k
                y4[i]  = (adj_w[:,None] * fl[tr[top_i]]).sum(0)
            for a in [0.7, 0.8, 0.9, 1.0]:
                for b in [0.8, 1.0, 1.2, 1.5]:
                    auc4 = eval_logspace(y4, a, b)
                    if auc4 > best4:
                        best4 = auc4; best4_cfg = {'lof_k': lof_k, 'gamma': gamma, 'k': k, 'a': a, 'b': b}
    print(f"  lof_k={lof_k}: best so far={best4:.4f}", flush=True)
results['lof_weighted_knn'] = (best4, best4_cfg)
print(f"  Best: {best4:.4f}  cfg={best4_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Harmonic-mean KNN ensemble
# Harmonic mean of geo_k5 + win_k1 predictions (less sensitive to near-zero values)
# sigmoid(a * base_logit + b * log(harmonic_mean(y_geo, y_win)))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Harmonic-mean KNN ensemble ===", flush=True)

# Compute geo_k5 LOO
y_geo_k5 = np.zeros((n_files, n_species), np.float32)
sc_ref = sim_ref.copy(); np.fill_diagonal(sc_ref, -np.inf)
for i in range(n_files):
    tr = np.array([j for j in range(n_files) if j != i])
    sims_i = sc_ref[i, tr]
    y_geo_k5[i] = softmax_knn(sims_i, fl[tr], k=5)

# Compute win_k1 LOO
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = (win_file_id != i)
    X_tr   = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    sims   = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :1]
    wp = np.zeros((te_e-te_s, n_species), np.float32)
    for wi in range(te_e-te_s):
        ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
        ww = ww/ws if ws > 1e-8 else np.ones(1)
        wp[wi] = (ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)

best5 = 0; best5_cfg = {}
for wg in [0.3, 0.4, 0.5, 0.6, 0.7]:
    # Harmonic mean: 2*y_geo*y_win / (y_geo + y_win)
    y_harm = 2 * y_geo_k5 * y_win_k1 / (y_geo_k5 + y_win_k1 + EPS)
    # Also try plain blend as comparison
    y_arith = wg * y_geo_k5 + (1-wg) * y_win_k1
    for y_ep, tag in [(y_harm, 'harm'), (y_arith, 'arith')]:
        for a in [0.7, 0.8, 0.9, 1.0]:
            for b in [0.8, 1.0, 1.2, 1.5, 1.8]:
                auc5 = eval_logspace(y_ep, a, b)
                if auc5 > best5:
                    best5 = auc5; best5_cfg = {'tag': tag, 'wg': wg, 'a': a, 'b': b}
results['harmonic_knn'] = (best5, best5_cfg)
print(f"  Best: {best5:.4f}  cfg={best5_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
EP_BEST = 0.9199
FULL_BEST = 0.9444
print(f"\n{'='*60}")
print(f"SPECTRAL/KDE/CENTROID/LOF SUMMARY")
print(f"EP-only best: interaction_knn={EP_BEST}")
print(f"Full pipeline best: {FULL_BEST}")
print(f"{'='*60}")
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    delta_ep   = auc - EP_BEST
    delta_full = auc - FULL_BEST
    ep_marker  = " *** NEW EP BEST ***"   if auc > EP_BEST   else ""
    fp_marker  = " *** NEW FULL BEST ***" if auc > FULL_BEST else ""
    print(f"  {name}: {auc:.4f}  (vs EP:{delta_ep:+.4f} / full:{delta_full:+.4f}){ep_marker}{fp_marker}")

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
for name, (auc, cfg) in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc), 'config': cfg})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"\n*** NEW BEST: {name} = {auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")
