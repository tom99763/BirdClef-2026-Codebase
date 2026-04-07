"""
Batch 15: Genuinely new embed prior methods (non-KDE-tuning)
Goal: beat 0.9738
Methods:
  1. Nearest-Neighbor Ratio (NNR): ratio of dist to nearest positive vs nearest negative
  2. Local Outlier Factor style: compare local density to neighbor densities
  3. Kernel Mean Embedding (KME): compare mean kernel embedding of test vs class
  4. Mahalanobis distance: per-class Mahalanobis with diagonal covariance
  5. Soft-NN: softmax-weighted nearest neighbor in PCA space
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

EPS = 1e-7
def vlom_blend(a, b):
    return sigmoid(0.5*np.log(a.clip(EPS)/(1-a).clip(EPS)) + 0.5*np.log(b.clip(EPS)/(1-b).clip(EPS)))
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_by_file = {}
for i, rid in enumerate(sed_npz['row_ids']):
    sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]), []).append(i)
file_sed_max = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fb = fname.replace('.ogg','').replace('.flac','')
    if fb in sed_by_file: file_sed_max[fi] = sed_npz['probs'][sed_by_file[fb]].max(0)
base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))
mask = file_labels.sum(0) > 0
PCA_N = 32
CURRENT_BEST = 0.9738

def sweep(scores, name=""):
    best = 0; best_cfg = None
    for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for b in [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]:
            pred = sigmoid(a * base_logit + b * np.log(scores.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best: best = auc; best_cfg = (a, b)
    flag = " *** NEW BEST ***" if best > CURRENT_BEST else ""
    print(f"  {name}: {best:.4f}{flag}  cfg={best_cfg}", flush=True)
    return best, best_cfg

results = {}

# ─── Method 1: Nearest-Neighbor Ratio (NNR) per window ───────────────────────
# For each test window: score = dist_to_nearest_neg / dist_to_nearest_pos (per species)
# Higher ratio = test window is closer to positives than negatives
print("\n=== Method 1: Nearest-Neighbor Ratio (per-window) ===", flush=True)
t0 = time.time()
out_nnr = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    tr_fids = win_file_id[tr_mask]
    # Pairwise distances: test (n_te, PCA_N) vs train (n_tr, PCA_N)
    # D[i,j] = ||X_te[i] - X_tr[j]||^2
    sq_te = (X_te**2).sum(1, keepdims=True)
    sq_tr = (X_tr_l**2).sum(1)
    D2 = sq_te + sq_tr - 2 * X_te @ X_tr_l.T  # (n_te, n_tr)
    D2 = np.maximum(D2, 0)
    win_scores = np.zeros((te_e - te_s, n_species), np.float32)
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        neg_mask = ~pos_mask
        if not pos_mask.any():
            win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
        D2_pos = D2[:, pos_mask]; D2_neg = D2[:, neg_mask]
        d_pos = D2_pos.min(1)  # (n_te,) nearest positive distance
        d_neg = D2_neg.min(1) if neg_mask.any() else np.ones(te_e - te_s)
        # NNR score: neg_dist / (pos_dist + neg_dist) in [0,1], higher = more positive
        win_scores[:, si] = d_neg / (d_pos + d_neg + EPS)
    out_nnr[fi] = win_scores.mean(0)
print(f"  ({time.time()-t0:.0f}s)", flush=True)
auc1, cfg1 = sweep(out_nnr, "NNR")
results['nnr_perwin'] = (auc1, cfg1)

# ─── Method 2: Kernel Mean Embedding (KME) ───────────────────────────────────
# Compare mean kernel similarity of test windows to class mean kernel
# Score = k(x_te, μ_pos) where μ_pos is the mean RKHS embedding
# Equivalent to mean Gaussian kernel between test window and all pos windows
print("\n=== Method 2: Kernel Mean Embedding (Gaussian kernel similarity) ===", flush=True)
t0 = time.time()
out_kme = np.zeros((n_files, n_species), np.float32)
KME_BW = 0.5
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    tr_fids = win_file_id[tr_mask]
    sq_te = (X_te**2).sum(1, keepdims=True)
    sq_tr = (X_tr_l**2).sum(1)
    D2 = np.maximum(sq_te + sq_tr - 2 * X_te @ X_tr_l.T, 0)
    # Kernel matrix K[te_win, tr_win] = exp(-D2/(2*bw^2))
    K = np.exp(-D2 / (2 * KME_BW**2))  # (n_te, n_tr)
    win_scores = np.zeros((te_e - te_s, n_species), np.float32)
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        neg_mask = ~pos_mask
        if not pos_mask.any():
            win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
        k_pos = K[:, pos_mask].mean(1)  # mean kernel similarity to pos class
        k_neg = K[:, neg_mask].mean(1) if neg_mask.any() else np.ones(te_e - te_s)
        # Log ratio of kernel similarities
        win_scores[:, si] = sigmoid(np.log(k_pos.clip(EPS)) - np.log(k_neg.clip(EPS)))
    out_kme[fi] = win_scores.mean(0)
print(f"  ({time.time()-t0:.0f}s)", flush=True)
auc2, cfg2 = sweep(out_kme, "KME")
results['kme_perwin'] = (auc2, cfg2)

# ─── Method 3: Mahalanobis distance (diagonal cov) ───────────────────────────
# Per-species: score based on Mahalanobis distance to class mean with diagonal covariance
print("\n=== Method 3: Mahalanobis (diagonal cov) per-window ===", flush=True)
t0 = time.time()
out_mah = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    tr_fids = win_file_id[tr_mask]
    win_scores = np.zeros((te_e - te_s, n_species), np.float32)
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        X_pos = X_tr_l[pos_mask]
        if len(X_pos) == 0:
            win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
        mu_pos = X_pos.mean(0)
        std_pos = X_pos.std(0).clip(1e-4)  # diagonal covariance
        # Mahalanobis: negative squared distance in whitened space
        delta = (X_te - mu_pos) / std_pos  # (n_te, PCA_N)
        mah_sq = (delta**2).sum(1)  # (n_te,)
        # Also compute for background (all training)
        delta_bg = (X_te - X_tr_l.mean(0)) / X_tr_l.std(0).clip(1e-4)
        mah_bg = (delta_bg**2).sum(1)
        # Score: bg_dist - pos_dist (higher = closer to class, further from bg)
        win_scores[:, si] = sigmoid((mah_bg - mah_sq) / PCA_N)
    out_mah[fi] = win_scores.mean(0)
print(f"  ({time.time()-t0:.0f}s)", flush=True)
auc3, cfg3 = sweep(out_mah, "Mahalanobis")
results['mahalanobis_perwin'] = (auc3, cfg3)

# ─── Method 4: Soft-NN with temperature ──────────────────────────────────────
# For each test window: soft-NN score = softmax-weighted fraction of positives
# Score = Σ_j softmax(-D2_j/τ) * is_pos_j
print("\n=== Method 4: Soft Nearest Neighbor per-window ===", flush=True)
t0 = time.time()
out_snn = np.zeros((n_files, n_species), np.float32)
SNN_TAU = 0.5  # temperature for softmax
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    tr_fids = win_file_id[tr_mask]
    sq_te = (X_te**2).sum(1, keepdims=True)
    sq_tr = (X_tr_l**2).sum(1)
    D2 = np.maximum(sq_te + sq_tr - 2 * X_te @ X_tr_l.T, 0)  # (n_te, n_tr)
    # Softmax weights for each test window
    log_w = -D2 / SNN_TAU  # (n_te, n_tr)
    log_w -= log_w.max(1, keepdims=True)
    w = np.exp(log_w); w /= w.sum(1, keepdims=True)  # (n_te, n_tr)
    win_scores = np.zeros((te_e - te_s, n_species), np.float32)
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids]).astype(np.float32)
        if not pos_mask.any():
            win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
        win_scores[:, si] = (w * pos_mask).sum(1)  # weighted fraction of positives
    out_snn[fi] = win_scores.mean(0)
print(f"  ({time.time()-t0:.0f}s)", flush=True)
auc4, cfg4 = sweep(out_snn, "Soft-NN")
results['soft_nn_perwin'] = (auc4, cfg4)

# ─── Method 5: KME + KDE ensemble ────────────────────────────────────────────
# Blend KME (kernel mean embedding) with existing best KDE
# Both are "kernel methods" but capture different aspects
print("\n=== Method 5: KME + KDE blend ===", flush=True)
from sklearn.neighbors import KernelDensity
def loo_kde_perwin_single(bw, pca_n=32):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_s, te_e = int(file_start[fi]), int(file_end[fi])
        tr_mask = (win_file_id != fi)
        X_tr_raw = emb_win_norm[tr_mask]
        pca_l = PCA(n_components=pca_n, random_state=42).fit(X_tr_raw)
        X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
        mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
        X_tr_l = (X_tr_l - mu_l) / std_l
        X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
        tr_fids = win_file_id[tr_mask]
        kde_bg = KernelDensity(bandwidth=bw).fit(X_tr_l)
        log_bg_wins = kde_bg.score_samples(X_te_pca)
        win_scores = np.zeros((te_e - te_s, n_species), np.float32)
        for si in range(n_species):
            pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
            X_pos = X_tr_l[pos_mask]
            if len(X_pos) == 0:
                win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
            kde_pos = KernelDensity(bandwidth=bw).fit(X_pos)
            win_scores[:, si] = sigmoid(kde_pos.score_samples(X_te_pca) - log_bg_wins)
        out[fi] = win_scores.mean(0)
    return out

# Use the best KDE blend (bw=0.3 15% + bw=0.5 85%)
print("  Computing KDE bw=0.3...", flush=True)
kde03 = loo_kde_perwin_single(0.3)
print("  Computing KDE bw=0.5...", flush=True)
kde05 = loo_kde_perwin_single(0.5)
kde_best = 0.15 * kde03 + 0.85 * kde05  # current best blend

best5 = 0; best_cfg5 = None
for w_kme in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blend = w_kme * out_kme + (1-w_kme) * kde_best
    auc_c, cfg_c = sweep(blend)
    if auc_c > best5: best5 = auc_c; best_cfg5 = (w_kme, cfg_c)
results['kme_kde_blend'] = (best5, best_cfg5)
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  KME+KDE: {best5:.4f}{flag}  w_kme={best_cfg5[0]}", flush=True)

# ─── Method 6: NNR + KDE blend ───────────────────────────────────────────────
print("\n=== Method 6: NNR + KDE blend ===", flush=True)
best6 = 0; best_cfg6 = None
for w_nnr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blend = w_nnr * out_nnr + (1-w_nnr) * kde_best
    auc_c, cfg_c = sweep(blend)
    if auc_c > best6: best6 = auc_c; best_cfg6 = (w_nnr, cfg_c)
results['nnr_kde_blend'] = (best6, best_cfg6)
flag = " *** NEW BEST ***" if best6 > CURRENT_BEST else ""
print(f"  NNR+KDE: {best6:.4f}{flag}  w_nnr={best_cfg6[0]}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 15 Summary ===", flush=True)
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
for name, (auc, cfg) in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc), 'config': str(cfg)})
    if auc > cur_best_json:
        cur_best_json = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
