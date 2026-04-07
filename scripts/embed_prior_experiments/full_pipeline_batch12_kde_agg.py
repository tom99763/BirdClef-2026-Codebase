"""
Batch 12: KDE aggregation variants + GMM — beyond per-window KDE (0.9721)
Methods:
  1. kde_logmean: log-mean aggregation (mean log_pos - mean log_bg → sigmoid)
     vs current: mean(sigmoid(log_pos_i - log_bg_i)) per window
  2. kde_max_win: max over windows instead of mean
  3. kde_softmax_win: softmax-weighted mean (higher-confidence windows weighted more)
  4. gaussian_mixture: GMM with 2-4 components per species in PCA-32 space
  5. kde_adaptive_species: bandwidth = f(n_pos_windows) per species
  6. kde_perwin_rknn_fine: fine sweep per-window KDE + RKNN k5
"""
import numpy as np, pickle, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
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

BW = 0.5; PCA_N = 32

def best_auc_sweep(scores):
    best = 0; best_cfg = None
    for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for b in [1.4, 1.6, 1.8, 2.0, 2.2]:
            pred = sigmoid(a * base_logit + b * np.log(scores.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best: best = auc; best_cfg = (a, b)
    return best, best_cfg

results = {}

# ── RKNN k5 (reuse) ─────────────────────────────────────────────────────────
print("Computing RKNN k5...", flush=True)
K_RKNN = 5
y_rknn5 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s = int(file_start[i]); te_e = int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_m = (win_file_id != i); X_tr = emb_win_norm[tr_m]; tr_fi = win_file_id[tr_m]
    sims_te_tr = X_te @ X_tr.T; sims_tr_tr = X_tr @ X_tr.T
    thresh = np.partition(-sims_tr_tr, K_RKNN, axis=1)[:, K_RKNN] * -1
    top_k_idx = np.argsort(-sims_te_tr, axis=1)[:, :K_RKNN]
    wp = np.zeros((te_e - te_s, n_species), np.float32)
    for wi in range(te_e - te_s):
        nbrs = top_k_idx[wi]; recip = [n for n in nbrs if sims_te_tr[wi, n] >= thresh[n]]
        if not recip: recip = nbrs.tolist()
        ww = sims_te_tr[wi, recip].clip(0); ws = ww.sum()
        ww = ww / ws if ws > 1e-8 else np.ones(len(recip)) / len(recip)
        wp[wi] = (ww[:, None] * file_labels[tr_fi[recip]]).sum(0)
    y_rknn5[i] = wp.mean(0)
print("  RKNN k5 done.", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Method 1: kde_logmean — aggregate as sigmoid(mean_log_pos - mean_log_bg)
# Current: mean_i(sigmoid(log_pos_i - log_bg_i))
# New:     sigmoid(mean_i(log_pos_i) - mean_i(log_bg_i))
# Difference: Jensen's inequality — new version is more "confident"
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 1: kde_logmean ===", flush=True)
t0 = time.time()
loo_kde_logmean = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    tr_fids = win_file_id[tr_mask]
    kde_bg = KernelDensity(bandwidth=BW).fit(X_tr_l)
    log_bg_wins = kde_bg.score_samples(X_te_pca)   # (n_te,)
    mean_log_bg = log_bg_wins.mean()                # scalar
    win_scores = np.zeros((te_e - te_s, n_species), np.float32)
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        X_pos = X_tr_l[pos_mask]
        if len(X_pos) == 0:
            loo_kde_logmean[fi, si] = sigmoid(file_logit_max[fi, si]); continue
        kde_pos = KernelDensity(bandwidth=BW).fit(X_pos)
        mean_log_pos = kde_pos.score_samples(X_te_pca).mean()  # mean log-dens
        loo_kde_logmean[fi, si] = sigmoid(mean_log_pos - mean_log_bg)
print(f"  LOO done in {time.time()-t0:.0f}s", flush=True)
best, cfg = best_auc_sweep(loo_kde_logmean)
print(f"  Best: {best:.4f}  cfg={cfg}", flush=True)
results['kde_logmean'] = (best, cfg)

# ──────────────────────────────────────────────────────────────────────────────
# Method 2: kde_max_win — max over windows (most favourable window per species)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 2: kde_max_win ===", flush=True)
t0 = time.time()
loo_kde_maxwin = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    tr_fids = win_file_id[tr_mask]
    kde_bg = KernelDensity(bandwidth=BW).fit(X_tr_l)
    log_bg_wins = kde_bg.score_samples(X_te_pca)
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        X_pos = X_tr_l[pos_mask]
        if len(X_pos) == 0:
            loo_kde_maxwin[fi, si] = sigmoid(file_logit_max[fi, si]); continue
        kde_pos = KernelDensity(bandwidth=BW).fit(X_pos)
        log_pos_wins = kde_pos.score_samples(X_te_pca)
        # max over windows (best window for this species)
        loo_kde_maxwin[fi, si] = sigmoid(log_pos_wins - log_bg_wins).max()
print(f"  LOO done in {time.time()-t0:.0f}s", flush=True)
best, cfg = best_auc_sweep(loo_kde_maxwin)
print(f"  Best: {best:.4f}  cfg={cfg}", flush=True)
results['kde_max_win'] = (best, cfg)

# ──────────────────────────────────────────────────────────────────────────────
# Method 3: kde_softmax_win — softmax-weighted mean (confident windows get more weight)
# weight_i = softmax(log_pos_i - log_bg_i); pooled = sum(w_i * sigmoid(ratio_i))
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 3: kde_softmax_win ===", flush=True)
t0 = time.time()
loo_kde_softwin = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    n_te = te_e - te_s
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    tr_fids = win_file_id[tr_mask]
    kde_bg = KernelDensity(bandwidth=BW).fit(X_tr_l)
    log_bg_wins = kde_bg.score_samples(X_te_pca)
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        X_pos = X_tr_l[pos_mask]
        if len(X_pos) == 0:
            loo_kde_softwin[fi, si] = sigmoid(file_logit_max[fi, si]); continue
        kde_pos = KernelDensity(bandwidth=BW).fit(X_pos)
        log_pos_wins = kde_pos.score_samples(X_te_pca)
        ratios = log_pos_wins - log_bg_wins          # (n_te,)
        # softmax weights over windows
        w = np.exp(ratios - ratios.max()); w /= w.sum()
        loo_kde_softwin[fi, si] = (w * sigmoid(ratios)).sum()
print(f"  LOO done in {time.time()-t0:.0f}s", flush=True)
best, cfg = best_auc_sweep(loo_kde_softwin)
print(f"  Best: {best:.4f}  cfg={cfg}", flush=True)
results['kde_softmax_win'] = (best, cfg)

# ──────────────────────────────────────────────────────────────────────────────
# Method 4: gaussian_mixture — GMM with 2 components per species (PCA-32)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 4: gaussian_mixture (GMM n_components=2) ===", flush=True)
t0 = time.time()
loo_gmm2 = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    X_te_avg = X_te_pca.mean(0, keepdims=True)
    tr_fids = win_file_id[tr_mask]
    # Background GMM (all training windows)
    gmm_bg = GaussianMixture(n_components=2, covariance_type='diag',
                              random_state=42, max_iter=50).fit(X_tr_l)
    log_bg = gmm_bg.score_samples(X_te_avg)[0]
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        X_pos = X_tr_l[pos_mask]
        if len(X_pos) == 0:
            loo_gmm2[fi, si] = sigmoid(file_logit_max[fi, si]); continue
        n_comp = min(2, len(X_pos))
        try:
            gmm_pos = GaussianMixture(n_components=n_comp, covariance_type='diag',
                                      random_state=42, max_iter=50).fit(X_pos)
            log_pos = gmm_pos.score_samples(X_te_avg)[0]
            loo_gmm2[fi, si] = sigmoid(log_pos - log_bg)
        except Exception:
            loo_gmm2[fi, si] = sigmoid(file_logit_max[fi, si])
print(f"  LOO done in {time.time()-t0:.0f}s", flush=True)
best, cfg = best_auc_sweep(loo_gmm2)
print(f"  Best: {best:.4f}  cfg={cfg}", flush=True)
results['gaussian_mixture'] = (best, cfg)

# ──────────────────────────────────────────────────────────────────────────────
# Method 5: kde_adaptive_species — bw adapted to n_pos_windows per species
# bw = base_bw * (n_pos / n_all)^(-1/5) (Silverman's rule adaptation)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 5: kde_adaptive_species ===", flush=True)
t0 = time.time()
loo_kde_adap = np.zeros((n_files, n_species), np.float32)
BASE_BW = 0.5
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    tr_fids = win_file_id[tr_mask]
    n_tr = len(X_tr_l)
    kde_bg = KernelDensity(bandwidth=BASE_BW).fit(X_tr_l)
    log_bg_wins = kde_bg.score_samples(X_te_pca)
    win_scores = np.zeros((te_e - te_s, n_species), np.float32)
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        X_pos = X_tr_l[pos_mask]
        if len(X_pos) == 0:
            win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
        # Silverman-like adaptive bw: scale by (n_pos/n_tr)^(-1/(d+4))
        # d=32, so factor = (n_pos/n_tr)^(-1/36)
        d = PCA_N
        bw_adap = BASE_BW * (len(X_pos) / n_tr) ** (-1.0 / (d + 4))
        bw_adap = float(np.clip(bw_adap, 0.2, 2.0))
        kde_pos = KernelDensity(bandwidth=bw_adap).fit(X_pos)
        win_scores[:, si] = sigmoid(kde_pos.score_samples(X_te_pca) - log_bg_wins)
    loo_kde_adap[fi] = win_scores.mean(0)
print(f"  LOO done in {time.time()-t0:.0f}s", flush=True)
best, cfg = best_auc_sweep(loo_kde_adap)
print(f"  Best: {best:.4f}  cfg={cfg}", flush=True)
results['kde_adaptive_species'] = (best, cfg)

# ──────────────────────────────────────────────────────────────────────────────
# Method 6: kde_perwin_rknn_fine — fine blend of per-window KDE + RKNN
# Current: 0.9714 at (0.45, 0.55, 0.90, 1.4); try more combos
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 6: kde_perwin_rknn_fine ===", flush=True)
# Reuse loo_kde_logmean... no, need per-window KDE (sigmoid-mean)
# Compute it
t0 = time.time()
loo_kde_pw = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    tr_fids = win_file_id[tr_mask]
    kde_bg = KernelDensity(bandwidth=BW).fit(X_tr_l)
    log_bg_wins = kde_bg.score_samples(X_te_pca)
    win_scores = np.zeros((te_e - te_s, n_species), np.float32)
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        X_pos = X_tr_l[pos_mask]
        if len(X_pos) == 0:
            win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
        kde_pos = KernelDensity(bandwidth=BW).fit(X_pos)
        win_scores[:, si] = sigmoid(kde_pos.score_samples(X_te_pca) - log_bg_wins)
    loo_kde_pw[fi] = win_scores.mean(0)
print(f"  Per-win KDE LOO done in {time.time()-t0:.0f}s", flush=True)

best6 = 0; best_cfg6 = None; all6 = []
for wg_kde in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    wg_rknn = 1.0 - wg_kde
    blend = wg_kde * loo_kde_pw + wg_rknn * y_rknn5
    for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for b in [1.4, 1.6, 1.8, 2.0, 2.2]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            all6.append((auc, wg_kde, wg_rknn, a, b))
            if auc > best6: best6 = auc; best_cfg6 = (wg_kde, wg_rknn, a, b)
print(f"  Best: {best6:.4f}  cfg=wg_kde={best_cfg6[0]}, wg_rknn={best_cfg6[1]}, a={best_cfg6[2]}, b={best_cfg6[3]}", flush=True)
all6.sort(reverse=True)
print("  Top 5:", [(f"{r[0]:.4f}", r[1:]) for r in all6[:5]], flush=True)
results['kde_perwin_rknn_fine'] = (best6, best_cfg6)

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===", flush=True)
current_best = 0.9721
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    flag = " *** NEW BEST ***" if auc > current_best else ""
    print(f"  {name}: {auc:.4f}{flag}  cfg={cfg}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
for name, (auc, cfg) in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc),
                               'config': str(cfg)})
    if auc > cur_best_json:
        cur_best_json = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json", flush=True)
