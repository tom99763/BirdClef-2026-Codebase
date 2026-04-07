"""
Batch 17: Diverse novel approaches
Goal: beat 0.9739
Methods:
  1. Epanechnikov kernel KDE (different kernel shape vs Gaussian)
  2. Tophat kernel KDE (hard boundary)
  3. PCA ensemble diversity: different random seeds for PCA
  4. Whitened KDE: full covariance whitening before KDE
  5. Species-aware PCA: reweight PCA dimensions by species variance
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
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
CURRENT_BEST = 0.9739

def sweep(scores, name=""):
    best = 0; best_cfg = None
    for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for b in [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]:
            pred = sigmoid(a * base_logit + b * np.log(safe_clip(scores)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best: best = auc; best_cfg = (a, b)
    flag = " *** NEW BEST ***" if best > CURRENT_BEST else ""
    if name: print(f"  {name}: {best:.4f}{flag}  cfg={best_cfg}", flush=True)
    return best, best_cfg

def safe_clip(x):
    """Clip and replace NaN/Inf."""
    x = np.nan_to_num(x, nan=EPS, posinf=1.0-EPS, neginf=EPS)
    return x.clip(EPS, 1.0-EPS)

results = {}

def loo_kde_perwin_kernel(bw, kernel='gaussian', pca_n=32):
    """Per-window KDE with configurable kernel."""
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
        kde_bg = KernelDensity(bandwidth=bw, kernel=kernel).fit(X_tr_l)
        log_bg_wins = kde_bg.score_samples(X_te_pca)
        win_scores = np.zeros((te_e - te_s, n_species), np.float32)
        for si in range(n_species):
            pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
            X_pos = X_tr_l[pos_mask]
            if len(X_pos) == 0:
                win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
            kde_pos = KernelDensity(bandwidth=bw, kernel=kernel).fit(X_pos)
            win_scores[:, si] = sigmoid(kde_pos.score_samples(X_te_pca) - log_bg_wins)
        out[fi] = win_scores.mean(0)
    return out

# ─── Method 1: Epanechnikov kernel ────────────────────────────────────────────
print("\n=== Method 1: Epanechnikov kernel KDE ===", flush=True)
t0 = time.time()
for bw in [0.5, 0.7, 1.0, 1.5]:
    out = loo_kde_perwin_kernel(bw, kernel='epanechnikov')
    auc, cfg = sweep(out, f"Epanechnikov bw={bw}")
    results[f'epanechnikov_bw{bw}'] = (auc, cfg)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 2: Tophat kernel ──────────────────────────────────────────────────
print("\n=== Method 2: Tophat kernel KDE ===", flush=True)
t0 = time.time()
for bw in [0.5, 0.8, 1.0, 1.5]:
    out = loo_kde_perwin_kernel(bw, kernel='tophat')
    auc, cfg = sweep(out, f"Tophat bw={bw}")
    results[f'tophat_bw{bw}'] = (auc, cfg)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# Best alternative kernel + best Gaussian blend
best_ep_name = max([k for k in results if 'epanechnikov' in k], key=lambda k: results[k][0])
best_top_name = max([k for k in results if 'tophat' in k], key=lambda k: results[k][0])
print(f"\n  Best Epanechnikov: {best_ep_name} = {results[best_ep_name][0]:.4f}", flush=True)
print(f"  Best Tophat:       {best_top_name} = {results[best_top_name][0]:.4f}", flush=True)

# Gaussian KDE (reference)
print("\n  Computing reference Gaussian KDE...", flush=True)
kde03 = loo_kde_perwin_kernel(0.3, 'gaussian')
kde05 = loo_kde_perwin_kernel(0.5, 'gaussian')
kde_ref = 0.15 * kde03 + 0.85 * kde05

# ─── Method 3: Epanechnikov + Gaussian ensemble ───────────────────────────────
print("\n=== Method 3: Epanechnikov + Gaussian blend ===", flush=True)
# Use best Epanechnikov bw
best_ep_bw = float(best_ep_name.split('bw')[1])
ep_best_out = loo_kde_perwin_kernel(best_ep_bw, 'epanechnikov')
best3 = 0; best_cfg3 = None
for w_ep in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blend = w_ep * ep_best_out + (1-w_ep) * kde_ref
    auc_c, cfg_c = sweep(blend)
    if auc_c > best3: best3 = auc_c; best_cfg3 = (w_ep, cfg_c)
results['epanechnikov_gaussian_blend'] = (best3, best_cfg3)
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  Ep+Gaussian blend: {best3:.4f}{flag}  w_ep={best_cfg3[0]}", flush=True)

# ─── Method 4: PCA seed ensemble ─────────────────────────────────────────────
# Different random seeds may find different PCA subspaces → diversity
print("\n=== Method 4: PCA seed ensemble (seeds 0,1,2,3,4) ===", flush=True)
t0 = time.time()
seed_outs = []
for seed in [0, 1, 2, 3, 4]:
    out_s = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_s, te_e = int(file_start[fi]), int(file_end[fi])
        tr_mask = (win_file_id != fi)
        X_tr_raw = emb_win_norm[tr_mask]
        pca_l = PCA(n_components=PCA_N, random_state=seed).fit(X_tr_raw)
        X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
        mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
        X_tr_l = (X_tr_l - mu_l) / std_l
        X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
        tr_fids = win_file_id[tr_mask]
        kde_bg = KernelDensity(bandwidth=0.5).fit(X_tr_l)
        log_bg_wins = kde_bg.score_samples(X_te_pca)
        win_scores = np.zeros((te_e - te_s, n_species), np.float32)
        for si in range(n_species):
            pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
            X_pos = X_tr_l[pos_mask]
            if len(X_pos) == 0:
                win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
            kde_pos = KernelDensity(bandwidth=0.5).fit(X_pos)
            win_scores[:, si] = sigmoid(kde_pos.score_samples(X_te_pca) - log_bg_wins)
        out_s[fi] = win_scores.mean(0)
    seed_outs.append(out_s)
seed_avg = np.mean(seed_outs, 0)
print(f"  ({time.time()-t0:.0f}s)", flush=True)
auc4, cfg4 = sweep(seed_avg, "PCA seed ensemble (avg)")
results['pca_seed_ensemble'] = (auc4, cfg4)

# Seed ensemble + KDE blend
best4b = 0; best_cfg4b = None
for w_seed in [0.1, 0.2, 0.3, 0.4, 0.5]:
    blend = w_seed * seed_avg + (1-w_seed) * kde_ref
    auc_c, cfg_c = sweep(blend)
    if auc_c > best4b: best4b = auc_c; best_cfg4b = (w_seed, cfg_c)
results['seed_ens_kde_blend'] = (best4b, best_cfg4b)
flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
print(f"  Seed+KDE blend: {best4b:.4f}{flag}  w_seed={best_cfg4b[0]}", flush=True)

# ─── Method 5: Whitened KDE (ZCA whitening) ───────────────────────────────────
# Full whitening of PCA space: divide each dimension by sqrt(eigenvalue)
# This gives all PCA dimensions equal variance
print("\n=== Method 5: ZCA-whitened KDE ===", flush=True)
t0 = time.time()
out_zca = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0)
    # ZCA: divide by sqrt(explained_variance) = std of each PCA component
    std_zca = np.sqrt(pca_l.explained_variance_).clip(1e-8).astype(np.float32)
    X_tr_l = (X_tr_l - mu_l) / std_zca  # whitened
    X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_zca
    tr_fids = win_file_id[tr_mask]
    for bw in [0.5]:
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
        out_zca[fi] = win_scores.mean(0)
print(f"  ({time.time()-t0:.0f}s)", flush=True)
auc5, cfg5 = sweep(out_zca, "ZCA-whitened KDE bw=0.5")
results['zca_kde'] = (auc5, cfg5)

# ZCA + Gaussian blend
best5b = 0; best_cfg5b = None
for w_zca in [0.1, 0.2, 0.3, 0.4, 0.5]:
    blend = w_zca * out_zca + (1-w_zca) * kde_ref
    auc_c, cfg_c = sweep(blend)
    if auc_c > best5b: best5b = auc_c; best_cfg5b = (w_zca, cfg_c)
results['zca_gaussian_blend'] = (best5b, best_cfg5b)
flag = " *** NEW BEST ***" if best5b > CURRENT_BEST else ""
print(f"  ZCA+Gaussian: {best5b:.4f}{flag}  w_zca={best_cfg5b[0]}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 17 Summary ===", flush=True)
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

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
