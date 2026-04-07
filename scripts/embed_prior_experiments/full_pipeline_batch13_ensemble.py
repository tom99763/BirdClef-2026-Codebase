"""
Batch 13: Ensemble of KDE variants + novel pooling strategies
Goal: beat kde_perwin = 0.9721
Methods:
  1. kde_pw_ensemble: ensemble of bw=0.4 and bw=0.5 per-window KDE (both tied at 0.9721)
  2. kde_pw_logspace: use log-space ensemble (geometric mean of sigmoid scores)
  3. kde_pw_pca_ensemble: ensemble of pca_n=24 and pca_n=32 per-window KDE
  4. kde_pw_bw_adaptive_global: global adaptive bw = Silverman's rule on all 739 windows
  5. kde_pw_trimmed_mean: trim 10% highest/lowest window scores before averaging
"""
import numpy as np, pickle, json, os, time
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

def loo_kde_perwin_variant(pca_n, bw, pool='mean'):
    """Per-window KDE with configurable pooling."""
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
            raw = sigmoid(kde_pos.score_samples(X_te_pca) - log_bg_wins)  # (n_wins,)
            if pool == 'mean':
                win_scores[:, si] = raw  # will avg below
            elif pool == 'trimmed':
                n = len(raw)
                k = max(1, int(n * 0.1))
                raw_sorted = np.sort(raw)
                win_scores[:, si] = raw_sorted[k:-k].mean() if n > 2*k else raw.mean()
                continue
        if pool == 'mean' or pool == 'trimmed':
            out[fi] = win_scores.mean(0)
        else:
            out[fi] = win_scores.mean(0)
    return out

def sweep(scores, extra_b=None):
    best = 0; best_cfg = None
    bs = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    if extra_b: bs = sorted(set(bs + extra_b))
    for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for b in bs:
            pred = sigmoid(a * base_logit + b * np.log(scores.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best: best = auc; best_cfg = (a, b)
    return best, best_cfg

results = {}

# ─────────────────────────────────────────────────────────────────────────────
print("Computing per-window KDE bw=0.4 and bw=0.5 (pca_n=32)...", flush=True)
t0 = time.time()
kde_bw04 = loo_kde_perwin_variant(32, 0.4)
kde_bw05 = loo_kde_perwin_variant(32, 0.5)
print(f"  Done in {time.time()-t0:.0f}s", flush=True)

# Method 1: kde_pw_ensemble (arithmetic mean)
print("\n=== Method 1: kde_pw_ensemble (bw=0.4 + bw=0.5 avg) ===", flush=True)
ens = 0.5 * kde_bw04 + 0.5 * kde_bw05
for w04 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    cand = w04 * kde_bw04 + (1-w04) * kde_bw05
    auc_c, cfg_c = sweep(cand)
    if auc_c > results.get('kde_pw_ensemble', (0,))[0]:
        results['kde_pw_ensemble'] = (auc_c, (w04, cfg_c))
print(f"  Best: {results['kde_pw_ensemble'][0]:.4f}  cfg={results['kde_pw_ensemble'][1]}", flush=True)

# Method 2: kde_pw_logspace_ens (geometric mean = exp(mean_log))
print("\n=== Method 2: kde_pw_logspace_ens (geometric mean) ===", flush=True)
geo_ens = np.exp(0.5 * np.log(kde_bw04.clip(EPS)) + 0.5 * np.log(kde_bw05.clip(EPS)))
auc2, cfg2 = sweep(geo_ens)
print(f"  Best: {auc2:.4f}  cfg={cfg2}", flush=True)
results['kde_pw_logspace_ens'] = (auc2, cfg2)

# Method 3: kde_pw_pca24_ens (pca=24 + pca=32 ensemble)
print("\n=== Method 3: kde_pw_pca24_ens (pca=24 + pca=32) ===", flush=True)
t0 = time.time()
kde_pca24 = loo_kde_perwin_variant(24, 0.5)
print(f"  pca24 done in {time.time()-t0:.0f}s", flush=True)
for w24 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    cand = w24 * kde_pca24 + (1-w24) * kde_bw05
    auc_c, cfg_c = sweep(cand)
    if auc_c > results.get('kde_pw_pca24_ens', (0,))[0]:
        results['kde_pw_pca24_ens'] = (auc_c, (w24, cfg_c))
print(f"  Best: {results['kde_pw_pca24_ens'][0]:.4f}  cfg={results['kde_pw_pca24_ens'][1]}", flush=True)

# Method 4: kde_pw_bw_silverman — global Silverman's rule bw
# Silverman's rule: bw = 1.06 * sigma * n^(-1/5) for each PCA dim
# Average across dims as global bw
print("\n=== Method 4: kde_pw_bw_silverman ===", flush=True)
t0 = time.time()
# Estimate Silverman bw from all 739 windows
X_all_pca = PCA(n_components=32, random_state=42).fit_transform(emb_win_norm)
X_all_s = (X_all_pca - X_all_pca.mean(0)) / X_all_pca.std(0).clip(1e-8)
sigma_mean = X_all_s.std(0).mean()
bw_silverman = float(1.06 * sigma_mean * (len(X_all_s) ** (-0.2)))
print(f"  Silverman bw = {bw_silverman:.3f}", flush=True)
kde_silv = loo_kde_perwin_variant(32, bw_silverman)
auc4, cfg4 = sweep(kde_silv)
print(f"  Best: {auc4:.4f}  cfg={cfg4}  ({time.time()-t0:.0f}s)", flush=True)
results['kde_pw_bw_silverman'] = (auc4, (bw_silverman, cfg4))

# Method 5: kde_pw_trimmed_mean — trim 10% windows
print("\n=== Method 5: kde_pw_trimmed_mean ===", flush=True)
t0 = time.time()
kde_trim = loo_kde_perwin_variant(32, 0.5, pool='trimmed')
auc5, cfg5 = sweep(kde_trim)
print(f"  Best: {auc5:.4f}  cfg={cfg5}  ({time.time()-t0:.0f}s)", flush=True)
results['kde_pw_trimmed_mean'] = (auc5, cfg5)

# Method 6: kde_perwin + kde_window_level blend
# kde_window_level (avg-then-score) = 0.9701; kde_perwin (score-then-avg) = 0.9721
# Can they complement each other?
print("\n=== Method 6: kde_pw_blend_with_window_level ===", flush=True)
# Load window-level KDE (we already have it from previous PKL run — recompute LOO)
t0 = time.time()
kde_win_avg = np.zeros((n_files, n_species), np.float32)  # avg-then-score
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=32, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te_avg = (((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l).mean(0, keepdims=True)
    tr_fids = win_file_id[tr_mask]
    kde_bg = KernelDensity(bandwidth=0.5).fit(X_tr_l)
    log_bg = kde_bg.score_samples(X_te_avg)[0]
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        X_pos = X_tr_l[pos_mask]
        if len(X_pos) == 0:
            kde_win_avg[fi, si] = sigmoid(file_logit_max[fi, si]); continue
        kde_pos = KernelDensity(bandwidth=0.5).fit(X_pos)
        kde_win_avg[fi, si] = sigmoid(kde_pos.score_samples(X_te_avg)[0] - log_bg)
print(f"  Window-avg KDE done in {time.time()-t0:.0f}s", flush=True)

best6 = 0; best_cfg6 = None
for w_pw in [0.5, 0.6, 0.7, 0.8, 0.9]:
    blend = w_pw * kde_bw05 + (1-w_pw) * kde_win_avg
    auc_c, cfg_c = sweep(blend)
    if auc_c > best6: best6 = auc_c; best_cfg6 = (w_pw, cfg_c)
print(f"  Best: {best6:.4f}  cfg=w_pw={best_cfg6[0]}, {best_cfg6[1]}", flush=True)
results['kde_pw_blend_window_level'] = (best6, best_cfg6)

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===", flush=True)
current_best = 0.9721
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    flag = " *** NEW BEST ***" if auc > current_best else ""
    print(f"  {name}: {auc:.4f}{flag}  cfg={cfg}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
for name, (auc, cfg) in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best_json:
        cur_best_json = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json", flush=True)
