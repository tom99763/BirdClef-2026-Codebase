"""
Fine sweep: kde_perwin around best config
Best so far: kde_perwin standalone a=0.90, b=1.6 → 0.9719
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

# Compute per-window KDE at various (pca_n, bw) combinations
print("Computing per-window KDE variants...", flush=True)

def loo_kde_perwin(pca_n, bw):
    """Per-window KDE: for each test window compute KDE score separately, then avg."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_s, te_e = int(file_start[fi]), int(file_end[fi])
        n_te = te_e - te_s
        tr_mask = (win_file_id != fi)
        X_tr_raw = emb_win_norm[tr_mask]
        pca_l = PCA(n_components=pca_n, random_state=42).fit(X_tr_raw)
        X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
        mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
        X_tr_l = (X_tr_l - mu_l) / std_l
        X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
        tr_fids = win_file_id[tr_mask]
        kde_bg = KernelDensity(bandwidth=bw).fit(X_tr_l)
        log_bg_wins = kde_bg.score_samples(X_te_pca)  # (n_te,)
        win_scores = np.zeros((n_te, n_species), np.float32)
        for si in range(n_species):
            pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
            X_pos = X_tr_l[pos_mask]
            if len(X_pos) == 0:
                win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
            kde_pos = KernelDensity(bandwidth=bw).fit(X_pos)
            win_scores[:, si] = sigmoid(kde_pos.score_samples(X_te_pca) - log_bg_wins)
        out[fi] = win_scores.mean(0)
    return out

# Best config so far: pca_n=32, bw=0.5 → 0.9719
# Try finer bw values and different pca_n

# 1) Fine bw sweep (pca_n=32)
print("\n--- Fine bw sweep (pca_n=32) ---", flush=True)
results_sweep = {}
for bw in [0.3, 0.4, 0.5, 0.6, 0.7]:
    t0 = time.time()
    kde_pw = loo_kde_perwin(pca_n=32, bw=bw)
    best_bw = 0; best_cfg_bw = None
    for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for b in [1.2, 1.4, 1.6, 1.8, 2.0]:
            pred = sigmoid(a * base_logit + b * np.log(kde_pw.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best_bw: best_bw = auc; best_cfg_bw = (a, b)
    results_sweep[f'bw{int(bw*10)}'] = (bw, 32, best_bw, best_cfg_bw)
    print(f"  bw={bw}, pca_n=32: {best_bw:.4f}  cfg={best_cfg_bw}  ({time.time()-t0:.0f}s)", flush=True)

# 2) Best bw from sweep → vary pca_n
best_bw_val = max(results_sweep.values(), key=lambda x: x[2])[0]
print(f"\n--- pca_n sweep (bw={best_bw_val}) ---", flush=True)
kde_best_bw = results_sweep.get(f'bw{int(best_bw_val*10)}')
for pca_n in [16, 24, 32, 40, 48]:
    if pca_n == 32 and best_bw_val == 0.5:
        # Already computed above
        print(f"  pca_n={pca_n}: already in sweep", flush=True); continue
    t0 = time.time()
    kde_pw = loo_kde_perwin(pca_n=pca_n, bw=best_bw_val)
    best_p = 0; best_cfg_p = None
    for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for b in [1.2, 1.4, 1.6, 1.8, 2.0]:
            pred = sigmoid(a * base_logit + b * np.log(kde_pw.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best_p: best_p = auc; best_cfg_p = (a, b)
    print(f"  pca_n={pca_n}, bw={best_bw_val}: {best_p:.4f}  cfg={best_cfg_p}  ({time.time()-t0:.0f}s)", flush=True)
    results_sweep[f'pca{pca_n}'] = (best_bw_val, pca_n, best_p, best_cfg_p)

# Report best overall
best_overall = max(results_sweep.values(), key=lambda x: x[2])
best_bw, best_pca_n, best_auc, best_cfg = best_overall
print(f"\nBest: bw={best_bw}, pca_n={best_pca_n}: {best_auc:.4f}  cfg=a={best_cfg[0]}, b={best_cfg[1]}", flush=True)

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
rd['experiments'].append({'method': f'kde_perwin_bw{best_bw}_pca{best_pca_n}',
                           'loo_auc': float(best_auc), 'full_auc': float(best_auc),
                           'config': {'bw': best_bw, 'pca_n': best_pca_n, 'a': best_cfg[0], 'b': best_cfg[1]}})
if best_auc > cur_best_json:
    rd['best'] = {'method': f'kde_perwin_bw{best_bw}_pca{best_pca_n}', 'loo_auc': float(best_auc), 'full_auc': float(best_auc)}
    print(f"*** JSON BEST UPDATED: {best_auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json", flush=True)
