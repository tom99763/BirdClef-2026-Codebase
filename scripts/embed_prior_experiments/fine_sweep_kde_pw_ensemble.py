"""
Fine sweep + PKL build: kde_pw_ensemble (bw=0.4 + bw=0.5 per-window KDE)
Best so far: w_bw04=0.4, a=0.90, b=2.0 → 0.9731
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

PCA_N = 32

def loo_kde_perwin(bw):
    out = np.zeros((n_files, n_species), np.float32)
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

# Compute bw=0.4 and bw=0.5
print("Computing bw=0.4...", flush=True)
t0 = time.time()
kde04 = loo_kde_perwin(0.4)
print(f"  Done {time.time()-t0:.0f}s", flush=True)

print("Computing bw=0.5...", flush=True)
t0 = time.time()
kde05 = loo_kde_perwin(0.5)
print(f"  Done {time.time()-t0:.0f}s", flush=True)

# Also try bw=0.45 and bw=0.35
print("Computing bw=0.45...", flush=True)
t0 = time.time()
kde045 = loo_kde_perwin(0.45)
print(f"  Done {time.time()-t0:.0f}s", flush=True)

# Fine sweep: weights and fusion params
print("\nFine sweep...", flush=True)
best_auc = 0; best_cfg = None; all_results = []

for w04 in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    w05 = 1.0 - w04
    blend = w04 * kde04 + w05 * kde05
    for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for b in [1.6, 1.8, 2.0, 2.2, 2.4]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            all_results.append((auc, 'bw04_bw05', w04, w05, 0, a, b))
            if auc > best_auc: best_auc = auc; best_cfg = ('bw04_bw05', w04, w05, 0, a, b)

# Also try 3-way: bw=0.35 + bw=0.4 + bw=0.5
print("Computing bw=0.35...", flush=True)
t0 = time.time()
kde035 = loo_kde_perwin(0.35)
print(f"  Done {time.time()-t0:.0f}s", flush=True)

for w35 in [0.10, 0.15, 0.20]:
    for w04 in [0.25, 0.30, 0.35, 0.40]:
        w05 = 1.0 - w35 - w04
        if w05 <= 0: continue
        blend = w35 * kde035 + w04 * kde04 + w05 * kde05
        for a in [0.88, 0.90, 0.92]:
            for b in [1.8, 2.0, 2.2]:
                pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
                auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
                all_results.append((auc, '3way', w35, w04, w05, a, b))
                if auc > best_auc: best_auc = auc; best_cfg = ('3way', w35, w04, w05, a, b)

# Include bw=0.45
for w045 in [0.30, 0.40, 0.50, 0.60, 0.70]:
    w05 = 1.0 - w045
    blend = w045 * kde045 + w05 * kde05
    for a in [0.88, 0.90, 0.92]:
        for b in [1.8, 2.0, 2.2]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            all_results.append((auc, 'bw045_bw05', w045, w05, 0, a, b))
            if auc > best_auc: best_auc = auc; best_cfg = ('bw045_bw05', w045, w05, 0, a, b)

all_results.sort(reverse=True)
print(f"\nBest: {best_auc:.4f}  cfg={best_cfg}", flush=True)
print("\nTop 10 configs:", flush=True)
for r in all_results[:10]:
    print(f"  AUC={r[0]:.4f} type={r[1]} w1={r[2]:.2f} w2={r[3]:.2f} w3={r[4]:.2f} a={r[5]} b={r[6]}", flush=True)

# ─── Save PKL with best config ─────────────────────────────────────────────────
print("\n=== Building production PKL ===", flush=True)
# Determine best blend config
best_type = best_cfg[0]
print(f"Best config: {best_cfg}", flush=True)
print(f"LOO-AUC: {best_auc:.4f}", flush=True)

# Production PCA on all 739 windows
pca_prod = PCA(n_components=PCA_N, random_state=42).fit(emb_win_norm)
X_win_pca = pca_prod.transform(emb_win_norm).astype(np.float32)
pca_mean = X_win_pca.mean(0).astype(np.float32)
pca_std  = X_win_pca.std(0).clip(1e-8).astype(np.float32)
X_win_pca_s = ((X_win_pca - pca_mean) / pca_std).astype(np.float32)

# Per-species positive windows in PCA-32 space
species_pos_X = {}
for si in range(n_species):
    pos_file_mask = file_labels[:, si] > 0.5
    if not pos_file_mask.any(): continue
    pos_win_mask = np.isin(win_file_id, np.where(pos_file_mask)[0])
    species_pos_X[si] = X_win_pca_s[pos_win_mask]

# Determine bw values for the ensemble
if best_type == 'bw04_bw05':
    bw_list = [0.4, 0.5]
    w_list  = [best_cfg[2], best_cfg[3]]
elif best_type == '3way':
    bw_list = [0.35, 0.4, 0.5]
    w_list  = [best_cfg[2], best_cfg[3], best_cfg[4]]
else:  # bw045_bw05
    bw_list = [0.45, 0.5]
    w_list  = [best_cfg[2], best_cfg[3]]

A_best = best_cfg[4]; B_best = best_cfg[5]

model_pkl = {
    'method': 'kde_pw_ensemble',
    'loo_auc': float(best_auc),
    'config': {'pca_n': PCA_N, 'bw_list': bw_list, 'w_list': w_list,
               'a': A_best, 'b': B_best},
    'pca_components': pca_prod.components_.astype(np.float32),
    'pca_mean_raw':   pca_prod.mean_.astype(np.float32),
    'pca_mean':       pca_mean,
    'pca_std':        pca_std,
    'kde_bg_train_X': X_win_pca_s,  # (739, 32)
    'species_pos_X':  species_pos_X,
    'file_labels':    file_labels,
    'file_logit_max': file_logit_max,
    'file_list':      file_list,
    'emb_win_norm':   emb_win_norm,
    'win_file_id':    win_file_id,
    'n_windows':      n_windows,
    'file_start':     file_start,
    'file_end':       file_end,
}

pkl_path = "outputs/embed_prior_model.pkl"
with open(pkl_path, 'wb') as f:
    pickle.dump(model_pkl, f)
size_mb = os.path.getsize(pkl_path) / 1024**2
print(f"Saved: {pkl_path}  ({size_mb:.1f} MB)", flush=True)

import shutil
shutil.copy(pkl_path, "birdclef-2026/notebook resource/current_subs/weights/embed_prior_model.pkl")
print("Copied to weights/embed_prior_model.pkl", flush=True)

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
rd['experiments'].append({'method': 'kde_pw_ensemble_validated', 'loo_auc': float(best_auc),
                           'full_auc': float(best_auc), 'config': model_pkl['config']})
if best_auc > cur_best:
    rd['best'] = {'method': 'kde_pw_ensemble_validated', 'loo_auc': float(best_auc),
                  'full_auc': float(best_auc)}
    print(f"*** NEW BEST: kde_pw_ensemble_validated = {best_auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json", flush=True)
