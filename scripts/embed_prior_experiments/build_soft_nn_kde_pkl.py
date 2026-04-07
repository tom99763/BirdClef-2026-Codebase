"""Build PKL for soft_nn_kde_blend: 0.10*SoftNN(tau=0.1) + 0.90*KDE(bw0.3*0.15+bw0.5*0.85)"""
import numpy as np, pickle, json, os, shutil
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

def loo_soft_nn(tau):
    out = np.zeros((n_files, n_species), np.float32)
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
        log_w = -D2 / tau
        log_w -= log_w.max(1, keepdims=True)
        w = np.exp(log_w); w /= w.sum(1, keepdims=True)
        win_scores = np.zeros((te_e - te_s, n_species), np.float32)
        for si in range(n_species):
            pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids]).astype(np.float32)
            if not pos_mask.any():
                win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
            win_scores[:, si] = (w * pos_mask).sum(1)
        out[fi] = win_scores.mean(0)
    return out

print("Computing KDE bw=0.3...", flush=True)
kde03 = loo_kde_perwin(0.3)
print("Computing KDE bw=0.5...", flush=True)
kde05 = loo_kde_perwin(0.5)
kde_blend = 0.15 * kde03 + 0.85 * kde05

print("Computing Soft-NN tau=0.1...", flush=True)
snn = loo_soft_nn(0.1)

# Fine sweep
print("\nFine sweep...", flush=True)
best_auc = 0; best_cfg = None
W_SNN = 0.10
combined = W_SNN * snn + (1-W_SNN) * kde_blend
for a in [0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.95]:
    for b in [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]:
        pred = sigmoid(a * base_logit + b * np.log(combined.clip(EPS)))
        auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
        if auc > best_auc: best_auc = auc; best_cfg = (a, b)
print(f"Best LOO-AUC: {best_auc:.4f}  cfg=a={best_cfg[0]}, b={best_cfg[1]}", flush=True)

# Build production PKL
print("\n=== Building production PKL ===", flush=True)
A_best, B_best = best_cfg

pca_prod = PCA(n_components=PCA_N, random_state=42).fit(emb_win_norm)
X_win_pca = pca_prod.transform(emb_win_norm).astype(np.float32)
pca_mean = X_win_pca.mean(0).astype(np.float32)
pca_std  = X_win_pca.std(0).clip(1e-8).astype(np.float32)
X_win_pca_s = ((X_win_pca - pca_mean) / pca_std).astype(np.float32)

species_pos_X = {}
for si in range(n_species):
    pos_file_mask = file_labels[:, si] > 0.5
    if not pos_file_mask.any(): continue
    pos_win_mask = np.isin(win_file_id, np.where(pos_file_mask)[0])
    species_pos_X[si] = X_win_pca_s[pos_win_mask]

model_pkl = {
    'method': 'soft_nn_kde_blend',
    'loo_auc': float(best_auc),
    'config': {
        'pca_n': PCA_N,
        'bw_list': [0.3, 0.5], 'w_list': [0.15, 0.85],  # KDE part
        'snn_tau': 0.1, 'w_snn': W_SNN,                   # Soft-NN part
        'a': A_best, 'b': B_best
    },
    'pca_components': pca_prod.components_.astype(np.float32),
    'pca_mean_raw':   pca_prod.mean_.astype(np.float32),
    'pca_mean':       pca_mean,
    'pca_std':        pca_std,
    'kde_bg_train_X': X_win_pca_s,
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
shutil.copy(pkl_path, "birdclef-2026/notebook resource/current_subs/weights/embed_prior_model.pkl")
print("Copied to weights/", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
rd['experiments'].append({'method': 'soft_nn_kde_blend_validated', 'loo_auc': float(best_auc),
                           'full_auc': float(best_auc), 'config': model_pkl['config']})
if best_auc > rd['best'].get('loo_auc', 0):
    rd['best'] = {'method': 'soft_nn_kde_blend_validated', 'loo_auc': float(best_auc), 'full_auc': float(best_auc)}
    print(f"*** NEW BEST: {best_auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json", flush=True)
