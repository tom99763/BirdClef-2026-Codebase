"""
Build KDE+RKNN Blend PKL
- Method: kde_win_rknn_blend
- Config: pca_n=32, bw=0.5, wg_kde=0.35, wg_rknn=0.65, a=0.92, b=1.4
- Validated LOO-AUC: 0.9711
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
file_embs_avg  = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_embs_avg[fi]  = emb_win[s:e].mean(0)
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

EPS = 1e-7
PCA_N = 32; BW = 0.5
WG_KDE = 0.35; WG_RKNN = 0.65
A = 0.92; B = 1.4
K_RKNN = 5
print(f"Config: pca_n={PCA_N}, bw={BW}, wg_kde={WG_KDE}, wg_rknn={WG_RKNN}, a={A}, b={B}", flush=True)

# ─── Fit production PCA on all 739 windows ─────────────────────────────────────
print("Fitting PCA-32 on all 739 window embeddings...", flush=True)
pca = PCA(n_components=PCA_N, random_state=42).fit(emb_win_norm)
X_win_pca = pca.transform(emb_win_norm).astype(np.float32)
pca_mean = X_win_pca.mean(0).astype(np.float32)
pca_std  = X_win_pca.std(0).clip(1e-8).astype(np.float32)
X_win_pca_s = ((X_win_pca - pca_mean) / pca_std).astype(np.float32)
print(f"  Window PCA shape: {X_win_pca_s.shape}", flush=True)

# ─── Build per-species positive feature matrix ─────────────────────────────────
print("Building per-species window features...", flush=True)
species_pos_X = {}
for si in range(n_species):
    pos_file_mask = file_labels[:, si] > 0.5
    if not pos_file_mask.any(): continue
    pos_win_mask = np.isin(win_file_id, np.where(pos_file_mask)[0])
    species_pos_X[si] = X_win_pca_s[pos_win_mask]
print(f"  Species with positive windows: {len(species_pos_X)}/{n_species}", flush=True)

# ─── LOO-CV verification ────────────────────────────────────────────────────────
print("\nRunning LOO-CV verification...", flush=True)
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

# LOO window-level KDE (proper LOO-window PCA)
t0 = time.time()
loo_kde_win = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]; X_te_raw = emb_win_norm[te_s:te_e]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    X_te_l = pca_l.transform(X_te_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te_avg = ((X_te_l - mu_l) / std_l).mean(0, keepdims=True)
    tr_fids = win_file_id[tr_mask]
    kde_bg = KernelDensity(bandwidth=BW).fit(X_tr_l)
    log_bg = kde_bg.score_samples(X_te_avg)[0]
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        X_pos = X_tr_l[pos_mask]
        if len(X_pos) == 0:
            loo_kde_win[fi, si] = sigmoid(file_logit_max[fi, si]); continue
        kde_pos = KernelDensity(bandwidth=BW).fit(X_pos)
        loo_kde_win[fi, si] = sigmoid(kde_pos.score_samples(X_te_avg)[0] - log_bg)
print(f"  KDE LOO done in {time.time()-t0:.0f}s", flush=True)

# LOO RKNN k5
y_rknn_k5 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s = int(file_start[i]); te_e = int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_m = (win_file_id != i); X_tr = emb_win_norm[tr_m]; tr_fi = win_file_id[tr_m]
    sims_te_tr = X_te @ X_tr.T
    sims_tr_tr = X_tr @ X_tr.T
    thresh = np.partition(-sims_tr_tr, K_RKNN, axis=1)[:, K_RKNN] * -1
    top_k_idx = np.argsort(-sims_te_tr, axis=1)[:, :K_RKNN]
    wp = np.zeros((te_e - te_s, n_species), np.float32)
    for wi in range(te_e - te_s):
        nbrs = top_k_idx[wi]
        recip = [n for n in nbrs if sims_te_tr[wi, n] >= thresh[n]]
        if not recip: recip = nbrs.tolist()
        ww = sims_te_tr[wi, recip].clip(0); ws = ww.sum()
        ww = ww / ws if ws > 1e-8 else np.ones(len(recip)) / len(recip)
        wp[wi] = (ww[:, None] * file_labels[tr_fi[recip]]).sum(0)
    y_rknn_k5[i] = wp.mean(0)

blend = WG_KDE * loo_kde_win + WG_RKNN * y_rknn_k5
pred = sigmoid(A * base_logit + B * np.log(blend.clip(EPS)))
mask = file_labels.sum(0) > 0
loo_auc = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
print(f"LOO-CV AUC: {loo_auc:.4f}  (expected ~0.9711)", flush=True)

# ─── Save PKL ─────────────────────────────────────────────────────────────────
print("\nSaving PKL...", flush=True)
model_pkl = {
    'method': 'kde_win_rknn_blend',
    'loo_auc': float(loo_auc),
    'config': {'pca_n': PCA_N, 'bw': BW, 'wg_kde': WG_KDE, 'wg_rknn': WG_RKNN,
               'k_rknn': K_RKNN, 'a': A, 'b': B},
    # PCA (fit on all 739 windows)
    'pca_components': pca.components_.astype(np.float32),  # (32, 1536)
    'pca_mean_raw':   pca.mean_.astype(np.float32),         # (1536,)
    'pca_mean':       pca_mean,                              # (32,)
    'pca_std':        pca_std,                               # (32,)
    # KDE data
    'kde_bg_train_X': X_win_pca_s,                          # (739, 32) all windows
    'kde_bandwidth':  BW,
    'species_pos_X':  species_pos_X,                        # {si: (n_pos, 32)}
    # Training data (for RKNN inference)
    'file_labels':    file_labels,
    'file_logit_max': file_logit_max,
    'file_list':      file_list,
    'emb_win_norm':   emb_win_norm,    # (739, 1536) for RKNN
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
entry = {'method': 'kde_win_rknn_blend_validated', 'loo_auc': float(loo_auc), 'full_auc': float(loo_auc),
         'config': model_pkl['config']}
rd['experiments'].append(entry)
if loo_auc > cur_best:
    rd['best'] = {'method': 'kde_win_rknn_blend_validated', 'loo_auc': float(loo_auc), 'full_auc': float(loo_auc)}
    print(f"\n*** NEW BEST: kde_win_rknn_blend_validated = {loo_auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")
