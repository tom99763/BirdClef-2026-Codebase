"""
Build KDE Embed Prior PKL
- Method: kde_per_species with LOO-validated AUC = 0.9560 (proper LOO-PCA)
- Config: bw=1.0, pca_n=32, blend 30% KDE + 70% win_k1, a=0.95, b=1.2
- Full dataset fit (all 66 labeled soundscape files)
- Inference: PCA-32 transform → KDE scoring → blend with win_k1
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────────
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
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi
emb_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)

EPS = 1e-7
print(f"Files: {n_files}, Species: {n_species}, Windows: {len(emb_win)}", flush=True)

# ─── Best config from LOO-validated experiment ─────────────────────────────────
BW     = 1.0    # KDE bandwidth
PCA_N  = 32     # PCA dimensions
WG_KDE = 0.30   # blend weight for KDE
A      = 0.95   # logspace fusion: base coefficient
B      = 1.2    # logspace fusion: embed prior coefficient
print(f"Config: bw={BW}, pca_n={PCA_N}, wg_kde={WG_KDE}, a={A}, b={B}", flush=True)

# ─── Fit PCA on ALL 66 files (production model) ───────────────────────────────
print("Fitting PCA on all 66 files...", flush=True)
pca = PCA(n_components=PCA_N, random_state=42)
pca.fit(emb_norm)
X_pca = pca.transform(emb_norm).astype(np.float32)
pca_mean = X_pca.mean(0).astype(np.float32)
pca_std  = X_pca.std(0).clip(1e-8).astype(np.float32)
X_pca_s  = ((X_pca - pca_mean) / pca_std).astype(np.float32)
print(f"  PCA shape: {X_pca_s.shape}", flush=True)

# ─── Fit KDE models on ALL 66 files ───────────────────────────────────────────
print("Fitting KDE models (background + per species)...", flush=True)
kde_bg = KernelDensity(kernel='gaussian', bandwidth=BW)
kde_bg.fit(X_pca_s)
print("  Background KDE fitted.", flush=True)

# Per-species KDE (only for species with at least 1 positive file)
kde_per_species = {}
n_pos_species = 0
for si in range(n_species):
    pos_idx = np.where(file_labels[:, si] > 0.5)[0]
    if len(pos_idx) == 0:
        continue
    kde_s = KernelDensity(kernel='gaussian', bandwidth=BW)
    kde_s.fit(X_pca_s[pos_idx])
    kde_per_species[si] = kde_s
    n_pos_species += 1
print(f"  Fitted KDE for {n_pos_species}/{n_species} species.", flush=True)

# ─── LOO-CV verification with production PCA (should match ~0.9560) ──────────
print("\nRunning LOO-CV verification (proper LOO-PCA)...", flush=True)
loo_kde_proper = np.zeros((n_files, n_species), np.float32)
for fi_test in range(n_files):
    tr = np.arange(n_files) != fi_test
    # Proper LOO-PCA
    pca_loo = PCA(n_components=PCA_N, random_state=42).fit(emb_norm[tr])
    X_tr_pca = pca_loo.transform(emb_norm[tr]).astype(np.float32)
    X_te_pca = pca_loo.transform(emb_norm[[fi_test]]).astype(np.float32)
    mu_loo = X_tr_pca.mean(0); std_loo = X_tr_pca.std(0).clip(1e-8)
    X_tr_s = (X_tr_pca - mu_loo) / std_loo
    X_te_s = (X_te_pca - mu_loo) / std_loo

    kde_bg_loo = KernelDensity(kernel='gaussian', bandwidth=BW).fit(X_tr_s)
    log_bg_loo = kde_bg_loo.score_samples(X_te_s)[0]
    for si in range(n_species):
        pos = np.where(file_labels[tr, si] > 0.5)[0]
        if len(pos) == 0:
            loo_kde_proper[fi_test, si] = sigmoid(file_logit_max[fi_test, si])
            continue
        kde_pos = KernelDensity(kernel='gaussian', bandwidth=BW).fit(X_tr_s[pos])
        log_pos = kde_pos.score_samples(X_te_s)[0]
        loo_kde_proper[fi_test, si] = sigmoid(log_pos - log_bg_loo)

# Compute win_k1
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = (win_file_id != i)
    X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :1]
    wp = np.zeros((te_e-te_s, n_species), np.float32)
    for wi in range(te_e-te_s):
        ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
        ww = ww/ws if ws > 1e-8 else np.ones(1)
        wp[wi] = (ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)

# Load VLOM base for verification
def vlom_blend(a, b):
    return sigmoid(0.5*np.log(a.clip(EPS)/(1-a).clip(EPS)) + 0.5*np.log(b.clip(EPS)/(1-b).clip(EPS)))
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_by_file = {}
for i, rid in enumerate(sed_npz['row_ids']):
    sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]), []).append(i)
file_sed_max = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fb = fname.replace('.ogg','').replace('.flac','')
    if fb in sed_by_file:
        file_sed_max[fi] = sed_npz['probs'][sed_by_file[fb]].max(0)
base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))

y_blend = WG_KDE * loo_kde_proper + (1-WG_KDE) * y_win_k1
pred = sigmoid(A * base_logit + B * np.log(y_blend.clip(EPS)))
mask = file_labels.sum(0) > 0
loo_auc = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
print(f"LOO-CV AUC (proper LOO-PCA): {loo_auc:.4f}  (expected ~0.9560)", flush=True)

# ─── Save PKL ─────────────────────────────────────────────────────────────────
print("\nSaving PKL...", flush=True)
model_pkl = {
    'method': 'kde_per_species',
    'loo_auc': float(loo_auc),
    'config': {'bw': BW, 'pca_n': PCA_N, 'wg_kde': WG_KDE, 'a': A, 'b': B},
    # PCA + normalization (fit on all 66 files)
    'pca_components': pca.components_.astype(np.float32),   # (32, 1536)
    'pca_mean_raw': pca.mean_.astype(np.float32),           # (1536,)
    'pca_mean': pca_mean,     # mean of PCA output
    'pca_std': pca_std,       # std of PCA output
    # KDE models
    'kde_bg_train_X': X_pca_s,       # (66, 32) training features for bg KDE
    'kde_bandwidth': BW,
    # Per-species positive features
    'species_pos_X': {},              # {si: X_pos (n_pos, 32)}
    # Training data
    'file_embs_norm': emb_norm,       # (66, 1536)
    'file_labels': file_labels,       # (66, 234)
    'file_prob_max': sigmoid(file_logit_max),  # (66, 234)
    'file_logit_max': file_logit_max, # (66, 234)
    'file_list': file_list,
    # Win-k1 training data
    'emb_win_norm': emb_win_norm,     # (739, 1536)
    'win_file_id': win_file_id,       # (739,)
    'n_windows': n_windows,
    'file_start': file_start,
    'file_end': file_end,
}

for si in range(n_species):
    pos_idx = np.where(file_labels[:, si] > 0.5)[0]
    if len(pos_idx) > 0:
        model_pkl['species_pos_X'][si] = X_pca_s[pos_idx]

pkl_path = "outputs/embed_prior_model.pkl"
with open(pkl_path, 'wb') as f:
    pickle.dump(model_pkl, f)
print(f"Saved to {pkl_path}", flush=True)

# File size check
size_mb = os.path.getsize(pkl_path) / 1024**2
print(f"PKL size: {size_mb:.1f} MB", flush=True)

# ─── Update JSON ──────────────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
method_name = 'kde_per_species_validated'
rd['experiments'].append({
    'method': method_name,
    'loo_auc': float(loo_auc),
    'full_auc': float(loo_auc),
    'config': model_pkl['config']
})
if loo_auc > cur_best:
    rd['best'] = {'method': method_name, 'loo_auc': float(loo_auc), 'full_auc': float(loo_auc)}
    print(f"\n*** NEW BEST: {method_name} = {loo_auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json", flush=True)
print("\nDone. Ready to create notebook.", flush=True)
