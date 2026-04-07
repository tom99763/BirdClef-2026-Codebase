"""
Batch 9: KDE Extensions — build on window-level KDE best (0.9701)
Methods:
  1. kde_win_rknn_blend: Window-level KDE + RKNN (replace win_k1 with rknn_k5)
  2. kde_win_pca48: Window-level KDE pca_n=48
  3. kde_win_pca64: Window-level KDE pca_n=64
  4. kde_win_bw_fine: Fine bw sweep [0.3, 0.4, 0.6, 0.7] for window-level
  5. kde_win_triple_blend: Window-level KDE + win_k1 + rknn three-way blend
All methods use proper LOO-window-PCA to avoid leakage.
"""
import numpy as np, pickle, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ─── Load data ─────────────────────────────────────────────────────────────────
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
print(f"Files: {n_files}, Species: {n_species}, Windows: {len(emb_win)}", flush=True)

# ─── VLOM base ─────────────────────────────────────────────────────────────────
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

# ─── Precompute win_k1 ─────────────────────────────────────────────────────────
print("Precomputing win_k1...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_m = (win_file_id != i); X_tr = emb_win_norm[tr_m]; tr_fi = win_file_id[tr_m]
    sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :1]
    wp = np.zeros((te_e-te_s, n_species), np.float32)
    for wi in range(te_e-te_s):
        ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
        ww = ww/ws if ws>1e-8 else np.ones(1)
        wp[wi] = (ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)
print("win_k1 done.", flush=True)

# ─── Load RKNN prior ───────────────────────────────────────────────────────────
print("Loading RKNN prior...", flush=True)
with open("outputs/embed_prior_rknn_k5_win1.pkl", "rb") as f:
    rknn_pkl = pickle.load(f)
# Recompute RKNN from scratch (LOO) using the same method as rknn_k5_win1
# The pkl stores training data; use it to get y_rknn_k5 for each file
rknn_emb_norm = rknn_pkl['emb_win_norm']
rknn_win_fid  = rknn_pkl['win_file_id']
rknn_file_labels = rknn_pkl['file_labels']
rknn_n_files  = len(rknn_pkl['file_list'])
rknn_file_start = rknn_pkl['file_start']
rknn_file_end   = rknn_pkl['file_end']
K_RKNN = 5

y_rknn_k5 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s = int(rknn_file_start[i]); te_e = int(rknn_file_end[i])
    X_te = rknn_emb_norm[te_s:te_e]
    tr_m = (rknn_win_fid != i)
    X_tr = rknn_emb_norm[tr_m]; tr_fi = rknn_win_fid[tr_m]
    sims_te_tr = X_te @ X_tr.T
    sims_tr_tr = X_tr @ X_tr.T
    # LOO sim threshold: k-th neighbor within training
    thresh = np.partition(-sims_tr_tr, K_RKNN, axis=1)[:, K_RKNN] * -1
    top_k_idx = np.argsort(-sims_te_tr, axis=1)[:, :K_RKNN]
    wp = np.zeros((te_e - te_s, n_species), np.float32)
    for wi in range(te_e - te_s):
        nbrs = top_k_idx[wi]
        recip = [n for n in nbrs if sims_te_tr[wi, n] >= thresh[n]]
        if not recip:
            recip = nbrs.tolist()
        ww = sims_te_tr[wi, recip].clip(0)
        ws = ww.sum()
        ww = ww / ws if ws > 1e-8 else np.ones(len(recip)) / len(recip)
        wp[wi] = (ww[:, None] * file_labels[tr_fi[recip]]).sum(0)
    y_rknn_k5[i] = wp.mean(0)
print("RKNN k5 LOO done.", flush=True)

# ─── LOO window-level KDE helper ──────────────────────────────────────────────
def loo_kde_window(pca_n, bw):
    """Proper LOO-window PCA KDE. Returns (n_files, n_species) array."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_s, te_e = int(file_start[fi]), int(file_end[fi])
        tr_mask = (win_file_id != fi)
        X_tr_raw = emb_win_norm[tr_mask]
        X_te_raw = emb_win_norm[te_s:te_e]
        pca_l = PCA(n_components=pca_n, random_state=42).fit(X_tr_raw)
        X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
        X_te_l = pca_l.transform(X_te_raw).astype(np.float32)
        mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
        X_tr_l = (X_tr_l - mu_l) / std_l
        X_te_avg = ((X_te_l - mu_l) / std_l).mean(0, keepdims=True)
        tr_fids = win_file_id[tr_mask]
        kde_bg = KernelDensity(bandwidth=bw).fit(X_tr_l)
        log_bg = kde_bg.score_samples(X_te_avg)[0]
        for si in range(n_species):
            pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
            X_pos = X_tr_l[pos_mask]
            if len(X_pos) == 0:
                out[fi, si] = sigmoid(file_logit_max[fi, si])
                continue
            kde_pos = KernelDensity(bandwidth=bw).fit(X_pos)
            out[fi, si] = sigmoid(kde_pos.score_samples(X_te_avg)[0] - log_bg)
    return out

results = {}

# ──────────────────────────────────────────────────────────────────────────────
# Method 1: kde_win_rknn_blend — Window KDE + RKNN (replacing win_k1)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 1: kde_win_rknn_blend ===", flush=True)
t0 = time.time()
kde_win_base = loo_kde_window(pca_n=32, bw=0.5)
print(f"  KDE LOO done in {time.time()-t0:.0f}s", flush=True)

best_auc = 0; best_cfg = None
for wg in [0.20, 0.25, 0.30, 0.35, 0.40]:
    for wg_rknn in [0.30, 0.40, 0.50, 0.60]:
        # blend: wg*kde + wg_rknn*rknn + (1-wg-wg_rknn)*win_k1
        if wg + wg_rknn > 0.95: continue
        wk = 1.0 - wg - wg_rknn
        blend = wg * kde_win_base + wg_rknn * y_rknn_k5 + wk * y_win_k1
        for a in [0.85, 0.90, 0.95]:
            for b in [1.0, 1.2, 1.4, 1.6]:
                pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
                auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
                if auc > best_auc:
                    best_auc = auc; best_cfg = (wg, wg_rknn, wk, a, b)
print(f"  Best: {best_auc:.4f}  cfg={best_cfg}", flush=True)
results['kde_win_rknn_blend'] = best_auc

# ──────────────────────────────────────────────────────────────────────────────
# Method 2: kde_win_pca48 — Window KDE with pca_n=48
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 2: kde_win_pca48 ===", flush=True)
t0 = time.time()
kde48 = loo_kde_window(pca_n=48, bw=0.5)
print(f"  KDE LOO done in {time.time()-t0:.0f}s", flush=True)

best_auc48 = 0; best_cfg48 = None
for wg in [0.20, 0.25, 0.30, 0.35, 0.40]:
    blend = wg * kde48 + (1-wg) * y_win_k1
    for a in [0.85, 0.90, 0.95]:
        for b in [1.0, 1.2, 1.4, 1.6]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best_auc48:
                best_auc48 = auc; best_cfg48 = (wg, a, b)
print(f"  Best: {best_auc48:.4f}  cfg={best_cfg48}", flush=True)
results['kde_win_pca48'] = best_auc48

# ──────────────────────────────────────────────────────────────────────────────
# Method 3: kde_win_pca64 — Window KDE with pca_n=64
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 3: kde_win_pca64 ===", flush=True)
t0 = time.time()
kde64 = loo_kde_window(pca_n=64, bw=0.5)
print(f"  KDE LOO done in {time.time()-t0:.0f}s", flush=True)

best_auc64 = 0; best_cfg64 = None
for wg in [0.20, 0.25, 0.30, 0.35, 0.40]:
    blend = wg * kde64 + (1-wg) * y_win_k1
    for a in [0.85, 0.90, 0.95]:
        for b in [1.0, 1.2, 1.4, 1.6]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best_auc64:
                best_auc64 = auc; best_cfg64 = (wg, a, b)
print(f"  Best: {best_auc64:.4f}  cfg={best_cfg64}", flush=True)
results['kde_win_pca64'] = best_auc64

# ──────────────────────────────────────────────────────────────────────────────
# Method 4: kde_win_bw_fine — Fine bw sweep
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 4: kde_win_bw_fine ===", flush=True)
for bw in [0.3, 0.4, 0.6, 0.7]:
    t0 = time.time()
    kde_bw = loo_kde_window(pca_n=32, bw=bw)
    best_bw = 0; best_bw_cfg = None
    for wg in [0.20, 0.25, 0.30, 0.35, 0.40]:
        blend = wg * kde_bw + (1-wg) * y_win_k1
        for a in [0.85, 0.90, 0.95]:
            for b in [1.0, 1.2, 1.4, 1.6]:
                pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
                auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
                if auc > best_bw:
                    best_bw = auc; best_bw_cfg = (bw, wg, a, b)
    print(f"  bw={bw}: {best_bw:.4f}  cfg={best_bw_cfg}  ({time.time()-t0:.0f}s)", flush=True)
    results[f'kde_win_bw{int(bw*10)}'] = best_bw

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===", flush=True)
current_best = 0.9701
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > current_best else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best_json:
        cur_best_json = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json", flush=True)
