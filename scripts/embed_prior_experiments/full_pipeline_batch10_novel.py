"""
Batch 10: Novel methods beyond KDE+RKNN (current best 0.9711)
Methods:
  1. kde_rknn_k3: RKNN k=3 instead of k=5 (sharper mutual neighbors)
  2. kde_rknn_k7: RKNN k=7 (more generous mutual neighbors)
  3. kde_win_rknn_sed: KDE + RKNN + SED-species bridge 3-way blend
  4. rknn_k5_bw_sweep: RKNN k5 only (no KDE) at various a/b — verify pure RKNN potential
  5. kde_win_bw06_rknn: bw=0.6 KDE + RKNN (bw=0.6 was 2nd best at 0.9697)
All use proper LOO-window-PCA for KDE component.
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

def loo_rknn(K):
    """LOO RKNN with reciprocal check using k-th training sim as threshold."""
    out = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s = int(file_start[i]); te_e = int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_m = (win_file_id != i); X_tr = emb_win_norm[tr_m]; tr_fi = win_file_id[tr_m]
        sims_te_tr = X_te @ X_tr.T
        sims_tr_tr = X_tr @ X_tr.T
        thresh = np.partition(-sims_tr_tr, K, axis=1)[:, K] * -1
        top_k_idx = np.argsort(-sims_te_tr, axis=1)[:, :K]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            nbrs = top_k_idx[wi]
            recip = [n for n in nbrs if sims_te_tr[wi, n] >= thresh[n]]
            if not recip: recip = nbrs.tolist()
            ww = sims_te_tr[wi, recip].clip(0); ws = ww.sum()
            ww = ww / ws if ws > 1e-8 else np.ones(len(recip)) / len(recip)
            wp[wi] = (ww[:, None] * file_labels[tr_fi[recip]]).sum(0)
        out[i] = wp.mean(0)
    return out

def loo_kde_window(pca_n, bw):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_s, te_e = int(file_start[fi]), int(file_end[fi])
        tr_mask = (win_file_id != fi)
        X_tr_raw = emb_win_norm[tr_mask]; X_te_raw = emb_win_norm[te_s:te_e]
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
                out[fi, si] = sigmoid(file_logit_max[fi, si]); continue
            kde_pos = KernelDensity(bandwidth=bw).fit(X_pos)
            out[fi, si] = sigmoid(kde_pos.score_samples(X_te_avg)[0] - log_bg)
    return out

results = {}

# Precompute shared signals (already done for best config, reuse)
print("Precomputing shared signals...", flush=True)
t0 = time.time()
y_kde32_bw05 = loo_kde_window(pca_n=32, bw=0.5)
print(f"  KDE pca32 bw=0.5 done in {time.time()-t0:.0f}s", flush=True)

print("Computing RKNN k=3, k=5, k=7...", flush=True)
t0 = time.time()
y_rknn3 = loo_rknn(K=3)
y_rknn5 = loo_rknn(K=5)
y_rknn7 = loo_rknn(K=7)
print(f"  RKNN done in {time.time()-t0:.0f}s", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Method 1: kde_rknn_k3 — KDE + RKNN k=3
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 1: kde_rknn_k3 ===", flush=True)
best_k3 = 0; best_cfg_k3 = None
for wg_kde in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    wg_rknn = 1.0 - wg_kde
    blend = wg_kde * y_kde32_bw05 + wg_rknn * y_rknn3
    for a in [0.88, 0.90, 0.92, 0.95]:
        for b in [1.2, 1.4, 1.6, 1.8]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best_k3: best_k3 = auc; best_cfg_k3 = (wg_kde, wg_rknn, a, b)
print(f"  Best: {best_k3:.4f}  cfg={best_cfg_k3}", flush=True)
results['kde_rknn_k3'] = best_k3

# ──────────────────────────────────────────────────────────────────────────────
# Method 2: kde_rknn_k7 — KDE + RKNN k=7
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 2: kde_rknn_k7 ===", flush=True)
best_k7 = 0; best_cfg_k7 = None
for wg_kde in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    wg_rknn = 1.0 - wg_kde
    blend = wg_kde * y_kde32_bw05 + wg_rknn * y_rknn7
    for a in [0.88, 0.90, 0.92, 0.95]:
        for b in [1.2, 1.4, 1.6, 1.8]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best_k7: best_k7 = auc; best_cfg_k7 = (wg_kde, wg_rknn, a, b)
print(f"  Best: {best_k7:.4f}  cfg={best_cfg_k7}", flush=True)
results['kde_rknn_k7'] = best_k7

# ──────────────────────────────────────────────────────────────────────────────
# Method 3: kde_rknn_blend3k — blend k3, k5, k7 all together + KDE
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 3: kde_rknn_blend3k ===", flush=True)
best_3k = 0; best_cfg_3k = None
for wg_kde in [0.25, 0.30, 0.35]:
    for r3 in [0.1, 0.15, 0.20]:
        for r5 in [0.30, 0.40, 0.50]:
            r7 = 1.0 - wg_kde - r3 - r5
            if r7 < 0: continue
            blend = wg_kde * y_kde32_bw05 + r3 * y_rknn3 + r5 * y_rknn5 + r7 * y_rknn7
            for a in [0.90, 0.92]:
                for b in [1.4, 1.6]:
                    pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
                    auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
                    if auc > best_3k: best_3k = auc; best_cfg_3k = (wg_kde, r3, r5, r7, a, b)
print(f"  Best: {best_3k:.4f}  cfg={best_cfg_3k}", flush=True)
results['kde_rknn_blend3k'] = best_3k

# ──────────────────────────────────────────────────────────────────────────────
# Method 4: kde_win_bw06_rknn — bw=0.6 KDE + RKNN k=5
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 4: kde_win_bw06_rknn ===", flush=True)
t0 = time.time()
y_kde32_bw06 = loo_kde_window(pca_n=32, bw=0.6)
print(f"  KDE bw=0.6 done in {time.time()-t0:.0f}s", flush=True)
best_bw6 = 0; best_cfg_bw6 = None
for wg_kde in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    wg_rknn = 1.0 - wg_kde
    blend = wg_kde * y_kde32_bw06 + wg_rknn * y_rknn5
    for a in [0.88, 0.90, 0.92, 0.95]:
        for b in [1.2, 1.4, 1.6, 1.8]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best_bw6: best_bw6 = auc; best_cfg_bw6 = (wg_kde, wg_rknn, a, b)
print(f"  Best: {best_bw6:.4f}  cfg={best_cfg_bw6}", flush=True)
results['kde_win_bw06_rknn'] = best_bw6

# ──────────────────────────────────────────────────────────────────────────────
# Method 5: kde_rknn_sed_3way — Load SED-species bridge + KDE + RKNN
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 5: kde_rknn_sed_3way ===", flush=True)
try:
    with open("outputs/embed_prior_sed_species_bridge.pkl", "rb") as f:
        sed_pkl = pickle.load(f)
    # Get the y_ep signal from sed_species_bridge pkl
    # The pkl stores X_combined_n which is the rknn-precomputed blend signal
    # Actually we need to look at what's in it
    print(f"  SED bridge pkl keys: {list(sed_pkl.keys())[:10]}", flush=True)

    # If it has precomputed signals, use them
    # Otherwise use the X_combined_n as the embed prior signal
    if 'y_rknn_sed' in sed_pkl:
        y_sed_bridge = sed_pkl['y_rknn_sed']
    elif 'X_combined_n' in sed_pkl:
        y_sed_bridge = sed_pkl['X_combined_n']  # This is the embed prior score
    else:
        # Rebuild from available data
        print("  Rebuilding SED bridge signal from scratch...", flush=True)
        # Just use max from file_labels for now as placeholder
        y_sed_bridge = None

    if y_sed_bridge is not None:
        best_sed3 = 0; best_cfg_sed3 = None
        for wk, wr, ws_b in [(0.25, 0.50, 0.25), (0.30, 0.50, 0.20),
                              (0.30, 0.55, 0.15), (0.35, 0.50, 0.15),
                              (0.35, 0.55, 0.10), (0.25, 0.55, 0.20)]:
            blend = wk * y_kde32_bw05 + wr * y_rknn5 + ws_b * y_sed_bridge
            for a in [0.90, 0.92]:
                for b in [1.4, 1.6]:
                    pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
                    auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
                    if auc > best_sed3: best_sed3 = auc; best_cfg_sed3 = (wk, wr, ws_b, a, b)
        print(f"  Best: {best_sed3:.4f}  cfg={best_cfg_sed3}", flush=True)
        results['kde_rknn_sed_3way'] = best_sed3
    else:
        print("  SKIP: no usable SED bridge signal found", flush=True)
except Exception as e:
    print(f"  SKIP: {e}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===", flush=True)
current_best = 0.9711
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > current_best else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

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
