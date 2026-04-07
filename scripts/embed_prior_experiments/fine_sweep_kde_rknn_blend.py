"""
Fine sweep: kde_win_rknn_blend around best config
Best so far: wg_kde=0.30, wg_rknn=0.60, wk=0.10, a=0.95, b=1.6 → 0.9711
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

# win_k1
print("Precomputing win_k1 + RKNN k5...", flush=True)
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

# RKNN k5
K_RKNN = 5
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
print("Done.", flush=True)

# Window-level KDE (pca_n=32, bw=0.5)
print("Computing window-level KDE (pca_n=32, bw=0.5)...", flush=True)
t0 = time.time()
BW, PCA_N = 0.5, 32
kde_win = np.zeros((n_files, n_species), np.float32)
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
            kde_win[fi, si] = sigmoid(file_logit_max[fi, si]); continue
        kde_pos = KernelDensity(bandwidth=BW).fit(X_pos)
        kde_win[fi, si] = sigmoid(kde_pos.score_samples(X_te_avg)[0] - log_bg)
print(f"KDE done in {time.time()-t0:.0f}s", flush=True)

# Fine sweep
print("\nFine sweep...", flush=True)
best_auc = 0; best_cfg = None; all_results = []
for wg_kde in [0.20, 0.25, 0.30, 0.35, 0.40]:
    for wg_rknn in [0.50, 0.55, 0.60, 0.65, 0.70]:
        wg_win = 1.0 - wg_kde - wg_rknn
        if wg_win < 0: continue
        blend = wg_kde * kde_win + wg_rknn * y_rknn_k5 + wg_win * y_win_k1
        for a in [0.90, 0.92, 0.95, 0.97, 1.00]:
            for b in [1.4, 1.5, 1.6, 1.7, 1.8]:
                pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
                auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
                all_results.append((auc, wg_kde, wg_rknn, wg_win, a, b))
                if auc > best_auc:
                    best_auc = auc; best_cfg = (wg_kde, wg_rknn, wg_win, a, b)

# Also try: no win_k1 (pure kde + rknn)
for wg_kde in [0.25, 0.30, 0.35, 0.40, 0.45]:
    wg_rknn = 1.0 - wg_kde
    blend = wg_kde * kde_win + wg_rknn * y_rknn_k5
    for a in [0.90, 0.95, 1.00]:
        for b in [1.4, 1.6, 1.8, 2.0]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            all_results.append((auc, wg_kde, wg_rknn, 0.0, a, b))
            if auc > best_auc:
                best_auc = auc; best_cfg = (wg_kde, wg_rknn, 0.0, a, b)

print(f"\nBest: {best_auc:.4f}  cfg=wg_kde={best_cfg[0]}, wg_rknn={best_cfg[1]}, wg_win={best_cfg[2]}, a={best_cfg[3]}, b={best_cfg[4]}", flush=True)

# Top 10
all_results.sort(reverse=True)
print("\nTop 10 configs:", flush=True)
for auc, wk, wr, ww, a, b in all_results[:10]:
    print(f"  AUC={auc:.4f} kde={wk:.2f} rknn={wr:.2f} win={ww:.2f} a={a} b={b}", flush=True)

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
rd['experiments'].append({'method': 'kde_win_rknn_fine', 'loo_auc': float(best_auc), 'full_auc': float(best_auc),
                           'config': {'wg_kde': best_cfg[0], 'wg_rknn': best_cfg[1], 'wg_win': best_cfg[2],
                                      'a': best_cfg[3], 'b': best_cfg[4], 'pca_n': 32, 'bw': 0.5}})
if best_auc > cur_best_json:
    rd['best'] = {'method': 'kde_win_rknn_fine', 'loo_auc': float(best_auc), 'full_auc': float(best_auc)}
    print(f"\n*** JSON BEST UPDATED: kde_win_rknn_fine = {best_auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json", flush=True)
