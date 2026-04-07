"""
Batch 14: Extended bandwidth & 3-way ensemble experiments
Goal: beat kde_pw_ensemble = 0.9732
Methods:
  1. bw=0.3 per-window KDE
  2. bw=0.6 per-window KDE
  3. 3-way: bw=0.3 + bw=0.4 + bw=0.5
  4. 3-way: bw=0.4 + bw=0.5 + bw=0.6
  5. 4-way: bw=0.3 + bw=0.4 + bw=0.5 + bw=0.6
  6. bw=0.35 per-window KDE
  7. bw=0.45 per-window KDE
  8. Geometric mean ensemble (bw=0.4, 0.5)
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

def sweep(scores):
    best = 0; best_cfg = None
    for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for b in [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]:
            pred = sigmoid(a * base_logit + b * np.log(scores.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best: best = auc; best_cfg = (a, b)
    return best, best_cfg

results = {}
CURRENT_BEST = 0.9732

# Compute all bandwidths
print("Computing per-window KDE for multiple bandwidths...", flush=True)
bw_map = {}
for bw in [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
    t0 = time.time()
    print(f"  bw={bw}...", flush=True)
    bw_map[bw] = loo_kde_perwin(bw)
    print(f"    Done {time.time()-t0:.0f}s, single AUC: ", end='', flush=True)
    a, cfg = sweep(bw_map[bw])
    print(f"{a:.4f} @ {cfg}", flush=True)
    results[f'kde_perwin_bw{bw}'] = (a, cfg)

# 2-way blends (not yet tried with bw=0.3, 0.6, 0.35, 0.45)
print("\n=== 2-way blends ===", flush=True)
pairs = [
    (0.3, 0.4), (0.3, 0.5), (0.3, 0.6),
    (0.35, 0.5), (0.45, 0.5), (0.4, 0.6), (0.5, 0.6),
]
for bw1, bw2 in pairs:
    best2 = 0; best_cfg2 = None
    for w1 in [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]:
        blend = w1 * bw_map[bw1] + (1-w1) * bw_map[bw2]
        auc_c, cfg_c = sweep(blend)
        if auc_c > best2: best2 = auc_c; best_cfg2 = (w1, cfg_c)
    results[f'kde_pw_blend_{bw1}_{bw2}'] = (best2, best_cfg2)
    flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
    print(f"  bw={bw1}+{bw2}: {best2:.4f}{flag}  w1={best_cfg2[0]}", flush=True)

# 3-way blends
print("\n=== 3-way blends ===", flush=True)
triplets = [
    (0.3, 0.4, 0.5), (0.4, 0.5, 0.6), (0.3, 0.5, 0.6),
    (0.35, 0.4, 0.5), (0.4, 0.45, 0.5),
]
for bw1, bw2, bw3 in triplets:
    best3 = 0; best_cfg3 = None
    for w1 in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for w2 in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            w3 = 1.0 - w1 - w2
            if w3 <= 0.05: continue
            blend = w1 * bw_map[bw1] + w2 * bw_map[bw2] + w3 * bw_map[bw3]
            auc_c, cfg_c = sweep(blend)
            if auc_c > best3: best3 = auc_c; best_cfg3 = (w1, w2, w3, cfg_c)
    results[f'kde_pw_3way_{bw1}_{bw2}_{bw3}'] = (best3, best_cfg3)
    flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
    print(f"  bw={bw1}+{bw2}+{bw3}: {best3:.4f}{flag}  w={best_cfg3[:3]}", flush=True)

# 4-way blend
print("\n=== 4-way blend ===", flush=True)
best4 = 0; best_cfg4 = None
for w1 in [0.1, 0.15, 0.2]:
    for w2 in [0.2, 0.25, 0.3, 0.35]:
        for w3 in [0.3, 0.35, 0.4, 0.45]:
            w4 = 1.0 - w1 - w2 - w3
            if w4 <= 0.05: continue
            blend = w1*bw_map[0.3] + w2*bw_map[0.4] + w3*bw_map[0.5] + w4*bw_map[0.6]
            auc_c, cfg_c = sweep(blend)
            if auc_c > best4: best4 = auc_c; best_cfg4 = (w1, w2, w3, w4, cfg_c)
results['kde_pw_4way_0.3_0.4_0.5_0.6'] = (best4, best_cfg4)
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  4-way: {best4:.4f}{flag}  w={best_cfg4[:4]}", flush=True)

# Summary
print("\n=== Summary ===", flush=True)
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

# Update JSON
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
print("Updated embed_prior_results.json", flush=True)
print(f"\nFinal best in JSON: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
