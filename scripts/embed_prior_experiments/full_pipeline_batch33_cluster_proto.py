"""
Batch 33: Clustering-based positive prototype methods
Goal: beat pca80_wm5_kn5 = 0.9652
Methods:
  1. K-means cluster pos prototype: cluster pos windows, use nearest centroid
  2. Geometric median positive prototype (robust to outliers)
  3. Max-of-clusters: score = max over cluster centroids
  4. Multi-prototype: use 2-3 cluster centroids for pos, avg neg
  5. Combine: PCA-80 + clustering methods
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[file_start[fi]:file_end[fi]] = fi

file_labels = np.zeros((n_files, n_species), np.float32)
file_embs   = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_embs[fi]   = emb_win[s:e].mean(0)

# PCA-80 space (best so far)
pca80 = PCA(n_components=80, random_state=42)
emb_win_pca = pca80.fit_transform(emb_win).astype(np.float32)
emb_win_pca_norm = normalize(emb_win_pca, norm='l2').astype(np.float32)
file_embs_pca = np.zeros((n_files, 80), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs_pca[fi] = emb_win_pca[s:e].mean(0)
file_embs_pca_norm = normalize(file_embs_pca, norm='l2').astype(np.float32)
EPS = 1e-7; mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9652

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# Base: PCA-80 max_pos (reference)
def pca80_base():
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_lab = file_labels[tr_idx]
        te_wins = emb_win_pca_norm[win_file_id == fi]
        tr_wins_all = emb_win_pca_norm[win_file_id != fi]
        tr_fids_all = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win[:,si] > 0.5
            neg_win_mask = ~pos_win_mask
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp = 0.5 * pos_sims.max(1) + 0.5 * (te_wins @ pp_mean)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(5, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    return out

out_base = pca80_base()
print(f"PCA-80 base: {eval_loo(out_base):.4f}", flush=True)

# ─── Method 1: Multi-centroid positive prototype ──────────────────────────────
print("\n=== Method 1: K-means multi-centroid pos prototype ===", flush=True)
t0 = time.time()
best1 = 0; best_k1 = None; best_out1 = None
for n_clusters in [2, 3]:
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_lab = file_labels[tr_idx]
        te_wins = emb_win_pca_norm[win_file_id == fi]
        tr_wins_all = emb_win_pca_norm[win_file_id != fi]
        tr_fids_all = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win[:,si] > 0.5
            neg_win_mask = ~pos_win_mask
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            # Multi-centroid: cluster positive windows
            if len(pos_wins) >= n_clusters:
                km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42, max_iter=50)
                km.fit(pos_wins)
                centroids = normalize(km.cluster_centers_, norm='l2').astype(np.float32)
                # Score: max over centroid similarities
                centroid_sims = te_wins @ centroids.T  # [n_te, n_clusters]
                sp = centroid_sims.max(1)
            else:
                pp = pos_wins.mean(0); pp /= (np.linalg.norm(pp) + EPS)
                sp = te_wins @ pp
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(5, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    auc = eval_loo(out)
    results[f'kmeans_{n_clusters}centroid'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  k={n_clusters}: {auc:.4f}{flag}", flush=True)
    if auc > best1: best1 = auc; best_k1 = n_clusters; best_out1 = out
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 2: Geometric median (Weiszfeld) positive prototype ───────────────
print("\n=== Method 2: Geometric median positive prototype ===", flush=True)
t0 = time.time()

def geom_median(X, max_iter=20):
    """Approximate geometric median by Weiszfeld algorithm."""
    y = X.mean(0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - y, axis=1) + 1e-8
        w = 1.0 / d
        y_new = (w[:, None] * X).sum(0) / w.sum()
        if np.linalg.norm(y_new - y) < 1e-6: break
        y = y_new
    return y

out_gmed = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_pca_norm[win_file_id == fi]
    tr_wins_all = emb_win_pca_norm[win_file_id != fi]
    tr_fids_all = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos_win_mask = tr_lab_win[:,si] > 0.5
        neg_win_mask = ~pos_win_mask
        if not pos_win_mask.any(): ws[:,si]=0.5; continue
        pos_wins = tr_wins_all[pos_win_mask]
        pos_sims = te_wins @ pos_wins.T
        # Geometric median for positive prototype
        pp_gm = geom_median(pos_wins)
        pp_gm /= (np.linalg.norm(pp_gm) + EPS)
        sp_gm = te_wins @ pp_gm
        # Blend: 0.5 * max_pos + 0.5 * geom_median
        sp = 0.5 * pos_sims.max(1) + 0.5 * sp_gm
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(5, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
        else: ws[:,si] = (sp+1)/2
    out_gmed[fi] = ws.mean(0)
auc2 = eval_loo(out_gmed)
results['geom_median_pos'] = auc2
flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
print(f"  geom_median_pos: {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 3: Trimmed mean positive prototype ────────────────────────────────
print("\n=== Method 3: Trimmed mean positive (remove outlier windows) ===", flush=True)
t0 = time.time()
out_trimmed = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_pca_norm[win_file_id == fi]
    tr_wins_all = emb_win_pca_norm[win_file_id != fi]
    tr_fids_all = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos_win_mask = tr_lab_win[:,si] > 0.5
        neg_win_mask = ~pos_win_mask
        if not pos_win_mask.any(): ws[:,si]=0.5; continue
        pos_wins = tr_wins_all[pos_win_mask]
        pos_sims_te = te_wins @ pos_wins.T
        # Trimmed mean: remove bottom 20% of positive windows by avg similarity to test
        if len(pos_wins) >= 5:
            avg_sim_to_test = pos_sims_te.mean(0)  # [n_pos]
            threshold = np.percentile(avg_sim_to_test, 20)
            keep_mask = avg_sim_to_test >= threshold
            if keep_mask.sum() >= 2:
                pos_wins_trimmed = pos_wins[keep_mask]
            else:
                pos_wins_trimmed = pos_wins
        else:
            pos_wins_trimmed = pos_wins
        pos_sims2 = te_wins @ pos_wins_trimmed.T
        pp_trimmed = pos_wins_trimmed.mean(0); pp_trimmed /= (np.linalg.norm(pp_trimmed) + EPS)
        sp = 0.5 * pos_sims2.max(1) + 0.5 * (te_wins @ pp_trimmed)
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(5, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
        else: ws[:,si] = (sp+1)/2
    out_trimmed[fi] = ws.mean(0)
auc3 = eval_loo(out_trimmed)
results['trimmed_mean_pos'] = auc3
flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
print(f"  trimmed_mean_pos: {auc3:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: Blend cluster + base ──────────────────────────────────────────
print("\n=== Method 4: Blend cluster methods + pca80 base ===", flush=True)
for name, out in [('kmeans', best_out1), ('geom_med', out_gmed), ('trimmed', out_trimmed)]:
    if out is None: continue
    best_blend = 0; best_wb = None
    for w_new in [0.2, 0.3, 0.4, 0.5]:
        blend = w_new * out + (1-w_new) * out_base
        auc_c = eval_loo(blend)
        if auc_c > best_blend: best_blend = auc_c; best_wb = w_new
    results[f'{name}_base_blend'] = best_blend
    flag = " *** NEW BEST ***" if best_blend > CURRENT_BEST else ""
    print(f"  {name}+base: {best_blend:.4f}{flag}  w={best_wb}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 33 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
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
print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
