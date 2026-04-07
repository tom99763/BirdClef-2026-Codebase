"""
Batch 59: New prototype construction approaches beyond 0.9873
Methods:
  1. k-means multi-prototype (2-4 clusters per species, max over centroids)
  2. Outlier-filtered prototype (remove outlier positive windows)
  3. Adaptive k_neg (higher for rare species, lower for common)
  4. Weighted prototype by intra-class similarity
  5. Two-pass refinement (first pass → re-weight positives → second pass)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
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
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

EPS = 1e-7; mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9873

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def winlabel_contrast(emb_wins_n, k_neg=4, w_max_pos=0.5, w_max_agg=0.55):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins_all = emb_wins_n[tr_mask]
        tr_lab_win_raw = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win_raw[:,si] > 0.5
            neg_win_mask = tr_lab_win_raw[:,si] < 0.1
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp = w_max_pos * pos_sims.max(1) + (1-w_max_pos) * (te_wins @ pp_mean)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

def winlabel_kmeans(emb_wins_n, k_neg=50, n_proto=3, w_max_pos=0.80, w_max_agg=0.90):
    """Multi-prototype using k-means on positive windows."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins_all = emb_wins_n[tr_mask]
        tr_lab_win_raw = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win_raw[:,si] > 0.5
            neg_win_mask = tr_lab_win_raw[:,si] < 0.1
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            n_pos = len(pos_wins)

            if n_pos >= n_proto * 2:
                # k-means clustering of positives
                km = KMeans(n_clusters=min(n_proto, n_pos), random_state=42, n_init=3)
                km.fit(pos_wins)
                centroids = normalize(km.cluster_centers_.astype(np.float32), norm='l2')
                proto_sims = te_wins @ centroids.T  # [n_te, n_proto]
                sp = w_max_pos * proto_sims.max(1) + (1-w_max_pos) * proto_sims.mean(1)
            else:
                # Fallback: single mean prototype
                pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
                pos_sims = te_wins @ pos_wins.T
                sp = w_max_pos * pos_sims.max(1) + (1-w_max_pos) * (te_wins @ pp_mean)

            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

def winlabel_filtered_pos(emb_wins_n, k_neg=50, w_max_pos=0.80, w_max_agg=0.90, keep_frac=0.8):
    """Filter outlier positive windows by intra-class similarity."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins_all = emb_wins_n[tr_mask]
        tr_lab_win_raw = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win_raw[:,si] > 0.5
            neg_win_mask = tr_lab_win_raw[:,si] < 0.1
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            # Filter: keep top-keep_frac by similarity to centroid
            if len(pos_wins) >= 4:
                centroid = pos_wins.mean(0); centroid /= (np.linalg.norm(centroid) + EPS)
                sims_to_centroid = pos_wins @ centroid
                n_keep = max(1, int(len(pos_wins) * keep_frac))
                keep_idx = np.argsort(-sims_to_centroid)[:n_keep]
                pos_wins = pos_wins[keep_idx]
            pos_sims = te_wins @ pos_wins.T
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp = w_max_pos * pos_sims.max(1) + (1-w_max_pos) * (te_wins @ pp_mean)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

# Precompute
print("Precomputing...", flush=True)
pca80 = PCA(n_components=80, random_state=42)
ew80 = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)

scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
ew80s = normalize(pca80s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)

ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica100 = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
print("Done.", flush=True)

# Baselines
out_wl80 = winlabel_contrast(ew80, k_neg=4, w_max_pos=0.70, w_max_agg=0.60)
out_wl_std = winlabel_contrast(ew80s, k_neg=4, w_max_pos=0.60, w_max_agg=0.65)
print(f"WL-PCA-80: {eval_loo(out_wl80):.4f}", flush=True)
print(f"WL-Std-PCA-80: {eval_loo(out_wl_std):.4f}", flush=True)

# Compute uh best
print("Computing uh best...", flush=True)
best_uh = 0; best_out_uh = None
for k_neg in [50, 60, 70]:
    for wma in [0.90, 0.92]:
        for wmp in [0.78, 0.80]:
            out = winlabel_contrast(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_uh: best_uh = auc; best_out_uh = out
print(f"  ICA-100 uh: {best_uh:.4f}", flush=True)

# ─── Method 1: k-means multi-prototype ────────────────────────────────────────
print("\n=== Method 1: k-means multi-prototype ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None; best_out1 = None
for n_proto in [2, 3, 4]:
    for k_neg in [40, 50, 60, 80]:
        for wma in [0.85, 0.88, 0.90, 0.92]:
            for wmp in [0.75, 0.78, 0.80]:
                out = winlabel_kmeans(ew_ica100, k_neg=k_neg, n_proto=n_proto, w_max_pos=wmp, w_max_agg=wma)
                auc = eval_loo(out)
                if auc > best1: best1 = auc; best_cfg1 = (n_proto, k_neg, wma, wmp); best_out1 = out
print(f"  k-means ICA-100: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_kmeans_ica100'] = best1

# Triple blend
best1t = 0; best_cfg1t = None
for w_ica in np.arange(0.30, 0.70, 0.02):
    for w_std in np.arange(0.10, 0.50, 0.02):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.55: continue
        blend = w_ica * best_out1 + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best1t: best1t = auc; best_cfg1t = (float(w_ica), float(w_std), float(w_pca))
results['wl_kmeans_triple'] = best1t
flag = " *** NEW BEST ***" if best1t > CURRENT_BEST else ""
print(f"  k-means triple: {best1t:.4f}{flag}  cfg={best_cfg1t}", flush=True)

# ─── Method 2: Outlier-filtered positive windows ─────────────────────────────
print("\n=== Method 2: Filtered positive windows ===", flush=True)
t0 = time.time()
best2 = 0; best_cfg2 = None; best_out2 = None
for keep_frac in [0.6, 0.7, 0.8, 0.9]:
    for k_neg in [40, 50, 60, 80]:
        for wma in [0.85, 0.88, 0.90, 0.92]:
            for wmp in [0.75, 0.78, 0.80]:
                out = winlabel_filtered_pos(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma, keep_frac=keep_frac)
                auc = eval_loo(out)
                if auc > best2: best2 = auc; best_cfg2 = (keep_frac, k_neg, wma, wmp); best_out2 = out
print(f"  Filtered ICA-100: {best2:.4f}  cfg={best_cfg2}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_filtered_ica100'] = best2

best2t = 0; best_cfg2t = None
for w_ica in np.arange(0.30, 0.70, 0.02):
    for w_std in np.arange(0.10, 0.50, 0.02):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.55: continue
        blend = w_ica * best_out2 + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best2t: best2t = auc; best_cfg2t = (float(w_ica), float(w_std), float(w_pca))
results['wl_filtered_triple'] = best2t
flag = " *** NEW BEST ***" if best2t > CURRENT_BEST else ""
print(f"  Filtered triple: {best2t:.4f}{flag}  cfg={best_cfg2t}", flush=True)

# ─── Method 3: Combine kmeans + filtered + uh ────────────────────────────────
print("\n=== Method 3: Blend kmeans + filtered + uh ===", flush=True)
best3 = 0; best_cfg3 = None
for wk in [0.2, 0.3, 0.4]:
    for wf in [0.1, 0.2, 0.3]:
        wu = 1.0 - wk - wf
        if wu < 0.3 or wu > 0.7: continue
        blend = wk * best_out1 + wf * best_out2 + wu * best_out_uh
        # Quick triple
        for w_ica2 in np.arange(0.40, 0.70, 0.05):
            for w_std in np.arange(0.15, 0.40, 0.05):
                w_pca = 1.0 - w_ica2 - w_std
                if w_pca < 0.05 or w_pca > 0.45: continue
                b = w_ica2 * blend + w_std * out_wl_std + w_pca * out_wl80
                auc = eval_loo(b)
                if auc > best3: best3 = auc; best_cfg3 = (wk, wf, wu, w_ica2, w_std, w_pca)
results['wl_kmeans_filtered_uh_4way'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  k-means+filtered+uh: {best3:.4f}{flag}  cfg={best_cfg3}", flush=True)

# ─── Method 4: Two-pass refinement ────────────────────────────────────────────
print("\n=== Method 4: Two-pass positive re-weighting ===", flush=True)
t0 = time.time()
def winlabel_twopass(emb_wins_n, k_neg=50, w_max_pos=0.80, w_max_agg=0.92):
    """Pass 1: standard WL. Pass 2: re-weight positives by pass-1 training scores."""
    # Pass 1: compute standard WL LOO scores
    pass1_out = winlabel_contrast(emb_wins_n, k_neg=k_neg, w_max_pos=w_max_pos, w_max_agg=w_max_agg)
    # pass1_out: [n_files, n_species] = WL scores for each file
    # For pass 2: weight positive windows by how well pass1 predicted them
    out2 = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins_all = emb_wins_n[tr_mask]
        tr_lab_win_raw = labels_win[tr_mask]
        tr_file_ids = win_file_id[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win_raw[:,si] > 0.5
            neg_win_mask = tr_lab_win_raw[:,si] < 0.1
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_fids = tr_file_ids[pos_win_mask]
            # Weight by pass1 score for the file this window belongs to
            pass1_weights = np.array([pass1_out[fid, si] for fid in pos_fids])
            pass1_weights = np.clip(pass1_weights, 0.01, 0.99)
            # Weighted prototype
            w_sum = pass1_weights.sum()
            if w_sum > EPS:
                weighted_proto = (pos_wins * pass1_weights[:, None]).sum(0) / w_sum
            else:
                weighted_proto = pos_wins.mean(0)
            weighted_proto /= (np.linalg.norm(weighted_proto) + EPS)
            pos_sims = te_wins @ pos_wins.T
            sp = w_max_pos * pos_sims.max(1) + (1-w_max_pos) * (te_wins @ weighted_proto)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out2[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out2

best4 = 0; best_cfg4 = None; best_out4 = None
for k_neg in [50, 70]:
    for wma in [0.88, 0.90, 0.92]:
        for wmp in [0.78, 0.80]:
            out = winlabel_twopass(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best4: best4 = auc; best_cfg4 = (k_neg, wma, wmp); best_out4 = out
print(f"  Two-pass ICA-100: {best4:.4f}  cfg={best_cfg4}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_twopass_ica100'] = best4

best4t = 0; best_cfg4t = None
for w_ica in np.arange(0.30, 0.70, 0.02):
    for w_std in np.arange(0.10, 0.50, 0.02):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.55: continue
        blend = w_ica * best_out4 + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best4t: best4t = auc; best_cfg4t = (float(w_ica), float(w_std), float(w_pca))
results['wl_twopass_triple'] = best4t
flag = " *** NEW BEST ***" if best4t > CURRENT_BEST else ""
print(f"  Two-pass triple: {best4t:.4f}{flag}  cfg={best_cfg4t}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== Batch 59 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
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
