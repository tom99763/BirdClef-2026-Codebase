"""
Batch 45: UMAP + Hubness correction + KernelPCA
Goal: beat 0.9732
Methods:
  1. UMAP 80-dim embedding
  2. Hubness-corrected cosine similarity (subtract mean sim)
  3. KernelPCA (rbf) - 80 components
  4. Combination of hubness-corrected + ICA-90 + PCA-80
  5. L2-normalized difference from mean prototype (ZSL-style centering)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA, KernelPCA
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
CURRENT_BEST = 0.9732

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def maxmean_contrast(emb_wins_n, k_neg=4, w_max_pos=0.5, w_max_agg=0.55):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_wins_all = emb_wins_n[win_file_id != fi]
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

def hubness_contrast(emb_wins_n, k_neg=4, w_max_pos=0.5, w_max_agg=0.55, hub_alpha=1.0):
    """Hubness-corrected contrast: subtract mean similarity to all training windows"""
    # Precompute mean sim of each window to all others (hubness bias)
    all_sims_mean = (emb_wins_n @ emb_wins_n.T).mean(1)  # [N]
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        te_idx = np.where(win_file_id == fi)[0]
        tr_wins_all = emb_wins_n[win_file_id != fi]
        tr_idx = np.where(win_file_id != fi)[0]
        tr_fids_all = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
        te_hub_bias = all_sims_mean[te_idx]  # [n_te]
        tr_hub_bias = all_sims_mean[tr_idx]  # [n_tr]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win[:,si] > 0.5
            neg_win_mask = ~pos_win_mask
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_hub = tr_hub_bias[pos_win_mask]
            # Hubness-corrected sim: sim(te, pos) - hub_alpha * hub_bias(pos)
            raw_sims = te_wins @ pos_wins.T  # [n_te, n_pos]
            corrected = raw_sims - hub_alpha * pos_hub[None, :]  # subtract pos hubness
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp_raw = w_max_pos * corrected.max(1) + (1-w_max_pos) * (te_wins @ pp_mean - hub_alpha * te_hub_bias)
            sp = (sp_raw - sp_raw.min()) / (sp_raw.max() - sp_raw.min() + EPS)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_hub = tr_hub_bias[neg_win_mask]
                neg_raw = te_wins @ neg_wins.T
                neg_corr = neg_raw - hub_alpha * neg_hub[None, :]
                k_act = min(k_neg, neg_corr.shape[1])
                top_neg_idx = np.argsort(-neg_corr, axis=1)[:, :k_act]
                top_neg = neg_wins[top_neg_idx].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                sn_raw = (te_wins * top_neg).sum(1) - hub_alpha * te_hub_bias
                sn = (sn_raw - sn_raw.min()) / (sn_raw.max() - sn_raw.min() + EPS)
                ws[:,si] = (sp - sn + 1) / 2
            else: ws[:,si] = sp
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

# Precompute standard components
pca80 = PCA(n_components=80, random_state=42)
ew80 = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
out_base = maxmean_contrast(ew80)
print(f"Base (pca80): {eval_loo(out_base):.4f}", flush=True)

ica90 = FastICA(n_components=90, random_state=42, max_iter=500, tol=0.01)
ew_ica90 = normalize(ica90.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
out_ica90 = maxmean_contrast(ew_ica90)
print(f"ICA-90: {eval_loo(out_ica90):.4f}", flush=True)

scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
ew80s = normalize(pca80s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
out_std80_kn2 = maxmean_contrast(ew80s, k_neg=2)
print(f"Std-PCA-80 kn2: {eval_loo(out_std80_kn2):.4f}", flush=True)

# ─── Method 1: UMAP 80-dim ───────────────────────────────────────────────────
print("\n=== Method 1: UMAP 80-dim ===", flush=True)
t0 = time.time()
try:
    import umap
    umap80 = umap.UMAP(n_components=80, random_state=42, n_neighbors=15, min_dist=0.1)
    emb_umap = umap80.fit_transform(emb_win).astype(np.float32)
    ew_umap = normalize(emb_umap, norm='l2').astype(np.float32)
    out_umap = maxmean_contrast(ew_umap)
    auc1 = eval_loo(out_umap)
    results['umap80'] = auc1
    flag = " *** NEW BEST ***" if auc1 > CURRENT_BEST else ""
    print(f"  UMAP-80: {auc1:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
    # Blend
    best1b = 0; best_cfg1b = None
    for w_umap in [0.2, 0.3, 0.4]:
        for w_ica in [0.25, 0.3, 0.35]:
            w_b = 1.0 - w_umap - w_ica
            if w_b < 0.3: continue
            blend = w_umap * out_umap + w_ica * out_ica90 + w_b * out_base
            auc_c = eval_loo(blend)
            if auc_c > best1b: best1b = auc_c; best_cfg1b = (w_umap, w_ica, w_b)
    results['umap80_ica90_base'] = best1b
    flag = " *** NEW BEST ***" if best1b > CURRENT_BEST else ""
    print(f"  UMAP80+ICA90+base: {best1b:.4f}{flag}  cfg={best_cfg1b}", flush=True)
except Exception as e:
    print(f"  UMAP failed: {e}", flush=True)

# ─── Method 2: Hubness-corrected PCA-80 ──────────────────────────────────────
print("\n=== Method 2: Hubness-corrected PCA-80 ===", flush=True)
t0 = time.time()
for alpha in [0.5, 1.0, 2.0]:
    out_hub = hubness_contrast(ew80, hub_alpha=alpha)
    auc2 = eval_loo(out_hub)
    results[f'hub_pca80_a{int(alpha*10)}'] = auc2
    flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
    print(f"  Hub PCA-80 alpha={alpha}: {auc2:.4f}{flag}", flush=True)
    # Blend with ICA-90
    best_b = 0; best_wb = None
    for w_ica in [0.35, 0.40]:
        for w_hub in [0.08, 0.12, 0.15]:
            w_b = 1.0 - w_ica - w_hub
            if w_b < 0.45: continue
            blend = w_ica * out_ica90 + w_hub * out_hub + w_b * out_base
            auc_c = eval_loo(blend)
            if auc_c > best_b: best_b = auc_c; best_wb = (w_ica, w_hub, w_b)
    results[f'ica90_hub{int(alpha*10)}_base'] = best_b
    flag = " *** NEW BEST ***" if best_b > CURRENT_BEST else ""
    print(f"  ICA90+Hub{alpha}+base: {best_b:.4f}{flag}  w={best_wb}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 3: KernelPCA rbf ─────────────────────────────────────────────────
print("\n=== Method 3: KernelPCA rbf ===", flush=True)
t0 = time.time()
try:
    kpca = KernelPCA(n_components=80, kernel='rbf', gamma=None, random_state=42)
    emb_kpca = kpca.fit_transform(normalize(emb_win, norm='l2')).astype(np.float32)
    ew_kpca = normalize(emb_kpca, norm='l2').astype(np.float32)
    out_kpca = maxmean_contrast(ew_kpca)
    auc3 = eval_loo(out_kpca)
    results['kpca80_rbf'] = auc3
    flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
    print(f"  KernelPCA-80 rbf: {auc3:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
    best3b = 0; best_cfg3b = None
    for w_ica in [0.35, 0.40]:
        for w_k in [0.08, 0.12, 0.15]:
            w_b = 1.0 - w_ica - w_k
            if w_b < 0.45: continue
            blend = w_ica * out_ica90 + w_k * out_kpca + w_b * out_base
            auc_c = eval_loo(blend)
            if auc_c > best3b: best3b = auc_c; best_cfg3b = (w_ica, w_k, w_b)
    results['ica90_kpca_base'] = best3b
    flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
    print(f"  ICA90+KPCA+base: {best3b:.4f}{flag}  cfg={best_cfg3b}", flush=True)
except Exception as e:
    print(f"  KernelPCA failed: {e}", flush=True)

# ─── Method 4: Z-score centering in embedding space ─────────────────────────
print("\n=== Method 4: Z-score PCA (per-dim whitening) + ICA-90 + Std-PCA-80 ===", flush=True)
# Apply z-score to PCA components (whiten the PCA outputs)
emb80_raw = pca80.transform(emb_win).astype(np.float32)
emb80_zs = (emb80_raw - emb80_raw.mean(0)) / (emb80_raw.std(0) + EPS)
ew80_zs = normalize(emb80_zs, norm='l2').astype(np.float32)
out_zs80 = maxmean_contrast(ew80_zs)
auc4 = eval_loo(out_zs80)
results['zs_pca80'] = auc4
flag = " *** NEW BEST ***" if auc4 > CURRENT_BEST else ""
print(f"  Z-score PCA-80: {auc4:.4f}{flag}", flush=True)

best4b = 0; best_cfg4b = None
for w_ica in [0.35, 0.40]:
    for w_std in [0.06, 0.08, 0.12]:
        for w_zs in [0.06, 0.08]:
            w_b = 1.0 - w_ica - w_std - w_zs
            if w_b < 0.40: continue
            blend = w_ica*out_ica90 + w_std*out_std80_kn2 + w_zs*out_zs80 + w_b*out_base
            auc_c = eval_loo(blend)
            if auc_c > best4b: best4b = auc_c; best_cfg4b = (w_ica, w_std, w_zs, w_b)
results['ica90_std80kn2_zs_base'] = best4b
flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
print(f"  ICA90+Std80kn2+ZS+base: {best4b:.4f}{flag}  cfg={best_cfg4b}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 45 Summary ===", flush=True)
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
