"""
Batch 44: Aggregation parameter tuning + new decompositions
Goal: beat 0.9732
Methods:
  1. Per-component w_max_agg/w_max_pos sweep for each base component
  2. Geometric mean aggregation (sqrt(max*mean))
  3. Sparse PCA / TruncatedSVD on correlation matrix
  4. ICA-90 with different w_max_pos (0.4, 0.5, 0.6, 0.7)
  5. Std-PCA-80 with different w_max_pos / w_max_agg
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
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

def geomean_contrast(emb_wins_n, k_neg=4, w_max_pos=0.5):
    """Geometric mean aggregation: sqrt(max * mean)"""
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
        mx = ws.max(0); mn = ws.mean(0)
        out[fi] = np.sqrt(np.clip(mx * mn, 0, None))
    return out

# Precompute
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

# ─── Method 1: ICA-90 w_max_pos sweep ─────────────────────────────────────────
print("\n=== Method 1: ICA-90 w_max_pos sweep ===", flush=True)
best1 = 0; best_cfg1 = None; best_out1 = None
for wmp in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    out_ica_wmp = maxmean_contrast(ew_ica90, w_max_pos=wmp)
    auc_solo = eval_loo(out_ica_wmp)
    # Blend with Std80kn2 + base
    for w_ica in [0.35, 0.40, 0.45]:
        for w_std in [0.08, 0.12, 0.15]:
            w_b = 1.0 - w_ica - w_std
            if w_b < 0.40: continue
            blend = w_ica * out_ica_wmp + w_std * out_std80_kn2 + w_b * out_base
            auc = eval_loo(blend)
            if auc > best1: best1 = auc; best_cfg1 = (wmp, w_ica, w_std, w_b); best_out1 = out_ica_wmp
    print(f"  wmp={wmp:.1f}: solo={auc_solo:.4f}", flush=True)
results['ica90_wmp_std80kn2_base'] = best1
flag = " *** NEW BEST ***" if best1 > CURRENT_BEST else ""
print(f"  Best: {best1:.4f}{flag}  cfg={best_cfg1}", flush=True)

# ─── Method 2: Std-PCA-80 w_max_pos sweep ────────────────────────────────────
print("\n=== Method 2: Std-PCA-80 w_max_pos sweep (k_neg=2) ===", flush=True)
best2 = 0; best_cfg2 = None
for wmp in [0.3, 0.4, 0.5, 0.6, 0.7]:
    out_std_wmp = maxmean_contrast(ew80s, k_neg=2, w_max_pos=wmp)
    for w_ica in [0.35, 0.40, 0.45]:
        for w_std in [0.08, 0.12, 0.15]:
            w_b = 1.0 - w_ica - w_std
            if w_b < 0.40: continue
            blend = w_ica * out_ica90 + w_std * out_std_wmp + w_b * out_base
            auc = eval_loo(blend)
            if auc > best2: best2 = auc; best_cfg2 = (wmp, w_ica, w_std, w_b)
results['ica90_std80kn2_wmp_base'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  Best: {best2:.4f}{flag}  cfg={best_cfg2}", flush=True)

# ─── Method 3: Std-PCA-80 w_max_agg sweep ────────────────────────────────────
print("\n=== Method 3: Std-PCA-80 w_max_agg sweep (k_neg=2) ===", flush=True)
best3 = 0; best_cfg3 = None
for wma in [0.4, 0.5, 0.6, 0.65, 0.7, 0.8]:
    out_std_wma = maxmean_contrast(ew80s, k_neg=2, w_max_agg=wma)
    for w_ica in [0.35, 0.40]:
        for w_std in [0.08, 0.12, 0.15]:
            w_b = 1.0 - w_ica - w_std
            if w_b < 0.45: continue
            blend = w_ica * out_ica90 + w_std * out_std_wma + w_b * out_base
            auc = eval_loo(blend)
            if auc > best3: best3 = auc; best_cfg3 = (wma, w_ica, w_std, w_b)
results['ica90_std80kn2_wma_base'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  Best: {best3:.4f}{flag}  cfg={best_cfg3}", flush=True)

# ─── Method 4: Geometric mean aggregation ────────────────────────────────────
print("\n=== Method 4: Geometric mean aggregation ===", flush=True)
t0 = time.time()
out_geo80 = geomean_contrast(ew80)
out_geo_ica90 = geomean_contrast(ew_ica90)
out_geo_std80 = geomean_contrast(ew80s)
print(f"  Geo PCA-80: {eval_loo(out_geo80):.4f}", flush=True)
print(f"  Geo ICA-90: {eval_loo(out_geo_ica90):.4f}", flush=True)
print(f"  Geo Std-PCA-80: {eval_loo(out_geo_std80):.4f}", flush=True)
# Blend geo with regular
best4 = 0; best_cfg4 = None
for w_geo in [0.2, 0.3, 0.4]:
    blend = w_geo * out_geo80 + (1-w_geo) * out_base
    auc = eval_loo(blend)
    if auc > best4: best4 = auc; best_cfg4 = ('geo80_base', w_geo)
for w_ica in [0.35, 0.40]:
    for w_geo in [0.08, 0.12]:
        w_b = 1.0 - w_ica - w_geo
        if w_b < 0.45: continue
        blend = w_ica * out_ica90 + w_geo * out_geo80 + w_b * out_base
        auc = eval_loo(blend)
        if auc > best4: best4 = auc; best_cfg4 = ('ica90_geo80_base', w_ica, w_geo, w_b)
results['best_geo_blend'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  Best geo blend: {best4:.4f}{flag}  cfg={best_cfg4}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 5: TruncatedSVD on correlation matrix ────────────────────────────
print("\n=== Method 5: TruncatedSVD on L2-normalized embeddings ===", flush=True)
t0 = time.time()
# Normalize rows first (unit sphere), then SVD
emb_l2 = emb_win / (np.linalg.norm(emb_win, axis=1, keepdims=True) + EPS)
svd = TruncatedSVD(n_components=80, random_state=42)
emb_svd = svd.fit_transform(emb_l2).astype(np.float32)
ew_svd = normalize(emb_svd, norm='l2').astype(np.float32)
out_svd = maxmean_contrast(ew_svd)
auc5 = eval_loo(out_svd)
results['svd80_l2input'] = auc5
flag = " *** NEW BEST ***" if auc5 > CURRENT_BEST else ""
print(f"  SVD-80 (L2-normed input): {auc5:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
# Blend
best5b = 0; best_cfg5b = None
for w_ica in [0.35, 0.40]:
    for w_svd in [0.08, 0.12, 0.15]:
        w_b = 1.0 - w_ica - w_svd
        if w_b < 0.45: continue
        blend = w_ica * out_ica90 + w_svd * out_svd + w_b * out_base
        auc_c = eval_loo(blend)
        if auc_c > best5b: best5b = auc_c; best_cfg5b = (w_ica, w_svd, w_b)
# Also with Std-PCA-80 kn2
for w_ica in [0.35, 0.40]:
    for w_std in [0.08, 0.12]:
        for w_svd in [0.06, 0.08]:
            w_b = 1.0 - w_ica - w_std - w_svd
            if w_b < 0.42: continue
            blend = w_ica*out_ica90 + w_std*out_std80_kn2 + w_svd*out_svd + w_b*out_base
            auc_c = eval_loo(blend)
            if auc_c > best5b: best5b = auc_c; best_cfg5b = (w_ica, w_std, w_svd, w_b)
results['ica90_std80kn2_svd_base'] = best5b
flag = " *** NEW BEST ***" if best5b > CURRENT_BEST else ""
print(f"  ICA90+Std80kn2+SVD+base: {best5b:.4f}{flag}  cfg={best_cfg5b}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 44 Summary ===", flush=True)
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
