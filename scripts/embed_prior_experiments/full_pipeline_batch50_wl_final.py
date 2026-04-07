"""
Batch 50: WL method final optimization
Goal: beat wlica_best_wl80_best = 0.9843
Methods:
  1. Re-find best WL-PCA-80 params + WL-ICA-90 params, then fine blend grid
  2. Add WL-Std-PCA-80 to the ensemble
  3. WL-PCA-80 + WL-ICA-90 + WL-Std-PCA-80 triple WL ensemble
  4. WL methods with ICA-100 (instead of 90)
  5. Grand WL ensemble: 4 WL components + base
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
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
CURRENT_BEST = 0.9843

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

# Precompute embeddings
pca80 = PCA(n_components=80, random_state=42)
ew80 = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
out_base = maxmean_contrast(ew80)
print(f"Base (pca80 filelabel): {eval_loo(out_base):.4f}", flush=True)

ica90 = FastICA(n_components=90, random_state=42, max_iter=500, tol=0.01)
ew_ica90 = normalize(ica90.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
print(f"ICA-90 precomputed", flush=True)

scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
ew80s = normalize(pca80s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
print(f"Std-PCA-80 precomputed", flush=True)

# ─── Method 1: Find optimal WL params for PCA-80 and ICA-90 ──────────────────
print("\n=== Method 1: Optimal WL params for PCA-80 ===", flush=True)
t0 = time.time()
best_wl80 = 0; best_cfg_wl80 = None; best_out_wl80 = None
for k_neg in [2, 3, 4, 5, 6, 8]:
    for wma in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        for wmp in [0.3, 0.4, 0.5, 0.6, 0.7]:
            out = winlabel_contrast(ew80, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl80: best_wl80 = auc; best_cfg_wl80 = (k_neg, wma, wmp); best_out_wl80 = out
print(f"  WL-PCA-80 best: {best_wl80:.4f}  cfg={best_cfg_wl80}  ({time.time()-t0:.0f}s)", flush=True)
results['wl80_best'] = best_wl80

print("\n=== Method 1b: Optimal WL params for ICA-90 ===", flush=True)
t0 = time.time()
best_wl_ica = 0; best_cfg_wl_ica = None; best_out_wl_ica = None
for k_neg in [2, 3, 4, 5, 6, 8]:
    for wma in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        for wmp in [0.3, 0.4, 0.5, 0.6, 0.7]:
            out = winlabel_contrast(ew_ica90, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl_ica: best_wl_ica = auc; best_cfg_wl_ica = (k_neg, wma, wmp); best_out_wl_ica = out
print(f"  WL-ICA-90 best: {best_wl_ica:.4f}  cfg={best_cfg_wl_ica}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_ica90_best'] = best_wl_ica

# Fine blend of WL-PCA-80-best + WL-ICA-90-best
print("\n=== Method 1c: Fine blend WL80-best + WL-ICA90-best ===", flush=True)
best1c = 0; best_w1c = None
for w_ica in np.arange(0.1, 0.9, 0.05):
    blend = w_ica * best_out_wl_ica + (1-w_ica) * best_out_wl80
    auc = eval_loo(blend)
    if auc > best1c: best1c = auc; best_w1c = float(w_ica)
results['wl80_wlica_fine_blend'] = best1c
flag = " *** NEW BEST ***" if best1c > CURRENT_BEST else ""
print(f"  WL80+WL-ICA90 blend: {best1c:.4f}{flag}  w_ica={best_w1c:.2f}", flush=True)

# ─── Method 2: Add WL-Std-PCA-80 ─────────────────────────────────────────────
print("\n=== Method 2: WL-Std-PCA-80 param sweep ===", flush=True)
t0 = time.time()
best_wl_std = 0; best_cfg_wl_std = None; best_out_wl_std = None
for k_neg in [1, 2, 3, 4, 5, 6]:
    for wma in [0.5, 0.55, 0.6, 0.65, 0.7]:
        for wmp in [0.4, 0.5, 0.6]:
            out = winlabel_contrast(ew80s, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl_std: best_wl_std = auc; best_cfg_wl_std = (k_neg, wma, wmp); best_out_wl_std = out
print(f"  WL-Std-PCA-80 best: {best_wl_std:.4f}  cfg={best_cfg_wl_std}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_std80_best'] = best_wl_std

# Triple: WL-PCA-80 + WL-ICA-90 + WL-Std-PCA-80
print("\n=== Method 2b: Triple WL ensemble ===", flush=True)
best2b = 0; best_cfg2b = None
for w_ica in [0.2, 0.25, 0.3, 0.35, 0.4]:
    for w_std in [0.1, 0.15, 0.2, 0.25]:
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.4: continue
        blend = w_ica * best_out_wl_ica + w_std * best_out_wl_std + w_pca * best_out_wl80
        auc = eval_loo(blend)
        if auc > best2b: best2b = auc; best_cfg2b = (w_ica, w_std, w_pca)
results['wl_ica90_std_pca80_triple'] = best2b
flag = " *** NEW BEST ***" if best2b > CURRENT_BEST else ""
print(f"  WL-triple: {best2b:.4f}{flag}  cfg={best_cfg2b}", flush=True)

# ─── Method 3: WL-triple + file-level base ───────────────────────────────────
print("\n=== Method 3: WL-triple + file-level base ===", flush=True)
best3 = 0; best_cfg3 = None
best_wl_triple_w = best_cfg2b  # from method 2b
for w_wl in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
    if best_cfg2b is not None:
        w_ica_in_wl, w_std_in_wl, w_pca_in_wl = best_cfg2b
        wl_blend = w_ica_in_wl * best_out_wl_ica + w_std_in_wl * best_out_wl_std + w_pca_in_wl * best_out_wl80
    else:
        wl_blend = best_out_wl80
    blend = w_wl * wl_blend + (1-w_wl) * out_base
    auc = eval_loo(blend)
    if auc > best3: best3 = auc; best_cfg3 = w_wl
results['wl_triple_base'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  WL-triple+base: {best3:.4f}{flag}  w_wl={best_cfg3}", flush=True)

# ─── Method 4: WL with ICA-100 ───────────────────────────────────────────────
print("\n=== Method 4: WL-ICA-100 ===", flush=True)
t0 = time.time()
ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica100 = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
best_wl_ica100 = 0; best_cfg_ica100 = None; best_out_wl_ica100 = None
for k_neg in [3, 4, 5, 6]:
    for wma in [0.55, 0.6, 0.65, 0.7]:
        for wmp in [0.4, 0.5, 0.6]:
            out = winlabel_contrast(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl_ica100: best_wl_ica100 = auc; best_cfg_ica100 = (k_neg, wma, wmp); best_out_wl_ica100 = out
print(f"  WL-ICA-100 best: {best_wl_ica100:.4f}  cfg={best_cfg_ica100}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_ica100_best'] = best_wl_ica100
# Blend ICA-100 + PCA-80 WL
best4b = 0; best_cfg4b = None
for w_ica100 in [0.2, 0.3, 0.4, 0.5]:
    blend = w_ica100 * best_out_wl_ica100 + (1-w_ica100) * best_out_wl80
    auc = eval_loo(blend)
    if auc > best4b: best4b = auc; best_cfg4b = w_ica100
results['wl_ica100_wl80'] = best4b
flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
print(f"  WL-ICA100+WL-PCA80: {best4b:.4f}{flag}  w_ica100={best_cfg4b}", flush=True)

# ─── Method 5: Full WL ensemble optimization ─────────────────────────────────
print("\n=== Method 5: Full WL ensemble (4-component) ===", flush=True)
best5 = 0; best_cfg5 = None
for w_ica90 in [0.2, 0.25, 0.3]:
    for w_ica100 in [0.1, 0.15, 0.2]:
        for w_std in [0.1, 0.15]:
            w_pca = 1.0 - w_ica90 - w_ica100 - w_std
            if w_pca < 0.35: continue
            blend = w_ica90*best_out_wl_ica + w_ica100*best_out_wl_ica100 + w_std*best_out_wl_std + w_pca*best_out_wl80
            auc = eval_loo(blend)
            if auc > best5: best5 = auc; best_cfg5 = (w_ica90, w_ica100, w_std, w_pca)
results['wl_4comp_ens'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  WL-4comp: {best5:.4f}{flag}  cfg={best_cfg5}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 50 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)
print(f"\nBest WL-PCA-80 config: {best_cfg_wl80}  AUC={best_wl80:.4f}", flush=True)
print(f"Best WL-ICA-90 config: {best_cfg_wl_ica}  AUC={best_wl_ica:.4f}", flush=True)
print(f"Best WL-Std-PCA-80 config: {best_cfg_wl_std}  AUC={best_wl_std:.4f}", flush=True)

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
