"""
Batch 41: Whitened PCA + k_neg tuning + diverse dim ensembles
Goal: beat ica90_base = 0.9729
Methods:
  1. Whitened PCA (svd_solver + whiten=True)
  2. ICA-90 with k_neg sweep (2,3,4,5,6,8)
  3. ICA multi-dim ensemble: best 3 dims (90, 92, 100) ensemble
  4. ICA-90 + PCA-64 + PCA-80 triple blend
  5. Standardized embedding (z-score) → PCA-80
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
CURRENT_BEST = 0.9729

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

# PCA-80 base
pca80 = PCA(n_components=80, random_state=42)
emb80 = pca80.fit_transform(emb_win).astype(np.float32)
ew80 = normalize(emb80, norm='l2').astype(np.float32)
out_base = maxmean_contrast(ew80)
print(f"Base (pca80): {eval_loo(out_base):.4f}", flush=True)

# ICA-90 reference
ica90 = FastICA(n_components=90, random_state=42, max_iter=500, tol=0.01)
emb_ica90 = ica90.fit_transform(emb_win).astype(np.float32)
ew_ica90 = normalize(emb_ica90, norm='l2').astype(np.float32)
out_ica90 = maxmean_contrast(ew_ica90)
print(f"ICA-90: {eval_loo(out_ica90):.4f}", flush=True)

# ─── Method 1: Whitened PCA-80 ────────────────────────────────────────────────
print("\n=== Method 1: Whitened PCA-80 ===", flush=True)
t0 = time.time()
pca80w = PCA(n_components=80, whiten=True, random_state=42)
emb80w = pca80w.fit_transform(emb_win).astype(np.float32)
ew80w = normalize(emb80w, norm='l2').astype(np.float32)
out_w80 = maxmean_contrast(ew80w)
auc1 = eval_loo(out_w80)
results['whitened_pca80'] = auc1
flag = " *** NEW BEST ***" if auc1 > CURRENT_BEST else ""
print(f"  Whitened PCA-80: {auc1:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
# Blend with ICA-90
best1b = 0; best_w1b = None
for w_ica in [0.3, 0.35, 0.4, 0.45, 0.5]:
    blend = w_ica * out_ica90 + (1-w_ica) * out_w80
    auc_c = eval_loo(blend)
    if auc_c > best1b: best1b = auc_c; best_w1b = w_ica
results['ica90_wpca80'] = best1b
flag = " *** NEW BEST ***" if best1b > CURRENT_BEST else ""
print(f"  ICA90+WhitenedPCA80: {best1b:.4f}{flag}  w_ica={best_w1b}", flush=True)
# Triple: ICA-90 + whitened + normal PCA-80
best1c = 0; best_cfg1c = None
for w_ica in [0.3, 0.4]:
    for w_wh in [0.2, 0.3, 0.4]:
        w_n = 1.0 - w_ica - w_wh
        if w_n < 0.2: continue
        blend = w_ica * out_ica90 + w_wh * out_w80 + w_n * out_base
        auc_c = eval_loo(blend)
        if auc_c > best1c: best1c = auc_c; best_cfg1c = (w_ica, w_wh, w_n)
results['ica90_wpca80_pca80'] = best1c
flag = " *** NEW BEST ***" if best1c > CURRENT_BEST else ""
print(f"  ICA90+WPCA80+PCA80: {best1c:.4f}{flag}  cfg={best_cfg1c}", flush=True)

# ─── Method 2: ICA-90 k_neg sweep ─────────────────────────────────────────────
print("\n=== Method 2: ICA-90 k_neg sweep ===", flush=True)
best2 = 0; best_cfg2 = None
for k_neg in [2, 3, 5, 6, 8, 10]:
    out_kn = maxmean_contrast(ew_ica90, k_neg=k_neg)
    auc_kn = eval_loo(out_kn)
    # Blend with PCA-80
    best_b = 0; best_wb = None
    for w_ica in [0.35, 0.40, 0.45]:
        blend = w_ica * out_kn + (1-w_ica) * out_base
        auc_c = eval_loo(blend)
        if auc_c > best_b: best_b = auc_c; best_wb = w_ica
    results[f'ica90_kn{k_neg}_base'] = best_b
    flag = " *** NEW BEST ***" if best_b > CURRENT_BEST else ""
    print(f"  k_neg={k_neg}: ICA={auc_kn:.4f}  +base={best_b:.4f}{flag}  w={best_wb}", flush=True)
    if best_b > best2: best2 = best_b; best_cfg2 = (k_neg, best_wb)

# ─── Method 3: Multi-dim ICA ensemble (90+92+100) ─────────────────────────────
print("\n=== Method 3: Multi-dim ICA ensemble (90+92+100) ===", flush=True)
t0 = time.time()
ica92 = FastICA(n_components=92, random_state=42, max_iter=500, tol=0.01)
emb_ica92 = ica92.fit_transform(emb_win).astype(np.float32)
ew_ica92 = normalize(emb_ica92, norm='l2').astype(np.float32)
out_ica92 = maxmean_contrast(ew_ica92)
print(f"  ICA-92: {eval_loo(out_ica92):.4f}", flush=True)

ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
emb_ica100 = ica100.fit_transform(emb_win).astype(np.float32)
ew_ica100 = normalize(emb_ica100, norm='l2').astype(np.float32)
out_ica100 = maxmean_contrast(ew_ica100)

best3 = 0; best_cfg3 = None
for w90 in [0.2, 0.25, 0.3]:
    for w92 in [0.1, 0.15, 0.2]:
        for w100 in [0.1, 0.15]:
            w_base = 1.0 - w90 - w92 - w100
            if w_base < 0.35 or w_base > 0.65: continue
            blend = w90*out_ica90 + w92*out_ica92 + w100*out_ica100 + w_base*out_base
            auc_c = eval_loo(blend)
            if auc_c > best3: best3 = auc_c; best_cfg3 = (w90, w92, w100, w_base)
results['ica90_92_100_base'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  ICA90+92+100+base: {best3:.4f}{flag}  cfg={best_cfg3}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: ICA-90 + PCA-64 + PCA-80 triple ───────────────────────────────
print("\n=== Method 4: ICA-90 + PCA-64 + PCA-80 triple ===", flush=True)
pca64 = PCA(n_components=64, random_state=42)
emb64 = pca64.fit_transform(emb_win).astype(np.float32)
ew64 = normalize(emb64, norm='l2').astype(np.float32)
out64 = maxmean_contrast(ew64)
print(f"  PCA-64: {eval_loo(out64):.4f}", flush=True)

best4 = 0; best_cfg4 = None
for w_ica in [0.3, 0.35, 0.4]:
    for w64 in [0.1, 0.15, 0.2, 0.25]:
        w80 = 1.0 - w_ica - w64
        if w80 < 0.3 or w80 > 0.6: continue
        blend = w_ica * out_ica90 + w64 * out64 + w80 * out_base
        auc_c = eval_loo(blend)
        if auc_c > best4: best4 = auc_c; best_cfg4 = (w_ica, w64, w80)
results['ica90_pca64_pca80'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  ICA90+PCA64+PCA80: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Method 5: Standardized embedding → PCA-80 ───────────────────────────────
print("\n=== Method 5: Standardized embedding → PCA-80 ===", flush=True)
t0 = time.time()
scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
emb80s = pca80s.fit_transform(emb_std).astype(np.float32)
ew80s = normalize(emb80s, norm='l2').astype(np.float32)
out_std = maxmean_contrast(ew80s)
auc5 = eval_loo(out_std)
results['std_pca80'] = auc5
flag = " *** NEW BEST ***" if auc5 > CURRENT_BEST else ""
print(f"  Standardized PCA-80: {auc5:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
# Blend with ICA-90 + base
best5b = 0; best_cfg5b = None
for w_ica in [0.35, 0.40]:
    for w_std in [0.1, 0.15, 0.2]:
        w_base2 = 1.0 - w_ica - w_std
        if w_base2 < 0.4: continue
        blend = w_ica * out_ica90 + w_std * out_std + w_base2 * out_base
        auc_c = eval_loo(blend)
        if auc_c > best5b: best5b = auc_c; best_cfg5b = (w_ica, w_std, w_base2)
results['ica90_std_base'] = best5b
flag = " *** NEW BEST ***" if best5b > CURRENT_BEST else ""
print(f"  ICA90+std+base: {best5b:.4f}{flag}  cfg={best_cfg5b}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 41 Summary ===", flush=True)
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
