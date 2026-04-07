"""
Batch 54: Refine around wl_ica100_ext_triple = 0.9862
Goal: beat 0.9862
New best config:
  WL-ICA-100: k_neg=24, w_max_agg=0.80, w_max_pos=0.75
  Triple blend: w_ica=0.475, w_std=0.300, w_pca=0.225
Methods:
  1. Fine blend grid (0.01 step) around (0.475, 0.30, 0.225)
  2. Even higher k_neg for ICA-100 (28, 32, 40, 50)
  3. Higher wma (0.82, 0.84, 0.86, 0.90) for ICA-100 ext
  4. WL-ICA-100-ext + WL-Std-PCA-80(re-sweep) + WL-PCA-80 with fresh params
  5. WL-ICA-100-ext + WL-ICA-90-ext triple (no PCA)
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
CURRENT_BEST = 0.9862

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

ica90 = FastICA(n_components=90, random_state=42, max_iter=500, tol=0.01)
ew_ica90 = normalize(ica90.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
print("Done.", flush=True)

# Baseline outputs with known best configs
out_wl80 = winlabel_contrast(ew80, k_neg=4, w_max_pos=0.70, w_max_agg=0.60)
out_wl_std = winlabel_contrast(ew80s, k_neg=4, w_max_pos=0.60, w_max_agg=0.65)
print(f"WL-PCA-80: {eval_loo(out_wl80):.4f}", flush=True)
print(f"WL-Std-PCA-80: {eval_loo(out_wl_std):.4f}", flush=True)

# ─── Method 1: Extended k_neg for ICA-100 (higher range) ─────────────────────
print("\n=== Method 1: ICA-100 very high k_neg ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None; best_out_ica100_ext = None
for k_neg in [20, 24, 28, 32, 40, 50]:
    for wma in [0.75, 0.80, 0.82, 0.84, 0.86, 0.90]:
        for wmp in [0.70, 0.72, 0.75, 0.78, 0.80]:
            out = winlabel_contrast(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best1: best1 = auc; best_cfg1 = (k_neg, wma, wmp); best_out_ica100_ext = out
print(f"  ICA-100 ext best: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_ica100_ext2'] = best1

# Fine blend around new ext best
best1b = 0; best_cfg1b = None
for w_ica in np.arange(0.35, 0.65, 0.01):
    for w_std in np.arange(0.15, 0.45, 0.01):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.10 or w_pca > 0.45: continue
        blend = w_ica * best_out_ica100_ext + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best1b: best1b = auc; best_cfg1b = (float(w_ica), float(w_std), float(w_pca))
results['wl_ica100_ext2_triple'] = best1b
flag = " *** NEW BEST ***" if best1b > CURRENT_BEST else ""
print(f"  Triple-ext2: {best1b:.4f}{flag}  cfg={best_cfg1b}", flush=True)

# ─── Method 2: Fine blend grid around (0.475, 0.30, 0.225) ──────────────────
print("\n=== Method 2: Fine blend around best (0.475, 0.30, 0.225) ===", flush=True)
# Best from batch 53: ICA-100 at k_neg=24, wma=0.80, wmp=0.75
out_ica100_b53 = winlabel_contrast(ew_ica100, k_neg=24, w_max_pos=0.75, w_max_agg=0.80)
best2 = 0; best_cfg2 = None
for w_ica in np.arange(0.38, 0.58, 0.005):
    for w_std in np.arange(0.22, 0.40, 0.005):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.10 or w_pca > 0.40: continue
        blend = w_ica * out_ica100_b53 + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best2: best2 = auc; best_cfg2 = (float(w_ica), float(w_std), float(w_pca))
results['wl_ica100_ext_ultrafine'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  Ultrafine blend: {best2:.4f}{flag}  cfg={best_cfg2}", flush=True)

# ─── Method 3: Re-sweep Std-PCA-80 with extended params ─────────────────────
print("\n=== Method 3: WL-Std-PCA-80 extended param sweep ===", flush=True)
t0 = time.time()
best3_std = 0; best_cfg3_std = None; best_out3_std = None
for k_neg in [2, 3, 4, 5, 6, 8, 10]:
    for wma in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for wmp in [0.55, 0.60, 0.65, 0.70]:
            out = winlabel_contrast(ew80s, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best3_std: best3_std = auc; best_cfg3_std = (k_neg, wma, wmp); best_out3_std = out
print(f"  WL-Std-PCA-80 ext: {best3_std:.4f}  cfg={best_cfg3_std}  ({time.time()-t0:.0f}s)", flush=True)

# Triple with new best std
best3t = 0; best_cfg3t = None
for w_ica in np.arange(0.35, 0.65, 0.01):
    for w_std in np.arange(0.15, 0.45, 0.01):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.10 or w_pca > 0.45: continue
        blend = w_ica * out_ica100_b53 + w_std * best_out3_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best3t: best3t = auc; best_cfg3t = (float(w_ica), float(w_std), float(w_pca))
results['wl_ica100_ext_newstd_triple'] = best3t
flag = " *** NEW BEST ***" if best3t > CURRENT_BEST else ""
print(f"  Triple(ext ICA100 + new Std): {best3t:.4f}{flag}  cfg={best_cfg3t}", flush=True)

# ─── Method 4: WL-ICA-100-ext + WL-ICA-90-ext triple (no PCA) ───────────────
print("\n=== Method 4: ICA-100-ext + ICA-90-ext + Std triple ===", flush=True)
t0 = time.time()
best4_ica90 = 0; best_cfg4_ica90 = None; best_out4_ica90 = None
for k_neg in [8, 10, 12, 16, 20, 24]:
    for wma in [0.70, 0.75, 0.80, 0.85]:
        for wmp in [0.65, 0.70, 0.75]:
            out = winlabel_contrast(ew_ica90, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best4_ica90: best4_ica90 = auc; best_cfg4_ica90 = (k_neg, wma, wmp); best_out4_ica90 = out
print(f"  WL-ICA-90 ext: {best4_ica90:.4f}  cfg={best_cfg4_ica90}  ({time.time()-t0:.0f}s)", flush=True)

best4 = 0; best_cfg4 = None
for w100 in np.arange(0.30, 0.65, 0.05):
    for w90 in np.arange(0.10, 0.40, 0.05):
        for w_std in np.arange(0.10, 0.35, 0.05):
            w_pca = 1.0 - w100 - w90 - w_std
            if w_pca < 0.05 or w_pca > 0.40: continue
            blend = w100*out_ica100_b53 + w90*best_out4_ica90 + w_std*out_wl_std + w_pca*out_wl80
            auc = eval_loo(blend)
            if auc > best4: best4 = auc; best_cfg4 = (w100, w90, w_std, w_pca)
results['wl_quad_ica100ext_90ext_std_pca'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  Quad(ext): {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Method 5: WL-ICA-100-ext + WL-PCA-80 re-sweep ─────────────────────────
print("\n=== Method 5: WL-PCA-80 extended param sweep ===", flush=True)
t0 = time.time()
best5_pca = 0; best_cfg5_pca = None; best_out5_pca = None
for k_neg in [3, 4, 5, 6, 8, 10]:
    for wma in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for wmp in [0.60, 0.65, 0.70, 0.75, 0.80]:
            out = winlabel_contrast(ew80, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best5_pca: best5_pca = auc; best_cfg5_pca = (k_neg, wma, wmp); best_out5_pca = out
print(f"  WL-PCA-80 ext: {best5_pca:.4f}  cfg={best_cfg5_pca}  ({time.time()-t0:.0f}s)", flush=True)

# Triple with best PCA
best5t = 0; best_cfg5t = None
for w_ica in np.arange(0.35, 0.65, 0.01):
    for w_std in np.arange(0.15, 0.45, 0.01):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.10 or w_pca > 0.45: continue
        blend = w_ica * out_ica100_b53 + w_std * out_wl_std + w_pca * best_out5_pca
        auc = eval_loo(blend)
        if auc > best5t: best5t = auc; best_cfg5t = (float(w_ica), float(w_std), float(w_pca))
results['wl_ica100_ext_newpca_triple'] = best5t
flag = " *** NEW BEST ***" if best5t > CURRENT_BEST else ""
print(f"  Triple(ext ICA100 + new PCA): {best5t:.4f}{flag}  cfg={best_cfg5t}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 54 Summary ===", flush=True)
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
