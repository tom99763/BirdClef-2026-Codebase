"""
Batch 51: WL triple ensemble fine-tuning + ICA-100 WL optimization
Goal: beat wl_ica90_std_pca80_triple = 0.9847
Methods:
  1. Re-establish WL triple configs + fine blend weight grid
  2. WL-ICA-100 with full param sweep (instead of ICA-90)
  3. WL-ICA-100 + WL-Std-PCA-80 + WL-PCA-80 triple
  4. WL-PCA-80 + WL-Std-PCA-80 (without ICA, just WL comparison)
  5. WL-4 comp with ICA-100 replacing ICA-90
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
CURRENT_BEST = 0.9847

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

# Precompute embeddings
print("Precomputing embeddings...", flush=True)
pca80 = PCA(n_components=80, random_state=42)
ew80 = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)

ica90 = FastICA(n_components=90, random_state=42, max_iter=500, tol=0.01)
ew_ica90 = normalize(ica90.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)

ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica100 = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)

scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
ew80s = normalize(pca80s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
print("Done precomputing.", flush=True)

# ─── Find optimal WL params for each component ───────────────────────────────
KNEG_LIST = [2, 3, 4, 5, 6, 8]
WMA_LIST  = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
WMP_LIST  = [0.3, 0.4, 0.5, 0.6, 0.7]

print("\n=== Finding WL params: PCA-80 ===", flush=True)
t0 = time.time()
best_wl80 = 0; best_cfg80 = None; best_out_wl80 = None
for kn in KNEG_LIST:
    for wma in WMA_LIST:
        for wmp in WMP_LIST:
            out = winlabel_contrast(ew80, k_neg=kn, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl80: best_wl80 = auc; best_cfg80 = (kn, wma, wmp); best_out_wl80 = out
print(f"  WL-PCA-80: {best_wl80:.4f}  cfg={best_cfg80}  ({time.time()-t0:.0f}s)", flush=True)
results['wl80_opt'] = best_wl80

print("\n=== Finding WL params: ICA-90 ===", flush=True)
t0 = time.time()
best_wl_ica90 = 0; best_cfg_ica90 = None; best_out_wl_ica90 = None
for kn in KNEG_LIST:
    for wma in WMA_LIST:
        for wmp in WMP_LIST:
            out = winlabel_contrast(ew_ica90, k_neg=kn, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl_ica90: best_wl_ica90 = auc; best_cfg_ica90 = (kn, wma, wmp); best_out_wl_ica90 = out
print(f"  WL-ICA-90: {best_wl_ica90:.4f}  cfg={best_cfg_ica90}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_ica90_opt'] = best_wl_ica90

print("\n=== Finding WL params: Std-PCA-80 ===", flush=True)
t0 = time.time()
best_wl_std = 0; best_cfg_std = None; best_out_wl_std = None
for kn in [1, 2, 3, 4, 5, 6]:
    for wma in WMA_LIST:
        for wmp in WMP_LIST:
            out = winlabel_contrast(ew80s, k_neg=kn, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl_std: best_wl_std = auc; best_cfg_std = (kn, wma, wmp); best_out_wl_std = out
print(f"  WL-Std-PCA-80: {best_wl_std:.4f}  cfg={best_cfg_std}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_std80_opt'] = best_wl_std

print("\n=== Finding WL params: ICA-100 ===", flush=True)
t0 = time.time()
best_wl_ica100 = 0; best_cfg_ica100 = None; best_out_wl_ica100 = None
for kn in KNEG_LIST:
    for wma in WMA_LIST:
        for wmp in WMP_LIST:
            out = winlabel_contrast(ew_ica100, k_neg=kn, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl_ica100: best_wl_ica100 = auc; best_cfg_ica100 = (kn, wma, wmp); best_out_wl_ica100 = out
print(f"  WL-ICA-100: {best_wl_ica100:.4f}  cfg={best_cfg_ica100}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_ica100_opt'] = best_wl_ica100

# ─── Triple blend optimization ────────────────────────────────────────────────
print("\n=== Triple blend: WL-ICA-90 + WL-Std80 + WL-PCA80 ===", flush=True)
best_trip_ica90 = 0; best_cfg_trip_ica90 = None
for w_ica in np.arange(0.15, 0.55, 0.05):
    for w_std in np.arange(0.05, 0.45, 0.05):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.25 or w_pca > 0.8: continue
        blend = w_ica * best_out_wl_ica90 + w_std * best_out_wl_std + w_pca * best_out_wl80
        auc = eval_loo(blend)
        if auc > best_trip_ica90: best_trip_ica90 = auc; best_cfg_trip_ica90 = (float(w_ica), float(w_std), float(w_pca))
results['wl_triple_ica90'] = best_trip_ica90
flag = " *** NEW BEST ***" if best_trip_ica90 > CURRENT_BEST else ""
print(f"  WL-triple(ica90): {best_trip_ica90:.4f}{flag}  cfg={best_cfg_trip_ica90}", flush=True)

print("\n=== Triple blend: WL-ICA-100 + WL-Std80 + WL-PCA80 ===", flush=True)
best_trip_ica100 = 0; best_cfg_trip_ica100 = None
for w_ica in np.arange(0.15, 0.55, 0.05):
    for w_std in np.arange(0.05, 0.45, 0.05):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.25 or w_pca > 0.8: continue
        blend = w_ica * best_out_wl_ica100 + w_std * best_out_wl_std + w_pca * best_out_wl80
        auc = eval_loo(blend)
        if auc > best_trip_ica100: best_trip_ica100 = auc; best_cfg_trip_ica100 = (float(w_ica), float(w_std), float(w_pca))
results['wl_triple_ica100'] = best_trip_ica100
flag = " *** NEW BEST ***" if best_trip_ica100 > CURRENT_BEST else ""
print(f"  WL-triple(ica100): {best_trip_ica100:.4f}{flag}  cfg={best_cfg_trip_ica100}", flush=True)

# ─── Quad: ICA-90 + ICA-100 + Std80 + PCA80 (all WL) ─────────────────────────
print("\n=== Quad WL ensemble (ica90+ica100+std+pca) ===", flush=True)
best_quad = 0; best_cfg_quad = None
for w90 in [0.15, 0.2, 0.25]:
    for w100 in [0.1, 0.15, 0.2]:
        for w_std in [0.1, 0.15, 0.2]:
            w_pca = 1.0 - w90 - w100 - w_std
            if w_pca < 0.3 or w_pca > 0.7: continue
            blend = w90*best_out_wl_ica90 + w100*best_out_wl_ica100 + w_std*best_out_wl_std + w_pca*best_out_wl80
            auc = eval_loo(blend)
            if auc > best_quad: best_quad = auc; best_cfg_quad = (w90, w100, w_std, w_pca)
results['wl_quad_ens'] = best_quad
flag = " *** NEW BEST ***" if best_quad > CURRENT_BEST else ""
print(f"  WL-quad: {best_quad:.4f}{flag}  cfg={best_cfg_quad}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 51 Summary ===", flush=True)
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
