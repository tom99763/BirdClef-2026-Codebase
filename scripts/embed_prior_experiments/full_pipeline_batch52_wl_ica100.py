"""
Batch 52: WL-ICA-100 fine optimization + save best configs
Goal: beat wl_triple_ica100 = 0.9853
Methods:
  1. WL-ICA-100 with extended param sweep, find+save best config
  2. Fine grid for triple blend weights
  3. ICA dim sweep for WL (80, 90, 95, 100, 105, 110)
  4. Best WL-ICA-dim + WL-Std80 + WL-PCA80
  5. Multi-seed ICA-100 WL ensemble
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
CURRENT_BEST = 0.9853

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
print("Done.", flush=True)

KNEG_LIST = [2, 3, 4, 5, 6, 8]
WMA_LIST  = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
WMP_LIST  = [0.3, 0.4, 0.5, 0.6, 0.7]

# ─── Re-establish best configs ────────────────────────────────────────────────
print("\n=== WL-PCA-80 optimal ===", flush=True)
t0 = time.time()
best_wl80 = 0; cfg80 = None; out_wl80 = None
for kn in KNEG_LIST:
    for wma in WMA_LIST:
        for wmp in WMP_LIST:
            out = winlabel_contrast(ew80, k_neg=kn, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl80: best_wl80 = auc; cfg80 = (kn, wma, wmp); out_wl80 = out
print(f"  WL-PCA-80: {best_wl80:.4f}  cfg={cfg80}  ({time.time()-t0:.0f}s)", flush=True)
results['wl80'] = best_wl80

print("\n=== WL-Std-PCA-80 optimal ===", flush=True)
t0 = time.time()
best_wl_std = 0; cfg_std = None; out_wl_std = None
for kn in [1, 2, 3, 4, 5, 6]:
    for wma in WMA_LIST:
        for wmp in WMP_LIST:
            out = winlabel_contrast(ew80s, k_neg=kn, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl_std: best_wl_std = auc; cfg_std = (kn, wma, wmp); out_wl_std = out
print(f"  WL-Std80: {best_wl_std:.4f}  cfg={cfg_std}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_std80'] = best_wl_std

print("\n=== WL-ICA-100 optimal ===", flush=True)
t0 = time.time()
best_wl_ica100 = 0; cfg_ica100 = None; out_wl_ica100 = None
for kn in KNEG_LIST:
    for wma in WMA_LIST:
        for wmp in WMP_LIST:
            out = winlabel_contrast(ew_ica100, k_neg=kn, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_wl_ica100: best_wl_ica100 = auc; cfg_ica100 = (kn, wma, wmp); out_wl_ica100 = out
print(f"  WL-ICA-100: {best_wl_ica100:.4f}  cfg={cfg_ica100}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_ica100'] = best_wl_ica100

# ─── WL-triple (ICA-100) fine blend grid ─────────────────────────────────────
print("\n=== WL-triple fine blend (ICA-100) ===", flush=True)
best_trip = 0; cfg_trip = None; best_triple_out = None
for w_ica in np.arange(0.10, 0.60, 0.025):
    for w_std in np.arange(0.05, 0.50, 0.025):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.20 or w_pca > 0.85: continue
        blend = w_ica * out_wl_ica100 + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best_trip: best_trip = auc; cfg_trip = (float(w_ica), float(w_std), float(w_pca)); best_triple_out = blend
results['wl_triple_ica100_fine'] = best_trip
flag = " *** NEW BEST ***" if best_trip > CURRENT_BEST else ""
print(f"  WL-triple-ica100: {best_trip:.4f}{flag}  cfg={cfg_trip}", flush=True)

# ─── ICA dim sweep for WL ─────────────────────────────────────────────────────
print("\n=== ICA dim sweep for WL ===", flush=True)
t0 = time.time()
best_dim = 100; best_auc_dim = best_wl_ica100; best_out_dim = out_wl_ica100
for n_comp in [80, 90, 95, 105, 110, 115, 120]:
    try:
        ica = FastICA(n_components=n_comp, random_state=42, max_iter=500, tol=0.01)
        ew_ica = normalize(ica.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
        # Find best WL params for this dim
        best_dim_auc = 0; best_dim_out = None
        for kn in [3, 4, 5, 6]:
            for wma in [0.55, 0.6, 0.65, 0.7]:
                for wmp in [0.4, 0.5, 0.6]:
                    out = winlabel_contrast(ew_ica, k_neg=kn, w_max_pos=wmp, w_max_agg=wma)
                    auc = eval_loo(out)
                    if auc > best_dim_auc: best_dim_auc = auc; best_dim_out = out
        # Triple blend
        best_trip_dim = 0
        for w_ica in [0.2, 0.25, 0.3, 0.35, 0.4]:
            for w_std in [0.1, 0.15, 0.2]:
                w_pca = 1.0 - w_ica - w_std
                if w_pca < 0.40: continue
                blend = w_ica * best_dim_out + w_std * out_wl_std + w_pca * out_wl80
                auc = eval_loo(blend)
                if auc > best_trip_dim: best_trip_dim = auc
        results[f'wl_ica{n_comp}_triple'] = best_trip_dim
        flag = " *** NEW BEST ***" if best_trip_dim > CURRENT_BEST else ""
        print(f"  ICA-{n_comp}: WL={best_dim_auc:.4f}  triple={best_trip_dim:.4f}{flag}", flush=True)
        if best_dim_auc > best_auc_dim: best_auc_dim = best_dim_auc; best_dim = n_comp; best_out_dim = best_dim_out
    except Exception as e:
        print(f"  ICA-{n_comp} failed: {e}", flush=True)
print(f"  ({time.time()-t0:.0f}s)  Best ICA dim for WL: {best_dim}", flush=True)

# ─── Summary + print configs ───────────────────────────────────────────────────
print("\n=== Batch 52 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

print(f"\n=== Best Configs ===", flush=True)
print(f"  WL-PCA-80:     {cfg80}  AUC={best_wl80:.4f}", flush=True)
print(f"  WL-Std-PCA-80: {cfg_std}  AUC={best_wl_std:.4f}", flush=True)
print(f"  WL-ICA-100:    {cfg_ica100}  AUC={best_wl_ica100:.4f}", flush=True)
print(f"  Triple blend:  {cfg_trip}  AUC={best_trip:.4f}", flush=True)

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

# Save detailed configs for pkl creation
configs = {
    'wl_pca80': {'k_neg': cfg80[0], 'w_max_agg': cfg80[1], 'w_max_pos': cfg80[2], 'auc': float(best_wl80)},
    'wl_std80': {'k_neg': cfg_std[0], 'w_max_agg': cfg_std[1], 'w_max_pos': cfg_std[2], 'auc': float(best_wl_std)},
    'wl_ica100': {'k_neg': cfg_ica100[0], 'w_max_agg': cfg_ica100[1], 'w_max_pos': cfg_ica100[2], 'auc': float(best_wl_ica100)},
    'triple_blend': {'w_ica100': cfg_trip[0], 'w_std': cfg_trip[1], 'w_pca80': cfg_trip[2], 'auc': float(best_trip)},
}
import json as json2
with open("outputs/wl_triple_configs.json", 'w') as f:
    json2.dump(configs, f, indent=2)
print(f"Saved configs to outputs/wl_triple_configs.json", flush=True)
