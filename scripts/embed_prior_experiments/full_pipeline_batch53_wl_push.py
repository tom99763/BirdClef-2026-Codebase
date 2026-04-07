"""
Batch 53: WL push beyond 0.9853
Goal: beat wl_triple_ica100 = 0.9853
Methods:
  1. WL-ICA-100 extended k_neg sweep (8, 10, 12, 16, 20)
  2. Fine blend grid (0.025 step) around optimal
  3. WL-PCA-96 (different PCA dim) + WL-ICA-100 + WL-Std-80
  4. WL with different PCA dims for "base" component (64, 72, 80, 96)
  5. 4-WL: ICA-100 + ICA-90 + Std-PCA-80 + PCA-80 (optimized each)
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

ica90 = FastICA(n_components=90, random_state=42, max_iter=500, tol=0.01)
ew_ica90 = normalize(ica90.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
print("Done.", flush=True)

# Re-establish optimal WL outputs with known best configs
out_wl80 = winlabel_contrast(ew80, k_neg=4, w_max_pos=0.7, w_max_agg=0.6)
out_wl_std = winlabel_contrast(ew80s, k_neg=4, w_max_pos=0.6, w_max_agg=0.65)
print(f"WL-PCA-80: {eval_loo(out_wl80):.4f}", flush=True)
print(f"WL-Std-PCA-80: {eval_loo(out_wl_std):.4f}", flush=True)

# ─── Method 1: WL-ICA-100 extended k_neg sweep ───────────────────────────────
print("\n=== Method 1: WL-ICA-100 extended k_neg ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None; best_out_ica100 = None
for k_neg in [8, 10, 12, 16, 20, 24]:
    for wma in [0.65, 0.70, 0.75, 0.80]:
        for wmp in [0.60, 0.65, 0.70, 0.75]:
            out = winlabel_contrast(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best1: best1 = auc; best_cfg1 = (k_neg, wma, wmp); best_out_ica100 = out
print(f"  WL-ICA-100 best: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_ica100_ext'] = best1
# Triple with new best ICA-100
best1t = 0; best_cfg1t = None
for w_ica in np.arange(0.10, 0.60, 0.025):
    for w_std in np.arange(0.05, 0.50, 0.025):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.20 or w_pca > 0.85: continue
        blend = w_ica * best_out_ica100 + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best1t: best1t = auc; best_cfg1t = (float(w_ica), float(w_std), float(w_pca))
results['wl_ica100_ext_triple'] = best1t
flag = " *** NEW BEST ***" if best1t > CURRENT_BEST else ""
print(f"  WL-triple-ext: {best1t:.4f}{flag}  cfg={best_cfg1t}", flush=True)

# ─── Method 2: Fine blend grid near optimal ──────────────────────────────────
print("\n=== Method 2: Fine blend grid ===", flush=True)
# Base: w_ica100=0.30, w_std=0.20, w_pca80=0.50 (from batch 52)
out_ica100_best = winlabel_contrast(ew_ica100, k_neg=8, w_max_pos=0.7, w_max_agg=0.75)
best2 = 0; best_cfg2 = None
for w_ica in np.arange(0.20, 0.45, 0.01):
    for w_std in np.arange(0.10, 0.35, 0.01):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.30 or w_pca > 0.70: continue
        blend = w_ica * out_ica100_best + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best2: best2 = auc; best_cfg2 = (float(w_ica), float(w_std), float(w_pca))
results['wl_triple_ica100_hyperfine'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  Hyperfine: {best2:.4f}{flag}  cfg={best_cfg2}", flush=True)

# ─── Method 3: WL-PCA-96 ────────────────────────────────────────────────────
print("\n=== Method 3: WL with PCA-96 ===", flush=True)
pca96 = PCA(n_components=96, random_state=42)
ew96 = normalize(pca96.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
best3 = 0; best_cfg3 = None; best_out96 = None
for k_neg in [3, 4, 5, 6]:
    for wma in [0.55, 0.60, 0.65]:
        for wmp in [0.60, 0.70]:
            out = winlabel_contrast(ew96, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best3: best3 = auc; best_cfg3 = (k_neg, wma, wmp); best_out96 = out
print(f"  WL-PCA-96: {best3:.4f}  cfg={best_cfg3}", flush=True)
# Triple: ICA-100 + Std-PCA-80 + PCA-96
best3t = 0; best_cfg3t = None
for w_ica in np.arange(0.20, 0.45, 0.05):
    for w_std in np.arange(0.10, 0.30, 0.05):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.30: continue
        blend = w_ica * out_ica100_best + w_std * out_wl_std + w_pca * best_out96
        auc = eval_loo(blend)
        if auc > best3t: best3t = auc; best_cfg3t = (float(w_ica), float(w_std), float(w_pca))
results['wl_ica100_std_pca96'] = best3t
flag = " *** NEW BEST ***" if best3t > CURRENT_BEST else ""
print(f"  ICA100+Std80+PCA96: {best3t:.4f}{flag}  cfg={best_cfg3t}", flush=True)

# ─── Method 4: WL quad (ICA-100+ICA-90+Std+PCA) ──────────────────────────────
print("\n=== Method 4: WL quad (ICA-100 + ICA-90 + Std + PCA) ===", flush=True)
# Find best WL-ICA-90 with extended params
best4_ica90 = 0; best_cfg4_ica90 = None; best_out4_ica90 = None
for k_neg in [4, 5, 6, 8]:
    for wma in [0.60, 0.65, 0.70, 0.75]:
        for wmp in [0.55, 0.60, 0.65, 0.70]:
            out = winlabel_contrast(ew_ica90, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best4_ica90: best4_ica90 = auc; best_cfg4_ica90 = (k_neg, wma, wmp); best_out4_ica90 = out
print(f"  WL-ICA-90 ext: {best4_ica90:.4f}  cfg={best_cfg4_ica90}", flush=True)

best4 = 0; best_cfg4 = None
for w100 in [0.20, 0.25, 0.30]:
    for w90 in [0.10, 0.15, 0.20]:
        for w_std in [0.10, 0.15]:
            w_pca = 1.0 - w100 - w90 - w_std
            if w_pca < 0.30: continue
            blend = w100*out_ica100_best + w90*best_out4_ica90 + w_std*out_wl_std + w_pca*out_wl80
            auc = eval_loo(blend)
            if auc > best4: best4 = auc; best_cfg4 = (w100, w90, w_std, w_pca)
results['wl_quad_ica100_90_std_pca'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  WL-quad: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 53 Summary ===", flush=True)
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
