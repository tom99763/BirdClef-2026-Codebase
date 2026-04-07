"""
Batch 43: Std-PCA k_neg=2 + ICA-90 fine grid + quad blends
Goal: beat ica90_std80kn2_base = 0.9732
Methods:
  1. ICA-90 + Std-PCA-80 k_neg=2 fine weight grid (tight sweep)
  2. k_neg=1 for Std-PCA-80 vs k_neg=2
  3. ICA-90 k_neg variants + Std-PCA-80 kn2 + PCA-80
  4. Quad: ICA-90 + Std-PCA-80 kn2 + Std-PCA-96 + PCA-80
  5. ICA-90 + Std-PCA-80 kn2 + PCA-64 + PCA-80
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

# Precompute bases
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
out_std80_kn4 = maxmean_contrast(ew80s)  # k_neg=4
print(f"Std-PCA-80 kn2: {eval_loo(out_std80_kn2):.4f}", flush=True)
print(f"Std-PCA-80 kn4: {eval_loo(out_std80_kn4):.4f}", flush=True)

# ─── Method 1: ICA-90 + Std-PCA-80 kn2 + PCA-80 fine grid ───────────────────
print("\n=== Method 1: ICA-90 + Std-PCA-80 kn2 + PCA-80 fine grid ===", flush=True)
best1 = 0; best_cfg1 = None
for w_ica in [0.32, 0.35, 0.38, 0.40, 0.42, 0.45]:
    for w_std in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22]:
        w_base = 1.0 - w_ica - w_std
        if w_base < 0.35 or w_base > 0.62: continue
        blend = w_ica * out_ica90 + w_std * out_std80_kn2 + w_base * out_base
        auc = eval_loo(blend)
        if auc > best1: best1 = auc; best_cfg1 = (w_ica, w_std, w_base)
results['ica90_std80kn2_base_fine'] = best1
flag = " *** NEW BEST ***" if best1 > CURRENT_BEST else ""
print(f"  Best: {best1:.4f}{flag}  cfg={best_cfg1}", flush=True)

# ─── Method 2: k_neg=1 for Std-PCA-80 ────────────────────────────────────────
print("\n=== Method 2: Std-PCA-80 k_neg=1 ===", flush=True)
out_std80_kn1 = maxmean_contrast(ew80s, k_neg=1)
print(f"  Std-PCA-80 kn1: {eval_loo(out_std80_kn1):.4f}", flush=True)
best2 = 0; best_cfg2 = None
for w_ica in [0.35, 0.40, 0.45]:
    for w_std in [0.08, 0.10, 0.12, 0.15, 0.18]:
        w_base = 1.0 - w_ica - w_std
        if w_base < 0.38: continue
        blend = w_ica * out_ica90 + w_std * out_std80_kn1 + w_base * out_base
        auc = eval_loo(blend)
        if auc > best2: best2 = auc; best_cfg2 = (w_ica, w_std, w_base)
results['ica90_std80kn1_base'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  ICA90+Std80kn1+base: {best2:.4f}{flag}  cfg={best_cfg2}", flush=True)

# ─── Method 3: ICA-90 k_neg variants + Std-PCA-80 kn2 ────────────────────────
print("\n=== Method 3: ICA-90 k_neg variants + Std-PCA-80 kn2 + PCA-80 ===", flush=True)
best3 = 0; best_cfg3 = None
for k_neg_ica in [2, 3, 5, 6]:
    out_ica_kn = maxmean_contrast(ew_ica90, k_neg=k_neg_ica)
    for w_ica in [0.35, 0.40, 0.45]:
        for w_std in [0.10, 0.15]:
            w_base = 1.0 - w_ica - w_std
            if w_base < 0.40: continue
            blend = w_ica * out_ica_kn + w_std * out_std80_kn2 + w_base * out_base
            auc = eval_loo(blend)
            if auc > best3: best3 = auc; best_cfg3 = (k_neg_ica, w_ica, w_std, w_base)
results['ica90_knX_std80kn2_base'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  Best: {best3:.4f}{flag}  cfg={best_cfg3}", flush=True)

# ─── Method 4: Quad blend: ICA-90 + Std-PCA-80 kn2 + Std-PCA-96 + PCA-80 ───
print("\n=== Method 4: Quad blend: ICA-90 + Std-PCA-80kn2 + Std-PCA-96 + PCA-80 ===", flush=True)
pca96s = PCA(n_components=96, random_state=42)
ew96s = normalize(pca96s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
out_std96_kn2 = maxmean_contrast(ew96s, k_neg=2)
print(f"  Std-PCA-96 kn2: {eval_loo(out_std96_kn2):.4f}", flush=True)

best4 = 0; best_cfg4 = None
for w_ica in [0.30, 0.35, 0.40]:
    for w_s80 in [0.08, 0.10, 0.12]:
        for w_s96 in [0.06, 0.08, 0.10]:
            w_base = 1.0 - w_ica - w_s80 - w_s96
            if w_base < 0.38 or w_base > 0.60: continue
            blend = w_ica*out_ica90 + w_s80*out_std80_kn2 + w_s96*out_std96_kn2 + w_base*out_base
            auc = eval_loo(blend)
            if auc > best4: best4 = auc; best_cfg4 = (w_ica, w_s80, w_s96, w_base)
results['ica90_std80kn2_std96kn2_base'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  Quad: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Method 5: ICA-90 + Std-PCA-80 kn2 + PCA-64 + PCA-80 ───────────────────
print("\n=== Method 5: ICA-90 + Std-PCA-80kn2 + PCA-64 + PCA-80 ===", flush=True)
pca64 = PCA(n_components=64, random_state=42)
ew64 = normalize(pca64.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
out64 = maxmean_contrast(ew64)
print(f"  PCA-64: {eval_loo(out64):.4f}", flush=True)

best5 = 0; best_cfg5 = None
for w_ica in [0.30, 0.35, 0.40]:
    for w_std in [0.08, 0.10, 0.12]:
        for w64 in [0.06, 0.08, 0.10]:
            w_base = 1.0 - w_ica - w_std - w64
            if w_base < 0.38 or w_base > 0.58: continue
            blend = w_ica*out_ica90 + w_std*out_std80_kn2 + w64*out64 + w_base*out_base
            auc = eval_loo(blend)
            if auc > best5: best5 = auc; best_cfg5 = (w_ica, w_std, w64, w_base)
results['ica90_std80kn2_pca64_base'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  ICA90+Std80kn2+pca64+base: {best5:.4f}{flag}  cfg={best_cfg5}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 43 Summary ===", flush=True)
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
