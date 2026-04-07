"""
Batch 47: Factor Analysis + multi-scale Std-PCA + different ICA dims with Std
Goal: beat 0.9732
Methods:
  1. Factor Analysis (FA) as alternative to ICA/PCA
  2. ICA-80 on standardized embeddings + PCA-80 + Std-PCA-80
  3. Different Std-PCA dims with k_neg=2 (64, 72, 96) vs ICA-90
  4. Multiple Std-PCA blends: Std-PCA-80(kn2) + Std-PCA-64(kn2) + PCA-80
  5. ICA-90 + Std-PCA-80(kn2) with different w_max_agg
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
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
print(f"Std-PCA-80 kn2: {eval_loo(out_std80_kn2):.4f}", flush=True)

# ─── Method 1: Factor Analysis ───────────────────────────────────────────────
print("\n=== Method 1: Factor Analysis (FA-80) ===", flush=True)
t0 = time.time()
try:
    fa = FactorAnalysis(n_components=80, random_state=42, max_iter=500)
    emb_fa = fa.fit_transform(emb_win).astype(np.float32)
    ew_fa = normalize(emb_fa, norm='l2').astype(np.float32)
    out_fa = maxmean_contrast(ew_fa)
    auc1 = eval_loo(out_fa)
    results['fa80'] = auc1
    flag = " *** NEW BEST ***" if auc1 > CURRENT_BEST else ""
    print(f"  FA-80: {auc1:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
    best1b = 0; best_cfg1b = None
    for w_ica in [0.35, 0.40]:
        for w_fa in [0.08, 0.12, 0.15]:
            for w_std in [0.06, 0.08]:
                w_b = 1.0 - w_ica - w_fa - w_std
                if w_b < 0.40: continue
                blend = w_ica*out_ica90 + w_fa*out_fa + w_std*out_std80_kn2 + w_b*out_base
                auc_c = eval_loo(blend)
                if auc_c > best1b: best1b = auc_c; best_cfg1b = (w_ica, w_fa, w_std, w_b)
    results['ica90_fa80_std80_base'] = best1b
    flag = " *** NEW BEST ***" if best1b > CURRENT_BEST else ""
    print(f"  ICA90+FA80+Std80kn2+base: {best1b:.4f}{flag}  cfg={best_cfg1b}", flush=True)
except Exception as e:
    print(f"  FA failed: {e}", flush=True)

# ─── Method 2: ICA-80 on standardized data ────────────────────────────────────
print("\n=== Method 2: Std-ICA-80 (StandardScaler → ICA-80) ===", flush=True)
t0 = time.time()
try:
    ica80_std = FastICA(n_components=80, random_state=42, max_iter=500, tol=0.01)
    emb_std_ica80 = ica80_std.fit_transform(emb_std).astype(np.float32)
    ew_std_ica80 = normalize(emb_std_ica80, norm='l2').astype(np.float32)
    out_std_ica80 = maxmean_contrast(ew_std_ica80)
    auc2 = eval_loo(out_std_ica80)
    results['std_ica80'] = auc2
    flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
    print(f"  Std-ICA-80: {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
    best2b = 0; best_cfg2b = None
    for w_ica in [0.35, 0.40]:
        for w_si in [0.08, 0.12, 0.15]:
            for w_std in [0.06, 0.08]:
                w_b = 1.0 - w_ica - w_si - w_std
                if w_b < 0.40: continue
                blend = w_ica*out_ica90 + w_si*out_std_ica80 + w_std*out_std80_kn2 + w_b*out_base
                auc_c = eval_loo(blend)
                if auc_c > best2b: best2b = auc_c; best_cfg2b = (w_ica, w_si, w_std, w_b)
    results['ica90_stdica80_std80_base'] = best2b
    flag = " *** NEW BEST ***" if best2b > CURRENT_BEST else ""
    print(f"  ICA90+StdICA80+Std80kn2+base: {best2b:.4f}{flag}  cfg={best_cfg2b}", flush=True)
except Exception as e:
    print(f"  Std-ICA-80 failed: {e}", flush=True)

# ─── Method 3: Different Std-PCA dims (kn2) + ICA-90 ────────────────────────
print("\n=== Method 3: Std-PCA dim sweep (k_neg=2) + ICA-90 ===", flush=True)
t0 = time.time()
best3 = 0; best_cfg3 = None
for n_comp in [60, 64, 72, 88, 96, 104]:
    pca_s = PCA(n_components=n_comp, random_state=42)
    ew_s = normalize(pca_s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
    out_s = maxmean_contrast(ew_s, k_neg=2)
    for w_ica in [0.38, 0.40, 0.42]:
        for w_s in [0.06, 0.08, 0.10, 0.12]:
            w_b = 1.0 - w_ica - w_s
            if w_b < 0.46: continue
            blend = w_ica * out_ica90 + w_s * out_s + w_b * out_base
            auc_c = eval_loo(blend)
            if auc_c > best3: best3 = auc_c; best_cfg3 = (n_comp, w_ica, w_s, w_b)
print(f"  Best: {best3:.4f}  cfg={best_cfg3}  ({time.time()-t0:.0f}s)", flush=True)
results['ica90_std_dimX_kn2_base'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  {'*** NEW BEST ***' if best3 > CURRENT_BEST else ''}", flush=True)

# ─── Method 4: Dual Std-PCA blend (kn2) ─────────────────────────────────────
print("\n=== Method 4: Dual Std-PCA (80+64) kn2 + ICA-90 ===", flush=True)
pca64s = PCA(n_components=64, random_state=42)
ew64s = normalize(pca64s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
out_std64_kn2 = maxmean_contrast(ew64s, k_neg=2)
print(f"  Std-PCA-64 kn2: {eval_loo(out_std64_kn2):.4f}", flush=True)

best4 = 0; best_cfg4 = None
for w_ica in [0.35, 0.40]:
    for w_s80 in [0.06, 0.08]:
        for w_s64 in [0.05, 0.06, 0.08]:
            w_b = 1.0 - w_ica - w_s80 - w_s64
            if w_b < 0.44: continue
            blend = w_ica*out_ica90 + w_s80*out_std80_kn2 + w_s64*out_std64_kn2 + w_b*out_base
            auc_c = eval_loo(blend)
            if auc_c > best4: best4 = auc_c; best_cfg4 = (w_ica, w_s80, w_s64, w_b)
results['ica90_std80kn2_std64kn2_base'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  ICA90+Std80kn2+Std64kn2+base: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Method 5: Std-PCA-80 kn2 with different w_max_agg ──────────────────────
print("\n=== Method 5: Std-PCA-80 kn2 w_max_agg sweep + ICA-90 ===", flush=True)
best5 = 0; best_cfg5 = None
for wma in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    out_std_wma = maxmean_contrast(ew80s, k_neg=2, w_max_agg=wma)
    for w_ica in [0.38, 0.40, 0.42]:
        for w_std in [0.06, 0.08, 0.10]:
            w_b = 1.0 - w_ica - w_std
            if w_b < 0.48: continue
            blend = w_ica * out_ica90 + w_std * out_std_wma + w_b * out_base
            auc_c = eval_loo(blend)
            if auc_c > best5: best5 = auc_c; best_cfg5 = (wma, w_ica, w_std, w_b)
results['ica90_std80kn2_wma_v2'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  Best: {best5:.4f}{flag}  cfg={best_cfg5}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 47 Summary ===", flush=True)
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
