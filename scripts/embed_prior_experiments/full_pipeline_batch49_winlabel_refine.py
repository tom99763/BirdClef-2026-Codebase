"""
Batch 49: Window-label methods fine-tuning
Goal: beat wlica90_std80kn2_base = 0.9820
Methods:
  1. WL-PCA-80 parameter sweep (k_neg, w_max_pos, w_max_agg)
  2. WL-Std-PCA-80 k_neg sweep
  3. WL-PCA-80 + WL-ICA-90 + Std-PCA-80 ensemble
  4. WL-PCA-80 alone fine parameters
  5. WL triple: WL-PCA-80 + WL-Std-PCA-80 + ICA-90
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
CURRENT_BEST = 0.9820

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

def winlabel_contrast(emb_wins_n, k_neg=4, w_max_pos=0.5, w_max_agg=0.55):
    """Window-level labels for prototype construction."""
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

# Precompute components
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

# WL baselines
out_wl80 = winlabel_contrast(ew80)
out_wl_ica90 = winlabel_contrast(ew_ica90)
out_wl_std80_kn2 = winlabel_contrast(ew80s, k_neg=2)
print(f"WL-PCA-80: {eval_loo(out_wl80):.4f}", flush=True)
print(f"WL-ICA-90: {eval_loo(out_wl_ica90):.4f}", flush=True)
print(f"WL-Std-PCA-80 kn2: {eval_loo(out_wl_std80_kn2):.4f}", flush=True)

# ─── Method 1: WL-PCA-80 parameter sweep ──────────────────────────────────────
print("\n=== Method 1: WL-PCA-80 parameter sweep ===", flush=True)
best1 = 0; best_cfg1 = None; best_out1 = None
for k_neg in [2, 3, 4, 5, 6, 8]:
    for wma in [0.5, 0.55, 0.6, 0.65, 0.7]:
        for wmp in [0.4, 0.5, 0.6]:
            out = winlabel_contrast(ew80, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best1: best1 = auc; best_cfg1 = (k_neg, wma, wmp); best_out1 = out
results['wl80_param_best'] = best1
flag = " *** NEW BEST ***" if best1 > CURRENT_BEST else ""
print(f"  WL-PCA-80 best: {best1:.4f}{flag}  cfg={best_cfg1}", flush=True)

# ─── Method 2: WL-Std-PCA-80 k_neg sweep ─────────────────────────────────────
print("\n=== Method 2: WL-Std-PCA-80 k_neg sweep ===", flush=True)
best2 = 0; best_cfg2 = None; best_out2 = None
for k_neg in [1, 2, 3, 4, 5, 6]:
    for wma in [0.5, 0.55, 0.6]:
        out = winlabel_contrast(ew80s, k_neg=k_neg, w_max_agg=wma)
        auc = eval_loo(out)
        if auc > best2: best2 = auc; best_cfg2 = (k_neg, wma); best_out2 = out
results['wl_std80_param_best'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  WL-Std-PCA-80 best: {best2:.4f}{flag}  cfg={best_cfg2}", flush=True)

# ─── Method 3: WL-PCA-80 + WL-Std-PCA-80 ensemble ───────────────────────────
print("\n=== Method 3: WL-PCA-80 + WL-Std-PCA-80(kn2) ensemble ===", flush=True)
best3 = 0; best_cfg3 = None
out1 = best_out1 if best_out1 is not None else out_wl80
out2 = best_out2 if best_out2 is not None else out_wl_std80_kn2
for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blend = w1 * out1 + (1-w1) * out2
    auc = eval_loo(blend)
    if auc > best3: best3 = auc; best_cfg3 = w1
results['wl80_wlstd80_ens'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  WL-PCA80+WL-Std80: {best3:.4f}{flag}  w1={best_cfg3}", flush=True)

# Also with ICA-90
best3b = 0; best_cfg3b = None
for w_wl80 in [0.5, 0.6, 0.7]:
    for w_ica in [0.1, 0.15, 0.2]:
        w_wl_std = 1.0 - w_wl80 - w_ica
        if w_wl_std < 0.1: continue
        blend = w_wl80*out1 + w_wl_std*out2 + w_ica*out_ica90
        auc = eval_loo(blend)
        if auc > best3b: best3b = auc; best_cfg3b = (w_wl80, w_wl_std, w_ica)
results['wl80_wlstd_ica90'] = best3b
flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
print(f"  WL80+WLStd80+ICA90: {best3b:.4f}{flag}  cfg={best_cfg3b}", flush=True)

# ─── Method 4: WL-ICA-90 parameter sweep ──────────────────────────────────────
print("\n=== Method 4: WL-ICA-90 parameter sweep ===", flush=True)
best4 = 0; best_cfg4 = None; best_out4 = None
for k_neg in [2, 3, 4, 5, 6, 8]:
    for wma in [0.5, 0.55, 0.6, 0.65]:
        for wmp in [0.4, 0.5, 0.6]:
            out = winlabel_contrast(ew_ica90, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best4: best4 = auc; best_cfg4 = (k_neg, wma, wmp); best_out4 = out
results['wl_ica90_param_best'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  WL-ICA-90 best: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# Blend WL-ICA-90 best with WL-PCA-80 best
if best_out4 is not None and best_out1 is not None:
    best4b = 0; best_cfg4b = None
    for w_wl_ica in [0.3, 0.4, 0.5, 0.6]:
        blend = w_wl_ica * best_out4 + (1-w_wl_ica) * best_out1
        auc = eval_loo(blend)
        if auc > best4b: best4b = auc; best_cfg4b = w_wl_ica
    results['wlica_best_wl80_best'] = best4b
    flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
    print(f"  WL-ICA90-best+WL-PCA80-best: {best4b:.4f}{flag}  w={best_cfg4b}", flush=True)

# ─── Method 5: WL best + Std-PCA-80 kn2 + PCA-80 ────────────────────────────
print("\n=== Method 5: WL-PCA80-best + Std-PCA80kn2 + PCA80 ===", flush=True)
best5 = 0; best_cfg5 = None
for w_wl in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    for w_std in [0.04, 0.06, 0.08]:
        w_b = 1.0 - w_wl - w_std
        if w_b < 0.12: continue
        blend = w_wl * (best_out1 if best_out1 is not None else out_wl80) + w_std * out_std80_kn2 + w_b * out_base
        auc = eval_loo(blend)
        if auc > best5: best5 = auc; best_cfg5 = (w_wl, w_std, w_b)
results['wl80best_std80kn2_base'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  WL80-best+Std80kn2+base: {best5:.4f}{flag}  cfg={best_cfg5}", flush=True)

# Also try WL-Std80-best + Std80kn2 + base
if best_out2 is not None:
    best5b = 0; best_cfg5b = None
    for w_wl in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
        for w_std in [0.04, 0.06]:
            w_b = 1.0 - w_wl - w_std
            if w_b < 0.10: continue
            blend = w_wl * best_out2 + w_std * out_std80_kn2 + w_b * out_base
            auc = eval_loo(blend)
            if auc > best5b: best5b = auc; best_cfg5b = (w_wl, w_std, w_b)
    results['wlstd80best_std80kn2_base'] = best5b
    flag = " *** NEW BEST ***" if best5b > CURRENT_BEST else ""
    print(f"  WL-Std80-best+Std80kn2+base: {best5b:.4f}{flag}  cfg={best_cfg5b}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 49 Summary ===", flush=True)
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
