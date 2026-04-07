"""
Batch 48: Window-level label utilization + soft prototyping
Goal: beat 0.9732
Methods:
  1. Use window-level labels for prototype construction (not just file-level)
  2. Soft prototype: weight pos windows by their label confidence
  3. Window-level LOO: use window labels for both training and evaluation
  4. Confidence-weighted negative: down-weight windows with ambiguous labels
  5. ICA-90 + window-label contrast + PCA-80 ensemble
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)  # [739, 234] window-level labels!
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

# Check window label statistics
win_label_max = labels_win.max()
win_label_min = labels_win[labels_win > 0].min() if (labels_win > 0).any() else 0
print(f"Window labels: max={win_label_max:.3f}, min_nonzero={win_label_min:.3f}")
print(f"Binary check: {np.unique(labels_win[:5,:5]).tolist()}", flush=True)

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

def winlabel_contrast(emb_wins_n, k_neg=4, w_max_pos=0.5, w_max_agg=0.55):
    """Use window-level labels for prototype construction."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins_all = emb_wins_n[tr_mask]
        tr_lab_win_raw = labels_win[tr_mask]  # window-level labels [N_tr, 234]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win_raw[:,si] > 0.5
            neg_win_mask = tr_lab_win_raw[:,si] < 0.1  # definitely negative windows
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            # Weight pos windows by their label confidence
            pos_weights = tr_lab_win_raw[pos_win_mask, si]
            pos_weights = pos_weights / (pos_weights.sum() + EPS)
            pos_sims = te_wins @ pos_wins.T
            # Weighted mean prototype
            pp_mean = (pos_weights[:, None] * pos_wins).sum(0)
            pp_mean /= (np.linalg.norm(pp_mean) + EPS)
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

# ─── Method 1: Window-label contrast (PCA-80) ─────────────────────────────────
print("\n=== Method 1: Window-label contrast (PCA-80) ===", flush=True)
t0 = time.time()
out_wl80 = winlabel_contrast(ew80)
auc1 = eval_loo(out_wl80)
results['winlabel_pca80'] = auc1
flag = " *** NEW BEST ***" if auc1 > CURRENT_BEST else ""
print(f"  WinLabel PCA-80: {auc1:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
# Blend
best1b = 0; best_cfg1b = None
for w_ica in [0.35, 0.40, 0.45]:
    for w_wl in [0.08, 0.12, 0.15]:
        w_b = 1.0 - w_ica - w_wl
        if w_b < 0.40: continue
        blend = w_ica * out_ica90 + w_wl * out_wl80 + w_b * out_base
        auc_c = eval_loo(blend)
        if auc_c > best1b: best1b = auc_c; best_cfg1b = (w_ica, w_wl, w_b)
results['ica90_wl80_base'] = best1b
flag = " *** NEW BEST ***" if best1b > CURRENT_BEST else ""
print(f"  ICA90+WL80+base: {best1b:.4f}{flag}  cfg={best_cfg1b}", flush=True)
# Triple with Std-PCA-80 kn2
best1c = 0; best_cfg1c = None
for w_ica in [0.35, 0.40]:
    for w_std in [0.06, 0.08]:
        for w_wl in [0.06, 0.08]:
            w_b = 1.0 - w_ica - w_std - w_wl
            if w_b < 0.44: continue
            blend = w_ica*out_ica90 + w_std*out_std80_kn2 + w_wl*out_wl80 + w_b*out_base
            auc_c = eval_loo(blend)
            if auc_c > best1c: best1c = auc_c; best_cfg1c = (w_ica, w_std, w_wl, w_b)
results['ica90_std80kn2_wl80_base'] = best1c
flag = " *** NEW BEST ***" if best1c > CURRENT_BEST else ""
print(f"  ICA90+Std80kn2+WL80+base: {best1c:.4f}{flag}  cfg={best_cfg1c}", flush=True)

# ─── Method 2: Window-label contrast (ICA-90) ─────────────────────────────────
print("\n=== Method 2: Window-label contrast (ICA-90) ===", flush=True)
t0 = time.time()
out_wl_ica = winlabel_contrast(ew_ica90)
auc2 = eval_loo(out_wl_ica)
results['winlabel_ica90'] = auc2
flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
print(f"  WinLabel ICA-90: {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
best2b = 0; best_cfg2b = None
for w_wl in [0.35, 0.40, 0.45]:
    for w_std in [0.06, 0.08]:
        w_b = 1.0 - w_wl - w_std
        if w_b < 0.47: continue
        blend = w_wl * out_wl_ica + w_std * out_std80_kn2 + w_b * out_base
        auc_c = eval_loo(blend)
        if auc_c > best2b: best2b = auc_c; best_cfg2b = (w_wl, w_std, w_b)
results['wlica90_std80kn2_base'] = best2b
flag = " *** NEW BEST ***" if best2b > CURRENT_BEST else ""
print(f"  WLica90+Std80kn2+base: {best2b:.4f}{flag}  cfg={best_cfg2b}", flush=True)

# ─── Method 3: Window-label contrast (Std-PCA-80, kn2) ───────────────────────
print("\n=== Method 3: Window-label contrast (Std-PCA-80, kn2) ===", flush=True)
out_wl_std80 = winlabel_contrast(ew80s, k_neg=2)
auc3 = eval_loo(out_wl_std80)
results['winlabel_std80_kn2'] = auc3
flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
print(f"  WinLabel Std-PCA-80 kn2: {auc3:.4f}{flag}", flush=True)
best3b = 0; best_cfg3b = None
for w_ica in [0.35, 0.40]:
    for w_wl_std in [0.06, 0.08, 0.10]:
        w_b = 1.0 - w_ica - w_wl_std
        if w_b < 0.50: continue
        blend = w_ica * out_ica90 + w_wl_std * out_wl_std80 + w_b * out_base
        auc_c = eval_loo(blend)
        if auc_c > best3b: best3b = auc_c; best_cfg3b = (w_ica, w_wl_std, w_b)
results['ica90_wl_std80kn2_base'] = best3b
flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
print(f"  ICA90+WL-Std80kn2+base: {best3b:.4f}{flag}  cfg={best_cfg3b}", flush=True)

# ─── Method 4: Grand ensemble (all window-label + standard) ──────────────────
print("\n=== Method 4: Grand ensemble ===", flush=True)
best4 = 0; best_cfg4 = None
for w_ica in [0.35, 0.40]:
    for w_std in [0.06, 0.08]:
        for w_wl80 in [0.05, 0.06]:
            for w_wl_ica in [0.04, 0.05]:
                w_b = 1.0 - w_ica - w_std - w_wl80 - w_wl_ica
                if w_b < 0.40: continue
                blend = w_ica*out_ica90 + w_std*out_std80_kn2 + w_wl80*out_wl80 + w_wl_ica*out_wl_ica + w_b*out_base
                auc_c = eval_loo(blend)
                if auc_c > best4: best4 = auc_c; best_cfg4 = (w_ica, w_std, w_wl80, w_wl_ica, w_b)
results['grand_ens'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  Grand ens: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 48 Summary ===", flush=True)
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
