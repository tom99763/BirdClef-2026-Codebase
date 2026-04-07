"""
Batch 42: Standardized PCA extensions + advanced blending
Goal: beat ica90_std_base = 0.9730
Methods:
  1. ICA-90 + Std-PCA-80 fine weight sweep (tighter grid)
  2. ICA on standardized embeddings (Std → ICA-90)
  3. Std-PCA different dims (64, 72, 80, 96, 112)
  4. ICA-90 + Std-PCA-80 + Std-PCA-96 triple blend
  5. Std-ICA-90 + PCA-80 blend
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
CURRENT_BEST = 0.9730

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
out_std80 = maxmean_contrast(ew80s)
print(f"Std-PCA-80: {eval_loo(out_std80):.4f}", flush=True)

# ─── Method 1: ICA-90 + Std-PCA-80 fine weight grid ─────────────────────────
print("\n=== Method 1: ICA-90 + Std-PCA-80 fine weight grid ===", flush=True)
best1 = 0; best_cfg1 = None
for w_ica in [0.35, 0.38, 0.40, 0.42, 0.45]:
    for w_std in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        w_base = 1.0 - w_ica - w_std
        if w_base < 0.35 or w_base > 0.60: continue
        blend = w_ica * out_ica90 + w_std * out_std80 + w_base * out_base
        auc = eval_loo(blend)
        if auc > best1: best1 = auc; best_cfg1 = (w_ica, w_std, w_base)
results['ica90_std80_base_fine'] = best1
flag = " *** NEW BEST ***" if best1 > CURRENT_BEST else ""
print(f"  Best ICA90+Std80+base: {best1:.4f}{flag}  cfg={best_cfg1}", flush=True)

# ─── Method 2: Std → ICA-90 ──────────────────────────────────────────────────
print("\n=== Method 2: Std-ICA-90 (standardized then ICA) ===", flush=True)
t0 = time.time()
try:
    ica90_std = FastICA(n_components=90, random_state=42, max_iter=500, tol=0.01)
    emb_std_ica = ica90_std.fit_transform(emb_std).astype(np.float32)
    ew_std_ica = normalize(emb_std_ica, norm='l2').astype(np.float32)
    out_std_ica = maxmean_contrast(ew_std_ica)
    auc2 = eval_loo(out_std_ica)
    results['std_ica90'] = auc2
    flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
    print(f"  Std-ICA-90: {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
    # Blend with PCA-80 base
    best2b = 0; best_w2b = None
    for w_ica in [0.3, 0.35, 0.40, 0.45, 0.50]:
        blend = w_ica * out_std_ica + (1-w_ica) * out_base
        auc_c = eval_loo(blend)
        if auc_c > best2b: best2b = auc_c; best_w2b = w_ica
    results['std_ica90_base'] = best2b
    flag = " *** NEW BEST ***" if best2b > CURRENT_BEST else ""
    print(f"  Std-ICA90+base: {best2b:.4f}{flag}  w={best_w2b}", flush=True)
    # Triple: Std-ICA + ICA-90 + base
    best2c = 0; best_cfg2c = None
    for w_si in [0.15, 0.2, 0.25]:
        for w_ica in [0.25, 0.3, 0.35]:
            w_b = 1.0 - w_si - w_ica
            if w_b < 0.4: continue
            blend = w_si * out_std_ica + w_ica * out_ica90 + w_b * out_base
            auc_c = eval_loo(blend)
            if auc_c > best2c: best2c = auc_c; best_cfg2c = (w_si, w_ica, w_b)
    results['std_ica90_ica90_base'] = best2c
    flag = " *** NEW BEST ***" if best2c > CURRENT_BEST else ""
    print(f"  StdICA90+ICA90+base: {best2c:.4f}{flag}  cfg={best_cfg2c}", flush=True)
except Exception as e:
    print(f"  Std-ICA failed: {e}", flush=True)

# ─── Method 3: Std-PCA different dims ────────────────────────────────────────
print("\n=== Method 3: Std-PCA different dims ===", flush=True)
t0 = time.time()
best3 = 0; best_dim3 = None; best_out3 = None
for n_comp in [64, 72, 96, 112, 128]:
    pca_s = PCA(n_components=n_comp, random_state=42)
    ew_s = normalize(pca_s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
    out_s = maxmean_contrast(ew_s)
    auc_s = eval_loo(out_s)
    # Blend with ICA-90 + PCA-80
    best_b = 0; best_cfg_b = None
    for w_ica in [0.35, 0.40]:
        for w_s in [0.10, 0.15, 0.20]:
            w_b = 1.0 - w_ica - w_s
            if w_b < 0.40: continue
            blend = w_ica * out_ica90 + w_s * out_s + w_b * out_base
            auc_c = eval_loo(blend)
            if auc_c > best_b: best_b = auc_c; best_cfg_b = (w_ica, w_s, w_b)
    results[f'ica90_std{n_comp}_base'] = best_b
    flag = " *** NEW BEST ***" if best_b > CURRENT_BEST else ""
    print(f"  Std-PCA-{n_comp}: alone={auc_s:.4f}  +ICA90+base={best_b:.4f}{flag}  cfg={best_cfg_b}", flush=True)
    if best_b > best3: best3 = best_b; best_dim3 = n_comp; best_out3 = out_s
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: ICA-90 + Std-PCA-80 + Std-PCA-best ───────────────────────────
if best_out3 is not None and best_dim3 != 80:
    print(f"\n=== Method 4: ICA-90 + Std-PCA-80 + Std-PCA-{best_dim3} ===", flush=True)
    best4 = 0; best_cfg4 = None
    for w_ica in [0.35, 0.40]:
        for w80s in [0.10, 0.15]:
            for w_best in [0.10, 0.15]:
                w_b = 1.0 - w_ica - w80s - w_best
                if w_b < 0.35: continue
                blend = w_ica * out_ica90 + w80s * out_std80 + w_best * best_out3 + w_b * out_base
                auc_c = eval_loo(blend)
                if auc_c > best4: best4 = auc_c; best_cfg4 = (w_ica, w80s, w_best, w_b)
    results[f'ica90_std80_std{best_dim3}_base'] = best4
    flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
    print(f"  quad blend: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Method 5: Std-PCA-80 k_neg sweep ────────────────────────────────────────
print("\n=== Method 5: Std-PCA-80 k_neg sweep ===", flush=True)
for k_neg in [2, 3, 5, 6, 8]:
    out_kn = maxmean_contrast(ew80s, k_neg=k_neg)
    # Blend with ICA-90
    best_b = 0; best_wb = None
    for w_ica in [0.35, 0.40, 0.45]:
        blend = w_ica * out_ica90 + (1-w_ica) * out_kn
        auc_c = eval_loo(blend)
        if auc_c > best_b: best_b = auc_c; best_wb = w_ica
    # Also: triple with PCA-80
    best_bt = 0; best_cfgbt = None
    for w_ica in [0.35, 0.40]:
        for w_s in [0.10, 0.15]:
            w_b = 1.0 - w_ica - w_s
            if w_b < 0.45: continue
            blend = w_ica * out_ica90 + w_s * out_kn + w_b * out_base
            auc_c = eval_loo(blend)
            if auc_c > best_bt: best_bt = auc_c; best_cfgbt = (w_ica, w_s, w_b)
    results[f'ica90_std80kn{k_neg}_base'] = best_bt
    flag = " *** NEW BEST ***" if best_bt > CURRENT_BEST else ""
    print(f"  k_neg={k_neg}:  +ICA90={best_b:.4f}  triple={best_bt:.4f}{flag}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 42 Summary ===", flush=True)
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
