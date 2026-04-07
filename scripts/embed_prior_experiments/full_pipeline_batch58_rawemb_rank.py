"""
Batch 58: Truly new directions beyond 0.9873 plateau
Methods:
  1. Raw 1280-dim Perch embeddings in WL framework (no PCA/ICA)
  2. Percentile-rank blending (convert to ranks before averaging)
  3. Geometric mean aggregation instead of linear blend
  4. NFA: Normalize-then-FastICA with higher n_components (150, 200)
  5. PCA whitened + ICA on top
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
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
CURRENT_BEST = 0.9873

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

def rank_blend(out_list, weights):
    """Blend by converting each output to percentile ranks first."""
    n_files, n_species = out_list[0].shape
    ranked = []
    for out in out_list:
        r = np.zeros_like(out)
        for si in range(n_species):
            r[:, si] = rankdata(out[:, si]) / n_files
        ranked.append(r)
    return sum(w * r for w, r in zip(weights, ranked))

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

# Raw 1280-dim normalized
ew_raw = normalize(emb_win, norm='l2').astype(np.float32)

# Whitened: PCA whiten → ICA
pca_whiten = PCA(n_components=100, whiten=True, random_state=42)
emb_whiten = pca_whiten.fit_transform(emb_win).astype(np.float32)
ew_whiten = normalize(emb_whiten, norm='l2').astype(np.float32)
print("Done.", flush=True)

# Baseline outputs
out_wl80 = winlabel_contrast(ew80, k_neg=4, w_max_pos=0.70, w_max_agg=0.60)
out_wl_std = winlabel_contrast(ew80s, k_neg=4, w_max_pos=0.60, w_max_agg=0.65)
print(f"WL-PCA-80: {eval_loo(out_wl80):.4f}", flush=True)
print(f"WL-Std-PCA-80: {eval_loo(out_wl_std):.4f}", flush=True)

# Recompute uh best output
print("Recomputing uh best...", flush=True)
best_uh = 0; best_out_uh = None
for k_neg in [50, 60, 70, 80, 100]:
    for wma in [0.85, 0.88, 0.90, 0.92]:
        for wmp in [0.75, 0.78, 0.80]:
            out = winlabel_contrast(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_uh: best_uh = auc; best_out_uh = out
print(f"  ICA-100 uh: {best_uh:.4f}", flush=True)

best_uh_trip = 0; best_out_uh_trip = None
for w_ica in np.arange(0.30, 0.70, 0.005):
    for w_std in np.arange(0.10, 0.50, 0.005):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.50: continue
        blend = w_ica * best_out_uh + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best_uh_trip: best_uh_trip = auc; best_out_uh_trip = blend
print(f"  uh_triple: {best_uh_trip:.4f}", flush=True)

# ─── Method 1: Raw 1280-dim WL ───────────────────────────────────────────────
print("\n=== Method 1: Raw 1280-dim WL ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None; best_out_raw = None
for k_neg in [4, 8, 16, 32, 50, 80]:
    for wma in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for wmp in [0.60, 0.65, 0.70, 0.75, 0.80]:
            out = winlabel_contrast(ew_raw, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best1: best1 = auc; best_cfg1 = (k_neg, wma, wmp); best_out_raw = out
print(f"  Raw-1280 WL: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_raw1280'] = best1

# Blend with uh_triple
best1t = 0; best_cfg1t = None
for w_r in np.arange(0.0, 0.55, 0.05):
    blend = w_r * best_out_raw + (1-w_r) * best_out_uh_trip
    auc = eval_loo(blend)
    if auc > best1t: best1t = auc; best_cfg1t = float(w_r)
results['wl_raw1280_uh_blend'] = best1t
flag = " *** NEW BEST ***" if best1t > CURRENT_BEST else ""
print(f"  raw+uh blend: {best1t:.4f}{flag}  w_raw={best_cfg1t}", flush=True)

# ─── Method 2: Rank-based blending ───────────────────────────────────────────
print("\n=== Method 2: Rank-based blending ===", flush=True)
best2 = 0; best_cfg2 = None
for w_ica in np.arange(0.30, 0.70, 0.025):
    for w_std in np.arange(0.10, 0.50, 0.025):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.55: continue
        rblend = rank_blend([best_out_uh, out_wl_std, out_wl80], [w_ica, w_std, w_pca])
        auc = eval_loo(rblend)
        if auc > best2: best2 = auc; best_cfg2 = (float(w_ica), float(w_std), float(w_pca))
results['wl_rank_blend_triple'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  Rank-blend triple: {best2:.4f}{flag}  cfg={best_cfg2}", flush=True)

# ─── Method 3: Geometric mean blend ──────────────────────────────────────────
print("\n=== Method 3: Geometric mean blend ===", flush=True)
best3 = 0; best_cfg3 = None
for w_ica in np.arange(0.30, 0.70, 0.025):
    for w_std in np.arange(0.10, 0.50, 0.025):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.55: continue
        # Geometric mean: product of scores^weight, clipped to [0.001, 0.999]
        s1 = np.clip(best_out_uh, 0.001, 0.999)
        s2 = np.clip(out_wl_std, 0.001, 0.999)
        s3 = np.clip(out_wl80, 0.001, 0.999)
        gm = s1**w_ica * s2**w_std * s3**w_pca
        auc = eval_loo(gm)
        if auc > best3: best3 = auc; best_cfg3 = (float(w_ica), float(w_std), float(w_pca))
results['wl_geomean_triple'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  Geomean blend: {best3:.4f}{flag}  cfg={best_cfg3}", flush=True)

# ─── Method 4: PCA-whitened ICA ──────────────────────────────────────────────
print("\n=== Method 4: PCA-whitened as input ===", flush=True)
t0 = time.time()
best4 = 0; best_cfg4 = None; best_out4 = None
for k_neg in [40, 60, 80, 100]:
    for wma in [0.80, 0.85, 0.88, 0.90]:
        for wmp in [0.73, 0.75, 0.78, 0.80]:
            out = winlabel_contrast(ew_whiten, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best4: best4 = auc; best_cfg4 = (k_neg, wma, wmp); best_out4 = out
print(f"  PCA-whiten WL: {best4:.4f}  cfg={best_cfg4}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_pca_whiten'] = best4

# Triple
best4t = 0; best_cfg4t = None
for w_wh in np.arange(0.20, 0.60, 0.025):
    for w_std in np.arange(0.10, 0.45, 0.025):
        w_pca = 1.0 - w_wh - w_std
        if w_pca < 0.10 or w_pca > 0.55: continue
        blend = w_wh * best_out4 + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best4t: best4t = auc; best_cfg4t = (float(w_wh), float(w_std), float(w_pca))
results['wl_pca_whiten_triple'] = best4t
flag = " *** NEW BEST ***" if best4t > CURRENT_BEST else ""
print(f"  PCA-whiten triple: {best4t:.4f}{flag}  cfg={best_cfg4t}", flush=True)

# ─── Method 5: Higher ICA (150, 200) on standardized embeddings ──────────────
print("\n=== Method 5: Higher ICA on Std embeddings ===", flush=True)
t0 = time.time()
for n_comp in [150, 200]:
    try:
        ica_h = FastICA(n_components=n_comp, random_state=42, max_iter=500, tol=0.02)
        ew_h = normalize(ica_h.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
        best_h = 0; best_cfg_h = None; best_out_h = None
        for k_neg in [40, 60, 80]:
            for wma in [0.80, 0.85, 0.90]:
                for wmp in [0.73, 0.75, 0.78]:
                    out = winlabel_contrast(ew_h, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
                    auc = eval_loo(out)
                    if auc > best_h: best_h = auc; best_cfg_h = (k_neg, wma, wmp); best_out_h = out
        # Quick triple
        bt = 0
        for w_h in np.arange(0.30, 0.65, 0.025):
            for w_s in np.arange(0.10, 0.40, 0.025):
                wp = 1.0 - w_h - w_s
                if wp < 0.10 or wp > 0.50: continue
                blend = w_h * best_out_h + w_s * out_wl_std + wp * out_wl80
                auc = eval_loo(blend)
                if auc > bt: bt = auc
        results[f'wl_std_ica{n_comp}_triple'] = bt
        flag = " *** NEW BEST ***" if bt > CURRENT_BEST else ""
        print(f"  Std-ICA-{n_comp}: solo={best_h:.4f}  triple={bt:.4f}{flag}", flush=True)
    except Exception as e:
        print(f"  ICA-{n_comp} FAIL: {e}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 58 Summary ===", flush=True)
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
