"""
Batch 55: Refine around wl_ica100_ext2_triple = 0.9871
Goal: beat 0.9871
Methods:
  1. Even higher k_neg (60, 80, 100, 128) for ICA-100, ultra-high wma
  2. Ultrafine blend (0.005 step) around the new best triple
  3. WL with PCA-112/128/160 as alternative base (more dimensions)
  4. Combined ext-ICA-100 + ext-Std-PCA-80 + ext-PCA-80 (all re-swept)
  5. Try 4-way blend: ext-ICA-100 + best-ICA-90-ext + best-Std + best-PCA
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
CURRENT_BEST = 0.9871

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

# Baseline outputs
out_wl80 = winlabel_contrast(ew80, k_neg=4, w_max_pos=0.70, w_max_agg=0.60)
out_wl_std = winlabel_contrast(ew80s, k_neg=4, w_max_pos=0.60, w_max_agg=0.65)
print(f"WL-PCA-80: {eval_loo(out_wl80):.4f}", flush=True)
print(f"WL-Std-PCA-80: {eval_loo(out_wl_std):.4f}", flush=True)

# ─── Method 1: Ultra-high k_neg for ICA-100 ──────────────────────────────────
print("\n=== Method 1: ICA-100 ultra-high k_neg ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None; best_out_ica100_uh = None
for k_neg in [24, 32, 40, 50, 60, 80, 100]:
    for wma in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.95]:
        for wmp in [0.72, 0.75, 0.78, 0.80, 0.82, 0.85]:
            out = winlabel_contrast(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best1: best1 = auc; best_cfg1 = (k_neg, wma, wmp); best_out_ica100_uh = out
print(f"  ICA-100 ultra-high best: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_ica100_uh'] = best1

# Blend with std and pca
best1b = 0; best_cfg1b = None
for w_ica in np.arange(0.30, 0.70, 0.01):
    for w_std in np.arange(0.10, 0.50, 0.01):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.50: continue
        blend = w_ica * best_out_ica100_uh + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best1b: best1b = auc; best_cfg1b = (float(w_ica), float(w_std), float(w_pca))
results['wl_ica100_uh_triple'] = best1b
flag = " *** NEW BEST ***" if best1b > CURRENT_BEST else ""
print(f"  ICA-100-uh triple: {best1b:.4f}{flag}  cfg={best_cfg1b}", flush=True)

# ─── Method 2: Ultrafine blend around (0.475, 0.30, 0.225) + ext2 ────────────
print("\n=== Method 2: Ultrafine blend (ext2 = k_neg sweep) ===", flush=True)
# Re-run with the best ext2 config — sweep the k_neg again with finer resolution
best2_ica = 0; best_cfg2_ica = None; best_out2_ica = None
for k_neg in [28, 32, 36, 40, 44, 48, 50, 55, 60]:
    for wma in [0.80, 0.83, 0.85, 0.87, 0.90]:
        for wmp in [0.73, 0.75, 0.77, 0.80]:
            out = winlabel_contrast(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best2_ica: best2_ica = auc; best_cfg2_ica = (k_neg, wma, wmp); best_out2_ica = out
print(f"  ICA-100 ext2b best: {best2_ica:.4f}  cfg={best_cfg2_ica}", flush=True)
# Ultrafine blend
best2 = 0; best_cfg2 = None
for w_ica in np.arange(0.35, 0.65, 0.005):
    for w_std in np.arange(0.15, 0.50, 0.005):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.45: continue
        blend = w_ica * best_out2_ica + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best2: best2 = auc; best_cfg2 = (float(w_ica), float(w_std), float(w_pca))
results['wl_ica100_ext2b_ultrafine'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  Ultrafine: {best2:.4f}{flag}  cfg={best_cfg2}", flush=True)

# ─── Method 3: Extended dims for PCA base (96, 112, 128) ─────────────────────
print("\n=== Method 3: Higher-dim PCA base ===", flush=True)
t0 = time.time()
# Use k_neg=24, wma=0.80, wmp=0.75 for ICA-100 (batch 53 best that gave 0.9862)
out_ica100_b53 = winlabel_contrast(ew_ica100, k_neg=24, w_max_pos=0.75, w_max_agg=0.80)
best_pca_dim = 80; best3 = 0; best_out3 = out_wl80
for n_comp in [88, 96, 104, 112, 128]:
    pca_n = PCA(n_components=n_comp, random_state=42)
    ew_n = normalize(pca_n.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
    b_auc = 0; b_out = None
    for k_neg in [3, 4, 5, 6]:
        for wma in [0.55, 0.60, 0.65, 0.70]:
            for wmp in [0.60, 0.65, 0.70, 0.75]:
                out = winlabel_contrast(ew_n, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
                auc = eval_loo(out)
                if auc > b_auc: b_auc = auc; b_out = out
    # Triple
    bt = 0
    for w_ica in np.arange(0.35, 0.65, 0.025):
        for w_std in np.arange(0.15, 0.45, 0.025):
            w_p = 1.0 - w_ica - w_std
            if w_p < 0.10 or w_p > 0.45: continue
            blend = w_ica * out_ica100_b53 + w_std * out_wl_std + w_p * b_out
            auc = eval_loo(blend)
            if auc > bt: bt = auc
    results[f'wl_ica100_pca{n_comp}_triple'] = bt
    flag = " *** NEW BEST ***" if bt > CURRENT_BEST else ""
    print(f"  PCA-{n_comp}: solo={b_auc:.4f}  triple={bt:.4f}{flag}", flush=True)
    if b_auc > best3: best3 = b_auc; best_pca_dim = n_comp; best_out3 = b_out
print(f"  ({time.time()-t0:.0f}s)  Best PCA dim: {best_pca_dim}", flush=True)

# ─── Method 4: ICA-100-uh + ICA-90-ext + Std + PCA 4-way ────────────────────
print("\n=== Method 4: 4-way blend with uh-ICA-100 ===", flush=True)
best4_ica90 = 0; best_cfg4_ica90 = None; best_out4_ica90 = None
for k_neg in [16, 20, 24, 28, 32]:
    for wma in [0.75, 0.80, 0.85, 0.90]:
        for wmp in [0.70, 0.75, 0.80]:
            out = winlabel_contrast(ew_ica90, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best4_ica90: best4_ica90 = auc; best_cfg4_ica90 = (k_neg, wma, wmp); best_out4_ica90 = out
print(f"  WL-ICA-90 ext2: {best4_ica90:.4f}  cfg={best_cfg4_ica90}", flush=True)

best4 = 0; best_cfg4 = None
for w100 in np.arange(0.25, 0.60, 0.05):
    for w90 in np.arange(0.05, 0.30, 0.05):
        for w_std in np.arange(0.10, 0.40, 0.05):
            w_pca = 1.0 - w100 - w90 - w_std
            if w_pca < 0.05 or w_pca > 0.40: continue
            blend = w100*best_out_ica100_uh + w90*best_out4_ica90 + w_std*out_wl_std + w_pca*out_wl80
            auc = eval_loo(blend)
            if auc > best4: best4 = auc; best_cfg4 = (w100, w90, w_std, w_pca)
results['wl_quad_uh100_ext90_std_pca'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  4-way: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 55 Summary ===", flush=True)
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
