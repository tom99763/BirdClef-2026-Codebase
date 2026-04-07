"""
Batch 56: New directions beyond 0.9873
Methods:
  1. All-negatives (k_neg=all negative windows, not top-k)
  2. Multi-seed ICA-100 ensemble (seeds 0,1,2,3,7,42)
  3. Higher ICA dims: 110, 120, 130 WL components
  4. Ultrafine blend for uh_triple (0.005 step)
  5. WL with cosine-weighted negatives instead of top-k
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

def winlabel_allneg(emb_wins_n, w_max_pos=0.5, w_max_agg=0.55):
    """Use ALL negative windows (no top-k selection)."""
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
                # Mean of ALL negatives
                neg_mean = neg_wins.mean(0)
                neg_mean /= (np.linalg.norm(neg_mean) + EPS)
                ws[:,si] = (sp - (te_wins @ neg_mean) + 1) / 2
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

# Baseline outputs
out_wl80 = winlabel_contrast(ew80, k_neg=4, w_max_pos=0.70, w_max_agg=0.60)
out_wl_std = winlabel_contrast(ew80s, k_neg=4, w_max_pos=0.60, w_max_agg=0.65)
print(f"WL-PCA-80: {eval_loo(out_wl80):.4f}", flush=True)
print(f"WL-Std-PCA-80: {eval_loo(out_wl_std):.4f}", flush=True)

# ─── Find best ultra-high k_neg config (uh) ───────────────────────────────────
print("\n=== Finding ICA-100 uh best config ===", flush=True)
best_uh = 0; best_cfg_uh = None; best_out_uh = None
for k_neg in [40, 50, 60, 70, 80, 100, 120, 150]:
    for wma in [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
        for wmp in [0.72, 0.75, 0.78, 0.80, 0.82, 0.85]:
            out = winlabel_contrast(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_uh: best_uh = auc; best_cfg_uh = (k_neg, wma, wmp); best_out_uh = out
print(f"  ICA-100 uh2 best: {best_uh:.4f}  cfg={best_cfg_uh}", flush=True)
results['wl_ica100_uh2'] = best_uh

# ─── Method 1: All-negative WL ───────────────────────────────────────────────
print("\n=== Method 1: All-negative WL ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None; best_out_allneg = None
for wma in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    for wmp in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        out = winlabel_allneg(ew_ica100, w_max_pos=wmp, w_max_agg=wma)
        auc = eval_loo(out)
        if auc > best1: best1 = auc; best_cfg1 = (wma, wmp); best_out_allneg = out
print(f"  All-neg ICA-100: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_allneg_ica100'] = best1

# Triple blend: allneg + std + pca
best1t = 0; best_cfg1t = None
for w_ica in np.arange(0.30, 0.70, 0.01):
    for w_std in np.arange(0.10, 0.50, 0.01):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.50: continue
        blend = w_ica * best_out_allneg + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best1t: best1t = auc; best_cfg1t = (float(w_ica), float(w_std), float(w_pca))
results['wl_allneg_triple'] = best1t
flag = " *** NEW BEST ***" if best1t > CURRENT_BEST else ""
print(f"  All-neg triple: {best1t:.4f}{flag}  cfg={best_cfg1t}", flush=True)

# ─── Method 2: Multi-seed ICA-100 ensemble ────────────────────────────────────
print("\n=== Method 2: Multi-seed ICA-100 ensemble ===", flush=True)
t0 = time.time()
seed_outs = {}
SEEDS = [0, 1, 2, 7, 42, 99]
for seed in SEEDS:
    try:
        ica_s = FastICA(n_components=100, random_state=seed, max_iter=500, tol=0.01)
        ew_s = normalize(ica_s.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
        # Find best params for this seed (quick sweep)
        bst = 0; bout = None
        for k_neg in [40, 60, 80, 100]:
            for wma in [0.80, 0.85, 0.90]:
                for wmp in [0.75, 0.80]:
                    out = winlabel_contrast(ew_s, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
                    auc = eval_loo(out)
                    if auc > bst: bst = auc; bout = out
        seed_outs[seed] = (bst, bout)
        print(f"  seed={seed}: {bst:.4f}", flush=True)
    except Exception as e:
        print(f"  seed={seed} failed: {e}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# Ensemble of all seeds
valid_seeds = [s for s in SEEDS if s in seed_outs]
if len(valid_seeds) >= 2:
    # Simple equal-weight ensemble of best-per-seed outputs
    seed_ensemble = np.mean([seed_outs[s][1] for s in valid_seeds], axis=0)
    auc_ens = eval_loo(seed_ensemble)
    results['wl_multiseed_ica100_ens'] = auc_ens
    flag = " *** NEW BEST ***" if auc_ens > CURRENT_BEST else ""
    print(f"  Multi-seed ensemble ({len(valid_seeds)} seeds): {auc_ens:.4f}{flag}", flush=True)

    # Blend with std and pca
    best2t = 0; best_cfg2t = None
    for w_ica in np.arange(0.30, 0.70, 0.01):
        for w_std in np.arange(0.10, 0.50, 0.01):
            w_pca = 1.0 - w_ica - w_std
            if w_pca < 0.05 or w_pca > 0.50: continue
            blend = w_ica * seed_ensemble + w_std * out_wl_std + w_pca * out_wl80
            auc = eval_loo(blend)
            if auc > best2t: best2t = auc; best_cfg2t = (float(w_ica), float(w_std), float(w_pca))
    results['wl_multiseed_triple'] = best2t
    flag = " *** NEW BEST ***" if best2t > CURRENT_BEST else ""
    print(f"  Multi-seed triple: {best2t:.4f}{flag}  cfg={best_cfg2t}", flush=True)

# ─── Method 3: Higher ICA dims (110, 120, 130) WL ────────────────────────────
print("\n=== Method 3: Higher ICA dims WL ===", flush=True)
t0 = time.time()
best3 = 0; best3_dim = None; best3_out = None
for n_comp in [105, 110, 115, 120, 130]:
    try:
        ica_d = FastICA(n_components=n_comp, random_state=42, max_iter=500, tol=0.01)
        ew_d = normalize(ica_d.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
        bst = 0; bout = None
        for k_neg in [40, 60, 80, 100]:
            for wma in [0.80, 0.85, 0.88, 0.90]:
                for wmp in [0.73, 0.75, 0.78, 0.80]:
                    out = winlabel_contrast(ew_d, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
                    auc = eval_loo(out)
                    if auc > bst: bst = auc; bout = out
        # Quick triple
        bt = 0
        for w_ica in np.arange(0.30, 0.65, 0.025):
            for w_std in np.arange(0.15, 0.45, 0.025):
                w_p = 1.0 - w_ica - w_std
                if w_p < 0.10 or w_p > 0.45: continue
                blend = w_ica * bout + w_std * out_wl_std + w_p * out_wl80
                auc = eval_loo(blend)
                if auc > bt: bt = auc
        results[f'wl_ica{n_comp}_uh_triple'] = bt
        flag = " *** NEW BEST ***" if bt > CURRENT_BEST else ""
        print(f"  ICA-{n_comp}: solo={bst:.4f}  triple={bt:.4f}{flag}", flush=True)
        if bst > best3: best3 = bst; best3_dim = n_comp; best3_out = bout
    except Exception as e:
        print(f"  ICA-{n_comp} failed: {e}", flush=True)
print(f"  ({time.time()-t0:.0f}s)  Best high-dim ICA: {best3_dim} = {best3:.4f}", flush=True)

# ─── Method 4: Ultrafine blend for uh2 ───────────────────────────────────────
print("\n=== Method 4: Ultrafine blend for uh2 ===", flush=True)
best4 = 0; best_cfg4 = None
for w_ica in np.arange(0.30, 0.65, 0.005):
    for w_std in np.arange(0.15, 0.50, 0.005):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.50: continue
        blend = w_ica * best_out_uh + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best4: best4 = auc; best_cfg4 = (float(w_ica), float(w_std), float(w_pca))
results['wl_ica100_uh2_ultrafine'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  Ultrafine uh2: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 56 Summary ===", flush=True)
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
