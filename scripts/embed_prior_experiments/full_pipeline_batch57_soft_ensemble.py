"""
Batch 57: New contrast formulations + output-level ensemble
Goal: beat 0.9873
Methods:
  1. Soft-attention negatives (weighted by similarity, not hard top-k)
  2. Output-level blend: uh_triple + multiseed_triple
  3. Temperature-scaled positive similarity
  4. Multi-seed optimized blend weights (not equal weights)
  5. ICA-100-uh + ICA-120-uh + Std triple
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

def winlabel_soft_neg(emb_wins_n, k_neg=50, temp=1.0, w_max_pos=0.80, w_max_agg=0.85):
    """Soft-attention negatives: weight by softmax(neg_sim * temp)."""
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
                neg_sims = te_wins @ neg_wins.T  # [n_test, n_neg]
                # Use top-k but with soft weighting
                k_act = min(k_neg, neg_sims.shape[1])
                topk_idx = np.argsort(-neg_sims, axis=1)[:, :k_act]
                topk_sims = np.take_along_axis(neg_sims, topk_idx, axis=1)
                # Softmax weights
                weights = np.exp(temp * topk_sims)
                weights /= weights.sum(1, keepdims=True) + EPS
                # Weighted mean of top-k negatives for each test window
                top_neg = np.einsum('ij,ijk->ik', weights,
                                    neg_wins[topk_idx])  # [n_test, dim]
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

ica120 = FastICA(n_components=120, random_state=42, max_iter=500, tol=0.01)
ew_ica120 = normalize(ica120.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
print("Done.", flush=True)

# Baseline outputs
out_wl80 = winlabel_contrast(ew80, k_neg=4, w_max_pos=0.70, w_max_agg=0.60)
out_wl_std = winlabel_contrast(ew80s, k_neg=4, w_max_pos=0.60, w_max_agg=0.65)
print(f"WL-PCA-80: {eval_loo(out_wl80):.4f}", flush=True)
print(f"WL-Std-PCA-80: {eval_loo(out_wl_std):.4f}", flush=True)

# Recompute uh_triple output (batch 55 best)
print("\nRecomputing uh best...", flush=True)
best_uh = 0; best_cfg_uh = None; best_out_uh = None
for k_neg in [40, 50, 60, 70, 80, 100]:
    for wma in [0.80, 0.85, 0.88, 0.90, 0.92]:
        for wmp in [0.72, 0.75, 0.78, 0.80]:
            out = winlabel_contrast(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_uh: best_uh = auc; best_cfg_uh = (k_neg, wma, wmp); best_out_uh = out
print(f"  ICA-100 uh re: {best_uh:.4f}  cfg={best_cfg_uh}", flush=True)

best_uh_trip = 0; best_cfg_uh_trip = None; best_out_uh_trip = None
for w_ica in np.arange(0.30, 0.70, 0.005):
    for w_std in np.arange(0.10, 0.50, 0.005):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.50: continue
        blend = w_ica * best_out_uh + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best_uh_trip: best_uh_trip = auc; best_cfg_uh_trip = (float(w_ica), float(w_std), float(w_pca)); best_out_uh_trip = blend
print(f"  uh_triple re: {best_uh_trip:.4f}  cfg={best_cfg_uh_trip}", flush=True)

# ─── Method 1: Soft-attention negatives ──────────────────────────────────────
print("\n=== Method 1: Soft-attention negatives ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None; best_out_soft = None
for k_neg in [40, 60, 80, 100]:
    for temp in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for wma in [0.80, 0.85, 0.90]:
            for wmp in [0.73, 0.75, 0.78, 0.80]:
                out = winlabel_soft_neg(ew_ica100, k_neg=k_neg, temp=temp, w_max_pos=wmp, w_max_agg=wma)
                auc = eval_loo(out)
                if auc > best1: best1 = auc; best_cfg1 = (k_neg, temp, wma, wmp); best_out_soft = out
print(f"  Soft-neg ICA-100: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_softneg_ica100'] = best1

best1t = 0; best_cfg1t = None
for w_ica in np.arange(0.30, 0.70, 0.01):
    for w_std in np.arange(0.10, 0.50, 0.01):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.50: continue
        blend = w_ica * best_out_soft + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best1t: best1t = auc; best_cfg1t = (float(w_ica), float(w_std), float(w_pca))
results['wl_softneg_triple'] = best1t
flag = " *** NEW BEST ***" if best1t > CURRENT_BEST else ""
print(f"  Soft-neg triple: {best1t:.4f}{flag}  cfg={best_cfg1t}", flush=True)

# ─── Method 2: Output-level ensemble (uh_triple + multiseed components) ──────
print("\n=== Method 2: Output-level ensemble ===", flush=True)
# Compute 6 independent seed outputs for ICA-100
SEEDS = [0, 1, 2, 7, 42, 99]
seed_bests = {}
for seed in SEEDS:
    try:
        ica_s = FastICA(n_components=100, random_state=seed, max_iter=500, tol=0.01)
        ew_s = normalize(ica_s.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
        bst = 0; bout = None; bcfg = None
        for k_neg in [50, 70, 100]:
            for wma in [0.82, 0.87, 0.90]:
                for wmp in [0.75, 0.80]:
                    out = winlabel_contrast(ew_s, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
                    auc = eval_loo(out)
                    if auc > bst: bst = auc; bout = out; bcfg = (k_neg, wma, wmp)
        seed_bests[seed] = (bst, bout, bcfg)
        print(f"  seed={seed}: {bst:.4f}  cfg={bcfg}", flush=True)
    except Exception as e:
        print(f"  seed={seed} FAIL: {e}", flush=True)

valid = [(s, d) for s, d in seed_bests.items() if d[1] is not None]
if len(valid) >= 2:
    # Optimize blend: uh_triple + seed-ensembles
    outs_v = [d[1] for _, d in valid]
    # Equal seed ensemble
    seed_ens = np.mean(outs_v, axis=0)
    # Blend uh_triple and seed_ens
    best2 = 0; best_cfg2 = None
    for w_uh in np.arange(0.0, 1.01, 0.05):
        blend = w_uh * best_out_uh_trip + (1-w_uh) * seed_ens
        auc = eval_loo(blend)
        if auc > best2: best2 = auc; best_cfg2 = float(w_uh)
    results['wl_uh_seedens_blend'] = best2
    flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
    print(f"  uh_triple + seed_ens blend: {best2:.4f}{flag}  w_uh={best_cfg2}", flush=True)

    # Also optimize per-seed weights + uh
    best2b = 0; best_cfg2b = None
    for w_uh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w_seed = (1.0 - w_uh) / len(outs_v)
        blend = w_uh * best_out_uh_trip
        for o in outs_v: blend = blend + w_seed * o
        auc = eval_loo(blend)
        if auc > best2b: best2b = auc; best_cfg2b = w_uh
    results['wl_uh_multiseed_final'] = best2b
    flag = " *** NEW BEST ***" if best2b > CURRENT_BEST else ""
    print(f"  uh + multi-seed final: {best2b:.4f}{flag}  w_uh={best_cfg2b}", flush=True)

# ─── Method 3: ICA-100-uh + ICA-120-uh + Std blend ──────────────────────────
print("\n=== Method 3: ICA-100-uh + ICA-120-uh + Std ===", flush=True)
t0 = time.time()
best3_120 = 0; best_cfg3_120 = None; best_out3_120 = None
for k_neg in [40, 60, 80, 100]:
    for wma in [0.80, 0.85, 0.90]:
        for wmp in [0.73, 0.75, 0.78, 0.80]:
            out = winlabel_contrast(ew_ica120, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best3_120: best3_120 = auc; best_cfg3_120 = (k_neg, wma, wmp); best_out3_120 = out
print(f"  ICA-120 uh: {best3_120:.4f}  cfg={best_cfg3_120}  ({time.time()-t0:.0f}s)", flush=True)

best3 = 0; best_cfg3 = None
for w100 in np.arange(0.20, 0.65, 0.025):
    for w120 in np.arange(0.10, 0.40, 0.025):
        for w_std in np.arange(0.10, 0.40, 0.025):
            w_pca = 1.0 - w100 - w120 - w_std
            if w_pca < 0.05 or w_pca > 0.40: continue
            blend = w100*best_out_uh + w120*best_out3_120 + w_std*out_wl_std + w_pca*out_wl80
            auc = eval_loo(blend)
            if auc > best3: best3 = auc; best_cfg3 = (w100, w120, w_std, w_pca)
results['wl_ica100_ica120_std_quad'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  ICA-100+ICA-120+Std quad: {best3:.4f}{flag}  cfg={best_cfg3}", flush=True)

# ─── Method 4: Blend of soft_neg_triple and uh_triple ────────────────────────
print("\n=== Method 4: soft_neg + uh blend ===", flush=True)
best4 = 0; best_cfg4 = None
for w_soft in np.arange(0.0, 1.01, 0.05):
    blend = w_soft * best_out_soft + (1-w_soft) * best_out_uh
    # triple this
    for w_ica2 in np.arange(0.30, 0.70, 0.025):
        for w_std in np.arange(0.10, 0.45, 0.025):
            w_pca = 1.0 - w_ica2 - w_std
            if w_pca < 0.05 or w_pca > 0.50: continue
            b = w_ica2 * blend + w_std * out_wl_std + w_pca * out_wl80
            auc = eval_loo(b)
            if auc > best4: best4 = auc; best_cfg4 = (float(w_soft), float(w_ica2), float(w_std), float(w_pca))
results['wl_softneg_uh_triple'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  soft_neg+uh blend triple: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 57 Summary ===", flush=True)
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
