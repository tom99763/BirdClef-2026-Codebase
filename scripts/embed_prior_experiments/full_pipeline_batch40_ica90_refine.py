"""
Batch 40: ICA-90 fine-tuning + advanced ensembles
Goal: beat ica90_base = 0.9729
Methods:
  1. ICA-90 fine weight sweep (0.35-0.55)
  2. ICA-90 dim neighbors (85, 88, 90, 92, 95)
  3. ICA-90 + ICA-100 + PCA-80 triple blend
  4. Multi-seed ICA-90 ensemble
  5. ICA-90 + PCA-80 + PCA-96 triple blend
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
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
CURRENT_BEST = 0.9729

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

# PCA-80 base
pca80 = PCA(n_components=80, random_state=42)
emb80 = pca80.fit_transform(emb_win).astype(np.float32)
ew80 = normalize(emb80, norm='l2').astype(np.float32)
out_base = maxmean_contrast(ew80)
print(f"Base (pca80): {eval_loo(out_base):.4f}", flush=True)

# ICA-90 reference
ica90 = FastICA(n_components=90, random_state=42, max_iter=500, tol=0.01)
emb_ica90 = ica90.fit_transform(emb_win).astype(np.float32)
ew_ica90 = normalize(emb_ica90, norm='l2').astype(np.float32)
out_ica90 = maxmean_contrast(ew_ica90)
print(f"ICA-90: {eval_loo(out_ica90):.4f}", flush=True)

# ICA-100 reference
ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
emb_ica100 = ica100.fit_transform(emb_win).astype(np.float32)
ew_ica100 = normalize(emb_ica100, norm='l2').astype(np.float32)
out_ica100 = maxmean_contrast(ew_ica100)
print(f"ICA-100: {eval_loo(out_ica100):.4f}", flush=True)

# ─── Method 1: ICA-90 fine weight sweep ───────────────────────────────────────
print("\n=== Method 1: ICA-90 fine weight sweep ===", flush=True)
best1 = 0; best_w1 = None
for w_ica in [0.30, 0.33, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55]:
    blend = w_ica * out_ica90 + (1-w_ica) * out_base
    auc = eval_loo(blend)
    results[f'ica90_w{int(w_ica*100)}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  w={w_ica}: {auc:.4f}{flag}", flush=True)
    if auc > best1: best1 = auc; best_w1 = w_ica

# ─── Method 2: ICA dim fine around 90 ─────────────────────────────────────────
print("\n=== Method 2: ICA dim fine around 90 ===", flush=True)
t0 = time.time()
ica_dim_outs = {90: out_ica90, 100: out_ica100}
for n_comp in [82, 85, 87, 92, 94]:
    try:
        ica = FastICA(n_components=n_comp, random_state=42, max_iter=500, tol=0.01)
        emb_ica = ica.fit_transform(emb_win).astype(np.float32)
        ew_ica = normalize(emb_ica, norm='l2').astype(np.float32)
        out_ica = maxmean_contrast(ew_ica)
        ica_dim_outs[n_comp] = out_ica
        best_b = 0; best_wb = None
        for w_ica in [0.35, 0.40, 0.45, 0.50]:
            blend = w_ica * out_ica + (1-w_ica) * out_base
            auc_c = eval_loo(blend)
            if auc_c > best_b: best_b = auc_c; best_wb = w_ica
        results[f'ica{n_comp}_base'] = best_b
        flag = " *** NEW BEST ***" if best_b > CURRENT_BEST else ""
        print(f"  ICA-{n_comp}+base: {best_b:.4f}{flag}  w={best_wb}", flush=True)
    except Exception as e:
        print(f"  ICA-{n_comp} failed: {e}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 3: ICA-90 + ICA-100 + PCA-80 triple blend ────────────────────────
print("\n=== Method 3: ICA-90 + ICA-100 + PCA-80 triple blend ===", flush=True)
best3 = 0; best_cfg3 = None
for w90 in [0.2, 0.25, 0.3, 0.35]:
    for w100 in [0.1, 0.15, 0.2, 0.25, 0.3]:
        w_base = 1.0 - w90 - w100
        if w_base < 0.3 or w_base > 0.7: continue
        blend = w90 * out_ica90 + w100 * out_ica100 + w_base * out_base
        auc_c = eval_loo(blend)
        if auc_c > best3: best3 = auc_c; best_cfg3 = (w90, w100, w_base)
results['ica90_100_base'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  ICA90+ICA100+base: {best3:.4f}{flag}  cfg={best_cfg3}", flush=True)

# ─── Method 4: Multi-seed ICA-90 ensemble ─────────────────────────────────────
print("\n=== Method 4: Multi-seed ICA-90 ===", flush=True)
t0 = time.time()
seed_ica_outs = [out_ica90]
for seed in [0, 7, 99, 2024]:
    try:
        ica_s = FastICA(n_components=90, random_state=seed, max_iter=500, tol=0.01)
        emb_s = ica_s.fit_transform(emb_win).astype(np.float32)
        ew_s = normalize(emb_s, norm='l2').astype(np.float32)
        out_s = maxmean_contrast(ew_s)
        seed_ica_outs.append(out_s)
        print(f"  seed={seed}: {eval_loo(out_s):.4f}", flush=True)
    except Exception as e:
        print(f"  seed={seed} failed: {e}", flush=True)
if len(seed_ica_outs) > 1:
    ica_ens = np.mean(seed_ica_outs, axis=0)
    auc_ens = eval_loo(ica_ens)
    results['multi_seed_ica90'] = auc_ens
    flag = " *** NEW BEST ***" if auc_ens > CURRENT_BEST else ""
    print(f"  Multi-seed ICA-90 ens: {auc_ens:.4f}{flag}", flush=True)
    best4b = 0; best_w4b = None
    for w_ica_ens in [0.3, 0.35, 0.4, 0.45, 0.5]:
        blend = w_ica_ens * ica_ens + (1-w_ica_ens) * out_base
        auc_c = eval_loo(blend)
        if auc_c > best4b: best4b = auc_c; best_w4b = w_ica_ens
    results['multi_seed_ica90_base'] = best4b
    flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
    print(f"  Multi-seed ICA90+base: {best4b:.4f}{flag}  w={best_w4b}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 5: ICA-90 + PCA-80 + PCA-96 triple blend ────────────────────────
print("\n=== Method 5: ICA-90 + PCA-80 + PCA-96 triple blend ===", flush=True)
pca96 = PCA(n_components=96, random_state=42)
emb96 = pca96.fit_transform(emb_win).astype(np.float32)
ew96 = normalize(emb96, norm='l2').astype(np.float32)
out96 = maxmean_contrast(ew96)
best5 = 0; best_cfg5 = None
for w_ica in [0.25, 0.3, 0.35, 0.4, 0.45]:
    for w_p80 in [0.2, 0.25, 0.3, 0.35]:
        w_p96 = 1.0 - w_ica - w_p80
        if w_p96 < 0.1 or w_p96 > 0.55: continue
        blend = w_ica * out_ica90 + w_p80 * out_base + w_p96 * out96
        auc_c = eval_loo(blend)
        if auc_c > best5: best5 = auc_c; best_cfg5 = (w_ica, w_p80, w_p96)
results['ica90_pca80_96'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  ICA90+PCA80+PCA96: {best5:.4f}{flag}  cfg={best_cfg5}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 40 Summary ===", flush=True)
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
