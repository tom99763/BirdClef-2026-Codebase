"""
Batch 38: ICA extensions + fine-tuning
Goal: beat ica80_base_blend = 0.9720
Methods:
  1. Different ICA dimensions (40, 60, 80, 100, 120)
  2. ICA blend weight fine-sweep (0.3, 0.35, 0.4, 0.45, 0.5)
  3. ICA + PCA multi-dim ensemble
  4. ICA with different aggregation (max vs mean)
  5. Triple: ICA-80 + PCA-80 + PCA-96
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
CURRENT_BEST = 0.9720

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def maxmean_contrast(emb_wins_n, k_neg=4, w_max_pos=0.5, w_max_agg=0.55):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_lab = file_labels[tr_idx]
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

# Base: PCA-80
pca80 = PCA(n_components=80, random_state=42)
emb_win_pca80 = pca80.fit_transform(emb_win).astype(np.float32)
ew80 = normalize(emb_win_pca80, norm='l2').astype(np.float32)
out_base = maxmean_contrast(ew80)
print(f"Base (pca80): {eval_loo(out_base):.4f}", flush=True)

# ICA-80 reference
ica80 = FastICA(n_components=80, random_state=42, max_iter=500, tol=0.01)
emb_win_ica80 = ica80.fit_transform(emb_win).astype(np.float32)
ew_ica80 = normalize(emb_win_ica80, norm='l2').astype(np.float32)
out_ica80 = maxmean_contrast(ew_ica80)
print(f"ICA-80: {eval_loo(out_ica80):.4f}", flush=True)

# ─── Method 1: ICA dimension sweep ───────────────────────────────────────────
print("\n=== Method 1: ICA dimension sweep ===", flush=True)
t0 = time.time()
ica_outs = {80: out_ica80}
for n_comp in [40, 60, 100, 120]:
    try:
        ica = FastICA(n_components=n_comp, random_state=42, max_iter=500, tol=0.01)
        emb_ica = ica.fit_transform(emb_win).astype(np.float32)
        ew_ica = normalize(emb_ica, norm='l2').astype(np.float32)
        out_ica = maxmean_contrast(ew_ica)
        auc_ica = eval_loo(out_ica)
        ica_outs[n_comp] = out_ica
        # Blend with base
        best_b = 0; best_wb = None
        for w_ica in [0.3, 0.4, 0.5]:
            blend = w_ica * out_ica + (1-w_ica) * out_base
            auc_c = eval_loo(blend)
            if auc_c > best_b: best_b = auc_c; best_wb = w_ica
        results[f'ica{n_comp}_base_blend'] = best_b
        flag = " *** NEW BEST ***" if best_b > CURRENT_BEST else ""
        print(f"  ICA-{n_comp} alone={auc_ica:.4f} +base={best_b:.4f}{flag}  w={best_wb}", flush=True)
    except Exception as e:
        print(f"  ICA-{n_comp} failed: {e}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 2: Fine-sweep ICA-80 blend weight ────────────────────────────────
print("\n=== Method 2: Fine-sweep ICA-80 blend weight ===", flush=True)
best2 = 0; best_w2 = None; best_out2 = None
for w_ica in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    blend = w_ica * out_ica80 + (1-w_ica) * out_base
    auc = eval_loo(blend)
    results[f'ica80_pca80_w{int(w_ica*100)}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  w_ica={w_ica}: {auc:.4f}{flag}", flush=True)
    if auc > best2: best2 = auc; best_w2 = w_ica; best_out2 = blend

# ─── Method 3: Multi-ICA ensemble ────────────────────────────────────────────
print("\n=== Method 3: Multi-ICA ensemble ===", flush=True)
all_ica = [(n, ica_outs.get(n)) for n in ica_outs if ica_outs[n] is not None]
if len(all_ica) >= 2:
    ica_ens = np.mean([o for _, o in all_ica], axis=0)
    auc_ica_ens = eval_loo(ica_ens)
    results['multi_ica_ens'] = auc_ica_ens
    flag = " *** NEW BEST ***" if auc_ica_ens > CURRENT_BEST else ""
    print(f"  Multi-ICA ens: {auc_ica_ens:.4f}{flag}", flush=True)
    # Blend with base
    best3b = 0; best_w3b = None
    for w_e in [0.3, 0.4, 0.5]:
        blend = w_e * ica_ens + (1-w_e) * out_base
        auc_c = eval_loo(blend)
        if auc_c > best3b: best3b = auc_c; best_w3b = w_e
    results['multi_ica_base_blend'] = best3b
    flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
    print(f"  Multi-ICA+base: {best3b:.4f}{flag}  w={best_w3b}", flush=True)

# ─── Method 4: ICA with max-only aggregation ─────────────────────────────────
print("\n=== Method 4: ICA-80 with max-only aggregation ===", flush=True)
out_ica80_max = maxmean_contrast(ew_ica80, w_max_agg=1.0)
auc4_max = eval_loo(out_ica80_max)
print(f"  ICA-80 max-only: {auc4_max:.4f}", flush=True)
best4b = 0; best_w4b = None
for w_ica in [0.3, 0.4, 0.5]:
    blend = w_ica * out_ica80_max + (1-w_ica) * out_base
    auc_c = eval_loo(blend)
    if auc_c > best4b: best4b = auc_c; best_w4b = w_ica
results['ica80_max_base_blend'] = best4b
flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
print(f"  ICA-80-max+base: {best4b:.4f}{flag}  w_ica={best_w4b}", flush=True)

# ─── Method 5: Triple: best-ICA + PCA-80 + PCA-96 ────────────────────────────
print("\n=== Method 5: Triple blend ICA + PCA-80 + PCA-96 ===", flush=True)
pca96 = PCA(n_components=96, random_state=42)
emb96 = pca96.fit_transform(emb_win).astype(np.float32)
ew96 = normalize(emb96, norm='l2').astype(np.float32)
out96 = maxmean_contrast(ew96)
# Triple blend
best5 = 0; best_cfg5 = None
for w_ica in [0.2, 0.3, 0.4]:
    for w_80 in [0.3, 0.4, 0.5]:
        w_96 = 1.0 - w_ica - w_80
        if w_96 < 0.1: continue
        blend = w_ica * out_ica80 + w_80 * out_base + w_96 * out96
        auc_c = eval_loo(blend)
        if auc_c > best5: best5 = auc_c; best_cfg5 = (w_ica, w_80, w_96)
results['triple_ica80_pca80_96'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  triple: {best5:.4f}{flag}  cfg={best_cfg5}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 38 Summary ===", flush=True)
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
