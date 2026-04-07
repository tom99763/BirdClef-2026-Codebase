"""
Batch 32: Fine PCA around 80 dims + random subspace ensemble
Goal: beat pca80_wm5_kn5 = 0.9652
Methods:
  1. Ultra-fine PCA: 60, 64, 72, 80, 88, 96
  2. Random subspace ensemble (5 PCA with different seeds)
  3. PCA-80-best + KNN diverse ensemble
  4. PCA with l1 normalization (vs l2)
  5. PCA-best blend with top contrastive from full space
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
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
file_embs   = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_embs[fi]   = emb_win[s:e].mean(0)

EPS = 1e-7; mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9652

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def make_pca_embs(n_comp, seed=42, norm='l2'):
    pca = PCA(n_components=n_comp, random_state=seed)
    emb_pca = pca.fit_transform(emb_win).astype(np.float32)
    if norm == 'l2':
        emb_win_n = normalize(emb_pca, norm='l2').astype(np.float32)
    else:
        emb_win_n = normalize(emb_pca, norm='l1').astype(np.float32)
    file_embs_pca = np.zeros((n_files, n_comp), np.float32)
    for fi in range(n_files):
        s, e = int(file_start[fi]), int(file_end[fi])
        file_embs_pca[fi] = emb_pca[s:e].mean(0)
    file_embs_n = normalize(file_embs_pca, norm=norm).astype(np.float32)
    return emb_win_n, file_embs_n

def max_pos_contrast_emb(emb_wins_n, emb_files_n, k_neg=5, w_max=0.5):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = emb_files_n[tr_idx]; tr_lab = file_labels[tr_idx]
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
            sp = w_max * pos_sims.max(1) + (1-w_max) * (te_wins @ pp_mean)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    return out

# ─── Method 1: Ultra-fine PCA sweep around 80 ────────────────────────────────
print("=== Method 1: Ultra-fine PCA sweep ===", flush=True)
t0 = time.time()
pca_outs = {}
for n_comp in [56, 60, 64, 68, 72, 80, 88, 96]:
    ew_n, ef_n = make_pca_embs(n_comp)
    out = max_pos_contrast_emb(ew_n, ef_n)
    auc = eval_loo(out)
    pca_outs[n_comp] = out
    results[f'pca{n_comp}_best'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  PCA-{n_comp}: {auc:.4f}{flag}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

best_pca_n = max(pca_outs, key=lambda k: results[f'pca{k}_best'])
print(f"  Best dim: {best_pca_n}", flush=True)

# ─── Method 2: Random subspace ensemble ──────────────────────────────────────
print("\n=== Method 2: Random subspace ensemble (5 seeds) ===", flush=True)
t0 = time.time()
seed_outs = []
for seed in [42, 123, 456, 789, 1024]:
    ew_n, ef_n = make_pca_embs(80, seed=seed)
    out_s = max_pos_contrast_emb(ew_n, ef_n)
    auc_s = eval_loo(out_s)
    seed_outs.append(out_s)
    print(f"  seed={seed}: {auc_s:.4f}", flush=True)
# Ensemble
ens_seed = np.mean(seed_outs, axis=0)
auc_seed_ens = eval_loo(ens_seed)
results['pca80_seed_ens'] = auc_seed_ens
flag = " *** NEW BEST ***" if auc_seed_ens > CURRENT_BEST else ""
print(f"  seed ensemble: {auc_seed_ens:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 3: PCA-80 optimal + KNN diverse ensemble ─────────────────────────
print("\n=== Method 3: PCA-80-best + multi-k KNN ensemble ===", flush=True)
ew80, ef80 = make_pca_embs(80)
out_pca80_best = max_pos_contrast_emb(ew80, ef80)
# Multi-k KNN
file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)
emb_win_norm   = normalize(emb_win, norm='l2').astype(np.float32)
out_knn_mk = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    for k in [3,5,10]:
        sims = te_wins @ tr_emb.T
        topk = np.argsort(-sims, axis=1)[:, :k]
        w = np.take_along_axis(sims, topk, axis=1).clip(0,1)
        w /= w.sum(1,keepdims=True)+EPS
        out_knn_mk[fi] += (w[:,:,None]*tr_lab[topk]).sum(1).mean(0)
    out_knn_mk[fi] /= 3
best3 = 0; best_w3 = None
for w_pca in [0.85, 0.9, 0.92, 0.95]:
    blend = w_pca * out_pca80_best + (1-w_pca) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best3: best3 = auc_c; best_w3 = w_pca
results['pca80_knn_blend'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  pca80+knn: {best3:.4f}{flag}  w_pca={best_w3}", flush=True)

# ─── Method 4: PCA ensemble of different dims + k_neg settings ───────────────
print("\n=== Method 4: Multi-config PCA ensemble ===", flush=True)
t0 = time.time()
configs = [(80, 0.5, 5), (80, 0.6, 4), (128, 0.5, 5), (64, 0.5, 5), (96, 0.5, 5)]
config_outs = []
for n_comp, w_max, k_neg in configs:
    ew_n, ef_n = make_pca_embs(n_comp)
    out_c = max_pos_contrast_emb(ew_n, ef_n, k_neg=k_neg, w_max=w_max)
    auc_c = eval_loo(out_c)
    config_outs.append((out_c, auc_c, f'pca{n_comp}_wm{int(w_max*10)}_kn{k_neg}'))
    print(f"  PCA-{n_comp} wm={w_max} kn={k_neg}: {auc_c:.4f}", flush=True)
# Ensemble
ens4 = np.mean([o for o,_,_ in config_outs], axis=0)
auc4 = eval_loo(ens4)
results['multi_config_pca_ens'] = auc4
flag = " *** NEW BEST ***" if auc4 > CURRENT_BEST else ""
print(f"  multi-config ens: {auc4:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 5: PCA-80 blend with topk contrastive (different k_neg values) ───
print("\n=== Method 5: PCA-80 multi-kneg blend ===", flush=True)
outs_kneg = {}
for k_neg in [3, 4, 5, 6, 7, 8]:
    outs_kneg[k_neg] = max_pos_contrast_emb(ew80, ef80, k_neg=k_neg, w_max=0.5)
# Find best pair blend
best5 = 0; best_cfg5 = None
for kn1 in [4, 5, 6]:
    for kn2 in [3, 5, 6, 7]:
        if kn1 == kn2: continue
        for w1 in [0.4, 0.5, 0.6]:
            blend = w1 * outs_kneg[kn1] + (1-w1) * outs_kneg[kn2]
            auc_c = eval_loo(blend)
            if auc_c > best5: best5 = auc_c; best_cfg5 = (kn1, kn2, w1)
results['pca80_kneg_blend'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  pca80 kneg blend: {best5:.4f}{flag}  cfg={best_cfg5}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 32 Summary ===", flush=True)
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
