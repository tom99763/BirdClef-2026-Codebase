"""
Batch 35: Max-aggregation extensions
Goal: beat agg_max_mean_blend = 0.9698
Methods:
  1. Fine-sweep max/mean blend weights
  2. p90+mean blend (fine weights)
  3. Max aggregation + different k_neg values
  4. Triple blend: max + p90 + mean
  5. PCA-80 + max agg with different PCA dims
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
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

# PCA-80
pca80 = PCA(n_components=80, random_state=42)
emb_win_pca = pca80.fit_transform(emb_win).astype(np.float32)
emb_win_pca_norm = normalize(emb_win_pca, norm='l2').astype(np.float32)
EPS = 1e-7; mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9698

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def compute_agg_scores(emb_wins_n, k_neg=5, w_max_pos=0.5):
    """Compute per-window scores (all files), return raw ws arrays."""
    out_mean = np.zeros((n_files, n_species), np.float32)
    out_max  = np.zeros((n_files, n_species), np.float32)
    out_p90  = np.zeros((n_files, n_species), np.float32)
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
        out_mean[fi] = ws.mean(0)
        out_max[fi]  = ws.max(0)
        out_p90[fi]  = np.percentile(ws, 90, axis=0)
    return out_mean, out_max, out_p90

# Base: PCA-80, k_neg=5, w_max_pos=0.5
out_mean, out_max, out_p90 = compute_agg_scores(emb_win_pca_norm)
print(f"Base: mean={eval_loo(out_mean):.4f}, max={eval_loo(out_max):.4f}, p90={eval_loo(out_p90):.4f}", flush=True)

# ─── Method 1: Fine-sweep max/mean blend ─────────────────────────────────────
print("\n=== Method 1: Fine-sweep max/mean blend ===", flush=True)
best1 = 0; best_w1 = None; best_out1 = None
for w_max in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
    blend = w_max * out_max + (1-w_max) * out_mean
    auc = eval_loo(blend)
    results[f'max_mean_w{int(w_max*100)}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  w_max={w_max}: {auc:.4f}{flag}", flush=True)
    if auc > best1: best1 = auc; best_w1 = w_max; best_out1 = blend

# ─── Method 2: p90/mean fine blend ───────────────────────────────────────────
print("\n=== Method 2: p90/mean fine blend ===", flush=True)
best2 = 0; best_w2 = None
for w_p90 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blend = w_p90 * out_p90 + (1-w_p90) * out_mean
    auc = eval_loo(blend)
    results[f'p90_mean_w{int(w_p90*10)}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  w_p90={w_p90}: {auc:.4f}{flag}", flush=True)
    if auc > best2: best2 = auc; best_w2 = w_p90

# ─── Method 3: Triple blend max+p90+mean ─────────────────────────────────────
print("\n=== Method 3: Triple blend (max+p90+mean) ===", flush=True)
best3 = 0; best_cfg3 = None
for w_max in [0.3, 0.4, 0.5]:
    for w_p90 in [0.1, 0.2, 0.3]:
        w_mean = 1.0 - w_max - w_p90
        if w_mean < 0.1: continue
        blend = w_max * out_max + w_p90 * out_p90 + w_mean * out_mean
        auc = eval_loo(blend)
        if auc > best3: best3 = auc; best_cfg3 = (w_max, w_p90, w_mean)
results['triple_max_p90_mean'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  triple: {best3:.4f}{flag}  cfg={best_cfg3}", flush=True)

# ─── Method 4: Max agg with different k_neg ──────────────────────────────────
print("\n=== Method 4: Max agg with different k_neg ===", flush=True)
t0 = time.time()
best4 = 0; best_kneg4 = None
for k_neg in [3, 4, 5, 6, 7, 8]:
    m, mx, p9 = compute_agg_scores(emb_win_pca_norm, k_neg=k_neg)
    # Blend max+mean at best w_max_blend
    blend = best_w1 * mx + (1-best_w1) * m
    auc = eval_loo(blend)
    results[f'maxmean_kn{k_neg}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  k_neg={k_neg}: {auc:.4f}{flag}", flush=True)
    if auc > best4: best4 = auc; best_kneg4 = k_neg
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 5: Different PCA dims with max+mean ──────────────────────────────
print("\n=== Method 5: PCA dim sweep with max+mean agg ===", flush=True)
t0 = time.time()
for n_comp in [64, 96, 112, 128]:
    pca = PCA(n_components=n_comp, random_state=42)
    ew_pca = pca.fit_transform(emb_win).astype(np.float32)
    ew_pca_n = normalize(ew_pca, norm='l2').astype(np.float32)
    m, mx, p9 = compute_agg_scores(ew_pca_n)
    blend = 0.5 * mx + 0.5 * m
    auc = eval_loo(blend)
    results[f'pca{n_comp}_maxmean'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  PCA-{n_comp}: {auc:.4f}{flag}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 35 Summary ===", flush=True)
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
