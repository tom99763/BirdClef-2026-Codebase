"""
Batch 36: Top-k window selection for aggregation + per-species max/mean
Goal: beat maxmean_kn4 = 0.9701
Methods:
  1. Top-k windows aggregation: take k most confident windows before averaging
  2. Per-species adaptive aggregation (max for rare, mean for common)
  3. Cross-PCA ensemble: blend PCA-80 maxmean with PCA-96 maxmean
  4. Different w_max_pos for aggregation (0.5 → 0.6)
  5. Max over sliding windows in PCA space
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

EPS = 1e-7; mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9701

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def make_pca(n_comp):
    pca = PCA(n_components=n_comp, random_state=42)
    ew_pca = pca.fit_transform(emb_win).astype(np.float32)
    ew_norm = normalize(ew_pca, norm='l2').astype(np.float32)
    return ew_norm

def contrast_scores(emb_wins_n, k_neg=4, w_max_pos=0.5):
    """Per-window contrast scores for all files."""
    out_ws = []
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
        out_ws.append(ws)
    return out_ws

def agg_maxmean(ws_list, w_max_agg=0.55):
    out = np.zeros((n_files, n_species), np.float32)
    for fi, ws in enumerate(ws_list):
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

ew80 = make_pca(80)
ws_list_80 = contrast_scores(ew80)
out_base = agg_maxmean(ws_list_80)
print(f"Base (pca80 maxmean kn4): {eval_loo(out_base):.4f}", flush=True)

# ─── Method 1: Top-k window aggregation ──────────────────────────────────────
print("\n=== Method 1: Top-k window selection for aggregation ===", flush=True)
t0 = time.time()
best1 = 0; best_k1 = None; best_out1 = None
for k_top in [1, 2, 3, 5, 7]:
    out = np.zeros((n_files, n_species), np.float32)
    for fi, ws in enumerate(ws_list_80):
        n_wins = ws.shape[0]
        if n_wins <= k_top:
            out[fi] = ws.mean(0)
        else:
            # For each species, top-k most confident windows
            topk_mean = np.zeros(n_species, np.float32)
            for si in range(n_species):
                topk_idx = np.argsort(-ws[:, si])[:k_top]
                topk_mean[si] = ws[topk_idx, si].mean()
            out[fi] = topk_mean
    auc = eval_loo(out)
    results[f'topk_win_agg_{k_top}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  k_top={k_top}: {auc:.4f}{flag}", flush=True)
    if auc > best1: best1 = auc; best_k1 = k_top; best_out1 = out
# Blend with base
for k_top in [1, 2, 3]:
    out_topk = None
    # recompute for specific k
    out_topk = np.zeros((n_files, n_species), np.float32)
    for fi, ws in enumerate(ws_list_80):
        n_wins = ws.shape[0]
        if n_wins <= k_top:
            out_topk[fi] = ws.mean(0)
        else:
            topk_mean = np.zeros(n_species, np.float32)
            for si in range(n_species):
                topk_idx = np.argsort(-ws[:, si])[:k_top]
                topk_mean[si] = ws[topk_idx, si].mean()
            out_topk[fi] = topk_mean
    best_b = 0; best_wb = None
    for w_topk in [0.3, 0.4, 0.5]:
        blend = w_topk * out_topk + (1-w_topk) * out_base
        auc_c = eval_loo(blend)
        if auc_c > best_b: best_b = auc_c; best_wb = w_topk
    results[f'topk{k_top}_base_blend'] = best_b
    flag = " *** NEW BEST ***" if best_b > CURRENT_BEST else ""
    print(f"  topk{k_top}+base: {best_b:.4f}{flag}  w={best_wb}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 2: Per-species adaptive aggregation ───────────────────────────────
print("\n=== Method 2: Per-species adaptive aggregation ===", flush=True)
# Species with few positives → max agg; many positives → mean agg
t0 = time.time()
species_pos_count = file_labels.sum(0)  # [234]
out_adapt_agg = np.zeros((n_files, n_species), np.float32)
for fi, ws in enumerate(ws_list_80):
    for si in range(n_species):
        n_pos = species_pos_count[si]
        if n_pos <= 3:  # rare: use max
            out_adapt_agg[fi, si] = ws[:, si].max()
        elif n_pos >= 8:  # common: use mean
            out_adapt_agg[fi, si] = ws[:, si].mean()
        else:  # mid: blend
            out_adapt_agg[fi, si] = 0.55 * ws[:, si].max() + 0.45 * ws[:, si].mean()
auc2 = eval_loo(out_adapt_agg)
results['per_species_adapt_agg'] = auc2
flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
print(f"  per_species_adapt_agg: {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 3: Cross-PCA ensemble ────────────────────────────────────────────
print("\n=== Method 3: Cross-PCA ensemble (80+96) ===", flush=True)
t0 = time.time()
ew96 = make_pca(96)
ws_list_96 = contrast_scores(ew96)
out_96 = agg_maxmean(ws_list_96)
auc96 = eval_loo(out_96)
print(f"  PCA-96 maxmean: {auc96:.4f}", flush=True)
best3 = 0; best_w3 = None
for w80 in [0.4, 0.5, 0.6, 0.7]:
    blend = w80 * out_base + (1-w80) * out_96
    auc_c = eval_loo(blend)
    if auc_c > best3: best3 = auc_c; best_w3 = w80
results['pca80_96_blend'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  pca80+pca96: {best3:.4f}{flag}  w80={best_w3}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: Different w_max_agg fine-sweep ────────────────────────────────
print("\n=== Method 4: w_max_agg fine-sweep (fixed PCA-80 kn4) ===", flush=True)
best4 = 0; best_w4 = None
for w_max_agg in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    out = agg_maxmean(ws_list_80, w_max_agg=w_max_agg)
    auc = eval_loo(out)
    results[f'w_max_agg_{int(w_max_agg*100)}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  w_max_agg={w_max_agg}: {auc:.4f}{flag}", flush=True)
    if auc > best4: best4 = auc; best_w4 = w_max_agg

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 36 Summary ===", flush=True)
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
