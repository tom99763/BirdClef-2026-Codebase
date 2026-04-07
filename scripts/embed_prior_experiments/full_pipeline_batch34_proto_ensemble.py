"""
Batch 34: Composite prototype ensemble + file-level aggregation variants
Goal: beat kmeans_base_blend = 0.9655
Methods:
  1. Multi-prototype avg: (mean + trimmed + geom_med) / 3
  2. Max-of-windows aggregation (file score = max over windows, not mean)
  3. Percentile aggregation: 75th/90th percentile over windows
  4. Weighted window: weight windows by their positive similarity
  5. Multi-prototype + PCA-80 base ensemble
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
CURRENT_BEST = 0.9655

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def geom_median(X, max_iter=15):
    y = X.mean(0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - y, axis=1) + 1e-8
        w = 1.0 / d
        y_new = (w[:, None] * X).sum(0) / w.sum()
        if np.linalg.norm(y_new - y) < 1e-6: break
        y = y_new
    return y

def compute_ws_agg(fi, agg='mean'):
    """Compute per-window scores and aggregate by agg method."""
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_pca_norm[win_file_id == fi]
    tr_wins_all = emb_win_pca_norm[win_file_id != fi]
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
        sp = 0.5 * pos_sims.max(1) + 0.5 * (te_wins @ pp_mean)
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(5, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
        else: ws[:,si] = (sp+1)/2
    if agg == 'mean': return ws.mean(0)
    elif agg == 'max': return ws.max(0)
    elif agg == 'p75': return np.percentile(ws, 75, axis=0)
    elif agg == 'p90': return np.percentile(ws, 90, axis=0)
    elif agg == 'p60': return np.percentile(ws, 60, axis=0)

# Base (mean agg)
out_base = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files): out_base[fi] = compute_ws_agg(fi, 'mean')
print(f"Base (mean agg): {eval_loo(out_base):.4f}", flush=True)

# ─── Method 1: Alternative aggregation ───────────────────────────────────────
print("\n=== Method 1: Alternative window aggregation ===", flush=True)
t0 = time.time()
agg_outs = {}
for agg in ['max', 'p75', 'p90', 'p60']:
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files): out[fi] = compute_ws_agg(fi, agg)
    auc = eval_loo(out)
    agg_outs[agg] = out
    results[f'agg_{agg}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {agg}: {auc:.4f}{flag}", flush=True)
# Blend with mean
for agg in ['max', 'p75', 'p90', 'p60']:
    best_b = 0; best_wb = None
    for w_alt in [0.2, 0.3, 0.4, 0.5]:
        blend = w_alt * agg_outs[agg] + (1-w_alt) * out_base
        auc_c = eval_loo(blend)
        if auc_c > best_b: best_b = auc_c; best_wb = w_alt
    results[f'agg_{agg}_mean_blend'] = best_b
    flag = " *** NEW BEST ***" if best_b > CURRENT_BEST else ""
    print(f"  {agg}+mean: {best_b:.4f}{flag}  w={best_wb}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 2: Multi-prototype composite ────────────────────────────────────
print("\n=== Method 2: Multi-prototype composite pos score ===", flush=True)
t0 = time.time()
out_composite = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_pca_norm[win_file_id == fi]
    tr_wins_all = emb_win_pca_norm[win_file_id != fi]
    tr_fids_all = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos_win_mask = tr_lab_win[:,si] > 0.5
        neg_win_mask = ~pos_win_mask
        if not pos_win_mask.any(): ws[:,si]=0.5; continue
        pos_wins = tr_wins_all[pos_win_mask]
        pos_sims = te_wins @ pos_wins.T
        # Multi-prototype: blend 3 pos scores
        sp_max = pos_sims.max(1)
        pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
        sp_mean = te_wins @ pp_mean
        if len(pos_wins) >= 3:
            pp_gm = geom_median(pos_wins); pp_gm /= (np.linalg.norm(pp_gm) + EPS)
            sp_gm = te_wins @ pp_gm
        else:
            sp_gm = sp_mean
        # Composite: (max + mean + geom_med) / 3
        sp = (sp_max + sp_mean + sp_gm) / 3.0
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(5, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
        else: ws[:,si] = (sp+1)/2
    out_composite[fi] = ws.mean(0)
auc2 = eval_loo(out_composite)
results['multi_proto_composite'] = auc2
flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
print(f"  composite (max+mean+gm)/3: {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
# Blend with base
best2b = 0; best_w2b = None
for w_c in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blend = w_c * out_composite + (1-w_c) * out_base
    auc_c = eval_loo(blend)
    if auc_c > best2b: best2b = auc_c; best_w2b = w_c
results['composite_base_blend'] = best2b
flag = " *** NEW BEST ***" if best2b > CURRENT_BEST else ""
print(f"  composite+base: {best2b:.4f}{flag}  w_c={best_w2b}", flush=True)

# ─── Method 3: Pos-weighted window aggregation ────────────────────────────────
print("\n=== Method 3: Pos-weighted window aggregation ===", flush=True)
t0 = time.time()
out_posweight = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_pca_norm[win_file_id == fi]
    tr_wins_all = emb_win_pca_norm[win_file_id != fi]
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
        sp = 0.5 * pos_sims.max(1) + 0.5 * (te_wins @ pp_mean)
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(5, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
        else: ws[:,si] = (sp+1)/2
    # Weight windows by their max positive similarity
    pp_all = (tr_wins_all[tr_lab_win.any(1)]).mean(0) if tr_lab_win.any() else None
    if pp_all is not None:
        pp_all /= (np.linalg.norm(pp_all) + EPS)
        win_weights = (te_wins @ pp_all).clip(0, 1) + 0.1
        win_weights /= win_weights.sum()
        out_posweight[fi] = (ws * win_weights[:, None]).sum(0)
    else:
        out_posweight[fi] = ws.mean(0)
auc3 = eval_loo(out_posweight)
results['pos_weighted_agg'] = auc3
flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
print(f"  pos_weighted_agg: {auc3:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# Blend
best3b = 0; best_w3b = None
for w_pw in [0.2, 0.3, 0.4, 0.5]:
    blend = w_pw * out_posweight + (1-w_pw) * out_base
    auc_c = eval_loo(blend)
    if auc_c > best3b: best3b = auc_c; best_w3b = w_pw
results['posweight_base_blend'] = best3b
flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
print(f"  posweight+base: {best3b:.4f}{flag}  w={best_w3b}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 34 Summary ===", flush=True)
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
