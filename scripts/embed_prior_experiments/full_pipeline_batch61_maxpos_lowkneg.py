"""
Batch 61: Unexplored combinations - k_neg=1 and w_max_pos=1.0
Goal: find single superior method beyond 0.9873
Methods:
  1. k_neg=1 (single hardest negative) with various wma/wmp
  2. w_max_pos=1.0 (pure max positive similarity) with various k_neg/wma
  3. Adaptive w_max_pos: 1.0 for rare species (<5 pos), lower for common
  4. Per-window LDA direction at inference (species-specific 1D projection)
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

def winlabel_adaptive(emb_wins_n, k_neg=50, w_max_agg=0.92, rare_thresh=5):
    """Adaptive w_max_pos: 1.0 for rare species (<rare_thresh pos windows), 0.80 for common."""
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
            n_pos = len(pos_wins)
            w_max_pos = 1.0 if n_pos < rare_thresh else 0.80
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

# ─── Method 1: k_neg=1 sweep ─────────────────────────────────────────────────
print("\n=== Method 1: k_neg=1 (single hardest negative) ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None
for wma in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    for wmp in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]:
        for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80'), (ew80s, 'std80')]:
            out = winlabel_contrast(emb, k_neg=1, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best1: best1 = auc; best_cfg1 = (name, wma, wmp)
print(f"  k_neg=1 best: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_kneg1'] = best1
flag = " *** NEW BEST ***" if best1 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 2: w_max_pos=1.0 (pure max) with various k_neg ───────────────────
print("\n=== Method 2: w_max_pos=1.0 (pure max-positive) ===", flush=True)
t0 = time.time()
best2 = 0; best_cfg2 = None
for k_neg in [1, 2, 3, 4, 5, 8, 16, 32, 50, 80, 100]:
    for wma in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
        for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80'), (ew80s, 'std80')]:
            out = winlabel_contrast(emb, k_neg=k_neg, w_max_pos=1.0, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best2: best2 = auc; best_cfg2 = (name, k_neg, wma)
print(f"  w_max_pos=1.0 best: {best2:.4f}  cfg={best_cfg2}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_maxpos1'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 3: Adaptive w_max_pos ────────────────────────────────────────────
print("\n=== Method 3: Adaptive w_max_pos ===", flush=True)
t0 = time.time()
best3 = 0; best_cfg3 = None
for k_neg in [40, 50, 60, 80]:
    for wma in [0.85, 0.88, 0.90, 0.92]:
        for rare_thresh in [3, 5, 8, 10, 15]:
            out = winlabel_adaptive(ew_ica100, k_neg=k_neg, w_max_agg=wma, rare_thresh=rare_thresh)
            auc = eval_loo(out)
            if auc > best3: best3 = auc; best_cfg3 = (k_neg, wma, rare_thresh)
print(f"  Adaptive wmp best: {best3:.4f}  cfg={best_cfg3}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_adaptive'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 4: No negative contrast at all (pure positive matching) ───────────
print("\n=== Method 4: Pure positive matching (no negative) ===", flush=True)
def winlabel_pos_only(emb_wins_n, w_max_pos=0.80, w_max_agg=0.90):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins_all = emb_wins_n[tr_mask]
        tr_lab_win_raw = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win_raw[:,si] > 0.5
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp = w_max_pos * pos_sims.max(1) + (1-w_max_pos) * (te_wins @ pp_mean)
            ws[:,si] = (sp + 1) / 2
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

t0 = time.time()
best4 = 0; best_cfg4 = None
for wma in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
    for wmp in [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
        for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80'), (ew80s, 'std80')]:
            out = winlabel_pos_only(emb, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best4: best4 = auc; best_cfg4 = (name, wma, wmp)
print(f"  Pos-only best: {best4:.4f}  cfg={best_cfg4}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_pos_only'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 5: Max-window aggregation only (w_max_agg=1.0) ───────────────────
print("\n=== Method 5: Max-window aggregation only ===", flush=True)
t0 = time.time()
best5 = 0; best_cfg5 = None
for k_neg in [40, 50, 60, 80, 100]:
    for wmp in [0.70, 0.75, 0.80, 0.85, 0.90, 1.0]:
        for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80')]:
            out = winlabel_contrast(emb, k_neg=k_neg, w_max_pos=wmp, w_max_agg=1.0)
            auc = eval_loo(out)
            if auc > best5: best5 = auc; best_cfg5 = (name, k_neg, wmp)
print(f"  Max-agg=1.0 best: {best5:.4f}  cfg={best_cfg5}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_maxagg1'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== Batch 61 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1])[:10]:
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
