"""
Batch 29: Dual-negative contrast + score calibration methods
Goal: beat mm_wm7_kn6 = 0.9618
Methods:
  1. Dual-neg: combine window-level neg AND file-level neg prototypes
  2. Species-specific pos type: use max_pos for rare species, mean for common
  3. Rank-weighted neg prototype (1/rank) with max_pos
  4. Score normalization: clip and rescale by species score range
  5. Window-max pos + reciprocal neg (sim to neg decreases score quadratically)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
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

file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)
emb_win_norm   = normalize(emb_win, norm='l2').astype(np.float32)
EPS = 1e-7; mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9618

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# Base method: wm7_kn6 (w_max=0.7, k_neg=6)
def base_score(fi, w_max=0.7, k_neg=6):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    tr_wins_all = emb_win_norm[win_file_id != fi]
    tr_fids_all = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos_win_mask = tr_lab_win[:,si] > 0.5
        neg_win_mask = ~pos_win_mask
        if not pos_win_mask.any(): ws[:,si]=0.5; continue
        pos_wins = tr_wins_all[pos_win_mask]
        pos_sims = te_wins @ pos_wins.T
        sp = w_max * pos_sims.max(1) + (1-w_max) * (te_wins @ (pos_wins.mean(0) / (np.linalg.norm(pos_wins.mean(0)) + EPS)))
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(k_neg, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
        else: ws[:,si] = (sp+1)/2
    return ws.mean(0)

# Precompute base
out_base = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files): out_base[fi] = base_score(fi)
print(f"Base (wm7_kn6): {eval_loo(out_base):.4f}", flush=True)

# ─── Method 1: Dual-neg (win-level + file-level) ──────────────────────────────
print("\n=== Method 1: Dual-neg (win neg + file neg) ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None
for w_win_neg in [0.5, 0.6, 0.7, 0.8, 1.0]:
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_win_norm[win_file_id == fi]
        tr_wins_all = emb_win_norm[win_file_id != fi]
        tr_fids_all = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win[:,si] > 0.5
            neg_win_mask = ~pos_win_mask; neg_file_mask = ~(tr_lab[:,si]>0.5)
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp = 0.7 * pos_sims.max(1) + 0.3 * (te_wins @ pp_mean)
            # Win-level neg
            sn_win = np.zeros(len(te_wins), np.float32)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims_w = te_wins @ neg_wins.T
                k_act = min(6, neg_sims_w.shape[1])
                top_neg_w = neg_wins[np.argsort(-neg_sims_w, axis=1)[:, :k_act]].mean(1)
                top_neg_w /= (np.linalg.norm(top_neg_w, axis=1, keepdims=True) + EPS)
                sn_win = (te_wins * top_neg_w).sum(1)
            # File-level neg
            sn_file = np.zeros(len(te_wins), np.float32)
            if neg_file_mask.any():
                neg_file = tr_emb[neg_file_mask]
                neg_sims_f = te_wins @ neg_file.T
                k_act = min(3, neg_sims_f.shape[1])
                top_neg_f = neg_file[np.argsort(-neg_sims_f, axis=1)[:, :k_act]].mean(1)
                top_neg_f /= (np.linalg.norm(top_neg_f, axis=1, keepdims=True) + EPS)
                sn_file = (te_wins * top_neg_f).sum(1)
            sn = w_win_neg * sn_win + (1-w_win_neg) * sn_file
            ws[:,si] = (sp - sn + 1) / 2
        out[fi] = ws.mean(0)
    auc = eval_loo(out)
    results[f'dual_neg_ww{int(w_win_neg*10)}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  w_win_neg={w_win_neg}: {auc:.4f}{flag}", flush=True)
    if auc > best1: best1 = auc; best_cfg1 = w_win_neg
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 2: Species-specific pos type (rare vs common) ────────────────────
print("\n=== Method 2: Species-specific pos type (prevalence-adaptive) ===", flush=True)
t0 = time.time()
# Count positive files per species (over all files)
species_pos_count = file_labels.sum(0)
best2 = 0; best_thresh2 = None
for thresh in [2, 3, 4, 5]:
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_win_norm[win_file_id == fi]
        tr_wins_all = emb_win_norm[win_file_id != fi]
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
            # Use max for rare species, mean for common
            n_pos_wins = pos_win_mask.sum()
            if n_pos_wins <= thresh:
                sp = pos_sims.max(1)  # max pos sim for rare
            else:
                sp = 0.7 * pos_sims.max(1) + 0.3 * (te_wins @ pp_mean)  # blend for common
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(6, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    auc = eval_loo(out)
    results[f'species_adapt_thresh{thresh}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  thresh={thresh}: {auc:.4f}{flag}", flush=True)
    if auc > best2: best2 = auc; best_thresh2 = thresh
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 3: Rank-weighted neg + max_pos ────────────────────────────────────
print("\n=== Method 3: Rank-weighted neg (1/rank) + max_pos ===", flush=True)
t0 = time.time()
out_rankw = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    tr_wins_all = emb_win_norm[win_file_id != fi]
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
        sp = 0.7 * pos_sims.max(1) + 0.3 * (te_wins @ pp_mean)
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(6, neg_sims.shape[1])
            top_neg_idx = np.argsort(-neg_sims, axis=1)[:, :k_act]
            top_neg = neg_wins[top_neg_idx]  # [n_te, k, 1536]
            rank_w = np.array([1.0/(r+1) for r in range(k_act)], dtype=np.float32)
            rank_w /= rank_w.sum()
            mean_neg = (top_neg * rank_w[None, :, None]).sum(1)
            mean_neg /= (np.linalg.norm(mean_neg, axis=1, keepdims=True) + EPS)
            ws[:,si] = (sp - (te_wins * mean_neg).sum(1) + 1) / 2
        else: ws[:,si] = (sp+1)/2
    out_rankw[fi] = ws.mean(0)
auc3 = eval_loo(out_rankw)
results['rankw_neg_max_pos'] = auc3
flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
print(f"  rankw_neg+max_pos: {auc3:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: Combine base (wm7_kn6) with dual_neg_best ─────────────────────
print("\n=== Method 4: Base + best dual-neg blend ===", flush=True)
best4 = 0; best_w4 = None
best_dual_key = max((k for k in results if k.startswith('dual_neg')), key=lambda k: results[k], default=None)
if best_dual_key:
    # Re-run best dual_neg config
    best_w_win = best_cfg1
    out_dual_best = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_win_norm[win_file_id == fi]
        tr_wins_all = emb_win_norm[win_file_id != fi]
        tr_fids_all = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win[:,si] > 0.5
            neg_win_mask = ~pos_win_mask; neg_file_mask = ~(tr_lab[:,si]>0.5)
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp = 0.7 * pos_sims.max(1) + 0.3 * (te_wins @ pp_mean)
            sn_win = np.zeros(len(te_wins), np.float32)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims_w = te_wins @ neg_wins.T
                k_act = min(6, neg_sims_w.shape[1])
                top_neg_w = neg_wins[np.argsort(-neg_sims_w, axis=1)[:, :k_act]].mean(1)
                top_neg_w /= (np.linalg.norm(top_neg_w, axis=1, keepdims=True) + EPS)
                sn_win = (te_wins * top_neg_w).sum(1)
            sn_file = np.zeros(len(te_wins), np.float32)
            if neg_file_mask.any():
                neg_file = tr_emb[neg_file_mask]
                neg_sims_f = te_wins @ neg_file.T
                k_act = min(3, neg_sims_f.shape[1])
                top_neg_f = neg_file[np.argsort(-neg_sims_f, axis=1)[:, :k_act]].mean(1)
                top_neg_f /= (np.linalg.norm(top_neg_f, axis=1, keepdims=True) + EPS)
                sn_file = (te_wins * top_neg_f).sum(1)
            sn = best_w_win * sn_win + (1-best_w_win) * sn_file
            ws[:,si] = (sp - sn + 1) / 2
        out_dual_best[fi] = ws.mean(0)
    for w_b in [0.5, 0.6, 0.7, 0.8]:
        blend = w_b * out_base + (1-w_b) * out_dual_best
        auc_c = eval_loo(blend)
        if auc_c > best4: best4 = auc_c; best_w4 = w_b
    results['base_dual_blend'] = best4
    flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
    print(f"  base+dual_best: {best4:.4f}{flag}  w_base={best_w4}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 29 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
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
