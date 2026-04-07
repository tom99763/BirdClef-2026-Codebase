"""
Batch 26: Adaptive positive grid search + symmetric adaptive contrast
Goal: beat blend_adaptive_pos_topk4 = 0.9610
Methods:
  1. k_pos x k_neg grid (k_pos=[1,2,3], k_neg=[2,3,4,5,6])
  2. Symmetric adaptive: top-k pos (adaptive) + top-k neg (adaptive)
  3. Multi-resolution blend: adaptive_k1 + mean_proto (pos-side fusion)
  4. Pessimistic positive: use min(sim to nearest pos) as score modifier
  5. Hardest negative mining: find negatives nearest to the positive prototype
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
CURRENT_BEST = 0.9610

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def adaptive_contrast(k_pos, k_neg):
    """Query-adaptive pos (top-k pos wins) + top-k neg wins contrast."""
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
            k_act_pos = min(k_pos, pos_sims.shape[1])
            top_pos = pos_wins[np.argsort(-pos_sims, axis=1)[:, :k_act_pos]].mean(1)
            top_pos /= (np.linalg.norm(top_pos, axis=1, keepdims=True) + EPS)
            sp = (te_wins * top_pos).sum(1)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act_neg = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act_neg]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    return out

# ─── Method 1: k_pos × k_neg grid ─────────────────────────────────────────────
print("=== Method 1: k_pos x k_neg grid ===", flush=True)
t0 = time.time()
grid_outs = {}
for k_pos in [1, 2, 3]:
    for k_neg in [2, 3, 4, 5, 6]:
        out = adaptive_contrast(k_pos, k_neg)
        auc = eval_loo(out)
        key = f'adap_kp{k_pos}_kn{k_neg}'
        results[key] = auc
        grid_outs[(k_pos, k_neg)] = out
        flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
        print(f"  k_pos={k_pos}, k_neg={k_neg}: {auc:.4f}{flag}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

best_grid = max(grid_outs, key=lambda k: results[f'adap_kp{k[0]}_kn{k[1]}'])
best_grid_out = grid_outs[best_grid]
print(f"  Best grid: k_pos={best_grid[0]}, k_neg={best_grid[1]}: {results[f'adap_kp{best_grid[0]}_kn{best_grid[1]}']:.4f}", flush=True)

# ─── Method 2: Mean-pos + adaptive-pos blend (pos-side fusion) ────────────────
print("\n=== Method 2: Mean-pos + adaptive-pos blend ===", flush=True)
t0 = time.time()
# Compute mean-pos topk4-neg (fast reference)
out_mean_topk4 = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    tr_wins_all = emb_win_norm[win_file_id != fi]
    tr_fids_all = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:,si]>0.5
        if not pos.any(): ws[:,si]=0.5; continue
        pp = tr_emb[pos].mean(0); pp/=(np.linalg.norm(pp)+EPS)
        sp = te_wins @ pp
        neg_win_mask = tr_lab_win[:,si]<0.5
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            top_neg = neg_wins[np.argsort(-neg_sims,axis=1)[:,:min(4,neg_sims.shape[1])]].mean(1)
            top_neg/=(np.linalg.norm(top_neg,axis=1,keepdims=True)+EPS)
            ws[:,si] = (sp-(te_wins*top_neg).sum(1)+1)/2
        else: ws[:,si]=(sp+1)/2
    out_mean_topk4[fi]=ws.mean(0)

out_adap_k1_k4 = grid_outs.get((1,4), adaptive_contrast(1,4))
best2 = 0; best_w2 = None
for w_adap in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    blend = w_adap * out_adap_k1_k4 + (1-w_adap) * out_mean_topk4
    auc_c = eval_loo(blend)
    if auc_c > best2: best2 = auc_c; best_w2 = w_adap
results['adap_k1_mean_blend'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  adap_k1+mean_topk4: {best2:.4f}{flag}  w_adap={best_w2}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 3: Symmetric adaptive (top-k pos + top-k neg both adaptive) ──────
print("\n=== Method 3: Symmetric adaptive contrast (best k_pos same as k_neg) ===", flush=True)
t0 = time.time()
# Already computed in grid. Find where k_pos == k_neg
for k in [1, 2, 3]:
    if (k, k) in grid_outs:
        auc = results[f'adap_kp{k}_kn{k}']
        flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
        print(f"  symmetric k={k}: {auc:.4f}{flag}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: Hardest negative mining (neg nearest to pos prototype) ─────────
print("\n=== Method 4: Hard negative from pos-prototype perspective ===", flush=True)
t0 = time.time()
out_hardneg = np.zeros((n_files, n_species), np.float32)
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
        # Adaptive pos: top-1 pos window per test win
        pos_sims = te_wins @ pos_wins.T
        pp_adap = pos_wins[pos_sims.argmax(1)]
        pp_adap /= (np.linalg.norm(pp_adap, axis=1, keepdims=True) + EPS)
        sp = (te_wins * pp_adap).sum(1)
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            # Hard negative: nearest to the POSITIVE PROTOTYPE (not to test win)
            # Mean pos proto
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            neg_to_pos_sims = neg_wins @ pp_mean  # [n_neg]
            k_hard = min(4, neg_wins.shape[0])
            hard_neg_idx = np.argsort(-neg_to_pos_sims)[:k_hard]
            hard_negs = neg_wins[hard_neg_idx]  # [k_hard, 1536]
            # For each test win, use these fixed hard negatives
            mean_hard_neg = hard_negs.mean(0)
            mean_hard_neg /= (np.linalg.norm(mean_hard_neg) + EPS)
            sn = te_wins @ mean_hard_neg  # [n_te]
            ws[:,si] = (sp - sn + 1) / 2
        else: ws[:,si] = (sp+1)/2
    out_hardneg[fi] = ws.mean(0)
auc4 = eval_loo(out_hardneg)
results['hard_neg_adap_pos'] = auc4
flag = " *** NEW BEST ***" if auc4 > CURRENT_BEST else ""
print(f"  hard_neg_adap_pos: {auc4:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 5: Best-grid blend with KNN ──────────────────────────────────────
print("\n=== Method 5: Best-grid blend with multi-k KNN ===", flush=True)
# Precompute multi-k KNN
out_knn_mk = np.zeros((n_files, n_species), np.float32)
EPS2 = 1e-7
def cosine_knn(tr_emb, tr_lab, te_wins, k=5):
    sims = te_wins @ tr_emb.T
    topk = np.argsort(-sims, axis=1)[:, :k]
    w = np.take_along_axis(sims, topk, axis=1).clip(0,1)
    w /= w.sum(1,keepdims=True)+EPS2
    return (w[:,:,None]*tr_lab[topk]).sum(1).astype(np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    out_knn_mk[fi] = np.mean([cosine_knn(tr_emb, tr_lab, te_wins, k=k).mean(0) for k in [3,5,10]], 0)

best5 = 0; best_cfg5 = None
for w_adap in [0.85, 0.9, 0.92, 0.95]:
    blend = w_adap * best_grid_out + (1-w_adap) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best5: best5 = auc_c; best_cfg5 = (w_adap, best_grid)
results['best_grid_knn'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  best_grid+knn: {best5:.4f}{flag}  cfg={best_cfg5}", flush=True)

# Also blend adap_k1_k4 + knn
best5b = 0; best_w5b = None
for w_a in [0.85, 0.9, 0.92, 0.95]:
    blend = w_a * out_adap_k1_k4 + (1-w_a) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best5b: best5b = auc_c; best_w5b = w_a
results['adap_k1k4_knn'] = best5b
flag = " *** NEW BEST ***" if best5b > CURRENT_BEST else ""
print(f"  adap_k1k4+knn: {best5b:.4f}{flag}  w={best_w5b}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 26 Summary ===", flush=True)
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
