"""
Batch 25: Query-adaptive prototype methods
Goal: beat best_k_contrast_knn = 0.9556
Methods:
  1. Nearest-positive prototype (top-k pos windows, query-adaptive)
  2. Symmetric top-k (top-k pos + top-k neg, query-adaptive)
  3. Rank-weighted negative (1/rank weight: 1, 0.5, 0.33...)
  4. Max-sim positive score (nearest positive window, not mean)
  5. Soft attention positive prototype (softmax over sim to test)
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
CURRENT_BEST = 0.9556

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')

def cosine_knn(tr_emb, tr_lab, te_wins, k=5):
    sims = te_wins @ tr_emb.T
    topk = np.argsort(-sims, axis=1)[:, :k]
    w = np.take_along_axis(sims, topk, axis=1).clip(0,1)
    w /= w.sum(1,keepdims=True)+EPS
    return (w[:,:,None]*tr_lab[topk]).sum(1).astype(np.float32)

results = {}

# ─── Method 1: Nearest-positive prototype (top-k pos windows, query-adaptive) ─
print("=== Method 1: Query-adaptive pos prototype (top-k pos windows) ===", flush=True)
t0 = time.time()
best1 = 0; best_k1 = None; best_out1 = None
for k_pos in [1, 3, 5, 10, 20]:
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
            # Query-adaptive pos: top-k most similar positive windows
            pos_sims = te_wins @ pos_wins.T  # [n_te, n_pos_wins]
            k_act_pos = min(k_pos, pos_sims.shape[1])
            top_pos_idx = np.argsort(-pos_sims, axis=1)[:, :k_act_pos]
            pp_adaptive = pos_wins[top_pos_idx].mean(1)  # [n_te, 1536]
            pp_adaptive /= (np.linalg.norm(pp_adaptive, axis=1, keepdims=True) + EPS)
            sp = (te_wins * pp_adaptive).sum(1)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act_neg = min(4, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act_neg]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    auc = eval_loo(out)
    results[f'adaptive_pos_k{k_pos}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  k_pos={k_pos}: {auc:.4f}{flag}", flush=True)
    if auc > best1: best1 = auc; best_k1 = k_pos; best_out1 = out
print(f"  ({time.time()-t0:.0f}s)\n  Best: k_pos={best_k1}: {best1:.4f}", flush=True)

# ─── Method 2: Rank-weighted negative (1/rank: w_1=1, w_2=0.5, w_3=0.33...) ──
print("\n=== Method 2: Rank-weighted negative ===", flush=True)
t0 = time.time()
out_rank = np.zeros((n_files, n_species), np.float32)
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
        neg_win_mask = tr_lab_win[:,si] < 0.5
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(5, neg_sims.shape[1])
            top_neg_idx = np.argsort(-neg_sims, axis=1)[:, :k_act]
            top_neg = neg_wins[top_neg_idx]  # [n_te, k, 1536]
            # Rank weights: 1/1, 1/2, 1/3, ...
            rank_w = np.array([1.0/(r+1) for r in range(k_act)], dtype=np.float32)
            rank_w /= rank_w.sum()
            mean_neg = (top_neg * rank_w[None, :, None]).sum(1)  # [n_te, 1536]
            mean_neg /= (np.linalg.norm(mean_neg, axis=1, keepdims=True) + EPS)
            ws[:,si] = (sp - (te_wins * mean_neg).sum(1) + 1) / 2
        else: ws[:,si] = (sp+1)/2
    out_rank[fi] = ws.mean(0)
auc2 = eval_loo(out_rank)
results['rank_weighted_neg'] = auc2
flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
print(f"  rank_weighted_neg: {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 3: Soft-attention positive prototype ──────────────────────────────
# Weight each positive window by softmax(sim_to_test / tau)
print("\n=== Method 3: Soft-attention positive prototype ===", flush=True)
t0 = time.time()
best3 = 0; best_tau3 = None; best_out3 = None
for tau in [0.1, 0.3, 0.5, 1.0, 2.0]:
    out_t = np.zeros((n_files, n_species), np.float32)
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
            # Soft attention: softmax(pos_sims / tau)
            pos_sims = te_wins @ pos_wins.T  # [n_te, n_pos]
            attn = pos_sims / tau
            attn -= attn.max(1, keepdims=True)  # numerical stability
            attn = np.exp(attn); attn /= attn.sum(1, keepdims=True) + EPS
            pp_attn = (attn[:, :, None] * pos_wins[None, :, :]).sum(1)  # [n_te, 1536]
            pp_attn /= (np.linalg.norm(pp_attn, axis=1, keepdims=True) + EPS)
            sp = (te_wins * pp_attn).sum(1)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(4, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out_t[fi] = ws.mean(0)
    auc_t = eval_loo(out_t)
    results[f'attn_pos_tau{tau}'] = auc_t
    flag = " *** NEW BEST ***" if auc_t > CURRENT_BEST else ""
    print(f"  tau={tau}: {auc_t:.4f}{flag}", flush=True)
    if auc_t > best3: best3 = auc_t; best_tau3 = tau; best_out3 = out_t
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: Max-sim positive score (nearest positive window only) ──────────
print("\n=== Method 4: Max-sim positive (nearest positive window score) ===", flush=True)
t0 = time.time()
out_maxpos = np.zeros((n_files, n_species), np.float32)
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
        neg_wins = tr_wins_all[neg_win_mask] if neg_win_mask.any() else None
        # Max similarity over all positive windows (nearest positive)
        pos_sims = te_wins @ pos_wins.T  # [n_te, n_pos]
        sp_max = pos_sims.max(1)  # [n_te] - nearest positive sim
        if neg_wins is not None:
            neg_sims = te_wins @ neg_wins.T
            k_act = min(4, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            sn = (te_wins * top_neg).sum(1)
            ws[:,si] = (sp_max - sn + 1) / 2
        else: ws[:,si] = (sp_max + 1) / 2
    out_maxpos[fi] = ws.mean(0)
auc4 = eval_loo(out_maxpos)
results['max_pos_topk4_neg'] = auc4
flag = " *** NEW BEST ***" if auc4 > CURRENT_BEST else ""
print(f"  max_pos+topk4_neg: {auc4:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 5: Blend of adaptive-pos + base contrast ─────────────────────────
print("\n=== Method 5: Blend best-new + topk4-contrast ===", flush=True)
# Recompute topk4 contrast as reference
out_topk4 = np.zeros((n_files, n_species), np.float32)
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
            k_act = min(4, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims,axis=1)[:,:k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg,axis=1,keepdims=True)+EPS)
            ws[:,si] = (sp-(te_wins*top_neg).sum(1)+1)/2
        else: ws[:,si]=(sp+1)/2
    out_topk4[fi] = ws.mean(0)

# Blend with best new method
for cand_name, cand_out in [('adaptive_pos', best_out1), ('attn_pos', best_out3), ('maxpos', out_maxpos), ('rank_neg', out_rank)]:
    if cand_out is None: continue
    best5_loc = 0; best_w5 = None
    for w_new in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = w_new * cand_out + (1 - w_new) * out_topk4
        auc_c = eval_loo(blend)
        if auc_c > best5_loc: best5_loc = auc_c; best_w5 = w_new
    results[f'blend_{cand_name}_topk4'] = best5_loc
    flag = " *** NEW BEST ***" if best5_loc > CURRENT_BEST else ""
    print(f"  {cand_name}+topk4: {best5_loc:.4f}{flag}  w_new={best_w5}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 25 Summary ===", flush=True)
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
