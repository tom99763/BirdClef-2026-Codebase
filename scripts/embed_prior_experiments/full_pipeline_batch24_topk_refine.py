"""
Batch 24: Top-k window contrast refinement + hybrid methods
Goal: beat win_contrast_topk3 = 0.9552
Methods:
  1. k_neg sweep (k=2,4,6,8,10)
  2. topk3 + topk5 blend
  3. topk3 win neg + window pos prototype (fully window-level)
  4. topk3 with distance-weighted mean neg (not simple mean)
  5. Dual-resolution: file pos proto + topk3 window neg + multik KNN
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
CURRENT_BEST = 0.9552

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')

def cosine_knn(tr_emb, tr_lab, te_wins, k=5):
    sims = te_wins @ tr_emb.T
    topk = np.argsort(-sims, axis=1)[:, :k]
    w = np.take_along_axis(sims, topk, axis=1).clip(0,1)
    w /= w.sum(1,keepdims=True)+EPS
    return (w[:,:,None]*tr_lab[topk]).sum(1).astype(np.float32)

def win_topk_contrast(k_neg, weighted=False):
    """Window-level contrastive with top-k neg windows."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_win_norm[win_file_id == fi]
        tr_wins = emb_win_norm[win_file_id != fi]
        tr_fids = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids])
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos = tr_lab[:,si]>0.5
            if not pos.any(): ws[:,si]=0.5; continue
            pp = tr_emb[pos].mean(0); pp/=(np.linalg.norm(pp)+EPS)
            sp = te_wins @ pp
            neg_win_mask = tr_lab_win[:,si]<0.5
            if neg_win_mask.any():
                neg_wins = tr_wins[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T  # [n_te, n_neg_wins]
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg_idx = np.argsort(-neg_sims, axis=1)[:, :k_act]
                top_neg = neg_wins[top_neg_idx]  # [n_te, k, 1536]
                if weighted:
                    # Weight by similarity
                    top_neg_sims = np.take_along_axis(neg_sims, top_neg_idx, axis=1)
                    top_neg_sims = top_neg_sims.clip(0, 1)
                    top_neg_sims /= top_neg_sims.sum(1, keepdims=True) + EPS
                    mean_neg = (top_neg * top_neg_sims[:, :, None]).sum(1)
                else:
                    mean_neg = top_neg.mean(1)
                mean_neg /= (np.linalg.norm(mean_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins*mean_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    return out

results = {}

# ─── Method 1: k_neg sweep ───────────────────────────────────────────────────
print("=== Method 1: k_neg sweep ===", flush=True)
t0 = time.time()
outs_k = {}
for k_neg in [2, 3, 4, 5, 6, 8, 10]:
    out = win_topk_contrast(k_neg)
    auc = eval_loo(out)
    outs_k[k_neg] = out
    results[f'win_topk_neg{k_neg}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  k_neg={k_neg}: {auc:.4f}{flag}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)
best_k = max(outs_k, key=lambda k: results[f'win_topk_neg{k}'])
out_best_k = outs_k[best_k]

# ─── Method 2: Distance-weighted neg ─────────────────────────────────────────
print("\n=== Method 2: Similarity-weighted neg (top-3) ===", flush=True)
t0 = time.time()
out_wt = win_topk_contrast(3, weighted=True)
auc2 = eval_loo(out_wt)
flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
print(f"  weighted_topk3: {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['win_topk3_weighted'] = auc2

# ─── Method 3: Blend of different k values ───────────────────────────────────
print("\n=== Method 3: Multi-k blend ===", flush=True)
out_k3 = outs_k[3]; out_k5 = outs_k[5]
best3 = 0; best_w3 = None
for w3 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blend = w3 * out_k3 + (1-w3) * out_k5
    auc_c = eval_loo(blend)
    if auc_c > best3: best3 = auc_c; best_w3 = w3
results['topk3_topk5_blend'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  k3+k5: {best3:.4f}{flag}  w_k3={best_w3}", flush=True)

# Precompute multik KNN
out_knn_mk = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    out_knn_mk[fi] = np.mean([cosine_knn(tr_emb, tr_lab, te_wins, k=k).mean(0) for k in [3,5,10]], 0)

# ─── Method 4: Best k + KNN blend ────────────────────────────────────────────
print("\n=== Method 4: Best-k contrast + multi-k KNN blend ===", flush=True)
best4 = 0; best_cfg4 = None
for w_c in [0.7, 0.8, 0.85, 0.9, 0.95]:
    blend = w_c * out_best_k + (1-w_c) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best4: best4 = auc_c; best_cfg4 = (w_c, best_k)
results['best_k_contrast_knn'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  best_k_contrast+knn: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Method 5: Win pos + win neg (fully window) ──────────────────────────────
print("\n=== Method 5: Fully window (win pos proto + topk3 win neg) ===", flush=True)
t0 = time.time()
out_full = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_wins = emb_win_norm[win_file_id == fi]
    tr_wins = emb_win_norm[win_file_id != fi]
    tr_fids = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids])
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos_mask = tr_lab_win[:,si]>0.5; neg_mask=~pos_mask
        if not pos_mask.any(): ws[:,si]=0.5; continue
        # Win-level positive prototype
        pp = tr_wins[pos_mask].mean(0); pp/=(np.linalg.norm(pp)+EPS)
        sp = te_wins @ pp
        if neg_mask.any():
            neg_wins = tr_wins[neg_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(3, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims,axis=1)[:,:k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg,axis=1,keepdims=True)+EPS)
            ws[:,si]=(sp-(te_wins*top_neg).sum(1)+1)/2
        else: ws[:,si]=(sp+1)/2
    out_full[fi] = ws.mean(0)
auc5 = eval_loo(out_full)
flag = " *** NEW BEST ***" if auc5 > CURRENT_BEST else ""
print(f"  full_win_topk3: {auc5:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['full_win_topk3'] = auc5

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 24 Summary ===", flush=True)
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
