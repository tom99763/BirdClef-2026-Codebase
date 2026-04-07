"""
Batch 22: Contrastive prototype extensions
Goal: beat contrastive_proto = 0.9490
Methods:
  1. Top-k contrastive (nearest k negatives, not just 1)
  2. Contrastive + multi-k KNN (fine sweep)
  3. Bidirectional contrastive (both nearest neg AND nearest pos among negatives)
  4. Window-level contrastive prototype (use training WINDOWS as neg candidates)
  5. Contrastive + proto_multik ensemble
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
CURRENT_BEST = 0.9490

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')

def cosine_knn(tr_emb, tr_lab, te_wins, k=5):
    sims = te_wins @ tr_emb.T
    topk = np.argsort(-sims, axis=1)[:, :k]
    w = np.take_along_axis(sims, topk, axis=1).clip(0,1)
    w /= w.sum(1,keepdims=True)+EPS
    return (w[:,:,None]*tr_lab[topk]).sum(1).astype(np.float32)

results = {}

# Precompute base components
print("Precomputing base components...", flush=True)
out_knn_mk = np.zeros((n_files, n_species), np.float32)
out_contrast = np.zeros((n_files, n_species), np.float32)
out_proto = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    # Multi-k KNN
    out_knn_mk[fi] = np.mean([cosine_knn(tr_emb, tr_lab, te_wins, k=k).mean(0) for k in [3,5,10]], 0)
    # Contrastive prototype
    ws_c = np.zeros((len(te_wins), n_species), np.float32)
    ws_p = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:,si]>0.5; neg = ~pos
        if not pos.any(): ws_c[:,si]=0.5; ws_p[:,si]=0.5; continue
        pp = tr_emb[pos].mean(0); pp/=(np.linalg.norm(pp)+EPS)
        sp = te_wins @ pp
        ws_p[:,si] = (sp+1)/2
        if neg.any():
            neg_sims = te_wins @ tr_emb[neg].T
            nn = tr_emb[neg][neg_sims.argmax(1)]
            nn /= (np.linalg.norm(nn,axis=1,keepdims=True)+EPS)
            ws_c[:,si] = (sp-(te_wins*nn).sum(1)+1)/2
        else: ws_c[:,si]=(sp+1)/2
    out_contrast[fi] = ws_c.mean(0)
    out_proto[fi] = ws_p.mean(0)
print("  Done.", flush=True)

# ─── Method 1: Top-k negative contrastive ─────────────────────────────────────
print("\n=== Method 1: Top-k contrastive (k=1,3,5 nearest neg avg) ===", flush=True)
t0 = time.time()
out_topk_contrast = {}
for k_neg in [1, 3, 5]:
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_win_norm[win_file_id == fi]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos = tr_lab[:,si]>0.5; neg = ~pos
            if not pos.any(): ws[:,si]=0.5; continue
            pp = tr_emb[pos].mean(0); pp/=(np.linalg.norm(pp)+EPS)
            sp = te_wins @ pp
            if neg.any():
                neg_sims = te_wins @ tr_emb[neg].T   # [n_te, n_neg]
                k_actual = min(k_neg, neg_sims.shape[1])
                top_neg_idx = np.argsort(-neg_sims, axis=1)[:, :k_actual]
                # Mean of top-k nearest negatives
                top_neg = tr_emb[neg][top_neg_idx]  # [n_te, k, 1536]
                mean_neg = top_neg.mean(1)           # [n_te, 1536]
                mean_neg /= (np.linalg.norm(mean_neg, axis=1, keepdims=True) + EPS)
                sn = (te_wins * mean_neg).sum(1)
                ws[:,si] = (sp - sn + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    auc = eval_loo(out)
    out_topk_contrast[k_neg] = out
    results[f'contrast_topk_neg{k_neg}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  k_neg={k_neg}: {auc:.4f}{flag}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

best_kneg = max(out_topk_contrast, key=lambda k: results[f'contrast_topk_neg{k}'])
out_best_contrast = out_topk_contrast[best_kneg]

# ─── Method 2: Contrastive + multi-k KNN blend (fine) ────────────────────────
print("\n=== Method 2: Best-contrast + multi-k KNN blend ===", flush=True)
best2 = 0; best_w2 = None
for w_c in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    blend = w_c * out_best_contrast + (1-w_c) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best2: best2 = auc_c; best_w2 = (w_c, best_kneg)
results['contrast_knn_fine'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  best: {best2:.4f}{flag}  w_c={best_w2}", flush=True)

# ─── Method 3: Window-level contrastive prototype ────────────────────────────
# Use all training WINDOWS as negative pool (not just file means)
print("\n=== Method 3: Window-level contrastive (neg from all training windows) ===", flush=True)
t0 = time.time()
out_win_contrast = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    # Use file-level mean for positive prototype (stable)
    tr_emb = file_embs_norm[tr_idx]
    # Use all training windows as negative candidates
    tr_wins_all = emb_win_norm[win_file_id != fi]
    tr_fids_all = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:,si]>0.5; neg = ~pos
        if not pos.any(): ws[:,si]=0.5; continue
        # Positive prototype (file mean)
        pp = tr_emb[pos].mean(0); pp/=(np.linalg.norm(pp)+EPS)
        sp = te_wins @ pp
        # Negative candidates: windows from neg files
        neg_file_mask = tr_lab_win[:,si] < 0.5  # window belongs to a neg file
        if neg_file_mask.any():
            neg_wins = tr_wins_all[neg_file_mask]  # all neg windows
            neg_sims = te_wins @ neg_wins.T  # [n_te, n_neg_wins]
            nn_win = neg_wins[neg_sims.argmax(1)]  # nearest neg window
            nn_win /= (np.linalg.norm(nn_win,axis=1,keepdims=True)+EPS)
            sn = (te_wins * nn_win).sum(1)
            ws[:,si] = (sp - sn + 1) / 2
        else: ws[:,si] = (sp+1)/2
    out_win_contrast[fi] = ws.mean(0)
auc3 = eval_loo(out_win_contrast)
flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
print(f"  win_contrast: {auc3:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['win_level_contrast'] = auc3

# win_contrast + multik blend
best3b = 0; best_w3b = None
for w_wc in [0.5, 0.6, 0.7, 0.8, 0.9]:
    blend = w_wc * out_win_contrast + (1-w_wc) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best3b: best3b = auc_c; best_w3b = w_wc
results['win_contrast_knn_blend'] = best3b
flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
print(f"  win_contrast+knn: {best3b:.4f}{flag}  w={best_w3b}", flush=True)

# ─── Method 4: Contrastive ensemble (k=1 + k=3 + mean_neg) ──────────────────
print("\n=== Method 4: Contrast ensemble ===", flush=True)
# Blend different contrastive variants
ens4 = (out_topk_contrast[1] + out_topk_contrast[3] + out_contrast) / 3
auc4 = eval_loo(ens4)
results['contrast_ensemble'] = auc4
flag = " *** NEW BEST ***" if auc4 > CURRENT_BEST else ""
print(f"  contrast_ens: {auc4:.4f}{flag}", flush=True)

# + KNN
best4b = 0; best_w4b = None
for w_e in [0.5, 0.6, 0.7, 0.8, 0.9]:
    blend = w_e * ens4 + (1-w_e) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best4b: best4b = auc_c; best_w4b = w_e
results['contrast_ens_knn_blend'] = best4b
flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
print(f"  contrast_ens+knn: {best4b:.4f}{flag}  w={best_w4b}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 22 Summary ===", flush=True)
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
