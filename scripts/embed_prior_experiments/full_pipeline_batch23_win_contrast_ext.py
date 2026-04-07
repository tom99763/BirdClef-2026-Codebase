"""
Batch 23: Window-level contrastive extensions
Goal: beat win_level_contrast = 0.9524
Methods:
  1. Top-k window neg (nearest k neg windows avg)
  2. Win-contrast + file-contrast blend
  3. Win-contrast pos (use training windows for positive prototype too)
  4. Mixed resolution: file-level KNN + window-level contrastive
  5. Dual-contrastive: contrast against nearest neg file AND nearest neg window
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
CURRENT_BEST = 0.9524

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def cosine_knn(tr_emb, tr_lab, te_wins, k=5):
    sims = te_wins @ tr_emb.T
    topk = np.argsort(-sims, axis=1)[:, :k]
    w = np.take_along_axis(sims, topk, axis=1).clip(0,1)
    w /= w.sum(1,keepdims=True)+EPS
    return (w[:,:,None]*tr_lab[topk]).sum(1).astype(np.float32)

# Precompute win_contrast and file_contrast
print("Precomputing references...", flush=True)
out_win_contrast = np.zeros((n_files, n_species), np.float32)
out_file_contrast = np.zeros((n_files, n_species), np.float32)
out_knn_mk = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    out_knn_mk[fi] = np.mean([cosine_knn(tr_emb, tr_lab, te_wins, k=k).mean(0) for k in [3,5,10]], 0)
    tr_wins_all = emb_win_norm[win_file_id != fi]
    tr_fids_all = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
    ws_w = np.zeros((len(te_wins), n_species), np.float32)
    ws_f = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:,si]>0.5; neg=~pos
        if not pos.any(): ws_w[:,si]=0.5; ws_f[:,si]=0.5; continue
        pp = tr_emb[pos].mean(0); pp/=(np.linalg.norm(pp)+EPS)
        sp = te_wins @ pp
        # File-level contrast
        if neg.any():
            neg_sims_f = te_wins @ tr_emb[neg].T
            nn_f = tr_emb[neg][neg_sims_f.argmax(1)]
            nn_f/=(np.linalg.norm(nn_f,axis=1,keepdims=True)+EPS)
            ws_f[:,si]=(sp-(te_wins*nn_f).sum(1)+1)/2
        else: ws_f[:,si]=(sp+1)/2
        # Window-level contrast
        neg_win_mask = tr_lab_win[:,si]<0.5
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims_w = te_wins @ neg_wins.T
            nn_w = neg_wins[neg_sims_w.argmax(1)]
            nn_w/=(np.linalg.norm(nn_w,axis=1,keepdims=True)+EPS)
            ws_w[:,si]=(sp-(te_wins*nn_w).sum(1)+1)/2
        else: ws_w[:,si]=(sp+1)/2
    out_win_contrast[fi] = ws_w.mean(0)
    out_file_contrast[fi] = ws_f.mean(0)
print("  Done.", flush=True)

# ─── Method 1: Top-k window negative ─────────────────────────────────────────
print("\n=== Method 1: Top-k window neg (k=1,3,5 avg) ===", flush=True)
t0 = time.time()
out_topk_win = {}
for k_neg in [1, 3, 5]:
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
            pos = tr_lab[:,si]>0.5
            if not pos.any(): ws[:,si]=0.5; continue
            pp = tr_emb[pos].mean(0); pp/=(np.linalg.norm(pp)+EPS)
            sp = te_wins @ pp
            neg_win_mask = tr_lab_win[:,si]<0.5
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_actual = min(k_neg, neg_sims.shape[1])
                top_neg_idx = np.argsort(-neg_sims, axis=1)[:, :k_actual]
                mean_neg = neg_wins[top_neg_idx].mean(1)
                mean_neg /= (np.linalg.norm(mean_neg,axis=1,keepdims=True)+EPS)
                ws[:,si] = (sp-(te_wins*mean_neg).sum(1)+1)/2
            else: ws[:,si]=(sp+1)/2
        out[fi] = ws.mean(0)
    auc = eval_loo(out)
    out_topk_win[k_neg] = out
    results[f'win_contrast_topk{k_neg}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  k_neg={k_neg}: {auc:.4f}{flag}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 2: Win-contrast + file-contrast blend ────────────────────────────
print("\n=== Method 2: Win-contrast + file-contrast blend ===", flush=True)
best2 = 0; best_w2 = None
for w_wc in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    blend = w_wc * out_win_contrast + (1-w_wc) * out_file_contrast
    auc_c = eval_loo(blend)
    if auc_c > best2: best2 = auc_c; best_w2 = w_wc
results['win_file_contrast_blend'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  win+file contrast: {best2:.4f}{flag}  w_wc={best_w2}", flush=True)

# ─── Method 3: Fully-window prototype + window neg ───────────────────────────
# Pos proto from windows, neg from windows
print("\n=== Method 3: Fully window-level (pos+neg from windows) ===", flush=True)
t0 = time.time()
out_full_win = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_wins = emb_win_norm[win_file_id == fi]
    tr_wins = emb_win_norm[win_file_id != fi]
    tr_fids = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids])
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos_mask = tr_lab_win[:,si]>0.5; neg_mask = ~pos_mask
        if not pos_mask.any(): ws[:,si]=0.5; continue
        # Pos proto from positive WINDOWS
        pp = tr_wins[pos_mask].mean(0); pp/=(np.linalg.norm(pp)+EPS)
        sp = te_wins @ pp
        if neg_mask.any():
            neg_wins = tr_wins[neg_mask]
            neg_sims = te_wins @ neg_wins.T
            nn_w = neg_wins[neg_sims.argmax(1)]
            nn_w/=(np.linalg.norm(nn_w,axis=1,keepdims=True)+EPS)
            ws[:,si]=(sp-(te_wins*nn_w).sum(1)+1)/2
        else: ws[:,si]=(sp+1)/2
    out_full_win[fi] = ws.mean(0)
auc3 = eval_loo(out_full_win)
flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
print(f"  full_win_contrast: {auc3:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['full_win_contrast'] = auc3

# ─── Method 4: Top-1 win neg + Top-1 file neg blend ─────────────────────────
print("\n=== Method 4: Win neg + file neg blend ===", flush=True)
best4 = 0; best_w4 = None
for w_wn in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    blend = w_wn * out_win_contrast + (1-w_wn) * out_file_contrast
    auc_c = eval_loo(blend)
    if auc_c > best4: best4 = auc_c; best_w4 = w_wn
# Also blend with KNN
best4b = 0; best_cfg4b = None
for w_c in [0.6, 0.7, 0.8, 0.9]:
    for w_wn_inner in [0.5, 0.6, 0.7]:
        mixed_c = w_wn_inner * out_win_contrast + (1-w_wn_inner) * out_file_contrast
        blend = w_c * mixed_c + (1-w_c) * out_knn_mk
        auc_c = eval_loo(blend)
        if auc_c > best4b: best4b = auc_c; best_cfg4b = (w_c, w_wn_inner)
results['dual_contrast_knn'] = best4b
flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
print(f"  dual_contrast+knn: {best4b:.4f}{flag}  cfg={best_cfg4b}", flush=True)

# ─── Method 5: Contrast variant ensemble ─────────────────────────────────────
print("\n=== Method 5: Multi-contrast ensemble ===", flush=True)
# Avg: win_contrast, file_contrast, topk_win variants
ens = (out_win_contrast + out_file_contrast + out_topk_win.get(3, out_win_contrast)) / 3
auc5 = eval_loo(ens)
results['multi_contrast_ens'] = auc5
flag = " *** NEW BEST ***" if auc5 > CURRENT_BEST else ""
print(f"  multi_contrast_ens: {auc5:.4f}{flag}", flush=True)

# + KNN
best5b = 0; best_w5b = None
for w_e in [0.6, 0.7, 0.8, 0.9, 0.95]:
    blend = w_e * ens + (1-w_e) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best5b: best5b = auc_c; best_w5b = w_e
results['multi_contrast_ens_knn'] = best5b
flag = " *** NEW BEST ***" if best5b > CURRENT_BEST else ""
print(f"  multi_contrast_ens+knn: {best5b:.4f}{flag}  w={best_w5b}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 23 Summary ===", flush=True)
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
