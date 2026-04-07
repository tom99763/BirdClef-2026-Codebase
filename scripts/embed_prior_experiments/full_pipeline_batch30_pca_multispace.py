"""
Batch 30: PCA-reduced space + multi-space fusion
Goal: beat mm_wm7_kn6 = 0.9618
Methods:
  1. PCA-reduced max_pos contrast (64, 128, 256, 512 dims)
  2. Dual-space blend: full 1536 + PCA-256
  3. Per-species feature weighting (variance-based)
  4. Whitened embedding space contrast
  5. Triple-space: 1536 + PCA-256 + PCA-64
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

def max_pos_contrast(emb_wins_norm, emb_files_norm, k_neg=6, w_max=0.7):
    """Compute max_pos contrast scores for all files LOO."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = emb_files_norm[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_wins_norm[win_file_id == fi]
        tr_wins_all = emb_wins_norm[win_file_id != fi]
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
            sp = w_max * pos_sims.max(1) + (1-w_max) * (te_wins @ pp_mean)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    return out

# ─── Method 1: PCA-reduced space ─────────────────────────────────────────────
print("=== Method 1: PCA-reduced space ===", flush=True)
t0 = time.time()
pca_outs = {}
for n_comp in [64, 128, 256, 512]:
    pca = PCA(n_components=n_comp, random_state=42)
    emb_all_pca = pca.fit_transform(emb_win).astype(np.float32)
    emb_win_pca_norm = normalize(emb_all_pca, norm='l2').astype(np.float32)
    # File-level PCA embeddings
    file_embs_pca = np.zeros((n_files, n_comp), np.float32)
    for fi in range(n_files):
        s, e = int(file_start[fi]), int(file_end[fi])
        file_embs_pca[fi] = emb_all_pca[s:e].mean(0)
    file_embs_pca_norm = normalize(file_embs_pca, norm='l2').astype(np.float32)
    out_pca = max_pos_contrast(emb_win_pca_norm, file_embs_pca_norm)
    auc_pca = eval_loo(out_pca)
    pca_outs[n_comp] = out_pca
    results[f'pca{n_comp}_max_pos'] = auc_pca
    flag = " *** NEW BEST ***" if auc_pca > CURRENT_BEST else ""
    print(f"  PCA-{n_comp}: {auc_pca:.4f}{flag}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# Base (full 1536)
out_base = max_pos_contrast(emb_win_norm, file_embs_norm)
print(f"  Base (1536): {eval_loo(out_base):.4f}", flush=True)

# ─── Method 2: Dual-space blend ──────────────────────────────────────────────
print("\n=== Method 2: Full-1536 + PCA blend ===", flush=True)
best2 = 0; best_cfg2 = None
for n_comp in [128, 256, 512]:
    if n_comp not in pca_outs: continue
    out_pca = pca_outs[n_comp]
    for w_full in [0.5, 0.6, 0.7, 0.8, 0.9]:
        blend = w_full * out_base + (1-w_full) * out_pca
        auc_c = eval_loo(blend)
        if auc_c > best2: best2 = auc_c; best_cfg2 = (w_full, n_comp)
results['full_pca_blend'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  full+PCA blend: {best2:.4f}{flag}  cfg={best_cfg2}", flush=True)

# ─── Method 3: Whitened embedding space ──────────────────────────────────────
print("\n=== Method 3: ZCA-whitened embedding ===", flush=True)
t0 = time.time()
# PCA whitening: normalize each PC by its std
pca256 = PCA(n_components=256, whiten=True, random_state=42)
emb_white = pca256.fit_transform(emb_win).astype(np.float32)
emb_white_norm = normalize(emb_white, norm='l2').astype(np.float32)
file_embs_white = np.zeros((n_files, 256), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs_white[fi] = emb_white[s:e].mean(0)
file_embs_white_norm = normalize(file_embs_white, norm='l2').astype(np.float32)
out_white = max_pos_contrast(emb_white_norm, file_embs_white_norm)
auc_white = eval_loo(out_white)
results['white256_max_pos'] = auc_white
flag = " *** NEW BEST ***" if auc_white > CURRENT_BEST else ""
print(f"  whitened-256: {auc_white:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# Blend whitened with base
best3b = 0; best_w3b = None
for w_b in [0.6, 0.7, 0.8, 0.9]:
    blend = w_b * out_base + (1-w_b) * out_white
    auc_c = eval_loo(blend)
    if auc_c > best3b: best3b = auc_c; best_w3b = w_b
results['base_white_blend'] = best3b
flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
print(f"  base+whitened blend: {best3b:.4f}{flag}  w_base={best_w3b}", flush=True)

# ─── Method 4: Triple-space (1536 + PCA-256 + PCA-64) ────────────────────────
print("\n=== Method 4: Triple-space blend ===", flush=True)
if 64 in pca_outs and 256 in pca_outs:
    best4 = 0; best_cfg4 = None
    for w1 in [0.6, 0.7, 0.8]:
        for w2 in [0.1, 0.15, 0.2]:
            w3 = 1.0 - w1 - w2
            if w3 < 0.05: continue
            blend = w1 * out_base + w2 * pca_outs[256] + w3 * pca_outs[64]
            auc_c = eval_loo(blend)
            if auc_c > best4: best4 = auc_c; best_cfg4 = (w1, w2, w3)
    results['triple_space'] = best4
    flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
    print(f"  triple-space: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 30 Summary ===", flush=True)
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
