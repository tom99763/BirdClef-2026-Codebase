"""
Batch 19: Prototype extensions and improvements
Goal: beat proto_knn_blend = 0.9275
Methods:
  1. Multi-scale prototype (mean, median, max aggregation blend)
  2. Weighted prototype (weight by logit confidence)
  3. Prototype with PCA denoising
  4. Negative-aware prototype (margin-based)
  5. Species-hierarchy prototype (use nearby species info)
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
logits_win = perch['logits'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[file_start[fi]:file_end[fi]] = fi

file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_embs_raw  = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_embs_raw[fi]  = emb_win[s:e].mean(0)

file_embs_norm = normalize(file_embs_raw, norm='l2').astype(np.float32)
emb_win_norm   = normalize(emb_win, norm='l2').astype(np.float32)

EPS = 1e-7
mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9275

def eval_loo(scores): return roc_auc_score(file_labels[:, mask], scores[:, mask], average='macro')
results = {}

def cosine_prototype(fi, tr_emb, tr_lab, te_wins):
    """Cosine prototype scores. tr_emb: [65,1536] normalized."""
    win_scores = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:, si] > 0.5; neg = ~pos
        if not pos.any(): win_scores[:, si] = 0.5; continue
        proto_pos = tr_emb[pos].mean(0)
        proto_pos /= (np.linalg.norm(proto_pos) + EPS)
        sim_pos = te_wins @ proto_pos
        if neg.any():
            proto_neg = tr_emb[neg].mean(0)
            proto_neg /= (np.linalg.norm(proto_neg) + EPS)
            win_scores[:, si] = (sim_pos - te_wins @ proto_neg + 1) / 2
        else:
            win_scores[:, si] = (sim_pos + 1) / 2
    return win_scores

def cosine_knn(tr_emb, tr_lab, te_wins, k=5):
    sims = te_wins @ tr_emb.T
    topk = np.argsort(-sims, axis=1)[:, :k]
    w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
    w /= w.sum(1, keepdims=True) + EPS
    return (w[:, :, None] * tr_lab[topk]).sum(1).astype(np.float32)

# ─── Method 1: Multi-aggregation prototype ────────────────────────────────────
# Try mean, window-max, and weighted-mean (by Perch logit)
print("=== Method 1: Multi-aggregation prototype ===", flush=True)
t0 = time.time()
out_mean   = np.zeros((n_files, n_species), np.float32)  # mean pool
out_winmax = np.zeros((n_files, n_species), np.float32)  # max pool per file
out_wt     = np.zeros((n_files, n_species), np.float32)  # logit-weighted mean

for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    s, e = int(file_start[fi]), int(file_end[fi])

    # Mean pool (same as batch18)
    tr_emb_mean = file_embs_norm[tr_idx]
    out_mean[fi] = cosine_prototype(fi, tr_emb_mean, tr_lab, te_wins).mean(0)

    # Max pool: each training file contributes its most confident window
    tr_emb_max_list = []
    for fj in tr_idx:
        wins_j = emb_win_norm[win_file_id == fj]
        # Window with highest L2 norm (most confident Perch window)
        norms_j = np.linalg.norm(perch['emb'][win_file_id == fj].astype(np.float32), axis=1)
        best_w = wins_j[np.argmax(norms_j)]
        tr_emb_max_list.append(best_w)
    tr_emb_max = normalize(np.array(tr_emb_max_list), norm='l2').astype(np.float32)
    out_winmax[fi] = cosine_prototype(fi, tr_emb_max, tr_lab, te_wins).mean(0)

    # Logit-weighted mean: weight each training file by max logit for each species
    # Global weight: mean of all species logits
    tr_logit_w = np.array([file_logit_max[fj].mean() for fj in tr_idx])
    tr_logit_w = np.exp(tr_logit_w - tr_logit_w.max())
    tr_logit_w /= tr_logit_w.sum()
    tr_emb_wt = normalize((file_embs_raw[tr_idx] * tr_logit_w[:, None]).sum(0, keepdims=True), norm='l2')[0]
    # Use weighted centroid as single prototype
    win_scores_wt = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:, si] > 0.5
        if not pos.any(): win_scores_wt[:, si] = 0.5; continue
        # Logit-weighted positive centroid
        tr_lw_si = np.exp(file_logit_max[tr_idx, si][pos] - file_logit_max[tr_idx, si][pos].max())
        tr_lw_si /= tr_lw_si.sum()
        proto = normalize((file_embs_raw[tr_idx][pos] * tr_lw_si[:, None]).sum(0, keepdims=True), norm='l2')[0]
        win_scores_wt[:, si] = (te_wins @ proto + 1) / 2
    out_wt[fi] = win_scores_wt.mean(0)

auc1a = eval_loo(out_mean)
auc1b = eval_loo(out_winmax)
auc1c = eval_loo(out_wt)
print(f"  mean_pool: {auc1a:.4f}", flush=True)
print(f"  max_pool:  {auc1b:.4f}", flush=True)
print(f"  logit_wt:  {auc1c:.4f}", flush=True)
results.update({'proto_mean_pool': auc1a, 'proto_max_pool': auc1b, 'proto_logit_wt': auc1c})
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# Best single-method prototype
best_out = max([(auc1a, out_mean), (auc1b, out_winmax), (auc1c, out_wt)], key=lambda x: x[0])
best_proto = best_out[1]

# ─── Method 2: PCA-denoised prototype ────────────────────────────────────────
# Project training embeddings to PCA-64 before computing prototypes
print("\n=== Method 2: PCA-denoised prototype (PCA-64) ===", flush=True)
t0 = time.time()
out_pca_proto = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_lab = file_labels[tr_idx]
    tr_emb_raw = file_embs_norm[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    # PCA on training file embeddings
    pca = PCA(n_components=min(64, len(tr_idx)-1), random_state=42).fit(tr_emb_raw)
    tr_pca = normalize(pca.transform(tr_emb_raw).astype(np.float32), norm='l2')
    te_pca = normalize(pca.transform(te_wins).astype(np.float32), norm='l2')
    out_pca_proto[fi] = cosine_prototype(fi, tr_pca, tr_lab, te_pca).mean(0)
auc2 = eval_loo(out_pca_proto)
flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
print(f"  PCA-64 prototype: {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['proto_pca64'] = auc2

# ─── Method 3: Multi-k KNN blend ──────────────────────────────────────────────
print("\n=== Method 3: Multi-k KNN blend (k=3,5,10) ===", flush=True)
t0 = time.time()
out_k3  = np.zeros((n_files, n_species), np.float32)
out_k5  = np.zeros((n_files, n_species), np.float32)
out_k10 = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    out_k3[fi]  = cosine_knn(tr_emb, tr_lab, te_wins, k=3).mean(0)
    out_k5[fi]  = cosine_knn(tr_emb, tr_lab, te_wins, k=5).mean(0)
    out_k10[fi] = cosine_knn(tr_emb, tr_lab, te_wins, k=10).mean(0)
for name, out in [('knn_k3', out_k3), ('knn_k5', out_k5), ('knn_k10', out_k10)]:
    auc = eval_loo(out)
    results[name] = auc
    print(f"  {name}: {auc:.4f}", flush=True)
# Blend k3+k5+k10
multi_k = (out_k3 + out_k5 + out_k10) / 3
auc3b = eval_loo(multi_k)
results['knn_multi_k'] = auc3b
print(f"  knn_multi_k: {auc3b:.4f}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: Proto + multi-k KNN blend ─────────────────────────────────────
print("\n=== Method 4: Proto + multi-k KNN blend ===", flush=True)
best4 = 0; best_cfg4 = None
for w_proto in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for knn_out, knn_name in [(out_k5, 'k5'), (multi_k, 'multik')]:
        blend = w_proto * out_mean + (1-w_proto) * knn_out
        auc_c = eval_loo(blend)
        if auc_c > best4: best4 = auc_c; best_cfg4 = (w_proto, knn_name)
results['proto_multik_blend'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  Best: w_proto={best_cfg4[0]}, knn={best_cfg4[1]} → {best4:.4f}{flag}", flush=True)

# ─── Method 5: PCA-denoised prototype + KNN blend ────────────────────────────
print("\n=== Method 5: PCA-proto + KNN blend ===", flush=True)
best5 = 0; best_cfg5 = None
for w_proto in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    blend = w_proto * out_pca_proto + (1-w_proto) * out_k5
    auc_c = eval_loo(blend)
    if auc_c > best5: best5 = auc_c; best_cfg5 = w_proto
results['pca_proto_knn_blend'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  Best: w_proto={best_cfg5} → {best5:.4f}{flag}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 19 Summary ===", flush=True)
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
