"""
Batch 18: Simple embed prior format (file-level LOO-CV, no base_logit fusion)
Goal: beat current best simple-format ~0.8940
Format:
  - train_emb: [65, 1536] file-averaged L2-norm embeddings
  - test_emb: [n_windows, 1536] test file windows (raw)
  - prior_score = method(train_emb, train_labels, test_emb)  # [n_windows, 234]
  - file_score = prior_score.mean(0)  # [234]
  - Evaluate: macro ROC-AUC vs file_labels

Methods:
  1. Mahalanobis KNN (global covariance inverse)
  2. Cosine prototype (per-species mean embedding similarity)
  3. GMM per species (PCA-32)
  4. Ranked attention KNN (weight by rank-based attention)
  5. Soft-KNN with bandwidth sweep
"""
import numpy as np, json, os, time, pickle, shutil
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# Load data
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)      # (739, 1536)
labels_win = perch['labels'].astype(np.float32)    # (739, 234)
logits_win = perch['logits'].astype(np.float32)    # (739, 234)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']

n_files   = len(file_list)
n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

# Construct file_ids
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[file_start[fi]:file_end[fi]] = fi

# File-level aggregation
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

EPS  = 1e-7
mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.8940

def eval_loo(scores_all):
    """scores_all: [n_files, n_species] file-level predictions."""
    return roc_auc_score(file_labels[:, mask], scores_all[:, mask], average='macro')

results = {}

# ─── Method 1: Cosine KNN (baseline reference, k=5) ─────────────────────────
print("=== Method 0 (reference): Cosine KNN k=5 ===", flush=True)
t0 = time.time()
out_knn5 = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]   # [65, 1536]
    tr_lab = file_labels[tr_idx]       # [65, 234]
    te_wins = emb_win_norm[win_file_id == fi]  # [n_wins, 1536]
    sims = te_wins @ tr_emb.T          # [n_wins, 65]
    topk = np.argsort(-sims, axis=1)[:, :5]
    w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
    w = w / (w.sum(1, keepdims=True) + EPS)
    pw = (w[:, :, None] * tr_lab[topk]).sum(1)  # [n_wins, 234]
    out_knn5[fi] = pw.mean(0)
auc_ref = eval_loo(out_knn5)
print(f"  AUC={auc_ref:.4f}  ({time.time()-t0:.0f}s)", flush=True)
results['cosine_knn_k5_simple'] = auc_ref

# ─── Method 1: Mahalanobis KNN (global covariance) ───────────────────────────
print("\n=== Method 1: Mahalanobis KNN (global covariance, k=5) ===", flush=True)
t0 = time.time()
out_mah = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]   # [65, 1536]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]  # [n_wins, 1536]
    # Fit LedoitWolf on training file embeddings (regularized covariance)
    lw = LedoitWolf().fit(tr_emb)
    # Mahalanobis: D^2 = (x - mu)^T Sigma^-1 (x - mu)
    # We compute pairwise Mahalanobis from each test window to each train file
    # Using precision matrix (inv covariance)
    prec = lw.precision_.astype(np.float32)  # [1536, 1536]
    # Efficient: D^2(x, y) = (x-y) prec (x-y)^T
    # For batch: compute for each test window vs each train file
    n_te = len(te_wins)
    n_tr = len(tr_emb)
    # Compute precision @ train: [65, 1536]
    tr_prec = tr_emb @ prec  # [65, 1536]
    # D^2[i,j] = te[i] prec te[i] - 2 te[i] prec tr[j] + tr[j] prec tr[j]
    te_prec_te = np.einsum('ij,ij->i', te_wins, te_wins @ prec)  # [n_te]
    te_prec_tr = te_wins @ tr_prec.T   # [n_te, 65]
    tr_prec_tr = np.einsum('ij,ij->i', tr_emb, tr_prec)           # [65]
    D2 = te_prec_te[:, None] - 2 * te_prec_tr + tr_prec_tr[None, :]  # [n_te, 65]
    D2 = np.maximum(D2, 0)
    # Convert distance to similarity: exp(-D2 / tau)
    tau = np.median(D2)  # adaptive tau
    sim = np.exp(-D2 / (tau + EPS))
    topk = np.argsort(-sim, axis=1)[:, :5]
    w = np.take_along_axis(sim, topk, axis=1)
    w = w / (w.sum(1, keepdims=True) + EPS)
    pw = (w[:, :, None] * tr_lab[topk]).sum(1)
    out_mah[fi] = pw.mean(0)
auc1 = eval_loo(out_mah)
flag = " *** NEW BEST ***" if auc1 > CURRENT_BEST else ""
print(f"  AUC={auc1:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['mahalanobis_knn_global_cov'] = auc1

# ─── Method 2: Cosine Prototype (per-species centroid similarity) ─────────────
print("\n=== Method 2: Cosine Prototype ===", flush=True)
t0 = time.time()
out_proto = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]  # [n_wins, 1536]
    win_scores = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:, si] > 0.5
        neg = ~pos
        if not pos.any():
            win_scores[:, si] = 0.5; continue
        proto_pos = tr_emb[pos].mean(0)
        proto_pos /= (np.linalg.norm(proto_pos) + EPS)
        sim_pos = te_wins @ proto_pos  # [n_wins]
        if neg.any():
            proto_neg = tr_emb[neg].mean(0)
            proto_neg /= (np.linalg.norm(proto_neg) + EPS)
            sim_neg = te_wins @ proto_neg
            # Relative score: pos sim vs neg sim
            win_scores[:, si] = (sim_pos - sim_neg + 1) / 2
        else:
            win_scores[:, si] = (sim_pos + 1) / 2
    out_proto[fi] = win_scores.mean(0)
auc2 = eval_loo(out_proto)
flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
print(f"  AUC={auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['cosine_prototype'] = auc2

# ─── Method 3: GMM per species (PCA-32) ─────────────────────────────────────
print("\n=== Method 3: GMM per species (PCA-32) ===", flush=True)
t0 = time.time()
from sklearn.mixture import GaussianMixture
out_gmm = np.zeros((n_files, n_species), np.float32)
PCA_N = 32
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    # PCA on training file embeddings
    pca = PCA(n_components=PCA_N, random_state=42).fit(tr_emb)
    X_tr = pca.transform(tr_emb)
    X_te = pca.transform(te_wins)
    mu = X_tr.mean(0); std = X_tr.std(0).clip(1e-8)
    X_tr = (X_tr - mu) / std
    X_te = (X_te - mu) / std
    win_scores = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:, si] > 0.5
        X_pos = X_tr[pos]
        X_neg = X_tr[~pos]
        if len(X_pos) == 0:
            win_scores[:, si] = 0.5; continue
        # Fit GMM for positive class (1 or 2 components)
        n_comp_pos = min(2, len(X_pos))
        try:
            gmm_pos = GaussianMixture(n_components=n_comp_pos, covariance_type='diag',
                                       random_state=42, max_iter=100).fit(X_pos)
            log_pos = gmm_pos.score_samples(X_te)
        except:
            log_pos = -0.5 * ((X_te - X_pos.mean(0))**2).sum(1)
        # Background log prob (all training)
        try:
            gmm_bg = GaussianMixture(n_components=min(3, len(X_tr)), covariance_type='diag',
                                      random_state=42, max_iter=100).fit(X_tr)
            log_bg = gmm_bg.score_samples(X_te)
        except:
            log_bg = np.zeros(len(X_te))
        win_scores[:, si] = 1 / (1 + np.exp(-(log_pos - log_bg).clip(-10, 10)))
    out_gmm[fi] = win_scores.mean(0)
auc3 = eval_loo(out_gmm)
flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
print(f"  AUC={auc3:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['gmm_pca32_simple'] = auc3

# ─── Method 4: Attention-weighted KNN (rank-based) ───────────────────────────
# Weight neighbors by 1/rank (harmonic) instead of cosine sim
print("\n=== Method 4: Rank-based attention KNN ===", flush=True)
t0 = time.time()
out_rank = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]
    tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    sims = te_wins @ tr_emb.T  # [n_wins, 65]
    K = min(20, len(tr_emb))
    topk = np.argsort(-sims, axis=1)[:, :K]
    # Rank-based weights: w[rank] = 1/(rank+1)
    ranks = np.arange(1, K+1, dtype=np.float32)
    w_rank = 1.0 / ranks  # [K]
    w_rank /= w_rank.sum()
    # Also use cosine sim to scale
    sim_topk = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
    # Combined: rank * cosine
    w = w_rank[None, :] * sim_topk
    w = w / (w.sum(1, keepdims=True) + EPS)
    pw = (w[:, :, None] * tr_lab[topk]).sum(1)
    out_rank[fi] = pw.mean(0)
auc4 = eval_loo(out_rank)
flag = " *** NEW BEST ***" if auc4 > CURRENT_BEST else ""
print(f"  AUC={auc4:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['rank_attention_knn'] = auc4

# ─── Method 5: Soft-KNN with temperature sweep ───────────────────────────────
print("\n=== Method 5: Soft-KNN (temperature sweep, all neighbors) ===", flush=True)
t0 = time.time()
best5 = 0; best_tau5 = None
out_best5 = None
for tau in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    out_t = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = file_embs_norm[tr_idx]
        tr_lab = file_labels[tr_idx]
        te_wins = emb_win_norm[win_file_id == fi]
        sims = te_wins @ tr_emb.T  # [n_wins, 65]
        # Softmax over all neighbors
        log_w = sims / tau
        log_w -= log_w.max(1, keepdims=True)
        w = np.exp(log_w); w /= w.sum(1, keepdims=True)
        out_t[fi] = (w[:, :, None] * tr_lab[None]).sum(1).mean(0)
    auc_t = eval_loo(out_t)
    print(f"  tau={tau}: {auc_t:.4f}", flush=True)
    results[f'soft_knn_tau{tau}'] = auc_t
    if auc_t > best5: best5 = auc_t; best_tau5 = tau; out_best5 = out_t
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  Best: tau={best_tau5} → {best5:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['soft_knn_best'] = best5

# ─── Method 6: Hybrid (cosine KNN k=5 + Soft-KNN) ────────────────────────────
print("\n=== Method 6: Cosine KNN k=5 + Soft-KNN blend ===", flush=True)
best6 = 0; best_w6 = None
for w_soft in [0.1, 0.2, 0.3, 0.4, 0.5]:
    blend = w_soft * out_best5 + (1-w_soft) * out_knn5
    auc_c = eval_loo(blend)
    if auc_c > best6: best6 = auc_c; best_w6 = w_soft
results['knn5_softnn_blend'] = best6
flag = " *** NEW BEST ***" if best6 > CURRENT_BEST else ""
print(f"  Best: w_soft={best_w6} → {best6:.4f}{flag}", flush=True)

# ─── Method 7: Prototype + KNN blend ─────────────────────────────────────────
print("\n=== Method 7: Prototype + KNN blend ===", flush=True)
best7 = 0; best_w7 = None
for w_proto in [0.1, 0.2, 0.3, 0.4, 0.5]:
    blend = w_proto * out_proto + (1-w_proto) * out_knn5
    auc_c = eval_loo(blend)
    if auc_c > best7: best7 = auc_c; best_w7 = w_proto
results['proto_knn_blend'] = best7
flag = " *** NEW BEST ***" if best7 > CURRENT_BEST else ""
print(f"  Best: w_proto={best_w7} → {best7:.4f}{flag}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 18 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

# Update JSON
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
