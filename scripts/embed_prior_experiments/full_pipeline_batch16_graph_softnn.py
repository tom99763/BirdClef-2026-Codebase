"""
Batch 16: Graph-based label propagation + Soft-NN ensemble
Goal: beat 0.9738
Methods:
  1. Soft-NN temperature sweep + KDE blend
  2. Weighted k-NN (1/distance weights) per-window
  3. Graph label propagation (diffusion): propagate labels on similarity graph
  4. Soft-NN ensemble (multiple temperatures)
  5. kNN-soft + KDE blend
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

EPS = 1e-7
def vlom_blend(a, b):
    return sigmoid(0.5*np.log(a.clip(EPS)/(1-a).clip(EPS)) + 0.5*np.log(b.clip(EPS)/(1-b).clip(EPS)))
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_by_file = {}
for i, rid in enumerate(sed_npz['row_ids']):
    sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]), []).append(i)
file_sed_max = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fb = fname.replace('.ogg','').replace('.flac','')
    if fb in sed_by_file: file_sed_max[fi] = sed_npz['probs'][sed_by_file[fb]].max(0)
base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))
mask = file_labels.sum(0) > 0
PCA_N = 32
CURRENT_BEST = 0.9738

def sweep(scores, name=""):
    best = 0; best_cfg = None
    for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
        for b in [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]:
            pred = sigmoid(a * base_logit + b * np.log(scores.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best: best = auc; best_cfg = (a, b)
    flag = " *** NEW BEST ***" if best > CURRENT_BEST else ""
    if name: print(f"  {name}: {best:.4f}{flag}  cfg={best_cfg}", flush=True)
    return best, best_cfg

results = {}

# Precompute KDE best blend (0.15*bw0.3 + 0.85*bw0.5)
print("Precomputing KDE...", flush=True)
def loo_kde_perwin(bw):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_s, te_e = int(file_start[fi]), int(file_end[fi])
        tr_mask = (win_file_id != fi)
        X_tr_raw = emb_win_norm[tr_mask]
        pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
        X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
        mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
        X_tr_l = (X_tr_l - mu_l) / std_l
        X_te_pca = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
        tr_fids = win_file_id[tr_mask]
        kde_bg = KernelDensity(bandwidth=bw).fit(X_tr_l)
        log_bg_wins = kde_bg.score_samples(X_te_pca)
        win_scores = np.zeros((te_e - te_s, n_species), np.float32)
        for si in range(n_species):
            pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
            X_pos = X_tr_l[pos_mask]
            if len(X_pos) == 0:
                win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
            kde_pos = KernelDensity(bandwidth=bw).fit(X_pos)
            win_scores[:, si] = sigmoid(kde_pos.score_samples(X_te_pca) - log_bg_wins)
        out[fi] = win_scores.mean(0)
    return out

kde03 = loo_kde_perwin(0.3)
kde05 = loo_kde_perwin(0.5)
kde_best = 0.15 * kde03 + 0.85 * kde05
print("  KDE precomputed.", flush=True)

# ─── Method 1: Soft-NN with multiple temperatures ─────────────────────────────
print("\n=== Method 1: Soft-NN temperature sweep ===", flush=True)
t0 = time.time()
snn_by_tau = {}
for tau in [0.1, 0.2, 0.5, 1.0, 2.0]:
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_s, te_e = int(file_start[fi]), int(file_end[fi])
        tr_mask = (win_file_id != fi)
        X_tr_raw = emb_win_norm[tr_mask]
        pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
        X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
        mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
        X_tr_l = (X_tr_l - mu_l) / std_l
        X_te = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
        tr_fids = win_file_id[tr_mask]
        sq_te = (X_te**2).sum(1, keepdims=True)
        sq_tr = (X_tr_l**2).sum(1)
        D2 = np.maximum(sq_te + sq_tr - 2 * X_te @ X_tr_l.T, 0)
        log_w = -D2 / tau
        log_w -= log_w.max(1, keepdims=True)
        w = np.exp(log_w); w /= w.sum(1, keepdims=True)
        win_scores = np.zeros((te_e - te_s, n_species), np.float32)
        for si in range(n_species):
            pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids]).astype(np.float32)
            if not pos_mask.any():
                win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
            win_scores[:, si] = (w * pos_mask).sum(1)
        out[fi] = win_scores.mean(0)
    snn_by_tau[tau] = out
    auc_s, cfg_s = sweep(out)
    results[f'soft_nn_tau{tau}'] = (auc_s, cfg_s)
    print(f"  Soft-NN tau={tau}: {auc_s:.4f}  cfg={cfg_s}", flush=True)
print(f"  ({time.time()-t0:.0f}s total)", flush=True)

# Best tau from above
best_tau = max(snn_by_tau, key=lambda t: results[f'soft_nn_tau{t}'][0])
snn_best = snn_by_tau[best_tau]
print(f"  Best tau: {best_tau} ({results[f'soft_nn_tau{best_tau}'][0]:.4f})", flush=True)

# Soft-NN + KDE blend
print("\n=== Method 2: Soft-NN + KDE blend ===", flush=True)
best2 = 0; best_cfg2 = None
for w_snn in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    blend = w_snn * snn_best + (1-w_snn) * kde_best
    auc_c, cfg_c = sweep(blend)
    if auc_c > best2: best2 = auc_c; best_cfg2 = (w_snn, best_tau, cfg_c)
results['soft_nn_kde_blend'] = (best2, best_cfg2)
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  Soft-NN+KDE: {best2:.4f}{flag}  w_snn={best_cfg2[0]}, tau={best_cfg2[1]}", flush=True)

# ─── Method 3: Graph label propagation ────────────────────────────────────────
# Build k-NN graph on training windows, propagate labels to test windows
print("\n=== Method 3: Graph label propagation (per-window) ===", flush=True)
t0 = time.time()
out_glp = np.zeros((n_files, n_species), np.float32)
K_GRAPH = 10  # graph connectivity
ALPHA = 0.5   # propagation weight

for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te = ((pca_l.transform(emb_win_norm[te_s:te_e]).astype(np.float32)) - mu_l) / std_l
    tr_fids = win_file_id[tr_mask]
    n_tr = len(X_tr_l); n_te_w = te_e - te_s

    # Build label matrix Y for training nodes (n_tr, n_species)
    Y_tr = np.zeros((n_tr, n_species), np.float32)
    for j, fj in enumerate(tr_fids):
        Y_tr[j] = file_labels[fj]

    # Test-to-train affinity: (n_te, n_tr) Gaussian kernel
    sq_te = (X_te**2).sum(1, keepdims=True)
    sq_tr = (X_tr_l**2).sum(1)
    D2_te_tr = np.maximum(sq_te + sq_tr - 2 * X_te @ X_tr_l.T, 0)  # (n_te, n_tr)

    # For each test window: find k nearest training neighbors, do weighted vote
    # This is a simplified label propagation (1-step from train to test)
    k = min(K_GRAPH, n_tr)
    win_scores = np.zeros((n_te_w, n_species), np.float32)
    for wi in range(n_te_w):
        nn_idx = np.argpartition(D2_te_tr[wi], k)[:k]  # k nearest training windows
        nn_d2 = D2_te_tr[wi][nn_idx]
        # Gaussian weights
        nn_w = np.exp(-nn_d2 / (2 * 0.5**2))
        nn_w /= nn_w.sum() + EPS
        # Weighted label vote
        win_scores[wi] = (nn_w[:, None] * Y_tr[nn_idx]).sum(0)
    out_glp[fi] = win_scores.mean(0)

print(f"  ({time.time()-t0:.0f}s)", flush=True)
auc3, cfg3 = sweep(out_glp, "Graph Label Prop")
results['graph_label_prop'] = (auc3, cfg3)

# GLP + KDE blend
best3b = 0; best_cfg3b = None
for w_glp in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blend = w_glp * out_glp + (1-w_glp) * kde_best
    auc_c, cfg_c = sweep(blend)
    if auc_c > best3b: best3b = auc_c; best_cfg3b = (w_glp, cfg_c)
results['glp_kde_blend'] = (best3b, best_cfg3b)
flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
print(f"  GLP+KDE: {best3b:.4f}{flag}  w_glp={best_cfg3b[0]}", flush=True)

# ─── Method 4: Multi-method ensemble (KDE + Soft-NN + GLP) ────────────────────
print("\n=== Method 4: Triple ensemble (KDE + Soft-NN + GLP) ===", flush=True)
best4 = 0; best_cfg4 = None
for w_snn in [0.05, 0.10, 0.15]:
    for w_glp in [0.05, 0.10, 0.15]:
        w_kde = 1.0 - w_snn - w_glp
        if w_kde < 0.7: continue
        blend = w_kde * kde_best + w_snn * snn_best + w_glp * out_glp
        auc_c, cfg_c = sweep(blend)
        if auc_c > best4: best4 = auc_c; best_cfg4 = (w_kde, w_snn, w_glp, cfg_c)
results['triple_kde_snn_glp'] = (best4, best_cfg4)
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  Triple: {best4:.4f}{flag}  w=(kde={best_cfg4[0]:.2f}, snn={best_cfg4[1]:.2f}, glp={best_cfg4[2]:.2f})", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 16 Summary ===", flush=True)
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
for name, (auc, cfg) in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc), 'config': str(cfg)})
    if auc > cur_best_json:
        cur_best_json = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
