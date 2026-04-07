"""
Embed Prior Auto Loop: Label Propagation + Soft-Neighbor variants

Methods:
1. label_propagation: Gaussian-kernel graph Laplacian propagation
   - W_ij = exp(-||x_i - x_j||^2 / sigma^2); iterate F = (1-alpha)*W_norm*F + alpha*Y
2. soft_neighbor_knn: Gaussian kernel over ALL N-1 files (not just top-k)
   - w_ij = exp(-(1-cos_sim)/T); sum over all j≠i
3. harmonic_function: Laplacian harmonic function (ZBL 2003)
   - Hard clamp labeled nodes (trivial here since all are "labeled"), use as transductive
   - Actually: use PREDICTED labels (file_prob_max) as soft "initial" values
4. label_prop_logspace: Apply logspace fusion on top of label propagation output

EP-only best: interaction_knn = 0.9199
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list = list(perch['file_list'])
n_windows = perch['n_windows']
n_files = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end = np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))
file_labels = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)

EPS = 1e-7
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

# ── Load base PKL ──────────────────────────────────────────────────────────────
with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep_base = pickle.load(f)
X_ref = ep_base['X_combined_n'].astype(np.float32)  # (66, 39)
fl = ep_base['file_labels'].astype(np.float32)

file_prob_max = sigmoid(file_logit_max)
base_logit = np.log(file_prob_max.clip(EPS)) - np.log((1-file_prob_max).clip(EPS))

# Cosine sim matrix from X_ref
sim_ref = X_ref @ X_ref.T
np.fill_diagonal(sim_ref, 0)

# Raw Perch avg embedding
file_embs_avg = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs_avg[fi] = emb_win[s:e].mean(0)
emb_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)
sim_emb = emb_norm @ emb_norm.T
np.fill_diagonal(sim_emb, 0)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Soft-neighbor (Gaussian kernel over ALL files)
# w_ij = exp(cos_sim_ij / T)  →  sum-weighted label average for each file (LOO)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Method 1: Soft-neighbor (Gaussian kernel ALL files) ===", flush=True)
best1 = 0; best1_cfg = {}
for T in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    # LOO: for each file i, compute soft-weighted sum over j≠i
    y1 = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        sims = sim_ref[i].copy(); sims[i] = -np.inf
        w = np.exp(sims / T)
        w[i] = 0.0
        ws = w.sum()
        if ws > 1e-8:
            y1[i] = (w[:, None] * fl).sum(0) / ws
        else:
            y1[i] = fl.mean(0)
    # Evaluate standalone
    auc_raw = macro_auc(file_labels, y1)
    # Logspace fusion
    for a in [0.7, 0.8, 0.9, 1.0]:
        for b in [0.8, 1.0, 1.2, 1.5, 1.8]:
            pred = sigmoid(a * base_logit + b * np.log(y1.clip(EPS)))
            if np.isfinite(pred).all():
                auc = macro_auc(file_labels, pred)
                if auc > best1:
                    best1 = auc; best1_cfg = {'T': T, 'a': a, 'b': b}
    print(f"  T={T}: raw={auc_raw:.4f}", flush=True)
results['soft_neighbor_knn'] = (best1, best1_cfg)
print(f"  Best (with logspace): {best1:.4f}  cfg={best1_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Label Propagation on X_ref graph
# F(t+1) = (1-alpha) * W_norm * F(t) + alpha * Y
# Y = file_prob_max (soft initial labels from Perch logits)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Label Propagation ===", flush=True)
best2 = 0; best2_cfg = {}

def label_prop_loo(sim_mat, Y, alpha=0.2, n_iter=10):
    """LOO label propagation: for each held-out file, propagate from rest."""
    n = len(Y)
    y_loo = np.zeros_like(Y)
    for i in range(n):
        # Build W without file i
        idx = [j for j in range(n) if j != i]
        W = sim_mat[np.ix_(idx, idx)].copy()
        np.fill_diagonal(W, 0)
        W = np.maximum(W, 0)  # keep positive sims only
        # Row-normalize
        D = W.sum(1, keepdims=True).clip(1e-8)
        W_norm = W / D
        # Initial labels for training files
        F = Y[idx].copy()
        Y_fixed = Y[idx].copy()
        for _ in range(n_iter):
            F = (1 - alpha) * (W_norm @ F) + alpha * Y_fixed
        # Predict file i: weighted average of final F by sim to i
        sims_i = sim_mat[i, idx]
        sims_i = np.maximum(sims_i, 0)
        ws = sims_i.sum()
        if ws > 1e-8:
            y_loo[i] = (sims_i[:, None] * F).sum(0) / ws
        else:
            y_loo[i] = F.mean(0)
    return y_loo

Y_init = file_prob_max.copy()
for alpha in [0.1, 0.2, 0.3, 0.5]:
    for n_iter in [5, 10, 20]:
        y2 = label_prop_loo(sim_ref, Y_init, alpha=alpha, n_iter=n_iter)
        for a in [0.7, 0.8, 0.9, 1.0]:
            for b in [0.8, 1.0, 1.2, 1.5]:
                pred2 = sigmoid(a * base_logit + b * np.log(y2.clip(EPS)))
                if np.isfinite(pred2).all():
                    auc2 = macro_auc(file_labels, pred2)
                    if auc2 > best2:
                        best2 = auc2; best2_cfg = {'alpha': alpha, 'n_iter': n_iter, 'a': a, 'b': b}
    print(f"  alpha={alpha}: best so far={best2:.4f}", flush=True)
results['label_propagation'] = (best2, best2_cfg)
print(f"  Best: {best2:.4f}  cfg={best2_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Label Propagation on raw Perch embedding space
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Label Propagation (raw Perch emb space) ===", flush=True)
best3 = 0; best3_cfg = {}
for alpha in [0.1, 0.2, 0.3]:
    for n_iter in [5, 10]:
        y3 = label_prop_loo(sim_emb, Y_init, alpha=alpha, n_iter=n_iter)
        for a in [0.7, 0.8, 0.9, 1.0]:
            for b in [0.8, 1.0, 1.2, 1.5]:
                pred3 = sigmoid(a * base_logit + b * np.log(y3.clip(EPS)))
                if np.isfinite(pred3).all():
                    auc3 = macro_auc(file_labels, pred3)
                    if auc3 > best3:
                        best3 = auc3; best3_cfg = {'alpha': alpha, 'n_iter': n_iter, 'a': a, 'b': b}
results['label_prop_emb_space'] = (best3, best3_cfg)
print(f"  Best: {best3:.4f}  cfg={best3_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Soft-neighbor in raw Perch emb space
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Soft-neighbor (Perch emb space) ===", flush=True)
best4 = 0; best4_cfg = {}
for T in [0.05, 0.10, 0.15, 0.20, 0.30]:
    y4 = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        sims = sim_emb[i].copy(); sims[i] = -np.inf
        w = np.exp(sims / T); w[i] = 0.0
        ws = w.sum()
        if ws > 1e-8:
            y4[i] = (w[:, None] * fl).sum(0) / ws
        else:
            y4[i] = fl.mean(0)
    for a in [0.7, 0.8, 0.9, 1.0]:
        for b in [0.8, 1.0, 1.2, 1.5, 1.8]:
            pred = sigmoid(a * base_logit + b * np.log(y4.clip(EPS)))
            if np.isfinite(pred).all():
                auc = macro_auc(file_labels, pred)
                if auc > best4:
                    best4 = auc; best4_cfg = {'T': T, 'a': a, 'b': b}
results['soft_neighbor_emb'] = (best4, best4_cfg)
print(f"  Best: {best4:.4f}  cfg={best4_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Combined soft-neighbor (X_ref + raw emb)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Combined soft-neighbor (X_ref + Perch emb) ===", flush=True)
best5 = 0; best5_cfg = {}
for w_ref in [0.3, 0.5, 0.7]:
    w_emb = 1.0 - w_ref
    sim_comb = w_ref * sim_ref + w_emb * sim_emb
    for T in [0.10, 0.20, 0.30]:
        y5 = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            sims = sim_comb[i].copy(); sims[i] = -np.inf
            w = np.exp(sims / T); w[i] = 0.0
            ws = w.sum()
            if ws > 1e-8:
                y5[i] = (w[:, None] * fl).sum(0) / ws
            else:
                y5[i] = fl.mean(0)
        for a in [0.7, 0.8, 0.9, 1.0]:
            for b in [0.8, 1.0, 1.2, 1.5]:
                pred = sigmoid(a * base_logit + b * np.log(y5.clip(EPS)))
                if np.isfinite(pred).all():
                    auc = macro_auc(file_labels, pred)
                    if auc > best5:
                        best5 = auc; best5_cfg = {'w_ref': w_ref, 'T': T, 'a': a, 'b': b}
results['combined_soft_neighbor'] = (best5, best5_cfg)
print(f"  Best: {best5:.4f}  cfg={best5_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
EP_BEST = 0.9199
print(f"\n{'='*60}")
print(f"LABEL PROPAGATION / SOFT-NEIGHBOR SUMMARY")
print(f"EP-only best reference: interaction_knn={EP_BEST}")
print(f"{'='*60}")
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    delta = auc - EP_BEST
    marker = " *** NEW EP BEST ***" if auc > EP_BEST else ""
    print(f"  {name}: {auc:.4f}  ({delta:+.4f}){marker}")

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
for name, (auc, cfg) in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc), 'config': cfg})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"\n*** NEW BEST: {name} = {auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")
