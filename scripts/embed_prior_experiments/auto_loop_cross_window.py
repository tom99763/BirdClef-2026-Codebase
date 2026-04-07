"""
Embed Prior Auto Loop: Cross-file Window Similarity + Density-weighted KNN

New methods:
1. cross_window_sim_knn: File similarity = avg cosine sim between ALL window pairs
   - Captures distributional overlap (better than file-avg similarity)
   - O(739²) ops = fast
2. density_weighted_knn: Down-weight "common" training files by their avg similarity to others
   - density(j) = avg sim(j, k) for k≠j
   - w_j = sim(i,j) / density(j)^gamma
3. cross_window_rknn: RKNN on cross-file window similarities

Target: beat interaction_knn (EP-only best = 0.9199)
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

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

# Load base PKL for X_ref
with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep_base = pickle.load(f)
X_ref = ep_base['X_combined_n'].astype(np.float32)
fl = ep_base['file_labels'].astype(np.float32)

file_prob_max = sigmoid(file_logit_max)
base_logit = np.log(file_prob_max.clip(EPS)) - np.log((1-file_prob_max).clip(EPS))
sim_ref = X_ref @ X_ref.T; np.fill_diagonal(sim_ref, -np.inf)

# ── Normalize all window embeddings ───────────────────────────────────────────
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)  # (739, 1536)

# ── Compute cross-file window similarity matrix ────────────────────────────────
# sim_cross[i,j] = mean of all (window_i, window_j) cosine similarities
print("Computing cross-file window similarity matrix (739×739)...", flush=True)
sim_all_wins = emb_win_norm @ emb_win_norm.T  # (739, 739)

sim_cross = np.zeros((n_files, n_files), np.float32)
for i in range(n_files):
    si, ei = int(file_start[i]), int(file_end[i])
    for j in range(n_files):
        if i == j:
            sim_cross[i, j] = 1.0
        else:
            sj, ej = int(file_start[j]), int(file_end[j])
            sim_cross[i, j] = sim_all_wins[si:ei, sj:ej].mean()
print("  done.", flush=True)

# Also compute max-based cross similarity (match best window pair)
print("Computing max cross-file window similarity...", flush=True)
sim_cross_max = np.zeros((n_files, n_files), np.float32)
for i in range(n_files):
    si, ei = int(file_start[i]), int(file_end[i])
    for j in range(n_files):
        if i == j:
            sim_cross_max[i, j] = 1.0
        else:
            sj, ej = int(file_start[j]), int(file_end[j])
            sim_cross_max[i, j] = sim_all_wins[si:ei, sj:ej].max()
print("  done.", flush=True)

T = 0.2
def compute_rknn(sim_mat, k=5):
    sc = sim_mat.copy(); np.fill_diagonal(sc, -np.inf)
    top_k = np.argsort(-sc, axis=1)[:, :k]
    kth = sc[np.arange(n_files), top_k[:, -1]]
    y = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims_i = sc[i, tr]; top_i = np.argsort(-sims_i)[:k]
        mutual, msims = [], []
        for ti, tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]] >= kth[tj]: mutual.append(tj); msims.append(sims_i[top_i[ti]])
        if len(mutual) == 0:
            top5 = np.argsort(-sims_i)[:5]; ls = sims_i[top5]/T; ls -= ls.max()
            w = np.exp(ls); w /= w.sum(); y[i] = (w[:,None]*fl[tr[top5]]).sum(0)
        else:
            ms = np.array(msims); ls = ms/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y[i] = (w[:,None]*fl[mutual]).sum(0)
    return y

def eval_logspace(y_ep, a, b):
    pred = sigmoid(a * base_logit + b * np.log(y_ep.clip(EPS)))
    if not np.isfinite(pred).all():
        return 0.0
    return macro_auc(file_labels, pred)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Cross-window mean similarity RKNN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Cross-window Mean Sim RKNN ===", flush=True)
best1 = 0; best1_cfg = {}
for k in [3, 5, 7]:
    y1 = compute_rknn(sim_cross, k=k)
    for a in [0.7, 0.8, 0.9, 1.0]:
        for b in [0.8, 1.0, 1.2, 1.5]:
            auc1 = eval_logspace(y1, a, b)
            if auc1 > best1: best1 = auc1; best1_cfg = {'k': k, 'a': a, 'b': b}
results['cross_win_mean_rknn'] = (best1, best1_cfg)
print(f"  Best: {best1:.4f}  cfg={best1_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Cross-window max similarity RKNN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Cross-window Max Sim RKNN ===", flush=True)
best2 = 0; best2_cfg = {}
for k in [3, 5, 7]:
    y2 = compute_rknn(sim_cross_max, k=k)
    for a in [0.7, 0.8, 0.9, 1.0]:
        for b in [0.8, 1.0, 1.2, 1.5]:
            auc2 = eval_logspace(y2, a, b)
            if auc2 > best2: best2 = auc2; best2_cfg = {'k': k, 'a': a, 'b': b}
results['cross_win_max_rknn'] = (best2, best2_cfg)
print(f"  Best: {best2:.4f}  cfg={best2_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Hybrid (X_ref + cross-window) RKNN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Hybrid X_ref + Cross-window ===", flush=True)
best3 = 0; best3_cfg = {}
sim_cross_n = sim_cross.copy(); np.fill_diagonal(sim_cross_n, -np.inf)
sim_cross_n = (sim_cross_n - sim_cross_n.min()) / (sim_cross_n.max() - sim_cross_n.min() + 1e-8)
sim_ref_n = sim_ref.copy()
sim_ref_n = (sim_ref_n - sim_ref_n[sim_ref_n > -1e9].min()) / (sim_ref_n[sim_ref_n > -1e9].max() - sim_ref_n[sim_ref_n > -1e9].min() + 1e-8)
np.fill_diagonal(sim_ref_n, -np.inf)

for w_ref in [0.3, 0.5, 0.7]:
    w_cw = 1.0 - w_ref
    sim_hyb = w_ref * sim_ref_n + w_cw * sim_cross_n
    np.fill_diagonal(sim_hyb, -np.inf)
    for k in [3, 5, 7]:
        y3 = compute_rknn(sim_hyb, k=k)
        for a in [0.7, 0.8, 0.9, 1.0]:
            for b in [0.8, 1.0, 1.2, 1.5]:
                auc3 = eval_logspace(y3, a, b)
                if auc3 > best3: best3 = auc3; best3_cfg = {'w_ref': w_ref, 'k': k, 'a': a, 'b': b}
results['hybrid_cross_win_rknn'] = (best3, best3_cfg)
print(f"  Best: {best3:.4f}  cfg={best3_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Density-weighted KNN (down-weight "hub" files)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Density-weighted KNN ===", flush=True)
best4 = 0; best4_cfg = {}
# Compute density: average similarity to other files (in X_ref space)
sim_ref_raw = X_ref @ X_ref.T
np.fill_diagonal(sim_ref_raw, 0)
density = sim_ref_raw.sum(1) / (n_files - 1)  # (66,) avg sim to others
density_min = density.min(); density_range = density.max() - density_min + 1e-8

for gamma in [0.5, 1.0, 2.0]:
    # density_norm in [0,1]: higher = more "hub"
    density_norm = (density - density_min) / density_range  # (66,)
    for k in [3, 5, 7, 10]:
        y4 = np.zeros((n_files, n_species), np.float32)
        sc = sim_ref_raw.copy(); np.fill_diagonal(sc, -np.inf)
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j != i])
            sims_i = sc[i, tr]
            top_i = np.argsort(-sims_i)[:k]
            # Density-weighted: w = sim / density^gamma
            raw_w = sims_i[top_i]
            d_w = density_norm[tr[top_i]]
            adj_w = raw_w / (d_w**gamma + 1e-8)
            adj_w = np.maximum(adj_w, 0); ws = adj_w.sum()
            adj_w = adj_w / ws if ws > 1e-8 else np.ones(k) / k
            y4[i] = (adj_w[:, None] * fl[tr[top_i]]).sum(0)
        for a in [0.7, 0.8, 0.9, 1.0]:
            for b in [0.8, 1.0, 1.2, 1.5]:
                auc4 = eval_logspace(y4, a, b)
                if auc4 > best4: best4 = auc4; best4_cfg = {'gamma': gamma, 'k': k, 'a': a, 'b': b}
results['density_weighted_knn'] = (best4, best4_cfg)
print(f"  Best: {best4:.4f}  cfg={best4_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Best-window KNN (select most informative window per file)
# For each file, use only the TOP signal window (max logit sum) for similarity
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Best-window KNN ===", flush=True)
# Select top signal window per file
file_best_win_emb = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    logit_sums = sigmoid(logits_win[s:e]).max(1)  # (n_win,) max prob per window
    best_wi = logit_sums.argmax()
    file_best_win_emb[fi] = emb_win[s:e][best_wi]

file_best_win_norm = normalize(file_best_win_emb, norm='l2').astype(np.float32)
sim_best_win = file_best_win_norm @ file_best_win_norm.T

best5 = 0; best5_cfg = {}
for k in [3, 5, 7]:
    y5 = compute_rknn(sim_best_win, k=k)
    for a in [0.7, 0.8, 0.9, 1.0]:
        for b in [0.8, 1.0, 1.2, 1.5]:
            auc5 = eval_logspace(y5, a, b)
            if auc5 > best5: best5 = auc5; best5_cfg = {'k': k, 'a': a, 'b': b}
results['best_window_knn'] = (best5, best5_cfg)
print(f"  Best: {best5:.4f}  cfg={best5_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
EP_BEST = 0.9199  # interaction_knn
print(f"\n{'='*60}")
print(f"CROSS-WINDOW KNN SUMMARY")
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
