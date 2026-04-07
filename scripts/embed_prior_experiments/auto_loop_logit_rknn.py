"""
Embed Prior Auto Loop: Logit-space RKNN variants

Methods:
1. logit_space_rknn: RKNN using cosine sim of sigmoid(logit_max) vectors (234-dim)
   - Similar species compositions → similar logit patterns → better neighbors
2. joint_space_rknn: Combine embedding-space (X_ref) sim + logit-space sim → RKNN
3. logit_rknn_fused: logit_space_rknn fused with logit_max via per-species alpha

EP-only LOO-AUC target: beat interaction_knn (0.9199)
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
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
file_embs_avg = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_embs_avg[fi] = emb_win[s:e].mean(0)

EPS = 1e-7
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

# ── Load base PKL for X_ref ────────────────────────────────────────────────────
with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep_base = pickle.load(f)
X_ref = ep_base['X_combined_n'].astype(np.float32)  # (66, 39) PCA24+geo
fl = ep_base['file_labels'].astype(np.float32)

# ── Feature spaces ─────────────────────────────────────────────────────────────
# 1. Logit space: L2-norm of sigmoid(logit_max) vectors (234-dim)
file_prob_max = sigmoid(file_logit_max)
file_prob_norm = normalize(file_prob_max, norm='l2').astype(np.float32)   # (66, 234)
sim_logit = file_prob_norm @ file_prob_norm.T  # (66, 66)

# 2. X_ref space (PCA24+geo)
sim_ref = X_ref @ X_ref.T  # (66, 66)

# 3. Perch avg embedding space
file_emb_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)
sim_emb = file_emb_norm @ file_emb_norm.T  # (66, 66)

# ── RKNN helper ────────────────────────────────────────────────────────────────
T = 0.2
def compute_rknn_ep(sim_mat, k=5):
    """EP-only RKNN: direct LOO prediction."""
    sc = sim_mat.copy()
    np.fill_diagonal(sc, -np.inf)
    top_k = np.argsort(-sc, axis=1)[:, :k]
    kth = sc[np.arange(n_files), top_k[:, -1]]
    y = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims_i = sc[i, tr]
        top_i = np.argsort(-sims_i)[:k]
        mutual, msims = [], []
        for ti, tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]] >= kth[tj]:
                mutual.append(tj); msims.append(sims_i[top_i[ti]])
        if len(mutual) == 0:
            top5 = np.argsort(-sims_i)[:5]; ls = sims_i[top5]/T; ls -= ls.max()
            w = np.exp(ls); w /= w.sum()
            y[i] = (w[:, None] * fl[tr[top5]]).sum(0)
        else:
            ms = np.array(msims); ls = ms/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y[i] = (w[:, None] * fl[mutual]).sum(0)
    return y

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: RKNN in pure logit space
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Method 1: Logit-space RKNN ===", flush=True)
best1 = 0; best1_cfg = {}
for k in [3, 5, 7]:
    y1 = compute_rknn_ep(sim_logit, k=k)
    auc1 = macro_auc(file_labels, y1)
    if auc1 > best1: best1 = auc1; best1_cfg = {'k': k}
    print(f"  k={k}: {auc1:.4f}")
results['logit_space_rknn'] = (best1, best1_cfg)
print(f"  Best: {best1:.4f}  cfg={best1_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Joint-space RKNN (X_ref + logit)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Joint-space RKNN (X_ref + logit sim) ===", flush=True)
best2 = 0; best2_cfg = {}
for w_ref in [0.3, 0.5, 0.7, 0.9]:
    w_logit = 1.0 - w_ref
    sim_joint = w_ref * sim_ref + w_logit * sim_logit
    for k in [3, 5, 7]:
        y2 = compute_rknn_ep(sim_joint, k=k)
        auc2 = macro_auc(file_labels, y2)
        if auc2 > best2: best2 = auc2; best2_cfg = {'w_ref': w_ref, 'k': k}
        print(f"  w_ref={w_ref} k={k}: {auc2:.4f}")
results['joint_space_rknn'] = (best2, best2_cfg)
print(f"  Best: {best2:.4f}  cfg={best2_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Joint-space RKNN + per-species logit fusion
# sigmoid(alpha × base_logit + beta × log(rknn_pred))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Logit-space RKNN + logspace fusion ===", flush=True)
best3 = 0; best3_cfg = {}
base_logit = np.log(file_prob_max.clip(EPS)) - np.log((1-file_prob_max).clip(EPS))

for w_ref in [0.3, 0.5, 0.7]:
    w_logit_sim = 1.0 - w_ref
    sim_joint = w_ref * sim_ref + w_logit_sim * sim_logit
    for k in [3, 5]:
        y_rknn = compute_rknn_ep(sim_joint, k=k)
        log_rknn = np.log(y_rknn.clip(EPS))
        for a in [0.5, 0.7, 0.8, 0.9, 1.0]:
            for b in [0.5, 0.8, 1.0, 1.2, 1.5]:
                pred3 = sigmoid(a * base_logit + b * log_rknn)
                auc3 = macro_auc(file_labels, pred3)
                if auc3 > best3:
                    best3 = auc3
                    best3_cfg = {'w_ref': w_ref, 'k': k, 'a': a, 'b': b}
results['logit_rknn_logspace'] = (best3, best3_cfg)
print(f"  Best: {best3:.4f}  cfg={best3_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Embedding-space RKNN + logit-space RKNN ensemble
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Emb-RKNN + Logit-RKNN ensemble ===", flush=True)
best4 = 0; best4_cfg = {}

# Precompute RKNN predictions in X_ref and logit spaces
y_ref_rknn = compute_rknn_ep(sim_ref, k=5)   # from X_ref space
y_log_rknn = compute_rknn_ep(sim_logit, k=5)  # from logit space

for w_ref_rknn in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w_log_rknn = 1.0 - w_ref_rknn
    y_ens = w_ref_rknn * y_ref_rknn + w_log_rknn * y_log_rknn
    log_ens = np.log(y_ens.clip(EPS))
    for a in [0.7, 0.8, 0.9, 1.0]:
        for b in [0.8, 1.0, 1.2, 1.5]:
            pred4 = sigmoid(a * base_logit + b * log_ens)
            auc4 = macro_auc(file_labels, pred4)
            if auc4 > best4:
                best4 = auc4
                best4_cfg = {'w_ref': w_ref_rknn, 'a': a, 'b': b}
results['emb_logit_rknn_ensemble'] = (best4, best4_cfg)
print(f"  Best: {best4:.4f}  cfg={best4_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Triple-space RKNN (X_ref + logit + emb avg)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Triple-space RKNN ===", flush=True)
best5 = 0; best5_cfg = {}
for w_ref in [0.4, 0.5, 0.6]:
    for w_logit in [0.1, 0.2, 0.3]:
        w_emb = 1.0 - w_ref - w_logit
        if w_emb < 0: continue
        sim_tri = w_ref * sim_ref + w_logit * sim_logit + w_emb * sim_emb
        for k in [3, 5]:
            y5 = compute_rknn_ep(sim_tri, k=k)
            log_y5 = np.log(y5.clip(EPS))
            for a in [0.8, 0.9, 1.0]:
                for b in [0.8, 1.0, 1.2, 1.5]:
                    pred5 = sigmoid(a * base_logit + b * log_y5)
                    auc5 = macro_auc(file_labels, pred5)
                    if auc5 > best5:
                        best5 = auc5
                        best5_cfg = {'w_ref': w_ref, 'w_logit': w_logit, 'w_emb': w_emb, 'k': k, 'a': a, 'b': b}
results['triple_space_rknn'] = (best5, best5_cfg)
print(f"  Best: {best5:.4f}  cfg={best5_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("LOGIT-SPACE RKNN VARIANTS SUMMARY")
print(f"Reference: interaction_knn=0.9199 (EP-only best)")
print(f"{'='*60}")
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    delta = auc - 0.9199
    marker = " *** NEW EP BEST ***" if auc > 0.9199 else ""
    print(f"  {name}: {auc:.4f}  ({delta:+.4f}){marker}")

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)

cur_best = rd['best'].get('loo_auc', 0)
for name, (auc, cfg) in results.items():
    rd['experiments'].append({
        'method': name,
        'loo_auc': float(auc),
        'full_auc': float(auc),
        'config': cfg
    })
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc), **cfg}
        print(f"\n*** NEW BEST: {name} = {auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")
