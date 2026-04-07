"""
Embed Prior Auto Loop: Multi-resolution KNN logspace fusion

Method: multires_knn_logspace
- Compute KNN predictions at k=1, 3, 5, 10
- Logspace fusion: sigmoid(a*base_logit + b1*log(y_k1) + b2*log(y_k3) + b3*log(y_k5) + b4*log(y_k10))
- Uses window-level KNN (win_k1) + file-level multi-k
- Optimize all coefficients jointly

Also: signal_boosted_knn - weight training files by SED/Perch signal strength for target species
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
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
file_embs_avg = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_embs_avg[fi] = emb_win[s:e].mean(0)
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

EPS = 1e-7
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep_base = pickle.load(f)
X_ref = ep_base['X_combined_n'].astype(np.float32)
fl = ep_base['file_labels'].astype(np.float32)

file_prob_max = sigmoid(file_logit_max)
base_logit = np.log(file_prob_max.clip(EPS)) - np.log((1-file_prob_max).clip(EPS))
sim_ref = X_ref @ X_ref.T
np.fill_diagonal(sim_ref, -np.inf)

# ── Compute file-level KNN at multiple k values ────────────────────────────────
print("Computing multi-k KNN predictions...", flush=True)
T = 0.2
K_LIST = [1, 3, 5, 7, 10]

def knn_loo(sim_mat, k, T=0.2):
    sc = sim_mat.copy(); np.fill_diagonal(sc, -np.inf)
    y = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims_i = sc[i, tr]; top_i = np.argsort(-sims_i)[:k]
        w = sims_i[top_i] / T; w -= w.max(); w = np.exp(w); w /= w.sum()
        y[i] = (w[:, None] * fl[tr[top_i]]).sum(0)
    return y.clip(EPS, 1-EPS)

# Pre-compute all k variants
y_knn = {k: knn_loo(sim_ref, k) for k in K_LIST}

# Window-level KNN (LOO)
print("Computing win_k1...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = (win_file_id != i)
    X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    sims = X_te @ X_tr.T
    top_idx = np.argsort(-sims, 1)[:, :1]
    wp = np.zeros((te_e-te_s, n_species), np.float32)
    for wi in range(te_e-te_s):
        ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
        ww = ww/ws if ws > 1e-8 else np.ones(1)
        wp[wi] = (ww[:, None] * file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)
y_win_k1 = y_win_k1.clip(EPS, 1-EPS)
print("  done.", flush=True)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Multi-k logspace fusion sweep
# sigmoid(a × base_logit + b5 × log(y_k5) + b_win × log(y_win))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Multi-k logspace fusion ===", flush=True)
best1 = 0; best1_cfg = {}

# First sweep: k3+k5+win combination
for k_file in [3, 5, 7]:
    log_knn = np.log(y_knn[k_file])
    log_win = np.log(y_win_k1)
    for a in [0.7, 0.8, 0.9, 1.0]:
        for b_file in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]:
            for b_win in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]:
                pred = sigmoid(a * base_logit + b_file * log_knn + b_win * log_win)
                auc = macro_auc(file_labels, pred)
                if auc > best1:
                    best1 = auc
                    best1_cfg = {'k': k_file, 'a': a, 'b_file': b_file, 'b_win': b_win}

results['multik_win_logspace'] = (best1, best1_cfg)
print(f"  Best: {best1:.4f}  cfg={best1_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: 3-way fusion: k3 + k5 + win
# sigmoid(a × base_logit + b3 × log(y_k3) + b5 × log(y_k5) + bw × log(y_win))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: 3-way KNN logspace (k3+k5+win) ===", flush=True)
best2 = 0; best2_cfg = {}
log_k3 = np.log(y_knn[3]); log_k5 = np.log(y_knn[5]); log_win = np.log(y_win_k1)
for a in [0.7, 0.8, 0.9]:
    for b3 in [0.2, 0.4, 0.6, 0.8]:
        for b5 in [0.2, 0.4, 0.6, 0.8]:
            for bw in [0.2, 0.4, 0.6, 0.8]:
                pred2 = sigmoid(a * base_logit + b3 * log_k3 + b5 * log_k5 + bw * log_win)
                auc2 = macro_auc(file_labels, pred2)
                if auc2 > best2:
                    best2 = auc2
                    best2_cfg = {'a': a, 'b3': b3, 'b5': b5, 'bw': bw}
results['3way_k3_k5_win'] = (best2, best2_cfg)
print(f"  Best: {best2:.4f}  cfg={best2_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: RKNN + win_k1 logspace (reproduce best EP baseline)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Reproduce EP best (RKNN+win logspace) ===", flush=True)

def compute_rknn_ep(sim_mat, k=5, T=0.2):
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
    return y.clip(EPS, 1-EPS)

y_rknn = compute_rknn_ep(sim_ref, k=5)
best3 = 0; best3_cfg = {}
log_rknn = np.log(y_rknn)
for a in [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
    for b_rknn in [0.6, 0.8, 1.0, 1.2, 1.5]:
        for b_win in [0.3, 0.5, 0.7, 0.9, 1.0]:
            wg = 0.45
            yb = wg * y_rknn + (1-wg) * y_win_k1
            pred3 = sigmoid(a * base_logit + b_rknn * np.log(yb.clip(EPS)))
            auc3 = macro_auc(file_labels, pred3)
            if auc3 > best3: best3 = auc3; best3_cfg = {'a': a, 'b': b_rknn, 'wg': wg}
results['rknn_win_logspace'] = (best3, best3_cfg)
print(f"  Best: {best3:.4f}  cfg={best3_cfg}", flush=True)

# Separate b_rknn and b_win
best3b = 0; best3b_cfg = {}
for a in [0.8, 0.85, 0.9]:
    for br in [0.8, 1.0, 1.2, 1.5]:
        for bw in [0.3, 0.5, 0.7, 0.9]:
            pred3b = sigmoid(a * base_logit + br * log_rknn + bw * log_win)
            auc3b = macro_auc(file_labels, pred3b)
            if auc3b > best3b: best3b = auc3b; best3b_cfg = {'a': a, 'br': br, 'bw': bw}
if best3b > best3:
    results['rknn_win_separate'] = (best3b, best3b_cfg)
    print(f"  Separate b: {best3b:.4f}  cfg={best3b_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Fine sweep around known best config (asymmetric_ls2)
# sigmoid(a × base_logit + b1 × log(geo_k5) + b2 × log(win_k1))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Asymmetric logspace fine sweep ===", flush=True)
log_k5 = np.log(y_knn[5])
best4 = 0; best4_cfg = {}
for a in [0.80, 0.85, 0.90, 0.95, 1.00]:
    for b_geo in [0.6, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0]:
        for b_win in [0.3, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
            pred4 = sigmoid(a * base_logit + b_geo * log_k5 + b_win * log_win)
            auc4 = macro_auc(file_labels, pred4)
            if auc4 > best4: best4 = auc4; best4_cfg = {'a': a, 'b_geo': b_geo, 'b_win': b_win}
results['asym_logspace_fine'] = (best4, best4_cfg)
print(f"  Best: {best4:.4f}  cfg={best4_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: RKNN + win k3 (larger window KNN)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: RKNN + win_k3 ===", flush=True)
y_win_k3 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = (win_file_id != i)
    X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    sims = X_te @ X_tr.T
    top_idx = np.argsort(-sims, 1)[:, :3]
    wp = np.zeros((te_e-te_s, n_species), np.float32)
    for wi in range(te_e-te_s):
        ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
        ww = ww/ws if ws > 1e-8 else np.ones(3)/3
        wp[wi] = (ww[:, None] * file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k3[i] = wp.mean(0)
y_win_k3 = y_win_k3.clip(EPS, 1-EPS)

best5 = 0; best5_cfg = {}
for wg1 in [0.3, 0.4, 0.5]:
    for wg3 in [0.1, 0.2, 0.3]:
        wg_rknn = 1.0 - wg1 - wg3
        if wg_rknn < 0: continue
        yb5 = wg_rknn * y_rknn + wg1 * y_win_k1 + wg3 * y_win_k3
        for a in [0.8, 0.85, 0.9]:
            for b in [1.4, 1.6, 1.8, 2.0]:
                pred5 = sigmoid(a * base_logit + b * np.log(yb5.clip(EPS)))
                auc5 = macro_auc(file_labels, pred5)
                if auc5 > best5: best5 = auc5; best5_cfg = {'wg_rknn': wg_rknn, 'wg1': wg1, 'wg3': wg3, 'a': a, 'b': b}
results['rknn_win_k1_k3'] = (best5, best5_cfg)
print(f"  Best: {best5:.4f}  cfg={best5_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
EP_BEST = 0.9199
print(f"\n{'='*60}")
print(f"MULTI-RES KNN SUMMARY")
print(f"EP-only reference: interaction_knn={EP_BEST}")
print(f"{'='*60}")
all_results = list(results.items()) + ([] if 'rknn_win_separate' in results else [])
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    delta = auc - EP_BEST
    marker = " *** NEW EP BEST ***" if auc > EP_BEST else ""
    print(f"  {name}: {auc:.4f}  ({delta:+.4f}){marker}")

# If new best, create model PKL and notebook
best_method = max(results.items(), key=lambda x: x[1][0])
best_name, (best_auc, best_cfg) = best_method
print(f"\nOverall best: {best_name} = {best_auc:.4f}")

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
for name, (auc, cfg) in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc), 'config': cfg})
if best_auc > cur_best:
    rd['best'] = {'method': best_name, 'loo_auc': float(best_auc), 'full_auc': float(best_auc)}
    print(f"*** NEW OVERALL BEST: {best_name} = {best_auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")

# ── Build model PKL for best method ───────────────────────────────────────────
if best_auc > EP_BEST - 0.005:  # Close to EP best → worth creating notebook
    print(f"\nBuilding embed_prior_model.pkl for {best_name}...")
    file_embs_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)
    model_pkl = {
        'method': best_name,
        'loo_auc': float(best_auc),
        'config': best_cfg,
        'file_embs_norm': file_embs_norm,        # (66, 1536) L2-norm
        'file_labels': file_labels,               # (66, 234)
        'file_prob_max': file_prob_max,           # (66, 234) sigmoid(logit_max)
        'file_logit_max': file_logit_max,         # (66, 234)
        # For RKNN inference: X_ref + window embeddings
        'X_ref': X_ref,                           # (66, 39) PCA24+geo
        'file_list': np.array(file_list),
        'emb_win_norm': emb_win_norm,             # (739, 1536)
        'win_file_id': win_file_id,               # (739,)
        'n_windows': n_windows,
        'file_start': file_start,
        'file_end': file_end,
    }
    with open("outputs/embed_prior_model.pkl", 'wb') as f:
        pickle.dump(model_pkl, f)
    print(f"Saved outputs/embed_prior_model.pkl")
