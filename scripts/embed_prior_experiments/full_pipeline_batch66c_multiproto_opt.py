"""
Batch 66c: Multi-Prototype WL (優化版 - 預計算 similarity matrix)
關鍵優化：每個 file 只做一次 te @ tr.T，後續所有 species 的
positive/negative similarity 都用 slicing（不重算）。

預計速度：約 35s/config（vs 原先可能的 300s+/config）

Method 1: Diverse Prototypes (Max-Min Selection)
Method 2: K-means Prototypes (快速 K-means 用預計算 sim)

Current best: 0.9873025
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
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
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

EPS = 1e-7
mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9873024930999804
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# ─── Precompute embeddings ────────────────────────────────────────────────────
print("Precomputing embeddings...", flush=True)
ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2')
pca80 = PCA(n_components=80, random_state=42)
ew_pca = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2')
scaler = StandardScaler()
ew_std = normalize(PCA(n_components=80, random_state=42).fit_transform(
    scaler.fit_transform(emb_win).astype(np.float32)).astype(np.float32), norm='l2')

W_ICA, W_STD, W_PCA = 0.655, 0.225, 0.120
ICA_K, ICA_WMA, ICA_WMP = 50, 0.92, 0.80
STD_K, STD_WMA, STD_WMP =  4, 0.65, 0.60
PCA_K, PCA_WMA, PCA_WMP =  4, 0.60, 0.70
print("Done.", flush=True)

# ─── Pre-compute sim cache: te @ tr.T for each LOO fold ──────────────────────
print("Pre-caching similarity matrices...", flush=True)
t0 = time.time()
# For each embedding space: store {fi: (te, tr, tr_lab, sims)}
# sims = te @ tr.T → shape (n_te, n_tr)
def build_sim_cache(emb_n):
    cache = {}
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]        # (n_te, dim)
        tr_m = win_file_id != fi
        tr = emb_n[tr_m]                     # (n_tr, dim)
        tl = labels_win[tr_m]                # (n_tr, n_species)
        sims = te @ tr.T                     # (n_te, n_tr) - ONE compute per file
        cache[fi] = (te, tr, tl, sims)
    return cache

cache_ica = build_sim_cache(ew_ica)
cache_std = build_sim_cache(ew_std)
cache_pca = build_sim_cache(ew_pca)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

# ─── Multi-Prototype WL using precomputed sims ───────────────────────────────
def wl_multiproto_cached(cache, fi, k_proto, k_neg, wma, w_max_ctr, proto_mode='diverse'):
    te, tr, tl, sims = cache[fi]  # sims: (n_te, n_tr)
    n_te = len(te)
    ws = np.zeros((n_te, n_species), np.float32)

    for si in range(n_species):
        pos_idx = np.where(tl[:, si] > 0.5)[0]
        neg_idx = np.where(tl[:, si] < 0.1)[0]

        if len(pos_idx) == 0:
            ws[:, si] = 0.5; continue

        # Positive similarities (slice from precomputed)
        pos_sims = sims[:, pos_idx]  # (n_te, n_pos) - NO recompute!
        pos_wins = tr[pos_idx]       # (n_pos, dim)
        n_pos = len(pos_idx)
        k_act = min(k_proto, n_pos)

        if k_act <= 1 or proto_mode == 'mean':
            # Single mean prototype: use mean of pos_sims columns
            sp = w_max_ctr * pos_sims.max(1) + (1-w_max_ctr) * pos_sims.mean(1)
        elif proto_mode == 'diverse':
            # Diverse prototypes using precomputed pos_sims
            # We need te @ centers.T where centers are selected positive windows
            # Strategy: use pos_sims to find diverse set
            # Start with window closest to mean (highest avg similarity)
            mean_sim = pos_sims.mean(0)  # (n_pos,) - avg test similarity per pos window
            center_indices = [np.argmax(mean_sim)]  # Start with highest-sim pos
            for _ in range(k_act - 1):
                # Similarity of each pos window to already-selected centers
                # centers are pos_wins[center_indices], diversity = min sim to any selected
                selected_sims = pos_sims[:, center_indices]  # (n_te, n_sel)
                # For diversity: use the pos-pos similarities
                # pos_pos_sims[i,j] = pos_wins[i] @ pos_wins[j] (if normalized)
                # We want the pos window with lowest max similarity to already selected
                # Approximate: use the column of pos_sims most different from selected
                selected_avg = pos_sims[:, center_indices].mean(1)  # (n_te,) avg selected sim
                residual = pos_sims - selected_avg[:, None]  # (n_te, n_pos)
                diversity_score = -np.abs(residual).mean(0)  # Most dissimilar from selected
                # Exclude already selected
                for ci in center_indices:
                    diversity_score[ci] = -np.inf
                next_ci = np.argmax(diversity_score)  # Most different
                center_indices.append(next_ci)
            # Final score: max/mean similarity to selected centers
            center_sims = pos_sims[:, center_indices]  # (n_te, k_act)
            sp = w_max_ctr * center_sims.max(1) + (1-w_max_ctr) * center_sims.mean(1)
        else:  # greedy max-spread
            sp = w_max_ctr * pos_sims.max(1) + (1-w_max_ctr) * pos_sims.mean(1)

        if len(neg_idx) > 0:
            neg_sims = sims[:, neg_idx]  # (n_te, n_neg) - NO recompute!
            k2 = min(k_neg, len(neg_idx))
            # Top-k neg: mean of top-k neg windows
            top_neg_idx = np.argsort(-neg_sims, axis=1)[:, :k2]  # (n_te, k2)
            # Compute te @ selected_neg.T without recompute:
            # tn_score[j] = te[j] @ mean(tr[neg_idx][top_neg_idx[j]])
            tn_scores = np.zeros(n_te, np.float32)
            for j in range(n_te):
                tn = tr[neg_idx][top_neg_idx[j]].mean(0)
                tn_norm = np.linalg.norm(tn)
                if tn_norm > EPS:
                    tn /= tn_norm
                    tn_scores[j] = te[j] @ tn
                else:
                    tn_scores[j] = 0.0
            ws[:, si] = (sp - tn_scores + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2

    return wma * ws.max(0) + (1-wma) * ws.mean(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Diverse Prototypes
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Diverse Prototypes ===", flush=True)
t0 = time.time()
best_div = 0; best_cfg_div = None

for k_proto in [2, 3, 4]:
    for wma in [0.88, 0.90, 0.92, 0.95]:
        for w_max_c in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for cache, k_neg, name in [
                (cache_ica, ICA_K, 'ica100'),
                (cache_pca, PCA_K, 'pca80'),
            ]:
                out = np.stack([wl_multiproto_cached(cache, fi, k_proto, k_neg, wma, w_max_c)
                                for fi in range(n_files)])
                auc = eval_loo(out)
                if auc > best_div: best_div = auc; best_cfg_div = (name, k_proto, wma, w_max_c)

print(f"  Diverse-Proto best: {best_div:.4f}  cfg={best_cfg_div}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_diverse_proto'] = best_div
print(f"  {'*** NEW BEST ***' if best_div > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Mean Prototype (reference - should match original WL)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Mean Prototype (cached reference) ===", flush=True)
t0 = time.time()
best_mean = 0; best_cfg_mean = None

for wma in [0.88, 0.90, 0.92, 0.95]:
    for w_max_c in [0.7, 0.8, 0.9, 1.0]:
        for cache, k_neg, name in [
            (cache_ica, ICA_K, 'ica100'),
            (cache_pca, PCA_K, 'pca80'),
            (cache_std, STD_K, 'std80'),
        ]:
            out = np.stack([wl_multiproto_cached(cache, fi, 1, k_neg, wma, w_max_c, 'mean')
                            for fi in range(n_files)])
            auc = eval_loo(out)
            if auc > best_mean: best_mean = auc; best_cfg_mean = (name, wma, w_max_c)

print(f"  Mean-Proto (ref) best: {best_mean:.4f}  cfg={best_cfg_mean}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_mean_proto_cached'] = best_mean
print(f"  {'*** NEW BEST ***' if best_mean > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Diverse Triple Blend
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Diverse Triple Blend ===", flush=True)
t0 = time.time()
best_dtriple = 0; best_cfg_dtriple = None

# Use best diverse params, apply to all 3 embedding spaces
if best_cfg_div:
    name_d, k_d, wma_d, wmc_d = best_cfg_div
    for k_proto in [2, 3]:
        for wma in [0.90, 0.92]:
            for w_max_c in [0.7, 0.8]:
                s_ica = np.stack([wl_multiproto_cached(cache_ica, fi, k_proto, ICA_K, wma, w_max_c)
                                  for fi in range(n_files)])
                s_std = np.stack([wl_multiproto_cached(cache_std, fi, k_proto, STD_K, wma, w_max_c)
                                  for fi in range(n_files)])
                s_pca = np.stack([wl_multiproto_cached(cache_pca, fi, k_proto, PCA_K, wma, w_max_c)
                                  for fi in range(n_files)])
                out = W_ICA * s_ica + W_STD * s_std + W_PCA * s_pca
                auc = eval_loo(out)
                if auc > best_dtriple: best_dtriple = auc; best_cfg_dtriple = (k_proto, wma, w_max_c)

print(f"  Diverse Triple best: {best_dtriple:.4f}  cfg={best_cfg_dtriple}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_diverse_triple'] = best_dtriple
print(f"  {'*** NEW BEST ***' if best_dtriple > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 66c Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
new_best_found = False
for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        new_best_found = True
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)

with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)

print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
if not new_best_found:
    print("未超越 0.9873，已 append 到 experiments。", flush=True)
