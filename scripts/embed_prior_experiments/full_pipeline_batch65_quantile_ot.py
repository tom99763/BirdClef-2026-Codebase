"""
Batch 65: Quantile Aggregation + Optimal Transport (Wasserstein) 距離
兩個全新方向，從未在之前實驗中出現

Method 1: Quantile Aggregation (取代 max/mean)
  - 現有 wl_contrast 用 wma*max + (1-wma)*mean 聚合 window scores
  - 改用 percentile (e.g., 80th, 90th, 95th) 作為聚合函數
  - 掃描 percentile + 與 mean 的混合比例

Method 2: 1D Wasserstein Distance Scoring
  - 對每個 species，計算 test window 相似度分布 vs positive prototype 的 Wasserstein 距離
  - W1 距離用 sorted 陣列差計算（1D 最優傳輸有解析解）
  - 分數 = -W1(test, pos) + W1(test, neg) → 越高越好

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

ICA_K, ICA_WMP = 50, 0.80
STD_K, STD_WMP =  4, 0.60
PCA_K, PCA_WMP =  4, 0.70
W_ICA, W_STD, W_PCA = 0.655, 0.225, 0.120
print("Done.", flush=True)

# ─── WL window scores (raw ~[0,1]) ───────────────────────────────────────────
def wl_scores(emb_n, fi, k_neg, wmp):
    te = emb_n[win_file_id == fi]
    tr_m = win_file_id != fi
    tr = emb_n[tr_m]; tl = labels_win[tr_m]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
        if not pm.any(): ws[:, si] = 0.5; continue
        pw = tr[pm]; ps = te @ pw.T
        pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
        sp = wmp * ps.max(1) + (1 - wmp) * (te @ pp)
        if nm.any():
            nw = tr[nm]; ns2 = te @ nw.T; k2 = min(k_neg, ns2.shape[1])
            tn = nw[np.argsort(-ns2, axis=1)[:, :k2]].mean(1)
            tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
            ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return ws

print("Pre-caching WL window scores...", flush=True)
t0 = time.time()
ica_wins = [wl_scores(ew_ica, fi, ICA_K, ICA_WMP) for fi in range(n_files)]
std_wins = [wl_scores(ew_std, fi, STD_K, STD_WMP) for fi in range(n_files)]
pca_wins = [wl_scores(ew_pca, fi, PCA_K, PCA_WMP) for fi in range(n_files)]
print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

# ─── WL triple blend (reference) ─────────────────────────────────────────────
def blend_wins(fi):
    return W_ICA * ica_wins[fi] + W_STD * std_wins[fi] + W_PCA * pca_wins[fi]

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Quantile Aggregation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Quantile Aggregation ===", flush=True)
t0 = time.time()
best_q = 0; best_cfg_q = None

def quantile_agg(ws, q, w_q, w_mean):
    """q-percentile * w_q + mean * w_mean aggregation."""
    qval = np.percentile(ws, q, axis=0)
    mval = ws.mean(0)
    return w_q * qval + w_mean * mval

for q in [70, 75, 80, 85, 90, 92, 95, 97, 99, 100]:
    for w_q in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        w_mean = 1.0 - w_q
        # Apply to triple blend window scores
        out = np.stack([quantile_agg(blend_wins(fi), q, w_q, w_mean)
                        for fi in range(n_files)])
        auc = eval_loo(out)
        if auc > best_q:
            best_q = auc; best_cfg_q = (q, w_q)

print(f"  Quantile-agg best: {best_q:.4f}  cfg={best_cfg_q}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_quantile_agg'] = best_q
print(f"  {'*** NEW BEST ***' if best_q > CURRENT_BEST else ''}", flush=True)

# Also try ICA alone with quantile
t0 = time.time()
best_q_ica = 0; best_cfg_q_ica = None
for q in [75, 80, 85, 90, 92, 95, 97, 99, 100]:
    for w_q in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        w_mean = 1.0 - w_q
        out = np.stack([quantile_agg(ica_wins[fi], q, w_q, w_mean) for fi in range(n_files)])
        auc = eval_loo(out)
        if auc > best_q_ica:
            best_q_ica = auc; best_cfg_q_ica = (q, w_q)

print(f"  Quantile-ICA best: {best_q_ica:.4f}  cfg={best_cfg_q_ica}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_quantile_ica'] = best_q_ica
print(f"  {'*** NEW BEST ***' if best_q_ica > CURRENT_BEST else ''}", flush=True)

# Three-way quantile: q_pct * w1 + max * w2 + mean * w3
t0 = time.time()
best_q3 = 0; best_cfg_q3 = None
for q in [80, 85, 90, 95]:
    for w_q in [0.2, 0.3, 0.4]:
        for w_max in [0.5, 0.6, 0.65, 0.7]:
            w_mean = max(0, 1 - w_q - w_max)
            if w_mean < 0: continue
            out2 = np.stack([
                w_q * np.percentile(blend_wins(fi), q, axis=0) +
                w_max * blend_wins(fi).max(0) +
                w_mean * blend_wins(fi).mean(0)
                for fi in range(n_files)
            ])
            auc = eval_loo(out2)
            if auc > best_q3:
                best_q3 = auc; best_cfg_q3 = (q, w_q, w_max)

print(f"  Quantile3way best: {best_q3:.4f}  cfg={best_cfg_q3}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_quantile3'] = best_q3
print(f"  {'*** NEW BEST ***' if best_q3 > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: 1D Wasserstein Distance Scoring
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: 1D Wasserstein Distance Scoring ===", flush=True)
print("理念：用 test-window 相似度分布 vs pos/neg 的 W1 距離作為分數", flush=True)

def w1_dist_1d(a, b):
    """1D Wasserstein distance between two arrays (sorted CDF diff)."""
    # Interpolate to same length using sorted quantiles
    n = max(len(a), len(b))
    qa = np.quantile(a, np.linspace(0, 1, n))
    qb = np.quantile(b, np.linspace(0, 1, n))
    return np.mean(np.abs(qa - qb))

def wl_wasserstein(emb_n, fi, k_neg, wmp, w_max_agg):
    """Wasserstein-based per-species scoring."""
    te = emb_n[win_file_id == fi]
    tr_m = win_file_id != fi
    tr = emb_n[tr_m]; tl = labels_win[tr_m]
    ws = np.zeros(n_species, np.float32)
    for si in range(n_species):
        pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
        if not pm.any(): ws[si] = 0.5; continue
        pw = tr[pm]
        pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
        sp_wmp = wmp * (te @ pw.T).max(1) + (1-wmp) * (te @ pp)
        if nm.any():
            nw = tr[nm]
            np_ = nw.mean(0); np_ /= np.linalg.norm(np_) + EPS
            sn_wmp = wmp * (te @ nw.T).max(1) + (1-wmp) * (te @ np_)
            # Wasserstein: closer to positive, farther from negative
            w_pos = w1_dist_1d(sp_wmp, np.ones(len(pw)))   # how far from "1" distribution
            w_neg = w1_dist_1d(sp_wmp, np.zeros(len(nw)))  # how far from "0" distribution
            # Score: probability-like (closer to pos is better)
            ws[si] = w_neg / (w_pos + w_neg + EPS)
        else:
            ws[si] = np.mean(sp_wmp > 0.5)
    return ws

t0 = time.time()
best_w1 = 0; best_cfg_w1 = None
for wmp in [0.60, 0.70, 0.80, 0.90, 1.0]:
    for wma in [0.0]:  # unused for wasserstein (file-level directly)
        for emb, name in [(ew_ica, 'ica100'), (ew_pca, 'pca80'), (ew_std, 'std80')]:
            out = np.stack([wl_wasserstein(emb, fi, 50, wmp, 0.9) for fi in range(n_files)])
            auc = eval_loo(out)
            if auc > best_w1: best_w1 = auc; best_cfg_w1 = (name, wmp)

print(f"  Wasserstein best: {best_w1:.4f}  cfg={best_cfg_w1}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_wasserstein'] = best_w1
print(f"  {'*** NEW BEST ***' if best_w1 > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Harmonic mean aggregation (instead of max/mean)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Harmonic Mean Aggregation ===", flush=True)
t0 = time.time()
best_hm = 0; best_cfg_hm = None

def harm_agg(ws, w_harm, w_max, w_mean):
    """Harmonic mean of window scores."""
    ws_clip = np.clip(ws, EPS, 1-EPS)
    h_mean = 1.0 / (1.0 / ws_clip).mean(0)
    return w_harm * h_mean + w_max * ws.max(0) + w_mean * ws.mean(0)

for w_harm in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for w_max in [0.5, 0.6, 0.7, 0.8]:
        w_mean = max(0, 1 - w_harm - w_max)
        out = np.stack([harm_agg(blend_wins(fi), w_harm, w_max, w_mean)
                        for fi in range(n_files)])
        auc = eval_loo(out)
        if auc > best_hm:
            best_hm = auc; best_cfg_hm = (w_harm, w_max, w_mean)

print(f"  Harmonic-mean best: {best_hm:.4f}  cfg={best_cfg_hm}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_harmonic_agg'] = best_hm
print(f"  {'*** NEW BEST ***' if best_hm > CURRENT_BEST else ''}", flush=True)

# ─── Also: blend quantile3 best with UH triple ───────────────────────────────
print("\n=== Method 4: Quantile3 blend with WL triple ===", flush=True)
t0 = time.time()
best_qblend = 0; best_cfg_qblend = None

# Re-evaluate UH triple reference
uh_triple = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    bw = blend_wins(fi)
    uh_triple[fi] = 0.92 * bw.max(0) + 0.08 * bw.mean(0)
uh_auc = eval_loo(uh_triple)
print(f"  UH-triple reference: {uh_auc:.4f}", flush=True)

# Best quantile3 config
if best_cfg_q3:
    q_best, wq_best, wmax_best = best_cfg_q3
    wmean_best = max(0, 1 - wq_best - wmax_best)
    q3_scores = np.stack([
        wq_best * np.percentile(blend_wins(fi), q_best, axis=0) +
        wmax_best * blend_wins(fi).max(0) +
        wmean_best * blend_wins(fi).mean(0)
        for fi in range(n_files)
    ])
    for w_q3 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        blend_final = (1 - w_q3) * uh_triple + w_q3 * q3_scores
        auc = eval_loo(blend_final)
        if auc > best_qblend:
            best_qblend = auc; best_cfg_qblend = w_q3

print(f"  Quantile3+UH blend: {best_qblend:.4f}  cfg=w_q3={best_cfg_qblend}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_q3_uh_blend'] = best_qblend
print(f"  {'*** NEW BEST ***' if best_qblend > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 65 Summary ===", flush=True)
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
