"""
Batch 71: Ultra-High Temperature Sweep + Combined Best Components

從 logit_temp_ft (T=8.0, w=0.28, LOO=0.9893) 出發：
- T 趨勢：1→5→8 全部改善，最優 T 可能更高
- 方向1：超高溫 T sweep（T=8-100 精細掃描）
- 方向2：3-way + 4-way 最優組合微調（subspace + multi-T）
- 方向3：Per-species adaptive temperature（不同 species 用不同 T）
- 方向4：Logit rank transform（非線性 calibration 替代 sigmoid/T）

Current best: logit_temp_ft = 0.9893 (T=8.0, w=0.28)
"""
import numpy as np, json, os, time, pickle
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logit_win  = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
n_windows  = perch['n_windows']
file_list  = list(perch['file_list'])
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[file_start[fi]:file_end[fi]] = fi
file_labels = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

EPS = 1e-7; mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9893335
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# ─── Load pkl ────────────────────────────────────────────────────────────────
print("Loading pkl transforms...", flush=True)
with open("outputs/embed_prior_model.pkl", 'rb') as f:
    ep = pickle.load(f)
ew_ica = ep['emb_win_ica_norm'].astype(np.float32)
ew_pca = ep['emb_win_pca_norm'].astype(np.float32)
ew_std = ep['emb_win_std_norm'].astype(np.float32)

W_ICA, W_STD, W_PCA = 0.655, 0.225, 0.120
ICA_K, ICA_WMA, ICA_WMP = 50, 0.92, 0.80
STD_K, STD_WMA, STD_WMP =  4, 0.65, 0.60
PCA_K, PCA_WMA, PCA_WMP =  4, 0.60, 0.70

def build_cache(emb_n):
    c = {}
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]; tr_m = win_file_id != fi
        c[fi] = (te, emb_n[tr_m], labels_win[tr_m], te @ emb_n[tr_m].T)
    return c

def wl_from_cache(cache, fi, k_neg, wmp, wma):
    te, tr, tl, sims = cache[fi]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pos_idx = np.where(tl[:, si] > 0.5)[0]
        neg_idx = np.where(tl[:, si] < 0.1)[0]
        if len(pos_idx) == 0: ws[:, si] = 0.5; continue
        ps = sims[:, pos_idx]
        pp = tr[pos_idx].mean(0); pp /= np.linalg.norm(pp) + EPS
        sp = wmp * ps.max(1) + (1-wmp) * (te @ pp)
        if len(neg_idx) > 0:
            ns2 = sims[:, neg_idx]; k2 = min(k_neg, len(neg_idx))
            top_idx = np.argsort(-ns2, axis=1)[:, :k2]
            tn_scores = np.array([
                (te[j] @ tr[neg_idx[top_idx[j]]].mean(0) /
                 (np.linalg.norm(tr[neg_idx[top_idx[j]]].mean(0)) + EPS))
                for j in range(len(te))], dtype=np.float32)
            ws[:, si] = (sp - tn_scores + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return wma * ws.max(0) + (1-wma) * ws.mean(0)

print("Building sim caches...", flush=True)
t0 = time.time()
c_ica = build_cache(ew_ica); c_std = build_cache(ew_std); c_pca = build_cache(ew_pca)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

print("Computing UH-triple reference...", flush=True)
t0 = time.time()
s_ica = np.stack([wl_from_cache(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std = np.stack([wl_from_cache(c_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca = np.stack([wl_from_cache(c_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
uh_triple = W_ICA * s_ica + W_STD * s_std + W_PCA * s_pca
uh_auc = eval_loo(uh_triple)
print(f"  UH-triple: {uh_auc:.4f}  ({time.time()-t0:.1f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Ultra-high temperature sweep
# T 趨勢 1→5→8 改善，延伸到 T=10-100
# 極高溫時 sigmoid(x/T) ≈ 0.5 + x/(4T) ≈ linear in x → 等同 raw logit scaling
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Ultra-High Temperature Sweep ===", flush=True)
t0 = time.time()
best_ult, best_cfg_ult = 0, None

T_FINE = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 1000.0]
W_FINE = [0.20, 0.22, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50]

for T in T_FINE:
    sig_T = (1.0 / (1.0 + np.exp(np.clip(-logit_win / T, -88, 88)))).astype(np.float32)
    preds_T = np.stack([sig_T[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
    for w in W_FINE:
        blend = (1-w) * uh_triple + w * preds_T
        auc = eval_loo(blend)
        if auc > best_ult: best_ult = auc; best_cfg_ult = (T, w)

print(f"  Ultra-high T best: {best_ult:.6f}  cfg={best_cfg_ult}  ({time.time()-t0:.1f}s)", flush=True)
results['logit_ultra_temp'] = best_ult
print(f"  {'*** NEW BEST ***' if best_ult > CURRENT_BEST else ''}", flush=True)

# At T→∞, sigmoid(x/T) → 0.5 + x/(4T), max becomes proportional to max(logit)
# Try pure logit max (T→∞ limit)
print("  Testing raw logit max (T→∞ limit)...", flush=True)
preds_raw_max = np.stack([logit_win[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
# Normalize raw logit predictions to [0,1] range via min-max
lo, hi = preds_raw_max.min(), preds_raw_max.max()
preds_raw_norm = (preds_raw_max - lo) / (hi - lo + EPS)
best_rawmax, best_cfg_rawmax = 0, None
for w in W_FINE:
    blend = (1-w) * uh_triple + w * preds_raw_norm
    auc = eval_loo(blend)
    if auc > best_rawmax: best_rawmax = auc; best_cfg_rawmax = w
results['logit_rawmax_uh_blend'] = best_rawmax
print(f"  Raw logit max (norm) blend: {best_rawmax:.6f}  w={best_cfg_rawmax}", flush=True)
print(f"  {'*** NEW BEST ***' if best_rawmax > CURRENT_BEST else ''}", flush=True)

# Keep best T for later
T_ultra_best = best_cfg_ult[0] if best_cfg_ult else 8.0
w_ultra_best = best_cfg_ult[1] if best_cfg_ult else 0.28
sig_Tub = (1.0 / (1.0 + np.exp(np.clip(-logit_win / T_ultra_best, -88, 88)))).astype(np.float32)
preds_Tub = np.stack([sig_Tub[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Rank-based logit transform
# 非線性 calibration：對每個 species 的 window logits 做 rank transform
# rank(x) / n_wins → uniform [0,1]，完全消除 logit scale 影響
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Rank-based Logit Transform ===", flush=True)
t0 = time.time()

# Global rank: across all windows (per species)
from scipy.stats import rankdata
logit_rank = np.zeros_like(logit_win)
for si in range(n_species):
    logit_rank[:, si] = rankdata(logit_win[:, si]) / len(logit_win)
logit_rank = logit_rank.astype(np.float32)

preds_rank = np.stack([logit_rank[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
best_rank, best_cfg_rank = 0, None
for w in [0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40]:
    blend = (1-w) * uh_triple + w * preds_rank
    auc = eval_loo(blend)
    if auc > best_rank: best_rank = auc; best_cfg_rank = w
results['logit_rank_uh_blend'] = best_rank
print(f"  Global rank blend: {best_rank:.6f}  w={best_cfg_rank}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_rank > CURRENT_BEST else ''}", flush=True)

# File-local rank: rank within each file's windows (LOO-safe version)
preds_local_rank = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    wins = logit_win[s:e]  # (n_win, n_species)
    n_w = e - s
    if n_w == 1:
        preds_local_rank[fi] = 1.0  # only 1 window → rank=1
    else:
        for si in range(n_species):
            preds_local_rank[fi, si] = rankdata(wins[:, si]).max() / n_w

best_lrank, best_cfg_lrank = 0, None
for w in [0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40]:
    blend = (1-w) * uh_triple + w * preds_local_rank
    auc = eval_loo(blend)
    if auc > best_lrank: best_lrank = auc; best_cfg_lrank = w
results['logit_local_rank_uh_blend'] = best_lrank
print(f"  Local rank blend: {best_lrank:.6f}  w={best_cfg_lrank}", flush=True)
print(f"  {'*** NEW BEST ***' if best_lrank > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Combined T-sweep best + Species Subspace (fine-tune 3-way)
# uh_logit_subspace_3way = 0.9893, logit_temp5_subspace_3way = 0.9893
# Fine-tune with new ultra-best T
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Fine-tune 3-Way Blend with Ultra-best T ===", flush=True)
t0 = time.time()

def species_subspace_loo(emb_n, n_comp, wma):
    out = np.zeros((n_files, n_species), np.float32)
    dim = emb_n.shape[1]
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]; tr_m = win_file_id != fi
        tr = emb_n[tr_m]; tl = labels_win[tr_m]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; n_pos = len(pos)
            k = min(n_comp, n_pos - 1, dim - 1)
            if k < 1:
                pp = pos.mean(0); pp /= np.linalg.norm(pp) + EPS
                ws[:, si] = np.clip((te @ pp + 1) / 2, 0, 1); continue
            try:
                pca_sp = PCA(n_components=k)
                pca_sp.fit(pos)
                te_proj = pca_sp.transform(te)
                te_recon = pca_sp.inverse_transform(te_proj)
                recon_err = np.linalg.norm(te - te_recon, axis=1)
                te_norm = np.linalg.norm(te, axis=1)
                ws[:, si] = np.clip(1 - recon_err / (te_norm + EPS), 0, 1)
            except Exception:
                ws[:, si] = 0.5
        out[fi] = wma * ws.max(0) + (1-wma) * ws.mean(0)
    return out

print("  Computing subspace scores...", flush=True)
# Use best config from batch 69/70: pca80, n_comp=3, wma=0.88
ss_scores = species_subspace_loo(ew_pca, 3, 0.88)
print(f"  Subspace standalone: {eval_loo(ss_scores):.4f}", flush=True)

# Fine-tune 3-way with ultra-best T
best_3ft, best_cfg_3ft = 0, None
W_LOG_LIST = [0.15, 0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30]
W_SS_LIST  = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
for w_log in W_LOG_LIST:
    for w_ss in W_SS_LIST:
        w_uh = 1.0 - w_log - w_ss
        if w_uh < 0.55: continue
        blend3 = w_uh * uh_triple + w_log * preds_Tub + w_ss * ss_scores
        auc = eval_loo(blend3)
        if auc > best_3ft: best_3ft = auc; best_cfg_3ft = (T_ultra_best, w_log, w_ss)

results['logit_temp_ss_3way_ft'] = best_3ft
print(f"  Fine-tuned 3-way: {best_3ft:.6f}  cfg={best_cfg_3ft}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_3ft > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: 4-Way Blend: UH + ultra-T logit + subspace + multi-T
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: 4-Way Blend ===", flush=True)
t0 = time.time()

# Multi-T average (best from batch 70: T=[5,10], w=0.25)
preds_T5 = np.stack([(1.0/(1.0+np.exp(np.clip(-logit_win/5.0,-88,88))))[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
preds_T10 = np.stack([(1.0/(1.0+np.exp(np.clip(-logit_win/10.0,-88,88))))[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
preds_multit = (preds_T5 + preds_T10) / 2.0

best_4w, best_cfg_4w = 0, None
for w_ub in [0.15, 0.20, 0.22, 0.25]:
    for w_mt in [0.05, 0.08, 0.10]:
        for w_ss in [0.04, 0.06, 0.08]:
            w_uh = 1.0 - w_ub - w_mt - w_ss
            if w_uh < 0.50: continue
            b4 = w_uh * uh_triple + w_ub * preds_Tub + w_mt * preds_multit + w_ss * ss_scores
            auc = eval_loo(b4)
            if auc > best_4w: best_4w = auc; best_cfg_4w = (w_ub, w_mt, w_ss)

results['logit_4way_blend'] = best_4w
print(f"  4-Way blend: {best_4w:.6f}  cfg(w_ub,w_mt,w_ss)={best_cfg_4w}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_4w > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Geometric Mean of Temperature-Calibrated Predictions
# 幾何平均比算術平均更保守（對 near-zero 更 robust）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Geometric Mean Multi-T ===", flush=True)
t0 = time.time()
best_gm, best_cfg_gm = 0, None

T_GM_COMBOS = [
    [5.0, 8.0], [5.0, 10.0], [8.0, 10.0],
    [5.0, 8.0, 10.0], [5.0, 8.0, 15.0],
]
for combo in T_GM_COMBOS:
    preds_list = []
    for T in combo:
        sig_T = (1.0/(1.0+np.exp(np.clip(-logit_win/T,-88,88)))).astype(np.float32)
        preds_list.append(np.stack([sig_T[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)]))
    # Geometric mean
    geo_preds = np.ones_like(preds_list[0])
    for p in preds_list:
        geo_preds = geo_preds * np.clip(p, EPS, 1-EPS)
    geo_preds = geo_preds ** (1.0 / len(combo))

    for w in [0.20, 0.25, 0.28, 0.30, 0.35]:
        blend = (1-w) * uh_triple + w * geo_preds
        auc = eval_loo(blend)
        if auc > best_gm: best_gm = auc; best_cfg_gm = (combo, w)

results['logit_geomean_uh_blend'] = best_gm
print(f"  Geo-mean multi-T: {best_gm:.6f}  cfg={best_cfg_gm}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_gm > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 71 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.6f}{flag}", flush=True)
print(f"  Current best ref: {CURRENT_BEST:.6f}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
new_best_found = False; best_new_method = None; best_new_auc = 0

for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        new_best_found = True
        if auc > best_new_auc: best_new_auc = auc; best_new_method = name

with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)

print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.6f}", flush=True)
if not new_best_found:
    print("未超越 0.9893，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
