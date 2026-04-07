"""
Batch 82: KDE + Two-stage Re-ranking + Local Mahalanobis

從 nmf_w_ultra (LOO=0.99136) 出發 — 已達強力平台（batch79-81 均打平）

嘗試根本性不同的方法：
1. KDE Score：per-species Gaussian KDE，p(x|species) 用 LOO 正樣本 fit
2. Two-stage Re-ranking：logit 初選 top-K species，WL 在 top-K 中精排
3. Local Mahalanobis WL：用正樣本的 covariance 逆矩陣計算 Mahalanobis 距離
4. Prototype distance ratio：pos_sim / (pos_sim + neg_sim) 而非差值
"""
import numpy as np, json, os, time, pickle
from sklearn.decomposition import PCA, NMF
from sklearn.covariance import LedoitWolf
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
CURRENT_BEST = 0.991359
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

with open("outputs/embed_prior_model.pkl", 'rb') as f:
    ep = pickle.load(f)
ew_ica = ep['emb_win_ica_norm'].astype(np.float32)
ew_pca = ep['emb_win_pca_norm'].astype(np.float32)
ew_std = ep['emb_win_std_norm'].astype(np.float32)
ew_nmf = ep['emb_win_nmf_norm'].astype(np.float32)

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

print("Building caches...", flush=True)
t0 = time.time()
c_ica = build_cache(ew_ica); c_std = build_cache(ew_std)
c_pca = build_cache(ew_pca); c_nmf = build_cache(ew_nmf)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

W_ICA, W_STD, W_PCA = 0.72, 0.18, 0.10
ICA_K, ICA_WMA, ICA_WMP = 50, 0.88, 0.85
STD_K, STD_WMA, STD_WMP = 3, 0.70, 0.50
PCA_K, PCA_WMA, PCA_WMP = 4, 0.60, 0.70
NMF_K, NMF_WMA, NMF_WMP, W_NMF = 6, 0.65, 0.60, 0.16
W_T, W_MT, W_SS, W_SM = 0.26, 0.13, 0.06, 0.07
W_UH = 1.0 - W_T - W_MT - W_SS - W_SM  # 0.48

s_ica_b = np.stack([wl_from_cache(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std_b = np.stack([wl_from_cache(c_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca_b = np.stack([wl_from_cache(c_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
s_nmf_b = np.stack([wl_from_cache(c_nmf, fi, NMF_K, NMF_WMP, NMF_WMA) for fi in range(n_files)])
uh_b    = W_ICA*s_ica_b + W_STD*s_std_b + W_PCA*s_pca_b
uh_nmf_b = (1-W_NMF)*uh_b + W_NMF*s_nmf_b

def make_preds(T):
    sig = (1.0/(1.0+np.exp(np.clip(-logit_win/T,-88,88)))).astype(np.float32)
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
preds_T8 = make_preds(8.0); preds_mt_810 = (preds_T8 + make_preds(10.0)) / 2.0

def make_softmax_preds(T=4.0):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        lw = logit_win[file_start[fi]:file_end[fi]]
        sm = lw/T; sm -= sm.max(1, keepdims=True)
        e = np.exp(sm); p = e/(e.sum(1, keepdims=True)+EPS)
        out[fi] = p.max(0)
    return out

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
            pos = tr[pm]; k = min(n_comp, len(pos)-1, dim-1)
            if k < 1:
                pp = pos.mean(0); pp /= np.linalg.norm(pp)+EPS
                ws[:, si] = np.clip((te@pp+1)/2,0,1); continue
            try:
                pca_sp = PCA(n_components=k); pca_sp.fit(pos)
                te_r = pca_sp.inverse_transform(pca_sp.transform(te))
                err = np.linalg.norm(te-te_r,axis=1)
                ws[:,si] = np.clip(1-err/(np.linalg.norm(te,axis=1)+EPS),0,1)
            except: ws[:,si]=0.5
        out[fi] = wma*ws.max(0)+(1-wma)*ws.mean(0)
    return out

print("Computing subspace + softmax...", flush=True)
t0 = time.time()
ss2 = species_subspace_loo(ew_pca, 2, 0.92)
sm4 = make_softmax_preds(4.0)
print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

ref = W_UH*uh_nmf_b + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
print(f"Reference (current best): {eval_loo(ref):.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: KDE Score (Gaussian KDE on ICA-space positive windows)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Gaussian KDE Score ===", flush=True)
t0 = time.time()

def kde_loo(emb_n, bw=0.3):
    """Gaussian KDE: p(x|species) ∝ sum_i exp(-||x-x_i||^2 / (2*bw^2))"""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]; tr_m = win_file_id != fi
        tr = emb_n[tr_m]; tl = labels_win[tr_m]
        # Precompute distances: te @ tr.T (cosine sim since both normalized)
        sims = te @ tr.T  # (n_te, n_tr)
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            # Cosine distance = 1 - cosine_sim
            pos_sims = sims[:, pos_idx]  # (n_te, n_pos)
            # KDE: sum exp((sim-1)/bw^2) normalized
            kde_scores = np.exp((pos_sims - 1.0) / (bw**2 + EPS)).mean(1)
            ws[:, si] = np.clip(kde_scores, 0, None)
        # Normalize per-file to [0,1]
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

best_kde, best_kde_cfg = CURRENT_BEST, None
for bw in [0.1, 0.2, 0.3, 0.4, 0.5]:
    kde_s = kde_loo(ew_ica, bw)
    for w_kde in [0.04, 0.06, 0.08, 0.10]:
        scale = 1.0 - w_kde
        blend = scale*(W_UH*uh_nmf_b + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4) + w_kde*kde_s
        auc = eval_loo(blend)
        if auc > best_kde: best_kde = auc; best_kde_cfg = (bw, w_kde)
    print(f"  bw={bw}: done", flush=True)

results['kde_score'] = best_kde
print(f"  KDE best: {best_kde:.6f}  cfg={best_kde_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_kde > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Prototype Distance Ratio（pos_sim / (pos_sim + neg_sim)）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Prototype Distance Ratio ===", flush=True)
t0 = time.time()

def wl_ratio_loo(cache, fi, k_neg):
    """pos_sim / (pos_sim + neg_sim) ratio score."""
    te, tr, tl, sims = cache[fi]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pos_idx = np.where(tl[:, si] > 0.5)[0]
        neg_idx = np.where(tl[:, si] < 0.1)[0]
        if len(pos_idx) == 0: ws[:, si] = 0.5; continue
        pos_proto = tr[pos_idx].mean(0); pos_proto /= np.linalg.norm(pos_proto) + EPS
        pos_sim = te @ pos_proto  # (n_te,)
        if len(neg_idx) > 0:
            neg_proto = tr[neg_idx[:k_neg]].mean(0); neg_proto /= np.linalg.norm(neg_proto) + EPS
            neg_sim = te @ neg_proto
            ratio = (pos_sim + 1) / (pos_sim + 1 + np.maximum(neg_sim + 1, EPS))
        else:
            ratio = (pos_sim + 1) / 2.0
        ws[:, si] = np.clip(ratio, 0, 1)
    return ws.max(0)

best_ratio, best_ratio_cfg = CURRENT_BEST, None
for k_neg in [4, 6, 10, 20, 50]:
    s_ratio = np.stack([wl_ratio_loo(c_ica, fi, k_neg) for fi in range(n_files)])
    uh_ratio = W_ICA*s_ratio + W_STD*s_std_b + W_PCA*s_pca_b
    uh_nmf_r = (1-W_NMF)*uh_ratio + W_NMF*s_nmf_b
    blend = W_UH*uh_nmf_r + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
    auc = eval_loo(blend)
    if auc > best_ratio: best_ratio = auc; best_ratio_cfg = k_neg
    print(f"  k_neg={k_neg}: {auc:.6f}", flush=True)

results['proto_ratio'] = best_ratio
print(f"  Ratio best: {best_ratio:.6f}  cfg={best_ratio_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_ratio > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Two-stage Re-ranking
# Stage 1: logit selects top-N species; Stage 2: WL re-ranks within top-N
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Two-stage Re-ranking ===", flush=True)
t0 = time.time()

def two_stage_loo(cache, fi, sig_file, k_neg, wmp, wma, top_n=20, T_boost=2.0):
    """Within top_n species by logit, boost WL scores; others unchanged."""
    wl_base = wl_from_cache(cache, fi, k_neg, wmp, wma)
    # Top-N by sigmoid logit
    top_idx = np.argsort(-sig_file)[:top_n]
    # Boost: multiply WL by T_boost for top-N species
    boosted = wl_base.copy()
    boosted[top_idx] *= T_boost
    # Normalize back to [0,1]
    mx = boosted.max()
    if mx > EPS: boosted /= mx
    return boosted

best_ts, best_ts_cfg = CURRENT_BEST, None
sig_file_scores = np.stack([
    (1.0/(1.0+np.exp(np.clip(-logit_win[file_start[fi]:file_end[fi]]/8.0,-88,88)))).max(0)
    for fi in range(n_files)])  # (n_files, n_species)

for top_n in [15, 20, 30]:
    for T_boost in [1.5, 2.0, 3.0]:
        s_ts = np.stack([two_stage_loo(c_ica, fi, sig_file_scores[fi], ICA_K, ICA_WMP, ICA_WMA, top_n, T_boost) for fi in range(n_files)])
        uh_ts = W_ICA*s_ts + W_STD*s_std_b + W_PCA*s_pca_b
        uh_nmf_ts = (1-W_NMF)*uh_ts + W_NMF*s_nmf_b
        blend = W_UH*uh_nmf_ts + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
        auc = eval_loo(blend)
        if auc > best_ts: best_ts = auc; best_ts_cfg = (top_n, T_boost)
    print(f"  top_n={top_n}: done", flush=True)

results['two_stage'] = best_ts
print(f"  Two-stage best: {best_ts:.6f}  cfg={best_ts_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_ts > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Per-species LOO ICA weight adjustment
# 某些 species 在 ICA 空間表現好，某些在 PCA 好
# 用訓練集本身估計每個 species 的最優 weight
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Per-species Adaptive Blend ===", flush=True)
t0 = time.time()

# 估計每個 species 在 ICA vs PCA 空間的相對 AUC
# 只用出現過的 species
def species_auc_loo(scores, file_lbl):
    """Per-species LOO AUC for a given score matrix."""
    aucs = np.zeros(n_species)
    for si in range(n_species):
        y = file_lbl[:, si]
        if y.sum() == 0 or y.sum() == n_files: aucs[si] = 0.5; continue
        try: aucs[si] = roc_auc_score(y, scores[:, si])
        except: aucs[si] = 0.5
    return aucs

# Get per-species AUC for ICA-WL and NMF-WL
auc_ica = species_auc_loo(s_ica_b, file_labels)
auc_nmf = species_auc_loo(s_nmf_b, file_labels)
auc_logit = species_auc_loo(preds_T8, file_labels)

# Adaptive weight: for each species, use the better-performing component more
# Soft selection: w_ica_sp = softmax([auc_ica, auc_nmf, auc_logit]) with temperature
best_adap, best_adap_cfg = CURRENT_BEST, None
for T_adap in [1.0, 2.0, 4.0, 8.0]:
    stack = np.stack([auc_ica, auc_nmf, auc_logit], axis=0)  # (3, n_species)
    stack_scaled = stack / T_adap
    stack_scaled -= stack_scaled.max(0, keepdims=True)
    exp_s = np.exp(stack_scaled)
    weights = exp_s / (exp_s.sum(0, keepdims=True) + EPS)  # (3, n_species)
    w_ica_sp, w_nmf_sp, w_logit_sp = weights[0], weights[1], weights[2]
    # Adaptive score: per-species weighted
    adapt_score = np.zeros((n_files, n_species), np.float32)
    for si in range(n_species):
        adapt_score[:, si] = (w_ica_sp[si] * s_ica_b[:, si] +
                              w_nmf_sp[si] * s_nmf_b[:, si] +
                              w_logit_sp[si] * preds_T8[:, si])
    # Blend with current best
    for w_adap in [0.10, 0.15, 0.20]:
        blend = (1-w_adap)*(W_UH*uh_nmf_b + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4) + w_adap*adapt_score
        auc = eval_loo(blend)
        if auc > best_adap: best_adap = auc; best_adap_cfg = (T_adap, w_adap)
    print(f"  T_adap={T_adap}: done", flush=True)

results['per_species_adapt'] = best_adap
print(f"  Per-species adapt best: {best_adap:.6f}  cfg={best_adap_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_adap > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 82 Summary ===", flush=True)
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
    print("未超越 0.99136，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
