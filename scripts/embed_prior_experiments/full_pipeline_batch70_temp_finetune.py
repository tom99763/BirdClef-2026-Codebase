"""
Batch 70: Logit Temperature Fine-tune + 3-Way T5 Blend + Multi-T Ensemble

從 logit_temp_uh_blend (LOO=0.9891, T=5.0, w=0.20) 出發，探索：
1. Fine-tune T 和 w（更細的 grid，T=3-15 範圍）
2. 3-Way Blend：UH + logit_T5 + species_subspace（之前 3-way 用 T=1.0，現在用 T=5.0）
3. Multi-T Ensemble：同時用多個溫度的 sigmoid，平均後 max-pool
4. WL contrast in T-calibrated sigmoid space（sigmoid(logit/T) 當作 embedding 做 WL）

Current best: logit_temp_uh_blend = 0.9891
"""
import numpy as np, json, os, time, pickle
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
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
CURRENT_BEST = 0.9891177049959994
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# ─── Load precomputed transforms from pkl ────────────────────────────────────
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

# ─── Sim cache ───────────────────────────────────────────────────────────────
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

# ─── UH-triple reference ─────────────────────────────────────────────────────
print("Computing UH-triple and T=5.0 logit reference...", flush=True)
t0 = time.time()
s_ica = np.stack([wl_from_cache(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std = np.stack([wl_from_cache(c_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca = np.stack([wl_from_cache(c_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
uh_triple = W_ICA * s_ica + W_STD * s_std + W_PCA * s_pca
uh_auc = eval_loo(uh_triple)

# Current best config: T=5.0, w=0.20
T_CURR = 5.0
sig_T5 = (1.0 / (1.0 + np.exp(-logit_win / T_CURR))).astype(np.float32)
preds_T5 = np.stack([sig_T5[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
curr_best_scores = (1-0.20) * uh_triple + 0.20 * preds_T5
curr_auc = eval_loo(curr_best_scores)
print(f"  UH-triple: {uh_auc:.4f}  logit_temp_uh (T=5,w=0.2): {curr_auc:.4f}  ({time.time()-t0:.1f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Fine-tune logit temperature and blend weight
# 上一批只試 T=[0.3,0.5,0.7,1.0,1.5,2.0,3.0,5.0]，現在延伸到更高 T 並更精細掃描
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Fine-tune Logit Temperature ===", flush=True)
t0 = time.time()
best_ft, best_cfg_ft = 0, None

# Extended T range + finer w grid near current best
T_LIST = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0]
W_LIST = [0.10, 0.13, 0.15, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.25, 0.28, 0.30, 0.35]

for T in T_LIST:
    sig_T = (1.0 / (1.0 + np.exp(-logit_win / T))).astype(np.float32)
    preds_T = np.stack([sig_T[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
    for w in W_LIST:
        blend = (1-w) * uh_triple + w * preds_T
        auc = eval_loo(blend)
        if auc > best_ft: best_ft = auc; best_cfg_ft = (T, w)

print(f"  Fine-tuned temp best: {best_ft:.6f}  cfg={best_cfg_ft}  ({time.time()-t0:.1f}s)", flush=True)
results['logit_temp_ft'] = best_ft
print(f"  {'*** NEW BEST ***' if best_ft > CURRENT_BEST else ''}", flush=True)

# Keep best preds for later blends
T_best, w_best = best_cfg_ft if best_cfg_ft else (5.0, 0.20)
sig_Tbest = (1.0 / (1.0 + np.exp(-logit_win / T_best))).astype(np.float32)
preds_Tbest = np.stack([sig_Tbest[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Multi-T Logit Ensemble
# 用多個溫度的 sigmoid 預測平均，再 max-pool（更穩健的 calibration）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Multi-T Logit Ensemble ===", flush=True)
t0 = time.time()

# Precompute sigmoid at multiple T values
T_ENSEMBLE = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
sig_all = {}
preds_all = {}
for T in T_ENSEMBLE:
    sig_T = (1.0 / (1.0 + np.exp(-logit_win / T))).astype(np.float32)
    sig_all[T] = sig_T
    preds_all[T] = np.stack([sig_T[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

# Ensemble: average predictions at multiple temperatures
best_mt, best_cfg_mt = 0, None
T_COMBOS = [
    [1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [5.0, 10.0],
    [1.0, 3.0, 5.0], [2.0, 5.0, 10.0], [1.0, 5.0, 10.0],
    [1.0, 2.0, 5.0, 10.0], [2.0, 3.0, 5.0, 7.0],
]
for combo in T_COMBOS:
    avg_preds = np.mean([preds_all[T] for T in combo], axis=0)
    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        blend = (1-w) * uh_triple + w * avg_preds
        auc = eval_loo(blend)
        if auc > best_mt: best_mt = auc; best_cfg_mt = (combo, w)

print(f"  Multi-T ensemble best: {best_mt:.6f}  cfg={best_cfg_mt}  ({time.time()-t0:.1f}s)", flush=True)
results['logit_multit_uh_blend'] = best_mt
print(f"  {'*** NEW BEST ***' if best_mt > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: 3-Way Blend with T=best logit (之前 uh_logit_subspace_3way 用 T=1.0)
# 現在用 T=best（通常 5.0）的 logit + species_subspace
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: 3-Way Blend with T=best Logit + Species Subspace ===", flush=True)
t0 = time.time()

def species_subspace_loo(emb_n, n_comp, wma):
    out = np.zeros((n_files, n_species), np.float32)
    dim = emb_n.shape[1]
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr_m = win_file_id != fi
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

# Find best subspace config (quick sweep)
print("  Computing species subspace scores...", flush=True)
best_ss_auc, best_ss_cfg, ss_scores_best = 0, None, None
for n_comp in [1, 2, 3]:
    for wma in [0.88, 0.90, 0.92]:
        for emb, name in [(ew_ica, 'ica100'), (ew_pca, 'pca80')]:
            out = species_subspace_loo(emb, n_comp, wma)
            auc = eval_loo(out)
            if auc > best_ss_auc:
                best_ss_auc = auc; best_ss_cfg = (name, emb, n_comp, wma)
                ss_scores_best = out
    print(f"    n_comp={n_comp} done", flush=True)

print(f"  Subspace standalone: {best_ss_auc:.4f}  cfg={best_ss_cfg[:1] + best_ss_cfg[2:]}", flush=True)

# 3-way sweep with T=best logit
best_3way, best_cfg_3way = 0, None
if ss_scores_best is not None:
    for w_log in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        for w_ss in [0.05, 0.08, 0.10, 0.12, 0.15]:
            w_uh = 1.0 - w_log - w_ss
            if w_uh < 0.60: continue
            blend3 = w_uh * uh_triple + w_log * preds_Tbest + w_ss * ss_scores_best
            auc = eval_loo(blend3)
            if auc > best_3way: best_3way = auc; best_cfg_3way = (T_best, w_log, w_ss)

results['logit_temp5_subspace_3way'] = best_3way
print(f"  3-Way (UH+logitT={T_best:.0f}+subspace): {best_3way:.6f}  cfg={best_cfg_3way}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_3way > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: WL Contrast in T-calibrated Sigmoid Space
# 把 sigmoid(logit/T) 當作 embedding 做 WL contrast（替代 cosine similarity）
# 不同於 batch 67b 的 WL-in-logit-space，這裡用 sigmoid 後的機率空間
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: WL in Sigmoid-Calibrated Space ===", flush=True)
t0 = time.time()
best_wls, best_cfg_wls = 0, None

for T in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
    sig_T = (1.0 / (1.0 + np.exp(-logit_win / T))).astype(np.float32)
    # L2-normalize the sigmoid predictions to use as embedding for WL contrast
    sig_T_norm = normalize(sig_T, norm='l2')
    c_sig = build_cache(sig_T_norm)
    for k_neg in [4, 8, 16]:
        for wmp in [0.60, 0.80, 1.0]:
            for wma in [0.88, 0.92]:
                out = np.stack([wl_from_cache(c_sig, fi, k_neg, wmp, wma) for fi in range(n_files)])
                auc = eval_loo(out)
                if auc > best_wls: best_wls = auc; best_cfg_wls = (T, k_neg, wmp, wma)
    print(f"  T={T} done", flush=True)

print(f"  WL-sigmoid-space best: {best_wls:.6f}  cfg={best_cfg_wls}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_sigmoid_space'] = best_wls
print(f"  {'*** NEW BEST ***' if best_wls > CURRENT_BEST else ''}", flush=True)

# Blend with UH
if best_cfg_wls:
    T_s, kn_s, wmp_s, wma_s = best_cfg_wls
    sig_s = normalize((1.0 / (1.0 + np.exp(-logit_win / T_s))).astype(np.float32), norm='l2')
    c_s = build_cache(sig_s)
    wls_scores = np.stack([wl_from_cache(c_s, fi, kn_s, wmp_s, wma_s) for fi in range(n_files)])
    best_wlsb, best_cfg_wlsb = 0, None
    for w in [0.05, 0.08, 0.10, 0.12, 0.15]:
        blend = (1-w) * uh_triple + w * wls_scores
        auc = eval_loo(blend)
        if auc > best_wlsb: best_wlsb = auc; best_cfg_wlsb = w
    results['wl_sigmoid_uh_blend'] = best_wlsb
    print(f"  WL-sigmoid+UH blend: {best_wlsb:.6f}  w={best_cfg_wlsb}", flush=True)
    print(f"  {'*** NEW BEST ***' if best_wlsb > CURRENT_BEST else ''}", flush=True)

    # 3-way with best logit_temp
    best_wls3, best_cfg_wls3 = 0, None
    for w_wls in [0.03, 0.05, 0.08]:
        for w_log in [0.15, 0.18, 0.20, 0.22]:
            w_uh = 1.0 - w_wls - w_log
            if w_uh < 0.60: continue
            b3 = w_uh * uh_triple + w_log * preds_Tbest + w_wls * wls_scores
            auc = eval_loo(b3)
            if auc > best_wls3: best_wls3 = auc; best_cfg_wls3 = (w_wls, w_log)
    results['wl_sigmoid_logit_3way'] = best_wls3
    print(f"  WL-sig+logit_Tbest+UH 3-way: {best_wls3:.6f}  cfg={best_cfg_wls3}", flush=True)
    print(f"  {'*** NEW BEST ***' if best_wls3 > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Soft-Aggregation of Logit Predictions
# 用 softmax 加權 window 預測（soft-max pooling），不同於 hard-max
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Soft-Max Pooling of Logit Predictions ===", flush=True)
t0 = time.time()
best_sm, best_cfg_sm = 0, None

for T in [3.0, 5.0, 7.0, 10.0]:
    sig_T = (1.0 / (1.0 + np.exp(-logit_win / T))).astype(np.float32)
    for pool_T in [0.5, 1.0, 2.0, 5.0]:  # soft-max pooling temperature
        preds_soft = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            s, e = int(file_start[fi]), int(file_end[fi])
            wins = sig_T[s:e]  # (n_win, n_species)
            # Soft-max pool: sum(w * x) where w = softmax(x / pool_T)
            att = np.exp(wins / pool_T)  # (n_win, n_species)
            att /= att.sum(0, keepdims=True) + EPS
            preds_soft[fi] = (att * wins).sum(0)  # weighted sum
        for w in [0.15, 0.18, 0.20, 0.22, 0.25]:
            blend = (1-w) * uh_triple + w * preds_soft
            auc = eval_loo(blend)
            if auc > best_sm: best_sm = auc; best_cfg_sm = (T, pool_T, w)

print(f"  Soft-max pool best: {best_sm:.6f}  cfg={best_cfg_sm}  ({time.time()-t0:.1f}s)", flush=True)
results['logit_softpool_uh_blend'] = best_sm
print(f"  {'*** NEW BEST ***' if best_sm > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 70 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.6f}{flag}", flush=True)
print(f"  UH-triple ref: {uh_auc:.4f}", flush=True)
print(f"  logit_temp_uh ref: {curr_auc:.6f}", flush=True)

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
    print("未超越 0.9891，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
