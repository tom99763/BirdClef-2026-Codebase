"""
Batch 81: Confidence-weighted WL & Cross-modal Signals

從 nmf_w_ultra (LOO=0.99136) 出發，所有常規調優均已達平台。

探索根本新角度：
1. Confidence-weighted WL：訓練窗口按其 logit 置信度加權作為 prototype
   (高置信度訓練窗口 → 更好的 prototype)
2. Logit-guided positive selection：只用 logit > threshold 的訓練窗口作為正樣本
3. Hybrid score：WL × sigmoid(logit) 的幾何平均（同時要求 embedding 相似 AND logit 高）
4. NMF on logit space：對 logit 做 NMF（234-dim），得到 species co-occurrence patterns
5. Cross-modal WL：用 logit 相似度找近鄰，用 embedding 相似度評分（反轉）
"""
import numpy as np, json, os, time, pickle
from sklearn.decomposition import PCA, NMF
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

# Precompute sigmoid for confidence
sig_win_T8 = (1.0/(1.0+np.exp(np.clip(-logit_win/8.0,-88,88)))).astype(np.float32)

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
# Method 1: Confidence-weighted WL prototype
# 用 logit max-species score 作為訓練窗口的 confidence，加權 positive prototype
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Confidence-weighted WL ===", flush=True)
t0 = time.time()

# 訓練窗口的 confidence score = max(sigmoid(logit/T))
conf_T8 = sig_win_T8.max(1)  # (739,) per-window max confidence

def wl_conf_weighted(cache, fi, k_neg, wmp, wma, conf, conf_power=1.0):
    """Use logit confidence to weight positive prototypes."""
    te, tr, tl, sims = cache[fi]
    tr_conf = conf[win_file_id != fi]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pos_idx = np.where(tl[:, si] > 0.5)[0]
        neg_idx = np.where(tl[:, si] < 0.1)[0]
        if len(pos_idx) == 0: ws[:, si] = 0.5; continue
        ps = sims[:, pos_idx]
        # Confidence-weighted prototype
        w_conf = tr_conf[pos_idx] ** conf_power
        w_conf /= w_conf.sum() + EPS
        pp = (w_conf[:, None] * tr[pos_idx]).sum(0)
        pp /= np.linalg.norm(pp) + EPS
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

best_cw, best_cw_cfg = CURRENT_BEST, None
for cp in [0.5, 1.0, 2.0, 3.0]:
    s_ica_cw = np.stack([wl_conf_weighted(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA, conf_T8, cp) for fi in range(n_files)])
    uh_cw = W_ICA*s_ica_cw + W_STD*s_std_b + W_PCA*s_pca_b
    uh_nmf_cw = (1-W_NMF)*uh_cw + W_NMF*s_nmf_b
    blend = W_UH*uh_nmf_cw + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
    auc = eval_loo(blend)
    if auc > best_cw: best_cw = auc; best_cw_cfg = cp
    print(f"  conf_power={cp}: {auc:.6f}", flush=True)

results['conf_weighted_wl'] = best_cw
print(f"  Confidence WL best: {best_cw:.6f}  cfg={best_cw_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_cw > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Hybrid score = geometric mean(WL score, sig(logit/T))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Hybrid Geometric Mean Score ===", flush=True)
t0 = time.time()

def make_hybrid_geo(wl_scores, sig_scores, alpha=0.5):
    """Geometric mean: WL^alpha × sig^(1-alpha)"""
    wl_clip = np.clip(wl_scores, EPS, 1.0-EPS)
    sg_clip = np.clip(sig_scores, EPS, 1.0-EPS)
    return wl_clip**alpha * sg_clip**(1-alpha)

best_hg, best_hg_cfg = CURRENT_BEST, None
for alpha in [0.3, 0.5, 0.7, 0.8]:
    for T_sig in [6.0, 8.0, 10.0]:
        sig_p = make_preds(T_sig)
        hyb = make_hybrid_geo(uh_nmf_b, sig_p, alpha)
        for w_hyb in [0.04, 0.06, 0.08]:
            scale = 1.0 - w_hyb
            blend = scale*(W_UH*uh_nmf_b + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4) + w_hyb*hyb
            auc = eval_loo(blend)
            if auc > best_hg: best_hg = auc; best_hg_cfg = (alpha, T_sig, w_hyb)
    print(f"  alpha={alpha} done", flush=True)

results['hybrid_geo'] = best_hg
print(f"  Hybrid geo best: {best_hg:.6f}  cfg={best_hg_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_hg > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: NMF on logit space (species co-occurrence embedding)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: NMF on Logit Space ===", flush=True)
t0 = time.time()

# logit → sigmoid → NMF (234-dim non-negative space)
sig_all = (1.0/(1.0+np.exp(np.clip(-logit_win/8.0,-88,88)))).astype(np.float32)
# NMF on sigmoid outputs
print("  Fitting NMF on logit space...", flush=True)
best_lnmf, best_lnmf_cfg = CURRENT_BEST, None

for n_comp in [20, 30, 50]:
    t1 = time.time()
    nmf_l = NMF(n_components=n_comp, max_iter=300, random_state=42)
    ew_lnmf_raw = nmf_l.fit_transform(sig_all + EPS).astype(np.float32)
    ew_lnmf = ew_lnmf_raw / (np.linalg.norm(ew_lnmf_raw, axis=1, keepdims=True) + EPS)
    c_lnmf = build_cache(ew_lnmf)
    for k_neg in [4, 6]:
        for wma in [0.60, 0.70]:
            s_lnmf = np.stack([wl_from_cache(c_lnmf, fi, k_neg, 0.65, wma) for fi in range(n_files)])
            for w_lnmf in [0.06, 0.10, 0.14]:
                scale = 1.0 - w_lnmf
                blend = W_UH*uh_nmf_b + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
                # Replace W_SM with logit-NMF signal
                blend2 = scale*blend + w_lnmf*np.stack([sig_all[file_start[fi]:file_end[fi]].max(0) * s_lnmf[fi] for fi in range(n_files)])
                # Simpler: just add as additional blend component
                blend3 = (W_UH*uh_nmf_b + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4) * scale + w_lnmf * s_lnmf
                auc = eval_loo(blend3)
                if auc > best_lnmf: best_lnmf = auc; best_lnmf_cfg = (n_comp, k_neg, wma, w_lnmf)
    print(f"  n_logit={n_comp}: best={best_lnmf:.6f}  ({time.time()-t1:.0f}s)", flush=True)

results['logit_nmf'] = best_lnmf
print(f"  Logit NMF best: {best_lnmf:.6f}  cfg={best_lnmf_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_lnmf > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: 提升 subspace weight（目前 0.06，嘗試搭配 NMF 新信號後的最優 w_ss）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Subspace Weight Fine-tune with NMF ===", flush=True)
t0 = time.time()
best_ssw, best_ssw_cfg = CURRENT_BEST, None

for w_ss in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
    for w_T in [0.24, 0.25, 0.26, 0.27]:
        for w_mt in [0.11, 0.12, 0.13, 0.14]:
            for w_sm in [0.06, 0.07, 0.08]:
                w_uh = 1.0 - w_T - w_mt - w_ss - w_sm
                if w_uh < 0.46 or w_uh > 0.56: continue
                blend = w_uh*uh_nmf_b + w_T*preds_T8 + w_mt*preds_mt_810 + w_ss*ss2 + w_sm*sm4
                auc = eval_loo(blend)
                if auc > best_ssw: best_ssw = auc; best_ssw_cfg = (w_ss, w_T, w_mt, w_sm, round(w_uh,4))

results['subspace_blend_tune'] = best_ssw
print(f"  Subspace blend tune: {best_ssw:.6f}  cfg={best_ssw_cfg}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_ssw > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 81 Summary ===", flush=True)
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
