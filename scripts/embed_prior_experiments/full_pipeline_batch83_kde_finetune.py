"""
Batch 83: KDE Fine-tune

從 kde_score (LOO=0.99177) 出發：
- base = NMF-Ultra base (0.96×base + 0.04×KDE)
- KDE: bw=0.1, space=ICA, w_kde=0.04

重大突破：Gaussian KDE (bw=0.1) +0.0004 打破 4 batch 平台！

現在精細調優：
1. KDE bandwidth 精細掃描（0.05~0.5，以 0.05 間隔）
2. KDE weight 精細掃描（0.02~0.12）
3. KDE on PCA / STD / NMF 空間（目前只試了 ICA）
4. 多空間 KDE ensemble
5. 全局 blend 重新優化（帶 KDE 後）
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
CURRENT_BEST = 0.991769
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

def kde_score_loo(emb_n, bw):
    """LOO Gaussian KDE on embedding space."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]; tr_m = win_file_id != fi
        tr = emb_n[tr_m]; tl = labels_win[tr_m]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_sims = sims[:, pos_idx]
            kde_s = np.exp((pos_sims - 1.0) / (bw**2 + EPS)).mean(1)
            ws[:, si] = np.clip(kde_s, 0, None)
        mx = ws.max(0, keepdims=True); mn = ws.min(0, keepdims=True)
        ws = (ws - mn) / (mx - mn + EPS)
        out[fi] = ws.max(0)
    return out

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

base_b = W_UH*uh_nmf_b + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
kde_ica_01 = kde_score_loo(ew_ica, 0.1)
ref = 0.96*base_b + 0.04*kde_ica_01
print(f"Reference (current best): {eval_loo(ref):.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: KDE bandwidth 精細掃描
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: KDE Bandwidth Fine Scan ===", flush=True)
t0 = time.time()
best_bw, best_bw_cfg = CURRENT_BEST, None

BW_LIST = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
kde_cache = {0.10: kde_ica_01}  # already computed

for bw in BW_LIST:
    if bw not in kde_cache:
        kde_cache[bw] = kde_score_loo(ew_ica, bw)
    for w_kde in [0.03, 0.04, 0.05, 0.06]:
        blend = (1-w_kde)*base_b + w_kde*kde_cache[bw]
        auc = eval_loo(blend)
        if auc > best_bw: best_bw = auc; best_bw_cfg = (bw, w_kde)
    print(f"  bw={bw}: done", flush=True)

results['kde_bw_scan'] = best_bw
print(f"  BW best: {best_bw:.6f}  cfg={best_bw_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_bw > CURRENT_BEST else ''}", flush=True)

best_bw_val = best_bw_cfg[0] if best_bw_cfg else 0.10
best_kde = kde_cache.get(best_bw_val, kde_ica_01)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: KDE weight 精細掃描（用最優 bw）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: KDE Weight Fine Scan ===", flush=True)
t0 = time.time()
best_wkde, best_wkde_val = CURRENT_BEST, 0.04

for w_kde in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
    blend = (1-w_kde)*base_b + w_kde*best_kde
    auc = eval_loo(blend)
    if auc > best_wkde: best_wkde = auc; best_wkde_val = w_kde
    print(f"  w_kde={w_kde}: {auc:.6f}", flush=True)

results['kde_w_scan'] = best_wkde
print(f"  KDE weight best: {best_wkde:.6f}  w={best_wkde_val}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_wkde > CURRENT_BEST else ''}", flush=True)
w_kde_best = best_wkde_val

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: KDE on other spaces (PCA, STD, NMF)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: KDE on Other Spaces ===", flush=True)
t0 = time.time()
best_sp, best_sp_cfg = CURRENT_BEST, None

spaces = {'pca': ew_pca, 'std': ew_std, 'nmf': ew_nmf}
kde_by_space = {}
for sp_name, sp_emb in spaces.items():
    for bw in [0.05, 0.10, 0.15, 0.20]:
        k_s = kde_score_loo(sp_emb, bw)
        kde_by_space[(sp_name, bw)] = k_s
        for w_kde in [0.03, 0.04, 0.05, 0.06]:
            blend = (1-w_kde)*base_b + w_kde*k_s
            auc = eval_loo(blend)
            if auc > best_sp: best_sp = auc; best_sp_cfg = (sp_name, bw, w_kde)
    print(f"  {sp_name} done", flush=True)

results['kde_space_scan'] = best_sp
print(f"  Space KDE best: {best_sp:.6f}  cfg={best_sp_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_sp > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Multi-space KDE ensemble
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Multi-space KDE Ensemble ===", flush=True)
t0 = time.time()
best_ens, best_ens_cfg = CURRENT_BEST, None

# ICA+PCA KDE ensemble
kde_pca_b = kde_score_loo(ew_pca, best_bw_val)
for w_ica_k in [0.5, 0.6, 0.7, 0.8]:
    kde_ens = w_ica_k*best_kde + (1-w_ica_k)*kde_pca_b
    for w_kde in [0.03, 0.04, 0.05, 0.06]:
        blend = (1-w_kde)*base_b + w_kde*kde_ens
        auc = eval_loo(blend)
        if auc > best_ens: best_ens = auc; best_ens_cfg = (w_ica_k, w_kde)

# ICA+STD KDE ensemble
kde_std_b = kde_score_loo(ew_std, best_bw_val)
for w_ica_k in [0.5, 0.6, 0.7, 0.8]:
    kde_ens = w_ica_k*best_kde + (1-w_ica_k)*kde_std_b
    for w_kde in [0.03, 0.04, 0.05, 0.06]:
        blend = (1-w_kde)*base_b + w_kde*kde_ens
        auc = eval_loo(blend)
        if auc > best_ens: best_ens = auc; best_ens_cfg = ('ica+std', w_ica_k, w_kde)

print(f"  Multi-space KDE best: {best_ens:.6f}  cfg={best_ens_cfg}  ({time.time()-t0:.0f}s)", flush=True)
results['kde_multi_space'] = best_ens
print(f"  {'*** NEW BEST ***' if best_ens > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Full blend re-optimize with KDE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Full Blend Re-optimize ===", flush=True)
t0 = time.time()
best_full, best_cfg_full = CURRENT_BEST, None

for w_kde in [0.03, 0.04, 0.05, 0.06]:
    for w_T in [0.24, 0.25, 0.26, 0.27]:
        for w_mt in [0.11, 0.12, 0.13, 0.14]:
            for w_ss in [0.04, 0.05, 0.06]:
                for w_sm in [0.06, 0.07, 0.08]:
                    w_uh = 1.0 - w_T - w_mt - w_ss - w_sm
                    if w_uh < 0.44 or w_uh > 0.58: continue
                    base_v = w_uh*uh_nmf_b + w_T*preds_T8 + w_mt*preds_mt_810 + w_ss*ss2 + w_sm*sm4
                    blend = (1-w_kde)*base_v + w_kde*best_kde
                    auc = eval_loo(blend)
                    if auc > best_full: best_full = auc; best_cfg_full = (w_kde, w_T, w_mt, w_ss, w_sm, round(w_uh,4))

results['kde_full_blend'] = best_full
print(f"  Full blend: {best_full:.6f}  cfg={best_cfg_full}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_full > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 83 Summary ===", flush=True)
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
    print("未超越 0.99177，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
    # 更新 pkl
    with open("outputs/embed_prior_model.pkl", 'rb') as f:
        ep_save = pickle.load(f)

    if best_cfg_full:
        bw_kd, bw_T, bw_mt, bw_ss, bw_sm, bw_uh = best_cfg_full
    else:
        bw_kd = w_kde_best; bw_T, bw_mt, bw_ss, bw_sm, bw_uh = W_T, W_MT, W_SS, W_SM, W_UH

    ep_save["config"]["kde"]["bandwidth"] = float(best_bw_val)
    ep_save["config"]["kde"]["w_kde"] = float(bw_kd)
    ep_save["config"]["w_logit"]   = float(bw_T)
    ep_save["config"]["w_multit"]  = float(bw_mt)
    ep_save["config"]["w_subspace"] = float(bw_ss)
    ep_save["config"]["w_softmax"] = float(bw_sm)
    ep_save["config"]["description"] = (
        f"Batch83 KDE fine: bw={best_bw_val},w={bw_kd}; blend=uh×{bw_uh}+T×{bw_T}+mt×{bw_mt}+ss×{bw_ss}+sm×{bw_sm}. LOO={best_new_auc:.4f}"
    )
    ep_save["method"]  = "kde_fine"
    ep_save["loo_auc"] = best_new_auc

    with open("outputs/embed_prior_model.pkl", 'wb') as f:
        pickle.dump(ep_save, f)
    print(f"  pkl 已更新：kde_fine, loo_auc={best_new_auc:.6f}", flush=True)
