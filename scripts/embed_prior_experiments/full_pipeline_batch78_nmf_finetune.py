"""
Batch 78: NMF Fine-tune & Extended NMF Exploration

從 nmf_wl (LOO=0.99125) 出發：
- uh_nmf = 0.84×UH + 0.16×NMF(n=100, k=6, wma=0.60, wmp=0.70)
- 全局: W_UH_B×uh_nmf + T8×0.2538 + mt×0.141 + ss×0.0564 + sm4×0.06

Batch 77 只掃了 n=40/60/80/100，且 k/wma/wmp 只用了少量組合。
現在精細調優：
1. NMF n_components 繼續增加（120, 150, 200）
2. NMF k_neg / wma / wmp 精細掃描（當前 k=6, wma=0.60, wmp=0.70）
3. w_nmf 精細掃描（當前 w_nmf=0.16）
4. 全局 blend 重新優化（帶 NMF 的新 UH 基礎）
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
CURRENT_BEST = 0.9912498
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

with open("outputs/embed_prior_model.pkl", 'rb') as f:
    ep = pickle.load(f)
ew_ica = ep['emb_win_ica_norm'].astype(np.float32)
ew_pca = ep['emb_win_pca_norm'].astype(np.float32)
ew_std = ep['emb_win_std_norm'].astype(np.float32)

# NMF shift
emb_min = ep['emb_min_shift']
emb_shifted = emb_win - emb_min + 1e-6

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

print("Building ICA/PCA/STD caches...", flush=True)
t0 = time.time()
c_ica = build_cache(ew_ica); c_std = build_cache(ew_std); c_pca = build_cache(ew_pca)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

# Current best WL
W_ICA, W_STD, W_PCA = 0.72, 0.18, 0.10
ICA_K, ICA_WMA, ICA_WMP = 50, 0.88, 0.85
STD_K, STD_WMA, STD_WMP = 3, 0.70, 0.50
PCA_K, PCA_WMA, PCA_WMP = 4, 0.60, 0.70
W_UH_B, W_T_B, W_MT_B, W_SS_B, W_SM_B = 0.4888, 0.2538, 0.141, 0.0564, 0.06

s_ica_b = np.stack([wl_from_cache(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std_b = np.stack([wl_from_cache(c_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca_b = np.stack([wl_from_cache(c_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
uh_b = W_ICA * s_ica_b + W_STD * s_std_b + W_PCA * s_pca_b

def make_preds(T):
    sig = (1.0/(1.0+np.exp(np.clip(-logit_win/T,-88,88)))).astype(np.float32)
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
preds_T8 = make_preds(8.0)
preds_mt_810 = (preds_T8 + make_preds(10.0)) / 2.0

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

# 載入已有 NMF-100 (batch77)
ew_nmf100 = ep['emb_win_nmf_norm'].astype(np.float32)
c_nmf100 = build_cache(ew_nmf100)

s_nmf100_b = np.stack([wl_from_cache(c_nmf100, fi, 6, 0.70, 0.60) for fi in range(n_files)])
uh_nmf_curr = 0.84 * uh_b + 0.16 * s_nmf100_b

ref = W_UH_B*uh_nmf_curr + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*sm4
print(f"Reference (current best): {eval_loo(ref):.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: NMF n_components 繼續增加
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: NMF n_components Extended ===", flush=True)
t0 = time.time()
best_nmf_n, best_nmf_n_cfg = CURRENT_BEST, None

for n_comp in [120, 150, 200]:
    t1 = time.time()
    nmf_v = NMF(n_components=n_comp, max_iter=300, random_state=42)
    ew_v_raw = nmf_v.fit_transform(emb_shifted).astype(np.float32)
    ew_v = ew_v_raw / (np.linalg.norm(ew_v_raw, axis=1, keepdims=True) + EPS)
    c_v = build_cache(ew_v)
    s_v = np.stack([wl_from_cache(c_v, fi, 6, 0.70, 0.60) for fi in range(n_files)])
    for w_nmf in [0.12, 0.16, 0.20]:
        uh_v = (1-w_nmf)*uh_b + w_nmf*s_v
        blend = W_UH_B*uh_v + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*sm4
        auc = eval_loo(blend)
        if auc > best_nmf_n: best_nmf_n = auc; best_nmf_n_cfg = (n_comp, 6, 0.60, 0.70, w_nmf)
    print(f"  n={n_comp}: best={best_nmf_n:.6f}  ({time.time()-t1:.0f}s)", flush=True)

results['nmf_n_ext'] = best_nmf_n
print(f"  NMF n_ext best: {best_nmf_n:.6f}  cfg={best_nmf_n_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_nmf_n > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: NMF-100 k_neg / wma / wmp 精細掃描
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: NMF-100 Sub-param Fine-tune ===", flush=True)
t0 = time.time()
best_nmf_p, best_nmf_p_cfg = CURRENT_BEST, None

for k_neg in [4, 5, 6, 8, 10]:
    for wma in [0.55, 0.60, 0.65, 0.70]:
        for wmp in [0.60, 0.65, 0.70, 0.75, 0.80]:
            s_v = np.stack([wl_from_cache(c_nmf100, fi, k_neg, wmp, wma) for fi in range(n_files)])
            uh_v = 0.84*uh_b + 0.16*s_v
            blend = W_UH_B*uh_v + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*sm4
            auc = eval_loo(blend)
            if auc > best_nmf_p: best_nmf_p = auc; best_nmf_p_cfg = (k_neg, wma, wmp)
    print(f"  k={k_neg} done", flush=True)

results['nmf_param_fine'] = best_nmf_p
print(f"  NMF sub-param best: {best_nmf_p:.6f}  cfg={best_nmf_p_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_nmf_p > CURRENT_BEST else ''}", flush=True)

# Update NMF WL scores with best sub-params
if best_nmf_p_cfg:
    k_b, wma_b, wmp_b = best_nmf_p_cfg
else:
    k_b, wma_b, wmp_b = 6, 0.60, 0.70
s_nmf_best = np.stack([wl_from_cache(c_nmf100, fi, k_b, wmp_b, wma_b) for fi in range(n_files)])

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: w_nmf 精細掃描
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: w_nmf Fine Scan ===", flush=True)
t0 = time.time()
best_wmf, best_wmf_val = CURRENT_BEST, 0.16

for w_nmf in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.28]:
    uh_v = (1-w_nmf)*uh_b + w_nmf*s_nmf_best
    blend = W_UH_B*uh_v + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*sm4
    auc = eval_loo(blend)
    if auc > best_wmf: best_wmf = auc; best_wmf_val = w_nmf
    print(f"  w_nmf={w_nmf}: {auc:.6f}", flush=True)

results['nmf_w_scan'] = best_wmf
print(f"  w_nmf best: {best_wmf:.6f}  w={best_wmf_val}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_wmf > CURRENT_BEST else ''}", flush=True)

# Update UH with best w_nmf
w_nmf_best = best_wmf_val
uh_nmf_best = (1-w_nmf_best)*uh_b + w_nmf_best*s_nmf_best

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: 全局 blend 重新優化（帶新 NMF UH 基礎）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Full Blend Re-optimize with NMF ===", flush=True)
t0 = time.time()
best_full, best_cfg_full = CURRENT_BEST, None

for w_T in [0.23, 0.24, 0.25, 0.26, 0.27]:
    for w_mt in [0.12, 0.13, 0.14, 0.15, 0.16]:
        for w_ss in [0.04, 0.05, 0.06]:
            for w_sm in [0.04, 0.05, 0.06, 0.07]:
                w_uh = 1.0 - w_T - w_mt - w_ss - w_sm
                if w_uh < 0.46 or w_uh > 0.58: continue
                blend = w_uh*uh_nmf_best + w_T*preds_T8 + w_mt*preds_mt_810 + w_ss*ss2 + w_sm*sm4
                auc = eval_loo(blend)
                if auc > best_full: best_full = auc; best_cfg_full = (w_T, w_mt, w_ss, w_sm, round(w_uh,4))

results['nmf_full_blend'] = best_full
print(f"  Full blend: {best_full:.6f}  cfg={best_cfg_full}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_full > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 78 Summary ===", flush=True)
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
    print("未超越 0.99125，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
    # pkl 更新
    with open("outputs/embed_prior_model.pkl", 'rb') as f:
        ep_save = pickle.load(f)

    # 選取 best NMF n（若有更好）
    if best_nmf_n_cfg and best_nmf_n > CURRENT_BEST:
        # 需要重新 fit 更大的 NMF
        best_n = best_nmf_n_cfg[0]
        print(f"  重新 fit NMF n={best_n}...", flush=True)
        nmf_new = NMF(n_components=best_n, max_iter=300, random_state=42)
        ew_new_raw = nmf_new.fit_transform(emb_shifted).astype(np.float32)
        ew_new = ew_new_raw / (np.linalg.norm(ew_new_raw, axis=1, keepdims=True) + EPS)
        ep_save['nmf_model'] = nmf_new
        ep_save['emb_win_nmf_norm'] = ew_new
        ep_save['config']['nmf']['n_components'] = best_n

    new_cfg = ep_save["config"].copy()
    new_cfg["nmf"]["k_neg"]      = k_b
    new_cfg["nmf"]["w_max_agg"]  = wma_b
    new_cfg["nmf"]["w_max_pos"]  = wmp_b
    new_cfg["nmf"]["w_nmf"]      = w_nmf_best
    new_cfg["nmf"]["uh_scale"]   = round(1.0 - w_nmf_best, 4)
    if best_cfg_full:
        new_cfg["w_logit"]   = best_cfg_full[0]
        new_cfg["w_multit"]  = best_cfg_full[1]
        new_cfg["w_subspace"] = best_cfg_full[2]
        new_cfg["w_softmax"] = best_cfg_full[3]
    new_cfg["description"] = f"Batch78 NMF fine: n={new_cfg['nmf']['n_components']},k={k_b},w={w_nmf_best}. LOO={best_new_auc:.4f}"

    ep_save["method"]  = "nmf_fine"
    ep_save["loo_auc"] = best_new_auc
    ep_save["config"]  = new_cfg

    with open("outputs/embed_prior_model.pkl", 'wb') as f:
        pickle.dump(ep_save, f)
    print(f"  pkl 已更新：method=nmf_fine, loo_auc={best_new_auc:.6f}", flush=True)
