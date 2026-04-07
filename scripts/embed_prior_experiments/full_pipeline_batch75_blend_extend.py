"""
Batch 75: Blend Extension & New Signal Sources

從 wl_full_fine (LOO=0.9911) 出發：
- w_ica=0.72, w_std=0.18, w_pca=0.10
- ICA: k=50, wma=0.88, wmp=0.85
- STD: k=3, wma=0.70, wmp=0.50  ← batch74 新發現
- PCA: k=4, wma=0.60, wmp=0.70
- T=8.0, w_logit=0.27, mt=[8,10], w_mt=0.15, ss×0.06

Batch 74 已微調：STD-PCA80 sub-params → +0.0001

現在嘗試：
1. ICA k_neg 更細密掃描（Batch 74 前 ICA k=50 固定）
2. Multi-T 擴展：更多溫度組合 [8,10,12], [6,8,10], [7,9,11]
3. 額外第五信號：softmax-normalized logit max（logit本身的 softmax → max species）
4. WL global weight 更細密掃描（w_ica 從 0.72 附近細掃）
"""
import numpy as np, json, os, time, pickle
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
CURRENT_BEST = 0.991136
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

with open("outputs/embed_prior_model.pkl", 'rb') as f:
    ep = pickle.load(f)
ew_ica = ep['emb_win_ica_norm'].astype(np.float32)
ew_pca = ep['emb_win_pca_norm'].astype(np.float32)
ew_std = ep['emb_win_std_norm'].astype(np.float32)

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

# Current best WL component params
W_ICA, W_STD, W_PCA = 0.72, 0.18, 0.10
ICA_K, ICA_WMA, ICA_WMP = 50, 0.88, 0.85
STD_K, STD_WMA, STD_WMP = 3, 0.70, 0.50  # batch74 updated
PCA_K, PCA_WMA, PCA_WMP = 4, 0.60, 0.70

# Precompute WL with best params
s_ica_b = np.stack([wl_from_cache(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std_b = np.stack([wl_from_cache(c_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca_b = np.stack([wl_from_cache(c_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
uh_b = W_ICA * s_ica_b + W_STD * s_std_b + W_PCA * s_pca_b

# Logit components
def make_preds(T):
    sig = (1.0/(1.0+np.exp(np.clip(-logit_win/T,-88,88)))).astype(np.float32)
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

preds_T8  = make_preds(8.0)
preds_T10 = make_preds(10.0)
preds_T12 = make_preds(12.0)
preds_T6  = make_preds(6.0)
preds_T7  = make_preds(7.0)
preds_T9  = make_preds(9.0)
preds_T11 = make_preds(11.0)
preds_mt_810 = (preds_T8 + preds_T10) / 2.0  # current best

# Softmax logit: softmax over species → max window
def make_softmax_preds(T=1.0):
    """Softmax over species dimension → max over windows"""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        lw = logit_win[file_start[fi]:file_end[fi]]  # (n_win, n_sp)
        sm = lw / T
        sm -= sm.max(1, keepdims=True)
        exp_sm = np.exp(sm)
        prob_sm = exp_sm / (exp_sm.sum(1, keepdims=True) + EPS)
        out[fi] = prob_sm.max(0)
    return out

# Subspace
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

print("Computing subspace...", flush=True)
t0 = time.time()
ss2 = species_subspace_loo(ew_pca, 2, 0.92)
print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

# Reference
W_T, W_MT, W_SS = 0.27, 0.15, 0.06
W_UH = 1.0 - W_T - W_MT - W_SS  # = 0.52
ref = W_UH*uh_b + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2
print(f"Reference (current best): {eval_loo(ref):.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: ICA k_neg 精細掃描 (batch74 前未調過)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: ICA k_neg Fine Scan ===", flush=True)
t0 = time.time()
best_ica_k, best_cfg_ica_k = 0, None

ICA_K_LIST = [20, 30, 40, 50, 60, 70, 80, 100]

for k_neg in ICA_K_LIST:
    s_ica_v = np.stack([wl_from_cache(c_ica, fi, k_neg, ICA_WMP, ICA_WMA) for fi in range(n_files)])
    uh_v = W_ICA * s_ica_v + W_STD * s_std_b + W_PCA * s_pca_b
    blend = W_UH*uh_v + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2
    auc = eval_loo(blend)
    if auc > best_ica_k: best_ica_k = auc; best_cfg_ica_k = k_neg
    print(f"  k_neg={k_neg}: {auc:.6f}", flush=True)

print(f"  ICA k_neg best: {best_ica_k:.6f}  k={best_cfg_ica_k}  ({time.time()-t0:.0f}s)", flush=True)
results['ica_k_scan'] = best_ica_k
print(f"  {'*** NEW BEST ***' if best_ica_k > CURRENT_BEST else ''}", flush=True)

# Update ICA if improved
ica_k_best = best_cfg_ica_k if best_cfg_ica_k else ICA_K
s_ica_b2 = np.stack([wl_from_cache(c_ica, fi, ica_k_best, ICA_WMP, ICA_WMA) for fi in range(n_files)])
uh_b2 = W_ICA * s_ica_b2 + W_STD * s_std_b + W_PCA * s_pca_b

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Multi-T 組合擴展
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Multi-T Extension ===", flush=True)
t0 = time.time()
best_mt_ext, best_cfg_mt_ext = 0, None

# 更多溫度組合
mt_configs = {
    "T[8,10]": preds_mt_810,  # current best
    "T[8,10,12]": (preds_T8 + preds_T10 + preds_T12) / 3.0,
    "T[6,8,10]": (preds_T6 + preds_T8 + preds_T10) / 3.0,
    "T[7,9,11]": (preds_T7 + preds_T9 + preds_T11) / 3.0,
    "T[8,9,10]": (preds_T8 + preds_T9 + preds_T10) / 3.0,
    "T[8,10,12,14]": (preds_T8 + preds_T10 + preds_T12 + make_preds(14.0)) / 4.0,
    "T[6,8,10,12]": (preds_T6 + preds_T8 + preds_T10 + preds_T12) / 4.0,
    "T[7,8,9,10]": (preds_T7 + preds_T8 + preds_T9 + preds_T10) / 4.0,
}

for name, mt_preds in mt_configs.items():
    blend = W_UH*uh_b2 + W_T*preds_T8 + W_MT*mt_preds + W_SS*ss2
    auc = eval_loo(blend)
    if auc > best_mt_ext: best_mt_ext = auc; best_cfg_mt_ext = (name, mt_preds)
    print(f"  {name}: {auc:.6f}", flush=True)

print(f"  Multi-T best: {best_mt_ext:.6f}  cfg={best_cfg_mt_ext[0] if best_cfg_mt_ext else None}  ({time.time()-t0:.0f}s)", flush=True)
results['multi_t_ext'] = best_mt_ext
print(f"  {'*** NEW BEST ***' if best_mt_ext > CURRENT_BEST else ''}", flush=True)

best_mt_preds = best_cfg_mt_ext[1] if best_cfg_mt_ext else preds_mt_810

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Softmax logit 作為第五信號
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Softmax Logit Signal ===", flush=True)
t0 = time.time()
best_softmax, best_cfg_softmax = 0, None

# Precompute softmax preds at multiple temperatures
softmax_configs = {}
for T_sm in [0.5, 1.0, 2.0, 4.0]:
    softmax_configs[f"sm_T{T_sm}"] = make_softmax_preds(T_sm)
    print(f"  Softmax T={T_sm} computed", flush=True)

# Try adding softmax as additional signal (5th)
for sm_name, sm_preds in softmax_configs.items():
    for w_sm in [0.04, 0.06, 0.08, 0.10]:
        # 按比例縮小其他權重
        total_other = W_UH + W_T + W_MT + W_SS
        scale = (1.0 - w_sm) / total_other
        blend = (scale*W_UH)*uh_b2 + (scale*W_T)*preds_T8 + (scale*W_MT)*best_mt_preds + (scale*W_SS)*ss2 + w_sm*sm_preds
        auc = eval_loo(blend)
        if auc > best_softmax: best_softmax = auc; best_cfg_softmax = (sm_name, w_sm)

print(f"  Softmax signal best: {best_softmax:.6f}  cfg={best_cfg_softmax}  ({time.time()-t0:.0f}s)", flush=True)
results['softmax_signal'] = best_softmax
print(f"  {'*** NEW BEST ***' if best_softmax > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: WL global weight 更細密掃描（w_ica ± 0.02 範圍）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: WL Global Weight Fine Scan ===", flush=True)
t0 = time.time()
best_wg, best_cfg_wg = 0, None

# 只掃 w_ica（w_std+w_pca 比例維持不變）
# 當前: w_ica=0.72, w_std=0.18, w_pca=0.10 → std:pca ratio = 0.18:0.10 = 1.8
for w_ica in [0.65, 0.68, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.77, 0.80]:
    rem = 1.0 - w_ica
    w_std_v = round(rem * (0.18/0.28), 4)
    w_pca_v = round(rem - w_std_v, 4)
    uh_v = w_ica*s_ica_b2 + w_std_v*s_std_b + w_pca_v*s_pca_b
    blend = W_UH*uh_v + W_T*preds_T8 + W_MT*best_mt_preds + W_SS*ss2
    auc = eval_loo(blend)
    if auc > best_wg: best_wg = auc; best_cfg_wg = (w_ica, w_std_v, w_pca_v)
    print(f"  w_ica={w_ica}: {auc:.6f}  (std={w_std_v}, pca={w_pca_v})", flush=True)

print(f"  WL global weight best: {best_wg:.6f}  cfg={best_cfg_wg}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_global_weight'] = best_wg
print(f"  {'*** NEW BEST ***' if best_wg > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Full combined with all best sub-params
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Full Combined All Best ===", flush=True)
t0 = time.time()

# Use best ICA k, best MT, best WL weights
w_ica_fc = best_cfg_wg[0] if best_cfg_wg else W_ICA
w_std_fc = best_cfg_wg[1] if best_cfg_wg else W_STD
w_pca_fc = best_cfg_wg[2] if best_cfg_wg else W_PCA

uh_fc = w_ica_fc*s_ica_b2 + w_std_fc*s_std_b + w_pca_fc*s_pca_b

best_fc, best_cfg_fc = 0, None
for w_T in [0.25, 0.26, 0.27, 0.28, 0.29]:
    for w_mt in [0.12, 0.13, 0.14, 0.15, 0.16, 0.17]:
        for w_ss in [0.04, 0.05, 0.06, 0.07]:
            w_uh = 1.0 - w_T - w_mt - w_ss
            if w_uh < 0.48 or w_uh > 0.62: continue
            blend = w_uh*uh_fc + w_T*preds_T8 + w_mt*best_mt_preds + w_ss*ss2
            auc = eval_loo(blend)
            if auc > best_fc: best_fc = auc; best_cfg_fc = (w_T, w_mt, w_ss, round(w_uh,3))

print(f"  Full combined: {best_fc:.6f}  cfg={best_cfg_fc}  ({time.time()-t0:.1f}s)", flush=True)
results['full_combined_b75'] = best_fc
print(f"  {'*** NEW BEST ***' if best_fc > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 75 Summary ===", flush=True)
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
    print("未超越 0.9911，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)

    # 更新 pkl 的 best config
    with open("outputs/embed_prior_model.pkl", 'rb') as f:
        ep_save = pickle.load(f)

    # 確定最優 ICA k
    save_ica_k = ica_k_best
    save_w_ica = best_cfg_wg[0] if best_cfg_wg else W_ICA
    save_w_std = best_cfg_wg[1] if best_cfg_wg else W_STD
    save_w_pca = best_cfg_wg[2] if best_cfg_wg else W_PCA
    save_w_T   = best_cfg_fc[0] if best_cfg_fc else W_T
    save_w_mt  = best_cfg_fc[1] if best_cfg_fc else W_MT
    save_w_ss  = best_cfg_fc[2] if best_cfg_fc else W_SS

    best_mt_name = best_cfg_mt_ext[0] if best_cfg_mt_ext else "T[8,10]"
    # multi-T temps 解析
    import re
    mt_nums = list(map(float, re.findall(r'[\d.]+', best_mt_name)))
    if not mt_nums: mt_nums = [8.0, 10.0]

    new_cfg = ep_save["config"].copy()
    new_cfg["description"] = (
        f"Batch75 full opt: ICA_k={save_ica_k}, w_ica={save_w_ica}, "
        f"T=8, w_T={save_w_T}, mt={mt_nums}/w={save_w_mt}. LOO={best_new_auc:.4f}"
    )
    new_cfg["ica100"]["k_neg"] = save_ica_k
    new_cfg["w_ica100"] = save_w_ica
    new_cfg["w_std"] = save_w_std
    new_cfg["w_pca80"] = save_w_pca
    new_cfg["w_logit"] = save_w_T
    new_cfg["w_multit"] = save_w_mt
    new_cfg["w_subspace"] = save_w_ss
    new_cfg["multit_temps"] = mt_nums

    ep_save["method"]  = "b75_full_opt"
    ep_save["loo_auc"] = best_new_auc
    ep_save["config"]  = new_cfg

    with open("outputs/embed_prior_model.pkl", 'wb') as f:
        pickle.dump(ep_save, f)
    print(f"  pkl 已更新：method=b75_full_opt, loo_auc={best_new_auc:.6f}", flush=True)
