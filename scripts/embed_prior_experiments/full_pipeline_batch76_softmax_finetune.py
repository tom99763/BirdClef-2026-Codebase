"""
Batch 76: Softmax Signal Fine-tune & New Directions

從 b75_softmax (LOO=0.99115) 出發：
- WL-UH(0.4888) + T8×0.2538 + mt[8,10]×0.141 + ss×0.0564 + softmax(T=4)×0.06

Batch 75 發現 softmax(T=4, w=0.06) 提供第五正交信號。

現在探索：
1. Softmax 溫度精細掃描（T=2.0 ~ 6.0 細密）
2. Softmax 權重精細掃描（w=0.04~0.12）
3. Top-K softmax（只保留前K個最高分的 softmax 分量）
4. Logit 差異信號：max logit - 2nd max logit（margin signal）
5. 全局 fine-tune：softmax最優 T + 最優 w 組合
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
CURRENT_BEST = 0.9911523606969138
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

# Current best params
W_ICA, W_STD, W_PCA = 0.72, 0.18, 0.10
ICA_K, ICA_WMA, ICA_WMP = 50, 0.88, 0.85
STD_K, STD_WMA, STD_WMP = 3, 0.70, 0.50
PCA_K, PCA_WMA, PCA_WMP = 4, 0.60, 0.70
# batch75 blend weights (scale=0.94)
W_UH_B = 0.4888; W_T_B = 0.2538; W_MT_B = 0.141; W_SS_B = 0.0564; W_SM_B = 0.06

# Precompute WL
s_ica_b = np.stack([wl_from_cache(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std_b = np.stack([wl_from_cache(c_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca_b = np.stack([wl_from_cache(c_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
uh_b = W_ICA * s_ica_b + W_STD * s_std_b + W_PCA * s_pca_b

# Logit components
def make_preds(T):
    sig = (1.0/(1.0+np.exp(np.clip(-logit_win/T,-88,88)))).astype(np.float32)
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

preds_T8  = make_preds(8.0)
preds_mt_810 = (preds_T8 + make_preds(10.0)) / 2.0

def make_softmax_preds(T):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        lw = logit_win[file_start[fi]:file_end[fi]]
        sm = lw / T; sm -= sm.max(1, keepdims=True)
        exp_sm = np.exp(sm); prob_sm = exp_sm / (exp_sm.sum(1, keepdims=True) + EPS)
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

# Reference blend
ref_sm4 = make_softmax_preds(4.0)
ref = W_UH_B*uh_b + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*ref_sm4
print(f"Reference (current best): {eval_loo(ref):.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Softmax T 精細掃描 [1.5 ~ 8.0]
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Softmax Temperature Fine Scan ===", flush=True)
t0 = time.time()
best_sm_t, best_t_val = 0, 4.0

SM_T_LIST = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]
sm_preds_cache = {}

for T_sm in SM_T_LIST:
    sm_p = make_softmax_preds(T_sm)
    sm_preds_cache[T_sm] = sm_p
    blend = W_UH_B*uh_b + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*sm_p
    auc = eval_loo(blend)
    if auc > best_sm_t: best_sm_t = auc; best_t_val = T_sm
    print(f"  T_sm={T_sm}: {auc:.6f}", flush=True)

print(f"  Softmax T best: {best_sm_t:.6f}  T={best_t_val}  ({time.time()-t0:.0f}s)", flush=True)
results['softmax_t_scan'] = best_sm_t
print(f"  {'*** NEW BEST ***' if best_sm_t > CURRENT_BEST else ''}", flush=True)

best_sm_preds = sm_preds_cache[best_t_val]

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Softmax 權重精細掃描
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Softmax Weight Fine Scan ===", flush=True)
t0 = time.time()
best_sm_w, best_w_val = 0, 0.06

# 用 best T 做 weight scan
W_BASE_TOTAL = W_UH_B + W_T_B + W_MT_B + W_SS_B  # ≈ 0.94 (= 1-0.06 currently)
for w_sm in [0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]:
    scale = (1.0 - w_sm) / (W_BASE_TOTAL + W_SM_B)  # rescale all
    blend = (scale*W_UH_B)*uh_b + (scale*W_T_B)*preds_T8 + (scale*W_MT_B)*preds_mt_810 + (scale*W_SS_B)*ss2 + w_sm*best_sm_preds
    auc = eval_loo(blend)
    if auc > best_sm_w: best_sm_w = auc; best_w_val = w_sm
    print(f"  w_sm={w_sm}: {auc:.6f}", flush=True)

print(f"  Softmax weight best: {best_sm_w:.6f}  w={best_w_val}  ({time.time()-t0:.0f}s)", flush=True)
results['softmax_w_scan'] = best_sm_w
print(f"  {'*** NEW BEST ***' if best_sm_w > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Logit margin signal（max - 2nd_max）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Logit Margin Signal ===", flush=True)
t0 = time.time()

def make_margin_preds():
    """max logit - 2nd max logit → max over windows"""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        lw = logit_win[file_start[fi]:file_end[fi]]  # (n_win, n_sp)
        # For each window, compute margin for each species:
        # margin[j, si] = logit[j, si] - max_{sk != si} logit[j, sk]
        top2 = np.sort(lw, axis=1)[:, -2:]  # (n_win, 2): [2nd_max, max]
        max_other = top2[:, -1:] - lw  # how much each species dominates the max
        # When si IS the max: margin = logit[si] - 2nd_max
        # When si is NOT the max: negative
        # Margin per species = logit - (overall max if si != max, else 2nd max)
        # Simpler: use sigmoid of (logit - mean_other_logits)
        mean_logit = lw.mean(1, keepdims=True)
        margin = lw - mean_logit  # centered logit (vs mean instead of max)
        # Normalize to [0,1] via sigmoid
        score_margin = 1.0 / (1.0 + np.exp(np.clip(-margin, -88, 88)))
        out[fi] = score_margin.max(0)
    return out

margin_preds = make_margin_preds()
best_margin, best_margin_cfg = 0, None
for w_mg in [0.04, 0.06, 0.08, 0.10]:
    scale = (1.0 - w_mg)
    blend = scale*(W_UH_B*uh_b + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*best_sm_preds) + w_mg*margin_preds
    auc = eval_loo(blend)
    if auc > best_margin: best_margin = auc; best_margin_cfg = w_mg
    print(f"  w_margin={w_mg}: {auc:.6f}", flush=True)

print(f"  Margin signal best: {best_margin:.6f}  cfg={best_margin_cfg}  ({time.time()-t0:.0f}s)", flush=True)
results['margin_signal'] = best_margin
print(f"  {'*** NEW BEST ***' if best_margin > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: 全局 fine-tune（最優 softmax T + w）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Full Fine-tune (best softmax T + w) ===", flush=True)
t0 = time.time()
best_full, best_cfg_full = 0, None

for w_sm in [0.04, 0.05, 0.06, 0.07, 0.08]:
    for w_T in [0.24, 0.25, 0.26, 0.27]:
        for w_mt in [0.13, 0.14, 0.15, 0.16]:
            for w_ss in [0.04, 0.05, 0.06]:
                w_uh = 1.0 - w_T - w_mt - w_ss - w_sm
                if w_uh < 0.45 or w_uh > 0.56: continue
                blend = w_uh*uh_b + w_T*preds_T8 + w_mt*preds_mt_810 + w_ss*ss2 + w_sm*best_sm_preds
                auc = eval_loo(blend)
                if auc > best_full: best_full = auc; best_cfg_full = (w_sm, w_T, w_mt, w_ss, round(w_uh,4))

print(f"  Full fine-tune: {best_full:.6f}  cfg={best_cfg_full}  ({time.time()-t0:.1f}s)", flush=True)
results['softmax_full_tune'] = best_full
print(f"  {'*** NEW BEST ***' if best_full > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 76 Summary ===", flush=True)
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
    print("未超越 0.99115，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
    # 更新 pkl
    with open("outputs/embed_prior_model.pkl", 'rb') as f:
        ep_save = pickle.load(f)

    if best_cfg_full:
        bw_sm, bw_T, bw_mt, bw_ss, bw_uh = best_cfg_full
    else:
        bw_sm, bw_T, bw_mt, bw_ss, bw_uh = W_SM_B, W_T_B, W_MT_B, W_SS_B, W_UH_B

    new_cfg = ep_save["config"].copy()
    new_cfg["description"] = (
        f"Batch76: softmax(T={best_t_val})/w={bw_sm}+UH×{bw_uh}+T8×{bw_T}+mt×{bw_mt}+ss×{bw_ss}. LOO={best_new_auc:.4f}"
    )
    new_cfg["w_logit"]      = bw_T
    new_cfg["w_multit"]     = bw_mt
    new_cfg["w_subspace"]   = bw_ss
    new_cfg["w_softmax"]    = bw_sm
    new_cfg["softmax_temp"] = float(best_t_val)

    ep_save["method"]  = "b76_softmax_tune"
    ep_save["loo_auc"] = best_new_auc
    ep_save["config"]  = new_cfg

    with open("outputs/embed_prior_model.pkl", 'wb') as f:
        pickle.dump(ep_save, f)
    print(f"  pkl 已更新：method=b76_softmax_tune, loo_auc={best_new_auc:.6f}", flush=True)
