"""
Batch 73: Ultra Fine-tune All Components

從 logit_4way_wl_tune (LOO=0.9908) 出發，進一步精細調優：
1. WL 元件權重更細掃描（w_ica 0.65-0.75, w_std 0.15-0.22）
2. ICA100 WL 參數微調（k_neg, wma, wmp 在最優附近精細掃描）
3. 主 T 和 w_logit 在 7.0 附近精細掃描
4. multi-T 組合精細探索
5. 全局最優組合（合併所有最優設定）

Current best: logit_4way_wl_tune = 0.9908
(w_ica=0.70, w_std=0.18, w_pca=0.12, T=7.0, w_T=0.30, mt=[8,12], w_mt=0.12, ss×0.06)
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
CURRENT_BEST = 0.990793
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# ─── Load pkl ────────────────────────────────────────────────────────────────
print("Loading pkl...", flush=True)
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

# ─── Precompute logit preds ───────────────────────────────────────────────────
def make_preds(T):
    sig = (1.0/(1.0+np.exp(np.clip(-logit_win/T,-88,88)))).astype(np.float32)
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

T_RANGE = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
preds_T = {T: make_preds(T) for T in T_RANGE}

# Subspace (best: pca80, n_comp=2, wma=0.92 from batch 72 ss_variant)
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

print("Computing subspace (pca80, n_comp=2, wma=0.92)...", flush=True)
t0 = time.time()
ss2 = species_subspace_loo(ew_pca, 2, 0.92)
print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Fine-tune WL component weights (精細 grid)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Fine-tune WL Component Weights ===", flush=True)
t0 = time.time()
best_wlt, best_cfg_wlt = 0, None

# Current best: w_ica=0.70, w_std=0.18, T=7.0, w_T=0.30, mt=[8,12], w_mt=0.12, ss×0.06
preds_T7 = preds_T[7.0]
preds_mt_812 = (preds_T[8.0] + preds_T[12.0]) / 2.0
W_T_FIX, W_MT_FIX, W_SS_FIX = 0.30, 0.12, 0.06
W_UH_FIX = 1.0 - W_T_FIX - W_MT_FIX - W_SS_FIX  # = 0.52

W_ICA_LIST = [0.62, 0.65, 0.67, 0.68, 0.70, 0.72, 0.74, 0.75, 0.78]
W_STD_LIST = [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22]

for w_ica in W_ICA_LIST:
    for w_std in W_STD_LIST:
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.25: continue
        s_ica_v = np.stack([wl_from_cache(c_ica, fi, 50, 0.80, 0.92) for fi in range(n_files)])
        s_std_v = np.stack([wl_from_cache(c_std, fi, 4,  0.60, 0.65) for fi in range(n_files)])
        s_pca_v = np.stack([wl_from_cache(c_pca, fi, 4,  0.70, 0.60) for fi in range(n_files)])
        uh_v = w_ica * s_ica_v + w_std * s_std_v + w_pca * s_pca_v
        blend = W_UH_FIX*uh_v + W_T_FIX*preds_T7 + W_MT_FIX*preds_mt_812 + W_SS_FIX*ss2
        auc = eval_loo(blend)
        if auc > best_wlt: best_wlt = auc; best_cfg_wlt = (w_ica, w_std, round(w_pca,3))
    print(f"  w_ica={w_ica} done", flush=True)

print(f"  WL fine-tune best: {best_wlt:.6f}  cfg={best_cfg_wlt}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_weight_fine'] = best_wlt
print(f"  {'*** NEW BEST ***' if best_wlt > CURRENT_BEST else ''}", flush=True)

# Store best WL weights
w_ica_best = best_cfg_wlt[0] if best_cfg_wlt else 0.70
w_std_best = best_cfg_wlt[1] if best_cfg_wlt else 0.18
w_pca_best = best_cfg_wlt[2] if best_cfg_wlt else 0.12

s_ica_b = np.stack([wl_from_cache(c_ica, fi, 50, 0.80, 0.92) for fi in range(n_files)])
s_std_b = np.stack([wl_from_cache(c_std, fi, 4,  0.60, 0.65) for fi in range(n_files)])
s_pca_b = np.stack([wl_from_cache(c_pca, fi, 4,  0.70, 0.60) for fi in range(n_files)])
uh_best = w_ica_best * s_ica_b + w_std_best * s_std_b + w_pca_best * s_pca_b

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Fine-tune ICA100 WL params with best WL weights
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Fine-tune ICA100 WL Params ===", flush=True)
t0 = time.time()
best_ica_p, best_cfg_ica_p = 0, None

K_LIST  = [40, 50, 60, 70, 80]
WMA_LIST = [0.88, 0.90, 0.92, 0.94, 0.95]
WMP_LIST = [0.70, 0.75, 0.80, 0.85, 0.90]

for k_neg in K_LIST:
    for wma in WMA_LIST:
        for wmp in WMP_LIST:
            s_ica_v = np.stack([wl_from_cache(c_ica, fi, k_neg, wmp, wma) for fi in range(n_files)])
            uh_v = w_ica_best*s_ica_v + w_std_best*s_std_b + w_pca_best*s_pca_b
            blend = W_UH_FIX*uh_v + W_T_FIX*preds_T7 + W_MT_FIX*preds_mt_812 + W_SS_FIX*ss2
            auc = eval_loo(blend)
            if auc > best_ica_p: best_ica_p = auc; best_cfg_ica_p = (k_neg, wma, wmp)
    print(f"  k_neg={k_neg} done", flush=True)

print(f"  ICA param fine-tune: {best_ica_p:.6f}  cfg={best_cfg_ica_p}  ({time.time()-t0:.0f}s)", flush=True)
results['ica_param_fine'] = best_ica_p
print(f"  {'*** NEW BEST ***' if best_ica_p > CURRENT_BEST else ''}", flush=True)

# Update ICA scores with best params
k_ica_best, wma_ica_best, wmp_ica_best = best_cfg_ica_p if best_cfg_ica_p else (50, 0.92, 0.80)
s_ica_b2 = np.stack([wl_from_cache(c_ica, fi, k_ica_best, wmp_ica_best, wma_ica_best) for fi in range(n_files)])
uh_best2 = w_ica_best * s_ica_b2 + w_std_best * s_std_b + w_pca_best * s_pca_b

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Fine-tune logit T and weights with new best UH
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Fine-tune Logit T + Weights ===", flush=True)
t0 = time.time()
best_lt, best_cfg_lt = 0, None

MT_COMBOS = {
    '7_10': (preds_T[7.0]+preds_T[10.0])/2,
    '8_10': (preds_T[8.0]+preds_T[10.0])/2,
    '8_12': preds_mt_812,
    '7_12': (preds_T[7.0]+preds_T[12.0])/2,
    '6_10': (preds_T[6.0]+preds_T[10.0])/2,
    '7_8_12': (preds_T[7.0]+preds_T[8.0]+preds_T[12.0])/3,
}

W_T2   = [0.25, 0.27, 0.28, 0.30, 0.32, 0.33, 0.35]
W_MT2  = [0.08, 0.10, 0.12, 0.14, 0.16]
W_SS2  = [0.04, 0.05, 0.06, 0.07, 0.08]

for T in [6.0, 7.0, 8.0, 9.0]:
    for mt_name, preds_mt_v in MT_COMBOS.items():
        for w_T in W_T2:
            for w_mt in W_MT2:
                for w_ss in W_SS2:
                    w_uh2 = 1.0 - w_T - w_mt - w_ss
                    if w_uh2 < 0.45 or w_uh2 > 0.72: continue
                    blend = w_uh2*uh_best2 + w_T*preds_T[T] + w_mt*preds_mt_v + w_ss*ss2
                    auc = eval_loo(blend)
                    if auc > best_lt: best_lt = auc; best_cfg_lt = (T, mt_name, w_T, w_mt, w_ss)
    print(f"  T={T} done", flush=True)

print(f"  Logit T+w fine-tune: {best_lt:.6f}  cfg={best_cfg_lt}  ({time.time()-t0:.0f}s)", flush=True)
results['logit_T_fine'] = best_lt
print(f"  {'*** NEW BEST ***' if best_lt > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 73 Summary ===", flush=True)
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
    print("未超越 0.9908，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
