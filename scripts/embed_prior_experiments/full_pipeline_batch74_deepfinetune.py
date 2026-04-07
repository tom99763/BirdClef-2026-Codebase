"""
Batch 74: Deep Fine-tune All WL Sub-component Params

從 logit_T_fine (LOO=0.9910) 出發：
- w_ica=0.72, w_std=0.18, w_pca=0.10
- ICA: k=50, wma=0.88, wmp=0.85
- T=8.0, w_T=0.28, mt=[8,10], w_mt=0.14, ss×0.06

Batch 73 做了 ICA 參數調優（wma=0.88, wmp=0.85 優於 0.92/0.80）
現在進一步：
1. STD-PCA80 和 PCA80 子組件參數精細調優
2. 更細 WL 三元件組合（不同 k_neg/wma/wmp 組合）
3. 全局最優組合再 fine-tune 一輪
4. 嘗試 k_neg 在 40-80 更細密掃描 for ICA

Current best: logit_T_fine = 0.9910
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
CURRENT_BEST = 0.991025
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

# Current best WL components
W_ICA_CURR, W_STD_CURR, W_PCA_CURR = 0.72, 0.18, 0.10
ICA_K_CURR, ICA_WMA_CURR, ICA_WMP_CURR = 50, 0.88, 0.85
STD_K_CURR, STD_WMA_CURR, STD_WMP_CURR = 4, 0.65, 0.60
PCA_K_CURR, PCA_WMA_CURR, PCA_WMP_CURR = 4, 0.60, 0.70

def make_uh(ica_k, ica_wma, ica_wmp, std_k, std_wma, std_wmp, pca_k, pca_wma, pca_wmp, w_ica, w_std, w_pca):
    s_i = np.stack([wl_from_cache(c_ica, fi, ica_k, ica_wmp, ica_wma) for fi in range(n_files)])
    s_s = np.stack([wl_from_cache(c_std, fi, std_k, std_wmp, std_wma) for fi in range(n_files)])
    s_p = np.stack([wl_from_cache(c_pca, fi, pca_k, pca_wmp, pca_wma) for fi in range(n_files)])
    return w_ica*s_i + w_std*s_s + w_pca*s_p

# Precompute logit components
def make_preds(T):
    sig = (1.0/(1.0+np.exp(np.clip(-logit_win/T,-88,88)))).astype(np.float32)
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

preds_T8  = make_preds(8.0)
preds_T10 = make_preds(10.0)
preds_mt_810 = (preds_T8 + preds_T10) / 2.0

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

# Current best logit config
W_T_CURR, W_MT_CURR, W_SS_CURR = 0.28, 0.14, 0.06
W_UH_CURR = 1.0 - W_T_CURR - W_MT_CURR - W_SS_CURR  # = 0.52

# Reference: current best UH
uh_curr = make_uh(ICA_K_CURR, ICA_WMA_CURR, ICA_WMP_CURR,
                  STD_K_CURR, STD_WMA_CURR, STD_WMP_CURR,
                  PCA_K_CURR, PCA_WMA_CURR, PCA_WMP_CURR,
                  W_ICA_CURR, W_STD_CURR, W_PCA_CURR)
ref_blend = W_UH_CURR*uh_curr + W_T_CURR*preds_T8 + W_MT_CURR*preds_mt_810 + W_SS_CURR*ss2
print(f"Reference (current best): {eval_loo(ref_blend):.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Fine-tune STD-PCA80 params
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Fine-tune STD-PCA80 Params ===", flush=True)
t0 = time.time()
best_std_p, best_cfg_std_p = 0, None

# Current: k=4, wma=0.65, wmp=0.60
STD_K_LIST  = [2, 3, 4, 6, 8]
STD_WMA_LIST = [0.55, 0.60, 0.65, 0.70, 0.75]
STD_WMP_LIST = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

s_ica_c = np.stack([wl_from_cache(c_ica, fi, ICA_K_CURR, ICA_WMP_CURR, ICA_WMA_CURR) for fi in range(n_files)])
s_pca_c = np.stack([wl_from_cache(c_pca, fi, PCA_K_CURR, PCA_WMP_CURR, PCA_WMA_CURR) for fi in range(n_files)])

for k_neg in STD_K_LIST:
    for wma in STD_WMA_LIST:
        for wmp in STD_WMP_LIST:
            s_std_v = np.stack([wl_from_cache(c_std, fi, k_neg, wmp, wma) for fi in range(n_files)])
            uh_v = W_ICA_CURR*s_ica_c + W_STD_CURR*s_std_v + W_PCA_CURR*s_pca_c
            blend = W_UH_CURR*uh_v + W_T_CURR*preds_T8 + W_MT_CURR*preds_mt_810 + W_SS_CURR*ss2
            auc = eval_loo(blend)
            if auc > best_std_p: best_std_p = auc; best_cfg_std_p = (k_neg, wma, wmp)
    print(f"  k_std={k_neg} done", flush=True)

print(f"  STD-PCA param best: {best_std_p:.6f}  cfg={best_cfg_std_p}  ({time.time()-t0:.0f}s)", flush=True)
results['std_param_fine'] = best_std_p
print(f"  {'*** NEW BEST ***' if best_std_p > CURRENT_BEST else ''}", flush=True)

# Update STD scores
std_k_b, std_wma_b, std_wmp_b = best_cfg_std_p if best_cfg_std_p else (STD_K_CURR, STD_WMA_CURR, STD_WMP_CURR)
s_std_b = np.stack([wl_from_cache(c_std, fi, std_k_b, std_wmp_b, std_wma_b) for fi in range(n_files)])

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Fine-tune PCA80 params
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Fine-tune PCA80 Params ===", flush=True)
t0 = time.time()
best_pca_p, best_cfg_pca_p = 0, None

PCA_K_LIST  = [2, 3, 4, 6, 8]
PCA_WMA_LIST = [0.55, 0.60, 0.65, 0.70]
PCA_WMP_LIST = [0.60, 0.65, 0.70, 0.75, 0.80]

uh_with_new_std = W_ICA_CURR*s_ica_c + W_STD_CURR*s_std_b + W_PCA_CURR  # placeholder

for k_neg in PCA_K_LIST:
    for wma in PCA_WMA_LIST:
        for wmp in PCA_WMP_LIST:
            s_pca_v = np.stack([wl_from_cache(c_pca, fi, k_neg, wmp, wma) for fi in range(n_files)])
            uh_v = W_ICA_CURR*s_ica_c + W_STD_CURR*s_std_b + W_PCA_CURR*s_pca_v
            blend = W_UH_CURR*uh_v + W_T_CURR*preds_T8 + W_MT_CURR*preds_mt_810 + W_SS_CURR*ss2
            auc = eval_loo(blend)
            if auc > best_pca_p: best_pca_p = auc; best_cfg_pca_p = (k_neg, wma, wmp)
    print(f"  k_pca={k_neg} done", flush=True)

print(f"  PCA param best: {best_pca_p:.6f}  cfg={best_cfg_pca_p}  ({time.time()-t0:.0f}s)", flush=True)
results['pca_param_fine'] = best_pca_p
print(f"  {'*** NEW BEST ***' if best_pca_p > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Full combined fine-tune with all best sub-params
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Full Combined Fine-tune ===", flush=True)
t0 = time.time()

pca_k_b = best_cfg_pca_p[0] if best_cfg_pca_p else PCA_K_CURR
pca_wma_b = best_cfg_pca_p[1] if best_cfg_pca_p else PCA_WMA_CURR
pca_wmp_b = best_cfg_pca_p[2] if best_cfg_pca_p else PCA_WMP_CURR
s_pca_b = np.stack([wl_from_cache(c_pca, fi, pca_k_b, pca_wmp_b, pca_wma_b) for fi in range(n_files)])

# Build full best UH
uh_full = W_ICA_CURR*s_ica_c + W_STD_CURR*s_std_b + W_PCA_CURR*s_pca_b

# Final fine-tune: sweep all blend weights with full best UH
best_comb, best_cfg_comb = 0, None
for w_T in [0.25, 0.27, 0.28, 0.29, 0.30, 0.32]:
    for w_mt in [0.10, 0.12, 0.14, 0.15, 0.16]:
        for w_ss in [0.04, 0.05, 0.06, 0.07, 0.08]:
            w_uh = 1.0 - w_T - w_mt - w_ss
            if w_uh < 0.45 or w_uh > 0.68: continue
            blend = w_uh*uh_full + w_T*preds_T8 + w_mt*preds_mt_810 + w_ss*ss2
            auc = eval_loo(blend)
            if auc > best_comb: best_comb = auc; best_cfg_comb = (w_T, w_mt, w_ss, round(w_uh,3))

results['full_combined_fine'] = best_comb
print(f"  Full combined: {best_comb:.6f}  cfg={best_cfg_comb}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_comb > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 74 Summary ===", flush=True)
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
    print("未超越 0.9910，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
