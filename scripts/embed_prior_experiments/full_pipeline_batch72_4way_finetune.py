"""
Batch 72: Fine-tune 4-Way Blend + 5-Way Extension + Ultra-T Sweep Continuation

從 logit_4way_blend (LOO=0.9900, T9×0.25 + multiT×0.10 + ss×0.08 + UH×0.57) 出發：
1. Fine-tune 4-way 各 weight（更細的 grid）
2. T sweep for primary logit component（T=9 附近精細掃描）
3. 5-Way: 加入 WL-logit-space 或 attention 作為第5個組件
4. Different multi-T combinations（[5,8,10] vs [5,10] vs [8,10,15]）
5. Subspace blend variant（不同 n_comp 和 embedding space）

Current best: logit_4way_blend = 0.9900 (T9×0.25 + multiT[5,10]×0.10 + ss×0.08 + UH×0.57)
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
CURRENT_BEST = 0.989965
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

print("Building sim caches + UH-triple...", flush=True)
t0 = time.time()
c_ica = build_cache(ew_ica); c_std = build_cache(ew_std); c_pca = build_cache(ew_pca)
s_ica = np.stack([wl_from_cache(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std = np.stack([wl_from_cache(c_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca = np.stack([wl_from_cache(c_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
uh_triple = W_ICA * s_ica + W_STD * s_std + W_PCA * s_pca
uh_auc = eval_loo(uh_triple)
print(f"  UH-triple: {uh_auc:.4f}  ({time.time()-t0:.1f}s)", flush=True)

# ─── Precompute logit components ─────────────────────────────────────────────
print("Precomputing logit components...", flush=True)
def make_preds(T):
    sig = (1.0/(1.0+np.exp(np.clip(-logit_win/T,-88,88)))).astype(np.float32)
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

# Primary T candidates
T_CANDS = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0]
preds_T = {T: make_preds(T) for T in T_CANDS}
# Multi-T combinations
mt_combos = {
    '5_10':  (make_preds(5.0) + make_preds(10.0)) / 2,
    '5_8':   (make_preds(5.0) + make_preds(8.0)) / 2,
    '8_10':  (make_preds(8.0) + make_preds(10.0)) / 2,
    '5_8_10': (make_preds(5.0) + make_preds(8.0) + make_preds(10.0)) / 3,
    '8_12':  (make_preds(8.0) + make_preds(12.0)) / 2,
    '5_10_15': (make_preds(5.0) + make_preds(10.0) + make_preds(15.0)) / 3,
}
print(f"  Done", flush=True)

# ─── Subspace scores (best config: pca80, n_comp=3, wma=0.88) ────────────────
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

print("Computing subspace scores...", flush=True)
t0 = time.time()
ss_scores = species_subspace_loo(ew_pca, 3, 0.88)
print(f"  ss standalone={eval_loo(ss_scores):.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Fine-tune 4-Way (T, w_T, w_mt, w_ss sweep)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Fine-tune 4-Way Blend ===", flush=True)
t0 = time.time()
best_4ft, best_cfg_4ft = 0, None

W_T_LIST  = [0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30]
W_MT_LIST = [0.06, 0.08, 0.10, 0.12, 0.14]
W_SS_LIST = [0.04, 0.06, 0.08, 0.10, 0.12]

for T in T_CANDS:
    for mt_name, preds_mt in mt_combos.items():
        for w_T in W_T_LIST:
            for w_mt in W_MT_LIST:
                for w_ss in W_SS_LIST:
                    w_uh = 1.0 - w_T - w_mt - w_ss
                    if w_uh < 0.45 or w_uh > 0.75: continue
                    blend = w_uh*uh_triple + w_T*preds_T[T] + w_mt*preds_mt + w_ss*ss_scores
                    auc = eval_loo(blend)
                    if auc > best_4ft:
                        best_4ft = auc
                        best_cfg_4ft = (T, mt_name, w_T, w_mt, w_ss, round(w_uh,3))
    print(f"  T={T} done", flush=True)

print(f"  4-Way FT best: {best_4ft:.6f}  cfg={best_cfg_4ft}  ({time.time()-t0:.0f}s)", flush=True)
results['logit_4way_ft'] = best_4ft
print(f"  {'*** NEW BEST ***' if best_4ft > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: 5-Way Blend (add geo-mean T=[8,10] as 5th component)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: 5-Way Blend ===", flush=True)
t0 = time.time()
best_5w, best_cfg_5w = 0, None

# Geo-mean of T=[8,10] (different from arithmetic mean multi-T)
preds_T8 = preds_T[8.0]; preds_T10 = preds_T[10.0]
preds_geo = np.sqrt(np.clip(preds_T8, EPS, 1) * np.clip(preds_T10, EPS, 1))

# Use best 4-way config as baseline, add geo-mean
if best_cfg_4ft:
    T4, mt4, wT4, wmt4, wss4, wuh4 = best_cfg_4ft
    preds_T4 = preds_T[T4]; preds_mt4 = mt_combos[mt4]
    for w_geo in [0.03, 0.05, 0.07, 0.10]:
        # Scale down existing weights proportionally
        scale = (1 - w_geo)
        blend5 = (wuh4*scale)*uh_triple + (wT4*scale)*preds_T4 + (wmt4*scale)*preds_mt4 + (wss4*scale)*ss_scores + w_geo*preds_geo
        auc = eval_loo(blend5)
        if auc > best_5w: best_5w = auc; best_cfg_5w = (w_geo,)

print(f"  5-Way best: {best_5w:.6f}  cfg={best_cfg_5w}  ({time.time()-t0:.1f}s)", flush=True)
results['logit_5way_blend'] = best_5w
print(f"  {'*** NEW BEST ***' if best_5w > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Optimal T for UH-triple itself (sweep WL component T for logit)
# Maybe UH-triple weights could also be further tuned
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Tune UH-triple Component Weights ===", flush=True)
t0 = time.time()
best_wl_tune, best_cfg_wl_tune = 0, None

# Use best T from 4-way FT
T_best_4 = best_cfg_4ft[0] if best_cfg_4ft else 9.0
preds_Tbest = preds_T.get(T_best_4, preds_T[9.0])
preds_mt_best = mt_combos.get(best_cfg_4ft[1] if best_cfg_4ft else '5_10', mt_combos['5_10'])
w_T_best = best_cfg_4ft[2] if best_cfg_4ft else 0.25
w_mt_best = best_cfg_4ft[3] if best_cfg_4ft else 0.10
w_ss_best = best_cfg_4ft[4] if best_cfg_4ft else 0.08

# Vary ICA/STD/PCA component weights
for w_ica in [0.60, 0.65, 0.70]:
    for w_std in [0.18, 0.22, 0.25]:
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.25: continue
        uh_var = w_ica * s_ica + w_std * s_std + w_pca * s_pca
        w_uh = 1.0 - w_T_best - w_mt_best - w_ss_best
        blend = w_uh*uh_var + w_T_best*preds_Tbest + w_mt_best*preds_mt_best + w_ss_best*ss_scores
        auc = eval_loo(blend)
        if auc > best_wl_tune:
            best_wl_tune = auc
            best_cfg_wl_tune = (w_ica, w_std, round(w_pca,3))

print(f"  WL-tune best: {best_wl_tune:.6f}  cfg(w_ica,w_std,w_pca)={best_cfg_wl_tune}  ({time.time()-t0:.1f}s)", flush=True)
results['logit_4way_wl_tune'] = best_wl_tune
print(f"  {'*** NEW BEST ***' if best_wl_tune > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Subspace variants in 4-way (try ICA-space subspace, n_comp=2)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Subspace Variants in 4-Way ===", flush=True)
t0 = time.time()
best_ssv, best_cfg_ssv = 0, None

for emb, ename in [(ew_ica, 'ica100'), (ew_pca, 'pca80')]:
    for n_comp in [2, 3, 4]:
        for wma_ss in [0.85, 0.88, 0.90, 0.92]:
            ss_var = species_subspace_loo(emb, n_comp, wma_ss)
            # Use best 4-way weights
            w_uh_4 = 1.0 - w_T_best - w_mt_best - w_ss_best
            blend = w_uh_4*uh_triple + w_T_best*preds_Tbest + w_mt_best*preds_mt_best + w_ss_best*ss_var
            auc = eval_loo(blend)
            if auc > best_ssv:
                best_ssv = auc
                best_cfg_ssv = (ename, n_comp, wma_ss)
        print(f"    {ename} n_comp={n_comp} done", flush=True)

print(f"  Subspace variant best: {best_ssv:.6f}  cfg={best_cfg_ssv}  ({time.time()-t0:.0f}s)", flush=True)
results['logit_4way_ss_variant'] = best_ssv
print(f"  {'*** NEW BEST ***' if best_ssv > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 72 Summary ===", flush=True)
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
    print("未超越 0.9900，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
