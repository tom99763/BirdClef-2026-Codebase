"""
Batch 79: NMF Ultra Fine-tune & 2nd NMF Component

從 nmf_fine (LOO=0.99133) 出發：
- uh_nmf = 0.82×UH + 0.18×NMF(n=100, k=6, wma=0.65, wmp=0.60)
- final = 0.48×uh_nmf + 0.26×T8 + 0.13×mt[8,10] + 0.06×ss + 0.07×sm4

Batch 78 確認：
- NMF n=100 已是最優（120/150/200 無提升）
- w_nmf=0.18 最優（單調遞增到 0.18，之後下降）
- wma=0.65, wmp=0.60 最優

現在探索：
1. NMF 更細密 k_neg 掃描（當前 k=6，範圍 3~15）
2. 第二個 NMF 組件（不同 random_state → 不同局部最優）
3. NMF + UH 三元件獨立 weight 掃描（不再共用 w_nmf/scale）
4. w_nmf 更細密：0.16~0.20 之間的細掃
5. ICA WL weight 與 NMF 聯合調優
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
CURRENT_BEST = 0.9913290259975499
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

with open("outputs/embed_prior_model.pkl", 'rb') as f:
    ep = pickle.load(f)
ew_ica = ep['emb_win_ica_norm'].astype(np.float32)
ew_pca = ep['emb_win_pca_norm'].astype(np.float32)
ew_std = ep['emb_win_std_norm'].astype(np.float32)
ew_nmf = ep['emb_win_nmf_norm'].astype(np.float32)
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

print("Building caches...", flush=True)
t0 = time.time()
c_ica = build_cache(ew_ica); c_std = build_cache(ew_std)
c_pca = build_cache(ew_pca); c_nmf = build_cache(ew_nmf)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

W_ICA, W_STD, W_PCA = 0.72, 0.18, 0.10
ICA_K, ICA_WMA, ICA_WMP = 50, 0.88, 0.85
STD_K, STD_WMA, STD_WMP = 3, 0.70, 0.50
PCA_K, PCA_WMA, PCA_WMP = 4, 0.60, 0.70
NMF_K, NMF_WMA, NMF_WMP, W_NMF = 6, 0.65, 0.60, 0.18
W_T, W_MT, W_SS, W_SM = 0.26, 0.13, 0.06, 0.07
W_UH_FINAL = 1.0 - W_T - W_MT - W_SS - W_SM  # 0.48

s_ica_b = np.stack([wl_from_cache(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std_b = np.stack([wl_from_cache(c_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca_b = np.stack([wl_from_cache(c_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
uh_b    = W_ICA*s_ica_b + W_STD*s_std_b + W_PCA*s_pca_b
s_nmf_b = np.stack([wl_from_cache(c_nmf, fi, NMF_K, NMF_WMP, NMF_WMA) for fi in range(n_files)])
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

ref = W_UH_FINAL*uh_nmf_b + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
print(f"Reference (current best): {eval_loo(ref):.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: NMF k_neg 更細密掃描
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: NMF k_neg Ultra Fine Scan ===", flush=True)
t0 = time.time()
best_k, best_k_cfg = CURRENT_BEST, None

for k_neg in [3, 4, 5, 6, 7, 8, 10, 12, 15]:
    s_v = np.stack([wl_from_cache(c_nmf, fi, k_neg, NMF_WMP, NMF_WMA) for fi in range(n_files)])
    uh_v = (1-W_NMF)*uh_b + W_NMF*s_v
    blend = W_UH_FINAL*uh_v + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
    auc = eval_loo(blend)
    if auc > best_k: best_k = auc; best_k_cfg = k_neg
    print(f"  k={k_neg}: {auc:.6f}", flush=True)

results['nmf_k_ultra'] = best_k
print(f"  Best: {best_k:.6f}  k={best_k_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_k > CURRENT_BEST else ''}", flush=True)
k_best = best_k_cfg if best_k_cfg else NMF_K
s_nmf_k = np.stack([wl_from_cache(c_nmf, fi, k_best, NMF_WMP, NMF_WMA) for fi in range(n_files)])

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: 第二個 NMF（不同 random_state）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: 2nd NMF Component (diff random_state) ===", flush=True)
t0 = time.time()
best_nmf2, best_nmf2_cfg = CURRENT_BEST, None

for seed in [0, 1, 7, 13, 99]:
    t1 = time.time()
    nmf_v = NMF(n_components=100, max_iter=300, random_state=seed)
    ew_v_raw = nmf_v.fit_transform(emb_shifted).astype(np.float32)
    ew_v = ew_v_raw / (np.linalg.norm(ew_v_raw, axis=1, keepdims=True) + EPS)
    c_v = build_cache(ew_v)
    s_v = np.stack([wl_from_cache(c_v, fi, k_best, NMF_WMP, NMF_WMA) for fi in range(n_files)])
    # Ensemble of 2 NMFs
    for w_nmf2 in [0.08, 0.12, 0.16]:
        s_ens = (1 - w_nmf2/(W_NMF+w_nmf2)) * s_nmf_k + (w_nmf2/(W_NMF+w_nmf2)) * s_v
        uh_v2 = (1-W_NMF-w_nmf2)*uh_b + (W_NMF+w_nmf2)*s_ens
        # 只有在 weight 合法時才用
        if W_NMF + w_nmf2 > 0.4: continue
        uh_v2 = (1-W_NMF)*uh_b + W_NMF*s_ens
        blend = W_UH_FINAL*uh_v2 + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
        auc = eval_loo(blend)
        if auc > best_nmf2: best_nmf2 = auc; best_nmf2_cfg = (seed, w_nmf2)
    print(f"  seed={seed}: best={best_nmf2:.6f}  ({time.time()-t1:.0f}s)", flush=True)

results['nmf2_ensemble'] = best_nmf2
print(f"  2nd NMF best: {best_nmf2:.6f}  cfg={best_nmf2_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_nmf2 > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: 4-way WL 獨立 weight（ICA/STD/PCA/NMF 各自獨立，不再 UH-then-NMF）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: 4-Way WL Independent Weights ===", flush=True)
t0 = time.time()
best_4w, best_4w_cfg = CURRENT_BEST, None

for w_ica4 in [0.32, 0.35, 0.38, 0.40, 0.42]:
    for w_std4 in [0.06, 0.08, 0.10]:
        for w_pca4 in [0.04, 0.05, 0.06]:
            for w_nmf4 in [0.06, 0.08, 0.10, 0.12]:
                total_wl = w_ica4 + w_std4 + w_pca4 + w_nmf4
                if total_wl > 0.62 or total_wl < 0.46: continue
                wl_4way = w_ica4*s_ica_b + w_std4*s_std_b + w_pca4*s_pca_b + w_nmf4*s_nmf_k
                # Scale to same final UH weight
                wl_4way_n = wl_4way  # 已是各組件的加權和
                blend = W_UH_FINAL*wl_4way_n + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
                auc = eval_loo(blend)
                if auc > best_4w: best_4w = auc; best_4w_cfg = (w_ica4, w_std4, w_pca4, w_nmf4)

results['wl_4way_indep'] = best_4w
print(f"  4-way WL indep best: {best_4w:.6f}  cfg={best_4w_cfg}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_4w > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: w_nmf 更細密：0.15~0.22
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: w_nmf Ultra Fine [0.15~0.22] ===", flush=True)
t0 = time.time()
best_wmf2, best_wmf2_val = CURRENT_BEST, 0.18

for w_nmf in [0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22]:
    uh_v = (1-w_nmf)*uh_b + w_nmf*s_nmf_k
    blend = W_UH_FINAL*uh_v + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
    auc = eval_loo(blend)
    if auc > best_wmf2: best_wmf2 = auc; best_wmf2_val = w_nmf
    print(f"  w_nmf={w_nmf}: {auc:.6f}", flush=True)

results['nmf_w_ultra'] = best_wmf2
print(f"  w_nmf ultra best: {best_wmf2:.6f}  w={best_wmf2_val}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_wmf2 > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 79 Summary ===", flush=True)
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
    print("未超越 0.99133，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
