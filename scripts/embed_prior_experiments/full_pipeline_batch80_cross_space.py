"""
Batch 80: Cross-space Signals & Second Embedding Type

從 nmf_w_ultra (LOO=0.99136) 出發：
- uh_nmf = 0.84×UH + 0.16×NMF(n=100, k=6, wma=0.65, wmp=0.60)
- final = 0.48×uh_nmf + 0.26×T8 + 0.13×mt[8,10] + 0.06×ss + 0.07×sm4

Batch 79 確認局部平台：
- NMF k=6, wma=0.65, wmp=0.60 是最優
- w_nmf=0.16 最優（與新 sub-params 搭配）
- 多個 NMF seeds 無提升
- 4-way WL 獨立 weight 無提升

現在探索全新信號源：
1. Cross-space WL：用 ICA 空間找鄰居，但用 PCA 空間計算相似度（跨空間互補）
2. Logit percentile signal：不用 max，而是用 P95/P90 over windows
3. 多層次 subspace：同時用 n_comp=2 和 n_comp=3 的 subspace 平均
4. 對數 logit prior：log(sig(logit)) 的 max → 不同於線性 sigmoid
5. 全局 ICA weight 聯合 NMF 二維掃描
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
# Method 1: Logit Percentile Signal（P95/P90 over windows）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Logit Percentile Signal ===", flush=True)
t0 = time.time()

def make_percentile_preds(T, pct):
    """sigmoid(logit/T) 的 pct 百分位數（代替 max）"""
    sig = (1.0/(1.0+np.exp(np.clip(-logit_win/T,-88,88)))).astype(np.float32)
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        s_win = sig[file_start[fi]:file_end[fi]]  # (n_win, n_sp)
        out[fi] = np.percentile(s_win, pct, axis=0)
    return out

best_pct, best_pct_cfg = CURRENT_BEST, None
for T in [6.0, 8.0, 10.0]:
    for pct in [80, 85, 90, 95, 97]:
        pp = make_percentile_preds(T, pct)
        # Replace T8 max with percentile
        for w_pct in [0.22, 0.26, 0.30]:
            blend = W_UH*uh_nmf_b + w_pct*pp + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
            auc = eval_loo(blend)
            if auc > best_pct: best_pct = auc; best_pct_cfg = (T, pct, w_pct)
    print(f"  T={T} done", flush=True)

results['logit_pct'] = best_pct
print(f"  Percentile best: {best_pct:.6f}  cfg={best_pct_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_pct > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: 多層次 Subspace 平均（n_comp=2 + n_comp=3）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Multi-level Subspace ===", flush=True)
t0 = time.time()
ss3 = species_subspace_loo(ew_pca, 3, 0.92)
print("  ss3 computed", flush=True)

best_ss_ml, best_ss_ml_cfg = CURRENT_BEST, None
for w23 in [0.3, 0.4, 0.5, 0.6, 0.7]:  # ss2/ss3 mixing ratio
    ss_blend = w23*ss2 + (1-w23)*ss3
    for w_ss in [0.04, 0.05, 0.06, 0.07, 0.08]:
        blend = W_UH*uh_nmf_b + W_T*preds_T8 + W_MT*preds_mt_810 + w_ss*ss_blend + W_SM*sm4
        auc = eval_loo(blend)
        if auc > best_ss_ml: best_ss_ml = auc; best_ss_ml_cfg = (w23, w_ss)

results['multi_subspace'] = best_ss_ml
print(f"  Multi-subspace best: {best_ss_ml:.6f}  cfg={best_ss_ml_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_ss_ml > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Log-sigmoid logit（log(sig(logit/T)).max）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Log-Sigmoid Logit ===", flush=True)
t0 = time.time()

def make_logsig_preds(T):
    """log(sigmoid(logit/T)).max over windows → then normalize"""
    sig = (1.0/(1.0+np.exp(np.clip(-logit_win/T,-88,88)))).astype(np.float32)
    log_sig = np.log(sig + EPS)  # (739, n_sp)
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        ls = log_sig[file_start[fi]:file_end[fi]]
        raw_max = ls.max(0)
        # Normalize to [0,1] via min-max scaling over species
        mn = raw_max.min(); mx = raw_max.max()
        if mx > mn: out[fi] = (raw_max - mn) / (mx - mn)
        else: out[fi] = 0.5
    return out

best_ls, best_ls_cfg = CURRENT_BEST, None
for T in [4.0, 6.0, 8.0, 12.0]:
    ls_p = make_logsig_preds(T)
    for w_ls in [0.04, 0.06, 0.08]:
        scale = 1.0 - w_ls
        blend = scale*(W_UH*uh_nmf_b + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4) + w_ls*ls_p
        auc = eval_loo(blend)
        if auc > best_ls: best_ls = auc; best_ls_cfg = (T, w_ls)
    print(f"  T={T} done", flush=True)

results['log_sigmoid'] = best_ls
print(f"  Log-sigmoid best: {best_ls:.6f}  cfg={best_ls_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_ls > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: ICA weight × NMF weight 二維掃描
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: ICA×NMF Joint 2D Scan ===", flush=True)
t0 = time.time()
best_2d, best_2d_cfg = CURRENT_BEST, None

for w_ica in [0.68, 0.70, 0.72, 0.74, 0.76]:
    rem = 1.0 - w_ica
    w_std_v = round(rem * (0.18/0.28), 4)
    w_pca_v = round(rem - w_std_v, 4)
    uh_v = w_ica*s_ica_b + w_std_v*s_std_b + w_pca_v*s_pca_b
    for w_nmf_v in [0.12, 0.14, 0.16, 0.18, 0.20]:
        uh_nmf_v = (1-w_nmf_v)*uh_v + w_nmf_v*s_nmf_b
        blend = W_UH*uh_nmf_v + W_T*preds_T8 + W_MT*preds_mt_810 + W_SS*ss2 + W_SM*sm4
        auc = eval_loo(blend)
        if auc > best_2d: best_2d = auc; best_2d_cfg = (w_ica, w_nmf_v)
    print(f"  w_ica={w_ica} done", flush=True)

results['ica_nmf_joint'] = best_2d
print(f"  ICA×NMF joint best: {best_2d:.6f}  cfg={best_2d_cfg}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_2d > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 80 Summary ===", flush=True)
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
