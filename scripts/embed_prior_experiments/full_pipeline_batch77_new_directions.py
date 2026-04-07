"""
Batch 77: New Signal Directions

從 b75_softmax (LOO=0.99115) 出發，當前已確認平台：
- ICA k_neg=50 ✓ (最優)
- Multi-T T[8,10] ✓ (最優)
- WL global weights ✓ (w_ica=0.72 最優)
- Softmax T[3-4.5] / w[0.04-0.08] ✓ (平台)

探索全新方向：
1. NMF (非負矩陣分解) 作為第四 WL 組件
2. Attention-weighted prototype：用 similarity 加權平均正樣本，而非算術平均
3. Logit entropy signal：高確信度窗口（低 entropy）的 species scores
4. Log-space WL：用 log(clip(sim, 0.01, 1)) 代替線性 cosine sim
5. 組合：將 NMF WL 加入 UH triple → 4-way WL blend
"""
import numpy as np, json, os, time, pickle
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler
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

def wl_from_cache_attn(cache, fi, k_neg, wmp, wma):
    """Attention-weighted prototype: sim-weighted positive mean instead of arithmetic mean."""
    te, tr, tl, sims = cache[fi]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pos_idx = np.where(tl[:, si] > 0.5)[0]
        neg_idx = np.where(tl[:, si] < 0.1)[0]
        if len(pos_idx) == 0: ws[:, si] = 0.5; continue
        ps = sims[:, pos_idx]  # (n_te, n_pos)
        # Attention-weighted prototype: softmax over pos similarities → weighted avg
        attn = np.exp(ps - ps.max(1, keepdims=True))
        attn /= attn.sum(1, keepdims=True) + EPS
        pp_attn = (attn[:, :, None] * tr[pos_idx][None, :, :]).sum(1)  # (n_te, dim)
        pp_attn_n = pp_attn / (np.linalg.norm(pp_attn, axis=1, keepdims=True) + EPS)
        sp_attn = (te * pp_attn_n).sum(1)  # per-test-window attention similarity
        sp = wmp * ps.max(1) + (1-wmp) * sp_attn
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

def make_softmax_preds(T):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        lw = logit_win[file_start[fi]:file_end[fi]]
        sm = lw / T; sm -= sm.max(1, keepdims=True)
        exp_sm = np.exp(sm); prob_sm = exp_sm / (exp_sm.sum(1, keepdims=True) + EPS)
        out[fi] = prob_sm.max(0)
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

print("Computing subspace + preloading...", flush=True)
t0 = time.time()
ss2 = species_subspace_loo(ew_pca, 2, 0.92)
sm4 = make_softmax_preds(4.0)
print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

ref = W_UH_B*uh_b + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*sm4
print(f"Reference (current best): {eval_loo(ref):.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: NMF 作為第四 WL 組件
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: NMF Embedding as 4th WL Component ===", flush=True)
t0 = time.time()

# Fit NMF on all windows (shift to non-negative first)
emb_shifted = emb_win - emb_win.min(axis=0, keepdims=True)  # (739, 1536) non-negative
emb_shifted += 1e-6  # avoid zeros

print("  Fitting NMF...", flush=True)
best_nmf, best_nmf_cfg = 0, None

for n_comp in [40, 60, 80, 100]:
    t1 = time.time()
    nmf = NMF(n_components=n_comp, max_iter=300, random_state=42)
    ew_nmf_raw = nmf.fit_transform(emb_shifted).astype(np.float32)
    ew_nmf = ew_nmf_raw / (np.linalg.norm(ew_nmf_raw, axis=1, keepdims=True) + EPS)
    c_nmf = build_cache(ew_nmf)

    for k_neg in [4, 6]:
        for wma in [0.60, 0.70]:
            for wmp in [0.60, 0.70]:
                s_nmf_v = np.stack([wl_from_cache(c_nmf, fi, k_neg, wmp, wma) for fi in range(n_files)])
                # Add NMF as 4th WL component (w_nmf, shrink others)
                for w_nmf in [0.08, 0.12, 0.16]:
                    scale = 1.0 - w_nmf
                    uh_with_nmf = scale * uh_b + w_nmf * s_nmf_v
                    blend = W_UH_B*uh_with_nmf + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*sm4
                    auc = eval_loo(blend)
                    if auc > best_nmf: best_nmf = auc; best_nmf_cfg = (n_comp, k_neg, wma, wmp, w_nmf)

    print(f"  NMF n={n_comp}: best so far={best_nmf:.6f}  ({time.time()-t1:.0f}s)", flush=True)

print(f"  NMF WL best: {best_nmf:.6f}  cfg={best_nmf_cfg}  ({time.time()-t0:.0f}s)", flush=True)
results['nmf_wl'] = best_nmf
print(f"  {'*** NEW BEST ***' if best_nmf > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Attention-weighted WL prototype
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Attention-weighted WL Prototype ===", flush=True)
t0 = time.time()
best_attn, best_attn_cfg = 0, None

# Test on ICA (most important WL component)
for k_neg in [40, 50, 60]:
    for wmp in [0.80, 0.85, 0.90]:
        for wma in [0.85, 0.88, 0.92]:
            s_ica_attn = np.stack([wl_from_cache_attn(c_ica, fi, k_neg, wmp, wma) for fi in range(n_files)])
            uh_attn = W_ICA * s_ica_attn + W_STD * s_std_b + W_PCA * s_pca_b
            blend = W_UH_B*uh_attn + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*sm4
            auc = eval_loo(blend)
            if auc > best_attn: best_attn = auc; best_attn_cfg = (k_neg, wmp, wma)

print(f"  Attention WL best: {best_attn:.6f}  cfg={best_attn_cfg}  ({time.time()-t0:.0f}s)", flush=True)
results['attn_wl'] = best_attn
print(f"  {'*** NEW BEST ***' if best_attn > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Logit Entropy Signal（低 entropy = 高確信度）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Logit Entropy Signal ===", flush=True)
t0 = time.time()

def make_entropy_preds(T_sm=4.0):
    """低 entropy 窗口的 species 分數 → max over windows"""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        lw = logit_win[file_start[fi]:file_end[fi]]  # (n_win, n_sp)
        sm = lw / T_sm; sm -= sm.max(1, keepdims=True)
        exp_sm = np.exp(sm); prob_sm = exp_sm / (exp_sm.sum(1, keepdims=True) + EPS)
        # Entropy per window: H = -sum(p*log(p))
        H = -(prob_sm * np.log(prob_sm + EPS)).sum(1)  # (n_win,)
        # Weight each window by (1 - H/log(n_sp)): lower entropy = higher weight
        H_norm = H / (np.log(n_species) + EPS)  # normalize to [0,1]
        w_win = np.clip(1.0 - H_norm, 0.0, 1.0) + EPS
        w_win /= w_win.sum()
        # Weighted mean over windows (instead of max)
        out[fi] = (w_win[:, None] * prob_sm).sum(0)
    return out

best_ent, best_ent_cfg = 0, None
for T_sm in [2.0, 4.0, 6.0]:
    ent_preds = make_entropy_preds(T_sm)
    for w_ent in [0.04, 0.06, 0.08]:
        scale = (1.0 - w_ent)
        blend = scale*(W_UH_B*uh_b + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss2 + W_SM_B*sm4) + w_ent*ent_preds
        auc = eval_loo(blend)
        if auc > best_ent: best_ent = auc; best_ent_cfg = (T_sm, w_ent)
    print(f"  T_sm={T_sm}: done", flush=True)

print(f"  Entropy signal best: {best_ent:.6f}  cfg={best_ent_cfg}  ({time.time()-t0:.0f}s)", flush=True)
results['entropy_signal'] = best_ent
print(f"  {'*** NEW BEST ***' if best_ent > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Subspace n_comp 掃描 (目前固定為 2)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Subspace n_comp Scan ===", flush=True)
t0 = time.time()
best_ss_nc, best_ss_nc_cfg = 0, None

for n_comp in [1, 2, 3, 4]:
    for wma in [0.85, 0.88, 0.92, 0.95]:
        ss_v = species_subspace_loo(ew_pca, n_comp, wma)
        blend = W_UH_B*uh_b + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + W_SS_B*ss_v + W_SM_B*sm4
        auc = eval_loo(blend)
        if auc > best_ss_nc: best_ss_nc = auc; best_ss_nc_cfg = (n_comp, wma)
    print(f"  n_comp={n_comp}: done", flush=True)

print(f"  Subspace n_comp best: {best_ss_nc:.6f}  cfg={best_ss_nc_cfg}  ({time.time()-t0:.0f}s)", flush=True)
results['subspace_nc_scan'] = best_ss_nc
print(f"  {'*** NEW BEST ***' if best_ss_nc > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Subspace 在 ICA 空間（目前只在 PCA 空間）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Subspace in ICA Space ===", flush=True)
t0 = time.time()
best_ss_ica, best_ss_ica_cfg = 0, None

for n_comp in [2, 3]:
    for wma in [0.88, 0.92, 0.95]:
        ss_ica_v = species_subspace_loo(ew_ica, n_comp, wma)
        # Replace ss2 with ICA-space subspace
        for w_ss in [0.04, 0.05, 0.06, 0.07]:
            blend = W_UH_B*uh_b + W_T_B*preds_T8 + W_MT_B*preds_mt_810 + w_ss*ss_ica_v + W_SM_B*sm4
            auc = eval_loo(blend)
            if auc > best_ss_ica: best_ss_ica = auc; best_ss_ica_cfg = (n_comp, wma, w_ss)

print(f"  Subspace ICA best: {best_ss_ica:.6f}  cfg={best_ss_ica_cfg}  ({time.time()-t0:.0f}s)", flush=True)
results['subspace_ica'] = best_ss_ica
print(f"  {'*** NEW BEST ***' if best_ss_ica > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 77 Summary ===", flush=True)
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
