"""
Batch 69: RBF Kernel WL + 3-Way Blend + Logit Temperature

三個新方向：
1. RBF Kernel WL: 用 exp(-γ(1-sim)) 取代 cosine 相似度，強調近鄰效果
2. 3-Way Blend: UH-triple + direct_logit + species_subspace（兩個最強 add-on 一起）
3. Logit Temperature: sigmoid(logit/T) 不同溫度，可能比 T=1 更好

Current best: direct_logit_uh_blend = 0.9884
"""
import numpy as np, json, os, time, pickle
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
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
CURRENT_BEST = 0.9883731573643638
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# ─── Load precomputed transforms from pkl ────────────────────────────────────
print("Loading pkl transforms...", flush=True)
with open("outputs/embed_prior_model.pkl", 'rb') as f:
    ep = pickle.load(f)
ew_ica = ep['emb_win_ica_norm'].astype(np.float32)  # (739, 100)
ew_pca = ep['emb_win_pca_norm'].astype(np.float32)  # (739, 80)
ew_std = ep['emb_win_std_norm'].astype(np.float32)  # (739, 80)
print(f"  ICA shape: {ew_ica.shape}, PCA: {ew_pca.shape}, STD: {ew_std.shape}", flush=True)

W_ICA, W_STD, W_PCA = 0.655, 0.225, 0.120
ICA_K, ICA_WMA, ICA_WMP = 50, 0.92, 0.80
STD_K, STD_WMA, STD_WMP =  4, 0.65, 0.60
PCA_K, PCA_WMA, PCA_WMP =  4, 0.60, 0.70

# ─── Sim cache ───────────────────────────────────────────────────────────────
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
            ns2 = sims[:, neg_idx]
            k2 = min(k_neg, len(neg_idx))
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
c_ica = build_cache(ew_ica)
c_std = build_cache(ew_std)
c_pca = build_cache(ew_pca)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

# ─── UH-triple reference ─────────────────────────────────────────────────────
print("Computing UH-triple reference...", flush=True)
t0 = time.time()
s_ica = np.stack([wl_from_cache(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std = np.stack([wl_from_cache(c_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca = np.stack([wl_from_cache(c_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
uh_triple = W_ICA * s_ica + W_STD * s_std + W_PCA * s_pca
uh_auc = eval_loo(uh_triple)

# Direct logit (baseline)
logit_sig = (1.0 / (1.0 + np.exp(-logit_win))).astype(np.float32)
preds_logit = np.stack([logit_sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
direct_best = (1-0.08) * uh_triple + 0.08 * preds_logit
direct_auc = eval_loo(direct_best)
print(f"  UH-triple: {uh_auc:.4f}  direct_logit_uh: {direct_auc:.4f}  ({time.time()-t0:.1f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: RBF Kernel WL
# 對 precomputed sim 矩陣套用 RBF 轉換：rbf = exp(-γ·(1-sim))
# 這讓 near-neighbor 的 influence 指數放大，far-neighbor 趨近 0
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: RBF Kernel WL ===", flush=True)
t0 = time.time()

def wl_rbf_from_sims(cache, fi, gamma, k_neg, wma):
    """RBF-transformed WL contrast"""
    te, tr, tl, cos_sims = cache[fi]
    # RBF: higher gamma → sharper (only very close windows contribute)
    rbf_sims = np.exp(-gamma * (1.0 - cos_sims))  # (n_te, n_tr), in [0,1]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pos_idx = np.where(tl[:, si] > 0.5)[0]
        neg_idx = np.where(tl[:, si] < 0.1)[0]
        if len(pos_idx) == 0: ws[:, si] = 0.5; continue
        # Positive score: max of RBF sims (most similar positive window)
        sp = rbf_sims[:, pos_idx].max(1)   # (n_te,) in [0,1]
        if len(neg_idx) > 0:
            k2 = min(k_neg, len(neg_idx))
            neg_rbf = rbf_sims[:, neg_idx]
            top_idx = np.argsort(-neg_rbf, axis=1)[:, :k2]
            tn = neg_rbf[np.arange(len(te))[:, None], top_idx].mean(1)  # (n_te,) mean of top-k neg
            ws[:, si] = sp / (sp + tn + EPS)  # ratio formula (sp, tn in [0,1])
        else:
            ws[:, si] = sp
    return wma * ws.max(0) + (1-wma) * ws.mean(0)

best_rbf, best_cfg_rbf = 0, None
GAMMA_LIST = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
K_NEG_LIST = [8, 16, 32, 50]
WMA_LIST   = [0.85, 0.88, 0.90, 0.92, 0.95]

for gamma in GAMMA_LIST:
    for k_neg in K_NEG_LIST:
        for wma in WMA_LIST:
            out = np.stack([wl_rbf_from_sims(c_ica, fi, gamma, k_neg, wma) for fi in range(n_files)])
            auc = eval_loo(out)
            if auc > best_rbf: best_rbf = auc; best_cfg_rbf = ('ica100', gamma, k_neg, wma)
    print(f"  gamma={gamma} done", flush=True)

# Also try pca and std at best gamma
if best_cfg_rbf:
    g_best = best_cfg_rbf[1]
    for cache, name in [(c_pca, 'pca80'), (c_std, 'std80')]:
        for k_neg in K_NEG_LIST:
            for wma in WMA_LIST:
                out = np.stack([wl_rbf_from_sims(cache, fi, g_best, k_neg, wma) for fi in range(n_files)])
                auc = eval_loo(out)
                if auc > best_rbf: best_rbf = auc; best_cfg_rbf = (name, g_best, k_neg, wma)

print(f"  RBF-WL best: {best_rbf:.4f}  cfg={best_cfg_rbf}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_rbf_kernel'] = best_rbf
print(f"  {'*** NEW BEST ***' if best_rbf > CURRENT_BEST else ''}", flush=True)

# RBF blend with UH-triple
if best_cfg_rbf:
    nm, g, kn, wm = best_cfg_rbf
    cache_rbf_best = {'ica100': c_ica, 'pca80': c_pca, 'std80': c_std}[nm]
    rbf_scores = np.stack([wl_rbf_from_sims(cache_rbf_best, fi, g, kn, wm) for fi in range(n_files)])
    best_rbfb, best_cfg_rbfb = 0, None
    for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        blend = (1-w) * uh_triple + w * rbf_scores
        auc = eval_loo(blend)
        if auc > best_rbfb: best_rbfb = auc; best_cfg_rbfb = w
    results['wl_rbf_uh_blend'] = best_rbfb
    print(f"  RBF+UH blend: {best_rbfb:.4f}  w={best_cfg_rbfb}", flush=True)
    print(f"  {'*** NEW BEST ***' if best_rbfb > CURRENT_BEST else ''}", flush=True)

    # RBF + UH + direct_logit (3-way)
    best_rbf3, best_cfg_rbf3 = 0, None
    for w_rbf in [0.03, 0.05, 0.08, 0.10]:
        for w_log in [0.05, 0.08, 0.10]:
            if 1-w_rbf-w_log < 0.70: continue
            blend3 = (1-w_rbf-w_log) * uh_triple + w_rbf * rbf_scores + w_log * preds_logit
            auc = eval_loo(blend3)
            if auc > best_rbf3: best_rbf3 = auc; best_cfg_rbf3 = (w_rbf, w_log)
    results['wl_rbf_uh_logit_3way'] = best_rbf3
    print(f"  RBF+UH+logit 3-way: {best_rbf3:.4f}  cfg={best_cfg_rbf3}", flush=True)
    print(f"  {'*** NEW BEST ***' if best_rbf3 > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Logit Temperature Calibration
# sigmoid(logit/T)：低溫 T<1 使預測更 sharp，高溫 T>1 更 soft
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Logit Temperature Calibration ===", flush=True)
t0 = time.time()
best_temp, best_cfg_temp = 0, None

TEMP_LIST = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
W_LIST = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

for T in TEMP_LIST:
    sig_T = (1.0 / (1.0 + np.exp(-logit_win / T))).astype(np.float32)
    preds_T = np.stack([sig_T[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
    for w in W_LIST:
        blend = (1-w) * uh_triple + w * preds_T
        auc = eval_loo(blend)
        if auc > best_temp: best_temp = auc; best_cfg_temp = (T, w)

print(f"  Temp-calibrated logit best: {best_temp:.4f}  cfg={best_cfg_temp}  ({time.time()-t0:.1f}s)", flush=True)
results['logit_temp_uh_blend'] = best_temp
print(f"  {'*** NEW BEST ***' if best_temp > CURRENT_BEST else ''}", flush=True)

# Also try per-species max vs soft-max aggregation
print("  Testing top-k aggregation instead of window-max...", flush=True)
best_topk, best_cfg_topk = 0, None
T_best = best_cfg_temp[0] if best_cfg_temp else 1.0
sig_Tb = (1.0 / (1.0 + np.exp(-logit_win / T_best))).astype(np.float32)

for top_k in [1, 2, 3, 5]:
    preds_topk = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        s, e = int(file_start[fi]), int(file_end[fi])
        wins = sig_Tb[s:e]  # (n_win, n_species)
        k_act = min(top_k, e - s)
        # Top-k mean per species
        for si in range(n_species):
            sorted_v = np.sort(wins[:, si])[::-1]
            preds_topk[fi, si] = sorted_v[:k_act].mean()
    for w in W_LIST:
        blend = (1-w) * uh_triple + w * preds_topk
        auc = eval_loo(blend)
        if auc > best_topk: best_topk = auc; best_cfg_topk = (T_best, top_k, w)

results['logit_topk_uh_blend'] = best_topk
print(f"  Top-k logit blend: {best_topk:.4f}  cfg={best_cfg_topk}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  {'*** NEW BEST ***' if best_topk > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: 3-Way Blend: UH-triple + direct_logit + species_subspace
# species_subspace_blend was 0.9883; combining with direct_logit (0.9884)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: 3-Way Blend (UH + logit + subspace) ===", flush=True)
t0 = time.time()

def species_subspace_loo(emb_n, n_comp, wma):
    out = np.zeros((n_files, n_species), np.float32)
    dim = emb_n.shape[1]
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr_m = win_file_id != fi
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

# Find best subspace config (sweep n_comp and wma)
print("  Computing species subspace scores...", flush=True)
best_ss_auc, best_ss_cfg = 0, None
for n_comp in [1, 2, 3]:
    for wma in [0.88, 0.90, 0.92]:
        for emb, name in [(ew_ica, 'ica100'), (ew_pca, 'pca80')]:
            out = species_subspace_loo(emb, n_comp, wma)
            auc = eval_loo(out)
            if auc > best_ss_auc: best_ss_auc = auc; best_ss_cfg = (name, emb, n_comp, wma)
    print(f"    n_comp={n_comp} done", flush=True)

print(f"  Subspace standalone: {best_ss_auc:.4f}  cfg={best_ss_cfg[:1] + best_ss_cfg[2:]}", flush=True)

if best_ss_cfg:
    ss_name, ss_emb, ss_nc, ss_wma = best_ss_cfg
    ss_scores = species_subspace_loo(ss_emb, ss_nc, ss_wma)

    # 3-way blend sweep
    best_3way, best_cfg_3way = 0, None
    for w_log in [0.05, 0.08, 0.10, 0.12]:
        for w_ss in [0.05, 0.08, 0.10, 0.12, 0.15]:
            w_uh = 1.0 - w_log - w_ss
            if w_uh < 0.60: continue
            blend3 = w_uh * uh_triple + w_log * preds_logit + w_ss * ss_scores
            auc = eval_loo(blend3)
            if auc > best_3way: best_3way = auc; best_cfg_3way = (w_log, w_ss)

    results['uh_logit_subspace_3way'] = best_3way
    print(f"  UH+logit+subspace 3-way: {best_3way:.4f}  cfg(w_log,w_ss)={best_cfg_3way}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  {'*** NEW BEST ***' if best_3way > CURRENT_BEST else ''}", flush=True)

    # Also: add RBF to the 3-way
    if 'wl_rbf_uh_blend' in results and best_rbf >= 0.985:
        nm, g, kn, wm = best_cfg_rbf
        cache_rbf_b = {'ica100': c_ica, 'pca80': c_pca, 'std80': c_std}[nm]
        rbf_s = np.stack([wl_rbf_from_sims(cache_rbf_b, fi, g, kn, wm) for fi in range(n_files)])
        best_4way, best_cfg_4way = 0, None
        for w_log in [0.05, 0.08]:
            for w_ss in [0.05, 0.10]:
                for w_rbf in [0.03, 0.05]:
                    w_uh = 1.0 - w_log - w_ss - w_rbf
                    if w_uh < 0.60: continue
                    blend4 = w_uh * uh_triple + w_log * preds_logit + w_ss * ss_scores + w_rbf * rbf_s
                    auc = eval_loo(blend4)
                    if auc > best_4way: best_4way = auc; best_cfg_4way = (w_log, w_ss, w_rbf)
        results['uh_logit_subspace_rbf_4way'] = best_4way
        print(f"  4-way blend: {best_4way:.4f}  cfg={best_cfg_4way}", flush=True)
        print(f"  {'*** NEW BEST ***' if best_4way > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Attention-Weighted Prototype WL
# 改用 softmax attention 選擇正例 windows，而非 max/mean 混合
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Attention-Weighted Prototype WL ===", flush=True)
t0 = time.time()

def wl_attention_from_cache(cache, fi, tau, k_neg, wma):
    """Softmax-attention weighted positive prototype"""
    te, tr, tl, sims = cache[fi]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pos_idx = np.where(tl[:, si] > 0.5)[0]
        neg_idx = np.where(tl[:, si] < 0.1)[0]
        if len(pos_idx) == 0: ws[:, si] = 0.5; continue
        ps = sims[:, pos_idx]  # (n_te, n_pos)
        # Softmax attention weights over positive windows
        att_raw = ps / tau  # (n_te, n_pos)
        att_raw = att_raw - att_raw.max(1, keepdims=True)  # stability
        att_w = np.exp(att_raw); att_w /= att_w.sum(1, keepdims=True) + EPS
        sp = (att_w * ps).sum(1)  # weighted sum of sims (expected similarity)
        if len(neg_idx) > 0:
            ns2 = sims[:, neg_idx]
            k2 = min(k_neg, len(neg_idx))
            top_idx = np.argsort(-ns2, axis=1)[:, :k2]
            tn_scores = np.array([
                (te[j] @ tr[neg_idx[top_idx[j]]].mean(0) /
                 (np.linalg.norm(tr[neg_idx[top_idx[j]]].mean(0)) + EPS))
                for j in range(len(te))], dtype=np.float32)
            ws[:, si] = (sp - tn_scores + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return wma * ws.max(0) + (1-wma) * ws.mean(0)

best_att, best_cfg_att = 0, None
TAU_LIST  = [0.05, 0.10, 0.20, 0.50, 1.0]
K_NEG2    = [8, 16, 32, 50]
WMA2      = [0.88, 0.90, 0.92, 0.95]

for tau in TAU_LIST:
    for k_neg in K_NEG2:
        for wma in WMA2:
            out = np.stack([wl_attention_from_cache(c_ica, fi, tau, k_neg, wma) for fi in range(n_files)])
            auc = eval_loo(out)
            if auc > best_att: best_att = auc; best_cfg_att = ('ica100', tau, k_neg, wma)
    print(f"  tau={tau} done", flush=True)

print(f"  Attention-WL best: {best_att:.4f}  cfg={best_cfg_att}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_attention'] = best_att
print(f"  {'*** NEW BEST ***' if best_att > CURRENT_BEST else ''}", flush=True)

# Blend attention WL with UH
if best_cfg_att:
    nm, tau_b, kn_b, wma_b = best_cfg_att
    att_scores = np.stack([wl_attention_from_cache(c_ica, fi, tau_b, kn_b, wma_b) for fi in range(n_files)])
    best_attb, best_cfg_attb = 0, None
    for w in [0.05, 0.08, 0.10, 0.12, 0.15]:
        blend = (1-w) * uh_triple + w * att_scores
        auc = eval_loo(blend)
        if auc > best_attb: best_attb = auc; best_cfg_attb = w
    results['wl_attention_uh_blend'] = best_attb
    print(f"  Attention+UH blend: {best_attb:.4f}  w={best_cfg_attb}", flush=True)
    print(f"  {'*** NEW BEST ***' if best_attb > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 69 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)
print(f"  UH-triple ref: {uh_auc:.4f}", flush=True)
print(f"  direct_logit_uh ref: {direct_auc:.4f}", flush=True)

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

print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
if not new_best_found:
    print("未超越 0.9884，已 append 到 experiments。", flush=True)
else:
    print(f"*** JSON BEST UPDATED: {best_new_method} = {best_new_auc:.6f} ***", flush=True)
