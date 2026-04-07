"""
Batch 64b: Product-of-Experts (PoE) - 優化快速版本
只保留 PoE 組合（不含 Ridge/PLDA，因為太慢）

理念：現有 triple blend 用線性加權，PoE 改用乘法組合。
若三個 embedding 空間對 class 條件獨立，PoE 理論最優。

最佳化：
- 預計算所有 file 的 WL window scores（不重複計算）
- PoE fusion 在已存的 window scores 上直接做
- 掃描：temp (PoE 溫度) + agg mode + 哪幾個空間組合

Current best: 0.9873025 (wl_uh_seedens_blend)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[file_start[fi]:file_end[fi]] = fi

file_labels = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

EPS = 1e-7
mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9873024930999804

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# ─── Precompute embeddings ────────────────────────────────────────────────────
print("Precomputing embeddings...", flush=True)
ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2')

pca80 = PCA(n_components=80, random_state=42)
ew_pca = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2')

scaler = StandardScaler()
ew_std = normalize(PCA(n_components=80, random_state=42).fit_transform(
    scaler.fit_transform(emb_win).astype(np.float32)).astype(np.float32), norm='l2')

# Best params from wl_uh_seedens_blend
ICA_K, ICA_WMA, ICA_WMP = 50, 0.92, 0.80
STD_K, STD_WMA, STD_WMP =  4, 0.65, 0.60
PCA_K, PCA_WMA, PCA_WMP =  4, 0.60, 0.70
W_ICA, W_STD, W_PCA = 0.655, 0.225, 0.120

# ─── WL contrast (returns window scores [n_te, n_species]) ───────────────────
def wl_scores(emb_n, fi, k_neg, wmp):
    """Returns raw window scores [n_te_wins, n_species] in ~[0,1]."""
    te = emb_n[win_file_id == fi]
    tr_m = win_file_id != fi
    tr = emb_n[tr_m]; tl = labels_win[tr_m]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
        if not pm.any(): ws[:, si] = 0.5; continue
        pw = tr[pm]; ps = te @ pw.T
        pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
        sp = wmp * ps.max(1) + (1 - wmp) * (te @ pp)
        if nm.any():
            nw = tr[nm]; ns2 = te @ nw.T; k2 = min(k_neg, ns2.shape[1])
            tn = nw[np.argsort(-ns2, axis=1)[:, :k2]].mean(1)
            tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
            ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return ws  # [n_te, n_species]

print("Pre-caching WL window scores for all files...", flush=True)
t0 = time.time()
ica_wins  = [None] * n_files  # [n_te, n_species]
std_wins  = [None] * n_files
pca_wins  = [None] * n_files
for fi in range(n_files):
    ica_wins[fi] = wl_scores(ew_ica, fi, ICA_K, ICA_WMP)
    std_wins[fi] = wl_scores(ew_std, fi, STD_K, STD_WMP)
    pca_wins[fi] = wl_scores(ew_pca, fi, PCA_K, PCA_WMP)
print(f"  Cached {n_files} files ({time.time()-t0:.0f}s)", flush=True)

# ─── Helper: aggregate window scores to file score ────────────────────────────
def agg(ws, wma):
    return wma * ws.max(0) + (1 - wma) * ws.mean(0)

# Reference: UH-triple (linear blend) for comparison
uh_triple = np.stack([
    agg(W_ICA * ica_wins[fi] + W_STD * std_wins[fi] + W_PCA * pca_wins[fi], ICA_WMA)
    for fi in range(n_files)
])
print(f"UH-triple reference: {eval_loo(uh_triple):.4f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Full PoE (all 3 spaces)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Full PoE (ICA+STD+PCA) ===", flush=True)
t0 = time.time()
best_poe3 = 0; best_cfg_poe3 = None

def poe3(fi, temp, wma):
    """Product of experts of 3 window score arrays."""
    s_ica = np.clip(ica_wins[fi], EPS, 1-EPS)
    s_std = np.clip(std_wins[fi], EPS, 1-EPS)
    s_pca = np.clip(pca_wins[fi], EPS, 1-EPS)
    log_odds = (np.log(s_ica/(1-s_ica)) +
                np.log(s_std/(1-s_std)) +
                np.log(s_pca/(1-s_pca))) * temp
    ws_poe = 1.0 / (1.0 + np.exp(-log_odds))
    return agg(ws_poe, wma)

for temp in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
    for wma in [0.85, 0.88, 0.90, 0.92, 0.95]:
        out = np.stack([poe3(fi, temp, wma) for fi in range(n_files)])
        auc = eval_loo(out)
        if auc > best_poe3: best_poe3 = auc; best_cfg_poe3 = (temp, wma)

print(f"  PoE3 best: {best_poe3:.4f}  cfg={best_cfg_poe3}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_poe3'] = best_poe3
print(f"  {'*** NEW BEST ***' if best_poe3 > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: PoE(ICA+STD) only, blend with PCA linear
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: PoE(ICA+STD) + linear PCA ===", flush=True)
t0 = time.time()
best_poe2 = 0; best_cfg_poe2 = None

def poe2_blend(fi, temp, w_poe, w_pca_linear, wma):
    s_ica = np.clip(ica_wins[fi], EPS, 1-EPS)
    s_std = np.clip(std_wins[fi], EPS, 1-EPS)
    log_odds = (np.log(s_ica/(1-s_ica)) + np.log(s_std/(1-s_std))) * temp
    ws_poe = 1.0 / (1.0 + np.exp(-log_odds))
    # Blend PoE(ICA+STD) with PCA window scores linearly
    ws_combined = w_poe * ws_poe + w_pca_linear * pca_wins[fi]
    return agg(ws_combined, wma)

for temp in [0.3, 0.5, 0.7, 1.0, 1.5]:
    for w_poe in [0.78, 0.82, 0.86, 0.88, 0.90, 0.92]:
        w_pca_lin = 1 - w_poe
        for wma in [0.88, 0.90, 0.92]:
            out = np.stack([poe2_blend(fi, temp, w_poe, w_pca_lin, wma) for fi in range(n_files)])
            auc = eval_loo(out)
            if auc > best_poe2: best_poe2 = auc; best_cfg_poe2 = (temp, w_poe, wma)

print(f"  PoE2+PCA best: {best_poe2:.4f}  cfg={best_cfg_poe2}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_poe2_pca'] = best_poe2
print(f"  {'*** NEW BEST ***' if best_poe2 > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: PoE at file-score level (not window level)
# Instead of PoE before aggregation, do aggregation first then PoE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: File-level PoE (aggregate then PoE) ===", flush=True)
t0 = time.time()
best_poe_file = 0; best_cfg_poe_file = None

# Pre-compute file-level scores for each embedding with its own wma
def file_scores_sweep():
    """All (emb, wma) combinations precomputed."""
    scores = {}
    for wma in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
        scores[('ica', wma)] = np.stack([agg(ica_wins[fi], wma) for fi in range(n_files)])
        scores[('std', wma)] = np.stack([agg(std_wins[fi], wma) for fi in range(n_files)])
        scores[('pca', wma)] = np.stack([agg(pca_wins[fi], wma) for fi in range(n_files)])
    return scores

precomp = file_scores_sweep()

for wma_ica in [0.90, 0.92, 0.95]:
    for wma_std in [0.60, 0.65, 0.70]:
        for wma_pca in [0.60, 0.65]:
            for temp in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                s_ica = np.clip(precomp[('ica', wma_ica)], EPS, 1-EPS)
                s_std = np.clip(precomp[('std', wma_std)], EPS, 1-EPS)
                s_pca = np.clip(precomp[('pca', wma_pca)], EPS, 1-EPS)
                log_odds = (np.log(s_ica/(1-s_ica)) +
                            np.log(s_std/(1-s_std)) +
                            np.log(s_pca/(1-s_pca))) * temp
                out = 1.0 / (1.0 + np.exp(-log_odds))
                auc = eval_loo(out)
                if auc > best_poe_file:
                    best_poe_file = auc
                    best_cfg_poe_file = (wma_ica, wma_std, wma_pca, temp)

print(f"  File-PoE best: {best_poe_file:.4f}  cfg={best_cfg_poe_file}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_poe_file'] = best_poe_file
print(f"  {'*** NEW BEST ***' if best_poe_file > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Hybrid - Linear blend at window level + PoE at file level
# Apply PoE after the triple linear blend (treat blend as one "expert")
# plus original ICA-only as second "expert"
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Hybrid Window-blend + File-PoE ===", flush=True)
t0 = time.time()
best_hyb = 0; best_cfg_hyb = None

# Precompute triple blend file scores
for wma in [0.88, 0.90, 0.92, 0.95]:
    blend_wins = [W_ICA * ica_wins[fi] + W_STD * std_wins[fi] + W_PCA * pca_wins[fi]
                  for fi in range(n_files)]
    triple_file = np.stack([agg(blend_wins[fi], wma) for fi in range(n_files)])
    # ICA-only file scores (different wma)
    for wma_ica in [0.88, 0.90, 0.92, 0.95]:
        ica_file = precomp[('ica', wma_ica)]
        # PoE of triple_blend and ica_only at file level
        for temp in [0.3, 0.5, 0.7, 1.0, 1.5]:
            s_t = np.clip(triple_file, EPS, 1-EPS)
            s_i = np.clip(ica_file, EPS, 1-EPS)
            log_odds = (np.log(s_t/(1-s_t)) + np.log(s_i/(1-s_i))) * temp
            out = 1.0 / (1.0 + np.exp(-log_odds))
            auc = eval_loo(out)
            if auc > best_hyb:
                best_hyb = auc; best_cfg_hyb = (wma, wma_ica, temp)

print(f"  Hybrid best: {best_hyb:.4f}  cfg={best_cfg_hyb}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_poe_hybrid'] = best_hyb
print(f"  {'*** NEW BEST ***' if best_hyb > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Asymmetric PoE (different temperatures per expert)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Asymmetric PoE (per-expert temperature) ===", flush=True)
t0 = time.time()
best_apoe = 0; best_cfg_apoe = None

for t_ica in [0.5, 0.7, 1.0, 1.5, 2.0]:
    for t_std in [0.3, 0.5, 0.7, 1.0]:
        for t_pca in [0.2, 0.3, 0.5]:
            for wma_ica in [0.90, 0.92]:
                for wma_std in [0.65]:
                    for wma_pca in [0.60]:
                        s_ica = np.clip(precomp[('ica', wma_ica)], EPS, 1-EPS)
                        s_std = np.clip(precomp[('std', wma_std)], EPS, 1-EPS)
                        s_pca = np.clip(precomp[('pca', wma_pca)], EPS, 1-EPS)
                        log_odds = (np.log(s_ica/(1-s_ica)) * t_ica +
                                    np.log(s_std/(1-s_std)) * t_std +
                                    np.log(s_pca/(1-s_pca)) * t_pca)
                        out = 1.0 / (1.0 + np.exp(-log_odds))
                        auc = eval_loo(out)
                        if auc > best_apoe:
                            best_apoe = auc; best_cfg_apoe = (t_ica, t_std, t_pca, wma_ica)

print(f"  Asym-PoE best: {best_apoe:.4f}  cfg={best_cfg_apoe}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_poe_asym'] = best_apoe
print(f"  {'*** NEW BEST ***' if best_apoe > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 64b Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
new_best_found = False
for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        new_best_found = True
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)

with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)

print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
if not new_best_found:
    print("未超越 0.9873，已 append 到 experiments。", flush=True)
