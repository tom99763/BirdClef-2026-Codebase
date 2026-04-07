"""
Batch 68: WL on Raw 1536-dim + Extended Blends
全新：使用完整 1536-dim Perch embedding（只做 L2 norm）做 WL contrast

已知 wl_raw1280 standalone=0.9804, blend=0.9873
預計 wl_raw1536 可能更強（更多資訊）

方法：
1. WL-raw1536 standalone
2. WL-raw1536 + direct_logit blend
3. WL-raw1536 + WL-ICA100-triple blend (4-way)
4. WL-raw1536 + WL-ICA100-triple + direct_logit (5-way)
5. Per-species Logistic Regression (species_logreg) on ICA100 features

Current best: direct_logit_uh_blend = 0.9884
"""
import numpy as np, json, os, time, pickle
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression
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

# ─── Precompute all embeddings ────────────────────────────────────────────────
print("Precomputing embeddings...", flush=True)
t0 = time.time()
# Raw 1536-dim (just L2 normalize)
ew_raw1536 = normalize(emb_win, norm='l2').astype(np.float32)

# ICA / PCA / Std-PCA (same as UH-triple)
with open("outputs/embed_prior_model.pkl", "rb") as f:
    ep = pickle.load(f)
ew_ica = ep['emb_win_ica_norm']   # (739, 100)
ew_std = ep['emb_win_std_norm']   # (739, 80)
ew_pca = ep['emb_win_pca_norm']   # (739, 80)

# Direct logit sigmoid max per file
logit_sig = 1.0 / (1.0 + np.exp(-logit_win))
preds_logit = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    preds_logit[fi] = logit_sig[s:e].max(0)
logit_auc = eval_loo(preds_logit)

W_ICA, W_STD, W_PCA = 0.655, 0.225, 0.120
ICA_K, ICA_WMA, ICA_WMP = 50, 0.92, 0.80
STD_K, STD_WMA, STD_WMP =  4, 0.65, 0.60
PCA_K, PCA_WMA, PCA_WMP =  4, 0.60, 0.70
print(f"  Done ({time.time()-t0:.1f}s)  Direct logit AUC: {logit_auc:.4f}", flush=True)

# ─── Sim cache ────────────────────────────────────────────────────────────────
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
            top = np.argsort(-ns2, axis=1)[:, :k2]
            tn_scores = np.array([
                (te[j] @ tr[neg_idx[top[j]]].mean(0) /
                 (np.linalg.norm(tr[neg_idx[top[j]]].mean(0)) + EPS))
                for j in range(len(te))], dtype=np.float32)
            ws[:, si] = (sp - tn_scores + 1) / 2
        else: ws[:, si] = (sp + 1) / 2
    return wma * ws.max(0) + (1-wma) * ws.mean(0)

def wl_sweep(cache, k_list, wmp_list, wma_list):
    best, cfg = 0, None
    for k in k_list:
        for wmp in wmp_list:
            for wma in wma_list:
                out = np.stack([wl_from_cache(cache, fi, k, wmp, wma) for fi in range(n_files)])
                auc = eval_loo(out)
                if auc > best: best = auc; cfg = (k, wmp, wma)
    return best, cfg

print("Building caches...", flush=True)
t0 = time.time()
c_raw  = build_cache(ew_raw1536)
c_ica  = build_cache(ew_ica)
c_std  = build_cache(ew_std)
c_pca  = build_cache(ew_pca)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

# ─── UH-triple reference ─────────────────────────────────────────────────────
print("Computing UH-triple reference...", flush=True)
t0 = time.time()
s_ica_ref = np.stack([wl_from_cache(c_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std_ref = np.stack([wl_from_cache(c_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca_ref = np.stack([wl_from_cache(c_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
uh_triple = W_ICA * s_ica_ref + W_STD * s_std_ref + W_PCA * s_pca_ref
uh_auc = eval_loo(uh_triple)
# Current best (UH + direct_logit w=0.08)
best_known = (1-0.08) * uh_triple + 0.08 * preds_logit
known_auc = eval_loo(best_known)
print(f"  UH-triple: {uh_auc:.4f}  (UH+logit w=0.08): {known_auc:.4f}  ({time.time()-t0:.1f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: WL on Raw 1536-dim
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: WL-Raw1536 ===", flush=True)
t0 = time.time()
best_raw, best_cfg_raw = wl_sweep(c_raw,
    k_list=[4, 8, 16, 32, 50, 80],
    wmp_list=[0.60, 0.70, 0.80, 0.90, 1.0],
    wma_list=[0.80, 0.85, 0.88, 0.90, 0.92, 0.95])
print(f"  WL-raw1536 best: {best_raw:.4f}  cfg={best_cfg_raw}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_raw1536'] = best_raw
print(f"  {'*** NEW BEST ***' if best_raw > CURRENT_BEST else ''}", flush=True)

# Best raw1536 scores
k_r, wmp_r, wma_r = best_cfg_raw
s_raw = np.stack([wl_from_cache(c_raw, fi, k_r, wmp_r, wma_r) for fi in range(n_files)])

# ─── Raw1536 + Direct Logit blend ────────────────────────────────────────────
print("  WL-raw1536 + direct_logit blend...", flush=True)
t0 = time.time()
best_rb, best_wb = 0, None
for w in [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]:
    blend = (1-w) * s_raw + w * preds_logit
    auc = eval_loo(blend)
    if auc > best_rb: best_rb = auc; best_wb = w
print(f"  Raw1536+logit blend: {best_rb:.4f}  w={best_wb}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_raw1536_logit_blend'] = best_rb
print(f"  {'*** NEW BEST ***' if best_rb > CURRENT_BEST else ''}", flush=True)

# ─── Raw1536 + UH-triple blend ───────────────────────────────────────────────
print("  WL-raw1536 + UH-triple blend...", flush=True)
t0 = time.time()
best_ruh, best_cfg_ruh = 0, None
for w_r in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blend = (1-w_r) * uh_triple + w_r * s_raw
    auc = eval_loo(blend)
    if auc > best_ruh: best_ruh = auc; best_cfg_ruh = w_r
print(f"  Raw1536+UH blend: {best_ruh:.4f}  w_raw={best_cfg_ruh}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_raw1536_uh_blend'] = best_ruh
print(f"  {'*** NEW BEST ***' if best_ruh > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: WL-raw1536 + UH-triple + Direct Logit (3-way)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Raw1536 + UH + Direct-Logit (3-way) ===", flush=True)
t0 = time.time()
best_3way, best_cfg_3way = 0, None
for w_raw in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    for w_logit in [0.04, 0.06, 0.08, 0.10, 0.12]:
        w_uh = 1.0 - w_raw - w_logit
        if w_uh < 0.55: continue
        blend = w_uh * uh_triple + w_raw * s_raw + w_logit * preds_logit
        auc = eval_loo(blend)
        if auc > best_3way: best_3way = auc; best_cfg_3way = (w_uh, w_raw, w_logit)
print(f"  3-way best: {best_3way:.4f}  cfg={best_cfg_3way}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_raw1536_uh_logit_3way'] = best_3way
print(f"  {'*** NEW BEST ***' if best_3way > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Per-species Logistic Regression (species_logreg)
# Train LR per species on ICA100 window-level features (LOO-CV)
# Different from WL contrast — discriminative classifier, not prototype matching
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Per-species Logistic Regression ===", flush=True)
print("理念：每 species 在 window 層級 fit LR，完全監督式，不同於 WL prototype 方法", flush=True)

def species_logreg_loo(emb_n, C, wma):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr_m = win_file_id != fi
        tr = emb_n[tr_m]; tl = labels_win[tr_m]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            y_tr = (tl[:, si] > 0.5).astype(np.float32)
            n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
            if n_pos < 1: ws[:, si] = 0.0; continue
            if n_neg < 1: ws[:, si] = 1.0; continue
            try:
                lr = LogisticRegression(C=C, max_iter=100, solver='lbfgs',
                                        class_weight='balanced', random_state=42)
                lr.fit(tr, y_tr)
                ws[:, si] = lr.predict_proba(te)[:, 1]
            except Exception:
                ws[:, si] = 0.5
        out[fi] = wma * ws.max(0) + (1-wma) * ws.mean(0)
    return out

t0 = time.time()
best_lr, best_cfg_lr = 0, None
for C in [0.01, 0.1, 1.0, 10.0]:
    for wma in [0.85, 0.90, 0.92, 0.95]:
        out = species_logreg_loo(ew_ica, C, wma)
        auc = eval_loo(out)
        if auc > best_lr: best_lr = auc; best_cfg_lr = (C, wma)
    print(f"  C={C} done...", flush=True)

print(f"  Species-LR best: {best_lr:.4f}  cfg={best_cfg_lr}  ({time.time()-t0:.0f}s)", flush=True)
results['species_logreg'] = best_lr
print(f"  {'*** NEW BEST ***' if best_lr > CURRENT_BEST else ''}", flush=True)

# Blend LR with UH+logit
t0 = time.time()
if best_cfg_lr:
    C_b, wma_b = best_cfg_lr
    lr_scores = species_logreg_loo(ew_ica, C_b, wma_b)
    best_lrb, best_cfg_lrb = 0, None
    for w_lr in [0.05, 0.10, 0.15, 0.20]:
        blend = (1-0.08-w_lr) * uh_triple + 0.08 * preds_logit + w_lr * lr_scores
        if 1-0.08-w_lr < 0.5: continue
        auc = eval_loo(blend)
        if auc > best_lrb: best_lrb = auc; best_cfg_lrb = w_lr
    results['species_logreg_blend'] = best_lrb
    print(f"  LR+UH+logit blend: {best_lrb:.4f}  w_lr={best_cfg_lrb}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  {'*** NEW BEST ***' if best_lrb > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: WL-raw1536 as 4th component in quad blend + direct logit
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Quad WL (ICA+STD+PCA+RAW) + Direct Logit ===", flush=True)
t0 = time.time()
best_quad, best_cfg_quad = 0, None

# Normalise raw so it sums with others
for w_raw_4th in [0.05, 0.10, 0.15, 0.20, 0.25]:
    for w_logit_4th in [0.04, 0.06, 0.08, 0.10]:
        # Scale down ICA/STD/PCA to accommodate raw
        scale = (1.0 - w_raw_4th - w_logit_4th)
        if scale < 0.5: continue
        w_ica4 = scale * W_ICA; w_std4 = scale * W_STD; w_pca4 = scale * W_PCA
        quad = w_ica4 * s_ica_ref + w_std4 * s_std_ref + w_pca4 * s_pca_ref + \
               w_raw_4th * s_raw + w_logit_4th * preds_logit
        auc = eval_loo(quad)
        if auc > best_quad:
            best_quad = auc; best_cfg_quad = (w_raw_4th, w_logit_4th)

print(f"  Quad+logit best: {best_quad:.4f}  cfg={best_cfg_quad}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_quad_raw_logit'] = best_quad
print(f"  {'*** NEW BEST ***' if best_quad > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 68 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)
print(f"  (current best: direct_logit_uh_blend = {CURRENT_BEST:.4f})", flush=True)

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
if new_best_found:
    print(f"NEW BEST: {best_new_method} AUC={best_new_auc:.4f}", flush=True)
else:
    print("未超越 0.9884，已 append 到 experiments。", flush=True)
