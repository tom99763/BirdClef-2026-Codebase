"""
五個新 no-logit 方法搜尋：
A) Label Propagation (LabelSpreading) — 考慮全局圖結構
B) Max-pool window embedding — file emb = max over windows (取代 mean)
C) 高維直接 KNN — 1536-dim + geo (無 PCA)
D) 多空間 Ensemble — attn_pca24 + window_knn + mahal 加權組合
E) UMAP 降維 + Attn-KNN

No-logit best: attn_k10_T02_pca24_day = 0.8758
"""
import numpy as np, json, pickle, re, os, shutil
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.semi_supervised import LabelSpreading
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────
raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

# File-level aggregation (mean + max)
file_embs_mean = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_embs_max  = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species), dtype=np.float32)

for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs_mean[fi] = emb_win[s:e].mean(0)
    file_embs_max[fi]  = emb_win[s:e].max(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

file_embs_norm_mean = normalize(file_embs_mean, norm='l2').astype(np.float32)
file_embs_norm_max  = normalize(file_embs_max,  norm='l2').astype(np.float32)

# Window embeddings (for window KNN)
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), dtype=np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

# ── Geo features ────────────────────────────────────────────────────────────
SITES = ['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx = {s: i for i, s in enumerate(SITES)}
file_sites  = np.zeros(n_files, dtype=np.int32)
file_hours  = np.zeros(n_files, dtype=np.float32)
file_months = np.zeros(n_files, dtype=np.float32)
file_days   = np.zeros(n_files, dtype=np.float32)
for fi, fname in enumerate(file_list):
    m = re.match(r'BC2026_Train_\d+_(S\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})', str(fname))
    if m:
        site, yr, mo, dy, hr, mn = m.groups()
        file_sites[fi]  = site2idx.get(site, 0)
        file_hours[fi]  = int(hr)
        file_months[fi] = int(mo)
        dpm = [0,31,28,31,30,31,30,31,31,30,31,30,31]
        file_days[fi] = sum(dpm[:int(mo)]) + int(dy)

site_oh   = np.eye(len(SITES), dtype=np.float32)[file_sites]
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24),
                       np.cos(2*np.pi*file_hours/24)], axis=1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12),
                       np.cos(2*np.pi*(file_months-1)/12)], axis=1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365),
                       np.cos(2*np.pi*(file_days-1)/365)], axis=1).astype(np.float32)
geo_all   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], axis=1).astype(np.float32)

# Standard pca24+day space (best nologit baseline)
pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm_mean)
X24   = pca24.transform(file_embs_norm_mean).astype(np.float32)
X24  /= (X24.std(0) + 1e-6)
X_comb_mean = np.concatenate([X24, geo_all], axis=1).astype(np.float32)
X_nl_mean   = (X_comb_mean / np.linalg.norm(X_comb_mean, axis=1, keepdims=True)).astype(np.float32)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

def attn_knn_loo(X, labels, k=10, T=0.2):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * labels[tr[top]]).sum(0)
    return preds

EPS  = 1e-7
BEST_NL = 0.875789
results  = {}
best_so_far = BEST_NL

print(f"Files={n_files}, species={n_species}")
print(f"No-logit best to beat: {BEST_NL:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# A) Label Propagation / LabelSpreading
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("A) Label Spreading (graph-based label propagation)")
print("="*60, flush=True)

def label_spreading_loo(X, labels, kernel='knn', knn_k=7, alpha=0.2):
    """LOO with LabelSpreading: treat held-out file as unlabeled (-1)."""
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    active = np.where(labels.sum(0) > 0)[0]
    for i in range(n_files):
        y_init = labels.copy()
        y_init[i] = -1  # mark as unlabeled
        try:
            ls = LabelSpreading(kernel=kernel, n_neighbors=knn_k,
                                alpha=alpha, max_iter=30)
            # Fit per species (binary)
            for s in active:
                y_s = y_init[:, s].astype(int)
                if (y_s[y_s >= 0] > 0).sum() == 0:
                    continue
                ls.fit(X, y_s)
                preds[i, s] = ls.label_distributions_[i, 1]
        except Exception:
            pass
        if (i+1) % 20 == 0:
            print(f"  fold {i+1}/{n_files}", flush=True)
    return preds

for knn_k, alpha in [(5, 0.2), (7, 0.2), (10, 0.2), (7, 0.4), (5, 0.5)]:
    print(f"  LabelSpreading knn={knn_k} alpha={alpha} ...", flush=True)
    p = label_spreading_loo(X_nl_mean, file_labels, kernel='knn',
                             knn_k=knn_k, alpha=alpha)
    auc = macro_auc(file_labels, p)
    nm  = f'label_spread_k{knn_k}_a{alpha:.1f}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far:
        best_so_far = auc
        best_method_A = (nm, auc, p.copy())
    print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

print(f"After A, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# B) Max-pool window embedding
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("B) Max-pool window embedding (file emb = window max, not mean)")
print("="*60, flush=True)

# Build pca24+day space with max-pool embeddings
pca24_max = PCA(n_components=24, random_state=42).fit(file_embs_norm_max)
X24_max   = pca24_max.transform(file_embs_norm_max).astype(np.float32)
X24_max  /= (X24_max.std(0) + 1e-6)
X_comb_max = np.concatenate([X24_max, geo_all], axis=1).astype(np.float32)
X_nl_max   = (X_comb_max / np.linalg.norm(X_comb_max, axis=1, keepdims=True)).astype(np.float32)

for k, T in [(5, 0.2), (10, 0.2), (10, 0.15), (15, 0.2), (10, 0.3)]:
    p   = attn_knn_loo(X_nl_max, file_labels, k=k, T=T)
    auc = macro_auc(file_labels, p)
    nm  = f'maxpool_pca24day_k{k}_T{T}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far:
        best_so_far = auc
        best_method_B = (nm, auc, p.copy(),
                         pca24_max, X24_max.std(0)+1e-6, X_nl_max.copy())
    print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

# Blend mean + max embeddings
for blend_w in [0.3, 0.5, 0.7]:
    X_blend = blend_w * X_nl_mean + (1-blend_w) * X_nl_max
    X_blend /= np.linalg.norm(X_blend, axis=1, keepdims=True) + 1e-8
    for k, T in [(10, 0.2), (10, 0.15)]:
        p   = attn_knn_loo(X_blend, file_labels, k=k, T=T)
        auc = macro_auc(file_labels, p)
        nm  = f'blend_mean{blend_w:.1f}_max{1-blend_w:.1f}_k{k}_T{T}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After B, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# C) 高維直接 KNN (1536-dim + geo, 無 PCA)
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("C) High-dim direct KNN: 1536-dim normalized + geo (no PCA)")
print("="*60, flush=True)

# Scale geo to match embedding magnitude
geo_scaled = geo_all * 0.5  # geo features are already [0,1] or sinusoidal
X_highdim  = np.concatenate([file_embs_norm_mean, geo_scaled], axis=1).astype(np.float32)
X_highdim /= np.linalg.norm(X_highdim, axis=1, keepdims=True) + 1e-8

for geo_w in [0.3, 0.5, 1.0]:
    geo_s = geo_all * geo_w
    X_hd  = np.concatenate([file_embs_norm_mean, geo_s], axis=1).astype(np.float32)
    X_hd /= np.linalg.norm(X_hd, axis=1, keepdims=True) + 1e-8
    for k, T in [(5, 0.2), (10, 0.2), (10, 0.15), (15, 0.2)]:
        p   = attn_knn_loo(X_hd, file_labels, k=k, T=T)
        auc = macro_auc(file_labels, p)
        nm  = f'highdim_geo{geo_w:.1f}_k{k}_T{T}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

# Pure 1536-dim (no geo)
for k, T in [(5, 0.2), (10, 0.2), (10, 0.15)]:
    p   = attn_knn_loo(file_embs_norm_mean, file_labels, k=k, T=T)
    auc = macro_auc(file_labels, p)
    nm  = f'pure1536_k{k}_T{T}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

print(f"After C, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# D) 多空間 Ensemble (attn_pca24 + window_knn + mahal 加權)
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("D) Multi-space ensemble: attn_pca24 + window_knn + mahal")
print("="*60, flush=True)

# 1) attn_pca24+day (best nologit)
y_attn = attn_knn_loo(X_nl_mean, file_labels, k=10, T=0.2)
print(f"  attn_pca24+day k10: {macro_auc(file_labels, y_attn):.4f}", flush=True)

# 2) window KNN k=5 mean (already have pkl, recompute for blend)
print("  Computing window KNN...", flush=True)
y_win = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    nw_te = te_e - te_s
    tr_mask = win_file_id != i
    X_tr = emb_win_norm[tr_mask]
    tr_wi = np.where(tr_mask)[0]
    Y_tr = file_labels[win_file_id[tr_wi]]
    sims = X_te @ X_tr.T
    top  = np.argsort(-sims, axis=1)[:, :5]
    win_p = np.zeros((nw_te, n_species), dtype=np.float32)
    for wi in range(nw_te):
        w = sims[wi, top[wi]].clip(0)
        ws = w.sum(); w = w/ws if ws > 1e-8 else np.ones(5)/5
        win_p[wi] = (w[:, None] * Y_tr[top[wi]]).sum(0)
    y_win[i] = win_p.mean(0)
print(f"  window_knn k5 mean: {macro_auc(file_labels, y_win):.4f}", flush=True)

# 3) Mahalanobis KNN (pca32, k=5) — recompute
pca32 = PCA(n_components=32, random_state=42).fit(file_embs_norm_mean)
X32   = pca32.transform(file_embs_norm_mean).astype(np.float32)
cov   = np.cov(X32.T)
try:
    cov_inv = np.linalg.inv(cov + 1e-4 * np.eye(32))
except:
    cov_inv = np.eye(32)

y_mahal = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    tr = np.array([j for j in range(n_files) if j != i])
    diff = X32[tr] - X32[i]
    md2  = np.einsum('nd,dd,nd->n', diff, cov_inv, diff)
    md   = np.sqrt(md2 + 1e-8)
    top  = np.argsort(md)[:5]
    w    = 1.0 / (md[top] + 1e-6); w /= w.sum()
    y_mahal[i] = (w[:, None] * file_labels[tr[top]]).sum(0)
print(f"  mahal_pca32 k5:     {macro_auc(file_labels, y_mahal):.4f}", flush=True)

# Sweep ensemble weights
print("  Sweeping ensemble weights...", flush=True)
for w1 in [0.5, 0.6, 0.7, 0.8]:
    for w2 in [0.1, 0.15, 0.2, 0.25]:
        w3 = 1 - w1 - w2
        if w3 < 0: continue
        blend = w1 * y_attn + w2 * y_win + w3 * y_mahal
        auc   = macro_auc(file_labels, blend)
        nm    = f'ens3_a{w1:.2f}_w{w2:.2f}_m{w3:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_ens3 = (nm, auc, blend.copy(), w1, w2, w3)
        if auc > BEST_NL - 0.002:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

# Also try 2-way blends
for wa, wb in [(0.7, 0.3), (0.8, 0.2), (0.6, 0.4), (0.75, 0.25)]:
    for (ya, na), (yb, nb_) in [
        ((y_attn,'attn'), (y_win,'win')),
        ((y_attn,'attn'), (y_mahal,'mahal')),
        ((y_win,'win'), (y_mahal,'mahal')),
    ]:
        blend = wa * ya + wb * yb
        auc   = macro_auc(file_labels, blend)
        nm    = f'ens2_{na}{wa:.2f}_{nb_}{wb:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST_NL - 0.002:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After D, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# E) UMAP 降維 + Attn-KNN
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("E) UMAP reduction + Attn-KNN")
print("="*60, flush=True)

try:
    import umap
    UMAP_OK = True
except ImportError:
    UMAP_OK = False
    print("  umap-learn not installed, skipping E", flush=True)

if UMAP_OK:
    for n_comp in [8, 16, 24]:
        for n_neighbors in [5, 10, 15]:
            try:
                reducer = umap.UMAP(n_components=n_comp, n_neighbors=n_neighbors,
                                     metric='cosine', random_state=42, verbose=False)
                X_umap = reducer.fit_transform(file_embs_norm_mean).astype(np.float32)
                X_umap /= (X_umap.std(0) + 1e-6)
                X_umap_geo = np.concatenate([X_umap, geo_all], axis=1).astype(np.float32)
                X_umap_geo /= np.linalg.norm(X_umap_geo, axis=1, keepdims=True) + 1e-8
                for k, T in [(10, 0.2), (10, 0.15), (15, 0.2)]:
                    p   = attn_knn_loo(X_umap_geo, file_labels, k=k, T=T)
                    auc = macro_auc(file_labels, p)
                    nm  = f'umap{n_comp}_nn{n_neighbors}_k{k}_T{T}'
                    marker = " ← NEW BEST" if auc > best_so_far else ""
                    if auc > best_so_far:
                        best_so_far = auc
                        best_umap   = (nm, auc, p.copy(), reducer, X_umap.std(0)+1e-6)
                    if auc > BEST_NL - 0.005:
                        print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
                    results[nm] = auc
            except Exception as ex:
                print(f"  UMAP n_comp={n_comp} nn={n_neighbors} failed: {ex}", flush=True)

print(f"After E, best so far: {best_so_far:.4f}\n", flush=True)

# ── SUMMARY ────────────────────────────────────────────────────────────────
print("="*60)
print("SUMMARY (top 15 no-logit methods)")
print("="*60)
for nm, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    marker = " ← NEW BEST" if auc > BEST_NL else ""
    print(f"  {nm:<60s}  {auc:.4f}  {auc-BEST_NL:+.4f}{marker}")

global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\n整體最佳: {global_best_name} = {global_best_auc:.4f}", flush=True)

# ── Update results.json ────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best_nl = data.get('best_nologit', {}).get('loo_auc', BEST_NL)
new_nl_best = False

for nm, auc in results.items():
    note = 'nologit_new_methods'
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': note})
    if auc > cur_best_nl:
        cur_best_nl = auc
        data['best_nologit'] = {'method': nm, 'loo_auc': round(auc, 6),
                                 'note': 'no_logit NEW BEST'}
        new_nl_best = True

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\n已更新 embed_prior_results.json")
print(f"no-logit best: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")

if new_nl_best:
    print(f"\nNEW no-logit BEST: {global_best_name} = {global_best_auc:.4f}")
    print("→ 請執行 build_best_nologit_pkl.py 來儲存 pkl 並建立 notebook")

print("done", flush=True)
