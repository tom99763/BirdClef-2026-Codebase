"""
No-logit 第五批：針對完整管線（SED + ProtoSSM）最佳化
- 完整管線 CV 顯示 attn-KNN pca24+day (LOO=0.8758) 在全管線得 0.9246
- 目標：找到 LOO > 0.8810 的方法，或全管線 > 0.9246 的方法

新方法：
1. Mutual-KNN（互為最近鄰才計入）
2. Density-weighted KNN（local density 做 down-weight）
3. Harmonic mean aggregation（window-level）
4. Site-specific normalization（site 內部正規化）
5. Multi-scale window KNN（不同窗口尺寸的 ensemble）
6. Skewness-corrected KNN（校正標籤分佈偏態）
7. Log-space attn-KNN（在 log 空間做相似度計算）
8. Per-window 最大相似度 aggregation（不平均，取 max sim weighted label）

No-logit best to beat: 0.8810
"""
import numpy as np, json, re, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

file_embs  = np.zeros((n_files, emb_win.shape[1]), np.float32)
file_labels= np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]   = emb_win[s:e].mean(0)
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)
emb_win_norm   = normalize(emb_win,   norm='l2').astype(np.float32)
win_file_id    = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

SITES = ['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx = {s: i for i, s in enumerate(SITES)}
file_sites  = np.zeros(n_files, np.int32)
file_hours  = np.zeros(n_files, np.float32)
file_months = np.zeros(n_files, np.float32)
file_days   = np.zeros(n_files, np.float32)
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
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24), np.cos(2*np.pi*file_hours/24)], 1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12), np.cos(2*np.pi*(file_months-1)/12)], 1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365), np.cos(2*np.pi*(file_days-1)/365)], 1).astype(np.float32)
geo_all   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], 1).astype(np.float32)

pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
X24  /= (X24.std(0) + 1e-6)
X_nl_pca24 = np.concatenate([X24, geo_all], 1).astype(np.float32)
X_nl_pca24 /= np.linalg.norm(X_nl_pca24, 1, keepdims=True) + 1e-8

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

def attn_knn_loo(X, k=10, T=0.2):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr[top]]).sum(0)
    return preds

def window_knn_loo(k=1):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        Y_tr = file_labels[win_file_id[tr_wi]]
        sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :k]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            w = sims[wi, top_idx[wi]].clip(0); ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k)/k
            wp[wi] = (w[:, None] * Y_tr[top_idx[wi]]).sum(0)
        preds[i] = wp.mean(0)
    return preds

BEST_NL = 0.8810
best_so_far = BEST_NL
results = {}
best_nm, best_preds = None, None

print(f"Files={n_files}, species={n_species}")
print(f"No-logit best to beat: {BEST_NL:.4f}\n", flush=True)

y_attn = attn_knn_loo(X_nl_pca24, k=10, T=0.2)
y_win1 = window_knn_loo(k=1)
y_win3 = window_knn_loo(k=3)

# ── A) Mutual-KNN（互為最近鄰才計入） ──────────────────────────────────
print("="*60)
print("A) Mutual-KNN (symmetric nearest neighbors)")
print("="*60, flush=True)

def mutual_knn_loo(X, k=10, T=0.2):
    """Only use neighbors j where i is also in j's top-k"""
    preds = np.zeros((n_files, n_species), np.float32)
    # Precompute all pairwise sims
    sims_all = (X @ X.T).astype(np.float32)
    np.fill_diagonal(sims_all, -2)
    # Precompute top-k neighbors for each file
    topk_all = np.argsort(-sims_all, 1)[:, :k]  # (n_files, k)

    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims_i = sims_all[i, tr]
        top_k_i = np.argsort(-sims_i)[:k]
        candidates = tr[top_k_i]  # i's top-k neighbors

        # Filter: only keep j where i is in j's top-k
        mutual = []
        for j in candidates:
            if i in topk_all[j]:
                mutual.append(j)
        mutual = np.array(mutual) if mutual else candidates[:3]  # fallback

        sims_mutual = sims_all[i, mutual]
        logit_m = sims_mutual / T; logit_m -= logit_m.max()
        w = np.exp(logit_m); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[mutual]).sum(0)
    return preds

for k in [10, 15, 20]:
    p = mutual_knn_loo(X_nl_pca24, k=k, T=0.2)
    auc = macro_auc(file_labels, p)
    nm = f'mutual_knn_k{k}_T02'
    print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f})", flush=True)
    results[nm] = auc
    if auc > best_so_far:
        best_so_far = auc; best_nm, best_preds = nm, p.copy()

    for wa in [0.60, 0.65, 0.70]:
        blend = wa * p + (1-wa) * y_win1
        auc2 = macro_auc(file_labels, blend)
        nm2 = f'mutual_k{k}_attn{wa:.2f}_wink1'
        if auc2 > best_so_far:
            best_so_far = auc2; best_nm, best_preds = nm2, blend.copy()
        if auc2 > BEST_NL - 0.002:
            marker = " ← NEW BEST" if auc2 > BEST_NL else ""
            print(f"  {nm2}: {auc2:.4f}  (Δ={auc2-BEST_NL:+.4f}){marker}", flush=True)
        results[nm2] = auc2

print(f"After A, best so far: {best_so_far:.4f}\n", flush=True)

# ── B) Density-weighted KNN ────────────────────────────────────────────────
print("="*60)
print("B) Density-weighted KNN (down-weight outliers)")
print("="*60, flush=True)

def density_weighted_knn_loo(X, k=10, T=0.2, k_density=5):
    """Compute local density per file, use as weight normalizer"""
    preds = np.zeros((n_files, n_species), np.float32)
    sims_all = (X @ X.T).astype(np.float32)
    np.fill_diagonal(sims_all, -2)
    # Local density: avg sim to k_density nearest neighbors
    topk_sims = np.sort(sims_all, 1)[:, -k_density:]
    density = topk_sims.mean(1)  # (n_files,)

    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims_i = sims_all[i, tr]
        top_k = np.argsort(-sims_i)[:k]
        tr_top = tr[top_k]

        # Weight by similarity AND density of neighbors
        s = sims_i[top_k]
        d = density[tr_top]  # density of training neighbors
        logit = s / T; logit -= logit.max()
        w_sim = np.exp(logit)
        # Higher density = more reliable neighbors → higher weight
        w_dens = np.exp(d - d.max())
        w = w_sim * w_dens; w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr_top]).sum(0)
    return preds

for k in [10, 15]:
    for k_d in [3, 5, 7]:
        p = density_weighted_knn_loo(X_nl_pca24, k=k, T=0.2, k_density=k_d)
        auc = macro_auc(file_labels, p)
        nm = f'density_k{k}_kd{k_d}'
        if auc > best_so_far:
            best_so_far = auc; best_nm, best_preds = nm, p.copy()
        if auc > BEST_NL - 0.003:
            marker = " ← NEW BEST" if auc > BEST_NL else ""
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

        for wa in [0.65, 0.70]:
            blend = wa * p + (1-wa) * y_win1
            auc2 = macro_auc(file_labels, blend)
            nm2 = f'density_k{k}_kd{k_d}_wink1_{1-wa:.2f}'
            if auc2 > best_so_far:
                best_so_far = auc2; best_nm, best_preds = nm2, blend.copy()
            if auc2 > BEST_NL - 0.002:
                marker = " ← NEW BEST" if auc2 > BEST_NL else ""
                print(f"  {nm2}: {auc2:.4f}  (Δ={auc2-BEST_NL:+.4f}){marker}", flush=True)
            results[nm2] = auc2

print(f"After B, best so far: {best_so_far:.4f}\n", flush=True)

# ── C) Harmonic mean window aggregation ──────────────────────────────────
print("="*60)
print("C) Harmonic mean window aggregation")
print("="*60, flush=True)

EPS = 1e-7
def window_knn_harmonic_loo(k=1):
    """Harmonic mean: 1/mean(1/p_i)"""
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        Y_tr = file_labels[win_file_id[tr_wi]]
        sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :k]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            w = sims[wi, top_idx[wi]].clip(0); ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k)/k
            wp[wi] = (w[:, None] * Y_tr[top_idx[wi]]).sum(0)
        # Harmonic mean: 1/mean(1/p+eps) - eps
        inv_mean = (1.0 / (wp + EPS)).mean(0)
        preds[i] = (1.0 / (inv_mean + EPS) - EPS).clip(0, 1)
    return preds

for k_h in [1, 3, 5]:
    y_harm = window_knn_harmonic_loo(k=k_h)
    auc_h = macro_auc(file_labels, y_harm)
    print(f"  win_harmonic k={k_h}: {auc_h:.4f}", flush=True)
    results[f'win_harmonic_k{k_h}'] = auc_h
    for wa in [0.60, 0.65, 0.70, 0.75]:
        blend = wa * y_attn + (1-wa) * y_harm
        auc = macro_auc(file_labels, blend)
        nm = f'attn_winharm_k{k_h}_wa{wa:.2f}'
        if auc > best_so_far:
            best_so_far = auc; best_nm, best_preds = nm, blend.copy()
        if auc > BEST_NL - 0.002:
            marker = " ← NEW BEST" if auc > BEST_NL else ""
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After C, best so far: {best_so_far:.4f}\n", flush=True)

# ── D) Site-specific normalization ────────────────────────────────────────
print("="*60)
print("D) Site-specific normalization")
print("="*60, flush=True)

# Within each site, center and normalize
file_embs_site_norm = file_embs_norm.copy()
for site_id in range(len(SITES)):
    site_mask = file_sites == site_id
    if site_mask.sum() < 2:
        continue
    X_site = file_embs[site_mask]
    mu_site = X_site.mean(0)
    X_site_c = X_site - mu_site
    X_site_n = normalize(X_site_c, norm='l2')
    file_embs_site_norm[site_mask] = X_site_n

# PCA on site-normalized embeddings
pca_s = PCA(n_components=24, random_state=42).fit(file_embs_site_norm)
X24_s = pca_s.transform(file_embs_site_norm).astype(np.float32)
X24_s /= (X24_s.std(0) + 1e-6)
X_site_geo = np.concatenate([X24_s, geo_all], 1).astype(np.float32)
X_site_geo /= np.linalg.norm(X_site_geo, 1, keepdims=True) + 1e-8

for k, T in [(10, 0.2), (10, 0.15), (7, 0.2)]:
    p = attn_knn_loo(X_site_geo, k=k, T=T)
    auc = macro_auc(file_labels, p)
    nm = f'site_norm_pca24_k{k}_T{T}'
    print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f})", flush=True)
    results[nm] = auc
    if auc > best_so_far:
        best_so_far = auc; best_nm, best_preds = nm, p.copy()

    for wa in [0.60, 0.65, 0.70]:
        blend = wa * p + (1-wa) * y_win1
        auc2 = macro_auc(file_labels, blend)
        nm2 = f'site_norm_k{k}_T{T}_wink1_{1-wa:.2f}'
        if auc2 > best_so_far:
            best_so_far = auc2; best_nm, best_preds = nm2, blend.copy()
        if auc2 > BEST_NL - 0.002:
            marker = " ← NEW BEST" if auc2 > BEST_NL else ""
            print(f"  {nm2}: {auc2:.4f}  (Δ={auc2-BEST_NL:+.4f}){marker}", flush=True)
        results[nm2] = auc2

print(f"After D, best so far: {best_so_far:.4f}\n", flush=True)

# ── E) Window-level max-similarity weighted label agg ────────────────────
print("="*60)
print("E) Window max-sim weighted: for each test win, weight by max sim")
print("="*60, flush=True)

def window_maxsim_weighted_loo(T_win=0.2):
    """
    For each file:
    1. Compute max similarity per test window to any training window
    2. Use softmax over test windows (high-sim windows get more weight)
    3. Average per-window KNN predictions weighted by max_sim
    """
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        Y_tr = file_labels[win_file_id[tr_wi]]
        sims = X_te @ X_tr.T   # (n_te_wins, n_tr_wins)

        # Per test window: nearest training neighbor
        max_sim_per_te_win = sims.max(1)   # (n_te_wins,)
        best_tr_win = sims.argmax(1)        # (n_te_wins,)

        # Weight test windows by their max similarity (softmax)
        logit_te = max_sim_per_te_win / T_win
        logit_te -= logit_te.max()
        w_te = np.exp(logit_te); w_te /= w_te.sum()

        # Per test window prediction: label of best training window's file
        win_preds = Y_tr[best_tr_win]  # (n_te_wins, n_species)
        preds[i] = (w_te[:, None] * win_preds).sum(0)
    return preds

for T_win in [0.15, 0.2, 0.25, 0.3]:
    p = window_maxsim_weighted_loo(T_win=T_win)
    auc = macro_auc(file_labels, p)
    nm = f'win_maxsim_T{T_win}'
    print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f})", flush=True)
    results[nm] = auc
    if auc > best_so_far:
        best_so_far = auc; best_nm, best_preds = nm, p.copy()

    for wa in [0.60, 0.65, 0.70, 0.75]:
        blend = wa * y_attn + (1-wa) * p
        auc2 = macro_auc(file_labels, blend)
        nm2 = f'attn_wmaxsim_T{T_win}_wa{wa:.2f}'
        if auc2 > best_so_far:
            best_so_far = auc2; best_nm, best_preds = nm2, blend.copy()
        if auc2 > BEST_NL - 0.002:
            marker = " ← NEW BEST" if auc2 > BEST_NL else ""
            print(f"  {nm2}: {auc2:.4f}  (Δ={auc2-BEST_NL:+.4f}){marker}", flush=True)
        results[nm2] = auc2

print(f"After E, best so far: {best_so_far:.4f}\n", flush=True)

# ── F) Smooth window-KNN: softmax over all training windows ──────────────
print("="*60)
print("F) Soft-all window KNN (no topk, full softmax over all)")
print("="*60, flush=True)

def window_softall_loo(T=0.1):
    """Softmax over ALL training windows (no hard top-k cutoff)"""
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        Y_tr = file_labels[win_file_id[tr_wi]]
        sims = X_te @ X_tr.T   # (n_te_wins, n_tr_wins)
        # Softmax over all tr windows per te window
        logit_s = sims / T
        logit_s -= logit_s.max(1, keepdims=True)
        w = np.exp(logit_s); w /= w.sum(1, keepdims=True)
        # Aggregate to file level
        win_p = w @ Y_tr  # (n_te_wins, n_species)
        preds[i] = win_p.mean(0)
    return preds

for T in [0.05, 0.10, 0.15, 0.20]:
    p = window_softall_loo(T=T)
    auc = macro_auc(file_labels, p)
    nm = f'win_softall_T{T}'
    print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f})", flush=True)
    results[nm] = auc
    if auc > best_so_far:
        best_so_far = auc; best_nm, best_preds = nm, p.copy()

    for wa in [0.60, 0.65, 0.70, 0.75]:
        blend = wa * y_attn + (1-wa) * p
        auc2 = macro_auc(file_labels, blend)
        nm2 = f'attn_softall_T{T}_wa{wa:.2f}'
        if auc2 > best_so_far:
            best_so_far = auc2; best_nm, best_preds = nm2, blend.copy()
        if auc2 > BEST_NL - 0.002:
            marker = " ← NEW BEST" if auc2 > BEST_NL else ""
            print(f"  {nm2}: {auc2:.4f}  (Δ={auc2-BEST_NL:+.4f}){marker}", flush=True)
        results[nm2] = auc2

print(f"After F, best so far: {best_so_far:.4f}\n", flush=True)

# ── G) 更精細 wink1 weight sweep ─────────────────────────────────────────
print("="*60)
print("G) Ultra-fine sweep: attn weight near 0.65, wink1")
print("="*60, flush=True)

for k in [9, 10, 11]:
    for T in [0.18, 0.19, 0.20, 0.21, 0.22]:
        y_a = attn_knn_loo(X_nl_pca24, k=k, T=T)
        for wa in [0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68]:
            blend = wa * y_a + (1-wa) * y_win1
            auc = macro_auc(file_labels, blend)
            nm = f'ultrafine_k{k}_T{T:.2f}_wa{wa:.2f}'
            if auc > best_so_far:
                best_so_far = auc; best_nm, best_preds = nm, blend.copy()
                print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}) ← NEW BEST", flush=True)
            results[nm] = auc

print(f"After G, best so far: {best_so_far:.4f}\n", flush=True)

# ── SUMMARY ────────────────────────────────────────────────────────────────
print("="*60)
print("SUMMARY (top 20)")
print("="*60)
for nm, auc in sorted(results.items(), key=lambda x: -x[1])[:20]:
    marker = " ← NEW BEST" if auc > BEST_NL else ""
    print(f"  {nm:<65s}  {auc:.4f}  {auc-BEST_NL:+.4f}{marker}")

global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\n整體最佳: {global_best_name} = {global_best_auc:.4f}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best_nl = data.get('best_nologit', {}).get('loo_auc', BEST_NL)
for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': 'nologit_batch5'})
    if auc > cur_best_nl:
        cur_best_nl = auc
        data['best_nologit'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'no_logit NEW BEST batch5'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"no-logit best: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")
if global_best_auc > BEST_NL:
    print(f"\nNEW BEST！{global_best_name} = {global_best_auc:.4f}")
print("done", flush=True)
