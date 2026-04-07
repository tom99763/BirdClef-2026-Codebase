"""
8個進階 no-logit 方法（來自文獻搜尋）：
1. TIP-Adapter 風格軟快取（全局 softmax，非 top-k）
2. Ledoit-Wolf Mahalanobis（修正奇異矩陣問題）
3. Label Co-occurrence GCN（物種共現圖傳播）
4. Geo-temporal prior table（site × month 條件頻率表）
5. SGC 圖平滑嵌入 + Attn-KNN
6. KDE 每物種核密度估計分類器
7. Partial Aggregation Prototype（多標籤原型精煉）
8. TIP-Adapter + 現有 Attn-KNN ensemble

No-logit best to beat: 0.8796
"""
import numpy as np, json, pickle, re, os, shutil
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
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

file_embs  = np.zeros((n_files, emb_win.shape[1]), np.float32)
file_labels= np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]   = emb_win[s:e].mean(0)
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)

# ── Geo features ────────────────────────────────────────────────────────────
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
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24),
                       np.cos(2*np.pi*file_hours/24)], 1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12),
                       np.cos(2*np.pi*(file_months-1)/12)], 1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365),
                       np.cos(2*np.pi*(file_days-1)/365)], 1).astype(np.float32)
geo_all   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], 1).astype(np.float32)

# Standard pca24+day combined space
pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
pca24_std = X24.std(0) + 1e-6
X24s  = X24 / pca24_std
X_comb = np.concatenate([X24s, geo_all], 1).astype(np.float32)
X_nl   = (X_comb / np.linalg.norm(X_comb, axis=1, keepdims=True)).astype(np.float32)

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

EPS     = 1e-7
BEST_NL = 0.8796
results = {}
best_so_far = BEST_NL

print(f"Files={n_files}, species={n_species}")
print(f"No-logit best to beat: {BEST_NL:.4f}\n", flush=True)

# ── 預計算 Attn-KNN baseline ────────────────────────────────────────────────
y_attn = attn_knn_loo(X_nl, k=10, T=0.2)
print(f"Baseline attn_pca24+day k10 T0.2: {macro_auc(file_labels, y_attn):.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1) TIP-Adapter：全局 softmax（β 掃描，非硬性 top-k）
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("1) TIP-Adapter: exp(-β·d) soft cache over ALL training files")
print("="*60, flush=True)

def tip_adapter_loo(X, beta=10.0):
    """TIP-Adapter: score = sum_j exp(-β·(1-sim(q,k_j))) · v_j / Z"""
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()          # cosine sims
        dists = 1.0 - sims                           # cosine distance
        w = np.exp(-beta * dists)
        w /= (w.sum() + 1e-8)
        preds[i] = (w[:, None] * file_labels[tr]).sum(0)
    return preds

for beta in [2, 5, 10, 20, 40, 80, 160]:
    p   = tip_adapter_loo(X_nl, beta=beta)
    auc = macro_auc(file_labels, p)
    nm  = f'tip_adapter_b{beta}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    print(f"  beta={beta}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

# Also try on raw embedding space (no PCA)
for beta in [5, 10, 20]:
    p   = tip_adapter_loo(file_embs_norm, beta=beta)
    auc = macro_auc(file_labels, p)
    nm  = f'tip_adapter_raw_b{beta}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    if auc > BEST_NL - 0.005:
        print(f"  raw beta={beta}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

print(f"After 1, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2) Ledoit-Wolf Mahalanobis KNN
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("2) Ledoit-Wolf Mahalanobis KNN (proper shrinkage)")
print("="*60, flush=True)

for pca_d in [16, 24, 32]:
    pcaX = PCA(n_components=pca_d, random_state=42).fit(file_embs_norm)
    Xd   = pcaX.transform(file_embs_norm).astype(np.float32)

    # Ledoit-Wolf shrinkage covariance
    lw = LedoitWolf(assume_centered=False)
    lw.fit(Xd)
    prec = lw.precision_.astype(np.float32)  # (d, d) precision matrix

    for k in [3, 5, 7, 10]:
        preds = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr  = np.array([j for j in range(n_files) if j != i])
            diff = Xd[tr] - Xd[i]                          # (65, d)
            md2  = np.einsum('nd,dd,nd->n', diff, prec, diff)
            md   = np.sqrt(md2.clip(0) + 1e-8)
            top  = np.argsort(md)[:k]
            w    = 1.0 / (md[top] + 1e-6); w /= w.sum()
            preds[i] = (w[:, None] * file_labels[tr[top]]).sum(0)
        auc = macro_auc(file_labels, preds)
        nm  = f'lw_mahal_pca{pca_d}_k{k}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST_NL - 0.01:
            print(f"  pca{pca_d} k={k}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

    # Blend LW-Mahal with attn-KNN
    lw_preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr  = np.array([j for j in range(n_files) if j != i])
        diff = Xd[tr] - Xd[i]
        md2  = np.einsum('nd,dd,nd->n', diff, prec, diff)
        md   = np.sqrt(md2.clip(0) + 1e-8)
        top  = np.argsort(md)[:5]
        w    = 1.0 / (md[top] + 1e-6); w /= w.sum()
        lw_preds[i] = (w[:, None] * file_labels[tr[top]]).sum(0)
    for wa in [0.7, 0.8, 0.85, 0.9]:
        blend = wa * y_attn + (1-wa) * lw_preds
        auc   = macro_auc(file_labels, blend)
        nm    = f'lw_mahal_pca{pca_d}_attn_blend_wa{wa:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST_NL - 0.003:
            print(f"  pca{pca_d} attn+lw wa={wa:.2f}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After 2, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3) Label Co-occurrence 後處理（物種共現圖平滑）
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("3) Label Co-occurrence post-processing on KNN predictions")
print("="*60, flush=True)

# 計算全局共現矩陣（LOO：每次排除一個檔案後計算）
# 為了避免 leakage，使用 LOO 共現矩阣
# 簡化：用全部66個檔案的共現（LOO 時用65個）
cooccur_global = (file_labels.T @ file_labels)  # (234, 234)
np.fill_diagonal(cooccur_global, 0)
# 正規化：每行除以各物種出現次數
counts = file_labels.sum(0)  # (234,)
cooccur_norm = cooccur_global / (counts[:, None] + 1e-8)  # P(co | species_i present)

def cooccur_smooth_loo(base_preds, alpha=0.1):
    """後處理：base + alpha * C @ base"""
    preds = np.zeros_like(base_preds)
    for i in range(n_files):
        # LOO 共現（排除 file i）
        tr = [j for j in range(n_files) if j != i]
        C  = (file_labels[tr].T @ file_labels[tr])
        np.fill_diagonal(C, 0)
        cnt = file_labels[tr].sum(0)
        C_n = C / (cnt[:, None] + 1e-8)
        p = base_preds[i]
        preds[i] = p + alpha * (C_n @ p)
    return preds.clip(0, 1)

for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
    p   = cooccur_smooth_loo(y_attn, alpha=alpha)
    auc = macro_auc(file_labels, p)
    nm  = f'cooccur_smooth_a{alpha:.2f}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    if auc > BEST_NL - 0.003:
        print(f"  alpha={alpha:.2f}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

print(f"After 3, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4) Geo-temporal Prior Table（site × month 條件頻率）
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("4) Geo-temporal prior table: P(species | site, month)")
print("="*60, flush=True)

def geo_prior_loo(base_preds, geo_weight=1.0):
    """LOO geo-temporal prior: base × prior^geo_weight (log-space addition)."""
    preds = np.zeros_like(base_preds)
    for i in range(n_files):
        tr = [j for j in range(n_files) if j != i]
        # site × month prior from training files
        site_i  = int(file_sites[i])
        month_i = int(file_months[i])
        # P(species | site=s, month=m) = count(species in site_s, month_m files) / count(site_s, month_m files)
        mask_sm = [(j for j in tr if file_sites[j] == site_i and file_months[j] == month_i)]
        sm_files = [j for j in tr if file_sites[j] == site_i and file_months[j] == month_i]
        if len(sm_files) >= 1:
            prior = file_labels[sm_files].mean(0) + 0.05  # Laplace smoothing
        else:
            # Fall back to site-only prior
            s_files = [j for j in tr if file_sites[j] == site_i]
            if len(s_files) >= 1:
                prior = file_labels[s_files].mean(0) + 0.05
            else:
                prior = file_labels[tr].mean(0) + 0.05  # global

        prior = prior.clip(EPS, 1-EPS)
        log_base  = np.log(base_preds[i].clip(EPS, 1-EPS))
        log_prior = np.log(prior)
        log_fused = log_base + geo_weight * log_prior
        # Normalise back to [0,1] via softmax rescale (subtract global mean)
        log_fused -= log_fused.mean()
        preds[i] = 1.0 / (1.0 + np.exp(-log_fused))
    return preds.clip(EPS, 1-EPS)

for gw in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    p   = geo_prior_loo(y_attn, geo_weight=gw)
    auc = macro_auc(file_labels, p)
    nm  = f'geo_prior_gw{gw:.1f}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    if auc > BEST_NL - 0.003:
        print(f"  gw={gw:.1f}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

print(f"After 4, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5) SGC 圖平滑嵌入（Simple Graph Convolution）+ Attn-KNN
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("5) SGC graph-smoothed embeddings + Attn-KNN")
print("="*60, flush=True)

def sgc_smooth(X_feat, X_graph, k_graph=5, n_hops=1, alpha=0.5):
    """
    X_feat: 特徵空間 (用於 KNN 特徵)
    X_graph: 圖空間 (用於建立親和力圖)
    Graph: kNN on X_graph, then D^-1 A X_feat propagation
    """
    # 建立 kNN 親和力矩陣
    sims = X_graph @ X_graph.T                          # (n, n)
    np.fill_diagonal(sims, 0)
    A = np.zeros_like(sims)
    for i in range(len(X_graph)):
        top = np.argsort(-sims[i])[:k_graph]
        A[i, top] = sims[i, top].clip(0)
    # 列正規化
    row_sum = A.sum(1, keepdims=True)
    row_sum[row_sum < 1e-8] = 1e-8
    A_norm = A / row_sum

    # SGC: X_smooth = (1-alpha)*X + alpha * A_norm @ X (per hop)
    X_smooth = X_feat.copy()
    for _ in range(n_hops):
        X_smooth = (1 - alpha) * X_feat + alpha * (A_norm @ X_smooth)
    X_smooth_n = X_smooth / (np.linalg.norm(X_smooth, axis=1, keepdims=True) + 1e-8)
    return X_smooth_n.astype(np.float32)

for k_g in [5, 7, 10]:
    for n_hops in [1, 2]:
        for alpha in [0.3, 0.5, 0.7]:
            X_smooth = sgc_smooth(X_nl, file_embs_norm, k_graph=k_g, n_hops=n_hops, alpha=alpha)
            for k_knn, T in [(10, 0.2), (10, 0.15)]:
                p   = attn_knn_loo(X_smooth, k=k_knn, T=T)
                auc = macro_auc(file_labels, p)
                nm  = f'sgc_kg{k_g}_h{n_hops}_a{alpha:.1f}_k{k_knn}_T{T}'
                marker = " ← NEW BEST" if auc > best_so_far else ""
                if auc > best_so_far: best_so_far = auc
                if auc > BEST_NL - 0.003:
                    print(f"  kg={k_g} h={n_hops} a={alpha:.1f} k={k_knn}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
                results[nm] = auc

print(f"After 5, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6) KDE 每物種核密度估計分類器
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("6) KDE per-species classifier (PCA-8 space)")
print("="*60, flush=True)

for pca_d in [6, 8, 12]:
    pcaK = PCA(n_components=pca_d, random_state=42).fit(file_embs_norm)
    Xk   = pcaK.transform(file_embs_norm).astype(np.float32)
    Xk  /= (Xk.std(0) + 1e-6)

    for bw in [0.5, 1.0, 1.5]:
        active = np.where(file_labels.sum(0) >= 2)[0]
        kde_preds = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr = [j for j in range(n_files) if j != i]
            x_q = Xk[[i]]
            for s in active:
                pos = [j for j in tr if file_labels[j, s] > 0.5]
                neg = [j for j in tr if file_labels[j, s] <= 0.5]
                if len(pos) < 1:
                    kde_preds[i, s] = 0.0; continue
                kde_pos = KernelDensity(bandwidth=bw, kernel='gaussian')
                kde_pos.fit(Xk[pos])
                log_pos = kde_pos.score_samples(x_q)[0]
                if len(neg) >= 2:
                    kde_neg = KernelDensity(bandwidth=bw, kernel='gaussian')
                    kde_neg.fit(Xk[neg])
                    log_neg = kde_neg.score_samples(x_q)[0]
                else:
                    log_neg = 0.0
                kde_preds[i, s] = float(1.0 / (1.0 + np.exp(-(log_pos - log_neg))))
        auc = macro_auc(file_labels, kde_preds)
        nm  = f'kde_pca{pca_d}_bw{bw}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST_NL - 0.01:
            print(f"  pca{pca_d} bw={bw}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After 6, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 7) Partial Aggregation Prototype（選取最相關的 k 個正樣本建立原型）
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("7) Partial Aggregation Prototype (anti-label-pollution)")
print("="*60, flush=True)

def partial_agg_proto_loo(X, k_proto=3):
    """
    對每個物種，只用與 query 最相似的 top-k 正樣本建立原型。
    避免其他標籤「污染」原型。
    """
    preds = np.zeros((n_files, n_species), np.float32)
    active = np.where(file_labels.sum(0) >= 2)[0]
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims_all = (X[[i]] @ X[tr].T).ravel()   # (65,)
        for s in active:
            pos_mask = file_labels[tr, s] > 0.5
            pos_idx  = np.where(pos_mask)[0]     # indices within tr
            if len(pos_idx) == 0:
                preds[i, s] = 0.0; continue
            # Select top-k positive samples most similar to query
            pos_sims = sims_all[pos_idx]
            top_pos  = np.argsort(-pos_sims)[:min(k_proto, len(pos_idx))]
            # Build prototype from selected positives
            proto = X[tr[pos_idx[top_pos]]].mean(0)
            proto /= (np.linalg.norm(proto) + 1e-8)
            preds[i, s] = float((X[i] @ proto).clip(0))
    return preds

for k_p in [1, 2, 3, 5]:
    p   = partial_agg_proto_loo(X_nl, k_proto=k_p)
    auc = macro_auc(file_labels, p)
    nm  = f'partial_proto_k{k_p}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    if auc > BEST_NL - 0.01:
        print(f"  k_proto={k_p}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

    # Blend with attn
    for wa in [0.7, 0.8]:
        blend = wa * y_attn + (1-wa) * p
        auc   = macro_auc(file_labels, blend)
        nm_b  = f'partial_proto_k{k_p}_attn{wa:.1f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST_NL - 0.005:
            print(f"  k_proto={k_p} attn_wa={wa}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm_b] = auc

print(f"After 7, best so far: {best_so_far:.4f}\n", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 8) TIP-Adapter + Attn-KNN ensemble + window_knn_k3 三路 ensemble
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("8) TIP-Adapter + Attn-KNN + Window-KNN 三路 ensemble")
print("="*60, flush=True)

# Best TIP-Adapter
best_tip_beta = max([(b, results.get(f'tip_adapter_b{b}', 0)) for b in [2,5,10,20,40,80,160]],
                     key=lambda x: x[1])[0]
y_tip = tip_adapter_loo(X_nl, beta=best_tip_beta)
print(f"  Best TIP-Adapter beta={best_tip_beta}: {macro_auc(file_labels, y_tip):.4f}", flush=True)

# Window KNN k=3
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

y_win3 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = win_file_id != i
    X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
    Y_tr  = file_labels[win_file_id[tr_wi]]
    sims  = X_te @ X_tr.T; top = np.argsort(-sims, axis=1)[:, :3]
    wp = np.zeros((te_e - te_s, n_species), np.float32)
    for wi in range(te_e - te_s):
        w = sims[wi, top[wi]].clip(0); ws = w.sum()
        w = w/ws if ws > 1e-8 else np.ones(3)/3
        wp[wi] = (w[:, None] * Y_tr[top[wi]]).sum(0)
    y_win3[i] = wp.mean(0)

for w1, w2, w3 in [
    (0.50, 0.30, 0.20), (0.55, 0.25, 0.20), (0.60, 0.20, 0.20),
    (0.60, 0.25, 0.15), (0.65, 0.20, 0.15), (0.70, 0.20, 0.10),
    (0.50, 0.35, 0.15), (0.55, 0.30, 0.15), (0.60, 0.30, 0.10),
]:
    blend = w1 * y_attn + w2 * y_tip + w3 * y_win3
    auc   = macro_auc(file_labels, blend)
    nm    = f'ens3_attn{w1:.2f}_tip{w2:.2f}_win{w3:.2f}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    if auc > BEST_NL - 0.003:
        print(f"  attn={w1:.2f} tip={w2:.2f} win={w3:.2f}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

# 2-way TIP+Attn
for wa in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
    blend = wa * y_attn + (1-wa) * y_tip
    auc   = macro_auc(file_labels, blend)
    nm    = f'ens2_attn{wa:.2f}_tip{1-wa:.2f}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    if auc > BEST_NL - 0.003:
        print(f"  attn={wa:.2f} tip={1-wa:.2f}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

print(f"After 8, best so far: {best_so_far:.4f}\n", flush=True)

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

# ── Update results.json ────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best_nl = data.get('best_nologit', {}).get('loo_auc', BEST_NL)
new_best_info = None
for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': 'nologit_advanced'})
    if auc > cur_best_nl:
        cur_best_nl = auc
        data['best_nologit'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'no_logit NEW BEST'}
        new_best_info = (nm, auc)

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\nno-logit best: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")
if new_best_info:
    print(f"NEW BEST: {new_best_info[0]} = {new_best_info[1]:.4f}")
print("done", flush=True)
