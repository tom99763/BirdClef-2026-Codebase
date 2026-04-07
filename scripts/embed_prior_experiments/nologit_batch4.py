"""
No-logit 第四批：ZCA Whitening（真正的 ZCA）、TIM、Adaptive 融合
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
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24),
                       np.cos(2*np.pi*file_hours/24)], 1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12),
                       np.cos(2*np.pi*(file_months-1)/12)], 1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365),
                       np.cos(2*np.pi*(file_days-1)/365)], 1).astype(np.float32)
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
        sims = X_te @ X_tr.T; top = np.argsort(-sims, 1)[:, :k]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            w = sims[wi, top[wi]].clip(0); ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k)/k
            wp[wi] = (w[:, None] * Y_tr[top[wi]]).sum(0)
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

# ── A) True ZCA Whitening ─────────────────────────────────────────────────
print("="*60)
print("A) True ZCA Whitening (full covariance → W_zca)")
print("="*60, flush=True)

def zca_whiten(X, eps=1e-5):
    """ZCA whitening: W = U @ diag(1/sqrt(S+eps)) @ U.T"""
    X = X.astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    # Use PCA-based ZCA (equivalent but more stable)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt.T @ np.diag(1.0 / np.sqrt(S**2 / len(X) + eps)) @ Vt
    return ((Xc @ W) + mu @ W).astype(np.float32)

# ZCA on raw embeddings (too high dimensional—use PCA reduction first)
for pca_d in [32, 48, 64, 96]:
    try:
        pca_z = PCA(n_components=pca_d, random_state=42).fit(file_embs_norm)
        Xr = pca_z.transform(file_embs_norm).astype(np.float32)
        Xw = zca_whiten(Xr, eps=1e-3)
        Xw_n = normalize(Xw, norm='l2').astype(np.float32)
        # with geo
        Xwg = np.concatenate([Xw_n, geo_all], 1).astype(np.float32)
        Xwg /= np.linalg.norm(Xwg, 1, keepdims=True) + 1e-8

        for k, T in [(10, 0.2), (10, 0.15), (7, 0.2)]:
            p = attn_knn_loo(Xwg, k=k, T=T)
            auc = macro_auc(file_labels, p)
            nm = f'zca_pca{pca_d}_k{k}_T{T}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, p.copy()
            if auc > BEST_NL - 0.005:
                print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm] = auc

        # Ensemble with win_k1
        y_z = attn_knn_loo(Xwg, k=10, T=0.2)
        for wa in [0.60, 0.65, 0.70]:
            blend = wa * y_z + (1-wa) * y_win1
            auc = macro_auc(file_labels, blend)
            nm = f'zca_pca{pca_d}_ens_wa{wa:.2f}_wink1'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, blend.copy()
            if auc > BEST_NL - 0.003:
                print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm] = auc
    except Exception as ex:
        print(f"  ZCA pca{pca_d} failed: {ex}", flush=True)

print(f"After A, best so far: {best_so_far:.4f}\n", flush=True)

# ── B) TIM-ADM 簡化版（Transductive Info Maximization） ──────────────────
print("="*60)
print("B) TIM-ADM (Transductive Information Maximization) simplified")
print("="*60, flush=True)

def tim_adm_loo(X, k_init=10, T_init=0.2, n_iter=10, lam=0.5):
    """
    Simplified TIM for multi-label:
    Initialize with attn-KNN probs, then iterate:
      Maximize entropy of marginal distribution (diversity)
      Minimize entropy of per-sample distribution (confidence)
    Combined objective: H(ȳ) - H(Y|X) (maximize mutual information)
    """
    EPS = 1e-8
    preds = np.zeros((n_files, n_species), np.float32)

    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        # Initial probs via attn-KNN
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k_init]
        logit = sims[top] / T_init; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        q = (w[:, None] * file_labels[tr[top]]).sum(0)  # (n_species,)

        # TIM-style transductive refinement (per-species)
        # For multi-label, each species is independent binary problem
        # Maximize: H(q̄) - H(q) where q̄ = q (single sample = marginal itself)
        # So gradient = -(log q + log(1-q)) - marginal_entropy_grad
        # For single query: simply push toward extremes (confident prediction)
        for _ in range(n_iter):
            # Entropy minimization: push toward 0 or 1
            # H(q) = -q log q - (1-q) log(1-q)
            # dH/dq = -log(q) + log(1-q) → gradient descent on H
            # Simplified: q_new = sigmoid(lam * logit(q))
            q_clamped = q.clip(EPS, 1-EPS)
            logit_q = np.log(q_clamped) - np.log(1 - q_clamped)
            # Amplify by lam (push toward extremes)
            q = 1.0 / (1.0 + np.exp(-lam * logit_q))

        preds[i] = q
    return preds

for lam in [1.5, 2.0, 3.0, 5.0]:
    for n_iter in [5, 10, 20]:
        p = tim_adm_loo(X_nl_pca24, lam=lam, n_iter=n_iter)
        auc = macro_auc(file_labels, p)
        nm = f'tim_lam{lam:.1f}_n{n_iter}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, p.copy()
        if auc > BEST_NL - 0.005:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

        # Ensemble
        for wa in [0.55, 0.60, 0.65, 0.70]:
            blend = wa * p + (1-wa) * y_win1
            auc2 = macro_auc(file_labels, blend)
            nm2 = f'tim_lam{lam:.1f}_n{n_iter}_wink1_{1-wa:.2f}'
            marker2 = " ← NEW BEST" if auc2 > best_so_far else ""
            if auc2 > best_so_far:
                best_so_far = auc2
                best_nm, best_preds = nm2, blend.copy()
            if auc2 > BEST_NL - 0.003:
                print(f"  {nm2}: {auc2:.4f}  (Δ={auc2-BEST_NL:+.4f}){marker2}", flush=True)
            results[nm2] = auc2

print(f"After B, best so far: {best_so_far:.4f}\n", flush=True)

# ── C) Asymmetric L2N（只對 query normalize） ─────────────────────────────
print("="*60)
print("C) Asymmetric normalization: different query vs support")
print("="*60, flush=True)

def asym_knn_loo(X_all, X_query_n, X_support_n, k=10, T=0.2):
    """
    Query: use X_query_n (normalized)
    Support: use X_support_n (e.g., unnormalized or differently normalized)
    """
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X_query_n[[i]] @ X_support_n[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr[top]]).sum(0)
    return preds

# Raw embedding norm as query, pca24+geo as support
for T in [0.2, 0.25]:
    p = asym_knn_loo(None, file_embs_norm, X_nl_pca24, k=10, T=T)
    auc = macro_auc(file_labels, p)
    nm = f'asym_raw_q_pca24s_T{T}'
    if auc > BEST_NL - 0.005:
        print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f})", flush=True)
    results[nm] = auc
    if auc > best_so_far:
        best_so_far = auc; best_nm, best_preds = nm, p.copy()

# pca24+geo as query, raw norm as support
for T in [0.2, 0.25]:
    p = asym_knn_loo(None, X_nl_pca24, file_embs_norm, k=10, T=T)
    auc = macro_auc(file_labels, p)
    nm = f'asym_pca24q_raw_s_T{T}'
    if auc > BEST_NL - 0.005:
        print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f})", flush=True)
    results[nm] = auc
    if auc > best_so_far:
        best_so_far = auc; best_nm, best_preds = nm, p.copy()

print(f"After C, best so far: {best_so_far:.4f}\n", flush=True)

# ── D) Power Transform + CL2N combo ──────────────────────────────────────
print("="*60)
print("D) Power Transform + CL2N combo")
print("="*60, flush=True)

mu_all = file_embs.mean(0)
for beta in [0.4, 0.5, 0.6]:
    # Apply power transform to raw embeddings
    X_raw_clipped = file_embs.clip(0)  # non-negative (Perch embeddings might be mixed)
    X_pt = X_raw_clipped ** beta
    # CL2N: center and L2 normalize
    X_pt_c = X_pt - X_pt.mean(0)
    X_pt_n = normalize(X_pt_c, norm='l2').astype(np.float32)
    # PCA + geo
    pca_pt = PCA(n_components=24, random_state=42).fit(X_pt_n)
    X24_pt = pca_pt.transform(X_pt_n).astype(np.float32)
    X24_pt /= (X24_pt.std(0) + 1e-6)
    Xptg = np.concatenate([X24_pt, geo_all], 1).astype(np.float32)
    Xptg /= np.linalg.norm(Xptg, 1, keepdims=True) + 1e-8

    y_ptcl = attn_knn_loo(Xptg, k=10, T=0.2)
    auc_b = macro_auc(file_labels, y_ptcl)
    print(f"  PT_CL2N beta={beta}: {auc_b:.4f}", flush=True)

    for wa in [0.60, 0.65, 0.70]:
        blend = wa * y_ptcl + (1-wa) * y_win1
        auc = macro_auc(file_labels, blend)
        nm = f'ptcl2n_b{beta}_attn{wa:.2f}_wink1'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, blend.copy()
        if auc > BEST_NL - 0.002:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After D, best so far: {best_so_far:.4f}\n", flush=True)

# ── E) Window-level attn KNN (attn softmax on window sims) ───────────────
print("="*60)
print("E) Window-level Attn-KNN (softmax on window similarities)")
print("="*60, flush=True)

def window_attn_knn_loo(k=5, T=0.2):
    """
    For each test window, compute attn-KNN with SOFTMAX over similarities
    (instead of simple cosine-weighted mean)
    Then aggregate windows by mean.
    """
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        Y_tr = file_labels[win_file_id[tr_wi]]
        sims = X_te @ X_tr.T   # (n_test_wins, n_tr_wins)
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            top = np.argsort(-sims[wi])[:k]
            s_top = sims[wi, top]
            logit = s_top / T; logit -= logit.max()
            w = np.exp(logit); w /= w.sum()
            wp[wi] = (w[:, None] * Y_tr[top]).sum(0)
        preds[i] = wp.mean(0)
    return preds

for k, T in [(3, 0.2), (5, 0.2), (5, 0.15), (7, 0.2), (1, 0.2)]:
    y_wattn = window_attn_knn_loo(k=k, T=T)
    auc_w = macro_auc(file_labels, y_wattn)
    print(f"  win_attn k={k} T={T}: {auc_w:.4f}", flush=True)
    results[f'win_attn_k{k}_T{T}'] = auc_w

    for wa in [0.55, 0.60, 0.65, 0.70, 0.75]:
        blend = wa * y_attn + (1-wa) * y_wattn
        auc = macro_auc(file_labels, blend)
        nm = f'ens_attn_wattn_k{k}_T{T}_wa{wa:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, blend.copy()
        if auc > BEST_NL - 0.002:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After E, best so far: {best_so_far:.4f}\n", flush=True)

# ── F) 混合 window agg: mean(wink1_sim, wink3_sim) ───────────────────────
print("="*60)
print("F) Window: sim(query_win, train_win) aggregated differently")
print("="*60, flush=True)

def window_knn_max_sim_loo(k=3):
    """Take max similarity per training file, then use that as weight for label"""
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        tr_file_ids = win_file_id[tr_wi]
        sims = X_te @ X_tr.T   # (n_test_wins, n_tr_wins)
        # Max similarity between any test window and any training window (per training file)
        n_tr_files = n_files - 1
        max_sim_per_file = np.zeros((n_tr_files, ), np.float32)
        tr_file_unique = np.array([j for j in range(n_files) if j != i])
        for fi_idx, fi in enumerate(tr_file_unique):
            fi_wins = tr_file_ids == fi
            if fi_wins.sum() == 0:
                continue
            max_sim_per_file[fi_idx] = sims[:, fi_wins].max()

        # Weighted average of file labels by max sim
        w = max_sim_per_file.clip(0)
        ws = w.sum()
        if ws > 1e-8:
            w /= ws
        else:
            w = np.ones(n_tr_files) / n_tr_files
        preds[i] = (w[:, None] * file_labels[tr_file_unique]).sum(0)
    return preds

y_msf = window_knn_max_sim_loo(k=3)
auc_msf = macro_auc(file_labels, y_msf)
print(f"  window_max_sim_per_file: {auc_msf:.4f}", flush=True)
results['win_max_sim_per_file'] = auc_msf
for wa in [0.55, 0.60, 0.65, 0.70, 0.75]:
    blend = wa * y_attn + (1-wa) * y_msf
    auc = macro_auc(file_labels, blend)
    nm = f'ens_attn{wa:.2f}_max_sim_file'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far:
        best_so_far = auc
        best_nm, best_preds = nm, blend.copy()
    if auc > BEST_NL - 0.002:
        print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

print(f"After F, best so far: {best_so_far:.4f}\n", flush=True)

# ── G) 四路 ensemble: attn + win1 + win3 + win_attn ─────────────────────
print("="*60)
print("G) 4-way ensemble: file-attn + win1 + win3 + win-attn")
print("="*60, flush=True)

y_wattn_k3 = window_attn_knn_loo(k=3, T=0.2)
for w_a in [0.50, 0.55, 0.60]:
    for w1 in [0.10, 0.15, 0.20]:
        for w3 in [0.10, 0.15, 0.20]:
            w_wa = round(1 - w_a - w1 - w3, 3)
            if w_wa <= 0 or w_wa > 0.3:
                continue
            blend = w_a * y_attn + w1 * y_win1 + w3 * y_win3 + w_wa * y_wattn_k3
            auc = macro_auc(file_labels, blend)
            nm = f'4way_a{w_a:.2f}_w1{w1:.2f}_w3{w3:.2f}_wa{w_wa:.2f}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, blend.copy()
            if auc > BEST_NL - 0.002:
                print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
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
    import json
    data = json.load(f)

cur_best_nl = data.get('best_nologit', {}).get('loo_auc', BEST_NL)
for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': 'nologit_batch4'})
    if auc > cur_best_nl:
        cur_best_nl = auc
        data['best_nologit'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'no_logit NEW BEST batch4'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"no-logit best: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")
if global_best_auc > BEST_NL:
    print(f"\nNEW BEST！{global_best_name} = {global_best_auc:.4f}")
print("done", flush=True)
