"""
No-logit 第四批(修正版)：
- ZCA Whitening + ensemble win_k1
- TIM-ADM（已修正）
- Window Attn-KNN（softmax）
- 原始 1536-dim 空間直接 KNN
- 最佳視窗選取策略
- Geometric mean 聚合
- 更多組合

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

# ── A) ZCA Whitening（最多 min(66,dim) components） ───────────────────────
print("="*60)
print("A) ZCA Whitening + Attn-KNN + ensemble win_k1")
print("="*60, flush=True)

def zca_whiten(X, eps=1e-4):
    """ZCA whitening via SVD of centered data matrix"""
    X = X.astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    S2 = S**2 / len(X)
    W = Vt.T @ np.diag(1.0 / np.sqrt(S2 + eps)) @ Vt
    return (Xc @ W).astype(np.float32)

max_pca = min(n_files - 2, 60)  # safe max
for pca_d in [20, 30, 40, 50]:
    if pca_d > max_pca:
        pca_d = max_pca
    try:
        pca_z = PCA(n_components=pca_d, random_state=42).fit(file_embs_norm)
        Xr = pca_z.transform(file_embs_norm).astype(np.float32)
        Xw = zca_whiten(Xr, eps=1e-3)
        Xw_n = normalize(Xw, norm='l2').astype(np.float32)
        # with geo
        Xwg = np.concatenate([Xw_n, geo_all], 1).astype(np.float32)
        Xwg /= np.linalg.norm(Xwg, 1, keepdims=True) + 1e-8

        y_z = attn_knn_loo(Xwg, k=10, T=0.2)
        auc_z = macro_auc(file_labels, y_z)
        print(f"  ZCA pca{pca_d}: {auc_z:.4f}", flush=True)
        results[f'zca_pca{pca_d}_k10_T02'] = auc_z

        for wa in [0.55, 0.60, 0.65, 0.70]:
            blend = wa * y_z + (1-wa) * y_win1
            auc = macro_auc(file_labels, blend)
            nm = f'zca_pca{pca_d}_attn{wa:.2f}_wink1'
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

# ── B) TIM-ADM (Transductive Information Maximization) ───────────────────
print("="*60)
print("B) TIM-ADM (entropy amplification via logit scaling)")
print("="*60, flush=True)

def tim_adm_loo(X, k_init=10, T_init=0.2, lam=2.0, n_iter=10):
    """Amplify initial attn-KNN predictions via logit scaling"""
    EPS = 1e-7
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k_init]
        logit = sims[top] / T_init; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        q = (w[:, None] * file_labels[tr[top]]).sum(0)
        # Amplify via logit scaling (TIM-style entropy minimization)
        for _ in range(n_iter):
            q_c = q.clip(EPS, 1-EPS)
            logit_q = np.log(q_c) - np.log(1 - q_c)
            q = 1.0 / (1.0 + np.exp(-lam * logit_q))
        preds[i] = q
    return preds

for lam in [1.5, 2.0, 3.0, 5.0]:
    p = tim_adm_loo(X_nl_pca24, lam=lam, n_iter=10)
    auc_t = macro_auc(file_labels, p)
    print(f"  TIM lam={lam}: {auc_t:.4f}", flush=True)
    results[f'tim_lam{lam:.1f}'] = auc_t

    for wa in [0.55, 0.60, 0.65, 0.70]:
        blend = wa * p + (1-wa) * y_win1
        auc = macro_auc(file_labels, blend)
        nm = f'tim_lam{lam:.1f}_attn{wa:.2f}_wink1'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, blend.copy()
        if auc > BEST_NL - 0.003:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After B, best so far: {best_so_far:.4f}\n", flush=True)

# ── C) 原始 1536-dim 空間 直接 KNN（不做 PCA） ──────────────────────────
print("="*60)
print("C) High-dim raw cosine KNN (1536-dim) + ensemble")
print("="*60, flush=True)

# 用原始 L2-normalized embedding 直接做 attn-KNN
for k, T in [(10, 0.2), (10, 0.15), (5, 0.2), (15, 0.2)]:
    p = attn_knn_loo(file_embs_norm, k=k, T=T)
    auc_r = macro_auc(file_labels, p)
    nm = f'raw1536_k{k}_T{T}'
    if auc_r > BEST_NL - 0.005:
        print(f"  {nm}: {auc_r:.4f}  (Δ={auc_r-BEST_NL:+.4f})", flush=True)
    results[nm] = auc_r

    for wa in [0.55, 0.60, 0.65, 0.70, 0.75]:
        blend = wa * p + (1-wa) * y_win1
        auc = macro_auc(file_labels, blend)
        nm2 = f'raw1536_k{k}_T{T}_attn{wa:.2f}_wink1'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm2, blend.copy()
        if auc > BEST_NL - 0.003:
            print(f"  {nm2}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm2] = auc

print(f"After C, best so far: {best_so_far:.4f}\n", flush=True)

# ── D) 最佳視窗選取策略 ──────────────────────────────────────────────────
print("="*60)
print("D) Best-window selection as file embedding")
print("="*60, flush=True)

def best_window_embed_loo(top_n=1):
    """
    For each file, select the top_n windows most similar to ANY training window,
    use average of those as the file embedding for attn-KNN.
    """
    # Precompute: for each test file, which window is "most representative"?
    # Strategy: window with highest max similarity to training windows
    file_best_embs = np.zeros((n_files, emb_win.shape[1]), np.float32)
    for fi in range(n_files):
        te_s, te_e = int(file_start[fi]), int(file_end[fi])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != fi
        X_tr = emb_win_norm[tr_mask]
        sims = X_te @ X_tr.T   # (n_test_wins, n_tr_wins)
        max_sim_per_win = sims.max(1)  # (n_test_wins,)
        top_wins = np.argsort(-max_sim_per_win)[:top_n]
        file_best_embs[fi] = X_te[top_wins].mean(0)
    file_best_embs = normalize(file_best_embs, norm='l2').astype(np.float32)
    return file_best_embs

for top_n in [1, 2, 3]:
    best_embs = best_window_embed_loo(top_n=top_n)
    # PCA + geo on best embeddings
    pca_bw = PCA(n_components=24, random_state=42).fit(best_embs)
    X_bw = pca_bw.transform(best_embs).astype(np.float32)
    X_bw /= (X_bw.std(0) + 1e-6)
    X_bwg = np.concatenate([X_bw, geo_all], 1).astype(np.float32)
    X_bwg /= np.linalg.norm(X_bwg, 1, keepdims=True) + 1e-8

    for k, T in [(10, 0.2), (10, 0.15), (7, 0.2)]:
        p = attn_knn_loo(X_bwg, k=k, T=T)
        auc_bw = macro_auc(file_labels, p)
        nm = f'best_win{top_n}_pca24geo_k{k}_T{T}'
        if auc_bw > BEST_NL - 0.005:
            print(f"  {nm}: {auc_bw:.4f}  (Δ={auc_bw-BEST_NL:+.4f})", flush=True)
        results[nm] = auc_bw

        for wa in [0.55, 0.60, 0.65, 0.70]:
            blend = wa * p + (1-wa) * y_win1
            auc = macro_auc(file_labels, blend)
            nm2 = f'best_win{top_n}_attn{wa:.2f}_wink1'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm2, blend.copy()
            if auc > BEST_NL - 0.003:
                print(f"  {nm2}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm2] = auc

print(f"After D, best so far: {best_so_far:.4f}\n", flush=True)

# ── E) Window Attn-KNN（softmax 加權）+ ensemble ──────────────────────────
print("="*60)
print("E) Window-level Attn-KNN (softmax) + ensemble")
print("="*60, flush=True)

def window_attn_knn_loo(k=5, T=0.2):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        Y_tr = file_labels[win_file_id[tr_wi]]
        sims = X_te @ X_tr.T
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            top = np.argsort(-sims[wi])[:k]
            s_top = sims[wi, top]
            logit_s = s_top / T; logit_s -= logit_s.max()
            w = np.exp(logit_s); w /= w.sum()
            wp[wi] = (w[:, None] * Y_tr[top]).sum(0)
        preds[i] = wp.mean(0)
    return preds

for k, T in [(1, 0.2), (3, 0.2), (5, 0.2), (5, 0.15)]:
    y_wa = window_attn_knn_loo(k=k, T=T)
    auc_wa = macro_auc(file_labels, y_wa)
    print(f"  win_attn k={k} T={T}: {auc_wa:.4f}", flush=True)
    results[f'win_attn_k{k}_T{T}'] = auc_wa

    # Ensemble with file attn
    for wa in [0.55, 0.60, 0.65, 0.70, 0.75]:
        blend = wa * y_attn + (1-wa) * y_wa
        auc = macro_auc(file_labels, blend)
        nm = f'file_attn_x_win_attn_k{k}_wa{wa:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, blend.copy()
        if auc > BEST_NL - 0.002:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

    # 3-way: file_attn + win_attn + win_k1
    for w_a in [0.55, 0.60]:
        for w_wa in [0.15, 0.20, 0.25]:
            w1 = round(1 - w_a - w_wa, 3)
            if w1 <= 0:
                continue
            blend3 = w_a * y_attn + w_wa * y_wa + w1 * y_win1
            auc = macro_auc(file_labels, blend3)
            nm = f'3way_fa{w_a:.2f}_wa{w_wa:.2f}_w1{w1:.2f}_wattnk{k}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, blend3.copy()
            if auc > BEST_NL - 0.002:
                print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm] = auc

print(f"After E, best so far: {best_so_far:.4f}\n", flush=True)

# ── F) Geometric mean aggregation of window KNN predictions ──────────────
print("="*60)
print("F) Geometric mean window aggregation")
print("="*60, flush=True)

EPS = 1e-7

def window_knn_geo_loo(k=1):
    """Geometric mean (exp of mean of logs) instead of arithmetic mean"""
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
        # Geometric mean: exp(mean(log(p)))
        preds[i] = np.exp(np.log(wp.clip(EPS)).mean(0))
    return preds

for k_g in [1, 3, 5]:
    y_geo = window_knn_geo_loo(k=k_g)
    auc_g = macro_auc(file_labels, y_geo)
    print(f"  win_geo k={k_g}: {auc_g:.4f}", flush=True)
    results[f'win_geo_k{k_g}'] = auc_g

    for wa in [0.55, 0.60, 0.65, 0.70, 0.75]:
        blend = wa * y_attn + (1-wa) * y_geo
        auc = macro_auc(file_labels, blend)
        nm = f'attn{wa:.2f}_wgeo_k{k_g}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, blend.copy()
        if auc > BEST_NL - 0.002:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After F, best so far: {best_so_far:.4f}\n", flush=True)

# ── G) 4-way ensemble: file_attn + win_k1 + win_k3 + win_attn_k1 ────────
print("="*60)
print("G) 4-way ensemble fine sweep")
print("="*60, flush=True)

y_win_attn1 = window_attn_knn_loo(k=1, T=0.2)

for w_a in [0.50, 0.55, 0.60, 0.65]:
    for w1 in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for w_wa in [0.05, 0.10, 0.15]:
            w3 = round(1 - w_a - w1 - w_wa, 3)
            if w3 < 0.05 or w3 > 0.30:
                continue
            blend = w_a * y_attn + w1 * y_win1 + w3 * y_win3 + w_wa * y_win_attn1
            auc = macro_auc(file_labels, blend)
            nm = f'4way_fa{w_a:.2f}_w1{w1:.2f}_w3{w3:.2f}_wa{w_wa:.2f}'
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
    data = json.load(f)

cur_best_nl = data.get('best_nologit', {}).get('loo_auc', BEST_NL)
for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': 'nologit_batch4b'})
    if auc > cur_best_nl:
        cur_best_nl = auc
        data['best_nologit'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'no_logit NEW BEST batch4b'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"no-logit best: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")
if global_best_auc > BEST_NL:
    print(f"\nNEW BEST！{global_best_name} = {global_best_auc:.4f}")
print("done", flush=True)
