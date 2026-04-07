"""
No-logit 第二批實驗：PT-MAP、ZCA、NCM、多空間 ensemble
No-logit best to beat: 0.8796
"""
import numpy as np, json, pickle, re, os, shutil
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
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

# Current best space: pca24 + day
pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
X24  /= (X24.std(0) + 1e-6)
X_nl_pca24 = np.concatenate([X24, geo_all], 1).astype(np.float32)
X_nl_pca24 /= np.linalg.norm(X_nl_pca24, axis=1, keepdims=True) + 1e-8

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

def window_knn_loo(k=3):
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

BEST_NL = 0.8796
best_so_far = BEST_NL
results = {}

print(f"Files={n_files}, species={n_species}")
print(f"No-logit best to beat: {BEST_NL:.4f}\n", flush=True)

# ── A) Power Transform (PT-MAP style) ─────────────────────────────────────
print("="*60)
print("A) Power Transform β: sign(x)|x|^β on L2-normed embeddings")
print("="*60, flush=True)

for beta in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    # Power transform on raw embeddings
    X_pt = np.sign(file_embs_norm) * np.abs(file_embs_norm) ** beta
    X_pt = normalize(X_pt, norm='l2').astype(np.float32)
    # PCA24 on transformed
    pca_pt = PCA(n_components=24, random_state=42).fit(X_pt)
    X24_pt = pca_pt.transform(X_pt).astype(np.float32)
    X24_pt /= (X24_pt.std(0) + 1e-6)
    X_ptg = np.concatenate([X24_pt, geo_all], 1).astype(np.float32)
    X_ptg /= np.linalg.norm(X_ptg, 1, keepdims=True) + 1e-8

    for k, T in [(10, 0.2), (10, 0.15), (7, 0.2)]:
        p = attn_knn_loo(X_ptg, k=k, T=T)
        auc = macro_auc(file_labels, p)
        nm = f'pt_beta{beta:.1f}_pca24geo_k{k}_T{T}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, p.copy()
        if auc > BEST_NL - 0.005:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After A, best so far: {best_so_far:.4f}\n", flush=True)

# ── B) ZCA Whitening ──────────────────────────────────────────────────────
print("="*60)
print("B) ZCA Whitening + Attn-KNN")
print("="*60, flush=True)

for pca_d in [24, 32, 48]:
    try:
        pca_w = PCA(n_components=pca_d, random_state=42, whiten=True).fit(file_embs_norm)
        X_zca = pca_w.transform(file_embs_norm).astype(np.float32)
        X_zca = normalize(X_zca, norm='l2').astype(np.float32)
        # with geo
        X_zg = np.concatenate([X_zca, geo_all], 1).astype(np.float32)
        X_zg /= np.linalg.norm(X_zg, 1, keepdims=True) + 1e-8

        for k, T in [(10, 0.2), (10, 0.15), (7, 0.2), (15, 0.2)]:
            p = attn_knn_loo(X_zg, k=k, T=T)
            auc = macro_auc(file_labels, p)
            nm = f'zca_pca{pca_d}_k{k}_T{T}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, p.copy()
            if auc > BEST_NL - 0.005:
                print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm] = auc

        # ensemble with window_knn
        y_attn_zca = attn_knn_loo(X_zg, k=10, T=0.2)
        y_win3 = window_knn_loo(k=3)
        for wa in [0.65, 0.70, 0.75]:
            blend = wa * y_attn_zca + (1-wa) * y_win3
            auc = macro_auc(file_labels, blend)
            nm = f'zca_pca{pca_d}_ens_attn{wa:.2f}_win{1-wa:.2f}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, blend.copy()
            if auc > BEST_NL - 0.003:
                print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm] = auc
    except Exception as ex:
        print(f"  ZCA pca{pca_d} failed: {ex}", flush=True)

print(f"After B, best so far: {best_so_far:.4f}\n", flush=True)

# ── C) Nearest Class Mean (NCM) ───────────────────────────────────────────
print("="*60)
print("C) Nearest Class Mean (per-species prototype) + cosine sim LOO")
print("="*60, flush=True)

# For each species, compute prototype from training files, then compute cosine sim to query
for X_feat, feat_name in [(X_nl_pca24, 'pca24+geo'), (file_embs_norm, 'raw')]:
    preds_ncm = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = [j for j in range(n_files) if j != i]
        for s in range(n_species):
            pos_idx = [j for j in tr if file_labels[j, s] > 0.5]
            neg_idx = [j for j in tr if file_labels[j, s] < 0.5]
            if len(pos_idx) == 0:
                preds_ncm[i, s] = 0.0
                continue
            proto_pos = X_feat[pos_idx].mean(0)
            proto_neg = X_feat[neg_idx].mean(0) if len(neg_idx) > 0 else np.zeros_like(proto_pos)
            proto_pos = proto_pos / (np.linalg.norm(proto_pos) + 1e-8)
            proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)
            sim_pos = float(X_feat[i] @ proto_pos)
            sim_neg = float(X_feat[i] @ proto_neg) if len(neg_idx) > 0 else 0.0
            preds_ncm[i, s] = (sim_pos - sim_neg + 1.0) / 2.0
    auc = macro_auc(file_labels, preds_ncm)
    nm = f'ncm_{feat_name}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far:
        best_so_far = auc
        best_nm, best_preds = nm, preds_ncm.copy()
    print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

    # Blend NCM with attn-KNN
    y_attn = attn_knn_loo(X_nl_pca24, k=10, T=0.2)
    for wa in [0.60, 0.70, 0.75, 0.80]:
        blend = wa * y_attn + (1-wa) * preds_ncm
        auc = macro_auc(file_labels, blend)
        nm2 = f'ncm_{feat_name}_attn{wa:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm2, blend.copy()
        if auc > BEST_NL - 0.003:
            print(f"  {nm2}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm2] = auc

print(f"After C, best so far: {best_so_far:.4f}\n", flush=True)

# ── D) Window KNN k=1, 2 test ────────────────────────────────────────────
print("="*60)
print("D) Window KNN k=1, 2 + ensemble")
print("="*60, flush=True)

y_attn = attn_knn_loo(X_nl_pca24, k=10, T=0.2)

for k_w in [1, 2]:
    y_wk = window_knn_loo(k=k_w)
    auc_wk = macro_auc(file_labels, y_wk)
    print(f"  window_knn k={k_w}: {auc_wk:.4f}", flush=True)
    results[f'win_k{k_w}'] = auc_wk

    for wa in [0.60, 0.65, 0.70, 0.75, 0.80]:
        blend = wa * y_attn + (1-wa) * y_wk
        auc = macro_auc(file_labels, blend)
        nm = f'ens2_attn{wa:.2f}_wink{k_w}_{1-wa:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, blend.copy()
        if auc > BEST_NL - 0.003:
            print(f"  attn={wa:.2f} wk={k_w}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After D, best so far: {best_so_far:.4f}\n", flush=True)

# ── E) Multi-space ensemble: pca24+geo, pca16+geo, pca8+geo ─────────────
print("="*60)
print("E) Multi-space attn-KNN ensemble")
print("="*60, flush=True)

spaces = {}
for pca_d in [8, 12, 16, 20, 24, 32]:
    pca_s = PCA(n_components=pca_d, random_state=42).fit(file_embs_norm)
    Xs = pca_s.transform(file_embs_norm).astype(np.float32)
    Xs /= (Xs.std(0) + 1e-6)
    Xsg = np.concatenate([Xs, geo_all], 1).astype(np.float32)
    Xsg /= np.linalg.norm(Xsg, 1, keepdims=True) + 1e-8
    spaces[pca_d] = Xsg

# Precompute predictions for each space
space_preds = {}
for d, X_s in spaces.items():
    space_preds[d] = attn_knn_loo(X_s, k=10, T=0.2)
    auc = macro_auc(file_labels, space_preds[d])
    print(f"  pca{d}+geo k10 T0.2: {auc:.4f}", flush=True)
    results[f'attn_pca{d}_geo_k10_T02'] = auc

# Two-space ensemble
dims = [8, 12, 16, 20, 24, 32]
for i, d1 in enumerate(dims):
    for d2 in dims[i+1:]:
        for w1 in [0.4, 0.5, 0.6]:
            blend = w1 * space_preds[d1] + (1-w1) * space_preds[d2]
            auc = macro_auc(file_labels, blend)
            nm = f'ens_pca{d1}x{d2}_w{w1:.1f}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, blend.copy()
            if auc > BEST_NL - 0.003:
                print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm] = auc

print(f"After E, best so far: {best_so_far:.4f}\n", flush=True)

# ── F) Geo-only variants ──────────────────────────────────────────────────
print("="*60)
print("F) Geo feature variants: site-only, hour-only, day-only")
print("="*60, flush=True)

geo_variants = {
    'site_only': site_oh,
    'hour_only': hour_enc,
    'month_only': month_enc,
    'day_only': day_enc,
    'site+day': np.concatenate([site_oh, day_enc], 1).astype(np.float32),
    'site+month+day': np.concatenate([site_oh, month_enc, day_enc], 1).astype(np.float32),
    'no_site': np.concatenate([hour_enc, month_enc, day_enc], 1).astype(np.float32),
    'no_hour': np.concatenate([site_oh, month_enc, day_enc], 1).astype(np.float32),
}

y_win3 = window_knn_loo(k=3)
for geo_name, geo_vec in geo_variants.items():
    Xg = np.concatenate([X24, geo_vec], 1).astype(np.float32)
    Xg /= np.linalg.norm(Xg, 1, keepdims=True) + 1e-8

    for k, T in [(10, 0.2), (10, 0.15)]:
        p = attn_knn_loo(Xg, k=k, T=T)
        auc = macro_auc(file_labels, p)
        nm = f'attn_pca24_{geo_name}_k{k}_T{T}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, p.copy()
        if auc > BEST_NL - 0.005:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

    # Ensemble with win_k3
    p_geo = attn_knn_loo(Xg, k=10, T=0.2)
    for wa in [0.65, 0.70, 0.75]:
        blend = wa * p_geo + (1-wa) * y_win3
        auc = macro_auc(file_labels, blend)
        nm = f'{geo_name}_ens_attn{wa:.2f}_wink3'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, blend.copy()
        if auc > BEST_NL - 0.003:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After F, best so far: {best_so_far:.4f}\n", flush=True)

# ── G) Window-level aggregation variants ────────────────────────────────
print("="*60)
print("G) Window-level aggregation: median, max-then-mean")
print("="*60, flush=True)

# Window KNN with median aggregation
def window_knn_median_loo(k=3):
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
        preds[i] = np.median(wp, 0)  # median instead of mean
    return preds

# Window KNN with max aggregation
def window_knn_max_loo(k=3):
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
        preds[i] = wp.max(0)  # max pooling
    return preds

y_attn = attn_knn_loo(X_nl_pca24, k=10, T=0.2)
for k_w in [3, 5]:
    y_med = window_knn_median_loo(k=k_w)
    auc_med = macro_auc(file_labels, y_med)
    print(f"  window_knn_median k={k_w}: {auc_med:.4f}", flush=True)
    results[f'win_median_k{k_w}'] = auc_med

    y_mx = window_knn_max_loo(k=k_w)
    auc_mx = macro_auc(file_labels, y_mx)
    print(f"  window_knn_max k={k_w}: {auc_mx:.4f}", flush=True)
    results[f'win_max_k{k_w}'] = auc_mx

    for agg_name, y_agg in [('median', y_med), ('max', y_mx)]:
        for wa in [0.65, 0.70, 0.75, 0.80]:
            blend = wa * y_attn + (1-wa) * y_agg
            auc = macro_auc(file_labels, blend)
            nm = f'ens_attn{wa:.2f}_win{agg_name}_k{k_w}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, blend.copy()
            if auc > BEST_NL - 0.003:
                print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm] = auc

print(f"After G, best so far: {best_so_far:.4f}\n", flush=True)

# ── H) Temperature fine sweep for attn-KNN ───────────────────────────────
print("="*60)
print("H) Fine temperature sweep + window ensemble")
print("="*60, flush=True)

y_win3 = window_knn_loo(k=3)
for k in [7, 10, 12, 15]:
    for T in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
        p = attn_knn_loo(X_nl_pca24, k=k, T=T)
        for wa in [0.65, 0.70, 0.75]:
            blend = wa * p + (1-wa) * y_win3
            auc = macro_auc(file_labels, blend)
            nm = f'fine_k{k}_T{T:.2f}_win{1-wa:.2f}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, blend.copy()
            if auc > BEST_NL - 0.002:
                print(f"  k={k} T={T:.2f} wa={wa:.2f}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm] = auc

print(f"After H, best so far: {best_so_far:.4f}\n", flush=True)

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

# Update results.json
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best_nl = data.get('best_nologit', {}).get('loo_auc', BEST_NL)
for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': 'nologit_batch2'})
    if auc > cur_best_nl:
        cur_best_nl = auc
        data['best_nologit'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'no_logit NEW BEST batch2'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"no-logit best: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")

# Save new best pkl if better
if global_best_auc > BEST_NL:
    print(f"\nNEW BEST: {global_best_name} = {global_best_auc:.4f}")
    # Need to rebuild the space for pkl - skip for now, will handle in build script
    print("請用 build_*.py 建立對應 pkl")

print("done", flush=True)
