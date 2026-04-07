"""
No-logit 第三批：文獻方法實作
1. SimpleShot CL2N (mean-subtracted L2N + Nearest Centroid)
2. Label Spreading (sklearn graph-based)
3. LaplacianShot-style transductive KNN
4. Prototype Rectification (BD-CSPN style, confident pseudo-label)
5. Power Transform + ensemble (PT-MAP style)
6. CL2N + window ensemble
7. 更精細的 attn+wink1 sweep

No-logit best to beat: 0.8810
"""
import numpy as np, json, pickle, re, os, shutil
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.semi_supervised import LabelSpreading
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

# Precompute common predictions
y_attn = attn_knn_loo(X_nl_pca24, k=10, T=0.2)
y_win1 = window_knn_loo(k=1)
y_win3 = window_knn_loo(k=3)

# ── A) SimpleShot CL2N: mean-subtract then L2N, Nearest Centroid ──────────
print("="*60)
print("A) SimpleShot CL2N + species prototype")
print("="*60, flush=True)

# Centered + L2-normalized embeddings
mu_all = file_embs.mean(0)
X_cl2n = file_embs - mu_all
X_cl2n = normalize(X_cl2n, norm='l2').astype(np.float32)

# CL2N + PCA + geo
pca_cl = PCA(n_components=24, random_state=42).fit(X_cl2n)
X_cl_p = pca_cl.transform(X_cl2n).astype(np.float32)
X_cl_p /= (X_cl_p.std(0) + 1e-6)
X_cl_g = np.concatenate([X_cl_p, geo_all], 1).astype(np.float32)
X_cl_g /= np.linalg.norm(X_cl_g, 1, keepdims=True) + 1e-8

for k, T in [(10, 0.2), (10, 0.15), (7, 0.2), (10, 0.25)]:
    p = attn_knn_loo(X_cl_g, k=k, T=T)
    auc = macro_auc(file_labels, p)
    nm = f'cl2n_pca24geo_k{k}_T{T}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far:
        best_so_far = auc
        best_nm, best_preds = nm, p.copy()
    if auc > BEST_NL - 0.005:
        print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

# CL2N + ensemble with win_k1
y_cl_attn = attn_knn_loo(X_cl_g, k=10, T=0.2)
for wa in [0.55, 0.60, 0.65, 0.70, 0.75]:
    for k_w, y_w in [(1, y_win1), (3, y_win3)]:
        blend = wa * y_cl_attn + (1-wa) * y_w
        auc = macro_auc(file_labels, blend)
        nm = f'cl2n_attn{wa:.2f}_wink{k_w}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, blend.copy()
        if auc > BEST_NL - 0.003:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After A, best so far: {best_so_far:.4f}\n", flush=True)

# ── B) Power Transform β sweep with wink1 ensemble ───────────────────────
print("="*60)
print("B) Power Transform + wink1 ensemble")
print("="*60, flush=True)

for beta in [0.3, 0.4, 0.5, 0.6, 0.7]:
    X_pt = np.sign(file_embs_norm) * np.abs(file_embs_norm) ** beta
    X_pt = normalize(X_pt, norm='l2').astype(np.float32)
    pca_pt = PCA(n_components=24, random_state=42).fit(X_pt)
    X24_pt = pca_pt.transform(X_pt).astype(np.float32)
    X24_pt /= (X24_pt.std(0) + 1e-6)
    Xptg = np.concatenate([X24_pt, geo_all], 1).astype(np.float32)
    Xptg /= np.linalg.norm(Xptg, 1, keepdims=True) + 1e-8

    y_pt = attn_knn_loo(Xptg, k=10, T=0.2)
    for wa in [0.55, 0.60, 0.65, 0.70, 0.75]:
        for k_w, y_w in [(1, y_win1), (3, y_win3)]:
            blend = wa * y_pt + (1-wa) * y_w
            auc = macro_auc(file_labels, blend)
            nm = f'pt{beta:.1f}_attn{wa:.2f}_wink{k_w}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, blend.copy()
            if auc > BEST_NL - 0.002:
                print(f"  β={beta} wa={wa:.2f} k_w={k_w}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm] = auc

print(f"After B, best so far: {best_so_far:.4f}\n", flush=True)

# ── C) Prototype Rectification (BD-CSPN style) ────────────────────────────
print("="*60)
print("C) Prototype Rectification: confident pseudo-label refine")
print("="*60, flush=True)

def proto_rectify_loo(X, tau=0.7, n_rounds=2, k=10, T=0.2):
    """
    Round 1: Attn-KNN prediction → soft labels for query
    Round 2+: Add confident pseudo-labeled queries to class prototypes, re-predict
    """
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        # Round 1: initial KNN
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        p1 = (w[:, None] * file_labels[tr[top]]).sum(0)

        # For multi-label, we treat each species independently
        # Only do rectification if confidence > tau
        p_final = p1.copy()
        for rnd in range(n_rounds - 1):
            # Per species, create rectified prototype
            p_new = np.zeros(n_species, np.float32)
            for s in range(n_species):
                pos_tr = tr[file_labels[tr, s] > 0.5]
                neg_tr = tr[file_labels[tr, s] < 0.5]
                # "pseudo-positive" from high-confidence query prediction (only the query itself)
                if p_final[s] > tau:
                    pseudo_pos = np.concatenate([[i], pos_tr])
                elif p_final[s] < (1 - tau):
                    pseudo_neg = np.concatenate([[i], neg_tr])
                    pos_tr_eff = pos_tr
                else:
                    pos_tr_eff = pos_tr

                if len(pos_tr) == 0:
                    p_new[s] = 0.0
                    continue
                # Rectified prototype
                proto = X[pos_tr].mean(0)
                if p_final[s] > tau:
                    proto = (X[pos_tr].sum(0) + X[i]) / (len(pos_tr) + 1)
                proto /= (np.linalg.norm(proto) + 1e-8)
                p_new[s] = float(X[i] @ proto)
            p_final = p_new

        # Re-predict with rectified: blend with initial
        preds[i] = 0.5 * p1 + 0.5 * p_final
    return preds

# This is slow (O(n_files × n_species)), use smaller species count
print("  Running prototype rectification (may be slow)...", flush=True)
p_pr = proto_rectify_loo(X_nl_pca24, tau=0.7, n_rounds=2)
auc = macro_auc(file_labels, p_pr)
nm = 'proto_rectify_tau07'
print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f})", flush=True)
results[nm] = auc

# Blend with attn-KNN
for wa in [0.60, 0.70, 0.80]:
    blend = wa * y_attn + (1-wa) * p_pr
    auc = macro_auc(file_labels, blend)
    nm = f'proto_rectify_blend_attn{wa:.2f}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far:
        best_so_far = auc
        best_nm, best_preds = nm, blend.copy()
    if auc > BEST_NL - 0.003:
        print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

print(f"After C, best so far: {best_so_far:.4f}\n", flush=True)

# ── D) LaplacianShot-style: Laplacian 正則化 transductive ─────────────────
print("="*60)
print("D) LaplacianShot-style Laplacian regularized")
print("="*60, flush=True)

def laplacian_shot_loo(X, k_graph=5, lam=0.5, n_iter=10):
    """
    Simplified LaplacianShot for multi-label:
    F = initial KNN probs (unary)
    Y = F
    Iterate: Y = (1-lam)*F + lam * W_norm @ Y
    where W is kNN affinity graph among all files
    """
    preds = np.zeros((n_files, n_species), np.float32)

    # Build affinity graph (symmetric kNN) from ALL files
    sims_all = (X @ X.T).astype(np.float32)  # (n_files, n_files)
    np.fill_diagonal(sims_all, -1)  # exclude self
    W = np.zeros_like(sims_all)
    for i in range(n_files):
        top_k = np.argsort(-sims_all[i])[:k_graph]
        W[i, top_k] = sims_all[i, top_k].clip(0)
    W = (W + W.T) / 2  # symmetrize
    row_sum = W.sum(1, keepdims=True).clip(1e-8)
    W_norm = W / row_sum  # row-normalized

    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        # Initial prediction via attn-KNN
        sims_i = (X[[i]] @ X[tr].T).ravel()
        top = np.argsort(-sims_i)[:10]
        logit = sims_i[top] / 0.2; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        F_i = (w[:, None] * file_labels[tr[top]]).sum(0)  # (n_species,)

        # Laplacian smoothing on training labels using graph
        # Create label matrix for all files (training use GT, i-th use F_i)
        Y = file_labels.copy()
        Y[i] = F_i  # initialize i with prediction

        for _ in range(n_iter):
            # Only update file i's label (transductive for i)
            # Using other files' graph neighbors to smooth
            neighbors_i = W_norm[i]  # (n_files,)
            Y[i] = (1 - lam) * F_i + lam * (neighbors_i[:, None] * Y).sum(0)
            Y[i] = Y[i].clip(0, 1)

        preds[i] = Y[i]
    return preds

for k_graph in [5, 8, 10]:
    for lam in [0.3, 0.5, 0.7]:
        p = laplacian_shot_loo(X_nl_pca24, k_graph=k_graph, lam=lam)
        auc = macro_auc(file_labels, p)
        nm = f'laplacian_k{k_graph}_lam{lam}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, p.copy()
        if auc > BEST_NL - 0.005:
            print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

        # Blend with win_k1
        for wa in [0.65, 0.70]:
            blend = wa * p + (1-wa) * y_win1
            auc2 = macro_auc(file_labels, blend)
            nm2 = f'laplacian_k{k_graph}_lam{lam}_wink1_{1-wa:.2f}'
            marker2 = " ← NEW BEST" if auc2 > best_so_far else ""
            if auc2 > best_so_far:
                best_so_far = auc2
                best_nm, best_preds = nm2, blend.copy()
            if auc2 > BEST_NL - 0.003:
                print(f"  {nm2}: {auc2:.4f}  (Δ={auc2-BEST_NL:+.4f}){marker2}", flush=True)
            results[nm2] = auc2

print(f"After D, best so far: {best_so_far:.4f}\n", flush=True)

# ── E) Label Spreading (sklearn) ──────────────────────────────────────────
print("="*60)
print("E) Label Spreading (sklearn) per-species LOO")
print("="*60, flush=True)

def label_spreading_loo(X, alpha=0.2, n_neighbors=7):
    """
    Per-species label spreading LOO.
    For each held-out file i:
      - i is "unlabeled", all others are labeled
      - Run LabelSpreading on features X
    """
    preds = np.zeros((n_files, n_species), np.float32)

    for i in range(n_files):
        tr = [j for j in range(n_files) if j != i]
        # LabelSpreading needs integer labels; for multi-label do per-species
        X_all = X.copy()  # all files including i

        for s in range(n_species):
            y_tr = file_labels[tr, s].astype(np.int32)  # 0 or 1
            if y_tr.sum() == 0:
                preds[i, s] = 0.0
                continue

            # Labels: use -1 for unlabeled (file i)
            y_ls = np.full(n_files, -1, dtype=np.int32)
            y_ls[tr] = y_tr

            try:
                ls = LabelSpreading(kernel='knn', n_neighbors=n_neighbors,
                                    alpha=alpha, max_iter=30)
                ls.fit(X_all, y_ls)
                prob = ls.label_distributions_[i]  # [P(neg), P(pos)]
                preds[i, s] = prob[1] if len(prob) > 1 else 0.0
            except Exception:
                preds[i, s] = 0.0

        if (i + 1) % 10 == 0:
            print(f"  LOO fold {i+1}/{n_files} done", flush=True)

    return preds

# This is slow, try with reduced species or fast version
# Use first 50 species as test, or run full
print("  Running label spreading (full, may take a while)...", flush=True)
for alpha_ls in [0.2, 0.5]:
    for nn_ls in [5, 7]:
        p_ls = label_spreading_loo(X_nl_pca24, alpha=alpha_ls, n_neighbors=nn_ls)
        auc = macro_auc(file_labels, p_ls)
        nm = f'label_spread_alpha{alpha_ls}_nn{nn_ls}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, p_ls.copy()
        print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

        # Blend
        for wa in [0.65, 0.70]:
            blend = wa * p_ls + (1-wa) * y_win1
            auc2 = macro_auc(file_labels, blend)
            nm2 = f'ls_blend_a{alpha_ls}_nn{nn_ls}_wink1_{1-wa:.2f}'
            marker2 = " ← NEW BEST" if auc2 > best_so_far else ""
            if auc2 > best_so_far:
                best_so_far = auc2
                best_nm, best_preds = nm2, blend.copy()
            if auc2 > BEST_NL - 0.003:
                print(f"  {nm2}: {auc2:.4f}  (Δ={auc2-BEST_NL:+.4f}){marker2}", flush=True)
            results[nm2] = auc2

print(f"After E, best so far: {best_so_far:.4f}\n", flush=True)

# ── F) 更精細的 attn+wink1 sweep (fine-grained) ───────────────────────────
print("="*60)
print("F) Fine sweep: attn k/T + wink1 weight")
print("="*60, flush=True)

for k in [8, 9, 10, 11, 12]:
    for T in [0.17, 0.19, 0.20, 0.21, 0.23]:
        y_a = attn_knn_loo(X_nl_pca24, k=k, T=T)
        for wa in [0.60, 0.63, 0.65, 0.67, 0.70]:
            blend = wa * y_a + (1-wa) * y_win1
            auc = macro_auc(file_labels, blend)
            nm = f'fine2_k{k}_T{T:.2f}_wa{wa:.2f}_wink1'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far:
                best_so_far = auc
                best_nm, best_preds = nm, blend.copy()
            if auc > BEST_NL - 0.001:
                print(f"  k={k} T={T:.2f} wa={wa:.2f}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
            results[nm] = auc

print(f"After F, best so far: {best_so_far:.4f}\n", flush=True)

# ── G) 3-way: attn + wink1 + wink3 ──────────────────────────────────────
print("="*60)
print("G) 3-way: attn + wink1 + wink3")
print("="*60, flush=True)

for w_a in [0.55, 0.60, 0.65, 0.70]:
    for w1 in [0.10, 0.15, 0.20, 0.25, 0.30]:
        w3 = round(1 - w_a - w1, 3)
        if w3 <= 0:
            continue
        blend = w_a * y_attn + w1 * y_win1 + w3 * y_win3
        auc = macro_auc(file_labels, blend)
        nm = f'3way_attn{w_a:.2f}_w1{w1:.2f}_w3{w3:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_nm, best_preds = nm, blend.copy()
        if auc > BEST_NL - 0.002:
            print(f"  attn={w_a:.2f} w1={w1:.2f} w3={w3:.2f}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
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
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': 'nologit_literature'})
    if auc > cur_best_nl:
        cur_best_nl = auc
        data['best_nologit'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'no_logit NEW BEST literature'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"no-logit best: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")

if global_best_auc > BEST_NL:
    print(f"\nNEW BEST！請用 build_*.py 建立對應 pkl")

print("done", flush=True)
