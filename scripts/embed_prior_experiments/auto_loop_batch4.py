"""
Embed Prior Auto Loop - Batch 4: Breakthrough novel methods
Current best LOO-AUC: 0.9164 (ls2_geo_k5_win_k1)

Novel methods NOT previously tried:
1. attention_weighted_knn: use neighbor's logit as attention score (not embedding sim)
2. spectral_graph_knn: graph Laplacian eigenvectors as feature space
3. gaussian_process: GP regression per species
4. contrastive_ratio: similarity ratio (positive / background)
5. cross_attention: transformer-style cross-attention query=test, key/val=train
6. temporal_decay_knn: exponential decay by date distance
7. site_prototype_knn: per-site prototypes as intermediate nodes
8. density_ratio_knn: importance weighting by P(test_emb) / P(train_emb)
"""
import numpy as np, pickle, json, os, re
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = "outputs/embed_prior_results.json"

# ─── Load data ────────────────────────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-88,88)))

file_embs      = np.zeros((n_files, emb_win.shape[1]), np.float32)
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s,e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0)>0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)

file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)
emb_win_norm   = normalize(emb_win,   norm='l2').astype(np.float32)
win_file_id    = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

EPS = 1e-7
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:,mask], ys[:,mask], average='macro')

# ─── Load pkl (for X_combined_n) ─────────────────────────────────────────────
with open("outputs/embed_prior_logspace_geo5_win1.pkl","rb") as f:
    ep = pickle.load(f)
X_ref = ep['X_combined_n'].astype(np.float32)  # (66, 39) pkl space
fl    = ep['file_labels'].astype(np.float32)

# ─── Load current best ────────────────────────────────────────────────────────
with open(RESULTS_PATH) as f:
    results_data = json.load(f)
best_loo = results_data['best']['loo_auc']
tried_methods = set(e['method'] for e in results_data.get('experiments', []))
print(f"Current best LOO-AUC: {best_loo:.4f}")
print(f"Methods tried: {len(tried_methods)}")

def save_result(method, loo_auc, config, y_preds=None, is_best=False):
    entry = {'method': method, 'loo_auc': loo_auc, 'config': config}
    results_data['experiments'].append(entry)
    if is_best:
        results_data['best'] = {'method': method, 'loo_auc': loo_auc, **config}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"  Saved: {method} = {loo_auc:.4f}")

all_new_results = []

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Attention-weighted KNN
# Use neighbor's Perch logit (not embedding similarity) as attention weight.
# For each test file, find top-k neighbors by embedding sim,
# then reweight labels by exp(neighbor_logit_for_that_species / T)
# Key insight: a neighbor's label is more reliable if its own logit is high
# ═══════════════════════════════════════════════════════════════════════════════
method = 'attention_weighted_knn'
if method not in tried_methods:
    print(f"\n[1] {method}...", flush=True)
    best_cfg = None; best_auc = 0
    for k in [5, 10]:
        for T_sim in [0.2, 0.3]:
            for T_logit in [1.0, 2.0, 5.0]:
                preds = np.zeros((n_files, n_species), np.float32)
                for i in range(n_files):
                    tr = np.array([j for j in range(n_files) if j!=i])
                    # Step 1: find top-k by embedding similarity
                    sims = (X_ref[[i]]@X_ref[tr].T).ravel()
                    top = np.argsort(-sims)[:k]
                    top_tr = tr[top]
                    # Step 2: for each species, weight by neighbor's logit confidence
                    for s in range(n_species):
                        # neighbor logits for species s (how confident are they about this species)
                        neigh_logits = file_logit_max[top_tr, s]  # (k,)
                        neigh_labels = fl[top_tr, s]               # (k,)
                        # Attention weight = sim_weight * logit_weight
                        sim_w = np.exp(sims[top]/T_sim); sim_w /= sim_w.sum()
                        logit_w = np.exp(neigh_logits/T_logit); logit_w /= logit_w.sum()
                        combined_w = sim_w * logit_w; combined_w /= combined_w.sum()
                        preds[i,s] = (combined_w * neigh_labels).sum()
                auc = macro_auc(file_labels, preds.clip(EPS,1-EPS))
                if auc > best_auc:
                    best_auc = auc; best_cfg = {'k':k,'T_sim':T_sim,'T_logit':T_logit}
                    print(f"  k={k} Ts={T_sim} Tl={T_logit}: {auc:.4f}", flush=True)
    save_result(method, best_auc, best_cfg or {})
    all_new_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        results_data['best'] = {'method': method, 'loo_auc': best_auc, **(best_cfg or {})}
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Spectral Graph KNN
# Build similarity graph of 66 files → compute graph Laplacian eigenvectors
# Use spectral embedding (not PCA) as the feature space for KNN
# This captures global graph structure, not just local variance
# ═══════════════════════════════════════════════════════════════════════════════
method = 'spectral_graph_knn'
if method not in tried_methods:
    print(f"\n[2] {method}...", flush=True)
    best_cfg = None; best_auc = 0
    for n_components in [8, 16, 24, 32]:
        for k_knn in [3, 5, 8]:
            # Build affinity matrix (gaussian kernel on cosine sim)
            cos_sim = X_ref @ X_ref.T  # (66,66)
            np.fill_diagonal(cos_sim, 0)
            A = np.exp(cos_sim / 0.3)  # RBF-like
            np.fill_diagonal(A, 0)
            # Normalized graph Laplacian
            D = A.sum(1)
            D_inv_sqrt = np.diag(1.0/np.sqrt(D+1e-8))
            L_sym = np.eye(n_files) - D_inv_sqrt @ A @ D_inv_sqrt
            # Eigenvectors (smallest eigenvalues = smoothest functions on graph)
            eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
            X_spectral = eigenvectors[:, :n_components].astype(np.float32)
            # Normalize
            X_spectral /= np.linalg.norm(X_spectral, 1, keepdims=True)+1e-8
            # LOO KNN in spectral space
            preds = np.zeros((n_files, n_species), np.float32)
            for i in range(n_files):
                tr = np.array([j for j in range(n_files) if j!=i])
                sims = (X_spectral[[i]]@X_spectral[tr].T).ravel()
                top = np.argsort(-sims)[:k_knn]
                ls = sims[top]/0.2; ls -= ls.max(); w=np.exp(ls); w/=w.sum()
                preds[i] = (w[:,None]*fl[tr[top]]).sum(0)
            auc = macro_auc(file_labels, preds.clip(EPS,1-EPS))
            if auc > best_auc:
                best_auc = auc; best_cfg = {'n_components':n_components,'k_knn':k_knn}
                print(f"  nc={n_components} k={k_knn}: {auc:.4f}", flush=True)
    save_result(method, best_auc, best_cfg or {})
    all_new_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        results_data['best'] = {'method': method, 'loo_auc': best_auc, **(best_cfg or {})}
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Contrastive Ratio KNN
# For each species s, compute:
#   positive_sim = mean similarity to files with species s (weighted)
#   negative_sim = mean similarity to files WITHOUT species s
#   score = positive_sim / (positive_sim + negative_sim)  <- ratio
# This penalizes predictions when the test file looks similar to NEGATIVE examples
# ═══════════════════════════════════════════════════════════════════════════════
method = 'contrastive_ratio_knn'
if method not in tried_methods:
    print(f"\n[3] {method}...", flush=True)
    best_cfg = None; best_auc = 0
    for k in [5, 10]:
        for neg_weight in [0.5, 1.0, 2.0]:
            preds = np.zeros((n_files, n_species), np.float32)
            for i in range(n_files):
                tr = np.array([j for j in range(n_files) if j!=i])
                sims = (X_ref[[i]]@X_ref[tr].T).ravel()
                for s in range(n_species):
                    pos_mask = fl[tr, s] > 0.5
                    neg_mask = fl[tr, s] < 0.5
                    if pos_mask.sum() == 0:
                        preds[i,s] = 0; continue
                    # Top-k from positives
                    pos_sims = sims[pos_mask]
                    pos_top = np.sort(pos_sims)[-k:]
                    pos_score = np.exp(pos_top/0.2).mean()
                    # Top-k from negatives
                    if neg_mask.sum() > 0:
                        neg_sims = sims[neg_mask]
                        neg_top = np.sort(neg_sims)[-k:]
                        neg_score = np.exp(neg_top/0.2).mean()
                    else:
                        neg_score = 0
                    preds[i,s] = pos_score / (pos_score + neg_weight*neg_score + 1e-8)
            auc = macro_auc(file_labels, preds.clip(EPS,1-EPS))
            if auc > best_auc:
                best_auc = auc; best_cfg = {'k':k,'neg_weight':neg_weight}
                print(f"  k={k} nw={neg_weight}: {auc:.4f}", flush=True)
    save_result(method, best_auc, best_cfg or {})
    all_new_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        results_data['best'] = {'method': method, 'loo_auc': best_auc, **(best_cfg or {})}
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Temporal Decay KNN
# Weight neighbors by both embedding similarity AND temporal proximity
# Files from the same season / month get a bonus weight
# ═══════════════════════════════════════════════════════════════════════════════
method = 'temporal_decay_knn'
if method not in tried_methods:
    print(f"\n[4] {method}...", flush=True)
    # Extract file dates
    file_days = np.zeros(n_files, np.float32)
    for fi, fname in enumerate(file_list):
        m = re.match(r'BC2026_Train_\d+_S\d+_(\d{4})(\d{2})(\d{2})_', str(fname))
        if m:
            mo, dy = int(m.group(2)), int(m.group(3))
            dpm = [0,31,28,31,30,31,30,31,31,30,31,30,31]
            file_days[fi] = sum(dpm[:mo]) + dy
        else:
            file_days[fi] = 180
    # Circular day distance (accounting for year wrap)
    def day_dist(d1, d2):
        diff = np.abs(d1 - d2)
        return np.minimum(diff, 365 - diff)

    best_cfg = None; best_auc = 0
    for k in [5, 10]:
        for sigma_day in [15, 30, 60]:
            for w_temporal in [0.2, 0.5, 0.8]:
                preds = np.zeros((n_files, n_species), np.float32)
                for i in range(n_files):
                    tr = np.array([j for j in range(n_files) if j!=i])
                    sim_emb = (X_ref[[i]]@X_ref[tr].T).ravel()
                    # Temporal similarity: gaussian kernel on day distance
                    d_dist = day_dist(file_days[i], file_days[tr])
                    sim_time = np.exp(-d_dist**2 / (2*sigma_day**2))
                    # Combined similarity
                    sim_combined = (1-w_temporal)*sim_emb + w_temporal*sim_time
                    top = np.argsort(-sim_combined)[:k]
                    ls = sim_combined[top]/0.2; ls -= ls.max(); w=np.exp(ls); w/=w.sum()
                    preds[i] = (w[:,None]*fl[tr[top]]).sum(0)
                auc = macro_auc(file_labels, preds.clip(EPS,1-EPS))
                if auc > best_auc:
                    best_auc = auc; best_cfg = {'k':k,'sigma_day':sigma_day,'w_temporal':w_temporal}
                    print(f"  k={k} σ={sigma_day} wt={w_temporal}: {auc:.4f}", flush=True)
    save_result(method, best_auc, best_cfg or {})
    all_new_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        results_data['best'] = {'method': method, 'loo_auc': best_auc, **(best_cfg or {})}
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Site Prototype KNN
# Build one prototype embedding per geographic site (9 sites).
# Find nearest site prototype → then KNN within that site's files
# This respects the geographic clustering structure
# ═══════════════════════════════════════════════════════════════════════════════
method = 'site_prototype_knn'
if method not in tried_methods:
    print(f"\n[5] {method}...", flush=True)
    SITES=['S03','S08','S09','S13','S15','S18','S19','S22','S23']
    site2idx={s:i for i,s in enumerate(SITES)}
    file_sites = np.zeros(n_files, np.int32)
    for fi,fname in enumerate(file_list):
        m = re.match(r'BC2026_Train_\d+_(S\d+)_', str(fname))
        if m: file_sites[fi] = site2idx.get(m.group(1), 0)

    best_cfg = None; best_auc = 0
    for k_site in [1, 2, 3]:   # how many sites to use
        for k_file in [3, 5]:   # files per site
            for alpha_site in [0.3, 0.5, 0.7]:  # site vs file weight
                preds = np.zeros((n_files, n_species), np.float32)
                for i in range(n_files):
                    tr = np.array([j for j in range(n_files) if j!=i])
                    # Build site prototypes from training files
                    site_prototypes = {}
                    site_files_dict = {}
                    for si in range(len(SITES)):
                        mask = (file_sites[tr] == si)
                        if mask.sum() > 0:
                            site_prototypes[si] = X_ref[tr[mask]].mean(0)
                            site_files_dict[si] = tr[mask]
                    if not site_prototypes:
                        continue
                    # Find nearest sites by prototype similarity
                    proto_arr = np.array(list(site_prototypes.values())).astype(np.float32)
                    site_keys = list(site_prototypes.keys())
                    proto_arr /= np.linalg.norm(proto_arr, 1, keepdims=True)+1e-8
                    xi = X_ref[i:i+1]
                    site_sims = (xi @ proto_arr.T).ravel()
                    top_sites = np.argsort(-site_sims)[:k_site]
                    # For each top site, find nearest files
                    file_scores = []
                    file_label_list = []
                    for si_idx in top_sites:
                        si = site_keys[si_idx]
                        site_fl = site_files_dict[si]
                        fi_sims = (xi @ X_ref[site_fl].T).ravel()
                        top_f = np.argsort(-fi_sims)[:k_file]
                        w_s = site_sims[si_idx] * alpha_site
                        for fi_local, fi_sim in zip(top_f, fi_sims[top_f]):
                            file_scores.append(w_s * fi_sim)
                            file_label_list.append(fl[site_fl[fi_local]])
                    if file_scores:
                        ws = np.array(file_scores); ws = np.exp(ws/0.2); ws /= ws.sum()
                        preds[i] = (ws[:,None] * np.array(file_label_list)).sum(0)
                auc = macro_auc(file_labels, preds.clip(EPS,1-EPS))
                if auc > best_auc:
                    best_auc = auc
                    best_cfg = {'k_site':k_site,'k_file':k_file,'alpha_site':alpha_site}
                    print(f"  ks={k_site} kf={k_file} as={alpha_site}: {auc:.4f}", flush=True)
    save_result(method, best_auc, best_cfg or {})
    all_new_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        results_data['best'] = {'method': method, 'loo_auc': best_auc, **(best_cfg or {})}
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 6: Gaussian Process Regression
# GP with RBF kernel on PCA-reduced embeddings, per species
# GP provides calibrated uncertainty, potentially better than KNN voting
# ═══════════════════════════════════════════════════════════════════════════════
method = 'gp_regression'
if method not in tried_methods:
    print(f"\n[6] {method} (reduced species for speed)...", flush=True)
    # Use PCA to reduce X_ref to manageable size
    pca_gp = PCA(n_components=16, random_state=42).fit(X_ref)
    X_gp = pca_gp.transform(X_ref).astype(np.float32)
    # Only run on species with enough positive samples (speed)
    active_species = np.where(fl.sum(0) >= 3)[0]
    best_cfg = {'n_components': 16, 'kernel': 'rbf_white'}; best_auc = 0
    preds = np.zeros((n_files, n_species), np.float32)
    for s in active_species:
        y_s = fl[:, s]
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j!=i])
            X_tr = X_gp[tr]; y_tr = y_s[tr]; X_te = X_gp[[i]]
            if y_tr.sum() < 2:
                preds[i,s] = y_tr.mean(); continue
            try:
                kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
                gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)
                gpr.fit(X_tr, y_tr)
                pred_s = gpr.predict(X_te)[0]
                preds[i,s] = float(np.clip(pred_s, 0, 1))
            except:
                preds[i,s] = y_tr.mean()
    # Fill inactive species with KNN fallback
    for s in range(n_species):
        if s not in active_species:
            preds[:, s] = fl[:, s].mean()
    auc = macro_auc(file_labels, preds.clip(EPS,1-EPS))
    best_auc = auc
    print(f"  GP result: {auc:.4f}", flush=True)
    save_result(method, best_auc, best_cfg)
    all_new_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        results_data['best'] = {'method': method, 'loo_auc': best_auc, **best_cfg}
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 7: Logspace with LS2 + Co-occurrence correction
# Combine the best logspace formula with co-occurrence prior as post-processing
# P_final[i,s] = P_ls2[i,s] + alpha * sum_t(P_ls2[i,t] * cooc[t,s])
# ═══════════════════════════════════════════════════════════════════════════════
method = 'ls2_plus_cooc'
if method not in tried_methods:
    print(f"\n[7] {method}...", flush=True)
    # Recompute LS2 predictions
    k_win=1
    y_win_loo = np.zeros((n_files,n_species),np.float32)
    for i in range(n_files):
        te_s,te_e=int(file_start[i]),int(file_end[i]); X_te=emb_win_norm[te_s:te_e]
        tr_mask=(win_file_id!=i); X_tr=emb_win_norm[tr_mask]; tr_fi=win_file_id[tr_mask]
        sims=X_te@X_tr.T; top_idx=np.argsort(-sims,1)[:,:k_win]
        wp=np.zeros((te_e-te_s,n_species),np.float32)
        for wi in range(te_e-te_s):
            ww=sims[wi,top_idx[wi]].clip(0); ws=ww.sum(); ww=ww/ws if ws>1e-8 else np.ones(k_win)/k_win
            wp[wi]=(ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
        y_win_loo[i]=wp.mean(0)

    y_geo_loo = np.zeros((n_files,n_species),np.float32)
    for i in range(n_files):
        tr=np.array([j for j in range(n_files) if j!=i])
        sims=(X_ref[[i]]@X_ref[tr].T).ravel(); top=np.argsort(-sims)[:5]
        ls=sims[top]/0.2; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_geo_loo[i]=(w[:,None]*fl[tr[top]]).sum(0)

    y_ls2_base = 0.5*y_geo_loo + 0.5*y_win_loo

    # Co-occurrence matrix (LOO: exclude file i from co-occurrence stats)
    best_cfg = None; best_auc = 0
    for alpha_cooc in [0.05, 0.10, 0.20, 0.30]:
        preds = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j!=i])
            # Build co-occurrence from training files only
            C = (fl[tr].T @ fl[tr]) / (len(tr)+1e-6)
            np.fill_diagonal(C, 0)
            row_sum = fl[tr].sum(0) + 1e-6
            C_cond = C / row_sum[:, None]  # C_cond[s,t] = P(t|s in file)
            # Apply
            y_base = y_ls2_base[i]
            y_corrected = y_base + alpha_cooc * (y_base @ C_cond)
            preds[i] = y_corrected
        auc = macro_auc(file_labels, preds.clip(EPS,1-EPS))
        if auc > best_auc:
            best_auc = auc; best_cfg = {'alpha_cooc': alpha_cooc}
            print(f"  alpha={alpha_cooc}: {auc:.4f}", flush=True)
    save_result(method, best_auc, best_cfg or {})
    all_new_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        results_data['best'] = {'method': method, 'loo_auc': best_auc, **(best_cfg or {})}
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"BATCH 4 SUMMARY")
print(f"{'='*60}")
print(f"Previous best: {results_data.get('best',{}).get('loo_auc',0):.4f} vs original {0.9164:.4f}")
for method, auc, cfg in sorted(all_new_results, key=lambda x: -x[1]):
    marker = " *** NEW BEST ***" if auc > 0.9164 else ""
    print(f"  {method:35s}: {auc:.4f}{marker}")
print(f"\nCurrent best method: {results_data['best']['method']}")
print(f"Current best LOO-AUC: {results_data['best']['loo_auc']:.4f}")
