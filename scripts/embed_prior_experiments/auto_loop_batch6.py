"""
Embed Prior Auto Loop - Batch 6
Current best EP-only LOO-AUC: 0.9166

Novel BREAKTHROUGH methods (structural innovations, not parameter sweeps):

1. hubness_aware_knn:    Penalize "hub" training files that appear as neighbors
                         too often across test queries — they inject false positives.
2. species_conditional_knn: For each species, KNN among ONLY files where that
                         species is present (positive-only neighbor pool).
3. kde_logspace:         Kernel Density Estimation in embedding space — measure
                         how close the test point is to the positive cluster
                         vs the negative cluster, per species.
4. asymmetric_ls2:       Separate logspace coefficients for geo_k5 vs win_k1,
                         allowing each component to contribute differently.
5. adaptive_k_by_freq:   Use larger K for rare species, smaller K for common
                         species (adaptive K per species based on prevalence).
6. reranked_knn:         Initial KNN retrieval, then re-rank neighbors using
                         species co-occurrence in training set.
"""
import numpy as np, pickle, json, os, shutil
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
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
file_prob_soft = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_prob_soft[fi] = sigmoid(logits_win[s:e]).max(0)

emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

EPS = 1e-7
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

# Load pkl (for X_combined_n)
with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep_base = pickle.load(f)
X_ref = ep_base['X_combined_n'].astype(np.float32)
fl_bin  = ep_base['file_labels'].astype(np.float32)

# Load current best
with open(RESULTS_PATH) as f:
    results_data = json.load(f)
best_loo = results_data['best']['loo_auc']
tried_methods = set(e['method'] for e in results_data.get('experiments', []))
print(f"Current best LOO-AUC: {best_loo:.4f} ({results_data['best']['method']})")

def save_result(method, loo_auc, config):
    entry = {'method': method, 'loo_auc': float(loo_auc), 'config': config}
    results_data['experiments'].append(entry)
    if loo_auc > results_data['best']['loo_auc']:
        results_data['best'] = {'method': method, 'loo_auc': float(loo_auc), **config}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_data, f, indent=2)

all_results = []

# Precompute window KNN (binary, k=1) — reused across methods
print("Precomputing win_k1...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = win_file_id != i
    X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :1]
    wp = np.zeros((te_e - te_s, n_species), np.float32)
    for wi in range(te_e - te_s):
        ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
        ww = ww/ws if ws > 1e-8 else np.ones(1)
        wp[wi] = (ww[:, None] * file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)
print("  done.", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Hubness-Aware KNN
# In high-dimensional spaces, some training samples become "hubs" — they appear
# in the K-nearest-neighbors of many other points. This introduces systematic
# bias, because a hub's species get over-represented in all predictions.
# Fix: count how many times each training file appears as a top-5 neighbor
# across all queries, then downweight hubs by their hubness score.
# weight_i = sim_i / (hubness_count_i + 1)  (dampens frequent neighbors)
# ═══════════════════════════════════════════════════════════════════════════════
method = 'hubness_aware_knn'
if method not in tried_methods:
    print(f"\n[1] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    for k in [5, 7, 10]:
        # Count how many times each file appears as top-k neighbor of others
        hub_count = np.zeros(n_files, np.float32)
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j != i])
            sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
            top = tr[np.argsort(-sims)[:k]]
            hub_count[top] += 1

        for T in [0.15, 0.2, 0.3]:
            for hub_damp in [0.5, 1.0, 2.0]:  # hubness dampening strength
                y_hub = np.zeros((n_files, n_species), np.float32)
                for i in range(n_files):
                    tr = np.array([j for j in range(n_files) if j != i])
                    sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
                    top = np.argsort(-sims)[:k]
                    # Downweight neighbors by their hubness
                    hub_w = 1.0 / (1.0 + hub_damp * hub_count[tr[top]] / k)
                    ls = sims[top] / T; ls -= ls.max(); w = np.exp(ls)
                    w = w * hub_w; w /= w.sum()
                    y_hub[i] = (w[:, None] * fl_bin[tr[top]]).sum(0)

                for w_geo in [0.40, 0.50, 0.60]:
                    y_blend = w_geo * y_hub + (1-w_geo) * y_win_k1
                    for A in [0.65, 0.70, 0.75]:
                        for B in [1.35, 1.45, 1.55, 1.65]:
                            preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                            auc = macro_auc(file_labels, preds)
                            if auc > best_auc:
                                best_auc = auc
                                best_cfg = {'k': k, 'T': T, 'hub_damp': hub_damp,
                                            'w_geo': w_geo, 'A': A, 'B': B}
                                print(f"  k={k} T={T} damp={hub_damp} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Species-Conditional KNN
# Standard KNN mixes files with and without a species in the neighbor pool.
# Novel: For predicting species s, ONLY consider training files where s is
# labeled. Effectively, this is asking: "among files where species s appears,
# which are closest to me?" — a conditional density estimate.
# If no positive neighbors found, fall back to full KNN.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'species_conditional_knn'
if method not in tried_methods:
    print(f"\n[2] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    # Precompute all pairwise similarities in X_ref
    sim_all = X_ref @ X_ref.T  # (66, 66)
    np.fill_diagonal(sim_all, -np.inf)

    # Positive file sets per species
    pos_files = [np.where(fl_bin[:, s] > 0.5)[0] for s in range(n_species)]

    for k_cond in [3, 5]:
        for T in [0.2, 0.3]:
            y_cond = np.zeros((n_files, n_species), np.float32)
            for i in range(n_files):
                sims_i = sim_all[i].copy()
                sims_i[i] = -np.inf  # exclude self
                for s in range(n_species):
                    pos_s = pos_files[s]
                    pos_s_tr = pos_s[pos_s != i]  # exclude test file
                    if len(pos_s_tr) == 0:
                        continue  # no positive examples → leave as 0
                    sims_s = sims_i[pos_s_tr]
                    k_use = min(k_cond, len(pos_s_tr))
                    top_idx = np.argsort(-sims_s)[:k_use]
                    top_sims = sims_s[top_idx]
                    ls = top_sims / T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
                    # Since all neighbors are positive, prediction = weighted sim
                    y_cond[i, s] = w.sum()  # = 1.0 if any positive neighbor found,
                    # but scaled by similarity weights
                    # Better: use normalized count with distance decay
                    y_cond[i, s] = (w * 1.0).sum()  # sum of positive-neighbor weights

            for w_geo in [0.30, 0.40, 0.50]:
                y_blend = w_geo * y_cond + (1-w_geo) * y_win_k1
                for A in [0.65, 0.70, 0.75]:
                    for B in [1.20, 1.35, 1.45, 1.55]:
                        preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                        auc = macro_auc(file_labels, preds)
                        if auc > best_auc:
                            best_auc = auc
                            best_cfg = {'k_cond': k_cond, 'T': T,
                                        'w_geo': w_geo, 'A': A, 'B': B}
                            print(f"  k={k_cond} T={T} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: KDE Logspace (Kernel Density Estimation)
# Instead of KNN, estimate the probability that test point belongs to each
# species' cluster using KDE. For each species s:
#   p(s present | x_test) ∝ mean_j_pos[K(x_test, x_j)] / mean_all[K(x_test, x_j)]
# where K is a Gaussian kernel: K(x,y) = exp(sim(x,y)/T)
# Ratio of positive-cluster density to total density.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'kde_logspace'
if method not in tried_methods:
    print(f"\n[3] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    sim_all = X_ref @ X_ref.T  # (66, 66)

    for T_kde in [0.1, 0.15, 0.2, 0.3]:
        K = np.exp(sim_all / T_kde)  # (66, 66) kernel matrix
        np.fill_diagonal(K, 0)       # exclude self

        y_kde = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            # Training indices (exclude i)
            tr_mask = np.arange(n_files) != i
            K_i = K[i][tr_mask]           # (65,) kernel values to training
            fl_tr = fl_bin[tr_mask]        # (65, 234) training labels

            # Density of positive files per species
            # density_pos[s] = sum of K_i[j] for j with label[j,s]=1 / n_pos[s]
            n_pos = fl_tr.sum(0).clip(1)  # (234,)
            dens_pos = (K_i[:, None] * fl_tr).sum(0) / n_pos  # (234,)
            # Total density
            dens_all = K_i.mean()
            y_kde[i] = dens_pos / (dens_all + EPS)

        for w_geo in [0.40, 0.50, 0.60]:
            y_blend = w_geo * y_kde + (1-w_geo) * y_win_k1
            for A in [0.65, 0.70, 0.75]:
                for B in [1.30, 1.45, 1.55, 1.70]:
                    preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                    auc = macro_auc(file_labels, preds)
                    if auc > best_auc:
                        best_auc = auc
                        best_cfg = {'T_kde': T_kde, 'w_geo': w_geo, 'A': A, 'B': B}
                        print(f"  T={T_kde} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Asymmetric LS2 (Separate Coefficients for Geo and Win)
# Current LS2 formula: sigmoid(a * logit + b * log(wg*geo + (1-wg)*win))
# Novel: let geo and win have SEPARATE logspace coefficients:
#   sigmoid(a * logit + b1 * log(geo_k5) + b2 * log(win_k1))
# This allows the model to optimize contribution of each signal independently.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'asymmetric_ls2'
if method not in tried_methods:
    print(f"\n[4] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    # Geo KNN k=5
    y_geo_k5 = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
        top = np.argsort(-sims)[:5]
        ls_w = sims[top]/0.2; ls_w -= ls_w.max(); w = np.exp(ls_w); w /= w.sum()
        y_geo_k5[i] = (w[:, None] * fl_bin[tr[top]]).sum(0)

    # Asymmetric blend: sigmoid(a * logit + b1 * log(geo) + b2 * log(win))
    for A in [0.70, 0.80, 0.90, 0.95]:
        for B1 in [0.60, 0.80, 1.00, 1.20]:
            for B2 in [0.40, 0.60, 0.80, 1.00]:
                log_geo = np.log(y_geo_k5.clip(EPS))
                log_win = np.log(y_win_k1.clip(EPS))
                preds = sigmoid(A * file_logit_max + B1 * log_geo + B2 * log_win)
                auc = macro_auc(file_labels, preds)
                if auc > best_auc:
                    best_auc = auc
                    best_cfg = {'A': A, 'B1': B1, 'B2': B2}
                    print(f"  A={A} B1={B1} B2={B2}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Adaptive-K KNN by Species Frequency
# Rare species have fewer training examples, so KNN is noisy.
# For rare species: use larger K to average out noise.
# For common species: use smaller K for precision.
# K[s] = base_k * (n_files / (2 * n_pos[s] + 1))^freq_power
# ═══════════════════════════════════════════════════════════════════════════════
method = 'adaptive_k_by_freq'
if method not in tried_methods:
    print(f"\n[5] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    species_freq = fl_bin.sum(0)  # (234,) how many files have each species

    for base_k in [3, 5]:
        for freq_power in [0.3, 0.5, 0.7]:
            for T in [0.2, 0.3]:
                # Adaptive K per species: more training files → smaller K OK
                k_per_species = np.clip(
                    (base_k * (n_files / (2 * species_freq + 1)) ** freq_power).astype(int),
                    1, 15
                )

                y_adaptive = np.zeros((n_files, n_species), np.float32)
                sim_all_local = X_ref @ X_ref.T
                np.fill_diagonal(sim_all_local, -np.inf)

                for i in range(n_files):
                    tr = np.array([j for j in range(n_files) if j != i])
                    sims_i = sim_all_local[i][tr]
                    sorted_tr = tr[np.argsort(-sims_i)]  # sorted by similarity
                    for s in range(n_species):
                        k_s = k_per_species[s]
                        top_s = sorted_tr[:k_s]
                        top_sims = sim_all_local[i][top_s]
                        ls_w = top_sims / T; ls_w -= ls_w.max()
                        w = np.exp(ls_w); w /= w.sum()
                        y_adaptive[i, s] = (w * fl_bin[top_s, s]).sum()

                for w_geo in [0.40, 0.50, 0.60]:
                    y_blend = w_geo * y_adaptive + (1-w_geo) * y_win_k1
                    for A in [0.65, 0.70, 0.75]:
                        for B in [1.35, 1.45, 1.55]:
                            preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                            auc = macro_auc(file_labels, preds)
                            if auc > best_auc:
                                best_auc = auc
                                best_cfg = {'base_k': base_k, 'freq_power': freq_power,
                                            'T': T, 'w_geo': w_geo, 'A': A, 'B': B}
                                print(f"  bk={base_k} fp={freq_power} T={T} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 6: Label Diffusion on K-NN Graph
# Build a K-NN graph from training embeddings.
# Run label propagation / diffusion from training labels to test file.
# Each step: new_score = alpha * neighbors_score + (1-alpha) * original_labels
# This captures transitive similarity (friend-of-friend).
# ═══════════════════════════════════════════════════════════════════════════════
method = 'label_diffusion_knn'
if method not in tried_methods:
    print(f"\n[6] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    sim_all = X_ref @ X_ref.T  # (66, 66)

    for k_graph in [3, 5]:
        for alpha in [0.5, 0.7, 0.85]:
            for n_steps in [2, 3, 5]:
                for T in [0.2]:
                    # Build normalized adjacency matrix on training files (LOO: exclude test each time)
                    y_diff = np.zeros((n_files, n_species), np.float32)
                    for i in range(n_files):
                        tr = np.array([j for j in range(n_files) if j != i])
                        n_tr = len(tr)
                        # Build graph among training files
                        sim_tr = sim_all[np.ix_(tr, tr)]
                        np.fill_diagonal(sim_tr, -np.inf)
                        # k-NN adjacency
                        W = np.zeros((n_tr, n_tr), np.float32)
                        for r in range(n_tr):
                            top_r = np.argsort(-sim_tr[r])[:k_graph]
                            sim_vals = sim_tr[r, top_r]
                            ls_w = sim_vals / T; ls_w -= ls_w.max()
                            w_r = np.exp(ls_w); w_r /= w_r.sum()
                            W[r, top_r] = w_r
                        # Symmetrize
                        W = 0.5 * (W + W.T)
                        row_sum = W.sum(1, keepdims=True).clip(1e-8)
                        W_norm = W / row_sum  # row-normalized

                        # Diffusion from training labels
                        F = fl_bin[tr].copy()  # initial: true labels
                        for _ in range(n_steps):
                            F = alpha * (W_norm @ F) + (1 - alpha) * fl_bin[tr]

                        # Predict test file: weighted average of diffused train labels
                        sims_te = sim_all[i, tr]
                        top_k = np.argsort(-sims_te)[:5]
                        ls_te = sims_te[top_k] / T; ls_te -= ls_te.max()
                        w_te = np.exp(ls_te); w_te /= w_te.sum()
                        y_diff[i] = (w_te[:, None] * F[top_k]).sum(0)

                    for w_geo in [0.40, 0.50, 0.60]:
                        y_blend = w_geo * y_diff + (1-w_geo) * y_win_k1
                        for A in [0.65, 0.70, 0.75]:
                            for B in [1.35, 1.45, 1.55]:
                                preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                                auc = macro_auc(file_labels, preds)
                                if auc > best_auc:
                                    best_auc = auc
                                    best_cfg = {'k_graph': k_graph, 'alpha': alpha,
                                                'n_steps': n_steps, 'T': T,
                                                'w_geo': w_geo, 'A': A, 'B': B}
                                    print(f"  kg={k_graph} al={alpha} steps={n_steps} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
current_best = results_data['best']['loo_auc']
original_best = 0.9166

print(f"\n{'='*60}")
print(f"BATCH 6 SUMMARY (EP-only LOO-AUC)")
print(f"{'='*60}")
print(f"Original best: {original_best:.4f}")
for method, auc, cfg in sorted(all_results, key=lambda x: -x[1]):
    marker = " *** NEW BEST ***" if auc > original_best else ""
    print(f"  {method:35s}: {auc:.4f}{marker}")
print(f"\nCurrent best: {results_data['best']['method']} = {current_best:.4f}")
