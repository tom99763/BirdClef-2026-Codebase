"""
Embed Prior Auto Loop - Batch 7
Current best EP-only LOO-AUC: 0.9171

Novel BREAKTHROUGH methods (structural innovations):

1. negative_aware_knn:     Explicit negative evidence: similar-but-absent neighbors
                           reduce confidence (like a contrastive KNN signal).
2. confidence_filtered_knn: Filter low-confidence training labels — Perch uncertain
                           samples (0.2 < prob < 0.8) may have wrong labels. Only
                           use high-confidence training files per species.
3. poly_knn:               Polynomial combination: instead of log-linear, use
                           sigmoid(a*logit + b1*log(geo) + b2*log(win) + b3*log(geo)*log(win))
                           Adding an interaction term between geo and win.
4. cluster_prototype_knn:  K-Means cluster training embeddings, find nearest
                           cluster to test point, use cluster's species distribution.
5. rknn_k3_win:            Try RKNN with k=3 (tighter mutual condition) + win_k1,
                           vs current k=5. Fewer but more reliable mutual neighbors.
6. power_mean_knn:         Power mean of neighbor labels instead of weighted average:
                           p_mean(labels, p) = (mean(labels^p))^(1/p)
                           Generalizes arithmetic (p=1) and geometric (p→0) means.
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

file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_prob_soft = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
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

# Load pkl
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

# Precompute pairwise similarities
sim_all = X_ref @ X_ref.T  # (66, 66)
np.fill_diagonal(sim_all, -np.inf)

# Precompute window KNN (binary, k=1) — reused
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
# Method 1: Negative-Aware KNN
# Standard KNN: prediction = weighted sum of POSITIVE evidence from neighbors.
# Novel: also subtract NEGATIVE evidence from similar files that DON'T have species s.
# For each species s:
#   score[s] = pos_signal[s] - gamma * neg_signal[s]
# where pos_signal = weighted sum of neighbor labels
# and neg_signal = weighted sum of (1 - neighbor_labels) for top-k similar files
# Intuition: If nearby files consistently lack species s, it's strong evidence of absence.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'negative_aware_knn'
if method not in tried_methods:
    print(f"\n[1] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    for k in [5, 7]:
        for T in [0.2, 0.3]:
            y_pos = np.zeros((n_files, n_species), np.float32)
            y_neg = np.zeros((n_files, n_species), np.float32)
            for i in range(n_files):
                tr = np.array([j for j in range(n_files) if j != i])
                sims = sim_all[i, tr]
                top = np.argsort(-sims)[:k]
                ls = sims[top]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
                y_pos[i] = (w[:, None] * fl_bin[tr[top]]).sum(0)
                y_neg[i] = (w[:, None] * (1 - fl_bin[tr[top]])).sum(0)

            for gamma in [0.1, 0.2, 0.3, 0.5]:
                y_adjusted = (y_pos - gamma * y_neg).clip(EPS)
                for w_geo in [0.40, 0.50, 0.60]:
                    y_blend = w_geo * y_adjusted + (1-w_geo) * y_win_k1
                    for A in [0.65, 0.70, 0.75]:
                        for B in [1.35, 1.45, 1.55, 1.65, 1.80]:
                            preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                            auc = macro_auc(file_labels, preds)
                            if auc > best_auc:
                                best_auc = auc
                                best_cfg = {'k': k, 'T': T, 'gamma': gamma,
                                            'w_geo': w_geo, 'A': A, 'B': B}
                                print(f"  k={k} T={T} γ={gamma} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Confidence-Filtered KNN
# Training labels from Perch have different reliability levels.
# High-confidence files (Perch prob > thresh or < 1-thresh) have reliable labels.
# Low-confidence files (0.2 < prob < 0.8) may have wrong labels.
# Novel: For each species, only use confident training files as neighbors.
# Separate thresholds for positive and negative filtering.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'confidence_filtered_knn'
if method not in tried_methods:
    print(f"\n[2] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    for conf_thr in [0.7, 0.8, 0.9]:
        # Per-species confidence mask: is this training file's label reliable for species s?
        # positive: prob > conf_thr → reliably present
        # negative: prob < 1 - conf_thr → reliably absent
        conf_mask = (file_prob_soft > conf_thr) | (file_prob_soft < (1 - conf_thr))  # (66, 234)

        for k in [5, 7]:
            for T in [0.2]:
                y_conf = np.zeros((n_files, n_species), np.float32)
                for i in range(n_files):
                    tr = np.array([j for j in range(n_files) if j != i])
                    sims = sim_all[i, tr]
                    for s in range(n_species):
                        # Filter training set to confident files for species s
                        conf_tr = tr[conf_mask[tr, s]]
                        if len(conf_tr) < 2:
                            # Fall back to full KNN if too few confident files
                            conf_tr = tr
                        sims_s = sim_all[i, conf_tr]
                        k_use = min(k, len(conf_tr))
                        top = np.argsort(-sims_s)[:k_use]
                        ls = sims_s[top]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
                        y_conf[i, s] = (w * fl_bin[conf_tr[top], s]).sum()

                for w_geo in [0.40, 0.50, 0.60]:
                    y_blend = w_geo * y_conf + (1-w_geo) * y_win_k1
                    for A in [0.65, 0.70, 0.75]:
                        for B in [1.35, 1.45, 1.55, 1.65]:
                            preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                            auc = macro_auc(file_labels, preds)
                            if auc > best_auc:
                                best_auc = auc
                                best_cfg = {'conf_thr': conf_thr, 'k': k, 'T': T,
                                            'w_geo': w_geo, 'A': A, 'B': B}
                                print(f"  thr={conf_thr} k={k} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Interaction-Term KNN (Poly KNN)
# Current: sigmoid(a*logit + b1*log(geo) + b2*log(win))
# Novel: Add interaction term between geo and win signals:
#   sigmoid(a*logit + b1*log(geo) + b2*log(win) + b3*log(geo)*log(win))
# The interaction captures cases where BOTH signals agree (multiplicative boost).
# ═══════════════════════════════════════════════════════════════════════════════
method = 'interaction_knn'
if method not in tried_methods:
    print(f"\n[3] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    # Geo KNN k=5
    y_geo_k5 = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = sim_all[i, tr]
        top = np.argsort(-sims)[:5]
        ls = sims[top]/0.2; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
        y_geo_k5[i] = (w[:, None] * fl_bin[tr[top]]).sum(0)

    log_geo = np.log(y_geo_k5.clip(EPS))
    log_win = np.log(y_win_k1.clip(EPS))
    interaction = log_geo * log_win  # element-wise product of log signals

    for A in [0.70, 0.80, 0.90, 0.95]:
        for B1 in [0.50, 0.70, 0.90, 1.10]:
            for B2 in [0.70, 0.90, 1.10, 1.30]:
                for B3 in [-0.5, -0.2, 0.0, 0.2, 0.5]:
                    preds = sigmoid(A * file_logit_max + B1 * log_geo + B2 * log_win + B3 * interaction)
                    auc = macro_auc(file_labels, preds)
                    if auc > best_auc:
                        best_auc = auc
                        best_cfg = {'A': A, 'B1': B1, 'B2': B2, 'B3': B3}
                        print(f"  A={A} B1={B1} B2={B2} B3={B3}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: RKNN k=3 + Win
# Test tighter reciprocal KNN with k=3 (more strict mutual condition).
# With k=3, both files need to be in each other's top-3 → stricter filtering.
# Hypothesis: fewer but more reliable mutual neighbors improves precision.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'rknn_k3_win'
if method not in tried_methods:
    print(f"\n[4] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    # Precompute top-k for each training file (for fast reciprocal check)
    sim_train = X_ref @ X_ref.T  # (66, 66)
    np.fill_diagonal(sim_train, -np.inf)
    for k in [3]:
        top_k_train = np.argsort(-sim_train, axis=1)[:, :k]  # (66, k)
        kth_sim_train = sim_train[np.arange(n_files), top_k_train[:, -1]]  # k-th sim

        y_rknn3 = np.zeros((n_files, n_species), np.float32)
        T = 0.2
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j != i])
            sims_i = sim_train[i, tr]
            top_i = np.argsort(-sims_i)[:k]  # top-k candidates
            mutual = []; mutual_sims = []
            for ti, tj in enumerate(tr[top_i]):
                # Reciprocal check: is test file i in tj's top-k training neighbors?
                if sims_i[top_i[ti]] >= kth_sim_train[tj]:
                    mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
            if len(mutual) == 0:
                # Fallback: use top-5 with softmax weights
                top5 = np.argsort(-sims_i)[:5]
                ls = sims_i[top5]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
                y_rknn3[i] = (w[:, None] * fl_bin[tr[top5]]).sum(0)
            else:
                ms = np.array(mutual_sims)
                ls = ms/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
                y_rknn3[i] = (w[:, None] * fl_bin[mutual]).sum(0)

        for wg in [0.30, 0.40, 0.45, 0.50]:
            yb = wg * y_rknn3 + (1-wg) * y_win_k1
            log_yb = np.log(yb.clip(EPS))
            for a in [0.85, 0.90, 0.95, 1.00]:
                for b in [1.40, 1.50, 1.55, 1.60, 1.70, 1.80]:
                    preds = sigmoid(a * file_logit_max + b * log_yb)
                    auc = macro_auc(file_labels, preds)
                    if auc > best_auc:
                        best_auc = auc
                        best_cfg = {'k': k, 'T': T, 'wg': wg, 'a': a, 'b': b}
                        print(f"  k={k} wg={wg} a={a} b={b}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Power Mean KNN
# Generalized mean of neighbor labels: p-mean(x) = (mean(x^p))^(1/p)
# p=1: arithmetic mean (current approach)
# p→0: geometric mean
# p=-1: harmonic mean (more conservative, avoids FPs)
# p=2: quadratic mean (rewards high-confidence agreement)
# For species detection, p < 1 (closer to geometric) may be better calibrated.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'power_mean_knn'
if method not in tried_methods:
    print(f"\n[5] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    k, T = 5, 0.2

    # Precompute neighbor weights and labels
    neighbor_labels = []  # (n_files, k, n_species)
    neighbor_weights = []  # (n_files, k)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = sim_all[i, tr]
        top = np.argsort(-sims)[:k]
        ls = sims[top]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
        neighbor_labels.append(fl_bin[tr[top]])  # (k, 234)
        neighbor_weights.append(w)  # (k,)
    neighbor_labels = np.array(neighbor_labels)   # (66, k, 234)
    neighbor_weights = np.array(neighbor_weights) # (66, k)

    for p in [-2.0, -1.0, -0.5, 0.3, 0.5, 0.7, 1.0, 2.0]:
        if abs(p) < 0.05:
            # Geometric mean
            y_pm = np.exp((neighbor_weights[:, :, None] * np.log(neighbor_labels.clip(EPS))).sum(1))
        else:
            # Generalized power mean
            labs_p = (neighbor_labels + EPS) ** p
            y_pm = ((neighbor_weights[:, :, None] * labs_p).sum(1)) ** (1.0/p)
            y_pm = y_pm.clip(EPS)

        for w_geo in [0.40, 0.50, 0.60]:
            y_blend = w_geo * y_pm + (1-w_geo) * y_win_k1
            for A in [0.65, 0.70, 0.75, 0.80]:
                for B in [1.30, 1.45, 1.55, 1.65, 1.80]:
                    preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                    auc = macro_auc(file_labels, preds)
                    if auc > best_auc:
                        best_auc = auc
                        best_cfg = {'p': p, 'k': k, 'T': T, 'w_geo': w_geo, 'A': A, 'B': B}
                        print(f"  p={p} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 6: RKNN k=5 + Asymmetric Logspace (Full Asymmetric RKNN)
# Combine RKNN k=5 reciprocal signal with win_k1 using ASYMMETRIC logspace:
#   sigmoid(a * base_logit + b1 * log(rknn_k5) + b2 * log(win_k1))
# Instead of blending rknn+win first then taking log,
# we take log separately and add them with separate coefficients.
# This is the asymmetric_ls2 generalization applied to RKNN.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'rknn_k5_asym'
if method not in tried_methods:
    print(f"\n[6] rknn_k5_asym (EP-only)...", flush=True)
    best_auc = 0; best_cfg = None

    # Load rknn pkl for X_ref_rknn (same X_ref)
    # Recompute RKNN k=5 with current X_ref
    k = 5; T = 0.2
    sim_train = X_ref @ X_ref.T
    np.fill_diagonal(sim_train, -np.inf)
    top_k_train = np.argsort(-sim_train, axis=1)[:, :k]
    kth_sim_train = sim_train[np.arange(n_files), top_k_train[:, -1]]

    y_rknn5 = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims_i = sim_train[i, tr]
        top_i = np.argsort(-sims_i)[:k]
        mutual = []; mutual_sims = []
        for ti, tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]] >= kth_sim_train[tj]:
                mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
        if len(mutual) == 0:
            top5 = np.argsort(-sims_i)[:5]
            ls = sims_i[top5]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y_rknn5[i] = (w[:, None] * fl_bin[tr[top5]]).sum(0)
        else:
            ms = np.array(mutual_sims)
            ls = ms/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y_rknn5[i] = (w[:, None] * fl_bin[mutual]).sum(0)

    log_rknn = np.log(y_rknn5.clip(EPS))
    log_win = np.log(y_win_k1.clip(EPS))

    # EP-only sweep
    for A in [0.65, 0.70, 0.75, 0.80]:
        for B1 in [0.50, 0.70, 0.90, 1.10, 1.30]:
            for B2 in [0.30, 0.50, 0.70, 0.90, 1.10]:
                preds = sigmoid(A * file_logit_max + B1 * log_rknn + B2 * log_win)
                auc = macro_auc(file_labels, preds)
                if auc > best_auc:
                    best_auc = auc
                    best_cfg = {'k': k, 'T': T, 'A': A, 'B1': B1, 'B2': B2}
                    print(f"  A={A} B1={B1} B2={B2}: {auc:.4f}", flush=True)

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
original_best = 0.9171

print(f"\n{'='*60}")
print(f"BATCH 7 SUMMARY (EP-only LOO-AUC)")
print(f"{'='*60}")
print(f"Original best: {original_best:.4f}")
for method, auc, cfg in sorted(all_results, key=lambda x: -x[1]):
    marker = " *** NEW BEST ***" if auc > original_best else ""
    print(f"  {method:35s}: {auc:.4f}{marker}")
print(f"\nCurrent best: {results_data['best']['method']} = {current_best:.4f}")
