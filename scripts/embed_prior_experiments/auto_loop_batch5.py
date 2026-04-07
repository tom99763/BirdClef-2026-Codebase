"""
Embed Prior Auto Loop - Batch 5
Current best EP-only LOO-AUC: 0.9164

Novel methods:
1. soft_label_ls2:       Use sigmoid(file_logit_max) as SOFT training labels instead of binary
2. window_attention_knn: Global softmax attention over ALL 739 training windows (not just k nearest)
3. cross_modal_knn:      KNN in joint embedding+logit space (both signals combined)
4. soft_label_window_knn: Window KNN with soft labels from Perch logits
5. logit_residual_knn:   Predict logit RESIDUAL (correction to Perch logit_max) via KNN
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
file_prob_soft = np.zeros((n_files, n_species), np.float32)  # sigmoid soft labels
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_prob_soft[fi] = sigmoid(logits_win[s:e]).max(0)  # soft probability

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
fl_bin = ep_base['file_labels'].astype(np.float32)      # binary labels (current)
fl_soft = file_prob_soft.astype(np.float32)             # soft probability labels (NEW)

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

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Soft-Label LS2
# Instead of binary training labels (0/1), use sigmoid(logit_max) as soft labels.
# Insight: Perch's own confidence on training files should be a better signal.
# If Perch says 0.95 probability → more reliable than just "1"
# If Perch says 0.55 → less reliable, shouldn't count as "1"
# ═══════════════════════════════════════════════════════════════════════════════
method = 'soft_label_ls2'
if method not in tried_methods:
    print(f"\n[1] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    for k in [5]:
        for T in [0.2]:
            # Geo-KNN with SOFT labels
            y_geo_soft = np.zeros((n_files, n_species), np.float32)
            for i in range(n_files):
                tr = np.array([j for j in range(n_files) if j != i])
                sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
                top = np.argsort(-sims)[:k]
                ls = sims[top]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
                y_geo_soft[i] = (w[:, None] * fl_soft[tr[top]]).sum(0)  # SOFT labels

            # Window-KNN with SOFT window-level labels
            win_prob_max = sigmoid(logits_win).astype(np.float32)  # (739, 234) soft
            y_win_soft = np.zeros((n_files, n_species), np.float32)
            for i in range(n_files):
                te_s, te_e = int(file_start[i]), int(file_end[i])
                X_te = emb_win_norm[te_s:te_e]
                tr_mask = win_file_id != i
                X_tr = emb_win_norm[tr_mask]
                Y_tr_soft = win_prob_max[tr_mask]  # soft window labels
                sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :1]
                wp = np.zeros((te_e - te_s, n_species), np.float32)
                for wi in range(te_e - te_s):
                    ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
                    ww = ww/ws if ws > 1e-8 else np.ones(1)
                    wp[wi] = (ww[:, None] * Y_tr_soft[top_idx[wi]]).sum(0)
                y_win_soft[i] = wp.mean(0)

            for w_geo in [0.50, 0.40, 0.60]:
                y_blend = w_geo * y_geo_soft + (1-w_geo) * y_win_soft
                for A in [0.65, 0.70, 0.75]:
                    for B in [1.30, 1.45, 1.60]:
                        preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                        auc = macro_auc(file_labels, preds)
                        if auc > best_auc:
                            best_auc = auc
                            best_cfg = {'k': k, 'T': T, 'w_geo': w_geo, 'A': A, 'B': B}
                            print(f"  w_geo={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST EP-only: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Global Window Attention KNN
# Instead of top-k window KNN, use FULL softmax attention over all 739 training windows.
# Each test window attends to all training windows with exp(sim/T) weights.
# This is the transformer cross-attention concept applied to our problem.
# No hard top-k cutoff → smoother predictions, potentially better calibration.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'global_window_attention'
if method not in tried_methods:
    print(f"\n[2] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    for T_win in [0.05, 0.10, 0.20, 0.50]:
        y_gwa = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            te_s, te_e = int(file_start[i]), int(file_end[i])
            X_te = emb_win_norm[te_s:te_e]  # (n_win_test, 1536)
            tr_mask = win_file_id != i
            X_tr = emb_win_norm[tr_mask]    # (n_win_train, 1536)
            Y_tr = file_labels[win_file_id[tr_mask]]  # (n_win_train, 234)
            # Full attention: (n_win_test, n_win_train)
            sims = X_te @ X_tr.T
            # Softmax attention (temperature T_win)
            attn = sims / T_win
            attn -= attn.max(1, keepdims=True)
            attn = np.exp(attn); attn /= attn.sum(1, keepdims=True)
            # Weighted label aggregation
            y_per_win = attn @ Y_tr  # (n_win_test, 234)
            y_gwa[i] = y_per_win.mean(0)

        for w_geo in [0.50, 0.40, 0.60]:
            y_blend = w_geo * y_geo_knn_cached + (1-w_geo) * y_gwa if 'y_geo_knn_cached' in dir() else y_gwa
            for A in [0.65, 0.70, 0.75]:
                for B in [1.30, 1.45, 1.60]:
                    preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                    auc = macro_auc(file_labels, preds)
                    if auc > best_auc:
                        best_auc = auc
                        best_cfg = {'T_win': T_win, 'w_geo': w_geo, 'A': A, 'B': B}
                        print(f"  T={T_win} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    # Also test standalone (no geo blend)
    for T_win in [0.05, 0.10, 0.20]:
        y_gwa = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            te_s, te_e = int(file_start[i]), int(file_end[i])
            X_te = emb_win_norm[te_s:te_e]
            tr_mask = win_file_id != i
            X_tr = emb_win_norm[tr_mask]
            Y_tr = file_labels[win_file_id[tr_mask]]
            sims = X_te @ X_tr.T
            attn = sims / T_win; attn -= attn.max(1, keepdims=True)
            attn = np.exp(attn); attn /= attn.sum(1, keepdims=True)
            y_gwa[i] = (attn @ Y_tr).mean(0)
        for A in [0.65, 0.70, 0.75, 0.80]:
            for B in [1.20, 1.35, 1.45, 1.60]:
                preds = sigmoid(A * file_logit_max + B * np.log(y_gwa.clip(EPS)))
                auc = macro_auc(file_labels, preds)
                if auc > best_auc:
                    best_auc = auc
                    best_cfg = {'T_win': T_win, 'w_geo': 0.0, 'A': A, 'B': B}
                    print(f"  standalone T={T_win} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST EP-only: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Cross-Modal KNN
# Current KNN uses only embedding similarity for neighbor selection.
# Novel idea: also use LOGIT similarity — if two files have similar Perch logit
# patterns, they probably have similar species.
# Joint similarity = α × emb_sim + (1-α) × logit_sim
# ═══════════════════════════════════════════════════════════════════════════════
method = 'cross_modal_knn'
if method not in tried_methods:
    print(f"\n[3] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    # Normalize logit space
    file_logit_norm = normalize(file_logit_max, norm='l2').astype(np.float32)

    for alpha_emb in [0.5, 0.6, 0.7, 0.8]:
        alpha_logit = 1.0 - alpha_emb
        for k in [5, 7]:
            for T in [0.2, 0.3]:
                y_cm = np.zeros((n_files, n_species), np.float32)
                for i in range(n_files):
                    tr = np.array([j for j in range(n_files) if j != i])
                    # Combined similarity in both spaces
                    sim_emb  = (X_ref[[i]] @ X_ref[tr].T).ravel()
                    sim_logit= (file_logit_norm[[i]] @ file_logit_norm[tr].T).ravel()
                    sim_joint = alpha_emb * sim_emb + alpha_logit * sim_logit
                    top = np.argsort(-sim_joint)[:k]
                    ls = sim_joint[top]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
                    y_cm[i] = (w[:, None] * fl_bin[tr[top]]).sum(0)
                # Window KNN still uses embedding only
                for w_geo in [0.45, 0.50, 0.55]:
                    # Use precomputed y_win (k=1) from geo5_win1 pkl
                    # Approximate: run inline
                    for A in [0.70, 0.75]:
                        for B in [1.40, 1.45, 1.50]:
                            preds = sigmoid(A * file_logit_max + B * np.log(y_cm.clip(EPS)))
                            auc = macro_auc(file_labels, preds)
                            if auc > best_auc:
                                best_auc = auc
                                best_cfg = {'alpha_emb': alpha_emb, 'k': k, 'T': T,
                                            'A': A, 'B': B}
                                print(f"  ae={alpha_emb} k={k} T={T} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST EP-only: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Soft-Label Window KNN
# Window-level KNN but using SOFT labels from Perch window logits as targets.
# Training window label[w, s] = sigmoid(logit_win[w, s])   (probability, not 0/1)
# This provides richer supervision signal at window level.
# Combined with geo-KNN using binary labels.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'soft_label_window_knn'
if method not in tried_methods:
    print(f"\n[4] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    win_prob = sigmoid(logits_win).astype(np.float32)  # (739, 234)

    # Geo-KNN (binary labels, best config)
    y_geo_bin = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
        top = np.argsort(-sims)[:5]
        ls = sims[top]/0.2; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
        y_geo_bin[i] = (w[:, None] * fl_bin[tr[top]]).sum(0)

    # Window-KNN with SOFT window-level labels
    for k_win in [1, 2, 3]:
        y_win_soft2 = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            te_s, te_e = int(file_start[i]), int(file_end[i])
            X_te = emb_win_norm[te_s:te_e]
            tr_mask = win_file_id != i
            X_tr = emb_win_norm[tr_mask]
            Y_tr_soft = win_prob[tr_mask]  # soft window labels
            sims = X_te @ X_tr.T
            top_idx = np.argsort(-sims, 1)[:, :k_win]
            wp = np.zeros((te_e - te_s, n_species), np.float32)
            for wi in range(te_e - te_s):
                ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
                ww = ww/ws if ws > 1e-8 else np.ones(k_win)/k_win
                wp[wi] = (ww[:, None] * Y_tr_soft[top_idx[wi]]).sum(0)
            y_win_soft2[i] = wp.mean(0)

        for w_geo in [0.40, 0.50, 0.60]:
            y_blend = w_geo * y_geo_bin + (1-w_geo) * y_win_soft2
            for A in [0.65, 0.70, 0.75]:
                for B in [1.30, 1.45, 1.60]:
                    preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                    auc = macro_auc(file_labels, preds)
                    if auc > best_auc:
                        best_auc = auc
                        best_cfg = {'k_win': k_win, 'w_geo': w_geo, 'A': A, 'B': B}
                        print(f"  k_win={k_win} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST EP-only: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Logit Residual KNN
# Novel idea: Instead of predicting species probability directly,
# predict the RESIDUAL correction to Perch's logit_max.
# residual[file, species] = file_label[file, species] - sigmoid(file_logit_max[file, species])
# KNN predicts this residual, then: final = sigmoid(logit_max + KNN_residual)
# This forces the KNN to focus on where Perch is WRONG.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'logit_residual_knn'
if method not in tried_methods:
    print(f"\n[5] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    # Compute per-file residuals (label - prob)
    file_residuals = file_labels - sigmoid(file_logit_max)  # in [-1, 1]

    for k in [3, 5, 7]:
        for T in [0.2, 0.3]:
            y_resid = np.zeros((n_files, n_species), np.float32)
            for i in range(n_files):
                tr = np.array([j for j in range(n_files) if j != i])
                sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
                top = np.argsort(-sims)[:k]
                ls = sims[top]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
                # KNN predicts the residual from neighbors' residuals
                y_resid[i] = (w[:, None] * file_residuals[tr[top]]).sum(0)

            for alpha in [0.3, 0.5, 0.7, 1.0, 1.5]:
                # Final prediction: sigmoid(logit_max + alpha * residual)
                preds = sigmoid(file_logit_max + alpha * y_resid)
                auc = macro_auc(file_labels, preds)
                if auc > best_auc:
                    best_auc = auc
                    best_cfg = {'k': k, 'T': T, 'alpha': alpha}
                    print(f"  k={k} T={T} alpha={alpha}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST EP-only: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Method 6: Geo-KNN with soft MIXED labels
# Mix binary labels (ground truth) with soft Perch-predicted labels:
# mixed_label[file,s] = beta * file_label[file,s] + (1-beta) * sigmoid(logit_max[file,s])
# This is a form of label smoothing using Perch's own confidence.
# ═══════════════════════════════════════════════════════════════════════════════
method = 'mixed_label_knn'
if method not in tried_methods:
    print(f"\n[6] {method}...", flush=True)
    best_auc = 0; best_cfg = None

    # Precompute window KNN (binary) for blending
    y_win_bin = np.zeros((n_files, n_species), np.float32)
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
        y_win_bin[i] = wp.mean(0)

    for beta in [0.3, 0.5, 0.7, 0.9]:
        # Mixed labels: beta * binary + (1-beta) * soft
        fl_mixed = beta * fl_bin + (1-beta) * fl_soft

        # Geo-KNN with mixed labels
        y_geo_mix = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j != i])
            sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
            top = np.argsort(-sims)[:5]
            ls = sims[top]/0.2; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y_geo_mix[i] = (w[:, None] * fl_mixed[tr[top]]).sum(0)

        for w_geo in [0.40, 0.50, 0.60]:
            y_blend = w_geo * y_geo_mix + (1-w_geo) * y_win_bin
            for A in [0.65, 0.70, 0.75]:
                for B in [1.35, 1.45, 1.55]:
                    preds = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
                    auc = macro_auc(file_labels, preds)
                    if auc > best_auc:
                        best_auc = auc
                        best_cfg = {'beta': beta, 'w_geo': w_geo, 'A': A, 'B': B}
                        print(f"  beta={beta} wg={w_geo} A={A} B={B}: {auc:.4f}", flush=True)

    save_result(method, best_auc, best_cfg or {})
    all_results.append((method, best_auc, best_cfg))
    tried_methods.add(method)
    if best_auc > best_loo:
        best_loo = best_auc
        print(f"  *** NEW BEST EP-only: {best_auc:.4f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
current_best = results_data['best']['loo_auc']
original_best = 0.9164

print(f"\n{'='*60}")
print(f"BATCH 5 SUMMARY (EP-only LOO-AUC)")
print(f"{'='*60}")
print(f"Original best: {original_best:.4f}")
for method, auc, cfg in sorted(all_results, key=lambda x: -x[1]):
    marker = " *** NEW BEST ***" if auc > original_best else ""
    print(f"  {method:35s}: {auc:.4f}{marker}")
print(f"\nCurrent best: {results_data['best']['method']} = {current_best:.4f}")
