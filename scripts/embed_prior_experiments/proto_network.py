"""
Method 1: Prototypical Network (training-free)
- Per-species: compute mean embedding of positive windows as prototype
- Test window score = cosine_similarity(window_emb, prototype)
- File score = mean of window scores across all test windows
- LOO-CV at FILE level (leave one file out, not one window)

Variants:
  A) window-level proto: use all positive WINDOWS across train files
  B) file-level proto: use mean embedding per file, then average per species
  C) weighted proto: weight each file's contribution by its label confidence
  D) L2-distance-based: exp(-||x - proto||^2 / 2*sigma^2)
  E) top-k proto: only use the k most similar train windows as prototype
"""

import numpy as np
import json
import pickle
import sys
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH    = '/home/lab/BirdClef-2026-Codebase/outputs/perch_labeled_ss.npz'
RESULTS_PATH = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_results.json'
MODEL_PATH   = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_model.pkl'
BASELINE_AUC = 0.8411

# ── Load data ──────────────────────────────────────────────────────────────────
data      = np.load(DATA_PATH, allow_pickle=True)
emb_win   = data['emb'].astype(np.float32)        # (739, 1536) window-level
labels_win= data['labels'].astype(np.float32)     # (739, 234)
filenames = data['filenames']                      # (739,) per-window filename
file_list = data['file_list']                      # (66,)
n_windows = data['n_windows']                      # (66,)

n_files   = len(file_list)
n_species = labels_win.shape[1]

# Build window-to-file mapping
win_file_idx = np.zeros(len(emb_win), dtype=np.int32)
idx = 0
for fi, nw in enumerate(n_windows):
    win_file_idx[idx:idx+nw] = fi
    idx += nw

# Build file-level labels (OR of window labels)
file_labels = np.zeros((n_files, n_species), dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    lb = labels_win[idx:idx+nw]
    file_labels[fi] = (lb.max(0) > 0.5).astype(np.float32)
    idx += nw

# L2-normalise window embeddings for cosine ops
emb_norm = normalize(emb_win, norm='l2')  # (739, 1536)

print(f"Data: {n_files} files, {sum(n_windows)} windows, {n_species} species")
print(f"Species with >=1 positive file: {int((file_labels.sum(0) > 0).sum())}")

# ── AUC helper ─────────────────────────────────────────────────────────────────
def macro_auc(y_true, y_score):
    """Macro AUC over species with positive files across full 66 files."""
    mask = y_true.sum(0) > 0
    if mask.sum() < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')
    except Exception as e:
        return float('nan')

# ══════════════════════════════════════════════════════════════════════════════
# Variant A: Window-level Prototypical (cosine)
#   - Leave out all windows belonging to held-out file
#   - For each species, prototype = mean of POSITIVE windows in train set
#   - Test score per window = cosine_sim(window, prototype)
#   - File score = mean over test windows
# ══════════════════════════════════════════════════════════════════════════════
def proto_window_cosine_loo():
    """
    Full window-level LOO.
    Returns file-level predictions shape (66, 234).
    """
    file_preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fi in range(n_files):
        train_mask = (win_file_idx != fi)
        test_mask  = (win_file_idx == fi)

        X_tr = emb_norm[train_mask]        # (N_tr, 1536)
        Y_tr = labels_win[train_mask]      # (N_tr, 234)
        X_te = emb_norm[test_mask]         # (N_te, 1536)

        pred_win = np.zeros((X_te.shape[0], n_species), dtype=np.float32)

        for s in range(n_species):
            pos_mask = Y_tr[:, s] > 0.5
            if pos_mask.sum() == 0:
                continue
            proto = X_tr[pos_mask].mean(0)          # (1536,)
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            scores = X_te @ proto                    # (N_te,)
            pred_win[:, s] = scores

        # File score = mean over windows (shift from [-1,1] to [0,1])
        # Cosine similarity in [−1, 1] → rescale to [0, 1]
        pred_win = (pred_win + 1.0) / 2.0
        file_preds[fi] = pred_win.mean(0)

    return macro_auc(file_labels, file_preds), file_preds


# ══════════════════════════════════════════════════════════════════════════════
# Variant B: Window proto with L2 distance score (RBF kernel)
#   score = exp(-||x - proto||^2 / (2 * sigma^2))
# ══════════════════════════════════════════════════════════════════════════════
def proto_window_rbf_loo(sigma=1.0):
    file_preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fi in range(n_files):
        train_mask = (win_file_idx != fi)
        test_mask  = (win_file_idx == fi)

        X_tr = emb_win[train_mask]   # unnormalized for L2 dist
        Y_tr = labels_win[train_mask]
        X_te = emb_win[test_mask]

        for s in range(n_species):
            pos_mask = Y_tr[:, s] > 0.5
            if pos_mask.sum() == 0:
                continue
            proto = X_tr[pos_mask].mean(0)       # (1536,)
            diff  = X_te - proto[np.newaxis, :]  # (N_te, 1536)
            dist2 = (diff ** 2).sum(1)            # (N_te,)
            scores= np.exp(-dist2 / (2.0 * sigma**2))
            file_preds[fi, s] = scores.mean()

    return macro_auc(file_labels, file_preds), file_preds


# ══════════════════════════════════════════════════════════════════════════════
# Variant C: Soft prototype with temperature
#   Instead of hard positive mask, weight windows by label value (soft labels)
#   Then use cosine similarity
# ══════════════════════════════════════════════════════════════════════════════
def proto_soft_cosine_loo():
    file_preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fi in range(n_files):
        train_mask = (win_file_idx != fi)
        test_mask  = (win_file_idx == fi)

        X_tr = emb_norm[train_mask]
        Y_tr = labels_win[train_mask]
        X_te = emb_norm[test_mask]

        for s in range(n_species):
            weights = Y_tr[:, s]       # soft weights (0/1 here but allows extension)
            w_sum = weights.sum()
            if w_sum < 1e-8:
                continue
            proto = (weights[:, None] * X_tr).sum(0) / w_sum
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            scores = X_te @ proto
            file_preds[fi, s] = ((scores + 1.0) / 2.0).mean()

    return macro_auc(file_labels, file_preds), file_preds


# ══════════════════════════════════════════════════════════════════════════════
# Variant D: Negative-aware prototype (positive - negative centering)
#   proto_pos = mean of positive windows
#   proto_neg = mean of negative windows
#   score = cos_sim(x, proto_pos) - cos_sim(x, proto_neg)  (rescaled)
# ══════════════════════════════════════════════════════════════════════════════
def proto_pos_neg_cosine_loo(neg_weight=0.5):
    file_preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fi in range(n_files):
        train_mask = (win_file_idx != fi)
        test_mask  = (win_file_idx == fi)

        X_tr = emb_norm[train_mask]
        Y_tr = labels_win[train_mask]
        X_te = emb_norm[test_mask]

        for s in range(n_species):
            pos_mask = Y_tr[:, s] > 0.5
            neg_mask = Y_tr[:, s] < 0.5
            if pos_mask.sum() == 0:
                continue

            proto_pos = X_tr[pos_mask].mean(0)
            proto_pos = proto_pos / (np.linalg.norm(proto_pos) + 1e-8)

            pos_scores = X_te @ proto_pos  # (N_te,)

            if neg_mask.sum() > 0:
                proto_neg = X_tr[neg_mask].mean(0)
                proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)
                neg_scores = X_te @ proto_neg  # (N_te,)
                raw = pos_scores - neg_weight * neg_scores
            else:
                raw = pos_scores

            # Normalize to [0, 1] via sigmoid
            scores = 1.0 / (1.0 + np.exp(-raw * 5.0))
            file_preds[fi, s] = scores.mean()

    return macro_auc(file_labels, file_preds), file_preds


# ══════════════════════════════════════════════════════════════════════════════
# Variant E: Nearest-neighbor in prototype space
#   For each test window, find K nearest positive-window prototypes
#   Use similarity as score
# ══════════════════════════════════════════════════════════════════════════════
def proto_knn_cosine_loo(k=5):
    """
    Per-species: find K nearest train windows that are positive.
    Score = mean cosine similarity to top-K.
    """
    file_preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fi in range(n_files):
        train_mask = (win_file_idx != fi)
        test_mask  = (win_file_idx == fi)

        X_tr = emb_norm[train_mask]
        Y_tr = labels_win[train_mask]
        X_te = emb_norm[test_mask]

        for s in range(n_species):
            pos_mask = Y_tr[:, s] > 0.5
            if pos_mask.sum() == 0:
                continue

            pos_embs = X_tr[pos_mask]    # (n_pos, 1536)
            k_eff = min(k, pos_embs.shape[0])

            # For each test window, compute cosine sim to all positive windows
            sims = X_te @ pos_embs.T     # (N_te, n_pos)
            # Top-k mean
            if k_eff < sims.shape[1]:
                top_sims = np.sort(sims, axis=1)[:, -k_eff:]
            else:
                top_sims = sims
            scores = top_sims.mean(1)    # (N_te,)
            file_preds[fi, s] = ((scores + 1.0) / 2.0).mean()

    return macro_auc(file_labels, file_preds), file_preds


# ══════════════════════════════════════════════════════════════════════════════
# Variant F: Attention-weighted prototype
#   Use the test window itself to weight which train windows matter most.
#   proto(x_test) = sum_i [ softmax(x_test · x_i / temp) * x_i * (y_is == 1) ]
# ══════════════════════════════════════════════════════════════════════════════
def proto_attention_cosine_loo(temperature=0.1):
    file_preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fi in range(n_files):
        train_mask = (win_file_idx != fi)
        test_mask  = (win_file_idx == fi)

        X_tr = emb_norm[train_mask]
        Y_tr = labels_win[train_mask]
        X_te = emb_norm[test_mask]  # (N_te, 1536)

        for s in range(n_species):
            pos_mask = Y_tr[:, s] > 0.5
            if pos_mask.sum() == 0:
                continue

            pos_embs = X_tr[pos_mask]   # (n_pos, 1536)

            # For each test window, compute attention over positive windows
            attn_logits = X_te @ pos_embs.T / temperature  # (N_te, n_pos)
            attn_logits -= attn_logits.max(axis=1, keepdims=True)  # numerical stability
            attn = np.exp(attn_logits)
            attn /= attn.sum(axis=1, keepdims=True)  # softmax

            # Weighted prototype per test window
            proto_per_win = attn @ pos_embs  # (N_te, 1536)
            proto_per_win = proto_per_win / (np.linalg.norm(proto_per_win, axis=1, keepdims=True) + 1e-8)

            # Score = cosine sim between test window and its attended prototype
            scores = (X_te * proto_per_win).sum(1)  # (N_te,)
            file_preds[fi, s] = ((scores + 1.0) / 2.0).mean()

    return macro_auc(file_labels, file_preds), file_preds


# ══════════════════════════════════════════════════════════════════════════════
# Variant G: Ensemble of file-level and window-level prototype
#   Combines file-level mean embeddings and window-level prototypes
# ══════════════════════════════════════════════════════════════════════════════
def proto_file_level_cosine_loo():
    """
    File-level LOO using mean embeddings per file (matches baseline setup).
    Uses prototypical approach on file-level embeddings.
    """
    # Build file-level mean embeddings
    file_embs = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
    idx = 0
    for fi, nw in enumerate(n_windows):
        file_embs[fi] = emb_win[idx:idx+nw].mean(0)
        idx += nw
    file_embs_norm = normalize(file_embs, norm='l2')

    file_preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fi in range(n_files):
        train_mask = np.ones(n_files, dtype=bool)
        train_mask[fi] = False

        X_tr = file_embs_norm[train_mask]  # (65, 1536)
        Y_tr = file_labels[train_mask]      # (65, 234)
        x_te = file_embs_norm[fi]           # (1536,)

        for s in range(n_species):
            pos_mask = Y_tr[:, s] > 0.5
            if pos_mask.sum() == 0:
                continue
            proto = X_tr[pos_mask].mean(0)
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            score = float(x_te @ proto)
            file_preds[fi, s] = (score + 1.0) / 2.0

    return macro_auc(file_labels, file_preds), file_preds


# ══════════════════════════════════════════════════════════════════════════════
# Run all variants
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PROTOTYPICAL NETWORK EXPERIMENTS")
print("="*65)
print(f"Baseline KNN: {BASELINE_AUC:.4f}")
print()

results_list = []

print("Variant A: Window-level cosine prototype ...")
auc_a, preds_a = proto_window_cosine_loo()
marker = "  *** NEW BEST ***" if auc_a > BASELINE_AUC else ""
print(f"  Window cosine proto: {auc_a:.4f}  (delta={auc_a-BASELINE_AUC:+.4f}){marker}")
results_list.append(("proto_window_cosine", auc_a, {}, preds_a))

print("Variant B: Window RBF prototype (sigma=1.0) ...")
# First need to determine a good sigma based on data
# Estimate sigma as mean pairwise distance / sqrt(2) (median heuristic)
sample_dists = []
sample_idx = np.random.choice(len(emb_win), size=min(200, len(emb_win)), replace=False)
sample_embs = emb_win[sample_idx]
for i in range(len(sample_embs)):
    for j in range(i+1, min(i+10, len(sample_embs))):
        d = np.linalg.norm(sample_embs[i] - sample_embs[j])
        sample_dists.append(d)
sigma_median = np.median(sample_dists) / np.sqrt(2)
print(f"  Estimated sigma (median heuristic): {sigma_median:.4f}")

auc_b, preds_b = proto_window_rbf_loo(sigma=sigma_median)
marker = "  *** NEW BEST ***" if auc_b > BASELINE_AUC else ""
print(f"  Window RBF proto (sigma={sigma_median:.2f}): {auc_b:.4f}  (delta={auc_b-BASELINE_AUC:+.4f}){marker}")
results_list.append(("proto_window_rbf", auc_b, {"sigma": float(sigma_median)}, preds_b))

print("Variant C: Soft window prototype ...")
auc_c, preds_c = proto_soft_cosine_loo()
marker = "  *** NEW BEST ***" if auc_c > BASELINE_AUC else ""
print(f"  Soft cosine proto: {auc_c:.4f}  (delta={auc_c-BASELINE_AUC:+.4f}){marker}")
results_list.append(("proto_soft_cosine", auc_c, {}, preds_c))

print("Variant D: Negative-aware prototype ...")
best_d_auc, best_d_preds, best_d_nw = 0.0, None, 0.5
for nw in [0.3, 0.5, 0.7, 1.0]:
    auc_d, preds_d = proto_pos_neg_cosine_loo(neg_weight=nw)
    marker = "  *** NEW BEST ***" if auc_d > BASELINE_AUC else ""
    print(f"  Neg-aware proto (nw={nw}): {auc_d:.4f}  (delta={auc_d-BASELINE_AUC:+.4f}){marker}")
    if auc_d > best_d_auc:
        best_d_auc, best_d_preds, best_d_nw = auc_d, preds_d, nw
results_list.append(("proto_pos_neg_cosine", best_d_auc, {"neg_weight": best_d_nw}, best_d_preds))

print("Variant E: Top-K nearest positive prototype ...")
best_e_auc, best_e_preds, best_e_k = 0.0, None, 5
for k in [1, 3, 5, 10, 20]:
    auc_e, preds_e = proto_knn_cosine_loo(k=k)
    marker = "  *** NEW BEST ***" if auc_e > BASELINE_AUC else ""
    print(f"  KNN proto (k={k}): {auc_e:.4f}  (delta={auc_e-BASELINE_AUC:+.4f}){marker}")
    if auc_e > best_e_auc:
        best_e_auc, best_e_preds, best_e_k = auc_e, preds_e, k
results_list.append(("proto_knn_cosine", best_e_auc, {"k": best_e_k}, best_e_preds))

print("Variant F: Attention-weighted prototype ...")
best_f_auc, best_f_preds, best_f_t = 0.0, None, 0.1
for temp in [0.05, 0.1, 0.2, 0.5]:
    auc_f, preds_f = proto_attention_cosine_loo(temperature=temp)
    marker = "  *** NEW BEST ***" if auc_f > BASELINE_AUC else ""
    print(f"  Attention proto (temp={temp}): {auc_f:.4f}  (delta={auc_f-BASELINE_AUC:+.4f}){marker}")
    if auc_f > best_f_auc:
        best_f_auc, best_f_preds, best_f_t = auc_f, preds_f, temp
results_list.append(("proto_attention_cosine", best_f_auc, {"temperature": best_f_t}, best_f_preds))

print("Variant G: File-level cosine prototype ...")
auc_g, preds_g = proto_file_level_cosine_loo()
marker = "  *** NEW BEST ***" if auc_g > BASELINE_AUC else ""
print(f"  File-level cosine proto: {auc_g:.4f}  (delta={auc_g-BASELINE_AUC:+.4f}){marker}")
results_list.append(("proto_file_level_cosine", auc_g, {}, preds_g))

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
print(f"Baseline KNN: {BASELINE_AUC:.4f}")
for name, auc, params, _ in sorted(results_list, key=lambda x: -x[1]):
    marker = "  *** NEW BEST ***" if auc > BASELINE_AUC else ""
    print(f"  {name}: {auc:.4f}  (delta={auc-BASELINE_AUC:+.4f}){marker}")

# ── Find best result ───────────────────────────────────────────────────────────
best_result = max(results_list, key=lambda x: x[1])
best_name, best_auc, best_params, best_preds = best_result

# ── Update results JSON ────────────────────────────────────────────────────────
with open(RESULTS_PATH) as f:
    results_json = json.load(f)

for name, auc, params, _ in results_list:
    record = {"method": name, "loo_auc": round(float(auc), 4)}
    record.update({k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                   for k, v in params.items()})
    results_json["experiments"].append(record)

if best_auc > results_json["best"]["loo_auc"]:
    results_json["best"] = {"method": best_name, "loo_auc": round(float(best_auc), 4)}
    print(f"\nNEW BEST: {best_name} AUC={best_auc:.4f}")

    # Save model
    model_dict = {
        "method": best_name,
        "loo_auc": round(float(best_auc), 4),
        "params": best_params,
        "file_list": file_list.tolist(),
        "loo_preds": best_preds.tolist(),
        # Store the window embeddings for full refit
        "emb_norm": emb_norm.tolist(),
        "labels_win": labels_win.tolist(),
        "win_file_idx": win_file_idx.tolist(),
        "n_windows": n_windows.tolist(),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"Saved model → {MODEL_PATH}")
else:
    print(f"\nNo improvement over current best ({results_json['best']['loo_auc']:.4f})")
    print(f"Best this run: {best_name} AUC={best_auc:.4f}")

with open(RESULTS_PATH, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Results saved → {RESULTS_PATH}")
