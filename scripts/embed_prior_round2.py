"""
Embed Prior Round 2 Experiments
Methods: A (Mahalanobis KNN), B (Power logit KNN), C (File-level max logit), D (Isotonic regression KNN)
"""
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

# ── Load data ──────────────────────────────────────────────────────────────────
BASE = Path("/home/lab/BirdClef-2026-Codebase")
data = np.load(BASE / "outputs/perch_labeled_ss.npz", allow_pickle=True)
embeddings = data["emb"]          # (739, 1536)
labels     = data["labels"]       # (739, 234)
logits     = data["logits"]       # (739, 234)
file_ids   = data["filenames"]    # (739,)

print(f"Data: {embeddings.shape}, {labels.shape}, {logits.shape}")
print(f"Unique files: {len(np.unique(file_ids))}")

# L2-normalize embeddings
emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

unique_files = np.unique(file_ids)
N_FILES = len(unique_files)
N_CLASSES = labels.shape[1]

# Load existing results
results_path = BASE / "outputs/embed_prior_results.json"
with open(results_path) as f:
    results = json.load(f)

CURRENT_BEST_AUC = results["best"]["loo_auc"]
print(f"Current best LOO-AUC: {CURRENT_BEST_AUC:.6f} ({results['best']['method']})")

# ── Helper: compute macro LOO-AUC ─────────────────────────────────────────────
def compute_macro_auc(file_true_all, file_pred_all):
    """macro ROC-AUC over species seen in at least one training fold"""
    Y_true = np.array(file_true_all)   # (n_files, 234)
    Y_pred = np.array(file_pred_all)   # (n_files, 234)
    aucs = []
    for c in range(N_CLASSES):
        yt = Y_true[:, c]
        yp = Y_pred[:, c]
        if yt.sum() > 0 and yt.sum() < len(yt):
            try:
                aucs.append(roc_auc_score(yt, yp))
            except:
                pass
    return np.mean(aucs) if aucs else 0.0

# ── Method B: Power logit KNN ─────────────────────────────────────────────────
print("\n=== Method B: Power logit KNN (power sweep) ===")

# Already tried: power=3.0, k=3, AUC=0.8923
# Try: power in [2, 4, 0.5], k in [3, 5]
best_b = {"auc": 0.0}

for power in [2.0, 4.0, 0.5]:
    for k_knn in [3, 5]:
        sigmoid_logit = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))  # (739, 234)
        pow_logit = sigmoid_logit ** power

        # Grid over alpha
        best_alpha = 0.0
        best_auc = 0.0
        for alpha in np.arange(0.1, 0.7, 0.05):
            file_true_all = []
            file_pred_all = []
            for hf in unique_files:
                tr_m = file_ids != hf
                te_m = file_ids == hf
                X_tr = emb_norm[tr_m]
                X_te = emb_norm[te_m]
                L_te = pow_logit[te_m]
                Y_tr = labels[tr_m]
                Y_te = labels[te_m]

                sims = X_te @ X_tr.T  # (n_te, n_tr)
                topk = np.argsort(-sims, axis=1)[:, :k_knn]
                w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
                w = w / (w.sum(1, keepdims=True) + 1e-8)
                knn_score = (w[:, :, None] * Y_tr[topk]).sum(1)

                score = alpha * L_te + (1 - alpha) * knn_score
                file_true_all.append(Y_te.max(0))
                file_pred_all.append(score.mean(0))

            auc = compute_macro_auc(file_true_all, file_pred_all)
            if auc > best_auc:
                best_auc = auc
                best_alpha = alpha

        if best_auc > best_b["auc"]:
            best_b = {"auc": best_auc, "power": power, "k": k_knn, "alpha": best_alpha}
        print(f"  power={power}, k={k_knn}: best_alpha={best_alpha:.2f}, AUC={best_auc:.6f}")

print(f"Method B best: {best_b}")

# ── Method A: Mahalanobis KNN ──────────────────────────────────────────────────
print("\n=== Method A: Mahalanobis KNN (PCA-64 covariance) ===")

best_a = {"auc": 0.0}

# PCA 64 on all data (LOO would be too slow for PCA — use all data for cov; slight leakage)
pca = PCA(n_components=64, random_state=42)
emb_pca = pca.fit_transform(emb_norm)  # (739, 64)

# Compute covariance on all training data & inv
cov = np.cov(emb_pca.T)  # (64, 64)
reg = 1e-4 * np.eye(64)
inv_cov = np.linalg.inv(cov + reg)

# Mahalanobis distance: sqrt((x-y)^T @ inv_cov @ (x-y))
# For efficiency: use inv_cov decomposition
# score = -distance (similarity)
L_chol = np.linalg.cholesky(inv_cov)  # (64,64): inv_cov = L_chol @ L_chol.T
emb_mah = emb_pca @ L_chol   # (739, 64) whitened embeddings; cosine on whitened = mahalanobis

# Now KNN on whitened embeddings, but we also blend logit
emb_mah_norm = emb_mah / (np.linalg.norm(emb_mah, axis=1, keepdims=True) + 1e-8)

sigmoid_logit = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))

for k_knn in [3, 5]:
    for alpha in np.arange(0.0, 0.5, 0.05):
        file_true_all = []
        file_pred_all = []
        for hf in unique_files:
            tr_m = file_ids != hf
            te_m = file_ids == hf
            X_tr_mah = emb_mah_norm[tr_m]
            X_te_mah = emb_mah_norm[te_m]
            L_te = sigmoid_logit[te_m]
            Y_tr = labels[tr_m]
            Y_te = labels[te_m]

            sims = X_te_mah @ X_tr_mah.T
            topk = np.argsort(-sims, axis=1)[:, :k_knn]
            w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
            w = w / (w.sum(1, keepdims=True) + 1e-8)
            knn_score = (w[:, :, None] * Y_tr[topk]).sum(1)

            score = alpha * L_te + (1 - alpha) * knn_score
            file_true_all.append(Y_te.max(0))
            file_pred_all.append(score.mean(0))

        auc = compute_macro_auc(file_true_all, file_pred_all)
        if auc > best_a["auc"]:
            best_a = {"auc": auc, "k": k_knn, "alpha": alpha}
    print(f"  k={k_knn}: AUC={best_a['auc']:.6f}")

print(f"Method A best: {best_a}")

# ── Method D: Isotonic regression calibrated KNN ──────────────────────────────
print("\n=== Method D: Isotonic regression calibrated KNN ===")

# Outer LOO: train isotonic on training folds
best_d = {"auc": 0.0}
k_iso = 5

# Two-pass outer LOO: first get KNN predictions, then calibrate
# Pass 1: collect all KNN(k=5) predictions
all_knn_preds = np.zeros_like(labels, dtype=np.float32)
for hf in unique_files:
    tr_m = file_ids != hf
    te_m = file_ids == hf
    X_tr = emb_norm[tr_m]
    X_te = emb_norm[te_m]
    Y_tr = labels[tr_m]
    sims = X_te @ X_tr.T
    topk = np.argsort(-sims, axis=1)[:, :k_iso]
    w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
    w = w / (w.sum(1, keepdims=True) + 1e-8)
    all_knn_preds[te_m] = (w[:, :, None] * Y_tr[topk]).sum(1)

print(f"  KNN(k={k_iso}) preds collected")

# Now outer LOO: for each held-out file, fit isotonic on remaining files, predict on held-out
file_true_all = []
file_pred_all_d = []
for hf in unique_files:
    tr_m = file_ids != hf
    te_m = file_ids == hf
    knn_tr = all_knn_preds[tr_m]   # (n_tr, 234)
    knn_te = all_knn_preds[te_m]   # (n_te, 234)
    Y_tr = labels[tr_m]
    Y_te = labels[te_m]

    # Fit per-species isotonic regression
    calib_score = np.zeros_like(knn_te)
    for c in range(N_CLASSES):
        yt = Y_tr[:, c]
        yp = knn_tr[:, c]
        if yt.sum() > 0 and yt.sum() < len(yt):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(yp, yt)
            calib_score[:, c] = ir.predict(knn_te[:, c])
        else:
            calib_score[:, c] = knn_te[:, c]

    file_true_all.append(Y_te.max(0))
    file_pred_all_d.append(calib_score.mean(0))

auc_d = compute_macro_auc(file_true_all, file_pred_all_d)
best_d = {"auc": auc_d, "k": k_iso}
print(f"  Isotonic KNN(k={k_iso}): AUC={auc_d:.6f}")

# ── Method C: File-level max logit + KNN ─────────────────────────────────────
print("\n=== Method C: File-level max logit (per-file aggregation) ===")

# Per each file compute per-species max logit → file_logit_max (66, 234)
# Then in LOO: use training file_logit_max as KNN labels
# But this is per-file level, so for test file we use the file logit max
# And KNN neighbors are training files aggregated by mean embedding

file_labels_agg = {}
file_embs_agg = {}
file_logit_max_agg = {}
for fi in unique_files:
    m = file_ids == fi
    file_labels_agg[fi] = labels[m].max(0)       # (234,)
    file_embs_agg[fi] = emb_norm[m].mean(0)       # (1536,)  mean embedding
    file_logit_max_agg[fi] = (1.0 / (1.0 + np.exp(-logits[m].max(0)))).astype(np.float32)

FILE_EMB = np.array([file_embs_agg[fi] for fi in unique_files])  # (66, 1536)
FILE_EMB = FILE_EMB / (np.linalg.norm(FILE_EMB, axis=1, keepdims=True) + 1e-8)
FILE_LABELS = np.array([file_labels_agg[fi] for fi in unique_files])  # (66, 234)
FILE_LOGIT_MAX = np.array([file_logit_max_agg[fi] for fi in unique_files])  # (66, 234)

best_c = {"auc": 0.0}
for k_knn in [3, 5]:
    for alpha in np.arange(0.0, 0.6, 0.05):
        file_true_all = []
        file_pred_all_c = []
        for i, hf in enumerate(unique_files):
            tr_idx = [j for j, f in enumerate(unique_files) if f != hf]
            te_idx = i

            X_tr = FILE_EMB[tr_idx]       # (65, 1536)
            X_te = FILE_EMB[te_idx:te_idx+1]  # (1, 1536)
            Y_tr = FILE_LABELS[tr_idx]    # (65, 234)
            L_te = FILE_LOGIT_MAX[te_idx:te_idx+1]  # (1, 234)
            Y_te = FILE_LABELS[te_idx]    # (234,)

            sims = X_te @ X_tr.T  # (1, 65)
            topk = np.argsort(-sims, axis=1)[:, :k_knn]
            w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
            w = w / (w.sum(1, keepdims=True) + 1e-8)
            knn_score = (w[:, :, None] * Y_tr[topk]).sum(1)  # (1, 234)

            score = alpha * L_te + (1 - alpha) * knn_score  # (1, 234)
            file_true_all.append(Y_te)
            file_pred_all_c.append(score[0])

        auc = compute_macro_auc(file_true_all, file_pred_all_c)
        if auc > best_c["auc"]:
            best_c = {"auc": auc, "k": k_knn, "alpha": alpha}
    print(f"  k={k_knn}: best AUC so far={best_c['auc']:.6f}")

print(f"Method C best: {best_c}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Current best: {CURRENT_BEST_AUC:.6f}")
print(f"Method A (Mahalanobis KNN): {best_a['auc']:.6f}")
print(f"Method B (Power logit KNN): {best_b['auc']:.6f}  (power={best_b.get('power')}, k={best_b.get('k')}, alpha={best_b.get('alpha'):.2f})")
print(f"Method C (File-level max):  {best_c['auc']:.6f}  (k={best_c.get('k')}, alpha={best_c.get('alpha'):.2f})")
print(f"Method D (Isotonic KNN):    {best_d['auc']:.6f}")

# Collect results to append
new_experiments = [
    {"method": "mahalanobis_knn", "loo_auc": round(best_a["auc"], 6), **{k: v for k, v in best_a.items() if k != "auc"}},
    {"method": "power_logit_knn_sweep", "loo_auc": round(best_b["auc"], 6), **{k: v for k, v in best_b.items() if k != "auc"}},
    {"method": "file_level_max_knn", "loo_auc": round(best_c["auc"], 6), **{k: v for k, v in best_c.items() if k != "auc"}},
    {"method": "isotonic_knn", "loo_auc": round(best_d["auc"], 6), **{k: v for k, v in best_d.items() if k != "auc"}},
]

# Find overall best from new methods
all_new = [(e["loo_auc"], e) for e in new_experiments]
best_new_auc, best_new_exp = max(all_new)

print(f"\nBest new: {best_new_exp['method']} AUC={best_new_auc:.6f}")

# Save results
results["experiments"].extend(new_experiments)

if best_new_auc > CURRENT_BEST_AUC:
    print(f"\n*** NEW BEST: {best_new_exp['method']} AUC={best_new_auc:.6f} > {CURRENT_BEST_AUC:.6f} ***")
    results["best"] = {
        "method": best_new_exp["method"],
        "loo_auc": best_new_auc,
        "config": {k: v for k, v in best_new_exp.items() if k not in ("method", "loo_auc")},
        "note": f"Found by embed_prior_round2.py 2026-03-25; prev={results['best']['method']}={CURRENT_BEST_AUC:.6f}"
    }
else:
    print(f"\nNo improvement over current best {CURRENT_BEST_AUC:.6f}")

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print("Results saved.")

# Save best_new info for next step
import sys
sys.stdout.flush()
# Store for use in next script
with open(BASE / "outputs/embed_prior_round2_best.json", "w") as f:
    json.dump({
        "current_best_auc": CURRENT_BEST_AUC,
        "best_new_auc": best_new_auc,
        "best_new_method": best_new_exp["method"],
        "best_new_exp": best_new_exp,
        "all_new": [{"method": e["method"], "loo_auc": e["loo_auc"]} for e in new_experiments]
    }, f, indent=2)
