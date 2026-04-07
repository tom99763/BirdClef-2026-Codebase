"""
Bayesian / probabilistic embedding prior experiments.
Methods: Distance-weighted KNN, RBF-kernel SVM, Label Propagation
LOO-CV at file level (66 files, leave-one-file-out).

IMPORTANT: operates at FILE level (mean embedding per file), NOT window level.
This matches the baseline KNN AUC = 0.8411 from train_embed_prior.py.

  - 66 file-level mean embeddings (1536-dim)
  - 66 file-level binary labels (OR of window labels)
  - LOO-CV: for each fold, train on 65 files, predict 1 file
  - macro-AUC across species present in the FULL dataset (75 active species)
"""

import numpy as np
import json
import pickle
import sys
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────────
DATA_PATH    = '/home/lab/BirdClef-2026-Codebase/outputs/perch_labeled_ss.npz'
RESULTS_PATH = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_results.json'
MODEL_PATH   = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_model.pkl'

data = np.load(DATA_PATH, allow_pickle=True)
emb_win   = data['emb'].astype(np.float32)       # (739, 1536) window embeddings
labels_win= data['labels'].astype(np.float32)    # (739, 234)  window labels
filenames = data['filenames']                     # (739,) per-window filename
file_list = data['file_list']                     # (66,)  unique files
n_windows = data['n_windows']                     # (66,)  windows per file

n_files   = len(file_list)
n_species = labels_win.shape[1]

# ── Build file-level representations ─────────────────────────────────────────
# Mean embedding + binary OR labels (matches train_embed_prior.py)
file_embs   = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels = np.zeros((n_files, n_species),         dtype=np.float32)

idx = 0
for fi, (fname, nw) in enumerate(zip(file_list, n_windows)):
    ws = emb_win[idx: idx + nw]
    lb = labels_win[idx: idx + nw]
    file_embs[fi]   = ws.mean(0)
    file_labels[fi] = (lb.max(0) > 0.5).astype(np.float32)  # OR across windows
    idx += nw

# L2-normalise for cosine operations
file_embs_norm = normalize(file_embs, norm='l2')

print(f"Data: {n_files} files, {n_species} species", flush=True)
print(f"  Species present (>=1 file): {int((file_labels.sum(0) > 0).sum())}", flush=True)

# ── AUC helper ────────────────────────────────────────────────────────────────
def macro_auc(y_true, y_score):
    """Macro AUC over species present in at least one file."""
    mask = y_true.sum(0) > 0
    if mask.sum() < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')
    except Exception as e:
        return float('nan')

BASELINE_AUC = 0.8411

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1: Distance-Weighted KNN (cosine similarity as weight)
#   Improvement over original KNN: use cosine SIM as weight instead of uniform
#   Original used NearestNeighbors with euclidean on L2-normed embs (= cosine)
#   but used uniform weights. We use cosine similarity as weight.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("METHOD 1: Distance-Weighted KNN (cosine-sim weights)", flush=True)
print("="*60, flush=True)

def dw_knn_loo(k):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask  = np.ones(n_files, dtype=bool); mask[i] = False
        X_tr  = file_embs_norm[mask]     # (65, 1536)
        y_tr  = file_labels[mask]        # (65, 234)
        x_te  = file_embs_norm[[i]]      # (1, 1536)

        # Cosine similarity (dot product on L2-normed)
        sims = (x_te @ X_tr.T).ravel()  # (65,)
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff)[:k_eff]
        weights = np.clip(sims[nn_idx], 0, None)
        if weights.sum() < 1e-9:
            weights = np.ones(k_eff)
        preds[i] = (weights[:, None] * y_tr[nn_idx]).sum(0) / weights.sum()

    return macro_auc(file_labels, preds), preds

best_dw_auc, best_dw_k, best_dw_preds = 0.0, 5, None
for k in [3, 5, 7, 10, 15, 20]:
    auc, preds = dw_knn_loo(k)
    marker = "  *** IMPROVEMENT ***" if auc > BASELINE_AUC else ""
    print(f"  DW-KNN k={k:2d}: {auc:.4f}  (delta={auc-BASELINE_AUC:+.4f}){marker}", flush=True)
    if auc > best_dw_auc:
        best_dw_auc, best_dw_k, best_dw_preds = auc, k, preds

print(f"  Best DW-KNN: k={best_dw_k}, AUC={best_dw_auc:.4f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: RBF-kernel SVM (per-species binary, calibrated probability)
#   File-level: 65 training points, 1 test point per fold
#   Use cross-val=3 for Platt calibration; many species have >=3 positives.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("METHOD 2: RBF-SVM (per-species, calibrated)", flush=True)
print("="*60, flush=True)

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

def rbf_svm_loo(C=1.0, gamma='scale'):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        if i % 10 == 0:
            print(f"  SVM fold {i+1}/{n_files} ...", flush=True)
        mask  = np.ones(n_files, dtype=bool); mask[i] = False
        X_tr  = file_embs_norm[mask]     # (65, 1536)
        y_tr  = file_labels[mask]        # (65, 234)
        x_te  = file_embs_norm[[i]]      # (1, 1536)

        file_pred = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            ys = y_tr[:, s].astype(int)
            n_pos = ys.sum()
            n_neg = (1 - ys).sum()
            if n_pos == 0 or n_neg == 0:
                continue
            # Need at least 2 positives for cv=2 calibration
            cv = min(3, n_pos, n_neg)
            if cv < 2:
                # Just use SVC decision function score without calibration
                try:
                    clf = SVC(C=C, kernel='rbf', gamma=gamma, probability=False)
                    clf.fit(X_tr, ys)
                    score = clf.decision_function(x_te)[0]
                    # Sigmoid transform to [0,1]
                    file_pred[s] = 1.0 / (1.0 + np.exp(-score))
                except Exception:
                    pass
            else:
                try:
                    base = SVC(C=C, kernel='rbf', gamma=gamma, probability=False)
                    clf  = CalibratedClassifierCV(base, cv=cv, method='sigmoid')
                    clf.fit(X_tr, ys)
                    p = clf.predict_proba(x_te)[0]
                    if 1 in clf.classes_:
                        c1 = list(clf.classes_).index(1)
                        file_pred[s] = p[c1]
                except Exception:
                    pass

        preds[i] = file_pred

    return macro_auc(file_labels, preds), preds

svm_auc, svm_preds = rbf_svm_loo(C=1.0, gamma='scale')
print(f"  RBF-SVM C=1.0 gamma=scale: {svm_auc:.4f}  (delta={svm_auc-BASELINE_AUC:+.4f})", flush=True)

# Also try C=10 if first is good
svm_auc2, svm_preds2 = rbf_svm_loo(C=10.0, gamma='scale')
print(f"  RBF-SVM C=10.0 gamma=scale: {svm_auc2:.4f}  (delta={svm_auc2-BASELINE_AUC:+.4f})", flush=True)

if svm_auc2 > svm_auc:
    svm_auc, svm_preds = svm_auc2, svm_preds2
    best_svm_C = 10.0
else:
    best_svm_C = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3: Label Spreading (semi-supervised)
#   66-file graph; test file = unlabeled node.
#   knn kernel on L2-normed embeddings.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("METHOD 3: Label Spreading (semi-supervised, knn graph)", flush=True)
print("="*60, flush=True)

def label_spreading_loo(n_neighbors=5, alpha=0.2):
    """
    For each LOO fold:
      - all 66 files form the graph
      - 65 files are labeled, 1 test file is unlabeled (-1)
      - LabelSpreading propagates across the knn graph
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        if i % 10 == 0:
            print(f"  LP fold {i+1}/{n_files} ...", flush=True)

        file_pred = np.zeros(n_species, dtype=np.float32)

        for s in range(n_species):
            y_full = file_labels[:, s].astype(int).copy()  # (66,)
            if y_full.sum() == 0:
                continue
            # Set test file as unlabeled
            y_full[i] = -1
            # Skip if no positive labels remain
            if (y_full == 1).sum() == 0:
                continue
            try:
                ls = LabelSpreading(kernel='knn', n_neighbors=n_neighbors,
                                    alpha=alpha, max_iter=30, tol=1e-3)
                ls.fit(file_embs_norm, y_full)
                proba = ls.predict_proba(file_embs_norm[[i]])  # (1, n_classes)
                if 1 in ls.classes_:
                    c1 = list(ls.classes_).index(1)
                    file_pred[s] = proba[0, c1]
            except Exception:
                pass

        preds[i] = file_pred

    return macro_auc(file_labels, preds), preds

# Try multiple hyperparameter settings
lp_results = []
for n_nb, alpha in [(5, 0.2), (7, 0.2), (5, 0.5), (10, 0.3)]:
    auc, preds = label_spreading_loo(n_neighbors=n_nb, alpha=alpha)
    marker = "  *** IMPROVEMENT ***" if auc > BASELINE_AUC else ""
    print(f"  LabelSpreading n_nb={n_nb} alpha={alpha}: {auc:.4f}  (delta={auc-BASELINE_AUC:+.4f}){marker}", flush=True)
    lp_results.append((auc, preds, n_nb, alpha))

best_lp_auc, best_lp_preds, best_lp_nb, best_lp_alpha = max(lp_results, key=lambda x: x[0])

# ══════════════════════════════════════════════════════════════════════════════
# Summary & Save
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("RESULTS SUMMARY", flush=True)
print("="*60, flush=True)
print(f"  Baseline KNN (uniform weights, k=5):  {BASELINE_AUC:.4f}", flush=True)
print(f"  [1] Distance-Weighted KNN (k={best_dw_k}):    {best_dw_auc:.4f}  (delta={best_dw_auc-BASELINE_AUC:+.4f})", flush=True)
print(f"  [2] RBF-SVM (C={best_svm_C}):                 {svm_auc:.4f}  (delta={svm_auc-BASELINE_AUC:+.4f})", flush=True)
print(f"  [3] Label Spreading (n_nb={best_lp_nb}, a={best_lp_alpha}): {best_lp_auc:.4f}  (delta={best_lp_auc-BASELINE_AUC:+.4f})", flush=True)

all_results = [
    ("distance_weighted_knn", best_dw_auc,  {"best_k": best_dw_k},               best_dw_preds),
    ("rbf_svm",               svm_auc,      {"C": best_svm_C, "gamma": "scale"},  svm_preds),
    ("label_spreading",       best_lp_auc,  {"n_neighbors": best_lp_nb, "alpha": best_lp_alpha}, best_lp_preds),
]

# ── Load + update results.json ────────────────────────────────────────────────
with open(RESULTS_PATH) as f:
    results_json = json.load(f)

for name, auc, params, _ in all_results:
    record = {"method": name, "loo_auc": round(auc, 4)}
    record.update(params)
    results_json["experiments"].append(record)

# Best overall
best_exp = max(all_results, key=lambda x: x[1])
if best_exp[1] > results_json["best"]["loo_auc"]:
    results_json["best"] = {"method": best_exp[0], "loo_auc": round(best_exp[1], 4)}
    print(f"\nNew overall best: {best_exp[0]} AUC={best_exp[1]:.4f}", flush=True)
    model_dict = {
        "method":    best_exp[0],
        "loo_auc":   round(best_exp[1], 4),
        "params":    best_exp[2],
        "file_list": file_list.tolist(),
        "loo_preds": best_exp[3].tolist(),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"Saved model → {MODEL_PATH}", flush=True)
else:
    print(f"\nNo improvement over current best ({results_json['best']['loo_auc']:.4f})", flush=True)

with open(RESULTS_PATH, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Saved results → {RESULTS_PATH}", flush=True)
