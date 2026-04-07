#!/usr/bin/env python3
"""
research_advanced_clustering_v2.py
====================================
精確複製 train_cluster_stacker.py 資料管道，
在 708-sample 帶標籤集上測試進階 clustering 方法。

基準線：baseline OOF AUC = 0.9553 (5-model sigmoid mean)
目標：找出可超越 0.9553 的方法

方法：
  M0: Baseline (5-model sigmoid mean)
  M1: k-NN in Perch Embedding Space (LOO on 708 labeled)
  M2: Confidence-weighted k-NN (distance-gated)
  M3: Embedding-guided k-NN → blend with baseline
  M4: Embedding-guided calibration correction
  M5: Soft label propagation via graph
  M6: Pseudo-augmented k-NN (127K soundscape reference)
  M7: Per-species prototype-distance re-scoring
  M8: Adaptive temperature scaling via neighbors
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ── Paths (mirror train_cluster_stacker.py) ────────────────────────────────────
BASE     = Path("/home/lab/BirdClef-2026-Codebase")
PERCH_META = BASE / "birdclef-2026/notebook resource/current_subs 2/perch meta"
OUT_DIR    = BASE / "birdclef-2026/notebook resource/current_subs 2/stacker_weights"
OUTPUTS    = BASE / "outputs"
RESULTS_PATH = OUTPUTS / "advanced_clustering_v2_results.json"
N_CLASSES = 234

# ── Reproduce exact data pipeline from train_cluster_stacker.py ───────────────
print("[1/7] Loading labeled features (708 rows) …")

meta = pd.read_parquet(str(PERCH_META / "full_perch_meta.parquet"))
filenames_708 = meta["filename"].values
row_ids_708   = meta["row_id"].values
unique_files  = list(dict.fromkeys(filenames_708))
file_to_idx   = {f: i for i, f in enumerate(unique_files)}
groups        = np.array([file_to_idx[f] for f in filenames_708], dtype=np.int32)

# Perch raw scores + embeddings
perch_arr     = np.load(str(PERCH_META / "full_perch_arrays.npz"))
perch_raw_prob = perch_arr["scores_full_raw"].astype(np.float32)  # (708, 234)
emb_lab        = perch_arr["emb_full"].astype(np.float32)         # (708, 1536)

# OOF meta features
oof_data    = np.load(str(PERCH_META / "full_oof_meta_features.npz"))
perch_prior = oof_data["oof_base"].astype(np.float32)   # (708, 234) logit
mlp_probe   = oof_data["oof_prior"].astype(np.float32)  # (708, 234) logit
fold_id     = oof_data["fold_id"].astype(np.int32)      # (708,)

mlp_probe_path = OUTPUTS / "mlp_probe_oof.npy"
if mlp_probe_path.exists():
    mlp_probe = np.load(str(mlp_probe_path)).astype(np.float32)

# Proto SSM
proto_preds_59 = np.load(str(OUTPUTS / "proto_ssm_oof_preds.npy")).astype(np.float32)
proto_files_59 = np.load(str(OUTPUTS / "proto_ssm_oof_file_list.npy"), allow_pickle=True)
proto_708 = np.zeros((708, N_CLASSES), dtype=np.float32)
for wi, fname in enumerate(filenames_708):
    mask = proto_files_59 == fname
    if mask.any():
        proto_708[wi] = proto_preds_59[np.where(mask)[0][0]]

# SED
sed_csebbs = np.load(str(OUTPUTS / "stacker_train_sed_csebbs_v3.npy")).astype(np.float32)

# Ground truth labels
label_data = np.load(str(OUTPUTS / "perch_labeled_ss.npz"), allow_pickle=True)
rid_to_label = dict(zip(label_data["row_ids"], range(len(label_data["row_ids"]))))
Y_lab = np.zeros((708, N_CLASSES), dtype=np.float32)
for i, rid in enumerate(row_ids_708):
    if rid in rid_to_label:
        Y_lab[i] = label_data["labels"][rid_to_label[rid]]

print(f"  Y_lab: {Y_lab.shape}  pos_rate={Y_lab.mean():.4f}")

# Build X_lab_norm (same as train_cluster_stacker.py)
EPS = 1e-7
def safe_logit(p):
    p = np.clip(p.astype(np.float32), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -88, 88))).astype(np.float32)

def to_logit(arr, name):
    mn, mx = float(arr.min()), float(arr.max())
    if mn >= -0.05 and mx <= 1.05:
        return safe_logit(arr)
    return arr.astype(np.float32)

perch_raw_l   = to_logit(perch_raw_prob, "perch_raw")
perch_prior_l = to_logit(perch_prior,    "perch_prior")
mlp_probe_l   = to_logit(mlp_probe,      "mlp_probe")
proto_l       = to_logit(proto_708,      "proto_ssm")
sed_l         = to_logit(sed_csebbs,     "sed_csebbs")

X_lab_raw  = np.concatenate([perch_raw_l, perch_prior_l, mlp_probe_l, proto_l, sed_l], axis=1)
norm_data  = np.load(str(OUT_DIR / "stacker_norm_v3.npz"), allow_pickle=True)
X_mean, X_std = norm_data["mean"], norm_data["std"]
X_lab_norm = ((X_lab_raw - X_mean) / (X_std + 1e-8)).astype(np.float32)
emb_lab_n  = normalize(emb_lab, norm='l2').astype(np.float32)

print(f"  X_lab_norm: {X_lab_norm.shape}")
print(f"  emb_lab_n : {emb_lab_n.shape}")

# ── Helper functions ──────────────────────────────────────────────────────────
def compute_auc(Y_true, Y_pred, prefix="", ref=0.9553):
    aucs = []
    for c in range(Y_true.shape[1]):
        if 1 <= Y_true[:, c].sum() < len(Y_true):
            try:
                aucs.append(roc_auc_score(Y_true[:, c], Y_pred[:, c]))
            except:
                pass
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    flag = " ★ BEAT BASELINE ★" if mean_auc > ref else ""
    if prefix:
        print(f"  {prefix:<52} {mean_auc:.4f}{flag}")
    return mean_auc

# ── M0: Baseline ─────────────────────────────────────────────────────────────
print("\n[2/7] M0: Baseline")
base_pred = sigmoid(X_lab_norm.reshape(708, 5, N_CLASSES)).mean(axis=1).astype(np.float32)
m0_auc = compute_auc(Y_lab, base_pred, "M0: 5-model sigmoid mean (baseline)")

# Individual model baselines
model_names = ["perch_raw", "perch_prior", "mlp_probe", "proto_ssm", "sed_csebbs"]
model_logits = [perch_raw_l, perch_prior_l, mlp_probe_l, proto_l, sed_l]
for name, logit in zip(model_names, model_logits):
    compute_auc(Y_lab, sigmoid(logit), f"  M0-single: {name}")

# ── GroupKFold cross-validation utility ──────────────────────────────────────
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)

def oof_evaluate(predict_fn, name, n_folds=5):
    """Run GroupKFold OOF evaluation."""
    Y_pred_oof = np.zeros_like(Y_lab)
    for fold, (tr, va) in enumerate(gkf.split(X_lab_norm, groups=groups)):
        Y_pred_oof[va] = predict_fn(tr, va)
    return compute_auc(Y_lab, Y_pred_oof, name)

# ── M1: k-NN in embedding space (GroupKFold OOF) ─────────────────────────────
print("\n[3/7] M1: k-NN Label Propagation (GroupKFold OOF)")

def make_knn_predictor(k, use_emb=True, use_X=False, emb_weight=1.0):
    def predict(tr, va):
        if use_emb and use_X:
            feat_tr = np.hstack([emb_lab_n[tr] * emb_weight,
                                  X_lab_norm[tr] * (1.0 - emb_weight)])
            feat_va = np.hstack([emb_lab_n[va] * emb_weight,
                                  X_lab_norm[va] * (1.0 - emb_weight)])
        elif use_emb:
            feat_tr = emb_lab_n[tr]
            feat_va = emb_lab_n[va]
        else:
            feat_tr = X_lab_norm[tr]
            feat_va = X_lab_norm[va]

        nn = NearestNeighbors(n_neighbors=min(k, len(tr)), metric='cosine',
                              algorithm='brute', n_jobs=-1)
        nn.fit(feat_tr)
        dists, idxs = nn.kneighbors(feat_va)

        sigma = np.median(dists) + 1e-6
        W = np.exp(-dists / sigma)
        W_n = W / (W.sum(axis=1, keepdims=True) + 1e-8)

        Y_pred_va = np.zeros((len(va), N_CLASSES), dtype=np.float32)
        for i in range(len(va)):
            nb_Y = Y_lab[tr[idxs[i]]]
            Y_pred_va[i] = (W_n[i, :, None] * nb_Y).sum(axis=0)
        return Y_pred_va
    return predict

best_knn_k = None
best_knn_auc = 0
for k in [5, 10, 15, 20, 30, 40]:
    auc = oof_evaluate(make_knn_predictor(k, use_emb=True), f"M1: k-NN emb k={k}")
    if auc > best_knn_auc:
        best_knn_auc = auc
        best_knn_k = k

# Also test X-space k-NN
for k in [10, 20, 30]:
    oof_evaluate(make_knn_predictor(k, use_emb=False, use_X=True), f"M1x: k-NN X-space k={k}")

# ── M2: Blend k-NN with baseline ────────────────────────────────────────────
print(f"\n[4/7] M2: Blend k-NN (k={best_knn_k}) with baseline")

def make_blend_predictor(k, alpha, use_emb=True):
    knn_fn = make_knn_predictor(k, use_emb=use_emb)
    def predict(tr, va):
        knn_pred = knn_fn(tr, va)
        base = base_pred[va]
        return alpha * knn_pred + (1 - alpha) * base
    return predict

best_blend_auc = 0
best_alpha = 0
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    auc = oof_evaluate(make_blend_predictor(best_knn_k or 20, alpha),
                       f"M2: kNN*{alpha:.1f}+base*{1-alpha:.1f}")
    if auc > best_blend_auc:
        best_blend_auc = auc
        best_alpha = alpha

# ── M3: Distance-adaptive blending ──────────────────────────────────────────
print(f"\n[5/7] M3: Distance-adaptive blending")

def make_dist_adaptive_predictor(k, max_alpha=0.7, dist_pct=50):
    def predict(tr, va):
        nn = NearestNeighbors(n_neighbors=min(k, len(tr)), metric='cosine',
                              algorithm='brute', n_jobs=-1)
        nn.fit(emb_lab_n[tr])
        dists, idxs = nn.kneighbors(emb_lab_n[va])

        # Threshold based on training set NN distances
        nn_tr = NearestNeighbors(n_neighbors=2, metric='cosine', algorithm='brute')
        nn_tr.fit(emb_lab_n[tr])
        dists_tr, _ = nn_tr.kneighbors(emb_lab_n[tr])
        threshold = np.percentile(dists_tr[:, 1], dist_pct)

        sigma = np.median(dists) + 1e-6
        W = np.exp(-dists / sigma)
        W_n = W / (W.sum(axis=1, keepdims=True) + 1e-8)

        Y_pred_va = np.zeros((len(va), N_CLASSES), dtype=np.float32)
        for i in range(len(va)):
            nb_Y = Y_lab[tr[idxs[i]]]
            knn_pred = (W_n[i, :, None] * nb_Y).sum(axis=0)

            nn_dist = dists[i, 0]
            trust = max_alpha * max(0.0, 1.0 - nn_dist / (threshold + 1e-6))
            trust = min(trust, max_alpha)

            Y_pred_va[i] = trust * knn_pred + (1 - trust) * base_pred[va[i]]
        return Y_pred_va
    return predict

for max_a in [0.5, 0.7, 0.9]:
    for pct in [25, 50, 75]:
        oof_evaluate(make_dist_adaptive_predictor(best_knn_k or 20, max_a, pct),
                     f"M3: dist-adaptive (max_a={max_a}, pct={pct})")

# ── M4: Embedding calibration correction ─────────────────────────────────────
print("\n[6/7] M4: Embedding calibration correction")

def make_calib_predictor(k, alpha_calib):
    def predict(tr, va):
        nn = NearestNeighbors(n_neighbors=min(k, len(tr)), metric='cosine',
                              algorithm='brute', n_jobs=-1)
        nn.fit(emb_lab_n[tr])
        dists, idxs = nn.kneighbors(emb_lab_n[va])

        sigma = np.median(dists) + 1e-6
        W = np.exp(-dists / sigma)
        W_n = W / (W.sum(axis=1, keepdims=True) + 1e-8)

        Y_pred_va = np.zeros((len(va), N_CLASSES), dtype=np.float32)
        for i in range(len(va)):
            nb_Y = Y_lab[tr[idxs[i]]]        # (k, 234) ground truth
            nb_base = base_pred[tr[idxs[i]]]  # (k, 234) model predictions
            correction = (W_n[i, :, None] * (nb_Y - nb_base)).sum(axis=0)
            Y_pred_va[i] = np.clip(base_pred[va[i]] + alpha_calib * correction, 0, 1)
        return Y_pred_va
    return predict

for k in [10, 15, 20]:
    for alpha_c in [0.2, 0.4, 0.6, 0.8]:
        oof_evaluate(make_calib_predictor(k, alpha_c),
                     f"M4: calib-correction (k={k}, alpha={alpha_c})")

# ── M5: Graph Label Propagation ─────────────────────────────────────────────
print("\n[7/7] M5: Graph Label Propagation (within training fold)")

def make_graph_lp_predictor(k, alpha_lp, n_iter):
    def predict(tr, va):
        # Build graph on train set
        nn = NearestNeighbors(n_neighbors=min(k+1, len(tr)), metric='cosine',
                              algorithm='brute', n_jobs=-1)
        nn.fit(emb_lab_n[tr])
        dists_tr, idxs_tr = nn.kneighbors(emb_lab_n[tr])
        dists_tr = dists_tr[:, 1:]  # skip self
        idxs_tr  = idxs_tr[:, 1:]

        sigma = np.median(dists_tr) + 1e-6
        W_tr = np.exp(-dists_tr / sigma)

        # Propagate labels on training set using base_pred as seeds
        F = base_pred[tr].copy()
        Y_seed = Y_lab[tr].copy()  # ground truth as anchor (not available, use base_pred)
        # Use base_pred as initial signal; Y_lab as label anchor
        for it in range(n_iter):
            F_new = np.zeros_like(F)
            for i in range(len(tr)):
                nb_F = F[idxs_tr[i]]
                w_n = W_tr[i] / (W_tr[i].sum() + 1e-8)
                F_new[i] = (w_n[:, None] * nb_F).sum(axis=0)
            # Clamp to [0,1] and mix with seed
            F = alpha_lp * np.clip(F_new, 0, 1) + (1 - alpha_lp) * base_pred[tr]

        # Predict validation: weighted avg of train propagated labels
        dists_va, idxs_va = nn.kneighbors(emb_lab_n[va])
        sigma_va = np.median(dists_va) + 1e-6
        W_va = np.exp(-dists_va / sigma_va)
        W_va_n = W_va / (W_va.sum(axis=1, keepdims=True) + 1e-8)

        Y_pred_va = np.zeros((len(va), N_CLASSES), dtype=np.float32)
        for i in range(len(va)):
            Y_pred_va[i] = (W_va_n[i, :, None] * F[idxs_va[i]]).sum(axis=0)
        return Y_pred_va
    return predict

for n_it in [3, 5]:
    for a_lp in [0.3, 0.5, 0.7]:
        oof_evaluate(make_graph_lp_predictor(15, a_lp, n_it),
                     f"M5: graph-LP (k=15, alpha={a_lp}, iter={n_it})")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  ADVANCED CLUSTERING SUMMARY (v2 — 708 labeled, GroupKFold)")
print("="*70)
print(f"  Baseline (5-model mean):   {m0_auc:.4f}")
print(f"  Best k-NN (k={best_knn_k}):        {best_knn_auc:.4f}")
print(f"  Best blend (alpha={best_alpha}):   {best_blend_auc:.4f}")
print(f"  Reference cluster baseline: 0.9553")
print("="*70)

results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "m0_auc": float(m0_auc),
    "best_knn_k": int(best_knn_k) if best_knn_k else None,
    "best_knn_auc": float(best_knn_auc),
    "best_blend_auc": float(best_blend_auc),
    "best_alpha": float(best_alpha),
    "baseline_reference": 0.9553,
}
with open(str(RESULTS_PATH), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved: {RESULTS_PATH}")
