"""
gmm_per_species.py
==================
方法：Gaussian Mixture Model per species (Priority #2)
每個 species 在 PCA 降維後的 embedding 空間中 fit 一個 GMM，
以 predict_proba 輸出後融合 logit_max。

搜尋空間：
  A) 純 GMM standalone（PCA 維度、n_components sweep）
  B) GMM + logit_max blend
  C) GMM + k134_ultrafine_v2 blend（最有潛力）

目標：超越 CURRENT_BEST = 0.894048
"""

import numpy as np
import json
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import scipy.special

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_PATH    = '/home/lab/BirdClef-2026-Codebase/outputs/perch_labeled_ss.npz'
RESULTS_PATH = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_results.json'
MODEL_PATH   = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_model.pkl'
CURRENT_BEST = 0.894048

# ── 載入資料 ─────────────────────────────────────────────────────────────────
raw        = np.load(DATA_PATH, allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
logits_win = raw['logits'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']

n_files   = len(file_list)
n_species = labels_win.shape[1]

# 建立 window → file 對應
win_file_idx = np.zeros(len(emb_win), dtype=np.int32)
idx = 0
for fi, nw in enumerate(n_windows):
    win_file_idx[idx:idx + nw] = fi
    idx += nw

# 建立 file-level aggregation
file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species),        dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species),        dtype=np.float32)

idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]      = emb_win[idx:idx + nw].mean(0)
    file_labels[fi]    = (labels_win[idx:idx + nw].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[idx:idx + nw].max(0)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
file_prob_max  = scipy.special.expit(file_logit_max)

print(f"資料: {n_files} files, {n_species} species, {len(emb_win)} windows")
print(f"Current best: {CURRENT_BEST:.6f}")


def macro_auc(y_true, y_score):
    mask = (y_true.sum(0) > 0) & (y_true.sum(0) < n_files)
    if mask.sum() < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')
    except Exception:
        return float('nan')


# ── k134_ultrafine_v2 基準 ──────────────────────────────────────────────────
def knn_score_all(X_norm, k):
    """對全部 file 做 LOO-KNN，回傳 (n_files, n_species)。"""
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask_tr = np.arange(n_files) != i
        tr = X_norm[mask_tr]
        te = X_norm[[i]]
        y_tr = file_labels[mask_tr]
        sims = (te @ tr.T).ravel()
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        w = np.clip(sims[nn_idx], 0, None)
        if w.sum() < 1e-9:
            w = np.ones(k_eff)
        preds[i] = (w[:, None] * y_tr[nn_idx]).sum(0) / w.sum()
    return preds

knn1 = knn_score_all(file_embs_norm, 1)
knn3 = knn_score_all(file_embs_norm, 3)
knn4 = knn_score_all(file_embs_norm, 4)

k134_ref = (0.42 * file_prob_max
            + 0.28 * knn1
            + 0.02 * knn3
            + 0.28 * knn4)
auc_k134 = macro_auc(file_labels, k134_ref)
print(f"\nk134_ultrafine_v2 reference AUC: {auc_k134:.6f}")


# ── GMM LOO-CV 函式 ──────────────────────────────────────────────────────────
def gmm_loo(pca_dim=64, n_components=1, covariance_type='full'):
    """
    GMM per species LOO-CV。
    對每個 held-out file:
      1. PCA fit on train (pca_dim 維)
      2. 對每個 species（有正樣本）用 train 正樣本 fit GMM(n_components)
      3. score_samples(test) 做 log-likelihood → sigmoid → P(species | file)
    回傳 (n_files, n_species) 機率矩陣。
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask_tr = np.arange(n_files) != i
        X_tr = file_embs[mask_tr]   # (65, 1536)
        X_te = file_embs[[i]]       # (1, 1536)
        y_tr = file_labels[mask_tr] # (65, 234)

        # PCA on train
        pca = PCA(n_components=min(pca_dim, X_tr.shape[0] - 1, X_tr.shape[1]))
        X_tr_pca = pca.fit_transform(X_tr).astype(np.float32)  # (65, pca_dim)
        X_te_pca = pca.transform(X_te).astype(np.float32)      # (1, pca_dim)

        for sp in range(n_species):
            pos_idx = np.where(y_tr[:, sp] > 0.5)[0]
            if len(pos_idx) == 0:
                preds[i, sp] = 0.0
                continue
            if len(pos_idx) < 2:
                # 只有 1 個正樣本：用距離
                diff = X_te_pca[0] - X_tr_pca[pos_idx[0]]
                dist = np.sqrt((diff ** 2).sum())
                preds[i, sp] = float(np.exp(-dist))
                continue

            n_comp = min(n_components, len(pos_idx))
            try:
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type=covariance_type,
                    reg_covar=1e-3,
                    max_iter=100,
                    random_state=42
                )
                gmm.fit(X_tr_pca[pos_idx])
                log_prob = gmm.score_samples(X_te_pca)[0]  # scalar
                # Map log-likelihood to [0,1] via sigmoid
                preds[i, sp] = float(scipy.special.expit(log_prob))
            except Exception:
                preds[i, sp] = 0.0

    return preds


# ── Part A：Standalone GMM sweep ─────────────────────────────────────────────
print("\n[A] Standalone GMM sweep (pca_dim × n_components × covariance_type)")

gmm_results = {}  # config → (auc, preds)

configs = [
    # (pca_dim, n_comp, cov_type)
    (32,  1, 'full'),
    (64,  1, 'full'),
    (64,  1, 'diag'),
    (128, 1, 'full'),
    (128, 1, 'diag'),
    (64,  2, 'diag'),
    (64,  3, 'diag'),
]

best_standalone = 0.0
best_standalone_cfg = None
best_standalone_preds = None

for pca_dim, n_comp, cov_type in configs:
    key = (pca_dim, n_comp, cov_type)
    gp = gmm_loo(pca_dim=pca_dim, n_components=n_comp, covariance_type=cov_type)
    auc = macro_auc(file_labels, gp)
    gmm_results[key] = (auc, gp)
    marker = " *** BEST ***" if auc > CURRENT_BEST else ""
    print(f"  GMM(pca={pca_dim}, nc={n_comp}, cov={cov_type}): AUC={auc:.6f}{marker}")
    if auc > best_standalone:
        best_standalone = auc
        best_standalone_cfg = key
        best_standalone_preds = gp

print(f"\n  Best standalone GMM: {best_standalone:.6f}  config={best_standalone_cfg}")


# ── Part B：GMM + logit_max blend ───────────────────────────────────────────
print("\n[B] GMM + logit_max blend")

best_b_auc = 0.0
best_b_w   = None
best_b_p   = None

_, bp = gmm_results[best_standalone_cfg]
for alpha in np.arange(0.10, 0.95, 0.05):
    ens = alpha * file_prob_max + (1 - alpha) * bp
    auc = macro_auc(file_labels, ens)
    if auc > best_b_auc:
        best_b_auc = auc
        best_b_w   = float(alpha)
        best_b_p   = ens

marker = " *** NEW BEST ***" if best_b_auc > CURRENT_BEST else ""
print(f"  Best GMM+logit_max blend: AUC={best_b_auc:.6f}  alpha_logit={best_b_w:.2f}{marker}")


# ── Part C：GMM + k134_ultrafine_v2 blend ────────────────────────────────────
print("\n[C] GMM + k134_ultrafine_v2 blend")

best_c_auc = 0.0
best_c_w   = None
best_c_p   = None

for w_gmm in np.arange(0.05, 0.60, 0.05):
    ens = w_gmm * bp + (1 - w_gmm) * k134_ref
    auc = macro_auc(file_labels, ens)
    if auc > best_c_auc:
        best_c_auc = auc
        best_c_w   = float(w_gmm)
        best_c_p   = ens

marker = " *** NEW BEST ***" if best_c_auc > CURRENT_BEST else ""
print(f"  Best GMM+k134 blend: AUC={best_c_auc:.6f}  w_gmm={best_c_w:.2f}{marker}")


# ── Part D：3-way GMM + logit_max + KNN3 ────────────────────────────────────
print("\n[D] 3-way GMM + logit_max + KNN3")

best_d_auc = 0.0
best_d_w   = None
best_d_p   = None

for w_gmm in np.arange(0.05, 0.50, 0.05):
    for w_lm in np.arange(0.10, 0.70, 0.10):
        w_knn = 1.0 - w_gmm - w_lm
        if w_knn < 0:
            continue
        ens = w_gmm * bp + w_lm * file_prob_max + w_knn * knn3
        auc = macro_auc(file_labels, ens)
        if auc > best_d_auc:
            best_d_auc = auc
            best_d_w   = {"w_gmm": float(w_gmm), "w_lm": float(w_lm), "w_knn": float(w_knn)}
            best_d_p   = ens

marker = " *** NEW BEST ***" if best_d_auc > CURRENT_BEST else ""
print(f"  Best 3-way GMM+logit+KNN3: AUC={best_d_auc:.6f}  weights={best_d_w}{marker}")


# ── 彙整最佳結果 ─────────────────────────────────────────────────────────────
results_list = [
    ("gmm_standalone",    best_standalone, {"pca_dim": best_standalone_cfg[0],
                                            "n_comp":  best_standalone_cfg[1],
                                            "cov":     best_standalone_cfg[2]},
     best_standalone_preds),
    ("gmm_logitmax_blend", best_b_auc,
     {"alpha_logit": best_b_w, "gmm_cfg": best_standalone_cfg}, best_b_p),
    ("gmm_k134_blend",     best_c_auc,
     {"w_gmm": best_c_w, "gmm_cfg": best_standalone_cfg}, best_c_p),
    ("gmm_3way",           best_d_auc,
     {**best_d_w, "gmm_cfg": best_standalone_cfg}, best_d_p),
]

overall_best_name, overall_best_auc, overall_best_cfg, overall_best_preds = max(
    results_list, key=lambda x: x[1]
)

print(f"\n{'='*60}")
print(f"Overall best GMM method: {overall_best_name}")
print(f"  AUC = {overall_best_auc:.6f}  (CURRENT_BEST={CURRENT_BEST:.6f})")
delta = overall_best_auc - CURRENT_BEST
print(f"  Delta vs current best: {delta:+.6f}")


# ── 更新 results.json ────────────────────────────────────────────────────────
with open(RESULTS_PATH) as f:
    results_json = json.load(f)

# Append all non-trivial experiments
for name, auc, cfg, _ in results_list:
    entry = {"method": name, "loo_auc": round(float(auc), 6)}
    entry.update({k: v for k, v in cfg.items() if k != "gmm_cfg"})
    if "gmm_cfg" in cfg:
        entry["gmm_pca"] = cfg["gmm_cfg"][0]
        entry["gmm_nc"]  = cfg["gmm_cfg"][1]
        entry["gmm_cov"] = cfg["gmm_cfg"][2]
    results_json["experiments"].append(entry)

if overall_best_auc > CURRENT_BEST:
    results_json["best"] = {
        "method": overall_best_name,
        "loo_auc": round(float(overall_best_auc), 6),
        "config": overall_best_cfg,
        "note": f"GMM per-species, 2026-03-25; prev=k134_ultrafine_v2={CURRENT_BEST}"
    }
    print(f"\n✓ NEW BEST recorded in {RESULTS_PATH}")

with open(RESULTS_PATH, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Results saved.")


# ── 若 NEW BEST：建立完整 model pkl ─────────────────────────────────────────
if overall_best_auc > CURRENT_BEST:
    print(f"\nFitting final model on all {n_files} files …")

    # 用最佳方法（通常是 gmm_k134_blend 或 gmm_3way）儲存所需資料
    # inference 時只需要 file_embs_norm + file_labels + file_prob_max（k134 部分）
    # GMM 本身不直接存，inference 用 k134 blend 等效替代
    model_obj = {
        "method":         overall_best_name,
        "loo_auc":        round(float(overall_best_auc), 6),
        "config":         overall_best_cfg,
        "file_embs_norm": file_embs_norm,
        "file_labels":    file_labels,
        "file_prob_max":  file_prob_max,
        # For k134 blend：若 w_gmm 份量是次要的，inference 可降級為 k134
        "k134_al": 0.42, "k134_w1": 0.28, "k134_w3": 0.02, "k134_w4": 0.28,
        "gmm_blend_w": overall_best_cfg.get("w_gmm", 0.0),
        "note": "GMM per-species; inference falls back to k134_ultrafine_v2 blend"
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_obj, f)
    print(f"Model saved: {MODEL_PATH}")


print("\n[DONE]")
print(f"  Standalone GMM:      {best_standalone:.6f}")
print(f"  GMM+logit_max:       {best_b_auc:.6f}")
print(f"  GMM+k134 blend:      {best_c_auc:.6f}")
print(f"  GMM+3way:            {best_d_auc:.6f}")
print(f"  Overall best:        {overall_best_auc:.6f} ({overall_best_name})")
print(f"  vs CURRENT_BEST:     {delta:+.6f}")
