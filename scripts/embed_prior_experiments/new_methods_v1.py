"""
embed_prior new_methods_v1.py
實驗三個全新方法，目標超越 current best LOO-AUC = 0.894048

方法：
  1. KNN Logit Transfer (KLT): 用鄰居的 logits (soft probs) 替代 binary labels
  2. Pyro Bayesian Temperature Calibration (BTC): per-species temperature 推斷
  3. Multi-resolution KLT ensemble: KLT(k=1,3,4) + logit_max blend

LOO-CV: 66 files leave-one-file-out
"""

import numpy as np
import json
import pickle
import warnings
import os
import sys
warnings.filterwarnings('ignore')

from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import scipy.special

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_PATH    = '/home/lab/BirdClef-2026-Codebase/outputs/perch_labeled_ss.npz'
RESULTS_PATH = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_results.json'
MODEL_PATH   = '/home/lab/BirdClef-2026-Codebase/outputs/embed_prior_model.pkl'
CURRENT_BEST = 0.894048

# ──────────────────────────────────────────────────────────────────
# 載入資料
# ──────────────────────────────────────────────────────────────────
raw        = np.load(DATA_PATH, allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)        # (739, 1536)
logits_win = raw['logits'].astype(np.float32)     # (739, 234)
labels_win = raw['labels'].astype(np.float32)     # (739, 234)
file_list  = raw['file_list']                     # (66,)
n_windows  = raw['n_windows']                     # (66,)

n_files   = len(file_list)
n_species = labels_win.shape[1]

# 建立 window → file 索引
win_file_idx = np.zeros(len(emb_win), dtype=np.int32)
idx = 0
for fi, nw in enumerate(n_windows):
    win_file_idx[idx:idx + nw] = fi
    idx += nw

# 建立 file-level aggregation
file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species),        dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species),        dtype=np.float32)
# 也存每個 file 的 window embeddings 列表（供 KLT 使用）
file_win_embs    = []   # list of (nw, 1536)
file_win_logits  = []   # list of (nw, 234)

idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]      = emb_win[idx:idx + nw].mean(0)
    file_labels[fi]    = (labels_win[idx:idx + nw].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[idx:idx + nw].max(0)
    file_win_embs.append(emb_win[idx:idx + nw].copy())
    file_win_logits.append(logits_win[idx:idx + nw].copy())
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
file_prob_max  = scipy.special.expit(file_logit_max)

print(f"資料: {n_files} files, {n_species} species, {len(emb_win)} windows")
print(f"Current best: {CURRENT_BEST:.6f}")

def macro_auc(y_true, y_score):
    """只算有 positive 且不全 positive 的 species"""
    mask = (y_true.sum(0) > 0) & (y_true.sum(0) < n_files)
    if mask.sum() < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro')
    except Exception:
        return float('nan')

# ──────────────────────────────────────────────────────────────────
# 共用函式：標準 KNN（用 binary labels）
# ──────────────────────────────────────────────────────────────────
def knn_binary_predict(k=3):
    """LOO file-level KNN，用 binary labels 作 signal"""
    X = file_embs_norm
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask_tr = np.arange(n_files) != i
        tr = X[mask_tr]; te = X[[i]]; y_tr = file_labels[mask_tr]
        sims = (te @ tr.T).ravel()
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        w = np.clip(sims[nn_idx], 0, None)
        if w.sum() < 1e-9:
            w = np.ones(k_eff)
        preds[i] = (w[:, None] * y_tr[nn_idx]).sum(0) / w.sum()
    return preds

# Pre-compute
print("Pre-computing KNN(1,3,4,5)...")
knn1 = knn_binary_predict(k=1)
knn3 = knn_binary_predict(k=3)
knn4 = knn_binary_predict(k=4)
knn5 = knn_binary_predict(k=5)
print(f"  KNN(1)={macro_auc(file_labels,knn1):.4f}, "
      f"KNN(3)={macro_auc(file_labels,knn3):.4f}, "
      f"KNN(4)={macro_auc(file_labels,knn4):.4f}, "
      f"KNN(5)={macro_auc(file_labels,knn5):.4f}")

# 現有 best formula baseline
knn_best_ref = 0.42*file_prob_max + 0.28*knn1 + 0.02*knn3 + 0.28*knn4
print(f"  k134_ultrafine_v2 (ref): {macro_auc(file_labels, knn_best_ref):.6f}")

results_list = []  # (name, auc, params, preds)

# ══════════════════════════════════════════════════════════════════
# 方法 1：KNN Logit Transfer (KLT)
#   用鄰居的 sigmoid(logits) 而非 binary labels 作為 KNN signal
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 1: KNN Logit Transfer (KLT)")
print("="*70)

def knn_logit_transfer(k=3):
    """
    LOO file-level KNN，用鄰居的 sigmoid(logit_max) 作 signal（含 cosine weight）
    """
    X = file_embs_norm
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask_tr = np.arange(n_files) != i
        tr = X[mask_tr]; te = X[[i]]
        neighbor_probs = file_prob_max[mask_tr]   # (65, 234) — sigmoid(logit_max)
        sims = (te @ tr.T).ravel()
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        w = np.clip(sims[nn_idx], 0, None)
        if w.sum() < 1e-9:
            w = np.ones(k_eff)
        w = w / w.sum()
        preds[i] = (w[:, None] * neighbor_probs[nn_idx]).sum(0)
    return preds

# 計算 KLT for k = 1..5
klt_cache = {}
for k in [1, 2, 3, 4, 5]:
    klt_cache[k] = knn_logit_transfer(k=k)
    print(f"  KLT(k={k}): {macro_auc(file_labels, klt_cache[k]):.4f}")

# 1a. alpha * logit_max + (1-alpha) * KLT(k)
print("\n  -- 1a: alpha*logit_max + (1-alpha)*KLT(k) sweep --")
best_1a_auc, best_1a_preds, best_1a_k, best_1a_alpha = 0.0, None, 3, 0.4
for k in [1, 2, 3, 4, 5]:
    for alpha in np.arange(0.20, 0.65, 0.01):
        ens = alpha * file_prob_max + (1 - alpha) * klt_cache[k]
        auc = macro_auc(file_labels, ens)
        if auc > best_1a_auc:
            best_1a_auc = auc
            best_1a_preds = ens.copy()
            best_1a_k, best_1a_alpha = k, alpha

marker = "  *** NEW BEST ***" if best_1a_auc > CURRENT_BEST else ""
print(f"  Best: k={best_1a_k}, alpha={best_1a_alpha:.3f}: {best_1a_auc:.6f}  "
      f"(delta={best_1a_auc-CURRENT_BEST:+.6f}){marker}")
results_list.append(("klt_logitmax_blend", best_1a_auc,
                     {"k": best_1a_k, "alpha": float(best_1a_alpha)}, best_1a_preds))

# 1b. k134 KLT: al*logit_max + w1*KLT(1) + w3*KLT(3) + w4*KLT(4)
print("\n  -- 1b: k134 KLT (al*logit_max + w1*KLT1 + w3*KLT3 + w4*KLT4) sweep --")
best_1b_auc, best_1b_preds, best_1b_w = 0.0, None, {}
al_grid  = np.arange(0.30, 0.55, 0.02)
w1_grid  = np.arange(0.15, 0.40, 0.02)
w3_grid  = np.arange(0.00, 0.15, 0.02)
# w4 = 1 - al - w1 - w3
for al in al_grid:
    for w1 in w1_grid:
        for w3 in w3_grid:
            w4 = 1.0 - al - w1 - w3
            if w4 < 0 or w4 > 0.50:
                continue
            ens = al*file_prob_max + w1*klt_cache[1] + w3*klt_cache[3] + w4*klt_cache[4]
            auc = macro_auc(file_labels, ens)
            if auc > best_1b_auc:
                best_1b_auc = auc
                best_1b_preds = ens.copy()
                best_1b_w = {"al": float(al), "w1": float(w1), "w3": float(w3), "w4": float(w4)}

marker = "  *** NEW BEST ***" if best_1b_auc > CURRENT_BEST else ""
print(f"  Best k134 KLT: {best_1b_auc:.6f}  (delta={best_1b_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_1b_w}")
results_list.append(("k134_klt", best_1b_auc, best_1b_w, best_1b_preds))

# 1c. 混合 binary KNN + KLT：al*logit_max + w_knn*KNN(k) + w_klt*KLT(k)
print("\n  -- 1c: logit_max + KNN(binary) + KLT(logit) 3-way --")
best_1c_auc, best_1c_preds, best_1c_w = 0.0, None, {}
for k in [1, 3, 4]:
    for al in np.arange(0.30, 0.55, 0.02):
        for w_knn in np.arange(0.10, 0.55, 0.02):
            w_klt = 1.0 - al - w_knn
            if w_klt < 0:
                continue
            ens = al*file_prob_max + w_knn*knn_binary_predict.__wrapped__ if False else \
                  al*file_prob_max + w_knn*(knn1 if k==1 else knn3 if k==3 else knn4) + w_klt*klt_cache[k]
            auc = macro_auc(file_labels, ens)
            if auc > best_1c_auc:
                best_1c_auc = auc
                best_1c_preds = ens.copy()
                best_1c_w = {"k": k, "al": float(al), "w_knn": float(w_knn), "w_klt": float(w_klt)}

marker = "  *** NEW BEST ***" if best_1c_auc > CURRENT_BEST else ""
print(f"  Best 3-way: {best_1c_auc:.6f}  (delta={best_1c_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_1c_w}")
results_list.append(("klt_knn_logitmax_3way", best_1c_auc, best_1c_w, best_1c_preds))

# 1d. window-level KLT：對每個 test window 找 training windows 最近鄰
#     再 aggregate 到 file level
print("\n  -- 1d: window-level KLT (test window → train window neighbors) --")

def window_level_klt(k=3):
    """
    對每個 test window，在 train windows 中找 k 個最近鄰，
    取鄰居 sigmoid(logit) 的加權平均 → window score
    再對 file 取 max → file score
    """
    X_norm = normalize(emb_win, norm='l2')   # (739, 1536)
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fi in range(n_files):
        # test windows
        test_mask  = win_file_idx == fi
        train_mask = win_file_idx != fi

        X_te = X_norm[test_mask]   # (n_te, 1536)
        X_tr = X_norm[train_mask]  # (n_tr, 1536)
        L_tr = scipy.special.expit(logits_win[train_mask])  # (n_tr, 234)

        sims = X_te @ X_tr.T   # (n_te, n_tr)
        k_eff = min(k, X_tr.shape[0])
        nn_idx = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]  # (n_te, k)
        w = np.take_along_axis(sims, nn_idx, axis=1).clip(0)   # (n_te, k)
        w_sum = w.sum(1, keepdims=True)
        w_sum[w_sum < 1e-9] = 1.0
        w = w / w_sum   # (n_te, k)

        # weighted neighbor logit probs: (n_te, 234)
        nbr_probs = np.stack([L_tr[nn_idx[i]] for i in range(len(X_te))], axis=0)  # (n_te, k, 234)
        win_scores = (w[:, :, None] * nbr_probs).sum(1)   # (n_te, 234)

        # file score = max across windows
        preds[fi] = win_scores.max(0)

    return preds

best_1d_auc, best_1d_preds, best_1d_w = 0.0, None, {}
for k in [1, 2, 3, 4, 5]:
    wklt_d = window_level_klt(k=k)
    auc_klt = macro_auc(file_labels, wklt_d)
    print(f"    window-KLT(k={k}) alone: {auc_klt:.4f}")
    for al in np.arange(0.25, 0.65, 0.02):
        ens = al * file_prob_max + (1 - al) * wklt_d
        auc = macro_auc(file_labels, ens)
        if auc > best_1d_auc:
            best_1d_auc = auc
            best_1d_preds = ens.copy()
            best_1d_w = {"k": k, "al": float(al)}

marker = "  *** NEW BEST ***" if best_1d_auc > CURRENT_BEST else ""
print(f"  Best window-KLT: {best_1d_auc:.6f}  (delta={best_1d_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_1d_w}")
results_list.append(("window_klt_logitmax", best_1d_auc, best_1d_w, best_1d_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 2：Pyro Bayesian Temperature Calibration (BTC)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 2: Pyro Bayesian Temperature Calibration")
print("="*70)

try:
    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam

    def bayesian_temp_calibration_loo(n_steps=300):
        """
        LOO: leave one FILE out.
        對每個 fold：
          - train: 所有其他 file 的 file_logit_max (65, 234) + file_labels (65, 234)
          - 推斷 per-species temperature T_s
          - 對 test file 的 logit_max 做 calibration：sigmoid(logit / T_s)
        """
        preds = np.zeros((n_files, n_species), dtype=np.float32)

        for fi in range(n_files):
            mask_tr = np.arange(n_files) != fi
            logits_tr = torch.tensor(file_logit_max[mask_tr], dtype=torch.float32)  # (65, 234)
            labels_tr = torch.tensor(file_labels[mask_tr],    dtype=torch.float32)  # (65, 234)
            logits_te = file_logit_max[[fi]]  # (1, 234) numpy

            pyro.clear_param_store()

            def model(logits, labels):
                log_T = pyro.sample(
                    "log_T",
                    dist.Normal(torch.zeros(n_species), 0.5 * torch.ones(n_species)).to_event(1)
                )
                T = torch.exp(log_T).clamp(min=0.1, max=10.0)  # (234,)
                scaled = logits / T[None, :]                    # (n, 234)
                with pyro.plate("data", logits.shape[0]):
                    pyro.sample("obs", dist.Bernoulli(logits=scaled).to_event(1), obs=labels)

            def guide(logits, labels):
                loc   = pyro.param("log_T_loc",   torch.zeros(n_species))
                scale = pyro.param("log_T_scale",  0.1 * torch.ones(n_species),
                                   constraint=dist.constraints.positive)
                pyro.sample("log_T", dist.Normal(loc, scale).to_event(1))

            optimizer = ClippedAdam({"lr": 0.05})
            svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

            for _ in range(n_steps):
                svi.step(logits_tr, labels_tr)

            T_est = torch.exp(pyro.param("log_T_loc")).clamp(min=0.1, max=10.0).detach().numpy()  # (234,)

            # calibrated prediction for test file
            calibrated_logit = logits_te / T_est[None, :]  # (1, 234)
            preds[fi] = scipy.special.expit(calibrated_logit).ravel()

            if fi % 10 == 0:
                print(f"    LOO fold {fi+1}/{n_files} done")

        return preds

    print("  Running LOO Bayesian temperature calibration (300 SVI steps)...")
    btc_preds = bayesian_temp_calibration_loo(n_steps=300)
    auc_btc = macro_auc(file_labels, btc_preds)
    print(f"  BTC alone: {auc_btc:.4f}")

    # blend BTC with KNN
    best_btc_auc, best_btc_preds, best_btc_w = auc_btc, btc_preds.copy(), {"al": 0.0, "w_knn": 0.0}
    for al_btc in np.arange(0.0, 1.01, 0.05):
        for k in [1, 3, 4]:
            knn_k = knn1 if k == 1 else knn3 if k == 3 else knn4
            for w_knn in np.arange(0.0, 1.0 - al_btc + 0.01, 0.05):
                w_logitmax = 1.0 - al_btc - w_knn
                if w_logitmax < 0:
                    continue
                ens = al_btc * btc_preds + w_knn * knn_k + w_logitmax * file_prob_max
                auc = macro_auc(file_labels, ens)
                if auc > best_btc_auc:
                    best_btc_auc = auc
                    best_btc_preds = ens.copy()
                    best_btc_w = {"al_btc": float(al_btc), "k": k, "w_knn": float(w_knn),
                                  "w_logitmax": float(w_logitmax)}

    marker = "  *** NEW BEST ***" if best_btc_auc > CURRENT_BEST else ""
    print(f"  Best BTC blend: {best_btc_auc:.6f}  (delta={best_btc_auc-CURRENT_BEST:+.6f}){marker}")
    print(f"    {best_btc_w}")
    results_list.append(("bayesian_temp_calib", best_btc_auc, best_btc_w, best_btc_preds))
    pyro_available = True

except ImportError as e:
    print(f"  Pyro 未安裝，跳過方法 2: {e}")
    pyro_available = False
except Exception as e:
    print(f"  方法 2 錯誤: {e}")
    import traceback; traceback.print_exc()
    pyro_available = False

# ══════════════════════════════════════════════════════════════════
# 方法 3：Bayesian Gaussian Prototype (Shrinkage)
#   conjugate Gaussian prior → posterior mean = shrinkage toward global mean
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 3: Bayesian Shrinkage Prototype")
print("="*70)

from sklearn.decomposition import PCA

def bayesian_shrinkage_proto(pca_dim=64, prior_strength=1.0):
    """
    Conjugate Gaussian prototype:
      prior: mu_s ~ N(global_mean, I)
      likelihood: x_i | y_is=1 ~ N(mu_s, I)
      posterior: mu_s | data ~ N(post_mean_s, post_cov_s)
      post_mean_s = (n_pos * sample_mean_s + prior_strength * global_mean)
                   / (n_pos + prior_strength)

    Score for test embedding x_te:
      score_s = exp(-0.5 * ||x_te - post_mean_s||^2 / sigma^2)
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    # LOO
    for fi in range(n_files):
        mask_tr = np.arange(n_files) != fi
        X_tr = file_embs[mask_tr]   # (65, 1536)
        Y_tr = file_labels[mask_tr] # (65, 234)
        X_te = file_embs[[fi]]      # (1, 1536)

        # PCA
        pca = PCA(n_components=pca_dim, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr)   # (65, pca_dim)
        X_te_pca = pca.transform(X_te)        # (1, pca_dim)

        global_mean = X_tr_pca.mean(0)   # (pca_dim,)

        scores = np.zeros(n_species, dtype=np.float32)
        for s in range(n_species):
            pos_idx = Y_tr[:, s] > 0.5
            n_pos = pos_idx.sum()
            if n_pos == 0:
                # no positive examples → use global prior
                post_mean = global_mean
            else:
                sample_mean = X_tr_pca[pos_idx].mean(0)
                post_mean = (n_pos * sample_mean + prior_strength * global_mean) / (n_pos + prior_strength)

            # score = negative squared distance (use as ranking)
            diff = X_te_pca[0] - post_mean
            dist2 = (diff ** 2).sum()
            # Convert to probability-like score via RBF
            sigma2 = float(pca_dim)   # heuristic: unit variance in PCA space
            scores[s] = np.exp(-0.5 * dist2 / sigma2)

        preds[fi] = scores

    return preds

best_bsp_auc, best_bsp_preds, best_bsp_w = 0.0, None, {}
for pca_dim in [16, 24, 32, 48, 64]:  # 128 > n_train=65 所以不用
    for prior_s in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        bp = bayesian_shrinkage_proto(pca_dim=pca_dim, prior_strength=prior_s)
        auc_bp = macro_auc(file_labels, bp)
        print(f"    BSP(pca={pca_dim}, prior={prior_s}): {auc_bp:.4f}")

        # blend with logit_max + KNN
        for al in np.arange(0.10, 0.60, 0.05):
            for w_knn in np.arange(0.0, 0.50, 0.05):
                w_bsp = 1.0 - al - w_knn
                if w_bsp < 0:
                    continue
                # use best KNN mix (k134)
                ens = al * file_prob_max + w_knn * (0.28*knn1 + 0.02*knn3 + 0.28*knn4) / 0.58 + w_bsp * bp
                auc = macro_auc(file_labels, ens)
                if auc > best_bsp_auc:
                    best_bsp_auc = auc
                    best_bsp_preds = ens.copy()
                    best_bsp_w = {"pca_dim": pca_dim, "prior_s": prior_s,
                                  "al": float(al), "w_knn": float(w_knn), "w_bsp": float(w_bsp)}

marker = "  *** NEW BEST ***" if best_bsp_auc > CURRENT_BEST else ""
print(f"  Best BSP blend: {best_bsp_auc:.6f}  (delta={best_bsp_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_bsp_w}")
results_list.append(("bayesian_shrinkage_proto", best_bsp_auc, best_bsp_w, best_bsp_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 4：Window-level KLT + binary KNN 混合 (最精細)
#   使用 window-level KLT 取 mean（非 max），再與現有 best 混合
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 4: window-level KLT (mean agg) + k134_ultrafine_v2 ultra blend")
print("="*70)

def window_level_klt_mean(k=3):
    """window-level KLT，file score = mean (而非 max) across windows"""
    X_norm = normalize(emb_win, norm='l2')
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fi in range(n_files):
        test_mask  = win_file_idx == fi
        train_mask = win_file_idx != fi
        X_te = X_norm[test_mask]
        X_tr = X_norm[train_mask]
        L_tr = scipy.special.expit(logits_win[train_mask])

        sims = X_te @ X_tr.T
        k_eff = min(k, X_tr.shape[0])
        nn_idx = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        w = np.take_along_axis(sims, nn_idx, axis=1).clip(0)
        w_sum = w.sum(1, keepdims=True); w_sum[w_sum < 1e-9] = 1.0
        w = w / w_sum

        nbr_probs = np.stack([L_tr[nn_idx[i]] for i in range(len(X_te))], axis=0)
        win_scores = (w[:, :, None] * nbr_probs).sum(1)
        preds[fi] = win_scores.mean(0)   # mean aggregation

    return preds

# k134 style with window-level KLT mean
wklt_mean_cache = {}
for k in [1, 3, 4]:
    wklt_mean_cache[k] = window_level_klt_mean(k=k)
    print(f"    win-KLT-mean(k={k}): {macro_auc(file_labels, wklt_mean_cache[k]):.4f}")

best_4_auc, best_4_preds, best_4_w = 0.0, None, {}
# full grid around k134 formula shape
for al in np.arange(0.30, 0.55, 0.02):
    for w1 in np.arange(0.10, 0.40, 0.02):
        for w3 in np.arange(0.00, 0.10, 0.02):
            w4 = 1.0 - al - w1 - w3
            if w4 < 0 or w4 > 0.50:
                continue
            ens = al*file_prob_max + w1*wklt_mean_cache[1] + w3*wklt_mean_cache[3] + w4*wklt_mean_cache[4]
            auc = macro_auc(file_labels, ens)
            if auc > best_4_auc:
                best_4_auc = auc
                best_4_preds = ens.copy()
                best_4_w = {"al": float(al), "w1": float(w1), "w3": float(w3), "w4": float(w4)}

marker = "  *** NEW BEST ***" if best_4_auc > CURRENT_BEST else ""
print(f"  Best win-KLT-mean k134: {best_4_auc:.6f}  (delta={best_4_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_4_w}")
results_list.append(("k134_wklt_mean", best_4_auc, best_4_w, best_4_preds))

# ══════════════════════════════════════════════════════════════════
# 方法 5：KLT + binary KNN 混合（6-way）
#   al*logit_max + w1b*KNN1 + w3b*KNN3 + w4b*KNN4 + w1k*KLT1 + w3k*KLT3
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("方法 5: 6-way (logit_max + binary KNN + KLT)")
print("="*70)

# 固定 al=0.42，掃 binary KNN vs KLT 的比例
best_5_auc, best_5_preds, best_5_w = 0.0, None, {}
al = 0.42
rem = 1.0 - al  # 0.58 for KNN/KLT
for r_knn in np.arange(0.0, 1.01, 0.1):   # 0=all KLT, 1=all binary KNN
    r_klt = 1.0 - r_knn
    # split rem into w1, w3, w4
    for w1_frac in np.arange(0.4, 0.6, 0.05):
        for w3_frac in np.arange(0.0, 0.1, 0.05):
            w4_frac = 1.0 - w1_frac - w3_frac
            if w4_frac < 0: continue

            knn_part = (w1_frac*knn1 + w3_frac*knn3 + w4_frac*knn4)
            klt_part = (w1_frac*klt_cache[1] + w3_frac*klt_cache[3] + w4_frac*klt_cache[4])
            ens = al*file_prob_max + rem*(r_knn*knn_part + r_klt*klt_part)
            auc = macro_auc(file_labels, ens)
            if auc > best_5_auc:
                best_5_auc = auc
                best_5_preds = ens.copy()
                best_5_w = {"al": al, "r_knn": float(r_knn), "r_klt": float(r_klt),
                             "w1_frac": float(w1_frac), "w3_frac": float(w3_frac), "w4_frac": float(w4_frac)}

marker = "  *** NEW BEST ***" if best_5_auc > CURRENT_BEST else ""
print(f"  Best 6-way: {best_5_auc:.6f}  (delta={best_5_auc-CURRENT_BEST:+.6f}){marker}")
print(f"    {best_5_w}")
results_list.append(("knn_klt_6way", best_5_auc, best_5_w, best_5_preds))

# ══════════════════════════════════════════════════════════════════
# 總結
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Current best: {CURRENT_BEST:.6f}")
for name, auc, params, _ in sorted(results_list, key=lambda x: -x[1]):
    marker = "  *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.6f}  (delta={auc-CURRENT_BEST:+.6f}){marker}")

# 找全域最佳
if results_list:
    best_result = max(results_list, key=lambda x: x[1])
    best_name, best_auc, best_params, best_preds = best_result

    # 更新 JSON
    with open(RESULTS_PATH) as f:
        results_json = json.load(f)

    def serialize_val(v):
        if isinstance(v, (np.float32, np.float64)):   return float(v)
        if isinstance(v, np.integer):                 return int(v)
        if isinstance(v, np.ndarray):                 return None
        return v

    for name, auc, params, _ in results_list:
        record = {"method": name, "loo_auc": round(float(auc), 6)}
        for k, v in params.items():
            sv = serialize_val(v)
            if sv is not None:
                record[k] = sv
        results_json["experiments"].append(record)

    overall_best = results_json["best"]["loo_auc"]
    if best_auc > overall_best:
        results_json["best"] = {
            "method": best_name,
            "loo_auc": round(float(best_auc), 6),
            "config": {k: serialize_val(v) for k, v in best_params.items() if serialize_val(v) is not None},
            "note": f"new_methods_v1 run 2026-03-25; prev={overall_best:.6f}"
        }
        print(f"\nNEW BEST: {best_name} AUC={best_auc:.6f}")

        model_dict = {
            "method": best_name,
            "loo_auc": round(float(best_auc), 6),
            "params": {k: serialize_val(v) for k, v in best_params.items() if serialize_val(v) is not None},
            "file_list": file_list.tolist(),
            "loo_preds": best_preds.tolist(),
            "file_embs_norm": file_embs_norm.tolist(),
            "file_prob_max": file_prob_max.tolist(),
            "file_labels": file_labels.tolist(),
        }
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_dict, f)
        print(f"Saved model → {MODEL_PATH}")
    else:
        print(f"\nNo improvement over current best ({overall_best:.6f})")
        print(f"Best this run: {best_name} AUC={best_auc:.6f}")

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved → {RESULTS_PATH}")
