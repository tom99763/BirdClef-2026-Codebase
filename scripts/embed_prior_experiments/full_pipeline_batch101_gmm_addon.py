"""
Batch 101 — GMM/Mahalanobis/BayesianRidge as Additive Signals on Triple Chain
===============================================================================
Current best: triple_30w02_40w03_40bw6w01 LOO=0.992166

Priority methods from loop spec (all previously tested STANDALONE in early batches,
but NEVER as additive blend signals on top of the current triple Fisher chain):

1. GMM per-species: fit 2-component GMM in ICA space, use log-likelihood ratio
2. Mahalanobis KNN: per-species covariance-weighted distance score
3. Bayesian Ridge: per-species regression on ICA features
4. RBF Nystroem + LogReg: kernel feature map + logistic regression
5. Attention-weighted KNN: KNN with logit-attention reweighting

These are tested as additive blend signals: final = (1-w)*triple_ref + w*new_signal
"""
import numpy as np
import json
import pickle
import copy
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from numpy.linalg import norm

ROOT = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

DATA = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
labels_win  = DATA["labels"].astype(np.float32)
logit_win   = DATA["logits"].astype(np.float32)
n_windows   = DATA["n_windows"]
n_files     = len(n_windows)
n_species   = labels_win.shape[1]
file_start  = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end    = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(739, np.int32)
for fi in range(n_files):
    win_file_id[file_start[fi]:file_end[fi]] = fi
EPS = 1e-8

with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

ew_ica  = ep["emb_win_ica_norm"]
ew_pca  = ep["emb_win_pca_norm"]
ew_std  = ep["emb_win_std_norm"]
ew_nmf  = ep["emb_win_nmf_norm"]
file_labels = ep["file_labels"]
cfg = ep["config"]

print(f"[batch101] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch101] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

def macro_auc(s):
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            aucs.append(roc_auc_score(y, s[:, si]))
    return float(np.mean(aucs)) if aucs else 0.0

def wl_loo(ew, k_neg, wmp, wma):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]; ps = te @ pw.T
            pp = pw.mean(0); pp /= norm(pp) + EPS
            sp = wmp * ps.max(1) + (1 - wmp) * (te @ pp)
            if nm.any() and k_neg > 0:
                nw = tr[nm]; ns = te @ nw.T; k2 = min(k_neg, ns.shape[1])
                if k2 > 0:
                    tn = nw[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
                    tn /= norm(tn, axis=1, keepdims=True) + EPS
                    ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
                else:
                    ws[:, si] = (sp + 1) / 2
            else:
                ws[:, si] = (sp + 1) / 2
        out[fi] = wma * ws.max(0) + (1 - wma) * ws.mean(0)
    return out

def make_lp(T):
    sig = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T, -88, 88)))
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

def make_sp(T):
    sm_r = logit_win / T; sm_r -= sm_r.max(1, keepdims=True)
    sm_e = np.exp(sm_r); smp = sm_e / (sm_e.sum(1, keepdims=True) + EPS)
    return np.stack([smp[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

def compute_subspace(ew_sp, n_comp=2, wma_ss=0.92):
    ss = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_sp[win_file_id == fi]; tr = ew_sp[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32); dim = te.shape[1]
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; k = min(n_comp, len(pos) - 1, dim - 1)
            if k < 1:
                pp = pos.mean(0); pp /= norm(pp) + EPS
                ws[:, si] = np.clip((te @ pp + 1) / 2, 0, 1); continue
            try:
                p = SklearnPCA(n_components=k); p.fit(pos)
                te_r = p.inverse_transform(p.transform(te))
                err = norm(te - te_r, axis=1)
                ws[:, si] = np.clip(1 - err / (norm(te, axis=1) + EPS), 0, 1)
            except:
                ws[:, si] = 0.5
        ss[fi] = wma_ss * ws.max(0) + (1 - wma_ss) * ws.mean(0)
    return ss

def proto_kde_loo(ew, bw=0.08):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        sims = te @ tr.T; ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pw = tr[pos_idx]; centroid = pw.mean(0); centroid /= (norm(centroid) + EPS)
            proto_w = np.clip(pw @ centroid, 0, None); proto_w /= (proto_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def fisher_kde_loo(ew, bw=0.06):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            w_dim = fisher / (norm(fisher) + EPS)
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_w @ centroid, 0, None); proto_w /= (proto_w.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def fisher_hard_kde_loo(ew, bw=0.06, top_k=30):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_w @ centroid, 0, None); proto_w /= (proto_w.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ── Pre-compute base + triple ─────────────────────────────────────────────────
print("Pre-computing base chain...", flush=True)
t0 = time.time()
s_ica = wl_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
s_std = wl_loo(ew_std, cfg["std_pca80"]["k_neg"], cfg["std_pca80"]["w_max_pos"], cfg["std_pca80"]["w_max_agg"])
s_pca = wl_loo(ew_pca, cfg["pca80"]["k_neg"], cfg["pca80"]["w_max_pos"], cfg["pca80"]["w_max_agg"])
s_nmf = wl_loo(ew_nmf, cfg["nmf"]["k_neg"], cfg["nmf"]["w_max_pos"], cfg["nmf"]["w_max_agg"])
uh_b  = cfg["w_ica100"] * s_ica + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
uh_nmf = cfg["nmf"]["uh_scale"] * uh_b + cfg["nmf"]["w_nmf"] * s_nmf
pT8   = make_lp(cfg["logit_temperature"]); pmt = (pT8 + make_lp(10.0)) / 2
sm6   = make_sp(cfg["softmax_temp"]); ss2 = compute_subspace(ew_pca, n_comp=2, wma_ss=0.92)
kde08 = proto_kde_loo(ew_ica, bw=0.08)
w_uh = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur  = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
final_ref = 0.96 * base_cur + 0.04 * kde08
f06 = fisher_kde_loo(ew_ica, bw=0.06)
fin_ref = (1 - 0.05) * final_ref + 0.05 * f06
fh30_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
fh40_b7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
fh40_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=40)
triple_ref = (1-0.02-0.03-0.01)*fin_ref + 0.02*fh30_b6 + 0.03*fh40_b7 + 0.01*fh40_b6
auc_t = macro_auc(triple_ref)
print(f"  triple_ref: {auc_t:.6f} (expected 0.992166) [{time.time()-t0:.0f}s]", flush=True)

results = {}
new_best_loo = best_loo
new_best_method = None

def reg(name, auc):
    delta = auc - best_loo
    mark = " *** NEW BEST ***" if auc > best_loo else (" (near-best)" if auc > best_loo - 0.0003 else "")
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch101"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# Method 1: GMM per-species (log-likelihood ratio as score)
# Fit 2-component GMM on positive windows in ICA space (top Fisher dims)
# Score = log p(x|pos_GMM) - log p(x|background_GMM)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] GMM per-species log-likelihood ratio...", flush=True)
t1 = time.time()

def gmm_loo(ew, n_components_pos=2, n_components_bg=3, top_fisher_k=30):
    """
    Per-species GMM: fit GMM on positive windows, compare to background GMM.
    Score = sigmoid(log p(x|pos) - log p(x|bg)) averaged over windows.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        # Fit background GMM once per fold
        try:
            bg_gmm = GaussianMixture(n_components=n_components_bg, covariance_type='diag',
                                      max_iter=50, random_state=0)
            bg_gmm.fit(tr)
            bg_ll = bg_gmm.score_samples(te)  # log-likelihood
        except:
            bg_ll = np.zeros(len(te))
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]
            # Select top Fisher dimensions first
            nm = tl[:, si] < 0.1
            neg_wins = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_fisher_k]
            pos_sub = pos_wins[:, top_idx]
            te_sub  = te[:, top_idx]
            n_comp = min(n_components_pos, len(pos_sub))
            if n_comp < 1: ws[:, si] = 0.5; continue
            try:
                pos_gmm = GaussianMixture(n_components=n_comp, covariance_type='diag',
                                           max_iter=50, random_state=0)
                pos_gmm.fit(pos_sub)
                pos_ll = pos_gmm.score_samples(te_sub)
                bg_sub_ll = bg_gmm.score_samples(te[:, top_idx]) if hasattr(bg_gmm, '_means') else bg_ll
                log_ratio = pos_ll - bg_sub_ll
                score = 1.0 / (1.0 + np.exp(-np.clip(log_ratio * 0.5, -10, 10)))
                ws[:, si] = score
            except:
                ws[:, si] = 0.5
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# Test with different Fisher top-k dims
for k in [20, 30, 40]:
    s_gmm = gmm_loo(ew_ica, n_components_pos=2, n_components_bg=3, top_fisher_k=k)
    for w_int in [1, 2, 3]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s_gmm
        reg(f"gmm_addon_k{k}_np2_w{w_int:02d}", macro_auc(blend))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Mahalanobis KNN on Fisher-selected subspace
# Per-species Mahalanobis distance using positive-class covariance
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Mahalanobis distance KNN (Fisher subspace)...", flush=True)
t1 = time.time()

def mahal_knn_fisher_loo(ew, top_k_fisher=30, k_nn=5, reg=1e-3):
    """
    Mahalanobis KNN: use Fisher-selected dims, compute per-class Mahalanobis distance.
    Score = 1 / (1 + mean_mahal_dist_to_k_nearest_positives)
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            pos_sub = pos_wins[:, top_idx]  # (n_pos, k)
            te_sub  = te[:, top_idx]         # (n_te, k)
            # Compute covariance of positive class + regularize
            cov_p = np.cov(pos_sub.T) if len(pos_sub) > 1 else np.eye(top_k_fisher)
            cov_p = cov_p + reg * np.eye(top_k_fisher)
            try:
                inv_cov = np.linalg.inv(cov_p)
            except:
                inv_cov = np.eye(top_k_fisher)
            # Compute Mahalanobis distance from each test window to each positive
            diff = te_sub[:, None, :] - pos_sub[None, :, :]  # (n_te, n_pos, k)
            mahal_sq = np.einsum('tpd,de,tpe->tp', diff, inv_cov, diff)  # (n_te, n_pos)
            k2 = min(k_nn, mahal_sq.shape[1])
            # Score: inverse of mean Mahalanobis distance to k nearest positives
            knn_mahal = np.sort(mahal_sq, axis=1)[:, :k2].mean(1)
            score = 1.0 / (1.0 + knn_mahal)
            ws[:, si] = score
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for k_fisher, k_nn in [(30, 5), (40, 5), (30, 3), (40, 3)]:
    s_mah = mahal_knn_fisher_loo(ew_ica, top_k_fisher=k_fisher, k_nn=k_nn)
    for w_int in [1, 2, 3]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s_mah
        reg(f"mahal_addon_kf{k_fisher}_kn{k_nn}_w{w_int:02d}", macro_auc(blend))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Bayesian Ridge per-species on Fisher-selected ICA dims
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Bayesian Ridge per-species (Fisher ICA dims)...", flush=True)
t1 = time.time()

def bayesian_ridge_fisher_loo(ew, top_k_fisher=30):
    """Per-species Bayesian Ridge on Fisher-selected ICA dims."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            # Build training set: positive = 1, negative = 0
            all_mask = pm | nm
            X_tr = tr[all_mask][:, top_idx]
            y_tr = (tl[all_mask, si] > 0.5).astype(float)
            if y_tr.sum() < 1 or (1 - y_tr).sum() < 1:
                ws[:, si] = 0.5; continue
            X_te = te[:, top_idx]
            try:
                br = BayesianRidge(max_iter=100)
                br.fit(X_tr, y_tr)
                pred = br.predict(X_te)
                score = np.clip(pred, 0, 1)
                ws[:, si] = score
            except:
                ws[:, si] = 0.5
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for k_fisher in [20, 30, 40]:
    s_br = bayesian_ridge_fisher_loo(ew_ica, top_k_fisher=k_fisher)
    for w_int in [1, 2, 3]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s_br
        reg(f"brdg_addon_kf{k_fisher}_w{w_int:02d}", macro_auc(blend))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Method 4: RBF Nystroem + LogReg on Fisher-selected ICA dims
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] RBF Nystroem + LogReg (Fisher ICA dims)...", flush=True)
t1 = time.time()

def nystroem_logreg_fisher_loo(ew, top_k_fisher=30, n_components=64, gamma=0.5):
    """Per-species Nystroem features + LogReg on Fisher-selected dims."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            all_mask = pm | nm
            X_tr = tr[all_mask][:, top_idx]
            y_tr = (tl[all_mask, si] > 0.5).astype(int)
            if y_tr.sum() < 2 or (1 - y_tr).sum() < 2:
                ws[:, si] = 0.5; continue
            X_te = te[:, top_idx]
            n_comp = min(n_components, len(X_tr) - 1)
            try:
                nys = Nystroem(kernel='rbf', gamma=gamma, n_components=n_comp, random_state=0)
                lr = LogisticRegression(C=1.0, max_iter=100, random_state=0)
                X_tr_nys = nys.fit_transform(X_tr)
                X_te_nys = nys.transform(X_te)
                lr.fit(X_tr_nys, y_tr)
                prob = lr.predict_proba(X_te_nys)[:, 1]
                ws[:, si] = prob
            except:
                ws[:, si] = 0.5
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# Use smaller k_fisher and n_components for speed
for k_fisher, n_comp in [(30, 32), (40, 32)]:
    s_nys = nystroem_logreg_fisher_loo(ew_ica, top_k_fisher=k_fisher, n_components=n_comp)
    for w_int in [1, 2, 3]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s_nys
        reg(f"nys_addon_kf{k_fisher}_nc{n_comp}_w{w_int:02d}", macro_auc(blend))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Method 5: Attention-weighted KNN (logit-attention)
# Use current logit predictions to weight the KNN neighbors
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Attention-weighted KNN (logit-gated Fisher KDE)...", flush=True)
t1 = time.time()

def attn_knn_fisher_loo(ew, top_k_fisher=40, bw=0.07, attn_T=6.0):
    """
    Fisher hard KDE but weight training windows by their logit confidence.
    Attention weight = softmax(logit_si / T) for species si.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        tl_logit = logit_win[win_file_id != fi]  # logit scores for train windows
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            # Attention: weight by logit confidence of positive training windows
            pos_logit_si = tl_logit[pm, si]  # logit scores for this species
            attn = 1.0 / (1.0 + np.exp(-np.clip(pos_logit_si / attn_T, -10, 10)))
            attn /= (attn.sum() + EPS)
            # Centroid weighted by attention
            centroid = (tr_w * attn[:, None]).sum(0); centroid /= norm(centroid) + EPS
            proto_w = attn  # use attention as proto_w directly
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for k_fisher, bw, attn_T in [(40, 0.07, 6.0), (40, 0.07, 8.0), (30, 0.06, 6.0)]:
    s_attn = attn_knn_fisher_loo(ew_ica, top_k_fisher=k_fisher, bw=bw, attn_T=attn_T)
    for w_int in [1, 2, 3]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s_attn
        reg(f"attn_addon_kf{k_fisher}_bw{int(bw*100)}_T{int(attn_T)}_w{w_int:02d}", macro_auc(blend))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch101] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

if results:
    all_sorted = sorted(results.items(), key=lambda x: -x[1]["loo_auc"])
    top_tag, top_v = all_sorted[0]
    top_auc = top_v["loo_auc"]
    if top_auc > best_loo:
        print(f"  *** NEW BEST: {top_tag} LOO={top_auc:.6f} (+{top_auc-best_loo:.6f}) ***", flush=True)
        new_best_loo = top_auc
        new_best_method = top_tag
    else:
        print(f"  Best this batch: {top_auc:.6f} ({top_tag}) — no improvement", flush=True)
    print("\n  Top-10:", flush=True)
    for tag, v in all_sorted[:10]:
        d = v["loo_auc"] - best_loo
        print(f"    {tag}: {v['loo_auc']:.6f} ({d:+.6f})", flush=True)

res2 = json.load(open(RESULTS_PATH))
if isinstance(res2.get("experiments"), list):
    res2["experiments"].extend(list(results.values()))
else:
    res2["experiments"] = list(results.values())

if new_best_loo > best_loo:
    res2["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch101"}
    ep2 = copy.deepcopy(ep)
    ep2["loo_auc"] = new_best_loo
    ep2["method"] = new_best_method
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep2, f)
    print(f"\n  SAVED: {new_best_method} LOO={new_best_loo:.6f}", flush=True)

json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"\nSaved {len(results)} experiments to JSON.", flush=True)
