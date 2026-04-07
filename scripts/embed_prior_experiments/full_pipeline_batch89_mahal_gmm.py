"""
Batch 89: Mahalanobis KDE + GMM per Species
=============================================
Current best: softmax_T6_proto_kde LOO=0.991782

Priority methods from experiment queue:
1. Mahalanobis distance KDE:
   - Per species, compute covariance of positive windows in ICA space
   - Score = Gaussian kernel with Mahalanobis distance (instead of cosine)
   - Accounts for species-specific embedding spread/orientation

2. GMM per species (EM):
   - Fit GMM(k=1,2) on positive windows in ICA-100 space
   - score = log_prob(test_window) from GMM → max over windows

Both used as KDE replacement: final = (1-w)*base + w*new_prior

CRITICAL: Uses stored pkl embeddings + n_windows-based file ordering.
"""

import numpy as np
import json
import pickle
import time
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.mixture import GaussianMixture
from numpy.linalg import norm, eigh
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────
DATA = np.load("outputs/perch_labeled_ss.npz")
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

with open("outputs/embed_prior_model.pkl", "rb") as f:
    ep = pickle.load(f)

ew_ica = ep["emb_win_ica_norm"]
file_labels = ep["file_labels"]
cfg = ep["config"]

print(f"Loaded: ICA{ew_ica.shape}", flush=True)

# ── Standard helpers ───────────────────────────────────────────────────────
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

def make_logit_pred(T):
    sig = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T, -88, 88)))
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

def make_softmax_pred(T):
    sm_r = logit_win / T; sm_r -= sm_r.max(1, keepdims=True)
    sm_e = np.exp(sm_r); smp = sm_e / (sm_e.sum(1, keepdims=True) + EPS)
    return np.stack([smp[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

def compute_subspace(ew_sp, n_comp=2, wma_ss=0.92):
    ss = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_sp[win_file_id == fi]; tr = ew_sp[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        dim = te.shape[1]
        for fi2 in range(n_species):
            pass
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; k = min(n_comp, len(pos) - 1, dim - 1)
            if k < 1:
                pp = pos.mean(0); pp /= norm(pp) + EPS
                ws[:, si] = np.clip((te @ pp + 1) / 2, 0, 1); continue
            try:
                pca_sp = SklearnPCA(n_components=k); pca_sp.fit(pos)
                te_r = pca_sp.inverse_transform(pca_sp.transform(te))
                err = norm(te - te_r, axis=1)
                ws[:, si] = np.clip(1 - err / (norm(te, axis=1) + EPS), 0, 1)
            except Exception:
                ws[:, si] = 0.5
        ss[fi] = wma_ss * ws.max(0) + (1 - wma_ss) * ws.mean(0)
    return ss

def proto_kde_loo_ica(bw=0.08):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]
            centroid = pos_wins.mean(0); centroid /= (norm(centroid) + EPS)
            proto_w = np.clip(pos_wins @ centroid, 0, None)
            proto_w = proto_w / (proto_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ── New Methods ────────────────────────────────────────────────────────────
def mahal_kde_loo(bw_scale=1.0, reg=1e-3, proto_weighted=True):
    """Mahalanobis distance KDE:
    For each species, compute per-species covariance of positive windows.
    score(x) = exp(-0.5 * d_mahal(x, pos_i)^2 * bw_scale) weighted by proto_w
    Uses PCA-whitening for numerical stability (top k=min(n_pos-1, 20) components).
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]  # (n_pos, d)
            n_pos = len(pos_idx)

            if n_pos < 3:
                # Fall back to cosine proto KDE
                centroid = pos_wins.mean(0); centroid /= (norm(centroid) + EPS)
                sims = te @ pos_wins.T
                kern = np.exp((sims - 1.0) / (0.08**2 + EPS))
                ws[:, si] = np.clip(kern.mean(1), 0, None)
            else:
                # PCA-whiten positive subspace (top k components)
                k = min(n_pos - 1, 20)
                mu = pos_wins.mean(0)
                centered = pos_wins - mu
                try:
                    pca_m = SklearnPCA(n_components=k).fit(centered)
                    # Project test windows
                    te_proj = pca_m.transform(te - mu)  # (n_te, k)
                    tr_proj = pca_m.transform(pos_wins - mu)  # (n_pos, k)
                    # Mahalanobis in whitened space: sqrt(var) normalization
                    std = np.sqrt(pca_m.explained_variance_ + reg)
                    te_w = te_proj / (std + EPS)   # whitened test
                    tr_w = tr_proj / (std + EPS)   # whitened train positives

                    # Gaussian kernel with Mahalanobis distance
                    # d_mahal^2 = ||te_w - tr_w_i||^2
                    diff = te_w[:, None, :] - tr_w[None, :, :]  # (n_te, n_pos, k)
                    dists_sq = (diff**2).sum(-1)  # (n_te, n_pos)
                    kern = np.exp(-0.5 * dists_sq * bw_scale)

                    if proto_weighted and n_pos > 1:
                        centroid_w = tr_w.mean(0)
                        proto_w_s = np.exp(-0.5 * ((tr_w - centroid_w)**2).sum(1) * bw_scale)
                        proto_w_s = proto_w_s / (proto_w_s.sum() + EPS)
                        ws[:, si] = np.clip((kern * proto_w_s[None, :]).sum(1), 0, None)
                    else:
                        ws[:, si] = np.clip(kern.mean(1), 0, None)
                except Exception:
                    ws[:, si] = 0.5
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def gmm_kde_loo(n_components=1, covariance_type="diag"):
    """GMM per species in ICA space.
    Fit GMM(n_components) on positive windows.
    Score = exp(log_prob(test_window)) → per-file max.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx].astype(np.float64)
            n_pos = len(pos_idx)
            k = min(n_components, n_pos)
            if k < 1 or n_pos < 2:
                ws[:, si] = 0.5; continue
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    reg_covar=1e-4,
                    max_iter=50,
                    random_state=42
                )
                gmm.fit(pos_wins)
                log_probs = gmm.score_samples(te.astype(np.float64))  # (n_te,)
                # Normalize: sigmoid of log_prob to get [0,1]
                lp = log_probs - log_probs.max()
                ws[:, si] = np.clip(np.exp(lp), 0, None)
            except Exception:
                ws[:, si] = 0.5
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def gmm_pca_loo(n_components_gmm=1, n_pca=20):
    """GMM in reduced PCA space (top 20 dims of positive subspace)."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]
            n_pos = len(pos_idx)
            if n_pos < 3: ws[:, si] = 0.5; continue
            k_pca = min(n_pca, n_pos - 1, pos_wins.shape[1] - 1)
            if k_pca < 1: ws[:, si] = 0.5; continue
            try:
                pca_g = SklearnPCA(n_components=k_pca)
                pca_g.fit(pos_wins)
                pos_proj = pca_g.transform(pos_wins).astype(np.float64)
                te_proj  = pca_g.transform(te).astype(np.float64)
                k_gmm = min(n_components_gmm, n_pos)
                gmm = GaussianMixture(n_components=k_gmm, covariance_type="diag",
                                      reg_covar=1e-4, max_iter=50, random_state=42)
                gmm.fit(pos_proj)
                log_probs = gmm.score_samples(te_proj)
                lp = log_probs - log_probs.max()
                ws[:, si] = np.clip(np.exp(lp), 0, None)
            except Exception:
                ws[:, si] = 0.5
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ── Pre-compute base ───────────────────────────────────────────────────────
ew_pca = ep["emb_win_pca_norm"]
ew_std = ep["emb_win_std_norm"]
ew_nmf = ep["emb_win_nmf_norm"]

print("Pre-computing base...", flush=True)
t0 = time.time()
s_ica = wl_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
s_std = wl_loo(ew_std, cfg["std_pca80"]["k_neg"], cfg["std_pca80"]["w_max_pos"], cfg["std_pca80"]["w_max_agg"])
s_pca = wl_loo(ew_pca, cfg["pca80"]["k_neg"], cfg["pca80"]["w_max_pos"], cfg["pca80"]["w_max_agg"])
s_nmf = wl_loo(ew_nmf, cfg["nmf"]["k_neg"], cfg["nmf"]["w_max_pos"], cfg["nmf"]["w_max_agg"])
uh_b  = cfg["w_ica100"] * s_ica + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
uh_nmf = cfg["nmf"]["uh_scale"] * uh_b + cfg["nmf"]["w_nmf"] * s_nmf
pT8   = make_logit_pred(cfg["logit_temperature"])
pmt810 = (pT8 + make_logit_pred(10.0)) / 2
sm6   = make_softmax_pred(cfg["softmax_temp"])
ss2   = compute_subspace(ew_pca, n_comp=2, wma_ss=0.92)
w_uh  = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base  = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt810 + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
kde_ica = proto_kde_loo_ica(bw=0.08)
print(f"  Base done ({time.time()-t0:.0f}s)", flush=True)
print(f"Base AUC: {macro_auc(base):.6f}")
print(f"Reference (should ~0.991782): {macro_auc(0.96*base + 0.04*kde_ica):.6f}")

# ── Pre-compute new KDE variants ──────────────────────────────────────────
print("\nPre-computing Mahalanobis KDE...", flush=True)
t1 = time.time()
mahal_bw1_p  = mahal_kde_loo(bw_scale=1.0,  proto_weighted=True)
mahal_bw2_p  = mahal_kde_loo(bw_scale=2.0,  proto_weighted=True)
mahal_bw05_p = mahal_kde_loo(bw_scale=0.5,  proto_weighted=True)
mahal_bw1_np = mahal_kde_loo(bw_scale=1.0,  proto_weighted=False)
print(f"  Mahalanobis done ({time.time()-t1:.0f}s)", flush=True)

print("Pre-computing GMM...", flush=True)
t1 = time.time()
gmm1_full  = gmm_kde_loo(n_components=1, covariance_type="diag")
gmm2_full  = gmm_kde_loo(n_components=2, covariance_type="diag")
gmm1_sph   = gmm_kde_loo(n_components=1, covariance_type="spherical")
gmm1_pca10 = gmm_pca_loo(n_components_gmm=1, n_pca=10)
gmm1_pca20 = gmm_pca_loo(n_components_gmm=1, n_pca=20)
gmm2_pca10 = gmm_pca_loo(n_components_gmm=2, n_pca=10)
print(f"  GMM done ({time.time()-t1:.0f}s)", flush=True)

# ── Load results and evaluate ─────────────────────────────────────────────
RES_PATH = Path("outputs/embed_prior_results.json")
with open(RES_PATH) as f:
    results = json.load(f)
best_auc = results["best"]["loo_auc"]
best_method = results["best"]["method"]
print(f"\nCurrent best: {best_method} = {best_auc:.6f}\n", flush=True)

experiments = []
def eval_blend(name, kde_score, w_kde):
    final = (1 - w_kde) * base + w_kde * kde_score
    auc = macro_auc(final)
    delta = auc - best_auc
    mark = " ***NEW BEST***" if auc > best_auc + 1e-6 else ""
    print(f"  {name}: {auc:.6f}  (Δ={delta:+.6f}){mark}", flush=True)
    experiments.append({"method": name, "loo_auc": auc})
    return auc

print("=== Mahalanobis KDE (replace cosine KDE) ===")
for name, kde in [("mahal_bw1_p", mahal_bw1_p), ("mahal_bw2_p", mahal_bw2_p),
                   ("mahal_bw05_p", mahal_bw05_p), ("mahal_bw1_np", mahal_bw1_np)]:
    for wk in [0.03, 0.04, 0.05, 0.06]:
        eval_blend(f"{name}_wk{int(wk*100):02d}", kde, wk)

print("\n=== Mahalanobis + ICA cosine blend ===")
for w_m in [0.3, 0.5, 0.7]:
    blended = w_m * mahal_bw1_p + (1-w_m) * kde_ica
    for si in range(n_species):
        mx = blended[:, si].max()
        if mx > EPS: blended[:, si] /= mx
    for wk in [0.04, 0.05]:
        eval_blend(f"mahal_ica_wm{int(w_m*10)}_wk{int(wk*100):02d}", blended, wk)

print("\n=== GMM full ICA space ===")
for name, kde in [("gmm1_diag", gmm1_full), ("gmm2_diag", gmm2_full), ("gmm1_sph", gmm1_sph)]:
    for wk in [0.03, 0.04, 0.05, 0.06]:
        eval_blend(f"{name}_wk{int(wk*100):02d}", kde, wk)

print("\n=== GMM in PCA subspace ===")
for name, kde in [("gmm1_pca10", gmm1_pca10), ("gmm1_pca20", gmm1_pca20), ("gmm2_pca10", gmm2_pca10)]:
    for wk in [0.03, 0.04, 0.05, 0.06]:
        eval_blend(f"{name}_wk{int(wk*100):02d}", kde, wk)

print("\n=== GMM + ICA cosine blend ===")
for w_g in [0.3, 0.5, 0.7]:
    blended = w_g * gmm1_full + (1-w_g) * kde_ica
    for si in range(n_species):
        mx = blended[:, si].max()
        if mx > EPS: blended[:, si] /= mx
    for wk in [0.04, 0.05]:
        eval_blend(f"gmm1_ica_wg{int(w_g*10)}_wk{int(wk*100):02d}", blended, wk)

# ── Finalize ────────────────────────────────────────────────────────────────
best_new = max(experiments, key=lambda x: x["loo_auc"])
delta_best = best_new["loo_auc"] - best_auc

print(f"\n{'='*60}")
print(f"Batch 89 Summary")
print(f"Experiments run: {len(experiments)}")
print(f"Best new: {best_new['method']} = {best_new['loo_auc']:.6f}")
print(f"Current best: {best_method} = {best_auc:.6f}")
print(f"Delta: {delta_best:+.6f}")

if delta_best > 1e-6:
    print(f"\n*** NEW BEST: {best_new['method']} = {best_new['loo_auc']:.6f} ***")
    results["best"] = best_new

for e in experiments:
    results["experiments"].append(e)

with open(RES_PATH, "w") as f:
    json.dump(results, f)
print("Results saved.")

print(f"\nTop 5 new:")
for e in sorted(experiments, key=lambda x: -x["loo_auc"])[:5]:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
