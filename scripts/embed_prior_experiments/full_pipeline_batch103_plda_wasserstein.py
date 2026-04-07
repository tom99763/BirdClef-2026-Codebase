"""
Batch 103 — PLDA / Wasserstein / EVT / Sinkhorn as Additive Signals
===============================================================================
Current best: attn_addon_kf40_bw7_T6_w01 LOO=0.992180
              formula: 0.99*triple_ref + 0.01*attn_kf40_bw7_T6

After 14502 experiments exhausting KDE/KNN/Fisher/GMM/Attention variants,
trying genuinely novel signal types not yet in experiments:

1. PLDA (Probabilistic LDA) - between/within scatter log-likelihood ratio
2. Wasserstein-1 distance (Earth Mover's Distance) in Fisher subspace
3. Extreme Value Theory (EVT) - Weibull tail modeling of top similarities
4. Logit-filtered KDE (only positive windows with logit > threshold)
5. CovFisher (diagonal Mahalanobis in Fisher-weighted ICA space)

All tested as: attn_ref = 0.99*triple_ref + 0.01*attn, then blend w/ new signal
"""
import numpy as np
import json
import pickle
import copy
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
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

print(f"[batch103] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch103] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

def macro_auc(s):
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            aucs.append(roc_auc_score(y, s[:, si]))
    return float(np.mean(aucs)) if aucs else 0.0

# ── Exact helper functions from batch101 ─────────────────────────────────────
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

def attn_knn_fisher_loo(ew, top_k_fisher=40, bw=0.07, attn_T=6.0):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        tl_logit = logit_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos.mean(0); mu_n = neg.mean(0)
            var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            pos_logit_si = tl_logit[pm, si]
            if attn_T >= 100.0:
                attn = np.ones(len(pos_logit_si), np.float32) / len(pos_logit_si)
            else:
                attn = 1.0 / (1.0 + np.exp(-np.clip(pos_logit_si / attn_T, -10, 10)))
                attn /= (attn.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * attn[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ── Pre-compute full chain up to attn_ref ─────────────────────────────────────
print("Pre-computing chain...", flush=True)
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
s_attn = attn_knn_fisher_loo(ew_ica, top_k_fisher=40, bw=0.07, attn_T=6.0)
attn_ref = 0.99 * triple_ref + 0.01 * s_attn
auc_ar = macro_auc(attn_ref)
print(f"  attn_ref: {auc_ar:.6f} (expected 0.992180) [{time.time()-t0:.0f}s]", flush=True)

results = {}
new_best_loo = best_loo
new_best_method = None

def reg(name, auc):
    global new_best_loo, new_best_method
    delta = auc - best_loo
    if auc > new_best_loo:
        new_best_loo = auc
        new_best_method = name
    mark = " *** NEW BEST ***" if auc > best_loo else (" (near-best)" if auc > best_loo - 0.0003 else "")
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch103"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: PLDA (Probabilistic LDA) — between/within scatter log-likelihood ratio
# Score = log|S_B| - log|S_W| projected score for each test window
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] PLDA scoring (between/within scatter)...", flush=True)
t1 = time.time()

def plda_loo(ew, top_k_fisher=40, reg_coeff=1e-3):
    """
    PLDA-style score: for each species, project into 1D direction that maximizes
    between/within class variance ratio (= Fisher LDA direction), then score
    each test window by its projection value.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
            # Pre-select Fisher dims
            mu_p = pos.mean(0); mu_n = neg.mean(0)
            var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            pos_sub = pos[:, top_idx]; neg_sub = neg[:, top_idx]; te_sub = te[:, top_idx]
            # Within-class scatter (pooled)
            all_sub = np.vstack([pos_sub, neg_sub])
            grand_mean = all_sub.mean(0)
            Sw = (pos_sub - pos_sub.mean(0)).T @ (pos_sub - pos_sub.mean(0))
            Sw += (neg_sub - neg_sub.mean(0)).T @ (neg_sub - neg_sub.mean(0))
            Sw /= (len(pos_sub) + len(neg_sub))
            Sw += reg_coeff * np.eye(top_k_fisher)
            # Between-class direction: mu_p - mu_n
            mu_diff = (pos_sub.mean(0) - neg_sub.mean(0))
            try:
                Sw_inv = np.linalg.inv(Sw)
            except:
                Sw_inv = np.eye(top_k_fisher)
            # LDA direction (Fisher linear discriminant)
            lda_dir = Sw_inv @ mu_diff
            lda_dir /= (norm(lda_dir) + EPS)
            # Score = projection onto LDA direction
            score = te_sub @ lda_dir  # (n_te,)
            # Normalize by range
            pos_proj = pos_sub @ lda_dir
            mn, mx = pos_proj.min(), pos_proj.max()
            if mx > mn:
                score = (score - mn) / (mx - mn + EPS)
            else:
                score = np.clip(score, 0, None)
            ws[:, si] = np.clip(score, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for k in [30, 40, 50]:
    s = plda_loo(ew_ica, top_k_fisher=k)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * attn_ref + w_val * s
        reg(f"plda_kf{k}_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: Wasserstein-1 / Earth Mover's Distance in Fisher subspace
# Score = 1 / (1 + W1(test_windows, pos_train_windows))
# Use sorted cosine similarities as 1D distribution, compute W1
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Wasserstein-1 distance scoring...", flush=True)
t1 = time.time()

def wasserstein_loo(ew, top_k_fisher=40):
    """
    W1 distance in Fisher subspace between test windows and positive training windows.
    Project onto LDA direction, then compute 1D Wasserstein-1 distance.
    Score = 1 / (1 + W1)
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos.mean(0); mu_n = neg.mean(0)
            var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            pos_sub = pos[:, top_idx]; te_sub = te[:, top_idx]
            # Project onto mean difference direction
            mu_diff = pos_sub.mean(0) - (neg[:, top_idx].mean(0) if neg.shape[0] else 0)
            mu_diff /= (norm(mu_diff) + EPS)
            pos_proj = np.sort(pos_sub @ mu_diff)  # sorted 1D projections
            # For each test window: compute W1 as |E[pos_proj] - proj_te|
            te_proj = te_sub @ mu_diff  # (n_te,)
            n_pos = len(pos_proj)
            # W1 in 1D = mean |CDF_pos(x) - CDF_te(x)| over sorted points
            # Approximation: |mean(pos_proj) - te_proj| / std(pos_proj+eps)
            mu_pos = pos_proj.mean()
            std_pos = pos_proj.std() + EPS
            # Normalized distance from test window to positive distribution mean
            score = np.exp(-np.abs(te_proj - mu_pos) / (std_pos))
            ws[:, si] = np.clip(score, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for k in [30, 40, 50]:
    s = wasserstein_loo(ew_ica, top_k_fisher=k)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * attn_ref + w_val * s
        reg(f"wass_kf{k}_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: Extreme Value Theory (EVT) — Weibull tail modeling
# Model the distribution of top similarities with a Weibull/GEV distribution
# Score = probability under the fitted tail distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Extreme Value Theory (Weibull tail) scoring...", flush=True)
t1 = time.time()

def evt_loo(ew, top_k_fisher=40, top_sim_frac=0.3):
    """
    EVT scoring: for each test window, take top-p fraction of similarities
    to positive class windows. Fit Weibull/GEV to the tail of positive class
    similarities seen in training. Score = CDF(sim) under fitted tail.
    """
    from scipy.stats import weibull_min
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos.mean(0); mu_n = neg.mean(0)
            var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            # Compute all pairwise similarities within positive class
            pos_sims = tr_w @ tr_w.T
            n_top = max(3, int(len(pos) * len(pos) * top_sim_frac))
            pos_flat = pos_sims.flatten()
            tail_vals = np.sort(pos_flat)[-n_top:]  # top similarities
            # Fit Weibull to the tail
            if len(tail_vals) >= 3 and tail_vals.std() > 1e-6:
                try:
                    # Flip: Weibull for maxima uses (1 - tail_vals)
                    neg_tail = 1.0 - tail_vals
                    neg_tail = np.clip(neg_tail, 1e-6, None)
                    c, loc, scale = weibull_min.fit(neg_tail, floc=0, f0=1.0)
                    # Score test windows
                    te_sims = (te_w @ tr_w.T).max(1)  # max similarity to any positive
                    neg_te = np.clip(1.0 - te_sims, 1e-6, None)
                    score = weibull_min.cdf(neg_te, c, loc=loc, scale=scale)
                    # Higher neg_te → lower sim → score should be 0; invert
                    score = 1.0 - score
                except:
                    # fallback to max similarity
                    score = (te_w @ tr_w.T).max(1)
            else:
                score = (te_w @ tr_w.T).max(1)
            ws[:, si] = np.clip(score, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for k in [30, 40]:
    for frac in [0.2, 0.3, 0.5]:
        s = evt_loo(ew_ica, top_k_fisher=k, top_sim_frac=frac)
        for w_val in [0.005, 0.010]:
            blend = (1 - w_val) * attn_ref + w_val * s
            reg(f"evt_kf{k}_f{int(frac*10)}_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: Logit-filtered KDE (only use windows with logit above threshold)
# High-confidence positive windows as cleaner training signal
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] Logit-filtered KDE (high-confidence windows)...", flush=True)
t1 = time.time()

def logit_filtered_kde_loo(ew, top_k_fisher=40, bw=0.07, logit_thr_pct=75):
    """
    Fisher Hard KDE but only use positive windows whose logit score
    is above a percentile threshold (high-confidence positives).
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        tl_logit = logit_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
            pos_logit = tl_logit[pm, si]
            # Filter to high-confidence positives
            thr = np.percentile(pos_logit, logit_thr_pct)
            high_conf = pos_logit >= thr
            if high_conf.sum() < 1:
                high_conf = np.ones(len(pos_logit), bool)
            pos_hc = pos[high_conf]  # high-confidence positive windows
            mu_p = pos.mean(0); mu_n = neg.mean(0)
            var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_hc * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_w @ centroid, 0, None); proto_w /= proto_w.sum() + EPS
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for pct in [50, 60, 70, 75, 80, 90]:
    s = logit_filtered_kde_loo(ew_ica, top_k_fisher=40, bw=0.07, logit_thr_pct=pct)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * attn_ref + w_val * s
        reg(f"logfilt_p{pct}_kf40bw7_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: CovFisher — diagonal Mahalanobis in Fisher-selected subspace
# Instead of uniform 1/sqrt(k) mask, weight by inverse standard deviation
# within positive class (different from full Mahalanobis which uses full cov)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] CovFisher (diagonal Mahalanobis in Fisher dims)...", flush=True)
t1 = time.time()

def covfisher_loo(ew, top_k_fisher=40, bw=0.07, cov_reg=1e-3):
    """
    Like Fisher Hard KDE, but instead of uniform 1/sqrt(k) in the top-k dims,
    weight each selected dimension by 1/std_pos (inverse positive class std).
    This is equivalent to diagonal Mahalanobis in the Fisher-selected subspace.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos.mean(0); mu_n = neg.mean(0)
            var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            pos_sub = pos[:, top_idx]
            # Diagonal covariance weighting: 1 / std_pos per dimension
            std_pos = np.sqrt(pos_sub.var(0) + cov_reg)  # (k,)
            w_sub = 1.0 / std_pos  # inverse std
            w_sub /= (norm(w_sub) + EPS)  # normalize
            # Full w_dim in original ICA space
            w_dim = np.zeros(ew.shape[1], np.float32)
            w_dim[top_idx] = w_sub
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_w @ centroid, 0, None); proto_w /= proto_w.sum() + EPS
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for k in [30, 40, 50]:
    for bw in [0.06, 0.07, 0.08]:
        s = covfisher_loo(ew_ica, top_k_fisher=k, bw=bw)
        for w_val in [0.005, 0.010]:
            blend = (1 - w_val) * attn_ref + w_val * s
            reg(f"covfisher_k{k}_bw{int(bw*100)}_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M6: Window-count prior (number of positive windows as prior strength)
# Files with more matching positive windows should get higher score
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M6] Window-count prior...", flush=True)
t1 = time.time()

def wincount_prior_loo(ew, top_k_fisher=40, bw=0.07, count_T=1.0):
    """
    Weight each positive training file's contribution by its window count
    (files with many windows of a species contribute more).
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        tr_fids = win_file_id[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos.mean(0); mu_n = neg.mean(0)
            var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            # Count windows per file in positive set
            pos_fids = tr_fids[pm]
            unique_pos_fids, counts = np.unique(pos_fids, return_counts=True)
            # Assign weight to each positive window proportional to its file's count
            win_weight = np.zeros(len(pos), np.float32)
            for ufid, cnt in zip(unique_pos_fids, counts):
                mask = pos_fids == ufid
                # Weight = softmax of counts / T
                win_weight[mask] = float(cnt)
            win_weight = np.exp(win_weight / count_T)
            win_weight /= (win_weight.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * win_weight[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for T in [0.5, 1.0, 2.0, 5.0]:
    s = wincount_prior_loo(ew_ica, top_k_fisher=40, bw=0.07, count_T=T)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * attn_ref + w_val * s
        reg(f"wincount_T{int(T*10)}_kf40bw7_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary + Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch103] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

if new_best_method:
    top5 = sorted(results.values(), key=lambda x: -x["loo_auc"])[:5]
    for r2 in top5:
        delta = r2["loo_auc"] - best_loo
        print(f"    {r2['method']}: {r2['loo_auc']:.6f} ({delta:+.6f})", flush=True)
else:
    print("  No improvement found.", flush=True)

# Save results to JSON
res2 = json.load(open(RESULTS_PATH))
if isinstance(res2.get("experiments"), list):
    res2["experiments"].extend(list(results.values()))
else:
    res2["experiments"].update(results)
json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"  Saved {len(results)} results to JSON", flush=True)

# Save new best pkl if improved
if new_best_method and new_best_loo > best_loo:
    print(f"\n  SAVED: {new_best_method} LOO={new_best_loo:.6f}", flush=True)
    ep_new = copy.deepcopy(ep)
    ep_new["method"] = new_best_method
    ep_new["loo_auc"] = new_best_loo
    ep_new["batch"] = "batch103"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep_new, f)
    res3 = json.load(open(RESULTS_PATH))
    res3["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch103"}
    json.dump(res3, open(RESULTS_PATH, "w"), indent=2)
    print(f"  Updated JSON best → {new_best_method} {new_best_loo:.6f}", flush=True)
else:
    print(f"  No improvement → best remains {best_loo:.6f} ({best['method']})", flush=True)
