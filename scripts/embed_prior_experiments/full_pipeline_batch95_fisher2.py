"""
Batch 95 — Fisher Variations: LDA / QDA / Adaptive-BW / Hard Selection
========================================================================
Current best: fisher_kde_bw60_w5 LOO=0.992051

Batch94 showed all Fisher extensions plateau at 0.992051.
Now trying fundamentally different uses of Fisher discriminant:

1. lda_proj    — Project onto Fisher discriminant axis (1D per species), KDE in 1D space
2. qda_score   — Diagonal QDA log-likelihood ratio (no KDE, direct score)
3. fisher_hard — Hard top-K ICA dimension selection (K=20/30/40) before proto-KDE
4. fisher_power — Power transform on Fisher weights (alpha=0.25/0.75/1.0/2.0)
5. fisher_bg   — Fisher KDE with grand-mean background model (instead of per-species neg)
6. fisher_confusable — Use top-K confusable species as negatives (not all negatives)
7. fisher_rand_ensemble — Ensemble of 5 random Fisher subspace KDEs
8. fisher_adaptive_bw  — Per-species bandwidth from intra-class spread
9. fisher_nonneg — Use only positive Fisher dimensions (mu_p > mu_n), zero others
10. fisher_twolevel — Two-level: Fisher select top-K prototypes, then Fisher KDE on them
"""
import numpy as np
import json
import pickle
import copy
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
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

print(f"[batch95] ICA{ew_ica.shape} PCA{ew_pca.shape}", flush=True)

res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch95] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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
    """Reference Fisher KDE (best=0.992051)."""
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

# ── Pre-compute base ──────────────────────────────────────────────────────────
print("Pre-computing base...", flush=True)
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
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

w_uh = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur  = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
final_ref = 0.96 * base_cur + 0.04 * kde08

# Reference Fisher (best confirmed)
print("Computing fisher_ica_06 reference...", flush=True)
t1 = time.time()
f06 = fisher_kde_loo(ew_ica, bw=0.06)
fin_ref = (1 - 0.05) * final_ref + 0.05 * f06
auc_ref = macro_auc(fin_ref)
print(f"  fisher_ica bw=0.06 w=0.05: {auc_ref:.6f} (expected 0.992051)", flush=True)
print(f"  ({time.time()-t1:.0f}s)", flush=True)

results = {}
new_best_loo = best_loo
new_best_method = None
new_best_final = None

def try_blend(name, scores, w_list):
    """Try blending scores with final_ref at various weights."""
    for w in w_list:
        tag = f"{name}_w{int(w*100):02d}"
        if tag in res.get("experiments", {}):
            continue
        s = (1 - w) * final_ref + w * scores
        a = macro_auc(s)
        delta = a - best_loo
        mark = ""
        if a > best_loo - 0.0003:
            mark = " (near-best)"
        print(f"  {tag}: {a:.6f} {delta:+.6f}{mark}", flush=True)
        results[tag] = {"loo_auc": a, "method": tag, "batch": "batch95"}

def try_fisher_blend(name, scores, w_list):
    """Try blending scores on top of fisher final."""
    for w in w_list:
        tag = f"{name}_w{int(w*100):02d}"
        if tag in res.get("experiments", {}):
            continue
        s = (1 - w) * fin_ref + w * scores
        a = macro_auc(s)
        delta = a - best_loo
        mark = ""
        if a > best_loo - 0.0003:
            mark = " (near-best)"
        print(f"  {tag}: {a:.6f} {delta:+.6f}{mark}", flush=True)
        results[tag] = {"loo_auc": a, "method": tag, "batch": "batch95"}

# ─────────────────────────────────────────────────────────────────────────────
# EXP1: LDA projection (1D per species) → density score
# Project ICA onto Fisher discriminant axis, compute 1D kernel density ratio
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP1] LDA 1D projection KDE...", flush=True)
t1 = time.time()

def lda_proj_loo(ew, bw=0.10):
    """Project onto LDA axis per species, compute KDE score in 1D space."""
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
            # LDA axis = direction that maximizes class separation
            diff = mu_p - mu_n  # shape (D,)
            # Compute within-class scatter (diagonal approx)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            sw_diag = var_p + var_n  # pooled within-class variance (diagonal)
            # Optimal LDA direction = Sw^-1 * (mu_p - mu_n)
            lda_axis = diff / sw_diag  # D-dim vector
            lda_axis /= norm(lda_axis) + EPS
            # Project all training pos and test onto LDA axis
            te_proj = te @ lda_axis  # scalar per window
            tr_proj = pos_wins @ lda_axis  # scalar per pos window
            mu_p_1d = tr_proj.mean()
            # Gaussian KDE around positive projections
            kern_1d = np.exp(-0.5 * ((te_proj[:, None] - tr_proj[None, :]) / bw)**2)
            score = kern_1d.mean(1)  # mean density at test point from positives
            mx = score.max()
            if mx > EPS:
                score /= mx
            ws[:, si] = score
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for bw_lda in [0.05, 0.08, 0.10, 0.15, 0.20]:
    lda_s = lda_proj_loo(ew_ica, bw=bw_lda)
    bw_tag = f"lda1d_bw{int(bw_lda*100)}"
    try_blend(bw_tag, lda_s, [0.03, 0.05, 0.07])
print(f"  EXP1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP2: Diagonal QDA log-likelihood ratio score (no KDE)
# log p(x|pos,Gaussian) - log p(x|neg,Gaussian) per species
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP2] Diagonal QDA log-likelihood ratio...", flush=True)
t1 = time.time()

def qda_loo(ew, min_var=1e-3):
    """Diagonal QDA: log p(x|pos) - log p(x|neg) per species."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.0; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos_wins.mean(0); var_p = np.maximum(pos_wins.var(0), min_var)
            mu_n = neg_wins.mean(0); var_n = np.maximum(neg_wins.var(0), min_var)
            # log p(x|pos) - log p(x|neg) in diagonal Gaussian
            # = -0.5 * sum[(x-mu_p)^2 / var_p + log(var_p)] - (-0.5 * sum[(x-mu_n)^2 / var_n + log(var_n)])
            # = 0.5 * [sum((x-mu_n)^2 / var_n - (x-mu_p)^2 / var_p) + sum(log(var_n) - log(var_p))]
            # Only use Fisher-relevant dimensions to reduce noise
            fisher = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_k = np.argsort(-fisher)[:30]  # top-30 discriminative dims
            # Score per test window
            diff_n = te[:, top_k] - mu_n[None, top_k]  # (N, k)
            diff_p = te[:, top_k] - mu_p[None, top_k]
            log_ratio = 0.5 * (
                ((diff_n**2) / var_n[None, top_k]).sum(1)
                - ((diff_p**2) / var_p[None, top_k]).sum(1)
                + np.log(var_n[top_k]).sum() - np.log(var_p[top_k]).sum()
            )
            score = 1.0 / (1.0 + np.exp(-np.clip(log_ratio * 0.1, -10, 10)))
            ws[:, si] = score
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

qda_s = qda_loo(ew_ica)
try_blend("qda_top30", qda_s, [0.03, 0.05, 0.07, 0.10])
for topk in [10, 20, 50]:
    def qda_topk_loo(ew, k=topk, min_var=1e-3):
        out = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
            tl = labels_win[win_file_id != fi]
            ws = np.zeros((len(te), n_species), np.float32)
            for si in range(n_species):
                pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
                if not pm.any(): ws[:, si] = 0.0; continue
                pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
                mu_p = pos_wins.mean(0); var_p = np.maximum(pos_wins.var(0), min_var)
                mu_n = neg_wins.mean(0); var_n = np.maximum(neg_wins.var(0), min_var)
                fisher = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
                top_dims = np.argsort(-fisher)[:k]
                diff_n = te[:, top_dims] - mu_n[None, top_dims]
                diff_p = te[:, top_dims] - mu_p[None, top_dims]
                log_ratio = 0.5 * (
                    ((diff_n**2) / var_n[None, top_dims]).sum(1)
                    - ((diff_p**2) / var_p[None, top_dims]).sum(1)
                    + np.log(var_n[top_dims]).sum() - np.log(var_p[top_dims]).sum()
                )
                score = 1.0 / (1.0 + np.exp(-np.clip(log_ratio * 0.1, -10, 10)))
                ws[:, si] = score
            for si in range(n_species):
                mx = ws[:, si].max()
                if mx > EPS: ws[:, si] /= mx
            out[fi] = ws.max(0)
        return out
    s_qda_k = qda_topk_loo(ew_ica)
    try_blend(f"qda_top{topk}", s_qda_k, [0.03, 0.05, 0.07])
print(f"  EXP2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP3: Fisher hard dimension selection (top-K dims set to 1, rest to 0)
# Fundamentally different from soft Fisher weights
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP3] Fisher hard dimension selection KDE...", flush=True)
t1 = time.time()

def fisher_hard_kde_loo(ew, bw=0.06, top_k=30):
    """Set top-K Fisher dimensions to uniform 1, rest to 0, then proto-KDE."""
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
            # Hard selection: top-K dims = 1, rest = 0
            top_idx = np.argsort(-fisher)[:top_k]
            w_dim = np.zeros(len(fisher), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k))
            # Apply hard mask
            te_w = te * w_dim[None, :]
            te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_dim[None, :]
            tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
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

for k in [15, 20, 25, 30, 40, 50]:
    s_h = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=k)
    try_blend(f"fisher_hard_k{k}_bw6", s_h, [0.03, 0.05, 0.07])
    try_fisher_blend(f"fisher_hard_k{k}_fbw6", s_h, [0.03, 0.05])
print(f"  EXP3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP4: Fisher power transform (alpha != 0.5 for sqrt)
# Current: sqrt(F_d). Try alpha=0.25, 0.75, 1.0, 1.5, 2.0
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP4] Fisher weight power transform...", flush=True)
t1 = time.time()

def fisher_power_kde_loo(ew, bw=0.06, alpha=0.5):
    """Fisher KDE with customizable power transform on weights."""
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
            # alpha-power transform of Fisher score
            fisher_raw = np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None)
            fisher = fisher_raw ** alpha  # power transform
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

for alpha in [0.25, 0.33, 0.75, 1.0, 1.5, 2.0]:
    s_p = fisher_power_kde_loo(ew_ica, bw=0.06, alpha=alpha)
    tag = f"fisher_pow{int(alpha*100):03d}"
    try_blend(tag, s_p, [0.04, 0.05, 0.06])
print(f"  EXP4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP5: Non-negative Fisher (only dims where mu_p > mu_n contribute)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP5] Non-negative Fisher KDE...", flush=True)
t1 = time.time()

def fisher_nonneg_kde_loo(ew, bw=0.06):
    """Use only dimensions where positive mean > negative mean."""
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
            fisher_raw = np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None)
            # Zero out dims where mu_p < mu_n (these are anti-informative)
            fisher = np.where(mu_p > mu_n, np.sqrt(fisher_raw), 0.0)
            if fisher.sum() < EPS:
                fisher = np.sqrt(fisher_raw)  # fallback to regular fisher
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

s_nn = fisher_nonneg_kde_loo(ew_ica, bw=0.06)
try_blend("fisher_nonneg_bw6", s_nn, [0.03, 0.05, 0.07, 0.10])
s_nn8 = fisher_nonneg_kde_loo(ew_ica, bw=0.08)
try_blend("fisher_nonneg_bw8", s_nn8, [0.03, 0.05, 0.07])
print(f"  EXP5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP6: Fisher with confusable-species negatives
# For each species, identify the K most "confusable" species (closest centroid)
# Use only those as negatives in Fisher computation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP6] Fisher with confusable-species negatives...", flush=True)
t1 = time.time()

def fisher_confusable_kde_loo(ew, bw=0.06, n_confusable=5):
    """Fisher KDE using only the most confusable species as negatives."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        # Compute per-species centroids from train
        centroids = np.zeros((n_species, ew.shape[1]), np.float32)
        has_pos = np.zeros(n_species, bool)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if pm.any():
                c = tr[pm].mean(0); c /= norm(c) + EPS
                centroids[si] = c; has_pos[si] = True
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]
            # Find most confusable species: highest centroid similarity
            if has_pos[si]:
                sims_to_others = centroids[has_pos] @ centroids[si]
                confusable_idx = np.argsort(-sims_to_others)
                # Exclude self
                confusable_idx = [idx for idx in confusable_idx
                                  if np.where(has_pos)[0][idx] != si][:n_confusable]
                actual_species = np.where(has_pos)[0]
                confusable_species = actual_species[confusable_idx]
                # Build negative windows from confusable species
                neg_mask = np.zeros(len(tl), bool)
                for cs in confusable_species:
                    neg_mask |= (tl[:, cs] > 0.5)
                if neg_mask.any():
                    neg_wins = tr[neg_mask]
                else:
                    neg_wins = tr[tl[:, si] < 0.1]
            else:
                neg_wins = tr[tl[:, si] < 0.1]
            if not len(neg_wins): neg_wins = tr[~pm]
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

for nc in [3, 5, 10]:
    s_c = fisher_confusable_kde_loo(ew_ica, bw=0.06, n_confusable=nc)
    try_blend(f"fisher_conf_nc{nc}_bw6", s_c, [0.03, 0.05, 0.07])
print(f"  EXP6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP7: Fisher adaptive bandwidth (per-species bw = f(intra-class spread))
# Narrower bw for well-clustered species, wider for spread ones
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP7] Fisher adaptive bandwidth...", flush=True)
t1 = time.time()

def fisher_adaptive_bw_loo(ew, bw_scale=1.0, bw_min=0.04, bw_max=0.12):
    """Adaptive per-species bandwidth based on intra-class Fisher-weighted spread."""
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
            # Compute adaptive bw from intra-class spread in Fisher space
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            sims_intra = tr_w @ centroid  # (n_pos,)
            # bw = 1 - mean_cosine_sim (spread measure)
            spread = 1.0 - np.clip(sims_intra.mean(), 0, 1)
            bw = np.clip(spread * bw_scale, bw_min, bw_max)
            proto_w = np.clip(tr_w @ centroid, 0, None); proto_w /= (proto_w.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for bws in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    s_abw = fisher_adaptive_bw_loo(ew_ica, bw_scale=bws)
    try_blend(f"fisher_abw_s{int(bws*10):02d}", s_abw, [0.04, 0.05, 0.06])
print(f"  EXP7 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP8: Fisher + background model (grand mean as negative reference)
# Instead of per-species negatives, use global background
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP8] Fisher with global background model...", flush=True)
t1 = time.time()

def fisher_bg_kde_loo(ew, bw=0.06):
    """Fisher KDE with global background model as negative."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        # Compute global background
        bg = tr.mean(0); bg /= norm(bg) + EPS
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]
            # Use global mean as "negative" reference
            mu_p = pos_wins.mean(0); mu_n = bg
            var_p = pos_wins.var(0) + EPS
            var_n = tr.var(0) + EPS  # global variance
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

for bw_bg in [0.05, 0.06, 0.08, 0.10]:
    s_bg = fisher_bg_kde_loo(ew_ica, bw=bw_bg)
    try_blend(f"fisher_bg_bw{int(bw_bg*100):02d}", s_bg, [0.03, 0.05, 0.07])
print(f"  EXP8 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP9: Random Fisher subspace ensemble
# 5 random subsets of ICA dimensions, each with their own Fisher weighting
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP9] Random Fisher subspace ensemble...", flush=True)
t1 = time.time()

def fisher_randsubspace_kde_loo(ew, bw=0.06, n_dims=50, n_ensemble=5, seed=42):
    """Average Fisher KDE over random dimension subsets."""
    rng = np.random.default_rng(seed)
    all_out = []
    dim_total = ew.shape[1]
    for e_idx in range(n_ensemble):
        dims = rng.choice(dim_total, n_dims, replace=False)
        ew_sub = ew[:, dims]
        ew_sub_n = ew_sub / (norm(ew_sub, axis=1, keepdims=True) + EPS)
        out_e = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            te = ew_sub_n[win_file_id == fi]; tr = ew_sub_n[win_file_id != fi]
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
            out_e[fi] = ws.max(0)
        all_out.append(out_e)
    return np.mean(all_out, axis=0)

for nd in [30, 50, 70]:
    for ne in [5, 10]:
        s_rs = fisher_randsubspace_kde_loo(ew_ica, bw=0.06, n_dims=nd, n_ensemble=ne)
        try_blend(f"fisher_rs_d{nd}_e{ne}", s_rs, [0.04, 0.05, 0.06])
print(f"  EXP9 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch95] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

if results:
    all_sorted = sorted(results.items(), key=lambda x: -x[1]["loo_auc"])
    current_best_tag, current_best_v = all_sorted[0]
    current_best_auc = current_best_v["loo_auc"]

    if current_best_auc > best_loo:
        print(f"  NEW BEST: {current_best_tag} LOO={current_best_auc:.6f} (+{current_best_auc-best_loo:.6f})", flush=True)
        new_best_loo = current_best_auc
        new_best_method = current_best_tag
    else:
        print(f"  New best: {current_best_auc:.6f} ({current_best_tag}) — no improvement", flush=True)

    print(f"\n  Top-10 results:", flush=True)
    for tag, v in all_sorted[:10]:
        d = v["loo_auc"] - best_loo
        print(f"    {tag}: {v['loo_auc']:.6f} ({d:+.6f})", flush=True)

# ── Save to JSON ─────────────────────────────────────────────────────────────
res2 = json.load(open(RESULTS_PATH))
if "experiments" not in res2:
    res2["experiments"] = {}
res2["experiments"].update(results)

if new_best_loo > best_loo:
    res2["best"]["loo_auc"] = new_best_loo
    res2["best"]["method"] = new_best_method
    res2["best"]["batch"] = "batch95"
    # Save updated pkl
    ep2 = copy.deepcopy(ep)
    ep2["loo_auc"] = new_best_loo
    ep2["method"] = new_best_method
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep2, f)
    print(f"\n  *** SAVED new best pkl: {new_best_method} LOO={new_best_loo:.6f} ***", flush=True)

json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"\nSaved {len(results)} experiments to JSON.", flush=True)
