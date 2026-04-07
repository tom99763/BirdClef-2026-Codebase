"""
Batch 104 — Novel Signal Types (Bootstrap / Transductive / Conformal / MultiScale)
===============================================================================
Current best: attn_addon_kf40_bw7_T6_w01 LOO=0.992180
Reference: attn_ref = 0.99*triple_ref + 0.01*attn_kf40_bw7_T6

Genuinely novel methods not in 14580-experiment history:
1. Bootstrap ensemble KDE — average over bootstrap samples of positive windows
2. Transductive KDE — use test windows to adaptively re-weight training set
3. Conformal non-conformity score — rank test windows against calibration set
4. Multi-scale adaptive KDE — per-species optimal bandwidth selection
5. Hierarchical KDE — file-level then window-level two-stage scoring
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

print(f"[batch104] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch104] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

def macro_auc(s):
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            aucs.append(roc_auc_score(y, s[:, si]))
    return float(np.mean(aucs)) if aucs else 0.0

# ── Exact helpers from batch101 ───────────────────────────────────────────────
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

# ── Pre-compute full chain → attn_ref ────────────────────────────────────────
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
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch104"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: Bootstrap ensemble KDE
# Average KDE scores over multiple bootstrap samples of positive windows.
# Reduces variance, might reveal signal hidden by outlier positive windows.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] Bootstrap ensemble KDE...", flush=True)
t1 = time.time()

def bootstrap_kde_loo(ew, top_k_fisher=40, bw=0.07, n_boot=20, seed=42):
    """
    For each species, draw n_boot bootstrap samples of positive windows,
    compute Fisher Hard KDE for each, and average the scores.
    """
    rng = np.random.default_rng(seed)
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
            n_pos = len(pos)
            boot_scores = np.zeros((n_boot, len(te)), np.float32)
            for b in range(n_boot):
                idx = rng.integers(0, n_pos, size=n_pos)  # bootstrap
                pos_b = pos[idx]
                tr_w = pos_b * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
                centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
                proto_w = np.clip(tr_w @ centroid, 0, None); proto_w /= proto_w.sum() + EPS
                sims_w = te_w @ tr_w.T
                kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
                boot_scores[b] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
            ws[:, si] = boot_scores.mean(0)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for n_b in [10, 20, 50]:
    s = bootstrap_kde_loo(ew_ica, top_k_fisher=40, bw=0.07, n_boot=n_b)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * attn_ref + w_val * s
        reg(f"bootstrap_nb{n_b}_kf40bw7_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: Transductive KDE
# Use test windows to re-weight training windows:
# training windows more similar to test windows get higher weight.
# This is a form of importance weighting / distribution shift correction.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Transductive KDE (test-adaptive weighting)...", flush=True)
t1 = time.time()

def transductive_kde_loo(ew, top_k_fisher=40, bw=0.07, trans_bw=0.15):
    """
    Weight each positive training window by its mean similarity to test windows.
    This adapts the KDE to the test distribution (transductive learning).
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        # Transductive weight: how similar is each training window to test set?
        # Use mean cosine similarity to test windows
        sim_tr_te = tr @ te.T  # (n_tr, n_te)
        trans_w = np.exp((sim_tr_te.mean(1) - 1.0) / (trans_bw**2 + EPS))  # (n_tr,)
        trans_w /= (trans_w.sum() + EPS)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
            pos_tw = trans_w[pm]  # transductive weights for positive windows
            mu_p = pos.mean(0); mu_n = neg.mean(0)
            var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            # Combine proto_w with transductive weight
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_w @ centroid, 0, None)
            # Blend: proto_w * transductive_weight
            combined_w = proto_w * pos_tw
            combined_w /= (combined_w.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * combined_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for tbw in [0.10, 0.15, 0.20]:
    s = transductive_kde_loo(ew_ica, top_k_fisher=40, bw=0.07, trans_bw=tbw)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * attn_ref + w_val * s
        reg(f"transductive_tbw{int(tbw*100)}_kf40bw7_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: Conformal non-conformity score
# For each test window, measure how "non-conforming" it is with positive class.
# Non-conformity = 1 - (sim to nearest positive) / (sim to nearest negative)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Conformal non-conformity score...", flush=True)
t1 = time.time()

def conformal_loo(ew, top_k_fisher=40, k_nn=5):
    """
    Conformal prediction non-conformity measure:
    alpha = 1 - (nearest_pos_sim / nearest_neg_sim)
    Lower alpha = more conforming to positive class = higher score
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
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            pos_w = pos * w_dim[None, :]; pos_w /= norm(pos_w, axis=1, keepdims=True) + EPS
            neg_w = neg * w_dim[None, :]; neg_w /= norm(neg_w, axis=1, keepdims=True) + EPS
            # k-NN similarity to positive and negative
            k_p = min(k_nn, len(pos_w))
            k_n = min(k_nn, len(neg_w))
            sims_p = te_w @ pos_w.T  # (n_te, n_pos)
            sims_n = te_w @ neg_w.T  # (n_te, n_neg)
            knn_pos_sim = np.sort(sims_p, axis=1)[:, -k_p:].mean(1)  # top-k mean
            knn_neg_sim = np.sort(sims_n, axis=1)[:, -k_n:].mean(1)
            # Conformity score: how much more similar to pos than neg
            conformity = knn_pos_sim / (knn_neg_sim + EPS)
            ws[:, si] = np.clip(conformity - 1.0, 0, None)  # > 0 when pos > neg
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for k in [3, 5, 10]:
    s = conformal_loo(ew_ica, top_k_fisher=40, k_nn=k)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * attn_ref + w_val * s
        reg(f"conformal_k{k}_kf40_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: Multi-scale adaptive KDE
# For each species, select the bandwidth that maximizes within-class similarity
# relative to between-class similarity on training set.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] Multi-scale adaptive per-species KDE...", flush=True)
t1 = time.time()

def multiscale_kde_loo(ew, top_k_fisher=40, bw_candidates=[0.05, 0.06, 0.07, 0.08, 0.09]):
    """
    For each species in each LOO fold, select the bandwidth that maximizes
    the ratio of positive-positive similarity to positive-negative similarity.
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
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            pos_w = pos * w_dim[None, :]; pos_w /= norm(pos_w, axis=1, keepdims=True) + EPS
            neg_w = neg * w_dim[None, :]; neg_w /= norm(neg_w, axis=1, keepdims=True) + EPS
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            # Select best bandwidth by within/between class KDE ratio
            best_bw = bw_candidates[0]
            best_ratio = -1.0
            pp_sim = pos_w @ pos_w.T  # pos-pos similarity matrix
            pn_sim = pos_w @ neg_w.T  # pos-neg similarity matrix
            for bw in bw_candidates:
                pp_kde = np.exp((pp_sim - 1.0) / (bw**2 + EPS)).mean()
                pn_kde = np.exp((pn_sim - 1.0) / (bw**2 + EPS)).mean()
                ratio = pp_kde / (pn_kde + EPS)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_bw = bw
            # Use selected bandwidth
            centroid = pos_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(pos_w @ centroid, 0, None); proto_w /= proto_w.sum() + EPS
            sims_w = te_w @ pos_w.T
            kern = np.exp((sims_w - 1.0) / (best_bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

s = multiscale_kde_loo(ew_ica, top_k_fisher=40)
for w_val in [0.005, 0.010, 0.015]:
    blend = (1 - w_val) * attn_ref + w_val * s
    reg(f"multiscale_kf40_w{int(w_val*1000):04d}", macro_auc(blend))

s = multiscale_kde_loo(ew_ica, top_k_fisher=30)
for w_val in [0.005, 0.010]:
    blend = (1 - w_val) * attn_ref + w_val * s
    reg(f"multiscale_kf30_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: Hierarchical KDE — file-level then window-level two-stage scoring
# Stage 1: For each training file, compute a file-level embedding (mean of windows)
# Stage 2: Find nearest neighbor files, then weight their windows
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Hierarchical two-stage KDE...", flush=True)
t1 = time.time()

def hierarchical_kde_loo(ew, top_k_fisher=40, bw_file=0.12, bw_win=0.07):
    """
    Two-stage KDE:
    Stage 1: Score test file against training files using file-mean embeddings
    Stage 2: Re-weight training windows by their file's stage-1 score
    """
    # Pre-compute file-level mean embeddings for all files
    file_embs = np.zeros((n_files, ew.shape[1]), np.float32)
    for fi in range(n_files):
        fe = ew[win_file_id == fi]
        fm = fe.mean(0); fm /= (norm(fm) + EPS)
        file_embs[fi] = fm
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        tr_fids = win_file_id[win_file_id != fi]
        te_file_emb = file_embs[fi]
        train_file_ids_unique = [f for f in range(n_files) if f != fi]
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
            # Stage 1: file-level similarity score for positive training files
            pos_fids = tr_fids[pm]
            pos_file_embs = np.array([file_embs[f] * w_dim for f in pos_fids])
            pos_file_embs /= (norm(pos_file_embs, axis=1, keepdims=True) + EPS)
            te_file_w = te_file_emb * w_dim
            te_file_w /= (norm(te_file_w) + EPS)
            file_sim = pos_file_embs @ te_file_w  # (n_pos,) file similarity
            file_weight = np.exp((file_sim - 1.0) / (bw_file**2 + EPS))
            file_weight /= (file_weight.sum() + EPS)
            # Stage 2: window-level KDE with file weights
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_w @ centroid, 0, None)
            combined_w = proto_w * file_weight
            combined_w /= (combined_w.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw_win**2 + EPS))
            ws[:, si] = np.clip((kern * combined_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for bf, bw in [(0.10, 0.07), (0.12, 0.07), (0.15, 0.07), (0.12, 0.06)]:
    s = hierarchical_kde_loo(ew_ica, top_k_fisher=40, bw_file=bf, bw_win=bw)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * attn_ref + w_val * s
        reg(f"hierarchical_bf{int(bf*100)}_bw{int(bw*100)}_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary + Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch104] SUMMARY", flush=True)
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
    ep_new["batch"] = "batch104"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep_new, f)
    res3 = json.load(open(RESULTS_PATH))
    res3["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch104"}
    json.dump(res3, open(RESULTS_PATH, "w"), indent=2)
    print(f"  Updated JSON best → {new_best_method} {new_best_loo:.6f}", flush=True)
else:
    print(f"  No improvement → best remains {best_loo:.6f} ({best['method']})", flush=True)
