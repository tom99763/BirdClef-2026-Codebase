"""
Batch 94 — Fisher Score KDE Extensions
========================================
Current best: fisher_kde_bw60_w5 LOO=0.992051

Building on the Fisher score KDE breakthrough (batch93).
Exploring extensions to push further:

1. fisher_std_kde    — Fisher KDE in STD-PCA80 space
2. fisher_pca_kde    — Fisher KDE in PCA80 space
3. fisher_nmf_kde    — Fisher KDE in NMF100 space
4. fisher_multi_kde  — Average Fisher KDE across ICA+STD+PCA
5. fisher_wl         — WL scoring with Fisher-weighted ICA dimensions
6. fisher_neg_push   — Fisher KDE + negative repulsion
7. fisher_logit_kde  — Fisher KDE in logit-PCA50 space + ICA blend
8. fisher_bw_fine    — Even finer bw sweep around 0.055-0.065
9. fisher_boost      — Second-pass Fisher: apply Fisher on Fisher residuals
10. fisher_hybrid    — Fisher KDE (bw=0.06) blended with proto-KDE (bw=0.08)
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

# ── Load data ──────────────────────────────────────────────────────────────
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

print(f"[batch94] ICA{ew_ica.shape} PCA{ew_pca.shape} STD{ew_std.shape} NMF{ew_nmf.shape}", flush=True)

res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch94] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

# ── Helpers ────────────────────────────────────────────────────────────────
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
    """Fisher score KDE (confirmed new best, bw=0.06)."""
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
final_ref = 0.96 * base_cur + 0.04 * kde08  # prev best base

# Reference Fisher (best confirmed)
print("Computing fisher_ica_06 reference...", flush=True)
t1 = time.time()
f_ica_06 = fisher_kde_loo(ew_ica, bw=0.06)
auc_ref   = macro_auc((1 - 0.05) * final_ref + 0.05 * f_ica_06)
print(f"  fisher_ica bw=0.06 w=0.05: {auc_ref:.6f} (expected {best_loo:.6f})", flush=True)
print(f"  ({time.time()-t1:.0f}s)", flush=True)

results = {}
new_best_score = best_loo
new_best_method = None
new_best_scores_arr = None

def check_and_record(key, score, scores_arr=None):
    global new_best_score, new_best_method, new_best_scores_arr
    results[key] = score
    marker = ""
    if score > best_loo:
        marker = " *** IMPROVED ***"
        if score > new_best_score:
            new_best_score = score
            new_best_method = key
            new_best_scores_arr = scores_arr
    elif score >= best_loo - 0.0003:
        marker = " (near-best)"
    if score >= best_loo - 0.0005:
        print(f"  {key}: {score:.6f} {score-best_loo:+.6f}{marker}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# EXP 1: Fisher KDE in other spaces
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP1] Fisher KDE in STD/PCA/NMF spaces...", flush=True)

for space_name, ew_sp in [("std", ew_std), ("pca", ew_pca), ("nmf", ew_nmf)]:
    for bw in [0.04, 0.05, 0.06, 0.07, 0.08]:
        sf = fisher_kde_loo(ew_sp, bw=bw)
        for w in [0.03, 0.04, 0.05, 0.06, 0.08]:
            cand = (1-w) * final_ref + w * sf
            a = macro_auc(cand)
            check_and_record(f"fisher_{space_name}_bw{int(bw*100)}_w{int(w*100)}", a, cand)
        # Replace kde08
        cand2 = 0.96 * base_cur + 0.04 * sf
        a2 = macro_auc(cand2)
        check_and_record(f"fisher_{space_name}_bw{int(bw*100)}_replace", a2, cand2)

# ═══════════════════════════════════════════════════════════════════════════════
# EXP 2: Multi-space Fisher KDE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP2] Multi-space Fisher KDE...", flush=True)

f_std_06  = fisher_kde_loo(ew_std, bw=0.06)
f_pca_06  = fisher_kde_loo(ew_pca, bw=0.06)

# 2-space blends: ICA + STD
for a1 in [0.5, 0.6, 0.7, 0.8]:
    f_blend = a1 * f_ica_06 + (1 - a1) * f_std_06
    for w in [0.04, 0.05, 0.06]:
        cand = (1-w) * final_ref + w * f_blend
        a = macro_auc(cand)
        check_and_record(f"fisher_ica_std_a{int(a1*10)}_w{int(w*100)}", a, cand)

# ICA + PCA
for a1 in [0.5, 0.6, 0.7, 0.8]:
    f_blend = a1 * f_ica_06 + (1 - a1) * f_pca_06
    for w in [0.04, 0.05, 0.06]:
        cand = (1-w) * final_ref + w * f_blend
        a = macro_auc(cand)
        check_and_record(f"fisher_ica_pca_a{int(a1*10)}_w{int(w*100)}", a, cand)

# 3-space blend: ICA + STD + PCA
for a_ica, a_std, a_pca in [(0.6, 0.2, 0.2), (0.7, 0.15, 0.15), (0.5, 0.3, 0.2)]:
    f3 = a_ica * f_ica_06 + a_std * f_std_06 + a_pca * f_pca_06
    for w in [0.04, 0.05, 0.06]:
        cand = (1-w) * final_ref + w * f3
        a = macro_auc(cand)
        check_and_record(f"fisher_3space_{int(a_ica*10)}_{int(a_std*10)}_{int(a_pca*10)}_w{int(w*100)}", a, cand)

# ═══════════════════════════════════════════════════════════════════════════════
# EXP 3: Fisher WL (Fisher-weighted dimensions in WL scoring)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP3] Fisher WL (Fisher-weighted WL)...", flush=True)

def fisher_wl_loo(k_neg=50, wmp=0.85, wma=0.88):
    """WL scoring in Fisher-weighted ICA space (per-species dimension weighting)."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]

            # Fisher weights
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            w_dim = fisher / (norm(fisher) + EPS)

            # Apply weights
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            pw_w = pos_wins * w_dim[None, :]; pw_w /= norm(pw_w, axis=1, keepdims=True) + EPS

            # WL scoring
            ps = te_w @ pw_w.T
            pp = pw_w.mean(0); pp /= norm(pp) + EPS
            sp = wmp * ps.max(1) + (1 - wmp) * (te_w @ pp)

            if nm.any() and k_neg > 0:
                nw_w = neg_wins * w_dim[None, :]; nw_w /= norm(nw_w, axis=1, keepdims=True) + EPS
                ns = te_w @ nw_w.T; k2 = min(k_neg, nw_w.shape[0])
                if k2 > 0:
                    tn = nw_w[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
                    tn /= norm(tn, axis=1, keepdims=True) + EPS
                    ws[:, si] = (sp - (te_w * tn).sum(1) + 1) / 2
                else:
                    ws[:, si] = (sp + 1) / 2
            else:
                ws[:, si] = (sp + 1) / 2
        out[fi] = wma * ws.max(0) + (1 - wma) * ws.mean(0)
    return out

print("  Computing fisher_wl...", flush=True)
s_fwl = fisher_wl_loo(k_neg=50, wmp=0.85, wma=0.88)
for w in [0.03, 0.05, 0.07, 0.10]:
    cand = (1-w) * final_ref + w * s_fwl
    a = macro_auc(cand)
    check_and_record(f"fisher_wl_k50_w{int(w*100)}", a, cand)

# Replace ica wl component
w_uh2 = w_uh  # same w_uh
uh_b2 = cfg["w_ica100"] * s_fwl + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
uh_nmf2 = cfg["nmf"]["uh_scale"] * uh_b2 + cfg["nmf"]["w_nmf"] * s_nmf
base_fwl = w_uh2 * uh_nmf2 + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
for w_kde in [0.03, 0.04, 0.05]:
    cand = (1-w_kde) * base_fwl + w_kde * f_ica_06
    a = macro_auc(cand)
    check_and_record(f"fisher_wl_base_kde{int(w_kde*100)}", a, cand)

# ═══════════════════════════════════════════════════════════════════════════════
# EXP 4: Fisher KDE with negative repulsion
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP4] Fisher KDE with negative repulsion...", flush=True)

def fisher_neg_kde_loo(bw_pos=0.06, bw_neg=0.10, w_neg=0.3):
    """
    Fisher KDE: positive score - w_neg * negative Fisher KDE.
    Negative windows that are similar to test get subtracted.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]

            # Shared Fisher weights
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            w_dim = fisher / (norm(fisher) + EPS)

            # Positive KDE
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            pw_w = pos_wins * w_dim[None, :]; pw_w /= norm(pw_w, axis=1, keepdims=True) + EPS
            centroid_p = pw_w.mean(0); centroid_p /= norm(centroid_p) + EPS
            proto_w_p = np.clip(pw_w @ centroid_p, 0, None); proto_w_p /= proto_w_p.sum() + EPS
            sims_p = te_w @ pw_w.T
            kern_p = np.exp((sims_p - 1.0) / (bw_pos**2 + EPS))
            score_p = np.clip((kern_p * proto_w_p[None, :]).sum(1), 0, None)

            # Negative KDE (repulsion)
            nw_w = neg_wins * w_dim[None, :]; nw_w /= norm(nw_w, axis=1, keepdims=True) + EPS
            sims_n = te_w @ nw_w.T
            kern_n = np.exp((sims_n - 1.0) / (bw_neg**2 + EPS))
            score_n = np.clip(kern_n.mean(1), 0, None)

            ws[:, si] = np.clip(score_p - w_neg * score_n, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for bw_neg, w_neg in [(0.10, 0.2), (0.10, 0.3), (0.08, 0.2), (0.12, 0.3)]:
    s_fneg = fisher_neg_kde_loo(bw_pos=0.06, bw_neg=bw_neg, w_neg=w_neg)
    for w in [0.04, 0.05, 0.06]:
        cand = (1-w) * final_ref + w * s_fneg
        a = macro_auc(cand)
        check_and_record(f"fisher_neg_bn{int(bw_neg*100)}_wn{int(w_neg*10)}_w{int(w*100)}", a, cand)

# ═══════════════════════════════════════════════════════════════════════════════
# EXP 5: Fisher KDE with logit weighting of Fisher scores
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP5] Fisher KDE with logit-weighted Fisher scores...", flush=True)

def fisher_logit_weighted_loo(bw=0.06, T=8.0):
    """
    Weight the Fisher score per dimension by logit confidence:
    w_d = Fisher_d * mean_logit_conf_d_of_positives
    """
    logit_conf = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T, -88, 88)))
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]; lc = logit_conf[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]

            # Base Fisher score
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))

            # Logit confidence weighting: high logit conf → trust Fisher more
            pos_logit_conf = lc[pm, si].mean()  # mean Perch confidence for positives
            logit_boost = 0.5 + 0.5 * pos_logit_conf  # [0.5, 1.0]
            fisher_weighted = fisher * logit_boost

            w_dim = fisher_weighted / (norm(fisher_weighted) + EPS)

            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
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

for T in [6.0, 8.0, 10.0]:
    s_flw = fisher_logit_weighted_loo(bw=0.06, T=T)
    for w in [0.04, 0.05, 0.06]:
        cand = (1-w) * final_ref + w * s_flw
        a = macro_auc(cand)
        check_and_record(f"fisher_logit_T{int(T)}_w{int(w*100)}", a, cand)

# ═══════════════════════════════════════════════════════════════════════════════
# EXP 6: Fisher + KDE hybrid (blend Fisher-KDE with proto-KDE)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP6] Fisher + proto-KDE hybrid...", flush=True)

f06 = f_ica_06  # already computed
k08 = kde08      # already computed

for a_f in [0.5, 0.6, 0.7, 0.8]:
    hybrid = a_f * f06 + (1 - a_f) * k08
    for w in [0.03, 0.04, 0.05, 0.06]:
        cand = (1-w) * base_cur + w * hybrid  # use base_cur (not final_ref)
        a = macro_auc(cand)
        check_and_record(f"fisher_kde_hybrid_af{int(a_f*10)}_w{int(w*100)}", a, cand)

# ═══════════════════════════════════════════════════════════════════════════════
# EXP 7: Fisher bw fine sweep (even finer than batch93)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP7] Fisher bw ultra-fine sweep...", flush=True)

for bw in [0.055, 0.057, 0.059, 0.060, 0.061, 0.063, 0.065]:
    sf_bw = fisher_kde_loo(ew_ica, bw=bw)
    for w in [0.04, 0.05, 0.06, 0.07, 0.08]:
        cand = (1-w) * final_ref + w * sf_bw
        a = macro_auc(cand)
        check_and_record(f"fisher_ica_bw{int(bw*1000)}_w{int(w*100)}", a, cand)

# ═══════════════════════════════════════════════════════════════════════════════
# EXP 8: Fisher + Softmax blend (extend base_cur with stronger softmax)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP8] Fisher + enhanced base components...", flush=True)

# Try adding more softmax and fisher together
sm5 = make_sp(5.0)
sm7 = make_sp(7.0)
sm4 = make_sp(4.0)

for T_sm, sm_extra in [(5.0, sm5), (7.0, sm7), (4.0, sm4)]:
    for w_sm, w_f in [(0.03, 0.05), (0.05, 0.05), (0.07, 0.05)]:
        if w_sm + w_f >= 0.15:
            continue
        cand = (1-w_sm-w_f) * final_ref + w_sm * sm_extra + w_f * f06
        a = macro_auc(cand)
        check_and_record(f"fisher_sm{int(T_sm)}_wsm{int(w_sm*100)}_wf{int(w_f*100)}", a, cand)

# ═══════════════════════════════════════════════════════════════════════════════
# EXP 9: Fisher with top-K positive selection
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP9] Fisher KDE with top-K positive selection...", flush=True)

def fisher_topk_kde_loo(bw=0.06, k_pos=5):
    """Use only top-K most positive windows (by logit) as prototypes."""
    logit_conf = 1.0 / (1.0 + np.exp(np.clip(-logit_win / 8.0, -88, 88)))
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]; lc = logit_conf[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            all_pos = tr[pm]; all_pos_conf = lc[pm, si]

            # Select top-K most confident positives
            k2 = min(k_pos, len(all_pos))
            top_idx = np.argsort(-all_pos_conf)[:k2]
            pos_wins = all_pos[top_idx]
            neg_wins = tr[nm] if nm.any() else tr[~pm]

            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            w_dim = fisher / (norm(fisher) + EPS)

            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
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

for k_pos in [3, 5, 8, 12]:
    s_ftopk = fisher_topk_kde_loo(bw=0.06, k_pos=k_pos)
    for w in [0.04, 0.05, 0.06]:
        cand = (1-w) * final_ref + w * s_ftopk
        a = macro_auc(cand)
        check_and_record(f"fisher_topk{k_pos}_w{int(w*100)}", a, cand)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"[batch94] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)
print(f"  New best: {new_best_score:.6f} ({new_best_method})", flush=True)
print(f"  Delta: {new_best_score - best_loo:+.6f}", flush=True)

top10 = sorted(results.items(), key=lambda x: -x[1])[:10]
print(f"\n  Top-10 results:", flush=True)
for k, v in top10:
    print(f"    {k}: {v:.6f} ({v-best_loo:+.6f})", flush=True)

# Update JSON
res2 = json.load(open(RESULTS_PATH))
new_exps = [{"method": k, "loo_auc": float(v), "batch": "batch94"} for k, v in results.items()]
res2["experiments"].extend(new_exps)

if new_best_score > best_loo and new_best_method is not None:
    res2["best"] = {
        "method": new_best_method,
        "loo_auc": float(new_best_score),
        "full_auc": None,
        "batch": "batch94"
    }
    print(f"\n*** NEW BEST: {new_best_method} = {new_best_score:.6f} ***", flush=True)

    # Save pkl
    ep_new = {k: v for k, v in ep.items()}
    ep_new["method"]  = new_best_method
    ep_new["loo_auc"] = float(new_best_score)
    ep_new["config"]  = {**cfg, "batch94_best": {"method": new_best_method, "loo": new_best_score}}
    ep_new["file_prob_max"] = new_best_scores_arr.copy()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep_new, f)
    print(f"  Saved updated pkl.", flush=True)

with open(RESULTS_PATH, "w") as f:
    json.dump(res2, f, indent=2)
print(f"\nSaved {len(new_exps)} experiments to JSON.", flush=True)
