"""
Batch 111 — KDE Bandwidth Systematic Retune
===============================================================================
Current best: fine2_c1k40w00020_c5k50w00010 LOO=0.992312
Formula: 0.997*attn_ref + 0.002*conf_k1_kf40 + 0.001*conf_k5_kf50

Batches 108-110: 3 consecutive zero improvements. Chain parameters (attn weight,
fisher_kde weight, triple_ref weights) confirmed robust. Conformal variants exhausted.

This batch: systematically retune KDE bandwidth parameters that were fixed early
(proto_kde bw=0.08, fisher_kde bw=0.06, attn bw=0.07, fh bw=0.06/0.07).
These have never been swept with the FULL chain including conformal on top.

1. M1: Rebuild final_ref with different proto_kde bw (0.05-0.12)
2. M2: Rebuild fin_ref with different fisher_kde bw (0.04-0.10)
3. M3: Rebuild triple_ref with different fh_bw combinations
4. M4: Rebuild attn_ref with different attn KNN bw (0.05-0.12)
5. M5: Joint: best proto_kde + best attn_bw
6. M6: Second KDE at different bw added to chain
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

print(f"[batch111] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch111] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

def conformal_score_loo(ew, top_k_fisher=40, k_nn=1):
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
            k_p = min(k_nn, len(pos_w)); k_n = min(k_nn, len(neg_w))
            sims_p = te_w @ pos_w.T; sims_n = te_w @ neg_w.T
            knn_pos = np.sort(sims_p, axis=1)[:, -k_p:].mean(1)
            knn_neg = np.sort(sims_n, axis=1)[:, -k_n:].mean(1)
            score = knn_pos / (knn_neg + EPS) - 1.0
            ws[:, si] = np.clip(score, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ── Pre-compute FIXED components (WL, logit, subspace) ──────────────────────
print("Pre-computing fixed base...", flush=True)
t0 = time.time()
s_ica = wl_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
s_std = wl_loo(ew_std, cfg["std_pca80"]["k_neg"], cfg["std_pca80"]["w_max_pos"], cfg["std_pca80"]["w_max_agg"])
s_pca = wl_loo(ew_pca, cfg["pca80"]["k_neg"], cfg["pca80"]["w_max_pos"], cfg["pca80"]["w_max_agg"])
s_nmf = wl_loo(ew_nmf, cfg["nmf"]["k_neg"], cfg["nmf"]["w_max_pos"], cfg["nmf"]["w_max_agg"])
uh_b  = cfg["w_ica100"] * s_ica + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
uh_nmf = cfg["nmf"]["uh_scale"] * uh_b + cfg["nmf"]["w_nmf"] * s_nmf
pT8   = make_lp(cfg["logit_temperature"]); pmt = (pT8 + make_lp(10.0)) / 2
sm6   = make_sp(cfg["softmax_temp"]); ss2 = compute_subspace(ew_pca, n_comp=2, wma_ss=0.92)
w_uh  = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur = w_uh*uh_nmf + cfg["w_logit"]*pT8 + cfg["w_multit"]*pmt + cfg["w_subspace"]*ss2 + cfg["w_softmax"]*sm6
# Fixed conformal signals
s_conf_k1_40 = conformal_score_loo(ew_ica, top_k_fisher=40, k_nn=1)
s_conf_k5_50 = conformal_score_loo(ew_ica, top_k_fisher=50, k_nn=5)
# Also fixed chain components
fh30_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
fh40_b7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
fh40_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=40)
s_attn_ref = attn_knn_fisher_loo(ew_ica, top_k_fisher=40, bw=0.07, attn_T=6.0)
print(f"  Fixed base done [{time.time()-t0:.0f}s]", flush=True)

def build_chain(kde_bw=0.08, fkde_bw=0.06, attn_bw=0.07, attn_T=6.0,
                use_fixed_attn=True, fh30_bw=0.06, fh40b7_bw=0.07, fh40b6_bw=0.06):
    kde = proto_kde_loo(ew_ica, bw=kde_bw)
    final_ref_ = 0.96 * base_cur + 0.04 * kde
    fkde = fisher_kde_loo(ew_ica, bw=fkde_bw)
    fin_ref_ = 0.95 * final_ref_ + 0.05 * fkde
    # Use stored or recomputed fh components
    if fh30_bw == 0.06 and fh40b7_bw == 0.07 and fh40b6_bw == 0.06:
        fh30 = fh30_b6; fh40b7 = fh40_b7; fh40b6 = fh40_b6
    else:
        fh30 = fisher_hard_kde_loo(ew_ica, bw=fh30_bw, top_k=30)
        fh40b7 = fisher_hard_kde_loo(ew_ica, bw=fh40b7_bw, top_k=40)
        fh40b6 = fisher_hard_kde_loo(ew_ica, bw=fh40b6_bw, top_k=40)
    triple_ = 0.94*fin_ref_ + 0.02*fh30 + 0.03*fh40b7 + 0.01*fh40b6
    if use_fixed_attn:
        attn_ = s_attn_ref
    else:
        attn_ = attn_knn_fisher_loo(ew_ica, top_k_fisher=40, bw=attn_bw, attn_T=attn_T)
    return 0.99 * triple_ + 0.01 * attn_

# Current best reference
attn_ref_cur = build_chain()
double_best = 0.997 * attn_ref_cur + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
print(f"  double_best: {macro_auc(double_best):.6f} (expected {best_loo:.6f})", flush=True)

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
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch111"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: Proto-KDE bandwidth sweep (currently bw=0.08)
# proto_kde is the first KDE in the chain → final_ref = 0.96*base + 0.04*kde
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] Proto-KDE bandwidth sweep...", flush=True)
t1 = time.time()
for bw_kde in [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]:
    ar = build_chain(kde_bw=bw_kde)
    blend = 0.997 * ar + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
    reg(f"kde_bw{int(bw_kde*100):03d}_conf", macro_auc(blend))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: Fisher-KDE (soft) bandwidth sweep (currently bw=0.06)
# fisher_kde is the second KDE → fin_ref = 0.95*final_ref + 0.05*fkde
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Fisher-KDE bandwidth sweep...", flush=True)
t1 = time.time()
for bw_fkde in [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
    ar = build_chain(fkde_bw=bw_fkde)
    blend = 0.997 * ar + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
    reg(f"fkde_bw{int(bw_fkde*100):03d}_conf", macro_auc(blend))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: Fisher-Hard KDE bandwidth combinations
# Current: fh30_bw=0.06, fh40b7_bw=0.07, fh40b6_bw=0.06
# Try shifting all bw values up or down
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Fisher-Hard KDE bandwidth combinations...", flush=True)
t1 = time.time()
fh_configs = [
    (0.05, 0.06, 0.05),  # all shifted down
    (0.06, 0.07, 0.06),  # baseline
    (0.07, 0.08, 0.07),  # all shifted up
    (0.06, 0.06, 0.06),  # uniform
    (0.07, 0.07, 0.07),  # uniform up
    (0.05, 0.07, 0.06),  # fh30 narrower
    (0.06, 0.08, 0.07),  # fh40s wider
    (0.06, 0.07, 0.05),  # fh40b6 narrower
]
for fh30_bw, fh40b7_bw, fh40b6_bw in fh_configs:
    ar = build_chain(fh30_bw=fh30_bw, fh40b7_bw=fh40b7_bw, fh40b6_bw=fh40b6_bw)
    blend = 0.997 * ar + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
    reg(f"fh_bw{int(fh30_bw*100):02d}_{int(fh40b7_bw*100):02d}_{int(fh40b6_bw*100):02d}_conf", macro_auc(blend))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: Attention KNN bandwidth sweep (currently bw=0.07)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] Attention KNN bandwidth sweep...", flush=True)
t1 = time.time()
for bw_attn in [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]:
    ar = build_chain(attn_bw=bw_attn, use_fixed_attn=False)
    blend = 0.997 * ar + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
    reg(f"attn_bw{int(bw_attn*100):03d}_conf", macro_auc(blend))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: Attention KNN temperature sweep (currently T=6.0)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Attention temperature sweep...", flush=True)
t1 = time.time()
for attn_T in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]:
    ar = build_chain(attn_T=attn_T, use_fixed_attn=False)
    blend = 0.997 * ar + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
    reg(f"attn_T{int(attn_T*10):03d}_conf", macro_auc(blend))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M6: Best bw found above × conformal w fine-tune
# Joint sweep of best proto_kde bw and attn bw with conformal weights
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M6] Joint bw + conformal weight sweep...", flush=True)
t1 = time.time()
# Try the most promising bw combos
best_bw_combos = [
    (0.07, 0.05, 0.07),  # proto narrower, fisher narrower, attn baseline
    (0.09, 0.07, 0.07),  # proto wider, fisher wider, attn baseline
    (0.08, 0.06, 0.06),  # baseline proto+fisher, attn narrower
    (0.08, 0.06, 0.08),  # baseline proto+fisher, attn wider
]
for kde_bw, fkde_bw, attn_bw in best_bw_combos:
    ar = build_chain(kde_bw=kde_bw, fkde_bw=fkde_bw, attn_bw=attn_bw, use_fixed_attn=False)
    # Try w variations around double-best
    for w40 in [0.002, 0.003]:
        for w50 in [0.001, 0.0015]:
            blend = (1-w40-w50)*ar + w40*s_conf_k1_40 + w50*s_conf_k5_50
            reg(f"joint_kde{int(kde_bw*100):02d}_fkde{int(fkde_bw*100):02d}_attn{int(attn_bw*100):02d}_w{int(w40*1000):04d}_{int(w50*1000):04d}",
                macro_auc(blend))
print(f"  M6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary + Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch111] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

top10 = sorted(results.values(), key=lambda x: -x["loo_auc"])[:10]
print("  Top-10 this batch:", flush=True)
for r2 in top10:
    delta = r2["loo_auc"] - best_loo
    mark = " ***" if r2["loo_auc"] > best_loo else ""
    print(f"    {r2['method']}: {r2['loo_auc']:.6f} ({delta:+.6f}){mark}", flush=True)

if new_best_method:
    print(f"\n  NEW BEST: {new_best_method} LOO={new_best_loo:.6f} (+{new_best_loo-best_loo:.6f})", flush=True)

res2 = json.load(open(RESULTS_PATH))
if isinstance(res2.get("experiments"), list):
    res2["experiments"].extend(list(results.values()))
else:
    res2["experiments"].update(results)
json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"  Saved {len(results)} results to JSON", flush=True)

if new_best_method and new_best_loo > best_loo:
    ep_new = copy.deepcopy(ep)
    ep_new["method"] = new_best_method
    ep_new["loo_auc"] = new_best_loo
    ep_new["batch"] = "batch111"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep_new, f)
    res3 = json.load(open(RESULTS_PATH))
    res3["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch111"}
    json.dump(res3, open(RESULTS_PATH, "w"), indent=2)
    print(f"  PKL + JSON updated → {new_best_method} {new_best_loo:.6f}", flush=True)
else:
    print(f"  No improvement → best remains {best_loo:.6f} ({best['method']})", flush=True)
