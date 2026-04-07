"""
Batch 106 — Conformal k=1 Deep Refinement
===============================================================================
Current best: conf_k1_kf40_w0003 LOO=0.992282
Reference: conf_ref = 0.997*attn_ref + 0.003*conf_k1_kf40

Top-10 findings from batch105:
  k=1,kf=40,w=0.003 → 0.992282 (BEST)
  k=5,kf=50,w=0.003 → 0.992241
  k=1,kf=40,w=0.007 → 0.992236

This batch:
1. k=1 × kf sweep: try kf=50 with k=1 (promising combo not yet tried)
2. Fine w sweep for k=1 kf=50
3. Double conformal stacking: conf_k1_kf40 + conf_k1_kf50
4. Conformal in soft Fisher space (vs hard selection)
5. Conformal on PCA/STD embedding spaces
6. Combined: conf on triple_ref + attn vs conf on attn_ref
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

print(f"[batch106] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch106] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

def conformal_score_loo(ew, top_k_fisher=40, k_nn=1, mode="ratio"):
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
            if mode == "ratio":
                score = knn_pos / (knn_neg + EPS) - 1.0
            elif mode == "diff":
                score = knn_pos - knn_neg
            ws[:, si] = np.clip(score, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def conformal_soft_loo(ew, k_nn=1):
    """Conformal using soft Fisher weights (not hard top-k)."""
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
            # Soft Fisher weighting (not hard mask)
            w_dim = fisher_raw / (norm(fisher_raw) + EPS)
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
# Pre-compute best conformal signal (k=1, kf=40)
s_conf_k1_40 = conformal_score_loo(ew_ica, top_k_fisher=40, k_nn=1, mode="ratio")
conf_ref = 0.997 * attn_ref + 0.003 * s_conf_k1_40
auc_cr = macro_auc(conf_ref)
print(f"  conf_ref: {auc_cr:.6f} (expected 0.992282) [{time.time()-t0:.0f}s]", flush=True)

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
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch106"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: k=1 × kf sweep (kf=50 was best for k=5, need to test with k=1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] k=1 × kf sweep on attn_ref...", flush=True)
t1 = time.time()
for kf in [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80]:
    s = conformal_score_loo(ew_ica, top_k_fisher=kf, k_nn=1, mode="ratio")
    for w_val in [0.002, 0.003, 0.004, 0.005]:
        blend = (1 - w_val) * attn_ref + w_val * s
        reg(f"c1kf{kf}_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: Fine w sweep for best k=1 configs (k=1,kf=40 and k=1,kf=50)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Fine w sweep (k=1, kf=40 & kf=50)...", flush=True)
t1 = time.time()
s_c1_50 = conformal_score_loo(ew_ica, top_k_fisher=50, k_nn=1, mode="ratio")
for w_val in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005, 0.006, 0.007, 0.008]:
    # kf=40
    blend = (1 - w_val) * attn_ref + w_val * s_conf_k1_40
    reg(f"c1k40_fine_w{int(w_val*10000):05d}", macro_auc(blend))
    # kf=50
    blend = (1 - w_val) * attn_ref + w_val * s_c1_50
    reg(f"c1k50_fine_w{int(w_val*10000):05d}", macro_auc(blend))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: Double conformal stacking on attn_ref
# Two complementary conformal signals
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Double conformal stacking...", flush=True)
t1 = time.time()
s_c1_30 = conformal_score_loo(ew_ica, top_k_fisher=30, k_nn=1, mode="ratio")
s_c5_40 = conformal_score_loo(ew_ica, top_k_fisher=40, k_nn=5, mode="ratio")
s_c5_50 = conformal_score_loo(ew_ica, top_k_fisher=50, k_nn=5, mode="ratio")

secondary = [("c1k30", s_c1_30), ("c1k50", s_c1_50), ("c5k40", s_c5_40), ("c5k50", s_c5_50)]
for name2, s2 in secondary:
    for w_val in [0.001, 0.002, 0.003]:
        # conf_k1_40 fixed at w=0.003, add second conformal
        blend = (1 - 0.003 - w_val) * attn_ref + 0.003 * s_conf_k1_40 + w_val * s2
        reg(f"double_c1k40+{name2}_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: Conformal k=1 on conf_ref (stack on top of the full chain incl. conformal)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] Conformal k=1 stacked on conf_ref...", flush=True)
t1 = time.time()
for w_val in [0.001, 0.002, 0.003, 0.004, 0.005]:
    blend = (1 - w_val) * conf_ref + w_val * s_conf_k1_40
    reg(f"conf_on_confref_k1k40_w{int(w_val*1000):04d}", macro_auc(blend))
    blend2 = (1 - w_val) * conf_ref + w_val * s_c1_50
    reg(f"conf_on_confref_k1k50_w{int(w_val*1000):04d}", macro_auc(blend2))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: Soft Fisher conformal (k=1)
# Use continuous Fisher weights instead of hard top-k selection
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Soft Fisher conformal (k=1)...", flush=True)
t1 = time.time()
s_soft_c1 = conformal_soft_loo(ew_ica, k_nn=1)
for w_val in [0.002, 0.003, 0.004, 0.005]:
    blend = (1 - w_val) * attn_ref + w_val * s_soft_c1
    reg(f"soft_conf_k1_w{int(w_val*1000):04d}", macro_auc(blend))
    # Also stack with hard conformal
    blend2 = (1 - 0.003 - w_val) * attn_ref + 0.003 * s_conf_k1_40 + w_val * s_soft_c1
    reg(f"hard+soft_c1k40+sc1_w{int(w_val*1000):04d}", macro_auc(blend2))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M6: Conformal on PCA/STD embeddings (cross-space)
# Different embedding spaces may give complementary conformal signal
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M6] Conformal on PCA/STD embeddings...", flush=True)
t1 = time.time()
s_conf_pca = conformal_score_loo(ew_pca, top_k_fisher=40, k_nn=1, mode="ratio")
s_conf_std = conformal_score_loo(ew_std, top_k_fisher=40, k_nn=1, mode="ratio")
for name2, s2 in [("pca", s_conf_pca), ("std", s_conf_std)]:
    for w_val in [0.002, 0.003, 0.004]:
        blend = (1 - w_val) * attn_ref + w_val * s2
        reg(f"conf_k1_{name2}_w{int(w_val*1000):04d}", macro_auc(blend))
    # Stack with ICA conformal
    for w_val in [0.001, 0.002]:
        blend = (1 - 0.003 - w_val) * attn_ref + 0.003 * s_conf_k1_40 + w_val * s2
        reg(f"conf_k1_ica+{name2}_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M7: Triple combination: attn + conf_k1_kf40 + conf_k1_kf50
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M7] Triple: attn_ref + conf_k1_kf40 + conf_k1_kf50...", flush=True)
t1 = time.time()
for w40, w50 in [(0.002, 0.001), (0.003, 0.001), (0.002, 0.002), (0.003, 0.002), (0.002, 0.003)]:
    base_w = 1 - w40 - w50
    if base_w < 0: continue
    blend = base_w * attn_ref + w40 * s_conf_k1_40 + w50 * s_c1_50
    reg(f"triple_attn_c1k40w{int(w40*1000):03d}_c1k50w{int(w50*1000):03d}", macro_auc(blend))
print(f"  M7 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary + Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch106] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

if new_best_method:
    top5 = sorted(results.values(), key=lambda x: -x["loo_auc"])[:5]
    for r2 in top5:
        delta = r2["loo_auc"] - best_loo
        print(f"    {r2['method']}: {r2['loo_auc']:.6f} ({delta:+.6f})", flush=True)
else:
    print("  No improvement found.", flush=True)

res2 = json.load(open(RESULTS_PATH))
if isinstance(res2.get("experiments"), list):
    res2["experiments"].extend(list(results.values()))
else:
    res2["experiments"].update(results)
json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"  Saved {len(results)} results to JSON", flush=True)

if new_best_method and new_best_loo > best_loo:
    print(f"\n  SAVED: {new_best_method} LOO={new_best_loo:.6f}", flush=True)
    ep_new = copy.deepcopy(ep)
    ep_new["method"] = new_best_method
    ep_new["loo_auc"] = new_best_loo
    ep_new["batch"] = "batch106"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep_new, f)
    res3 = json.load(open(RESULTS_PATH))
    res3["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch106"}
    json.dump(res3, open(RESULTS_PATH, "w"), indent=2)
    print(f"  Updated JSON best → {new_best_method} {new_best_loo:.6f}", flush=True)
else:
    print(f"  No improvement → best remains {best_loo:.6f} ({best['method']})", flush=True)
