"""
Batch 96 — Fisher Hard KDE Refinement
=======================================
Current best: fisher_hard_k30_fbw6_w03 LOO=0.992112

Batch95 showed Fisher hard top-30 dimension selection beats soft Fisher.
Key question: can we refine further?

1. hard_k_sweep     — Fine sweep around k=30: k=18,20,22,25,28,30,32,35,38
2. hard_w_sweep     — Fine sweep around w=0.03 for k=30: w=0.01..0.06
3. hard_bw_sweep    — Different bw for hard k30: bw=0.04,0.05,0.06,0.07,0.08
4. hard_double      — Apply fisher_hard_k30 twice (sequential stacking)
5. hard_weighted    — Hard select k=30, but use Fisher scores as weights (not uniform)
6. hard_k_bw_grid   — 2D grid: k in {20,25,30,35} x bw in {0.05,0.06,0.07}
7. hard_nonneg      — Hard top-30 but only positive-direction dimensions
8. hard_twolevel    — Level1: hard k=30 on fin_ref (w=0.03); Level2: add another fisher_hard on result
9. hard_layered     — hard_k30_bw06 + hard_k15_bw04 (two-layer selection)
10. soft_then_hard  — (already done) vs hard_then_soft
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

print(f"[batch96] ICA{ew_ica.shape}", flush=True)

res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch96] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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
    """Soft Fisher KDE."""
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

def fisher_hard_kde_loo(ew, bw=0.06, top_k=30, weighted=False):
    """Hard top-K Fisher dimension selection KDE.
    weighted=False: uniform 1/sqrt(k) weights for top-K
    weighted=True: use Fisher scores as weights (not uniform) for top-K
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
            top_idx = np.argsort(-fisher_raw)[:top_k]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            if weighted:
                w_dim[top_idx] = fisher_raw[top_idx]
                w_dim /= (norm(w_dim) + EPS)
            else:
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

# Reproduce fisher soft (fin_ref)
print("Computing soft Fisher reference (fin_ref)...", flush=True)
f06 = fisher_kde_loo(ew_ica, bw=0.06)
fin_ref = (1 - 0.05) * final_ref + 0.05 * f06
auc_finref = macro_auc(fin_ref)
print(f"  fin_ref AUC: {auc_finref:.6f} (expected 0.992051)", flush=True)

# Reproduce new best (fisher_hard_k30 on fin_ref w=0.03)
print("Computing hard Fisher k30 reference...", flush=True)
fh30 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
best_ref = (1 - 0.03) * fin_ref + 0.03 * fh30
auc_bestref = macro_auc(best_ref)
print(f"  fisher_hard_k30 w=0.03: {auc_bestref:.6f} (expected 0.992112)", flush=True)

results = {}
new_best_loo = best_loo
new_best_method = None

def reg(name, auc):
    delta = auc - best_loo
    mark = ""
    if auc > best_loo:
        mark = f" *** NEW BEST ***"
    elif auc > best_loo - 0.0003:
        mark = " (near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch96"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# EXP1: Fine k sweep around 30 (on fin_ref)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP1] Fine k sweep (blend on fin_ref, w=0.03)...", flush=True)
t1 = time.time()
for k in [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 42, 45]:
    fh = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=k)
    s = (1 - 0.03) * fin_ref + 0.03 * fh
    reg(f"fhard_k{k:02d}_w03", macro_auc(s))
print(f"  EXP1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP2: Fine w sweep for k=30
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP2] Fine w sweep for k=30...", flush=True)
t1 = time.time()
fh30 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
for w_int in range(1, 9):  # 0.01 to 0.08
    w = w_int * 0.01
    s = (1 - w) * fin_ref + w * fh30
    reg(f"fhard_k30_w{w_int:02d}", macro_auc(s))
print(f"  EXP2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP3: bw sweep for k=30, w=0.03
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP3] bw sweep for k=30, w=0.03...", flush=True)
t1 = time.time()
for bw_int in [4, 5, 6, 7, 8, 10, 12]:
    bw = bw_int * 0.01
    fh = fisher_hard_kde_loo(ew_ica, bw=bw, top_k=30)
    s = (1 - 0.03) * fin_ref + 0.03 * fh
    reg(f"fhard_k30_bw{bw_int:02d}_w03", macro_auc(s))
print(f"  EXP3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP4: 2D grid k x bw (w=0.03)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP4] 2D grid k x bw (w=0.03)...", flush=True)
t1 = time.time()
for k in [20, 25, 30, 35, 40]:
    for bw in [0.05, 0.06, 0.07]:
        if k == 30 and bw == 0.06:
            continue  # already done
        fh = fisher_hard_kde_loo(ew_ica, bw=bw, top_k=k)
        s = (1 - 0.03) * fin_ref + 0.03 * fh
        reg(f"fhard_k{k}_bw{int(bw*100)}_w03", macro_auc(s))
print(f"  EXP4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP5: Double stacking — apply fisher_hard_k30 twice
# Level 1: fin_ref + 0.03 * fh30 (= best_ref)
# Level 2: best_ref + w2 * fh30
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP5] Double stacking fisher_hard_k30...", flush=True)
t1 = time.time()
fh30 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
best_ref_l1 = (1 - 0.03) * fin_ref + 0.03 * fh30
for w2_int in [1, 2, 3]:
    w2 = w2_int * 0.01
    s = (1 - w2) * best_ref_l1 + w2 * fh30
    reg(f"fhard_k30_double_w2_{w2_int:02d}", macro_auc(s))
print(f"  EXP5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP6: Weighted hard Fisher (use Fisher scores as weights for top-K, not uniform)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP6] Weighted hard Fisher (Fisher-weighted top-K)...", flush=True)
t1 = time.time()
for k in [20, 25, 30, 35, 40]:
    fhw = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=k, weighted=True)
    for w_int in [2, 3, 4, 5]:
        w = w_int * 0.01
        s = (1 - w) * fin_ref + w * fhw
        reg(f"fhard_wt_k{k}_w{w_int:02d}", macro_auc(s))
print(f"  EXP6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP7: Non-negative hard Fisher (only dims where mu_p > mu_n, then top-K)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP7] Non-negative hard Fisher...", flush=True)
t1 = time.time()

def fisher_hard_nonneg_kde_loo(ew, bw=0.06, top_k=30):
    """Hard top-K but only from positive-direction dimensions."""
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
            # Zero out anti-informative dims
            fisher_pos = np.where(mu_p > mu_n, fisher_raw, 0.0)
            if fisher_pos.sum() < EPS:
                fisher_pos = fisher_raw  # fallback
            top_idx = np.argsort(-fisher_pos)[:top_k]
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

for k in [20, 25, 30, 35]:
    fhnn = fisher_hard_nonneg_kde_loo(ew_ica, bw=0.06, top_k=k)
    for w_int in [2, 3, 4]:
        w = w_int * 0.01
        s = (1 - w) * fin_ref + w * fhnn
        reg(f"fhard_nn_k{k}_w{w_int:02d}", macro_auc(s))
print(f"  EXP7 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP8: Layered selection (two Fisher hard passes with different k/bw)
# Layer1: hard k=30 bw=0.06 → Layer2: hard k=15 bw=0.04 stacked on layer1
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP8] Layered Fisher hard (k30 + k15)...", flush=True)
t1 = time.time()
fh30 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
fh15 = fisher_hard_kde_loo(ew_ica, bw=0.04, top_k=15)
fh20 = fisher_hard_kde_loo(ew_ica, bw=0.05, top_k=20)
for w1, w2 in [(0.03, 0.02), (0.03, 0.03), (0.03, 0.01), (0.02, 0.02)]:
    s = (1 - w1) * fin_ref + w1 * fh30
    s2 = (1 - w2) * s + w2 * fh15
    reg(f"fhard_layer_w1{int(w1*100):02d}_w2{int(w2*100):02d}", macro_auc(s2))
# Also try k20 as second layer
for w1, w2 in [(0.03, 0.02), (0.03, 0.03)]:
    s = (1 - w1) * fin_ref + w1 * fh30
    s2 = (1 - w2) * s + w2 * fh20
    reg(f"fhard_layer2_w1{int(w1*100):02d}_w2{int(w2*100):02d}", macro_auc(s2))
print(f"  EXP8 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP9: Fisher hard k=30 on top of best_ref (current best) — further stacking
# blend with different w on top of fisher_hard_k30_fbw6_w03 result
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP9] Further stacking on best_ref (0.992112)...", flush=True)
t1 = time.time()
fh30 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
best_ref = (1 - 0.03) * fin_ref + 0.03 * fh30

# Try adding other methods on top of best_ref
fh25 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=25)
fh35 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=35)
fh30_bw5 = fisher_hard_kde_loo(ew_ica, bw=0.05, top_k=30)
fh30_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=30)

for tag, sig in [("fh25", fh25), ("fh35", fh35), ("fh30bw5", fh30_bw5), ("fh30bw7", fh30_bw7)]:
    for w_int in [1, 2, 3]:
        w = w_int * 0.01
        s = (1 - w) * best_ref + w * sig
        reg(f"stack_{tag}_w{w_int:02d}", macro_auc(s))
print(f"  EXP9 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch96] SUMMARY", flush=True)
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

# ── Save ─────────────────────────────────────────────────────────────────────
res2 = json.load(open(RESULTS_PATH))
if isinstance(res2.get("experiments"), list):
    # append as list entries
    res2["experiments"].extend(list(results.values()))
elif isinstance(res2.get("experiments"), dict):
    res2["experiments"].update(results)
else:
    res2["experiments"] = list(results.values())

if new_best_loo > best_loo:
    res2["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch96"}
    ep2 = copy.deepcopy(ep)
    ep2["loo_auc"] = new_best_loo
    ep2["method"] = new_best_method
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep2, f)
    print(f"\n  SAVED new best: {new_best_method} LOO={new_best_loo:.6f}", flush=True)

json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"\nSaved {len(results)} experiments to JSON.", flush=True)
