"""
Batch 98 — Breaking the 0.9922 Ceiling
========================================
Current best: fh40bw7_w04 LOO=0.992153
Increments are shrinking (0.000013 last batch).

Trying fundamentally new approaches to break through the 0.9922 ceiling:

1. mahalanobis_kde — Full (regularized) Mahalanobis distance in Fisher space
2. quantile_fisher — Quantile-based Fisher (use median/IQR instead of mean/var)
3. per_species_k    — Different k per species based on #positive examples
4. fisher_combo_dim — Blend of k=30_bw6 + k=40_bw7 at the score level (not embedding level)
5. fisher_hard_neg  — Add negative Fisher hard KDE as repulsion term
6. fisher_proto_refine — After Fisher hard KDE, refine prototype by re-weighting outliers
7. soft_hard_interp — Interpolate between soft and hard Fisher as a function of alpha
8. fisher_knn_hybrid — Use Fisher-weighted KNN instead of KDE
9. wl_fisher_combo  — WL score in Fisher-weighted space (different from batch93)
10. extra_w_sweep   — Extend w sweep beyond 0.08 for k=40, bw=0.07
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

print(f"[batch98] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch98] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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
f06 = fisher_kde_loo(ew_ica, bw=0.06)
fin_ref = (1 - 0.05) * final_ref + 0.05 * f06
fh40_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
cur_best = (1 - 0.04) * fin_ref + 0.04 * fh40_bw7
auc_cur = macro_auc(cur_best)
print(f"  cur_best: {auc_cur:.6f} (expected 0.992153)", flush=True)

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
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch98"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# EXP1: Extend w sweep beyond 0.08 for k=40, bw=0.07
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP1] Extended w sweep for k=40, bw=0.07...", flush=True)
t1 = time.time()
for w_int in [9, 10, 12, 15, 20]:
    w = w_int * 0.01
    s = (1 - w) * fin_ref + w * fh40_bw7
    reg(f"fh40bw7_xw{w_int:02d}", macro_auc(s))
print(f"  EXP1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP2: Quantile-based Fisher (use robust statistics instead of mean/var)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP2] Quantile-based Fisher KDE...", flush=True)
t1 = time.time()

def fisher_quantile_kde_loo(ew, bw=0.06, top_k=40, q_lo=0.25, q_hi=0.75):
    """Use median/IQR instead of mean/var for Fisher statistic."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            # Use median instead of mean, IQR instead of variance
            med_p = np.median(pos_wins, axis=0); med_n = np.median(neg_wins, axis=0)
            iqr_p = np.percentile(pos_wins, q_hi*100, axis=0) - np.percentile(pos_wins, q_lo*100, axis=0) + EPS
            iqr_n = np.percentile(neg_wins, q_hi*100, axis=0) - np.percentile(neg_wins, q_lo*100, axis=0) + EPS
            fisher_q = np.abs(med_p - med_n) / (iqr_p + iqr_n)  # robust Fisher
            top_idx = np.argsort(-fisher_q)[:top_k]
            w_dim = np.zeros(len(fisher_q), np.float32)
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

for k, bw in [(30, 0.06), (40, 0.07), (40, 0.06)]:
    s_q = fisher_quantile_kde_loo(ew_ica, bw=bw, top_k=k)
    for w_int in [3, 4, 5]:
        w = w_int * 0.01
        s = (1 - w) * fin_ref + w * s_q
        reg(f"fquant_k{k}_bw{int(bw*100)}_w{w_int:02d}", macro_auc(s))
print(f"  EXP2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP3: Per-species adaptive k (larger k for species with more positives)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP3] Per-species adaptive k...", flush=True)
t1 = time.time()

def fisher_adaptive_k_kde_loo(ew, bw=0.07, k_base=30, k_max=50):
    """Adaptive k: k = min(k_base + n_pos/2, k_max) per species."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            n_pos = pm.sum()
            k = min(k_base + n_pos // 2, k_max, ew.shape[1])
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:k]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(k))
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

for k_base, k_max, bw in [(30, 50, 0.07), (30, 60, 0.07), (25, 50, 0.07), (35, 55, 0.07)]:
    s_ak = fisher_adaptive_k_kde_loo(ew_ica, bw=bw, k_base=k_base, k_max=k_max)
    for w_int in [3, 4]:
        w = w_int * 0.01
        s = (1 - w) * fin_ref + w * s_ak
        reg(f"fadaptk_b{k_base}_m{k_max}_bw7_w{w_int:02d}", macro_auc(s))
print(f"  EXP3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP4: Mahalanobis distance in Fisher-selected subspace
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP4] Mahalanobis KDE in Fisher subspace...", flush=True)
t1 = time.time()

def fisher_mahal_kde_loo(ew, bw=0.07, top_k=40):
    """Use Mahalanobis (full diagonal covariance) in Fisher-selected subspace."""
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
            # In Fisher subspace, use diagonal Mahalanobis (1/std scaling)
            pos_sub = pos_wins[:, top_idx]
            std_sub = pos_sub.std(0) + EPS
            # Scale by inverse std → unit variance in each dimension
            scale = 1.0 / (std_sub * np.sqrt(float(top_k)))
            te_m = te[:, top_idx] * scale[None, :]
            tr_m = pos_sub * scale[None, :]
            # Normalize to unit sphere
            te_m /= norm(te_m, axis=1, keepdims=True) + EPS
            tr_m /= norm(tr_m, axis=1, keepdims=True) + EPS
            centroid = tr_m.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_m @ centroid, 0, None); proto_w /= (proto_w.sum() + EPS)
            sims_m = te_m @ tr_m.T
            kern = np.exp((sims_m - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for k, bw in [(40, 0.06), (40, 0.07), (40, 0.08), (30, 0.07)]:
    s_mah = fisher_mahal_kde_loo(ew_ica, bw=bw, top_k=k)
    for w_int in [3, 4, 5]:
        w = w_int * 0.01
        s = (1 - w) * fin_ref + w * s_mah
        reg(f"fmahal_k{k}_bw{int(bw*100)}_w{w_int:02d}", macro_auc(s))
print(f"  EXP4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP5: Fisher KNN (k-nearest neighbor instead of KDE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP5] Fisher-weighted KNN score...", flush=True)
t1 = time.time()

def fisher_knn_loo(ew, top_k_fisher=40, k_nn=5):
    """In Fisher-weighted space, use k-NN score instead of KDE."""
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
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            # Fisher-weighted embeddings
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            # KNN: score = mean similarity of top-K nearest neighbors
            sims = te_w @ tr_w.T  # (N_test, N_pos)
            k2 = min(k_nn, sims.shape[1])
            top_sims = np.sort(sims, axis=1)[:, -k2:]
            knn_score = top_sims.mean(1)
            knn_score = (knn_score + 1) / 2  # normalize to [0,1]
            ws[:, si] = np.clip(knn_score, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for k_nn in [3, 5, 7, 10]:
    s_knn = fisher_knn_loo(ew_ica, top_k_fisher=40, k_nn=k_nn)
    for w_int in [3, 4, 5]:
        w = w_int * 0.01
        s = (1 - w) * fin_ref + w * s_knn
        reg(f"fknn_k40_nn{k_nn}_w{w_int:02d}", macro_auc(s))
print(f"  EXP5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP6: Score-level fusion of hard Fisher k30+k40 (already computed)
# Try triple combination: fh30_bw6 + fh40_bw7 + fh40_bw6
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP6] Triple Fisher combination...", flush=True)
t1 = time.time()
fh30_bw6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
fh40_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
fh40_bw6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=40)
fh37_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=37)

for w1, w2, w3 in [(0.02, 0.03, 0.01), (0.02, 0.02, 0.02), (0.01, 0.03, 0.02)]:
    s = (1 - w1 - w2 - w3) * fin_ref + w1 * fh30_bw6 + w2 * fh40_bw7 + w3 * fh40_bw6
    reg(f"triple_30w{int(w1*100):02d}_40w{int(w2*100):02d}_40bw6w{int(w3*100):02d}", macro_auc(s))

# fh37 vs fh40
for w_int in [3, 4]:
    w = w_int * 0.01
    s = (1 - w) * fin_ref + w * fh37_bw7
    reg(f"fh37bw7_w{w_int:02d}", macro_auc(s))

# combined fh37 + fh40
for w1_int, w2_int in [(2, 2), (2, 3), (3, 2)]:
    w1, w2 = w1_int * 0.01, w2_int * 0.01
    s = (1 - w1 - w2) * fin_ref + w1 * fh37_bw7 + w2 * fh40_bw7
    reg(f"fh37_40_w{w1_int}_w{w2_int}", macro_auc(s))

print(f"  EXP6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP7: Try adding to current_best (0.992153) with small corrections
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP7] Small corrections on top of current best (0.992153)...", flush=True)
t1 = time.time()
cur_best = (1 - 0.04) * fin_ref + 0.04 * fh40_bw7

# Try small additions of proto_kde, fisher soft, etc.
extra_signals = [
    ("pkde08", proto_kde_loo(ew_ica, bw=0.08)),
    ("pkde06", proto_kde_loo(ew_ica, bw=0.06)),
    ("fsoft06", fisher_kde_loo(ew_ica, bw=0.06)),
    ("fhard30bw6", fh30_bw6),
]
for tag, sig in extra_signals:
    for w_int in [1, 2]:
        w = w_int * 0.01
        s = (1 - w) * cur_best + w * sig
        reg(f"cb_plus_{tag}_w{w_int:02d}", macro_auc(s))
print(f"  EXP7 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP8: Interpolation between soft Fisher and hard Fisher (alpha blending)
# alpha=0: pure soft Fisher; alpha=1: pure hard Fisher
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP8] Soft-to-hard Fisher interpolation...", flush=True)
t1 = time.time()

def fisher_interp_kde_loo(ew, bw=0.07, top_k=40, alpha=0.5):
    """Interpolate between soft and hard Fisher weights."""
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
            # Soft weights
            w_soft = fisher_raw / (norm(fisher_raw) + EPS)
            # Hard weights (top-k uniform)
            top_idx = np.argsort(-fisher_raw)[:top_k]
            w_hard = np.zeros(len(fisher_raw), np.float32)
            w_hard[top_idx] = 1.0 / np.sqrt(float(top_k))
            # Interpolate
            w_dim = (1 - alpha) * w_soft + alpha * w_hard
            w_dim /= (norm(w_dim) + EPS)
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

for alpha in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
    s_int = fisher_interp_kde_loo(ew_ica, bw=0.07, top_k=40, alpha=alpha)
    for w_int in [3, 4, 5]:
        w = w_int * 0.01
        s = (1 - w) * fin_ref + w * s_int
        reg(f"finterp_a{int(alpha*10):02d}_bw7_w{w_int:02d}", macro_auc(s))
print(f"  EXP8 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch98] SUMMARY", flush=True)
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
    res2["experiments"].extend(list(results.values()))
elif isinstance(res2.get("experiments"), dict):
    res2["experiments"].update(results)
else:
    res2["experiments"] = list(results.values())

if new_best_loo > best_loo:
    res2["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch98"}
    ep2 = copy.deepcopy(ep)
    ep2["loo_auc"] = new_best_loo
    ep2["method"] = new_best_method
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep2, f)
    print(f"\n  SAVED new best: {new_best_method} LOO={new_best_loo:.6f}", flush=True)

json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"\nSaved {len(results)} experiments to JSON.", flush=True)
