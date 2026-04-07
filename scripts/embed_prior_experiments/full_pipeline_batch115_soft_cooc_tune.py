"""
Batch 115 — Fine-tune Soft-gate Co-occurrence Smoothing
===============================================================================
Current best: soft_c055_sl20_a0075 LOO=0.994086 (+0.000407 from batch114)
Key trends: slope=20>>12>>8>>5, center=0.55>0.50>0.45, alpha still rising at 0.075

Strategy: Push all three dimensions further:
1. M1: Larger slope (30, 50, 100, 200) with center=0.55, alpha=0.075
2. M2: Higher center (0.55→0.70) with slope=20, alpha=0.075
3. M3: Larger alpha (0.080→0.150) with best center/slope from M1/M2
4. M4: Joint fine sweep of (center × slope × alpha) in promising region
5. M5: Ultra-sharp sigmoid approaches (approaching hard threshold from soft side)
6. M6: Stack soft_cooc + conformal more carefully with best params
"""
import numpy as np
import json
import pickle
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

print(f"[batch115] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch115] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

# ── Pre-compute full chain ────────────────────────────────────────────────────
print("Pre-computing full chain...", flush=True)
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
w_uh  = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur  = w_uh*uh_nmf + cfg["w_logit"]*pT8 + cfg["w_multit"]*pmt + cfg["w_subspace"]*ss2 + cfg["w_softmax"]*sm6
final_ref = 0.96 * base_cur + 0.04 * kde08
f06 = fisher_kde_loo(ew_ica, bw=0.06)
fin_ref = 0.95 * final_ref + 0.05 * f06
fh30_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
fh40_b7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
fh40_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=40)
triple_ref = 0.94*fin_ref + 0.02*fh30_b6 + 0.03*fh40_b7 + 0.01*fh40_b6
s_attn = attn_knn_fisher_loo(ew_ica, top_k_fisher=40, bw=0.07, attn_T=6.0)
attn_ref = 0.99 * triple_ref + 0.01 * s_attn
s_conf_k1_40 = conformal_score_loo(ew_ica, top_k_fisher=40, k_nn=1)
s_conf_k5_50 = conformal_score_loo(ew_ica, top_k_fisher=50, k_nn=5)
double_best = 0.997 * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
auc_db = macro_auc(double_best)
print(f"  double_best: {auc_db:.6f} (expected 0.992312) [{time.time()-t0:.0f}s]", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Soft-gate co-occurrence smoothing
# ─────────────────────────────────────────────────────────────────────────────
def _cooc_matrix():
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0) + EPS
    cooc_norm = cooc / count_i[:, None]
    np.fill_diagonal(cooc_norm, 0)
    return cooc_norm

COOC_NORM = _cooc_matrix()

def soft_cooc(scores, center=0.55, slope=20.0, alpha=0.075):
    """Soft sigmoid gate co-occurrence smoothing."""
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(-slope * (s - center)))
        s_gated = s * gate
        total = np.abs(s_gated).sum()
        if total < EPS:
            smoothed[fi] = s
            continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS:
            contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def hard_cooc(scores, threshold=0.50, alpha=0.060):
    """Hard threshold gate co-occurrence smoothing."""
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        s_thresh = s * (s > threshold).astype(np.float32)
        if s_thresh.sum() < EPS:
            smoothed[fi] = s
            continue
        contrib = COOC_NORM.T @ s_thresh
        max_c = np.abs(contrib).max()
        if max_c > EPS:
            contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

# Results tracking
results = {}
new_best_loo = best_loo
new_best_method = best["method"]

def reg(name, auc):
    global new_best_loo, new_best_method
    delta = auc - best_loo
    if auc > new_best_loo:
        new_best_loo = auc
        new_best_method = name
    mark = " *** NEW BEST ***" if auc > best_loo else (" (near-best)" if auc > best_loo - 0.0010 else "")
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch115"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: Larger slope sweep (center=0.55, alpha=0.075)
# batch114: sl20=0.994086; slope trend still rising
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] Larger slope sweep (center=0.55, alpha=0.075)...", flush=True)
t1 = time.time()
for slope in [25.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0]:
    s = soft_cooc(double_best, center=0.55, slope=slope, alpha=0.075)
    reg(f"soft_c055_sl{int(slope):03d}_a0075", macro_auc(s))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: Higher center sweep (slope=20, alpha=0.075)
# batch114: c050<c055; explore c055→c075
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Higher center sweep (slope=20, alpha=0.075)...", flush=True)
t1 = time.time()
for center in [0.57, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]:
    s = soft_cooc(double_best, center=float(center), slope=20.0, alpha=0.075)
    reg(f"soft_c{int(round(center*100)):03d}_sl020_a0075", macro_auc(s))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: Larger alpha sweep (center=0.55, slope=20)
# batch114: a075=0.994086; explore beyond
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Larger alpha sweep (center=0.55, slope=20)...", flush=True)
t1 = time.time()
for alpha in [0.080, 0.085, 0.090, 0.100, 0.110, 0.125, 0.150, 0.175, 0.200]:
    s = soft_cooc(double_best, center=0.55, slope=20.0, alpha=float(alpha))
    reg(f"soft_c055_sl020_a{int(round(alpha*1000)):04d}", macro_auc(s))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: Joint fine sweep after M1-M3 findings
# Use the best slope from M1 and best center from M2
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] Joint center × slope × alpha sweep...", flush=True)
t1 = time.time()

# Find best from M1-M3
best_m123_auc = best_loo
best_m123_slope = 20.0
best_m123_center = 0.55
best_m123_alpha = 0.075
for name, r in results.items():
    if r["loo_auc"] > best_m123_auc:
        best_m123_auc = r["loo_auc"]
        # Parse params
        if "soft_c" in name and "_sl" in name and "_a" in name:
            try:
                parts = name.replace("soft_c", "").split("_sl")
                best_m123_center = float(parts[0]) / 100.0
                rest = parts[1].split("_a")
                best_m123_slope = float(rest[0])
                best_m123_alpha = float(rest[1]) / 1000.0
            except:
                pass

print(f"  Best from M1-M3: center={best_m123_center:.2f}, slope={best_m123_slope:.0f}, alpha={best_m123_alpha:.3f}, LOO={best_m123_auc:.6f}", flush=True)

# Sweep around the new best
for center in [best_m123_center - 0.02, best_m123_center, best_m123_center + 0.02]:
    for slope_mult in [0.75, 1.0, 1.25, 1.5]:
        slope = best_m123_slope * slope_mult
        for alpha_mult in [0.85, 1.0, 1.15]:
            alpha = best_m123_alpha * alpha_mult
            if center <= 0 or center >= 1: continue
            s = soft_cooc(double_best, center=float(center), slope=float(slope), alpha=float(alpha))
            reg(f"joint_c{int(round(center*100)):03d}_sl{int(slope):03d}_a{int(round(alpha*1000)):04d}", macro_auc(s))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: Comparing soft vs hard at equivalent sharpness + extreme slopes
# At slope=100-200, the sigmoid ≈ hard threshold
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Extreme slope variants + hard threshold comparison...", flush=True)
t1 = time.time()

# Hard threshold at same effective cutoffs
for thr in [0.53, 0.55, 0.57, 0.60]:
    for alpha in [0.060, 0.075, 0.090]:
        s = hard_cooc(double_best, threshold=float(thr), alpha=float(alpha))
        reg(f"hard_t{int(thr*100):03d}_a{int(alpha*1000):04d}", macro_auc(s))

# Very large slope soft (approximate hard)
for slope in [300.0, 500.0, 1000.0]:
    s = soft_cooc(double_best, center=0.55, slope=slope, alpha=0.075)
    reg(f"soft_c055_sl{int(slope):04d}_a0075", macro_auc(s))

print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M6: Best soft_cooc + conformal stacking
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M6] Best soft_cooc + conformal stacking...", flush=True)
t1 = time.time()

# Find overall best from this batch
best_b115_auc = best_loo
best_b115_params = (0.55, 20.0, 0.075)
for name, r in results.items():
    if r["loo_auc"] > best_b115_auc:
        best_b115_auc = r["loo_auc"]
        if "soft_c" in name and "_sl" in name and "_a" in name:
            try:
                parts = name.replace("soft_c", "").split("_sl")
                c = float(parts[0]) / 100.0
                rest = parts[1].split("_a")
                sl = float(rest[0])
                al = float(rest[1]) / 1000.0
                best_b115_params = (c, sl, al)
            except:
                pass

c_best, sl_best, al_best = best_b115_params
print(f"  Best params: center={c_best:.2f}, slope={sl_best:.0f}, alpha={al_best:.3f} LOO={best_b115_auc:.6f}", flush=True)
s_best_soft = soft_cooc(double_best, center=c_best, slope=sl_best, alpha=al_best)

# Stack conformal
for w_conf in [0.001, 0.002, 0.003, 0.005]:
    blend = (1 - w_conf) * s_best_soft + w_conf * s_conf_k1_40
    reg(f"best_soft+conf_w{int(w_conf*1000):04d}", macro_auc(blend))

# Apply soft_cooc before conformal
s_soft_before = soft_cooc(attn_ref, center=c_best, slope=sl_best, alpha=al_best)
blend = 0.997 * s_soft_before + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
reg("best_soft_on_attn+conf", macro_auc(blend))

# Two rounds of soft_cooc
s_soft2 = soft_cooc(s_best_soft, center=c_best, slope=sl_best, alpha=al_best * 0.5)
reg("best_soft_x2_half", macro_auc(s_soft2))

print(f"  M6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print(f"[batch115] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

sorted_results = sorted(results.items(), key=lambda x: -x[1]["loo_auc"])
print(f"  Top-10 this batch:", flush=True)
for name, r in sorted_results[:10]:
    delta = r["loo_auc"] - best_loo
    print(f"    {name}: {r['loo_auc']:.6f} ({delta:+.6f})", flush=True)

for name, r in results.items():
    res["experiments"].append(r)

if new_best_loo > best_loo:
    res["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch115"}
    print(f"\n  NEW BEST: {new_best_method} LOO={new_best_loo:.6f} (+{new_best_loo-best_loo:.6f})", flush=True)
    ep["best_method"] = new_best_method
    ep["best_loo"] = new_best_loo
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep, f)
    print(f"  PKL + JSON updated → {new_best_method} {new_best_loo:.6f}", flush=True)
else:
    print(f"\n  No improvement → best remains {best_loo:.6f} ({best['method']})", flush=True)

with open(RESULTS_PATH, "w") as f:
    json.dump(res, f, indent=2)
print(f"  Saved {len(results)} results to JSON", flush=True)
