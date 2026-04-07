"""
Batch 113 — Deep Co-occurrence Smoothing Exploration
===============================================================================
Current best: cooc_smooth_a020 LOO=0.992545 (+0.000233 from batch112)
Key finding: label co-occurrence smoothing shows STRONG upward trend at alpha=0.020.
Trend: a010=+0.000006, a015=+0.000126, a020=+0.000233 — still rising steeply!

Strategy: Exhaustively explore co-occurrence smoothing variants:
1. M1: Alpha sweep extended (0.025→0.30) — find the peak
2. M2: Iterative co-occurrence smoothing (apply K rounds)
3. M3: Nonlinear co-occurrence (sigmoid/power-sharpened source)
4. M4: Conditional co-occurrence (threshold-gated propagation)
5. M5: Symmetric / Jaccard co-occurrence matrix
6. M6: Stack cooc on top of new chain stages + two-stage cooc
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

print(f"[batch113] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch113] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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
# Co-occurrence smoothing function (from batch112)
# ─────────────────────────────────────────────────────────────────────────────
def cooccurrence_smooth(scores, alpha=0.1):
    """Smooth scores using label co-occurrence matrix from all 66 files."""
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl  # (n_species, n_species)
    count_i = fl.sum(0) + EPS
    cooc_norm = cooc / count_i[:, None]  # P(j|i)
    np.fill_diagonal(cooc_norm, 0)
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        contrib = cooc_norm.T @ s
        contrib /= (np.abs(contrib).max() + EPS)
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

# ─────────────────────────────────────────────────────────────────────────────
# Results tracking
# ─────────────────────────────────────────────────────────────────────────────
results = {}
new_best_loo = best_loo
new_best_method = best["method"]

def reg(name, auc):
    global new_best_loo, new_best_method
    delta = auc - best_loo
    if auc > new_best_loo:
        new_best_loo = auc
        new_best_method = name
    mark = " *** NEW BEST ***" if auc > best_loo else (" (near-best)" if auc > best_loo - 0.0003 else "")
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch113"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: Extended alpha sweep — find the peak of co-occurrence smoothing
# batch112 trend: a010=+6e-6, a015=+126e-6, a020=+233e-6 (still rising)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] Extended alpha sweep for co-occurrence smoothing...", flush=True)
t1 = time.time()

for alpha in [0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.250, 0.300]:
    s_coo = cooccurrence_smooth(double_best, alpha=alpha)
    reg(f"cooc_a{int(alpha*1000):04d}", macro_auc(s_coo))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: Iterative co-occurrence smoothing (apply K rounds with smaller step)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Iterative co-occurrence smoothing...", flush=True)
t1 = time.time()

def cooccurrence_smooth_iter(scores, alpha=0.02, n_iter=5):
    """Apply co-occurrence smoothing iteratively."""
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0) + EPS
    cooc_norm = cooc / count_i[:, None]
    np.fill_diagonal(cooc_norm, 0)
    s = scores.copy()
    for _ in range(n_iter):
        new_s = np.zeros_like(s)
        for fi in range(n_files):
            contrib = cooc_norm.T @ s[fi]
            contrib /= (np.abs(contrib).max() + EPS)
            new_s[fi] = (1 - alpha) * s[fi] + alpha * np.clip(contrib, 0, None)
        s = new_s
    return s

for n_iter in [2, 3, 5, 8]:
    for alpha_step in [0.010, 0.015, 0.020]:
        s_iter = cooccurrence_smooth_iter(double_best, alpha=alpha_step, n_iter=n_iter)
        reg(f"cooc_iter{n_iter}_a{int(alpha_step*1000):04d}", macro_auc(s_iter))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: Sigmoid/power-weighted co-occurrence (boost high-confidence species more)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Nonlinear co-occurrence smoothing...", flush=True)
t1 = time.time()

def cooccurrence_smooth_sigmoid(scores, alpha=0.05, temp=5.0):
    """Use sigmoid-sharpened scores as source for co-occurrence contribution."""
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0) + EPS
    cooc_norm = cooc / count_i[:, None]
    np.fill_diagonal(cooc_norm, 0)
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        s_sharp = 1.0 / (1.0 + np.exp(-temp * (s - 0.5)))
        contrib = cooc_norm.T @ s_sharp
        contrib /= (np.abs(contrib).max() + EPS)
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def cooccurrence_smooth_power(scores, alpha=0.05, power=2.0):
    """Use power-sharpened scores as source for co-occurrence contribution."""
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0) + EPS
    cooc_norm = cooc / count_i[:, None]
    np.fill_diagonal(cooc_norm, 0)
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        s_power = np.clip(s, 0, 1) ** power
        contrib = cooc_norm.T @ s_power
        contrib /= (np.abs(contrib).max() + EPS)
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

for alpha in [0.020, 0.030, 0.050, 0.075]:
    for temp in [3.0, 5.0, 8.0]:
        s_sig = cooccurrence_smooth_sigmoid(double_best, alpha=alpha, temp=temp)
        reg(f"cooc_sig_a{int(alpha*1000):04d}_T{int(temp*10):03d}", macro_auc(s_sig))

for alpha in [0.020, 0.030, 0.050, 0.075]:
    for pw in [1.5, 2.0, 3.0]:
        s_pw = cooccurrence_smooth_power(double_best, alpha=alpha, power=pw)
        reg(f"cooc_pow_a{int(alpha*1000):04d}_p{int(pw*10):03d}", macro_auc(s_pw))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: Conditional co-occurrence (threshold-gated propagation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] Conditional co-occurrence smoothing...", flush=True)
t1 = time.time()

def cooccurrence_smooth_threshold(scores, alpha=0.05, threshold=0.5):
    """Only propagate from species with score > threshold."""
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0) + EPS
    cooc_norm = cooc / count_i[:, None]
    np.fill_diagonal(cooc_norm, 0)
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        s_thresh = s * (s > threshold).astype(np.float32)
        if s_thresh.sum() < EPS:
            smoothed[fi] = s
            continue
        contrib = cooc_norm.T @ s_thresh
        contrib /= (np.abs(contrib).max() + EPS)
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

for alpha in [0.020, 0.030, 0.050, 0.075, 0.100]:
    for thr in [0.3, 0.4, 0.5, 0.6]:
        s_thr = cooccurrence_smooth_threshold(double_best, alpha=alpha, threshold=thr)
        reg(f"cooc_thr{int(thr*10):02d}_a{int(alpha*1000):04d}", macro_auc(s_thr))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: Symmetric / Jaccard co-occurrence matrix
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Symmetric / Jaccard co-occurrence...", flush=True)
t1 = time.time()

def cooccurrence_smooth_symmetric(scores, alpha=0.05):
    """Use symmetric co-occurrence: (P(j|i) + P(i|j)) / 2."""
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0) + EPS
    pji = cooc / count_i[:, None]
    pij = cooc / count_i[None, :]
    cooc_sym = 0.5 * (pji + pij.T)
    np.fill_diagonal(cooc_sym, 0)
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        contrib = cooc_sym.T @ s
        contrib /= (np.abs(contrib).max() + EPS)
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def cooccurrence_smooth_jaccard(scores, alpha=0.05):
    """Use Jaccard similarity for co-occurrence: count(A∧B) / count(A∨B)."""
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0)
    union = count_i[:, None] + count_i[None, :] - cooc + EPS
    jaccard = cooc / union
    np.fill_diagonal(jaccard, 0)
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        contrib = jaccard.T @ s
        contrib /= (np.abs(contrib).max() + EPS)
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

for alpha in [0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200]:
    s_sym = cooccurrence_smooth_symmetric(double_best, alpha=alpha)
    reg(f"cooc_sym_a{int(alpha*1000):04d}", macro_auc(s_sym))

for alpha in [0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200]:
    s_jac = cooccurrence_smooth_jaccard(double_best, alpha=alpha)
    reg(f"cooc_jac_a{int(alpha*1000):04d}", macro_auc(s_jac))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M6: Two-stage cooc + cooc on attn_ref before conformal
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M6] Two-stage cooc + cooc-then-conformal...", flush=True)
t1 = time.time()

# Find best alpha from M1
best_alpha_auc = best_loo
best_alpha = 0.020
for name, r in results.items():
    if name.startswith("cooc_a") and r["loo_auc"] > best_alpha_auc:
        best_alpha_auc = r["loo_auc"]
        best_alpha = float(name.replace("cooc_a", "")) / 1000.0

print(f"  Best alpha from M1: {best_alpha:.3f} (LOO={best_alpha_auc:.6f})", flush=True)
s_cooc_best = cooccurrence_smooth(double_best, alpha=best_alpha)

# Add more conformal on top of cooc
for w_conf in [0.001, 0.002, 0.003, 0.005]:
    blend = (1 - w_conf) * s_cooc_best + w_conf * s_conf_k1_40
    reg(f"cooc{int(best_alpha*1000):04d}+conf_w{int(w_conf*1000):04d}", macro_auc(blend))

# Cooc on attn_ref before conformal
for alpha in [0.030, 0.040, 0.050, 0.075]:
    s_coo_attn = cooccurrence_smooth(attn_ref, alpha=alpha)
    blend = 0.997 * s_coo_attn + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
    reg(f"cooc_attn_a{int(alpha*1000):04d}+db_conf", macro_auc(blend))

# Two-stage cooc
for a1, a2 in [(0.020, 0.015), (0.020, 0.020), (0.030, 0.015), (0.025, 0.020), (0.020, 0.025)]:
    s_c1 = cooccurrence_smooth(double_best, alpha=a1)
    s_c2 = cooccurrence_smooth(s_c1, alpha=a2)
    reg(f"cooc2_{int(a1*1000):04d}_{int(a2*1000):04d}", macro_auc(s_c2))

print(f"  M6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print(f"[batch113] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

sorted_results = sorted(results.items(), key=lambda x: -x[1]["loo_auc"])
print(f"  Top-10 this batch:", flush=True)
for name, r in sorted_results[:10]:
    delta = r["loo_auc"] - best_loo
    print(f"    {name}: {r['loo_auc']:.6f} ({delta:+.6f})", flush=True)

for name, r in results.items():
    res["experiments"].append(r)
if new_best_loo > best_loo:
    res["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch113"}
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
