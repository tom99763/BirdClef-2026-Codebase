"""
Batch 114 — Fine-tune Threshold Co-occurrence Smoothing
===============================================================================
Current best: cooc_thr05_a0050 LOO=0.993679 (+0.001134 from batch113)
Key finding: threshold=0.5, alpha=0.050 is optimal — but we only sampled coarsely.

Strategy: Fine-tune every hyperparameter of threshold co-occurrence:
1. M1: Fine alpha sweep around 0.050 (0.035→0.075, step 0.005)
2. M2: Fine threshold sweep (0.40→0.55, step 0.02)
3. M3: Joint 2D sweep (threshold × alpha) around optimum
4. M4: Iterative threshold smoothing (apply K rounds)
5. M5: Soft threshold gate (sigmoid instead of hard step)
6. M6: Stack thr_cooc on top + additional conformal blend
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

print(f"[batch114] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch114] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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
# Threshold-gated co-occurrence smoothing (best from batch113)
# ─────────────────────────────────────────────────────────────────────────────
def _cooc_matrix():
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0) + EPS
    cooc_norm = cooc / count_i[:, None]
    np.fill_diagonal(cooc_norm, 0)
    return cooc_norm

COOC_NORM = _cooc_matrix()

def cooc_thr_smooth(scores, alpha=0.05, threshold=0.5):
    """Threshold-gated co-occurrence smoothing."""
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        s_thresh = s * (s > threshold).astype(np.float32)
        if s_thresh.sum() < EPS:
            smoothed[fi] = s
            continue
        contrib = COOC_NORM.T @ s_thresh
        contrib /= (np.abs(contrib).max() + EPS)
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def cooc_soft_thr_smooth(scores, alpha=0.05, thr_center=0.5, thr_slope=10.0):
    """Soft threshold gate (sigmoid) co-occurrence smoothing."""
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(-thr_slope * (s - thr_center)))
        s_gated = s * gate
        total = s_gated.sum()
        if total < EPS:
            smoothed[fi] = s
            continue
        contrib = COOC_NORM.T @ s_gated
        contrib /= (np.abs(contrib).max() + EPS)
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
    mark = " *** NEW BEST ***" if auc > best_loo else (" (near-best)" if auc > best_loo - 0.0005 else "")
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch114"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: Fine alpha sweep (threshold=0.5 fixed)
# batch113: thr05_a030=+0.000858, thr05_a050=+0.001134, thr05_a075=+0.001032
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] Fine alpha sweep (threshold=0.5)...", flush=True)
t1 = time.time()
for alpha in np.arange(0.030, 0.090, 0.005):
    s = cooc_thr_smooth(double_best, alpha=float(alpha), threshold=0.5)
    reg(f"thr05_a{int(round(alpha*1000)):04d}", macro_auc(s))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: Fine threshold sweep (alpha=0.050 fixed)
# batch113: thr03=+0.000011, thr04=+0.000754, thr05=+0.001134, thr06=-0.000924
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Fine threshold sweep (alpha=0.050)...", flush=True)
t1 = time.time()
for thr in np.arange(0.38, 0.62, 0.02):
    s = cooc_thr_smooth(double_best, alpha=0.050, threshold=float(thr))
    reg(f"thr{int(round(thr*100)):03d}_a0050", macro_auc(s))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: Joint 2D sweep (threshold × alpha) fine grid around optimum
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Joint threshold × alpha sweep...", flush=True)
t1 = time.time()
for thr in [0.44, 0.46, 0.48, 0.50, 0.52, 0.54]:
    for alpha in [0.040, 0.045, 0.050, 0.055, 0.060, 0.065]:
        s = cooc_thr_smooth(double_best, alpha=float(alpha), threshold=float(thr))
        reg(f"thr{int(round(thr*100)):03d}_a{int(round(alpha*1000)):04d}", macro_auc(s))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: Iterative threshold smoothing (K rounds with smaller step)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] Iterative threshold smoothing...", flush=True)
t1 = time.time()

def cooc_thr_smooth_iter(scores, alpha=0.02, threshold=0.5, n_iter=3):
    s = scores.copy()
    for _ in range(n_iter):
        s_new = np.zeros_like(s)
        for fi in range(n_files):
            sv = s[fi]
            s_thresh = sv * (sv > threshold).astype(np.float32)
            if s_thresh.sum() < EPS:
                s_new[fi] = sv
                continue
            contrib = COOC_NORM.T @ s_thresh
            contrib /= (np.abs(contrib).max() + EPS)
            s_new[fi] = (1 - alpha) * sv + alpha * np.clip(contrib, 0, None)
        s = s_new
    return s

for n_iter in [2, 3, 5]:
    for alpha_step in [0.015, 0.020, 0.025, 0.030]:
        s_iter = cooc_thr_smooth_iter(double_best, alpha=float(alpha_step), threshold=0.5, n_iter=n_iter)
        reg(f"thr05_iter{n_iter}_a{int(round(alpha_step*1000)):04d}", macro_auc(s_iter))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: Soft threshold gate (sigmoid instead of hard step)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Soft threshold gate (sigmoid)...", flush=True)
t1 = time.time()
for alpha in [0.040, 0.050, 0.060, 0.075]:
    for slope in [5.0, 8.0, 12.0, 20.0]:
        for center in [0.45, 0.50, 0.55]:
            s = cooc_soft_thr_smooth(double_best, alpha=float(alpha), thr_center=float(center), thr_slope=float(slope))
            reg(f"soft_c{int(center*100):03d}_sl{int(slope):02d}_a{int(alpha*1000):04d}", macro_auc(s))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M6: Stack thr_cooc on top + additional conformal blend
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M6] Best thr_cooc + conformal stacking...", flush=True)
t1 = time.time()

# Find the best from M1-M3
best_thr_auc = best_loo
best_thr_name = "cooc_thr05_a0050"
for name, r in results.items():
    if r["loo_auc"] > best_thr_auc:
        best_thr_auc = r["loo_auc"]
        best_thr_name = name

# Find the best threshold/alpha to use
best_thr_val = 0.50
best_alpha_val = 0.050
if best_thr_name in results:
    nm = best_thr_name
    if nm.startswith("thr") and "_a" in nm:
        parts = nm.split("_a")
        best_thr_val = float(parts[0].replace("thr", "")) / 100.0
        best_alpha_val = float(parts[1]) / 1000.0

print(f"  Best thr_cooc: {best_thr_name} (thr={best_thr_val:.2f}, alpha={best_alpha_val:.3f}) LOO={best_thr_auc:.6f}", flush=True)
s_best_cooc = cooc_thr_smooth(double_best, alpha=best_alpha_val, threshold=best_thr_val)

# Add additional conformal on top of cooc
for w_conf in [0.001, 0.002, 0.003]:
    blend = (1 - w_conf) * s_best_cooc + w_conf * s_conf_k1_40
    reg(f"best_cooc+conf_w{int(w_conf*1000):04d}", macro_auc(blend))

# Apply cooc before conformal (on attn_ref)
for thr, alpha in [(0.48, 0.050), (0.50, 0.050), (0.50, 0.055)]:
    s_coo_attn = cooc_thr_smooth(attn_ref, alpha=float(alpha), threshold=float(thr))
    blend = 0.997 * s_coo_attn + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
    reg(f"cooc_attn_t{int(thr*100):03d}_a{int(alpha*1000):04d}+conf", macro_auc(blend))

# Two-stage: cooc on double_best, then another round
for thr2, alpha2 in [(0.50, 0.020), (0.50, 0.030), (0.50, 0.040)]:
    s_c2 = cooc_thr_smooth(s_best_cooc, alpha=float(alpha2), threshold=float(thr2))
    reg(f"cooc2_best_t{int(thr2*100):03d}_a{int(alpha2*1000):04d}", macro_auc(s_c2))

print(f"  M6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print(f"[batch114] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

sorted_results = sorted(results.items(), key=lambda x: -x[1]["loo_auc"])
print(f"  Top-10 this batch:", flush=True)
for name, r in sorted_results[:10]:
    delta = r["loo_auc"] - best_loo
    print(f"    {name}: {r['loo_auc']:.6f} ({delta:+.6f})", flush=True)

for name, r in results.items():
    res["experiments"].append(r)

if new_best_loo > best_loo:
    res["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch114"}
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
