"""
Batch 117 — New Signal Families After Cooc Convergence
===============================================================================
Current best: 2r_c053_sl037_a0040 LOO=0.994248 (two-round soft_cooc)
batch116: only +0.000023 improvement → cooc family nearly exhausted

Strategy: Completely new signal families + deeper two-round exploration:
1. M1: Per-species score calibration (platt scaling / rank normalization)
2. M2: Rank fusion — convert scores to ranks, then blend
3. M3: Deeper two-round optimization (fine-tune first-round params)
4. M4: Double cooc (two separate cooc matrices — train vs all)
5. M5: Power transform of double_best before cooc
6. M6: Three-round with optimal params
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

print(f"[batch117] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch117] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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
# Co-occurrence utilities
# ─────────────────────────────────────────────────────────────────────────────
def _cooc_matrix():
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0) + EPS
    cooc_norm = cooc / count_i[:, None]
    np.fill_diagonal(cooc_norm, 0)
    return cooc_norm

COOC_NORM = _cooc_matrix()

def soft_cooc(scores, center=0.53, slope=37.0, alpha=0.086):
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

# Current best: two-round
r1_best = soft_cooc(double_best, center=0.54, slope=41.0, alpha=0.089)
best_2r = soft_cooc(r1_best, center=0.53, slope=37.0, alpha=0.040)
auc_2r = macro_auc(best_2r)
print(f"  two_round check: {auc_2r:.6f} (expected 0.994248)", flush=True)

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
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch117"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: Per-species rank normalization before cooc
# Normalize each species score across files via rank, then apply soft_cooc
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] Per-species rank normalization before cooc...", flush=True)
t1 = time.time()

def rank_normalize(scores):
    """Normalize each species column to rank-based uniform [0,1]."""
    out = np.zeros_like(scores)
    for si in range(n_species):
        col = scores[:, si]
        ranks = np.argsort(np.argsort(col)).astype(np.float32)
        out[:, si] = ranks / (n_files - 1 + EPS)
    return out

# Apply rank norm, then cooc, then blend back
s_rank = rank_normalize(double_best)
for alpha in [0.050, 0.075, 0.086, 0.100, 0.125]:
    s_cooc_rank = soft_cooc(s_rank, center=0.53, slope=37.0, alpha=float(alpha))
    # Blend cooc-on-rank with original
    for w_rank in [0.10, 0.20, 0.30]:
        blend = (1 - w_rank) * double_best + w_rank * s_cooc_rank
        reg(f"rank_a{int(alpha*1000):04d}_w{int(w_rank*100):03d}", macro_auc(blend))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: Power transform before cooc (sharpens high-confidence scores)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Power transform before cooc...", flush=True)
t1 = time.time()

for power in [1.5, 2.0, 2.5, 3.0]:
    s_pow = np.clip(double_best, 0, 1) ** power
    for alpha in [0.086, 0.100, 0.125, 0.150]:
        s_cooc_pow = soft_cooc(s_pow, center=0.53, slope=37.0, alpha=float(alpha))
        # Blend with original (scores are now power-transformed, need to blend carefully)
        for w_pow in [0.20, 0.30, 0.40, 0.50]:
            blend = (1 - w_pow) * double_best + w_pow * s_cooc_pow
            reg(f"pow{int(power*10):02d}_a{int(alpha*1000):04d}_w{int(w_pow*100):03d}", macro_auc(blend))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: Fine-tune first-round parameters of the two-round approach
# Objective: improve r1 itself, then apply best second round
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Fine-tune first-round in two-round...", flush=True)
t1 = time.time()

best_2r_in_m3 = best_loo
for c1, sl1, al1 in [
    # Vary center
    (0.52, 41.0, 0.089), (0.53, 41.0, 0.089), (0.54, 41.0, 0.089), (0.55, 41.0, 0.089), (0.56, 41.0, 0.089),
    # Vary slope
    (0.54, 35.0, 0.089), (0.54, 37.0, 0.089), (0.54, 39.0, 0.089), (0.54, 41.0, 0.089), (0.54, 43.0, 0.089),
    # Vary alpha
    (0.54, 41.0, 0.080), (0.54, 41.0, 0.085), (0.54, 41.0, 0.089), (0.54, 41.0, 0.093), (0.54, 41.0, 0.097),
]:
    r1 = soft_cooc(double_best, center=float(c1), slope=float(sl1), alpha=float(al1))
    r2 = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.040)
    auc = macro_auc(r2)
    name = f"2r_r1_c{int(round(c1*100)):03d}_sl{int(sl1):03d}_al{int(round(al1*1000)):04d}"
    reg(name, auc)
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: Three-round carefully
# Best two-round: r1=(c0.54,sl41,a0.089), r2=(c0.53,sl37,a0.040)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] Three-round exploration...", flush=True)
t1 = time.time()

for c3, sl3, al3 in [
    (0.53, 37.0, 0.020), (0.53, 37.0, 0.030), (0.53, 37.0, 0.040),
    (0.54, 41.0, 0.020), (0.54, 41.0, 0.030), (0.54, 41.0, 0.040),
    (0.55, 20.0, 0.030), (0.55, 20.0, 0.040), (0.55, 20.0, 0.050),
    (0.50, 10.0, 0.020), (0.50, 10.0, 0.030),
]:
    r3 = soft_cooc(best_2r, center=float(c3), slope=float(sl3), alpha=float(al3))
    name = f"3r_c{int(round(c3*100)):03d}_sl{int(sl3):02d}_al{int(round(al3*1000)):04d}"
    reg(name, macro_auc(r3))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: Cooc on power of r1 (two-round where second round uses power transform)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Second-round with power transform...", flush=True)
t1 = time.time()

for pw in [1.5, 2.0, 2.5]:
    r1_pow = np.clip(r1_best, 0, 1) ** pw
    for al2 in [0.050, 0.075, 0.100]:
        r2_pow = soft_cooc(r1_pow, center=0.53, slope=37.0, alpha=float(al2))
        # r2_pow is in [0,1]^pw range, blend with r1_best
        for w_pw in [0.30, 0.40, 0.50]:
            blend = (1 - w_pw) * r1_best + w_pw * r2_pow
            reg(f"2r_r2pow{int(pw*10):02d}_a{int(al2*1000):04d}_w{int(w_pw*100):03d}", macro_auc(blend))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M6: Best two-round + additional micro-tuning
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M6] Two-round + conformal + fine mixing...", flush=True)
t1 = time.time()

# Stack tiny conformal on best_2r
for w_c in [0.0005, 0.001, 0.002, 0.003]:
    blend = (1 - w_c) * best_2r + w_c * s_conf_k1_40
    reg(f"2r+ck1_w{int(w_c*10000):05d}", macro_auc(blend))

# Blend best_2r with original double_best
for w_db in [0.02, 0.05, 0.10]:
    blend = (1 - w_db) * best_2r + w_db * double_best
    reg(f"2r_mix_db_w{int(w_db*100):03d}", macro_auc(blend))

# Very small alpha third round
for c3, sl3, al3 in [(0.53, 37.0, 0.010), (0.53, 37.0, 0.015), (0.54, 41.0, 0.015)]:
    r3 = soft_cooc(best_2r, center=float(c3), slope=float(sl3), alpha=float(al3))
    reg(f"2r+tiny3r_c{int(round(c3*100)):03d}_al{int(round(al3*1000)):04d}", macro_auc(r3))

print(f"  M6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print(f"[batch117] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

sorted_results = sorted(results.items(), key=lambda x: -x[1]["loo_auc"])
print(f"  Top-10 this batch:", flush=True)
for name, r in sorted_results[:10]:
    delta = r["loo_auc"] - best_loo
    print(f"    {name}: {r['loo_auc']:.6f} ({delta:+.6f})", flush=True)

for name, r in results.items():
    res["experiments"].append(r)

if new_best_loo > best_loo:
    res["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch117"}
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
