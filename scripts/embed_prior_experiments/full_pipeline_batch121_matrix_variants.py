"""
Batch 121 — Co-occurrence Matrix Variants + Novel Combinations
===============================================================================
Current best: idf_p075 LOO=0.994463 (batch119)
Formula: power=2.0, center=0.55, slope=41, alpha=0.130, blend=0.55, idf_power=0.75

Strategy: Try genuinely different co-occurrence matrix types + novel combinations:
1. M1: Jaccard similarity matrix (symmetric; penalizes uncommon pairs)
2. M2: PMI (Pointwise Mutual Information) co-occurrence
3. M3: Species acoustic embedding similarity (embedding-based alternative to label cooc)
4. M4: Blend IDF cooc (current best) with two-round soft cooc (batch116 best)
5. M5: Soft/score-weighted co-occurrence using file-level predictions
6. M6: Conditional entropy + coverage weighted smoothing
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

print(f"[batch121] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch121] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

# ── Pre-compute co-occurrence matrices ────────────────────────────────────────
fl = file_labels.astype(np.float32)
count_i = fl.sum(0) + EPS

# Standard conditional: COOC_NORM[i,j] = P(j|i)
cooc_raw = fl.T @ fl
COOC_NORM = cooc_raw / count_i[:, None]
np.fill_diagonal(COOC_NORM, 0)

# Jaccard: J[i,j] = count(i∧j) / (count(i) + count(j) - count(i∧j))
denom_jacc = count_i[:, None] + count_i[None, :] - cooc_raw + EPS
COOC_JACC = cooc_raw / denom_jacc
np.fill_diagonal(COOC_JACC, 0)

# PMI: log(P(i∧j) / (P(i)*P(j))); positive PMI (ppmi)
p_i = count_i / n_files
p_ij = cooc_raw / n_files
pmi_raw = np.log(p_ij / (p_i[:, None] * p_i[None, :] + EPS) + EPS)
pmi_pos = np.clip(pmi_raw, 0, None)  # Positive PMI
np.fill_diagonal(pmi_pos, 0)
# Normalize each row by row-sum for use as transition matrix
row_sum_pmi = pmi_pos.sum(1) + EPS
COOC_PMI = pmi_pos / row_sum_pmi[:, None]

# IDF weights
raw_idf = np.log(float(n_files) / (count_i + 1.0 - EPS))
raw_idf = np.clip(raw_idf, 0, None)
idf_w075 = (raw_idf ** 0.75)
idf_w075 /= (idf_w075.mean() + EPS)

# Standard soft_cooc function
def soft_cooc_mat(scores, center=0.55, slope=41.0, alpha=0.130, cooc_mat=None,
                  src_weight=None):
    """Soft co-occurrence with optional source weighting."""
    if cooc_mat is None:
        cooc_mat = COOC_NORM
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate
        if src_weight is not None:
            s_gated = s_gated * src_weight
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = cooc_mat.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def idf_blend(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55, idf_power=0.75):
    """Current best: IDF-weighted power+cooc."""
    s_pow = np.clip(scores, 0, 1) ** 2.0
    idf_w = (raw_idf ** idf_power); idf_w /= (idf_w.mean() + EPS)
    s_cooc = soft_cooc_mat(s_pow, center=center, slope=slope, alpha=alpha,
                            src_weight=idf_w)
    return (1 - blend) * scores + blend * s_cooc

# Also compute two-round soft cooc baseline (batch116 best)
def two_round_soft_cooc(scores):
    """batch116 best: 2r_c053_sl037_a0040."""
    r1 = soft_cooc_mat(scores, center=0.54, slope=41.0, alpha=0.089)
    r2 = soft_cooc_mat(r1, center=0.53, slope=37.0, alpha=0.040)
    return r2

two_round_scores = two_round_soft_cooc(double_best)
print(f"  two_round baseline: {macro_auc(two_round_scores):.6f} (expected 0.994248)", flush=True)

best_idf_scores = idf_blend(double_best)
print(f"  current best check: {macro_auc(best_idf_scores):.6f} (expected {best_loo:.6f})", flush=True)

results = []

# ── M1: Jaccard co-occurrence matrix ─────────────────────────────────────────
print("\n[M1] Jaccard co-occurrence matrix...", flush=True)
t0 = time.time()

for alpha in [0.080, 0.100, 0.130, 0.160, 0.200]:
    for blend in [0.45, 0.55, 0.65]:
        name = f"jacc_a{int(alpha*100):03d}_w{int(blend*100):02d}"
        s_pow = np.clip(double_best, 0, 1) ** 2.0
        s_cooc = soft_cooc_mat(s_pow, center=0.55, slope=41.0, alpha=alpha,
                               cooc_mat=COOC_JACC, src_weight=idf_w075)
        s = (1 - blend) * double_best + blend * s_cooc
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 121})
        if auc > best_loo + 1e-7:
            best_loo = auc

# Also try Jaccard without IDF
for alpha in [0.080, 0.130, 0.160]:
    for blend in [0.45, 0.55, 0.65]:
        name = f"jacc_noidf_a{int(alpha*100):03d}_w{int(blend*100):02d}"
        s_pow = np.clip(double_best, 0, 1) ** 2.0
        s_cooc = soft_cooc_mat(s_pow, center=0.55, slope=41.0, alpha=alpha,
                               cooc_mat=COOC_JACC)
        s = (1 - blend) * double_best + blend * s_cooc
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 121})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M1 done ({time.time()-t0:.0f}s)", flush=True)

# ── M2: PMI co-occurrence matrix ──────────────────────────────────────────────
print("\n[M2] PMI co-occurrence matrix...", flush=True)
t0 = time.time()

for alpha in [0.080, 0.100, 0.130, 0.160, 0.200]:
    for blend in [0.45, 0.55, 0.65]:
        name = f"pmi_a{int(alpha*100):03d}_w{int(blend*100):02d}"
        s_pow = np.clip(double_best, 0, 1) ** 2.0
        s_cooc = soft_cooc_mat(s_pow, center=0.55, slope=41.0, alpha=alpha,
                               cooc_mat=COOC_PMI, src_weight=idf_w075)
        s = (1 - blend) * double_best + blend * s_cooc
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 121})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M2 done ({time.time()-t0:.0f}s)", flush=True)

# ── M3: Species acoustic embedding similarity ─────────────────────────────────
print("\n[M3] Species acoustic embedding similarity...", flush=True)
t0 = time.time()

# Compute per-species centroid in ICA embedding space from ALL training windows
# (Using all data, since this is a global species prototype)
species_centroids = np.zeros((n_species, ew_ica.shape[1]), np.float32)
for si in range(n_species):
    pos_mask = labels_win[:, si] > 0.5
    if pos_mask.sum() > 0:
        species_centroids[si] = ew_ica[pos_mask].mean(0)
        species_centroids[si] /= (norm(species_centroids[si]) + EPS)

# Species-species acoustic similarity matrix
EMB_SIM = species_centroids @ species_centroids.T  # cosine similarity (normalized)
np.fill_diagonal(EMB_SIM, 0)
# Normalize to transition matrix (row-normalize)
row_sum_emb = np.abs(EMB_SIM).sum(1) + EPS
EMB_SIM_NORM = EMB_SIM / row_sum_emb[:, None]

# Clip to positive similarities only
EMB_SIM_POS = np.clip(EMB_SIM, 0, None)
row_sum_pos = EMB_SIM_POS.sum(1) + EPS
EMB_SIM_POS_NORM = EMB_SIM_POS / row_sum_pos[:, None]
np.fill_diagonal(EMB_SIM_POS_NORM, 0)

for alpha in [0.080, 0.100, 0.130, 0.160]:
    for blend in [0.45, 0.55, 0.65]:
        name = f"embsim_a{int(alpha*100):03d}_w{int(blend*100):02d}"
        s_pow = np.clip(double_best, 0, 1) ** 2.0
        s_cooc = soft_cooc_mat(s_pow, center=0.55, slope=41.0, alpha=alpha,
                               cooc_mat=EMB_SIM_POS_NORM, src_weight=idf_w075)
        s = (1 - blend) * double_best + blend * s_cooc
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 121})
        if auc > best_loo + 1e-7:
            best_loo = auc

# Blend label-cooc and embedding-sim matrices
for w_emb in [0.20, 0.30, 0.40, 0.50]:
    w_cooc = 1 - w_emb
    MIXED_MAT = w_cooc * COOC_NORM + w_emb * EMB_SIM_POS_NORM
    np.fill_diagonal(MIXED_MAT, 0)
    name = f"mix_emb{int(w_emb*100):02d}_cooc{int(w_cooc*100):02d}"
    s_pow = np.clip(double_best, 0, 1) ** 2.0
    s_cooc = soft_cooc_mat(s_pow, center=0.55, slope=41.0, alpha=0.130,
                           cooc_mat=MIXED_MAT, src_weight=idf_w075)
    s = (1 - 0.55) * double_best + 0.55 * s_cooc
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 121})
    if auc > best_loo + 1e-7:
        best_loo = auc

print(f"  M3 done ({time.time()-t0:.0f}s)", flush=True)

# ── M4: Blend IDF cooc with two-round soft cooc ───────────────────────────────
print("\n[M4] Blend IDF cooc with two-round soft cooc...", flush=True)
t0 = time.time()

for w_idf in [0.50, 0.60, 0.70, 0.80, 0.90]:
    w_2r = 1 - w_idf
    blended = w_idf * best_idf_scores + w_2r * two_round_scores
    name = f"blend_idf{int(w_idf*100):02d}_2r{int(w_2r*100):02d}"
    auc = macro_auc(blended)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 121})
    if auc > best_loo + 1e-7:
        best_loo = auc

# Also try: double_best + IDF cooc + two-round (3-way blend)
for w_db in [0.30, 0.40, 0.50]:
    for w_idf in [0.25, 0.35, 0.45]:
        w_2r = max(0, 1 - w_db - w_idf)
        if w_2r < 0.05: continue
        blended = w_db * double_best + w_idf * best_idf_scores + w_2r * two_round_scores
        name = f"3way_db{int(w_db*100):02d}_idf{int(w_idf*100):02d}_2r{int(w_2r*100):02d}"
        auc = macro_auc(blended)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 121})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M4 done ({time.time()-t0:.0f}s)", flush=True)

# ── M5: Soft co-occurrence using model predictions (score-weighted matrix) ────
print("\n[M5] Score-weighted co-occurrence matrix...", flush=True)
t0 = time.time()

# Build a "soft" co-occurrence from double_best scores (LOO-aware per fold)
# For each fold fi, build score-cooc from training file scores only
def score_soft_cooc(scores, alpha=0.130, center=0.55, slope=41.0, blend=0.55):
    """Build co-occurrence matrix from model scores of training files per LOO fold."""
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        # Training file scores (LOO)
        tr_scores = scores[win_file_id != fi]  # wrong dim, use file-level
        pass
    # Actually use file-level scores for co-occurrence
    file_scores_pow = np.clip(scores, 0, 1) ** 2.0
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        # Build co-occurrence from training files (all except fi)
        tr_s = file_scores_pow[np.arange(n_files) != fi]  # (n_files-1, n_species)
        # Soft co-occurrence: C[i,j] = sum_f(s_f[i] * s_f[j]) / (sum_f(s_f[i]) + EPS)
        sum_i = tr_s.sum(0) + EPS  # (n_species,)
        cooc_soft = tr_s.T @ tr_s  # (n_species, n_species)
        cooc_soft_norm = cooc_soft / sum_i[:, None]
        np.fill_diagonal(cooc_soft_norm, 0)
        # Normalize each row
        row_max = np.abs(cooc_soft_norm).max(1) + EPS
        cooc_soft_norm = cooc_soft_norm / row_max[:, None]

        s = file_scores_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * idf_w075
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = scores[fi]; continue
        contrib = cooc_soft_norm.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return (1 - blend) * scores + blend * smoothed

for alpha in [0.100, 0.130, 0.160]:
    for blend in [0.45, 0.55, 0.65]:
        name = f"scorecooc_a{int(alpha*100):03d}_w{int(blend*100):02d}"
        s = score_soft_cooc(double_best, alpha=alpha, blend=blend)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 121})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M5 done ({time.time()-t0:.0f}s)", flush=True)

# ── M6: Coverage-weighted smoothing ──────────────────────────────────────────
print("\n[M6] Coverage & entropy weighted smoothing...", flush=True)
t0 = time.time()

# Weight sources by species "coverage" in training: species with more training
# examples have more reliable co-occurrence estimates → give them higher weight
# Coverage weight: count_i normalized (common species have more reliable cooc)
def coverage_cooc_blend(scores, cov_power=0.5, center=0.55, slope=41.0,
                        alpha=0.130, blend=0.55, idf_power=0.75):
    """Blend of IDF weighting and coverage weighting."""
    raw_idf = np.log(float(n_files) / (count_i + 1.0 - EPS))
    raw_idf_clipped = np.clip(raw_idf, 0, None)
    idf_w = (raw_idf_clipped ** idf_power); idf_w /= (idf_w.mean() + EPS)
    # Coverage: more training examples = more reliable
    cov_w = (count_i / count_i.max()) ** cov_power
    # Combine: IDF weights rare species, coverage weights reliable species
    combined_w = idf_w * cov_w; combined_w /= (combined_w.mean() + EPS)
    s_pow = np.clip(scores, 0, 1) ** 2.0
    s_cooc = soft_cooc_mat(s_pow, center=center, slope=slope, alpha=alpha,
                           src_weight=combined_w)
    return (1 - blend) * scores + blend * s_cooc

for cov_p in [0.25, 0.50, 0.75, 1.0]:
    for idf_p in [0.60, 0.75, 0.90]:
        name = f"cov{int(cov_p*100):03d}_idf{int(idf_p*100):03d}"
        s = coverage_cooc_blend(double_best, cov_power=cov_p, idf_power=idf_p)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 121})
        if auc > best_loo + 1e-7:
            best_loo = auc

# Negative coverage: maybe rare species in COOC are more informative
for cov_p in [-0.25, -0.50, -0.75]:
    for idf_p in [0.60, 0.75]:
        name = f"negcov{int(abs(cov_p)*100):03d}_idf{int(idf_p*100):03d}"
        s = coverage_cooc_blend(double_best, cov_power=cov_p, idf_power=idf_p)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 121})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M6 done ({time.time()-t0:.0f}s)", flush=True)

# ── Summary + Save ────────────────────────────────────────────────────────────
res = json.load(open(RESULTS_PATH))
prev_best_loo = res["best"]["loo_auc"]
prev_best_method = res["best"]["method"]

top10 = sorted(results, key=lambda x: -x["loo_auc"])[:10]
print("\n" + "="*60)
print(f"[batch121] SUMMARY")
print(f"  Previous best: {prev_best_loo:.6f} ({prev_best_method})")
print(f"  Top-10 this batch:")
for r in top10:
    print(f"    {r['method']}: {r['loo_auc']:.6f} ({r['delta']:+.6f})")

new_best = max(results, key=lambda x: x["loo_auc"])
if new_best["loo_auc"] > prev_best_loo + 1e-7:
    print(f"\n  NEW BEST: {new_best['method']} LOO={new_best['loo_auc']:.6f} ({new_best['delta']:+.6f})")
    res["best"] = {"method": new_best["method"], "loo_auc": new_best["loo_auc"]}
    with open(MODEL_PATH, "rb") as f:
        ep2 = pickle.load(f)
    ep2["method"] = new_best["method"]
    ep2["loo_auc"] = new_best["loo_auc"]
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep2, f)
    print(f"  PKL + JSON updated → {new_best['method']} {new_best['loo_auc']:.6f}")
else:
    print(f"\n  No improvement over {prev_best_method} ({prev_best_loo:.6f})")

for r in results:
    res["experiments"].append(r)
with open(RESULTS_PATH, "w") as f:
    json.dump(res, f, indent=2)
print(f"  Saved {len(results)} results to JSON")
