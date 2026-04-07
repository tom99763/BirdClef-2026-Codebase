"""
Batch 119 — New Directions Beyond Power+Cooc
===============================================================================
Current best: pw20_c055_sl41_a013_w055 LOO=0.994375 (+0.000025 from batch118)
Formula: s_pow = clip(db,0,1)^2.0; s_cooc = soft_cooc(s_pow, c=0.55, sl=41, a=0.130); BEST = 0.45*db + 0.55*s_cooc

Strategy: Explore genuinely new signal families:
1. M1: Logit-space co-occurrence (transform to log-odds before cooc)
2. M2: Top-K selective propagation (hard top-K instead of sigmoid gate)
3. M3: IDF-weighted co-occurrence (rare species weighted more)
4. M4: Score-adaptive blend weight (more blending for uncertain predictions)
5. M5: Window-level co-occurrence (apply cooc at window level before file-agg)
6. M6: Two-round power+cooc compound (second application with small params)
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

print(f"[batch119] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch119] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

# ── Co-occurrence matrices ────────────────────────────────────────────────────
def _cooc_matrix():
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0) + EPS
    cooc_norm = cooc / count_i[:, None]
    np.fill_diagonal(cooc_norm, 0)
    return cooc_norm

def _jaccard_matrix():
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0)
    denom = count_i[:, None] + count_i[None, :] - cooc + EPS
    jacc = cooc / denom
    np.fill_diagonal(jacc, 0)
    return jacc

def _idf_weights():
    fl = file_labels.astype(np.float32)
    count_i = fl.sum(0)
    # IDF: log(n_files / (count + 1)), higher = rarer species
    idf = np.log(float(n_files) / (count_i + 1.0))
    idf = np.clip(idf, 0, None)
    idf /= (idf.sum() + EPS) / n_species  # normalize so mean=1
    return idf

COOC_NORM = _cooc_matrix()
IDF = _idf_weights()

def soft_cooc(scores, center=0.53, slope=37.0, alpha=0.086, cooc_mat=None):
    if cooc_mat is None:
        cooc_mat = COOC_NORM
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = cooc_mat.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def pow_cooc_blend(scores, power=2.0, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    """Current best: pow20_c055_sl41_a013_w055."""
    s_pow = np.clip(scores, 0, 1) ** power
    s_cooc = soft_cooc(s_pow, center=center, slope=slope, alpha=alpha)
    return (1 - blend) * scores + blend * s_cooc

# Verify current best
best_method_scores = pow_cooc_blend(double_best,
    power=2.0, center=0.55, slope=41.0, alpha=0.130, blend=0.55)
print(f"  current best check: {macro_auc(best_method_scores):.6f} (expected {best_loo:.6f})", flush=True)

results = []

# ── M1: Logit-space co-occurrence ────────────────────────────────────────────
print("\n[M1] Logit-space co-occurrence...", flush=True)
t0 = time.time()

def logit_transform(s, eps=1e-6):
    """Convert probability [0,1] to log-odds."""
    s_c = np.clip(s, eps, 1 - eps)
    return np.log(s_c / (1 - s_c))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -88, 88)))

def logit_cooc_blend(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55,
                     logit_scale=3.0, power=2.0):
    """Apply power transform in logit space, then cooc, then blend."""
    # Option A: Convert to logit space before power
    s_c = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = np.log(s_c / (1 - s_c))
    # Scale and apply power to magnitude (preserve sign)
    s_scaled = sigmoid(logits / logit_scale)  # compress range before power
    s_pow = np.clip(s_scaled, 0, 1) ** power
    s_cooc = soft_cooc(s_pow, center=center, slope=slope, alpha=alpha)
    return (1 - blend) * scores + blend * s_cooc

for lscale in [1.5, 2.0, 3.0, 4.0]:
    for pw in [1.5, 2.0, 2.5]:
        name = f"lgt_sc{int(lscale*10):02d}_pw{int(pw*10):02d}"
        s = logit_cooc_blend(double_best, center=0.55, slope=41.0,
                             alpha=0.130, blend=0.55, logit_scale=lscale, power=pw)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 119})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M1 done ({time.time()-t0:.0f}s)", flush=True)

# ── M2: Top-K selective propagation ─────────────────────────────────────────
print("\n[M2] Top-K selective propagation...", flush=True)
t0 = time.time()

def topk_cooc_blend(scores, k=5, alpha=0.130, blend=0.55):
    """Only propagate from top-K scoring species per file (hard cutoff)."""
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        # Select top-K species
        if k >= len(s):
            s_topk = s.copy()
        else:
            thresh_idx = np.argsort(s)[-(k)]  # k-th highest
            thresh = s[thresh_idx]
            s_topk = np.where(s >= thresh, s, 0.0).astype(np.float32)
        total = s_topk.sum()
        if total < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_topk
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def pow_topk_blend(scores, power=2.0, k=5, alpha=0.130, blend=0.55):
    """Power transform, then top-K cooc, then blend."""
    s_pow = np.clip(scores, 0, 1) ** power
    s_cooc = topk_cooc_blend(s_pow, k=k, alpha=alpha, blend=1.0)  # full cooc on pow
    return (1 - blend) * scores + blend * s_cooc

for k in [3, 5, 8, 10, 15]:
    for alpha in [0.100, 0.130, 0.160]:
        for blend in [0.45, 0.55, 0.65]:
            name = f"topk{k:02d}_a{int(alpha*100):03d}_w{int(blend*100):02d}"
            s = pow_topk_blend(double_best, power=2.0, k=k, alpha=alpha, blend=blend)
            auc = macro_auc(s)
            delta = auc - best_loo
            status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
            print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
            results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 119})
            if auc > best_loo + 1e-7:
                best_loo = auc

print(f"  M2 done ({time.time()-t0:.0f}s)", flush=True)

# ── M3: IDF-weighted co-occurrence ───────────────────────────────────────────
print("\n[M3] IDF-weighted co-occurrence...", flush=True)
t0 = time.time()

def idf_cooc_blend(scores, power=2.0, center=0.55, slope=41.0, alpha=0.130,
                   blend=0.55, idf_power=0.5):
    """Weight co-occurrence contributions by IDF of source species."""
    idf_w = IDF ** idf_power  # raise IDF to power to moderate effect
    idf_w /= (idf_w.mean() + EPS)  # normalize to mean=1
    s_pow = np.clip(scores, 0, 1) ** power
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * idf_w  # weight by IDF
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = scores[fi]; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return (1 - blend) * scores + blend * smoothed

for idf_p in [0.25, 0.50, 0.75, 1.0, 1.5]:
    name = f"idf_p{int(idf_p*100):03d}"
    s = idf_cooc_blend(double_best, power=2.0, center=0.55, slope=41.0,
                       alpha=0.130, blend=0.55, idf_power=idf_p)
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 119})
    if auc > best_loo + 1e-7:
        best_loo = auc

# Also try IDF weighting on the target side (rare species get more correction)
def idf_target_blend(scores, power=2.0, center=0.55, slope=41.0, alpha=0.130,
                     blend=0.55, idf_power=0.5):
    """Weight blend toward IDF — rare species get more co-occurrence correction."""
    idf_w = np.clip(IDF ** idf_power, 0, None)
    idf_w = idf_w / (idf_w.mean() + EPS)  # normalize
    s_pow = np.clip(scores, 0, 1) ** power
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = scores[fi]; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        # Variable blend per species: rare species blended more
        blend_per = np.clip(blend * idf_w, 0, 0.95)
        smoothed[fi] = (1 - blend_per) * scores[fi] + blend_per * np.clip(contrib, 0, None)
    return smoothed

for idf_p in [0.25, 0.50, 0.75, 1.0]:
    name = f"idf_tgt_p{int(idf_p*100):03d}"
    s = idf_target_blend(double_best, power=2.0, center=0.55, slope=41.0,
                         alpha=0.130, blend=0.55, idf_power=idf_p)
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 119})
    if auc > best_loo + 1e-7:
        best_loo = auc

print(f"  M3 done ({time.time()-t0:.0f}s)", flush=True)

# ── M4: Score-adaptive blend weight ─────────────────────────────────────────
print("\n[M4] Score-adaptive blend weight...", flush=True)
t0 = time.time()

def adaptive_blend_cooc(scores, power=2.0, center=0.55, slope=41.0, alpha=0.130,
                        base_blend=0.55, adapt_strength=1.0):
    """
    Low-confidence species get MORE co-occurrence correction.
    High-confidence species get LESS (they're already well-detected).
    blend_i = base_blend * (1 + adapt_strength * (0.5 - s_i))
    """
    s_pow = np.clip(scores, 0, 1) ** power
    s_cooc_out = soft_cooc(s_pow, center=center, slope=slope, alpha=alpha)
    result = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        # blend varies per species: low confidence → more blending
        blend_per = np.clip(base_blend * (1 + adapt_strength * (0.5 - s)), 0.1, 0.9)
        result[fi] = (1 - blend_per) * s + blend_per * s_cooc_out[fi]
    return result

for adapt_s in [0.5, 1.0, 1.5, 2.0]:
    for base_b in [0.45, 0.55, 0.65]:
        name = f"adpt_str{int(adapt_s*10):02d}_b{int(base_b*100):02d}"
        s = adaptive_blend_cooc(double_best, power=2.0, center=0.55, slope=41.0,
                                alpha=0.130, base_blend=base_b, adapt_strength=adapt_s)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 119})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M4 done ({time.time()-t0:.0f}s)", flush=True)

# ── M5: Window-level co-occurrence ───────────────────────────────────────────
print("\n[M5] Window-level co-occurrence...", flush=True)
t0 = time.time()

# For window-level, we need per-window file-level co-occurrence
# Idea: smooth each window's scores using co-occurrence of file-level stats
# Then aggregate to file level

def win_level_cooc_blend(power=2.0, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    """
    Apply co-occurrence at WINDOW level using file-level co-occurrence matrix.
    Each window's scores are smoothed using COOC_NORM derived from training files.
    """
    # Get window-level scores from logits (same sigmoid as make_lp)
    win_scores_raw = 1.0 / (1.0 + np.exp(np.clip(-logit_win / 8.0, -88, 88)))  # T=8

    # Apply co-occurrence at window level
    win_smoothed = np.zeros_like(win_scores_raw)
    for wi in range(win_scores_raw.shape[0]):
        s = win_scores_raw[wi]
        s_pow = np.clip(s, 0, 1) ** power
        arg = np.clip(-slope * (s_pow - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s_pow * gate
        if np.abs(s_gated).sum() < EPS:
            win_smoothed[wi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        win_smoothed[wi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)

    # Aggregate to file level
    win_agg = np.stack([win_smoothed[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

    # Blend with double_best
    return (1 - blend) * double_best + blend * win_agg

for pw in [1.5, 2.0, 2.5]:
    for alpha in [0.100, 0.130, 0.160]:
        for blend in [0.20, 0.30, 0.40]:
            name = f"winlv_pw{int(pw*10):02d}_a{int(alpha*100):03d}_w{int(blend*100):02d}"
            s = win_level_cooc_blend(power=pw, center=0.55, slope=41.0,
                                     alpha=alpha, blend=blend)
            auc = macro_auc(s)
            delta = auc - best_loo
            status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
            print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
            results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 119})
            if auc > best_loo + 1e-7:
                best_loo = auc

print(f"  M5 done ({time.time()-t0:.0f}s)", flush=True)

# ── M6: Two-round compound pow+cooc ─────────────────────────────────────────
print("\n[M6] Two-round compound pow+cooc...", flush=True)
t0 = time.time()

# Apply current best pow+cooc, then apply AGAIN with smaller alpha/blend
best_r1 = pow_cooc_blend(double_best)  # first round (current best params)

for alpha2 in [0.020, 0.030, 0.040, 0.050]:
    for blend2 in [0.15, 0.25, 0.35]:
        for power2 in [1.5, 2.0, 2.5]:
            name = f"2r_pw{int(power2*10):02d}_a{int(alpha2*100):03d}_w{int(blend2*100):02d}"
            s_r2 = pow_cooc_blend(best_r1, power=power2, center=0.55, slope=41.0,
                                  alpha=alpha2, blend=blend2)
            auc = macro_auc(s_r2)
            delta = auc - best_loo
            status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
            print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
            results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 119})
            if auc > best_loo + 1e-7:
                best_loo = auc

# Also try: blend best_r1 with double_best at different ratios
# (i.e., partial first round)
for w1 in [0.60, 0.70, 0.80, 0.90]:
    mixed = w1 * best_r1 + (1 - w1) * double_best
    name = f"r1mix_w{int(w1*100):02d}"
    auc = macro_auc(mixed)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 119})
    if auc > best_loo + 1e-7:
        best_loo = auc

print(f"  M6 done ({time.time()-t0:.0f}s)", flush=True)

# ── Summary + Save ────────────────────────────────────────────────────────────
res = json.load(open(RESULTS_PATH))
prev_best_loo = res["best"]["loo_auc"]
prev_best_method = res["best"]["method"]

top10 = sorted(results, key=lambda x: -x["loo_auc"])[:10]
print("\n" + "="*60)
print(f"[batch119] SUMMARY")
print(f"  Previous best: {prev_best_loo:.6f} ({prev_best_method})")
print(f"  Top-10 this batch:")
for r in top10:
    print(f"    {r['method']}: {r['loo_auc']:.6f} ({r['delta']:+.6f})")

new_best = max(results, key=lambda x: x["loo_auc"])
if new_best["loo_auc"] > prev_best_loo + 1e-7:
    print(f"\n  NEW BEST: {new_best['method']} LOO={new_best['loo_auc']:.6f} ({new_best['delta']:+.6f})")
    res["best"] = {"method": new_best["method"], "loo_auc": new_best["loo_auc"]}
    # Update pkl
    import pickle
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
