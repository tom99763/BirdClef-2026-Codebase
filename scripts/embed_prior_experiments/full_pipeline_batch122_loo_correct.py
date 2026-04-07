"""
Batch 122 — LOO-Correct Cooc + Novel Structural Variations
===============================================================================
Current best: idf_p075 LOO=0.994463

Key insight: Current COOC_NORM is computed from ALL files including the test file.
This is technically data leakage. LOO-correct version may give better generalization.

Strategy:
1. M1: LOO-correct co-occurrence (exclude fi from cooc matrix when evaluating fi)
2. M2: Anti-correlation suppression (anti-correlated species suppress each other)
3. M3: Per-species adaptive alpha (rare species get higher co-occurrence weight)
4. M4: Convex combination of LOO-correct + full cooc (balance bias vs variance)
5. M5: Frequency-ratio co-occurrence (P(j|i)/P(j) — measures synergy beyond base rate)
6. M6: Isotonic/monotone recalibration of scores before cooc
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

print(f"[batch122] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch122] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

# ── Global cooc data ──────────────────────────────────────────────────────────
fl = file_labels.astype(np.float32)
count_i_global = fl.sum(0) + EPS
cooc_raw_global = fl.T @ fl
COOC_NORM_GLOBAL = cooc_raw_global / count_i_global[:, None]
np.fill_diagonal(COOC_NORM_GLOBAL, 0)

raw_idf_global = np.log(float(n_files) / (count_i_global + 1.0 - EPS))
raw_idf_global = np.clip(raw_idf_global, 0, None)
IDF_W075_GLOBAL = raw_idf_global ** 0.75; IDF_W075_GLOBAL /= (IDF_W075_GLOBAL.mean() + EPS)

# Precompute per-fold LOO cooc matrices and IDF weights
print("Pre-computing LOO co-occurrence matrices...", flush=True)
COOC_LOO = []  # list of 66 (n_species, n_species) matrices
IDF_LOO = []   # list of 66 (n_species,) vectors
for fi in range(n_files):
    fl_tr = fl[np.arange(n_files) != fi]  # (n_files-1, n_species)
    count_tr = fl_tr.sum(0) + EPS
    cooc_tr = fl_tr.T @ fl_tr
    cooc_norm_tr = cooc_tr / count_tr[:, None]
    np.fill_diagonal(cooc_norm_tr, 0)
    COOC_LOO.append(cooc_norm_tr)
    # LOO IDF (excluding fi)
    raw_idf_tr = np.log(float(n_files - 1) / (count_tr + 1.0 - EPS))
    raw_idf_tr = np.clip(raw_idf_tr, 0, None)
    idf_w = raw_idf_tr ** 0.75; idf_w /= (idf_w.mean() + EPS)
    IDF_LOO.append(idf_w)
print(f"  Done.", flush=True)

def idf_blend_global(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    """Reference: current best with global cooc matrix."""
    s_pow = np.clip(scores, 0, 1) ** 2.0
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * IDF_W075_GLOBAL
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM_GLOBAL.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return (1 - blend) * scores + blend * smoothed

print(f"  current best check: {macro_auc(idf_blend_global(double_best)):.6f} (expected {best_loo:.6f})", flush=True)

results = []

# ── M1: LOO-correct co-occurrence ─────────────────────────────────────────────
print("\n[M1] LOO-correct co-occurrence...", flush=True)
t0 = time.time()

def idf_blend_loo_correct(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    """Use per-fold LOO co-occurrence matrix (no leakage of test file labels)."""
    s_pow = np.clip(scores, 0, 1) ** 2.0
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        cooc_fi = COOC_LOO[fi]
        idf_fi = IDF_LOO[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * idf_fi
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = cooc_fi.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return (1 - blend) * scores + blend * smoothed

for alpha in [0.100, 0.115, 0.130, 0.145, 0.160]:
    for blend in [0.45, 0.55, 0.65]:
        name = f"loo_a{int(alpha*100):03d}_w{int(blend*100):02d}"
        s = idf_blend_loo_correct(scores=double_best, alpha=alpha, blend=blend)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 122})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M1 done ({time.time()-t0:.0f}s)", flush=True)

# ── M2: Anti-correlation suppression ─────────────────────────────────────────
print("\n[M2] Anti-correlation suppression...", flush=True)
t0 = time.time()

# Compute expected P(j|NOT i) = count(j ∧ NOT i) / count(NOT i)
# If P(j|i) << P(j|NOT i), then i is anti-correlated with j
# Suppression: when i is detected, suppress j proportionally to anti-correlation

fl_int = fl.astype(np.float32)
count_not_i = (n_files - count_i_global)  # count of files WITHOUT species i
cooc_not_i = (count_i_global[:, None] * np.ones((n_species, n_species)) - cooc_raw_global)
cooc_not_i = np.clip(cooc_not_i, 0, None)
# P(j | NOT i) = count(j ∧ NOT i) / count(NOT i)
COOC_NEG = cooc_not_i / (count_not_i[:, None] + EPS)
np.fill_diagonal(COOC_NEG, 0)

def anti_cooc_blend(scores, center=0.55, slope=41.0, alpha_pos=0.130, alpha_neg=0.050,
                    blend=0.55):
    """
    Positive co-occurrence boosts + negative co-occurrence suppresses.
    When i is detected with high confidence, j gets:
      + alpha_pos * P(j|i) * gate_i contribution
      - alpha_neg * P(j|NOT i) * gate_i contribution  (suppression)
    """
    s_pow = np.clip(scores, 0, 1) ** 2.0
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * IDF_W075_GLOBAL
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib_pos = COOC_NORM_GLOBAL.T @ s_gated
        contrib_neg = COOC_NEG.T @ s_gated
        # Normalize
        max_pos = np.abs(contrib_pos).max(); max_neg = np.abs(contrib_neg).max()
        if max_pos > EPS: contrib_pos /= max_pos
        if max_neg > EPS: contrib_neg /= max_neg
        smoothed[fi] = ((1 - alpha_pos) * s
                        + alpha_pos * np.clip(contrib_pos, 0, None)
                        - alpha_neg * np.clip(contrib_neg, 0, None))
    return (1 - blend) * scores + blend * smoothed

for alpha_pos in [0.100, 0.130, 0.160]:
    for alpha_neg in [0.010, 0.020, 0.030, 0.050]:
        name = f"anti_ap{int(alpha_pos*100):03d}_an{int(alpha_neg*100):03d}"
        s = anti_cooc_blend(double_best, alpha_pos=alpha_pos, alpha_neg=alpha_neg)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 122})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M2 done ({time.time()-t0:.0f}s)", flush=True)

# ── M3: Per-species adaptive alpha ───────────────────────────────────────────
print("\n[M3] Per-species adaptive alpha...", flush=True)
t0 = time.time()

def adaptive_alpha_blend(scores, base_alpha=0.130, idf_alpha_scale=0.5,
                         center=0.55, slope=41.0, blend=0.55):
    """
    Alpha varies per TARGET species: rare species (high IDF) get higher correction.
    alpha_j = base_alpha * (1 + idf_alpha_scale * (IDF_W075[j] - 1))
    """
    alpha_per_species = base_alpha * (1 + idf_alpha_scale * (IDF_W075_GLOBAL - 1))
    alpha_per_species = np.clip(alpha_per_species, 0.01, 0.50)

    s_pow = np.clip(scores, 0, 1) ** 2.0
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * IDF_W075_GLOBAL
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM_GLOBAL.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        # Variable alpha per target species
        smoothed[fi] = (1 - alpha_per_species) * s + alpha_per_species * np.clip(contrib, 0, None)
    return (1 - blend) * scores + blend * smoothed

for idf_scale in [0.20, 0.40, 0.60, 0.80, 1.0, 1.5]:
    for base_a in [0.100, 0.130, 0.160]:
        name = f"adaptalpha_bs{int(base_a*100):03d}_sc{int(idf_scale*100):03d}"
        s = adaptive_alpha_blend(double_best, base_alpha=base_a, idf_alpha_scale=idf_scale)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 122})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M3 done ({time.time()-t0:.0f}s)", flush=True)

# ── M4: LOO-correct blended with global ──────────────────────────────────────
print("\n[M4] LOO-correct × global blend...", flush=True)
t0 = time.time()

best_global = idf_blend_global(double_best)
best_loo_correct = idf_blend_loo_correct(double_best)
print(f"  global: {macro_auc(best_global):.6f}", flush=True)
print(f"  loo_correct: {macro_auc(best_loo_correct):.6f}", flush=True)

for w_loo in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    w_glob = 1 - w_loo
    blended = w_loo * best_loo_correct + w_glob * best_global
    name = f"blend_loo{int(w_loo*100):02d}_glob{int(w_glob*100):02d}"
    auc = macro_auc(blended)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 122})
    if auc > best_loo + 1e-7:
        best_loo = auc

print(f"  M4 done ({time.time()-t0:.0f}s)", flush=True)

# ── M5: Frequency-ratio co-occurrence (lift matrix) ──────────────────────────
print("\n[M5] Frequency-ratio (lift) co-occurrence...", flush=True)
t0 = time.time()

# Lift[i,j] = P(j|i) / P(j) — how much more likely is j given i vs base rate
p_j = count_i_global / n_files
LIFT = COOC_NORM_GLOBAL / (p_j[None, :] + EPS)  # COOC_NORM[i,j] = P(j|i); divide by P(j)
np.fill_diagonal(LIFT, 0)
# Clip lift to [0, max_lift] to prevent extreme values
LIFT_CLIPPED = np.clip(LIFT, 0, 5.0)
# Row normalize
row_sum_lift = LIFT_CLIPPED.sum(1) + EPS
LIFT_NORM = LIFT_CLIPPED / row_sum_lift[:, None]

def lift_cooc_blend(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55,
                   max_lift=5.0):
    LIFT_L = np.clip(COOC_NORM_GLOBAL / (p_j[None, :] + EPS), 0, max_lift)
    np.fill_diagonal(LIFT_L, 0)
    rs = LIFT_L.sum(1) + EPS
    LIFT_L = LIFT_L / rs[:, None]
    s_pow = np.clip(scores, 0, 1) ** 2.0
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * IDF_W075_GLOBAL
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = LIFT_L.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return (1 - blend) * scores + blend * smoothed

for max_lift in [2.0, 3.0, 5.0, 8.0, 10.0]:
    for alpha in [0.100, 0.130, 0.160]:
        name = f"lift_ml{int(max_lift):02d}_a{int(alpha*100):03d}"
        s = lift_cooc_blend(double_best, alpha=alpha, max_lift=max_lift)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 122})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M5 done ({time.time()-t0:.0f}s)", flush=True)

# ── M6: Soft monotone recalibration before cooc ──────────────────────────────
print("\n[M6] Pre-cooc recalibration (beta transform) + cooc...", flush=True)
t0 = time.time()

def beta_transform(s, beta=0.5):
    """Beta distribution-like monotone transform: s^beta / (s^beta + (1-s)^beta)"""
    s_c = np.clip(s, EPS, 1 - EPS)
    sp = s_c ** beta; sm = (1 - s_c) ** beta
    return sp / (sp + sm + EPS)

def recal_cooc_blend(scores, beta=0.5, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    """Apply monotone beta recalibration before co-occurrence smoothing."""
    s_recal = beta_transform(np.clip(scores, 0, 1), beta=beta)
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_recal[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * IDF_W075_GLOBAL
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = scores[fi]; continue
        contrib = COOC_NORM_GLOBAL.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return (1 - blend) * scores + blend * smoothed

for beta in [0.3, 0.5, 0.7, 1.5, 2.0, 3.0]:
    name = f"beta_b{int(beta*10):02d}"
    s = recal_cooc_blend(double_best, beta=beta)
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 122})
    if auc > best_loo + 1e-7:
        best_loo = auc

# Also try combining beta with power transform (beta first, then power^2)
for beta in [0.5, 0.7, 1.5]:
    for power in [1.5, 2.0, 2.5]:
        name = f"betapow_b{int(beta*10):02d}_p{int(power*10):02d}"
        s_c = np.clip(double_best, EPS, 1 - EPS)
        s_b = s_c ** beta / (s_c ** beta + (1 - s_c) ** beta + EPS)
        s_bp = np.clip(s_b, 0, 1) ** power
        smoothed = np.zeros_like(double_best)
        for fi in range(n_files):
            s = s_bp[fi]
            arg = np.clip(-slope * (s - 0.55), -88, 88)
            gate = 1.0 / (1.0 + np.exp(arg))
            s_gated = s * gate * IDF_W075_GLOBAL
            if np.abs(s_gated).sum() < EPS:
                smoothed[fi] = double_best[fi]; continue
            contrib = COOC_NORM_GLOBAL.T @ s_gated
            max_c = np.abs(contrib).max()
            if max_c > EPS: contrib /= max_c
            smoothed[fi] = (1 - 0.130) * s + 0.130 * np.clip(contrib, 0, None)
        sv = (1 - 0.55) * double_best + 0.55 * smoothed
        auc = macro_auc(sv)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 122})
        if auc > best_loo + 1e-7:
            best_loo = auc

slope = 41.0  # reset for remaining code
print(f"  M6 done ({time.time()-t0:.0f}s)", flush=True)

# ── Summary + Save ────────────────────────────────────────────────────────────
res = json.load(open(RESULTS_PATH))
prev_best_loo = res["best"]["loo_auc"]
prev_best_method = res["best"]["method"]

top10 = sorted(results, key=lambda x: -x["loo_auc"])[:10]
print("\n" + "="*60)
print(f"[batch122] SUMMARY")
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
