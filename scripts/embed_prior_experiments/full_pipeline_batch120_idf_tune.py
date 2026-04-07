"""
Batch 120 — Fine-tune IDF-weighted Co-occurrence
===============================================================================
Current best: idf_p075 LOO=0.994463 (+0.000088 from batch119)
Formula: power=2.0, center=0.55, slope=41, alpha=0.130, blend=0.55
         IDF[i] = clip(log(n_files/(count_i+1)), 0); idf_w = IDF^0.75; normalize
         s_gated = s_pow * gate * idf_w (weight rare source species more)

Strategy:
1. M1: Fine idf_power sweep (0.50→1.00, step 0.05) — find optimal IDF strength
2. M2: Joint idf_power × alpha sweep around (0.75, 0.130)
3. M3: Joint idf_power × blend sweep around (0.75, 0.55)
4. M4: Joint idf_power × center/slope sweep
5. M5: IDF normalization variants (different ways to normalize idf_w)
6. M6: Combined source+target IDF (careful small target weighting)
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

print(f"[batch120] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch120] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

# ── IDF weight computation ────────────────────────────────────────────────────
def _cooc_matrix():
    fl = file_labels.astype(np.float32)
    cooc = fl.T @ fl
    count_i = fl.sum(0) + EPS
    cooc_norm = cooc / count_i[:, None]
    np.fill_diagonal(cooc_norm, 0)
    return cooc_norm, count_i - EPS

COOC_NORM, COUNT_I = _cooc_matrix()

def _compute_idf(idf_power):
    """IDF weights: rare species weighted more strongly."""
    raw_idf = np.log(float(n_files) / (COUNT_I + 1.0))
    raw_idf = np.clip(raw_idf, 0, None)
    idf_w = raw_idf ** idf_power
    mean_idf = idf_w.mean() + EPS
    return idf_w / mean_idf  # normalize so mean=1

def idf_cooc(scores, center=0.55, slope=41.0, alpha=0.130, idf_power=0.75):
    """Power+soft_cooc with IDF source weighting."""
    s_pow = np.clip(scores, 0, 1) ** 2.0
    idf_w = _compute_idf(idf_power)
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * idf_w  # IDF-weighted source signal
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def idf_blend(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55, idf_power=0.75):
    s_cooc = idf_cooc(scores, center=center, slope=slope, alpha=alpha, idf_power=idf_power)
    return (1 - blend) * scores + blend * s_cooc

# Verify current best
best_check = idf_blend(double_best, idf_power=0.75)
print(f"  current best check: {macro_auc(best_check):.6f} (expected {best_loo:.6f})", flush=True)

results = []

# ── M1: Fine idf_power sweep ─────────────────────────────────────────────────
print("\n[M1] Fine idf_power sweep...", flush=True)
t0 = time.time()

for idf_p in np.arange(0.50, 1.01, 0.05):
    idf_p = round(idf_p, 2)
    name = f"idf_fp{int(idf_p*100):03d}"
    s = idf_blend(double_best, idf_power=idf_p)
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 120})
    if auc > best_loo + 1e-7:
        best_loo = auc

# Also try values below 0.50 and above 1.0
for idf_p in [0.10, 0.20, 0.30, 0.40, 1.10, 1.20, 1.30, 1.40, 1.50]:
    name = f"idf_ext{int(idf_p*100):03d}"
    s = idf_blend(double_best, idf_power=idf_p)
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 120})
    if auc > best_loo + 1e-7:
        best_loo = auc

print(f"  M1 done ({time.time()-t0:.0f}s)", flush=True)

# ── M2: Joint idf_power × alpha sweep ────────────────────────────────────────
print("\n[M2] Joint idf_power × alpha sweep...", flush=True)
t0 = time.time()

# Based on M1 findings, narrow around best idf_power
best_idf = 0.75  # will update dynamically but start here
for idf_p in [0.60, 0.70, 0.75, 0.80, 0.85]:
    for alpha in [0.100, 0.115, 0.130, 0.145, 0.160]:
        name = f"j_idf{int(idf_p*100):03d}_a{int(alpha*100):03d}"
        s = idf_blend(double_best, alpha=alpha, idf_power=idf_p)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 120})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M2 done ({time.time()-t0:.0f}s)", flush=True)

# ── M3: Joint idf_power × blend sweep ────────────────────────────────────────
print("\n[M3] Joint idf_power × blend sweep...", flush=True)
t0 = time.time()

for idf_p in [0.60, 0.70, 0.75, 0.80, 0.85]:
    for blend in [0.45, 0.50, 0.55, 0.60, 0.65]:
        name = f"j_idf{int(idf_p*100):03d}_w{int(blend*100):02d}"
        s = idf_blend(double_best, blend=blend, idf_power=idf_p)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 120})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M3 done ({time.time()-t0:.0f}s)", flush=True)

# ── M4: Joint idf_power × center/slope sweep ─────────────────────────────────
print("\n[M4] Joint idf_power × center/slope sweep...", flush=True)
t0 = time.time()

for idf_p in [0.70, 0.75, 0.80]:
    for center in [0.51, 0.53, 0.55, 0.57]:
        for slope in [33, 37, 41, 45]:
            name = f"j_idf{int(idf_p*100):03d}_c{int(center*100):03d}_sl{slope:02d}"
            s = idf_blend(double_best, center=center, slope=slope, idf_power=idf_p)
            auc = macro_auc(s)
            delta = auc - best_loo
            status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
            print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
            results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 120})
            if auc > best_loo + 1e-7:
                best_loo = auc

print(f"  M4 done ({time.time()-t0:.0f}s)", flush=True)

# ── M5: IDF normalization variants ───────────────────────────────────────────
print("\n[M5] IDF normalization variants...", flush=True)
t0 = time.time()

def idf_norm_variant(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55,
                     idf_power=0.75, norm_mode="mean"):
    """Different ways to normalize idf_w."""
    raw_idf = np.log(float(n_files) / (COUNT_I + 1.0))
    raw_idf = np.clip(raw_idf, 0, None)
    idf_w_raw = raw_idf ** idf_power

    if norm_mode == "mean":
        idf_w = idf_w_raw / (idf_w_raw.mean() + EPS)
    elif norm_mode == "max":
        idf_w = idf_w_raw / (idf_w_raw.max() + EPS)
    elif norm_mode == "l2":
        idf_w = idf_w_raw / (np.linalg.norm(idf_w_raw) + EPS) * np.sqrt(n_species)
    elif norm_mode == "softmax":
        w_sm = idf_w_raw - idf_w_raw.max()
        idf_w = np.exp(w_sm) / (np.exp(w_sm).sum() + EPS) * n_species
    elif norm_mode == "rank":
        # Rank-based: replace with rank/n_species (uniform spacing)
        ranks = np.argsort(np.argsort(idf_w_raw))  # 0..n-1
        idf_w = (ranks + 0.5) / n_species
    elif norm_mode == "binary":
        # Binary: IDF above median gets weight 2, below gets weight 0.5
        med = np.median(idf_w_raw)
        idf_w = np.where(idf_w_raw >= med, 2.0, 0.5)
    else:
        idf_w = idf_w_raw

    s_pow = np.clip(scores, 0, 1) ** 2.0
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * idf_w
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return (1 - blend) * scores + blend * smoothed

for nmode in ["mean", "max", "l2", "softmax", "rank", "binary"]:
    for idf_p in [0.60, 0.75, 0.90]:
        name = f"idfnm_{nmode[:4]}_p{int(idf_p*100):03d}"
        s = idf_norm_variant(double_best, idf_power=idf_p, norm_mode=nmode)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 120})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M5 done ({time.time()-t0:.0f}s)", flush=True)

# ── M6: IDF + second power transform ─────────────────────────────────────────
print("\n[M6] IDF combined with different source power...", flush=True)
t0 = time.time()

def idf_cooc_srcpow(scores, src_power=2.0, center=0.55, slope=41.0, alpha=0.130,
                    blend=0.55, idf_power=0.75):
    """Apply power transform to source signal (before gate) separately from cooc smoothing."""
    idf_w = _compute_idf(idf_power)
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        s_pow_src = np.clip(s, 0, 1) ** src_power  # power for source signal
        arg = np.clip(-slope * (s_pow_src - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s_pow_src * gate * idf_w
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        # Apply smoothing to ORIGINAL scores (not powered)
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return (1 - blend) * scores + blend * smoothed

for src_pw in [1.5, 2.0, 2.5, 3.0]:
    for idf_p in [0.60, 0.75, 0.90]:
        name = f"srcpow{int(src_pw*10):02d}_idf{int(idf_p*100):03d}"
        s = idf_cooc_srcpow(double_best, src_power=src_pw, idf_power=idf_p)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 120})
        if auc > best_loo + 1e-7:
            best_loo = auc

# Also: apply current idf_blend result to itself (compound)
curr_best_s = idf_blend(double_best, idf_power=0.75)
for alpha2 in [0.020, 0.030, 0.040]:
    for blend2 in [0.15, 0.20, 0.25]:
        name = f"idf2r_a{int(alpha2*100):03d}_w{int(blend2*100):02d}"
        s = idf_blend(curr_best_s, alpha=alpha2, blend=blend2, idf_power=0.75)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 120})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M6 done ({time.time()-t0:.0f}s)", flush=True)

# ── Summary + Save ────────────────────────────────────────────────────────────
res = json.load(open(RESULTS_PATH))
prev_best_loo = res["best"]["loo_auc"]
prev_best_method = res["best"]["method"]

top10 = sorted(results, key=lambda x: -x["loo_auc"])[:10]
print("\n" + "="*60)
print(f"[batch120] SUMMARY")
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
