"""
Batch 123 — Final Push: Fine-tune IDF×Two-round + New Base Signals
===============================================================================
Current best: idf_p075 LOO=0.994463
Key observation: blend_idf80_2r19 = 0.994455 (only 0.000008 below best from batch121)

Strategy:
1. M1: Fine IDF×two-round blend (around 80/20 to find sweet spot)
2. M2: 3-way blend (double_best + IDF cooc + two-round)
3. M3: Multi-IDF ensemble (average different idf_power values)
4. M4: Confidence-gated co-occurrence (only for uncertain files)
5. M5: New logit pooling (top-k mean instead of max, different temperatures)
6. M6: Max-ceiling blend (take max of cooc output and original score)
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

print(f"[batch123] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch123] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

def make_lp_topk(T, k=3):
    """Top-K mean pooling over windows (instead of max)."""
    sig = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T, -88, 88)))
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        wins = sig[file_start[fi]:file_end[fi]]  # (n_wins, n_species)
        n_w = wins.shape[0]
        if n_w <= k:
            out[fi] = wins.max(0)
        else:
            # Top-K mean for each species
            top_k_idx = np.argsort(-wins, axis=0)[:k, :]  # (k, n_species)
            for si in range(n_species):
                out[fi, si] = wins[top_k_idx[:, si], si].mean()
    return out

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
print(f"  double_best: {auc_db:.6f} [{time.time()-t0:.0f}s]", flush=True)

# ── IDF co-occurrence + two-round ─────────────────────────────────────────────
fl = file_labels.astype(np.float32)
count_i_global = fl.sum(0) + EPS
cooc_raw_global = fl.T @ fl
COOC_NORM = cooc_raw_global / count_i_global[:, None]
np.fill_diagonal(COOC_NORM, 0)
raw_idf = np.log(float(n_files) / (count_i_global + 1.0 - EPS))
raw_idf = np.clip(raw_idf, 0, None)
IDF_W075 = raw_idf ** 0.75; IDF_W075 /= (IDF_W075.mean() + EPS)

def soft_cooc_idf(scores, center=0.55, slope=41.0, alpha=0.130):
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * IDF_W075
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def idf_blend_full(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    s_pow = np.clip(scores, 0, 1) ** 2.0
    return (1 - blend) * scores + blend * soft_cooc_idf(s_pow, center=center, slope=slope, alpha=alpha)

def soft_cooc_plain(scores, center=0.53, slope=37.0, alpha=0.086):
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

def two_round_scores_fn(scores):
    r1 = soft_cooc_plain(scores, center=0.54, slope=41.0, alpha=0.089)
    r2 = soft_cooc_plain(r1, center=0.53, slope=37.0, alpha=0.040)
    return r2

# Pre-compute reference signals
idf_result = idf_blend_full(double_best)
two_round_result = two_round_scores_fn(double_best)
print(f"  idf_result: {macro_auc(idf_result):.6f} (expected {best_loo:.6f})", flush=True)
print(f"  two_round_result: {macro_auc(two_round_result):.6f} (expected 0.994248)", flush=True)

results = []

# ── M1: Fine IDF × two-round blend ───────────────────────────────────────────
print("\n[M1] Fine IDF × two-round blend...", flush=True)
t0 = time.time()

for w_idf in np.arange(0.70, 1.01, 0.02):
    w_idf = round(w_idf, 2)
    w_2r = round(1 - w_idf, 2)
    name = f"idf2r_w{int(w_idf*100):03d}"
    s = w_idf * idf_result + w_2r * two_round_result
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
    if auc > best_loo + 1e-7:
        best_loo = auc

# Also try IDF blend with two_round as secondary
for w_idf in [0.85, 0.90, 0.95]:
    for w_2r in [0.05, 0.10, 0.15]:
        w_db = round(1 - w_idf - w_2r, 2)
        if w_db < -0.01: continue
        name = f"3way_i{int(w_idf*100):02d}_r{int(w_2r*100):02d}_d{int(max(0,w_db)*100):02d}"
        s = w_idf * idf_result + w_2r * two_round_result + max(0, w_db) * double_best
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M1 done ({time.time()-t0:.0f}s)", flush=True)

# ── M2: Multi-IDF ensemble (average different idf_power values) ───────────────
print("\n[M2] Multi-IDF ensemble...", flush=True)
t0 = time.time()

# Pre-compute IDF results at different powers
idf_results = {}
for idf_p in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
    idf_w_p = raw_idf ** idf_p; idf_w_p /= (idf_w_p.mean() + EPS)
    tmp = np.zeros_like(double_best)
    s_pow = np.clip(double_best, 0, 1) ** 2.0
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-41.0 * (s - 0.55), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * idf_w_p
        if np.abs(s_gated).sum() < EPS:
            tmp[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        tmp[fi] = (1 - 0.130) * s + 0.130 * np.clip(contrib, 0, None)
    idf_results[idf_p] = (1 - 0.55) * double_best + 0.55 * tmp

# Try simple averages and weighted averages
# Average of idf_p=0.65, 0.70, 0.75
ensemble_3 = (idf_results[0.65] + idf_results[0.70] + idf_results[0.75]) / 3
name = "multi_idf_3way"
auc = macro_auc(ensemble_3)
delta = auc - best_loo
print(f"  {name}: {auc:.6f} {delta:+.6f}", flush=True)
results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
if auc > best_loo + 1e-7:
    best_loo = auc

# Average of idf_p=0.60, 0.65, 0.70, 0.75, 0.80
ensemble_5 = (idf_results[0.60] + idf_results[0.65] + idf_results[0.70] + idf_results[0.75] + idf_results[0.80]) / 5
name = "multi_idf_5way"
auc = macro_auc(ensemble_5)
delta = auc - best_loo
print(f"  {name}: {auc:.6f} {delta:+.6f}", flush=True)
results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
if auc > best_loo + 1e-7:
    best_loo = auc

# Weight by performance
aucs_by_idf = {p: macro_auc(idf_results[p]) for p in idf_results}
for name_sfx, ps in [("top3", [0.65, 0.70, 0.75]), ("top4", [0.60, 0.65, 0.70, 0.75])]:
    aucs_arr = np.array([aucs_by_idf[p] for p in ps])
    w_arr = aucs_arr - aucs_arr.min() + 1e-4
    w_arr /= w_arr.sum()
    ens = sum(w * idf_results[p] for w, p in zip(w_arr, ps))
    name = f"wt_idf_{name_sfx}"
    auc = macro_auc(ens)
    delta = auc - best_loo
    print(f"  {name}: {auc:.6f} {delta:+.6f}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
    if auc > best_loo + 1e-7:
        best_loo = auc

print(f"  M2 done ({time.time()-t0:.0f}s)", flush=True)

# ── M3: Confidence-gated co-occurrence ───────────────────────────────────────
print("\n[M3] Confidence-gated co-occurrence...", flush=True)
t0 = time.time()

def confidence_gate_blend(scores, conf_thresh=0.6, blend_high=0.30, blend_low=0.70,
                          center=0.55, slope=41.0, alpha=0.130):
    """
    High-confidence files (max score > conf_thresh): less co-occurrence blending.
    Low-confidence files (max score <= conf_thresh): more co-occurrence blending.
    """
    s_pow = np.clip(scores, 0, 1) ** 2.0
    s_cooc = soft_cooc_idf(s_pow, center=center, slope=slope, alpha=alpha)
    result = np.zeros_like(scores)
    for fi in range(n_files):
        max_s = scores[fi].max()
        blend = blend_high if max_s > conf_thresh else blend_low
        result[fi] = (1 - blend) * scores[fi] + blend * s_cooc[fi]
    return result

for conf_t in [0.40, 0.50, 0.60, 0.70]:
    for b_high in [0.30, 0.45, 0.55]:
        for b_low in [0.55, 0.65, 0.75]:
            if b_high >= b_low: continue
            name = f"confgate_t{int(conf_t*100):02d}_h{int(b_high*100):02d}_l{int(b_low*100):02d}"
            s = confidence_gate_blend(double_best, conf_thresh=conf_t,
                                      blend_high=b_high, blend_low=b_low)
            auc = macro_auc(s)
            delta = auc - best_loo
            status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
            print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
            results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
            if auc > best_loo + 1e-7:
                best_loo = auc

print(f"  M3 done ({time.time()-t0:.0f}s)", flush=True)

# ── M4: Max-ceiling blend (only increase, never decrease) ────────────────────
print("\n[M4] Max-ceiling blend...", flush=True)
t0 = time.time()

def max_ceil_blend(scores, blend=0.55, center=0.55, slope=41.0, alpha=0.130):
    """Take max of original and co-occurrence output: only boost, never suppress."""
    s_pow = np.clip(scores, 0, 1) ** 2.0
    s_cooc_raw = soft_cooc_idf(s_pow, center=center, slope=slope, alpha=alpha)
    s_blended = (1 - blend) * scores + blend * s_cooc_raw
    # Only increase: take max
    return np.maximum(scores, s_blended)

for blend in [0.40, 0.55, 0.70, 0.85]:
    name = f"maxceil_w{int(blend*100):02d}"
    s = max_ceil_blend(double_best, blend=blend)
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
    if auc > best_loo + 1e-7:
        best_loo = auc

# Also: max of IDF result and two-round result (species-wise max)
max_idf_2r = np.maximum(idf_result, two_round_result)
name = "max_idf_2r"
auc = macro_auc(max_idf_2r)
delta = auc - best_loo
print(f"  {name}: {auc:.6f} {delta:+.6f}", flush=True)
results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
if auc > best_loo + 1e-7:
    best_loo = auc

# Geometric mean
geo_mean = np.sqrt(np.clip(idf_result, EPS, None) * np.clip(two_round_result, EPS, None))
name = "geo_idf_2r"
auc = macro_auc(geo_mean)
delta = auc - best_loo
print(f"  {name}: {auc:.6f} {delta:+.6f}", flush=True)
results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
if auc > best_loo + 1e-7:
    best_loo = auc

print(f"  M4 done ({time.time()-t0:.0f}s)", flush=True)

# ── M5: New logit pooling strategies ─────────────────────────────────────────
print("\n[M5] New logit pooling strategies...", flush=True)
t0 = time.time()

# Top-K mean pooling (instead of max)
for k in [2, 3, 5]:
    for T in [6.0, 8.0, 10.0, 12.0]:
        pT_topk = make_lp_topk(T, k=k)
        # Try blending this with current double_best
        for w_new in [0.02, 0.04, 0.06, 0.08]:
            w_db = 1 - w_new
            new_base = w_db * double_best + w_new * pT_topk
            # Apply IDF cooc to the new base
            s = idf_blend_full(new_base)
            auc = macro_auc(s)
            delta = auc - best_loo
            name = f"topk{k}_T{int(T):02d}_wnew{int(w_new*100):02d}"
            status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
            print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
            results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
            if auc > best_loo + 1e-7:
                best_loo = auc

# Different temperature pooling blends
for T_high in [15.0, 20.0, 25.0]:
    pT_high = make_lp(T_high)
    for w_high in [0.02, 0.04, 0.06]:
        new_base = (1 - w_high) * double_best + w_high * pT_high
        s = idf_blend_full(new_base)
        auc = macro_auc(s)
        delta = auc - best_loo
        name = f"hiT{int(T_high):02d}_w{int(w_high*100):02d}"
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M5 done ({time.time()-t0:.0f}s)", flush=True)

# ── M6: Species-weighted cooc output normalization ───────────────────────────
print("\n[M6] Output normalization variants...", flush=True)
t0 = time.time()

def sum_norm_cooc(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    """Normalize co-occurrence contribution by sum instead of max."""
    s_pow = np.clip(scores, 0, 1) ** 2.0
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * IDF_W075
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        # Sum normalize instead of max normalize
        sum_c = contrib.sum()
        if abs(sum_c) > EPS: contrib /= abs(sum_c) / n_species  # scale to mean=1
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return (1 - blend) * scores + blend * smoothed

for alpha in [0.080, 0.100, 0.130]:
    for blend in [0.45, 0.55, 0.65]:
        name = f"sumnorm_a{int(alpha*100):03d}_w{int(blend*100):02d}"
        s = sum_norm_cooc(double_best, alpha=alpha, blend=blend)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
        if auc > best_loo + 1e-7:
            best_loo = auc

# Apply sigmoid to smoothed output (to keep in [0,1] range)
def sigmoid_post_cooc(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55,
                      post_scale=1.0):
    """Apply sigmoid to the co-occurrence contribution before blending."""
    s_pow = np.clip(scores, 0, 1) ** 2.0
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * IDF_W075
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        # Apply sigmoid to contrib (maps [-1,1] to [0.27, 0.73])
        contrib_sig = 1.0 / (1.0 + np.exp(np.clip(-post_scale * contrib, -88, 88)))
        smoothed[fi] = (1 - alpha) * s + alpha * contrib_sig
    return (1 - blend) * scores + blend * smoothed

for post_s in [1.0, 2.0, 4.0]:
    for alpha in [0.130, 0.160]:
        name = f"sigpost_sc{int(post_s):01d}_a{int(alpha*100):03d}"
        s = sigmoid_post_cooc(double_best, alpha=alpha, post_scale=post_s)
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 123})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M6 done ({time.time()-t0:.0f}s)", flush=True)

# ── Summary + Save ────────────────────────────────────────────────────────────
res = json.load(open(RESULTS_PATH))
prev_best_loo = res["best"]["loo_auc"]
prev_best_method = res["best"]["method"]

top10 = sorted(results, key=lambda x: -x["loo_auc"])[:10]
print("\n" + "="*60)
print(f"[batch123] SUMMARY")
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
