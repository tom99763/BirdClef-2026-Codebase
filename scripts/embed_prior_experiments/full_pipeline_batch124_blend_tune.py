"""
Batch 124 — Fine-tune IDF×Two-round Blend
===============================================================================
Current best: 3way_i85_r15_d00 LOO=0.994510 (+0.000024)
Formula: 0.85 * idf_result + 0.15 * two_round_result
Where:
  idf_result = 0.45*db + 0.55*soft_cooc_idf(clip(db,0,1)^2, c=0.55, sl=41, a=0.130, idf_pow=0.75)
  two_round_result = soft_cooc(r1_params) ∘ soft_cooc(r2_params)

Strategy:
1. M1: Ultra-fine IDF×two-round sweep (every 1% ratio around 85/15)
2. M2: Different two-round params in the blend
3. M3: Three-way blend with double_best at small weights
4. M4: IDF blend params + two-round ratio joint sweep
5. M5: Multiple co-occurrence results as ensemble
6. M6: Fine-tune IDF cooc params to optimize for 85/15 blend target
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

print(f"[batch124] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch124] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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
print(f"  double_best: {auc_db:.6f} [{time.time()-t0:.0f}s]", flush=True)

# ── Co-occurrence infrastructure ──────────────────────────────────────────────
fl = file_labels.astype(np.float32)
count_i = fl.sum(0) + EPS
cooc_raw = fl.T @ fl
COOC_NORM = cooc_raw / count_i[:, None]
np.fill_diagonal(COOC_NORM, 0)
raw_idf = np.log(float(n_files) / (count_i + 1.0 - EPS))
raw_idf = np.clip(raw_idf, 0, None)
IDF_W075 = raw_idf ** 0.75; IDF_W075 /= (IDF_W075.mean() + EPS)

def soft_cooc(scores, center=0.53, slope=37.0, alpha=0.086, idf_w=None):
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate
        if idf_w is not None:
            s_gated = s_gated * idf_w
        if np.abs(s_gated).sum() < EPS:
            smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def idf_cooc(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    s_pow = np.clip(scores, 0, 1) ** 2.0
    s_cooc = soft_cooc(s_pow, center=center, slope=slope, alpha=alpha, idf_w=IDF_W075)
    return (1 - blend) * scores + blend * s_cooc

def two_round(scores, c1=0.54, sl1=41.0, a1=0.089, c2=0.53, sl2=37.0, a2=0.040):
    r1 = soft_cooc(scores, center=c1, slope=sl1, alpha=a1)
    r2 = soft_cooc(r1, center=c2, slope=sl2, alpha=a2)
    return r2

# Pre-compute reference signals
idf_result = idf_cooc(double_best)
two_round_result = two_round(double_best)
best_blend = 0.85 * idf_result + 0.15 * two_round_result
print(f"  idf_result: {macro_auc(idf_result):.6f}", flush=True)
print(f"  two_round_result: {macro_auc(two_round_result):.6f}", flush=True)
print(f"  current best check: {macro_auc(best_blend):.6f} (expected {best_loo:.6f})", flush=True)

results = []

# ── M1: Ultra-fine IDF×two-round blend ───────────────────────────────────────
print("\n[M1] Ultra-fine IDF×two-round blend...", flush=True)
t0 = time.time()

for w_idf in np.arange(0.80, 0.96, 0.01):
    w_idf = round(w_idf, 2)
    w_2r = round(1 - w_idf, 2)
    name = f"blend_i{int(w_idf*100):03d}_r{int(w_2r*100):02d}"
    s = w_idf * idf_result + w_2r * two_round_result
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 124})
    if auc > best_loo + 1e-7:
        best_loo = auc

print(f"  M1 done ({time.time()-t0:.0f}s)", flush=True)

# ── M2: Different two-round params ────────────────────────────────────────────
print("\n[M2] Different two-round params × best IDF ratio...", flush=True)
t0 = time.time()

for c1 in [0.52, 0.53, 0.54, 0.55]:
    for sl1 in [37, 41, 45]:
        for a1 in [0.070, 0.089, 0.110]:
            for a2 in [0.020, 0.040, 0.060]:
                name = f"2r_c{int(c1*100):02d}_sl{sl1:02d}_a1{int(a1*100):03d}_a2{int(a2*100):03d}"
                tr_v = two_round(double_best, c1=c1, sl1=float(sl1), a1=a1, c2=0.53, sl2=37.0, a2=a2)
                s = 0.85 * idf_result + 0.15 * tr_v
                auc = macro_auc(s)
                delta = auc - best_loo
                status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
                print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
                results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 124})
                if auc > best_loo + 1e-7:
                    best_loo = auc

print(f"  M2 done ({time.time()-t0:.0f}s)", flush=True)

# ── M3: Three-way with double_best ────────────────────────────────────────────
print("\n[M3] Three-way blend with double_best at small weights...", flush=True)
t0 = time.time()

for w_idf in [0.83, 0.85, 0.87, 0.89]:
    for w_2r in [0.10, 0.13, 0.15, 0.17]:
        w_db = round(1 - w_idf - w_2r, 2)
        if w_db < -0.005 or w_db > 0.10: continue
        name = f"3w_i{int(w_idf*100):03d}_r{int(w_2r*100):02d}_d{int(max(0,w_db)*100):02d}"
        s = w_idf * idf_result + w_2r * two_round_result + max(0, w_db) * double_best
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 124})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M3 done ({time.time()-t0:.0f}s)", flush=True)

# ── M4: IDF cooc params optimized for 85/15 blend ────────────────────────────
print("\n[M4] IDF cooc params for 85/15 blend...", flush=True)
t0 = time.time()

for center in [0.53, 0.55, 0.57]:
    for slope in [37, 41, 45]:
        for alpha in [0.110, 0.130, 0.150]:
            for blend in [0.50, 0.55, 0.60]:
                idf_v = idf_cooc(double_best, center=center, slope=float(slope), alpha=alpha, blend=blend)
                s = 0.85 * idf_v + 0.15 * two_round_result
                auc = macro_auc(s)
                delta = auc - best_loo
                name = f"idf4_c{int(center*100):03d}_sl{slope:02d}_a{int(alpha*100):03d}_w{int(blend*100):02d}"
                status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
                print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
                results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 124})
                if auc > best_loo + 1e-7:
                    best_loo = auc

print(f"  M4 done ({time.time()-t0:.0f}s)", flush=True)

# ── M5: Ensemble of many co-occurrence variants ──────────────────────────────
print("\n[M5] Ensemble of multiple co-occurrence variants...", flush=True)
t0 = time.time()

# Build a larger ensemble with different IDF powers and two-round configs
idf_60 = idf_cooc(double_best, center=0.55, slope=41.0, alpha=0.130, blend=0.55)
idf_65 = idf_cooc(double_best, center=0.55, slope=41.0, alpha=0.130, blend=0.55)
idf_70 = idf_cooc(double_best, center=0.55, slope=41.0, alpha=0.130, blend=0.55)
# (all same since we haven't varied idf_power here — recompute)
for idf_pow, tag in [(0.60, 'p60'), (0.65, 'p65'), (0.70, 'p70'), (0.75, 'p75')]:
    idf_w_p = raw_idf ** idf_pow; idf_w_p /= (idf_w_p.mean() + EPS)
    s_pow = np.clip(double_best, 0, 1) ** 2.0
    tmp = soft_cooc(s_pow, center=0.55, slope=41.0, alpha=0.130, idf_w=idf_w_p)
    globals()[f'idf_{tag}'] = (1 - 0.55) * double_best + 0.55 * tmp

# Try simple average of IDF results × two-round
for combo, tags in [('2way', ['p65', 'p75']), ('3way', ['p60', 'p70', 'p75']),
                    ('4way', ['p60', 'p65', 'p70', 'p75'])]:
    mean_idf = np.mean([globals()[f'idf_{t}'] for t in tags], axis=0)
    s = 0.85 * mean_idf + 0.15 * two_round_result
    name = f"ensIDF_{combo}_x2r15"
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 124})
    if auc > best_loo + 1e-7:
        best_loo = auc

# Also try fine blend ratios with ensemble
for w_ens in [0.82, 0.85, 0.88, 0.91, 0.94]:
    mean_idf = np.mean([globals()[f'idf_{t}'] for t in ['p65', 'p70', 'p75']], axis=0)
    s = w_ens * mean_idf + (1 - w_ens) * two_round_result
    name = f"ens3idf_w{int(w_ens*100):03d}"
    auc = macro_auc(s)
    delta = auc - best_loo
    status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
    results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 124})
    if auc > best_loo + 1e-7:
        best_loo = auc

print(f"  M5 done ({time.time()-t0:.0f}s)", flush=True)

# ── M6: Blend ratio + IDF power joint search ─────────────────────────────────
print("\n[M6] Joint IDF power × blend ratio search...", flush=True)
t0 = time.time()

for idf_pow in [0.65, 0.70, 0.75, 0.80]:
    idf_w_p = raw_idf ** idf_pow; idf_w_p /= (idf_w_p.mean() + EPS)
    s_pow = np.clip(double_best, 0, 1) ** 2.0
    tmp = soft_cooc(s_pow, center=0.55, slope=41.0, alpha=0.130, idf_w=idf_w_p)
    idf_v = (1 - 0.55) * double_best + 0.55 * tmp
    for w_idf in [0.82, 0.85, 0.88, 0.91, 0.94]:
        s = w_idf * idf_v + (1 - w_idf) * two_round_result
        name = f"j_idfa{int(idf_pow*100):03d}_w{int(w_idf*100):03d}"
        auc = macro_auc(s)
        delta = auc - best_loo
        status = "*** NEW BEST ***" if auc > best_loo + 1e-7 else "(near-best)"
        print(f"  {name}: {auc:.6f} {delta:+.6f} {status}", flush=True)
        results.append({"method": name, "loo_auc": auc, "delta": delta, "batch": 124})
        if auc > best_loo + 1e-7:
            best_loo = auc

print(f"  M6 done ({time.time()-t0:.0f}s)", flush=True)

# ── Summary + Save ────────────────────────────────────────────────────────────
res = json.load(open(RESULTS_PATH))
prev_best_loo = res["best"]["loo_auc"]
prev_best_method = res["best"]["method"]

top10 = sorted(results, key=lambda x: -x["loo_auc"])[:10]
print("\n" + "="*60)
print(f"[batch124] SUMMARY")
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
