"""
batch127b: Cascade/nested co-occurrence combinations (FIXED: proper full chain)
M1: Nested: two_round(idf_result) — 先 IDF 再 two-round
M2: Nested: idf_cooc(two_round_result)
M3: Cascade: 3-component blend with nested-2r as third component
M4: Per-species adaptive alpha (alpha scales with IDF weight)
M5: Three-round co-occurrence
M6: Non-linear blends (geometric mean, max-blend, power post-proc)
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

print(f"[batch127b] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch127b] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

def macro_auc(s):
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            aucs.append(roc_auc_score(y, s[:, si]))
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(method, score, config, note=''):
    global best_loo
    delta = score - best_loo
    r = {'method': method, 'loo_auc': score, 'config': config, 'note': note}
    res['experiments'].append(r)
    if score > best_loo:
        best_loo = score
        res['best'] = {'method': method, 'loo_auc': score}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(res, f, indent=2)
    return delta

tried = {e['method'] for e in res['experiments']}

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

# ── Build full chain → double_best ───────────────────────────────────────────
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
print(f"  double_best: {macro_auc(double_best):.6f} [{time.time()-t0:.0f}s]", flush=True)

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
        if idf_w is not None: s_gated = s_gated * idf_w
        if np.abs(s_gated).sum() < EPS: smoothed[fi] = s; continue
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
idf_result       = idf_cooc(double_best)
two_round_result = two_round(double_best)
best_blend       = 0.85 * idf_result + 0.15 * two_round_result
print(f"  idf_result: {macro_auc(idf_result):.6f}", flush=True)
print(f"  two_round_result: {macro_auc(two_round_result):.6f}", flush=True)
print(f"  3way check: {macro_auc(best_blend):.6f} (expected {best_loo:.6f})", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M1: Nested — two_round(idf_result)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M1] Nested: two_round on top of idf_result...", flush=True)

m1_best = 0.0
for c1, sl1, a1, c2, sl2, a2 in [
    (0.54, 41.0, 0.040, 0.53, 37.0, 0.020),
    (0.54, 41.0, 0.060, 0.53, 37.0, 0.030),
    (0.55, 41.0, 0.050, 0.54, 37.0, 0.025),
    (0.54, 41.0, 0.020, 0.53, 37.0, 0.010),
    (0.54, 41.0, 0.030, 0.53, 37.0, 0.015),
]:
    mname = f'nest_idf_2r_c{int(c1*100):d}_a1{int(a1*1000):03d}_a2{int(a2*1000):03d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    result = two_round(idf_result, c1=c1, sl1=sl1, a1=a1, c2=c2, sl2=sl2, a2=a2)
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'c1':c1,'sl1':sl1,'a1':a1,'c2':c2,'sl2':sl2,'a2':a2})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)
    m1_best = max(m1_best, score)

# Blend: idf_result blended with two_round(idf_result)
r2_on_idf = two_round(idf_result, c1=0.54, sl1=41.0, a1=0.040, c2=0.53, sl2=37.0, a2=0.020)
for w_orig in [0.90, 0.85, 0.80, 0.95, 0.75]:
    mname = f'nest_idf_blend_w{int(w_orig*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    result = w_orig * idf_result + (1 - w_orig) * r2_on_idf
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'w_orig': w_orig})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)
    m1_best = max(m1_best, score)

print(f"  M1 done, best={m1_best:.6f}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M2: Nested — idf_cooc(two_round_result)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M2] Nested: idf_cooc on top of two_round_result...", flush=True)

m2_best = 0.0
for center, alpha, blend_w in [
    (0.55, 0.060, 0.55),
    (0.55, 0.090, 0.55),
    (0.55, 0.040, 0.55),
    (0.55, 0.060, 0.70),
    (0.54, 0.060, 0.55),
    (0.55, 0.060, 0.40),
]:
    mname = f'nest_2r_idf_c{int(center*100):d}_a{int(alpha*1000):03d}_w{int(blend_w*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    result = idf_cooc(two_round_result, center=center, slope=41.0, alpha=alpha, blend=blend_w)
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'center':center,'alpha':alpha,'blend':blend_w})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)
    m2_best = max(m2_best, score)

# Blend: 3way with nested idf(two_round)
nested_idf_on_2r = idf_cooc(two_round_result, center=0.55, slope=41.0, alpha=0.060, blend=0.55)
for w_best in [0.90, 0.85, 0.95]:
    mname = f'nest_2r_idf_3way_blend_w{int(w_best*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    result = w_best * best_blend + (1 - w_best) * nested_idf_on_2r
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'w_best': w_best})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)
    m2_best = max(m2_best, score)

print(f"  M2 done, best={m2_best:.6f}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M3: Cascade blend with nested component
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M3] Cascade blend...", flush=True)

for w_idf, w_2r, w_nest, tag in [
    (0.85, 0.10, 0.05, 'cas_i85_r10_n05'),
    (0.85, 0.05, 0.10, 'cas_i85_r05_n10'),
    (0.85, 0.08, 0.07, 'cas_i85_r08_n07'),
    (0.85, 0.00, 0.15, 'cas_i85_r00_n15'),
    (0.80, 0.10, 0.10, 'cas_i80_r10_n10'),
    (0.90, 0.05, 0.05, 'cas_i90_r05_n05'),
]:
    if tag in tried:
        print(f'  {tag}: skip', flush=True); continue
    result = w_idf * idf_result + w_2r * two_round_result + w_nest * r2_on_idf
    score  = macro_auc(result)
    delta  = save_result(tag, score, {'w_idf':w_idf,'w_2r':w_2r,'w_nest':w_nest})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {tag}: {score:.6f} {delta:+.6f}{flag}', flush=True)

print(f"  M3 done", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M4: Per-species adaptive alpha
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M4] Per-species adaptive alpha...", flush=True)

m4_best = 0.0
for idf_scale in [0.3, 0.5, -0.3, 0.1]:
    mname = f'adapt_alpha_is{"p" if idf_scale>=0 else "m"}{int(abs(idf_scale)*10):d}_ba130'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    base_alpha = 0.130
    alpha_sp = base_alpha * (1 + idf_scale * (IDF_W075 - 1.0))
    alpha_sp = np.clip(alpha_sp, 0.01, 0.5)
    s_pow = np.clip(double_best, 0, 1) ** 2.0
    smoothed = np.zeros_like(s_pow)
    for fi in range(n_files):
        s = s_pow[fi]
        arg = np.clip(-41.0 * (s - 0.55), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * IDF_W075
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha_sp) * s + alpha_sp * np.clip(contrib, 0, None)
    result = (1 - 0.55) * double_best + 0.55 * smoothed
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'idf_scale': idf_scale, 'base_alpha': 0.130})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)
    m4_best = max(m4_best, score)

print(f"  M4 done, best={m4_best:.6f}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M5: Three-round co-occurrence
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M5] Three-round co-occurrence...", flush=True)

m5_best = 0.0
for c3, sl3, a3 in [
    (0.53, 33.0, 0.010),
    (0.53, 33.0, 0.020),
    (0.53, 33.0, 0.030),
    (0.52, 30.0, 0.020),
    (0.54, 37.0, 0.015),
]:
    mname = f'3round_c{int(c3*100):d}_sl{int(sl3):d}_a{int(a3*1000):03d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    r1 = soft_cooc(double_best, center=0.54, slope=41.0, alpha=0.089)
    r2 = soft_cooc(r1,          center=0.53, slope=37.0, alpha=0.040)
    r3 = soft_cooc(r2,          center=c3,   slope=sl3,  alpha=a3)
    score  = macro_auc(r3)
    delta  = save_result(mname, score, {'c3':c3,'sl3':sl3,'a3':a3})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)
    m5_best = max(m5_best, score)

# 4-way blend with three_round
three_round_best = soft_cooc(
    soft_cooc(soft_cooc(double_best, 0.54,41.0,0.089), 0.53,37.0,0.040),
    0.53, 33.0, 0.020)
for w_idf, w_2r, w_3r in [
    (0.82, 0.13, 0.05),
    (0.80, 0.13, 0.07),
    (0.84, 0.13, 0.03),
    (0.83, 0.12, 0.05),
]:
    mname = f'4way_i{int(w_idf*100):d}_r{int(w_2r*100):d}_3r{int(w_3r*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    result = w_idf * idf_result + w_2r * two_round_result + w_3r * three_round_best
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'w_idf':w_idf,'w_2r':w_2r,'w_3r':w_3r})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)
    m5_best = max(m5_best, score)

print(f"  M5 done, best={m5_best:.6f}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M6: Non-linear blends
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M6] Non-linear blends...", flush=True)

EPS_BLEND = 1e-6
for gamma in [0.85, 0.70, 0.50, 0.90, 0.95]:
    mname = f'geomean_g{int(gamma*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    result = (np.clip(idf_result, EPS_BLEND, 1) ** gamma *
              np.clip(two_round_result, EPS_BLEND, 1) ** (1 - gamma))
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'gamma': gamma})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)

# Max-element blend
for w_max in [0.03, 0.05, 0.08, 0.10]:
    mname = f'maxblend_wm{int(w_max*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    element_max = np.maximum(idf_result, two_round_result)
    result = (1 - w_max) * best_blend + w_max * element_max
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'w_max': w_max})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)

# Power post-processing
for pow_post in [1.02, 0.98, 1.05, 0.95]:
    mname = f'postpow_{int(pow_post*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    result = np.clip(best_blend, 0, 1) ** pow_post
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'pow_post': pow_post})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)

print(f"  M6 done", flush=True)

print("\n" + "="*60)
print(f"[batch127b] SUMMARY")
print(f"  Global best: {res['best']['method']} LOO={res['best']['loo_auc']:.6f}")
print(f"  M1 Nested idf→2r:     {m1_best:.6f}")
print(f"  M2 Nested 2r→idf:     {m2_best:.6f}")
print(f"  M4 Adaptive alpha:    {m4_best:.6f}")
print(f"  M5 Three-round/4way:  {m5_best:.6f}")
