"""
Batch 109 — Structural New Signals: Multi-kf Ensemble, Label Propagation, MMD
===============================================================================
Current best: fine2_c1k40w00020_c5k50w00010 LOO=0.992312
Formula: 0.997*attn_ref + 0.002*conf_k1_kf40 + 0.001*conf_k5_kf50

Batch108 confirmed ceiling for:
- Sub-grid w sweep (no improvement with 0.0001 precision)
- Attention-weighted conformal (same as standard)
- Min-similarity aggregation (weaker)
- Adaptive kf (weaker)
- Geometric mean scoring (weaker)

This batch — fundamentally different signal types:
1. M1: Multi-kf conformal ensemble (average of kf=20,30,40,50,60)
2. M2: Rank-based conformal (rank position among all training windows)
3. M3: Label propagation / graph diffusion (k-NN graph over all windows)
4. M4: MMD-based signal (max mean discrepancy between test and positive class)
5. M5: Weighted Fisher conformal (weight kf by explained variance per species)
6. M6: Species-conditional conformal (calibrate per-species weight by LOO performance)
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

print(f"[batch109] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch109] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

def conformal_score_loo(ew, top_k_fisher=40, k_nn=1, mode="ratio"):
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

# ── Pre-compute full chain → attn_ref ────────────────────────────────────────
print("Pre-computing chain...", flush=True)
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
w_uh = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur  = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
final_ref = 0.96 * base_cur + 0.04 * kde08
f06 = fisher_kde_loo(ew_ica, bw=0.06)
fin_ref = (1 - 0.05) * final_ref + 0.05 * f06
fh30_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
fh40_b7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
fh40_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=40)
triple_ref = (1-0.02-0.03-0.01)*fin_ref + 0.02*fh30_b6 + 0.03*fh40_b7 + 0.01*fh40_b6
s_attn = attn_knn_fisher_loo(ew_ica, top_k_fisher=40, bw=0.07, attn_T=6.0)
attn_ref = 0.99 * triple_ref + 0.01 * s_attn

s_conf_k1_40 = conformal_score_loo(ew_ica, top_k_fisher=40, k_nn=1, mode="ratio")
s_conf_k5_50 = conformal_score_loo(ew_ica, top_k_fisher=50, k_nn=5, mode="ratio")
double_best = (1 - 0.002 - 0.001) * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
auc_db = macro_auc(double_best)
print(f"  double_best: {auc_db:.6f} (expected {best_loo:.6f}) [{time.time()-t0:.0f}s]", flush=True)

results = {}
new_best_loo = best_loo
new_best_method = None

def reg(name, auc):
    global new_best_loo, new_best_method
    delta = auc - best_loo
    if auc > new_best_loo:
        new_best_loo = auc
        new_best_method = name
    mark = " *** NEW BEST ***" if auc > best_loo else (" (near-best)" if auc > best_loo - 0.0003 else "")
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch109"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: Multi-kf ensemble conformal (average of kf=20,30,40,50,60)
# Ensemble of conformal signals captures different scale Fisher directions
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] Multi-kf conformal ensemble...", flush=True)
t1 = time.time()
kf_list = [20, 30, 40, 50, 60]
conf_kf_signals = {}
for kf in kf_list:
    conf_kf_signals[kf] = conformal_score_loo(ew_ica, top_k_fisher=kf, k_nn=1, mode="ratio")

# Equal-weight average of all 5
s_multi_kf_mean = np.mean([conf_kf_signals[kf] for kf in kf_list], axis=0)
# Linearly decaying weight (more weight to kf=40 which is best)
w_kf = np.array([0.1, 0.15, 0.5, 0.15, 0.1])  # center-weighted
s_multi_kf_cwt = sum(w_kf[i] * conf_kf_signals[kf_list[i]] for i in range(5))
# Top-heavy: kf=40,50 dominate
s_multi_kf_top = 0.5 * conf_kf_signals[40] + 0.3 * conf_kf_signals[50] + 0.2 * conf_kf_signals[30]

for name_e, s_e in [("mkf_mean", s_multi_kf_mean), ("mkf_cwt", s_multi_kf_cwt), ("mkf_top", s_multi_kf_top)]:
    for w_val in [0.001, 0.002, 0.003, 0.004]:
        blend = (1 - w_val) * attn_ref + w_val * s_e
        reg(f"multikf_{name_e}_w{int(w_val*1000):04d}", macro_auc(blend))
    # Stack on double_best
    for w_val in [0.0005, 0.001]:
        base_w = 1.0 - 0.002 - 0.001 - w_val
        if base_w < 0.99: continue
        blend = base_w * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50 + w_val * s_e
        reg(f"triple_db+{name_e}_w{int(w_val*10000):05d}", macro_auc(blend))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: Rank-based conformal
# Use rank of positive KNN similarity among all training similarities
# More robust to absolute scale differences
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Rank-based conformal...", flush=True)
t1 = time.time()

def conformal_rank_loo(ew, top_k_fisher=40, k_nn=1):
    """Rank-based: score = percentile rank of knn_pos among all train similarities."""
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
            all_w = np.concatenate([pos_w, neg_w], axis=0)
            k_p = min(k_nn, len(pos_w)); k_n = min(k_nn, len(neg_w))
            sims_p = te_w @ pos_w.T; sims_n = te_w @ neg_w.T
            sims_all = te_w @ all_w.T   # all training sims
            knn_pos = np.sort(sims_p, axis=1)[:, -k_p:].mean(1)
            knn_neg = np.sort(sims_n, axis=1)[:, -k_n:].mean(1)
            n_all = sims_all.shape[1]
            # Rank of knn_pos among all training sims (higher = better)
            rank_pos = np.array([(knn_pos[j] > sims_all[j]).sum() / n_all for j in range(len(te))])
            rank_neg = np.array([(knn_neg[j] > sims_all[j]).sum() / n_all for j in range(len(te))])
            score = rank_pos - rank_neg
            ws[:, si] = np.clip(score, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

s_rank_k1_40 = conformal_rank_loo(ew_ica, top_k_fisher=40, k_nn=1)
s_rank_k1_50 = conformal_rank_loo(ew_ica, top_k_fisher=50, k_nn=1)
for name_r, s_r in [("rank_k1kf40", s_rank_k1_40), ("rank_k1kf50", s_rank_k1_50)]:
    for w_val in [0.002, 0.003, 0.004, 0.005]:
        blend = (1 - w_val) * attn_ref + w_val * s_r
        reg(f"conf_{name_r}_w{int(w_val*1000):04d}", macro_auc(blend))
    for w_val in [0.0005, 0.001]:
        base_w = 1.0 - 0.002 - 0.001 - w_val
        if base_w < 0.99: continue
        blend = base_w * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50 + w_val * s_r
        reg(f"triple_db+{name_r}_w{int(w_val*10000):05d}", macro_auc(blend))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: Label propagation via k-NN graph
# Build k-NN graph over all training windows, propagate labels to test windows
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Label propagation (k-NN graph)...", flush=True)
t1 = time.time()

def label_prop_loo(ew, k_graph=10, alpha=0.5, top_k_fisher=40):
    """Label propagation: F = alpha * W * F + (1-alpha) * Y, iterated."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
            # Fisher dim selection
            mu_p = pos.mean(0); mu_n = neg.mean(0)
            var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            # All windows in Fisher subspace
            all_w = np.concatenate([tr, te], axis=0)
            all_w_proj = all_w * w_dim[None, :]; all_w_proj /= norm(all_w_proj, axis=1, keepdims=True) + EPS
            n_tr = len(tr); n_te = len(te); n_total = n_tr + n_te
            # Build similarity matrix
            sims = all_w_proj @ all_w_proj.T   # (n_total, n_total)
            # Row-normalize to get transition matrix (keep top-k only)
            W = np.zeros_like(sims)
            for i in range(n_total):
                top_k_idx = np.argsort(-sims[i])[:k_graph + 1]
                top_k_idx = top_k_idx[top_k_idx != i][:k_graph]
                W[i, top_k_idx] = sims[i, top_k_idx]
            # Row-normalize
            row_sum = W.sum(1, keepdims=True) + EPS
            W /= row_sum
            # Initial labels: training windows get their labels, test windows get 0
            y = np.zeros(n_total, np.float32)
            y[:n_tr] = (tl[:, si] > 0.5).astype(np.float32)
            # Iterate label propagation (5 steps)
            f = y.copy()
            for _ in range(5):
                f = alpha * (W @ f) + (1 - alpha) * y
            ws[:, si] = np.clip(f[n_tr:], 0, 1)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

s_lp_k10 = label_prop_loo(ew_ica, k_graph=10, alpha=0.5, top_k_fisher=40)
s_lp_k5  = label_prop_loo(ew_ica, k_graph=5,  alpha=0.5, top_k_fisher=40)
s_lp_k10_a7 = label_prop_loo(ew_ica, k_graph=10, alpha=0.7, top_k_fisher=40)
for name_lp, s_lp in [("lp_k10_a5", s_lp_k10), ("lp_k5_a5", s_lp_k5), ("lp_k10_a7", s_lp_k10_a7)]:
    for w_val in [0.002, 0.003, 0.004, 0.005]:
        blend = (1 - w_val) * attn_ref + w_val * s_lp
        reg(f"{name_lp}_w{int(w_val*1000):04d}", macro_auc(blend))
    for w_val in [0.001, 0.002]:
        base_w = 1.0 - 0.002 - 0.001 - w_val
        if base_w < 0.99: continue
        blend = base_w * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50 + w_val * s_lp
        reg(f"triple_db+{name_lp}_w{int(w_val*10000):05d}", macro_auc(blend))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: MMD-based signal (maximum mean discrepancy)
# Measures if test windows come from same distribution as positive training
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] MMD-based signal...", flush=True)
t1 = time.time()

def mmd_loo(ew, top_k_fisher=40, bw=0.1):
    """MMD score: similarity between test and positive class distribution."""
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
            # For each test window, compute kernel similarity to positive set
            # k(x,y) = exp((sim-1)/bw^2), so k(x,x)=1, k(x,y) in (0,1] for cosine
            sims_te_pos = te_w @ pos_w.T
            sims_te_neg = te_w @ neg_w.T
            sims_pos_pos = pos_w @ pos_w.T
            # MMD per test window: E[k(te,pos)] - E[k(te,neg)] - 0.5*E[k(pos,pos)]
            kern_te_pos = np.exp((sims_te_pos - 1.0) / (bw**2 + EPS))
            kern_te_neg = np.exp((sims_te_neg - 1.0) / (bw**2 + EPS))
            kern_pp = np.exp((sims_pos_pos - 1.0) / (bw**2 + EPS))
            score = kern_te_pos.mean(1) - kern_te_neg.mean(1) - 0.5 * (kern_pp.mean() - 1.0)
            ws[:, si] = np.clip(score, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

s_mmd_40_10 = mmd_loo(ew_ica, top_k_fisher=40, bw=0.1)
s_mmd_40_07 = mmd_loo(ew_ica, top_k_fisher=40, bw=0.07)
s_mmd_50_10 = mmd_loo(ew_ica, top_k_fisher=50, bw=0.1)
for name_m, s_m in [("mmd_kf40_bw10", s_mmd_40_10), ("mmd_kf40_bw07", s_mmd_40_07), ("mmd_kf50_bw10", s_mmd_50_10)]:
    for w_val in [0.002, 0.003, 0.004, 0.005]:
        blend = (1 - w_val) * attn_ref + w_val * s_m
        reg(f"{name_m}_w{int(w_val*1000):04d}", macro_auc(blend))
    for w_val in [0.0005, 0.001]:
        base_w = 1.0 - 0.002 - 0.001 - w_val
        if base_w < 0.99: continue
        blend = base_w * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50 + w_val * s_m
        reg(f"triple_db+{name_m}_w{int(w_val*10000):05d}", macro_auc(blend))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: Cross-space conformal voting
# ICA conformal + PCA conformal (geometric mean of scores, different inductive bias)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Cross-space conformal voting...", flush=True)
t1 = time.time()
s_conf_pca_k1_40 = conformal_score_loo(ew_pca, top_k_fisher=40, k_nn=1, mode="ratio")
# Geometric mean of ICA and PCA conformal (both must agree)
s_geom_vote = np.sqrt(np.clip(s_conf_k1_40 * s_conf_pca_k1_40, 0, None))
# Harmonic mean (penalizes disagreement more)
s_harm_vote = 2 * s_conf_k1_40 * s_conf_pca_k1_40 / (s_conf_k1_40 + s_conf_pca_k1_40 + EPS)
# Min (conservative voting)
s_min_vote  = np.minimum(s_conf_k1_40, s_conf_pca_k1_40)

for name_v, s_v in [("geom_ica_pca", s_geom_vote), ("harm_ica_pca", s_harm_vote), ("min_ica_pca", s_min_vote)]:
    for w_val in [0.002, 0.003, 0.004]:
        blend = (1 - w_val) * attn_ref + w_val * s_v
        reg(f"vote_{name_v}_w{int(w_val*1000):04d}", macro_auc(blend))
    for w_val in [0.0005, 0.001]:
        base_w = 1.0 - 0.002 - 0.001 - w_val
        if base_w < 0.99: continue
        blend = base_w * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50 + w_val * s_v
        reg(f"triple_db+{name_v}_w{int(w_val*10000):05d}", macro_auc(blend))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M6: Squished conformal (apply sigmoid normalization instead of linear clip)
# Different from tanh — uses relative ranking within the score distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M6] Per-file normalized conformal...", flush=True)
t1 = time.time()

def conformal_zscore_loo(ew, top_k_fisher=40, k_nn=1):
    """Conformal with z-score normalization per test file (mean/std of scores)."""
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
            # Z-score normalize within this test file
            score_mean = score.mean()
            score_std  = score.std() + EPS
            score_z = (score - score_mean) / score_std
            ws[:, si] = np.clip(score_z, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

s_zscore_k1_40 = conformal_zscore_loo(ew_ica, top_k_fisher=40, k_nn=1)
s_zscore_k1_50 = conformal_zscore_loo(ew_ica, top_k_fisher=50, k_nn=1)
for name_z, s_z in [("zscore_k1kf40", s_zscore_k1_40), ("zscore_k1kf50", s_zscore_k1_50)]:
    for w_val in [0.002, 0.003, 0.004, 0.005]:
        blend = (1 - w_val) * attn_ref + w_val * s_z
        reg(f"conf_{name_z}_w{int(w_val*1000):04d}", macro_auc(blend))
    for w_val in [0.0005, 0.001]:
        base_w = 1.0 - 0.002 - 0.001 - w_val
        if base_w < 0.99: continue
        blend = base_w * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50 + w_val * s_z
        reg(f"triple_db+{name_z}_w{int(w_val*10000):05d}", macro_auc(blend))
print(f"  M6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary + Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch109] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

top10 = sorted(results.values(), key=lambda x: -x["loo_auc"])[:10]
print("  Top-10 this batch:", flush=True)
for r2 in top10:
    delta = r2["loo_auc"] - best_loo
    mark = " ***" if r2["loo_auc"] > best_loo else ""
    print(f"    {r2['method']}: {r2['loo_auc']:.6f} ({delta:+.6f}){mark}", flush=True)

if new_best_method:
    print(f"\n  NEW BEST: {new_best_method} LOO={new_best_loo:.6f} (+{new_best_loo-best_loo:.6f})", flush=True)

res2 = json.load(open(RESULTS_PATH))
if isinstance(res2.get("experiments"), list):
    res2["experiments"].extend(list(results.values()))
else:
    res2["experiments"].update(results)
json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"  Saved {len(results)} results to JSON", flush=True)

if new_best_method and new_best_loo > best_loo:
    ep_new = copy.deepcopy(ep)
    ep_new["method"] = new_best_method
    ep_new["loo_auc"] = new_best_loo
    ep_new["batch"] = "batch109"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep_new, f)
    res3 = json.load(open(RESULTS_PATH))
    res3["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch109"}
    json.dump(res3, open(RESULTS_PATH, "w"), indent=2)
    print(f"  PKL + JSON updated → {new_best_method} {new_best_loo:.6f}", flush=True)
else:
    print(f"  No improvement → best remains {best_loo:.6f} ({best['method']})", flush=True)
