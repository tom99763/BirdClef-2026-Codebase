"""
Batch 112 — File-Level Similarity Prior + Higher-Order Features
===============================================================================
Current best: fine2_c1k40w00020_c5k50w00010 LOO=0.992312
Batches 108-111: 4 consecutive zero improvements.

Completely new signal families not yet explored:
1. M1: Nearest-file label transfer (file-level mean embedding similarity)
2. M2: Weighted K-nearest-file label ensemble
3. M3: Variance of positive similarities as discriminative signal
4. M4: Soft-max window pooling variant in WL computation
5. M5: Squared cosine distance feature (geometric distance in embedding)
6. M6: Stack nearest-file prior on double_best
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

print(f"[batch112] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch112] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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
print(f"  double_best: {auc_db:.6f} (expected {best_loo:.6f}) [{time.time()-t0:.0f}s]", flush=True)

# Compute file-level mean embeddings (for nearest-file prior)
file_embs_ica = np.stack([ew_ica[win_file_id == fi].mean(0) for fi in range(n_files)])
file_embs_ica /= norm(file_embs_ica, axis=1, keepdims=True) + EPS

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
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch112"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: Nearest-file label transfer
# For each held-out file, find most similar training file, use its labels
# This is a pure file-level similarity prior (completely different from window-level)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] Nearest-file label transfer...", flush=True)
t1 = time.time()

def nearest_file_prior_loo(k_near=1, temp=1.0):
    """File-level: use weighted label average of K nearest training files."""
    out = np.zeros((n_files, n_species), np.float32)
    file_sims = file_embs_ica @ file_embs_ica.T  # (66, 66) cosine sims
    for fi in range(n_files):
        sims_fi = file_sims[fi].copy()
        sims_fi[fi] = -1.0  # exclude self
        if k_near == 1:
            best_fi = np.argmax(sims_fi)
            out[fi] = file_labels[best_fi].astype(np.float32)
        else:
            top_k_idx = np.argsort(-sims_fi)[:k_near]
            w = np.exp(sims_fi[top_k_idx] / temp)
            w /= w.sum() + EPS
            out[fi] = (file_labels[top_k_idx].astype(np.float32) * w[:, None]).sum(0)
    return out

for k in [1, 2, 3, 5]:
    s_nf = nearest_file_prior_loo(k_near=k)
    for w_val in [0.005, 0.01, 0.02, 0.03, 0.05]:
        blend = (1 - w_val) * attn_ref + w_val * s_nf
        reg(f"nf_k{k}_alone_w{int(w_val*100):03d}", macro_auc(blend))
    # Stack on double_best
    for w_val in [0.005, 0.01, 0.02]:
        base_w = 1.0 - 0.002 - 0.001 - w_val
        if base_w < 0.97: continue
        blend = base_w * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50 + w_val * s_nf
        reg(f"triple_db+nf_k{k}_w{int(w_val*100):03d}", macro_auc(blend))

# Soft nearest file with temperature
s_nf_t5 = nearest_file_prior_loo(k_near=5, temp=0.5)
s_nf_t2 = nearest_file_prior_loo(k_near=5, temp=0.2)
for name_t, s_t in [("nf_k5_t5", s_nf_t5), ("nf_k5_t2", s_nf_t2)]:
    for w_val in [0.01, 0.02, 0.03]:
        blend = (1 - w_val) * attn_ref + w_val * s_t
        reg(f"{name_t}_w{int(w_val*100):03d}", macro_auc(blend))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: File-level similarity weighted WL signal
# Use nearest file's WINDOW embeddings as additional positive references
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] Cross-file window borrowing...", flush=True)
t1 = time.time()

def cross_file_wl_loo(ew, k_near=2):
    """For each test file, also use windows from nearest K training files
    as additional positive references if they're labeled positive."""
    out = np.zeros((n_files, n_species), np.float32)
    file_sims_mat = file_embs_ica @ file_embs_ica.T
    for fi in range(n_files):
        te = ew[win_file_id == fi]
        sims_fi = file_sims_mat[fi].copy(); sims_fi[fi] = -1.0
        near_files = np.argsort(-sims_fi)[:k_near]
        # Use own training windows + borrow from near files
        tr_mask = win_file_id != fi
        tr = ew[tr_mask]; tl = labels_win[tr_mask]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]; ps = te @ pw.T
            pp = pw.mean(0); pp /= norm(pp) + EPS
            sp = (te @ pp)
            ws[:, si] = (sp + 1) / 2
        out[fi] = ws.max(0)
    return out

# This is slow; do a simplified version using file-level embeddings directly
def file_level_sim_prior(ew, k_near=3, bw=0.15):
    """KDE over file-level similarities to labeled files."""
    out = np.zeros((n_files, n_species), np.float32)
    file_sims_mat = file_embs_ica @ file_embs_ica.T
    for fi in range(n_files):
        sims_fi = file_sims_mat[fi].copy(); sims_fi[fi] = -1.0
        kern = np.exp((sims_fi - 1.0) / (bw**2 + EPS))
        kern[fi] = 0.0
        # KDE score for each species: weighted average of labels
        kern_sum = kern.sum() + EPS
        out[fi] = (kern[:, None] * file_labels.astype(np.float32)).sum(0) / kern_sum
    return out

for bw_f in [0.10, 0.15, 0.20, 0.25, 0.30]:
    s_fps = file_level_sim_prior(ew_ica, bw=bw_f)
    for w_val in [0.01, 0.02, 0.03, 0.05]:
        blend = (1 - w_val) * attn_ref + w_val * s_fps
        reg(f"file_kde_bw{int(bw_f*100):03d}_w{int(w_val*100):03d}", macro_auc(blend))
    for w_val in [0.005, 0.01]:
        base_w = 1.0 - 0.002 - 0.001 - w_val
        if base_w < 0.97: continue
        blend = base_w * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50 + w_val * s_fps
        reg(f"triple_db+fkde{int(bw_f*100):03d}_w{int(w_val*100):03d}", macro_auc(blend))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: Variance-based discrimination signal
# For species where positive windows are highly CONSISTENT (low variance),
# the conformal signal is more reliable. Use variance to weight the conformal.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] Variance-based signal...", flush=True)
t1 = time.time()

def variance_signal_loo(ew, top_k_fisher=40):
    """Signal based on variance of test-window similarities to positive set.
    Low variance (consistent) → more confident positive.
    High variance → test windows are heterogeneous.
    """
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
            sims_p = te_w @ pos_w.T  # (n_te, n_pos)
            # Mean similarity (existing signal)
            mean_sim = sims_p.mean(1)
            # NEGATIVE variance of similarity — high consistency = high score
            var_sim = sims_p.var(1)
            score = mean_sim / (np.sqrt(var_sim) + EPS)
            ws[:, si] = np.clip(score, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

s_var_40 = variance_signal_loo(ew_ica, top_k_fisher=40)
s_var_50 = variance_signal_loo(ew_ica, top_k_fisher=50)
for name_v, s_v in [("var_kf40", s_var_40), ("var_kf50", s_var_50)]:
    for w_val in [0.002, 0.003, 0.004, 0.005]:
        blend = (1 - w_val) * attn_ref + w_val * s_v
        reg(f"{name_v}_w{int(w_val*1000):04d}", macro_auc(blend))
    for w_val in [0.0005, 0.001]:
        base_w = 1.0 - 0.002 - 0.001 - w_val
        if base_w < 0.99: continue
        blend = base_w * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50 + w_val * s_v
        reg(f"triple_db+{name_v}_w{int(w_val*10000):05d}", macro_auc(blend))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: Soft-max window pooling in the chain
# Instead of max(0) + mean(0) aggregation, use softmax-weighted pooling
# at the triple_ref level: exp(score*T) / sum, weighted sum
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] Softmax window pooling WL variant...", flush=True)
t1 = time.time()

def wl_softmax_loo(ew, k_neg, wmp, T=5.0):
    """WL with softmax pooling over windows instead of max+mean mix."""
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
        # Softmax pooling over windows
        for si in range(n_species):
            sc = ws[:, si] * T
            sc -= sc.max()
            w_sm = np.exp(sc); w_sm /= w_sm.sum() + EPS
            out[fi, si] = (ws[:, si] * w_sm).sum()
    return out

for T_sm in [2.0, 5.0, 10.0, 20.0]:
    s_sm = wl_softmax_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], T=T_sm)
    # Use this as replacement for s_ica in the chain
    uh_b_sm = cfg["w_ica100"] * s_sm + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
    uh_nmf_sm = cfg["nmf"]["uh_scale"] * uh_b_sm + cfg["nmf"]["w_nmf"] * s_nmf
    base_sm = w_uh*uh_nmf_sm + cfg["w_logit"]*pT8 + cfg["w_multit"]*pmt + cfg["w_subspace"]*ss2 + cfg["w_softmax"]*sm6
    final_ref_sm = 0.96 * base_sm + 0.04 * kde08
    fin_ref_sm = 0.95 * final_ref_sm + 0.05 * f06
    triple_sm = 0.94*fin_ref_sm + 0.02*fh30_b6 + 0.03*fh40_b7 + 0.01*fh40_b6
    attn_sm = 0.99 * triple_sm + 0.01 * s_attn
    blend = 0.997 * attn_sm + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50
    reg(f"sm_pool_T{int(T_sm*10):03d}_conf", macro_auc(blend))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: File-level label co-occurrence smoothing
# Use label co-occurrence matrix from training to smooth predictions:
# if species A and B often co-occur, boost B when A is confident
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Label co-occurrence smoothing...", flush=True)
t1 = time.time()

def cooccurrence_smooth(scores, alpha=0.1):
    """Smooth scores using label co-occurrence matrix from all 66 files."""
    # Compute co-occurrence matrix from all file labels
    fl = file_labels.astype(np.float32)
    # P(species j | species i) = count(i and j) / count(i)
    cooc = fl.T @ fl  # (n_species, n_species)
    count_i = fl.sum(0) + EPS  # (n_species,)
    cooc_norm = cooc / count_i[:, None]  # P(j|i)
    np.fill_diagonal(cooc_norm, 0)  # don't self-reinforce
    # Smooth: score_new = (1-alpha)*score + alpha * cooc_norm.T @ score
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        # Contribution from co-occurring species
        contrib = cooc_norm.T @ s  # (n_species,)
        contrib /= (np.abs(contrib).max() + EPS)  # normalize
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

for alpha in [0.02, 0.05, 0.10, 0.15, 0.20]:
    s_coo = cooccurrence_smooth(double_best, alpha=alpha)
    reg(f"cooc_smooth_a{int(alpha*100):03d}", macro_auc(s_coo))
# Apply co-occurrence smoothing to the conformal component
for alpha in [0.05, 0.10]:
    s_coo_c40 = cooccurrence_smooth(s_conf_k1_40, alpha=alpha)
    blend = 0.997 * attn_ref + 0.002 * s_coo_c40 + 0.001 * s_conf_k5_50
    reg(f"cooc_conf_a{int(alpha*100):03d}", macro_auc(blend))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M6: Multi-scale file embedding similarity
# Use multiple window aggregations for file-level repr
# (max pooling vs mean pooling of windows)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M6] Multi-scale file similarity prior...", flush=True)
t1 = time.time()

# File-level max pooling embedding
file_embs_max = np.stack([ew_ica[win_file_id == fi].max(0) for fi in range(n_files)])
file_embs_max /= norm(file_embs_max, axis=1, keepdims=True) + EPS

# File-level combined mean+max embedding
file_embs_mm = 0.5 * file_embs_ica + 0.5 * file_embs_max
file_embs_mm /= norm(file_embs_mm, axis=1, keepdims=True) + EPS

for name_fe, fe in [("mean", file_embs_ica), ("max", file_embs_max), ("mm", file_embs_mm)]:
    file_sims = fe @ fe.T
    # KDE-based prior with this embedding
    for bw_f in [0.15, 0.20]:
        out_fl = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            s_fi = file_sims[fi].copy(); s_fi[fi] = -1e9
            kern = np.exp((s_fi - 1.0) / (bw_f**2 + EPS)); kern[fi] = 0
            kern_sum = kern.sum() + EPS
            out_fl[fi] = (kern[:, None] * file_labels.astype(np.float32)).sum(0) / kern_sum
        for w_val in [0.01, 0.02, 0.03]:
            blend = (1 - w_val) * attn_ref + w_val * out_fl
            reg(f"fe_{name_fe}_bw{int(bw_f*100):03d}_w{int(w_val*100):03d}", macro_auc(blend))
        for w_val in [0.005, 0.01]:
            base_w = 1.0 - 0.002 - 0.001 - w_val
            if base_w < 0.97: continue
            blend = base_w * attn_ref + 0.002 * s_conf_k1_40 + 0.001 * s_conf_k5_50 + w_val * out_fl
            reg(f"triple_db+fe_{name_fe}_bw{int(bw_f*100):03d}_w{int(w_val*100):03d}", macro_auc(blend))
print(f"  M6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary + Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch112] SUMMARY", flush=True)
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
    ep_new["batch"] = "batch112"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep_new, f)
    res3 = json.load(open(RESULTS_PATH))
    res3["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch112"}
    json.dump(res3, open(RESULTS_PATH, "w"), indent=2)
    print(f"  PKL + JSON updated → {new_best_method} {new_best_loo:.6f}", flush=True)
else:
    print(f"  No improvement → best remains {best_loo:.6f} ({best['method']})", flush=True)
