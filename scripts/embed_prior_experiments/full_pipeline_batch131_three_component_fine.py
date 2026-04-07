"""
batch131 — Fine-tune three-component blend + explore 4th component
===============================================================================
Current best: three3w_b80_i15_s5 LOO=0.994782
Formula: 0.80×3way_best + 0.15×3way_ica_ens + 0.05×3way_std_ens

Strategy:
M1: Fine sweep three-component ratios (small steps around 0.80/0.15/0.05)
M2: Try M2-best ICA composition (ds65/m10/a25) in three-component blend
M3: Vary STD ensemble composition in three-component blend
M4: Add 4th component (PCA-based)
M5: Different ICA+STD ensemble blends as 2nd component
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.covariance import LedoitWolf
from numpy.linalg import norm
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-8
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

with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

ew_ica  = ep["emb_win_ica_norm"]
ew_pca  = ep["emb_win_pca_norm"]
ew_std  = ep["emb_win_std_norm"]
ew_nmf  = ep["emb_win_nmf_norm"]
file_labels = ep["file_labels"]
cfg = ep["config"]

print(f"[batch131] ICA{ew_ica.shape}, n_files={n_files}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
tried = {e["method"] for e in res["experiments"]}
print(f"[batch131] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

def macro_auc(s):
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            aucs.append(roc_auc_score(y, s[:, si]))
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, config_dict=None):
    global best_loo
    delta = score - best_loo
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 131}
    res["experiments"].append(entry)
    if score > best_loo + 1e-7:
        best_loo = score
        res["best"] = {"method": mname, "loo_auc": float(score)}
        with open(MODEL_PATH, "rb") as f:
            ep_up = pickle.load(f)
        ep_up["method"] = mname
        ep_up["loo_auc"] = float(score)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(ep_up, f)
        print(f"  [SAVED] New best PKL!", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return delta

# ─── Core functions ────────────────────────────────────────────────────────────
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

def mahal_ica_loo(ew, k=5, w_max_agg=0.8):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        lw = LedoitWolf().fit(tr); VI = lw.precision_
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]
            XV_pos = pos @ VI; diag_pos = (XV_pos * pos).sum(1)
            XV_te = te @ VI; diag_te = (XV_te * te).sum(1)
            cross = te @ (pos @ VI).T
            d2 = diag_te[:, None] - 2 * cross + diag_pos[None, :]
            d2 = np.clip(d2, 0, None)
            k2 = min(k, len(pos)); idx = np.argsort(d2, axis=1)[:, :k2]
            sim = 1.0 / (1.0 + d2[np.arange(len(te))[:, None], idx].mean(1))
            ws[:, si] = sim
        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

def attn_ica_loo(ew, tau=0.3, w_max_agg=0.8):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        tl_logit = logit_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]
            pos_logit = tl_logit[pm, si]
            attn = 1.0 / (1.0 + np.exp(-np.clip(pos_logit / tau, -10, 10)))
            attn /= (attn.sum() + EPS)
            sims = te @ pos.T
            ws[:, si] = (sims * attn[None, :]).sum(1)
        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

def wl_dual_softmax(ew, tau=0.3, w_max_agg=0.8):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 1.0; continue
            pos = tr[pm]; nw = tr[nm] if nm.any() else tr[~pm]
            k2 = min(5, len(nw))
            pos_sims = te @ pos.T
            fwd_sm = pos_sims / tau; fwd_sm -= fwd_sm.max(1, keepdims=True)
            fwd_e = np.exp(fwd_sm); fwd = fwd_e / (fwd_e.sum(1, keepdims=True) + EPS)
            best_pos_idx = fwd.argmax(1)
            forward_score = fwd[np.arange(len(te)), best_pos_idx]
            backward_score = np.zeros(len(te), np.float32)
            all_others = np.concatenate([te, nw[:k2]], axis=0) if k2 > 0 else te
            for ti in range(len(te)):
                pi = best_pos_idx[ti]
                bwd_sims = pos[pi] @ all_others.T
                bwd = np.exp(pos_sims[ti, pi] / tau - np.log(np.exp(bwd_sims / tau).sum() + EPS))
                backward_score[ti] = bwd
            ws[:, si] = np.sqrt(forward_score * backward_score + EPS)
        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

# ─── Build full chain ──────────────────────────────────────────────────────────
print("\nPre-computing full chain...", flush=True)
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

# ─── ICA/STD ensemble components ──────────────────────────────────────────────
print("\nComputing ensemble components...", flush=True)
t0 = time.time()
ref_ds_ica   = wl_dual_softmax(ew_ica, tau=0.3, w_max_agg=0.8)
ref_mahal    = mahal_ica_loo(ew_ica, k=5, w_max_agg=0.80)
ref_attn_ica = attn_ica_loo(ew_ica, tau=0.3, w_max_agg=0.80)
ref_ds_std   = wl_dual_softmax(ew_std, tau=0.3, w_max_agg=0.8)
ref_attn_std = attn_ica_loo(ew_std, tau=0.3, w_max_agg=0.80)
ref_ds_pca   = wl_dual_softmax(ew_pca, tau=0.3, w_max_agg=0.8)

ica_ens_ref  = 0.6 * ref_ds_ica + 0.2 * ref_mahal + 0.2 * ref_attn_ica  # batch129 best
std_ens_ref  = 0.7 * ref_ds_std + 0.3 * ref_attn_std                     # batch130 M4
print(f"  ica_ens_ref: {macro_auc(ica_ens_ref):.6f}  std_ens_ref: {macro_auc(std_ens_ref):.6f} [{time.time()-t0:.0f}s]", flush=True)

# ─── Co-occurrence infrastructure ─────────────────────────────────────────────
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

def apply_3way(s):
    return 0.85 * idf_cooc(s) + 0.15 * two_round(s)

# Pre-compute 3way signals
print("\nPre-computing 3way signals...", flush=True)
current_3way  = apply_3way(double_best)
ica_3way_ref  = apply_3way(ica_ens_ref)
std_3way_ref  = apply_3way(std_ens_ref)
pca_3way_ref  = apply_3way(ref_ds_pca)
print(f"  3way_best: {macro_auc(current_3way):.6f}", flush=True)
print(f"  3way_ica: {macro_auc(ica_3way_ref):.6f}", flush=True)
print(f"  3way_std: {macro_auc(std_3way_ref):.6f}", flush=True)
print(f"  3way_pca: {macro_auc(pca_3way_ref):.6f}", flush=True)

# Verify batch130 best
check_130 = 0.80 * current_3way + 0.15 * ica_3way_ref + 0.05 * std_3way_ref
print(f"  batch130_best check: {macro_auc(check_130):.6f} (expected 0.994782)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# M1: Fine sweep three-component ratios (small steps)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[M1] Fine sweep three-component ratios...", flush=True)

# Parameterize as (w_best, w_ica), w_std = 1 - w_best - w_ica
for w_best in np.arange(0.76, 0.84, 0.01):
    for w_ica in np.arange(0.12, 0.22, 0.01):
        w_best_r = round(float(w_best), 2)
        w_ica_r  = round(float(w_ica), 2)
        w_std_r  = round(1 - w_best_r - w_ica_r, 2)
        if w_std_r < 0.01 or w_std_r > 0.15: continue
        mname = f'fine3w_b{int(w_best_r*100):d}_i{int(w_ica_r*100):d}_s{int(w_std_r*100):d}'
        if mname in tried:
            continue
        result = w_best_r * current_3way + w_ica_r * ica_3way_ref + w_std_r * std_3way_ref
        score  = macro_auc(result)
        delta  = save_result(mname, score, {'w_best': w_best_r, 'w_ica': w_ica_r, 'w_std': w_std_r})
        flag   = ' ← NEW BEST!' if score > best_loo else ''
        print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)

print("  M1 done", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# M2: Try M2-best ICA composition (ds65/m10/a25) in three-component
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[M2] Alternative ICA composition in 3-component...", flush=True)

# Best ICA composition from batch130 M2: ds65_m10_a25 = 0.994769
ica_ens_alt = 0.65 * ref_ds_ica + 0.10 * ref_mahal + 0.25 * ref_attn_ica
ica_3way_alt = apply_3way(ica_ens_alt)
print(f"  ica_ens_alt (ds65/m10/a25): {macro_auc(ica_ens_alt):.6f} → 3way: {macro_auc(ica_3way_alt):.6f}", flush=True)

for w_best, w_ica, w_std in [
    (0.80, 0.15, 0.05),
    (0.82, 0.13, 0.05),
    (0.83, 0.12, 0.05),
    (0.79, 0.16, 0.05),
    (0.79, 0.15, 0.06),
    (0.80, 0.14, 0.06),
]:
    mname = f'alt3w_b{int(w_best*100):d}_i{int(w_ica*100):d}_s{int(w_std*100):d}_ds65m10a25'
    if mname in tried:
        continue
    result = w_best * current_3way + w_ica * ica_3way_alt + w_std * std_3way_ref
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'w_best': w_best, 'w_ica': w_ica, 'w_std': w_std, 'ica_comp': 'ds65m10a25'})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)

print("  M2 done", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# M3: Vary STD ensemble composition
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[M3] Vary STD ensemble composition...", flush=True)

std_variants = {
    'ds80a20': 0.8 * ref_ds_std + 0.2 * ref_attn_std,
    'ds90a10': 0.9 * ref_ds_std + 0.1 * ref_attn_std,
    'ds_only': ref_ds_std,
    'ds60a40': 0.6 * ref_ds_std + 0.4 * ref_attn_std,
}
for svname, std_v in std_variants.items():
    std_3way_v = apply_3way(std_v)
    for w_best, w_ica, w_std in [(0.80, 0.15, 0.05), (0.80, 0.14, 0.06), (0.81, 0.14, 0.05)]:
        mname = f'std3w_{svname}_b{int(w_best*100):d}_i{int(w_ica*100):d}_s{int(w_std*100):d}'
        if mname in tried:
            continue
        result = w_best * current_3way + w_ica * ica_3way_ref + w_std * std_3way_v
        score  = macro_auc(result)
        delta  = save_result(mname, score, {'std_comp': svname, 'w_best': w_best, 'w_ica': w_ica, 'w_std': w_std})
        flag   = ' ← NEW BEST!' if score > best_loo else ''
        print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)

print("  M3 done", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# M4: Add 4th component (PCA-based)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[M4] Four-component blend...", flush=True)

for w_best, w_ica, w_std, w_pca in [
    (0.78, 0.14, 0.05, 0.03),
    (0.78, 0.13, 0.05, 0.04),
    (0.79, 0.13, 0.05, 0.03),
    (0.77, 0.14, 0.05, 0.04),
    (0.80, 0.12, 0.05, 0.03),
    (0.78, 0.15, 0.04, 0.03),
]:
    mname = f'four3w_b{int(w_best*100):d}_i{int(w_ica*100):d}_s{int(w_std*100):d}_p{int(w_pca*100):d}'
    if mname in tried:
        continue
    result = w_best * current_3way + w_ica * ica_3way_ref + w_std * std_3way_ref + w_pca * pca_3way_ref
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'w_best': w_best, 'w_ica': w_ica, 'w_std': w_std, 'w_pca': w_pca})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)

print("  M4 done", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# M5: ICA+STD combined as single 2nd component
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[M5] Combined ICA+STD ensemble as 2nd component...", flush=True)

for w_i, w_s in [(0.7, 0.3), (0.75, 0.25), (0.6, 0.4), (0.8, 0.2)]:
    combined = w_i * ica_ens_ref + w_s * std_ens_ref
    combined_3way = apply_3way(combined)
    auc_comb = macro_auc(combined_3way)
    for w_best in [0.84, 0.85, 0.83, 0.82]:
        w_c = round(1 - w_best, 2)
        mname = f'comb3w_b{int(w_best*100):d}_ica{int(w_i*10):d}std{int(w_s*10):d}'
        if mname in tried:
            continue
        result = w_best * current_3way + w_c * combined_3way
        score  = macro_auc(result)
        delta  = save_result(mname, score, {'w_best': w_best, 'w_c': w_c, 'w_i_in_c': w_i, 'w_s_in_c': w_s})
        flag   = ' ← NEW BEST!' if score > best_loo else ''
        print(f'  {mname} ({auc_comb:.4f}): {score:.6f} {delta:+.6f}{flag}', flush=True)

print("  M5 done", flush=True)

print("\n" + "="*60, flush=True)
print(f"[batch131] DONE. Best: {res['best']['method']} LOO={res['best']['loo_auc']:.6f}", flush=True)
