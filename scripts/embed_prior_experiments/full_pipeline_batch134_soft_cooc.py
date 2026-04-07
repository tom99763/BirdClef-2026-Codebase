"""
batch134 — Soft label co-occurrence + IDF parameter search
===============================================================================
Current best: altfine_b76_i16_s8 LOO=0.994870
Formula: 0.76×3way_best + 0.16×3way_ica_alt + 0.08×3way_std

New directions (all fast — operate on pre-computed LOO arrays):
 A: Soft label co-occurrence  — use file_prob_max confidence instead of binary labels
    Hypothesis: P(A|B) computed from soft predictions is more nuanced than binary
 B: IDF exponent search       — try idf_exp = 0.5, 0.6, 0.65, 0.80, 1.0, 1.25
 C: Asymmetric co-occurrence  — use P(B|A) directional, not P(A and B)
 D: Double 3way smoothing     — apply_3way(apply_3way(double_best))
 E: Adaptive alpha co-occ     — alpha scales with species rarity
 F: Blend improvements from above with current 3-component formula
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
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
win_file_id = np.zeros(sum(n_windows), np.int32)
for fi in range(n_files):
    win_file_id[file_start[fi]:file_end[fi]] = fi

with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

ew_ica = ep["emb_win_ica_norm"]
ew_pca = ep["emb_win_pca_norm"]
ew_std = ep["emb_win_std_norm"]
ew_nmf = ep["emb_win_nmf_norm"]
file_labels = ep["file_labels"]          # binary [66, 234]
file_prob_max = ep["file_prob_max"]      # batch112 preds — used for soft cooc
cfg = ep["config"]

print(f"[batch134] ICA{ew_ica.shape}, n_files={n_files}", flush=True)
res = json.load(open(RESULTS_PATH))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch134] Current best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

def macro_auc(s, fl=file_labels):
    aucs = []
    for si in range(n_species):
        y = fl[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try: aucs.append(roc_auc_score(y, s[:, si]))
            except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, config_dict=None):
    global best_loo
    if mname in tried: return score - best_loo
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 134}
    res["experiments"].append(entry)
    tried.add(mname)
    if score > best_loo + 1e-7:
        best_loo = score
        res["best"] = {"method": mname, "loo_auc": float(score)}
        with open(MODEL_PATH, "rb") as f:
            ep_up = pickle.load(f)
        ep_up["method"] = mname
        ep_up["loo_auc"] = float(score)
        # Also update file_prob_max with best available predictions
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(ep_up, f)
        print(f"  [SAVED] New best PKL!", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

# ─── Core LOO functions (from batch132) ────────────────────────────────────────
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
    from sklearn.decomposition import PCA as SklearnPCA
    ss = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_sp[win_file_id == fi]; tr = ew_sp[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; k = min(n_comp, len(pos) - 1, te.shape[1] - 1)
            if k < 1:
                pp = pos.mean(0); pp /= norm(pp) + EPS
                ws[:, si] = np.clip((te @ pp + 1) / 2, 0, 1); continue
            try:
                p = SklearnPCA(n_components=k); p.fit(pos)
                te_r = p.inverse_transform(p.transform(te))
                err = norm(te - te_r, axis=1)
                ws[:, si] = np.clip(1 - err / (norm(te, axis=1) + EPS), 0, 1)
            except: ws[:, si] = 0.5
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
            w_dim = np.zeros(len(fisher_raw), np.float32); w_dim[top_idx] = 1.0 / np.sqrt(float(top_k))
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
            w_dim = np.zeros(len(fisher_raw), np.float32); w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
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
            w_dim = np.zeros(len(fisher_raw), np.float32); w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
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
            XV_te  = te  @ VI; diag_te  = (XV_te  * te ).sum(1)
            cross  = te @ (pos @ VI).T
            d2     = diag_te[:, None] - 2 * cross + diag_pos[None, :]
            d2     = np.clip(d2, 0, None)
            k2     = min(k, len(pos)); idx = np.argsort(d2, axis=1)[:, :k2]
            sim    = 1.0 / (1.0 + d2[np.arange(len(te))[:, None], idx].mean(1))
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
            fwd_e  = np.exp(fwd_sm); fwd = fwd_e / (fwd_e.sum(1, keepdims=True) + EPS)
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

# ─── Build full chain (same as batch132) ───────────────────────────────────────
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
print(f"  double_best: {macro_auc(double_best):.6f} [{time.time()-t0:.0f}s]", flush=True)

print("\nComputing ensemble components...", flush=True)
t0 = time.time()
ref_ds_ica   = wl_dual_softmax(ew_ica, tau=0.3, w_max_agg=0.8)
ref_mahal    = mahal_ica_loo(ew_ica, k=5, w_max_agg=0.80)
ref_attn_ica = attn_ica_loo(ew_ica, tau=0.3, w_max_agg=0.80)
ref_ds_std   = wl_dual_softmax(ew_std, tau=0.3, w_max_agg=0.8)
ref_attn_std = attn_ica_loo(ew_std, tau=0.3, w_max_agg=0.80)
ica_ens_alt  = 0.65 * ref_ds_ica + 0.10 * ref_mahal + 0.25 * ref_attn_ica
std_ens_ref  = 0.70 * ref_ds_std + 0.30 * ref_attn_std
print(f"  ica_alt={macro_auc(ica_ens_alt):.6f}  std={macro_auc(std_ens_ref):.6f} [{time.time()-t0:.0f}s]", flush=True)

# ─── STANDARD co-occ (binary labels — for reference) ──────────────────────────
fl_hard = file_labels.astype(np.float32)
count_i = fl_hard.sum(0) + EPS
cooc_hard = fl_hard.T @ fl_hard
COOC_HARD = cooc_hard / count_i[:, None]
np.fill_diagonal(COOC_HARD, 0)
raw_idf   = np.log(float(n_files) / (count_i + 1.0 - EPS))
raw_idf   = np.clip(raw_idf, 0, None)
IDF_W075  = raw_idf ** 0.75; IDF_W075 /= (IDF_W075.mean() + EPS)

def soft_cooc_fn(scores, cooc_norm, center=0.53, slope=37.0, alpha=0.086, idf_w=None):
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope * (s - center), -88, 88)))
        s_gated = s * gate
        if idf_w is not None: s_gated = s_gated * idf_w
        if np.abs(s_gated).sum() < EPS: smoothed[fi] = s; continue
        contrib = cooc_norm.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def idf_cooc_fn(scores, cooc_norm=COOC_HARD, idf_w=IDF_W075, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    s_pow = np.clip(scores, 0, 1) ** 2.0
    s_cooc = soft_cooc_fn(s_pow, cooc_norm, center=center, slope=slope, alpha=alpha, idf_w=idf_w)
    return (1 - blend) * scores + blend * s_cooc

def two_round_fn(scores, cooc_norm=COOC_HARD, c1=0.54, sl1=41.0, a1=0.089, c2=0.53, sl2=37.0, a2=0.040):
    r1 = soft_cooc_fn(scores, cooc_norm, center=c1, slope=sl1, alpha=a1)
    return soft_cooc_fn(r1, cooc_norm, center=c2, slope=sl2, alpha=a2)

def apply_3way_fn(s, cooc_norm=COOC_HARD, idf_w=IDF_W075):
    return 0.85 * idf_cooc_fn(s, cooc_norm, idf_w) + 0.15 * two_round_fn(s, cooc_norm)

# Reference: binary labels 3way
print("\nPre-computing reference 3way (binary labels)...", flush=True)
current_3way  = apply_3way_fn(double_best)
ica_3way_alt  = apply_3way_fn(ica_ens_alt)
std_3way_ref  = apply_3way_fn(std_ens_ref)
print(f"  3way_best={macro_auc(current_3way):.6f}  3way_ica_alt={macro_auc(ica_3way_alt):.6f}  3way_std={macro_auc(std_3way_ref):.6f}", flush=True)
chk_ref = 0.76*current_3way + 0.16*ica_3way_alt + 0.08*std_3way_ref
print(f"  batch132_best check: {macro_auc(chk_ref):.6f} (expected 0.994870)", flush=True)

# Update file_prob_max in PKL to reflect real best
best_preds = 0.76*current_3way + 0.16*ica_3way_alt + 0.08*std_3way_ref
with open(MODEL_PATH, "rb") as f:
    ep_up = pickle.load(f)
ep_up["file_prob_max"] = best_preds.astype(np.float32)
ep_up["file_prob_max_3way_best"]    = current_3way.astype(np.float32)
ep_up["file_prob_max_3way_ica_alt"] = ica_3way_alt.astype(np.float32)
ep_up["file_prob_max_3way_std"]     = std_3way_ref.astype(np.float32)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(ep_up, f)
print(f"  PKL updated: file_prob_max, 3way components saved", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Direction A: SOFT LABEL CO-OCCURRENCE
# Use file_prob_max predictions (soft) instead of binary labels for cooc matrix
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Direction A: Soft label co-occurrence ===", flush=True)

# Soft labels: use the current best prediction (batch112 file_prob_max from PKL)
# Try various thresholds / soft levels
for soft_thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    fl_soft = np.clip(best_preds, 0, 1).astype(np.float32)  # continuous soft labels
    # Threshold soft labels at various levels for the cooc matrix
    fl_thresh = (fl_soft > soft_thr).astype(np.float32)
    cnt_thresh = fl_thresh.sum(0) + EPS
    cooc_thresh = fl_thresh.T @ fl_thresh
    COOC_T = cooc_thresh / cnt_thresh[:, None]
    np.fill_diagonal(COOC_T, 0)
    raw_idf_t = np.log(float(n_files) / (cnt_thresh + 1.0 - EPS))
    raw_idf_t = np.clip(raw_idf_t, 0, None)
    idf_t = raw_idf_t ** 0.75; idf_t /= (idf_t.mean() + EPS)

    c3way_t = apply_3way_fn(double_best, cooc_norm=COOC_T, idf_w=idf_t)
    ica3_t   = apply_3way_fn(ica_ens_alt, cooc_norm=COOC_T, idf_w=idf_t)
    std3_t   = apply_3way_fn(std_ens_ref, cooc_norm=COOC_T, idf_w=idf_t)
    best_t   = 0.76*c3way_t + 0.16*ica3_t + 0.08*std3_t
    auc_t    = macro_auc(best_t)
    mname    = f"soft_cooc_t{int(soft_thr*10):d}"
    delta    = save_result(mname, auc_t, {"soft_thr": soft_thr})
    flag     = " ← NEW BEST!" if auc_t > best_loo else ""
    print(f"  {mname}: {auc_t:.6f} {delta:+.6f}{flag}", flush=True)

# True soft co-occ: use prediction values directly
fl_soft = np.clip(best_preds, 0, 1).astype(np.float32)
cooc_soft_raw = fl_soft.T @ fl_soft
cnt_soft = fl_soft.sum(0) + EPS
COOC_SOFT = cooc_soft_raw / cnt_soft[:, None]
np.fill_diagonal(COOC_SOFT, 0)
# IDF from soft frequency
sp_freq_soft = fl_soft.mean(0).clip(EPS, 1-EPS)
idf_soft = (np.log(1/sp_freq_soft) ** 0.75)
idf_soft /= (idf_soft.mean() + EPS)

c3way_soft = apply_3way_fn(double_best, cooc_norm=COOC_SOFT, idf_w=idf_soft)
ica3_soft  = apply_3way_fn(ica_ens_alt, cooc_norm=COOC_SOFT, idf_w=idf_soft)
std3_soft  = apply_3way_fn(std_ens_ref, cooc_norm=COOC_SOFT, idf_w=idf_soft)
best_soft  = 0.76*c3way_soft + 0.16*ica3_soft + 0.08*std3_soft
auc_soft   = macro_auc(best_soft)
delta      = save_result("soft_cooc_full", auc_soft, {"desc": "soft float labels for cooc"})
flag       = " ← NEW BEST!" if auc_soft > best_loo else ""
print(f"  soft_cooc_full: {auc_soft:.6f} {delta:+.6f}{flag}", flush=True)

# Hybrid: blend binary and soft cooc matrices
for alpha_h in [0.2, 0.4, 0.5, 0.6]:
    COOC_H = (1-alpha_h) * COOC_HARD + alpha_h * COOC_SOFT
    np.fill_diagonal(COOC_H, 0)
    idf_h = (1-alpha_h) * IDF_W075 + alpha_h * idf_soft
    c3h = apply_3way_fn(double_best, cooc_norm=COOC_H, idf_w=idf_h)
    i3h = apply_3way_fn(ica_ens_alt, cooc_norm=COOC_H, idf_w=idf_h)
    s3h = apply_3way_fn(std_ens_ref, cooc_norm=COOC_H, idf_w=idf_h)
    bh  = 0.76*c3h + 0.16*i3h + 0.08*s3h
    ah  = macro_auc(bh)
    mname = f"hybrid_cooc_a{int(alpha_h*10):d}"
    delta = save_result(mname, ah, {"alpha_h": alpha_h})
    flag  = " ← NEW BEST!" if ah > best_loo else ""
    print(f"  {mname}: {ah:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Direction B: IDF exponent search
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Direction B: IDF exponent search ===", flush=True)

for idf_exp in [0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00, 1.10, 1.25]:
    idf_exp = round(idf_exp, 2)
    idf_var = raw_idf ** idf_exp
    idf_var /= (idf_var.mean() + EPS)
    c3_idf  = apply_3way_fn(double_best, cooc_norm=COOC_HARD, idf_w=idf_var)
    i3_idf  = apply_3way_fn(ica_ens_alt, cooc_norm=COOC_HARD, idf_w=idf_var)
    s3_idf  = apply_3way_fn(std_ens_ref, cooc_norm=COOC_HARD, idf_w=idf_var)
    b_idf   = 0.76*c3_idf + 0.16*i3_idf + 0.08*s3_idf
    a_idf   = macro_auc(b_idf)
    mname   = f"idf_exp_{int(idf_exp*100):d}"
    delta   = save_result(mname, a_idf, {"idf_exp": idf_exp})
    flag    = " ← NEW BEST!" if a_idf > best_loo else ""
    print(f"  {mname}: {a_idf:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Direction C: Asymmetric co-occurrence P(B|A) directional
# Current: COOC = P(A,B) / P(B) ≈ P(A|B)
# Asymmetric: COOC_A = P(A,B) / P(A) ≈ P(B|A) (complement direction)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Direction C: Asymmetric co-occurrence ===", flush=True)

cooc_asym = fl_hard.T @ fl_hard
count_row = fl_hard.sum(0) + EPS
COOC_ASYM = cooc_asym / count_row[None, :]   # P(B|A): row=A, col=B, normalize by count(A)
np.fill_diagonal(COOC_ASYM, 0)

# Average of symmetric and asymmetric
COOC_AVG = 0.5 * COOC_HARD + 0.5 * COOC_ASYM
np.fill_diagonal(COOC_AVG, 0)

for cooc_label, cooc_mat in [("asym", COOC_ASYM), ("avg_asym", COOC_AVG)]:
    c3_a = apply_3way_fn(double_best, cooc_norm=cooc_mat, idf_w=IDF_W075)
    i3_a = apply_3way_fn(ica_ens_alt, cooc_norm=cooc_mat, idf_w=IDF_W075)
    s3_a = apply_3way_fn(std_ens_ref, cooc_norm=cooc_mat, idf_w=IDF_W075)
    b_a  = 0.76*c3_a + 0.16*i3_a + 0.08*s3_a
    a_a  = macro_auc(b_a)
    mname = f"cooc_{cooc_label}"
    delta = save_result(mname, a_a, {"cooc": cooc_label})
    flag  = " ← NEW BEST!" if a_a > best_loo else ""
    print(f"  {mname}: {a_a:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Direction D: Double 3way smoothing (apply_3way twice)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Direction D: Double 3way smoothing ===", flush=True)

double_3way_best = apply_3way_fn(current_3way)
double_3way_ica  = apply_3way_fn(ica_3way_alt)
double_3way_std  = apply_3way_fn(std_3way_ref)
double_3way_auc  = macro_auc(double_3way_best)
print(f"  double_3way_best: {double_3way_auc:.6f} (baseline double_3way)", flush=True)

for w1, w2, w3 in [
    (0.76, 0.16, 0.08), (0.80, 0.14, 0.06), (0.70, 0.20, 0.10),
    (0.85, 0.10, 0.05), (0.72, 0.18, 0.10),
]:
    b_d2 = w1*double_3way_best + w2*double_3way_ica + w3*double_3way_std
    a_d2 = macro_auc(b_d2)
    mname = f"d2_3way_b{int(w1*100)}_i{int(w2*100)}_s{int(w3*100)}"
    delta = save_result(mname, a_d2, {"w1": w1, "w2": w2, "w3": w3})
    flag  = " ← NEW BEST!" if a_d2 > best_loo else ""
    print(f"  {mname}: {a_d2:.6f} {delta:+.6f}{flag}", flush=True)

# Blend double_3way with single_3way
for a_d, a_s in [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85), (0.20, 0.80), (0.30, 0.70)]:
    blend_d_s = a_d * (0.76*double_3way_best + 0.16*double_3way_ica + 0.08*double_3way_std) \
              + a_s * chk_ref
    auc_ds  = macro_auc(blend_d_s)
    mname   = f"d2blend_d{int(a_d*100)}_s{int(a_s*100)}"
    delta   = save_result(mname, auc_ds, {"a_double": a_d, "a_single": a_s})
    flag    = " ← NEW BEST!" if auc_ds > best_loo else ""
    print(f"  {mname}: {auc_ds:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Direction E: Alpha sweep (co-occ contribution strength)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Direction E: Alpha sweep ===", flush=True)

for alpha_val in [0.08, 0.10, 0.115, 0.12, 0.145, 0.16, 0.18, 0.20]:
    # Modify idf_cooc alpha
    def idf_cooc_alpha(scores, a=alpha_val):
        s_pow = np.clip(scores, 0, 1) ** 2.0
        s_cooc = soft_cooc_fn(s_pow, COOC_HARD, center=0.55, slope=41.0, alpha=a, idf_w=IDF_W075)
        return (1 - 0.55) * scores + 0.55 * s_cooc

    def apply_3way_alpha(s):
        return 0.85 * idf_cooc_alpha(s) + 0.15 * two_round_fn(s, COOC_HARD)

    c3_a = apply_3way_alpha(double_best)
    i3_a = apply_3way_alpha(ica_ens_alt)
    s3_a = apply_3way_alpha(std_ens_ref)
    b_a  = 0.76*c3_a + 0.16*i3_a + 0.08*s3_a
    a_val = macro_auc(b_a)
    mname = f"alpha_{int(alpha_val*1000):d}"
    delta = save_result(mname, a_val, {"alpha": alpha_val})
    flag  = " ← NEW BEST!" if a_val > best_loo else ""
    print(f"  {mname}: {a_val:.6f} {delta:+.6f}{flag}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"Batch134 complete", flush=True)
print(f"Best LOO: {best_loo:.6f}  (best method: {res['best']['method']})", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 134]
print(f"Experiments this run: {len(exps_this)}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
