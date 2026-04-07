"""
batch135 — Alpha parameter extended sweep + joint parameter optimization
===============================================================================
Previous best: alpha_200 LOO=0.995120
Key insight: increasing alpha (co-occ contribution) from 0.130→0.200 gained +0.000250
Alpha still increasing at 0.200, so continue sweep upward.

Also explore:
 - blend parameter (controls idf_cooc vs original signal ratio)
 - s_pow exponent (currently 2.0)
 - two_round alpha values (currently 0.089, 0.040)
 - Joint optimization of (alpha, blend)
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
file_labels = ep["file_labels"]
cfg = ep["config"]

# Load pre-computed 3way components (saved in batch134)
if "file_prob_max_3way_best" in ep:
    current_3way = ep["file_prob_max_3way_best"]
    ica_3way_alt = ep["file_prob_max_3way_ica_alt"]
    std_3way_ref = ep["file_prob_max_3way_std"]
    print(f"[batch135] Loaded pre-computed 3way components from PKL", flush=True)
    # verify
    chk = 0.76*current_3way + 0.16*ica_3way_alt + 0.08*std_3way_ref
    auc_chk = 0.0
    from sklearn.metrics import roc_auc_score
    fl = file_labels
    aucs = []
    for si in range(n_species):
        y = fl[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try: aucs.append(roc_auc_score(y, chk[:, si]))
            except: pass
    auc_chk = float(np.mean(aucs))
    print(f"  3way verify: {auc_chk:.6f} (expect 0.994870)", flush=True)
    need_recompute = abs(auc_chk - 0.994870) > 0.0001
else:
    need_recompute = True
    print("[batch135] No pre-computed 3way found, will recompute", flush=True)

print(f"[batch135] ICA{ew_ica.shape}, n_files={n_files}", flush=True)
res = json.load(open(RESULTS_PATH))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch135] Current best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 135}
    res["experiments"].append(entry)
    tried.add(mname)
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
    return score - best_loo

# ─── Recompute full chain if needed ─────────────────────────────────────────
if need_recompute:
    print("\nRecomputing full chain...", flush=True)
    # (Full chain code from batch132/134)
    from sklearn.decomposition import PCA as SklearnPCA

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
                    else: ws[:, si] = (sp + 1) / 2
                else: ws[:, si] = (sp + 1) / 2
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
            tl = labels_win[win_file_id != fi]; ws = np.zeros((len(te), n_species), np.float32)
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
            te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]; tl = labels_win[win_file_id != fi]
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
            te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]; tl = labels_win[win_file_id != fi]
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
            te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]; tl = labels_win[win_file_id != fi]
            ws = np.zeros((len(te), n_species), np.float32)
            for si in range(n_species):
                pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
                if not pm.any(): ws[:, si] = 0.5; continue
                pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
                mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
                var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
                fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
                top_idx = np.argsort(-fisher_raw)[:top_k]
                w_dim = np.zeros(len(fisher_raw), np.float32); w_dim[top_idx] = 1.0/np.sqrt(float(top_k))
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
            tl = labels_win[win_file_id != fi]; tl_logit = logit_win[win_file_id != fi]
            ws = np.zeros((len(te), n_species), np.float32)
            for si in range(n_species):
                pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
                if not pm.any(): ws[:, si] = 0.5; continue
                pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
                mu_p = pos.mean(0); mu_n = neg.mean(0)
                var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
                fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
                top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
                w_dim = np.zeros(len(fisher_raw), np.float32); w_dim[top_idx] = 1.0/np.sqrt(float(top_k_fisher))
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
            te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]; tl = labels_win[win_file_id != fi]
            ws = np.zeros((len(te), n_species), np.float32)
            for si in range(n_species):
                pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
                if not pm.any(): ws[:, si] = 0.5; continue
                pos = tr[pm]; neg = tr[nm] if nm.any() else tr[~pm]
                mu_p = pos.mean(0); mu_n = neg.mean(0)
                var_p = pos.var(0) + EPS; var_n = neg.var(0) + EPS
                fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
                top_idx = np.argsort(-fisher_raw)[:top_k_fisher]
                w_dim = np.zeros(len(fisher_raw), np.float32); w_dim[top_idx] = 1.0/np.sqrt(float(top_k_fisher))
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
            te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]; tl = labels_win[win_file_id != fi]
            lw = LedoitWolf().fit(tr); VI = lw.precision_
            ws = np.zeros((len(te), n_species), np.float32)
            for si in range(n_species):
                pm = tl[:, si] > 0.5
                if not pm.any(): ws[:, si] = 0.5; continue
                pos = tr[pm]
                XV_pos = pos @ VI; diag_pos = (XV_pos * pos).sum(1)
                XV_te  = te  @ VI; diag_te  = (XV_te  * te ).sum(1)
                cross  = te @ (pos @ VI).T
                d2     = np.clip(diag_te[:, None] - 2 * cross + diag_pos[None, :], 0, None)
                k2     = min(k, len(pos)); idx = np.argsort(d2, axis=1)[:, :k2]
                sim    = 1.0 / (1.0 + d2[np.arange(len(te))[:, None], idx].mean(1))
                ws[:, si] = sim
            out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
        return out

    def attn_ica_loo(ew, tau=0.3, w_max_agg=0.8):
        out = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
            tl = labels_win[win_file_id != fi]; tl_logit = logit_win[win_file_id != fi]
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
            te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]; tl = labels_win[win_file_id != fi]
            ws = np.zeros((len(te), n_species), np.float32)
            for si in range(n_species):
                pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
                if not pm.any(): ws[:, si] = 1.0; continue
                pos = tr[pm]; nw = tr[nm] if nm.any() else tr[~pm]; k2 = min(5, len(nw))
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
    ref_ds_ica   = wl_dual_softmax(ew_ica, tau=0.3, w_max_agg=0.8)
    ref_mahal    = mahal_ica_loo(ew_ica, k=5, w_max_agg=0.80)
    ref_attn_ica = attn_ica_loo(ew_ica, tau=0.3, w_max_agg=0.80)
    ref_ds_std   = wl_dual_softmax(ew_std, tau=0.3, w_max_agg=0.8)
    ref_attn_std = attn_ica_loo(ew_std, tau=0.3, w_max_agg=0.80)
    ica_ens_alt  = 0.65 * ref_ds_ica + 0.10 * ref_mahal + 0.25 * ref_attn_ica
    std_ens_ref  = 0.70 * ref_ds_std + 0.30 * ref_attn_std
else:
    # Load pre-computed double_best = inverse of 3way_best (approximation)
    # We need the raw double_best for new co-occ experiments; use pkls cached values
    double_best = None  # Will use 3way components only for these experiments
    ica_ens_alt = None
    std_ens_ref = None
    print("  Using pre-computed 3way components (double_best not needed for blend experiments)", flush=True)

# ─── Co-occurrence setup ─────────────────────────────────────────────────────
fl_hard   = file_labels.astype(np.float32)
count_i   = fl_hard.sum(0) + EPS
cooc_raw  = fl_hard.T @ fl_hard
COOC_NORM = cooc_raw / count_i[:, None]
np.fill_diagonal(COOC_NORM, 0)
raw_idf   = np.log(float(n_files) / (count_i + 1.0 - EPS))
raw_idf   = np.clip(raw_idf, 0, None)

def soft_cooc(scores, center=0.53, slope=37.0, alpha=0.086, idf_w=None):
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope * (s - center), -88, 88)))
        s_gated = s * gate
        if idf_w is not None: s_gated = s_gated * idf_w
        if np.abs(s_gated).sum() < EPS: smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def idf_cooc_custom(scores, alpha=0.130, blend=0.55, idf_exp=0.75, center=0.55, slope=41.0, s_pow=2.0):
    idf_w = raw_idf ** idf_exp; idf_w /= (idf_w.mean() + EPS)
    s_p = np.clip(scores, 0, 1) ** s_pow
    s_c = soft_cooc(s_p, center=center, slope=slope, alpha=alpha, idf_w=idf_w)
    return (1 - blend) * scores + blend * s_c

def two_round(scores, c1=0.54, sl1=41.0, a1=0.089, c2=0.53, sl2=37.0, a2=0.040):
    return soft_cooc(soft_cooc(scores, center=c1, slope=sl1, alpha=a1), center=c2, slope=sl2, alpha=a2)

def apply_3way_custom(s, alpha=0.130, blend=0.55, idf_exp=0.75, center=0.55, slope=41.0, s_pow=2.0):
    return 0.85 * idf_cooc_custom(s, alpha=alpha, blend=blend, idf_exp=idf_exp,
                                   center=center, slope=slope, s_pow=s_pow) + 0.15 * two_round(s)

def apply_blend(s_best, s_ica, s_std, wb=0.76, wi=0.16, ws_=0.08):
    return wb * s_best + wi * s_ica + ws_ * s_std

# ═══════════════════════════════════════════════════════════════════════════════
# Alpha extended sweep (0.22 to 0.60)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Alpha extended sweep ===", flush=True)
t0 = time.time()
for alpha_val in [0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.50, 0.55, 0.60]:
    c3 = apply_3way_custom(current_3way, alpha=alpha_val)
    i3 = apply_3way_custom(ica_3way_alt, alpha=alpha_val)
    s3 = apply_3way_custom(std_3way_ref, alpha=alpha_val)

    # Note: we apply to already-3way-smoothed inputs here → double smoothing with modified alpha
    # Pure approach: also compute from raw components if available
    b = apply_blend(c3, i3, s3)
    a = macro_auc(b)
    mname = f"alpha_ext_{int(alpha_val*100):d}_on3way"
    delta = save_result(mname, a, {"alpha": alpha_val, "input": "3way"})
    flag  = " ← NEW BEST!" if a > best_loo else ""
    print(f"  alpha={alpha_val:.2f} (on 3way): {a:.6f} {delta:+.6f}{flag}", flush=True)

# Also apply directly to double_best components if available
if double_best is not None:
    print("\n--- Alpha sweep on RAW components ---", flush=True)
    for alpha_val in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.50, 0.55, 0.60]:
        c3r = apply_3way_custom(double_best, alpha=alpha_val)
        i3r = apply_3way_custom(ica_ens_alt, alpha=alpha_val)
        s3r = apply_3way_custom(std_ens_ref, alpha=alpha_val)
        br  = apply_blend(c3r, i3r, s3r)
        ar  = macro_auc(br)
        mname = f"alpha_{int(alpha_val*100):d}"
        delta = save_result(mname, ar, {"alpha": alpha_val})
        flag  = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  alpha={alpha_val:.2f} (raw): {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Alpha sweep complete [{time.time()-t0:.1f}s]", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Joint (alpha, blend) grid search
# ═══════════════════════════════════════════════════════════════════════════════
if double_best is not None:
    print("\n=== Joint (alpha, blend) search ===", flush=True)
    best_ab = {"alpha": 0.20, "blend": 0.55, "auc": 0.0}
    for alpha_val in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        for blend_val in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            c3j = apply_3way_custom(double_best, alpha=alpha_val, blend=blend_val)
            i3j = apply_3way_custom(ica_ens_alt, alpha=alpha_val, blend=blend_val)
            s3j = apply_3way_custom(std_ens_ref, alpha=alpha_val, blend=blend_val)
            bj  = apply_blend(c3j, i3j, s3j)
            aj  = macro_auc(bj)
            mname = f"ab_a{int(alpha_val*100)}_b{int(blend_val*100)}"
            delta = save_result(mname, aj, {"alpha": alpha_val, "blend": blend_val})
            flag  = " ← NEW BEST!" if aj > best_loo else ""
            if aj > best_ab["auc"]:
                best_ab = {"alpha": alpha_val, "blend": blend_val, "auc": aj}
            print(f"  a={alpha_val:.2f} b={blend_val:.2f}: {aj:.6f} {delta:+.6f}{flag}", flush=True)
    print(f"  Best (alpha,blend): a={best_ab['alpha']:.2f} b={best_ab['blend']:.2f} auc={best_ab['auc']:.6f}", flush=True)

    # ─── s_pow search ──────────────────────────────────────────────────────────
    print("\n=== s_pow (score power) search ===", flush=True)
    for s_pow_val in [1.0, 1.5, 2.5, 3.0]:
        c3sp = apply_3way_custom(double_best, alpha=0.20, s_pow=s_pow_val)
        i3sp = apply_3way_custom(ica_ens_alt, alpha=0.20, s_pow=s_pow_val)
        s3sp = apply_3way_custom(std_ens_ref, alpha=0.20, s_pow=s_pow_val)
        bsp  = apply_blend(c3sp, i3sp, s3sp)
        asp  = macro_auc(bsp)
        mname = f"spow_{int(s_pow_val*10)}_a20"
        delta = save_result(mname, asp, {"s_pow": s_pow_val, "alpha": 0.20})
        flag  = " ← NEW BEST!" if asp > best_loo else ""
        print(f"  s_pow={s_pow_val:.1f}: {asp:.6f} {delta:+.6f}{flag}", flush=True)

    # ─── two_round alpha search ────────────────────────────────────────────────
    print("\n=== two_round alpha values search ===", flush=True)
    for tr_a1, tr_a2 in [(0.089, 0.040), (0.10, 0.050), (0.12, 0.060), (0.15, 0.070), (0.12, 0.040), (0.10, 0.030)]:
        def apply_3way_tr(s, a_=0.20):
            idf_w = raw_idf ** 0.75; idf_w /= (idf_w.mean() + EPS)
            s_p = np.clip(s, 0, 1) ** 2.0
            s_c = soft_cooc(s_p, center=0.55, slope=41.0, alpha=a_, idf_w=idf_w)
            idf_s = (1 - 0.55) * s + 0.55 * s_c
            tr_r = soft_cooc(soft_cooc(s, center=0.54, slope=41.0, alpha=tr_a1), center=0.53, slope=37.0, alpha=tr_a2)
            return 0.85 * idf_s + 0.15 * tr_r

        c3t = apply_3way_tr(double_best)
        i3t = apply_3way_tr(ica_ens_alt)
        s3t = apply_3way_tr(std_ens_ref)
        bt  = apply_blend(c3t, i3t, s3t)
        at  = macro_auc(bt)
        mname = f"tr_a1_{int(tr_a1*1000)}_a2_{int(tr_a2*1000)}_a20"
        delta = save_result(mname, at, {"tr_a1": tr_a1, "tr_a2": tr_a2, "alpha": 0.20})
        flag  = " ← NEW BEST!" if at > best_loo else ""
        print(f"  tr_a1={tr_a1:.3f} a2={tr_a2:.3f}: {at:.6f} {delta:+.6f}{flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 135]
print(f"Batch135 complete: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
