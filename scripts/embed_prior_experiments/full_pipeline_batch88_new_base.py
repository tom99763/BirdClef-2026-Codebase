"""
Batch 88: New Base Component Strategies
========================================
Current best: softmax_T6_proto_kde LOO=0.991782

All KDE space variants exhausted. Now targeting base improvements via:
1. Logit aggregation: mean/median vs max over windows
2. Softmax aggregation: mean over windows
3. NMF subspace (reconstruction error in NMF space)
4. Two-class LDA projection (positive vs negative)
5. k-NN density estimate in ICA space
6. WL with logit weighting of positives (logit-weighted prototype)
7. Blended logit: mix of max and mean aggregation

All use exact pkl config for base components.
"""

import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy.linalg import norm

# ── Load data ──────────────────────────────────────────────────────────────
DATA = np.load("outputs/perch_labeled_ss.npz")
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

with open("outputs/embed_prior_model.pkl", "rb") as f:
    ep = pickle.load(f)

ew_ica = ep["emb_win_ica_norm"]
ew_pca = ep["emb_win_pca_norm"]
ew_std = ep["emb_win_std_norm"]
ew_nmf = ep["emb_win_nmf_norm"]
file_labels = ep["file_labels"]
cfg = ep["config"]

print(f"Loaded: ICA{ew_ica.shape} NMF{ew_nmf.shape}", flush=True)

# ── Helpers ────────────────────────────────────────────────────────────────
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

def make_logit_pred(T, agg="max"):
    sig = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T, -88, 88)))
    if agg == "max":
        return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
    elif agg == "mean":
        return np.stack([sig[file_start[fi]:file_end[fi]].mean(0) for fi in range(n_files)])
    elif agg == "median":
        return np.stack([np.median(sig[file_start[fi]:file_end[fi]], axis=0) for fi in range(n_files)])
    elif agg == "maxmean":
        mx = np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
        mn = np.stack([sig[file_start[fi]:file_end[fi]].mean(0) for fi in range(n_files)])
        return 0.7 * mx + 0.3 * mn
    elif agg == "top2":
        # mean of top-2 window scores per file per species
        out = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            s = sig[file_start[fi]:file_end[fi]]
            n = s.shape[0]
            if n >= 2:
                idx = np.argsort(-s, axis=0)[:2]
                out[fi] = s[idx[0], np.arange(n_species)] * 0.5 + s[idx[1], np.arange(n_species)] * 0.5
            else:
                out[fi] = s.max(0)
        return out

def make_softmax_pred(T, agg="max"):
    sm_r = logit_win / T; sm_r -= sm_r.max(1, keepdims=True)
    sm_e = np.exp(sm_r); smp = sm_e / (sm_e.sum(1, keepdims=True) + EPS)
    if agg == "max":
        return np.stack([smp[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
    elif agg == "mean":
        return np.stack([smp[file_start[fi]:file_end[fi]].mean(0) for fi in range(n_files)])
    elif agg == "maxmean":
        mx = np.stack([smp[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
        mn = np.stack([smp[file_start[fi]:file_end[fi]].mean(0) for fi in range(n_files)])
        return 0.7 * mx + 0.3 * mn

def compute_subspace(ew_sp, n_comp=2, wma_ss=0.92):
    ss = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_sp[win_file_id == fi]; tr = ew_sp[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        dim = te.shape[1]
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; k = min(n_comp, len(pos) - 1, dim - 1)
            if k < 1:
                pp = pos.mean(0); pp /= norm(pp) + EPS
                ws[:, si] = np.clip((te @ pp + 1) / 2, 0, 1); continue
            try:
                pca_sp = SklearnPCA(n_components=k); pca_sp.fit(pos)
                te_r = pca_sp.inverse_transform(pca_sp.transform(te))
                err = norm(te - te_r, axis=1)
                ws[:, si] = np.clip(1 - err / (norm(te, axis=1) + EPS), 0, 1)
            except Exception:
                ws[:, si] = 0.5
        ss[fi] = wma_ss * ws.max(0) + (1 - wma_ss) * ws.mean(0)
    return ss

def proto_kde_loo_ica(bw=0.08):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]
            centroid = pos_wins.mean(0); centroid /= (norm(centroid) + EPS)
            proto_w = np.clip(pos_wins @ centroid, 0, None)
            proto_w = proto_w / (proto_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def lda_score_loo():
    """Two-class LDA projection score per species in ICA space."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any() or not nm.any(): ws[:, si] = 0.5; continue
            if pm.sum() < 2 or nm.sum() < 2: ws[:, si] = 0.5; continue
            X = np.vstack([tr[pm], tr[nm]])
            y = np.array([1]*pm.sum() + [0]*nm.sum())
            try:
                lda = LinearDiscriminantAnalysis(n_components=1)
                lda.fit(X, y)
                scores = lda.transform(te).ravel()
                # ensure positive class → higher score
                pos_mean = lda.transform(tr[pm]).mean()
                neg_mean = lda.transform(tr[nm]).mean()
                if pos_mean < neg_mean:
                    scores = -scores
                ws[:, si] = np.clip(scores, 0, None)
            except Exception:
                ws[:, si] = 0.5
        for si in range(n_species):
            mn_v = ws[:, si].min(); mx_v = ws[:, si].max()
            if mx_v > mn_v: ws[:, si] = (ws[:, si] - mn_v) / (mx_v - mn_v)
        out[fi] = ws.max(0)
    return out

def knn_density_loo(k=5):
    """k-NN density: score = mean similarity to k-nearest positives in ICA space."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_sims = sims[:, pos_idx]  # (n_te, n_pos)
            k2 = min(k, len(pos_idx))
            if k2 == 1:
                ws[:, si] = np.clip(pos_sims.max(1), 0, None)
            else:
                top_k = np.sort(pos_sims, axis=1)[:, -k2:]
                ws[:, si] = np.clip(top_k.mean(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def wl_logit_weighted_loo():
    """WL where positive windows are weighted by their logit confidence."""
    lw_full = 1.0 / (1.0 + np.exp(np.clip(-logit_win / 8.0, -88, 88)))
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]; lw = lw_full[win_file_id != fi]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]
            # Logit-weighted prototype
            logit_w = lw[pm, si]
            logit_w = logit_w / (logit_w.sum() + EPS)
            pp = (pw * logit_w[:, None]).sum(0); pp /= norm(pp) + EPS
            ps = te @ pw.T
            sp = 0.5 * ps.max(1) + 0.5 * (te @ pp)
            if nm.any():
                nw = tr[nm]; ns_s = te @ nw.T; k2 = min(50, ns_s.shape[1])
                if k2 > 0:
                    tn = nw[np.argsort(-ns_s, axis=1)[:, :k2]].mean(1)
                    tn /= norm(tn, axis=1, keepdims=True) + EPS
                    ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
                else:
                    ws[:, si] = (sp + 1) / 2
            else:
                ws[:, si] = (sp + 1) / 2
        out[fi] = 0.88 * ws.max(0) + 0.12 * ws.mean(0)
    return out

# ── Pre-compute base ───────────────────────────────────────────────────────
print("Pre-computing base components...", flush=True)
t0 = time.time()
s_ica = wl_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
s_std = wl_loo(ew_std, cfg["std_pca80"]["k_neg"], cfg["std_pca80"]["w_max_pos"], cfg["std_pca80"]["w_max_agg"])
s_pca = wl_loo(ew_pca, cfg["pca80"]["k_neg"], cfg["pca80"]["w_max_pos"], cfg["pca80"]["w_max_agg"])
s_nmf = wl_loo(ew_nmf, cfg["nmf"]["k_neg"], cfg["nmf"]["w_max_pos"], cfg["nmf"]["w_max_agg"])
uh_b  = cfg["w_ica100"] * s_ica + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
uh_nmf = cfg["nmf"]["uh_scale"] * uh_b + cfg["nmf"]["w_nmf"] * s_nmf
pT8  = make_logit_pred(cfg["logit_temperature"])
pmt_810 = (pT8 + make_logit_pred(10.0)) / 2
sm6 = make_softmax_pred(cfg["softmax_temp"])
ss2 = compute_subspace(ew_pca, n_comp=2, wma_ss=0.92)
print(f"  Base done ({time.time()-t0:.0f}s)", flush=True)

w_uh = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt_810 + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
kde_ref = proto_kde_loo_ica(bw=0.08)

print(f"\nBase AUC: {macro_auc(base_cur):.6f} (should ~0.991359)", flush=True)
print(f"Reference (should ~0.991782): {macro_auc(0.96*base_cur + 0.04*kde_ref):.6f}", flush=True)

# New components
print("\nPre-computing new components...", flush=True)
t1 = time.time()
pT8_mean   = make_logit_pred(8.0, "mean")
pT8_median = make_logit_pred(8.0, "median")
pT8_mm     = make_logit_pred(8.0, "maxmean")
pT8_top2   = make_logit_pred(8.0, "top2")
pT6_mean   = make_logit_pred(6.0, "mean")
pT10_mean  = make_logit_pred(10.0, "mean")
pmt_810_mean = (pT8_mean + pT10_mean) / 2

sm6_mean   = make_softmax_pred(6.0, "mean")
sm6_mm     = make_softmax_pred(6.0, "maxmean")

ss2_ica = compute_subspace(ew_ica, n_comp=2, wma_ss=0.92)
ss2_nmf = compute_subspace(ew_nmf, n_comp=2, wma_ss=0.92)

lda_s   = lda_score_loo()
knn5_s  = knn_density_loo(k=5)
knn10_s = knn_density_loo(k=10)
wl_lw   = wl_logit_weighted_loo()

print(f"  New components done ({time.time()-t1:.0f}s)", flush=True)

# ── Load results JSON ──────────────────────────────────────────────────────
RES_PATH = Path("outputs/embed_prior_results.json")
with open(RES_PATH) as f:
    results = json.load(f)
best_auc = results["best"]["loo_auc"]
best_method = results["best"]["method"]
print(f"\nCurrent best: {best_method} = {best_auc:.6f}\n", flush=True)

experiments = []

def eval_full(name, base, w_kde=0.04, kde=None):
    """Evaluate full = (1-w)*base + w*kde."""
    if kde is None: kde = kde_ref
    final = (1 - w_kde) * base + w_kde * kde
    auc = macro_auc(final)
    delta = auc - best_auc
    mark = " ***NEW BEST***" if auc > best_auc + 1e-6 else ""
    print(f"  {name}: {auc:.6f}  (Δ={delta:+.6f}){mark}", flush=True)
    experiments.append({"method": name, "loo_auc": auc})
    return auc

def eval_base_only(name, base):
    auc = macro_auc(base)
    delta = auc - best_auc
    print(f"  {name} [base]: {auc:.6f}  (Δ={delta:+.6f})", flush=True)

print("=== Group 1: Logit aggregation variants ===")
# Replace pT8 (max) with mean/median/maxmean/top2
for agg_name, pT8_alt in [("mean", pT8_mean), ("median", pT8_median), ("mm", pT8_mm), ("top2", pT8_top2)]:
    for pmt_alt, pmt_name in [(pmt_810, "mt810max"), (pmt_810_mean, "mt810mean")]:
        b = w_uh * uh_nmf + cfg["w_logit"] * pT8_alt + cfg["w_multit"] * pmt_alt + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
        for wk in [0.04, 0.05]:
            eval_full(f"logit_{agg_name}_{pmt_name}_wk{int(wk*100):02d}", b, wk)

print("\n=== Group 2: Softmax aggregation variants ===")
for agg_name, sm_alt in [("mean", sm6_mean), ("mm", sm6_mm)]:
    b = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt_810 + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm_alt
    for wk in [0.04, 0.05]:
        eval_full(f"softmax_{agg_name}_wk{int(wk*100):02d}", b, wk)

print("\n=== Group 3: Subspace in different spaces ===")
for ss_name, ss_alt in [("ica", ss2_ica), ("nmf", ss2_nmf)]:
    b = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt_810 + cfg["w_subspace"] * ss_alt + cfg["w_softmax"] * sm6
    for wk in [0.04, 0.05]:
        eval_full(f"subspace_{ss_name}_wk{int(wk*100):02d}", b, wk)

print("\n=== Group 4: LDA score as component ===")
for w_lda in [0.03, 0.05, 0.07, 0.10]:
    b = (1 - w_lda) * base_cur + w_lda * lda_s
    for wk in [0.04, 0.05]:
        eval_full(f"lda_w{int(w_lda*100):02d}_wk{int(wk*100):02d}", b, wk)

print("\n=== Group 5: k-NN density as component or KDE ===")
# Use knn as replacement for KDE
for k_name, knn_s in [("k5", knn5_s), ("k10", knn10_s)]:
    for wk in [0.03, 0.04, 0.05]:
        eval_full(f"knn_{k_name}_wk{int(wk*100):02d}", base_cur, wk, knn_s)
# Add knn as base component
for k_name, knn_s in [("k5", knn5_s), ("k10", knn10_s)]:
    for w_knn in [0.03, 0.05]:
        b = (1 - w_knn) * base_cur + w_knn * knn_s
        eval_full(f"knn_{k_name}_base_w{int(w_knn*100):02d}_wk04", b, 0.04)

print("\n=== Group 6: Logit-weighted WL as ICA replacement ===")
for w_lw in [0.3, 0.5, 0.7]:
    uh_b_lw = cfg["w_ica100"] * ((1-w_lw)*s_ica + w_lw*wl_lw) + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
    uh_nmf_lw = cfg["nmf"]["uh_scale"] * uh_b_lw + cfg["nmf"]["w_nmf"] * s_nmf
    b = w_uh * uh_nmf_lw + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt_810 + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
    for wk in [0.04, 0.05]:
        eval_full(f"wl_lw_w{int(w_lw*10)}_wk{int(wk*100):02d}", b, wk)

print("\n=== Group 7: Combined logit aggregation + softmax mean ===")
for agg_name, pT8_alt in [("mm", pT8_mm), ("top2", pT8_top2)]:
    b = w_uh * uh_nmf + cfg["w_logit"] * pT8_alt + cfg["w_multit"] * pmt_810 + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6_mm
    for wk in [0.04, 0.05]:
        eval_full(f"combo_{agg_name}_smm_wk{int(wk*100):02d}", b, wk)

print("\n=== Group 8: LDA as KDE replacement (no base change) ===")
# Treat LDA as the prior signal
for w_lda in [0.02, 0.03, 0.04, 0.05]:
    auc = macro_auc(0.96 * base_cur + 0.04 * kde_ref + w_lda * lda_s)
    # Normalize: base + kde + lda sum to weights
    tot = 0.96 + 0.04 + w_lda
    auc2 = macro_auc((0.96/tot) * base_cur + (0.04/tot) * kde_ref + (w_lda/tot) * lda_s)
    delta = auc2 - best_auc
    name = f"base_kde_lda_w{int(w_lda*100):02d}"
    print(f"  {name}: {auc2:.6f}  (Δ={delta:+.6f})", flush=True)
    experiments.append({"method": name, "loo_auc": auc2})

# ── Finalize ────────────────────────────────────────────────────────────────
best_new = max(experiments, key=lambda x: x["loo_auc"])
delta_best = best_new["loo_auc"] - best_auc

print(f"\n{'='*60}")
print(f"Batch 88 Summary")
print(f"Experiments run: {len(experiments)}")
print(f"Best new: {best_new['method']} = {best_new['loo_auc']:.6f}")
print(f"Current best: {best_method} = {best_auc:.6f}")
print(f"Delta: {delta_best:+.6f}")

if delta_best > 1e-6:
    print(f"\n*** NEW BEST FOUND: {best_new['method']} = {best_new['loo_auc']:.6f} ***")
    results["best"] = best_new
else:
    print(f"\nNo improvement. Best stays at {best_method} = {best_auc:.6f}")

for e in experiments:
    results["experiments"].append(e)

with open(RES_PATH, "w") as f:
    json.dump(results, f)
print("Results saved.")

print(f"\nTop 5 new:")
for e in sorted(experiments, key=lambda x: -x["loo_auc"])[:5]:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
