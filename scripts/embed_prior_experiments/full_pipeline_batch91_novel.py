"""
Batch 91: Novel Aggregation + One-Class Methods
================================================
Current best: softmax_T6_proto_kde LOO=0.991782

Genuinely new methods not yet tried:
1. product_of_experts: multiply ICA KDE × centroid cosine (log-space sum → exp)
2. geometric_mean_blend: base^(1-w) * kde^w instead of arithmetic mix
3. ocsvm: One-Class SVM trained on positive ICA embeddings per species
4. isolation_forest: IsolationForest on positives, use -anomaly_score as prior
5. rank_aggregation: convert base+kde to rank percentiles, then blend
6. perspecies_adaptive: per-species w_kde ~ 1/(1+n_pos) (rare species get more KDE)

CRITICAL: Uses stored pkl embeddings + n_windows-based file ordering.
"""

import numpy as np
import json
import pickle
import time
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from numpy.linalg import norm
warnings.filterwarnings("ignore")

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

print(f"Loaded: ICA{ew_ica.shape}", flush=True)

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

def make_logit_pred(T):
    sig = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T, -88, 88)))
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

def make_softmax_pred(T):
    sm_r = logit_win / T; sm_r -= sm_r.max(1, keepdims=True)
    sm_e = np.exp(sm_r); smp = sm_e / (sm_e.sum(1, keepdims=True) + EPS)
    return np.stack([smp[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

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

def proto_kde_loo(bw=0.08):
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

# ── New Methods ────────────────────────────────────────────────────────────

def product_of_experts_loo(bw=0.08):
    """Product of Experts: multiply proto-KDE × centroid-cosine scores.
    poe(x,s) = kde(x,s) * cosim(x, centroid_s)  → per-species max normalization
    Implements "consensus signal" where both kernels must agree.
    """
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
            # Expert 1: proto-weighted KDE
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
            kde_s = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
            # Expert 2: centroid cosine similarity
            cent_s = np.clip(te @ centroid, 0, None)
            # Product (geometric mean for stability)
            ws[:, si] = np.sqrt(kde_s * cent_s + EPS)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def geometric_blend(base, kde, alpha):
    """Geometric mean blend: base^(1-alpha) * kde^alpha (component-wise)."""
    b = np.clip(base, EPS, None)
    k = np.clip(kde, EPS, None)
    return (b ** (1 - alpha)) * (k ** alpha)

def rank_aggregation_blend(base, kde, w_kde=0.04):
    """Convert each score to rank-percentile [0,1], then arithmetic blend."""
    def to_rank(s):
        r = np.zeros_like(s)
        for si in range(s.shape[1]):
            col = s[:, si]
            order = np.argsort(col)
            r[order, si] = np.arange(len(col)) / (len(col) - 1 + EPS)
        return r
    rb = to_rank(base); rk = to_rank(kde)
    return (1 - w_kde) * rb + w_kde * rk

def ocsvm_loo(kernel='rbf', nu=0.5, n_pca=20):
    """One-Class SVM per species: fit on positive windows, score = decision_function."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        k_pca = min(n_pca, tr.shape[0] - 1)
        try:
            pca_s = SklearnPCA(n_components=k_pca).fit(tr)
            tr_p = pca_s.transform(tr).astype(np.float64)
            te_p = pca_s.transform(te).astype(np.float64)
        except Exception:
            ws[:] = 0.5; out[fi] = ws.max(0); continue
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) < 3: ws[:, si] = 0.5; continue
            try:
                svm = OneClassSVM(kernel=kernel, nu=nu, gamma='scale')
                svm.fit(tr_p[pos_idx])
                scores = svm.decision_function(te_p)
                ws[:, si] = np.clip(scores - scores.min(), 0, None).astype(np.float32)
            except Exception:
                ws[:, si] = 0.5
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def isolation_forest_loo(n_estimators=50, contamination=0.1, n_pca=20):
    """IsolationForest per species: score = -anomaly_score (higher = more normal = more likely positive)."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        k_pca = min(n_pca, tr.shape[0] - 1)
        try:
            pca_s = SklearnPCA(n_components=k_pca).fit(tr)
            tr_p = pca_s.transform(tr).astype(np.float64)
            te_p = pca_s.transform(te).astype(np.float64)
        except Exception:
            ws[:] = 0.5; out[fi] = ws.max(0); continue
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) < 4: ws[:, si] = 0.5; continue
            try:
                clf = IsolationForest(n_estimators=n_estimators,
                                      contamination=contamination,
                                      random_state=42)
                clf.fit(tr_p[pos_idx])
                # score_samples returns -1 * anomaly_score: higher = more normal
                scores = clf.score_samples(te_p)
                ws[:, si] = np.clip(scores - scores.min(), 0, None).astype(np.float32)
            except Exception:
                ws[:, si] = 0.5
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def perspecies_adaptive_loo(kde_scores, base_scores, max_w=0.08, min_w=0.02):
    """Per-species adaptive KDE weight: w_si = max_w / (1 + log(1 + n_pos_si)).
    Species with fewer positive training examples get higher KDE weight.
    """
    # Count positives per species in full training set
    n_pos_per_species = (labels_win > 0.5).sum(0).astype(np.float32)  # (234,)
    w_spec = max_w / (1.0 + np.log1p(n_pos_per_species + EPS))
    w_spec = np.clip(w_spec, min_w, max_w)
    # Apply per-species blend
    out = np.zeros_like(base_scores)
    for si in range(n_species):
        w = w_spec[si]
        out[:, si] = (1 - w) * base_scores[:, si] + w * kde_scores[:, si]
    return out

# ── Pre-compute base ───────────────────────────────────────────────────────
print("Pre-computing base...", flush=True)
t0 = time.time()
s_ica = wl_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
s_std = wl_loo(ew_std, cfg["std_pca80"]["k_neg"], cfg["std_pca80"]["w_max_pos"], cfg["std_pca80"]["w_max_agg"])
s_pca = wl_loo(ew_pca, cfg["pca80"]["k_neg"], cfg["pca80"]["w_max_pos"], cfg["pca80"]["w_max_agg"])
s_nmf = wl_loo(ew_nmf, cfg["nmf"]["k_neg"], cfg["nmf"]["w_max_pos"], cfg["nmf"]["w_max_agg"])
uh_b  = cfg["w_ica100"] * s_ica + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
uh_nmf = cfg["nmf"]["uh_scale"] * uh_b + cfg["nmf"]["w_nmf"] * s_nmf
pT8   = make_logit_pred(cfg["logit_temperature"])
pmt810 = (pT8 + make_logit_pred(10.0)) / 2
sm6   = make_softmax_pred(cfg["softmax_temp"])
ss2   = compute_subspace(ew_pca, n_comp=2, wma_ss=0.92)
w_uh  = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base  = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt810 + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
kde   = proto_kde_loo(bw=0.08)
print(f"  Base done ({time.time()-t0:.0f}s)", flush=True)
print(f"Base: {macro_auc(base):.6f}  Reference: {macro_auc(0.96*base + 0.04*kde):.6f}", flush=True)

# ── New components ────────────────────────────────────────────────────────
print("\nPre-computing new components...", flush=True)
t1 = time.time()
poe    = product_of_experts_loo(bw=0.08)
ocsvm5 = ocsvm_loo(nu=0.5, n_pca=20)
ocsvm2 = ocsvm_loo(nu=0.2, n_pca=20)
isofor = isolation_forest_loo(n_estimators=50, n_pca=20)
print(f"  Done ({time.time()-t1:.0f}s)", flush=True)

# ── Load results ──────────────────────────────────────────────────────────
RES_PATH = Path("outputs/embed_prior_results.json")
with open(RES_PATH) as f:
    results = json.load(f)
best_auc = results["best"]["loo_auc"]
best_method = results["best"]["method"]
print(f"\nCurrent best: {best_method} = {best_auc:.6f}\n", flush=True)

experiments = []
def ev(name, s):
    auc = macro_auc(s)
    delta = auc - best_auc
    mark = " ***NEW BEST***" if auc > best_auc + 1e-6 else ""
    print(f"  {name}: {auc:.6f}  (Δ={delta:+.6f}){mark}", flush=True)
    experiments.append({"method": name, "loo_auc": auc})
    return auc

print("=== Group 1: Product of Experts as KDE replacement ===")
for wk in [0.03, 0.04, 0.05, 0.06, 0.08]:
    ev(f"product_of_experts_wk{int(wk*100):02d}", (1-wk)*base + wk*poe)

print("\n=== Group 2: Geometric Mean Blend ===")
for alpha in [0.02, 0.03, 0.04, 0.05, 0.06]:
    ev(f"geometric_mean_a{int(alpha*100):02d}", geometric_blend(base, kde, alpha))
# Geometric with PoE
for alpha in [0.03, 0.04, 0.05]:
    ev(f"geometric_poe_a{int(alpha*100):02d}", geometric_blend(base, poe, alpha))

print("\n=== Group 3: Rank Aggregation ===")
for wk in [0.03, 0.04, 0.05, 0.06, 0.10]:
    ev(f"rank_agg_kde_wk{int(wk*100):02d}", rank_aggregation_blend(base, kde, wk))
for wk in [0.03, 0.04, 0.05]:
    ev(f"rank_agg_poe_wk{int(wk*100):02d}", rank_aggregation_blend(base, poe, wk))

print("\n=== Group 4: One-Class SVM ===")
for name, score in [("ocsvm_nu05", ocsvm5), ("ocsvm_nu02", ocsvm2)]:
    for wk in [0.03, 0.04, 0.05, 0.06]:
        ev(f"{name}_wk{int(wk*100):02d}", (1-wk)*base + wk*score)
# OC-SVM + KDE blend
for w_s in [0.3, 0.5]:
    blend_s = w_s * ocsvm5 + (1-w_s) * kde
    for si in range(n_species):
        mx = blend_s[:, si].max();
        if mx > EPS: blend_s[:, si] /= mx
    for wk in [0.04, 0.05]:
        ev(f"ocsvm_kde_ws{int(w_s*10)}_wk{int(wk*100):02d}", (1-wk)*base + wk*blend_s)

print("\n=== Group 5: Isolation Forest ===")
for wk in [0.03, 0.04, 0.05, 0.06]:
    ev(f"isolation_forest_wk{int(wk*100):02d}", (1-wk)*base + wk*isofor)
# IsoFor + KDE blend
for w_i in [0.3, 0.5]:
    blend_i = w_i * isofor + (1-w_i) * kde
    for si in range(n_species):
        mx = blend_i[:, si].max()
        if mx > EPS: blend_i[:, si] /= mx
    for wk in [0.04, 0.05]:
        ev(f"isofor_kde_wi{int(w_i*10)}_wk{int(wk*100):02d}", (1-wk)*base + wk*blend_i)

print("\n=== Group 6: Per-Species Adaptive KDE Weight ===")
for max_w in [0.06, 0.08, 0.10]:
    s = perspecies_adaptive_loo(kde, base, max_w=max_w, min_w=0.02)
    ev(f"perspecies_adaptive_maxw{int(max_w*100):02d}", s)
# Adaptive with PoE
for max_w in [0.06, 0.08]:
    s = perspecies_adaptive_loo(poe, base, max_w=max_w, min_w=0.02)
    ev(f"perspecies_poe_maxw{int(max_w*100):02d}", s)

print("\n=== Group 7: Three-way KDE+PoE+OcSVM ===")
for wa, wb, wc in [(0.02, 0.01, 0.01), (0.03, 0.01, 0.01)]:
    s = (1-wa-wb-wc)*base + wa*kde + wb*poe + wc*ocsvm5
    ev(f"triple_kde_poe_svm_wa{int(wa*100):02d}_{int(wb*100):02d}_{int(wc*100):02d}", s)

# ── Finalize ────────────────────────────────────────────────────────────────
best_new = max(experiments, key=lambda x: x["loo_auc"])
delta_best = best_new["loo_auc"] - best_auc

print(f"\n{'='*60}")
print(f"Batch 91 Summary: {len(experiments)} experiments")
print(f"Best new: {best_new['method']} = {best_new['loo_auc']:.6f}")
print(f"Current best: {best_method} = {best_auc:.6f}")
print(f"Delta: {delta_best:+.6f}")

if delta_best > 1e-6:
    print(f"\n*** NEW BEST: {best_new['method']} = {best_new['loo_auc']:.6f} ***")
    results["best"] = best_new

for e in experiments:
    results["experiments"].append(e)
with open(RES_PATH, "w") as f:
    json.dump(results, f)
print("Results saved.")

print(f"\nTop 5:")
for e in sorted(experiments, key=lambda x: -x["loo_auc"])[:5]:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
