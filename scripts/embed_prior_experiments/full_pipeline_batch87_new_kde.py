"""
Batch 87: New KDE Spaces + Adaptive Strategies
================================================
Current best: softmax_T6_proto_kde LOO=0.991782

Uses EXACT base computation from Batch 86 (with correct pkl config params):
  s_ica = wl_loo(ew_ica, k_neg=50, wmp=0.85, wma=0.88)
  s_std = wl_loo(ew_std, k_neg=3, wmp=0.5, wma=0.7)
  s_pca = wl_loo(ew_pca, k_neg=4, wmp=0.7, wma=0.6)
  s_nmf = wl_loo(ew_nmf, k_neg=6, wmp=0.6, wma=0.65)
  uh_b  = 0.72*s_ica + 0.18*s_std + 0.10*s_pca
  uh_nmf = 0.84*uh_b + 0.16*s_nmf
  base = 0.48*uh_nmf + 0.26*logit_T8 + 0.13*mt_810 + 0.06*ss2 + 0.07*sm6
  final = 0.96*base + 0.04*proto_kde(bw=0.08)  → 0.991782

New hypothesis: KDE in other embedding spaces may unlock new diversity.
1. NMF-space proto KDE (different manifold from ICA)
2. STD-space proto KDE
3. ICA+NMF KDE blend
4. Top-P central positives KDE (50/75%)
5. Centroid cosine score blend
6. Adaptive bandwidth KDE
7. Multi-bandwidth KDE blend
8. Reciprocal-rank kernel
"""

import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
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

# ── Load pkl ────────────────────────────────────────────────────────────────
with open("outputs/embed_prior_model.pkl", "rb") as f:
    ep = pickle.load(f)

ew_ica = ep["emb_win_ica_norm"]
ew_pca = ep["emb_win_pca_norm"]
ew_std = ep["emb_win_std_norm"]
ew_nmf = ep["emb_win_nmf_norm"]
file_labels = ep["file_labels"]
cfg = ep["config"]

print(f"Loaded: ICA{ew_ica.shape} PCA{ew_pca.shape} NMF{ew_nmf.shape} STD{ew_std.shape}", flush=True)

# ── Helpers (exact Batch 86 versions) ─────────────────────────────────────
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
        sims = te @ tr.T
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
    """Reconstruction-error based subspace score (exact Batch 86 method)."""
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

def proto_kde_loo(ew, bw, proto_weighted=True):
    """Proto-weighted Gaussian KDE LOO."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]
            centroid = pos_wins.mean(0); centroid /= (norm(centroid) + EPS)
            if proto_weighted and len(pos_idx) > 1:
                proto_w = np.clip(pos_wins @ centroid, 0, None)
                proto_w = proto_w / (proto_w.sum() + EPS)
                kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
                ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
            else:
                ws[:, si] = np.clip(np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS)).mean(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def topp_kde_loo(bw, top_p=0.5):
    """KDE using only top-P most central positive windows (ICA space)."""
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
            centrality = pos_wins @ centroid
            k = max(1, int(np.ceil(len(pos_idx) * top_p)))
            top_idx = np.argsort(-centrality)[:k]
            filtered = pos_idx[top_idx]
            kern = np.exp((sims[:, filtered] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip(kern.mean(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def centroid_score_loo(ew):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            centroid = tr[pos_idx].mean(0); centroid /= (norm(centroid) + EPS)
            ws[:, si] = np.clip(te @ centroid, 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def adaptive_bw_kde_loo(bw_scale=1.0):
    """Proto KDE with per-species adaptive bandwidth."""
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
            if len(pos_idx) > 1:
                spread = float(1.0 - (pos_wins @ centroid).mean())
                bw = float(np.clip(bw_scale * (0.06 + 0.08 * spread), 0.04, 0.20))
            else:
                bw = bw_scale * 0.08
            proto_w = np.clip(pos_wins @ centroid, 0, None)
            proto_w = proto_w / (proto_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ── Pre-compute base components (exact Batch 86 params from pkl cfg) ────────
print("Pre-computing WL components...", flush=True)
t0 = time.time()
s_ica = wl_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
s_std = wl_loo(ew_std, cfg["std_pca80"]["k_neg"], cfg["std_pca80"]["w_max_pos"], cfg["std_pca80"]["w_max_agg"])
s_pca = wl_loo(ew_pca, cfg["pca80"]["k_neg"], cfg["pca80"]["w_max_pos"], cfg["pca80"]["w_max_agg"])
s_nmf = wl_loo(ew_nmf, cfg["nmf"]["k_neg"], cfg["nmf"]["w_max_pos"], cfg["nmf"]["w_max_agg"])
uh_b  = cfg["w_ica100"] * s_ica + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
uh_nmf = cfg["nmf"]["uh_scale"] * uh_b + cfg["nmf"]["w_nmf"] * s_nmf
print(f"  WL done ({time.time()-t0:.0f}s)", flush=True)

pT8  = make_logit_pred(cfg["logit_temperature"])
pmt_810 = (pT8 + make_logit_pred(10.0)) / 2
sm6 = make_softmax_pred(cfg["softmax_temp"])
print(f"  Logit done ({time.time()-t0:.0f}s)", flush=True)

ss2 = compute_subspace(ew_pca, n_comp=2, wma_ss=0.92)
print(f"  Subspace done ({time.time()-t0:.0f}s)", flush=True)

# Base and KDE reference
w_uh = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt_810 + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
print(f"\nBase AUC (should ~0.991359): {macro_auc(base_cur):.6f}", flush=True)

# ── Pre-compute KDE variants ──────────────────────────────────────────────
print("\nPre-computing KDE variants...", flush=True)
t1 = time.time()

kde_ica_pw08  = proto_kde_loo(ew_ica, bw=0.08, proto_weighted=True)
kde_ica_pw06  = proto_kde_loo(ew_ica, bw=0.06, proto_weighted=True)
kde_ica_pw10  = proto_kde_loo(ew_ica, bw=0.10, proto_weighted=True)

kde_nmf_pw06  = proto_kde_loo(ew_nmf, bw=0.06, proto_weighted=True)
kde_nmf_pw08  = proto_kde_loo(ew_nmf, bw=0.08, proto_weighted=True)
kde_nmf_pw10  = proto_kde_loo(ew_nmf, bw=0.10, proto_weighted=True)
kde_nmf_pw12  = proto_kde_loo(ew_nmf, bw=0.12, proto_weighted=True)

kde_std_pw06  = proto_kde_loo(ew_std, bw=0.06, proto_weighted=True)
kde_std_pw08  = proto_kde_loo(ew_std, bw=0.08, proto_weighted=True)
kde_std_pw10  = proto_kde_loo(ew_std, bw=0.10, proto_weighted=True)

kde_top50_08  = topp_kde_loo(bw=0.08, top_p=0.50)
kde_top75_08  = topp_kde_loo(bw=0.08, top_p=0.75)

cent_ica = centroid_score_loo(ew_ica)
cent_nmf = centroid_score_loo(ew_nmf)

kde_adap08 = adaptive_bw_kde_loo(bw_scale=1.0)
kde_adap12 = adaptive_bw_kde_loo(bw_scale=1.5)

print(f"  All KDE done ({time.time()-t1:.0f}s)", flush=True)

# Multi-bandwidth blends (normalized)
def blend_kde(kdes, weights):
    b = sum(w * k for w, k in zip(weights, kdes))
    for si in range(n_species):
        mx = b[:, si].max()
        if mx > EPS: b[:, si] /= mx
    return b

kde_multibw_ica = blend_kde([kde_ica_pw06, kde_ica_pw08, kde_ica_pw10], [0.25, 0.5, 0.25])
kde_multibw_nmf = blend_kde([kde_nmf_pw06, kde_nmf_pw08, kde_nmf_pw10], [0.25, 0.5, 0.25])

# Reference check
ref = macro_auc(0.96 * base_cur + 0.04 * kde_ica_pw08)
print(f"\nReference (should be ~0.991782): {ref:.6f}", flush=True)

# ── Load results JSON ──────────────────────────────────────────────────────
RES_PATH = Path("outputs/embed_prior_results.json")
with open(RES_PATH) as f:
    results = json.load(f)
best_auc = results["best"]["loo_auc"]
best_method = results["best"]["method"]
print(f"Current best: {best_method} = {best_auc:.6f}\n", flush=True)

experiments = []

def eval_blend(name, kde_score, w_kde):
    final = (1 - w_kde) * base_cur + w_kde * kde_score
    auc = macro_auc(final)
    delta = auc - best_auc
    print(f"  {name}: {auc:.6f}  (Δ={delta:+.6f})", flush=True)
    experiments.append({"method": name, "loo_auc": auc})
    return auc

print("=== Group 1: NMF-space proto KDE ===")
for bw_name, kde in [("06", kde_nmf_pw06), ("08", kde_nmf_pw08), ("10", kde_nmf_pw10), ("12", kde_nmf_pw12)]:
    for wk in [0.03, 0.04, 0.05, 0.06]:
        eval_blend(f"nmf_kde_bw{bw_name}_wk{int(wk*100):02d}", kde, wk)

print("\n=== Group 2: STD-space proto KDE ===")
for bw_name, kde in [("06", kde_std_pw06), ("08", kde_std_pw08), ("10", kde_std_pw10)]:
    for wk in [0.03, 0.04, 0.05, 0.06]:
        eval_blend(f"std_kde_bw{bw_name}_wk{int(wk*100):02d}", kde, wk)

print("\n=== Group 3: ICA+NMF KDE blend ===")
for w_ica_f in [0.3, 0.5, 0.7]:
    for bw_n, kde_n in [("06", kde_nmf_pw06), ("08", kde_nmf_pw08)]:
        b = blend_kde([kde_ica_pw08, kde_n], [w_ica_f, 1 - w_ica_f])
        tag = f"wi{int(w_ica_f*10)}_bn{bw_n}"
        for wk in [0.04, 0.05]:
            eval_blend(f"ica_nmf_blend_{tag}_wk{int(wk*100):02d}", b, wk)

print("\n=== Group 4: Top-P central positives KDE ===")
for name, kde in [("top50_08", kde_top50_08), ("top75_08", kde_top75_08)]:
    for wk in [0.03, 0.04, 0.05]:
        eval_blend(f"kde_{name}_wk{int(wk*100):02d}", kde, wk)

print("\n=== Group 5: Centroid score blend ===")
for name, cent in [("ica", cent_ica), ("nmf", cent_nmf)]:
    for wk in [0.02, 0.03, 0.04]:
        eval_blend(f"centroid_{name}_wk{int(wk*100):02d}", cent, wk)
cent_blend = blend_kde([cent_ica, cent_nmf], [0.5, 0.5])
for wk in [0.03, 0.04]:
    eval_blend(f"centroid_blend_wk{int(wk*100):02d}", cent_blend, wk)

print("\n=== Group 6: Adaptive bandwidth KDE ===")
for name, kde in [("adap08", kde_adap08), ("adap12", kde_adap12)]:
    for wk in [0.03, 0.04, 0.05]:
        eval_blend(f"kde_{name}_wk{int(wk*100):02d}", kde, wk)

print("\n=== Group 7: Multi-bandwidth blend ===")
for name, kde in [("ica_multibw", kde_multibw_ica), ("nmf_multibw", kde_multibw_nmf)]:
    for wk in [0.03, 0.04, 0.05]:
        eval_blend(f"{name}_wk{int(wk*100):02d}", kde, wk)

print("\n=== Group 8: ICA+NMF+STD three-way blend ===")
for w_ica_f in [0.5, 0.6]:
    for w_nmf_f in [0.3, 0.25]:
        w_std_f = 1.0 - w_ica_f - w_nmf_f
        if w_std_f < 0: continue
        b = blend_kde([kde_ica_pw08, kde_nmf_pw08, kde_std_pw08], [w_ica_f, w_nmf_f, w_std_f])
        tag = f"wi{int(w_ica_f*10)}_wn{int(w_nmf_f*10)}"
        for wk in [0.04, 0.05]:
            eval_blend(f"three_way_{tag}_wk{int(wk*100):02d}", b, wk)

# ── Finalize ────────────────────────────────────────────────────────────────
best_new = max(experiments, key=lambda x: x["loo_auc"])
delta_best = best_new["loo_auc"] - best_auc

print(f"\n{'='*60}")
print(f"Batch 87 Summary")
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
