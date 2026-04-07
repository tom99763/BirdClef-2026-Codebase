"""
Batch 85: Logit-Weighted KDE + Adaptive Bandwidth + Soft Expansion
===================================================================
Batch 84 found density ratio / Laplacian / Student-t KDE all below current best (0.991769).
Proto-weighted KDE (0.991597) was closest but still below.

Key hypothesis: Perch logit is a direct confidence signal per window per species.
Using logit as weights in KDE should focus the density on high-confidence training positives,
filtering out mislabeled or weak-signal windows.

Methods:
1. logit_kde_bXX_Tw_wYY   - KDE weighted by sig(logit_s / T_w) for each positive window
2. adaptive_bw_kde_wYY    - Per-species Silverman bandwidth based on # positives
3. soft_pos_kde_bXX_wYY   - Add soft positives (logit > thr) to hard positives, scale by p
4. geo_kde_bXX_wYY        - Geometric mean of KDE scores (product of kernels) vs arithmetic
5. topK_kde_bXX_K_wYY     - Use only top-K most similar positive windows (less noise)

All use per-species max normalization (correct, matching batch82).
Base: full NMF-Ultra + subspace (0.991187)
"""

import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.decomposition import PCA as SklearnPCA

# ── Load data ───────────────────────────────────────────────────────────────
DATA = np.load("outputs/perch_labeled_ss.npz")
embeddings  = DATA["emb"].astype(np.float32)
labels_raw  = DATA["labels"].astype(np.float32)
logits_raw  = DATA["logits"].astype(np.float32)
file_list   = DATA["file_list"]
n_windows   = DATA["n_windows"]
file_start  = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end    = np.cumsum(n_windows).astype(np.int32)
file_ids    = np.zeros(len(embeddings), dtype=np.int32)
for _fi in range(len(file_list)):
    file_ids[file_start[_fi]:file_end[_fi]] = _fi

n_files   = len(file_list)
n_species = labels_raw.shape[1]
EPS = 1e-8

# ── Load pkl ────────────────────────────────────────────────────────────────
with open("outputs/embed_prior_model.pkl", "rb") as f:
    ep = pickle.load(f)

ica = ep["ica"]; pca = ep["pca"]; scaler = ep["scaler"]; pca_std = ep["pca_std"]
nmf_model = ep["nmf_model"]; emb_min = ep["emb_min_shift"]; cfg = ep["config"]

# Pre-transform embeddings
print("Pre-transforming embeddings...", flush=True)
emb_ica_raw = ica.transform(embeddings).astype(np.float32)
emb_ica_n   = emb_ica_raw / (np.linalg.norm(emb_ica_raw, axis=1, keepdims=True) + EPS)

emb_shifted = np.maximum(embeddings - emb_min + 1e-6, 1e-8)
emb_nmf_raw = nmf_model.transform(emb_shifted).astype(np.float32)
emb_nmf_n   = emb_nmf_raw / (np.linalg.norm(emb_nmf_raw, axis=1, keepdims=True) + EPS)

emb_std_raw = scaler.transform(embeddings).astype(np.float32)
emb_std_pca = pca_std.transform(emb_std_raw).astype(np.float32)
emb_std_n   = emb_std_pca / (np.linalg.norm(emb_std_pca, axis=1, keepdims=True) + EPS)

emb_pca_raw = pca.transform(embeddings).astype(np.float32)
emb_pca_n   = emb_pca_raw / (np.linalg.norm(emb_pca_raw, axis=1, keepdims=True) + EPS)

# Pre-compute sigmoid logit signals (for KDE weighting)
logit_sig8  = (1.0 / (1.0 + np.exp(np.clip(-logits_raw / 8.0, -88, 88)))).astype(np.float32)
logit_sig4  = (1.0 / (1.0 + np.exp(np.clip(-logits_raw / 4.0, -88, 88)))).astype(np.float32)
print(f"Embeddings ready. ICA{emb_ica_n.shape}", flush=True)

# ── ROC-AUC helper ──────────────────────────────────────────────────────────
def roc_auc_macro(file_scores, file_labels):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            aucs.append(roc_auc_score(y, file_scores[:, si]))
    return float(np.mean(aucs)) if aucs else 0.0

# ── File-level labels ────────────────────────────────────────────────────────
file_labels = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    file_labels[fi] = (labels_raw[file_ids == fi].max(0) > 0.5).astype(np.float32)

# ── WL helpers for base computation ─────────────────────────────────────────
def wl_contrast_emb(te, tr, tl, k_neg, w_max_pos, w_max_agg):
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pos_m = tl[:, si] > 0.5; neg_m = tl[:, si] < 0.1
        if not pos_m.any(): ws[:, si] = 0.5; continue
        pos_w = tr[pos_m]; ps = te @ pos_w.T
        pp = pos_w.mean(0); pp /= np.linalg.norm(pp) + EPS
        sp = w_max_pos * ps.max(1) + (1 - w_max_pos) * (te @ pp)
        if neg_m.any():
            neg_w = tr[neg_m]; ns = te @ neg_w.T; k2 = min(k_neg, ns.shape[1])
            tn = neg_w[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
            tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
            ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)

def compute_base_loo():
    w_ica  = cfg["w_ica100"]; w_std = cfg["w_std"]; w_pca = cfg["w_pca80"]
    k_neg_ica = cfg["ica100"]["k_neg"]; wma_ica = cfg["ica100"]["w_max_agg"]; wmp_ica = cfg["ica100"]["w_max_pos"]
    k_neg_std = cfg["std_pca80"]["k_neg"]; wma_std = cfg["std_pca80"]["w_max_agg"]; wmp_std = cfg["std_pca80"]["w_max_pos"]
    k_neg_pca = cfg["pca80"]["k_neg"]; wma_pca = cfg["pca80"]["w_max_agg"]; wmp_pca = cfg["pca80"]["w_max_pos"]
    k_neg_nmf = cfg["nmf"]["k_neg"]; wma_nmf = cfg["nmf"]["w_max_agg"]; wmp_nmf = cfg["nmf"]["w_max_pos"]
    w_nmf = cfg["nmf"]["w_nmf"]
    w_logit = cfg["w_logit"]; w_mt = cfg["w_multit"]; w_sm = cfg["w_softmax"]
    T_sig = cfg.get("logit_temperature", 8.0); mt_temps = cfg.get("multit_temps", [8.0, 10.0])
    T_sm = cfg.get("softmax_temp", 4.0); w_ss = cfg.get("w_subspace", 0.06)

    base_scores = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tm = file_ids != fi; te_m = file_ids == fi; tl = labels_raw[tm]
        te_ica = emb_ica_n[te_m]; tr_ica = emb_ica_n[tm]
        te_std = emb_std_n[te_m]; tr_std = emb_std_n[tm]
        te_pca = emb_pca_n[te_m]; tr_pca = emb_pca_n[tm]
        te_nmf = emb_nmf_n[te_m]; tr_nmf = emb_nmf_n[tm]

        s_ica = wl_contrast_emb(te_ica, tr_ica, tl, k_neg_ica, wmp_ica, wma_ica)
        s_std = wl_contrast_emb(te_std, tr_std, tl, k_neg_std, wmp_std, wma_std)
        s_pca = wl_contrast_emb(te_pca, tr_pca, tl, k_neg_pca, wmp_pca, wma_pca)
        s_uh  = w_ica * s_ica + w_std * s_std + w_pca * s_pca

        sims_nmf = te_nmf @ tr_nmf.T
        ws_nmf = np.zeros((len(te_nmf), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]; ni = np.where(tl[:, si] < 0.1)[0]
            if len(pi) == 0: ws_nmf[:, si] = 0.5; continue
            ps = sims_nmf[:, pi]; pp = tr_nmf[pi].mean(0); pp /= np.linalg.norm(pp) + EPS
            sp = wmp_nmf * ps.max(1) + (1 - wmp_nmf) * (te_nmf @ pp)
            if len(ni) > 0:
                k2 = min(k_neg_nmf, len(ni)); ns = sims_nmf[:, ni]
                ti_idx = np.argsort(-ns, axis=1)[:, :k2]
                tn = np.array([tr_nmf[ni[ti_idx[j]]].mean(0) for j in range(len(te_nmf))], dtype=np.float32)
                tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
                ws_nmf[:, si] = (sp - (te_nmf * tn).sum(1) + 1) / 2
            else:
                ws_nmf[:, si] = (sp + 1) / 2
        s_nmf = wma_nmf * ws_nmf.max(0) + (1 - wma_nmf) * ws_nmf.mean(0)
        s_uh_nmf = (1 - w_nmf) * s_uh + w_nmf * s_nmf

        lg = logits_raw[te_m]
        sig_T = 1.0/(1.0+np.exp(np.clip(-lg/T_sig,-88,88))); s_sig = sig_T.max(0)
        sig_mt = np.mean([1.0/(1.0+np.exp(np.clip(-lg/T,-88,88))) for T in mt_temps],axis=0); s_mt = sig_mt.max(0)
        sm_r = lg/T_sm; sm_r -= sm_r.max(1, keepdims=True)
        sm_e = np.exp(sm_r); sm_p = sm_e/(sm_e.sum(1,keepdims=True)+EPS); s_sm = sm_p.max(0)

        # Subspace component
        te_pca_ss = emb_pca_n[te_m]; tr_pca_ss = emb_pca_n[tm]
        ws_ss = np.zeros((len(te_pca_ss), n_species), np.float32); dim_ss = te_pca_ss.shape[1]
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws_ss[:, si] = 0.5; continue
            pos = tr_pca_ss[pm]; k = min(2, len(pos)-1, dim_ss-1)
            if k < 1:
                pp = pos.mean(0); pp /= np.linalg.norm(pp)+EPS
                ws_ss[:, si] = np.clip((te_pca_ss @ pp + 1)/2, 0, 1); continue
            try:
                pca_sp = SklearnPCA(n_components=k); pca_sp.fit(pos)
                te_r = pca_sp.inverse_transform(pca_sp.transform(te_pca_ss))
                err = np.linalg.norm(te_pca_ss - te_r, axis=1)
                ws_ss[:, si] = np.clip(1 - err/(np.linalg.norm(te_pca_ss,axis=1)+EPS), 0, 1)
            except: ws_ss[:, si] = 0.5
        s_ss = 0.92 * ws_ss.max(0) + 0.08 * ws_ss.mean(0)
        w_uh = 1.0 - w_logit - w_mt - w_sm - w_ss
        base_scores[fi] = w_uh*s_uh_nmf + w_logit*s_sig + w_mt*s_mt + w_sm*s_sm + w_ss*s_ss
    return base_scores

print("Computing base LOO...", flush=True)
t0 = time.time()
base_loo = compute_base_loo()
file_labels = ep["file_labels"]  # Use pkl's stored file_labels for consistency
base_auc = roc_auc_macro(base_loo, file_labels)
print(f"Base LOO AUC: {base_auc:.6f}  ({time.time()-t0:.1f}s)", flush=True)

# ── Load results ─────────────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    results = json.load(f)
best_auc = results["best"]["loo_auc"]; best_method = results["best"]["method"]
tried_methods = set(e["method"] for e in results["experiments"])
print(f"Current best: {best_method} = {best_auc:.6f}", flush=True)

new_experiments = []

def run_exp(name, scores_loo, w_kde):
    if name in tried_methods:
        print(f"  SKIP: {name}"); return None
    blend = (1.0 - w_kde) * base_loo + w_kde * scores_loo
    auc = roc_auc_macro(blend, file_labels)
    print(f"  {name}: {auc:.6f}  (Δ={auc-best_auc:+.6f})")
    return {"method": name, "loo_auc": float(auc), "config": {"w_kde": w_kde, "base_auc": base_auc}}

# ── Group 1: Logit-weighted KDE ──────────────────────────────────────────────
# Weight each positive training window by sig(logit_s / T_w)
print("\n[Group 1] Logit-weighted Gaussian KDE (ICA space)", flush=True)
def logit_weighted_kde_loo(emb_n, lw_win, bw, T_w):
    """KDE where each positive window is weighted by sig(logit/T_w)."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[file_ids == fi]
        tm = file_ids != fi
        tr = emb_n[tm]; tl = labels_raw[tm]; lw = lw_win[tm]  # logit weights for train
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]
            if len(pi) == 0: ws[:, si] = 0.5; continue
            pos_sims = sims[:, pi]  # (n_te, n_pos)
            # Weight by logit confidence of each positive window
            w_pos = lw[pi, si]  # sig(logit_s / T_w) for each positive window
            w_pos = w_pos / (w_pos.sum() + EPS)  # normalize
            kern = np.exp((pos_sims - 1.0) / (bw**2 + EPS))  # (n_te, n_pos)
            ws[:, si] = np.clip((kern * w_pos[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for T_w in [4.0, 8.0]:
    lw_name = f"T{int(T_w)}"
    lw_win = 1.0/(1.0+np.exp(np.clip(-logits_raw/T_w,-88,88)))
    for bw in [0.08, 0.10, 0.12]:
        t0 = time.time()
        scores = logit_weighted_kde_loo(emb_ica_n, lw_win, bw=bw, T_w=T_w)
        bw_tag = f"{int(bw*100):02d}"
        for w in [0.03, 0.04, 0.05]:
            r = run_exp(f"kde_lw{lw_name}_bw{bw_tag}_w{int(w*100):02d}", scores, w)
            if r: new_experiments.append(r)
        print(f"  Tw={T_w} bw={bw:.2f} done ({time.time()-t0:.1f}s)", flush=True)

# ── Group 2: Adaptive bandwidth KDE (Silverman's rule per species) ───────────
print("\n[Group 2] Adaptive bandwidth KDE (Silverman's rule)", flush=True)
def adaptive_bw_kde_loo(emb_n, scale=1.0):
    """Per-species bandwidth: bw_s = scale * sigma_s * n_s^(-1/5)  (Silverman)."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[file_ids == fi]
        tm = file_ids != fi
        tr = emb_n[tm]; tl = labels_raw[tm]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]
            if len(pi) == 0: ws[:, si] = 0.5; continue
            pos = tr[pi]
            # Silverman's bandwidth: use std of cosine distances to centroid
            centroid = pos.mean(0); centroid /= np.linalg.norm(centroid) + EPS
            dists = 1.0 - (pos @ centroid)  # cosine distances
            sigma = max(dists.std(), 0.01)  # avoid zero
            bw_s = scale * sigma * (len(pi) ** (-0.2))
            bw_s = np.clip(bw_s, 0.05, 0.5)
            pos_sims = sims[:, pi]
            ws[:, si] = np.clip(np.exp((pos_sims - 1.0) / (bw_s**2 + EPS)).mean(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for scale in [0.5, 1.0, 1.5, 2.0]:
    t0 = time.time()
    scores = adaptive_bw_kde_loo(emb_ica_n, scale=scale)
    for w in [0.03, 0.04, 0.05]:
        r = run_exp(f"kde_adaptive_s{int(scale*10):02d}_w{int(w*100):02d}", scores, w)
        if r: new_experiments.append(r)
    print(f"  scale={scale:.1f} done ({time.time()-t0:.1f}s)", flush=True)

# ── Group 3: Soft-positive expansion KDE ─────────────────────────────────────
# Add soft positives: windows with logit > thr (not just labeled positives)
print("\n[Group 3] Soft-positive expansion KDE", flush=True)
def soft_pos_kde_loo(emb_n, bw, hard_weight=1.0, soft_thr=0.6, soft_weight=0.3):
    """Expand positives to include windows with sig(logit/8) > soft_thr."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[file_ids == fi]
        tm = file_ids != fi
        tr = emb_n[tm]; tl = labels_raw[tm]; ls8 = logit_sig8[tm]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            hard_pi = np.where(tl[:, si] > 0.5)[0]
            soft_pi  = np.where((tl[:, si] <= 0.5) & (ls8[:, si] > soft_thr))[0]
            if len(hard_pi) == 0 and len(soft_pi) == 0:
                ws[:, si] = 0.5; continue
            all_idx = np.concatenate([hard_pi, soft_pi]) if len(soft_pi) > 0 else hard_pi
            w_arr = np.concatenate([
                np.full(len(hard_pi), hard_weight),
                np.full(len(soft_pi), soft_weight)
            ]) if len(soft_pi) > 0 else np.ones(len(hard_pi))
            w_arr /= w_arr.sum() + EPS
            pos_sims = sims[:, all_idx]
            kern = np.exp((pos_sims - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * w_arr[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for soft_thr in [0.65, 0.70, 0.75]:
    for sw in [0.2, 0.3]:
        t0 = time.time()
        scores = soft_pos_kde_loo(emb_ica_n, bw=0.1, hard_weight=1.0, soft_thr=soft_thr, soft_weight=sw)
        thr_tag = f"{int(soft_thr*100):02d}"; sw_tag = f"{int(sw*10):01d}"
        for w in [0.03, 0.04, 0.05]:
            r = run_exp(f"kde_soft_thr{thr_tag}_sw{sw_tag}_w{int(w*100):02d}", scores, w)
            if r: new_experiments.append(r)
        print(f"  thr={soft_thr:.2f} sw={sw:.1f} done ({time.time()-t0:.1f}s)", flush=True)

# ── Group 4: Top-K KDE (only use top-K most similar positive windows) ─────────
print("\n[Group 4] Top-K KDE (focus on nearest positives)", flush=True)
def topk_kde_loo(emb_n, bw, K):
    """Use only top-K most similar positive training windows for KDE."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[file_ids == fi]
        tm = file_ids != fi
        tr = emb_n[tm]; tl = labels_raw[tm]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]
            if len(pi) == 0: ws[:, si] = 0.5; continue
            pos_sims = sims[:, pi]  # (n_te, n_pos)
            k_use = min(K, len(pi))
            if k_use < len(pi):
                # For each test window, take top-k positives
                top_idx = np.argsort(-pos_sims, axis=1)[:, :k_use]  # (n_te, k_use)
                top_sims = pos_sims[np.arange(len(te))[:, None], top_idx]  # (n_te, k_use)
            else:
                top_sims = pos_sims
            ws[:, si] = np.clip(np.exp((top_sims - 1.0) / (bw**2 + EPS)).mean(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for K in [3, 5, 10, 20]:
    t0 = time.time()
    scores = topk_kde_loo(emb_ica_n, bw=0.1, K=K)
    for w in [0.03, 0.04, 0.05]:
        r = run_exp(f"kde_topk{K:02d}_bw10_w{int(w*100):02d}", scores, w)
        if r: new_experiments.append(r)
    print(f"  K={K} done ({time.time()-t0:.1f}s)", flush=True)

# ── Group 5: Geometric KDE (product of kernels = geometric mean) ─────────────
print("\n[Group 5] Geometric mean KDE (product of kernels)", flush=True)
def geo_kde_loo(emb_n, bw):
    """Geometric mean: exp(mean(log kernel)) = product^(1/n) of kernel values."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[file_ids == fi]
        tm = file_ids != fi
        tr = emb_n[tm]; tl = labels_raw[tm]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]
            if len(pi) == 0: ws[:, si] = 0.5; continue
            pos_sims = sims[:, pi]
            # log kernel = (sim - 1) / bw^2, then mean → geometric mean
            log_kern = (pos_sims - 1.0) / (bw**2 + EPS)
            ws[:, si] = np.clip(np.exp(log_kern.mean(1)), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for bw in [0.08, 0.10, 0.15]:
    t0 = time.time()
    scores = geo_kde_loo(emb_ica_n, bw=bw)
    bw_tag = f"{int(bw*100):02d}"
    for w in [0.03, 0.04, 0.05]:
        r = run_exp(f"kde_geo_bw{bw_tag}_w{int(w*100):02d}", scores, w)
        if r: new_experiments.append(r)
    print(f"  bw={bw:.2f} done ({time.time()-t0:.1f}s)", flush=True)

# ── Group 6: Max-KDE (max over positive windows instead of mean) ──────────────
print("\n[Group 6] Max-KDE (max over positives)", flush=True)
def max_kde_loo(emb_n, bw):
    """Max over positive training windows: max_i exp((cosim-1)/bw^2)."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[file_ids == fi]
        tm = file_ids != fi
        tr = emb_n[tm]; tl = labels_raw[tm]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]
            if len(pi) == 0: ws[:, si] = 0.5; continue
            pos_sims = sims[:, pi]
            ws[:, si] = np.clip(np.exp((pos_sims.max(1) - 1.0) / (bw**2 + EPS)), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for bw in [0.08, 0.10, 0.12, 0.15]:
    t0 = time.time()
    scores = max_kde_loo(emb_ica_n, bw=bw)
    bw_tag = f"{int(bw*100):02d}"
    for w in [0.03, 0.04, 0.05]:
        r = run_exp(f"kde_max_bw{bw_tag}_w{int(w*100):02d}", scores, w)
        if r: new_experiments.append(r)
    print(f"  bw={bw:.2f} done ({time.time()-t0:.1f}s)", flush=True)

# ── Group 7: Combined logit-weighted + adaptive BW ───────────────────────────
print("\n[Group 7] Logit-weighted + adaptive BW combined", flush=True)
def logit_adaptive_kde_loo(emb_n, T_w=8.0, scale=1.0):
    lw_win_ta = 1.0/(1.0+np.exp(np.clip(-logits_raw/T_w,-88,88)))
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[file_ids == fi]
        tm = file_ids != fi
        tr = emb_n[tm]; tl = labels_raw[tm]; lw = lw_win_ta[tm]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]
            if len(pi) == 0: ws[:, si] = 0.5; continue
            pos = tr[pi]
            centroid = pos.mean(0); centroid /= np.linalg.norm(centroid)+EPS
            dists = 1.0 - (pos @ centroid)
            sigma = max(dists.std(), 0.01)
            bw_s = np.clip(scale * sigma * (len(pi)**(-0.2)), 0.05, 0.5)
            w_pos = lw[pi, si]; w_pos = w_pos / (w_pos.sum() + EPS)
            kern = np.exp((sims[:, pi] - 1.0) / (bw_s**2 + EPS))
            ws[:, si] = np.clip((kern * w_pos[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for T_w in [4.0, 8.0]:
    for scale in [0.5, 1.0]:
        t0 = time.time()
        scores = logit_adaptive_kde_loo(emb_ica_n, T_w=T_w, scale=scale)
        for w in [0.04, 0.05]:
            r = run_exp(f"kde_lw_adap_T{int(T_w)}_s{int(scale*10):02d}_w{int(w*100):02d}", scores, w)
            if r: new_experiments.append(r)
        print(f"  Tw={T_w} scale={scale:.1f} done ({time.time()-t0:.1f}s)", flush=True)

# ── Summary ──────────────────────────────────────────────────────────────────
valid = [e for e in new_experiments if e is not None]
if valid:
    best_new = max(valid, key=lambda x: x["loo_auc"])
    print(f"\n=== Batch 85 Summary ===")
    print(f"Experiments: {len(valid)}")
    print(f"Best new: {best_new['method']} = {best_new['loo_auc']:.6f}")
    print(f"Current best: {best_method} = {best_auc:.6f}")
    print(f"Delta: {best_new['loo_auc'] - best_auc:+.6f}")

    results["experiments"].extend(valid)
    if best_new["loo_auc"] > best_auc:
        results["best"] = {"method": best_new["method"], "loo_auc": best_new["loo_auc"], "full_auc": best_new["loo_auc"]}
        print(f"*** NEW BEST ***")

    with open("outputs/embed_prior_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved.")

    print("\nTop 5:")
    for e in sorted(valid, key=lambda x: -x["loo_auc"])[:5]:
        print(f"  {e['method']}: {e['loo_auc']:.6f}")
