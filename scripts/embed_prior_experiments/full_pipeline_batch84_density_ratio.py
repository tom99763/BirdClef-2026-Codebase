"""
Batch 84: Density Ratio KDE + Kernel Variants
==============================================
Hypothesis: Positive/negative density ratio is strictly better than
positive-only KDE because it incorporates negative contrast information.

Methods:
1. kde_density_ratio_bXX_wYY  - LR(x,s) = pos_kde / (pos_kde + neg_kde)
   sweep: bw ∈ {0.08, 0.10, 0.12, 0.15}, w_kde ∈ {0.03, 0.04, 0.05, 0.06}
2. kde_student_t_bXX_wYY      - Student-t kernel: (1 + dist/nu)^(-(nu+1)/2)
3. kde_laplacian_bXX_wYY      - Laplacian kernel: exp(-|1-cosim|/bw)
4. kde_proto_weighted_wYY     - Weight positives by their prototypicality

All use per-species max normalization (correct, matching batch82).
Base: 0.96×NMF-Ultra + w_kde×new_component
"""

import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.decomposition import PCA as SklearnPCA

# ── Load data ───────────────────────────────────────────────────────────────
DATA = np.load("outputs/perch_labeled_ss.npz")
embeddings  = DATA["emb"].astype(np.float32)           # (739, 1536)
labels_raw  = DATA["labels"].astype(np.float32)        # (739, 234)
logits_raw  = DATA["logits"].astype(np.float32)        # (739, 234)
file_list   = DATA["file_list"]                        # (66,)  — unique file names
n_windows   = DATA["n_windows"]                        # (66,)  — windows per file
# Build file_ids using n_windows order (matches pkl's win_file_id)
file_start  = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end    = np.cumsum(n_windows).astype(np.int32)
file_ids    = np.zeros(len(embeddings), dtype=np.int32)
for _fi in range(len(file_list)):
    file_ids[file_start[_fi]:file_end[_fi]] = _fi

unique_files = np.arange(len(file_list), dtype=np.int32)
n_files   = len(unique_files)
n_species = labels_raw.shape[1]
EPS = 1e-8

# ── Load existing best pkl (for base score + ICA transform) ─────────────────
with open("outputs/embed_prior_model.pkl", "rb") as f:
    ep = pickle.load(f)

ica        = ep["ica"]
pca        = ep["pca"]
scaler     = ep["scaler"]
pca_std    = ep["pca_std"]
nmf_model  = ep["nmf_model"]
emb_min    = ep["emb_min_shift"]
cfg        = ep["config"]

# Pre-transform all embeddings once
print("Pre-transforming embeddings...")
# ICA space (normalized)
emb_ica_raw = ica.transform(embeddings).astype(np.float32)
norms = np.linalg.norm(emb_ica_raw, axis=1, keepdims=True) + EPS
emb_ica_n   = emb_ica_raw / norms

# Also need NMF-based base for the full blend
emb_shifted = np.maximum(embeddings - emb_min + 1e-6, 1e-8)
emb_nmf_raw = nmf_model.transform(emb_shifted).astype(np.float32)
nmf_norms   = np.linalg.norm(emb_nmf_raw, axis=1, keepdims=True) + EPS
emb_nmf_n   = emb_nmf_raw / nmf_norms

# STD/PCA space
emb_std_raw = scaler.transform(embeddings).astype(np.float32)
emb_std_pca = pca_std.transform(emb_std_raw).astype(np.float32)
std_norms   = np.linalg.norm(emb_std_pca, axis=1, keepdims=True) + EPS
emb_std_n   = emb_std_pca / std_norms

# PCA-80 space
emb_pca_raw = pca.transform(embeddings).astype(np.float32)
pca_norms   = np.linalg.norm(emb_pca_raw, axis=1, keepdims=True) + EPS
emb_pca_n   = emb_pca_raw / pca_norms

print(f"Embeddings transformed: ICA{emb_ica_n.shape}, NMF{emb_nmf_n.shape}")

# ── ROC-AUC helper ──────────────────────────────────────────────────────────
def roc_auc_macro(file_scores, file_labels):
    """macro ROC-AUC over species with at least one positive."""
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
    mask = file_ids == fi
    file_labels[fi] = labels_raw[mask].max(0)

# ── Base score (NMF-Ultra, same as batch82) ──────────────────────────────────
def wl_contrast_emb(te, tr, tl, k_neg, w_max_pos, w_max_agg):
    """WL contrast for one embedding space."""
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pos_m = tl[:, si] > 0.5
        neg_m = tl[:, si] < 0.1
        if not pos_m.any(): ws[:, si] = 0.5; continue
        pos_w = tr[pos_m]
        ps = te @ pos_w.T
        pp = pos_w.mean(0); pp /= np.linalg.norm(pp) + EPS
        sp = w_max_pos * ps.max(1) + (1 - w_max_pos) * (te @ pp)
        if neg_m.any():
            neg_w = tr[neg_m]
            ns = te @ neg_w.T
            k2 = min(k_neg, ns.shape[1])
            tn = neg_w[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
            tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
            ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)

def compute_base_loo():
    """Compute NMF-Ultra base LOO scores."""
    # Config from pkl
    w_ica  = cfg["w_ica100"]; w_std = cfg["w_std"]; w_pca = cfg["w_pca80"]
    k_neg_ica = cfg["ica100"]["k_neg"]; wma_ica = cfg["ica100"]["w_max_agg"]; wmp_ica = cfg["ica100"]["w_max_pos"]
    k_neg_std = cfg["std_pca80"]["k_neg"]; wma_std = cfg["std_pca80"]["w_max_agg"]; wmp_std = cfg["std_pca80"]["w_max_pos"]
    k_neg_pca = cfg["pca80"]["k_neg"]; wma_pca = cfg["pca80"]["w_max_agg"]; wmp_pca = cfg["pca80"]["w_max_pos"]
    k_neg_nmf = cfg["nmf"]["k_neg"]; wma_nmf = cfg["nmf"]["w_max_agg"]; wmp_nmf = cfg["nmf"]["w_max_pos"]
    w_nmf   = cfg["nmf"]["w_nmf"]
    w_logit = cfg["w_logit"]; w_mt = cfg["w_multit"]; w_sm = cfg["w_softmax"]
    T_sig = cfg.get("logit_temperature", 8.0)
    mt_temps = cfg.get("multit_temps", [8.0, 10.0])
    T_sm = cfg.get("softmax_temp", 4.0)

    base_scores = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tm = file_ids != fi; te_m = file_ids == fi
        te_ica = emb_ica_n[te_m]; tr_ica = emb_ica_n[tm]
        te_std = emb_std_n[te_m]; tr_std = emb_std_n[tm]
        te_pca = emb_pca_n[te_m]; tr_pca = emb_pca_n[tm]
        te_nmf = emb_nmf_n[te_m]; tr_nmf = emb_nmf_n[tm]
        tl = labels_raw[tm]

        s_ica = wl_contrast_emb(te_ica, tr_ica, tl, k_neg_ica, wmp_ica, wma_ica)
        s_std = wl_contrast_emb(te_std, tr_std, tl, k_neg_std, wmp_std, wma_std)
        s_pca = wl_contrast_emb(te_pca, tr_pca, tl, k_neg_pca, wmp_pca, wma_pca)
        s_uh  = w_ica * s_ica + w_std * s_std + w_pca * s_pca

        # NMF WL (simplified)
        sims_nmf = te_nmf @ tr_nmf.T
        ws_nmf   = np.zeros((len(te_nmf), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]
            ni = np.where(tl[:, si] < 0.1)[0]
            if len(pi) == 0: ws_nmf[:, si] = 0.5; continue
            ps = sims_nmf[:, pi]
            pp = tr_nmf[pi].mean(0); pp /= np.linalg.norm(pp) + EPS
            sp = wmp_nmf * ps.max(1) + (1 - wmp_nmf) * (te_nmf @ pp)
            if len(ni) > 0:
                k2 = min(k_neg_nmf, len(ni))
                ns = sims_nmf[:, ni]
                ti = np.argsort(-ns, axis=1)[:, :k2]
                tn = np.array([tr_nmf[ni[ti[j]]].mean(0) for j in range(len(te_nmf))], dtype=np.float32)
                tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
                ws_nmf[:, si] = (sp - (te_nmf * tn).sum(1) + 1) / 2
            else:
                ws_nmf[:, si] = (sp + 1) / 2
        s_nmf = wma_nmf * ws_nmf.max(0) + (1 - wma_nmf) * ws_nmf.mean(0)
        s_uh_nmf = (1 - w_nmf) * s_uh + w_nmf * s_nmf

        # Logit components
        lg = logits_raw[te_m]
        sig_T  = 1.0 / (1.0 + np.exp(np.clip(-lg / T_sig, -88, 88)))
        s_sig  = sig_T.max(0)
        sig_mt = np.mean([1.0/(1.0+np.exp(np.clip(-lg/T,-88,88))) for T in mt_temps], axis=0)
        s_mt   = sig_mt.max(0)
        sm_r   = lg / T_sm; sm_r -= sm_r.max(1, keepdims=True)
        sm_e   = np.exp(sm_r); sm_p = sm_e / (sm_e.sum(1, keepdims=True) + EPS)
        s_sm   = sm_p.max(0)

        # Species subspace LOO (PCA-2 per species on PCA-80 space, wma=0.92, W_SS=0.06)
        te_pca_ss = emb_pca_n[te_m]; tr_pca_ss = emb_pca_n[tm]
        ws_ss = np.zeros((len(te_pca_ss), n_species), np.float32)
        dim_ss = te_pca_ss.shape[1]
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws_ss[:, si] = 0.5; continue
            pos = tr_pca_ss[pm]; k = min(2, len(pos)-1, dim_ss-1)
            if k < 1:
                pp = pos.mean(0); pp /= np.linalg.norm(pp) + EPS
                ws_ss[:, si] = np.clip((te_pca_ss @ pp + 1) / 2, 0, 1); continue
            try:
                pca_sp = SklearnPCA(n_components=k); pca_sp.fit(pos)
                te_r = pca_sp.inverse_transform(pca_sp.transform(te_pca_ss))
                err = np.linalg.norm(te_pca_ss - te_r, axis=1)
                ws_ss[:, si] = np.clip(1 - err / (np.linalg.norm(te_pca_ss, axis=1) + EPS), 0, 1)
            except:
                ws_ss[:, si] = 0.5
        s_ss = 0.92 * ws_ss.max(0) + 0.08 * ws_ss.mean(0)

        w_ss   = cfg.get("w_subspace", 0.06)
        w_uh   = 1.0 - w_logit - w_mt - w_sm - w_ss
        base_scores[fi] = w_uh * s_uh_nmf + w_logit * s_sig + w_mt * s_mt + w_sm * s_sm + w_ss * s_ss
    return base_scores

print("Computing base LOO scores (NMF-Ultra)...")
t0 = time.time()
base_loo = compute_base_loo()
base_auc = roc_auc_macro(base_loo, file_labels)
print(f"Base LOO AUC: {base_auc:.6f}  ({time.time()-t0:.1f}s)")

# ── KDE helpers (per-species max normalization) ──────────────────────────────
def kde_positive_loo(emb_n, bw):
    """Positive-only Gaussian KDE (batch82 correct method)."""
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
            ws[:, si] = np.clip(np.exp((sims[:, pi] - 1.0) / (bw**2 + EPS)).mean(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def kde_density_ratio_loo(emb_n, bw):
    """Density ratio KDE: pos_kde / (pos_kde + neg_kde).
    Incorporates negative contrast → should beat positive-only KDE."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[file_ids == fi]
        tm = file_ids != fi
        tr = emb_n[tm]; tl = labels_raw[tm]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]
            ni = np.where(tl[:, si] < 0.1)[0]
            if len(pi) == 0: ws[:, si] = 0.5; continue
            pos_kde = np.clip(np.exp((sims[:, pi] - 1.0) / (bw**2 + EPS)).mean(1), 0, None)
            if len(ni) > 0:
                neg_kde = np.clip(np.exp((sims[:, ni] - 1.0) / (bw**2 + EPS)).mean(1), 0, None)
                ws[:, si] = pos_kde / (pos_kde + neg_kde + EPS)
            else:
                ws[:, si] = pos_kde
        # Per-species max normalization
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def kde_laplacian_loo(emb_n, bw):
    """Laplacian kernel KDE: exp(-|1-cosim|/bw)."""
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
            # Laplacian: exp(-(1 - cosim)/bw)  [dist = 1 - cosim >= 0]
            ws[:, si] = np.exp(-(1.0 - sims[:, pi]) / (bw + EPS)).mean(1)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def kde_student_t_loo(emb_n, bw, nu=3.0):
    """Student-t kernel KDE: (1 + (1-cosim)/(nu*bw^2))^(-(nu+1)/2)."""
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
            dist = np.clip(1.0 - sims[:, pi], 0, 2.0)  # (n_te, n_pos) cosine distance
            kernel = (1.0 + dist / (nu * bw**2 + EPS)) ** (-(nu + 1) / 2)  # (n_te, n_pos)
            ws[:, si] = kernel.mean(1)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def kde_proto_weighted_loo(emb_n, bw):
    """Proto-weighted KDE: weight each positive window by its similarity to centroid.
    Upweights 'typical' positives, downweights outliers."""
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
            # Prototype weights: similarity of each pos to mean pos
            centroid = pos.mean(0); centroid /= np.linalg.norm(centroid) + EPS
            proto_w = np.clip(pos @ centroid, 0, None)  # (n_pos,)
            proto_w = proto_w / (proto_w.sum() + EPS)   # normalize weights
            # Weighted KDE
            kern = np.exp((sims[:, pi] - 1.0) / (bw**2 + EPS))  # (n_te, n_pos)
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def kde_density_ratio_laplacian_loo(emb_n, bw):
    """Density ratio with Laplacian kernel."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[file_ids == fi]
        tm = file_ids != fi
        tr = emb_n[tm]; tl = labels_raw[tm]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]
            ni = np.where(tl[:, si] < 0.1)[0]
            if len(pi) == 0: ws[:, si] = 0.5; continue
            pos_kde = np.exp(-(1.0 - sims[:, pi]) / (bw + EPS)).mean(1)
            if len(ni) > 0:
                neg_kde = np.exp(-(1.0 - sims[:, ni]) / (bw + EPS)).mean(1)
                ws[:, si] = pos_kde / (pos_kde + neg_kde + EPS)
            else:
                ws[:, si] = pos_kde
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ── Load results JSON ────────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    results = json.load(f)

best_auc   = results["best"]["loo_auc"]
best_method = results["best"]["method"]
tried_methods = set(e["method"] for e in results["experiments"])
print(f"\nCurrent best: {best_method} = {best_auc:.6f}")

new_experiments = []

def run_experiment(method_name, scores_loo, w_kde):
    """Blend base + new component, compute AUC, log result."""
    if method_name in tried_methods:
        print(f"  SKIP (already tried): {method_name}")
        return None
    blend = (1.0 - w_kde) * base_loo + w_kde * scores_loo
    auc = roc_auc_macro(blend, file_labels)
    print(f"  {method_name}: {auc:.6f}  (Δ={auc - best_auc:+.6f})")
    return {"method": method_name, "loo_auc": float(auc),
            "config": {"w_kde": w_kde, "base_auc": base_auc}}

# ── Experiment group 1: Positive KDE reference (bw=0.1, per-species norm) ───
print("\n[Group 0] Baseline KDE (verify correct normalization)")
t0 = time.time()
kde_pos_01 = kde_positive_loo(emb_ica_n, bw=0.1)
for w in [0.04]:
    res = run_experiment(f"kde_pos_bw01_w{int(w*100):02d}_perspecies", kde_pos_01, w)
    if res: new_experiments.append(res)
print(f"  (time: {time.time()-t0:.1f}s)")

# ── Experiment group 2: Density ratio KDE (Gaussian) ────────────────────────
print("\n[Group 1] Density ratio KDE (Gaussian kernel)")
for bw in [0.08, 0.10, 0.12, 0.15]:
    t0 = time.time()
    dr_scores = kde_density_ratio_loo(emb_ica_n, bw=bw)
    bw_tag = f"{int(bw*100):02d}"
    for w in [0.03, 0.04, 0.05, 0.06]:
        res = run_experiment(f"kde_dr_bw{bw_tag}_w{int(w*100):02d}", dr_scores, w)
        if res: new_experiments.append(res)
    print(f"  bw={bw:.2f} done ({time.time()-t0:.1f}s)")

# ── Experiment group 3: Laplacian KDE ────────────────────────────────────────
print("\n[Group 2] Laplacian KDE")
for bw in [0.10, 0.15, 0.20, 0.30]:
    t0 = time.time()
    lap_scores = kde_laplacian_loo(emb_ica_n, bw=bw)
    bw_tag = f"{int(bw*100):02d}"
    for w in [0.03, 0.04, 0.05]:
        res = run_experiment(f"kde_lap_bw{bw_tag}_w{int(w*100):02d}", lap_scores, w)
        if res: new_experiments.append(res)
    print(f"  bw={bw:.2f} done ({time.time()-t0:.1f}s)")

# ── Experiment group 4: Student-t KDE ────────────────────────────────────────
print("\n[Group 3] Student-t KDE (nu=3)")
for bw in [0.10, 0.15, 0.20]:
    t0 = time.time()
    st_scores = kde_student_t_loo(emb_ica_n, bw=bw, nu=3.0)
    bw_tag = f"{int(bw*100):02d}"
    for w in [0.03, 0.04, 0.05]:
        res = run_experiment(f"kde_st3_bw{bw_tag}_w{int(w*100):02d}", st_scores, w)
        if res: new_experiments.append(res)
    print(f"  bw={bw:.2f} done ({time.time()-t0:.1f}s)")

# ── Experiment group 5: Proto-weighted KDE ───────────────────────────────────
print("\n[Group 4] Proto-weighted KDE")
for bw in [0.08, 0.10, 0.15]:
    t0 = time.time()
    pw_scores = kde_proto_weighted_loo(emb_ica_n, bw=bw)
    bw_tag = f"{int(bw*100):02d}"
    for w in [0.03, 0.04, 0.05]:
        res = run_experiment(f"kde_pw2_bw{bw_tag}_w{int(w*100):02d}", pw_scores, w)
        if res: new_experiments.append(res)
    print(f"  bw={bw:.2f} done ({time.time()-t0:.1f}s)")

# ── Experiment group 6: Density ratio Laplacian ──────────────────────────────
print("\n[Group 5] Density ratio Laplacian KDE")
for bw in [0.10, 0.15, 0.20]:
    t0 = time.time()
    drl_scores = kde_density_ratio_laplacian_loo(emb_ica_n, bw=bw)
    bw_tag = f"{int(bw*100):02d}"
    for w in [0.03, 0.04, 0.05]:
        res = run_experiment(f"kde_drl_bw{bw_tag}_w{int(w*100):02d}", drl_scores, w)
        if res: new_experiments.append(res)
    print(f"  bw={bw:.2f} done ({time.time()-t0:.1f}s)")

# ── Experiment group 7: Density ratio on NMF space ───────────────────────────
print("\n[Group 6] Density ratio KDE on NMF space")
for bw in [0.08, 0.10, 0.12]:
    t0 = time.time()
    dr_nmf = kde_density_ratio_loo(emb_nmf_n, bw=bw)
    bw_tag = f"{int(bw*100):02d}"
    for w in [0.03, 0.04, 0.05]:
        res = run_experiment(f"kde_dr_nmf_bw{bw_tag}_w{int(w*100):02d}", dr_nmf, w)
        if res: new_experiments.append(res)
    print(f"  bw={bw:.2f} NMF done ({time.time()-t0:.1f}s)")

# ── Experiment group 8: Blend ICA + NMF density ratios ───────────────────────
print("\n[Group 7] Blend ICA density ratio + NMF density ratio")
dr_ica_010 = kde_density_ratio_loo(emb_ica_n, bw=0.10)
dr_nmf_010 = kde_density_ratio_loo(emb_nmf_n, bw=0.10)
for alpha_ica in [0.5, 0.6, 0.7, 0.8]:
    mixed = alpha_ica * dr_ica_010 + (1 - alpha_ica) * dr_nmf_010
    for w in [0.04, 0.05]:
        tag = f"kde_dr_ica{int(alpha_ica*10)}_nmf{int((1-alpha_ica)*10)}_w{int(w*100):02d}"
        res = run_experiment(tag, mixed, w)
        if res: new_experiments.append(res)
print(f"  ICA+NMF blend done")

# ── Determine new best ────────────────────────────────────────────────────────
valid = [e for e in new_experiments if e is not None]
if not valid:
    print("\nNo new experiments ran (all already tried).")
else:
    best_new = max(valid, key=lambda x: x["loo_auc"])
    print(f"\n=== Batch 84 Summary ===")
    print(f"Experiments run: {len(valid)}")
    print(f"Best new: {best_new['method']} = {best_new['loo_auc']:.6f}")
    print(f"Current best: {best_method} = {best_auc:.6f}")
    print(f"Delta: {best_new['loo_auc'] - best_auc:+.6f}")

    # Update JSON
    results["experiments"].extend(valid)
    if best_new["loo_auc"] > best_auc:
        results["best"] = {
            "method": best_new["method"],
            "loo_auc": best_new["loo_auc"],
            "full_auc": best_new["loo_auc"]
        }
        print(f"\n*** NEW BEST: {best_new['method']} = {best_new['loo_auc']:.6f} ***")

    with open("outputs/embed_prior_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved.")

    # Print top 5 new results
    valid_sorted = sorted(valid, key=lambda x: -x["loo_auc"])
    print("\nTop 5 new experiments:")
    for e in valid_sorted[:5]:
        print(f"  {e['method']}: {e['loo_auc']:.6f}")
