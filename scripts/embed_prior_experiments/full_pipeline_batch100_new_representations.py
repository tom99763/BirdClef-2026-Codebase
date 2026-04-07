"""
Batch 100 — New Representation Approaches
==========================================
Current best: triple_30w02_40w03_40bw6w01 LOO=0.992166
Ceiling confirmed at ~0.9922 after 100 batches.

Trying fundamentally new representations NOT yet explored:

1. fisher_absica   — Fisher hard on |ICA| (absolute value embeddings)
2. fisher_sqica    — Fisher hard on ICA^2 (squared, captures 2nd-order structure)
3. fisher_l1norm   — Fisher with L1-normalized ICA instead of L2
4. fisher_raw_ica  — Fisher hard on ICA WITHOUT any normalization (raw coordinates)
5. fisher_concat2  — Concatenate [ICA_norm, |ICA|] → Fisher hard on 200-d space
6. fisher_sign     — Fisher on sign(ICA) + ICA_norm concatenated
7. fisher_diff_mean — Fisher on (ICA_win - global_mean_ICA) before normalizing
8. fisher_high_kurt — Fisher using only high-kurtosis ICA components (more non-Gaussian)
9. combined_repr   — Blend of multiple representation Fisher signals
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
from scipy.stats import kurtosis as scipy_kurtosis

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

ew_ica  = ep["emb_win_ica_norm"]    # L2-normalized ICA (739, 100)
ew_pca  = ep["emb_win_pca_norm"]
ew_std  = ep["emb_win_std_norm"]
ew_nmf  = ep["emb_win_nmf_norm"]
file_labels = ep["file_labels"]
cfg = ep["config"]

# Also load raw ICA (before L2-norm) - need to recompute from stored ICA transform
# ew_ica is L2-normalized; we can reconstruct the raw ICA by using the stored ICA model
ica_model = ep.get("ica", None)

print(f"[batch100] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch100] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

# ── Pre-compute base ──────────────────────────────────────────────────────────
print("Pre-computing base...", flush=True)
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
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

w_uh = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur  = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
final_ref = 0.96 * base_cur + 0.04 * kde08
f06 = fisher_kde_loo(ew_ica, bw=0.06)
fin_ref = (1 - 0.05) * final_ref + 0.05 * f06

# Pre-compute triple Fisher (current best)
fh30_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
fh40_b7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
fh40_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=40)
triple_ref = (1-0.02-0.03-0.01)*fin_ref + 0.02*fh30_b6 + 0.03*fh40_b7 + 0.01*fh40_b6
auc_triple = macro_auc(triple_ref)
print(f"  triple_ref: {auc_triple:.6f} (expected 0.992166)", flush=True)

results = {}
new_best_loo = best_loo
new_best_method = None

def reg(name, auc):
    delta = auc - best_loo
    mark = " *** NEW BEST ***" if auc > best_loo else (" (near-best)" if auc > best_loo - 0.0003 else "")
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch100"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# Prepare alternative ICA representations
# ─────────────────────────────────────────────────────────────────────────────
print("\nPreparing alternative ICA representations...", flush=True)

# Raw ICA (not L2-normalized) -- recover from ew_ica * stored_norms
# Since ew_ica = raw_ica / L2_norm, we need the norms
# Estimate: raw_ica proportional to ew_ica (but different norm)
# We'll use the stored raw transform via the ICA model if available
raw_ica_available = False
if ica_model is not None and hasattr(ica_model, 'transform'):
    # Check if we have raw embeddings
    raw_emb = ep.get("emb_win_raw", None)
    if raw_emb is None:
        # Try to get the scaler-transformed embeddings
        scaler = ep.get("scaler", None)
        if scaler is not None:
            raw_emb_data = DATA["emb"].astype(np.float32)
            raw_emb_scaled = scaler.transform(raw_emb_data)
            raw_ica_coords = ica_model.transform(raw_emb_scaled).astype(np.float32)
            raw_ica_available = True
            print(f"  Raw ICA computed: {raw_ica_coords.shape}", flush=True)

if not raw_ica_available:
    # Estimate raw ICA from ew_ica by computing norms from variance
    # Each ICA component has unit variance in training, so scale by global std
    raw_ica_coords = ew_ica  # fallback: same as L2-normalized
    print("  Raw ICA not available, using L2-normalized as fallback", flush=True)

# 1. |ICA| - absolute value
ew_abs_ica = np.abs(ew_ica)
ew_abs_ica_n = ew_abs_ica / (norm(ew_abs_ica, axis=1, keepdims=True) + EPS)
print(f"  |ICA| prepared: {ew_abs_ica_n.shape}", flush=True)

# 2. ICA^2
ew_sq_ica = ew_ica ** 2
ew_sq_ica_n = ew_sq_ica / (norm(ew_sq_ica, axis=1, keepdims=True) + EPS)
print(f"  ICA^2 prepared: {ew_sq_ica_n.shape}", flush=True)

# 3. L1-normalized ICA
ew_l1_ica = ew_ica / (np.abs(ew_ica).sum(axis=1, keepdims=True) + EPS)
ew_l1_ica_n = ew_l1_ica / (norm(ew_l1_ica, axis=1, keepdims=True) + EPS)  # then L2-norm for cosine
print(f"  L1-ICA prepared: {ew_l1_ica_n.shape}", flush=True)

# 4. Concatenated [ICA_norm, |ICA|] → PCA reduce to 100d
from sklearn.preprocessing import StandardScaler
ew_concat = np.concatenate([ew_ica, ew_abs_ica_n], axis=1)  # (739, 200)
from sklearn.decomposition import PCA as PCA2
pca_concat = PCA2(n_components=100, random_state=42)
ew_concat_pca = pca_concat.fit_transform(ew_concat).astype(np.float32)
ew_concat_pca_n = ew_concat_pca / (norm(ew_concat_pca, axis=1, keepdims=True) + EPS)
print(f"  Concat[ICA,|ICA|] PCA100 prepared: {ew_concat_pca_n.shape}", flush=True)

# 5. ICA sign features concatenated with ICA
ew_sign = np.sign(ew_ica).astype(np.float32)  # -1, 0, 1
ew_sign_concat = np.concatenate([ew_ica, ew_sign], axis=1)  # (739, 200)
pca_sign = PCA2(n_components=100, random_state=42)
ew_sign_pca = pca_sign.fit_transform(ew_sign_concat).astype(np.float32)
ew_sign_pca_n = ew_sign_pca / (norm(ew_sign_pca, axis=1, keepdims=True) + EPS)
print(f"  Sign+ICA PCA100 prepared: {ew_sign_pca_n.shape}", flush=True)

# 6. High-kurtosis ICA components (more non-Gaussian → more independent)
# Compute per-component kurtosis across all windows
kurtosis_per_dim = np.abs(scipy_kurtosis(ew_ica, axis=0))  # (100,)
high_kurt_idx = np.argsort(-kurtosis_per_dim)[:50]  # top-50 high kurtosis dims
ew_highkurt = ew_ica[:, high_kurt_idx]
ew_highkurt_n = ew_highkurt / (norm(ew_highkurt, axis=1, keepdims=True) + EPS)
print(f"  High-kurtosis ICA (top-50) prepared: {ew_highkurt_n.shape}", flush=True)
print(f"  Kurtosis range: {kurtosis_per_dim.min():.2f} to {kurtosis_per_dim.max():.2f}", flush=True)

# 7. Mean-subtracted ICA (deviation from global mean)
ica_global_mean = ew_ica.mean(0)
ew_meandev = ew_ica - ica_global_mean[None, :]
ew_meandev_n = ew_meandev / (norm(ew_meandev, axis=1, keepdims=True) + EPS)
print(f"  Mean-deviation ICA prepared: {ew_meandev_n.shape}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP1: Fisher hard on |ICA| (absolute value)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP1] Fisher hard on |ICA|...", flush=True)
t1 = time.time()
for k, bw in [(30, 0.06), (40, 0.07), (40, 0.06)]:
    s = fisher_hard_kde_loo(ew_abs_ica_n, bw=bw, top_k=k)
    for w_int in [2, 3, 4]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s
        reg(f"fh_abs_k{k}_bw{int(bw*100)}_w{w_int:02d}", macro_auc(blend))
print(f"  EXP1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP2: Fisher hard on ICA^2 (second-order features)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP2] Fisher hard on ICA^2...", flush=True)
t1 = time.time()
for k, bw in [(30, 0.06), (40, 0.07), (40, 0.06)]:
    s = fisher_hard_kde_loo(ew_sq_ica_n, bw=bw, top_k=k)
    for w_int in [2, 3, 4]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s
        reg(f"fh_sq_k{k}_bw{int(bw*100)}_w{w_int:02d}", macro_auc(blend))
print(f"  EXP2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP3: Fisher hard on L1-normalized ICA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP3] Fisher hard on L1-normalized ICA...", flush=True)
t1 = time.time()
for k, bw in [(30, 0.06), (40, 0.07), (40, 0.06)]:
    s = fisher_hard_kde_loo(ew_l1_ica_n, bw=bw, top_k=k)
    for w_int in [2, 3, 4]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s
        reg(f"fh_l1_k{k}_bw{int(bw*100)}_w{w_int:02d}", macro_auc(blend))
print(f"  EXP3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP4: Fisher hard on [ICA, |ICA|] PCA-100 space
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP4] Fisher hard on Concat[ICA,|ICA|] PCA100...", flush=True)
t1 = time.time()
for k, bw in [(30, 0.06), (40, 0.07), (50, 0.08)]:
    s = fisher_hard_kde_loo(ew_concat_pca_n, bw=bw, top_k=k)
    for w_int in [2, 3, 4]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s
        reg(f"fh_cat_k{k}_bw{int(bw*100)}_w{w_int:02d}", macro_auc(blend))
print(f"  EXP4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP5: Fisher hard on Sign+ICA PCA-100 space
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP5] Fisher hard on Sign+ICA PCA100...", flush=True)
t1 = time.time()
for k, bw in [(30, 0.06), (40, 0.07)]:
    s = fisher_hard_kde_loo(ew_sign_pca_n, bw=bw, top_k=k)
    for w_int in [2, 3, 4]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s
        reg(f"fh_sgn_k{k}_bw{int(bw*100)}_w{w_int:02d}", macro_auc(blend))
print(f"  EXP5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP6: Fisher hard restricted to high-kurtosis ICA components (top-50)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP6] Fisher hard on high-kurtosis ICA (top-50 dims)...", flush=True)
t1 = time.time()
for k in [15, 20, 25, 30]:
    for bw in [0.06, 0.07]:
        s = fisher_hard_kde_loo(ew_highkurt_n, bw=bw, top_k=k)
        for w_int in [2, 3, 4]:
            w = w_int * 0.01
            blend = (1 - w) * triple_ref + w * s
            reg(f"fh_hkurt_k{k}_bw{int(bw*100)}_w{w_int:02d}", macro_auc(blend))
print(f"  EXP6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP7: Fisher hard on mean-deviation ICA (deviation from global mean)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP7] Fisher hard on mean-deviation ICA...", flush=True)
t1 = time.time()
for k, bw in [(30, 0.06), (40, 0.07), (40, 0.06)]:
    s = fisher_hard_kde_loo(ew_meandev_n, bw=bw, top_k=k)
    for w_int in [2, 3, 4]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s
        reg(f"fh_mdev_k{k}_bw{int(bw*100)}_w{w_int:02d}", macro_auc(blend))
print(f"  EXP7 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP8: Combine new repr with original triple as 4th/5th component
# Best new repr signal + triple
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP8] Combined new repr + triple...", flush=True)
t1 = time.time()
# Soft Fisher on abs, sq, l1 representations
for tag, ew_alt in [("abs", ew_abs_ica_n), ("sq", ew_sq_ica_n), ("l1", ew_l1_ica_n), ("mdev", ew_meandev_n)]:
    s_soft = fisher_kde_loo(ew_alt, bw=0.06)
    for w_int in [1, 2, 3]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s_soft
        reg(f"triple_plus_fsoft_{tag}_w{w_int:02d}", macro_auc(blend))
print(f"  EXP8 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP9: WL scoring in absolute-ICA and squared-ICA spaces
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP9] WL in abs/sq ICA spaces...", flush=True)
t1 = time.time()
for tag, ew_alt in [("abs", ew_abs_ica_n), ("sq", ew_sq_ica_n)]:
    s_wl = wl_loo(ew_alt, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
    for w_int in [2, 3, 4]:
        w = w_int * 0.01
        blend = (1 - w) * triple_ref + w * s_wl
        reg(f"wl_{tag}_w{w_int:02d}", macro_auc(blend))
print(f"  EXP9 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch100] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

if results:
    all_sorted = sorted(results.items(), key=lambda x: -x[1]["loo_auc"])
    top_tag, top_v = all_sorted[0]
    top_auc = top_v["loo_auc"]
    if top_auc > best_loo:
        print(f"  *** NEW BEST: {top_tag} LOO={top_auc:.6f} (+{top_auc-best_loo:.6f}) ***", flush=True)
        new_best_loo = top_auc
        new_best_method = top_tag
    else:
        print(f"  Best this batch: {top_auc:.6f} ({top_tag}) — no improvement", flush=True)
    print("\n  Top-10:", flush=True)
    for tag, v in all_sorted[:10]:
        d = v["loo_auc"] - best_loo
        print(f"    {tag}: {v['loo_auc']:.6f} ({d:+.6f})", flush=True)

res2 = json.load(open(RESULTS_PATH))
if isinstance(res2.get("experiments"), list):
    res2["experiments"].extend(list(results.values()))
elif isinstance(res2.get("experiments"), dict):
    res2["experiments"].update(results)
else:
    res2["experiments"] = list(results.values())

if new_best_loo > best_loo:
    res2["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch100"}
    ep2 = copy.deepcopy(ep)
    ep2["loo_auc"] = new_best_loo
    ep2["method"] = new_best_method
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep2, f)
    print(f"\n  SAVED new best: {new_best_method} LOO={new_best_loo:.6f}", flush=True)

json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"\nSaved {len(results)} experiments to JSON.", flush=True)
