"""
Batch 93 — New Kernel & Signal Methods
=======================================
Current best: softmax_T6_proto_kde LOO=0.991782

Novel approaches not yet tried:
1. kernelized_wl        — RBF kernel similarity in WL (instead of cosine)
2. random_subspace_kde  — Average KDE over 10 random 50-dim ICA subspaces
3. logit_weight_kde     — Proto-KDE where train windows weighted by Perch logit confidence
4. temporal_pos_kde     — Proto-KDE where window position affects weight (first/last window)
5. student_t_kde        — Student-t kernel (heavier tails) instead of Gaussian
6. matern_kde           — Matérn kernel (ν=1.5) instead of Gaussian
7. class_covar_kde      — Per-class covariance-normalized KDE
8. tfidf_embed          — TF-IDF weighting of ICA dimensions before KDE
9. fisher_score_kde     — Fisher discrimination score to weight KDE dimensions
10. two_level_kde       — Hierarchical: file-level + window-level KDE
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
from numpy.linalg import norm

ROOT = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

# ── Load data ──────────────────────────────────────────────────────────────
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

ew_ica  = ep["emb_win_ica_norm"]   # (739, 100)
ew_pca  = ep["emb_win_pca_norm"]   # (739, 80)
ew_std  = ep["emb_win_std_norm"]   # (739, 80)
ew_nmf  = ep["emb_win_nmf_norm"]   # (739, 100)
ls_win  = ep["logit_sig_win"]      # (739, 234) sigmoid(logit)
file_labels = ep["file_labels"]    # (66, 234)
flm     = ep["file_logit_max"]     # (66, 234)
cfg     = ep["config"]

print(f"[batch93] Loaded ICA{ew_ica.shape} NMF{ew_nmf.shape}", flush=True)

# ── Load JSON ────────────────────────────────────────────────────────────────
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch93] Current best: {best['method']} LOO={best_loo:.6f}")

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

def proto_kde_loo(ew, bw=0.08):
    """Standard proto-KDE (cosine kernel). Reference implementation."""
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
            proto_w = np.clip(pos_wins @ centroid, 0, None)
            proto_w = proto_w / (proto_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ── Pre-compute base ──────────────────────────────────────────────────────────
print("Computing base...", flush=True)
t0 = time.time()
s_ica = wl_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
s_std = wl_loo(ew_std, cfg["std_pca80"]["k_neg"], cfg["std_pca80"]["w_max_pos"], cfg["std_pca80"]["w_max_agg"])
s_pca = wl_loo(ew_pca, cfg["pca80"]["k_neg"], cfg["pca80"]["w_max_pos"], cfg["pca80"]["w_max_agg"])
s_nmf = wl_loo(ew_nmf, cfg["nmf"]["k_neg"], cfg["nmf"]["w_max_pos"], cfg["nmf"]["w_max_agg"])
uh_b   = cfg["w_ica100"] * s_ica + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
uh_nmf = cfg["nmf"]["uh_scale"] * uh_b + cfg["nmf"]["w_nmf"] * s_nmf
pT8    = make_logit_pred(cfg["logit_temperature"])
pmt    = (pT8 + make_logit_pred(10.0)) / 2
sm6    = make_softmax_pred(cfg["softmax_temp"])
ss2    = compute_subspace(ew_pca, n_comp=2, wma_ss=0.92)
kde_ref = proto_kde_loo(ew_ica, bw=0.08)
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

w_uh = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur  = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
final_ref = 0.96 * base_cur + 0.04 * kde_ref
auc_ref   = macro_auc(final_ref)
print(f"Reproduced reference: {auc_ref:.6f} (expected {best_loo:.6f})", flush=True)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. kernelized_wl — RBF kernel similarity in WL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] kernelized_wl...", flush=True)

def kernelized_wl_loo(ew, k_neg, wmp, wma, gamma=10.0):
    """WL using RBF kernel similarity: K(x,y) = exp(-gamma * ||x-y||^2)"""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]
            # RBF kernel: K(te_i, pw_j) = exp(-gamma * ||te_i - pw_j||^2)
            # ||te - pw||^2 = ||te||^2 + ||pw||^2 - 2 te@pw.T
            # Since normalized: ||te||=1, ||pw||=1 → dist^2 = 2 - 2 cos_sim
            ps = te @ pw.T  # cosine sim
            dists2 = np.clip(2 - 2 * ps, 0, None)
            K_pos = np.exp(-gamma * dists2)  # (n_te, n_pos)
            sp = wmp * K_pos.max(1) + (1 - wmp) * K_pos.mean(1)
            if nm.any() and k_neg > 0:
                nw = tr[nm]; ns = te @ nw.T; k2 = min(k_neg, ns.shape[1])
                if k2 > 0:
                    tn = nw[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
                    tn /= norm(tn, axis=1, keepdims=True) + EPS
                    dists2_neg = np.clip(2 - 2 * (te * tn).sum(1, keepdims=False), 0, None)
                    K_neg = np.exp(-gamma * dists2_neg)
                    ws[:, si] = np.clip(sp - K_neg + 1, 0, 2) / 2
                else:
                    ws[:, si] = np.clip((sp + 1) / 2, 0, 1)
            else:
                ws[:, si] = np.clip((sp + 1) / 2, 0, 1)
        out[fi] = wma * ws.max(0) + (1 - wma) * ws.mean(0)
    return out

for gamma in [5.0, 10.0, 20.0, 50.0]:
    sk_rbf = kernelized_wl_loo(ew_ica, cfg["ica100"]["k_neg"],
                                cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"],
                                gamma=gamma)
    for w in [0.03, 0.05, 0.07]:
        cand = (1-w)*final_ref + w*sk_rbf
        a = macro_auc(cand)
        results[f"kernelized_wl_g{gamma}_w{int(w*100)}"] = a
        if a >= best_loo - 0.0005:
            print(f"  kernelized_wl gamma={gamma} w={w}: LOO={a:.6f} delta={a-best_loo:+.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. random_subspace_kde — Average KDE over N random subspaces
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] random_subspace_kde...", flush=True)

rng = np.random.RandomState(42)
n_subspaces = 8
subspace_dim = 50

def random_subspace_kde_loo(n_sub=8, sub_dim=50, bw=0.08, seed=42):
    rng2 = np.random.RandomState(seed)
    agg = np.zeros((n_files, n_species), np.float32)
    for _ in range(n_sub):
        idx = rng2.choice(ew_ica.shape[1], sub_dim, replace=False)
        ew_sub = ew_ica[:, idx]
        ew_sub_n = ew_sub / (norm(ew_sub, axis=1, keepdims=True) + EPS)
        agg += proto_kde_loo(ew_sub_n, bw=bw)
    return agg / n_sub

for n_sub, sub_dim in [(8, 50), (16, 50), (8, 70)]:
    s_rsub = random_subspace_kde_loo(n_sub=n_sub, sub_dim=sub_dim, bw=0.08)
    for w in [0.02, 0.04, 0.06]:
        cand = (1-w)*final_ref + w*s_rsub
        a = macro_auc(cand)
        key = f"random_subspace_n{n_sub}_d{sub_dim}_w{int(w*100)}"
        results[key] = a
        if a >= best_loo - 0.0005:
            print(f"  rsubspace n={n_sub} d={sub_dim} w={w}: LOO={a:.6f} delta={a-best_loo:+.6f}", flush=True)
    # Also replace kde_ref
    cand2 = 0.96 * base_cur + 0.04 * s_rsub
    a2 = macro_auc(cand2)
    results[f"random_subspace_n{n_sub}_d{sub_dim}_replace_kde"] = a2
    if a2 >= best_loo - 0.0005:
        print(f"  rsubspace n={n_sub} d={sub_dim} REPLACE_KDE: LOO={a2:.6f} delta={a2-best_loo:+.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. logit_weight_kde — Proto-KDE with logit-weighted prototypes
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] logit_weight_kde...", flush=True)

def logit_weight_kde_loo(bw=0.08, T=8.0):
    """Weight train windows by their Perch logit confidence for that species."""
    logit_conf = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T, -88, 88)))  # (739, 234)
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        lc = logit_conf[win_file_id != fi]  # train window logit confs
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]
            pos_confs = lc[pos_idx, si]  # Perch confidence for this species
            # Logit-weighted centroid
            w_lc = pos_confs / (pos_confs.sum() + EPS)
            centroid = (pos_wins * w_lc[:, None]).sum(0)
            centroid /= (norm(centroid) + EPS)
            # Proto weights: confidence-weighted
            proto_w = np.clip(pos_wins @ centroid, 0, None)
            proto_w = proto_w * w_lc  # additional logit weighting
            proto_w = proto_w / (proto_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for T in [6.0, 8.0, 10.0]:
    for bw in [0.06, 0.08, 0.10]:
        s_lwk = logit_weight_kde_loo(bw=bw, T=T)
        for w in [0.03, 0.04, 0.05]:
            cand = (1-w)*final_ref + w*s_lwk
            a = macro_auc(cand)
            key = f"logit_weight_kde_T{int(T)}_bw{int(bw*100)}_w{int(w*100)}"
            results[key] = a
            if a >= best_loo - 0.0005:
                print(f"  logit_kde T={T} bw={bw} w={w}: LOO={a:.6f} delta={a-best_loo:+.6f}", flush=True)
        # Replace kde_ref
        cand2 = 0.96 * base_cur + 0.04 * s_lwk
        a2 = macro_auc(cand2)
        results[f"logit_weight_kde_T{int(T)}_bw{int(bw*100)}_replace"] = a2
        if a2 >= best_loo - 0.0005:
            print(f"  logit_kde T={T} bw={bw} REPLACE: LOO={a2:.6f} delta={a2-best_loo:+.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. student_t_kde — Heavier-tailed kernel
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] student_t_kde...", flush=True)

def student_t_kde_loo(bw=0.08, nu=2.0):
    """Student-t kernel: K(x,y) = (1 + dist^2/(nu * bw^2))^(-(nu+1)/2)"""
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
            # cosine sim → dist^2 = 2 - 2*sim (for unit vectors)
            cos_sims = sims[:, pos_idx]  # (n_te, n_pos)
            dist2 = np.clip(2 - 2 * cos_sims, 0, None)
            kern = (1.0 + dist2 / (nu * bw**2 + EPS)) ** (-(nu + 1) / 2)
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for nu in [1.0, 2.0, 5.0]:
    for bw in [0.06, 0.08, 0.10]:
        s_stk = student_t_kde_loo(bw=bw, nu=nu)
        cand = 0.96 * base_cur + 0.04 * s_stk
        a = macro_auc(cand)
        key = f"student_t_kde_nu{int(nu)}_bw{int(bw*100)}"
        results[key] = a
        if a >= best_loo - 0.0005:
            print(f"  student_t nu={nu} bw={bw}: LOO={a:.6f} delta={a-best_loo:+.6f}", flush=True)
        # Blend with final_ref
        for w in [0.02, 0.04]:
            cand2 = (1-w)*final_ref + w*s_stk
            a2 = macro_auc(cand2)
            if a2 >= best_loo - 0.0002:
                print(f"  student_t nu={nu} bw={bw} w={w}: LOO={a2:.6f} delta={a2-best_loo:+.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. matern_kde — Matérn kernel (ν=1.5)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] matern_kde...", flush=True)

def matern15_kde_loo(bw=0.08):
    """Matérn ν=1.5 kernel: K(r) = (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)"""
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
            cos_sims = sims[:, pos_idx]
            dist = np.sqrt(np.clip(2 - 2 * cos_sims, 0, None))
            r = dist / (bw + EPS)
            kern = (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for bw in [0.05, 0.08, 0.10, 0.12]:
    s_mat = matern15_kde_loo(bw=bw)
    cand = 0.96 * base_cur + 0.04 * s_mat
    a = macro_auc(cand)
    key = f"matern_kde_bw{int(bw*100)}"
    results[key] = a
    if a >= best_loo - 0.0005:
        print(f"  matern bw={bw}: LOO={a:.6f} delta={a-best_loo:+.6f}", flush=True)
    for w in [0.02, 0.04]:
        cand2 = (1-w)*final_ref + w*s_mat
        a2 = macro_auc(cand2)
        if a2 >= best_loo - 0.0002:
            print(f"  matern bw={bw} w={w}: LOO={a2:.6f} delta={a2-best_loo:+.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. tfidf_embed — TF-IDF weighting of ICA dimensions
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6] tfidf_embed...", flush=True)

def tfidf_weighted_kde_loo(bw=0.08):
    """
    Compute TF-IDF weights for ICA dimensions:
    - TF: mean absolute activation of dimension per species prototype
    - IDF: inverse document frequency across species
    Then apply dimension-weighted proto-KDE.
    """
    # Build species prototypes from all 739 windows
    species_protos = np.zeros((n_species, ew_ica.shape[1]), np.float32)
    for si in range(n_species):
        pos_m = labels_win[:, si] > 0.5
        if pos_m.any():
            species_protos[si] = np.abs(ew_ica[pos_m]).mean(0)

    # IDF: how many species activate each dimension
    active = (species_protos > 0).sum(0).astype(np.float32)
    idf = np.log((n_species + 1) / (active + 1)) + 1  # (100,)

    # TF per species: normalized activation
    tf = species_protos / (norm(species_protos, axis=1, keepdims=True) + EPS)

    # TF-IDF weights per species
    tfidf_weights = tf * idf[None, :]  # (n_species, 100)

    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]
            # Apply TF-IDF weights to dimensions
            w_dim = tfidf_weights[si]  # (100,)
            w_norm = w_dim / (norm(w_dim) + EPS)
            # Weighted embedding space
            te_w = te * w_norm[None, :]
            te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_norm[None, :]
            tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_w @ centroid, 0, None)
            proto_w /= (proto_w.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for bw in [0.06, 0.08, 0.10]:
    s_tfidf = tfidf_weighted_kde_loo(bw=bw)
    cand = 0.96 * base_cur + 0.04 * s_tfidf
    a = macro_auc(cand)
    key = f"tfidf_embed_bw{int(bw*100)}"
    results[key] = a
    if a >= best_loo - 0.0005:
        print(f"  tfidf_embed bw={bw} (replace kde): LOO={a:.6f} delta={a-best_loo:+.6f}", flush=True)
    for w in [0.02, 0.04]:
        cand2 = (1-w)*final_ref + w*s_tfidf
        a2 = macro_auc(cand2)
        if a2 >= best_loo - 0.0002:
            print(f"  tfidf_embed bw={bw} w={w}: LOO={a2:.6f} delta={a2-best_loo:+.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. fisher_score_kde — Fisher discrimination per dimension
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7] fisher_score_kde...", flush=True)

def fisher_score_kde_loo(bw=0.08):
    """
    Per-species Fisher score per dimension:
    F_d = (mu_pos_d - mu_neg_d)^2 / (var_pos_d + var_neg_d)
    Use Fisher score as dimension weights for proto-KDE.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            pos_idx = np.where(pm)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]

            # Fisher score per dimension
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher = (mu_p - mu_n) ** 2 / (var_p + var_n)
            fisher = np.sqrt(np.clip(fisher, 0, None))
            w_dim = fisher / (norm(fisher) + EPS)

            # Apply Fisher weights
            te_w = te * w_dim[None, :]
            te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_dim[None, :]
            tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_w @ centroid, 0, None)
            proto_w /= (proto_w.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for bw in [0.06, 0.08, 0.10]:
    s_fisher = fisher_score_kde_loo(bw=bw)
    cand = 0.96 * base_cur + 0.04 * s_fisher
    a = macro_auc(cand)
    key = f"fisher_score_kde_bw{int(bw*100)}"
    results[key] = a
    if a >= best_loo - 0.0005:
        print(f"  fisher_kde bw={bw} (replace): LOO={a:.6f} delta={a-best_loo:+.6f}", flush=True)
    for w in [0.02, 0.04]:
        cand2 = (1-w)*final_ref + w*s_fisher
        a2 = macro_auc(cand2)
        if a2 >= best_loo - 0.0002:
            print(f"  fisher_kde bw={bw} w={w}: LOO={a2:.6f} delta={a2-best_loo:+.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. two_level_kde — Hierarchical file + window KDE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[8] two_level_kde...", flush=True)

def two_level_kde_loo(bw_win=0.08, bw_file=0.15, alpha=0.5):
    """
    Two-level KDE:
    1. Window-level: standard proto-KDE using train windows
    2. File-level: KDE using mean embeddings of train files
    Blend: alpha * window-level + (1-alpha) * file-level
    """
    # File mean embeddings
    file_emb = np.array([ew_ica[win_file_id==fi].mean(0) for fi in range(n_files)])
    file_emb_n = file_emb / (norm(file_emb, axis=1, keepdims=True) + EPS)

    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = ew_ica[win_file_id == fi]; tr_wins = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        te_file = file_emb_n[fi:fi+1]  # (1, 100)
        tr_fl = file_labels  # (66, 234)

        # Window-level KDE
        sims_win = te_wins @ tr_wins.T
        ws_win = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws_win[:, si] = 0.5; continue
            pos_wins = tr_wins[pos_idx]
            centroid = pos_wins.mean(0); centroid /= (norm(centroid) + EPS)
            proto_w = np.clip(pos_wins @ centroid, 0, None); proto_w /= (proto_w.sum() + EPS)
            kern = np.exp((sims_win[:, pos_idx] - 1.0) / (bw_win**2 + EPS))
            ws_win[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws_win[:, si].max()
            if mx > EPS: ws_win[:, si] /= mx
        score_win = ws_win.max(0)

        # File-level KDE
        tr_file_idx = np.where(np.arange(n_files) != fi)[0]
        tr_fl_emb = file_emb_n[tr_file_idx]
        tr_fl_lab = file_labels[tr_file_idx]
        sims_fl = te_file @ tr_fl_emb.T  # (1, n_train_files)
        score_fl = np.zeros(n_species, np.float32)
        for si in range(n_species):
            pos_f = np.where(tr_fl_lab[:, si] > 0.5)[0]
            if len(pos_f) == 0: continue
            kern_f = np.exp((sims_fl[0, pos_f] - 1.0) / (bw_file**2 + EPS))
            score_fl[si] = kern_f.mean()

        out[fi] = alpha * score_win + (1 - alpha) * score_fl
    return out

for bw_win, bw_file in [(0.08, 0.15), (0.08, 0.20), (0.06, 0.12)]:
    for alpha in [0.4, 0.5, 0.6, 0.7]:
        s_2l = two_level_kde_loo(bw_win=bw_win, bw_file=bw_file, alpha=alpha)
        cand = 0.96 * base_cur + 0.04 * s_2l
        a = macro_auc(cand)
        key = f"two_level_kde_bw{int(bw_win*100)}_{int(bw_file*100)}_a{int(alpha*10)}"
        results[key] = a
        if a >= best_loo - 0.0005:
            print(f"  two_level bw_win={bw_win} bw_file={bw_file} a={alpha}: LOO={a:.6f} delta={a-best_loo:+.6f}", flush=True)
        for w in [0.02, 0.04]:
            cand2 = (1-w)*final_ref + w*s_2l
            a2 = macro_auc(cand2)
            if a2 >= best_loo - 0.0002:
                print(f"  two_level bw_win={bw_win} bw_file={bw_file} a={alpha} w={w}: LOO={a2:.6f} delta={a2-best_loo:+.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON Update
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"[batch93] Results summary:", flush=True)
best_result = max(results.items(), key=lambda x: x[1]) if results else (None, 0)
print(f"  Best from this batch: {best_result[0]} = {best_result[1]:.6f}", flush=True)
print(f"  Reference best:       {best_loo:.6f}", flush=True)
print(f"  Delta: {best_result[1]-best_loo:+.6f}", flush=True)

# All results at or above best_loo - 0.0003
near_best = [(k, v) for k, v in sorted(results.items(), key=lambda x: -x[1]) if v >= best_loo - 0.0003]
print(f"  Near-best results ({len(near_best)}):", flush=True)
for k, v in near_best[:20]:
    print(f"    {k}: {v:.6f} ({v-best_loo:+.6f})", flush=True)

# Update JSON with new experiments
res2 = json.load(open(RESULTS_PATH))
new_exps = [{"method": k, "loo_auc": float(v), "batch": "batch93"} for k, v in results.items()]
res2["experiments"].extend(new_exps)

if best_result[1] > best_loo:
    res2["best"] = {
        "method": best_result[0],
        "loo_auc": float(best_result[1]),
        "full_auc": None,
        "batch": "batch93"
    }
    print(f"\n*** NEW BEST: {best_result[0]} = {best_result[1]:.6f} ***", flush=True)

with open(RESULTS_PATH, "w") as f:
    json.dump(res2, f, indent=2)
print(f"Saved {len(new_exps)} experiments to JSON.", flush=True)
