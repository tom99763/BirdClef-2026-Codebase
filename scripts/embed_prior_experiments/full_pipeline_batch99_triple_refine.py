"""
Batch 99 — Triple Fisher Combination Refinement
=================================================
Current best: triple_30w02_40w03_40bw6w01 LOO=0.992166
Formula: 0.94*fin_ref + 0.02*fh30_bw6 + 0.03*fh40_bw7 + 0.01*fh40_bw6

Key insight: Multiple Fisher Hard configs are complementary.
Fine-tune and explore variations.

1. triple_grid   — Fine sweep of (w30_bw6, w40_bw7, w40_bw6) weights
2. triple_k_var  — Vary k values: (k=25,bw6), (k=35,bw7), (k=45,bw6)
3. quad_combo    — Add 4th component (fh37_bw7)
4. bw6_7_variants — Try different bw for 3-combo components
5. fin_ref_only  — Use different fin_ref (with different soft Fisher bw)
6. k_fine_grid   — 2D k1xk2 grid for the two main hard components
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

ew_ica  = ep["emb_win_ica_norm"]
ew_pca  = ep["emb_win_pca_norm"]
ew_std  = ep["emb_win_std_norm"]
ew_nmf  = ep["emb_win_nmf_norm"]
file_labels = ep["file_labels"]
cfg = ep["config"]

print(f"[batch99] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch99] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

# Pre-compute key Fisher hard signals
print("Pre-computing Fisher hard signals...", flush=True)
t1 = time.time()
fh30_bw6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
fh40_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
fh40_bw6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=40)
# Verify current best
s_best = (1 - 0.02 - 0.03 - 0.01) * fin_ref + 0.02 * fh30_bw6 + 0.03 * fh40_bw7 + 0.01 * fh40_bw6
auc_best = macro_auc(s_best)
print(f"  current best: {auc_best:.6f} (expected 0.992166)", flush=True)
print(f"  done ({time.time()-t1:.0f}s)", flush=True)

# Additional signals
fh25_bw6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=25)
fh35_bw6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=35)
fh35_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=35)
fh45_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=45)
fh30_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=30)
fh50_bw6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=50)

results = {}
new_best_loo = best_loo
new_best_method = None

def reg(name, auc):
    delta = auc - best_loo
    mark = ""
    if auc > best_loo:
        mark = f" *** NEW BEST ***"
    elif auc > best_loo - 0.0003:
        mark = " (near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch99"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# EXP1: Fine grid of triple combination weights
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP1] Fine triple weight grid...", flush=True)
t1 = time.time()
# Reference: (w30_bw6=0.02, w40_bw7=0.03, w40_bw6=0.01) = best
for w1 in [0.01, 0.02, 0.03]:
    for w2 in [0.02, 0.03, 0.04]:
        for w3 in [0.005, 0.01, 0.015, 0.02]:
            if abs(w1 - 0.02) < 1e-6 and abs(w2 - 0.03) < 1e-6 and abs(w3 - 0.01) < 1e-6:
                continue  # skip known best
            total_w = w1 + w2 + w3
            if total_w > 0.12:
                continue
            name = f"t3_w1{int(w1*200):02d}_w2{int(w2*100):02d}_w3{int(w3*200):02d}"
            s = (1 - total_w) * fin_ref + w1 * fh30_bw6 + w2 * fh40_bw7 + w3 * fh40_bw6
            reg(name, macro_auc(s))
print(f"  EXP1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP2: Vary k values in the triple combination
# Best: (k30_bw6, k40_bw7, k40_bw6) with weights (0.02, 0.03, 0.01)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP2] Vary k in triple combination...", flush=True)
t1 = time.time()
combos = [
    ("t3_25b6_40b7_40b6", fh25_bw6, fh40_bw7, fh40_bw6),
    ("t3_35b6_40b7_40b6", fh35_bw6, fh40_bw7, fh40_bw6),
    ("t3_30b6_35b7_40b6", fh30_bw6, fh35_bw7, fh40_bw6),
    ("t3_30b6_45b7_40b6", fh30_bw6, fh45_bw7, fh40_bw6),
    ("t3_30b6_40b7_35b6", fh30_bw6, fh40_bw7, fh35_bw6),
    ("t3_30b6_40b7_50b6", fh30_bw6, fh40_bw7, fh50_bw6),
    ("t3_30b7_40b7_40b6", fh30_bw7, fh40_bw7, fh40_bw6),
]
for tag, s1, s2, s3 in combos:
    s = (1 - 0.02 - 0.03 - 0.01) * fin_ref + 0.02 * s1 + 0.03 * s2 + 0.01 * s3
    reg(tag + "_w2_3_1", macro_auc(s))
    s = (1 - 0.02 - 0.03 - 0.015) * fin_ref + 0.02 * s1 + 0.03 * s2 + 0.015 * s3
    reg(tag + "_w2_3_15", macro_auc(s))
print(f"  EXP2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP3: Quadruple combination (add 4th component)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP3] Quadruple Fisher combination...", flush=True)
t1 = time.time()
# Best triple: fin_ref + 0.02*fh30_bw6 + 0.03*fh40_bw7 + 0.01*fh40_bw6
for tag4, s4 in [("fh35b7", fh35_bw7), ("fh45b7", fh45_bw7), ("fh30b7", fh30_bw7)]:
    for w4 in [0.005, 0.01, 0.015]:
        s = (1 - 0.02 - 0.03 - 0.01 - w4) * fin_ref + 0.02 * fh30_bw6 + 0.03 * fh40_bw7 + 0.01 * fh40_bw6 + w4 * s4
        reg(f"quad_best_{tag4}_w{int(w4*200):02d}", macro_auc(s))
print(f"  EXP3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP4: Different soft Fisher bw in fin_ref chain
# Maybe fin_ref with different bw gives better base for the triple
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP4] Different soft Fisher bw in fin_ref...", flush=True)
t1 = time.time()
for soft_bw in [0.05, 0.07, 0.08]:
    f_alt = fisher_kde_loo(ew_ica, bw=soft_bw)
    fin_ref_alt = (1 - 0.05) * final_ref + 0.05 * f_alt
    s = (1 - 0.02 - 0.03 - 0.01) * fin_ref_alt + 0.02 * fh30_bw6 + 0.03 * fh40_bw7 + 0.01 * fh40_bw6
    reg(f"triple_finalt_fbw{int(soft_bw*100)}", macro_auc(s))
    # Also try with different triple weights on alt fin_ref
    s2 = (1 - 0.03 - 0.04 - 0.01) * fin_ref_alt + 0.03 * fh30_bw6 + 0.04 * fh40_bw7 + 0.01 * fh40_bw6
    reg(f"triple_finalt_fbw{int(soft_bw*100)}_w34", macro_auc(s2))
print(f"  EXP4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP5: 2D k1 x k2 grid for the two main hard Fisher components
# k1 (bw=0.06) x k2 (bw=0.07) with best weights (w1=0.02, w2=0.03, w3=0.01 for k1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP5] 2D k1 x k2 grid...", flush=True)
t1 = time.time()
# Pre-compute remaining signals
fh_cache = {
    (25, 6): fh25_bw6,
    (30, 6): fh30_bw6,
    (35, 6): fh35_bw6,
    (40, 6): fh40_bw6,
    (50, 6): fh50_bw6,
    (30, 7): fh30_bw7,
    (35, 7): fh35_bw7,
    (40, 7): fh40_bw7,
    (45, 7): fh45_bw7,
}
for k1 in [25, 30, 35]:
    for k2 in [35, 40, 45]:
        s1 = fh_cache.get((k1, 6), fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=k1))
        s2 = fh_cache.get((k2, 7), fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=k2))
        # Use third component as fh40_bw6
        s = 0.94 * fin_ref + 0.02 * s1 + 0.03 * s2 + 0.01 * fh40_bw6
        reg(f"t3_k{k1}b6_k{k2}b7_k40b6", macro_auc(s))
print(f"  EXP5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP6: Try adding a "wide" Fisher hard (k=60,70,80 bw=0.06) to triple
# Wide Fisher captures broader patterns
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP6] Wide Fisher component...", flush=True)
t1 = time.time()
for k_wide in [60, 70, 80]:
    fh_wide = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=k_wide)
    for w_wide in [0.01, 0.015]:
        s = (1 - 0.02 - 0.03 - 0.01 - w_wide) * fin_ref + 0.02 * fh30_bw6 + 0.03 * fh40_bw7 + 0.01 * fh40_bw6 + w_wide * fh_wide
        reg(f"triple_wide_k{k_wide}_w{int(w_wide*200):02d}", macro_auc(s))
print(f"  EXP6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP7: Alternative second component: proto_kde (non-Fisher) + triple
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP7] Proto-KDE as 4th component...", flush=True)
t1 = time.time()
pkde = proto_kde_loo(ew_ica, bw=0.08)
for w_pkde in [0.01, 0.015, 0.02]:
    s = (1 - 0.02 - 0.03 - 0.01 - w_pkde) * fin_ref + 0.02 * fh30_bw6 + 0.03 * fh40_bw7 + 0.01 * fh40_bw6 + w_pkde * pkde
    reg(f"triple_pkde_w{int(w_pkde*200):02d}", macro_auc(s))
print(f"  EXP7 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch99] SUMMARY", flush=True)
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

# ── Save ─────────────────────────────────────────────────────────────────────
res2 = json.load(open(RESULTS_PATH))
if isinstance(res2.get("experiments"), list):
    res2["experiments"].extend(list(results.values()))
elif isinstance(res2.get("experiments"), dict):
    res2["experiments"].update(results)
else:
    res2["experiments"] = list(results.values())

if new_best_loo > best_loo:
    res2["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch99"}
    ep2 = copy.deepcopy(ep)
    ep2["loo_auc"] = new_best_loo
    ep2["method"] = new_best_method
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep2, f)
    print(f"\n  SAVED new best: {new_best_method} LOO={new_best_loo:.6f}", flush=True)

json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"\nSaved {len(results)} experiments to JSON.", flush=True)
