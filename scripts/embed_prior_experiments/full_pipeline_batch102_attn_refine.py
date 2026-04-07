"""
Batch 102 — Attention-weighted KDE Refinement
===============================================================================
Current best: attn_addon_kf40_bw7_T6_w01 LOO=0.992180
              formula: (1-0.01)*triple_ref + 0.01*attn_kf40_bw7_T6

New breakthrough: Attention-weighted Fisher Hard KDE as additive signal.
This batch fine-tunes all hyperparameters around the winner.

Experiments:
1. Fine w sweep (0.005 – 0.03)
2. attn_T sweep: 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, "inf" (uniform)
3. bw sweep: 0.04 – 0.10
4. top_k sweep: 15, 20, 25, 30, 35, 40, 45, 50
5. 2D grid: k × T (winners from above)
6. Double attention signal stacking
7. Replace triple hard Fisher w/ attention
8. Attention on fin_ref directly
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

print(f"[batch102] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch102] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

def macro_auc(s):
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            aucs.append(roc_auc_score(y, s[:, si]))
    return float(np.mean(aucs)) if aucs else 0.0

# ── EXACT copy of helper functions from batch101 ─────────────────────────────
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

def attn_knn_fisher_loo(ew, top_k_fisher=40, bw=0.07, attn_T=6.0):
    """
    Fisher Hard KDE but weight training windows by logit confidence (attention).
    attn_T controls sigmoid sharpness; attn_T>=100 → uniform (= original hard KDE).
    """
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
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            pos_logit_si = tl_logit[pm, si]
            if attn_T >= 100.0:
                attn = np.ones(len(pos_logit_si), np.float32) / len(pos_logit_si)
            else:
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

# ── Pre-compute base chain ─────────────────────────────────────────────────
print("Pre-computing base chain...", flush=True)
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
w_uh = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur  = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
final_ref = 0.96 * base_cur + 0.04 * kde08
f06 = fisher_kde_loo(ew_ica, bw=0.06)
fin_ref = (1 - 0.05) * final_ref + 0.05 * f06
fh30_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
fh40_b7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
fh40_b6 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=40)
triple_ref = (1-0.02-0.03-0.01)*fin_ref + 0.02*fh30_b6 + 0.03*fh40_b7 + 0.01*fh40_b6
auc_t = macro_auc(triple_ref)
print(f"  triple_ref: {auc_t:.6f} (expected 0.992166) [{time.time()-t0:.0f}s]", flush=True)

# Pre-compute best attention signal (k=40, bw=0.07, T=6)
print("Pre-computing attention signal kf40_bw7_T6...", flush=True)
s_attn_best = attn_knn_fisher_loo(ew_ica, top_k_fisher=40, bw=0.07, attn_T=6.0)
auc_attn_best_raw = macro_auc(s_attn_best)
print(f"  attn_kf40_bw7_T6 standalone: {auc_attn_best_raw:.6f}", flush=True)

results = {}
new_best_loo = best_loo
new_best_method = None

def reg(name, auc):
    global new_best_loo, new_best_method
    delta = auc - best_loo
    if auc > new_best_loo:
        new_best_loo = auc
        new_best_method = name
    mark = " *** NEW BEST ***" if auc > best_loo else (" (near-best)" if auc > best_loo - 0.0003 else "")
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch102"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# M1: Fine blend weight sweep
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M1] Fine w sweep (attn_kf40_bw7_T6 on triple_ref)...", flush=True)
t1 = time.time()
for w_val in [0.005, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025, 0.030]:
    blend = (1 - w_val) * triple_ref + w_val * s_attn_best
    reg(f"attn_bw7_T6_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M2: attn_T sweep (temperature for logit attention)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M2] attn_T sweep (k=40, bw=0.07)...", flush=True)
t1 = time.time()
for T in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 100.0]:
    s = attn_knn_fisher_loo(ew_ica, top_k_fisher=40, bw=0.07, attn_T=T)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * triple_ref + w_val * s
        tag = "inf" if T >= 100 else f"{int(T)}"
        reg(f"attn_T{tag}_bw7_kf40_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M3: bw sweep (keep k=40, T=6)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M3] bw sweep (k=40, T=6)...", flush=True)
t1 = time.time()
for bw in [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
    s = attn_knn_fisher_loo(ew_ica, top_k_fisher=40, bw=bw, attn_T=6.0)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * triple_ref + w_val * s
        reg(f"attn_T6_bw{int(bw*100)}_kf40_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M4: top_k sweep (keep bw=0.07, T=6)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M4] top_k sweep (bw=0.07, T=6)...", flush=True)
t1 = time.time()
for k in [15, 20, 25, 30, 35, 40, 45, 50, 60]:
    s = attn_knn_fisher_loo(ew_ica, top_k_fisher=k, bw=0.07, attn_T=6.0)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * triple_ref + w_val * s
        reg(f"attn_T6_bw7_kf{k}_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M5: Double attention stacking (two complementary attn signals)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M5] Double attention stacking...", flush=True)
t1 = time.time()
s_attn_a = attn_knn_fisher_loo(ew_ica, top_k_fisher=30, bw=0.06, attn_T=6.0)
s_attn_b = attn_knn_fisher_loo(ew_ica, top_k_fisher=50, bw=0.08, attn_T=6.0)
s_attn_c = attn_knn_fisher_loo(ew_ica, top_k_fisher=40, bw=0.07, attn_T=4.0)
s_attn_d = attn_knn_fisher_loo(ew_ica, top_k_fisher=40, bw=0.07, attn_T=8.0)

secondary_signals = [
    ("kf30b6T6", s_attn_a),
    ("kf50b8T6", s_attn_b),
    ("kf40b7T4", s_attn_c),
    ("kf40b7T8", s_attn_d),
]
for name2, s2 in secondary_signals:
    for w_val in [0.005, 0.010]:
        # best_attn fixed at w=0.01
        blend = (1 - 0.01 - w_val) * triple_ref + 0.01 * s_attn_best + w_val * s2
        reg(f"double_attn_best+{name2}_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M6: Replace one hard Fisher in triple with attention
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M6] Replace fh40_bw7 in triple with attention...", flush=True)
t1 = time.time()
# Variant A: replace fh40b7 (w=0.03) with attn_kf40_bw7_T6 at same weight
attn_triple_A = 0.94 * fin_ref + 0.02 * fh30_b6 + 0.03 * s_attn_best + 0.01 * fh40_b6
reg("attn_triple_replaceB_fh40b7", macro_auc(attn_triple_A))
# Variant B: extend triple with small attn weight
for w_val in [0.005, 0.010, 0.015, 0.020]:
    blend = (1 - w_val) * triple_ref + w_val * s_attn_best
    reg(f"triple_plus_attn_w{int(w_val*1000):04d}", macro_auc(blend))
# Variant C: attn replaces entire triple hard component
quad_ref = (1 - 0.06) * fin_ref + 0.06 * s_attn_best
reg("attn_replace_triple_full", macro_auc(quad_ref))
print(f"  M6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M7: Apply attn signal directly on fin_ref (skip triple layer)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M7] Attn on fin_ref directly (bypass triple)...", flush=True)
t1 = time.time()
for w_val in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
    blend = (1 - w_val) * fin_ref + w_val * s_attn_best
    reg(f"attn_on_finref_w{int(w_val*100):02d}", macro_auc(blend))
print(f"  M7 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# M8: Alternative attention functions (sq_logit, rank, label_conf)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[M8] Alternative attention functions...", flush=True)
t1 = time.time()

def attn_alt_loo(ew, top_k_fisher=40, bw=0.07, mode="sq_logit"):
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
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k_fisher))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            pos_logit_si = tl_logit[pm, si]
            pos_label_si = tl[pm, si]
            sig = 1.0 / (1.0 + np.exp(-np.clip(pos_logit_si / 6.0, -10, 10)))
            if mode == "sq_logit":
                attn = sig ** 2
            elif mode == "exp_logit":
                raw = np.exp(np.clip(pos_logit_si / 6.0, -10, 10))
                attn = raw
            elif mode == "rank":
                ranks = np.argsort(np.argsort(pos_logit_si)).astype(np.float32) + 1.0
                attn = ranks
            elif mode == "label_conf":
                attn = np.clip(pos_label_si, 0.5, 1.0)
            else:
                attn = sig
            attn /= (attn.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * attn[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for mode in ["sq_logit", "exp_logit", "rank", "label_conf"]:
    s = attn_alt_loo(ew_ica, top_k_fisher=40, bw=0.07, mode=mode)
    for w_val in [0.005, 0.010, 0.015]:
        blend = (1 - w_val) * triple_ref + w_val * s
        reg(f"attn_{mode}_kf40bw7_w{int(w_val*1000):04d}", macro_auc(blend))
print(f"  M8 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary + Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch102] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

if new_best_method:
    top5 = sorted(results.values(), key=lambda x: -x["loo_auc"])[:5]
    for r2 in top5:
        delta = r2["loo_auc"] - best_loo
        print(f"    {r2['method']}: {r2['loo_auc']:.6f} ({delta:+.6f})", flush=True)
else:
    print("  No improvement found.", flush=True)

# Save results to JSON
res2 = json.load(open(RESULTS_PATH))
if isinstance(res2.get("experiments"), list):
    res2["experiments"].extend(list(results.values()))
else:
    res2["experiments"].update(results)
json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"  Saved {len(results)} results to JSON", flush=True)

# Save new best pkl if improved
if new_best_method and new_best_loo > best_loo:
    print(f"\n  SAVED: {new_best_method} LOO={new_best_loo:.6f}", flush=True)
    ep_new = copy.deepcopy(ep)
    ep_new["method"] = new_best_method
    ep_new["loo_auc"] = new_best_loo
    ep_new["batch"] = "batch102"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep_new, f)
    res3 = json.load(open(RESULTS_PATH))
    res3["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch102"}
    json.dump(res3, open(RESULTS_PATH, "w"), indent=2)
    print(f"  Updated JSON best → {new_best_method} {new_best_loo:.6f}", flush=True)
else:
    print(f"  No improvement → best remains {best_loo:.6f} ({best['method']})", flush=True)
