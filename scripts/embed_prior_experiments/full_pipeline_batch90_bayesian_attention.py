"""
Batch 90: Bayesian Ridge + Logit-Attention KNN as KDE Components
==================================================================
Current best: softmax_T6_proto_kde LOO=0.991782

Priority methods (from user queue):
3. Bayesian Ridge Regression (sklearn BayesianRidge, PCA 64-dim):
   - Per species: fit BayesianRidge(X=pos_embeddings_pca64, y=1) vs neg
   - Actually: fit BayesianRidge to classify pos vs neg windows
   - Score: predicted probability of being positive class
   - Use as a per-file score by averaging window probs

4. Logit-Attention KNN:
   - For each test window, find k-NN in training positives
   - Weight each neighbor by softmax(logit[neighbor, species])
   - Attention score = weighted sum → file-level max

Both implemented as KDE replacements: final = (1-w)*base + w*new_score
Using n_windows-based correct file ordering.
"""

import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import BayesianRidge, Ridge
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

print(f"Loaded: ICA{ew_ica.shape} PCA{ew_pca.shape}", flush=True)

# ── Standard helpers ───────────────────────────────────────────────────────
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

# ── New Methods ────────────────────────────────────────────────────────────
def bayesian_ridge_loo(n_pca=64, use_neg=True):
    """BayesianRidge per-species in PCA-n_pca space.
    For each species: fit BayesianRidge on pos(y=1) + neg(y=0) windows,
    predict probability-like scores for test windows.
    Score per file = mean of predicted scores over test windows.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        # Fit per-species PCA on all training data
        k_pca = min(n_pca, tr.shape[0] - 1, tr.shape[1] - 1)
        if k_pca < 4: ws[:] = 0.5; out[fi] = ws.max(0); continue
        try:
            pca_b = SklearnPCA(n_components=k_pca).fit(tr)
            tr_p = pca_b.transform(tr).astype(np.float64)
            te_p = pca_b.transform(te).astype(np.float64)
        except Exception:
            ws[:] = 0.5; out[fi] = ws.max(0); continue

        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            if use_neg and nm.any():
                idx = np.where(pm | nm)[0]
                y = np.where(tl[idx, si] > 0.5, 1.0, 0.0)
                X = tr_p[idx]
            else:
                idx = np.where(pm)[0]
                # Positive-only: one-class via centroid regression
                X = tr_p[idx]; y = np.ones(len(idx))
            if len(np.unique(y)) < 2:
                ws[:, si] = 0.5; continue
            try:
                br = BayesianRidge(max_iter=100)
                br.fit(X, y)
                preds = br.predict(te_p)
                ws[:, si] = np.clip(preds, 0, None).astype(np.float32)
            except Exception:
                ws[:, si] = 0.5
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def attention_knn_loo(k=10, T_attn=6.0):
    """Logit-attention KNN:
    For each test window, find k-nearest positives in ICA space.
    Weight each neighbor by softmax(logit[neighbor, species] / T_attn).
    Score = sum of cosim * attention_weight.
    """
    lw = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T_attn, -88, 88)))  # (739, 234)
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]; lw_tr = lw[win_file_id != fi]
        sims = te @ tr.T  # (n_te, n_tr)
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_sims = sims[:, pos_idx]  # (n_te, n_pos)
            pos_logits = lw_tr[pos_idx, si]  # (n_pos,) logit scores of positives
            k2 = min(k, len(pos_idx))
            # For each test window, get top-k nearest positives
            top_idx = np.argsort(-pos_sims, axis=1)[:, :k2]  # (n_te, k2)
            top_sims = np.take_along_axis(pos_sims, top_idx, axis=1)  # (n_te, k2)
            top_logits = pos_logits[top_idx]  # (n_te, k2)
            # Softmax attention over top-k
            attn_sm = np.exp((top_logits - top_logits.max(1, keepdims=True)) / T_attn)
            attn_sm = attn_sm / (attn_sm.sum(1, keepdims=True) + EPS)
            ws[:, si] = np.clip((top_sims * attn_sm).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def attention_knn_proto_loo(k=10, T_attn=6.0):
    """Hybrid: proto-weighted centroid sim × logit-attention KNN."""
    lw = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T_attn, -88, 88)))
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]; lw_tr = lw[win_file_id != fi]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]
            centroid = pos_wins.mean(0); centroid /= (norm(centroid) + EPS)
            # Proto weight: cosim to centroid
            proto_w = np.clip(pos_wins @ centroid, 0, None)
            # Logit weight
            logit_w = lw_tr[pos_idx, si]
            # Combined: 50% proto + 50% logit
            comb_w = 0.5 * proto_w + 0.5 * logit_w
            comb_w = comb_w / (comb_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (0.08**2 + EPS))
            ws[:, si] = np.clip((kern * comb_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def ridge_reg_loo(n_pca=32, alpha=1.0):
    """Ridge regression per-species (faster alternative to BayesianRidge)."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        k_pca = min(n_pca, tr.shape[0] - 1, tr.shape[1] - 1)
        if k_pca < 4: ws[:] = 0.5; out[fi] = ws.max(0); continue
        try:
            pca_r = SklearnPCA(n_components=k_pca).fit(tr)
            tr_p = pca_r.transform(tr).astype(np.float64)
            te_p = pca_r.transform(te).astype(np.float64)
        except Exception:
            ws[:] = 0.5; out[fi] = ws.max(0); continue
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            if nm.any():
                idx = np.where(pm | nm)[0]
                y = np.where(tl[idx, si] > 0.5, 1.0, 0.0)
                X = tr_p[idx]
                if len(np.unique(y)) < 2: ws[:, si] = 0.5; continue
                try:
                    rr = Ridge(alpha=alpha)
                    rr.fit(X, y)
                    ws[:, si] = np.clip(rr.predict(te_p), 0, None).astype(np.float32)
                except Exception:
                    ws[:, si] = 0.5
            else:
                ws[:, si] = 0.5
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
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
kde_ica = proto_kde_loo_ica(bw=0.08)
print(f"  Base done ({time.time()-t0:.0f}s)", flush=True)
print(f"Base AUC: {macro_auc(base):.6f}", flush=True)
print(f"Reference: {macro_auc(0.96*base + 0.04*kde_ica):.6f} (should ~0.991782)", flush=True)

# ── New components ────────────────────────────────────────────────────────
print("\nPre-computing new components...", flush=True)
t1 = time.time()
br64  = bayesian_ridge_loo(n_pca=64)
br32  = bayesian_ridge_loo(n_pca=32)
rr32_a1 = ridge_reg_loo(n_pca=32, alpha=1.0)
rr32_a01 = ridge_reg_loo(n_pca=32, alpha=0.1)
rr64_a1  = ridge_reg_loo(n_pca=64, alpha=1.0)
print(f"  Regression done ({time.time()-t1:.0f}s)", flush=True)

t2 = time.time()
attn_k5_T6   = attention_knn_loo(k=5,  T_attn=6.0)
attn_k10_T6  = attention_knn_loo(k=10, T_attn=6.0)
attn_k10_T8  = attention_knn_loo(k=10, T_attn=8.0)
attn_k20_T6  = attention_knn_loo(k=20, T_attn=6.0)
attn_proto_k10 = attention_knn_proto_loo(k=10, T_attn=6.0)
attn_proto_k20 = attention_knn_proto_loo(k=20, T_attn=6.0)
print(f"  Attention KNN done ({time.time()-t2:.0f}s)", flush=True)

# ── Load results & evaluate ────────────────────────────────────────────────
RES_PATH = Path("outputs/embed_prior_results.json")
with open(RES_PATH) as f:
    results = json.load(f)
best_auc = results["best"]["loo_auc"]
best_method = results["best"]["method"]
print(f"\nCurrent best: {best_method} = {best_auc:.6f}\n", flush=True)

experiments = []
def eval_blend(name, kde_score, w_kde):
    final = (1 - w_kde) * base + w_kde * kde_score
    auc = macro_auc(final)
    delta = auc - best_auc
    mark = " ***NEW BEST***" if auc > best_auc + 1e-6 else ""
    print(f"  {name}: {auc:.6f}  (Δ={delta:+.6f}){mark}", flush=True)
    experiments.append({"method": name, "loo_auc": auc})
    return auc

print("=== BayesianRidge PCA-64 ===")
for name, score in [("br64", br64), ("br32", br32)]:
    for wk in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        eval_blend(f"{name}_wk{int(wk*100):02d}", score, wk)

print("\n=== Ridge Regression ===")
for name, score in [("rr32_a1", rr32_a1), ("rr32_a01", rr32_a01), ("rr64_a1", rr64_a1)]:
    for wk in [0.03, 0.04, 0.05, 0.06]:
        eval_blend(f"{name}_wk{int(wk*100):02d}", score, wk)

print("\n=== Logit-Attention KNN ===")
for name, score in [("attn_k5_T6", attn_k5_T6), ("attn_k10_T6", attn_k10_T6),
                     ("attn_k10_T8", attn_k10_T8), ("attn_k20_T6", attn_k20_T6)]:
    for wk in [0.03, 0.04, 0.05, 0.06]:
        eval_blend(f"{name}_wk{int(wk*100):02d}", score, wk)

print("\n=== Attention + Proto Hybrid ===")
for name, score in [("attn_proto_k10", attn_proto_k10), ("attn_proto_k20", attn_proto_k20)]:
    for wk in [0.03, 0.04, 0.05, 0.06]:
        eval_blend(f"{name}_wk{int(wk*100):02d}", score, wk)

print("\n=== Ridge + ICA KDE blend ===")
for w_r in [0.3, 0.5, 0.7]:
    b_kde = w_r * rr32_a1 + (1-w_r) * kde_ica
    for si in range(n_species):
        mx = b_kde[:, si].max()
        if mx > EPS: b_kde[:, si] /= mx
    for wk in [0.04, 0.05]:
        eval_blend(f"rr32_ica_wr{int(w_r*10)}_wk{int(wk*100):02d}", b_kde, wk)

print("\n=== Attention + ICA KDE blend ===")
for w_a in [0.3, 0.5, 0.7]:
    b_kde = w_a * attn_k10_T6 + (1-w_a) * kde_ica
    for si in range(n_species):
        mx = b_kde[:, si].max()
        if mx > EPS: b_kde[:, si] /= mx
    for wk in [0.04, 0.05]:
        eval_blend(f"attn_ica_wa{int(w_a*10)}_wk{int(wk*100):02d}", b_kde, wk)

# ── Finalize ────────────────────────────────────────────────────────────────
best_new = max(experiments, key=lambda x: x["loo_auc"])
delta_best = best_new["loo_auc"] - best_auc

print(f"\n{'='*60}")
print(f"Batch 90 Summary")
print(f"Experiments run: {len(experiments)}")
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

print(f"\nTop 5 new:")
for e in sorted(experiments, key=lambda x: -x["loo_auc"])[:5]:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
