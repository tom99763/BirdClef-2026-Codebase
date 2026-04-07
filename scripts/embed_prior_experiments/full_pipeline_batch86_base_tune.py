"""
Batch 86: Base Component Tuning + Proto+Logit KDE
===================================================
Current best: softmax_T6_proto_kde LOO=0.991782
Base (correct): 0.991359 using stored pkl embeddings + n_windows ordering

Hypothesis: The base can be improved by:
1. Primary logit T sweep (T=5,6,7,9,10 instead of T=8)
2. Subspace n_comp=3 (currently 2)
3. Multi-T expansion: add T=12 → [8,10,12]
4. Proto+logit combined KDE: weight positives by centroid sim × logit confidence
5. NMF weight fine-tune: w_nmf ∈ {0.12, 0.14, 0.16, 0.18, 0.20}

CRITICAL: Use pkl stored embeddings (emb_win_*_norm) + n_windows ordering
to match batch82's exact base computation (0.991359).
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
labels_win  = DATA["labels"].astype(np.float32)    # (739, 234)
logit_win   = DATA["logits"].astype(np.float32)    # (739, 234)
n_windows   = DATA["n_windows"]                    # (66,)
n_files     = len(n_windows)
n_species   = labels_win.shape[1]
file_start  = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end    = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(739, np.int32)
for fi in range(n_files):
    win_file_id[file_start[fi]:file_end[fi]] = fi
EPS = 1e-8

# ── Load pkl (use STORED embeddings for exact reproducibility) ─────────────
with open("outputs/embed_prior_model.pkl", "rb") as f:
    ep = pickle.load(f)

ew_ica = ep["emb_win_ica_norm"]   # (739, 100)
ew_pca = ep["emb_win_pca_norm"]   # (739, 80)
ew_std = ep["emb_win_std_norm"]   # (739, 80)
ew_nmf = ep["emb_win_nmf_norm"]   # (739, 100)
file_labels = ep["file_labels"]   # (66, 234)
cfg = ep["config"]

print(f"Loaded: ICA{ew_ica.shape} PCA{ew_pca.shape} NMF{ew_nmf.shape}", flush=True)

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
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]; ps = te @ pw.T
            pp = pw.mean(0); pp /= norm(pp) + EPS
            sp = wmp * ps.max(1) + (1 - wmp) * (te @ pp)
            if nm.any():
                nw = tr[nm]; ns = te @ nw.T; k2 = min(k_neg, ns.shape[1])
                tn = nw[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
                tn /= norm(tn, axis=1, keepdims=True) + EPS
                ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
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

def compute_subspace(n_comp=2, wma_ss=0.92):
    ss = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_pca[win_file_id == fi]; tr = ew_pca[win_file_id != fi]
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
                pca_sp = SklearnPCA(n_components=k); pca_sp.fit(pos)
                te_r = pca_sp.inverse_transform(pca_sp.transform(te))
                err = norm(te - te_r, axis=1)
                ws[:, si] = np.clip(1 - err / (norm(te, axis=1) + EPS), 0, 1)
            except:
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
            pi = np.where(tl[:, si] > 0.5)[0]
            if len(pi) == 0: ws[:, si] = 0.5; continue
            pos = tr[pi]; c = pos.mean(0); c /= norm(c) + EPS
            pw = np.clip(pos @ c, 0, None); pw /= pw.sum() + EPS
            kern = np.exp((sims[:, pi] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * pw[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def proto_logit_kde_loo(bw=0.08, T_lw=8.0, alpha=0.5):
    """Blend centroid proto-weight with logit weight: w = alpha*proto + (1-alpha)*logit."""
    lw_full = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T_lw, -88, 88)))
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]; lw = lw_full[win_file_id != fi]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pi = np.where(tl[:, si] > 0.5)[0]
            if len(pi) == 0: ws[:, si] = 0.5; continue
            pos = tr[pi]; c = pos.mean(0); c /= norm(c) + EPS
            proto_w = np.clip(pos @ c, 0, None)
            logit_w  = lw[pi, si]
            w_comb   = alpha * proto_w + (1 - alpha) * logit_w
            w_comb   = w_comb / (w_comb.sum() + EPS)
            kern = np.exp((sims[:, pi] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * w_comb[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ── Pre-compute shared WL components (most expensive) ─────────────────────
print("Pre-computing WL components...", flush=True)
t0 = time.time()
s_ica = wl_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
s_std = wl_loo(ew_std, cfg["std_pca80"]["k_neg"], cfg["std_pca80"]["w_max_pos"], cfg["std_pca80"]["w_max_agg"])
s_pca = wl_loo(ew_pca, cfg["pca80"]["k_neg"], cfg["pca80"]["w_max_pos"], cfg["pca80"]["w_max_agg"])
s_nmf = wl_loo(ew_nmf, cfg["nmf"]["k_neg"], cfg["nmf"]["w_max_pos"], cfg["nmf"]["w_max_agg"])
uh_b  = cfg["w_ica100"] * s_ica + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
print(f"  WL components done ({time.time()-t0:.0f}s)", flush=True)

# Pre-compute logit predictions at various temperatures
pT5 = make_logit_pred(5.0); pT6 = make_logit_pred(6.0); pT7 = make_logit_pred(7.0)
pT8 = make_logit_pred(8.0); pT9 = make_logit_pred(9.0); pT10 = make_logit_pred(10.0)
pmt_810 = (pT8 + pT10) / 2
pmt_810_12 = (pT8 + pT10 + make_logit_pred(12.0)) / 3

# Pre-compute softmax at T=4 and T=6
sm4 = make_softmax_pred(4.0); sm6 = make_softmax_pred(6.0)

print(f"Logit predictions done ({time.time()-t0:.0f}s)", flush=True)

# Pre-compute subspace (n_comp=2 standard, n_comp=3 new)
print("Pre-computing subspace (n=2 and n=3)...", flush=True)
t1 = time.time()
ss2 = compute_subspace(n_comp=2, wma_ss=0.92)
ss3 = compute_subspace(n_comp=3, wma_ss=0.92)
print(f"  Subspace done ({time.time()-t1:.0f}s)", flush=True)

# Pre-compute proto KDE and proto+logit KDE
print("Pre-computing KDE variants...", flush=True)
t1 = time.time()
kde_proto_08 = proto_kde_loo(bw=0.08)
kde_pl_05    = proto_logit_kde_loo(bw=0.08, T_lw=8.0, alpha=0.5)
kde_pl_07    = proto_logit_kde_loo(bw=0.08, T_lw=8.0, alpha=0.7)
kde_pl_08    = proto_logit_kde_loo(bw=0.08, T_lw=4.0, alpha=0.5)
print(f"  KDE done ({time.time()-t1:.0f}s)", flush=True)

# Reference blend (current best)
uh_nmf_ref = (1 - 0.16) * uh_b + 0.16 * s_nmf
base_ref = 0.48 * uh_nmf_ref + 0.26 * pT8 + 0.13 * pmt_810 + 0.06 * ss2 + 0.07 * sm6
blend_ref = 0.96 * base_ref + 0.04 * kde_proto_08
ref_auc = macro_auc(blend_ref)
print(f"\nReference (should be ~0.991782): {ref_auc:.6f}", flush=True)

# ── Load results JSON ──────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    results = json.load(f)
best_auc    = results["best"]["loo_auc"]
best_method = results["best"]["method"]
tried       = set(e["method"] for e in results["experiments"])
print(f"Current best: {best_method} = {best_auc:.6f}\n", flush=True)

new_exps = []

def run(name, scores, config=None):
    if name in tried:
        return None
    auc = macro_auc(scores)
    mark = " ***NEW BEST***" if auc > best_auc else ""
    print(f"  {name}: {auc:.6f}  (Δ={auc-best_auc:+.6f}){mark}")
    return {"method": name, "loo_auc": float(auc), "config": config or {}}

# ═══════════════════════════════════════════════════════════════════════════
# Group 1: Primary logit T sweep (currently T=8 for sig)
# ═══════════════════════════════════════════════════════════════════════════
print("=== Group 1: Primary logit T sweep ===", flush=True)
for pT, T_name in [(pT5,'5'),(pT6,'6'),(pT7,'7'),(pT9,'9'),(pT10,'10')]:
    uh_nmf = (1-0.16)*uh_b + 0.16*s_nmf
    for sm, sm_name in [(sm4,'4'),(sm6,'6')]:
        w_uh = 1 - 0.26 - 0.13 - 0.06 - 0.07
        base = w_uh*uh_nmf + 0.26*pT + 0.13*pmt_810 + 0.06*ss2 + 0.07*sm
        for w_kde in [0.04, 0.05]:
            blend = (1-w_kde)*base + w_kde*kde_proto_08
            r = run(f"base_sigT{T_name}_smT{sm_name}_wk{int(w_kde*100):02d}",
                    blend, {"sig_T": float(T_name), "sm_T": float(sm_name), "w_kde": w_kde})
            if r: new_exps.append(r)

# ═══════════════════════════════════════════════════════════════════════════
# Group 2: Subspace n_comp=3
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Group 2: Subspace n_comp=3 ===", flush=True)
uh_nmf = (1-0.16)*uh_b + 0.16*s_nmf
for sm, sm_name in [(sm4,'4'),(sm6,'6')]:
    w_uh = 1 - 0.26 - 0.13 - 0.06 - 0.07
    base3 = w_uh*uh_nmf + 0.26*pT8 + 0.13*pmt_810 + 0.06*ss3 + 0.07*sm
    for w_kde in [0.03, 0.04, 0.05]:
        blend = (1-w_kde)*base3 + w_kde*kde_proto_08
        r = run(f"base_ss3_smT{sm_name}_wk{int(w_kde*100):02d}",
                blend, {"ss_n": 3, "sm_T": float(sm_name), "w_kde": w_kde})
        if r: new_exps.append(r)
    # Also with w_ss=0.08
    base3b = (1-0.26-0.13-0.08-0.07)*uh_nmf + 0.26*pT8 + 0.13*pmt_810 + 0.08*ss3 + 0.07*sm
    for w_kde in [0.04]:
        blend = (1-w_kde)*base3b + w_kde*kde_proto_08
        r = run(f"base_ss3_wss08_smT{sm_name}_wk{int(w_kde*100):02d}",
                blend, {"ss_n": 3, "w_ss": 0.08, "sm_T": float(sm_name), "w_kde": w_kde})
        if r: new_exps.append(r)

# ═══════════════════════════════════════════════════════════════════════════
# Group 3: Multi-T expansion [8,10,12]
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Group 3: Multi-T expansion [8,10,12] ===", flush=True)
uh_nmf = (1-0.16)*uh_b + 0.16*s_nmf
for sm, sm_name in [(sm4,'4'),(sm6,'6')]:
    w_uh = 1 - 0.26 - 0.13 - 0.06 - 0.07
    base_mt12 = w_uh*uh_nmf + 0.26*pT8 + 0.13*pmt_810_12 + 0.06*ss2 + 0.07*sm
    for w_kde in [0.03, 0.04, 0.05]:
        blend = (1-w_kde)*base_mt12 + w_kde*kde_proto_08
        r = run(f"base_mt81012_smT{sm_name}_wk{int(w_kde*100):02d}",
                blend, {"mt_temps": "8,10,12", "sm_T": float(sm_name), "w_kde": w_kde})
        if r: new_exps.append(r)

# ═══════════════════════════════════════════════════════════════════════════
# Group 4: NMF weight fine-tune
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Group 4: NMF weight fine-tune ===", flush=True)
for w_nmf in [0.12, 0.14, 0.18, 0.20]:
    uh_nmf_t = (1-w_nmf)*uh_b + w_nmf*s_nmf
    w_uh = 1 - 0.26 - 0.13 - 0.06 - 0.07
    base_nmf = w_uh*uh_nmf_t + 0.26*pT8 + 0.13*pmt_810 + 0.06*ss2 + 0.07*sm6
    for w_kde in [0.04]:
        blend = (1-w_kde)*base_nmf + w_kde*kde_proto_08
        r = run(f"base_nmf{int(w_nmf*100):02d}_smT6_wk{int(w_kde*100):02d}",
                blend, {"w_nmf": w_nmf, "sm_T": 6.0, "w_kde": w_kde})
        if r: new_exps.append(r)

# ═══════════════════════════════════════════════════════════════════════════
# Group 5: Proto+logit combined KDE
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Group 5: Proto+logit combined KDE ===", flush=True)
uh_nmf = (1-0.16)*uh_b + 0.16*s_nmf
w_uh = 1 - 0.26 - 0.13 - 0.06 - 0.07
base = w_uh*uh_nmf + 0.26*pT8 + 0.13*pmt_810 + 0.06*ss2 + 0.07*sm6
for kde_s, kde_name in [(kde_pl_05,'pl05'), (kde_pl_07,'pl07'), (kde_pl_08,'pl08')]:
    for w_kde in [0.03, 0.04, 0.05]:
        blend = (1-w_kde)*base + w_kde*kde_s
        r = run(f"kde_{kde_name}_wk{int(w_kde*100):02d}",
                blend, {"kde_type": kde_name, "w_kde": w_kde})
        if r: new_exps.append(r)

# ═══════════════════════════════════════════════════════════════════════════
# Group 6: Combining best improvements
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Group 6: Best-of combinations ===", flush=True)
uh_nmf = (1-0.16)*uh_b + 0.16*s_nmf
w_uh = 1 - 0.26 - 0.13 - 0.06 - 0.07

# ss3 + T6 sig + T6 softmax
base_combo1 = w_uh*uh_nmf + 0.26*pT6 + 0.13*pmt_810 + 0.06*ss3 + 0.07*sm6
for w_kde in [0.03, 0.04, 0.05]:
    blend = (1-w_kde)*base_combo1 + w_kde*kde_proto_08
    r = run(f"combo_sigT6_ss3_smT6_wk{int(w_kde*100):02d}", blend, {"combo": 1, "w_kde": w_kde})
    if r: new_exps.append(r)

# ss3 + multi-T [8,10,12] + T6 softmax
base_combo2 = w_uh*uh_nmf + 0.26*pT8 + 0.13*pmt_810_12 + 0.06*ss3 + 0.07*sm6
for w_kde in [0.03, 0.04, 0.05]:
    blend = (1-w_kde)*base_combo2 + w_kde*kde_proto_08
    r = run(f"combo_mt12_ss3_smT6_wk{int(w_kde*100):02d}", blend, {"combo": 2, "w_kde": w_kde})
    if r: new_exps.append(r)

# T6 sig + multi-T [8,10,12] + ss3 + T6 softmax
base_combo3 = w_uh*uh_nmf + 0.26*pT6 + 0.13*pmt_810_12 + 0.06*ss3 + 0.07*sm6
for w_kde in [0.03, 0.04, 0.05]:
    blend = (1-w_kde)*base_combo3 + w_kde*kde_proto_08
    r = run(f"combo_sigT6_mt12_ss3_smT6_wk{int(w_kde*100):02d}", blend, {"combo": 3, "w_kde": w_kde})
    if r: new_exps.append(r)

# ═══════════════════════════════════════════════════════════════════════════
# Group 7: Wider softmax T sweep with ss3
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Group 7: Softmax T with ss3 ===", flush=True)
uh_nmf = (1-0.16)*uh_b + 0.16*s_nmf
w_uh = 1 - 0.26 - 0.13 - 0.06 - 0.07
for T_sm in [5.0, 6.0, 7.0, 8.0]:
    sm = make_softmax_pred(T_sm)
    base_ss3 = w_uh*uh_nmf + 0.26*pT8 + 0.13*pmt_810 + 0.06*ss3 + 0.07*sm
    for w_kde in [0.04]:
        blend = (1-w_kde)*base_ss3 + w_kde*kde_proto_08
        r = run(f"ss3_smT{int(T_sm)}_wk04", blend, {"ss_n": 3, "sm_T": T_sm, "w_kde": w_kde})
        if r: new_exps.append(r)

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
valid = [e for e in new_exps if e is not None]
print(f"\n{'='*60}", flush=True)
print(f"Batch 86 Summary", flush=True)
print(f"Experiments run: {len(valid)}", flush=True)

if valid:
    best_new = max(valid, key=lambda x: x["loo_auc"])
    print(f"Best new: {best_new['method']} = {best_new['loo_auc']:.6f}")
    print(f"Current best: {best_method} = {best_auc:.6f}")
    print(f"Delta: {best_new['loo_auc'] - best_auc:+.6f}")

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

    print("\nTop 5 new:")
    for e in sorted(valid, key=lambda x: -x["loo_auc"])[:5]:
        print(f"  {e['method']}: {e['loo_auc']:.6f}")
