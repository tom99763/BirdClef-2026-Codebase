"""
batch138 — 3way ratio fine-tune + blend re-optimize at ratio=0.88
===============================================================================
Current best: ratio_idf88_tr12_a20 LOO=0.995152
  Formula: 0.76×3way_best + 0.21×3way_ica_alt + 0.03×3way_std
  apply_3way: 0.88×idf_cooc(alpha=0.200) + 0.12×two_round

New directions:
 A: Fine ratio sweep (0.84-0.92 step 0.01) + ultra-fine (0.86-0.90 step 0.005)
 B: Blend re-optimize at ratio=0.88 (full grid)
 C: Joint (ratio, blend) around best
 D: Alpha fine-tune at ratio=0.88 (alpha=0.18-0.24)
 E: Combined best settings fine sweep
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-8
ROOT = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

DATA = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
n_windows = DATA["n_windows"]
n_files   = len(n_windows)
n_species = DATA["labels"].shape[1]

with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

file_labels = ep["file_labels"]
double_best = ep["chain_double_best"]
ica_ens_alt = ep["chain_ica_ens_alt"]
std_ens_ref = ep["chain_std_ens_ref"]

res = json.load(open(RESULTS_PATH))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch138] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

def macro_auc(s, fl=file_labels):
    aucs = []
    for si in range(n_species):
        y = fl[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try: aucs.append(roc_auc_score(y, s[:, si]))
            except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, config_dict=None):
    global best_loo
    if mname in tried: return score - best_loo
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 138}
    res["experiments"].append(entry)
    tried.add(mname)
    if score > best_loo + 1e-7:
        best_loo = score
        res["best"] = {"method": mname, "loo_auc": float(score)}
        with open(MODEL_PATH, "rb") as f:
            ep_up = pickle.load(f)
        ep_up["method"] = mname
        ep_up["loo_auc"] = float(score)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(ep_up, f)
        print(f"  [SAVED] New best PKL!", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

# ─── Co-occurrence setup ─────────────────────────────────────────────────────
fl_hard  = file_labels.astype(np.float32)
count_i  = fl_hard.sum(0) + EPS
COOC     = (fl_hard.T @ fl_hard) / count_i[:, None]
np.fill_diagonal(COOC, 0)
raw_idf  = np.clip(np.log(float(n_files) / (count_i + 1.0 - EPS)), 0, None)
IDF075   = raw_idf ** 0.75; IDF075 /= (IDF075.mean() + EPS)

def soft_cooc(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope*(s-center), -88, 88)))
        sg = s * gate * (idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi] = s; continue
        c = COOC.T @ sg; mc = np.abs(c).max()
        if mc > EPS: c /= mc
        out[fi] = (1-alpha)*s + alpha*np.clip(c, 0, None)
    return out

def idf_cooc(scores, alpha=0.200, blend=0.55):
    sp = np.clip(scores, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    return (1-blend)*scores + blend*sc

def two_round(scores):
    r1 = soft_cooc(scores, center=0.54, slope=41.0, alpha=0.089)
    return soft_cooc(r1, center=0.53, slope=37.0, alpha=0.040)

def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.88, r_tr=0.12):
    return r_idf * idf_cooc(s, alpha=alpha, blend=blend) + r_tr * two_round(s)

def blend3(c, i, s, wb=0.76, wi=0.21, ws=0.03):
    return wb*c + wi*i + ws*s

# Verify
c3 = apply_3way(double_best); i3 = apply_3way(ica_ens_alt); s3 = apply_3way(std_ens_ref)
chk = blend3(c3, i3, s3)
print(f"Verify ratio_idf88 baseline: {macro_auc(chk):.6f} (expect 0.995152)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Fine ratio sweep around 0.88
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Fine ratio sweep ===", flush=True)
ratio_results = {}
for r_idf in [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92,
              0.860, 0.865, 0.870, 0.875, 0.880, 0.885, 0.890, 0.895]:
    r_idf_r = round(float(r_idf), 3)
    r_tr_r  = round(1.0 - r_idf_r, 3)
    if r_idf_r in ratio_results: continue
    c3r = apply_3way(double_best, r_idf=r_idf_r, r_tr=r_tr_r)
    i3r = apply_3way(ica_ens_alt, r_idf=r_idf_r, r_tr=r_tr_r)
    s3r = apply_3way(std_ens_ref, r_idf=r_idf_r, r_tr=r_tr_r)
    br  = blend3(c3r, i3r, s3r)
    ar  = macro_auc(br)
    ratio_results[r_idf_r] = ar
    mname = f"ratio_idf{int(r_idf_r*1000):d}_a20"
    delta = save_result(mname, ar, {"r_idf": r_idf_r, "r_tr": r_tr_r, "alpha": 0.200})
    flag  = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  r_idf={r_idf_r:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

best_r_idf = max(ratio_results, key=ratio_results.get)
print(f"  Best r_idf: {best_r_idf:.3f} → {ratio_results[best_r_idf]:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Blend re-optimize at best ratio
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Blend re-optimize at r_idf={best_r_idf:.3f} ===", flush=True)
r_tr_best = round(1.0 - best_r_idf, 3)
c3b = apply_3way(double_best, r_idf=best_r_idf, r_tr=r_tr_best)
i3b = apply_3way(ica_ens_alt, r_idf=best_r_idf, r_tr=r_tr_best)
s3b = apply_3way(std_ens_ref, r_idf=best_r_idf, r_tr=r_tr_best)

best_blend_auc = best_loo
best_blend = (0.76, 0.21, 0.03)
for wb in np.arange(0.68, 0.84, 0.01):
    for wi in np.arange(0.12, 0.26, 0.01):
        wb_r = round(float(wb), 2); wi_r = round(float(wi), 2)
        ws_r = round(1.0 - wb_r - wi_r, 2)
        if ws_r < 0.02 or ws_r > 0.16: continue
        mname = f"rb138_r{int(best_r_idf*1000)}_b{int(wb_r*100)}_i{int(wi_r*100)}_s{int(ws_r*100)}"
        if mname in tried: continue
        result = blend3(c3b, i3b, s3b, wb=wb_r, wi=wi_r, ws=ws_r)
        score  = macro_auc(result)
        delta  = save_result(mname, score, {"wb": wb_r, "wi": wi_r, "ws": ws_r, "r_idf": best_r_idf})
        if score > best_blend_auc:
            best_blend_auc = score
            best_blend = (wb_r, wi_r, ws_r)
        if score > best_loo - 0.00008:
            flag = " ← NEW BEST!" if score > best_loo else ""
            print(f"  {mname}: {score:.6f} {delta:+.6f}{flag}", flush=True)

wb_best, wi_best, ws_best = best_blend
print(f"  Best blend at r_idf={best_r_idf:.3f}: {best_blend} → {best_blend_auc:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Alpha fine-tune at best ratio + best blend
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Alpha fine-tune at r_idf={best_r_idf:.3f} ===", flush=True)
for alpha_val in [0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]:
    av = round(float(alpha_val), 2)
    c3a = apply_3way(double_best, alpha=av, r_idf=best_r_idf, r_tr=r_tr_best)
    i3a = apply_3way(ica_ens_alt, alpha=av, r_idf=best_r_idf, r_tr=r_tr_best)
    s3a = apply_3way(std_ens_ref, alpha=av, r_idf=best_r_idf, r_tr=r_tr_best)
    ba  = blend3(c3a, i3a, s3a, wb=wb_best, wi=wi_best, ws=ws_best)
    aa  = macro_auc(ba)
    mname = f"alpha_{int(av*100)}_r{int(best_r_idf*1000)}"
    delta = save_result(mname, aa, {"alpha": av, "r_idf": best_r_idf, "wb": wb_best, "wi": wi_best, "ws": ws_best})
    flag  = " ← NEW BEST!" if aa > best_loo else ""
    print(f"  alpha={av:.2f}: {aa:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: blend parameter sweep (currently 0.55) at best settings
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: idf_cooc blend param sweep at best settings ===", flush=True)
for blend_p in [0.40, 0.45, 0.50, 0.52, 0.55, 0.57, 0.60, 0.65]:
    bp = round(float(blend_p), 2)
    c3d = apply_3way(double_best, blend=bp, r_idf=best_r_idf, r_tr=r_tr_best)
    i3d = apply_3way(ica_ens_alt, blend=bp, r_idf=best_r_idf, r_tr=r_tr_best)
    s3d = apply_3way(std_ens_ref, blend=bp, r_idf=best_r_idf, r_tr=r_tr_best)
    bd  = blend3(c3d, i3d, s3d, wb=wb_best, wi=wi_best, ws=ws_best)
    ad  = macro_auc(bd)
    mname = f"blendp_{int(bp*100)}_r{int(best_r_idf*1000)}_a20"
    delta = save_result(mname, ad, {"blend": bp, "r_idf": best_r_idf})
    flag  = " ← NEW BEST!" if ad > best_loo else ""
    print(f"  blend={bp:.2f}: {ad:.6f} {delta:+.6f}{flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 138]
print(f"Batch138 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
