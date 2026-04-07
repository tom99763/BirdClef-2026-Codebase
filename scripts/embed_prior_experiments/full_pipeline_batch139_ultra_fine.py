"""
batch139 — Ultra-fine ratio + near-zero STD blend search
===============================================================================
Current best: rb138_r875_b75_i23_s2 LOO=0.995186
  r_idf=0.875, r_tr=0.125, alpha=0.200, blend=0.55
  Final blend: 0.75×3way_best + 0.23×3way_ica_alt + 0.02×3way_std

Observations:
- STD weight declining: 0.08 → 0.05 → 0.03 → 0.02 (nearly zero)
- r_idf converging at 0.875
- alpha=0.200 confirmed optimal at this ratio

Directions:
 A: Ultra-fine ratio (0.871-0.879 step 0.001)
 B: Near-zero STD blend (ws=0.00-0.03) with wb+wi grid
 C: Pure 2-component blend (no STD) fine sweep
 D: Fine alpha at best ratio+blend (0.19-0.21 step 0.005)
 E: Fine blend parameter (0.53-0.57 step 0.01)
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
n_files   = len(DATA["n_windows"])
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
print(f"[batch139] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 139}
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

# Co-occurrence
fl_hard = file_labels.astype(np.float32)
count_i = fl_hard.sum(0) + EPS
COOC    = (fl_hard.T @ fl_hard) / count_i[:, None]
np.fill_diagonal(COOC, 0)
raw_idf = np.clip(np.log(float(n_files) / (count_i + 1.0 - EPS)), 0, None)
IDF075  = raw_idf ** 0.75; IDF075 /= (IDF075.mean() + EPS)

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

def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.089)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.040)
    return r_idf * idf_s + r_tr * tr

def blend3(c, i, s, wb, wi, ws):
    return wb*c + wi*i + ws*s

# Pre-compute 3way at r_idf=0.875
c3_ref = apply_3way(double_best)
i3_ref = apply_3way(ica_ens_alt)
s3_ref = apply_3way(std_ens_ref)
chk = blend3(c3_ref, i3_ref, s3_ref, 0.75, 0.23, 0.02)
print(f"Verify: {macro_auc(chk):.6f} (expect 0.995186)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Ultra-fine ratio (0.870-0.880 step 0.001) with current best blend
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Ultra-fine ratio sweep ===", flush=True)
ratio_results = {}
for r_idf_int in range(870, 881):
    r_idf = r_idf_int / 1000.0
    r_tr  = round(1.0 - r_idf, 3)
    c3r = apply_3way(double_best, r_idf=r_idf, r_tr=r_tr)
    i3r = apply_3way(ica_ens_alt, r_idf=r_idf, r_tr=r_tr)
    s3r = apply_3way(std_ens_ref, r_idf=r_idf, r_tr=r_tr)
    br  = blend3(c3r, i3r, s3r, 0.75, 0.23, 0.02)
    ar  = macro_auc(br)
    ratio_results[r_idf] = ar
    mname = f"ufr_{r_idf_int}"
    delta = save_result(mname, ar, {"r_idf": r_idf, "wb": 0.75, "wi": 0.23, "ws": 0.02})
    flag  = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  r_idf={r_idf:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

best_r = max(ratio_results, key=ratio_results.get)
print(f"  Best ratio: {best_r:.3f} → {ratio_results[best_r]:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Near-zero STD blend at best ratio
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Near-zero STD blend at r_idf={best_r:.3f} ===", flush=True)
r_tr_best = round(1.0 - best_r, 3)
c3b = apply_3way(double_best, r_idf=best_r, r_tr=r_tr_best)
i3b = apply_3way(ica_ens_alt, r_idf=best_r, r_tr=r_tr_best)
s3b = apply_3way(std_ens_ref, r_idf=best_r, r_tr=r_tr_best)

best_blend_auc = best_loo
best_blend = (0.75, 0.23, 0.02)
for ws in [0.00, 0.01, 0.02, 0.03, 0.04]:
    for wi in np.arange(0.18, 0.28, 0.01):
        wi_r = round(float(wi), 2); ws_r = round(float(ws), 2)
        wb_r = round(1.0 - wi_r - ws_r, 2)
        if wb_r < 0.68 or wb_r > 0.84: continue
        mname = f"b139_r{int(best_r*1000)}_b{int(wb_r*100)}_i{int(wi_r*100)}_s{int(ws_r*100)}"
        if mname in tried: continue
        result = blend3(c3b, i3b, s3b, wb=wb_r, wi=wi_r, ws=ws_r)
        score  = macro_auc(result)
        delta  = save_result(mname, score, {"wb": wb_r, "wi": wi_r, "ws": ws_r, "r": best_r})
        if score > best_blend_auc:
            best_blend_auc = score
            best_blend = (wb_r, wi_r, ws_r)
        if score > best_loo - 0.00008:
            flag = " ← NEW BEST!" if score > best_loo else ""
            print(f"  wb={wb_r:.2f} wi={wi_r:.2f} ws={ws_r:.2f}: {score:.6f} {delta:+.6f}{flag}", flush=True)

wb_best, wi_best, ws_best = best_blend
print(f"  Best blend: {best_blend} → {best_blend_auc:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Pure 2-component blend (no STD) fine sweep
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: 2-component blend (no STD) ===", flush=True)
for wi in np.arange(0.18, 0.32, 0.01):
    wi_r = round(float(wi), 2)
    wb_r = round(1.0 - wi_r, 2)
    if wb_r < 0.68 or wb_r > 0.84: continue
    mname = f"2comp_r{int(best_r*1000)}_b{int(wb_r*100)}_i{int(wi_r*100)}"
    if mname in tried: continue
    result = blend3(c3b, i3b, s3b, wb=wb_r, wi=wi_r, ws=0.0)
    score  = macro_auc(result)
    delta  = save_result(mname, score, {"wb": wb_r, "wi": wi_r, "ws": 0.0, "r": best_r})
    if score > best_loo - 0.00008:
        flag = " ← NEW BEST!" if score > best_loo else ""
        print(f"  wb={wb_r:.2f} wi={wi_r:.2f} ws=0.00: {score:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Fine alpha at best settings
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Fine alpha at r={best_r:.3f}, blend={best_blend} ===", flush=True)
for alpha_val in [0.190, 0.195, 0.200, 0.205, 0.210, 0.215, 0.220]:
    av = round(float(alpha_val), 3)
    c3d = apply_3way(double_best, alpha=av, r_idf=best_r, r_tr=r_tr_best)
    i3d = apply_3way(ica_ens_alt, alpha=av, r_idf=best_r, r_tr=r_tr_best)
    s3d = apply_3way(std_ens_ref, alpha=av, r_idf=best_r, r_tr=r_tr_best)
    bd  = blend3(c3d, i3d, s3d, wb=wb_best, wi=wi_best, ws=ws_best)
    ad  = macro_auc(bd)
    mname = f"alpha_{int(av*1000)}_r{int(best_r*1000)}"
    delta = save_result(mname, ad, {"alpha": av, "r": best_r})
    flag  = " ← NEW BEST!" if ad > best_loo else ""
    print(f"  alpha={av:.3f}: {ad:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: idf_cooc blend parameter fine sweep at best settings
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: idf_cooc blend param fine sweep ===", flush=True)
for bp in [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58]:
    bpv = round(float(bp), 2)
    c3e = apply_3way(double_best, blend=bpv, r_idf=best_r, r_tr=r_tr_best)
    i3e = apply_3way(ica_ens_alt, blend=bpv, r_idf=best_r, r_tr=r_tr_best)
    s3e = apply_3way(std_ens_ref, blend=bpv, r_idf=best_r, r_tr=r_tr_best)
    be  = blend3(c3e, i3e, s3e, wb=wb_best, wi=wi_best, ws=ws_best)
    ae  = macro_auc(be)
    mname = f"blendfine_{int(bpv*100)}_r{int(best_r*1000)}"
    delta = save_result(mname, ae, {"blend": bpv, "r": best_r})
    flag  = " ← NEW BEST!" if ae > best_loo else ""
    print(f"  blend={bpv:.2f}: {ae:.6f} {delta:+.6f}{flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 139]
print(f"Batch139 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
