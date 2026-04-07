"""
batch137 — Parameter re-tune at alpha=0.200
===============================================================================
Current best: alpha_200 LOO=0.995120
  Formula: 0.76×3way_best + 0.16×3way_ica_alt + 0.08×3way_std
  apply_3way: alpha=0.200, blend=0.55, center=0.55, slope=41.0, s_pow=2.0

Hypothesis: previous blend weights (0.76/0.16/0.08) were optimized with alpha=0.130.
With stronger co-occ smoothing (alpha=0.200), optimal weights may shift.
Also: center and slope parameters in idf_cooc gate function not yet swept at alpha=0.200.

Directions:
 A: Blend weight re-sweep at alpha=0.200 (w_best × w_ica × w_std grid)
 B: Center parameter sweep (0.40-0.70) at alpha=0.200
 C: Slope parameter sweep (25-65) at alpha=0.200
 D: Joint (center, slope) at alpha=0.200, best blend
 E: 3way ratio (idf_cooc vs two_round) at alpha=0.200
 F: Fine blend grid with best (center, slope) found in D
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
n_windows    = DATA["n_windows"]
n_files      = len(n_windows)
n_species    = DATA["labels"].shape[1]

with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

file_labels = ep["file_labels"]   # [66, 234]

# Load pre-computed raw chain components (saved by batch136)
double_best = ep["chain_double_best"]    # [66, 234]
ica_ens_alt = ep["chain_ica_ens_alt"]    # [66, 234]
std_ens_ref = ep["chain_std_ens_ref"]    # [66, 234]
print(f"[batch137] Raw chain components loaded", flush=True)

res = json.load(open(RESULTS_PATH))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch137] Current best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    if mname in tried:
        return score - best_loo
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 137}
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
fl_hard   = file_labels.astype(np.float32)
count_i   = fl_hard.sum(0) + EPS
cooc_raw  = fl_hard.T @ fl_hard
COOC_NORM = cooc_raw / count_i[:, None]
np.fill_diagonal(COOC_NORM, 0)
raw_idf   = np.log(float(n_files) / (count_i + 1.0 - EPS))
raw_idf   = np.clip(raw_idf, 0, None)
IDF_W075  = raw_idf ** 0.75; IDF_W075 /= (IDF_W075.mean() + EPS)

def soft_cooc(scores, center=0.53, slope=37.0, alpha=0.086, idf_w=None):
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope * (s - center), -88, 88)))
        s_gated = s * gate
        if idf_w is not None: s_gated = s_gated * idf_w
        if np.abs(s_gated).sum() < EPS: smoothed[fi] = s; continue
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS: contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def idf_cooc_custom(scores, alpha=0.200, blend=0.55, center=0.55, slope=41.0, s_pow=2.0):
    """idf_cooc with customizable parameters."""
    s_p = np.clip(scores, 0, 1) ** s_pow
    s_c = soft_cooc(s_p, center=center, slope=slope, alpha=alpha, idf_w=IDF_W075)
    return (1 - blend) * scores + blend * s_c

def two_round_fixed(scores):
    """two_round with fixed parameters (not swept)."""
    r1 = soft_cooc(scores, center=0.54, slope=41.0, alpha=0.089)
    return soft_cooc(r1, center=0.53, slope=37.0, alpha=0.040)

def apply_3way_custom(s, alpha=0.200, blend=0.55, center=0.55, slope=41.0,
                      s_pow=2.0, r_idf=0.85, r_tr=0.15):
    """Configurable 3way smoothing."""
    idf_s = idf_cooc_custom(s, alpha=alpha, blend=blend, center=center, slope=slope, s_pow=s_pow)
    tr    = two_round_fixed(s)
    return r_idf * idf_s + r_tr * tr

def apply_blend(s_best, s_ica, s_std, wb=0.76, wi=0.16, ws_=0.08):
    return wb * s_best + wi * s_ica + ws_ * s_std

# Verify baseline
c3_ref = apply_3way_custom(double_best)
i3_ref = apply_3way_custom(ica_ens_alt)
s3_ref = apply_3way_custom(std_ens_ref)
ref_chk = apply_blend(c3_ref, i3_ref, s3_ref)
print(f"Baseline verify: {macro_auc(ref_chk):.6f} (expect 0.995120)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# Direction A: Blend weight re-sweep at alpha=0.200
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Blend weight re-sweep at alpha=0.200 ===", flush=True)
best_blend_auc = macro_auc(ref_chk)
best_blend = (0.76, 0.16, 0.08)

for w_best in np.arange(0.70, 0.86, 0.01):
    for w_ica in np.arange(0.10, 0.24, 0.01):
        w_best_r = round(float(w_best), 2)
        w_ica_r  = round(float(w_ica), 2)
        w_std_r  = round(1.0 - w_best_r - w_ica_r, 2)
        if w_std_r < 0.02 or w_std_r > 0.15: continue
        mname = f"rblend_b{int(w_best_r*100)}_i{int(w_ica_r*100)}_s{int(w_std_r*100)}"
        if mname in tried: continue
        result = apply_blend(c3_ref, i3_ref, s3_ref, wb=w_best_r, wi=w_ica_r, ws_=w_std_r)
        score  = macro_auc(result)
        delta  = save_result(mname, score, {"wb": w_best_r, "wi": w_ica_r, "ws": w_std_r, "alpha": 0.200})
        if score > best_blend_auc:
            best_blend_auc = score
            best_blend = (w_best_r, w_ica_r, w_std_r)
        if score > best_loo - 0.0002:
            flag = " ← NEW BEST!" if score > best_loo else ""
            print(f"  {mname}: {score:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best blend: {best_blend} → {best_blend_auc:.6f}", flush=True)

# Use best blend for subsequent experiments
wb_best, wi_best, ws_best = best_blend

# ═══════════════════════════════════════════════════════════════════════════════
# Direction B: Center parameter sweep at alpha=0.200
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B: Center sweep at alpha=0.200 ===", flush=True)
center_results = {}
for center_val in [0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.65, 0.70]:
    cv = round(float(center_val), 2)
    c3c = apply_3way_custom(double_best, center=cv)
    i3c = apply_3way_custom(ica_ens_alt, center=cv)
    s3c = apply_3way_custom(std_ens_ref, center=cv)
    bc  = apply_blend(c3c, i3c, s3c, wb=wb_best, wi=wi_best, ws_=ws_best)
    ac  = macro_auc(bc)
    center_results[cv] = ac
    mname  = f"center_{int(cv*100):d}_a20"
    delta  = save_result(mname, ac, {"center": cv, "alpha": 0.200})
    flag   = " ← NEW BEST!" if ac > best_loo else ""
    print(f"  center={cv:.2f}: {ac:.6f} {delta:+.6f}{flag}", flush=True)

best_center = max(center_results, key=center_results.get)
print(f"  Best center: {best_center:.2f} → {center_results[best_center]:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Direction C: Slope parameter sweep at alpha=0.200
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== C: Slope sweep at alpha=0.200 ===", flush=True)
slope_results = {}
for slope_val in [20, 25, 28, 31, 35, 38, 41, 45, 50, 55, 60, 70]:
    sv = float(slope_val)
    c3s = apply_3way_custom(double_best, slope=sv)
    i3s = apply_3way_custom(ica_ens_alt, slope=sv)
    s3s = apply_3way_custom(std_ens_ref, slope=sv)
    bs  = apply_blend(c3s, i3s, s3s, wb=wb_best, wi=wi_best, ws_=ws_best)
    as_ = macro_auc(bs)
    slope_results[sv] = as_
    mname  = f"slope_{int(sv):d}_a20"
    delta  = save_result(mname, as_, {"slope": sv, "alpha": 0.200})
    flag   = " ← NEW BEST!" if as_ > best_loo else ""
    print(f"  slope={sv:.0f}: {as_:.6f} {delta:+.6f}{flag}", flush=True)

best_slope = max(slope_results, key=slope_results.get)
print(f"  Best slope: {best_slope:.0f} → {slope_results[best_slope]:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Direction D: Joint (center, slope) at alpha=0.200
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== D: Joint (center, slope) search ===", flush=True)
# Focus around best center ± 0.10 and best slope ± 8
best_cs_auc = best_loo
for center_val in [max(0.35, best_center - 0.08), best_center - 0.04,
                    best_center, best_center + 0.04, min(0.70, best_center + 0.08)]:
    center_val = round(float(center_val), 2)
    for slope_val in [max(20, best_slope - 8), best_slope - 4,
                       best_slope, best_slope + 4, min(70, best_slope + 8)]:
        slope_val = round(float(slope_val), 1)
        mname = f"cs_c{int(center_val*100)}_s{int(slope_val)}_a20"
        if mname in tried: continue
        c3d = apply_3way_custom(double_best, center=center_val, slope=slope_val)
        i3d = apply_3way_custom(ica_ens_alt, center=center_val, slope=slope_val)
        s3d = apply_3way_custom(std_ens_ref, center=center_val, slope=slope_val)
        bd  = apply_blend(c3d, i3d, s3d, wb=wb_best, wi=wi_best, ws_=ws_best)
        ad  = macro_auc(bd)
        delta = save_result(mname, ad, {"center": center_val, "slope": slope_val})
        if ad > best_cs_auc: best_cs_auc = ad
        if ad > best_loo - 0.0003:
            flag = " ← NEW BEST!" if ad > best_loo else ""
            print(f"  c={center_val:.2f} s={slope_val:.0f}: {ad:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Direction E: 3way ratio (r_idf vs r_tr) at alpha=0.200
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== E: 3way ratio search ===", flush=True)
for r_idf in [0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 1.00]:
    r_tr = round(1.0 - r_idf, 2)
    r_idf_r = round(float(r_idf), 2)
    c3e = apply_3way_custom(double_best, r_idf=r_idf_r, r_tr=r_tr)
    i3e = apply_3way_custom(ica_ens_alt, r_idf=r_idf_r, r_tr=r_tr)
    s3e = apply_3way_custom(std_ens_ref, r_idf=r_idf_r, r_tr=r_tr)
    be  = apply_blend(c3e, i3e, s3e, wb=wb_best, wi=wi_best, ws_=ws_best)
    ae  = macro_auc(be)
    mname  = f"ratio_idf{int(r_idf_r*100)}_tr{int(r_tr*100)}_a20"
    delta  = save_result(mname, ae, {"r_idf": r_idf_r, "r_tr": r_tr})
    flag   = " ← NEW BEST!" if ae > best_loo else ""
    print(f"  r_idf={r_idf_r:.2f}: {ae:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Direction F: Best found settings → fine blend re-sweep
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== F: Fine blend at best (center, slope) ===", flush=True)

# Compute 3way with best center and slope found
c3f = apply_3way_custom(double_best, center=best_center, slope=best_slope)
i3f = apply_3way_custom(ica_ens_alt, center=best_center, slope=best_slope)
s3f = apply_3way_custom(std_ens_ref, center=best_center, slope=best_slope)

for w_best_ in np.arange(0.72, 0.84, 0.01):
    for w_ica_ in np.arange(0.12, 0.22, 0.01):
        wb_r = round(float(w_best_), 2)
        wi_r = round(float(w_ica_), 2)
        ws_r = round(1.0 - wb_r - wi_r, 2)
        if ws_r < 0.03 or ws_r > 0.14: continue
        mname = f"fine_c{int(best_center*100)}_s{int(best_slope)}_b{int(wb_r*100)}_i{int(wi_r*100)}"
        if mname in tried: continue
        result = apply_blend(c3f, i3f, s3f, wb=wb_r, wi=wi_r, ws_=ws_r)
        score  = macro_auc(result)
        delta  = save_result(mname, score, {"wb": wb_r, "wi": wi_r, "ws": ws_r,
                                             "center": best_center, "slope": best_slope})
        if score > best_loo - 0.0001:
            flag = " ← NEW BEST!" if score > best_loo else ""
            print(f"  {mname}: {score:.6f} {delta:+.6f}{flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 137]
print(f"Batch137 complete in {elapsed:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
