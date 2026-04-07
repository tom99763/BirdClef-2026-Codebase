"""
batch142 — Per-component alpha extended search + blend re-optimization
===============================================================================
Current best: pcalpha_b200_i225_s225 LOO=0.995235 (+0.000049)
  a_best=0.200, a_ica=0.225, a_std=0.225
  Final blend: 0.75×3way_best + 0.23×3way_ica_alt + 0.02×3way_std

batch141 key findings:
- Per-component alpha discovered: a_ica=0.225, a_std=0.225 > 0.200 (all same)
- a_best=0.200 still optimal (0.205 slightly worse)
- Symmetric COOC much worse
- slope=41 still optimal

Directions:
 A: Extended a_ica range (0.225-0.350 step 0.025)
 B: Extended a_std range (0.225-0.350 step 0.025)
 C: Joint fine-tune (a_ica: 0.210-0.260, a_std: 0.210-0.280) at a_best=0.200
 D: Blend re-optimize at new best alpha settings
 E: blend param (0.55) re-tune with per-component alpha
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
print(f"[batch142] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 142}
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

def blend3(c, i, s, wb=0.75, wi=0.23, ws=0.02):
    return wb*c + wi*i + ws*s

# Verify
c3_b = apply_3way(double_best, alpha=0.200)
i3_b = apply_3way(ica_ens_alt, alpha=0.225)
s3_b = apply_3way(std_ens_ref,  alpha=0.225)
chk = blend3(c3_b, i3_b, s3_b)
print(f"Verify best: {macro_auc(chk):.6f} (expect 0.995235)\n", flush=True)

t0 = time.time()

# Pre-compute 3way at many alpha values
print("Pre-computing 3way at various alphas...", flush=True)
alpha_range = [round(x/1000, 3) for x in range(175, 380, 25)]  # 0.175-0.375 step 0.025
cache_double = {av: apply_3way(double_best, alpha=av) for av in alpha_range}
cache_ica    = {av: apply_3way(ica_ens_alt, alpha=av) for av in alpha_range}
cache_std    = {av: apply_3way(std_ens_ref,  alpha=av) for av in alpha_range}
print(f"  Cached {len(alpha_range)} alpha values: {alpha_range}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# A: Extended a_ica range (a_best=0.200 fixed)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== A: Extended a_ica range (a_best=0.200) ===", flush=True)
c3_fixed = cache_double[0.200]
best_a_ica = best_loo
for a_ica in alpha_range:
    i3 = cache_ica[a_ica]
    for a_std in [0.200, 0.225, 0.250]:
        s3 = cache_std[a_std]
        ar = macro_auc(blend3(c3_fixed, i3, s3))
        mname = f"pcalpha_b200_i{int(a_ica*1000)}_s{int(a_std*1000)}"
        delta = save_result(mname, ar, {"a_best": 0.200, "a_ica": a_ica, "a_std": a_std})
        if ar > best_a_ica:
            best_a_ica = ar
        if ar > best_loo - 0.0001:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_ica={a_ica:.3f} a_std={a_std:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Fine joint grid (a_ica: 0.210-0.260 step 0.005, a_std: 0.210-0.280 step 0.010)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Fine joint grid a_ica × a_std (a_best=0.200) ===", flush=True)
fine_alpha = [round(x/1000, 3) for x in range(200, 301, 5)]  # 0.200-0.300 step 0.005
cache_ica_fine = {av: apply_3way(ica_ens_alt, alpha=av) for av in fine_alpha if av not in cache_ica}
cache_std_fine = {av: apply_3way(std_ens_ref, alpha=av) for av in fine_alpha if av not in cache_std}
cache_ica.update(cache_ica_fine); cache_std.update(cache_std_fine)

best_joint = best_loo
best_joint_params = (0.225, 0.225)
for a_ica in fine_alpha:
    i3 = cache_ica[a_ica]
    for a_std in fine_alpha:
        s3 = cache_std[a_std]
        ar = macro_auc(blend3(c3_fixed, i3, s3))
        mname = f"pcfine_b200_i{int(a_ica*1000)}_s{int(a_std*1000)}"
        delta = save_result(mname, ar, {"a_best": 0.200, "a_ica": a_ica, "a_std": a_std})
        if ar > best_joint:
            best_joint = ar
            best_joint_params = (a_ica, a_std)
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_ica={a_ica:.3f} a_std={a_std:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

best_a_ica_opt, best_a_std_opt = best_joint_params
print(f"  Best joint: a_ica={best_a_ica_opt:.3f} a_std={best_a_std_opt:.3f} → {best_joint:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Blend re-optimize at best per-component alpha
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Blend re-optimize at a_best=0.200, a_ica={best_a_ica_opt:.3f}, a_std={best_a_std_opt:.3f} ===", flush=True)
c3_c = cache_double[0.200]
i3_c = cache_ica[best_a_ica_opt]
s3_c = cache_std[best_a_std_opt]

best_blend_c = best_loo
best_blend_params = (0.75, 0.23, 0.02)
for ws in [0.00, 0.01, 0.02, 0.03]:
    for wi in np.arange(0.17, 0.30, 0.01):
        wi_r = round(float(wi), 2); ws_r = round(float(ws), 2)
        wb_r = round(1.0 - wi_r - ws_r, 2)
        if wb_r < 0.68 or wb_r > 0.84: continue
        br = blend3(c3_c, i3_c, s3_c, wb=wb_r, wi=wi_r, ws=ws_r)
        ar = macro_auc(br)
        mname = f"pcblend_b{int(wb_r*100)}_i{int(wi_r*100)}_s{int(ws_r*100)}_ai{int(best_a_ica_opt*1000)}"
        delta = save_result(mname, ar, {"wb": wb_r, "wi": wi_r, "ws": ws_r,
                                         "a_ica": best_a_ica_opt, "a_std": best_a_std_opt})
        if ar > best_blend_c:
            best_blend_c = ar
            best_blend_params = (wb_r, wi_r, ws_r)
        if ar > best_loo - 0.00008:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  wb={wb_r:.2f} wi={wi_r:.2f} ws={ws_r:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best blend: {best_blend_params} → {best_blend_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: blend param (0.55) re-tune with per-component alpha
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: idf_cooc blend param re-tune with per-comp alpha ===", flush=True)
wb_d, wi_d, ws_d = best_blend_params
for blend_p in [0.50, 0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60]:
    bp = round(float(blend_p), 2)
    c3d = apply_3way(double_best, alpha=0.200, blend=bp)
    i3d = apply_3way(ica_ens_alt, alpha=best_a_ica_opt, blend=bp)
    s3d = apply_3way(std_ens_ref,  alpha=best_a_std_opt, blend=bp)
    ar  = macro_auc(blend3(c3d, i3d, s3d, wb=wb_d, wi=wi_d, ws=ws_d))
    mname = f"pcblend_blp{int(bp*100)}_ai{int(best_a_ica_opt*1000)}_as{int(best_a_std_opt*1000)}"
    delta = save_result(mname, ar, {"blend": bp, "a_ica": best_a_ica_opt, "a_std": best_a_std_opt})
    flag  = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  blend={bp:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 142]
print(f"Batch142 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
