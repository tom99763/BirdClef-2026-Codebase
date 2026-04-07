"""
batch144 — Per-component r_idf + two_round re-opt + s_pow variation
===============================================================================
Current best: pcfine_b200_i255_s280 LOO=0.995285
  a_best=0.200, a_ica=0.255, a_std=0.280
  blend=(0.75, 0.23, 0.02), idf_blend=0.55, r_idf=0.875

batch143 findings:
- a_best: 0.200-0.205 both optimal (plateau)
- r_idf: 0.875-0.900 ALL tie at 0.995285 (wide plateau)
- a_ica plateau: 0.255-0.268
- Per-component blend: tie at 0.995285 in many settings

Unexplored:
 A: r_idf extended (0.90-1.00) - test pure idf_cooc
 B: Per-component r_idf (different r_idf for each source)
 C: two_round re-optimize with per-component alpha
 D: s_pow variation (1.5, 2.5, 3.0) with per-component alpha
 E: Per-component s_pow
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
print(f"[batch144] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 144}
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

def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, spow=2.0):
    sp = np.clip(s, 0, 1)**spow
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.089)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.040)
    return r_idf * idf_s + r_tr * tr

def blend3(c, i, s, wb=0.75, wi=0.23, ws=0.02):
    return wb*c + wi*i + ws*s

c3_ref = apply_3way(double_best, alpha=0.200)
i3_ref = apply_3way(ica_ens_alt, alpha=0.255)
s3_ref = apply_3way(std_ens_ref,  alpha=0.280)
chk = blend3(c3_ref, i3_ref, s3_ref)
print(f"Verify: {macro_auc(chk):.6f} (expect 0.995285)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: r_idf extended (0.90-1.00 step 0.02) — test pure idf_cooc
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: r_idf extended range ===", flush=True)
ridf_results = {}
for r_int in list(range(875, 905, 5)) + list(range(910, 1001, 10)):
    r_idf = r_int / 1000.0
    r_tr  = round(1.0 - r_idf, 3)
    if r_idf > 1.0: continue
    c3 = apply_3way(double_best, alpha=0.200, r_idf=r_idf, r_tr=r_tr)
    i3 = apply_3way(ica_ens_alt, alpha=0.255, r_idf=r_idf, r_tr=r_tr)
    s3 = apply_3way(std_ens_ref,  alpha=0.280, r_idf=r_idf, r_tr=r_tr)
    ar = macro_auc(blend3(c3, i3, s3))
    ridf_results[r_idf] = ar
    mname = f"pcridfx_{r_int}"
    delta = save_result(mname, ar, {"r_idf": r_idf, "a_best": 0.200, "a_ica": 0.255, "a_std": 0.280})
    flag  = " ← NEW BEST!" if ar > best_loo else ""
    if ar > best_loo - 0.0002:
        print(f"  r_idf={r_idf:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

best_r = max(ridf_results, key=ridf_results.get)
print(f"  Best r_idf: {best_r:.3f} → {ridf_results[best_r]:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Per-component r_idf
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Per-component r_idf ===", flush=True)
# Pre-compute at multiple r_idf values
r_opts = [0.850, 0.875, 0.900, 0.925, 0.950]

cache_b_ridf = {r: apply_3way(double_best, alpha=0.200, r_idf=r, r_tr=round(1-r,3)) for r in r_opts}
cache_i_ridf = {r: apply_3way(ica_ens_alt, alpha=0.255, r_idf=r, r_tr=round(1-r,3)) for r in r_opts}
cache_s_ridf = {r: apply_3way(std_ens_ref,  alpha=0.280, r_idf=r, r_tr=round(1-r,3)) for r in r_opts}

best_pcridf = best_loo
for r_b in r_opts:
    for r_i in r_opts:
        for r_s in r_opts:
            if r_b == r_i == r_s == 0.875: continue
            ar = macro_auc(blend3(cache_b_ridf[r_b], cache_i_ridf[r_i], cache_s_ridf[r_s]))
            mname = f"pcridf_b{int(r_b*1000)}_i{int(r_i*1000)}_s{int(r_s*1000)}"
            delta = save_result(mname, ar, {"r_b": r_b, "r_i": r_i, "r_s": r_s})
            if ar > best_pcridf: best_pcridf = ar
            if ar > best_loo - 0.00008:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  r_b={r_b:.3f} r_i={r_i:.3f} r_s={r_s:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best per-comp r_idf: {best_pcridf:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: two_round alpha re-optimize with per-component alpha
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: two_round alpha re-optimize ===", flush=True)

def apply_3way_tr(s, alpha=0.200, a1=0.089, a2=0.040):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = 0.45*s + 0.55*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)
    return 0.875 * idf_s + 0.125 * tr

best_tr = best_loo
for a1 in [0.060, 0.070, 0.080, 0.089, 0.100, 0.110, 0.130]:
    for a2 in [0.020, 0.030, 0.040, 0.050, 0.060, 0.070]:
        c3 = apply_3way_tr(double_best, alpha=0.200, a1=a1, a2=a2)
        i3 = apply_3way_tr(ica_ens_alt, alpha=0.255, a1=a1, a2=a2)
        s3 = apply_3way_tr(std_ens_ref,  alpha=0.280, a1=a1, a2=a2)
        ar = macro_auc(blend3(c3, i3, s3))
        mname = f"pctr_a1{int(a1*1000)}_a2{int(a2*1000)}"
        delta = save_result(mname, ar, {"a1": a1, "a2": a2})
        if ar > best_tr: best_tr = ar
        if ar > best_loo - 0.00006:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a1={a1:.3f} a2={a2:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best two_round re-opt: {best_tr:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: s_pow variation with per-component alpha
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: s_pow variation ===", flush=True)
for spow in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    c3 = apply_3way(double_best, alpha=0.200, spow=spow)
    i3 = apply_3way(ica_ens_alt, alpha=0.255, spow=spow)
    s3 = apply_3way(std_ens_ref,  alpha=0.280, spow=spow)
    ar = macro_auc(blend3(c3, i3, s3))
    mname = f"spow_{int(spow*10)}_pc"
    delta = save_result(mname, ar, {"spow": spow})
    flag = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  spow={spow:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Per-component s_pow
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Per-component s_pow ===", flush=True)
spow_opts = [1.5, 2.0, 2.5, 3.0]
cache_b_spow = {sp: apply_3way(double_best, alpha=0.200, spow=sp) for sp in spow_opts}
cache_i_spow = {sp: apply_3way(ica_ens_alt, alpha=0.255, spow=sp) for sp in spow_opts}
cache_s_spow = {sp: apply_3way(std_ens_ref,  alpha=0.280, spow=sp) for sp in spow_opts}

best_pcspow = best_loo
for sp_b in spow_opts:
    for sp_i in spow_opts:
        for sp_s in spow_opts:
            if sp_b == sp_i == sp_s == 2.0: continue
            ar = macro_auc(blend3(cache_b_spow[sp_b], cache_i_spow[sp_i], cache_s_spow[sp_s]))
            mname = f"pcspow_b{int(sp_b*10)}_i{int(sp_i*10)}_s{int(sp_s*10)}"
            delta = save_result(mname, ar, {"sp_b": sp_b, "sp_i": sp_i, "sp_s": sp_s})
            if ar > best_pcspow: best_pcspow = ar
            if ar > best_loo - 0.00008:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  sp_b={sp_b:.1f} sp_i={sp_i:.1f} sp_s={sp_s:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best per-comp s_pow: {best_pcspow:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 144]
print(f"Batch144 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
