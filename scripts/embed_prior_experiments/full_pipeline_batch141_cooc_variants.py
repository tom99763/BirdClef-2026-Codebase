"""
batch141 — Symmetric COOC + per-component alpha + two_round fine-tune
===============================================================================
Current best: rb138_r875_b75_i23_s2 LOO=0.995186
  r_idf=0.875, r_tr=0.125, alpha=0.200, blend=0.55
  Final blend: 0.75×3way_best + 0.23×3way_ica_alt + 0.02×3way_std

batch140 confirmed:
- IDF exponent 0.74-0.78 all tie at 0.995186 (0.75 optimal)
- L1/L2 normalization much worse
- Two-step idf_cooc worse

Unexplored directions:
 A: Symmetric COOC (sqrt normalization: P(i,j)/sqrt(P(i)P(j)) — Jaccard-like)
 B: Per-component alpha (different alpha for each of 3 blend sources)
 C: two_round param fine-tune (alpha1, alpha2, center1, center2, slope1, slope2)
 D: idf_cooc slope fine-tune (38-44, step 1)
 E: Per-component blend fine sweep at component level
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
print(f"[batch141] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 141}
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

# Standard co-occurrence
fl_hard = file_labels.astype(np.float32)
count_i = fl_hard.sum(0) + EPS
COOC    = (fl_hard.T @ fl_hard) / count_i[:, None]  # asymmetric: P(j|i)
np.fill_diagonal(COOC, 0)

# Symmetric COOC (sqrt normalization: PMI-like)
outer_sqrt = np.sqrt(count_i[:, None] * count_i[None, :]) + EPS
COOC_SYM   = (fl_hard.T @ fl_hard) / outer_sqrt
np.fill_diagonal(COOC_SYM, 0)
COOC_SYM  /= (COOC_SYM.max() + EPS)  # normalize to [0,1]

raw_idf = np.clip(np.log(float(n_files) / (count_i + 1.0 - EPS)), 0, None)
IDF075  = raw_idf ** 0.75; IDF075 /= (IDF075.mean() + EPS)

def soft_cooc_std(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    """Standard (asymmetric) co-occurrence"""
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

def soft_cooc_sym(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    """Symmetric co-occurrence"""
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope*(s-center), -88, 88)))
        sg = s * gate * (idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi] = s; continue
        c = COOC_SYM.T @ sg; mc = np.abs(c).max()
        if mc > EPS: c /= mc
        out[fi] = (1-alpha)*s + alpha*np.clip(c, 0, None)
    return out

soft_cooc = soft_cooc_std  # alias

def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, slope=41.0):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc_std(sp, alpha=alpha, idf_w=IDF075, slope=slope)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc_std(s, center=0.54, slope=41.0, alpha=0.089)
    tr = soft_cooc_std(r1, center=0.53, slope=37.0, alpha=0.040)
    return r_idf * idf_s + r_tr * tr

def blend3(c, i, s, wb=0.75, wi=0.23, ws=0.02):
    return wb*c + wi*i + ws*s

# Verify
c3_ref = apply_3way(double_best); i3_ref = apply_3way(ica_ens_alt); s3_ref = apply_3way(std_ens_ref)
chk = blend3(c3_ref, i3_ref, s3_ref)
print(f"Verify: {macro_auc(chk):.6f} (expect 0.995186)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Symmetric COOC variants
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Symmetric COOC variants ===", flush=True)

def apply_3way_sym(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc_sym(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc_std(s, center=0.54, slope=41.0, alpha=0.089)
    tr = soft_cooc_std(r1, center=0.53, slope=37.0, alpha=0.040)
    return r_idf * idf_s + r_tr * tr

def apply_3way_sym_both(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc_sym(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc_sym(s, center=0.54, slope=41.0, alpha=0.089)
    tr = soft_cooc_sym(r1, center=0.53, slope=37.0, alpha=0.040)
    return r_idf * idf_s + r_tr * tr

for sym_name, fn in [("sym_idf_only", apply_3way_sym), ("sym_both", apply_3way_sym_both)]:
    c3 = fn(double_best); i3 = fn(ica_ens_alt); s3 = fn(std_ens_ref)
    ar = macro_auc(blend3(c3, i3, s3))
    delta = save_result(f"cooc_{sym_name}", ar, {"cooc": sym_name})
    flag = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  {sym_name}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Blend of symmetric + asymmetric
for sym_w in [0.2, 0.3, 0.4, 0.5]:
    def apply_3way_blend_sym(s, sw=sym_w):
        sp = np.clip(s, 0, 1)**2
        sc_asym = soft_cooc_std(sp, alpha=0.200, idf_w=IDF075)
        sc_sym  = soft_cooc_sym(sp, alpha=0.200, idf_w=IDF075)
        sc = (1-sw)*sc_asym + sw*sc_sym
        idf_s = 0.45*s + 0.55*sc
        r1 = soft_cooc_std(s, center=0.54, slope=41.0, alpha=0.089)
        tr = soft_cooc_std(r1, center=0.53, slope=37.0, alpha=0.040)
        return 0.875 * idf_s + 0.125 * tr
    c3 = apply_3way_blend_sym(double_best); i3 = apply_3way_blend_sym(ica_ens_alt); s3 = apply_3way_blend_sym(std_ens_ref)
    ar = macro_auc(blend3(c3, i3, s3))
    mname = f"cooc_symmix_{int(sym_w*10)}"
    delta = save_result(mname, ar, {"sym_w": sym_w})
    flag = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  sym_mix={sym_w:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Per-component alpha (different alpha for each of 3 blend sources)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Per-component alpha ===", flush=True)

# Pre-compute at different alphas
alpha_vals = [0.150, 0.175, 0.200, 0.205, 0.225, 0.250]
cached = {}
for av in alpha_vals:
    cached[("double", av)] = apply_3way(double_best, alpha=av)
    cached[("ica",    av)] = apply_3way(ica_ens_alt, alpha=av)
    cached[("std",    av)] = apply_3way(std_ens_ref,  alpha=av)

best_percomp_auc = best_loo
for a_b in [0.175, 0.200, 0.205]:
    for a_i in [0.175, 0.200, 0.205, 0.225]:
        for a_s in [0.150, 0.175, 0.200, 0.225, 0.250]:
            if a_b == a_i == a_s == 0.200: continue  # already done
            c3 = cached.get(("double", a_b))
            i3 = cached.get(("ica",    a_i))
            s3 = cached.get(("std",    a_s))
            if c3 is None or i3 is None or s3 is None: continue
            ar = macro_auc(blend3(c3, i3, s3))
            mname = f"pcalpha_b{int(a_b*1000)}_i{int(a_i*1000)}_s{int(a_s*1000)}"
            delta = save_result(mname, ar, {"a_best": a_b, "a_ica": a_i, "a_std": a_s})
            if ar > best_percomp_auc:
                best_percomp_auc = ar
            if ar > best_loo - 0.0001:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  ab={a_b:.3f} ai={a_i:.3f} as={a_s:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best per-component alpha: {best_percomp_auc:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: two_round parameter fine-tune
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: two_round param fine-tune ===", flush=True)

def apply_3way_tr(s, c1=0.54, s1=41.0, a1=0.089, c2=0.53, s2=37.0, a2=0.040):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc_std(sp, alpha=0.200, idf_w=IDF075)
    idf_s = 0.45*s + 0.55*sc
    r1 = soft_cooc_std(s, center=c1, slope=s1, alpha=a1)
    tr = soft_cooc_std(r1, center=c2, slope=s2, alpha=a2)
    return 0.875 * idf_s + 0.125 * tr

# Vary alpha1 (round 1 alpha)
best_tr_auc = best_loo
for a1 in [0.060, 0.070, 0.080, 0.089, 0.100, 0.110, 0.120]:
    for a2 in [0.020, 0.030, 0.040, 0.050, 0.060]:
        c3 = apply_3way_tr(double_best, a1=a1, a2=a2)
        i3 = apply_3way_tr(ica_ens_alt, a1=a1, a2=a2)
        s3 = apply_3way_tr(std_ens_ref, a1=a1, a2=a2)
        ar = macro_auc(blend3(c3, i3, s3))
        mname = f"tr_a1_{int(a1*1000)}_a2_{int(a2*1000)}"
        delta = save_result(mname, ar, {"a1": a1, "a2": a2})
        if ar > best_tr_auc:
            best_tr_auc = ar
        if ar > best_loo - 0.00006:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a1={a1:.3f} a2={a2:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best two_round alpha: {best_tr_auc:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: idf_cooc slope fine-tune (38-44, step 1)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: idf_cooc slope fine-tune ===", flush=True)
for slope in range(38, 45):
    sl = float(slope)
    c3 = apply_3way(double_best, slope=sl)
    i3 = apply_3way(ica_ens_alt, slope=sl)
    s3 = apply_3way(std_ens_ref, slope=sl)
    ar = macro_auc(blend3(c3, i3, s3))
    mname = f"slope_{slope}"
    delta = save_result(mname, ar, {"slope": sl})
    flag = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  slope={slope}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 141]
print(f"Batch141 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
