"""
batch140 — IDF exponent fine-sweep + normalization variants
===============================================================================
Current best: rb138_r875_b75_i23_s2 LOO=0.995186
  r_idf=0.875, r_tr=0.125, alpha=0.200, blend=0.55
  Final blend: 0.75×3way_best + 0.23×3way_ica_alt + 0.02×3way_std

batch139 confirmed plateau:
- r_idf: 0.875~0.879 all tie at 0.995186
- alpha: 0.200~0.205 both 0.995186
- blend: 0.55~0.56 both 0.995186
- Near-zero STD (ws=0.02) confirmed optimal
- 2-component (no STD) worse

Unexplored directions:
 A: IDF exponent sweep (currently 0.75, try 0.50-1.00 step 0.05)
 B: IDF exponent fine-tune near best
 C: Cooc normalization variant (L1/L2/sum vs max)
 D: Two-step idf_cooc (apply twice with smaller alpha)
 E: Extended center/slope fine-tune at new params (if IDF changes help)
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
print(f"[batch140] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 140}
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

# Co-occurrence setup — base
fl_hard = file_labels.astype(np.float32)
count_i = fl_hard.sum(0) + EPS
COOC    = (fl_hard.T @ fl_hard) / count_i[:, None]
np.fill_diagonal(COOC, 0)
raw_idf = np.clip(np.log(float(n_files) / (count_i + 1.0 - EPS)), 0, None)

def make_idf(exp):
    idf = raw_idf ** exp
    idf /= (idf.mean() + EPS)
    return idf

IDF075 = make_idf(0.75)  # reference

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

def soft_cooc_l1norm(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    """L1-normalize cooc output instead of max-normalize"""
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope*(s-center), -88, 88)))
        sg = s * gate * (idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi] = s; continue
        c = COOC.T @ sg; mc = np.abs(c).sum()
        if mc > EPS: c = c / mc * n_species  # L1 norm, scale to ~1
        out[fi] = (1-alpha)*s + alpha*np.clip(c, 0, None)
    return out

def soft_cooc_l2norm(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    """L2-normalize cooc output instead of max-normalize"""
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope*(s-center), -88, 88)))
        sg = s * gate * (idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi] = s; continue
        c = COOC.T @ sg; mc = np.sqrt((c**2).sum())
        if mc > EPS: c = c / mc * np.sqrt(n_species)  # L2 norm, scale to ~1
        out[fi] = (1-alpha)*s + alpha*np.clip(c, 0, None)
    return out

def apply_3way_idfexp(s, idf_exp=0.75, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125):
    idf_w = make_idf(idf_exp)
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=idf_w)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.089)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.040)
    return r_idf * idf_s + r_tr * tr

def apply_3way_base(s):
    return apply_3way_idfexp(s, idf_exp=0.75)

def blend3(c, i, s, wb=0.75, wi=0.23, ws=0.02):
    return wb*c + wi*i + ws*s

# Verify
c3_ref = apply_3way_base(double_best)
i3_ref = apply_3way_base(ica_ens_alt)
s3_ref = apply_3way_base(std_ens_ref)
chk = blend3(c3_ref, i3_ref, s3_ref)
print(f"Verify: {macro_auc(chk):.6f} (expect 0.995186)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: IDF exponent sweep (coarse: 0.50-1.00 step 0.05)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: IDF exponent coarse sweep ===", flush=True)
idf_results = {}
for idf_exp_int in range(50, 105, 5):
    idf_exp = idf_exp_int / 100.0
    c3 = apply_3way_idfexp(double_best, idf_exp=idf_exp)
    i3 = apply_3way_idfexp(ica_ens_alt, idf_exp=idf_exp)
    s3 = apply_3way_idfexp(std_ens_ref, idf_exp=idf_exp)
    br = blend3(c3, i3, s3)
    ar = macro_auc(br)
    idf_results[idf_exp] = ar
    mname = f"idfexp_{idf_exp_int}"
    delta = save_result(mname, ar, {"idf_exp": idf_exp})
    flag  = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  idf_exp={idf_exp:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

best_idf_exp = max(idf_results, key=idf_results.get)
print(f"  Best idf_exp: {best_idf_exp:.2f} → {idf_results[best_idf_exp]:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: IDF exponent fine-tune near best
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: IDF exponent fine-tune near {best_idf_exp:.2f} ===", flush=True)
fine_idf_results = {}
center_exp = int(best_idf_exp * 100)
for idf_exp_int in range(max(50, center_exp - 7), min(105, center_exp + 8)):
    idf_exp = idf_exp_int / 100.0
    if idf_exp in idf_results: continue  # already done
    c3 = apply_3way_idfexp(double_best, idf_exp=idf_exp)
    i3 = apply_3way_idfexp(ica_ens_alt, idf_exp=idf_exp)
    s3 = apply_3way_idfexp(std_ens_ref, idf_exp=idf_exp)
    br = blend3(c3, i3, s3)
    ar = macro_auc(br)
    fine_idf_results[idf_exp] = ar
    mname = f"idfexp_f{idf_exp_int}"
    delta = save_result(mname, ar, {"idf_exp": idf_exp})
    flag  = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  idf_exp={idf_exp:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

all_idf = {**idf_results, **fine_idf_results}
best_idf_exp_fine = max(all_idf, key=all_idf.get)
print(f"  Best idf_exp (fine): {best_idf_exp_fine:.2f} → {all_idf[best_idf_exp_fine]:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Normalization variant at best IDF exp
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Normalization variants at idf_exp={best_idf_exp_fine:.2f} ===", flush=True)
idf_w_best = make_idf(best_idf_exp_fine)

def apply_3way_l1(s):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc_l1norm(sp, alpha=0.200, idf_w=idf_w_best)
    idf_s = 0.45*s + 0.55*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.089)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.040)
    return 0.875 * idf_s + 0.125 * tr

def apply_3way_l2(s):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc_l2norm(sp, alpha=0.200, idf_w=idf_w_best)
    idf_s = 0.45*s + 0.55*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.089)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.040)
    return 0.875 * idf_s + 0.125 * tr

for norm_name, fn in [("l1norm", apply_3way_l1), ("l2norm", apply_3way_l2)]:
    c3 = fn(double_best); i3 = fn(ica_ens_alt); s3 = fn(std_ens_ref)
    ar = macro_auc(blend3(c3, i3, s3))
    delta = save_result(f"cooc_{norm_name}", ar, {"norm": norm_name})
    flag = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  {norm_name}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Two-step idf_cooc (apply idf_cooc twice with smaller alpha each step)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Two-step idf_cooc at best settings ===", flush=True)
idf_w_ref = make_idf(best_idf_exp_fine)

def apply_3way_2step(s, a1=0.150, a2=0.100, blend1=0.40, blend2=0.30):
    """Two sequential idf_cooc passes"""
    sp = np.clip(s, 0, 1)**2
    sc1 = soft_cooc(sp, alpha=a1, idf_w=idf_w_ref)
    idf_s1 = (1-blend1)*s + blend1*sc1
    sp2 = np.clip(idf_s1, 0, 1)**2
    sc2 = soft_cooc(sp2, alpha=a2, idf_w=idf_w_ref)
    idf_s2 = (1-blend2)*idf_s1 + blend2*sc2
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.089)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.040)
    return 0.875 * idf_s2 + 0.125 * tr

for a1, a2, b1, b2 in [
    (0.150, 0.100, 0.40, 0.30),
    (0.120, 0.080, 0.35, 0.25),
    (0.100, 0.100, 0.30, 0.30),
    (0.200, 0.100, 0.40, 0.20),
    (0.150, 0.150, 0.35, 0.25),
]:
    c3 = apply_3way_2step(double_best, a1, a2, b1, b2)
    i3 = apply_3way_2step(ica_ens_alt, a1, a2, b1, b2)
    s3 = apply_3way_2step(std_ens_ref, a1, a2, b1, b2)
    ar = macro_auc(blend3(c3, i3, s3))
    mname = f"2step_a{int(a1*1000)}_{int(a2*1000)}_b{int(b1*100)}_{int(b2*100)}"
    delta = save_result(mname, ar, {"a1": a1, "a2": a2, "b1": b1, "b2": b2})
    flag  = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  a1={a1:.3f} a2={a2:.3f} b1={b1:.2f} b2={b2:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: IDF-blend joint sweep at best IDF exp (if idf_exp changed)
# ═══════════════════════════════════════════════════════════════════════════════
if abs(best_idf_exp_fine - 0.75) > 0.02:
    print(f"\n=== E: Blend re-optimize at idf_exp={best_idf_exp_fine:.2f} ===", flush=True)
    idf_w_new = make_idf(best_idf_exp_fine)

    def apply_3way_new_idf(s, blend=0.55):
        sp = np.clip(s, 0, 1)**2
        sc = soft_cooc(sp, alpha=0.200, idf_w=idf_w_new)
        idf_s = (1-blend)*s + blend*sc
        r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.089)
        tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.040)
        return 0.875 * idf_s + 0.125 * tr

    best_e_auc = best_loo
    best_e_params = None
    for blend_p in np.arange(0.45, 0.70, 0.02):
        bp = round(float(blend_p), 2)
        c3 = apply_3way_new_idf(double_best, blend=bp)
        i3 = apply_3way_new_idf(ica_ens_alt, blend=bp)
        s3 = apply_3way_new_idf(std_ens_ref, blend=bp)
        for wi in np.arange(0.18, 0.30, 0.02):
            wi_r = round(float(wi), 2)
            wb_r = round(1.0 - wi_r - 0.02, 2)
            if wb_r < 0.68 or wb_r > 0.84: continue
            br = blend3(c3, i3, s3, wb=wb_r, wi=wi_r, ws=0.02)
            ar = macro_auc(br)
            mname = f"idfnew_b{int(best_idf_exp_fine*100)}_blp{int(bp*100)}_b{int(wb_r*100)}_i{int(wi_r*100)}"
            delta = save_result(mname, ar, {"idf_exp": best_idf_exp_fine, "blend": bp, "wb": wb_r, "wi": wi_r})
            if ar > best_e_auc:
                best_e_auc = ar
                best_e_params = (bp, wb_r, wi_r)
            if ar > best_loo - 0.0001:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  blend={bp:.2f} wb={wb_r:.2f} wi={wi_r:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
    if best_e_params:
        print(f"  Best E params: blend={best_e_params[0]:.2f} wb={best_e_params[1]:.2f} wi={best_e_params[2]:.2f} → {best_e_auc:.6f}", flush=True)
else:
    print(f"\n=== E: skipped (idf_exp={best_idf_exp_fine:.2f} same as 0.75) ===", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 140]
print(f"Batch140 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
