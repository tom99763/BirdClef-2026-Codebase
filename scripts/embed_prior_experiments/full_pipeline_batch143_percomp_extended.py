"""
batch143 — Per-component alpha: a_best fine-tune + r_idf re-tune + per-component blend
===============================================================================
Current best: pcfine_b200_i255_s280 LOO=0.995285 (+0.000050)
  a_best=0.200, a_ica=0.255, a_std=0.280
  blend=(0.75, 0.23, 0.02), idf_blend=0.55, r_idf=0.875

batch142 findings:
- a_ica plateau: 0.255-0.265 all tie at 0.995285
- a_std plateau: 0.280+ all tie at 0.995285 (but 0.280-0.255=insensitive)
- Blend still (0.75, 0.23, 0.02) optimal
- blend=0.55/0.56 optimal

Directions:
 A: Fine-tune a_best (0.185-0.215, step 0.005) with best a_ica/a_std
 B: r_idf re-tune (0.86-0.90) with per-component alpha
 C: Per-component idf_cooc blend param (different blend for each source)
 D: Ultra-fine a_ica (0.252-0.268 step 0.001) with best a_std
 E: Joint r_idf × a_best fine search
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
print(f"[batch143] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 143}
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

# Best reference from batch142
A_BEST_REF = 0.200
A_ICA_REF  = 0.255
A_STD_REF  = 0.280
c3_ref = apply_3way(double_best, alpha=A_BEST_REF)
i3_ref = apply_3way(ica_ens_alt, alpha=A_ICA_REF)
s3_ref = apply_3way(std_ens_ref,  alpha=A_STD_REF)
chk = blend3(c3_ref, i3_ref, s3_ref)
print(f"Verify: {macro_auc(chk):.6f} (expect 0.995285)\n", flush=True)

t0 = time.time()

# Pre-compute caches
print("Pre-computing caches...", flush=True)
a_best_range = [round(x/1000, 3) for x in range(185, 220, 5)]
cache_double = {av: apply_3way(double_best, alpha=av) for av in a_best_range}
# a_ica fine range
a_ica_fine = [round(x/1000, 3) for x in range(252, 270)]
cache_ica_fine = {av: apply_3way(ica_ens_alt, alpha=av) for av in a_ica_fine}
# fixed refs
cache_ica_fine[0.255] = i3_ref
cache_double[0.200]   = c3_ref
print(f"  Done ({len(cache_double)} a_best, {len(cache_ica_fine)} a_ica_fine)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# A: Fine-tune a_best (0.185-0.215) with a_ica=0.255, a_std=0.280
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== A: Fine-tune a_best ===", flush=True)
best_a_best = best_loo
for a_b in a_best_range:
    c3 = cache_double[a_b]
    ar = macro_auc(blend3(c3, i3_ref, s3_ref))
    mname = f"pcbest_ab{int(a_b*1000)}_ai255_as280"
    delta = save_result(mname, ar, {"a_best": a_b, "a_ica": 0.255, "a_std": 0.280})
    if ar > best_a_best: best_a_best = ar
    flag = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  a_best={a_b:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: r_idf re-tune (0.860-0.900 step 0.005) with per-component alpha
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: r_idf re-tune with per-comp alpha ===", flush=True)
best_ridf = best_loo
for r_int in range(860, 905, 5):
    r_idf = r_int / 1000.0
    r_tr  = round(1.0 - r_idf, 3)
    c3 = apply_3way(double_best, alpha=0.200, r_idf=r_idf, r_tr=r_tr)
    i3 = apply_3way(ica_ens_alt, alpha=0.255, r_idf=r_idf, r_tr=r_tr)
    s3 = apply_3way(std_ens_ref,  alpha=0.280, r_idf=r_idf, r_tr=r_tr)
    ar = macro_auc(blend3(c3, i3, s3))
    mname = f"pcridf_{r_int}_ab200_ai255_as280"
    delta = save_result(mname, ar, {"r_idf": r_idf, "a_best": 0.200, "a_ica": 0.255, "a_std": 0.280})
    if ar > best_ridf: best_ridf = ar
    if ar > best_loo - 0.0001:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  r_idf={r_idf:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best r_idf search: {best_ridf:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Per-component idf_cooc blend param (different blend for each source)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Per-component blend param ===", flush=True)
blend_opts = [0.50, 0.53, 0.55, 0.56, 0.58, 0.60]
best_pcblend = best_loo
for bp_b in [0.55]:  # best fixed
    for bp_i in blend_opts:
        for bp_s in blend_opts:
            if bp_b == bp_i == bp_s == 0.55: continue
            c3 = apply_3way(double_best, alpha=0.200, blend=bp_b)
            i3 = apply_3way(ica_ens_alt, alpha=0.255, blend=bp_i)
            s3 = apply_3way(std_ens_ref,  alpha=0.280, blend=bp_s)
            ar = macro_auc(blend3(c3, i3, s3))
            mname = f"pcbp_b{int(bp_b*100)}_i{int(bp_i*100)}_s{int(bp_s*100)}"
            delta = save_result(mname, ar, {"bp_b": bp_b, "bp_i": bp_i, "bp_s": bp_s})
            if ar > best_pcblend: best_pcblend = ar
            if ar > best_loo - 0.00008:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  bp_b={bp_b:.2f} bp_i={bp_i:.2f} bp_s={bp_s:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best per-comp blend: {best_pcblend:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Ultra-fine a_ica (0.252-0.268 step 0.001)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Ultra-fine a_ica sweep ===", flush=True)
best_uf_ica = best_loo
for av, i3 in sorted(cache_ica_fine.items()):
    ar = macro_auc(blend3(c3_ref, i3, s3_ref))
    mname = f"ufaica_{int(av*1000)}"
    delta = save_result(mname, ar, {"a_ica": av, "a_best": 0.200, "a_std": 0.280})
    if ar > best_uf_ica: best_uf_ica = ar
    if ar > best_loo - 0.00004:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_ica={av:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best ultra-fine a_ica: {best_uf_ica:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 143]
print(f"Batch143 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
