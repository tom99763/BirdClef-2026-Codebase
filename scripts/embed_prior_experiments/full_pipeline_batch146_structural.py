"""
batch146 — Structural variants: pre-blend 3way, logit-space cooc, soft-label cooc
===============================================================================
Current best: pctr_a1110_a230 LOO=0.995298
  a_best=0.200, a_ica=0.255, a_std=0.280
  two_round: a1=0.110, a2=0.030
  blend=(0.75, 0.23, 0.02), idf_blend=0.55, r_idf=0.875

Structural ideas not yet tried:
 A: Pre-blend chains first, then apply single 3way (vs apply 3way separately then blend)
 B: Logit-space transformation before cooc (sigmoid → logit → cooc → sigmoid)
 C: Soft-label cooc matrix (build COOC from prediction scores instead of hard binary labels)
 D: Apply 3way twice (second pass on already-smoothed predictions)
 E: Geometric mean / harmonic mean of components instead of linear blend
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
print(f"[batch146] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 146}
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

def soft_cooc(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None, cooc=None):
    if cooc is None: cooc = COOC
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope*(s-center), -88, 88)))
        sg = s * gate * (idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi] = s; continue
        c = cooc.T @ sg; mc = np.abs(c).max()
        if mc > EPS: c /= mc
        out[fi] = (1-alpha)*s + alpha*np.clip(c, 0, None)
    return out

def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, a1=0.110, a2=0.030):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf * idf_s + r_tr * tr

def blend3(c, i, s, wb=0.75, wi=0.23, ws=0.02):
    return wb*c + wi*i + ws*s

# Reference best
c3_ref = apply_3way(double_best, alpha=0.200)
i3_ref = apply_3way(ica_ens_alt, alpha=0.255)
s3_ref = apply_3way(std_ens_ref,  alpha=0.280)
chk = blend3(c3_ref, i3_ref, s3_ref)
print(f"Verify: {macro_auc(chk):.6f} (expect 0.995298)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Pre-blend chains first, then apply single 3way
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Pre-blend then 3way ===", flush=True)

# Standard best blend of raw chains
pre_blend = blend3(double_best, ica_ens_alt, std_ens_ref)
for alpha in [0.200, 0.225, 0.255, 0.280, 0.300]:
    pb3 = apply_3way(pre_blend, alpha=alpha)
    ar = macro_auc(pb3)
    mname = f"preblend_a{int(alpha*1000)}"
    delta = save_result(mname, ar, {"alpha": alpha, "method": "preblend"})
    flag = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  pre_blend then 3way(alpha={alpha:.3f}): {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Weighted pre-blend then 3way at various alpha and final blend
best_preblend = best_loo
for wb in [0.75, 0.80, 0.85]:
    for wi in [0.18, 0.20, 0.23]:
        ws = round(1.0 - wb - wi, 2)
        if ws < 0 or ws > 0.10: continue
        pb = blend3(double_best, ica_ens_alt, std_ens_ref, wb=wb, wi=wi, ws=ws)
        for alpha in [0.200, 0.225, 0.255]:
            pb3 = apply_3way(pb, alpha=alpha)
            # Also try linear blend of pre-blend-3way and standard per-comp approach
            for fblend in [0.0, 0.3, 0.5, 0.7, 1.0]:
                final = fblend * pb3 + (1-fblend) * chk
                ar = macro_auc(final)
                if ar > best_preblend: best_preblend = ar
                mname = f"pba_wb{int(wb*100)}_wi{int(wi*100)}_a{int(alpha*1000)}_fb{int(fblend*10)}"
                delta = save_result(mname, ar, {"wb": wb, "wi": wi, "alpha": alpha, "fblend": fblend})
                if ar > best_loo - 0.00008:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  wb={wb} wi={wi} a={alpha:.3f} fbl={fblend:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best pre-blend approach: {best_preblend:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Logit-space transformation before cooc
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Logit-space cooc ===", flush=True)

def logit_safe(x, eps=1e-4):
    return np.log(np.clip(x, eps, 1-eps) / (1 - np.clip(x, eps, 1-eps)))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -88, 88)))

def apply_3way_logit(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125):
    """Apply cooc in logit space"""
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    # Logit-space version
    sl = logit_safe(s)
    sl_pow = logit_safe(np.clip(s, 0, 1)**2)
    sc_logit = soft_cooc(sigmoid(sl_pow), alpha=alpha, idf_w=IDF075)
    idf_logit = (1-blend)*s + blend*sc_logit
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.110)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.030)
    # Mix logit-space and normal
    for mix_logit in [0.3, 0.5, 0.7]:
        idf_mix = (1-mix_logit)*idf_s + mix_logit*idf_logit
        result = r_idf * idf_mix + r_tr * tr
        # Return single for now
    return r_idf * idf_s + r_tr * tr  # fallback

for mix_logit in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    def mk_3way_logit(s, a=0.200, ml=mix_logit):
        sp = np.clip(s, 0, 1)**2
        sc_normal = soft_cooc(sp, alpha=a, idf_w=IDF075)
        idf_normal = 0.45*s + 0.55*sc_normal
        sl_pow = logit_safe(np.clip(s, 0, 1)**2)
        sc_logit = soft_cooc(sigmoid(sl_pow), alpha=a, idf_w=IDF075)
        idf_logit = 0.45*s + 0.55*sc_logit
        idf_mix = (1-ml)*idf_normal + ml*idf_logit
        r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.110)
        tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.030)
        return 0.875 * idf_mix + 0.125 * tr

    c3 = mk_3way_logit(double_best, a=0.200)
    i3 = mk_3way_logit(ica_ens_alt, a=0.255)
    s3 = mk_3way_logit(std_ens_ref,  a=0.280)
    ar = macro_auc(blend3(c3, i3, s3))
    mname = f"logit_mix{int(mix_logit*10)}"
    delta = save_result(mname, ar, {"mix_logit": mix_logit})
    flag = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  mix_logit={mix_logit:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Soft-label co-occurrence matrix (COOC built from prediction scores)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Soft-label COOC matrix ===", flush=True)

# Build soft COOC from best predictions
pred_ref = chk  # [n_files, n_species]
pred_thr = [0.3, 0.5, 0.7]

for thr in pred_thr:
    soft_labels = (pred_ref > thr).astype(np.float32)
    cnt = soft_labels.sum(0) + EPS
    COOC_SOFT = (soft_labels.T @ soft_labels) / cnt[:, None]
    np.fill_diagonal(COOC_SOFT, 0)

    def apply_3way_softcooc(s, a=0.200, cooc=COOC_SOFT):
        sp = np.clip(s, 0, 1)**2
        sc = soft_cooc(sp, alpha=a, idf_w=IDF075, cooc=cooc)
        idf_s = 0.45*s + 0.55*sc
        r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.110)
        tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.030)
        return 0.875 * idf_s + 0.125 * tr

    c3 = apply_3way_softcooc(double_best, a=0.200)
    i3 = apply_3way_softcooc(ica_ens_alt, a=0.255)
    s3 = apply_3way_softcooc(std_ens_ref,  a=0.280)
    ar_soft = macro_auc(blend3(c3, i3, s3))
    mname = f"softcooc_thr{int(thr*10)}"
    delta = save_result(mname, ar_soft, {"thr": thr, "cooc": "soft"})
    flag = " ← NEW BEST!" if ar_soft > best_loo else ""
    print(f"  soft_cooc(thr={thr:.1f}): {ar_soft:.6f} {delta:+.6f}{flag}", flush=True)

    # Blend of hard COOC + soft COOC
    for mix_soft in [0.2, 0.4, 0.6]:
        COOC_MIX = (1-mix_soft)*COOC + mix_soft*COOC_SOFT
        c3 = apply_3way_softcooc(double_best, a=0.200, cooc=COOC_MIX)
        i3 = apply_3way_softcooc(ica_ens_alt, a=0.255, cooc=COOC_MIX)
        s3 = apply_3way_softcooc(std_ens_ref,  a=0.280, cooc=COOC_MIX)
        ar = macro_auc(blend3(c3, i3, s3))
        mname = f"softmix_t{int(thr*10)}_m{int(mix_soft*10)}"
        delta = save_result(mname, ar, {"thr": thr, "mix_soft": mix_soft})
        if ar > best_loo - 0.00008:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  softmix(thr={thr:.1f},mix={mix_soft:.1f}): {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Apply 3way twice (2nd pass on already-smoothed)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Double 3way pass ===", flush=True)
best_double = best_loo
for a2nd in [0.050, 0.080, 0.100, 0.120, 0.150]:
    # Apply small extra smoothing to already-3way predictions
    c3_2 = apply_3way(c3_ref, alpha=a2nd, a1=0.040, a2=0.020)
    i3_2 = apply_3way(i3_ref, alpha=a2nd, a1=0.040, a2=0.020)
    s3_2 = apply_3way(s3_ref, alpha=a2nd, a1=0.040, a2=0.020)
    for mix2 in [0.1, 0.2, 0.3, 0.5]:
        # Blend of original and double-pass
        c_f = (1-mix2)*c3_ref + mix2*c3_2
        i_f = (1-mix2)*i3_ref + mix2*i3_2
        s_f = (1-mix2)*s3_ref + mix2*s3_2
        ar = macro_auc(blend3(c_f, i_f, s_f))
        mname = f"dbl_a{int(a2nd*1000)}_m{int(mix2*10)}"
        delta = save_result(mname, ar, {"a2nd": a2nd, "mix2": mix2})
        if ar > best_double: best_double = ar
        if ar > best_loo - 0.00008:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a2nd={a2nd:.3f} mix2={mix2:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best double 3way: {best_double:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Geometric/harmonic mean of blend components
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Non-linear blend aggregation ===", flush=True)

# Geometric mean (equal weights, take root)
c3_n = (c3_ref - c3_ref.min()) / (c3_ref.max() - c3_ref.min() + EPS)
i3_n = (i3_ref - i3_ref.min()) / (i3_ref.max() - i3_ref.min() + EPS)
s3_n = (s3_ref - s3_ref.min()) / (s3_ref.max() - s3_ref.min() + EPS)

# Weighted geometric mean
geo_mean = np.clip(c3_ref, EPS, None)**0.75 * np.clip(i3_ref, EPS, None)**0.23 * np.clip(s3_ref, EPS, None)**0.02
ar = macro_auc(geo_mean)
delta = save_result("geo_mean_075_023_002", ar, {"type": "geometric_mean"})
flag = " ← NEW BEST!" if ar > best_loo else ""
print(f"  Geometric mean (0.75^c, 0.23^i, 0.02^s): {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Rank-based aggregation
rank_c = np.argsort(np.argsort(c3_ref, axis=0), axis=0).astype(float)
rank_i = np.argsort(np.argsort(i3_ref, axis=0), axis=0).astype(float)
rank_s = np.argsort(np.argsort(s3_ref, axis=0), axis=0).astype(float)
rank_blend = 0.75*rank_c + 0.23*rank_i + 0.02*rank_s
ar = macro_auc(rank_blend)
delta = save_result("rank_blend_075_023_002", ar, {"type": "rank_blend"})
flag = " ← NEW BEST!" if ar > best_loo else ""
print(f"  Rank blend (0.75, 0.23, 0.02): {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Linear blend with rank component mixed in
for rank_mix in [0.1, 0.2, 0.3]:
    final = (1-rank_mix)*chk + rank_mix*(rank_blend / n_files)
    ar = macro_auc(final)
    mname = f"rank_mix_{int(rank_mix*10)}"
    delta = save_result(mname, ar, {"rank_mix": rank_mix})
    if ar > best_loo - 0.0001:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  rank_mix={rank_mix:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 146]
print(f"Batch146 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
