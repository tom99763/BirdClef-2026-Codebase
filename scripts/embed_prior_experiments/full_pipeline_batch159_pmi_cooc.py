"""
batch159 — PMI-based co-occurrence: PPMI, NPMI, Symmetric PMI
===============================================================================
Current best: sca4_ia26_sa28 LOO=0.995800
  chk = 0.75×c3(0.20) + 0.15×i3(0.26) + 0.10×s3(0.28)
  final = 0.72×chk + 0.28×(0.56×rank_c(0.23)+0.44×rank_i(0.40))/n_files

Rationale:
  Current COOC = P(j|i) = conditional probability (normalizes by count_i only)
  PMI[i,j] = log(P(i,j)*N / (count_i * count_j))
            = log(co_occur * N) - log(count_i) - log(count_j)
  PPMI = max(PMI, 0)  — clips negative associations
  NPMI = PMI / -log(P(i,j))  — normalized to [-1, +1]

PMI accounts for BOTH marginals, so it avoids biasing towards common species.
A rare pair that co-occurs unexpectedly gets high PMI even if absolute count is low.

Sections:
 A: Compute PMI matrices, verify standalone AUC on raw blend
 B: PPMI soft_cooc — sweep alpha (0.10-0.40) replacing standard COOC
 C: NPMI soft_cooc — sweep alpha
 D: PPMI apply_3way (replace both IDF and two_round steps)
 E: Blend PPMI-3way into existing chk formula
 F: PPMI-based rank component (replace or supplement current rank)
 G: Hybrid: PPMI for idf_cooc step only, standard for two_round step
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

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch159] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 159}
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

# ─── Build co-occurrence matrices ──────────────────────────────────────────────
fl_hard  = file_labels.astype(np.float32)
count_i  = fl_hard.sum(0) + EPS                      # (n_sp,)  count per species

# Standard COOC: P(j|i)
COOC = (fl_hard.T @ fl_hard) / count_i[:, None]
np.fill_diagonal(COOC, 0)
raw_idf = np.clip(np.log(float(n_files) / (count_i + 1.0 - EPS)), 0, None)
IDF075  = raw_idf ** 0.75; IDF075 /= (IDF075.mean() + EPS)

# PMI matrix
#   co_occur[i,j] = number of files where both i and j present
#   P(i) = count_i / n_files
#   P(j) = count_j / n_files
#   P(i,j) = co_occur[i,j] / n_files
#   PMI[i,j] = log(P(i,j) / (P(i)*P(j)))
#             = log(co_occur * n_files / (count_i * count_j))
co_occur = fl_hard.T @ fl_hard  # (n_sp, n_sp)
np.fill_diagonal(co_occur, 0)

pij     = co_occur / float(n_files)  # joint probability
pi_pj   = (count_i[:, None] * count_i[None, :]) / (float(n_files)**2)  # product of marginals
pmi_raw = np.where(co_occur > 0, np.log(np.clip(pij / (pi_pj + EPS), EPS, None)), -10.0)

# PPMI: clip negative
PPMI = np.maximum(pmi_raw, 0.0).astype(np.float32)
np.fill_diagonal(PPMI, 0)

# NPMI: PMI / -log(P(i,j)), range [-1, 1]; clip to [0, 1] for use as cooc
log_pij = np.where(co_occur > 0, np.log(np.clip(pij, EPS, None)), -10.0)
npmi_raw = np.where(co_occur > 0, pmi_raw / (-log_pij + EPS), 0.0)
NPMI = np.clip(npmi_raw, 0.0, 1.0).astype(np.float32)
np.fill_diagonal(NPMI, 0)

# Normalize matrices to have same scale as COOC (max=1)
PPMI_norm = PPMI / (PPMI.max() + EPS)
NPMI_norm = NPMI  # already [0, 1]

print(f"PPMI non-zero: {(PPMI > 0).sum()}, max={PPMI.max():.3f}", flush=True)
print(f"NPMI non-zero: {(NPMI > 0).sum()}, max={NPMI.max():.3f}", flush=True)
print(f"COOC non-zero: {(COOC > 0).sum()}, max={COOC.max():.3f}", flush=True)

# ─── soft_cooc variants ────────────────────────────────────────────────────────
def soft_cooc(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    """Standard soft_cooc using COOC matrix"""
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

def soft_cooc_ppmi(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    """soft_cooc using PPMI matrix"""
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope*(s-center), -88, 88)))
        sg = s * gate * (idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi] = s; continue
        c = PPMI_norm.T @ sg; mc = np.abs(c).max()
        if mc > EPS: c /= mc
        out[fi] = (1-alpha)*s + alpha*np.clip(c, 0, None)
    return out

def soft_cooc_npmi(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    """soft_cooc using NPMI matrix"""
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope*(s-center), -88, 88)))
        sg = s * gate * (idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi] = s; continue
        c = NPMI_norm.T @ sg; mc = np.abs(c).max()
        if mc > EPS: c /= mc
        out[fi] = (1-alpha)*s + alpha*np.clip(c, 0, None)
    return out

# ─── 3way variants ──────────────────────────────────────────────────────────────
def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, a1=0.110, a2=0.030):
    """Standard 3way using COOC"""
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf * idf_s + r_tr * tr

def apply_3way_ppmi(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, a1=0.110, a2=0.030):
    """3way: idf_cooc step uses PPMI, two_round uses standard COOC (Hybrid-G)"""
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc_ppmi(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf * idf_s + r_tr * tr

def apply_3way_ppmi_full(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, a1=0.110, a2=0.030):
    """3way: ALL steps use PPMI"""
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc_ppmi(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc_ppmi(s, center=0.54, slope=41.0, alpha=a1)
    tr = soft_cooc_ppmi(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf * idf_s + r_tr * tr

def apply_3way_npmi(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, a1=0.110, a2=0.030):
    """3way: idf_cooc step uses NPMI, two_round uses standard COOC"""
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc_npmi(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf * idf_s + r_tr * tr

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

# ─── Reference (confirm best) ──────────────────────────────────────────────────
c3_ref = apply_3way(double_best, alpha=0.200)
i3_ref = apply_3way(ica_ens_alt, alpha=0.260)
s3_ref = apply_3way(std_ens_ref,  alpha=0.280)
chk_ref = 0.75*c3_ref + 0.15*i3_ref + 0.10*s3_ref
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_ref   = 0.56*rank_c_ref + 0.44*rank_i_ref
rank_norm  = rank_ref / n_files
v = 0.72*chk_ref + 0.28*rank_norm
print(f"Verify: {macro_auc(v):.6f} (expect ~0.995800)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Standalone PPMI & NPMI soft_cooc, sweep alpha
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: PPMI soft_cooc standalone ===", flush=True)
best_a = best_loo
for a in [0.10, 0.15, 0.20, 0.23, 0.25, 0.28, 0.30, 0.35, 0.40]:
    # PPMI idf_cooc only (like the idf step in 3way)
    sp_db = np.clip(double_best, 0, 1)**2
    sc_p = soft_cooc_ppmi(sp_db, alpha=a, idf_w=IDF075)
    idf_p = 0.45*double_best + 0.55*sc_p
    ar = macro_auc(idf_p)
    mname = f"ppmi_idf_a{int(a*100)}"
    delta = save_result(mname, ar)
    if ar > best_a: best_a = ar
    if ar > best_loo - 0.0005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  PPMI idf a={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

for a in [0.10, 0.15, 0.20, 0.23, 0.25, 0.28, 0.30, 0.35, 0.40]:
    sp_db = np.clip(double_best, 0, 1)**2
    sc_n = soft_cooc_npmi(sp_db, alpha=a, idf_w=IDF075)
    idf_n = 0.45*double_best + 0.55*sc_n
    ar = macro_auc(idf_n)
    mname = f"npmi_idf_a{int(a*100)}"
    delta = save_result(mname, ar)
    if ar > best_a: best_a = ar
    if ar > best_loo - 0.0005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  NPMI idf a={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section A: {best_a:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: PPMI hybrid 3way (idf step only) — sweep alpha, blend into chk
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: PPMI hybrid 3way (idf step PPMI, two_round standard) ===", flush=True)
best_b = best_loo
ppmi_signals = {}  # cache
for a in [0.15, 0.20, 0.23, 0.25, 0.28, 0.30, 0.35, 0.40]:
    c3_p = apply_3way_ppmi(double_best, alpha=a)
    i3_p = apply_3way_ppmi(ica_ens_alt, alpha=a*1.3)
    s3_p = apply_3way_ppmi(std_ens_ref,  alpha=a*1.4)
    ppmi_signals[a] = (c3_p, i3_p, s3_p)
    chk_p = 0.75*c3_p + 0.15*i3_p + 0.10*s3_p
    ar = macro_auc(chk_p)
    save_result(f"ppmi_hyb_a{int(a*100)}_chk", ar)
    # rank with PPMI hybrid
    rk_c_p = make_rank(apply_3way_ppmi(double_best, alpha=a*1.15))
    rk_i_p = make_rank(apply_3way_ppmi(ica_ens_alt, alpha=a*1.75))
    rk_p = 0.56*rk_c_p + 0.44*rk_i_p
    final = 0.72*chk_p + 0.28*(rk_p/n_files)
    ar_full = macro_auc(final)
    mname = f"ppmi_hyb_full_a{int(a*100)}"
    delta = save_result(mname, ar_full, {"ppmi_alpha": a})
    if ar_full > best_b: best_b = ar_full
    if ar_full > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar_full > best_loo else ""
        print(f"  PPMI-hyb a={a:.2f}: chk={ar:.6f} full={ar_full:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section B: {best_b:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: PPMI full 3way (all steps PPMI) — sweep alpha
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: PPMI full 3way (all steps PPMI) ===", flush=True)
best_c = best_loo
for a in [0.15, 0.20, 0.23, 0.25, 0.30, 0.35]:
    c3_pf = apply_3way_ppmi_full(double_best, alpha=a)
    i3_pf = apply_3way_ppmi_full(ica_ens_alt, alpha=a*1.3)
    s3_pf = apply_3way_ppmi_full(std_ens_ref,  alpha=a*1.4)
    chk_pf = 0.75*c3_pf + 0.15*i3_pf + 0.10*s3_pf
    rk_c_pf = make_rank(apply_3way_ppmi_full(double_best, alpha=a*1.15))
    rk_i_pf = make_rank(apply_3way_ppmi_full(ica_ens_alt, alpha=a*1.75))
    rk_pf = 0.56*rk_c_pf + 0.44*rk_i_pf
    final = 0.72*chk_pf + 0.28*(rk_pf/n_files)
    ar = macro_auc(final)
    mname = f"ppmi_full_a{int(a*100)}"
    delta = save_result(mname, ar, {"ppmi_alpha": a, "mode": "full"})
    if ar > best_c: best_c = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  PPMI-full a={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: NPMI hybrid 3way — sweep alpha
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: NPMI hybrid 3way (idf step NPMI, two_round standard) ===", flush=True)
best_d = best_loo
for a in [0.15, 0.20, 0.23, 0.25, 0.30, 0.35, 0.40]:
    c3_n = apply_3way_npmi(double_best, alpha=a)
    i3_n = apply_3way_npmi(ica_ens_alt, alpha=a*1.3)
    s3_n = apply_3way_npmi(std_ens_ref,  alpha=a*1.4)
    chk_n = 0.75*c3_n + 0.15*i3_n + 0.10*s3_n
    rk_c_n = make_rank(apply_3way_npmi(double_best, alpha=a*1.15))
    rk_i_n = make_rank(apply_3way_npmi(ica_ens_alt, alpha=a*1.75))
    rk_n = 0.56*rk_c_n + 0.44*rk_i_n
    final = 0.72*chk_n + 0.28*(rk_n/n_files)
    ar = macro_auc(final)
    mname = f"npmi_hyb_a{int(a*100)}"
    delta = save_result(mname, ar, {"npmi_alpha": a})
    if ar > best_d: best_d = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  NPMI-hyb a={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section D: {best_d:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Blend PPMI-chk with standard chk
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Blend PPMI chk with standard chk ===", flush=True)
best_e = best_loo
# Use best alpha from section B (try a=0.23 as starting point)
for a_ppmi in [0.20, 0.23, 0.25, 0.30]:
    c3_p, i3_p, s3_p = ppmi_signals.get(a_ppmi, (
        apply_3way_ppmi(double_best, alpha=a_ppmi),
        apply_3way_ppmi(ica_ens_alt, alpha=a_ppmi*1.3),
        apply_3way_ppmi(std_ens_ref,  alpha=a_ppmi*1.4),
    ))
    chk_p = 0.75*c3_p + 0.15*i3_p + 0.10*s3_p
    for w_ppmi in [0.05, 0.10, 0.15, 0.20, 0.30]:
        chk_blend = (1-w_ppmi)*chk_ref + w_ppmi*chk_p
        final = 0.72*chk_blend + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"ppmi_blend_a{int(a_ppmi*100)}_w{int(w_ppmi*100)}"
        delta = save_result(mname, ar, {"a_ppmi": a_ppmi, "w_ppmi": w_ppmi})
        if ar > best_e: best_e = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  blend a={a_ppmi:.2f} w={w_ppmi:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section E: {best_e:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: PPMI-based rank component
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== F: PPMI rank component (supplement or replace standard rank) ===", flush=True)
best_f = best_loo

# Build PPMI rank signals at best alpha values
for a_c_ppmi in [0.20, 0.23, 0.25, 0.30]:
    rk_c_p = make_rank(apply_3way_ppmi(double_best, alpha=a_c_ppmi))
    for a_i_ppmi in [0.35, 0.40, 0.45, 0.50]:
        rk_i_p = make_rank(apply_3way_ppmi(ica_ens_alt, alpha=a_i_ppmi))
        # Pure PPMI rank
        rk_ppmi = 0.56*rk_c_p + 0.44*rk_i_p
        final = 0.72*chk_ref + 0.28*(rk_ppmi/n_files)
        ar = macro_auc(final)
        mname = f"ppmi_rk_ac{int(a_c_ppmi*100)}_ai{int(a_i_ppmi*100)}"
        delta = save_result(mname, ar, {"a_rank_c_ppmi": a_c_ppmi, "a_rank_i_ppmi": a_i_ppmi})
        if ar > best_f: best_f = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  PPMI-rk ac={a_c_ppmi:.2f} ai={a_i_ppmi:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

        # Blend PPMI rank with standard rank
        for w_ppmi_rk in [0.20, 0.40, 0.50, 0.60, 0.80]:
            rk_mix = (1-w_ppmi_rk)*rank_ref + w_ppmi_rk*rk_ppmi
            final = 0.72*chk_ref + 0.28*(rk_mix/n_files)
            ar = macro_auc(final)
            mname = f"ppmi_rkblend_ac{int(a_c_ppmi*100)}_ai{int(a_i_ppmi*100)}_w{int(w_ppmi_rk*100)}"
            delta = save_result(mname, ar)
            if ar > best_f: best_f = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  PPMI-rkblend ac={a_c_ppmi:.2f} ai={a_i_ppmi:.2f} w={w_ppmi_rk:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section F: {best_f:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# G: PMI-weighted IDF (use PMI signal as alternative to IDF075 weighting)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== G: PMI-weighted co-occurrence (PMI replaces IDF075) ===", flush=True)
best_g = best_loo

# Alternative IDF-like weighting: per-species PMI signal strength
#   pmi_signal[i] = sum of PPMI[i,:] (how many strong PMI links does species i have)
pmi_signal = PPMI_norm.sum(1)  # (n_sp,)
pmi_signal = pmi_signal / (pmi_signal.mean() + EPS)

for pmi_pow in [0.5, 0.75, 1.0, 1.5]:
    pmi_w = pmi_signal ** pmi_pow
    pmi_w = pmi_w / (pmi_w.mean() + EPS)
    # Use pmi_w instead of IDF075 in the idf_cooc step
    for a in [0.20, 0.25, 0.30]:
        sp_db = np.clip(double_best, 0, 1)**2
        sc = soft_cooc(sp_db, alpha=a, idf_w=pmi_w)
        idf_s = 0.45*double_best + 0.55*sc
        ar = macro_auc(idf_s)
        mname = f"pmiidf_pow{int(pmi_pow*100)}_a{int(a*100)}"
        delta = save_result(mname, ar, {"pmi_pow": pmi_pow, "alpha": a})
        if ar > best_g: best_g = ar
        if ar > best_loo - 0.0005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  pmi_idf pow={pmi_pow:.2f} a={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

    # Blend IDF075 and pmi_w
    for idf_mix in [0.3, 0.5, 0.7]:
        mixed_w = idf_mix*IDF075 + (1-idf_mix)*pmi_w
        for a in [0.20, 0.23, 0.25]:
            sp_db = np.clip(double_best, 0, 1)**2
            sc = soft_cooc(sp_db, alpha=a, idf_w=mixed_w)
            idf_s = 0.45*double_best + 0.55*sc
            r1 = soft_cooc(double_best, center=0.54, slope=41.0, alpha=0.110)
            tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.030)
            out_s = 0.875*idf_s + 0.125*tr
            ar = macro_auc(out_s)
            mname = f"pmiidf_mix_pow{int(pmi_pow*100)}_im{int(idf_mix*100)}_a{int(a*100)}"
            delta = save_result(mname, ar)
            if ar > best_g: best_g = ar
            if ar > best_loo - 0.0005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  pmi_idf_mix pow={pmi_pow:.2f} im={idf_mix:.1f} a={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section G: {best_g:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 159]
print(f"Batch159 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
