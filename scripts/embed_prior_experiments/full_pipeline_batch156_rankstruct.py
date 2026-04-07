"""
batch156 — Structural rank variants + multi-alpha rank ensemble
===============================================================================
Current best: sca4_ia26_sa28 LOO=0.995800
  Formula confirmed: chk = 0.75×c3(a=0.20) + 0.15×i3(a=0.26) + 0.10×s3(a=0.28)
  rank_c = rank(apply_3way(db, a=0.23))
  rank_i = rank(apply_3way(ia, a=0.40))
  final = 0.72×chk + 0.28×(0.56×rank_c + 0.44×rank_i)/n_files

batch155 findings:
- All fine-tuning converges to LOO=0.995800 — clear plateau
- Score alphas confirmed: (0.20, 0.26, 0.28) optimal
- Rank alphas confirmed: (0.23, 0.40) optimal
- Need structural exploration

Directions:
 A: Multi-alpha rank ensemble: average ranks from 3-4 different alpha values
 B: Rank of only idf_cooc part (without two_round) vs rank of only two_round part
 C: Rank of chk blend itself (self-rank boost)
 D: Rank computed on per-species zscore normalized predictions
 E: Separate rank for score and IDF: rank(s^2 * IDF) type variants
 F: Two-round of ranks (apply soft_cooc to ranks themselves)
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
print(f"[batch156] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 156}
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

def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, a1=0.110, a2=0.030):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf * idf_s + r_tr * tr

def idf_only(s, alpha=0.200):
    """Only idf_cooc part, no two_round"""
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    return (1-0.55)*s + 0.55*sc

def two_round_only(s, a1=0.110, a2=0.030):
    """Only two_round part, no idf_cooc"""
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    return soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

# Reference formula
c3_ref = apply_3way(double_best, alpha=0.200)
i3_ref = apply_3way(ica_ens_alt, alpha=0.260)
s3_ref = apply_3way(std_ens_ref,  alpha=0.280)
chk_ref = 0.75*c3_ref + 0.15*i3_ref + 0.10*s3_ref

rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_ref = 0.56*rank_c_ref + 0.44*rank_i_ref
rank_norm = rank_ref / n_files

v = 0.72*chk_ref + 0.28*rank_norm
print(f"Verify: {macro_auc(v):.6f} (expect ~0.995800)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Multi-alpha rank ensemble (average ranks at multiple alphas)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Multi-alpha rank ensemble ===", flush=True)
# Pre-compute ranks at multiple alphas for ICA
alphas_c = [0.20, 0.23, 0.25, 0.28, 0.30]
alphas_i = [0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.50]
cache_rc = {a: make_rank(apply_3way(double_best, alpha=a)) for a in alphas_c}
cache_ri = {a: make_rank(apply_3way(ica_ens_alt, alpha=a)) for a in alphas_i}

best_ma = best_loo
# Try: average rank across multiple ICA alphas
for n_ai in [3, 4, 5]:  # use n_ai alphas from center of best range
    center_alphas = [0.38, 0.40, 0.42]
    if n_ai == 4: center_alphas = [0.35, 0.38, 0.40, 0.42]
    if n_ai == 5: center_alphas = [0.35, 0.38, 0.40, 0.42, 0.45]
    avg_ri = np.mean([cache_ri[a] for a in center_alphas], axis=0)
    rk = 0.56*rank_c_ref + 0.44*avg_ri
    for rm in [0.26, 0.28, 0.30]:
        final = (1-rm)*chk_ref + rm*(rk/n_files)
        ar = macro_auc(final)
        mname = f"marank_ni{n_ai}_nc1_rm{int(rm*100)}"
        delta = save_result(mname, ar, {"n_ai": n_ai, "rm": rm})
        if ar > best_ma: best_ma = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  avg_{n_ai}AI rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Also try averaging C alphas
for n_ac in [2, 3]:
    ac_alphas = [0.20, 0.23] if n_ac == 2 else [0.20, 0.23, 0.25]
    avg_rc = np.mean([cache_rc[a] for a in ac_alphas], axis=0)
    rk = 0.56*avg_rc + 0.44*rank_i_ref
    for rm in [0.26, 0.28, 0.30]:
        final = (1-rm)*chk_ref + rm*(rk/n_files)
        ar = macro_auc(final)
        mname = f"marank_nc{n_ac}_ni1_rm{int(rm*100)}"
        delta = save_result(mname, ar, {"n_ac": n_ac, "rm": rm})
        if ar > best_ma: best_ma = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  avg_{n_ac}AC rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best multi-alpha rank: {best_ma:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Rank of idf_only vs two_round_only
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Rank of idf_only vs two_round_only ===", flush=True)
best_rb = best_loo
for alpha_c in [0.20, 0.23, 0.26]:
    rk_c_idf = make_rank(idf_only(double_best, alpha=alpha_c))
    rk_c_tr  = make_rank(two_round_only(double_best))
    for alpha_i in [0.35, 0.40, 0.45]:
        rk_i_idf = make_rank(idf_only(ica_ens_alt, alpha=alpha_i))
        rk_i_tr  = make_rank(two_round_only(ica_ens_alt))
        # IDF-only rank
        for wb, wi in [(0.55, 0.45), (0.56, 0.44), (0.60, 0.40)]:
            rk_idf = wb*rk_c_idf + wi*rk_i_idf
            final = 0.72*chk_ref + 0.28*(rk_idf/n_files)
            ar = macro_auc(final)
            mname = f"idfonly_rk_ac{int(alpha_c*100)}_ai{int(alpha_i*100)}_wb{int(wb*100)}"
            delta = save_result(mname, ar, {"type": "idf_only", "a_c": alpha_c, "a_i": alpha_i, "wb": wb})
            if ar > best_rb: best_rb = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  idf_only ac={alpha_c:.2f} ai={alpha_i:.2f} wb={wb:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best idf/tr rank: {best_rb:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Rank of chk blend itself (self-rank boost)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Rank of chk blend itself ===", flush=True)
rank_chk = make_rank(chk_ref)
best_rc = best_loo
for rm in np.arange(0.10, 0.40, 0.02):
    rm_r = round(float(rm), 2)
    final = (1-rm_r)*chk_ref + rm_r*(rank_chk/n_files)
    ar = macro_auc(final)
    mname = f"chkrank_rm{int(rm_r*100)}"
    delta = save_result(mname, ar, {"rm": rm_r, "type": "chk_self_rank"})
    if ar > best_rc: best_rc = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  self-rank rm={rm_r:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Also try combining chk rank with component ranks
for w_chkrk in [0.05, 0.10, 0.15]:
    w_comp = 1 - w_chkrk
    combined_rank = w_comp*rank_ref + w_chkrk*rank_chk * n_files  # unnorm back
    for rm in [0.26, 0.28, 0.30]:
        final = (1-rm)*chk_ref + rm*(combined_rank/(n_files*(w_comp+w_chkrk)))
        ar = macro_auc(final)
        mname = f"mixrank_wc{int(w_chkrk*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar, {"w_chk_rank": w_chkrk, "rm": rm})
        if ar > best_rc: best_rc = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  mix-rank wc={w_chkrk:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best self/mix rank: {best_rc:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Per-species zscore normalized before ranking
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Zscore-normalized rank ===", flush=True)
best_zr = best_loo
for source in [double_best, ica_ens_alt]:
    name = "db" if source is double_best else "ia"
    for alpha in [0.23, 0.40]:
        s3w = apply_3way(source, alpha=alpha)
        # Zscore across files per species
        mu = s3w.mean(axis=0, keepdims=True)
        sigma = s3w.std(axis=0, keepdims=True) + EPS
        s_norm = (s3w - mu) / sigma
        rk_norm_sp = make_rank(s_norm)
        if name == "db":
            rk = 0.56*rk_norm_sp + 0.44*rank_i_ref
        else:
            rk = 0.56*rank_c_ref + 0.44*rk_norm_sp
        for rm in [0.26, 0.28, 0.30]:
            final = (1-rm)*chk_ref + rm*(rk/n_files)
            ar = macro_auc(final)
            mname = f"zrk_{name}_a{int(alpha*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar, {"source": name, "alpha": alpha, "rm": rm})
            if ar > best_zr: best_zr = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  zscore {name} a={alpha:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best zscore rank: {best_zr:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Two-round of ranks (soft_cooc applied to rank values)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Soft-cooc applied to rank values ===", flush=True)
best_sr = best_loo
rank_c_norm = rank_c_ref / n_files
rank_i_norm = rank_i_ref / n_files
for alpha in [0.05, 0.10, 0.15, 0.20]:
    rk_c_smooth = soft_cooc(rank_c_norm, center=0.50, slope=20.0, alpha=alpha)
    rk_i_smooth = soft_cooc(rank_i_norm, center=0.50, slope=20.0, alpha=alpha)
    for wb in [0.55, 0.56, 0.60]:
        rk_sm = wb*rk_c_smooth + (1-wb)*rk_i_smooth
        for rm in [0.26, 0.28, 0.30]:
            final = (1-rm)*chk_ref + rm*rk_sm
            ar = macro_auc(final)
            mname = f"smrk_a{int(alpha*100)}_wb{int(wb*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar, {"alpha_smooth": alpha, "wb": wb, "rm": rm})
            if ar > best_sr: best_sr = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  smooth-rank a={alpha:.2f} wb={wb:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best smooth rank: {best_sr:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 156]
print(f"Batch156 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
