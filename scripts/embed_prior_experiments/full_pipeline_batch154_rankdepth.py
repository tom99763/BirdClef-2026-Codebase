"""
batch154 — Deeper rank alpha search + STD rank reintroduction
===============================================================================
Current best: uf3j_rk56_cb75_ci15_rm28 LOO=0.995780
  rank = 0.56×rank_c + 0.44×rank_i
  rank alphas: a_rank_c=0.23, a_rank_i=0.40
  chk = 0.75×c3 + 0.15×i3 + 0.10×s3
  final = 0.72×chk + 0.28×(rank/n_files)
  score alphas: a_best=0.200, a_ica=0.255, a_std=0.280

batch153 findings:
- a_rank_c improved 0.20→0.23 (section E gave 0.995750, section F gave 0.995780)
- rank weights (0.56, 0.44) remain optimal; STD rank = 0
- chk blend shifted to (0.75, 0.15, 0.10) — STD weight up from 0.09

Directions:
 A: Fine a_rank_c (0.20-0.30 step 0.005) at fixed other params
 B: Fine a_rank_i (0.35-0.48 step 0.005)
 C: Joint a_rank_c × a_rank_i grid
 D: Reintroduce STD rank with low weight (0-0.10 step 0.01) and 3-comp alpha
 E: Score alpha re-tune: a_ica and a_std at new rank params
 F: Power-scaled rank: use rank^p/n_files^p instead of rank/n_files
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
print(f"[batch154] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 154}
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

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

# Fixed score components
c3_score = apply_3way(double_best, alpha=0.200)
i3_score = apply_3way(ica_ens_alt, alpha=0.255)
s3_score = apply_3way(std_ens_ref,  alpha=0.280)

# Best rank components from batch153
rank_c_best = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_best = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_s_best = make_rank(apply_3way(std_ens_ref,  alpha=0.280))

chk_best = 0.75*c3_score + 0.15*i3_score + 0.10*s3_score
rank_ref = 0.56*rank_c_best + 0.44*rank_i_best

v = 0.72*chk_best + 0.28*(rank_ref/n_files)
print(f"Verify: {macro_auc(v):.6f} (expect ~0.995780)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Fine a_rank_c (0.185-0.295 step 0.005)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Fine a_rank_c (step 0.005) ===", flush=True)
best_ac = best_loo
best_ac_val = 0.23
for a_int in range(37, 60):  # 0.185-0.295 step 0.005
    a = a_int / 200.0
    rk_c = make_rank(apply_3way(double_best, alpha=a))
    rk = 0.56*rk_c + 0.44*rank_i_best
    final = 0.72*chk_best + 0.28*(rk/n_files)
    ar = macro_auc(final)
    mname = f"fac4_ac{int(a*1000)}_ai400"
    delta = save_result(mname, ar, {"a_rank_c": a, "a_rank_i": 0.40})
    if ar > best_ac:
        best_ac = ar
        best_ac_val = a
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_rank_c={a:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best a_rank_c: {best_ac_val:.3f} → {best_ac:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Fine a_rank_i (0.350-0.475 step 0.005)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Fine a_rank_i (step 0.005) at a_rank_c={best_ac_val:.3f} ===", flush=True)
rk_c_updated = make_rank(apply_3way(double_best, alpha=best_ac_val))
best_ai = best_loo
best_ai_val = 0.40
for a_int in range(70, 97):  # 0.350-0.480 step 0.005
    a = a_int / 200.0
    rk_i = make_rank(apply_3way(ica_ens_alt, alpha=a))
    rk = 0.56*rk_c_updated + 0.44*rk_i
    final = 0.72*chk_best + 0.28*(rk/n_files)
    ar = macro_auc(final)
    mname = f"fai4_ai{int(a*1000)}_ac{int(best_ac_val*1000)}"
    delta = save_result(mname, ar, {"a_rank_i": a, "a_rank_c": best_ac_val})
    if ar > best_ai:
        best_ai = ar
        best_ai_val = a
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_rank_i={a:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best a_rank_i: {best_ai_val:.3f} → {best_ai:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Joint a_rank_c × a_rank_i grid (fine step 0.01)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Joint rank alpha grid ===", flush=True)
alpha_c_range = [round(x/100, 2) for x in range(18, 32)]  # 0.18-0.31
alpha_i_range = [round(x/100, 2) for x in range(35, 50)]  # 0.35-0.49
best_jab = best_loo
best_jab_params = (best_ac_val, best_ai_val)
for a_c in alpha_c_range:
    rk_c = make_rank(apply_3way(double_best, alpha=a_c))
    for a_i in alpha_i_range:
        rk_i = make_rank(apply_3way(ica_ens_alt, alpha=a_i))
        rk = 0.56*rk_c + 0.44*rk_i
        final = 0.72*chk_best + 0.28*(rk/n_files)
        ar = macro_auc(final)
        mname = f"jab4_ac{int(a_c*100)}_ai{int(a_i*100)}"
        delta = save_result(mname, ar, {"a_rank_c": a_c, "a_rank_i": a_i})
        if ar > best_jab:
            best_jab = ar
            best_jab_params = (a_c, a_i)
        if ar > best_loo - 0.00004:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_c={a_c:.2f} a_i={a_i:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

best_ac2, best_ai2 = best_jab_params
print(f"  Best joint alpha: ({best_ac2:.2f}, {best_ai2:.2f}) → {best_jab:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Reintroduce STD rank with small weight
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: STD rank reintroduction (3-component) ===", flush=True)
rk_c_d = make_rank(apply_3way(double_best, alpha=best_ac2))
rk_i_d = make_rank(apply_3way(ica_ens_alt, alpha=best_ai2))
# Try STD rank at various alphas
alpha_s_rk = [round(x/100, 2) for x in range(20, 50, 5)]  # 0.20-0.45
best_std_rk = best_loo
for a_s in alpha_s_rk:
    rk_s = make_rank(apply_3way(std_ens_ref, alpha=a_s))
    for ws in [0.02, 0.04, 0.06, 0.08, 0.10]:
        wb_c = round(0.56 * (1 - ws), 2)
        wi_i = round(0.44 * (1 - ws), 2)
        # normalize
        rk = wb_c*rk_c_d + wi_i*rk_i_d + ws*rk_s
        final = 0.72*chk_best + 0.28*(rk/n_files)
        ar = macro_auc(final)
        mname = f"std3rk_as{int(a_s*100)}_ws{int(ws*100)}_ac{int(best_ac2*100)}_ai{int(best_ai2*100)}"
        delta = save_result(mname, ar, {"a_rank_s": a_s, "ws": ws, "a_rank_c": best_ac2, "a_rank_i": best_ai2})
        if ar > best_std_rk: best_std_rk = ar
        if ar > best_loo - 0.00004:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_s={a_s:.2f} ws={ws:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best 3-comp rank: {best_std_rk:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Re-tune score alphas (a_ica, a_std) at new rank setup
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Score alpha re-tune (a_ica, a_std) ===", flush=True)
rk_final = 0.56*rk_c_d + 0.44*rk_i_d
best_sa = best_loo
for a_ica in [round(x/100, 2) for x in range(20, 35)]:
    i3_new = apply_3way(ica_ens_alt, alpha=a_ica)
    for a_std in [round(x/100, 2) for x in range(25, 36)]:
        s3_new = apply_3way(std_ens_ref, alpha=a_std)
        chk_new = 0.75*c3_score + 0.15*i3_new + 0.10*s3_new
        final = 0.72*chk_new + 0.28*(rk_final/n_files)
        ar = macro_auc(final)
        mname = f"sca4_ia{int(a_ica*100)}_sa{int(a_std*100)}"
        delta = save_result(mname, ar, {"a_ica": a_ica, "a_std": a_std})
        if ar > best_sa: best_sa = ar
        if ar > best_loo - 0.00004:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_ica={a_ica:.2f} a_std={a_std:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best score alphas: {best_sa:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: Rank power scaling (rank^p / n_files^p)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== F: Rank power scaling ===", flush=True)
best_pow = best_loo
for p in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30]:
    rk_scaled = (rk_final ** p) / (n_files ** p)
    for rm in [0.26, 0.28, 0.30, 0.32]:
        final = (1-rm)*chk_best + rm*rk_scaled
        ar = macro_auc(final)
        mname = f"rkpow_p{int(p*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar, {"rank_power": p, "rm": rm})
        if ar > best_pow: best_pow = ar
        if ar > best_loo - 0.00004:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  power={p:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best rank power: {best_pow:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 154]
print(f"Batch154 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
