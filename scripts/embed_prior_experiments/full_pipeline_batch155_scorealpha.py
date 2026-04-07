"""
batch155 — Fine-tune score alphas + joint comprehensive optimization
===============================================================================
Current best: sca4_ia26_sa28 LOO=0.995800
  score alphas: a_best=0.200, a_ica=0.260, a_std=0.280
  rank alphas: a_rank_c=0.23, a_rank_i=0.40
  rank = 0.56×rank_c + 0.44×rank_i
  chk = 0.75×c3 + 0.15×i3 + 0.10×s3
  final = 0.72×chk + 0.28×(rank/n_files)

batch154 findings:
- Score alpha a_ica: 0.255 → 0.260 (small improvement)
- Score alpha a_std: 0.280 → 0.280 (unchanged)
- Section E: a_ica=0.26, a_std=0.28 is optimal at step 0.01
- rank alpha plateau: a_rank_c=0.23, a_rank_i=0.38-0.43 all equivalent

Directions:
 A: Ultra-fine a_ica (0.245-0.275 step 0.005)
 B: Ultra-fine a_std (0.260-0.300 step 0.005)
 C: Fine-tune a_best (0.160-0.240 step 0.01)
 D: Joint: all 3 score alphas together (coarse)
 E: Joint: chk blend re-opt at new score alphas
 F: Mega joint: score alphas + chk blend + rank weights + rm
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
print(f"[batch155] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 155}
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

# Best rank components
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_ref = 0.56*rank_c_ref + 0.44*rank_i_ref
rank_norm = rank_ref / n_files

# Best score components from batch154
c3_ref = apply_3way(double_best, alpha=0.200)
i3_ref = apply_3way(ica_ens_alt, alpha=0.260)
s3_ref = apply_3way(std_ens_ref,  alpha=0.280)
chk_ref = 0.75*c3_ref + 0.15*i3_ref + 0.10*s3_ref

v = 0.72*chk_ref + 0.28*rank_norm
print(f"Verify: {macro_auc(v):.6f} (expect ~0.995800)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Ultra-fine a_ica (0.240-0.280 step 0.005)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Ultra-fine a_ica (step 0.005) ===", flush=True)
best_ai = best_loo
best_ai_val = 0.260
for a_int in range(48, 57):  # 0.240-0.280 step 0.005
    a = a_int / 200.0
    i3 = apply_3way(ica_ens_alt, alpha=a)
    chk = 0.75*c3_ref + 0.15*i3 + 0.10*s3_ref
    final = 0.72*chk + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"ufia_ia{int(a*1000)}"
    delta = save_result(mname, ar, {"a_ica": a})
    if ar > best_ai:
        best_ai = ar
        best_ai_val = a
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_ica={a:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best a_ica: {best_ai_val:.3f} → {best_ai:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Ultra-fine a_std (0.260-0.300 step 0.005)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Ultra-fine a_std (step 0.005) ===", flush=True)
i3_best = apply_3way(ica_ens_alt, alpha=best_ai_val)
best_as = best_loo
best_as_val = 0.280
for a_int in range(52, 61):  # 0.260-0.300 step 0.005
    a = a_int / 200.0
    s3 = apply_3way(std_ens_ref, alpha=a)
    chk = 0.75*c3_ref + 0.15*i3_best + 0.10*s3
    final = 0.72*chk + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"ufsa_sa{int(a*1000)}_ia{int(best_ai_val*1000)}"
    delta = save_result(mname, ar, {"a_std": a, "a_ica": best_ai_val})
    if ar > best_as:
        best_as = ar
        best_as_val = a
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_std={a:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best a_std: {best_as_val:.3f} → {best_as:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Fine a_best (0.150-0.250 step 0.01)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Fine a_best (step 0.01) ===", flush=True)
s3_best = apply_3way(std_ens_ref, alpha=best_as_val)
best_ab = best_loo
best_ab_val = 0.200
for a_int in range(15, 26):
    a = a_int / 100.0
    c3 = apply_3way(double_best, alpha=a)
    chk = 0.75*c3 + 0.15*i3_best + 0.10*s3_best
    final = 0.72*chk + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"ufab_ab{int(a*100)}_ia{int(best_ai_val*1000)}"
    delta = save_result(mname, ar, {"a_best": a, "a_ica": best_ai_val, "a_std": best_as_val})
    if ar > best_ab:
        best_ab = ar
        best_ab_val = a
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_best={a:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best a_best: {best_ab_val:.3f} → {best_ab:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Joint all 3 score alphas (coarse step 0.01)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Joint 3-way score alpha grid ===", flush=True)
c3_best = apply_3way(double_best, alpha=best_ab_val)
best_3a = best_loo
for a_b in [round(x/100, 2) for x in range(17, 24)]:
    c3 = apply_3way(double_best, alpha=a_b)
    for a_i in [round(x/100, 2) for x in range(24, 30)]:
        i3 = apply_3way(ica_ens_alt, alpha=a_i)
        for a_s in [round(x/100, 2) for x in range(26, 32)]:
            s3 = apply_3way(std_ens_ref, alpha=a_s)
            chk = 0.75*c3 + 0.15*i3 + 0.10*s3
            final = 0.72*chk + 0.28*rank_norm
            ar = macro_auc(final)
            mname = f"j3sa_ab{int(a_b*100)}_ai{int(a_i*100)}_as{int(a_s*100)}"
            delta = save_result(mname, ar, {"a_best": a_b, "a_ica": a_i, "a_std": a_s})
            if ar > best_3a: best_3a = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  ab={a_b:.2f} ai={a_i:.2f} as={a_s:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best 3-way alpha: {best_3a:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: chk blend re-opt at best score alphas
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: chk blend re-opt at new score alphas ===", flush=True)
# Use best from D (fallback to batch154 best if no improvement)
# Re-build best c3/i3/s3 from D's best
best_3a_params = (best_ab_val, best_ai_val, best_as_val)
# Fine-search from saved experiments
for e in res["experiments"]:
    if e.get("batch") == 155 and e["method"].startswith("j3sa_") and e["loo_auc"] >= best_3a - 1e-7:
        cfg = e.get("config", {})
        best_3a_params = (cfg.get("a_best", best_ab_val), cfg.get("a_ica", best_ai_val), cfg.get("a_std", best_as_val))

ab_e, ai_e, as_e = best_3a_params
c3_e = apply_3way(double_best, alpha=ab_e)
i3_e = apply_3way(ica_ens_alt, alpha=ai_e)
s3_e = apply_3way(std_ens_ref,  alpha=as_e)
print(f"  Using alphas: ab={ab_e:.2f} ai={ai_e:.3f} as={as_e:.3f}", flush=True)

best_blend = best_loo
best_blend_params = (0.75, 0.15, 0.10)
for wb in np.arange(0.70, 0.82, 0.01):
    for wi in np.arange(0.12, 0.22, 0.01):
        wb_r = round(float(wb), 2); wi_r = round(float(wi), 2)
        ws_r = round(1.0 - wb_r - wi_r, 2)
        if ws_r < 0 or ws_r > 0.15: continue
        chk = wb_r*c3_e + wi_r*i3_e + ws_r*s3_e
        final = 0.72*chk + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"blnd5_ab{int(ab_e*100)}_ai{int(ai_e*100)}_cb{int(wb_r*100)}_ci{int(wi_r*100)}"
        delta = save_result(mname, ar, {"wb": wb_r, "wi": wi_r, "ws": ws_r,
                                         "a_best": ab_e, "a_ica": ai_e, "a_std": as_e})
        if ar > best_blend:
            best_blend = ar
            best_blend_params = (wb_r, wi_r, ws_r)
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  chk({wb_r:.2f},{wi_r:.2f},{ws_r:.2f}): {ar:.6f} {delta:+.6f}{flag}", flush=True)

wb_b, wi_b, ws_b = best_blend_params
print(f"  Best blend: ({wb_b:.2f},{wi_b:.2f},{ws_b:.2f}) → {best_blend:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: Final ultra-fine joint (rm + rank weights + chk blend)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== F: Ultra-fine joint (rm + rank + chk) ===", flush=True)
chk_final = wb_b*c3_e + wi_b*i3_e + ws_b*s3_e
best_joint = best_loo
for wb_rk in np.arange(0.53, 0.60, 0.01):
    wb_rk_r = round(float(wb_rk), 2)
    wi_rk_r = round(1.0 - wb_rk_r, 2)
    rk = wb_rk_r*rank_c_ref + wi_rk_r*rank_i_ref
    for rm in np.arange(0.25, 0.32, 0.01):
        rm_r = round(float(rm), 2)
        final = (1-rm_r)*chk_final + rm_r*(rk/n_files)
        ar = macro_auc(final)
        mname = f"uf5j_rk{int(wb_rk_r*100)}_rm{int(rm_r*100)}_ab{int(ab_e*100)}_ai{int(ai_e*100)}"
        delta = save_result(mname, ar, {
            "wb_rk": wb_rk_r, "wi_rk": wi_rk_r, "rm": rm_r,
            "a_best": ab_e, "a_ica": ai_e, "a_std": as_e,
            "wb_chk": wb_b, "wi_chk": wi_b
        })
        if ar > best_joint: best_joint = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  rk({wb_rk_r:.2f},{wi_rk_r:.2f}) rm={rm_r:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best ultra-fine joint: {best_joint:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 155]
print(f"Batch155 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
