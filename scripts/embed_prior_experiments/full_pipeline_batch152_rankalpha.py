"""
batch152 — Fine-tune rank alpha + comprehensive joint optimization
===============================================================================
Current best: rkalpha_ac200_ai400 LOO=0.995715 (+0.000090)
  rank computed with: a_rank_c=0.200, a_rank_i=0.400
  rank = 0.56×rank_c + 0.44×rank_i
  chk = 0.77×c3 + 0.19×i3 + 0.04×s3
  final = 0.70×chk + 0.30×(rank/n_files)
  score computed with: a_best=0.200, a_ica=0.255, a_std=0.280

Insight: rank computation benefits from higher alpha (more co-occurrence signal)

Directions:
 A: Fine-tune a_rank_i (0.350-0.500 step 0.025) at fixed other params
 B: Fine-tune a_rank_c (0.150-0.300 step 0.025) at fixed other params
 C: Comprehensive joint: rank alpha + rank weights + rank_mix + chk blend
 D: Even higher a_rank_i (0.400-0.600)
 E: Use separate alpha for two_round in rank computation
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
print(f"[batch152] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 152}
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
c3_score = apply_3way(double_best, alpha=0.200)  # for chk
i3_score = apply_3way(ica_ens_alt, alpha=0.255)  # for chk
s3_score = apply_3way(std_ens_ref,  alpha=0.280)  # for chk

# Reference rank components
c3_rank_ref = apply_3way(double_best, alpha=0.200)  # same as score for best
i3_rank_ref = apply_3way(ica_ens_alt, alpha=0.400)  # higher alpha for rank
rank_c_ref = make_rank(c3_rank_ref)
rank_i_ref = make_rank(i3_rank_ref)

chk_ref = 0.77*c3_score + 0.19*i3_score + 0.04*s3_score
rank_ref = 0.56*rank_c_ref + 0.44*rank_i_ref

v = 0.70*chk_ref + 0.30*(rank_ref/n_files)
print(f"Verify: {macro_auc(v):.6f} (expect 0.995715)\n", flush=True)

t0 = time.time()

# Pre-compute 3way at many alphas for rank computation
print("Pre-computing 3way components for rank at various alphas...", flush=True)
alpha_for_rank = [round(x/100, 2) for x in range(20, 65, 5)]  # 0.20-0.60 step 0.05
cache_c_rk = {a: make_rank(apply_3way(double_best, alpha=a)) for a in alpha_for_rank}
cache_i_rk = {a: make_rank(apply_3way(ica_ens_alt, alpha=a)) for a in alpha_for_rank}
cache_s_rk = {a: make_rank(apply_3way(std_ens_ref, alpha=a)) for a in alpha_for_rank}
print(f"  Cached {len(alpha_for_rank)} alpha values for rank", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# A: Fine-tune a_rank_i (0.30-0.55 step 0.025)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== A: Fine-tune a_rank_i ===", flush=True)
best_ari = best_loo
best_ari_val = 0.40
for a in alpha_for_rank:
    rk_blend = 0.56*rank_c_ref + 0.44*cache_i_rk[a]
    final = 0.70*chk_ref + 0.30*(rk_blend/n_files)
    ar = macro_auc(final)
    mname = f"rkai_ai{int(a*100)}_ac200"
    delta = save_result(mname, ar, {"a_rank_i": a, "a_rank_c": 0.200})
    if ar > best_ari:
        best_ari = ar
        best_ari_val = a
    if ar > best_loo - 0.00006:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_rank_i={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best a_rank_i: {best_ari_val:.2f} → {best_ari:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Fine-tune a_rank_c at best a_rank_i
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Fine-tune a_rank_c at a_rank_i={best_ari_val:.2f} ===", flush=True)
best_arc = best_loo
best_arc_val = 0.200
for a in alpha_for_rank:
    rk_blend = 0.56*cache_c_rk[a] + 0.44*cache_i_rk[best_ari_val]
    final = 0.70*chk_ref + 0.30*(rk_blend/n_files)
    ar = macro_auc(final)
    mname = f"rkac_ac{int(a*100)}_ai{int(best_ari_val*100)}"
    delta = save_result(mname, ar, {"a_rank_c": a, "a_rank_i": best_ari_val})
    if ar > best_arc:
        best_arc = ar
        best_arc_val = a
    if ar > best_loo - 0.00006:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_rank_c={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best a_rank_c: {best_arc_val:.2f} → {best_arc:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Joint grid: rank weights + chk blend at new rank alphas
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Joint rank weights + chk blend at ac={best_arc_val:.2f} ai={best_ari_val:.2f} ===", flush=True)
rk_c_best = cache_c_rk[best_arc_val]
rk_i_best = cache_i_rk[best_ari_val]

best_joint = best_loo
for wb_rk in np.arange(0.45, 0.70, 0.01):
    wb_rk_r = round(float(wb_rk), 2)
    wi_rk_r = round(1.0 - wb_rk_r, 2)
    rk_blend = wb_rk_r*rk_c_best + wi_rk_r*rk_i_best
    for wb_c in np.arange(0.70, 0.84, 0.01):
        for wi_c in np.arange(0.14, 0.25, 0.01):
            wb_cr = round(float(wb_c), 2); wi_cr = round(float(wi_c), 2)
            ws_cr = round(1.0 - wb_cr - wi_cr, 2)
            if ws_cr < 0 or ws_cr > 0.15: continue
            chk = wb_cr*c3_score + wi_cr*i3_score + ws_cr*s3_score
            for rm in [0.28, 0.30, 0.32]:
                final = (1-rm)*chk + rm*(rk_blend/n_files)
                ar = macro_auc(final)
                mname = f"jnt2_rk{int(wb_rk_r*100)}_cb{int(wb_cr*100)}_ci{int(wi_cr*100)}_rm{int(rm*100)}"
                delta = save_result(mname, ar, {
                    "wb_rk": wb_rk_r, "wi_rk": wi_rk_r,
                    "wb_chk": wb_cr, "wi_chk": wi_cr, "ws_chk": ws_cr, "rm": rm
                })
                if ar > best_joint: best_joint = ar
                if ar > best_loo - 0.00004:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  rk({wb_rk_r},{wi_rk_r}) chk({wb_cr},{wi_cr},{ws_cr}) rm={rm}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best joint: {best_joint:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 152]
print(f"Batch152 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
