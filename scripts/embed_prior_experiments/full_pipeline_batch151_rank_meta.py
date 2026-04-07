"""
batch151 — Optimize chk blend for rank combination + rank with different alpha
===============================================================================
Current best: rkext_rm30_b56_i44 LOO=0.995625
  rank = 0.56×rank_c + 0.44×rank_i
  final = 0.70×chk + 0.30×(rank/n_files)
  chk = blend3(c3_ref, i3_ref, s3_ref) with wb=0.75, wi=0.23, ws=0.02

Key insight: we're optimizing the rank separately from the chk,
but the chk blend weights weren't re-optimized for this combination.

Directions:
 A: Re-optimize chk blend weights for rank combination
 B: Different alpha for rank computation (ranks of higher-alpha predictions)
 C: Use different per-comp alpha for rank vs score
 D: Rank of chk (self-rank boost)
 E: ICA-heavy chk + ICA-heavy rank together
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
print(f"[batch151] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 151}
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

# Standard 3way components
c3_ref = apply_3way(double_best, alpha=0.200)
i3_ref = apply_3way(ica_ens_alt, alpha=0.255)
s3_ref = apply_3way(std_ens_ref,  alpha=0.280)

rank_c = make_rank(c3_ref)
rank_i = make_rank(i3_ref)
rank_s = make_rank(s3_ref)
rank_ref = 0.56*rank_c + 0.44*rank_i

def chk_from_blend(wb, wi, ws):
    return wb*c3_ref + wi*i3_ref + ws*s3_ref

chk_std = chk_from_blend(0.75, 0.23, 0.02)

# Verify
v = 0.70*chk_std + 0.30*(rank_ref/n_files)
print(f"Verify: {macro_auc(v):.6f} (expect 0.995625)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Re-optimize chk blend weights for the rank combination
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Re-optimize chk blend weights (rm=0.30) ===", flush=True)
best_chk = best_loo
best_chk_params = (0.75, 0.23, 0.02)
for wb in np.arange(0.55, 0.86, 0.01):
    for wi in np.arange(0.14, 0.40, 0.01):
        wb_r = round(float(wb), 2); wi_r = round(float(wi), 2)
        ws_r = round(1.0 - wb_r - wi_r, 2)
        if ws_r < 0 or ws_r > 0.20: continue
        chk = chk_from_blend(wb_r, wi_r, ws_r)
        final = 0.70*chk + 0.30*(rank_ref/n_files)
        ar = macro_auc(final)
        mname = f"chkopt_b{int(wb_r*100)}_i{int(wi_r*100)}_s{int(ws_r*100)}"
        delta = save_result(mname, ar, {"wb": wb_r, "wi": wi_r, "ws": ws_r, "rm": 0.30})
        if ar > best_chk:
            best_chk = ar
            best_chk_params = (wb_r, wi_r, ws_r)
        if ar > best_loo - 0.00004:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  wb={wb_r:.2f} wi={wi_r:.2f} ws={ws_r:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

wb_c, wi_c, ws_c = best_chk_params
print(f"  Best chk blend: ({wb_c:.2f}, {wi_c:.2f}, {ws_c:.2f}) → {best_chk:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Different alpha for rank computation
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Different alpha for rank computation ===", flush=True)
best_rka = best_loo
for a_rank_c in [0.150, 0.200, 0.250, 0.300, 0.350, 0.400]:
    for a_rank_i in [0.200, 0.255, 0.300, 0.350, 0.400, 0.450]:
        c3_rk = apply_3way(double_best, alpha=a_rank_c)
        i3_rk = apply_3way(ica_ens_alt, alpha=a_rank_i)
        rk_c = make_rank(c3_rk)
        rk_i = make_rank(i3_rk)
        rb = 0.56*rk_c + 0.44*rk_i
        chk = chk_from_blend(wb_c, wi_c, ws_c)
        final = 0.70*chk + 0.30*(rb/n_files)
        ar = macro_auc(final)
        mname = f"rkalpha_ac{int(a_rank_c*1000)}_ai{int(a_rank_i*1000)}"
        delta = save_result(mname, ar, {"a_rank_c": a_rank_c, "a_rank_i": a_rank_i})
        if ar > best_rka: best_rka = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_rank_c={a_rank_c:.3f} a_rank_i={a_rank_i:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best rank alpha: {best_rka:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Independent re-optimization: best chk + best rank weights + best rm
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Joint chk + rank + rm optimization ===", flush=True)
best_joint = best_loo
chk_best = chk_from_blend(wb_c, wi_c, ws_c)
for wb_r in np.arange(0.48, 0.65, 0.01):
    wi_r = round(1.0 - round(float(wb_r), 2), 2)
    wb_rr = round(float(wb_r), 2)
    rb = wb_rr*rank_c + wi_r*rank_i
    for rm in np.arange(0.25, 0.40, 0.01):
        rm_r = round(float(rm), 2)
        final = (1-rm_r)*chk_best + rm_r*(rb/n_files)
        ar = macro_auc(final)
        mname = f"jopt_b{int(wb_rr*100)}_i{int(wi_r*100)}_rm{int(rm_r*100)}_cb{int(wb_c*100)}"
        delta = save_result(mname, ar, {"wb_rank": wb_rr, "wi_rank": wi_r, "rm": rm_r,
                                         "wb_chk": wb_c, "wi_chk": wi_c})
        if ar > best_joint: best_joint = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  rank ({wb_rr:.2f},{wi_r:.2f}) rm={rm_r:.2f} chk({wb_c:.2f},{wi_c:.2f}): {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best joint optimization: {best_joint:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 151]
print(f"Batch151 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
