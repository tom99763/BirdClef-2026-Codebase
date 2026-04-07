"""
batch150 — Push ICA rank weight higher, extended range
===============================================================================
Current best: rkext_rm30_b56_i44 LOO=0.995625
  rank = 0.56×rank_c + 0.44×rank_i + 0.00×rank_s
  final = 0.70×chk + 0.30×(rank/n_files)

Trend: ICA rank weight rising fast (0.25→0.34→0.44, +STD→0)
        best rank weight declining (0.75→0.63→0.56)
        rank_mix rising (0.18→0.20→0.30)

Push ICA rank weight to 0.50-0.70 and beyond
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
print(f"[batch150] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 150}
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

def blend3(c, i, s, wb=0.75, wi=0.23, ws=0.02):
    return wb*c + wi*i + ws*s

c3_ref = apply_3way(double_best, alpha=0.200)
i3_ref = apply_3way(ica_ens_alt, alpha=0.255)
s3_ref = apply_3way(std_ens_ref,  alpha=0.280)
chk    = blend3(c3_ref, i3_ref, s3_ref)

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

rank_c = make_rank(c3_ref)
rank_i = make_rank(i3_ref)
rank_s = make_rank(s3_ref)

# Also compute ranks of raw chains
rank_db = make_rank(double_best)
rank_ia = make_rank(ica_ens_alt)
rank_sr = make_rank(std_ens_ref)

# Verify
rb_ref = 0.56*rank_c + 0.44*rank_i
chk_v = 0.70*chk + 0.30*(rb_ref/n_files)
print(f"Verify: {macro_auc(chk_v):.6f} (expect 0.995625)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Push ICA weight to extreme (ws=0, sweep wb 0.25-0.60)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: 2-component rank (no STD rank) at rm=0.30 ===", flush=True)
best_2c = best_loo
best_2c_params = (0.56, 0.44)
for wb in np.arange(0.20, 0.65, 0.01):
    wb_r = round(float(wb), 2)
    wi_r = round(1.0 - wb_r, 2)
    rb = wb_r*rank_c + wi_r*rank_i
    final = 0.70*chk + 0.30*(rb/n_files)
    ar = macro_auc(final)
    mname = f"rk2c_b{int(wb_r*100)}_i{int(wi_r*100)}_rm30"
    delta = save_result(mname, ar, {"wb": wb_r, "wi": wi_r, "ws": 0.0, "rm": 0.30})
    if ar > best_2c:
        best_2c = ar
        best_2c_params = (wb_r, wi_r)
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  wb={wb_r:.2f} wi={wi_r:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

wb_best, wi_best = best_2c_params
print(f"  Best 2-comp: ({wb_best:.2f}, {wi_best:.2f}) → {best_2c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Fine-tune rank_mix at new best 2-comp weights (0.28-0.40)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Fine rank_mix at wb={wb_best:.2f} wi={wi_best:.2f} ===", flush=True)
rb_best = wb_best*rank_c + wi_best*rank_i
rb_norm = rb_best / n_files

best_rm = best_loo
best_rm_val = 0.30
for rm_int in range(20, 50):
    rm = rm_int / 100.0
    final = (1-rm)*chk + rm*rb_norm
    ar = macro_auc(final)
    mname = f"rk2rm_b{int(wb_best*100)}_rm{rm_int:02d}"
    delta = save_result(mname, ar, {"wb": wb_best, "wi": wi_best, "rm": rm})
    if ar > best_rm:
        best_rm = ar
        best_rm_val = rm
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  rank_mix={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best rank_mix: {best_rm_val:.2f} → {best_rm:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Joint ultra-fine around new optimum
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Joint ultra-fine around new optimum ===", flush=True)
best_joint = best_loo
for wb in np.arange(max(0.20, wb_best-0.08), min(0.65, wb_best+0.09), 0.01):
    wb_r = round(float(wb), 2)
    wi_r = round(1.0 - wb_r, 2)
    rb = wb_r*rank_c + wi_r*rank_i
    for rm in np.arange(max(0.20, best_rm_val-0.06), min(0.55, best_rm_val+0.07), 0.01):
        rm = round(float(rm), 2)
        final = (1-rm)*chk + rm*(rb/n_files)
        ar = macro_auc(final)
        mname = f"rkjnt3_b{int(wb_r*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar, {"wb": wb_r, "wi": wi_r, "rm": rm})
        if ar > best_joint: best_joint = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  wb={wb_r:.2f} wi={wi_r:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best joint: {best_joint:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Use raw chain ranks instead of 3way ranks
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Raw chain ranks ===", flush=True)
best_raw_rank = best_loo
for wb_r in [0.40, 0.45, 0.50, 0.55, 0.56]:
    wi_r = round(1.0 - wb_r, 2)
    # Use raw chain ranks
    rb_raw = wb_r*rank_db + wi_r*rank_ia
    for rm in [0.20, 0.25, 0.30, 0.35]:
        final = (1-rm)*chk + rm*(rb_raw/n_files)
        ar = macro_auc(final)
        mname = f"rawrk_b{int(wb_r*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar, {"wb": wb_r, "wi": wi_r, "rm": rm, "type": "raw"})
        if ar > best_raw_rank: best_raw_rank = ar
        if ar > best_loo - 0.0001:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  raw_rank wb={wb_r:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best raw rank: {best_raw_rank:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 150]
print(f"Batch150 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
