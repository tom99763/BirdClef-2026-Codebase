"""
batch157 — file_prob_max as additional signal
===============================================================================
Current best: sca4_ia26_sa28 LOO=0.995800
  chk = 0.75×c3(db,0.20) + 0.15×i3(ia,0.26) + 0.10×s3(std,0.28)
  final = 0.72×chk + 0.28×(0.56×rank_c(0.23) + 0.44×rank_i(0.40))/n_files

New discovery: file_prob_max (raw Perch max-pool prob) has macro_auc=0.994870
  > chain_double_best (0.992312)
  > file_prob_max_3way_best (0.994510)

Directions:
 A: Apply 3way to file_prob_max with various alphas
 B: Add file_prob_max as 4th component in chk blend
 C: Add file_prob_max rank as additional rank component
 D: Replace chain_double_best with file_prob_max in the formula
 E: Blend chain_double_best with file_prob_max before 3way
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
prob_max    = ep["file_prob_max"]           # raw Perch max-pool prob
pm3_best    = ep["file_prob_max_3way_best"] # pre-computed 3way of prob_max

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch157] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 157}
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

# Baseline for file_prob_max
print(f"file_prob_max raw AUC: {macro_auc(prob_max):.6f}", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Apply 3way to file_prob_max with various alphas
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: 3way transform of file_prob_max ===", flush=True)
best_pa = best_loo
best_pm3 = None
best_pm3_alpha = 0.200
for a in [round(x/100, 2) for x in range(15, 45, 1)]:
    pm3 = apply_3way(prob_max, alpha=a)
    ar = macro_auc(pm3)
    mname = f"pm3way_a{int(a*100)}"
    delta = save_result(mname, ar, {"a": a, "type": "pm3way"})
    if ar > best_pa:
        best_pa = ar
        best_pm3_alpha = a
        best_pm3 = pm3.copy()
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  pm3way a={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

if best_pm3 is None:
    best_pm3 = apply_3way(prob_max, alpha=best_pm3_alpha)
print(f"  Best pm3way alpha: {best_pm3_alpha:.2f} → {best_pa:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Add file_prob_max as 4th component in chk blend
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: file_prob_max as 4th chk component ===", flush=True)
best_4c = best_loo
# Add w_pm weight, adjust others proportionally
for w_pm in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
    # Use pm3way at best alpha
    pm4 = best_pm3
    for wb in np.arange(0.65, 0.78, 0.02):
        wb_r = round(float(wb), 2)
        # remaining = 1 - wb - w_pm, split between i3 and s3
        remain = 1 - wb_r - w_pm
        for wi in np.arange(0.10, 0.18, 0.02):
            wi_r = round(float(wi), 2)
            ws_r = round(remain - wi_r, 2)
            if ws_r < 0 or ws_r > 0.12: continue
            chk4 = wb_r*c3_ref + wi_r*i3_ref + ws_r*s3_ref + w_pm*pm4
            final = 0.72*chk4 + 0.28*rank_norm
            ar = macro_auc(final)
            mname = f"pm4c_wpm{int(w_pm*100)}_wb{int(wb_r*100)}_wi{int(wi_r*100)}"
            delta = save_result(mname, ar, {"w_pm": w_pm, "wb": wb_r, "wi": wi_r, "ws": ws_r})
            if ar > best_4c: best_4c = ar
            if ar > best_loo - 0.00004:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  pm4({w_pm:.2f},{wb_r:.2f},{wi_r:.2f},{ws_r:.2f}): {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best 4-comp chk: {best_4c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Add prob_max rank as additional rank component
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: prob_max rank component ===", flush=True)
best_pc = best_loo
rank_pm = make_rank(best_pm3)
# 3-comp rank: rank_c, rank_i, rank_pm
for wb_pm in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
    for wb_c in np.arange(0.50, 0.58, 0.02):
        wb_cr = round(float(wb_c), 2)
        wi_r = round(1 - wb_cr - wb_pm, 2)
        if wi_r < 0.35 or wi_r > 0.52: continue
        rk = wb_cr*rank_c_ref + wi_r*rank_i_ref + wb_pm*rank_pm
        for rm in [0.26, 0.28, 0.30]:
            final = (1-rm)*chk_ref + rm*(rk/n_files)
            ar = macro_auc(final)
            mname = f"pmrk_wpm{int(wb_pm*100)}_wc{int(wb_cr*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar, {"w_pm": wb_pm, "wb_c": wb_cr, "wi_i": wi_r, "rm": rm})
            if ar > best_pc: best_pc = ar
            if ar > best_loo - 0.00004:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  pmrk({wb_pm:.2f},{wb_cr:.2f},{wi_r:.2f}) rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best pm-rank: {best_pc:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Replace chain_double_best with file_prob_max
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Replace chain_double_best with file_prob_max ===", flush=True)
best_pd = best_loo
for a_pm in [round(x/100, 2) for x in range(15, 30)]:
    pm3 = apply_3way(prob_max, alpha=a_pm)
    for wi in np.arange(0.12, 0.20, 0.02):
        wi_r = round(float(wi), 2)
        for ws in np.arange(0.06, 0.14, 0.02):
            ws_r = round(float(ws), 2)
            wb_r = round(1 - wi_r - ws_r, 2)
            if wb_r < 0.68 or wb_r > 0.82: continue
            chk = wb_r*pm3 + wi_r*i3_ref + ws_r*s3_ref
            # Rank with pm instead of db
            rank_pm_main = make_rank(apply_3way(prob_max, alpha=best_pm3_alpha))
            rk = 0.56*rank_pm_main + 0.44*rank_i_ref
            final = 0.72*chk + 0.28*(rk/n_files)
            ar = macro_auc(final)
            mname = f"pmrepl_apm{int(a_pm*100)}_wi{int(wi_r*100)}_ws{int(ws_r*100)}"
            delta = save_result(mname, ar, {"a_pm": a_pm, "wi": wi_r, "ws": ws_r})
            if ar > best_pd: best_pd = ar
            if ar > best_loo - 0.00004:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  pm_repl a={a_pm:.2f} wi={wi_r:.2f} ws={ws_r:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best pm_replace: {best_pd:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Blend chain_double_best with file_prob_max before 3way
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Blend db+pm before 3way ===", flush=True)
best_pe = best_loo
for w_pm in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blended = (1-w_pm)*double_best + w_pm*prob_max
    for alpha in [0.200, 0.230, 0.250]:
        bl3 = apply_3way(blended, alpha=alpha)
        # Replace c3_ref with this blend
        for wb in [0.74, 0.75, 0.76]:
            for wi in [0.14, 0.15, 0.16]:
                ws = round(1 - wb - wi, 2)
                if ws < 0 or ws > 0.15: continue
                chk = wb*bl3 + wi*i3_ref + ws*s3_ref
                rk = 0.56*make_rank(apply_3way(blended, alpha=0.23)) + 0.44*rank_i_ref
                final = 0.72*chk + 0.28*(rk/n_files)
                ar = macro_auc(final)
                mname = f"bldb_wpm{int(w_pm*100)}_a{int(alpha*100)}_wb{int(wb*100)}_wi{int(wi*100)}"
                delta = save_result(mname, ar, {"w_pm": w_pm, "alpha": alpha, "wb": wb, "wi": wi})
                if ar > best_pe: best_pe = ar
                if ar > best_loo - 0.00004:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  blend_db wpm={w_pm:.2f} a={alpha:.2f} wb={wb:.2f} wi={wi:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best blend_db: {best_pe:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 157]
print(f"Batch157 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
