"""
batch153 — Ultra-fine joint optimization around batch152 best
===============================================================================
Current best: jnt2_rk55_cb76_ci15_rm28 LOO=0.995741
  rank = 0.55×rank_c + 0.45×rank_i
  rank alphas: a_rank_c=0.200, a_rank_i=0.400
  chk = 0.76×c3 + 0.15×i3 + 0.09×s3
  final = 0.72×chk + 0.28×(rank/n_files)
  score alphas: a_best=0.200, a_ica=0.255, a_std=0.280

batch152 findings:
- Rank weights optimum: (0.55, 0.45) no STD rank
- chk blend optimum: (0.76, 0.15, 0.09) — STD weight increased from 0.04→0.09!
- rm optimum: 0.28 (slightly lower than 0.30)
- Second best: rk55_cb77_ci14_rm28 → tied at 0.995741

Directions:
 A: Ultra-fine rank weights (0.50-0.60 step 0.005)
 B: Fine-tune rm (0.24-0.34 step 0.005)
 C: Ultra-fine chk blend near (0.76, 0.15, 0.09)
 D: Fine-tune a_rank_i (0.35-0.50 step 0.01) at new rank/chk config
 E: Fine-tune a_rank_c (0.15-0.30 step 0.01)
 F: Comprehensive ultra-fine joint
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
print(f"[batch153] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 153}
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

# Reference rank components (best152 alphas)
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.200))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.400))

chk_ref = 0.76*c3_score + 0.15*i3_score + 0.09*s3_score
rank_ref = 0.55*rank_c_ref + 0.45*rank_i_ref

v = 0.72*chk_ref + 0.28*(rank_ref/n_files)
print(f"Verify: {macro_auc(v):.6f} (expect ~0.995741)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Ultra-fine rank weights (0.50-0.60 step 0.005)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Ultra-fine rank weights (step 0.005) ===", flush=True)
best_rw = best_loo
best_rw_params = (0.55, 0.45)
for wb_int in range(100, 121):  # 0.50-0.60 step 0.005
    wb_r = wb_int / 200.0
    wi_r = round(1.0 - wb_r, 3)
    rk = wb_r*rank_c_ref + wi_r*rank_i_ref
    final = 0.72*chk_ref + 0.28*(rk/n_files)
    ar = macro_auc(final)
    mname = f"ufrkw_wb{int(wb_r*1000)}_rm28"
    delta = save_result(mname, ar, {"wb_rk": wb_r, "wi_rk": wi_r, "rm": 0.28})
    if ar > best_rw:
        best_rw = ar
        best_rw_params = (wb_r, wi_r)
    if ar > best_loo - 0.00004:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  wb={wb_r:.3f} wi={wi_r:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

wb_rk_best, wi_rk_best = best_rw_params
print(f"  Best rank weights: ({wb_rk_best:.3f}, {wi_rk_best:.3f}) → {best_rw:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Fine-tune rm (0.22-0.34 step 0.005)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Fine-tune rm at best rank weights ===", flush=True)
rk_best_blend = wb_rk_best*rank_c_ref + wi_rk_best*rank_i_ref
best_rm = best_loo
best_rm_val = 0.28
for rm_int in range(44, 69):  # 0.220-0.345 step 0.005
    rm = rm_int / 200.0
    final = (1-rm)*chk_ref + rm*(rk_best_blend/n_files)
    ar = macro_auc(final)
    mname = f"ufrm_wb{int(wb_rk_best*1000)}_rm{rm_int}"
    delta = save_result(mname, ar, {"wb_rk": wb_rk_best, "rm": rm})
    if ar > best_rm:
        best_rm = ar
        best_rm_val = rm
    if ar > best_loo - 0.00004:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  rm={rm:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best rm: {best_rm_val:.3f} → {best_rm:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Ultra-fine chk blend near (0.76, 0.15, 0.09)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Ultra-fine chk blend ===", flush=True)
best_chk = best_loo
best_chk_params = (0.76, 0.15, 0.09)
rk_norm = rk_best_blend / n_files
for wb in np.arange(0.73, 0.80, 0.01):
    for wi in np.arange(0.12, 0.20, 0.01):
        wb_r = round(float(wb), 2); wi_r = round(float(wi), 2)
        ws_r = round(1.0 - wb_r - wi_r, 2)
        if ws_r < 0 or ws_r > 0.15: continue
        chk = wb_r*c3_score + wi_r*i3_score + ws_r*s3_score
        final = (1-best_rm_val)*chk + best_rm_val*rk_norm
        ar = macro_auc(final)
        mname = f"ufchk_cb{int(wb_r*100)}_ci{int(wi_r*100)}_rm{int(best_rm_val*1000)}"
        delta = save_result(mname, ar, {"wb_chk": wb_r, "wi_chk": wi_r, "ws_chk": ws_r, "rm": best_rm_val})
        if ar > best_chk:
            best_chk = ar
            best_chk_params = (wb_r, wi_r, ws_r)
        if ar > best_loo - 0.00004:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  chk({wb_r:.2f},{wi_r:.2f},{ws_r:.2f}) rm={best_rm_val:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

wb_c, wi_c, ws_c = best_chk_params
print(f"  Best chk blend: ({wb_c:.2f}, {wi_c:.2f}, {ws_c:.2f}) → {best_chk:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Fine-tune a_rank_i (0.35-0.50 step 0.01)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Fine-tune a_rank_i (0.35-0.50 step 0.01) ===", flush=True)
alpha_i_range = [round(x/100, 2) for x in range(33, 53, 1)]
best_ai = best_loo
best_ai_val = 0.40
chk_best = wb_c*c3_score + wi_c*i3_score + ws_c*s3_score
for a in alpha_i_range:
    rk_i = make_rank(apply_3way(ica_ens_alt, alpha=a))
    rk = wb_rk_best*rank_c_ref + wi_rk_best*rk_i
    final = (1-best_rm_val)*chk_best + best_rm_val*(rk/n_files)
    ar = macro_auc(final)
    mname = f"ufai_ai{int(a*100)}_ac200_rm{int(best_rm_val*1000)}"
    delta = save_result(mname, ar, {"a_rank_i": a, "a_rank_c": 0.200})
    if ar > best_ai:
        best_ai = ar
        best_ai_val = a
    if ar > best_loo - 0.00004:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_rank_i={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best a_rank_i: {best_ai_val:.2f} → {best_ai:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Fine-tune a_rank_c (0.15-0.35 step 0.01)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Fine-tune a_rank_c at best a_rank_i={best_ai_val:.2f} ===", flush=True)
alpha_c_range = [round(x/100, 2) for x in range(13, 38, 1)]
best_ac = best_loo
best_ac_val = 0.20
rk_i_best = make_rank(apply_3way(ica_ens_alt, alpha=best_ai_val))
for a in alpha_c_range:
    rk_c = make_rank(apply_3way(double_best, alpha=a))
    rk = wb_rk_best*rk_c + wi_rk_best*rk_i_best
    final = (1-best_rm_val)*chk_best + best_rm_val*(rk/n_files)
    ar = macro_auc(final)
    mname = f"ufac_ac{int(a*100)}_ai{int(best_ai_val*100)}_rm{int(best_rm_val*1000)}"
    delta = save_result(mname, ar, {"a_rank_c": a, "a_rank_i": best_ai_val})
    if ar > best_ac:
        best_ac = ar
        best_ac_val = a
    if ar > best_loo - 0.00004:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_rank_c={a:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best a_rank_c: {best_ac_val:.2f} → {best_ac:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: Comprehensive ultra-fine joint (all best params together)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== F: Comprehensive joint near global best ===", flush=True)
# Re-compute with best alpha pair
rk_c_best = make_rank(apply_3way(double_best, alpha=best_ac_val))
rk_i_best2 = make_rank(apply_3way(ica_ens_alt, alpha=best_ai_val))

best_joint = best_loo
for wb_rk in np.arange(max(0.48, wb_rk_best-0.04), min(0.62, wb_rk_best+0.05), 0.01):
    wb_rk_r = round(float(wb_rk), 2)
    wi_rk_r = round(1.0 - wb_rk_r, 2)
    rk = wb_rk_r*rk_c_best + wi_rk_r*rk_i_best2
    for wb_c2 in np.arange(max(0.73, wb_c-0.03), min(0.80, wb_c+0.04), 0.01):
        for wi_c2 in np.arange(max(0.11, wi_c-0.03), min(0.20, wi_c+0.04), 0.01):
            wb_cr = round(float(wb_c2), 2); wi_cr = round(float(wi_c2), 2)
            ws_cr = round(1.0 - wb_cr - wi_cr, 2)
            if ws_cr < 0 or ws_cr > 0.15: continue
            chk = wb_cr*c3_score + wi_cr*i3_score + ws_cr*s3_score
            for rm in np.arange(max(0.22, best_rm_val-0.04), min(0.36, best_rm_val+0.05), 0.01):
                rm_r = round(float(rm), 2)
                final = (1-rm_r)*chk + rm_r*(rk/n_files)
                ar = macro_auc(final)
                mname = f"uf3j_rk{int(wb_rk_r*100)}_cb{int(wb_cr*100)}_ci{int(wi_cr*100)}_rm{int(rm_r*100)}"
                delta = save_result(mname, ar, {
                    "wb_rk": wb_rk_r, "wi_rk": wi_rk_r,
                    "wb_chk": wb_cr, "wi_chk": wi_cr, "ws_chk": ws_cr, "rm": rm_r,
                    "a_rank_c": best_ac_val, "a_rank_i": best_ai_val
                })
                if ar > best_joint: best_joint = ar
                if ar > best_loo - 0.00004:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  rk({wb_rk_r:.2f},{wi_rk_r:.2f}) chk({wb_cr:.2f},{wi_cr:.2f},{ws_cr:.2f}) rm={rm_r:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best joint: {best_joint:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 153]
print(f"Batch153 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
