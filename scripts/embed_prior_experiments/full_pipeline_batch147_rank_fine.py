"""
batch147 — Rank-mix fine-tune + rank blend weights + power transform variants
===============================================================================
Current best: rank_mix_2 LOO=0.995352 (+0.000054)
  final = 0.8×chk + 0.2×(rank_blend/n_files)
  rank_blend = 0.75×rank_c + 0.23×rank_i + 0.02×rank_s

batch146 key insight: rank-based component mixing helps!
- rank_mix=0.2 gives 0.995352 (best)
- rank_mix=0.1 gives 0.995339
- rank_mix=0.3 gives 0.995312

Directions:
 A: Fine-tune rank_mix weight (0.10-0.35 step 0.01)
 B: Different rank blend weights (vs 0.75/0.23/0.02)
 C: Per-species ranking + blend
 D: Rank + linear combo tuning
 E: Power transform instead of rank (raise predictions to power before blend)
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
print(f"[batch147] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 147}
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

# Reference best components
c3_ref = apply_3way(double_best, alpha=0.200)
i3_ref = apply_3way(ica_ens_alt, alpha=0.255)
s3_ref = apply_3way(std_ens_ref,  alpha=0.280)
chk = blend3(c3_ref, i3_ref, s3_ref)

# Rank components
def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

rank_c = make_rank(c3_ref)
rank_i = make_rank(i3_ref)
rank_s = make_rank(s3_ref)
rank_blend_ref = 0.75*rank_c + 0.23*rank_i + 0.02*rank_s

# Verify
chk_r = 0.8*chk + 0.2*(rank_blend_ref / n_files)
print(f"Verify: {macro_auc(chk_r):.6f} (expect 0.995352)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Fine-tune rank_mix weight
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Fine-tune rank_mix ===", flush=True)
rank_norm = rank_blend_ref / n_files

best_rm = best_loo
best_rm_val = 0.20
for rm_int in range(5, 50, 1):
    rm = rm_int / 100.0
    final = (1-rm)*chk + rm*rank_norm
    ar = macro_auc(final)
    mname = f"rkmix_{rm_int:02d}"
    delta = save_result(mname, ar, {"rank_mix": rm})
    if ar > best_rm:
        best_rm = ar
        best_rm_val = rm
    if ar > best_loo - 0.00003:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  rank_mix={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best rank_mix: {best_rm_val:.2f} → {best_rm:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Different rank blend weights
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Rank blend weights (at best rank_mix={best_rm_val:.2f}) ===", flush=True)
best_rw = best_loo
for wb in np.arange(0.65, 0.90, 0.05):
    for wi in np.arange(0.10, 0.30, 0.05):
        ws = round(1.0 - round(float(wb), 2) - round(float(wi), 2), 2)
        wb_r = round(float(wb), 2); wi_r = round(float(wi), 2)
        if ws < 0 or ws > 0.20: continue
        rb = wb_r*rank_c + wi_r*rank_i + ws*rank_s
        final = (1-best_rm_val)*chk + best_rm_val*(rb/n_files)
        ar = macro_auc(final)
        mname = f"rkwt_b{int(wb_r*100)}_i{int(wi_r*100)}_s{int(ws*100)}_rm{int(best_rm_val*100)}"
        delta = save_result(mname, ar, {"wb": wb_r, "wi": wi_r, "ws": ws, "rm": best_rm_val})
        if ar > best_rw: best_rw = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  wb={wb_r:.2f} wi={wi_r:.2f} ws={ws:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best rank weights: {best_rw:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Joint: rank blend weights + rank_mix
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Joint rank_mix × blend weights ===", flush=True)
# Try best blends with finer rank_mix range
wb_c = 0.75; wi_c = 0.23; ws_c = 0.02  # start from original
best_joint = best_loo
for rm in [0.15, 0.18, 0.20, 0.22, 0.25]:
    for wb_v in [0.70, 0.75, 0.80]:
        for wi_v in [0.20, 0.23, 0.26]:
            ws_v = round(1.0 - wb_v - wi_v, 2)
            if ws_v < 0 or ws_v > 0.15: continue
            rb = wb_v*rank_c + wi_v*rank_i + ws_v*rank_s
            final = (1-rm)*chk + rm*(rb/n_files)
            ar = macro_auc(final)
            mname = f"rkjnt_rm{int(rm*100)}_b{int(wb_v*100)}_i{int(wi_v*100)}"
            delta = save_result(mname, ar, {"rm": rm, "wb": wb_v, "wi": wi_v, "ws": ws_v})
            if ar > best_joint: best_joint = ar
            if ar > best_loo - 0.00004:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  rm={rm:.2f} wb={wb_v:.2f} wi={wi_v:.2f} ws={ws_v:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best joint rank: {best_joint:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Power transform instead of rank
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Power transform blend ===", flush=True)
best_pow = best_loo
for pw in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    # raise each component to power pw before blend
    c3_pw = np.clip(c3_ref, 0, 1)**pw
    i3_pw = np.clip(i3_ref, 0, 1)**pw
    s3_pw = np.clip(s3_ref, 0, 1)**pw
    # normalize
    c3_pn = c3_pw / (c3_pw.max(axis=0, keepdims=True) + EPS)
    i3_pn = i3_pw / (i3_pw.max(axis=0, keepdims=True) + EPS)
    s3_pn = s3_pw / (s3_pw.max(axis=0, keepdims=True) + EPS)
    pow_blend = 0.75*c3_pn + 0.23*i3_pn + 0.02*s3_pn
    for pm in [0.1, 0.2, 0.3]:
        final = (1-pm)*chk + pm*pow_blend
        ar = macro_auc(final)
        mname = f"pow_{int(pw*10)}_m{int(pm*10)}"
        delta = save_result(mname, ar, {"power": pw, "mix": pm})
        if ar > best_pow: best_pow = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  power={pw:.1f} mix={pm:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best power: {best_pow:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 147]
print(f"Batch147 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
