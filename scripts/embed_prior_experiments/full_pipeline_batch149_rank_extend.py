"""
batch149 — Extended rank weight search (ICA weight still rising)
===============================================================================
Current best: rkfm2_b63_rm20 LOO=0.995534 (+0.000129)
  rank = 0.63×rank_c + 0.34×rank_i + 0.03×rank_s
  final = 0.80×chk + 0.20×(rank/n_files)

Observations:
- ICA rank weight still rising (0.23→0.25→0.34, still going up)
- best rank weight declining (0.75→0.70→0.63)
- Optimal may be even more ICA-weighted

Directions:
 A: Extended rank weights (wb: 0.40-0.70, wi: 0.28-0.55 step 0.01) at rm=0.20
 B: Fine-tune rank_mix (0.15-0.30) at best weights
 C: Further joint refinement
 D: Test extreme end (pure ICA rank dominance)
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
print(f"[batch149] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 149}
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

# Verify
rb_ref = 0.63*rank_c + 0.34*rank_i + 0.03*rank_s
chk_v = 0.80*chk + 0.20*(rb_ref/n_files)
print(f"Verify: {macro_auc(chk_v):.6f} (expect 0.995534)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Extended rank weights (broader range)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Extended rank weights (rm=0.20) ===", flush=True)
best_rw = best_loo
best_rw_params = (0.63, 0.34, 0.03)
for wb in np.arange(0.38, 0.72, 0.01):
    for wi in np.arange(0.26, 0.60, 0.01):
        wb_r = round(float(wb), 2); wi_r = round(float(wi), 2)
        ws_r = round(1.0 - wb_r - wi_r, 2)
        if ws_r < 0 or ws_r > 0.25: continue
        rb = wb_r*rank_c + wi_r*rank_i + ws_r*rank_s
        final = 0.80*chk + 0.20*(rb/n_files)
        ar = macro_auc(final)
        mname = f"rkext_b{int(wb_r*100)}_i{int(wi_r*100)}_s{int(ws_r*100)}"
        delta = save_result(mname, ar, {"wb": wb_r, "wi": wi_r, "ws": ws_r, "rm": 0.20})
        if ar > best_rw:
            best_rw = ar
            best_rw_params = (wb_r, wi_r, ws_r)
        if ar > best_loo - 0.00003:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  wb={wb_r:.2f} wi={wi_r:.2f} ws={ws_r:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

wb_best, wi_best, ws_best = best_rw_params
print(f"  Best rank weights: ({wb_best:.2f}, {wi_best:.2f}, {ws_best:.2f}) → {best_rw:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Fine-tune rank_mix at best weights
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Fine rank_mix at wb={wb_best:.2f} wi={wi_best:.2f} ws={ws_best:.2f} ===", flush=True)
rb_best = wb_best*rank_c + wi_best*rank_i + ws_best*rank_s
rb_norm = rb_best / n_files

best_rm = best_loo
best_rm_val = 0.20
for rm_int in range(12, 35):
    rm = rm_int / 100.0
    final = (1-rm)*chk + rm*rb_norm
    ar = macro_auc(final)
    mname = f"rkext_rm{rm_int:02d}_b{int(wb_best*100)}_i{int(wi_best*100)}"
    delta = save_result(mname, ar, {"rm": rm, "wb": wb_best, "wi": wi_best, "ws": ws_best})
    if ar > best_rm:
        best_rm = ar
        best_rm_val = rm
    if ar > best_loo - 0.00003:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  rank_mix={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best rank_mix: {best_rm_val:.2f} → {best_rm:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Joint refinement near new best
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Joint refinement ===", flush=True)
best_joint = best_loo
for wb in np.arange(max(0.38, wb_best-0.05), min(0.72, wb_best+0.06), 0.01):
    for wi in np.arange(max(0.26, wi_best-0.05), min(0.60, wi_best+0.06), 0.01):
        wb_r = round(float(wb), 2); wi_r = round(float(wi), 2)
        ws_r = round(1.0 - wb_r - wi_r, 2)
        if ws_r < 0 or ws_r > 0.25: continue
        rb = wb_r*rank_c + wi_r*rank_i + ws_r*rank_s
        for rm in [best_rm_val - 0.02, best_rm_val, best_rm_val + 0.02]:
            rm = round(rm, 2)
            if rm < 0.08 or rm > 0.40: continue
            final = (1-rm)*chk + rm*(rb/n_files)
            ar = macro_auc(final)
            mname = f"rkjnt2_b{int(wb_r*100)}_i{int(wi_r*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar, {"wb": wb_r, "wi": wi_r, "ws": ws_r, "rm": rm})
            if ar > best_joint: best_joint = ar
            if ar > best_loo - 0.00003:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  wb={wb_r:.2f} wi={wi_r:.2f} ws={ws_r:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best joint: {best_joint:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 149]
print(f"Batch149 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
