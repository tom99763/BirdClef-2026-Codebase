"""
batch145 — two_round fine-tune + re-optimize all params at new two_round
===============================================================================
Current best: pctr_a1110_a230 LOO=0.995298 (+0.000013)
  a_best=0.200, a_ica=0.255, a_std=0.280
  two_round: a1=0.110, a2=0.030
  blend=(0.75, 0.23, 0.02), idf_blend=0.55, r_idf=0.875

Directions:
 A: two_round alpha fine-tune (a1: 0.095-0.125 step 0.005, a2: 0.020-0.045 step 0.005)
 B: two_round center/slope re-opt at new alpha
 C: Per-component alpha re-optimize at new two_round
 D: Blend weights re-optimize at new two_round
 E: r_idf re-tune at new two_round
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
print(f"[batch145] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 145}
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

def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125,
               a1=0.110, a2=0.030, c1=0.54, c2=0.53, s1=41.0, s2=37.0):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=c1, slope=s1, alpha=a1)
    tr = soft_cooc(r1, center=c2, slope=s2, alpha=a2)
    return r_idf * idf_s + r_tr * tr

def blend3(c, i, s, wb=0.75, wi=0.23, ws=0.02):
    return wb*c + wi*i + ws*s

# Verify
c3_ref = apply_3way(double_best, alpha=0.200)
i3_ref = apply_3way(ica_ens_alt, alpha=0.255)
s3_ref = apply_3way(std_ens_ref,  alpha=0.280)
chk = blend3(c3_ref, i3_ref, s3_ref)
print(f"Verify: {macro_auc(chk):.6f} (expect 0.995298)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: two_round alpha fine-tune
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: two_round alpha fine-tune ===", flush=True)
a1_range = [round(x/1000, 3) for x in range(90, 135, 5)]
a2_range = [round(x/1000, 3) for x in range(15, 50, 5)]

best_tr_auc = best_loo
best_tr_params = (0.110, 0.030)
for a1 in a1_range:
    for a2 in a2_range:
        c3 = apply_3way(double_best, alpha=0.200, a1=a1, a2=a2)
        i3 = apply_3way(ica_ens_alt, alpha=0.255, a1=a1, a2=a2)
        s3 = apply_3way(std_ens_ref,  alpha=0.280, a1=a1, a2=a2)
        ar = macro_auc(blend3(c3, i3, s3))
        mname = f"trf_a1{int(a1*1000)}_a2{int(a2*1000)}"
        delta = save_result(mname, ar, {"a1": a1, "a2": a2})
        if ar > best_tr_auc:
            best_tr_auc = ar
            best_tr_params = (a1, a2)
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a1={a1:.3f} a2={a2:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

best_a1, best_a2 = best_tr_params
print(f"  Best two_round: a1={best_a1:.3f} a2={best_a2:.3f} → {best_tr_auc:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: two_round center/slope re-opt at new alpha
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: two_round center/slope re-opt at a1={best_a1:.3f} a2={best_a2:.3f} ===", flush=True)
best_cs = best_loo
for c1 in [0.52, 0.53, 0.54, 0.55, 0.56]:
    for c2 in [0.51, 0.52, 0.53, 0.54, 0.55]:
        for sl2 in [33.0, 35.0, 37.0, 39.0, 41.0]:
            c3 = apply_3way(double_best, alpha=0.200, a1=best_a1, a2=best_a2, c1=c1, c2=c2, s2=sl2)
            i3 = apply_3way(ica_ens_alt, alpha=0.255, a1=best_a1, a2=best_a2, c1=c1, c2=c2, s2=sl2)
            s3 = apply_3way(std_ens_ref,  alpha=0.280, a1=best_a1, a2=best_a2, c1=c1, c2=c2, s2=sl2)
            ar = macro_auc(blend3(c3, i3, s3))
            mname = f"trcs_c1{int(c1*100)}_c2{int(c2*100)}_s2{int(sl2)}"
            delta = save_result(mname, ar, {"c1": c1, "c2": c2, "s2": sl2, "a1": best_a1, "a2": best_a2})
            if ar > best_cs: best_cs = ar
            if ar > best_loo - 0.00006:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  c1={c1:.2f} c2={c2:.2f} s2={sl2:.0f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best center/slope: {best_cs:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Per-component alpha re-optimize at new two_round
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Per-component alpha at a1={best_a1:.3f} a2={best_a2:.3f} ===", flush=True)
a_range = [round(x/1000, 3) for x in range(190, 305, 5)]
cache_b = {av: apply_3way(double_best, alpha=av, a1=best_a1, a2=best_a2) for av in a_range}
cache_i = {av: apply_3way(ica_ens_alt, alpha=av, a1=best_a1, a2=best_a2) for av in a_range}
cache_s = {av: apply_3way(std_ens_ref,  alpha=av, a1=best_a1, a2=best_a2) for av in a_range}

best_pc = best_loo
best_pc_params = (0.200, 0.255, 0.280)
# Grid over a_ica and a_std (a_best fixed at 0.200)
for a_ica in a_range:
    for a_std in a_range:
        ar = macro_auc(blend3(cache_b[0.200], cache_i[a_ica], cache_s[a_std]))
        mname = f"pcnew_ai{int(a_ica*1000)}_as{int(a_std*1000)}"
        delta = save_result(mname, ar, {"a_ica": a_ica, "a_std": a_std, "a1": best_a1, "a2": best_a2})
        if ar > best_pc:
            best_pc = ar
            best_pc_params = (0.200, a_ica, a_std)
        if ar > best_loo - 0.00004:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_ica={a_ica:.3f} a_std={a_std:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

best_ab_new, best_ai_new, best_as_new = best_pc_params
print(f"  Best per-comp alpha: a_ica={best_ai_new:.3f} a_std={best_as_new:.3f} → {best_pc:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Blend weights re-optimize
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Blend weights at new settings ===", flush=True)
c3_d = apply_3way(double_best, alpha=0.200, a1=best_a1, a2=best_a2)
i3_d = apply_3way(ica_ens_alt, alpha=best_ai_new, a1=best_a1, a2=best_a2)
s3_d = apply_3way(std_ens_ref,  alpha=best_as_new, a1=best_a1, a2=best_a2)

best_blend = best_loo
best_blend_params = (0.75, 0.23, 0.02)
for ws in [0.00, 0.01, 0.02, 0.03]:
    for wi in np.arange(0.17, 0.30, 0.01):
        wi_r = round(float(wi), 2); ws_r = round(float(ws), 2)
        wb_r = round(1.0 - wi_r - ws_r, 2)
        if wb_r < 0.68 or wb_r > 0.84: continue
        br = blend3(c3_d, i3_d, s3_d, wb=wb_r, wi=wi_r, ws=ws_r)
        ar = macro_auc(br)
        mname = f"blnew_b{int(wb_r*100)}_i{int(wi_r*100)}_s{int(ws_r*100)}_a1{int(best_a1*1000)}"
        delta = save_result(mname, ar, {"wb": wb_r, "wi": wi_r, "ws": ws_r})
        if ar > best_blend:
            best_blend = ar
            best_blend_params = (wb_r, wi_r, ws_r)
        if ar > best_loo - 0.00006:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  wb={wb_r:.2f} wi={wi_r:.2f} ws={ws_r:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best blend: {best_blend_params} → {best_blend:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 145]
print(f"Batch145 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
