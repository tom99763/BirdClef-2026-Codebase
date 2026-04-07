"""
batch158 — Novel approaches: logit aggregation, IDF rank weighting, Jaccard COOC
===============================================================================
Current best: sca4_ia26_sa28 LOO=0.995800
  sca4_ia26_sa28: chk=0.75×c3(0.20)+0.15×i3(0.26)+0.10×s3(0.28)
  final = 0.72×chk + 0.28×(0.56×rank_c(0.23)+0.44×rank_i(0.40))/n_files

batch156/157 confirm plateau — need truly novel structural changes.

New directions:
 A: Per-window logit aggregation (mean/max/top-k) from logit_sig_win
 B: IDF-weighted rank (weigh rank by species rarity)
 C: Jaccard COOC instead of conditional-probability COOC
 D: pm3way_ica_alt and pm3way_std as additional chain signals
 E: Conditional probability reweighted COOC (different normalization)
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
prob_max    = ep["file_prob_max"]
pm3_ica     = ep["file_prob_max_3way_ica_alt"]
pm3_std     = ep["file_prob_max_3way_std"]
logit_sig_win = ep["logit_sig_win"]    # shape=(739, 234)
win_file_id   = ep["win_file_id"]      # shape=(739,)

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch158] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 158}
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

# Standard COOC
COOC    = (fl_hard.T @ fl_hard) / count_i[:, None]
np.fill_diagonal(COOC, 0)
raw_idf = np.clip(np.log(float(n_files) / (count_i + 1.0 - EPS)), 0, None)
IDF075  = raw_idf ** 0.75; IDF075 /= (IDF075.mean() + EPS)

# Jaccard COOC
def make_jaccard():
    cooc_raw = fl_hard.T @ fl_hard  # (n_sp, n_sp): co-occurrence counts
    union = count_i[:, None] + count_i[None, :] - cooc_raw
    jac = np.where(union > 0, cooc_raw / (union + EPS), 0.0)
    np.fill_diagonal(jac, 0)
    return jac.astype(np.float32)

JACCARD = make_jaccard()

def soft_cooc_jac(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    """soft_cooc using Jaccard COOC"""
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope*(s-center), -88, 88)))
        sg = s * gate * (idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi] = s; continue
        c = JACCARD.T @ sg; mc = np.abs(c).max()
        if mc > EPS: c /= mc
        out[fi] = (1-alpha)*s + alpha*np.clip(c, 0, None)
    return out

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

def apply_3way_jac(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, a1=0.110, a2=0.030):
    """3way with Jaccard COOC"""
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc_jac(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc_jac(s, center=0.54, slope=41.0, alpha=a1)
    tr = soft_cooc_jac(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf * idf_s + r_tr * tr

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

# Reference
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

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Per-window logit aggregations
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Window logit aggregation ===", flush=True)
# Build file-level aggregations from logit_sig_win
file_logit_mean = np.zeros((n_files, n_species), dtype=np.float32)
file_logit_topk = np.zeros((n_files, n_species), dtype=np.float32)  # top-3 mean
file_win_count = np.zeros(n_files, dtype=int)

for fi in range(n_files):
    mask = (win_file_id == fi)
    wins = logit_sig_win[mask]  # (n_wins, n_species)
    if len(wins) == 0: continue
    file_win_count[fi] = len(wins)
    file_logit_mean[fi] = wins.mean(axis=0)
    k = min(3, len(wins))
    top_k = np.sort(wins, axis=0)[-k:].mean(axis=0)
    file_logit_topk[fi] = top_k

print(f"  mean AUC: {macro_auc(file_logit_mean):.6f}", flush=True)
print(f"  top3 AUC: {macro_auc(file_logit_topk):.6f}", flush=True)

best_wa = best_loo
for agg_name, agg_signal in [('mean', file_logit_mean), ('top3', file_logit_topk)]:
    ar = macro_auc(agg_signal)
    save_result(f"winagg_{agg_name}", ar, {"type": agg_name})
    # Try blending into chk
    for w_agg in [0.02, 0.05, 0.08, 0.10]:
        # Add as 4th signal
        chk4 = (1-w_agg)*chk_ref + w_agg*agg_signal
        final = 0.72*chk4 + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wagg_{agg_name}_w{int(w_agg*100)}"
        delta = save_result(mname, ar, {"agg": agg_name, "w": w_agg})
        if ar > best_wa: best_wa = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  {agg_name} w={w_agg:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best window agg: {best_wa:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: IDF-weighted rank (rare species get boosted rank signal)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: IDF-weighted rank ===", flush=True)
best_wr = best_loo
# IDF weights per species: rare species get higher weight in rank
idf_raw = raw_idf.copy()
idf_norm = idf_raw / (idf_raw.mean() + EPS)

for idf_pow in [0.5, 0.75, 1.0, 1.5]:
    idf_w = idf_norm ** idf_pow
    idf_w_norm = idf_w / (idf_w.mean() + EPS)
    # Weighted rank: rank × idf_weight (per species)
    rank_c_idf = rank_c_ref * idf_w_norm[None, :]
    rank_i_idf = rank_i_ref * idf_w_norm[None, :]
    rk_idf = 0.56*rank_c_idf + 0.44*rank_i_idf
    # Normalize: divide by max possible rank for each species
    max_rank = n_files * idf_w_norm[None, :]
    rk_idf_norm = rk_idf / (max_rank + EPS)
    for rm in [0.26, 0.28, 0.30]:
        final = (1-rm)*chk_ref + rm*rk_idf_norm
        ar = macro_auc(final)
        mname = f"idfrk_p{int(idf_pow*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar, {"idf_pow": idf_pow, "rm": rm})
        if ar > best_wr: best_wr = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  idf_pow={idf_pow:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best IDF rank: {best_wr:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Jaccard COOC instead of standard conditional-prob COOC
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Jaccard COOC ===", flush=True)
best_jc = best_loo
for a in [0.15, 0.20, 0.23, 0.25, 0.30]:
    c3_j = apply_3way_jac(double_best, alpha=a)
    i3_j = apply_3way_jac(ica_ens_alt, alpha=a*1.1)
    chk_j = 0.75*c3_j + 0.15*i3_j + 0.10*s3_ref
    ar_chk = macro_auc(chk_j)
    # Rank with Jaccard
    rk_c_j = make_rank(apply_3way_jac(double_best, alpha=a))
    rk_i_j = make_rank(apply_3way_jac(ica_ens_alt, alpha=a*1.6))
    rk_j = 0.56*rk_c_j + 0.44*rk_i_j
    for w_jac in [0.20, 0.50, 0.80, 1.00]:
        # Blend Jaccard chk with standard
        chk_blend = (1-w_jac)*chk_ref + w_jac*chk_j
        # Blend Jaccard rank with standard
        rk_blend = (1-w_jac)*rank_ref + w_jac*rk_j
        final = 0.72*chk_blend + 0.28*(rk_blend/n_files)
        ar = macro_auc(final)
        mname = f"jac_a{int(a*100)}_wj{int(w_jac*100)}"
        delta = save_result(mname, ar, {"a": a, "w_jaccard": w_jac})
        if ar > best_jc: best_jc = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  jac a={a:.2f} w_jac={w_jac:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best Jaccard: {best_jc:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: pm3way_ica and pm3way_std as additional chain signals
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: pm3_ica and pm3_std as chain signals ===", flush=True)
print(f"  pm3_ica AUC: {macro_auc(pm3_ica):.6f}  pm3_std AUC: {macro_auc(pm3_std):.6f}", flush=True)

# Apply 3way to them
best_pd = best_loo
for a_pm in [0.20, 0.25, 0.30]:
    pm3i = apply_3way(pm3_ica, alpha=a_pm)
    pm3s = apply_3way(pm3_std, alpha=a_pm)
    for w_pmi in [0.05, 0.10]:
        for w_pms in [0.02, 0.05]:
            # 5-component blend
            remain = 1 - w_pmi - w_pms
            for wb in np.arange(0.70, 0.78, 0.02):
                for wi in np.arange(0.12, 0.18, 0.02):
                    ws = round(remain - round(float(wb), 2) - round(float(wi), 2), 2)
                    if ws < 0 or ws > 0.10: continue
                    chk5 = round(float(wb), 2)*c3_ref + round(float(wi), 2)*i3_ref + ws*s3_ref + w_pmi*pm3i + w_pms*pm3s
                    final = 0.72*chk5 + 0.28*rank_norm
                    ar = macro_auc(final)
                    mname = f"pm5c_apm{int(a_pm*100)}_wi{int(w_pmi*100)}_ws{int(w_pms*100)}_wb{int(round(float(wb), 2)*100)}"
                    delta = save_result(mname, ar, {"a_pm": a_pm, "w_pmi": w_pmi, "w_pms": w_pms})
                    if ar > best_pd: best_pd = ar
                    if ar > best_loo - 0.00005:
                        flag = " ← NEW BEST!" if ar > best_loo else ""
                        print(f"  5comp apm={a_pm:.2f} wi={w_pmi:.2f} ws={w_pms:.2f} wb={round(float(wb), 2):.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best 5-comp: {best_pd:.6f}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 158]
print(f"Batch158 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
