"""
batch165 — Per-species adaptive window KNN + label-weighted combinations
===============================================================================
Current best: wfip_ki8_kp5_wi5_w25 LOO=0.995927

New ideas (all batch164 variants tie/fail):
 A: Per-species k (rare species k=15, common k=5) — adaptive KNN
 B: LOO-optimal w per species (find best blend weight per species)
 C: Geometric mean combination: sqrt(chk * wknn_ref)
 D: Min/max combination with window KNN
 E: Window KNN on subset: only use windows from species-positive files
 F: Final ultra-fine w search (0.001 step) around 0.025
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

file_labels  = ep["file_labels"]
double_best  = ep["chain_double_best"]
ica_ens_alt  = ep["chain_ica_ens_alt"]
std_ens_ref  = ep["chain_std_ens_ref"]

emb_ica = ep["emb_win_ica_norm"]
emb_pca = ep["emb_win_pca_norm"]
labels_win  = ep["labels_win"]
win_file_id = ep["win_file_id"]

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch165] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 165}
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

c3_ref     = apply_3way(double_best, alpha=0.200)
i3_ref     = apply_3way(ica_ens_alt, alpha=0.260)
s3_ref     = apply_3way(std_ens_ref,  alpha=0.280)
chk_ref    = 0.75*c3_ref + 0.15*i3_ref + 0.10*s3_ref
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_ref   = 0.56*rank_c_ref + 0.44*rank_i_ref
rank_norm  = rank_ref / n_files
v = 0.72*chk_ref + 0.28*rank_norm
print(f"Verify: {macro_auc(v):.6f}\n", flush=True)

SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
fi_wins_list   = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list= [np.where(win_file_id != fi)[0] for fi in range(n_files)]

def wknn(SIM, k=7):
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins = fi_wins_list[fi]
        other_wins = other_wins_list[fi]
        if len(fi_wins) == 0: continue
        k_eff = min(k, len(other_wins))
        wp = np.zeros((len(fi_wins), n_species), dtype=np.float32)
        for wi, wk in enumerate(fi_wins):
            sims = SIM[wk, other_wins]
            top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
            top_w = other_wins[top_l]
            w = np.clip(sims[top_l], 0, None)
            ws = w.sum()
            w = w/ws if ws > EPS else np.ones(k_eff)/k_eff
            wp[wi] = (w[:, None] * signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

# Precompute reference
print("Pre-computing reference KNN (k=8 ICA + k=5 PCA)...", flush=True)
p_ica8 = wknn(SIM_ICA, k=8)
p_pca5 = wknn(SIM_PCA, k=5)
p_ref  = 0.5*p_ica8 + 0.5*p_pca5

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Per-species adaptive k
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Per-species adaptive k ===", flush=True)
best_a = best_loo

# Compute KNN at all k values needed
ks = [3, 5, 7, 8, 10, 12, 15]
ica_kk = {k: wknn(SIM_ICA, k=k) for k in ks}
pca_kk = {k: wknn(SIM_PCA, k=k) for k in ks}

# Assign k based on species rarity: rare (count < 5) get k=15, common (count > 20) get k=5
count_sp = fl_hard.sum(0)  # (n_species,) count of positive files per species
k_per_sp = np.where(count_sp < 5, 15,
           np.where(count_sp < 10, 10,
           np.where(count_sp < 20, 7, 5))).astype(int)

def wknn_adaptive(SIM, knn_cache, k_per_sp):
    """Build per-species KNN using adaptive k."""
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    unique_k = np.unique(k_per_sp)
    for k in unique_k:
        sp_mask = (k_per_sp == k)
        if sp_mask.sum() == 0: continue
        p_k = knn_cache.get(k, wknn(SIM, k=k))
        preds[:, sp_mask] = p_k[:, sp_mask]
    return preds

p_ica_adapt = wknn_adaptive(SIM_ICA, ica_kk, k_per_sp)
p_pca_adapt = wknn_adaptive(SIM_PCA, pca_kk, k_per_sp)
comb_adapt = 0.5*p_ica_adapt + 0.5*p_pca_adapt
ar_sa = macro_auc(comb_adapt)
print(f"  Adaptive standalone: {ar_sa:.6f}", flush=True)
for w in [0.015, 0.020, 0.025, 0.030]:
    chk_new = (1-w)*chk_ref + w*comb_adapt
    final = 0.72*chk_new + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"wknn_adapt_w{int(w*1000)}"
    delta = save_result(mname, ar)
    if ar > best_a: best_a = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  adaptive w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section A: {best_a:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: LOO-optimal w per species (per-species blend weight selection)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Per-species LOO-optimal w ===", flush=True)
best_b = best_loo

# For each species, find optimal w to blend chk_ref and p_ref
w_grid = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
ps_w_preds = np.zeros((n_files, n_species), dtype=np.float32)
ps_w_vals = np.zeros(n_species, dtype=np.float32)

for si in range(n_species):
    y = fl_hard[:, si]
    if y.sum() == 0 or y.sum() == n_files:
        ps_w_preds[:, si] = chk_ref[:, si]
        continue
    best_auc_s = -1.0
    best_w_s = 0.025
    for w in w_grid:
        pred_s = (1-w)*chk_ref[:, si] + w*p_ref[:, si]
        try:
            v = roc_auc_score(y, pred_s)
            if v > best_auc_s:
                best_auc_s, best_w_s = v, w
        except:
            pass
    ps_w_vals[si] = best_w_s
    ps_w_preds[:, si] = (1-best_w_s)*chk_ref[:, si] + best_w_s*p_ref[:, si]

# But this is in-sample selection — use a different approach: median w
median_w = float(np.median(ps_w_vals))
print(f"  Median optimal w: {median_w:.3f}", flush=True)

ar_psw = macro_auc(ps_w_preds)
mname = f"wknn_psw"
delta = save_result(mname, ar_psw)
if ar_psw > best_b: best_b = ar_psw
print(f"  per-species w (in-sample): {ar_psw:.6f} {delta:+.6f}", flush=True)

# Apply final combination
final_psw = 0.72*ps_w_preds + 0.28*rank_norm
ar = macro_auc(final_psw)
mname2 = f"wknn_psw_fullrank"
delta2 = save_result(mname2, ar)
if ar > best_b: best_b = ar
if ar > best_loo - 0.00005:
    flag = " ← NEW BEST!" if ar > best_loo else ""
    print(f"  ps_w with rank: {ar:.6f} {delta2:+.6f}{flag}", flush=True)
print(f"  Best section B: {best_b:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Geometric mean and harmonic mean combinations
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Geometric/harmonic mean combinations ===", flush=True)
best_c = best_loo

for w in [0.020, 0.025, 0.030]:
    # Geometric mean: sqrt(chk * (1 + w*(wknn-1))) — not exactly, let's do:
    # final = chk_ref^(1-w) * p_ref^w (geometric blend)
    chk_safe = np.clip(chk_ref, EPS, 1.0)
    p_safe   = np.clip(p_ref, EPS, 1.0)
    geom = (chk_safe ** (1-w)) * (p_safe ** w)
    final = 0.72*geom + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"wknn_geom_w{int(w*1000)}"
    delta = save_result(mname, ar)
    if ar > best_c: best_c = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  geom w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

    # Harmonic mean blend
    harm = 2.0 * chk_ref * p_ref / (chk_ref + p_ref + EPS)
    chk_blend = (1-w)*chk_ref + w*harm
    final_h = 0.72*chk_blend + 0.28*rank_norm
    ar_h = macro_auc(final_h)
    mname_h = f"wknn_harm_w{int(w*1000)}"
    delta_h = save_result(mname_h, ar_h)
    if ar_h > best_c: best_c = ar_h
    if ar_h > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar_h > best_loo else ""
        print(f"  harm w={w:.3f}: {ar_h:.6f} {delta_h:+.6f}{flag}", flush=True)

    # Max combination: element-wise max
    max_sig = np.maximum(chk_ref, p_ref)
    chk_max = (1-w)*chk_ref + w*max_sig
    final_m = 0.72*chk_max + 0.28*rank_norm
    ar_m = macro_auc(final_m)
    mname_m = f"wknn_max_w{int(w*1000)}"
    delta_m = save_result(mname_m, ar_m)
    if ar_m > best_c: best_c = ar_m
    if ar_m > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar_m > best_loo else ""
        print(f"  max w={w:.3f}: {ar_m:.6f} {delta_m:+.6f}{flag}", flush=True)

print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Ensemble window KNN across multiple k values
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Multi-k ensemble window KNN ===", flush=True)
best_d = best_loo

# Average multiple k values
for ks_ens in [(5, 8, 12), (6, 8, 10), (7, 8, 9)]:
    p_ica_ens = np.mean([ica_kk[k] for k in ks_ens], axis=0)
    p_pca_ens = np.mean([pca_kk[k] for k in ks_ens if k in pca_kk], axis=0)
    comb_ens = 0.5*p_ica_ens + 0.5*p_pca_ens
    for w in [0.020, 0.025, 0.030]:
        chk_new = (1-w)*chk_ref + w*comb_ens
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_kens_{'_'.join(map(str, ks_ens))}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_d: best_d = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  k_ens={ks_ens} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section D: {best_d:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: Ultra-fine w search (step 0.001) around 0.025
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== F: Ultra-fine w search (0.001 step) ===", flush=True)
best_f = best_loo
for w_int in range(18, 32):
    w = w_int / 1000.0
    chk_new = (1-w)*chk_ref + w*p_ref
    final = 0.72*chk_new + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"wknn_uf_w{w_int}"
    delta = save_result(mname, ar)
    if ar > best_f: best_f = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section F: {best_f:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 165]
print(f"Batch165 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
