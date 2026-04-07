"""
batch161 — Fine-tune window-level ICA KNN parameters
===============================================================================
Current best: wknn_ica_k7_lbl_wchk2 LOO=0.995861
  chk = (1-0.02)*chk_ref + 0.02*wknn_ica_k7_lbl
  final = 0.72*chk + 0.28*(0.56*rank_c(0.23)+0.44*rank_i(0.40))/n_files

Key finding (batch160):
  Window-level ICA KNN (k=7, labels, mean-agg, w=0.02 in chk) gives +0.000061 improvement
  w=0.05 gives 0.995815 (worse), so blend weight is very sensitive

Directions:
 A: Fine-tune k (6, 7, 8, 9, 10, 12, 15, 20) at w=0.02
 B: Fine-tune w (0.005, 0.010, 0.015, 0.020, 0.025, 0.030) at k=7
 C: Top-k aggregation: top-2, top-3 mean aggregation
 D: Window KNN blend into rank instead of chk (or both)
 E: Combine ICA+PCA window KNN signals
 F: 3-way optimization: k × w × blend_target (chk vs rank)
 G: ICA KNN applied per chain signal: wknn as 4th signal in chk blend
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

emb_ica = ep["emb_win_ica_norm"]  # (739, 100)
emb_pca = ep["emb_win_pca_norm"]  # (739, 80)
emb_std = ep["emb_win_std_norm"]  # (739, 80)
labels_win  = ep["labels_win"]    # (739, 234)
win_file_id = ep["win_file_id"]   # (739,) int32
logit_sig   = ep["logit_sig_win"] # (739, 234)

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch161] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 161}
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
print(f"Verify: {macro_auc(v):.6f} (expect ~0.995861)\n", flush=True)

# ─── Window KNN helper ─────────────────────────────────────────────────────────
# Precompute all similarity matrices
SIM_ICA = emb_ica @ emb_ica.T  # (739, 739)
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T

# Precompute window-to-file membership
fi_wins_list = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id != fi)[0] for fi in range(n_files)]

def window_knn_fast(SIM, k=7, use_logit=False, agg='mean'):
    signal = logit_sig if use_logit else labels_win.astype(np.float32)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins = fi_wins_list[fi]
        other_wins = other_wins_list[fi]
        if len(fi_wins) == 0: continue
        k_eff = min(k, len(other_wins))
        win_preds = np.zeros((len(fi_wins), n_species), dtype=np.float32)
        for wi, wk in enumerate(fi_wins):
            sims_wk = SIM[wk, other_wins]
            top_local = np.argpartition(-sims_wk, k_eff - 1)[:k_eff]
            top_wins = other_wins[top_local]
            w = np.clip(sims_wk[top_local], 0, None)
            w_sum = w.sum()
            if w_sum < EPS: w = np.ones(k_eff) / k_eff
            else: w = w / w_sum
            win_preds[wi] = (w[:, None] * signal[top_wins]).sum(0)
        if agg == 'mean':
            preds[fi] = win_preds.mean(0)
        elif agg == 'max':
            preds[fi] = win_preds.max(0)
        elif agg == 'top2':
            if len(win_preds) >= 2:
                preds[fi] = np.sort(win_preds, axis=0)[-2:].mean(0)
            else:
                preds[fi] = win_preds[0]
        elif agg == 'top3':
            if len(win_preds) >= 3:
                preds[fi] = np.sort(win_preds, axis=0)[-3:].mean(0)
            else:
                preds[fi] = win_preds.mean(0)
    return preds

t0 = time.time()

# Precompute ICA KNN at various k values
print("Pre-computing ICA window KNN at k=[5,6,7,8,9,10,12,15,20]...", flush=True)
ica_knn = {}
for k in [5, 6, 7, 8, 9, 10, 12, 15, 20]:
    ica_knn[k] = window_knn_fast(SIM_ICA, k=k, use_logit=False, agg='mean')
    print(f"  k={k}: standalone={macro_auc(ica_knn[k]):.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# A: Fine k sweep at w=0.02 in chk
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Fine k sweep at w=0.02 ===", flush=True)
best_a = best_loo
for k in [5, 6, 7, 8, 9, 10, 12, 15, 20]:
    p = ica_knn[k]
    for w in [0.015, 0.020, 0.025]:
        chk_new = (1-w)*chk_ref + w*p
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wkf_ica_k{k}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_a: best_a = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  k={k} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section A: {best_a:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Fine w sweep at k=7
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Fine w sweep at k=7 ===", flush=True)
best_b = best_loo
p7 = ica_knn[7]
for w_int in range(3, 40, 1):  # 0.003 to 0.039 step 0.001
    w = w_int / 1000.0
    chk_new = (1-w)*chk_ref + w*p7
    final = 0.72*chk_new + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"wfw_ica_k7_w{w_int}"
    delta = save_result(mname, ar)
    if ar > best_b: best_b = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  k=7 w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section B: {best_b:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Aggregation variants (top-2, top-3, max)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Aggregation variants at k=7 ===", flush=True)
best_c = best_loo
for agg in ['top2', 'top3', 'max']:
    p_agg = window_knn_fast(SIM_ICA, k=7, use_logit=False, agg=agg)
    ar_sa = macro_auc(p_agg)
    print(f"  {agg} standalone: {ar_sa:.6f}", flush=True)
    for w in [0.01, 0.015, 0.02, 0.025, 0.03]:
        chk_new = (1-w)*chk_ref + w*p_agg
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wfagg_ica_k7_{agg}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_c: best_c = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  {agg} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Window KNN into rank, or both chk and rank
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Window KNN into rank ===", flush=True)
best_d = best_loo

# Find best k from section A
best_k = 7
best_w = 0.020

for k in [7, 8, 9, 10]:
    p = ica_knn[k]
    # Into rank only
    rk_wknn = make_rank(p)
    for w_rk in [0.01, 0.02, 0.05, 0.08, 0.10]:
        rk_mix = (1-w_rk)*rank_ref + w_rk*rk_wknn
        final = 0.72*chk_ref + 0.28*(rk_mix/n_files)
        ar = macro_auc(final)
        mname = f"wfrk_ica_k{k}_wrk{int(w_rk*100)}"
        delta = save_result(mname, ar)
        if ar > best_d: best_d = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  into rank k={k} w={w_rk:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
    # Into chk AND rank simultaneously
    for w_ch in [0.015, 0.020]:
        for w_rk in [0.01, 0.02, 0.03]:
            chk_new = (1-w_ch)*chk_ref + w_ch*p
            rk_mix = (1-w_rk)*rank_ref + w_rk*make_rank(p)
            final = 0.72*chk_new + 0.28*(rk_mix/n_files)
            ar = macro_auc(final)
            mname = f"wfboth_ica_k{k}_wch{int(w_ch*1000)}_wrk{int(w_rk*100)}"
            delta = save_result(mname, ar)
            if ar > best_d: best_d = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  both k={k} wch={w_ch:.3f} wrk={w_rk:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section D: {best_d:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Combine ICA + PCA window KNN
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Combine ICA + PCA window KNN ===", flush=True)
best_e = best_loo

print("  Computing PCA window KNN at k=[5,7,10]...", flush=True)
pca_knn = {}
for k in [5, 7, 10]:
    pca_knn[k] = window_knn_fast(SIM_PCA, k=k, use_logit=False, agg='mean')

# Best ICA: k=7 at w=0.02; try combining with PCA
for k_i in [7, 8]:
    for k_p in [5, 7, 10]:
        # Ensemble ICA and PCA window KNN
        for w_ica_frac in [0.3, 0.5, 0.7]:
            w_pca_frac = 1.0 - w_ica_frac
            combined = w_ica_frac * ica_knn[k_i] + w_pca_frac * pca_knn[k_p]
            for w in [0.015, 0.020, 0.025]:
                chk_new = (1-w)*chk_ref + w*combined
                final = 0.72*chk_new + 0.28*rank_norm
                ar = macro_auc(final)
                mname = f"wfip_ki{k_i}_kp{k_p}_wi{int(w_ica_frac*10)}_w{int(w*1000)}"
                delta = save_result(mname, ar)
                if ar > best_e: best_e = ar
                if ar > best_loo - 0.00005:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  ICA+PCA ki={k_i} kp={k_p} wi={w_ica_frac:.1f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section E: {best_e:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: 3-way: k × w × rm fine grid
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== F: Joint k × w × rm grid ===", flush=True)
best_f = best_loo

for k in [7, 8, 9]:
    p = ica_knn[k]
    for w in [0.010, 0.015, 0.020, 0.025]:
        chk_new = (1-w)*chk_ref + w*p
        for rm in [0.26, 0.27, 0.28, 0.29, 0.30]:
            final = (1-rm)*chk_new + rm*rank_norm
            ar = macro_auc(final)
            mname = f"wfjoint_k{k}_w{int(w*1000)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_f: best_f = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  joint k={k} w={w:.3f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section F: {best_f:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# G: Apply 3way co-occ smoothing on window KNN output
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== G: 3way smoothing of window KNN output ===", flush=True)
best_g = best_loo

p7_raw = ica_knn[7]
for alpha_smooth in [0.10, 0.15, 0.20, 0.23, 0.25]:
    p7_smooth = apply_3way(p7_raw, alpha=alpha_smooth)
    for w in [0.010, 0.015, 0.020, 0.025]:
        chk_new = (1-w)*chk_ref + w*p7_smooth
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wfsm_ica_k7_a{int(alpha_smooth*100)}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_g: best_g = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  smooth a={alpha_smooth:.2f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section G: {best_g:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 161]
print(f"Batch161 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
