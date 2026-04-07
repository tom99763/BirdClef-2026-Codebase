"""
batch166 — Re-optimize chk weights with window KNN as 4th component
===============================================================================
Current best: wfip_ki8_kp5_wi5_w25 LOO=0.995927
  chk_wknn = (1-0.025)*chk_ref + 0.025*wknn_comb
  chk_ref = 0.75*c3 + 0.15*i3 + 0.10*s3
  final = 0.72*chk_wknn + 0.28*rank_norm

Hypothesis: Now that we have window KNN signal, the (0.75, 0.15, 0.10) blend
in chk_ref may not be optimal anymore. Also, the rank mixing ratio 0.72/0.28
might shift.

Directions:
 A: Re-tune chk 3-component blend (wb, wi, ws) with wknn fixed at w=0.025
 B: Re-tune rank mixing rm while keeping optimal chk with wknn
 C: Joint optimization: (wb, wi, ws, ww, rm) where ww is wknn weight
 D: Re-tune rank alpha (a_rank_c, a_rank_i) with new chk+wknn
 E: Compare: wknn in chk vs wknn as 4th signal in final blend
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
print(f"[batch166] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 166}
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

# Precompute score components
c3_ref     = apply_3way(double_best, alpha=0.200)
i3_ref     = apply_3way(ica_ens_alt, alpha=0.260)
s3_ref     = apply_3way(std_ens_ref,  alpha=0.280)

# Precompute rank components
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_ref   = 0.56*rank_c_ref + 0.44*rank_i_ref
rank_norm  = rank_ref / n_files

# Precompute window KNN
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
        for wi, wkk in enumerate(fi_wins):
            sims = SIM[wkk, other_wins]
            top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
            top_w = other_wins[top_l]
            w = np.clip(sims[top_l], 0, None)
            ws = w.sum()
            w = w/ws if ws > EPS else np.ones(k_eff)/k_eff
            wp[wi] = (w[:, None] * signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

print("Pre-computing window KNN...", flush=True)
p_ica8 = wknn(SIM_ICA, k=8)
p_pca5 = wknn(SIM_PCA, k=5)
wknn_comb = 0.5*p_ica8 + 0.5*p_pca5

# Verify current best
chk_ref = 0.75*c3_ref + 0.15*i3_ref + 0.10*s3_ref
chk_wknn = 0.975*chk_ref + 0.025*wknn_comb
v = 0.72*chk_wknn + 0.28*rank_norm
print(f"Verify: {macro_auc(v):.6f} (expect ~0.995927)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Re-tune chk 3-component blend with wknn fixed at w=0.025
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Re-tune chk (wb, wi, ws) with wknn w=0.025 ===", flush=True)
best_a = best_loo

# Grid: wb in [0.68-0.80], wi in [0.10-0.20], ws = 1-wb-wi
for wb_int in range(68, 82, 2):
    wb = wb_int / 100.0
    for wi_int in range(10, 22, 2):
        wi = wi_int / 100.0
        ws = round(1.0 - wb - wi, 2)
        if ws < 0.04 or ws > 0.16: continue
        chk = wb*c3_ref + wi*i3_ref + ws*s3_ref
        chk_w = 0.975*chk + 0.025*wknn_comb
        final = 0.72*chk_w + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wkrw_wb{wb_int}_wi{wi_int}"
        delta = save_result(mname, ar)
        if ar > best_a: best_a = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  wb={wb:.2f} wi={wi:.2f} ws={ws:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section A: {best_a:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Re-tune rank mixing ratio with chk+wknn fixed
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Re-tune rank mixing rm ===", flush=True)
best_b = best_loo
for rm_int in range(22, 36):
    rm = rm_int / 100.0
    final = (1-rm)*chk_wknn + rm*rank_norm
    ar = macro_auc(final)
    mname = f"wkrm_rm{rm_int}"
    delta = save_result(mname, ar)
    if ar > best_b: best_b = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section B: {best_b:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Joint 4-component blend (c3, i3, s3, wknn) + rank mixing
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Joint 4-component blend + rm ===", flush=True)
best_c = best_loo

for wb in [0.70, 0.72, 0.74, 0.75, 0.76, 0.78]:
    for wi in [0.12, 0.14, 0.15, 0.16]:
        for ws in [0.07, 0.08, 0.09, 0.10]:
            ww = round(1.0 - wb - wi - ws, 3)
            if ww < 0.01 or ww > 0.08: continue
            chk4 = wb*c3_ref + wi*i3_ref + ws*s3_ref + ww*wknn_comb
            for rm in [0.26, 0.27, 0.28, 0.29, 0.30]:
                final = (1-rm)*chk4 + rm*rank_norm
                ar = macro_auc(final)
                mname = f"wk4j_wb{int(wb*100)}_wi{int(wi*100)}_ws{int(ws*100)}_ww{int(ww*100)}_rm{rm_int}"
                mname = f"wk4j_wb{int(wb*100)}_wi{int(wi*100)}_ws{int(ws*100)}_ww{int(ww*1000)}_rm{int(rm*100)}"
                delta = save_result(mname, ar)
                if ar > best_c: best_c = ar
                if ar > best_loo - 0.00005:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  4c wb={wb:.2f} wi={wi:.2f} ws={ws:.2f} ww={ww:.3f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Re-tune rank alpha (a_rank_c, a_rank_i) with new chk+wknn
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Re-tune rank alphas with wknn in chk ===", flush=True)
best_d = best_loo
for a_c in [0.20, 0.22, 0.23, 0.24, 0.25]:
    rk_c = make_rank(apply_3way(double_best, alpha=a_c))
    for a_i in [0.38, 0.40, 0.42, 0.44]:
        rk_i = make_rank(apply_3way(ica_ens_alt, alpha=a_i))
        rk = 0.56*rk_c + 0.44*rk_i
        final = 0.72*chk_wknn + 0.28*(rk/n_files)
        ar = macro_auc(final)
        mname = f"wkra_ac{int(a_c*100)}_ai{int(a_i*100)}"
        delta = save_result(mname, ar)
        if ar > best_d: best_d = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_c={a_c:.2f} a_i={a_i:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section D: {best_d:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: wknn as 4th signal in FINAL blend (not in chk)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: wknn in final blend directly ===", flush=True)
best_e = best_loo

chk_ref_std = 0.75*c3_ref + 0.15*i3_ref + 0.10*s3_ref  # standard chk without wknn
for w_wk in [0.01, 0.02, 0.025, 0.03, 0.04, 0.05]:
    for rm in [0.25, 0.26, 0.27, 0.28, 0.29, 0.30]:
        rw = 1.0 - rm - w_wk
        if rw < 0.60 or rw > 0.80: continue
        final = rw*chk_ref_std + rm*rank_norm + w_wk*wknn_comb
        ar = macro_auc(final)
        mname = f"wkfin_wk{int(w_wk*1000)}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_e: best_e = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  final wk={w_wk:.3f} rm={rm:.2f} rw={rw:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section E: {best_e:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 166]
print(f"Batch166 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
