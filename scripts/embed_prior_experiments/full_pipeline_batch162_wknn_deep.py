"""
batch162 — Deep fine-tune on window KNN ICA+PCA combination
===============================================================================
Current best: wfip_ki8_kp5_wi5_w25 LOO=0.995927
  chk = (1-0.025)*chk_ref + 0.025*(0.5*wknn_ica_k8 + 0.5*wknn_pca_k5)
  final = 0.72*chk + 0.28*rank_norm

batch161 findings:
- ICA k=8 + PCA k=5 (50/50) at w=0.025 = 0.995927 (new best)
- ICA k=8 + PCA k=7 (70/30 ICA) at w=0.025 = 0.995927 (tie)
- k=7 w=0.030 achieves 0.995914
- Plateau seems at 0.99592-0.99593

Directions:
 A: Fine ki × kp grid at fixed w=0.025 (ICA k: 6-10, PCA k: 3-8)
 B: Fine w sweep for best (ki=8, kp=5) from 0.020-0.035
 C: ICA+PCA+STD triple combination
 D: wi fraction fine grid (ICA weight vs PCA weight)
 E: Blend window KNN into rank_c and rank_i separately
 F: 4-component chk with window KNN as 4th signal
 G: Cascade: apply window KNN, then apply 3way smoothing on result
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
emb_std = ep["emb_win_std_norm"]
labels_win  = ep["labels_win"]
win_file_id = ep["win_file_id"]

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch162] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 162}
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

# Precompute window KNN
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
fi_wins_list   = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list= [np.where(win_file_id != fi)[0] for fi in range(n_files)]

def wknn(SIM, k=7, agg='mean'):
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
        preds[fi] = wp.mean(0) if agg == 'mean' else wp.max(0)
    return preds

print("Pre-computing ICA window KNN at k=[5..12]...", flush=True)
ica_k = {k: wknn(SIM_ICA, k=k) for k in range(5, 13)}
print("Pre-computing PCA window KNN at k=[3..10]...", flush=True)
pca_k = {k: wknn(SIM_PCA, k=k) for k in range(3, 11)}
print("Pre-computing STD window KNN at k=[5,7,10]...", flush=True)
std_k = {k: wknn(SIM_STD, k=k) for k in [5, 7, 10]}

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Fine ki × kp grid at w=0.025, wi=0.5
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Fine ki × kp grid ===", flush=True)
best_a = best_loo
for ki in range(5, 12):
    for kp in range(3, 10):
        comb = 0.5*ica_k[ki] + 0.5*pca_k[kp]
        chk_new = 0.975*chk_ref + 0.025*comb
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wfd_ki{ki}_kp{kp}_wi5_w25"
        delta = save_result(mname, ar)
        if ar > best_a: best_a = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  ki={ki} kp={kp}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section A: {best_a:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Fine w sweep for best (ki=8, kp=5, wi=0.5)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Fine w sweep (ki=8, kp=5, wi=0.5) ===", flush=True)
best_b = best_loo
comb85 = 0.5*ica_k[8] + 0.5*pca_k[5]
for w_int in range(15, 45):  # 0.015 to 0.044 step 0.001
    w = w_int / 1000.0
    chk_new = (1-w)*chk_ref + w*comb85
    final = 0.72*chk_new + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"wfbw_ki8_kp5_wi5_w{w_int}"
    delta = save_result(mname, ar)
    if ar > best_b: best_b = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section B: {best_b:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: ICA+PCA+STD triple combination
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: ICA+PCA+STD triple combination ===", flush=True)
best_c = best_loo
for ki, kp, ks in [(8, 5, 7), (8, 5, 10), (7, 5, 7), (7, 7, 7)]:
    for wi_frac in [0.4, 0.5, 0.6]:
        for wp_frac in [0.2, 0.3, 0.4]:
            ws_frac = round(1.0 - wi_frac - wp_frac, 2)
            if ws_frac < 0 or ws_frac > 0.4: continue
            comb = wi_frac*ica_k[ki] + wp_frac*pca_k[kp] + ws_frac*std_k[ks]
            for w in [0.020, 0.025, 0.030]:
                chk_new = (1-w)*chk_ref + w*comb
                final = 0.72*chk_new + 0.28*rank_norm
                ar = macro_auc(final)
                mname = f"wftri_i{ki}_{int(wi_frac*10)}_p{kp}_{int(wp_frac*10)}_s{ks}_{int(ws_frac*10)}_w{int(w*1000)}"
                delta = save_result(mname, ar)
                if ar > best_c: best_c = ar
                if ar > best_loo - 0.00005:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  IPS ki={ki} kp={kp} ks={ks} wi={wi_frac:.1f} wp={wp_frac:.1f} ws={ws_frac:.1f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: wi fraction fine grid around (0.5, 0.5) at (ki=8, kp=5)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: wi fraction fine grid ===", flush=True)
best_d = best_loo
for wi_int in range(3, 8):  # 0.3 to 0.7 step 0.1
    wi = wi_int / 10.0
    wp = 1.0 - wi
    comb = wi*ica_k[8] + wp*pca_k[5]
    for w in [0.020, 0.025, 0.030]:
        chk_new = (1-w)*chk_ref + w*comb
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wfwi_ki8_kp5_wi{wi_int}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_d: best_d = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  wi={wi:.1f} wp={wp:.1f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section D: {best_d:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Window KNN into individual rank components
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Window KNN supplementing rank components ===", flush=True)
best_e = best_loo
comb85 = 0.5*ica_k[8] + 0.5*pca_k[5]

# Use wknn to supplement rank_c: rank_c = make_rank(apply_3way(double_best) + w_wk * wknn)
for w_wk in [0.02, 0.05, 0.10, 0.20]:
    aug_signal_c = double_best + w_wk * comb85
    rk_c_aug = make_rank(apply_3way(aug_signal_c, alpha=0.23))
    rk_aug = 0.56*rk_c_aug + 0.44*rank_i_ref
    # chk with wknn blend
    chk_new = 0.975*chk_ref + 0.025*comb85
    final = 0.72*chk_new + 0.28*(rk_aug/n_files)
    ar = macro_auc(final)
    mname = f"wfe_augc_wk{int(w_wk*100)}"
    delta = save_result(mname, ar)
    if ar > best_e: best_e = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  aug_c w_wk={w_wk:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

    aug_signal_i = ica_ens_alt + w_wk * comb85
    rk_i_aug = make_rank(apply_3way(aug_signal_i, alpha=0.40))
    rk_aug2 = 0.56*rank_c_ref + 0.44*rk_i_aug
    chk_new2 = 0.975*chk_ref + 0.025*comb85
    final2 = 0.72*chk_new2 + 0.28*(rk_aug2/n_files)
    ar2 = macro_auc(final2)
    mname2 = f"wfe_augi_wk{int(w_wk*100)}"
    delta2 = save_result(mname2, ar2)
    if ar2 > best_e: best_e = ar2
    if ar2 > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar2 > best_loo else ""
        print(f"  aug_i w_wk={w_wk:.2f}: {ar2:.6f} {delta2:+.6f}{flag}", flush=True)

print(f"  Best section E: {best_e:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: 4-component chk with window KNN as 4th signal
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== F: 4-component chk with wKNN ===", flush=True)
best_f = best_loo
comb85 = 0.5*ica_k[8] + 0.5*pca_k[5]

for wb in [0.70, 0.72, 0.74, 0.75, 0.76]:
    for wi in [0.13, 0.15, 0.16]:
        for ws in [0.08, 0.09, 0.10]:
            ww = round(1.0 - wb - wi - ws, 3)
            if ww < 0.01 or ww > 0.10: continue
            chk4 = wb*c3_ref + wi*i3_ref + ws*s3_ref + ww*comb85
            final = 0.72*chk4 + 0.28*rank_norm
            ar = macro_auc(final)
            mname = f"wf4c_wb{int(wb*100)}_wi{int(wi*100)}_ws{int(ws*100)}_ww{int(ww*100)}"
            delta = save_result(mname, ar)
            if ar > best_f: best_f = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  4c wb={wb:.2f} wi={wi:.2f} ws={ws:.2f} ww={ww:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section F: {best_f:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 162]
print(f"Batch162 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
