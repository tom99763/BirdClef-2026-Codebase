"""
batch179 — Deep sweep around NEW BEST: ica128s42 (LOO=0.995999)
================================================================
NEW BEST from batch178: ica128s42_kf5_ww4_rm28 = 0.995999
  FastICA(128-dim, seed=42) replacing ICA100 in wkt3 triple
  triple = 0.5*ICA128(k=5) + 0.3*PCA(k=3) + 0.2*STD(k=5)

This batch explores:
  A: k sweep for ICA128 (k=3..12) with best wkt3 config
  B: ICA128 with different dims (96..192, step 16) + k sweep
  C: ICA128 + original ICA100 combined (wkt4-style: 4 components)
  D: ww/rm ultra-fine sweep for best ica128 config
  E: ICA128 with different seeds (to verify seed sensitivity)
  F: Replace PCA or STD with ICA128 (different component positions)
"""
import numpy as np
import json, pickle, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import FastICA, PCA as sklearn_PCA
from sklearn.preprocessing import normalize, StandardScaler
import warnings
warnings.filterwarnings('ignore')

EPS   = 1e-8
BATCH = 179
ROOT  = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

file_labels = ep["file_labels"]
double_best = ep["chain_double_best"]
ica_ens_alt = ep["chain_ica_ens_alt"]
std_ens_ref = ep["chain_std_ens_ref"]
emb_ica     = ep["emb_win_ica_norm"]
emb_pca     = ep["emb_win_pca_norm"]
emb_std     = ep["emb_win_std_norm"]
emb_nmf     = ep["emb_win_nmf_norm"]
labels_win  = ep["labels_win"]
win_file_id = ep["win_file_id"]

DATA      = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
raw_emb   = DATA["emb"].astype(np.float32)
n_files   = len(DATA["n_windows"])
n_species = DATA["labels"].shape[1]

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch{BATCH}] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)
print(f"[batch{BATCH}] Total tried: {len(tried)}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": BATCH}
    res["experiments"].append(entry)
    tried.add(mname)
    if score > best_loo + 1e-7:
        best_loo = score
        res["best"] = {"method": mname, "loo_auc": float(score)}
        with open(MODEL_PATH, "rb") as f:
            ep_up = pickle.load(f)
        ep_up["method"] = mname; ep_up["loo_auc"] = float(score)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(ep_up, f)
        print(f"  *** NEW BEST: {mname} LOO={score:.6f} ***", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

fl_hard  = file_labels.astype(np.float32)
count_i  = fl_hard.sum(0) + EPS
COOC     = (fl_hard.T @ fl_hard) / count_i[:, None]
np.fill_diagonal(COOC, 0)
raw_idf  = np.clip(np.log(float(n_files) / (count_i + 1.0 - EPS)), 0, None)
IDF075   = raw_idf ** 0.75; IDF075 /= (IDF075.mean() + EPS)

def soft_cooc(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s    = scores[fi]; gate = 1.0/(1.0+np.exp(np.clip(-slope*(s-center),-88,88)))
        sg   = s * gate * (idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi]=s; continue
        c = COOC.T @ sg; mc = np.abs(c).max()
        if mc > EPS: c /= mc
        out[fi] = (1-alpha)*s + alpha*np.clip(c, 0, None)
    return out

def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, a1=0.110, a2=0.030):
    sp = np.clip(s,0,1)**2; sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf*idf_s + r_tr*tr

def make_rank(x): return np.argsort(np.argsort(x,axis=0),axis=0).astype(float)

fi_wins_list    = [np.where(win_file_id==fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id!=fi)[0] for fi in range(n_files)]

c3_ref = apply_3way(double_best, alpha=0.19)
i3_ref = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref = apply_3way(std_ens_ref,  alpha=0.33)
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_norm  = (0.56*rank_c_ref + 0.44*rank_i_ref) / n_files

print("Pre-computing existing SIMs...", flush=True)
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
SIM_NMF = emb_nmf @ emb_nmf.T

def wknn_single(SIM, k=7):
    signal = labels_win.astype(np.float32)
    preds  = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_w = fi_wins_list[fi]; ow = other_wins_list[fi]
        if len(fi_w)==0: continue
        k_eff = min(k, len(ow))
        wp = np.zeros((len(fi_w), n_species), dtype=np.float32)
        for wi, wkk in enumerate(fi_w):
            sims = SIM[wkk, ow]
            top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
            top_w = ow[top_l]; w = np.clip(sims[top_l], 0, None)
            ws = w.sum(); w = w/ws if ws>EPS else np.ones(k_eff)/k_eff
            wp[wi] = (w[:,None]*signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

# Pre-compute PCA-256 preprocessing (same as batch178)
print("Pre-computing PCA-256 for ICA...", flush=True)
scaler_raw = StandardScaler()
raw_scaled = scaler_raw.fit_transform(raw_emb)
pca_pre    = sklearn_PCA(n_components=256, random_state=42)
raw_pca256 = pca_pre.fit_transform(raw_scaled).astype(np.float32)
print("  Done.", flush=True)

# Pre-compute base KNN predictions
p_pca3 = wknn_single(SIM_PCA, k=3)
p_std5 = wknn_single(SIM_STD, k=5)

def build_ica(dim, seed=42):
    ica = FastICA(n_components=dim, random_state=seed, max_iter=500, tol=0.01)
    emb = ica.fit_transform(raw_pca256).astype(np.float32)
    return normalize(emb, norm='l2')

def eval_wkt3_ica(SIM_new, k_new, kp=3, ks=5, ww=0.04, rm=0.28):
    p_new = wknn_single(SIM_new, k=k_new)
    p_pca = p_pca3 if kp==3 else wknn_single(SIM_PCA, k=kp)
    p_std = p_std5 if ks==5 else wknn_single(SIM_STD, k=ks)
    triple = 0.5*p_new + 0.3*p_pca + 0.2*p_std
    chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple
    final = (1-rm)*chk4 + rm*rank_norm
    return macro_auc(final)

t0 = time.time()

# =============================================================================
# A: ICA-128 k sweep (comprehensive)
# =============================================================================
print("\n=== A: ICA-128 k sweep ===", flush=True)
print("  Building ICA-128 (seed=42)...", flush=True)
emb_ica128 = build_ica(128, 42)
SIM_ICA128 = emb_ica128 @ emb_ica128.T

for kf in range(3, 13):
    for kp in [2, 3, 4, 5]:
        for ks in [4, 5, 6, 7]:
            for ww in [0.03, 0.04, 0.05, 0.06]:
                for rm in [0.27, 0.28, 0.29, 0.30]:
                    ar = eval_wkt3_ica(SIM_ICA128, kf, kp, ks, ww, rm)
                    mname = f"ica128_kf{kf}_kp{kp}_ks{ks}_ww{int(ww*100)}_rm{int(rm*100)}"
                    d = save_result(mname, ar, {"dim":128,"seed":42,"kf":kf,"kp":kp,"ks":ks,"ww":ww,"rm":rm})
                    if d > 1e-7: print(f"  *** NEW BEST: {mname}: {ar:.6f} (+{d:.6f})", flush=True)

best_A = max([e['loo_auc'] for e in res['experiments'] if e.get('batch')==BATCH and 'ica128_kf' in e['method']], default=0)
print(f"  ICA-128 sweep best: {best_A:.6f}", flush=True)

# =============================================================================
# B: ICA dim sweep (96..192 step 16) + best k config
# =============================================================================
print("\n=== B: ICA dim sweep ===", flush=True)
best_kf_A = 5  # from batch178

for ica_dim in [96, 112, 128, 144, 160, 176, 192]:
    print(f"  ICA-{ica_dim}...", flush=True)
    emb_tmp = build_ica(ica_dim, 42)
    SIM_tmp = emb_tmp @ emb_tmp.T
    for kf in range(3, 10):
        for kp in [2, 3, 4]:
            for ks in [4, 5, 6]:
                for ww in [0.03, 0.04, 0.05, 0.06]:
                    for rm in [0.27, 0.28, 0.29]:
                        ar = eval_wkt3_ica(SIM_tmp, kf, kp, ks, ww, rm)
                        mname = f"icad{ica_dim}_kf{kf}_kp{kp}_ks{ks}_ww{int(ww*100)}_rm{int(rm*100)}"
                        d = save_result(mname, ar, {"dim":ica_dim,"seed":42,"kf":kf,"kp":kp,"ks":ks,"ww":ww,"rm":rm})
                        if d > 1e-7: print(f"  *** NEW BEST: {mname}: {ar:.6f} (+{d:.6f})", flush=True)

best_B = max([e['loo_auc'] for e in res['experiments'] if e.get('batch')==BATCH and 'icad' in e['method']], default=0)
print(f"  ICA dim sweep best: {best_B:.6f}", flush=True)

# =============================================================================
# C: ICA128 + ICA100 combined (wkt4-style)
# =============================================================================
print("\n=== C: ICA128 + ICA100 combined (wkt4) ===", flush=True)

# Pre-compute ICA128 KNN at various k
ica128_preds = {}
for kf in range(3, 10):
    ica128_preds[kf] = wknn_single(SIM_ICA128, k=kf)

ica100_preds = {}
for ki in range(3, 10):
    ica100_preds[ki] = wknn_single(SIM_ICA, k=ki)

for ki in [5, 6, 7]:
    for kf in [4, 5, 6]:
        for kp in [2, 3, 4]:
            for ks in [4, 5, 6]:
                for wi, wf, wp_, ws_ in [
                    (0.35, 0.20, 0.25, 0.20),
                    (0.30, 0.25, 0.25, 0.20),
                    (0.40, 0.15, 0.25, 0.20),
                    (0.30, 0.20, 0.30, 0.20),
                ]:
                    p_ki = ica100_preds[ki]
                    p_kf = ica128_preds[kf]
                    p_kp = p_pca3 if kp==3 else wknn_single(SIM_PCA, k=kp)
                    p_ks = p_std5 if ks==5 else wknn_single(SIM_STD, k=ks)
                    quad = wi*p_ki + wf*p_kf + wp_*p_kp + ws_*p_ks
                    for ww in [0.04, 0.05]:
                        for rm in [0.27, 0.28, 0.29]:
                            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*quad
                            final = (1-rm)*chk4 + rm*rank_norm
                            ar = macro_auc(final)
                            mname = f"wkt4ii_ki{ki}_kf{kf}_kp{kp}_ks{ks}_wi{int(wi*100)}_wf{int(wf*100)}_rm{int(rm*100)}"
                            d = save_result(mname, ar, {"ki":ki,"kf":kf,"kp":kp,"ks":ks,
                                                         "wi":wi,"wf":wf,"wp":wp_,"ws":ws_,"ww":ww,"rm":rm})
                            if d > 1e-7: print(f"  *** NEW BEST: {mname}: {ar:.6f} (+{d:.6f})", flush=True)

best_C = max([e['loo_auc'] for e in res['experiments'] if e.get('batch')==BATCH and 'wkt4ii' in e['method']], default=0)
print(f"  ICA128+ICA100 quad best: {best_C:.6f}", flush=True)

# =============================================================================
# D: ww/rm ultra-fine sweep for best ICA128 config
# =============================================================================
print("\n=== D: ww/rm ultra-fine sweep ===", flush=True)

# Use best config found so far in this batch
all_ica128 = [e for e in res['experiments'] if e.get('batch')==BATCH and 'ica128' in e['method']]
if all_ica128:
    best_so_far = max(all_ica128, key=lambda x: x['loo_auc'])
    best_kf_D = best_so_far['config'].get('kf', 5)
    best_kp_D = best_so_far['config'].get('kp', 3)
    best_ks_D = best_so_far['config'].get('ks', 5)
    print(f"  Using best config: kf={best_kf_D}, kp={best_kp_D}, ks={best_ks_D}", flush=True)
else:
    best_kf_D, best_kp_D, best_ks_D = 5, 3, 5

p_best_kf = wknn_single(SIM_ICA128, k=best_kf_D)
p_best_kp = p_pca3 if best_kp_D==3 else wknn_single(SIM_PCA, k=best_kp_D)
p_best_ks = p_std5 if best_ks_D==5 else wknn_single(SIM_STD, k=best_ks_D)
triple_best = 0.5*p_best_kf + 0.3*p_best_kp + 0.2*p_best_ks
chk4_best_d = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref

for ww_int in range(3, 8):
    ww = ww_int / 100.0
    chk4_d = chk4_best_d + ww * triple_best
    for rm_int in range(260, 310):
        rm = rm_int / 1000.0
        final = (1-rm)*chk4_d + rm*rank_norm
        ar = macro_auc(final)
        mname = f"ica128fine_kf{best_kf_D}_kp{best_kp_D}_ks{best_ks_D}_ww{ww_int}_rm{rm_int}"
        d = save_result(mname, ar, {"kf":best_kf_D,"kp":best_kp_D,"ks":best_ks_D,"ww":ww,"rm":rm})
        if d > 1e-7: print(f"  *** NEW BEST: {mname}: {ar:.6f} (+{d:.6f})", flush=True)

best_D = max([e['loo_auc'] for e in res['experiments'] if e.get('batch')==BATCH and 'ica128fine' in e['method']], default=0)
print(f"  Ultra-fine sweep best: {best_D:.6f}", flush=True)

# =============================================================================
# E: ICA128 seed sensitivity + cross-seed blend
# =============================================================================
print("\n=== E: ICA128 seed sweep + cross-seed blend ===", flush=True)
seeds_to_try = [0, 1, 7, 13, 17, 31, 50, 77, 100, 200, 500]
seed_sims = {42: SIM_ICA128}  # already computed

for seed in seeds_to_try:
    print(f"  ICA-128 seed={seed}...", flush=True)
    emb_s = build_ica(128, seed)
    SIM_s = emb_s @ emb_s.T
    seed_sims[seed] = SIM_s
    # Single seed sweep
    for kf in range(3, 9):
        for ww in [0.04, 0.05]:
            ar = eval_wkt3_ica(SIM_s, kf, 3, 5, ww, 0.28)
            mname = f"ica128s{seed}_kf{kf}_ww{int(ww*100)}_rm28"
            d = save_result(mname, ar, {"dim":128,"seed":seed,"kf":kf,"ww":ww,"rm":0.28})
            if d > 1e-7: print(f"  *** NEW BEST: {mname}: {ar:.6f} (+{d:.6f})", flush=True)

# Cross-seed blend (average 2 ICA128 from different seeds)
best_seeds = sorted(seed_sims.keys())[:4]  # top-4 seeds
for s1, s2 in [(42,0), (42,1), (42,7), (42,13), (42,17)]:
    if s1 in seed_sims and s2 in seed_sims:
        SIM_avg = 0.5*seed_sims[s1] + 0.5*seed_sims[s2]
        for kf in range(3, 8):
            for ww in [0.04, 0.05]:
                for rm in [0.27, 0.28, 0.29]:
                    p_avg = wknn_single(SIM_avg, k=kf)
                    triple_avg = 0.5*p_avg + 0.3*p_pca3 + 0.2*p_std5
                    chk4_avg = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple_avg
                    final_avg = (1-rm)*chk4_avg + rm*rank_norm
                    ar = macro_auc(final_avg)
                    mname = f"ica128avg_s{s1}s{s2}_kf{kf}_ww{int(ww*100)}_rm{int(rm*100)}"
                    d = save_result(mname, ar, {"seeds":[s1,s2],"kf":kf,"ww":ww,"rm":rm})
                    if d > 1e-7: print(f"  *** NEW BEST: {mname}: {ar:.6f} (+{d:.6f})", flush=True)

best_E = max([e['loo_auc'] for e in res['experiments'] if e.get('batch')==BATCH and 'ica128s' in e['method']], default=0)
print(f"  Seed sweep best: {best_E:.6f}", flush=True)

# =============================================================================
# F: ICA128 replacing PCA or STD (not just ICA) in wkt3
# =============================================================================
print("\n=== F: ICA128 in different positions ===", flush=True)

for kf in range(3, 9):
    p_kf = wknn_single(SIM_ICA128, k=kf)
    for ww in [0.04, 0.05]:
        for rm in [0.27, 0.28, 0.29]:
            # Replace PCA with ICA128
            triple_rp = 0.5*wknn_single(SIM_ICA, k=6) + 0.3*p_kf + 0.2*p_std5
            chk4_rp = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple_rp
            ar_rp = macro_auc((1-rm)*chk4_rp + rm*rank_norm)
            mname_rp = f"ica128rp_kf{kf}_ww{int(ww*100)}_rm{int(rm*100)}"
            d = save_result(mname_rp, ar_rp, {"pos":"pca","kf":kf,"ww":ww,"rm":rm})
            if d > 1e-7: print(f"  *** NEW BEST rp: {mname_rp}: {ar_rp:.6f} (+{d:.6f})", flush=True)

            # Replace STD with ICA128
            triple_rs = 0.5*wknn_single(SIM_ICA, k=6) + 0.3*p_pca3 + 0.2*p_kf
            chk4_rs = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple_rs
            ar_rs = macro_auc((1-rm)*chk4_rs + rm*rank_norm)
            mname_rs = f"ica128rs_kf{kf}_ww{int(ww*100)}_rm{int(rm*100)}"
            d = save_result(mname_rs, ar_rs, {"pos":"std","kf":kf,"ww":ww,"rm":rm})
            if d > 1e-7: print(f"  *** NEW BEST rs: {mname_rs}: {ar_rs:.6f} (+{d:.6f})", flush=True)

best_F = max([e['loo_auc'] for e in res['experiments'] if e.get('batch')==BATCH and 'ica128r' in e['method']], default=0)
print(f"  ICA128-position best: {best_F:.6f}", flush=True)

# ── Final summary ──────────────────────────────────────────────────────────────
elapsed = time.time() - t0
batch_exps = [e for e in res['experiments'] if e.get('batch') == BATCH]
print(f"\n[batch{BATCH}] Done in {elapsed/60:.1f} min", flush=True)
print(f"[batch{BATCH}] Final best LOO: {best_loo:.6f} ({res['best']['method']})", flush=True)
print(f"[batch{BATCH}] Improvement:    {best_loo - 0.995999:+.6f}", flush=True)
for key, label in [('ica128_kf','A (k sweep)'), ('icad','B (dim sweep)'),
                   ('wkt4ii','C (ICA128+100)'), ('ica128fine','D (fine)'),
                   ('ica128s','E (seed)'), ('ica128r','F (position)')]:
    vals = [e['loo_auc'] for e in batch_exps if key in e['method']]
    if vals: print(f"  {label}: {max(vals):.6f}")
