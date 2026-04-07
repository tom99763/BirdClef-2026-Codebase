"""
batch177 — Novel unexplored directions beyond wkt3 saturation
==============================================================
Current best: wkt3_ki6_kp3_ks5_ww4_rm28 LOO=0.995986
All priority methods (Mahal/GMM/BayesianRidge/RBF/AttentionKNN) exhausted.
This batch tries genuinely novel structural changes:

  A: Verify baseline (wkt3_ki6_kp3_ks5_ww4_rm28 = 0.995986)
  B: Geometric mean aggregation (product instead of sum for wkt3 components)
  C: Per-species adaptive rm (rare species get more rank normalization)
  D: Temporal position-aware KNN (same-position windows weighted higher)
  E: Fine-grained rm ultra-sweep (0.270~0.290 step 0.001)
  F: wkt3 with harmonic mean aggregation
  G: wkt3 with softmax-temperature weighting (k-neighbor confidence)
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

EPS   = 1e-8
BATCH = 177
ROOT  = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

# ── Load data ─────────────────────────────────────────────────────────────────
DATA     = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
n_files  = len(DATA["n_windows"])
n_species= DATA["labels"].shape[1]

with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

file_labels = ep["file_labels"]
double_best = ep["chain_double_best"]
ica_ens_alt = ep["chain_ica_ens_alt"]
std_ens_ref = ep["chain_std_ens_ref"]
emb_ica     = ep["emb_win_ica_norm"]   # (739, 100)
emb_pca     = ep["emb_win_pca_norm"]   # (739, 80)
emb_std     = ep["emb_win_std_norm"]   # (739, 80)
emb_nmf     = ep["emb_win_nmf_norm"]   # (739, 100)
logit_sig   = ep["logit_sig_win"]      # (739, 234)
labels_win  = ep["labels_win"]         # (739, 234)
win_file_id = ep["win_file_id"]        # (739,)

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch{BATCH}] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)
print(f"[batch{BATCH}] Total tried: {len(tried)}", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    if mname in tried:
        return score - best_loo
    entry = {"method": mname, "loo_auc": float(score),
             "config": config_dict or {}, "batch": BATCH}
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
        print(f"  *** NEW BEST: {mname} LOO={score:.6f} ***", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

# ── Fixed components (identical to batch173/176) ──────────────────────────────
fl_hard  = file_labels.astype(np.float32)
count_i  = fl_hard.sum(0) + EPS
COOC     = (fl_hard.T @ fl_hard) / count_i[:, None]
np.fill_diagonal(COOC, 0)
raw_idf  = np.clip(np.log(float(n_files) / (count_i + 1.0 - EPS)), 0, None)
IDF075   = raw_idf ** 0.75; IDF075 /= (IDF075.mean() + EPS)

def soft_cooc(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s    = scores[fi]
        gate = 1.0 / (1.0 + np.exp(np.clip(-slope*(s-center), -88, 88)))
        sg   = s * gate * (idf_w if idf_w is not None else 1.0)
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

fi_wins_list    = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id != fi)[0] for fi in range(n_files)]

c3_ref     = apply_3way(double_best, alpha=0.19)
i3_ref     = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref     = apply_3way(std_ens_ref,  alpha=0.33)
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_norm  = (0.56*rank_c_ref + 0.44*rank_i_ref) / n_files

print("Pre-computing similarity matrices...", flush=True)
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
SIM_NMF = emb_nmf @ emb_nmf.T
print("  Done.", flush=True)

# Window temporal positions (0..11 within each file)
win_pos = np.zeros(len(win_file_id), dtype=np.int32)
for fi in range(n_files):
    for j, w in enumerate(fi_wins_list[fi]):
        win_pos[w] = j

def wknn_single(SIM, k=7):
    signal = labels_win.astype(np.float32)
    preds  = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_w = fi_wins_list[fi]; ow = other_wins_list[fi]
        if len(fi_w) == 0: continue
        k_eff = min(k, len(ow))
        wp = np.zeros((len(fi_w), n_species), dtype=np.float32)
        for wi, wkk in enumerate(fi_w):
            sims = SIM[wkk, ow]
            top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
            top_w = ow[top_l]; w = np.clip(sims[top_l], 0, None)
            ws = w.sum(); w = w/ws if ws > EPS else np.ones(k_eff)/k_eff
            wp[wi] = (w[:, None] * signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

t0 = time.time()

# =============================================================================
# A: VERIFY BASELINE
# =============================================================================
print("\n=== A: Verify baseline ===", flush=True)
print("Pre-computing best wkt3 triple...", flush=True)
p_ica6 = wknn_single(SIM_ICA, k=6)
p_pca3 = wknn_single(SIM_PCA, k=3)
p_std5 = wknn_single(SIM_STD, k=5)
wkt3_best = 0.5*p_ica6 + 0.3*p_pca3 + 0.2*p_std5
chk4_best = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wkt3_best
baseline  = (1-0.28)*chk4_best + 0.28*rank_norm
base_auc  = macro_auc(baseline)
print(f"  Verify: {base_auc:.6f}", flush=True)

# =============================================================================
# B: Geometric Mean Aggregation for wkt3
# =============================================================================
print("\n=== B: Geometric mean wkt3 aggregation ===", flush=True)

def geomean_blend(a, b, c, w1=0.5, w2=0.3, w3=0.2):
    eps2 = 1e-9
    return np.exp(w1*np.log(a+eps2) + w2*np.log(b+eps2) + w3*np.log(c+eps2))

# Pre-normalize preds to [0,1]
def norm01(x):
    mn = x.min(); mx = x.max()
    return (x - mn) / (mx - mn + EPS)

for ki, kp, ks in [(6,3,5), (7,3,5), (6,4,5), (6,3,6)]:
    _pi = wknn_single(SIM_ICA, k=ki) if ki != 6 else p_ica6
    _pp = wknn_single(SIM_PCA, k=kp) if kp != 3 else p_pca3
    _ps = wknn_single(SIM_STD, k=ks) if ks != 5 else p_std5
    for ww in [0.04, 0.05, 0.06]:
        for rm in [0.27, 0.28, 0.29]:
            # Geometric mean of components
            gm = geomean_blend(np.clip(_pi, EPS, 1), np.clip(_pp, EPS, 1),
                               np.clip(_ps, EPS, 1), 0.5, 0.3, 0.2)
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*gm
            final = (1-rm)*chk4 + rm*rank_norm
            ar = macro_auc(final)
            mname = f"wkt3geom_ki{ki}_kp{kp}_ks{ks}_ww{int(ww*100)}_rm{int(rm*100)}"
            d = save_result(mname, ar, {"ki":ki,"kp":kp,"ks":ks,"ww":ww,"rm":rm,"agg":"geom"})
            if d > 1e-7: print(f"    *** IMPROVEMENT: {mname}: {ar:.6f} (+{d:.6f})", flush=True)

bests_B = [(e['loo_auc'], e['method']) for e in res['experiments'] if e.get('batch')==BATCH and 'geom' in e['method']]
if bests_B:
    best_B = max(bests_B, key=lambda x: x[0])
    print(f"  Geom-mean wkt3 best: {best_B[0]:.6f} ({best_B[1]})", flush=True)

# =============================================================================
# C: Per-Species Adaptive rm
# =============================================================================
print("\n=== C: Per-species adaptive rm ===", flush=True)
# Species rarity: count = number of files where species present
species_count = file_labels.sum(0)  # (234,)

def adaptive_rm_blend(chk4, rank_norm, species_count,
                      rm_rare=0.35, rm_common=0.20, rare_thr=5, common_thr=15):
    # Interpolate rm between rare and common based on count
    rm_per_species = np.where(
        species_count < rare_thr, rm_rare,
        np.where(species_count > common_thr, rm_common,
                 rm_rare + (rm_common - rm_rare) * (species_count - rare_thr) / (common_thr - rare_thr + EPS))
    ).astype(np.float32)  # (234,)
    # Apply per-species blend
    final = (1 - rm_per_species[None, :]) * chk4 + rm_per_species[None, :] * rank_norm
    return final

for ki, kp, ks in [(6,3,5), (7,3,5)]:
    _pi = p_ica6 if ki==6 else wknn_single(SIM_ICA, k=ki)
    _pp = p_pca3 if kp==3 else wknn_single(SIM_PCA, k=kp)
    _ps = p_std5 if ks==5 else wknn_single(SIM_STD, k=ks)
    triple = 0.5*_pi + 0.3*_pp + 0.2*_ps
    for ww in [0.04, 0.05]:
        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple
        for rm_rare, rm_common in [(0.35, 0.20), (0.38, 0.18), (0.32, 0.22), (0.40, 0.15)]:
            for rare_thr, common_thr in [(5, 15), (4, 12), (6, 18)]:
                final = adaptive_rm_blend(chk4, rank_norm, species_count,
                                          rm_rare, rm_common, rare_thr, common_thr)
                ar = macro_auc(final)
                mname = f"wkt3arm_ki{ki}_ww{int(ww*100)}_rr{int(rm_rare*100)}_rc{int(rm_common*100)}_rt{rare_thr}_ct{common_thr}"
                d = save_result(mname, ar, {"ki":ki,"kp":kp,"ks":ks,"ww":ww,
                                             "rm_rare":rm_rare,"rm_common":rm_common,
                                             "rare_thr":rare_thr,"common_thr":common_thr})
                if d > 1e-7: print(f"    *** IMPROVEMENT: {mname}: {ar:.6f} (+{d:.6f})", flush=True)

bests_C = [(e['loo_auc'], e['method']) for e in res['experiments'] if e.get('batch')==BATCH and 'arm' in e['method']]
if bests_C:
    best_C = max(bests_C, key=lambda x: x[0])
    print(f"  Adaptive-rm wkt3 best: {best_C[0]:.6f} ({best_C[1]})", flush=True)

# =============================================================================
# D: Temporal Position-Aware KNN
# =============================================================================
print("\n=== D: Temporal position-aware KNN ===", flush=True)

def wknn_position_aware(SIM, k=7, pos_sigma=3.0, pos_w=0.15):
    signal = labels_win.astype(np.float32)
    preds  = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_w = fi_wins_list[fi]; ow = other_wins_list[fi]
        if len(fi_w) == 0: continue
        k_eff = min(k, len(ow))
        wp = np.zeros((len(fi_w), n_species), dtype=np.float32)
        for wi, wkk in enumerate(fi_w):
            sims = SIM[wkk, ow].copy()
            # Position similarity: gaussian decay
            pos_diff = np.abs(win_pos[wkk] - win_pos[ow]).astype(float)
            pos_sim  = np.exp(-0.5 * (pos_diff / pos_sigma)**2)
            # Combine: embedding sim + position bonus
            combined = sims * (1.0 + pos_w * pos_sim)
            top_l  = np.argpartition(-combined, k_eff-1)[:k_eff]
            top_w  = ow[top_l]
            w = np.clip(combined[top_l], 0, None)
            ws = w.sum(); w = w/ws if ws > EPS else np.ones(k_eff)/k_eff
            wp[wi] = (w[:, None] * signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

pos_results = {}
for pos_sigma in [2.0, 3.0, 4.0, 5.0]:
    for pos_w in [0.10, 0.15, 0.20, 0.30]:
        p_ica6_pos = wknn_position_aware(SIM_ICA, k=6, pos_sigma=pos_sigma, pos_w=pos_w)
        p_pca3_pos = wknn_position_aware(SIM_PCA, k=3, pos_sigma=pos_sigma, pos_w=pos_w)
        p_std5_pos = wknn_position_aware(SIM_STD, k=5, pos_sigma=pos_sigma, pos_w=pos_w)
        triple_pos = 0.5*p_ica6_pos + 0.3*p_pca3_pos + 0.2*p_std5_pos
        for ww in [0.04, 0.05]:
            for rm in [0.27, 0.28, 0.29]:
                chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple_pos
                final = (1-rm)*chk4 + rm*rank_norm
                ar = macro_auc(final)
                mname = f"wkt3pos_ps{int(pos_sigma*10)}_pw{int(pos_w*100)}_ww{int(ww*100)}_rm{int(rm*100)}"
                d = save_result(mname, ar, {"pos_sigma":pos_sigma,"pos_w":pos_w,"ww":ww,"rm":rm})
                if d > 1e-7: print(f"    *** IMPROVEMENT: {mname}: {ar:.6f} (+{d:.6f})", flush=True)
                pos_results[mname] = ar
        print(f"  pos_sigma={pos_sigma} pos_w={pos_w}: best={max(pos_results.values()):.6f}", flush=True)

# =============================================================================
# E: Fine-grained rm ultra-sweep (0.270~0.290 step 0.001)
# =============================================================================
print("\n=== E: Fine-grained rm ultra-sweep ===", flush=True)
chk4_base_e = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wkt3_best
for rm_int in range(270, 291):
    rm = rm_int / 1000.0
    final = (1-rm)*chk4_base_e + rm*rank_norm
    ar = macro_auc(final)
    mname = f"wkt3rm_fine_{rm_int}"
    d = save_result(mname, ar, {"rm": rm})
    if d > 1e-7: print(f"  *** NEW BEST: {mname}: {ar:.6f} (+{d:.6f})", flush=True)
    elif rm_int % 5 == 0: print(f"  rm={rm:.3f}: {ar:.6f}", flush=True)

# =============================================================================
# F: Harmonic Mean Aggregation
# =============================================================================
print("\n=== F: Harmonic mean wkt3 ===", flush=True)

def harmonic_blend(a, b, c, w1=0.5, w2=0.3, w3=0.2):
    eps2 = 1e-8
    return 1.0 / (w1/(a+eps2) + w2/(b+eps2) + w3/(c+eps2))

for ki, kp, ks in [(6,3,5), (7,3,5), (6,4,5)]:
    _pi = p_ica6 if ki==6 else wknn_single(SIM_ICA, k=ki)
    _pp = p_pca3 if kp==3 else wknn_single(SIM_PCA, k=kp)
    _ps = p_std5 if ks==5 else wknn_single(SIM_STD, k=ks)
    for ww in [0.04, 0.05]:
        for rm in [0.27, 0.28, 0.29]:
            hm = harmonic_blend(np.clip(_pi, EPS, 1), np.clip(_pp, EPS, 1),
                                np.clip(_ps, EPS, 1))
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*hm
            final = (1-rm)*chk4 + rm*rank_norm
            ar = macro_auc(final)
            mname = f"wkt3harm_ki{ki}_kp{kp}_ks{ks}_ww{int(ww*100)}_rm{int(rm*100)}"
            d = save_result(mname, ar, {"ki":ki,"kp":kp,"ks":ks,"ww":ww,"rm":rm,"agg":"harm"})
            if d > 1e-7: print(f"    *** IMPROVEMENT: {mname}: {ar:.6f} (+{d:.6f})", flush=True)

# =============================================================================
# G: Softmax-temperature weighted KNN
# =============================================================================
print("\n=== G: Softmax-temperature weighted KNN ===", flush=True)

def wknn_softmax_temp(SIM, k=7, temp=5.0):
    signal = labels_win.astype(np.float32)
    preds  = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_w = fi_wins_list[fi]; ow = other_wins_list[fi]
        if len(fi_w) == 0: continue
        k_eff = min(k, len(ow))
        wp = np.zeros((len(fi_w), n_species), dtype=np.float32)
        for wi, wkk in enumerate(fi_w):
            sims = SIM[wkk, ow]
            top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
            top_w = ow[top_l]
            raw_s = sims[top_l]
            # Softmax with temperature
            e = np.exp(temp * (raw_s - raw_s.max()))
            w = e / (e.sum() + EPS)
            wp[wi] = (w[:, None] * signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

for temp in [3.0, 5.0, 8.0, 12.0, 20.0]:
    p_ica_sm = wknn_softmax_temp(SIM_ICA, k=6, temp=temp)
    p_pca_sm = wknn_softmax_temp(SIM_PCA, k=3, temp=temp)
    p_std_sm = wknn_softmax_temp(SIM_STD, k=5, temp=temp)
    triple_sm = 0.5*p_ica_sm + 0.3*p_pca_sm + 0.2*p_std_sm
    for ww in [0.04, 0.05]:
        for rm in [0.27, 0.28, 0.29]:
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple_sm
            final = (1-rm)*chk4 + rm*rank_norm
            ar = macro_auc(final)
            mname = f"wkt3sm_t{int(temp*10)}_ww{int(ww*100)}_rm{int(rm*100)}"
            d = save_result(mname, ar, {"temp":temp,"ww":ww,"rm":rm})
            if d > 1e-7: print(f"    *** IMPROVEMENT: {mname}: {ar:.6f} (+{d:.6f})", flush=True)
    print(f"  temp={temp}: done", flush=True)

# ── Final summary ──────────────────────────────────────────────────────────────
elapsed = time.time() - t0
batch_exps = [e for e in res['experiments'] if e.get('batch') == BATCH]
print(f"\n[batch{BATCH}] Done in {elapsed/60:.1f} min", flush=True)
print(f"[batch{BATCH}] Final best LOO: {best_loo:.6f} ({res['best']['method']})", flush=True)
print(f"[batch{BATCH}] Baseline:       {base_auc:.6f}", flush=True)
print(f"[batch{BATCH}] Improvement:    {best_loo - base_auc:+.6f}", flush=True)
section_labels = {
    'geom': 'B (Geom Mean)', 'arm': 'C (Adaptive rm)',
    'pos': 'D (Pos-Aware)', 'rm_fine': 'E (Fine rm)',
    'harm': 'F (Harmonic)', 'sm': 'G (Softmax-Temp)'
}
print(f"[batch{BATCH}] Section bests:")
for key, label in section_labels.items():
    matches = [e['loo_auc'] for e in batch_exps if key in e['method']]
    if matches:
        print(f"  {label}: {max(matches):.6f}")
