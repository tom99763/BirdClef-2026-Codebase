"""
batch180 — Novel directions 2: Attention KNN, Multi-seed ICA-128, Spectral, Logit-fused
=========================================================================================
Current best: ica128s42_kf5_ww4_rm28 LOO=0.995999

batch179 covers: deep ICA-128 sweep (k, dim, ww/rm, seeds, positions)
batch180 tries genuinely NEW angles using CORRECT window-level SIM architecture:

  A: Softmax-Attention KNN — weight neighbors via softmax(sim/T) instead of top-k uniform
  B: Multi-seed ICA-128 ensemble — average window-level SIM from N seeds (more stable)
  C: Spectral embedding from window SIM (graph Laplacian eigenvectors → new KNN features)
  D: Logit-fused ICA-128 — concat ICA-128 + PCA(logit, n_dim) window embeddings
  E: VLOM-style SIM blend — geomean+RMS of two window SIM matrices (novel blend formula)
"""
import numpy as np
import json, pickle, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import normalize, StandardScaler
import warnings
warnings.filterwarnings('ignore')

EPS   = 1e-8
BATCH = 180
ROOT  = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

# ── Load PKL ──────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

file_labels = ep["file_labels"]
double_best = ep["chain_double_best"]
ica_ens_alt = ep["chain_ica_ens_alt"]
std_ens_ref = ep["chain_std_ens_ref"]
emb_ica     = ep["emb_win_ica_norm"]   # (739, 100) window-level ICA-100
emb_pca     = ep["emb_win_pca_norm"]   # (739, 80)  window-level PCA-80
emb_std     = ep["emb_win_std_norm"]   # (739, 80)  window-level STD-PCA-80
emb_nmf     = ep["emb_win_nmf_norm"]   # (739, 100) window-level NMF-100
labels_win  = ep["labels_win"]         # (739, 234)
win_file_id = ep["win_file_id"]        # (739,) — file index per window
logit_sig   = ep["logit_sig_win"]      # (739, 234) per-window sigmoid logits

# Load raw data
DATA      = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
raw_emb   = DATA["emb"].astype(np.float32)  # (739, 1536)
n_files   = int(len(DATA["n_windows"]))
n_species = int(DATA["labels"].shape[1])
filenames = DATA["filenames"]
file_list = DATA["file_list"]
fname_to_id = {f: i for i, f in enumerate(file_list)}
win_file_id_raw = np.array(
    [fname_to_id.get(str(f), -1) for f in filenames], dtype=np.int32)

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

def save_result(mname, score, cfg=None):
    global best_loo
    if mname in tried:
        return score - best_loo
    entry = {"method": mname, "loo_auc": float(score), "config": cfg or {}, "batch": BATCH}
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

# ── Co-occurrence chain helpers ───────────────────────────────────────────────
fl_hard = file_labels.astype(np.float32)
count_i = fl_hard.sum(0) + EPS
COOC    = (fl_hard.T @ fl_hard) / count_i[:, None]
np.fill_diagonal(COOC, 0)
raw_idf = np.clip(np.log(float(n_files) / (count_i + 1.0 - EPS)), 0, None)
IDF075  = raw_idf ** 0.75; IDF075 /= (IDF075.mean() + EPS)

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
    sp    = np.clip(s, 0, 1)**2
    sc    = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1    = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    tr    = soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf * idf_s + r_tr * tr

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

# ── Chain reference predictions (from PKL) ───────────────────────────────────
fi_wins_list    = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id != fi)[0] for fi in range(n_files)]

c3_ref     = apply_3way(double_best, alpha=0.19)
i3_ref     = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref     = apply_3way(std_ens_ref, alpha=0.33)
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_norm  = (0.56*rank_c_ref + 0.44*rank_i_ref) / n_files

# ── Window-level KNN (standard top-k) ────────────────────────────────────────
def wknn_single(SIM, k=7):
    """Window-level LOO KNN: for each file, hold out its windows, predict via KNN."""
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

# ── Attention KNN (softmax over ALL neighbors, not just top-k) ───────────────
def wknn_attn(SIM, tau=0.20):
    """Softmax-attention KNN: weight all training windows via softmax(sim/tau)."""
    signal = labels_win.astype(np.float32)
    preds  = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_w = fi_wins_list[fi]; ow = other_wins_list[fi]
        if len(fi_w) == 0: continue
        wp = np.zeros((len(fi_w), n_species), dtype=np.float32)
        for wi, wkk in enumerate(fi_w):
            sims = SIM[wkk, ow]
            # Softmax weights over all training windows
            sims_s = sims - sims.max()
            exp_w = np.exp(sims_s / max(tau, 1e-5))
            w = exp_w / (exp_w.sum() + EPS)
            wp[wi] = (w[:, None] * signal[ow]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

# ── Chain injection helper ────────────────────────────────────────────────────
def run_chain(triple, ww=0.04, rm=0.28):
    """Inject triple into chain: chk4 → blend with rank_norm."""
    chk4  = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple
    final = (1-rm)*chk4 + rm*rank_norm
    return final

# ── Pre-compute baseline SIM matrices ────────────────────────────────────────
print("\n[batch180] Pre-computing baseline SIM matrices...", flush=True)
SIM_ICA = emb_ica @ emb_ica.T  # (739, 739)
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T

# Verify baseline
p_ica6 = wknn_single(SIM_ICA, k=6)
p_pca3 = wknn_single(SIM_PCA, k=3)
p_std5 = wknn_single(SIM_STD, k=5)
wkt3_ref = 0.5*p_ica6 + 0.3*p_pca3 + 0.2*p_std5
base_check = run_chain(wkt3_ref, ww=0.04, rm=0.28)
print(f"  Baseline wkt3 k6/3/5 ww4 rm28: {macro_auc(base_check):.6f}", flush=True)

# ── Build ICA-128 (seed=42, batch178 new best) ────────────────────────────────
print("\n[batch180] Building ICA-128 (seed=42)...", flush=True)
t0 = time.time()
scaler_raw = StandardScaler()
raw_scaled = scaler_raw.fit_transform(raw_emb)
pca_pre256 = PCA(n_components=256, random_state=42)
raw_pca256 = pca_pre256.fit_transform(raw_scaled)
ica128s42  = FastICA(n_components=128, random_state=42, max_iter=500, tol=0.01)
emb_ica128 = normalize(ica128s42.fit_transform(raw_pca256), norm='l2')
SIM_ICA128 = emb_ica128 @ emb_ica128.T
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

# Verify new best
p_ica128_5 = wknn_single(SIM_ICA128, k=5)
wkt3_ica128 = 0.5*p_ica128_5 + 0.3*p_pca3 + 0.2*p_std5
best_check = run_chain(wkt3_ica128, ww=0.04, rm=0.28)
print(f"  ICA128 s42 k5 ww4 rm28: {macro_auc(best_check):.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Softmax-Attention KNN
# Replace top-k hard selection with softmax over ALL training windows
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== A: Softmax-Attention KNN ===", flush=True)
t0 = time.time()

# Use ICA-128 SIM (current best embedding) for attention KNN
for tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    p_attn = wknn_attn(SIM_ICA128, tau=tau)
    # Blend with PCA and STD (standard wkt3 ratios)
    for wa, wb, wc in [(0.5, 0.3, 0.2), (0.6, 0.2, 0.2), (0.4, 0.3, 0.3)]:
        triple = wa*p_attn + wb*p_pca3 + wc*p_std5
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.26, 0.28, 0.30]:
                ww_int = int(ww*100)
                wa_int = int(wa*10)
                tau_int = int(tau*100)
                rm_int = int(rm*100)
                mname = f"attn_tau{tau_int:02d}_wa{wa_int}_ww{ww_int}_rm{rm_int}"
                if mname in tried: continue
                final = run_chain(triple, ww=ww, rm=rm)
                auc = macro_auc(final)
                diff = save_result(mname, auc, {"tau": tau, "wa": wa, "ww": ww, "rm": rm})
                marker = " ***NEW BEST***" if diff > 1e-7 else ""
                if abs(diff) < 0.002:  # only print near-best
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

# Also try attention KNN on blended SIM (ICA128+PCA+STD blend first, then attention)
print("  Attention on blended SIM...", flush=True)
for w_blend in [(0.5, 0.3, 0.2), (0.6, 0.2, 0.2)]:
    wa, wb, wc = w_blend
    SIM_blend = wa*SIM_ICA128 + wb*SIM_PCA + wc*SIM_STD
    np.fill_diagonal(SIM_blend, 0.0)
    for tau in [0.10, 0.20, 0.30]:
        p_attn_b = wknn_attn(SIM_blend, tau=tau)
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.26, 0.28, 0.30]:
                wa_int = int(wa*10)
                tau_int = int(tau*100)
                ww_int = int(ww*100)
                mname = f"attn_blend{wa_int}_tau{tau_int:02d}_ww{ww_int}_rm{int(rm*100)}"
                if mname in tried: continue
                final = run_chain(p_attn_b, ww=ww, rm=rm)
                auc = macro_auc(final)
                diff = save_result(mname, auc, {"type": "attn_blend", "tau": tau, "wa": wa})
                marker = " ***NEW BEST***" if diff > 1e-7 else ""
                if abs(diff) < 0.001:
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section A done in {time.time()-t0:.1f}s. Best so far: {best_loo:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Multi-seed ICA-128 ensemble
# Average window-level SIM matrices from N different ICA-128 seeds
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B: Multi-seed ICA-128 ensemble ===", flush=True)
t0 = time.time()

SEEDS = [0, 1, 7, 13, 17, 31, 42, 50, 77, 100]
SIM_ica128_seeds = {42: SIM_ICA128}  # seed=42 already computed

for seed in SEEDS:
    if seed == 42:
        continue
    ica_s = FastICA(n_components=128, random_state=seed, max_iter=500, tol=0.01)
    emb_s = normalize(ica_s.fit_transform(raw_pca256), norm='l2')
    SIM_ica128_seeds[seed] = emb_s @ emb_s.T
    print(f"  ICA-128 seed={seed} done", flush=True)

# Try averaging N seeds
for n_seeds in [2, 3, 4, 5, 7, 10]:
    seed_list = SEEDS[:n_seeds]
    SIM_avg = np.mean([SIM_ica128_seeds[s] for s in seed_list], axis=0)
    np.fill_diagonal(SIM_avg, 0.0)
    for k in [4, 5, 6]:
        p_avg = wknn_single(SIM_avg, k=k)
        for wa, wb, wc in [(0.5, 0.3, 0.2), (0.6, 0.2, 0.2)]:
            triple = wa*p_avg + wb*p_pca3 + wc*p_std5
            for ww in [0.03, 0.04, 0.05]:
                for rm in [0.26, 0.28, 0.30]:
                    wa_int = int(wa*10)
                    mname = f"ica128ms{n_seeds}_k{k}_wa{wa_int}_ww{int(ww*100)}_rm{int(rm*100)}"
                    if mname in tried: continue
                    final = run_chain(triple, ww=ww, rm=rm)
                    auc = macro_auc(final)
                    diff = save_result(mname, auc, {"n_seeds": n_seeds, "k": k, "ww": ww})
                    marker = " ***NEW BEST***" if diff > 1e-7 else ""
                    if abs(diff) < 0.002:
                        print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

# VLOM-style multi-seed: geomean(sims)+RMS(sims) instead of arithmetic mean
print("  Multi-seed VLOM-SIM...", flush=True)
for n_seeds in [3, 5]:
    seed_list = SEEDS[:n_seeds]
    sims_list = [np.clip(SIM_ica128_seeds[s], 0, 1) for s in seed_list]
    # Geometric mean in log space
    log_stack = np.stack([np.log(s + EPS) for s in sims_list], axis=0)
    SIM_geomean = np.exp(log_stack.mean(0)) - EPS
    # RMS
    sq_stack = np.stack([s**2 for s in sims_list], axis=0)
    SIM_rms = np.sqrt(sq_stack.mean(0))
    # VLOM blend
    SIM_vlom_ms = 0.5 * (SIM_geomean + SIM_rms)
    np.fill_diagonal(SIM_vlom_ms, 0.0)

    for k in [4, 5]:
        p_vlom = wknn_single(SIM_vlom_ms, k=k)
        triple = 0.5*p_vlom + 0.3*p_pca3 + 0.2*p_std5
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.26, 0.28, 0.30]:
                mname = f"ica128vlomms{n_seeds}_k{k}_ww{int(ww*100)}_rm{int(rm*100)}"
                if mname in tried: continue
                final = run_chain(triple, ww=ww, rm=rm)
                auc = macro_auc(final)
                diff = save_result(mname, auc, {"type": "vlom_ms", "n_seeds": n_seeds})
                marker = " ***NEW BEST***" if diff > 1e-7 else ""
                if abs(diff) < 0.002:
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section B done in {time.time()-t0:.1f}s. Best so far: {best_loo:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Spectral embedding KNN
# Compute normalized graph Laplacian eigenvectors of window-level SIM,
# use as new window embeddings → new SIM matrix → wknn_single
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== C: Spectral embedding KNN ===", flush=True)
t0 = time.time()

def build_spectral_win_emb(SIM_win, n_components=64, sigma_clip=0.0):
    """
    Spectral embedding of windows from window-level SIM matrix.
    Returns normalized eigenvectors as (739, n_components) embedding.
    """
    n = SIM_win.shape[0]
    A = np.maximum(SIM_win - sigma_clip, 0.0)  # ReLU threshold
    D = A.sum(1)
    D_inv_sqrt = np.where(D > EPS, 1.0 / np.sqrt(D + EPS), 0.0)
    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    L = np.eye(n) - D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]
    # Compute smallest eigenvectors (skip trivial constant)
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[1:n_components+1]
    emb = eigvecs[:, idx]
    return normalize(emb, norm='l2')

# Build spectral embedding from ICA-128 SIM
# Use a clipped version to suppress noise
for n_comp in [32, 48, 64, 96]:
    for sigma in [0.0, 0.1, 0.2]:
        t_spec = time.time()
        spec_emb = build_spectral_win_emb(SIM_ICA128, n_components=n_comp, sigma_clip=sigma)
        SIM_spec = spec_emb @ spec_emb.T
        np.fill_diagonal(SIM_spec, 0.0)

        for k in [4, 5, 6, 8]:
            p_spec = wknn_single(SIM_spec, k=k)
            for wa, wb, wc in [(0.5, 0.3, 0.2), (0.6, 0.2, 0.2), (0.4, 0.4, 0.2)]:
                triple = wa*p_spec + wb*p_pca3 + wc*p_std5
                for ww in [0.03, 0.04, 0.05]:
                    for rm in [0.26, 0.28, 0.30]:
                        wa_i = int(wa*10)
                        sig_i = int(sigma*10)
                        mname = f"spectral{n_comp}sig{sig_i}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                        if mname in tried: continue
                        final = run_chain(triple, ww=ww, rm=rm)
                        auc = macro_auc(final)
                        diff = save_result(mname, auc, {"n_comp": n_comp, "sigma": sigma})
                        marker = " ***NEW BEST***" if diff > 1e-7 else ""
                        if abs(diff) < 0.002:
                            print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

# Also blend spectral + ICA128 as 4th component
print("  Spectral as 4th wkt component...", flush=True)
spec64 = build_spectral_win_emb(SIM_ICA128, n_components=64, sigma_clip=0.1)
SIM_spec64 = spec64 @ spec64.T
np.fill_diagonal(SIM_spec64, 0.0)

p_spec64_k5 = wknn_single(SIM_spec64, k=5)
# 4-way blend: ICA128 + PCA + STD + Spectral
for wi, wp_w, ws_w, wsp in [(5,3,1,1), (4,3,2,1), (5,2,2,1), (4,2,2,2)]:
    total = wi + wp_w + ws_w + wsp
    wa, wb, wc, wd = wi/total, wp_w/total, ws_w/total, wsp/total
    p_spec64_kx = wknn_single(SIM_spec64, k=5)
    triple4 = wa*p_ica128_5 + wb*p_pca3 + wc*p_std5 + wd*p_spec64_kx
    for ww in [0.03, 0.04, 0.05]:
        for rm in [0.26, 0.28, 0.30]:
            mname = f"wkt4_ica128spec64_{wi}{wp_w}{ws_w}{wsp}_ww{int(ww*100)}_rm{int(rm*100)}"
            if mname in tried: continue
            final = run_chain(triple4, ww=ww, rm=rm)
            auc = macro_auc(final)
            diff = save_result(mname, auc, {"type": "wkt4_spectral"})
            marker = " ***NEW BEST***" if diff > 1e-7 else ""
            if abs(diff) < 0.002:
                print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section C done in {time.time()-t0:.1f}s. Best so far: {best_loo:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Logit-fused ICA-128
# Concat ICA-128 window emb + PCA(logit_sig, n_dim) → new embedding → SIM → wknn
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== D: Logit-fused ICA-128 ===", flush=True)
t0 = time.time()

logit_arr = logit_sig.astype(np.float32)  # (739, 234)
logit_scaler = StandardScaler()
logit_scaled = logit_scaler.fit_transform(logit_arr)

for n_logit_comp in [8, 16, 24, 32]:
    pca_logit = PCA(n_components=n_logit_comp, random_state=42)
    emb_logit_pca = normalize(pca_logit.fit_transform(logit_scaled), norm='l2')

    for alpha_logit in [0.05, 0.10, 0.15, 0.20, 0.30]:
        # Weighted concat: main ICA-128 + logit-PCA
        emb_fused = np.concatenate([
            (1 - alpha_logit) * emb_ica128,
            alpha_logit * emb_logit_pca
        ], axis=1)
        emb_fused = normalize(emb_fused, norm='l2')
        SIM_fused = emb_fused @ emb_fused.T

        for k in [4, 5, 6]:
            p_fused = wknn_single(SIM_fused, k=k)
            for wa, wb, wc in [(0.5, 0.3, 0.2), (0.6, 0.2, 0.2)]:
                triple = wa*p_fused + wb*p_pca3 + wc*p_std5
                for ww in [0.03, 0.04, 0.05]:
                    for rm in [0.26, 0.28, 0.30]:
                        al_i = int(alpha_logit*100)
                        wa_i = int(wa*10)
                        mname = f"ica128logit{n_logit_comp}_al{al_i:02d}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                        if mname in tried: continue
                        final = run_chain(triple, ww=ww, rm=rm)
                        auc = macro_auc(final)
                        diff = save_result(mname, auc, {"n_logit": n_logit_comp,
                                                         "alpha": alpha_logit})
                        marker = " ***NEW BEST***" if diff > 1e-7 else ""
                        if abs(diff) < 0.002:
                            print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section D done in {time.time()-t0:.1f}s. Best so far: {best_loo:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Section E: VLOM-style SIM blending
# Apply VLOM concept (geomean + RMS) to window-level SIM matrices
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== E: VLOM-style SIM blending ===", flush=True)
t0 = time.time()

def vlom_sim(A, B, wa=0.5):
    """VLOM blend of two SIM matrices in [0,1] range."""
    a = np.clip(A, 0.0, 1.0)
    b = np.clip(B, 0.0, 1.0)
    wb = 1.0 - wa
    geomean = np.exp(wa * np.log(a + EPS) + wb * np.log(b + EPS))
    rms     = np.sqrt(wa * a**2 + wb * b**2)
    result  = 0.5 * (geomean + rms)
    np.fill_diagonal(result, 0.0)
    return result

# VLOM(ICA-128, ICA-100) with various weights
for wa in [0.3, 0.4, 0.5, 0.6, 0.7]:
    SIM_vlom_pair = vlom_sim(SIM_ICA128, SIM_ICA, wa=wa)
    for k in [4, 5, 6]:
        p_vlom_p = wknn_single(SIM_vlom_pair, k=k)
        for wb_pca, wb_std in [(0.3, 0.2), (0.25, 0.25), (0.4, 0.1)]:
            wa_ica = 1.0 - wb_pca - wb_std
            if wa_ica <= 0: continue
            triple = wa_ica*p_vlom_p + wb_pca*p_pca3 + wb_std*p_std5
            for ww in [0.03, 0.04, 0.05]:
                for rm in [0.26, 0.28, 0.30]:
                    wa_i = int(wa*10)
                    wb_i = int(wb_pca*10)
                    mname = f"vlom128x100_wa{wa_i}_wpc{wb_i}_k{k}_ww{int(ww*100)}_rm{int(rm*100)}"
                    if mname in tried: continue
                    final = run_chain(triple, ww=ww, rm=rm)
                    auc = macro_auc(final)
                    diff = save_result(mname, auc, {"type": "vlom_ica128x100", "wa": wa})
                    marker = " ***NEW BEST***" if diff > 1e-7 else ""
                    if abs(diff) < 0.002:
                        print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

# VLOM(multi-seed ICA-128 avg, ICA-100)
SIM_ms3_avg = np.mean([SIM_ica128_seeds[s] for s in SEEDS[:3]], axis=0)
np.fill_diagonal(SIM_ms3_avg, 0.0)
for wa in [0.4, 0.5, 0.6]:
    SIM_vlom_ms3 = vlom_sim(SIM_ms3_avg, SIM_ICA, wa=wa)
    for k in [4, 5]:
        p_vlom_ms3 = wknn_single(SIM_vlom_ms3, k=k)
        triple = 0.5*p_vlom_ms3 + 0.3*p_pca3 + 0.2*p_std5
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.26, 0.28, 0.30]:
                wa_i = int(wa*10)
                mname = f"vlom_ms3x100_wa{wa_i}_k{k}_ww{int(ww*100)}_rm{int(rm*100)}"
                if mname in tried: continue
                final = run_chain(triple, ww=ww, rm=rm)
                auc = macro_auc(final)
                diff = save_result(mname, auc, {"type": "vlom_ms3_ica100", "wa": wa})
                marker = " ***NEW BEST***" if diff > 1e-7 else ""
                if abs(diff) < 0.002:
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

# VLOM(ICA-128, PCA-80)
for wa in [0.4, 0.5, 0.6, 0.7]:
    SIM_vlom_ip = vlom_sim(SIM_ICA128, SIM_PCA, wa=wa)
    for k in [4, 5]:
        p_vlom_ip = wknn_single(SIM_vlom_ip, k=k)
        triple = 0.5*p_vlom_ip + 0.3*p_pca3 + 0.2*p_std5
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.26, 0.28, 0.30]:
                wa_i = int(wa*10)
                mname = f"vlom_ica128xpca_wa{wa_i}_k{k}_ww{int(ww*100)}_rm{int(rm*100)}"
                if mname in tried: continue
                final = run_chain(triple, ww=ww, rm=rm)
                auc = macro_auc(final)
                diff = save_result(mname, auc, {"type": "vlom_ica128_pca", "wa": wa})
                marker = " ***NEW BEST***" if diff > 1e-7 else ""
                if abs(diff) < 0.002:
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section E done in {time.time()-t0:.1f}s. Best so far: {best_loo:.6f}", flush=True)

# ─── Final summary ──────────────────────────────────────────────────────────
print(f"\n[batch{BATCH}] Done. Final best: {res['best']['method']} LOO={res['best']['loo_auc']:.6f}",
      flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == BATCH]
print(f"[batch{BATCH}] Experiments this batch: {len(exps_this)}", flush=True)
if exps_this:
    top = sorted(exps_this, key=lambda x: x["loo_auc"], reverse=True)[:5]
    print(f"[batch{BATCH}] Top-5 this batch:", flush=True)
    for e in top:
        print(f"  {e['method']}: {e['loo_auc']:.6f}", flush=True)
