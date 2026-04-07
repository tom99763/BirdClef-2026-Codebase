"""
batch178 — New Projection Spaces: FactorAnalysis / KernelPCA / ICA variants
=============================================================================
Current best: wkt3_ki6_kp3_ks5_ww4_rm28 LOO=0.995986
All existing methods exhausted. This batch computes BRAND NEW embedding spaces
from raw 1536-dim Perch embeddings (not in existing PKL) and adds them to
the wkt3/wkt4 framework.

  A: Verify baseline (0.995986)
  B: Factor Analysis (80-dim) — different generative model from PCA
  C: Kernel PCA RBF (32-dim) — nonlinear projection
  D: FastICA with 64-dim (vs existing 100-dim ICA)
  E: wkt5 — 5-component: ICA(k6)+PCA(k3)+STD(k5)+NMF(k5)+FA(k4)
  F: wkt4 with KernelPCA replacing STD
  G: Sparse random projection (Johnson-Lindenstrauss, 128-dim)
"""
import numpy as np
import json, pickle, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import FactorAnalysis, KernelPCA, FastICA
from sklearn.preprocessing import normalize
from sklearn.random_projection import SparseRandomProjection
import warnings
warnings.filterwarnings('ignore')

EPS   = 1e-8
BATCH = 178
ROOT  = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

# ── Load pre-computed PKL (existing embeddings + helpers) ─────────────────────
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
labels_win  = ep["labels_win"]         # (739, 234)
win_file_id = ep["win_file_id"]        # (739,)

# Load raw 1536-dim embeddings for new projections
RAW_NPZ   = ROOT / "outputs" / "perch_labeled_ss.npz"
DATA      = np.load(RAW_NPZ)
raw_emb   = DATA["emb"].astype(np.float32)   # (739, 1536)
n_files   = len(DATA["n_windows"])
n_species = DATA["labels"].shape[1]

# Filenames → file_id mapping
filenames = DATA["filenames"]
file_list = DATA["file_list"]
fname_to_id = {f: i for i, f in enumerate(file_list)}
win_file_id_raw = np.array([fname_to_id.get(str(f), -1) for f in filenames], dtype=np.int32)

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
        ep_up["method"] = mname; ep_up["loo_auc"] = float(score)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(ep_up, f)
        print(f"  *** NEW BEST: {mname} LOO={score:.6f} ***", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

# ── Co-occurrence / 3-way helpers ─────────────────────────────────────────────
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
    sp  = np.clip(s, 0, 1)**2
    sc  = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
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
s3_ref     = apply_3way(std_ens_ref, alpha=0.33)
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_norm  = (0.56*rank_c_ref + 0.44*rank_i_ref) / n_files

print("Pre-computing existing similarity matrices...", flush=True)
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
SIM_NMF = emb_nmf @ emb_nmf.T

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
# A: Verify baseline
# =============================================================================
print("\n=== A: Verify baseline ===", flush=True)
p_ica6 = wknn_single(SIM_ICA, k=6)
p_pca3 = wknn_single(SIM_PCA, k=3)
p_std5 = wknn_single(SIM_STD, k=5)
wkt3   = 0.5*p_ica6 + 0.3*p_pca3 + 0.2*p_std5
chk4   = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wkt3
base   = (1-0.28)*chk4 + 0.28*rank_norm
print(f"  Verify: {macro_auc(base):.6f}", flush=True)

# =============================================================================
# B: Factor Analysis (80-dim) — new projection space
# =============================================================================
print("\n=== B: Factor Analysis embedding ===", flush=True)

# PCA-whiten first (1536 → 256), then FA (256 → 80)
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.preprocessing import StandardScaler

scaler_raw = StandardScaler()
raw_scaled = scaler_raw.fit_transform(raw_emb)  # (739, 1536)
pca_pre    = sklearn_PCA(n_components=256, random_state=42)
raw_pca256 = pca_pre.fit_transform(raw_scaled)  # (739, 256)

for fa_dim in [48, 64, 80, 96]:
    print(f"  Computing FA({fa_dim})...", flush=True)
    try:
        fa = FactorAnalysis(n_components=fa_dim, random_state=42, max_iter=500)
        emb_fa = fa.fit_transform(raw_pca256).astype(np.float32)
        emb_fa = normalize(emb_fa, norm='l2')

        SIM_FA = emb_fa @ emb_fa.T
        for kf in [3, 4, 5, 6, 7]:
            p_fa = wknn_single(SIM_FA, k=kf)
            for ww in [0.03, 0.04, 0.05, 0.06]:
                for rm in [0.27, 0.28, 0.29]:
                    # Replace STD with FA in wkt3
                    triple_fa = 0.5*p_ica6 + 0.3*p_pca3 + 0.2*p_fa
                    chk4_fa = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple_fa
                    final_fa = (1-rm)*chk4_fa + rm*rank_norm
                    ar = macro_auc(final_fa)
                    mname = f"wkt3fa_d{fa_dim}_kf{kf}_ww{int(ww*100)}_rm{int(rm*100)}"
                    d = save_result(mname, ar, {"fa_dim":fa_dim,"kf":kf,"ww":ww,"rm":rm})
                    if d > 1e-7: print(f"    *** IMPROVEMENT: {mname}: {ar:.6f} (+{d:.6f})", flush=True)
    except Exception as e:
        print(f"  FA({fa_dim}) failed: {e}", flush=True)

bests_B = [e['loo_auc'] for e in res['experiments'] if e.get('batch')==BATCH and 'fa_d' in e['method']]
print(f"  FA best: {max(bests_B) if bests_B else 'N/A':.6f}", flush=True)

# =============================================================================
# C: Kernel PCA RBF (nonlinear projection)
# =============================================================================
print("\n=== C: Kernel PCA RBF embedding ===", flush=True)

for kpca_dim in [32, 48, 64]:
    for gamma in [0.001, 0.005, 0.01]:
        print(f"  KernelPCA(dim={kpca_dim}, γ={gamma})...", flush=True)
        try:
            kpca = KernelPCA(n_components=kpca_dim, kernel='rbf', gamma=gamma,
                             random_state=42, n_jobs=-1)
            emb_kp = kpca.fit_transform(raw_pca256).astype(np.float32)
            emb_kp = normalize(emb_kp, norm='l2')

            SIM_KP = emb_kp @ emb_kp.T
            for kf in [3, 4, 5, 6]:
                p_kp = wknn_single(SIM_KP, k=kf)
                # Replace STD with KernelPCA
                triple_kp = 0.5*p_ica6 + 0.3*p_pca3 + 0.2*p_kp
                for ww in [0.04, 0.05]:
                    for rm in [0.27, 0.28, 0.29]:
                        chk4_kp = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple_kp
                        final_kp = (1-rm)*chk4_kp + rm*rank_norm
                        ar = macro_auc(final_kp)
                        mname = f"kpca_d{kpca_dim}_g{int(gamma*1000)}_kf{kf}_ww{int(ww*100)}_rm{int(rm*100)}"
                        d = save_result(mname, ar, {"dim":kpca_dim,"gamma":gamma,"kf":kf,"ww":ww,"rm":rm})
                        if d > 1e-7: print(f"    *** IMPROVEMENT: {mname}: {ar:.6f} (+{d:.6f})", flush=True)
        except Exception as e:
            print(f"  KernelPCA failed: {e}", flush=True)

bests_C = [e['loo_auc'] for e in res['experiments'] if e.get('batch')==BATCH and e['method'].startswith('kpca')]
print(f"  KernelPCA best: {max(bests_C) if bests_C else 'N/A':.6f}", flush=True)

# =============================================================================
# D: FastICA with different n_components (64-dim, different from existing 100-dim)
# =============================================================================
print("\n=== D: FastICA 64-dim (new projection) ===", flush=True)

for ica_dim in [32, 48, 64, 128]:
    for seed in [42, 123, 999]:
        print(f"  FastICA(dim={ica_dim}, seed={seed})...", flush=True)
        try:
            ica_new = FastICA(n_components=ica_dim, random_state=seed,
                              max_iter=500, tol=0.01)
            emb_ica_new = ica_new.fit_transform(raw_pca256).astype(np.float32)
            emb_ica_new = normalize(emb_ica_new, norm='l2')

            SIM_ICA_NEW = emb_ica_new @ emb_ica_new.T
            for kf in [3, 4, 5, 6, 7, 8]:
                p_ica_new = wknn_single(SIM_ICA_NEW, k=kf)
                # Use as replacement for ICA in wkt3
                triple_new = 0.5*p_ica_new + 0.3*p_pca3 + 0.2*p_std5
                for ww in [0.04, 0.05]:
                    for rm in [0.27, 0.28, 0.29]:
                        chk4_n = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple_new
                        final_n = (1-rm)*chk4_n + rm*rank_norm
                        ar = macro_auc(final_n)
                        mname = f"ica{ica_dim}s{seed}_kf{kf}_ww{int(ww*100)}_rm{int(rm*100)}"
                        d = save_result(mname, ar, {"ica_dim":ica_dim,"seed":seed,"kf":kf,"ww":ww,"rm":rm})
                        if d > 1e-7: print(f"    *** IMPROVEMENT: {mname}: {ar:.6f} (+{d:.6f})", flush=True)
        except Exception as e:
            print(f"  FastICA({ica_dim}, {seed}) failed: {e}", flush=True)

bests_D = [e['loo_auc'] for e in res['experiments'] if e.get('batch')==BATCH and e['method'].startswith('ica')]
print(f"  FastICA-new best: {max(bests_D) if bests_D else 'N/A':.6f}", flush=True)

# =============================================================================
# E: wkt5 — 5-component: best existing 4 + Factor Analysis
# =============================================================================
print("\n=== E: wkt5 (5-component ensemble) ===", flush=True)

# Use best FA dim found so far
best_fa_entry = max(
    [e for e in res['experiments'] if e.get('batch')==BATCH and 'fa_d' in e['method']],
    key=lambda x: x['loo_auc'], default=None
)

if best_fa_entry:
    fa_dim_best = best_fa_entry['config'].get('fa_dim', 64)
    fa = FactorAnalysis(n_components=fa_dim_best, random_state=42, max_iter=500)
    emb_fa_best = fa.fit_transform(raw_pca256).astype(np.float32)
    emb_fa_best = normalize(emb_fa_best, norm='l2')
    SIM_FA_BEST = emb_fa_best @ emb_fa_best.T

    for kf in [3, 4, 5, 6]:
        p_fa_e = wknn_single(SIM_FA_BEST, k=kf)
        for ki, kp, ks, kn in [(6,3,5,5), (6,3,5,4), (7,3,5,5)]:
            _pi = p_ica6 if ki==6 else wknn_single(SIM_ICA, k=ki)
            _pp = p_pca3 if kp==3 else wknn_single(SIM_PCA, k=kp)
            _ps = p_std5 if ks==5 else wknn_single(SIM_STD, k=ks)
            p_nmf_e = wknn_single(SIM_NMF, k=kn)
            for w1, w2, w3, w4, w5 in [
                (0.35, 0.25, 0.15, 0.15, 0.10),
                (0.40, 0.25, 0.15, 0.10, 0.10),
                (0.35, 0.25, 0.15, 0.10, 0.15),
            ]:
                wkt5 = w1*_pi + w2*_pp + w3*_ps + w4*p_nmf_e + w5*p_fa_e
                for ww in [0.04, 0.05]:
                    for rm in [0.27, 0.28, 0.29]:
                        chk4_5 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*wkt5
                        final_5 = (1-rm)*chk4_5 + rm*rank_norm
                        ar = macro_auc(final_5)
                        mname = f"wkt5fa_ki{ki}_kp{kp}_ks{ks}_kn{kn}_kf{kf}_w{int(w1*10)}{int(w2*10)}{int(w3*10)}{int(w4*10)}{int(w5*10)}_rm{int(rm*100)}"
                        d = save_result(mname, ar, {"ki":ki,"kp":kp,"ks":ks,"kn":kn,"kf":kf,
                                                     "w1":w1,"w2":w2,"w3":w3,"w4":w4,"w5":w5,"rm":rm})
                        if d > 1e-7: print(f"    *** IMPROVEMENT: {mname}: {ar:.6f} (+{d:.6f})", flush=True)
    print(f"  wkt5 done", flush=True)

# =============================================================================
# F: Sparse Random Projection (Johnson-Lindenstrauss)
# =============================================================================
print("\n=== F: Sparse Random Projection ===", flush=True)

for proj_dim in [64, 128, 256]:
    for seed in [42, 7, 314]:
        try:
            srp = SparseRandomProjection(n_components=proj_dim, random_state=seed)
            emb_srp = srp.fit_transform(raw_emb).astype(np.float32)
            emb_srp = normalize(emb_srp, norm='l2')
            SIM_SRP = emb_srp @ emb_srp.T
            for kf in [3, 4, 5, 6, 7]:
                p_srp = wknn_single(SIM_SRP, k=kf)
                triple_srp = 0.5*p_ica6 + 0.3*p_pca3 + 0.2*p_srp
                for ww in [0.04, 0.05]:
                    for rm in [0.27, 0.28, 0.29]:
                        chk4_srp = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple_srp
                        final_srp = (1-rm)*chk4_srp + rm*rank_norm
                        ar = macro_auc(final_srp)
                        mname = f"srp_d{proj_dim}_s{seed}_k{kf}_ww{int(ww*100)}_rm{int(rm*100)}"
                        d = save_result(mname, ar, {"dim":proj_dim,"seed":seed,"kf":kf,"ww":ww,"rm":rm})
                        if d > 1e-7: print(f"    *** IMPROVEMENT: {mname}: {ar:.6f} (+{d:.6f})", flush=True)
        except Exception as e:
            print(f"  SRP({proj_dim},{seed}) failed: {e}", flush=True)
    print(f"  SRP dim={proj_dim}: done", flush=True)

# ── Final summary ──────────────────────────────────────────────────────────────
elapsed = time.time() - t0
batch_exps = [e for e in res['experiments'] if e.get('batch') == BATCH]
print(f"\n[batch{BATCH}] Done in {elapsed/60:.1f} min", flush=True)
print(f"[batch{BATCH}] Final best LOO: {best_loo:.6f} ({res['best']['method']})", flush=True)
print(f"[batch{BATCH}] Improvement:    {best_loo - macro_auc(base):+.6f}", flush=True)
for key, label in [('fa_d','B (FA)'), ('kpca','C (KernelPCA)'), ('ica','D (ICA-new)'),
                   ('wkt5','E (wkt5)'), ('srp','F (SRP)')]:
    vals = [e['loo_auc'] for e in batch_exps if key in e['method']]
    if vals: print(f"  {label}: {max(vals):.6f}")
