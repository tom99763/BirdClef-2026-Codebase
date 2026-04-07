"""
batch176 — Novel methods: Score-space KNN, Combined emb+logit, Stacked blend
===============================================================================
Current best: wkt3_ki6_kp3_ks5_ww4_rm28 LOO=0.995986

batch175 covered:
  A: Bayesian Ridge PCA-64..256 + blend
  B: Nystroem+LogReg
  C: Attention KNN sweep
  D: Covariance Pooling KNN (2nd-order file repr)
  E: Random Subspace KNN

batch176 focuses on genuinely new angles:
  A: Covariance Pooling KNN (file-level 2nd-order stats, PCA-32 space) — deeper blend search
  B: Random Subspace KNN — more configs + cosine distance variants
  C: Score-space KNN — use logit_sig_win (234-dim) as features for KNN
  D: Combined emb+logit KNN — concatenate PCA-128 + normalized logit → 362-dim
  E: Stacked blend optimization — find optimal blend of wkt3 + score-space + cov-pooling
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-8
ROOT = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

# ── Load data ─────────────────────────────────────────────────────────────────
DATA = np.load(ROOT / "outputs" / "perch_labeled_ss.npz", allow_pickle=True)
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
emb_nmf = ep["emb_win_nmf_norm"]
logit_sig  = ep["logit_sig_win"]      # (739, 234) window-level sigmoid logits
labels_win = ep["labels_win"]
win_file_id= ep["win_file_id"]

# Raw data
raw_emb    = DATA["emb"].astype(np.float32)
raw_labels = DATA["labels"].astype(np.float32)
file_list  = DATA["file_list"]
filenames  = DATA["filenames"]
fname2idx  = {fn: i for i, fn in enumerate(file_list)}
file_ids_raw = np.array([fname2idx[fn] for fn in filenames], dtype=np.int32)

# ── Load results ─────────────────────────────────────────────────────────────
with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch176] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)
print(f"[batch176] Total tried: {len(tried)}", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def macro_auc(s, fl=file_labels):
    aucs = []
    for si in range(n_species):
        y = fl[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try:
                aucs.append(roc_auc_score(y, s[:, si]))
            except:
                pass
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, batch_n=176, cfg=None):
    global best_loo
    if mname in tried:
        return score - best_loo
    res["experiments"].append({"method": mname, "loo_auc": float(score), "config": cfg or {}, "batch": batch_n})
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
        print(f"  *** NEW BEST: {mname} = {score:.6f} ***", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

# ── Precompute baseline components ────────────────────────────────────────────
fl_hard = file_labels.astype(np.float32)
count_i = fl_hard.sum(0) + EPS
COOC = (fl_hard.T @ fl_hard) / count_i[:, None]
np.fill_diagonal(COOC, 0)
raw_idf = np.clip(np.log(float(n_files)/(count_i+1.0-EPS)), 0, None)
IDF075 = raw_idf**0.75
IDF075 /= (IDF075.mean()+EPS)

def soft_cooc(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0/(1.0+np.exp(np.clip(-slope*(s-center),-88,88)))
        sg = s*gate*(idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS:
            out[fi] = s
            continue
        c = COOC.T@sg
        mc = np.abs(c).max()
        if mc > EPS:
            c /= mc
        out[fi] = (1-alpha)*s + alpha*np.clip(c,0,None)
    return out

def apply_3way(s, alpha=0.200):
    sp = np.clip(s,0,1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = 0.45*s + 0.55*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.110)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.030)
    return 0.875*idf_s + 0.125*tr

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

fi_wins_list    = [np.where(win_file_id==fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id!=fi)[0] for fi in range(n_files)]

c3_ref = apply_3way(double_best, alpha=0.19)
i3_ref = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref = apply_3way(std_ens_ref,  alpha=0.33)
rank_norm = (0.56*make_rank(apply_3way(double_best,0.23)) + 0.44*make_rank(apply_3way(ica_ens_alt,0.40))) / n_files

# Best triple: wkt3 = 0.5*ICA(k=6) + 0.3*PCA(k=3) + 0.2*STD(k=5)
print("Pre-computing best wkt3 triple...", flush=True)
SIM_ICA = emb_ica@emb_ica.T
SIM_PCA = emb_pca@emb_pca.T
SIM_STD = emb_std@emb_std.T
SIM_NMF = emb_nmf@emb_nmf.T

def wknn_s(SIM, k=7):
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files,n_species), dtype=np.float32)
    for fi in range(n_files):
        fw=fi_wins_list[fi]; ow=other_wins_list[fi]
        if len(fw)==0: continue
        ke=min(k,len(ow)); wp=np.zeros((len(fw),n_species), dtype=np.float32)
        for wi,wkk in enumerate(fw):
            sims=SIM[wkk,ow]; tl=np.argpartition(-sims,ke-1)[:ke]; tw=ow[tl]
            w=np.clip(sims[tl],0,None); ws=w.sum()
            w=w/ws if ws>EPS else np.ones(ke)/ke
            wp[wi]=(w[:,None]*signal[tw]).sum(0)
        preds[fi]=wp.mean(0)
    return preds

p6=wknn_s(SIM_ICA,6); p3=wknn_s(SIM_PCA,3); p5=wknn_s(SIM_STD,5)
wknn_best = 0.5*p6+0.3*p3+0.2*p5
chk4_base = 0.74*c3_ref+0.16*i3_ref+0.06*s3_ref+0.04*wknn_best
wkt3_final = 0.72*chk4_base+0.28*rank_norm
print(f"  wkt3 LOO verify: {macro_auc(wkt3_final):.6f}", flush=True)

# Scaler for raw embeddings
scaler = StandardScaler()
X_all = scaler.fit_transform(raw_emb)

t0 = time.time()

# =============================================================================
# A: Covariance Pooling KNN — deeper parameter sweep (PCA-32 space)
# Build file-level representation: mean(32) + std(32) + upper-triangle cov(528)
# =============================================================================
print("\n=== A: Covariance Pooling KNN (deep sweep) ===", flush=True)
best_cov = best_loo

pca32 = PCA(n_components=32, random_state=42)
Z32 = pca32.fit_transform(X_all).astype(np.float32)

def build_file_cov_features(Z, file_ids, include_upper=True):
    """Per-file: mean(d) + std(d) [+ upper-triangle cov (d*(d+1)/2)] feature."""
    feats = []
    for fi in range(n_files):
        mask = file_ids==fi
        z = Z[mask]
        mu = z.mean(0)
        sg = z.std(0)+EPS
        if include_upper:
            cov = np.cov(z.T)
            if cov.ndim==0:
                cov = np.array([[float(cov)]])
            upper = cov[np.triu_indices_from(cov,k=0)]
            feats.append(np.concatenate([mu, sg, upper]))
        else:
            feats.append(np.concatenate([mu, sg]))
    return np.array(feats, dtype=np.float32)

print("  Building covariance features (PCA-32)...", flush=True)
file_cov_feats = build_file_cov_features(Z32, file_ids_raw, include_upper=True)
cov_scaler = StandardScaler()
file_cov_norm = cov_scaler.fit_transform(file_cov_feats).astype(np.float32)
# L2-normalize for cosine similarity
file_cov_l2 = normalize(file_cov_norm, norm='l2')

# Mean+std only (no upper triangle) — lighter 64-dim features
file_ms_feats = build_file_cov_features(Z32, file_ids_raw, include_upper=False)
ms_scaler = StandardScaler()
file_ms_norm = ms_scaler.fit_transform(file_ms_feats).astype(np.float32)
file_ms_l2 = normalize(file_ms_norm, norm='l2')

def file_knn_loo(feat_l2, k=5):
    """File-level LOO KNN with cosine similarity."""
    SIM_f = feat_l2 @ feat_l2.T
    preds = np.zeros((n_files,n_species), dtype=np.float32)
    for fi in range(n_files):
        other = np.array([j for j in range(n_files) if j!=fi])
        sims = SIM_f[fi, other]
        ke = min(k, len(other))
        tl = np.argpartition(-sims, ke-1)[:ke]
        w = np.clip(sims[tl], 0, None)
        ws = w.sum()
        w = w/ws if ws > EPS else np.ones(ke)/ke
        top_files = other[tl]
        preds[fi] = (w[:,None]*file_labels[top_files]).sum(0)
    return preds

# Sweep k for both feature sets
for feat_name, feat_l2 in [('cov592', file_cov_l2), ('ms64', file_ms_l2)]:
    for k in [3, 4, 5, 6, 7, 8, 10, 12]:
        mn_base = f"b176_cov_{feat_name}_k{k}"
        if mn_base in tried:
            continue
        p_f = file_knn_loo(feat_l2, k=k)
        ar = macro_auc(p_f)
        save_result(mn_base, ar, cfg={"feat": feat_name, "k": k})
        # Blend with wkt3_final
        for cw in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            bl = (1-cw)*chk4_base + cw*p_f
            for rm in [0.27, 0.28, 0.29]:
                final = (1-rm)*bl + rm*rank_norm
                ar2 = macro_auc(final)
                mn2 = f"b176_cov_{feat_name}_k{k}_cw{int(cw*100)}_rm{int(rm*100)}"
                d = save_result(mn2, ar2)
                if ar2 > best_cov:
                    best_cov = ar2
                    print(f"  [BEST CovKNN] {mn2}: {ar2:.6f} ({d:+.6f})", flush=True)

# Also try mixing cov-knn prediction with wknn_best before blending into chain
for k in [4, 5, 6]:
    mn_base = f"b176_cov_cov592_k{k}"
    p_f = file_knn_loo(file_cov_l2, k=k)
    for ww in [0.03, 0.04, 0.05]:
        for mx in [0.3, 0.5, 0.7, 1.0]:
            pm = mx*p_f + (1-mx)*wknn_best
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*pm
            for rm in [0.27, 0.28]:
                final = (1-rm)*chk4 + rm*rank_norm
                ar2 = macro_auc(final)
                mn = f"b176_cov592_k{k}_ww{int(ww*100)}_mx{int(mx*10)}_rm{int(rm*100)}"
                d = save_result(mn, ar2)
                if ar2 > best_cov:
                    best_cov = ar2
                    print(f"  [BEST CovMix] {mn}: {ar2:.6f} ({d:+.6f})", flush=True)

print(f"  Covariance Pooling KNN best: {best_cov:.6f}", flush=True)

# =============================================================================
# B: Random Subspace KNN — wider sweep incl. cosine distance in proj space
# =============================================================================
print("\n=== B: Random Subspace KNN (extended) ===", flush=True)
best_rs = best_loo

X_norm = raw_emb / (np.linalg.norm(raw_emb, axis=1, keepdims=True)+EPS)

def rs_knn_ensemble(n_proj=30, proj_dim=128, k=5, seed=42):
    """Ensemble KNN over random projections of raw embeddings."""
    rng2 = np.random.RandomState(seed)
    preds_all = []
    for _ in range(n_proj):
        proj = rng2.randn(1536, proj_dim).astype(np.float32)
        proj /= (np.linalg.norm(proj, axis=0, keepdims=True)+EPS)
        Z_proj = X_norm @ proj
        Z_proj_n = Z_proj / (np.linalg.norm(Z_proj, axis=1, keepdims=True)+EPS)
        SIM_proj = Z_proj_n @ Z_proj_n.T
        preds_all.append(wknn_s(SIM_proj, k=k))
    return np.mean(preds_all, axis=0)

# Broader sweep: varying n_proj, proj_dim, k
for n_proj, proj_dim, k in [
    (20, 128, 5), (30, 128, 6), (20, 256, 5), (30, 256, 6), (40, 128, 5),
    (50, 128, 5), (25, 64,  5), (40, 64,  6), (20, 128, 7), (30, 128, 4),
    (60, 128, 5), (20, 512, 5), (30, 512, 6)
]:
    mn_base = f"b176_rs_n{n_proj}_d{proj_dim}_k{k}"
    if mn_base in tried:
        continue
    print(f"  RandomSubspace n={n_proj} d={proj_dim} k={k}...", flush=True)
    p_rs = rs_knn_ensemble(n_proj=n_proj, proj_dim=proj_dim, k=k)
    ar = macro_auc(p_rs)
    save_result(mn_base, ar, cfg={"n_proj": n_proj, "proj_dim": proj_dim, "k": k})
    print(f"    {mn_base}: {ar:.6f}", flush=True)

    # Blend with chain
    for ww in [0.03, 0.04, 0.05]:
        for mx in [0.0, 0.3, 0.5, 0.7, 1.0]:
            pm = mx*p_rs + (1-mx)*wknn_best
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*pm
            for rm in [0.27, 0.28]:
                final = (1-rm)*chk4 + rm*rank_norm
                ar2 = macro_auc(final)
                mn2 = f"b176_rs_n{n_proj}_d{proj_dim}_k{k}_ww{int(ww*100)}_mx{int(mx*10)}_rm{int(rm*100)}"
                d = save_result(mn2, ar2)
                if ar2 > best_rs:
                    best_rs = ar2
                    print(f"  [BEST RS] {mn2}: {ar2:.6f} ({d:+.6f})", flush=True)

print(f"  Random Subspace KNN best: {best_rs:.6f}", flush=True)

# =============================================================================
# C: Score-space KNN — use logit_sig_win (234-dim) as features
# KNN in logit-probability space instead of embedding space
# =============================================================================
print("\n=== C: Score-space KNN (logit space) ===", flush=True)
best_ss = best_loo

# Build file-level logit features: max, mean, p75, p90 across windows
logit_sig_f32 = logit_sig.astype(np.float32)
file_logit_max  = np.zeros((n_files, n_species), np.float32)
file_logit_mean = np.zeros((n_files, n_species), np.float32)
file_logit_p75  = np.zeros((n_files, n_species), np.float32)
file_logit_p90  = np.zeros((n_files, n_species), np.float32)

for fi in range(n_files):
    wins = fi_wins_list[fi]
    if len(wins) == 0:
        continue
    L = logit_sig_f32[wins]
    file_logit_max[fi]  = L.max(0)
    file_logit_mean[fi] = L.mean(0)
    file_logit_p75[fi]  = np.percentile(L, 75, axis=0)
    file_logit_p90[fi]  = np.percentile(L, 90, axis=0)

# Normalize logit features for cosine similarity
def l2norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True)+EPS)

flogit_max_l2  = l2norm(file_logit_max)
flogit_mean_l2 = l2norm(file_logit_mean)
flogit_p75_l2  = l2norm(file_logit_p75)
flogit_p90_l2  = l2norm(file_logit_p90)

# Score-space file-level LOO KNN
for agg_name, feat_l2 in [
    ('max',  flogit_max_l2),
    ('mean', flogit_mean_l2),
    ('p75',  flogit_p75_l2),
    ('p90',  flogit_p90_l2)
]:
    for k in [3, 4, 5, 6, 7, 8, 10]:
        mn_base = f"b176_ss_{agg_name}_k{k}"
        if mn_base in tried:
            continue
        p_ss = file_knn_loo(feat_l2, k=k)
        ar = macro_auc(p_ss)
        save_result(mn_base, ar, cfg={"agg": agg_name, "k": k})
        # Blend with chain
        for sw in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            bl = (1-sw)*chk4_base + sw*p_ss
            for rm in [0.27, 0.28, 0.29]:
                final = (1-rm)*bl + rm*rank_norm
                ar2 = macro_auc(final)
                mn2 = f"b176_ss_{agg_name}_k{k}_sw{int(sw*100)}_rm{int(rm*100)}"
                d = save_result(mn2, ar2)
                if ar2 > best_ss:
                    best_ss = ar2
                    print(f"  [BEST SS] {mn2}: {ar2:.6f} ({d:+.6f})", flush=True)

# Score-space KNN at window level — use logit similarity to find neighbor windows
print("  Score-space window KNN...", flush=True)
logit_win_l2 = l2norm(logit_sig_f32)
SIM_logit = logit_win_l2 @ logit_win_l2.T  # (739, 739) cosine in logit space

for k in [3, 4, 5, 6, 7, 8]:
    mn_base = f"b176_ssw_k{k}"
    if mn_base in tried:
        continue
    p_ssw = wknn_s(SIM_logit, k=k)  # window-level predictions using logit similarity
    ar = macro_auc(p_ssw)
    save_result(mn_base, ar, cfg={"k": k, "space": "logit_cosine"})
    for ww in [0.03, 0.04, 0.05]:
        for mx in [0.0, 0.3, 0.5, 0.7, 1.0]:
            pm = mx*p_ssw + (1-mx)*wknn_best
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*pm
            for rm in [0.27, 0.28]:
                final = (1-rm)*chk4 + rm*rank_norm
                ar2 = macro_auc(final)
                mn2 = f"b176_ssw_k{k}_ww{int(ww*100)}_mx{int(mx*10)}_rm{int(rm*100)}"
                d = save_result(mn2, ar2)
                if ar2 > best_ss:
                    best_ss = ar2
                    print(f"  [BEST SSW] {mn2}: {ar2:.6f} ({d:+.6f})", flush=True)

print(f"  Score-space KNN best: {best_ss:.6f}", flush=True)

# =============================================================================
# D: Combined embedding + logit KNN (joint 362-dim feature)
# Concatenate PCA-128 embeddings + L2-normalized logit features
# =============================================================================
print("\n=== D: Combined emb+logit KNN ===", flush=True)
best_comb = best_loo

# Build PCA-128 of embeddings
pca128 = PCA(n_components=128, random_state=42)
Z128 = pca128.fit_transform(X_all).astype(np.float32)
# Build file-level mean-pooled PCA-128
file_pca128 = np.zeros((n_files, 128), np.float32)
for fi in range(n_files):
    mask = file_ids_raw == fi
    file_pca128[fi] = Z128[mask].mean(0)
file_pca128_l2 = l2norm(file_pca128)  # (66, 128)

# Also PCA-64 for a lighter variant
pca64 = PCA(n_components=64, random_state=42)
Z64 = pca64.fit_transform(X_all).astype(np.float32)
file_pca64 = np.zeros((n_files, 64), np.float32)
for fi in range(n_files):
    mask = file_ids_raw == fi
    file_pca64[fi] = Z64[mask].mean(0)
file_pca64_l2 = l2norm(file_pca64)

# Concatenate emb + logit_agg features for file-level KNN
for emb_name, emb_feat in [('pca128', file_pca128_l2), ('pca64', file_pca64_l2)]:
    for agg_name, logit_feat in [
        ('max',  l2norm(file_logit_max)),
        ('mean', l2norm(file_logit_mean)),
        ('p90',  l2norm(file_logit_p90))
    ]:
        # Concatenate and re-normalize
        combo = np.concatenate([emb_feat, logit_feat], axis=1)
        combo_l2 = l2norm(combo)

        for k in [3, 4, 5, 6, 7, 8]:
            mn_base = f"b176_jt_{emb_name}_{agg_name}_k{k}"
            if mn_base in tried:
                continue
            p_jt = file_knn_loo(combo_l2, k=k)
            ar = macro_auc(p_jt)
            save_result(mn_base, ar, cfg={"emb": emb_name, "logit_agg": agg_name, "k": k})
            for jw in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
                bl = (1-jw)*chk4_base + jw*p_jt
                for rm in [0.27, 0.28, 0.29]:
                    final = (1-rm)*bl + rm*rank_norm
                    ar2 = macro_auc(final)
                    mn2 = f"b176_jt_{emb_name}_{agg_name}_k{k}_jw{int(jw*100)}_rm{int(rm*100)}"
                    d = save_result(mn2, ar2)
                    if ar2 > best_comb:
                        best_comb = ar2
                        print(f"  [BEST CombKNN] {mn2}: {ar2:.6f} ({d:+.6f})", flush=True)

print(f"  Combined emb+logit KNN best: {best_comb:.6f}", flush=True)

# =============================================================================
# E: Stacked blend optimization
# Use best component predictions from C and D to stack-blend with wkt3
# =============================================================================
print("\n=== E: Stacked blend optimization ===", flush=True)
best_stk = best_loo

# Collect best component predictions for stacking
p_ss_max5  = file_knn_loo(flogit_max_l2, k=5)
p_ss_p90_4 = file_knn_loo(flogit_p90_l2, k=4)
p_cov5     = file_knn_loo(file_cov_l2, k=5)

combo_pca128_max_l2 = l2norm(np.concatenate([file_pca128_l2, l2norm(file_logit_max)], axis=1))
p_jt5 = file_knn_loo(combo_pca128_max_l2, k=5)

# 2-way stacks: wkt3_final + each component
components = {
    'ssmax5':  p_ss_max5,
    'ssp90_4': p_ss_p90_4,
    'cov5':    p_cov5,
    'jt5':     p_jt5,
}
wkt3_logit = np.log(np.clip(wkt3_final,EPS,1-EPS)/(np.clip(1-wkt3_final,EPS,None)))

for cname, comp_pred in components.items():
    comp_logit = np.log(np.clip(comp_pred,EPS,1-EPS)/(np.clip(1-comp_pred,EPS,None)))
    for a in np.arange(0.0, 0.41, 0.05):
        # VLOM blend
        blended_logit = (1-a)*wkt3_logit + a*comp_logit
        blended = 1.0/(1.0+np.exp(-blended_logit))
        ar = macro_auc(blended)
        mn = f"b176_stk_{cname}_a{int(a*100)}"
        d = save_result(mn, ar)
        if ar > best_stk:
            best_stk = ar
            print(f"  [BEST Stk 2-way] {mn}: {ar:.6f} ({d:+.6f})", flush=True)

# 3-way stacks
for cname1, comp1 in [('ssmax5', p_ss_max5), ('cov5', p_cov5)]:
    for cname2, comp2 in [('ssp90_4', p_ss_p90_4), ('jt5', p_jt5)]:
        if cname1 == cname2:
            continue
        l1 = np.log(np.clip(comp1,EPS,1-EPS)/(np.clip(1-comp1,EPS,None)))
        l2 = np.log(np.clip(comp2,EPS,1-EPS)/(np.clip(1-comp2,EPS,None)))
        for a1 in [0.05, 0.10, 0.15, 0.20]:
            for a2 in [0.05, 0.10, 0.15]:
                if a1+a2 > 0.40:
                    continue
                blended_logit = (1-a1-a2)*wkt3_logit + a1*l1 + a2*l2
                blended = 1.0/(1.0+np.exp(-blended_logit))
                ar = macro_auc(blended)
                mn = f"b176_stk3_{cname1}_{cname2}_a1{int(a1*100)}_a2{int(a2*100)}"
                d = save_result(mn, ar)
                if ar > best_stk:
                    best_stk = ar
                    print(f"  [BEST Stk 3-way] {mn}: {ar:.6f} ({d:+.6f})", flush=True)

print(f"  Stacked blend best: {best_stk:.6f}", flush=True)

# =============================================================================
# Summary
# =============================================================================
elapsed = time.time()-t0
print(f"\n[batch176] Done in {elapsed/60:.1f} min", flush=True)
print(f"[batch176] Final best LOO: {best_loo:.6f} ({res['best']['method']})", flush=True)
print(f"[batch176] Baseline:       0.995986", flush=True)
print(f"[batch176] Improvement:    {best_loo-0.995986:+.6f}", flush=True)
print(f"[batch176] Section bests:", flush=True)
print(f"  A (Cov Pooling KNN):  {best_cov:.6f}", flush=True)
print(f"  B (Random Subspace):  {best_rs:.6f}", flush=True)
print(f"  C (Score-space KNN):  {best_ss:.6f}", flush=True)
print(f"  D (Comb emb+logit):   {best_comb:.6f}", flush=True)
print(f"  E (Stacked blend):    {best_stk:.6f}", flush=True)
