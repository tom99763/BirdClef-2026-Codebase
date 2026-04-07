"""
batch172 — Five New Methods: Mahalanobis KNN, GMM-centroid, Ridge Regression,
           RBF+LogReg (Nystroem), Attention-weighted KNN
===============================================================================
Current best: wkt3_w1_ks7_rm28 LOO=0.995970
  wknn_comb = 0.5*p_ica8 + 0.3*p_pca5 + 0.2*p_std7
  chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_comb
  final = 0.72*chk4 + 0.28*rank_norm

Strategy: Test each new method both:
  (1) As replacement for wknn_comb (4th component in chk4) — main comparison
  (2) Standalone macro-AUC — for reference

CRITICAL: Use ep["file_labels"] from PKL for COOC and macro_auc ground truth.
"""
import numpy as np
import json
import pickle
import time
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA as skPCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

EPS = 1e-8
ROOT = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

# ── Load PKL (single source of truth) ─────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

file_labels  = ep["file_labels"]       # (66, 234) GROUND TRUTH
double_best  = ep["chain_double_best"] # (66, 234)
ica_ens_alt  = ep["chain_ica_ens_alt"]
std_ens_ref  = ep["chain_std_ens_ref"]
emb_ica = ep["emb_win_ica_norm"]       # (739, 100)
emb_pca = ep["emb_win_pca_norm"]       # (739, 80)
emb_std = ep["emb_win_std_norm"]       # (739, 80)
labels_win  = ep["labels_win"]         # (739, 234) window labels
logit_sig   = ep["logit_sig_win"]      # (739, 234) sigmoid(logits)
win_file_id = ep["win_file_id"]        # (739,) file index per window

# Also load raw embeddings for methods needing them
DATA = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
emb_raw = DATA["emb"].astype(np.float32)  # (739, 1536)

n_files   = len(ep["file_list"])
n_species = file_labels.shape[1]

# ── Load results ───────────────────────────────────────────────────────────────
with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch172] Best: {res['best']['method']} LOO={best_loo:.6f}")
print(f"[batch172] Total tried: {len(tried)}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def macro_auc(s):
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try: aucs.append(roc_auc_score(y, s[:, si]))
            except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, config_dict=None):
    global best_loo
    if mname in tried:
        return score - best_loo
    entry = {"method": mname, "loo_auc": float(score),
             "config": config_dict or {}, "batch": 172}
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
        print(f"  *** NEW BEST SAVED: {mname} = {score:.6f} ***", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

# ── Baseline formula ───────────────────────────────────────────────────────────
COOC_fl = file_labels.astype(np.float32)
count_i = COOC_fl.sum(0) + EPS
COOC    = (COOC_fl.T @ COOC_fl) / count_i[:, None]
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

def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125,
               a1=0.110, a2=0.030):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf * idf_s + r_tr * tr

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

c3_ref     = apply_3way(double_best, alpha=0.19)
i3_ref     = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref     = apply_3way(std_ens_ref,  alpha=0.33)
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_norm  = (0.56*rank_c_ref + 0.44*rank_i_ref) / n_files

fi_wins_list    = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id != fi)[0] for fi in range(n_files)]

def std_wknn(SIM, k=8, signal=None):
    if signal is None: signal = labels_win
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

def formula(new_preds, ww=0.04, rm=0.28):
    chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*new_preds
    return (1-rm)*chk4 + rm*rank_norm

# Precompute baseline triple WKNN
print("Pre-computing baseline triple WKNN...", flush=True)
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
p_ica8 = std_wknn(SIM_ICA, k=8)
p_pca5 = std_wknn(SIM_PCA, k=5)
p_std7 = std_wknn(SIM_STD, k=7)
wknn_comb = 0.5*p_ica8 + 0.3*p_pca5 + 0.2*p_std7
v_base = formula(wknn_comb, ww=0.04, rm=0.28)
print(f"Baseline verify: {macro_auc(v_base):.6f} (expect ~0.995970)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Mahalanobis KNN
# Whiten PCA components by dividing by per-component std
# → Euclidean in whitened space = Mahalanobis distance
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Method 1: Mahalanobis KNN ===", flush=True)
ica_std = emb_ica.std(0) + EPS
pca_std = emb_pca.std(0) + EPS
std_std = emb_std.std(0) + EPS

emb_ica_w = emb_ica / ica_std
emb_pca_w = emb_pca / pca_std
emb_std_w = emb_std / std_std

# Row-normalize for cosine
emb_ica_wn = emb_ica_w / (np.linalg.norm(emb_ica_w, axis=1, keepdims=True) + EPS)
emb_pca_wn = emb_pca_w / (np.linalg.norm(emb_pca_w, axis=1, keepdims=True) + EPS)
emb_std_wn = emb_std_w / (np.linalg.norm(emb_std_w, axis=1, keepdims=True) + EPS)

SIM_ICA_W = emb_ica_wn @ emb_ica_wn.T
SIM_PCA_W = emb_pca_wn @ emb_pca_wn.T
SIM_STD_W = emb_std_wn @ emb_std_wn.T

best_m1 = best_loo
for ki, kp, ks in [(8,5,7),(8,5,6),(7,5,7),(9,5,7),(8,6,7),(8,4,7),(10,5,7)]:
    pm_ica = std_wknn(SIM_ICA_W, k=ki)
    pm_pca = std_wknn(SIM_PCA_W, k=kp)
    pm_std = std_wknn(SIM_STD_W, k=ks)
    for wi, wp, ws in [(0.5,0.3,0.2),(0.6,0.3,0.1),(0.4,0.4,0.2),(0.5,0.3,0.2)]:
        mahal = wi*pm_ica + wp*pm_pca + ws*pm_std
        # Standalone AUC
        sa = macro_auc(mahal)
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.27, 0.28, 0.29]:
                final = formula(mahal, ww=ww, rm=rm)
                ar = macro_auc(final)
                mname = f"mahl172_ki{ki}_kp{kp}_ks{ks}_wi{int(wi*10)}_ww{int(ww*100)}_rm{int(rm*100)}"
                delta = save_result(mname, ar, {"standalone_auc": sa})
                if ar > best_m1: best_m1 = ar
                if ar > best_loo - 0.00005:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  ki={ki} kp={kp} ks={ks} ww={ww:.2f} rm={rm:.2f}: {ar:.6f} (standalone={sa:.4f}) {delta:+.6f}{flag}", flush=True)
print(f"  Best M1: {best_m1:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: GMM-Centroid (per-species Gaussian centroid proximity)
# For each held-out file, compute cosine sim to per-species centroids
# Centroid = mean embedding of positive windows from other files
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Method 2: GMM-Centroid (per-species) ===", flush=True)

def gmm_centroid_loo(emb_w, k_neg_ratio=3.0):
    """
    Per-species centroid proximity LOO.
    Score = cosine_sim(query_mean, positive_centroid) - w_neg * cosine_sim(query_mean, all_mean)
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    # Global mean (for negative contrast)
    global_mean = emb_w.mean(0)
    global_mean_n = global_mean / (np.linalg.norm(global_mean) + EPS)

    for fi in range(n_files):
        fi_wins = fi_wins_list[fi]
        other_wins = other_wins_list[fi]
        if len(fi_wins) == 0: continue

        # Query: mean embedding of held-out file windows
        q = emb_w[fi_wins].mean(0)
        q_n = q / (np.linalg.norm(q) + EPS)

        for si in range(n_species):
            pos_mask = (labels_win[other_wins, si] > 0.5)
            if pos_mask.sum() == 0:
                preds[fi, si] = 0.0
                continue
            # Positive centroid (mean of positive windows)
            pos_centroid = emb_w[other_wins][pos_mask].mean(0)
            pos_c_n = pos_centroid / (np.linalg.norm(pos_centroid) + EPS)
            sim_pos = float(q_n @ pos_c_n)
            # Contrast with all-windows centroid (background)
            all_centroid = emb_w[other_wins].mean(0)
            all_c_n = all_centroid / (np.linalg.norm(all_centroid) + EPS)
            sim_all = float(q_n @ all_c_n)
            preds[fi, si] = max(0.0, sim_pos - 0.5 * sim_all)
    return preds

print("  Computing GMM-centroid (ICA)...", flush=True)
p_gmm_ica = gmm_centroid_loo(emb_ica)
sa_gmm = macro_auc(p_gmm_ica)
print(f"  GMM-ICA standalone AUC: {sa_gmm:.6f}", flush=True)

print("  Computing GMM-centroid (PCA)...", flush=True)
p_gmm_pca = gmm_centroid_loo(emb_pca)

best_m2 = best_loo
for wi, wp in [(0.6,0.4),(0.7,0.3),(0.5,0.5),(0.8,0.2)]:
    gmm_comb = wi*p_gmm_ica + wp*p_gmm_pca
    sa = macro_auc(gmm_comb)
    for ww in [0.03, 0.04, 0.05, 0.06]:
        for rm in [0.27, 0.28, 0.29]:
            final = formula(gmm_comb, ww=ww, rm=rm)
            ar = macro_auc(final)
            mname = f"gmm172_wi{int(wi*10)}_wp{int(wp*10)}_ww{int(ww*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar, {"standalone_auc": sa})
            if ar > best_m2: best_m2 = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  wi={wi:.1f} ww={ww:.2f} rm={rm:.2f}: {ar:.6f} (sa={sa:.4f}) {delta:+.6f}{flag}", flush=True)
# Also blend with existing wknn
for alpha_g in [0.3, 0.5]:
    hyb = alpha_g*(0.6*p_gmm_ica + 0.4*p_gmm_pca) + (1-alpha_g)*wknn_comb
    for ww in [0.04, 0.05]:
        final = formula(hyb, ww=ww, rm=0.28)
        ar = macro_auc(final)
        mname = f"gmm172_hyb_ag{int(alpha_g*10)}_ww{int(ww*100)}"
        delta = save_result(mname, ar)
        if ar > best_m2: best_m2 = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  gmm_hybrid ag={alpha_g:.1f} ww={ww:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best M2: {best_m2:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Ridge Regression (multi-output, LOO)
# Multi-output Ridge: predict all 234 species simultaneously
# Features: PCA-32 of ICA embeddings (fast)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Method 3: Ridge Regression (LOO) ===", flush=True)

# Reduce to 32 dims (ICA already decorrelated, PCA gives compact representation)
pca32 = skPCA(n_components=32, random_state=42)
pca32.fit(emb_ica)
emb32 = pca32.transform(emb_ica).astype(np.float32)

# Standardize
sc32 = StandardScaler()
sc32.fit(emb32)
emb32_s = sc32.transform(emb32).astype(np.float32)

def ridge_loo(alpha_ridge=1.0):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins = fi_wins_list[fi]
        other_wins = other_wins_list[fi]
        if len(fi_wins) == 0: continue
        X_tr = emb32_s[other_wins]
        X_te = emb32_s[fi_wins]
        Y_tr = labels_win[other_wins]  # (N_train, 234)
        # Multi-output Ridge
        rid = Ridge(alpha=alpha_ridge, fit_intercept=True)
        rid.fit(X_tr, Y_tr)
        Y_pred = rid.predict(X_te)  # (N_test, 234)
        preds[fi] = np.clip(Y_pred, 0, 1).mean(0)
        if fi % 15 == 0:
            print(f"    Ridge LOO {fi+1}/{n_files}...", flush=True)
    return preds

print("  Computing Ridge LOO (alpha=1.0)...", flush=True)
p_ridge = ridge_loo(alpha_ridge=1.0)
sa_ridge = macro_auc(p_ridge)
print(f"  Ridge standalone AUC: {sa_ridge:.6f}", flush=True)

best_m3 = best_loo
for ww in [0.02, 0.03, 0.04, 0.05, 0.06]:
    for rm in [0.27, 0.28, 0.29]:
        final = formula(p_ridge, ww=ww, rm=rm)
        ar = macro_auc(final)
        mname = f"ridge172_a100_ww{int(ww*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar, {"standalone_auc": sa_ridge, "alpha": 1.0})
        if ar > best_m3: best_m3 = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  Ridge ww={ww:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
# Try alpha=0.1
print("  Computing Ridge LOO (alpha=0.1)...", flush=True)
p_ridge01 = ridge_loo(alpha_ridge=0.1)
for ww in [0.03, 0.04, 0.05]:
    final = formula(p_ridge01, ww=ww, rm=0.28)
    ar = macro_auc(final)
    mname = f"ridge172_a010_ww{int(ww*100)}_rm28"
    delta = save_result(mname, ar)
    if ar > best_m3: best_m3 = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  Ridge(0.1) ww={ww:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
# Blend with wknn
for alpha_r in [0.3, 0.5, 0.7]:
    hyb = alpha_r*p_ridge + (1-alpha_r)*wknn_comb
    final = formula(hyb, ww=0.04, rm=0.28)
    ar = macro_auc(final)
    mname = f"ridge172_hyb_ar{int(alpha_r*10)}"
    delta = save_result(mname, ar)
    if ar > best_m3: best_m3 = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  Ridge hybrid ar={alpha_r:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best M3: {best_m3:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: RBF Kernel Approximation + Logistic Regression (Nystroem)
# Nystroem(n_components=64) + LogisticRegression per batch of species
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Method 4: RBF+LogReg (Nystroem, LOO) ===", flush=True)

def nystroem_logreg_loo(n_components=64, C=1.0):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    # Pre-fit Nystroem on full data for consistency
    nys = Nystroem(kernel='rbf', n_components=n_components, random_state=42)
    nys.fit(emb32_s)
    emb_nys_full = nys.transform(emb32_s).astype(np.float32)

    for fi in range(n_files):
        fi_wins = fi_wins_list[fi]
        other_wins = other_wins_list[fi]
        if len(fi_wins) == 0: continue
        X_tr = emb_nys_full[other_wins]
        X_te = emb_nys_full[fi_wins]
        Y_tr = labels_win[other_wins]  # (N_train, 234)

        # Multi-output Ridge (faster than per-species LR)
        rid = Ridge(alpha=1.0/C, fit_intercept=True)
        rid.fit(X_tr, Y_tr)
        Y_pred = rid.predict(X_te)
        preds[fi] = np.clip(Y_pred, 0, 1).mean(0)
        if fi % 15 == 0:
            print(f"    Nystroem LOO {fi+1}/{n_files}...", flush=True)
    return preds

print("  Computing Nystroem+Ridge LOO (n=64)...", flush=True)
p_nys = nystroem_logreg_loo(n_components=64, C=1.0)
sa_nys = macro_auc(p_nys)
print(f"  Nystroem standalone AUC: {sa_nys:.6f}", flush=True)

best_m4 = best_loo
for ww in [0.02, 0.03, 0.04, 0.05, 0.06]:
    for rm in [0.27, 0.28, 0.29]:
        final = formula(p_nys, ww=ww, rm=rm)
        ar = macro_auc(final)
        mname = f"nys172_n64_ww{int(ww*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar, {"standalone_auc": sa_nys})
        if ar > best_m4: best_m4 = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  Nystroem ww={ww:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
# Blend with wknn
for alpha_n in [0.3, 0.5]:
    hyb = alpha_n*p_nys + (1-alpha_n)*wknn_comb
    final = formula(hyb, ww=0.04, rm=0.28)
    ar = macro_auc(final)
    mname = f"nys172_hyb_an{int(alpha_n*10)}"
    delta = save_result(mname, ar)
    if ar > best_m4: best_m4 = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  Nystroem hybrid an={alpha_n:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best M4: {best_m4:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Attention-weighted KNN
# Window KNN where neighbor labels are weighted by their logit confidence
# w_combined = sim_geometric * (1 + temp * logit_sig_neighbor)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Method 5: Attention-weighted KNN ===", flush=True)

def attention_wknn(SIM, k=8, temp=1.0, conf_signal=None):
    if conf_signal is None: conf_signal = logit_sig
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins = fi_wins_list[fi]
        other_wins = other_wins_list[fi]
        if len(fi_wins) == 0: continue
        k_eff = min(k, len(other_wins))
        wp = np.zeros((len(fi_wins), n_species), dtype=np.float32)
        for wi_idx, wkk in enumerate(fi_wins):
            sims = SIM[wkk, other_wins]
            top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
            top_w = other_wins[top_l]
            w_geo = np.clip(sims[top_l], 0, None)    # (k,)
            w_conf = conf_signal[top_w] ** temp       # (k, 234)
            # Combined: geo weight × (1 + confidence) per species
            combined = w_geo[:, None] * (1.0 + w_conf)  # (k, 234)
            label_w  = (combined * signal[top_w]).sum(0)
            norm_w   = combined.sum(0) + EPS
            wp[wi_idx] = label_w / norm_w
        preds[fi] = wp.mean(0)
    return preds

print("  Computing Attention KNN (ICA, k=8, temp=1.0)...", flush=True)
best_m5 = best_loo
for k_attn in [6, 8, 10]:
    for temp in [0.5, 1.0, 1.5, 2.0]:
        p_attn = attention_wknn(SIM_ICA, k=k_attn, temp=temp)
        sa = macro_auc(p_attn)
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.27, 0.28, 0.29]:
                final = formula(p_attn, ww=ww, rm=rm)
                ar = macro_auc(final)
                mname = f"attnw172_k{k_attn}_t{int(temp*10)}_ww{int(ww*100)}_rm{int(rm*100)}"
                delta = save_result(mname, ar, {"standalone_auc": sa, "k": k_attn, "temp": temp})
                if ar > best_m5: best_m5 = ar
                if ar > best_loo - 0.00005:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  k={k_attn} temp={temp:.1f} ww={ww:.2f} rm={rm:.2f}: {ar:.6f} (sa={sa:.4f}) {delta:+.6f}{flag}", flush=True)
# Blend with wknn
for k_attn in [8]:
    for temp in [1.0, 2.0]:
        p_attn = attention_wknn(SIM_ICA, k=k_attn, temp=temp)
        for alpha_a in [0.3, 0.5, 0.7]:
            hyb = alpha_a*p_attn + (1-alpha_a)*wknn_comb
            final = formula(hyb, ww=0.04, rm=0.28)
            ar = macro_auc(final)
            mname = f"attnw172_hyb_k{k_attn}_t{int(temp*10)}_aa{int(alpha_a*10)}"
            delta = save_result(mname, ar)
            if ar > best_m5: best_m5 = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  attn_hyb k={k_attn} temp={temp:.1f} aa={alpha_a:.1f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best M5: {best_m5:.6f}\n", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
exps_this = [e for e in res["experiments"] if e.get("batch") == 172]
print(f"{'='*70}", flush=True)
print(f"Batch172 完成，耗時 {elapsed:.1f}s，共 {len(exps_this)} 個實驗", flush=True)
print(f"最終最佳 LOO: {best_loo:.6f}  方法: {res['best']['method']}", flush=True)

# Method-level summary
method_bests = {
    "Mahalanobis KNN": best_m1,
    "GMM-Centroid":    best_m2,
    "Ridge Regression": best_m3,
    "Nystroem+Ridge":  best_m4,
    "Attention KNN":   best_m5,
}
print("\n各方法最佳 LOO（blended）：")
for name, val in method_bests.items():
    diff = val - 0.995970
    print(f"  {name:22s}: {val:.6f}  ({diff:+.6f} vs 0.995970)")

improved = [e for e in exps_this if e["loo_auc"] > 0.995970]
if improved:
    print(f"\n*** {len(improved)} 個方法超越現有最佳！ ***")
    for e in sorted(improved, key=lambda x: -x["loo_auc"])[:5]:
        print(f"  {e['method']}: {e['loo_auc']:.6f}")
else:
    print("\n無方法超越 0.995970 — 確認 plateau。")

top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 本 batch：")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
