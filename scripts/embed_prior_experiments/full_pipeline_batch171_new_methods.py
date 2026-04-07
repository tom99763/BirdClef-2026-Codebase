"""
batch171 — New embedding methods: Mahalanobis KNN, Logit-space KNN,
           Attention-weighted KNN, Bayesian Ridge, RBF+LogReg
===============================================================================
Current best: wkt3_w1_ks7_rm28 LOO=0.995970
  wknn_comb = 0.5*p_ica8 + 0.3*p_pca5 + 0.2*p_std7
  chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_comb
  final = 0.72*chk4 + 0.28*rank_norm

Strategy: Test each new method as replacement for wknn_comb (4th component),
then fine-tune blend weight w_new. Also test as standalone and blend.

Methods:
  A: Mahalanobis-KNN   — whitened PCA → Euclidean (= Mahalanobis in PCA space)
  B: Logit-space KNN   — KNN in sigmoid(logits) feature space
  C: Attention-KNN     — window KNN reweighted by logit confidence
  D: Gaussian density  — per-species Gaussian centroid proximity (LOO)
  E: Bayesian Ridge    — per-species BayesianRidge on PCA-64 (LOO)
  F: Blend search      — best new method blended with existing wknn
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-8
ROOT = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

# ── Load data ──────────────────────────────────────────────────────────────────
DATA = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
emb_raw   = DATA["emb"].astype(np.float32)       # (739, 1536)
logits_raw = DATA["logits"].astype(np.float32)   # (739, 234)
labels_win = DATA["labels"].astype(np.float32)   # (739, 234)
filenames  = DATA["filenames"]                    # (739,)
n_windows  = DATA["n_windows"]                   # (66,)

# Build window→file mapping using PKL file_list (ground truth source)
file_list = list(DATA["file_list"])
n_files   = len(file_list)
n_species = labels_win.shape[1]

win_file_id = np.zeros(len(filenames), dtype=np.int32)
for fi, fn in enumerate(file_list):
    win_file_id[filenames == fn] = fi

# ── Load PKL (existing chains + embeddings) ────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

double_best  = ep["chain_double_best"]
ica_ens_alt  = ep["chain_ica_ens_alt"]
std_ens_ref  = ep["chain_std_ens_ref"]
emb_ica = ep["emb_win_ica_norm"]   # (739, 100)
emb_pca = ep["emb_win_pca_norm"]   # (739, 80)
emb_std = ep["emb_win_std_norm"]   # (739, 80)
logit_sig = ep["logit_sig_win"]    # (739, 234)  sigmoid(logits) — from PKL, not NPZ
# CRITICAL: use PKL file_labels (ground truth, not window label max)
file_labels = ep["file_labels"]    # (66, 234) — from real annotations

# ── Load results ───────────────────────────────────────────────────────────────
with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch171] Current best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def macro_auc(s, fl=file_labels):
    aucs = []
    for si in range(n_species):
        y = fl[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try: aucs.append(roc_auc_score(y, s[:, si]))
            except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, config_dict=None, batch=171):
    global best_loo
    if mname in tried:
        return score - best_loo
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": batch}
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
        print(f"  [SAVED] New best PKL: {mname} = {score:.6f}", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

# ── Baseline formula components ─────────────────────────────────────────────────
COOC_fl  = file_labels.astype(np.float32)
count_i  = COOC_fl.sum(0) + EPS
COOC     = (COOC_fl.T @ COOC_fl) / count_i[:, None]
np.fill_diagonal(COOC, 0)
raw_idf  = np.clip(np.log(float(n_files) / (count_i + 1.0 - EPS)), 0, None)
IDF075   = raw_idf ** 0.75; IDF075 /= (IDF075.mean() + EPS)

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

c3_ref     = apply_3way(double_best, alpha=0.19)
i3_ref     = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref     = apply_3way(std_ens_ref,  alpha=0.33)
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_norm  = (0.56*rank_c_ref + 0.44*rank_i_ref) / n_files

fi_wins_list    = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id != fi)[0] for fi in range(n_files)]

def eval_as_wknn(new_preds, ww=0.04, wb=0.74, wi=0.16, ws=0.06, rm=0.28):
    """Insert new_preds as 4th component replacing wknn_comb."""
    chk4 = wb*c3_ref + wi*i3_ref + ws*s3_ref + ww*new_preds
    return (1-rm)*chk4 + rm*rank_norm

def existing_wknn(SIM, k=7):
    """Compute window-level KNN predictions (existing formula)."""
    signal = labels_win
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
            w = np.clip(sims[top_l], 0, None)
            ws_sum = w.sum()
            w = w/ws_sum if ws_sum > EPS else np.ones(k_eff)/k_eff
            wp[wi_idx] = (w[:, None] * signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

print("Pre-computing existing wknn baseline...", flush=True)
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
p_ica8 = existing_wknn(SIM_ICA, k=8)
p_pca5 = existing_wknn(SIM_PCA, k=5)
p_std7 = existing_wknn(SIM_STD, k=7)
wknn_comb = 0.5*p_ica8 + 0.3*p_pca5 + 0.2*p_std7
v_base = eval_as_wknn(wknn_comb, ww=0.04, wb=0.74, wi=0.16, ws=0.06, rm=0.28)
print(f"Verify baseline: {macro_auc(v_base):.6f} (expect ~0.995970)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Mahalanobis-KNN
# Whiten embeddings in PCA space by dividing each component by its std
# → Euclidean in whitened space = Mahalanobis in original PCA space
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Mahalanobis-KNN ===", flush=True)

# Whiten ICA components (divide by std across windows)
ica_std = emb_ica.std(0) + EPS
emb_ica_whitened = emb_ica / ica_std  # (739, 100)
# Normalize rows to unit length for cosine
emb_ica_wn = emb_ica_whitened / (np.linalg.norm(emb_ica_whitened, axis=1, keepdims=True) + EPS)

# PCA whitened
pca_std = emb_pca.std(0) + EPS
emb_pca_whitened = emb_pca / pca_std
emb_pca_wn = emb_pca_whitened / (np.linalg.norm(emb_pca_whitened, axis=1, keepdims=True) + EPS)

SIM_ICA_W = emb_ica_wn @ emb_ica_wn.T
SIM_PCA_W = emb_pca_wn @ emb_pca_wn.T

best_a = best_loo
for ki, kp in [(8,5),(7,6),(9,5),(8,6),(7,5),(10,4),(8,4),(9,4)]:
    p_ica_w = existing_wknn(SIM_ICA_W, k=ki)
    p_pca_w = existing_wknn(SIM_PCA_W, k=kp)
    for wi_blend, wp_blend in [(0.5,0.5),(0.6,0.4),(0.4,0.6),(0.7,0.3)]:
        mahal_comb = wi_blend*p_ica_w + wp_blend*p_pca_w
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.27, 0.28, 0.29]:
                chk4 = (0.74+0.04-ww)*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*mahal_comb
                final = (1-rm)*chk4 + rm*rank_norm
                ar = macro_auc(final)
                mname = f"mahl_ki{ki}_kp{kp}_wi{int(wi_blend*10)}_ww{int(ww*100)}_rm{int(rm*100)}"
                delta = save_result(mname, ar)
                if ar > best_a: best_a = ar
                if ar > best_loo - 0.00005:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  ki={ki} kp={kp} wi={wi_blend:.1f} ww={ww:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section A: {best_a:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Logit-space KNN
# KNN using sigmoid(logits) as feature vectors (window-level)
# Rationale: logit space directly encodes species presence patterns
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== B: Logit-space KNN ===", flush=True)

# Normalize logit_sig rows to unit length
logit_sig_norm = logit_sig / (np.linalg.norm(logit_sig, axis=1, keepdims=True) + EPS)
SIM_LOGIT = logit_sig_norm @ logit_sig_norm.T  # cosine similarity in logit space

best_b = best_loo
for k_logit in [3, 5, 7, 8, 10, 12, 15]:
    p_logit = existing_wknn(SIM_LOGIT, k=k_logit)
    for ww in [0.03, 0.04, 0.05, 0.06]:
        for rm in [0.27, 0.28, 0.29]:
            # Replace wknn_comb with logit-space KNN
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*p_logit
            final = (1-rm)*chk4 + rm*rank_norm
            ar = macro_auc(final)
            mname = f"lknn_k{k_logit}_ww{int(ww*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_b: best_b = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  k={k_logit} ww={ww:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Blend logit-KNN with embedding-KNN
for k_logit in [5, 7, 8]:
    p_logit = existing_wknn(SIM_LOGIT, k=k_logit)
    for alpha_logit in [0.3, 0.5, 0.7]:
        hybrid_comb = alpha_logit*p_logit + (1-alpha_logit)*wknn_comb
        for ww in [0.04, 0.05]:
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*hybrid_comb
            final = 0.72*chk4 + 0.28*rank_norm
            ar = macro_auc(final)
            mname = f"lknn_hyb_k{k_logit}_al{int(alpha_logit*10)}_ww{int(ww*100)}"
            delta = save_result(mname, ar)
            if ar > best_b: best_b = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  hybrid k={k_logit} al={alpha_logit:.1f} ww={ww:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section B: {best_b:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Attention-weighted KNN
# Standard window KNN but reweight neighbor labels by their logit confidence
# Rationale: a neighbor that is highly confident about species X should count more
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== C: Attention-weighted KNN ===", flush=True)

def attn_wknn(SIM, k=8, temp=1.0):
    """
    KNN where label weights are multiplied by per-species logit confidence
    of the neighbor window. Higher confidence neighbor → more influence.
    """
    signal = labels_win.astype(np.float32)
    conf   = logit_sig.astype(np.float32) ** temp  # (739,234) confidence
    preds  = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins   = fi_wins_list[fi]
        other_wins= other_wins_list[fi]
        if len(fi_wins) == 0: continue
        k_eff = min(k, len(other_wins))
        wp = np.zeros((len(fi_wins), n_species), dtype=np.float32)
        for wi_idx, wkk in enumerate(fi_wins):
            sims = SIM[wkk, other_wins]
            top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
            top_w = other_wins[top_l]
            w_geo  = np.clip(sims[top_l], 0, None)          # (k,) geometric weight
            w_conf = conf[top_w]                              # (k, 234) attention
            # Combine: w_geo broadcast × w_conf → per-species weighted label
            w_geo_bc = w_geo[:, None]                         # (k,1)
            combined = w_geo_bc * (1 + w_conf)                # (k, 234) attention
            label_w  = (combined * signal[top_w]).sum(0)      # (234,)
            norm_w   = combined.sum(0) + EPS                  # (234,)
            wp[wi_idx] = label_w / norm_w
        preds[fi] = wp.mean(0)
    return preds

best_c = best_loo
for k_attn in [6, 8, 10]:
    for temp in [0.5, 1.0, 1.5, 2.0]:
        p_attn = attn_wknn(SIM_ICA, k=k_attn, temp=temp)
        for ww in [0.04, 0.05]:
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*p_attn
            final = 0.72*chk4 + 0.28*rank_norm
            ar = macro_auc(final)
            mname = f"attn_k{k_attn}_t{int(temp*10)}_ww{int(ww*100)}"
            delta = save_result(mname, ar)
            if ar > best_c: best_c = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  k={k_attn} temp={temp:.1f} ww={ww:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Blend attention with standard wknn
for k_attn in [8]:
    for temp in [1.0, 2.0]:
        p_attn = attn_wknn(SIM_ICA, k=k_attn, temp=temp)
        for alpha_attn in [0.3, 0.5, 0.7]:
            hyb = alpha_attn*p_attn + (1-alpha_attn)*wknn_comb
            for ww in [0.04, 0.05]:
                chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*hyb
                final = 0.72*chk4 + 0.28*rank_norm
                ar = macro_auc(final)
                mname = f"attn_hyb_k{k_attn}_t{int(temp*10)}_al{int(alpha_attn*10)}_ww{int(ww*100)}"
                delta = save_result(mname, ar)
                if ar > best_c: best_c = ar
                if ar > best_loo - 0.00005:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  hybrid k={k_attn} temp={temp:.1f} al={alpha_attn:.1f} ww={ww:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Gaussian centroid density (per-species, LOO)
# For each file: find per-species mean embedding (from positive windows in other files)
# Score = exp(-||file_emb_mean - species_centroid||^2 / (2*sigma^2))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== D: Gaussian centroid density ===", flush=True)

def gaussian_density_loo(emb_win, sigma=1.0):
    """
    LOO Gaussian density: for each held-out file, compute mean embedding,
    then score = similarity to per-species centroid from other files.
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins   = fi_wins_list[fi]
        other_wins= other_wins_list[fi]
        if len(fi_wins) == 0: continue
        # Mean embedding of held-out file
        q = emb_win[fi_wins].mean(0)  # (D,)
        q_norm = q / (np.linalg.norm(q) + EPS)
        # Per-species centroids from other files
        for si in range(n_species):
            # positive windows of species si in other files
            pos_mask = (labels_win[other_wins, si] > 0.5)
            if pos_mask.sum() == 0:
                # no positive examples: use marginal (all windows mean)
                centroid = emb_win[other_wins].mean(0)
            else:
                centroid = emb_win[other_wins][pos_mask].mean(0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + EPS)
            cos_sim = float(q_norm @ centroid_norm)
            preds[fi, si] = max(0.0, cos_sim)
    return preds

print("  Computing Gaussian density (ICA)...", flush=True)
p_gauss_ica = gaussian_density_loo(emb_ica, sigma=1.0)
print("  Computing Gaussian density (PCA)...", flush=True)
p_gauss_pca = gaussian_density_loo(emb_pca, sigma=1.0)
gauss_comb = 0.6*p_gauss_ica + 0.4*p_gauss_pca

best_d = best_loo
for ww in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
    for rm in [0.27, 0.28, 0.29]:
        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*gauss_comb
        final = (1-rm)*chk4 + rm*rank_norm
        ar = macro_auc(final)
        mname = f"gauss_ww{int(ww*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_d: best_d = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  ww={ww:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Blend with existing wknn
for alpha_g in [0.3, 0.5, 0.7]:
    hyb = alpha_g*gauss_comb + (1-alpha_g)*wknn_comb
    for ww in [0.04, 0.05]:
        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*hyb
        final = 0.72*chk4 + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"gauss_hyb_al{int(alpha_g*10)}_ww{int(ww*100)}"
        delta = save_result(mname, ar)
        if ar > best_d: best_d = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  gauss hybrid al={alpha_g:.1f} ww={ww:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section D: {best_d:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: File-level KNN (using mean file embedding, not window-level)
# Average all windows per file, then find KNN files
# This is coarser but less noisy than window-level
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== E: File-level mean-embed KNN ===", flush=True)

def file_knn_loo(emb_win, k=5):
    """File-level KNN: average windows per file, KNN on file embeddings."""
    # Compute file-level mean embeddings
    file_embs = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
    for fi in range(n_files):
        fi_wins = fi_wins_list[fi]
        file_embs[fi] = emb_win[fi_wins].mean(0)
    file_embs_norm = file_embs / (np.linalg.norm(file_embs, axis=1, keepdims=True) + EPS)
    SIM_FILE = file_embs_norm @ file_embs_norm.T  # (66, 66)

    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        other_files = [f for f in range(n_files) if f != fi]
        sims = SIM_FILE[fi, other_files]
        k_eff = min(k, len(other_files))
        top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
        top_f = [other_files[l] for l in top_l]
        w = np.clip(sims[top_l], 0, None)
        ws = w.sum()
        w = w/ws if ws > EPS else np.ones(k_eff)/k_eff
        preds[fi] = (w[:, None] * file_labels[top_f]).sum(0)
    return preds

best_e = best_loo
for k_file in [3, 4, 5, 6, 7, 8, 10]:
    p_file_ica = file_knn_loo(emb_ica, k=k_file)
    p_file_pca = file_knn_loo(emb_pca, k=k_file)
    p_file_std = file_knn_loo(emb_std, k=k_file)
    p_file_comb = 0.5*p_file_ica + 0.3*p_file_pca + 0.2*p_file_std
    for ww in [0.03, 0.04, 0.05, 0.06]:
        for rm in [0.27, 0.28, 0.29]:
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*p_file_comb
            final = (1-rm)*chk4 + rm*rank_norm
            ar = macro_auc(final)
            mname = f"fknn_k{k_file}_ww{int(ww*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_e: best_e = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  k={k_file} ww={ww:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
    # Blend with window-level wknn
    for alpha_f in [0.3, 0.5, 0.7]:
        hyb = alpha_f*p_file_comb + (1-alpha_f)*wknn_comb
        for ww in [0.04, 0.05]:
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*hyb
            final = 0.72*chk4 + 0.28*rank_norm
            ar = macro_auc(final)
            mname = f"fknn_hyb_k{k_file}_al{int(alpha_f*10)}_ww{int(ww*100)}"
            delta = save_result(mname, ar)
            if ar > best_e: best_e = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  file-hyb k={k_file} al={alpha_f:.1f} ww={ww:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section E: {best_e:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: Bayesian Ridge Regression (per-species, LOO, PCA-32 features)
# Fit BayesianRidge on window-level embeddings to predict window labels (LOO)
# Then average per-file
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== F: Bayesian Ridge (PCA-32) ===", flush=True)

from sklearn.linear_model import BayesianRidge
from sklearn.decomposition import PCA as skPCA

# Reduce to 32 dims for speed
pca32 = skPCA(n_components=32, random_state=42)
pca32.fit(emb_ica)
emb_ica32 = pca32.transform(emb_ica).astype(np.float32)  # (739, 32)

def bayesian_ridge_loo(emb_low, n_comp=32):
    """LOO Bayesian Ridge: per-species regression on PCA-32 embeddings."""
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    scaler = StandardScaler()

    # Pre-fit scaler on all data
    scaler.fit(emb_low)
    emb_scaled = scaler.transform(emb_low).astype(np.float32)

    for fi in range(n_files):
        fi_wins   = fi_wins_list[fi]
        other_wins= other_wins_list[fi]
        if len(fi_wins) == 0: continue

        X_train = emb_scaled[other_wins]  # (N_train, 32)
        X_test  = emb_scaled[fi_wins]     # (N_test, 32)
        Y_train = labels_win[other_wins]  # (N_train, 234)

        # Only fit species with positive examples in training
        file_pred = np.zeros(n_species, dtype=np.float32)
        for si in range(n_species):
            y_train = Y_train[:, si]
            if y_train.sum() == 0 or y_train.sum() == len(y_train):
                file_pred[si] = y_train.mean()
                continue
            br = BayesianRidge(max_iter=100)
            br.fit(X_train, y_train)
            pred_win = br.predict(X_test)
            file_pred[si] = float(np.clip(pred_win, 0, 1).mean())
        preds[fi] = file_pred

        if fi % 10 == 0:
            print(f"    BayesRidge LOO: {fi+1}/{n_files} done", flush=True)
    return preds

print("  Computing Bayesian Ridge (this takes ~10-15 min)...", flush=True)
p_br = bayesian_ridge_loo(emb_ica32)
print("  BayesRidge done.", flush=True)

best_f = best_loo
for ww in [0.02, 0.03, 0.04, 0.05, 0.06]:
    for rm in [0.27, 0.28, 0.29]:
        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*p_br
        final = (1-rm)*chk4 + rm*rank_norm
        ar = macro_auc(final)
        mname = f"br32_ww{int(ww*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_f: best_f = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  ww={ww:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Blend with existing wknn
for alpha_br in [0.3, 0.5, 0.7]:
    hyb = alpha_br*p_br + (1-alpha_br)*wknn_comb
    for ww in [0.04, 0.05]:
        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*hyb
        final = 0.72*chk4 + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"br32_hyb_al{int(alpha_br*10)}_ww{int(ww*100)}"
        delta = save_result(mname, ar)
        if ar > best_f: best_f = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  BayesRidge hybrid al={alpha_br:.1f} ww={ww:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section F: {best_f:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 171]
elapsed = time.time() - t0
print(f"Batch171 complete in {elapsed:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)

all_improved = [e for e in exps_this if e["loo_auc"] > 0.995970]
if all_improved:
    print(f"\n*** {len(all_improved)} methods beat previous best! ***")
    for e in sorted(all_improved, key=lambda x: -x["loo_auc"])[:10]:
        print(f"  {e['method']}: {e['loo_auc']:.6f}")
else:
    print("\nNo improvement over 0.995970 — plateau confirmed for these methods.")

top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
