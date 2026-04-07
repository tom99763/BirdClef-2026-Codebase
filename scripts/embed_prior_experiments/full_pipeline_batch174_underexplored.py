"""
batch174 — Deep exploration of under-explored methods
===============================================================================
Current best: wkt3_ki6_kp3_ks5_ww4_rm28 LOO=0.995986

Under-explored priority methods:
  - bayesian_ridge: only 1 variant tried!
  - nystroem (RBF+LogReg): only 4 variants tried!
  - attn_knn: only 2 variants tried!

New directions:
  A: Bayesian Ridge — sweep PCA dims {32,64,96,128,256}, alpha_1, alpha_2 priors
     Also: BayesianRidge on window-level + file-level aggregation
  B: Nystroem+LogReg — sweep gamma {0.001..1.0}, n_components {64,128,256,512},
     C_logistic {0.01..10}, kernel={rbf,poly,laplacian}
  C: Attention KNN — deeper sweep: k {3..15}, temperature {0.05..2.0},
     attention_fn={softmax, sigmoid, linear}, use as wknn_comb replacement
  D: Integrate best new method into the wkt3_ki6_kp3_ks5 baseline formula
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem, RBFSampler
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
emb_nmf = ep["emb_win_nmf_norm"]
logit_sig  = ep["logit_sig_win"]   # (739, 234)
labels_win = ep["labels_win"]      # (739, 234)
win_file_id= ep["win_file_id"]     # (739,)

# Raw 1536-dim embeddings for non-KNN methods
raw_emb = DATA["emb"].astype(np.float32)         # (739, 1536)
raw_labels = DATA["labels"].astype(np.float32)   # (739, 234)
raw_logits = DATA["logits"].astype(np.float32)   # (739, 234)
# Build file_ids from filenames + file_list
file_list_arr = DATA["file_list"]  # (66,) filenames
filenames_arr = DATA["filenames"]  # (739,) window filenames
fname_to_idx  = {fn: i for i, fn in enumerate(file_list_arr)}
file_ids_raw  = np.array([fname_to_idx[fn] for fn in filenames_arr], dtype=np.int32)
unique_files  = np.arange(n_files)

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch174] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)
print(f"[batch174] Total tried: {len(tried)}", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def macro_auc(s, fl=file_labels):
    aucs = []
    for si in range(n_species):
        y = fl[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try: aucs.append(roc_auc_score(y, s[:, si]))
            except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, batch_n=174, config_dict=None):
    global best_loo
    if mname in tried: return score - best_loo
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": batch_n}
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

# ── Co-occurrence helpers (for blending) ──────────────────────────────────────
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

def apply_3way(s, alpha=0.200):
    sp = np.clip(s, 0, 1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = 0.45*s + 0.55*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.110)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.030)
    return 0.875*idf_s + 0.125*tr

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

# ── Fixed baseline components ──────────────────────────────────────────────────
fi_wins_list    = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id != fi)[0] for fi in range(n_files)]

c3_ref = apply_3way(double_best, alpha=0.19)
i3_ref = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref = apply_3way(std_ens_ref,  alpha=0.33)
rank_c  = make_rank(apply_3way(double_best, alpha=0.23))
rank_i  = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_norm = (0.56*rank_c + 0.44*rank_i) / n_files

# Best wkt3 triple from batch173: ki=6, kp=3, ks=5
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T

def wknn_single(SIM, k=7):
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins = fi_wins_list[fi]; other_wins = other_wins_list[fi]
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

print("Pre-computing best wkt3 triple (ki=6,kp=3,ks=5)...", flush=True)
p_ica6 = wknn_single(SIM_ICA, k=6)
p_pca3 = wknn_single(SIM_PCA, k=3)
p_std5 = wknn_single(SIM_STD, k=5)
wknn_best_triple = 0.5*p_ica6 + 0.3*p_pca3 + 0.2*p_std5
chk4_base = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_best_triple
verify = 0.72*chk4_base + 0.28*rank_norm
print(f"  Verify batch173 best: {macro_auc(verify):.6f}", flush=True)

t0 = time.time()

# ── File-level aggregation helper ─────────────────────────────────────────────
def file_agg_mean(win_scores):
    """Aggregate window-level predictions to file level by mean."""
    f_scores = np.zeros((n_files, n_species), dtype=np.float32)
    for fi, fid in enumerate(unique_files):
        mask = file_ids_raw == fid
        f_scores[fi] = win_scores[mask].mean(0)
    return f_scores

def file_agg_max(win_scores):
    """Aggregate window-level predictions to file level by max."""
    f_scores = np.zeros((n_files, n_species), dtype=np.float32)
    for fi, fid in enumerate(unique_files):
        mask = file_ids_raw == fid
        f_scores[fi] = win_scores[mask].max(0)
    return f_scores

# =============================================================================
# A: Bayesian Ridge — comprehensive sweep (only 1 variant tried before!)
# =============================================================================
print("\n=== A: Bayesian Ridge comprehensive sweep ===", flush=True)

# Scaler for raw embeddings
scaler_full = StandardScaler()
X_all = scaler_full.fit_transform(raw_emb)

best_br = best_loo
for pca_dim in [32, 64, 96, 128, 192, 256]:
    print(f"  BayesianRidge PCA-{pca_dim}...", flush=True)
    pca = PCA(n_components=pca_dim, random_state=42)
    Z_all = pca.fit_transform(X_all).astype(np.float32)

    # LOO at file level
    file_preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi, fid in enumerate(unique_files):
        test_mask  = file_ids_raw == fid
        train_mask = ~test_mask
        Z_tr, Z_te = Z_all[train_mask], Z_all[test_mask]
        Y_tr = raw_labels[train_mask]

        win_preds = np.zeros((test_mask.sum(), n_species), dtype=np.float32)
        for si in range(n_species):
            y_tr = Y_tr[:, si]
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                win_preds[:, si] = y_tr.mean()
                continue
            clf = BayesianRidge(max_iter=300)
            clf.fit(Z_tr, y_tr)
            win_preds[:, si] = np.clip(clf.predict(Z_te), 0, 1)

        file_preds[fi] = win_preds.mean(0)

    ar = macro_auc(file_preds)
    mname = f"br_pca{pca_dim}"
    delta = save_result(mname, ar, config_dict={"pca_dim": pca_dim})
    print(f"    {mname}: {ar:.6f} ({delta:+.6f})", flush=True)
    if ar > best_br:
        best_br = ar

    # Try blending BR output with chk4_base
    for br_w in [0.05, 0.10, 0.15, 0.20]:
        blended = (1-br_w)*chk4_base + br_w*file_preds
        for rm in [0.27, 0.28, 0.29]:
            final = (1-rm)*blended + rm*rank_norm
            ar2 = macro_auc(final)
            mname2 = f"br_pca{pca_dim}_w{int(br_w*100)}_rm{int(rm*100)}"
            d2 = save_result(mname2, ar2)
            if ar2 > best_br:
                best_br = ar2
                print(f"    [BEST BR-blend] {mname2}: {ar2:.6f} (+{d2:.6f})", flush=True)

print(f"  BayesianRidge best: {best_br:.6f}", flush=True)

# =============================================================================
# B: Nystroem + LogReg — comprehensive sweep (only 4 variants tried!)
# =============================================================================
print("\n=== B: Nystroem+LogReg comprehensive sweep ===", flush=True)

best_nys = best_loo

# Use PCA-128 as input to Nystroem (reduces compute)
pca128 = PCA(n_components=128, random_state=42)
Z128 = pca128.fit_transform(X_all).astype(np.float32)

for kernel in ['rbf', 'laplacian']:
    for gamma in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        for n_comp in [128, 256]:
            print(f"  Nystroem {kernel} γ={gamma} n={n_comp}...", flush=True)
            try:
                nys = Nystroem(kernel=kernel, gamma=gamma, n_components=n_comp, random_state=42)
                Z_nys = nys.fit_transform(Z128).astype(np.float32)

                file_preds_nys = np.zeros((n_files, n_species), dtype=np.float32)
                for fi, fid in enumerate(unique_files):
                    test_mask  = file_ids_raw == fid
                    train_mask = ~test_mask
                    Z_tr, Z_te = Z_nys[train_mask], Z_nys[test_mask]
                    Y_tr = raw_labels[train_mask]

                    win_preds = np.zeros((test_mask.sum(), n_species), dtype=np.float32)
                    for si in range(n_species):
                        y_tr = Y_tr[:, si]
                        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                            win_preds[:, si] = y_tr.mean(); continue
                        try:
                            clf = LogisticRegression(C=1.0, max_iter=200, solver='lbfgs')
                            clf.fit(Z_tr, (y_tr > 0.5).astype(int))
                            win_preds[:, si] = clf.predict_proba(Z_te)[:, 1]
                        except:
                            win_preds[:, si] = y_tr.mean()
                    file_preds_nys[fi] = win_preds.mean(0)

                ar = macro_auc(file_preds_nys)
                mname = f"nys_{kernel[:3]}_g{int(gamma*1000)}_n{n_comp}"
                delta = save_result(mname, ar, config_dict={"kernel": kernel, "gamma": gamma, "n_comp": n_comp})
                print(f"    {mname}: {ar:.6f} ({delta:+.6f})", flush=True)

                # Blend with baseline
                for nys_w in [0.05, 0.10, 0.15]:
                    blended = (1-nys_w)*chk4_base + nys_w*file_preds_nys
                    final = 0.72*blended + 0.28*rank_norm
                    ar2 = macro_auc(final)
                    mname2 = f"nys_{kernel[:3]}_g{int(gamma*1000)}_n{n_comp}_w{int(nys_w*100)}"
                    d2 = save_result(mname2, ar2)
                    if ar2 > best_nys:
                        best_nys = ar2
                        print(f"    [BEST Nys-blend] {mname2}: {ar2:.6f} (+{d2:.6f})", flush=True)
            except Exception as ex:
                print(f"    Nystroem error: {ex}", flush=True)

print(f"  Nystroem best: {best_nys:.6f}", flush=True)

# =============================================================================
# C: Attention KNN — comprehensive sweep (only 2 variants tried!)
# =============================================================================
print("\n=== C: Attention KNN comprehensive sweep ===", flush=True)

SIM_NMF = emb_nmf @ emb_nmf.T
win_conf = logit_sig.max(axis=1)  # (739,) max sigmoid per window

def attn_wknn(SIM, k=8, temperature=0.10, agg='softmax'):
    """
    Attention-weighted KNN: use KNN logit similarities as attention scores
    to reweight the neighbor labels.
    agg: 'softmax' (sharp), 'sigmoid' (smooth), 'linear' (normalized cosine)
    """
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins = fi_wins_list[fi]; other_wins = other_wins_list[fi]
        if len(fi_wins) == 0: continue
        k_eff = min(k, len(other_wins))
        wp = np.zeros((len(fi_wins), n_species), dtype=np.float32)
        for wi, wkk in enumerate(fi_wins):
            sims = SIM[wkk, other_wins]
            top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
            top_w = other_wins[top_l]
            raw_w = sims[top_l]

            if agg == 'softmax':
                logit_w = raw_w / (temperature + EPS)
                logit_w -= logit_w.max()
                w = np.exp(logit_w); w /= (w.sum() + EPS)
            elif agg == 'sigmoid':
                w = 1.0 / (1.0 + np.exp(-raw_w / temperature))
                w /= (w.sum() + EPS)
            else:  # linear
                w = np.clip(raw_w, 0, None)
                ws = w.sum(); w = w/ws if ws > EPS else np.ones(k_eff)/k_eff

            wp[wi] = (w[:, None] * signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

best_attn = best_loo

for emb_name, SIM in [('ica', SIM_ICA), ('pca', SIM_PCA), ('std', SIM_STD), ('nmf', SIM_NMF)]:
    for agg in ['softmax', 'linear']:
        for temp in [0.05, 0.10, 0.20, 0.50]:
            for k in [5, 7, 8, 10, 12]:
                print(f"  AttnKNN {emb_name} agg={agg} T={temp} k={k}...", flush=True)
                p_attn = attn_wknn(SIM, k=k, temperature=temp, agg=agg)

                # Use as wknn_comb replacement in chk4
                for ww in [0.03, 0.04, 0.05]:
                    # Mix with best triple
                    for mix in [0.0, 0.3, 0.5, 0.7, 1.0]:
                        p_mix = mix*p_attn + (1-mix)*wknn_best_triple
                        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*p_mix
                        for rm in [0.27, 0.28]:
                            final = (1-rm)*chk4 + rm*rank_norm
                            ar = macro_auc(final)
                            t_str = str(int(temp*100)).zfill(3)
                            mname = f"attn_{emb_name}_{agg[:3]}_T{t_str}_k{k}_ww{int(ww*100)}_mx{int(mix*10)}_rm{int(rm*100)}"
                            delta = save_result(mname, ar)
                            if ar > best_attn:
                                best_attn = ar
                                print(f"  [BEST Attn] {mname}: {ar:.6f} (+{delta:.6f})", flush=True)

print(f"  Attention KNN best: {best_attn:.6f}", flush=True)

# =============================================================================
# Summary
# =============================================================================
elapsed = time.time() - t0
print(f"\n[batch174] Done in {elapsed/60:.1f} min", flush=True)
print(f"[batch174] Final best: {best_loo:.6f} ({res['best']['method']})", flush=True)
print(f"[batch174] Baseline was: 0.995986", flush=True)
print(f"[batch174] Improvement: {best_loo - 0.995986:+.6f}", flush=True)
