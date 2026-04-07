"""
batch164 — Window KNN with logit-blend signals and IDF-weighted outputs
===============================================================================
Current best: wfip_ki8_kp5_wi5_w25 LOO=0.995927
  chk = (1-0.025)*chk_ref + 0.025*(0.5*wknn_ica_k8 + 0.5*wknn_pca_k5)
  final = 0.72*chk + 0.28*rank_norm

batch162-163 findings:
- Plateau at 0.995927; NMF/raw/triple don't help
- Window KNN uses binary labels — try signal variants

New directions:
 A: Window KNN using soft labels (logit_sig instead of hard labels)
 B: Window KNN using probability-blended signals: alpha*label + (1-alpha)*logit_sig
 C: IDF-weighted window KNN output (boost rare species predictions)
 D: Asymmetric window KNN: more weight to windows with confident predictions
 E: Test-time chain: apply 3way on window KNN, then re-blend with chk_ref at finer w
 F: Window KNN with different embedding dimensionalities (PCA 40, 60 dims)
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

emb_ica = ep["emb_win_ica_norm"]   # (739, 100)
emb_pca = ep["emb_win_pca_norm"]   # (739, 80)
labels_win  = ep["labels_win"]     # (739, 234) hard labels
logit_sig   = ep["logit_sig_win"]  # (739, 234) probabilities
win_file_id = ep["win_file_id"]

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch164] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 164}
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

SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
fi_wins_list   = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list= [np.where(win_file_id != fi)[0] for fi in range(n_files)]

def wknn_signal(SIM, k=7, signal=None):
    """Window KNN with arbitrary signal (labels_win or logit_sig or blend)."""
    if signal is None:
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
        preds[fi] = wp.mean(0)
    return preds

def wknn_confident(SIM, k=7, conf_thr=0.5):
    """Window KNN: only use neighbor windows where max(logit_sig) > conf_thr."""
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins = fi_wins_list[fi]
        other_wins = other_wins_list[fi]
        if len(fi_wins) == 0: continue
        # Filter confident neighbors
        conf_mask = (logit_sig[other_wins].max(1) > conf_thr)
        confident_wins = other_wins[conf_mask]
        if len(confident_wins) < k:
            confident_wins = other_wins  # fallback
        k_eff = min(k, len(confident_wins))
        wp = np.zeros((len(fi_wins), n_species), dtype=np.float32)
        for wi, wk in enumerate(fi_wins):
            sims = SIM[wk, confident_wins]
            top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
            top_w = confident_wins[top_l]
            w = np.clip(sims[top_l], 0, None)
            ws = w.sum()
            w = w/ws if ws > EPS else np.ones(k_eff)/k_eff
            wp[wi] = (w[:, None] * signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

t0 = time.time()

# Reference best (ICA k=8 + PCA k=5)
print("Pre-computing reference window KNN...", flush=True)
p_ica8_lbl = wknn_signal(SIM_ICA, k=8, signal=labels_win.astype(np.float32))
p_pca5_lbl = wknn_signal(SIM_PCA, k=5, signal=labels_win.astype(np.float32))

# ═══════════════════════════════════════════════════════════════════════════════
# A: Window KNN using logit_sig (soft labels)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Window KNN with logit_sig signal ===", flush=True)
best_a = best_loo

p_ica8_lgit = wknn_signal(SIM_ICA, k=8, signal=logit_sig)
p_pca5_lgit = wknn_signal(SIM_PCA, k=5, signal=logit_sig)

# Pure logit KNN
for w in [0.015, 0.020, 0.025]:
    chk_new = (1-w)*chk_ref + w*p_ica8_lgit
    final = 0.72*chk_new + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"wknn_ica8_lgit_w{int(w*1000)}"
    delta = save_result(mname, ar)
    if ar > best_a: best_a = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  ica8_lgit w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ICA logit + PCA label
comb_lgit_lbl = 0.5*p_ica8_lgit + 0.5*p_pca5_lbl
for w in [0.015, 0.020, 0.025, 0.030]:
    chk_new = (1-w)*chk_ref + w*comb_lgit_lbl
    final = 0.72*chk_new + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"wknn_ilgitplbl_w{int(w*1000)}"
    delta = save_result(mname, ar)
    if ar > best_a: best_a = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  ica_lgit+pca_lbl w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# ICA label + PCA logit
comb_lbl_lgit = 0.5*p_ica8_lbl + 0.5*p_pca5_lgit
for w in [0.015, 0.020, 0.025, 0.030]:
    chk_new = (1-w)*chk_ref + w*comb_lbl_lgit
    final = 0.72*chk_new + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"wknn_ilblplgit_w{int(w*1000)}"
    delta = save_result(mname, ar)
    if ar > best_a: best_a = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  ica_lbl+pca_lgit w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section A: {best_a:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Mixed signals: alpha*label + (1-alpha)*logit
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Blended signal (label+logit) ===", flush=True)
best_b = best_loo

for alpha_mix in [0.3, 0.5, 0.7, 0.9]:
    mixed_sig = alpha_mix * labels_win.astype(np.float32) + (1-alpha_mix) * logit_sig
    p_ica8_mix = wknn_signal(SIM_ICA, k=8, signal=mixed_sig)
    p_pca5_mix = wknn_signal(SIM_PCA, k=5, signal=mixed_sig)
    comb_mix = 0.5*p_ica8_mix + 0.5*p_pca5_mix
    for w in [0.020, 0.025, 0.030]:
        chk_new = (1-w)*chk_ref + w*comb_mix
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_mix_a{int(alpha_mix*10)}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_b: best_b = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  mix a={alpha_mix:.1f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section B: {best_b:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: IDF-weighted window KNN output (boost rare species)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: IDF-weighted window KNN output ===", flush=True)
best_c = best_loo

comb_ref = 0.5*p_ica8_lbl + 0.5*p_pca5_lbl

for idf_pow in [0.5, 0.75, 1.0]:
    idf_w = raw_idf ** idf_pow
    idf_w = idf_w / (idf_w.mean() + EPS)
    # Scale the window KNN output by IDF
    comb_idf = comb_ref * idf_w[None, :]
    # Re-normalize to same range
    max_val = comb_idf.max()
    if max_val > EPS:
        comb_idf_norm = comb_idf / max_val
    else:
        comb_idf_norm = comb_idf
    for w in [0.015, 0.020, 0.025, 0.030]:
        chk_new = (1-w)*chk_ref + w*comb_idf_norm
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_idf_pow{int(idf_pow*100)}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_c: best_c = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  idf pow={idf_pow:.2f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Confidence-filtered window KNN
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Confidence-filtered window KNN ===", flush=True)
best_d = best_loo

for conf_thr in [0.3, 0.4, 0.5, 0.6]:
    p_conf = wknn_confident(SIM_ICA, k=8, conf_thr=conf_thr)
    comb_conf = 0.5*p_conf + 0.5*p_pca5_lbl
    ar_sa = macro_auc(p_conf)
    for w in [0.015, 0.020, 0.025, 0.030]:
        chk_new = (1-w)*chk_ref + w*comb_conf
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_conf_thr{int(conf_thr*10)}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_d: best_d = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  conf thr={conf_thr:.1f} sa={ar_sa:.6f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section D: {best_d:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: 3way smoothing on window KNN BEFORE blending (cascade)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Cascade — 3way on window KNN before blend ===", flush=True)
best_e = best_loo

comb_ref = 0.5*p_ica8_lbl + 0.5*p_pca5_lbl

for a_sm in [0.10, 0.15, 0.20, 0.23, 0.25]:
    comb_sm = apply_3way(comb_ref, alpha=a_sm)
    for w in [0.010, 0.015, 0.020, 0.025, 0.030]:
        chk_new = (1-w)*chk_ref + w*comb_sm
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_casc_a{int(a_sm*100)}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_e: best_e = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  casc a={a_sm:.2f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section E: {best_e:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: Different PCA dimensionalities for window embedding
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== F: Lower-dim PCA variants ===", flush=True)
best_f = best_loo

# Truncate existing PCA embeddings to lower dims
for n_dim in [20, 30, 40, 50, 60]:
    emb_pca_trunc = emb_pca[:, :n_dim]
    # Re-normalize
    norms = np.linalg.norm(emb_pca_trunc, axis=1, keepdims=True)
    emb_pca_trunc = emb_pca_trunc / (norms + EPS)
    SIM_PCA_trunc = emb_pca_trunc @ emb_pca_trunc.T
    p_pca_dim = wknn_signal(SIM_PCA_trunc, k=5)
    comb_dim = 0.5*p_ica8_lbl + 0.5*p_pca_dim
    ar_sa = macro_auc(p_pca_dim)
    for w in [0.020, 0.025]:
        chk_new = (1-w)*chk_ref + w*comb_dim
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_pcadim{n_dim}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_f: best_f = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  pca_dim={n_dim} sa={ar_sa:.6f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section F: {best_f:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 164]
print(f"Batch164 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
