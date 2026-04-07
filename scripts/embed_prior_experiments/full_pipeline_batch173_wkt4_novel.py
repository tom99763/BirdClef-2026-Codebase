"""
batch173 — wkt4 (NMF 4th WKNN) + Logit-Guided KNN + Temporal Position KNN
===============================================================================
Current best: wkt3_w1_ks7_rm28 LOO=0.995970
  wknn_triple = 0.5*wknn_ica(k=8) + 0.3*wknn_pca(k=5) + 0.2*wknn_std(k=7)
  chk4 = 0.74*c3 + 0.16*i3 + 0.06*s3 + 0.04*wknn_triple
  final = 0.72*chk4 + 0.28*rank_norm
  c3=apply_3way(double_best,0.19), i3=apply_3way(ica_alt,0.31), s3=apply_3way(std,0.33)
  rank_c=make_rank(apply_3way(double_best,0.23)), rank_i=make_rank(apply_3way(ica_alt,0.40))
  rank_blend=0.56*rank_c+0.44*rank_i; rank_norm=rank_blend/n_files

batch170 results were lost (JSON corruption). NMF as 4th WKNN NOT properly recorded.

New directions:
  A: Verify baseline wkt3_w1_ks7_rm28 = 0.995970
  B: wkt4 — ICA+PCA+STD+NMF quad window KNN (batch170 section E redo)
     - Sweep k_nmf in {3,4,5,6,7,8}
     - Sweep blend weights for NMF component
  C: Fine k sweep for wkt3 triple around best (batch170 lost results redo)
     - k_ica {6,7,8,9,10,11}, k_pca {3,4,5,6,7}, k_std {5,6,7,8,9,10}
  D: Logit-guided window KNN
     - Weight cosine similarity by window max logit confidence
     - Use emb_win normalized + logit_sig_win for confidence gating
  E: Position-aware window KNN
     - Weight similarity by window temporal position agreement
     - Windows at same relative position more likely to have same species
  F: NMF-only triple (replace STD with NMF in wkt3)
     - wkt3_nmf = ICA(k=8)+PCA(k=5)+NMF(k=7)
     - Compare vs wkt3 (ICA+PCA+STD)
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
emb_ica = ep["emb_win_ica_norm"]    # (739, 100)
emb_pca = ep["emb_win_pca_norm"]    # (739, 80)
emb_std = ep["emb_win_std_norm"]    # (739, 80)
emb_nmf = ep["emb_win_nmf_norm"]    # (739, 100)
logit_sig  = ep["logit_sig_win"]    # (739, 234) sigmoid of Perch logits
labels_win = ep["labels_win"]       # (739, 234)
win_file_id= ep["win_file_id"]      # (739,)

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch173] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)
print(f"[batch173] Total tried: {len(tried)}", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def macro_auc(s, fl=file_labels):
    aucs = []
    for si in range(n_species):
        y = fl[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try: aucs.append(roc_auc_score(y, s[:, si]))
            except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, batch_n=173, config_dict=None):
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
        print(f"  *** NEW BEST PKL SAVED: {mname} LOO={score:.6f} ***", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

# ── Co-occurrence helpers ──────────────────────────────────────────────────────
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

# ── Pre-compute fixed components ───────────────────────────────────────────────
fi_wins_list    = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id != fi)[0] for fi in range(n_files)]

c3_ref = apply_3way(double_best, alpha=0.19)
i3_ref = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref = apply_3way(std_ens_ref,  alpha=0.33)

rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_norm  = (0.56*rank_c_ref + 0.44*rank_i_ref) / n_files

# Pre-compute window similarity matrices
print("Pre-computing similarity matrices...", flush=True)
SIM_ICA = emb_ica @ emb_ica.T  # (739, 739)
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
SIM_NMF = emb_nmf @ emb_nmf.T
print("  Done.", flush=True)

def wknn_single(SIM, k=7):
    """Standard window-level LOO KNN."""
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins    = fi_wins_list[fi]
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

t0 = time.time()

# =============================================================================
# A: VERIFY BASELINE — wkt3_w1_ks7_rm28
# =============================================================================
print("\n=== A: Verify baseline ===", flush=True)

print("  Computing wknn ICA(k=8)...", flush=True)
p_ica8 = wknn_single(SIM_ICA, k=8)
print("  Computing wknn PCA(k=5)...", flush=True)
p_pca5 = wknn_single(SIM_PCA, k=5)
print("  Computing wknn STD(k=7)...", flush=True)
p_std7 = wknn_single(SIM_STD, k=7)

wknn_triple_w1 = 0.5*p_ica8 + 0.3*p_pca5 + 0.2*p_std7  # w1 blend
chk4_base = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_triple_w1
baseline = 0.72*chk4_base + 0.28*rank_norm
baseline_auc = macro_auc(baseline)
print(f"  [VERIFY] wkt3_w1_ks7_rm28 = {baseline_auc:.6f}  (expected ~0.995970)", flush=True)
save_result("wkt3_w1_ks7_rm28_173v", baseline_auc, config_dict={"ki":8,"kp":5,"ks":7,"wi":0.5,"wp":0.3,"ws":0.2,"ww":0.04,"rm":0.28})

# =============================================================================
# B: wkt4 — Add NMF as 4th WKNN component
# =============================================================================
print("\n=== B: wkt4 NMF quad WKNN ===", flush=True)

# Pre-compute wknn for NMF at various k values
nmf_k_list = [3, 4, 5, 6, 7, 8, 9, 10]
nmf_preds = {}
for kn in nmf_k_list:
    print(f"  Computing wknn NMF(k={kn})...", flush=True)
    nmf_preds[kn] = wknn_single(SIM_NMF, k=kn)

# wkt4 blend weights: ICA, PCA, STD, NMF
# Start with w1 blend (0.5/0.3/0.2) and add NMF as explicit 4th, reduce others
# Try: (0.45, 0.25, 0.15, 0.15), (0.40, 0.30, 0.15, 0.15), (0.45, 0.30, 0.15, 0.10)
wkt4_configs = [
    # (wi, wp, ws, wn, label)
    (0.45, 0.25, 0.15, 0.15, "q4515"),
    (0.40, 0.30, 0.15, 0.15, "q4315"),
    (0.45, 0.30, 0.15, 0.10, "q4531"),
    (0.50, 0.25, 0.15, 0.10, "q5251"),
    (0.45, 0.25, 0.20, 0.10, "q4521"),
    (0.40, 0.35, 0.15, 0.10, "q4351"),
    (0.40, 0.30, 0.20, 0.10, "q4321"),
    (0.50, 0.20, 0.20, 0.10, "q5221"),
    (0.35, 0.30, 0.20, 0.15, "q3321"),
    (0.45, 0.25, 0.15, 0.15, "q4515"),
]

best_wkt4 = best_loo
for kn in [5, 6, 7, 8]:
    p_nmf_k = nmf_preds[kn]
    for wi, wp, ws, wn, lbl in wkt4_configs:
        if abs(wi+wp+ws+wn - 1.0) > 0.01: continue
        wknn_quad = wi*p_ica8 + wp*p_pca5 + ws*p_std7 + wn*p_nmf_k
        for ww in [0.03, 0.04, 0.05]:
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*wknn_quad
            for rm in [0.27, 0.28, 0.29]:
                final = (1-rm)*chk4 + rm*rank_norm
                ar = macro_auc(final)
                mname = f"wkt4_{lbl}_kn{kn}_ww{int(ww*100)}_rm{int(rm*100)}"
                delta = save_result(mname, ar)
                if ar > best_wkt4:
                    best_wkt4 = ar
                    print(f"  [BEST wkt4] {mname}: {ar:.6f} (+{delta:.6f})", flush=True)

print(f"  wkt4 best: {best_wkt4:.6f}", flush=True)

# =============================================================================
# C: Fine k sweep for wkt3 (batch170 results lost)
# =============================================================================
print("\n=== C: Fine k sweep wkt3 ===", flush=True)

# Pre-compute missing k values
k_ica_list = [6, 7, 9, 10, 11]
k_pca_list = [3, 4, 6, 7]
k_std_list = [5, 6, 8, 9, 10]

ica_preds = {8: p_ica8}
pca_preds = {5: p_pca5}
std_preds = {7: p_std7}

for ki in k_ica_list:
    print(f"  Computing wknn ICA(k={ki})...", flush=True)
    ica_preds[ki] = wknn_single(SIM_ICA, k=ki)
for kp in k_pca_list:
    print(f"  Computing wknn PCA(k={kp})...", flush=True)
    pca_preds[kp] = wknn_single(SIM_PCA, k=kp)
for ks in k_std_list:
    print(f"  Computing wknn STD(k={ks})...", flush=True)
    std_preds[ks] = wknn_single(SIM_STD, k=ks)

best_wkt3 = best_loo

for ki in [6,7,8,9,10,11]:
    for kp in [3,4,5,6,7]:
        for ks in [5,6,7,8,9,10]:
            # w1 blend (0.5/0.3/0.2)
            triple = 0.5*ica_preds[ki] + 0.3*pca_preds[kp] + 0.2*std_preds[ks]
            for ww in [0.03, 0.04, 0.05]:
                chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple
                for rm in [0.27, 0.28, 0.29]:
                    final = (1-rm)*chk4 + rm*rank_norm
                    ar = macro_auc(final)
                    mname = f"wkt3_ki{ki}_kp{kp}_ks{ks}_ww{int(ww*100)}_rm{int(rm*100)}"
                    delta = save_result(mname, ar)
                    if ar > best_wkt3:
                        best_wkt3 = ar
                        print(f"  [BEST wkt3] {mname}: {ar:.6f} (+{delta:.6f})", flush=True)

# Also try w2 blend (0.4/0.4/0.2)
for ki in [7,8,9,10]:
    for kp in [4,5,6]:
        for ks in [6,7,8]:
            triple_w2 = 0.4*ica_preds[ki] + 0.4*pca_preds[kp] + 0.2*std_preds[ks]
            chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*triple_w2
            final = 0.72*chk4 + 0.28*rank_norm
            ar = macro_auc(final)
            mname = f"wkt3w2_ki{ki}_kp{kp}_ks{ks}"
            save_result(mname, ar)
            if ar > best_wkt3:
                best_wkt3 = ar
                print(f"  [BEST wkt3w2] {mname}: {ar:.6f}", flush=True)

print(f"  wkt3 sweep best: {best_wkt3:.6f}", flush=True)

# =============================================================================
# D: Logit-guided window KNN
# =============================================================================
print("\n=== D: Logit-guided window KNN ===", flush=True)

# Idea: weight cosine similarity by the "activeness" of each window (max logit)
# Active windows contribute more to the KNN prediction
win_confidence = logit_sig.max(axis=1)  # (739,) max sigmoid logit per window

def wknn_logit_guided(SIM, k=7, conf_power=0.5):
    """KNN where neighbor similarity is boosted by neighbor confidence."""
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins    = fi_wins_list[fi]
        other_wins = other_wins_list[fi]
        if len(fi_wins) == 0: continue
        k_eff = min(k, len(other_wins))
        wp = np.zeros((len(fi_wins), n_species), dtype=np.float32)
        for wi, wkk in enumerate(fi_wins):
            sims = SIM[wkk, other_wins]
            # Boost by neighbor confidence
            conf_boost = win_confidence[other_wins] ** conf_power
            guided_sims = sims * conf_boost
            top_l = np.argpartition(-guided_sims, k_eff-1)[:k_eff]
            top_w = other_wins[top_l]
            w = np.clip(sims[top_l], 0, None)  # use raw sim for weighting
            ws = w.sum()
            w = w/ws if ws > EPS else np.ones(k_eff)/k_eff
            wp[wi] = (w[:, None] * signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

# Try logit-guided on ICA and PCA
best_lgd = best_loo
for conf_power in [0.3, 0.5, 0.7, 1.0]:
    print(f"  Logit-guided cp={conf_power:.1f} ICA(k=8)...", flush=True)
    p_lgd_ica = wknn_logit_guided(SIM_ICA, k=8, conf_power=conf_power)
    print(f"  Logit-guided cp={conf_power:.1f} PCA(k=5)...", flush=True)
    p_lgd_pca = wknn_logit_guided(SIM_PCA, k=5, conf_power=conf_power)
    print(f"  Logit-guided cp={conf_power:.1f} STD(k=7)...", flush=True)
    p_lgd_std = wknn_logit_guided(SIM_STD, k=7, conf_power=conf_power)

    # Use as drop-in replacement for wknn_triple
    lgd_triple = 0.5*p_lgd_ica + 0.3*p_lgd_pca + 0.2*p_lgd_std
    chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*lgd_triple
    for rm in [0.27, 0.28, 0.29]:
        final = (1-rm)*chk4 + rm*rank_norm
        ar = macro_auc(final)
        cp_str = str(int(conf_power*10))
        mname = f"lgd_cp{cp_str}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_lgd:
            best_lgd = ar
            print(f"  [BEST lgd] {mname}: {ar:.6f} (+{delta:.6f})", flush=True)

    # Blend logit-guided with standard triple
    for blend_lgd in [0.20, 0.30, 0.40, 0.50]:
        mixed = blend_lgd*lgd_triple + (1-blend_lgd)*wknn_triple_w1
        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*mixed
        final = 0.72*chk4 + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"lgd_blend_cp{cp_str}_b{int(blend_lgd*100)}"
        delta = save_result(mname, ar)
        if ar > best_lgd:
            best_lgd = ar
            print(f"  [BEST lgd_blend] {mname}: {ar:.6f} (+{delta:.6f})", flush=True)

print(f"  Logit-guided best: {best_lgd:.6f}", flush=True)

# =============================================================================
# E: Position-aware window KNN
# =============================================================================
print("\n=== E: Position-aware window KNN ===", flush=True)

# Assign each window a position index (0 to n_win-1 for each file)
win_positions = np.zeros(len(win_file_id), dtype=np.float32)
for fi in range(n_files):
    fi_wins = fi_wins_list[fi]
    n_w = len(fi_wins)
    if n_w > 0:
        win_positions[fi_wins] = np.arange(n_w, dtype=np.float32) / max(n_w - 1, 1)

def wknn_position_aware(SIM, k=7, sigma_pos=0.5):
    """KNN where similarity is modulated by temporal position agreement."""
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins    = fi_wins_list[fi]
        other_wins = other_wins_list[fi]
        if len(fi_wins) == 0: continue
        k_eff = min(k, len(other_wins))
        wp = np.zeros((len(fi_wins), n_species), dtype=np.float32)
        for wi, wkk in enumerate(fi_wins):
            pos_i = win_positions[wkk]
            # Position similarity: Gaussian kernel on position difference
            pos_diff = win_positions[other_wins] - pos_i
            pos_sim = np.exp(-0.5 * (pos_diff / sigma_pos) ** 2)
            # Combined: base cosine + position boost
            sims = SIM[wkk, other_wins] * (1.0 + 0.3 * pos_sim)
            top_l = np.argpartition(-sims, k_eff-1)[:k_eff]
            top_w = other_wins[top_l]
            w = np.clip(SIM[wkk, other_wins][top_l], 0, None)  # raw cosine weights
            ws = w.sum()
            w = w/ws if ws > EPS else np.ones(k_eff)/k_eff
            wp[wi] = (w[:, None] * signal[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

best_pos = best_loo
for sigma in [0.3, 0.5, 0.7, 1.0]:
    print(f"  Position-aware sigma={sigma:.1f} ICA(k=8)...", flush=True)
    p_pos_ica = wknn_position_aware(SIM_ICA, k=8, sigma_pos=sigma)
    p_pos_pca = wknn_position_aware(SIM_PCA, k=5, sigma_pos=sigma)
    p_pos_std = wknn_position_aware(SIM_STD, k=7, sigma_pos=sigma)

    pos_triple = 0.5*p_pos_ica + 0.3*p_pos_pca + 0.2*p_pos_std
    chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*pos_triple
    for rm in [0.27, 0.28, 0.29]:
        final = (1-rm)*chk4 + rm*rank_norm
        ar = macro_auc(final)
        sig_str = str(int(sigma*10))
        mname = f"pos_s{sig_str}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_pos:
            best_pos = ar
            print(f"  [BEST pos] {mname}: {ar:.6f} (+{delta:.6f})", flush=True)

    # Blend position-aware with standard
    for blend_p in [0.25, 0.50]:
        mixed = blend_p*pos_triple + (1-blend_p)*wknn_triple_w1
        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*mixed
        final = 0.72*chk4 + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"pos_blend_s{sig_str}_b{int(blend_p*100)}"
        delta = save_result(mname, ar)
        if ar > best_pos:
            best_pos = ar
            print(f"  [BEST pos_blend] {mname}: {ar:.6f} (+{delta:.6f})", flush=True)

print(f"  Position-aware best: {best_pos:.6f}", flush=True)

# =============================================================================
# F: NMF triple (ICA + PCA + NMF, replace STD with NMF)
# =============================================================================
print("\n=== F: NMF-replacing-STD triple ===", flush=True)

best_nmf_triple = best_loo
for kn in [5, 6, 7, 8, 9]:
    p_nmf_k = nmf_preds[kn]
    # w1 blend but STD→NMF
    triple_nmf = 0.5*p_ica8 + 0.3*p_pca5 + 0.2*p_nmf_k
    for ww in [0.03, 0.04, 0.05]:
        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple_nmf
        for rm in [0.27, 0.28, 0.29]:
            final = (1-rm)*chk4 + rm*rank_norm
            ar = macro_auc(final)
            mname = f"wkt3nmf_kn{kn}_ww{int(ww*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_nmf_triple:
                best_nmf_triple = ar
                print(f"  [BEST wkt3nmf] {mname}: {ar:.6f} (+{delta:.6f})", flush=True)

# Also try 4-way with both STD and NMF
for kn in [5, 6, 7, 8]:
    for ks in [6, 7, 8]:
        p_nmf_k = nmf_preds[kn]
        p_std_k = std_preds.get(ks, wknn_single(SIM_STD, k=ks))
        # 4-way: ICA(0.4)+PCA(0.25)+STD(0.20)+NMF(0.15)
        quad = 0.40*p_ica8 + 0.25*p_pca5 + 0.20*p_std_k + 0.15*p_nmf_k
        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*quad
        final = 0.72*chk4 + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wk4sn_kn{kn}_ks{ks}"
        delta = save_result(mname, ar)
        if ar > best_nmf_triple:
            best_nmf_triple = ar
            print(f"  [BEST wk4sn] {mname}: {ar:.6f} (+{delta:.6f})", flush=True)

print(f"  NMF-triple best: {best_nmf_triple:.6f}", flush=True)

# =============================================================================
# G: Ultra-fine rank mix (rm) around best
# =============================================================================
print("\n=== G: Ultra-fine rank mix ===", flush=True)

chk4_best = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_triple_w1
best_rm = best_loo
for rm_int in range(240, 330, 2):  # 0.240 to 0.330 step 0.002
    rm = rm_int / 1000.0
    final = (1-rm)*chk4_best + rm*rank_norm
    ar = macro_auc(final)
    mname = f"wkt3_w1_ks7_rm{rm_int}"
    delta = save_result(mname, ar)
    if ar > best_rm:
        best_rm = ar
        print(f"  [BEST rm] {mname}: {ar:.6f} rm={rm:.3f}", flush=True)

print(f"  Ultra-fine rm best: {best_rm:.6f}", flush=True)

# =============================================================================
# Summary
# =============================================================================
elapsed = time.time() - t0
print(f"\n[batch173] Done in {elapsed/60:.1f} min", flush=True)
print(f"[batch173] Final best: {best_loo:.6f} ({res['best']['method']})", flush=True)
print(f"[batch173] Baseline was: 0.995970", flush=True)
delta_total = best_loo - 0.995970
print(f"[batch173] Improvement: {delta_total:+.6f}", flush=True)
