"""
batch170 — Triple WKNN Fine-tuning + Quad WKNN + Score Alpha
===============================================================================
Current best: wkt3_w1_ks7_rm28 LOO=0.995970
  wknn_triple = 0.5*wknn_ica(k=8) + 0.3*wknn_pca(k=5) + 0.2*wknn_std(k=7)
  chk4 = 0.74*c3 + 0.16*i3 + 0.06*s3 + 0.04*wknn_triple
  final = 0.72*chk4 + 0.28*rank_norm

batch169 coverage:
  - k_std swept {5,6,7,8,10,12}, fixed k_ica=8, k_pca=5
  - blend weights: eq(1/3), w1(0.5/0.3/0.2), w2(0.4/0.4/0.2)
  - rm: {0.26, 0.27, 0.28}

batch170 new directions:
  A: k_ica sweep (4..12) with k_pca=5, k_std=7 (best so far)
  B: k_pca sweep (3..10) with k_ica=8, k_std=7
  C: Finer blend weight grid (step 0.05) for ICA/PCA/STD
  D: Score alpha re-tune with triple WKNN (a_best, a_ica, a_std)
  E: NMF as 4th WKNN embedding (quad WKNN: ICA+PCA+STD+NMF)
  F: rm ultra-fine search (step 0.002) around 0.27-0.29
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
labels_win  = ep["labels_win"]      # (739, 234)
win_file_id = ep["win_file_id"]     # (739,)

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch170] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 170}
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

# Reference components (batch168 best alphas)
c3_ref     = apply_3way(double_best, alpha=0.19)
i3_ref     = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref     = apply_3way(std_ens_ref,  alpha=0.33)
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_norm  = (0.56*rank_c_ref + 0.44*rank_i_ref) / n_files

# ── Pre-compute similarity matrices ────────────────────────────────────────────
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
SIM_NMF = emb_nmf @ emb_nmf.T
fi_wins_list    = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id != fi)[0] for fi in range(n_files)]

def wknn(SIM, k):
    """Window KNN with label signal."""
    sig = labels_win.astype(np.float32)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_wins   = fi_wins_list[fi]
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
            wp[wi] = (w[:, None] * sig[top_w]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

print("Pre-computing reference WKNN (ICA k=8, PCA k=5, STD k=7)...", flush=True)
p_ica8 = wknn(SIM_ICA, k=8)
p_pca5 = wknn(SIM_PCA, k=5)
p_std7 = wknn(SIM_STD, k=7)
wknn_ref = 0.5*p_ica8 + 0.3*p_pca5 + 0.2*p_std7

# Verify
chk4_ref = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_ref
v_ref = 0.72*chk4_ref + 0.28*rank_norm
print(f"Verify reference: {macro_auc(v_ref):.6f} (expect ~0.995970)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: k_ica sweep (3..14), fixed k_pca=5, k_std=7
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: k_ica sweep (k_pca=5, k_std=7 fixed) ===", flush=True)
best_a = best_loo
ica_wknns = {}
for k_ica in [3, 4, 5, 6, 7, 9, 10, 11, 12, 14]:
    ica_wknns[k_ica] = wknn(SIM_ICA, k=k_ica)
    w3 = 0.5*ica_wknns[k_ica] + 0.3*p_pca5 + 0.2*p_std7
    chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*w3
    for rm in [0.26, 0.27, 0.28, 0.29]:
        ar = macro_auc((1-rm)*chk4 + rm*rank_norm)
        mname = f"wkt3a_ki{k_ica}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_a: best_a = ar
        if ar > best_loo - 0.00007:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  k_ica={k_ica} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best A: {best_a:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: k_pca sweep (3..12), fixed k_ica=8, k_std=7
# ═══════════════════════════════════════════════════════════════════════════════
print("=== B: k_pca sweep (k_ica=8, k_std=7 fixed) ===", flush=True)
best_b = best_loo
pca_wknns = {}
for k_pca in [3, 4, 6, 7, 8, 9, 10, 12]:
    pca_wknns[k_pca] = wknn(SIM_PCA, k=k_pca)
    w3 = 0.5*p_ica8 + 0.3*pca_wknns[k_pca] + 0.2*p_std7
    chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*w3
    for rm in [0.26, 0.27, 0.28, 0.29]:
        ar = macro_auc((1-rm)*chk4 + rm*rank_norm)
        mname = f"wkt3b_kp{k_pca}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_b: best_b = ar
        if ar > best_loo - 0.00007:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  k_pca={k_pca} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best B: {best_b:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Finer blend weight grid for triple WKNN (step 0.05)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== C: Finer blend weights (step 0.05) for ICA/PCA/STD ===", flush=True)
best_c = best_loo
for wi_int in range(30, 70, 5):   # ICA weight
    wi = wi_int / 100.0
    for wp_int in range(15, 50, 5):  # PCA weight
        wp = wp_int / 100.0
        ws = round(1.0 - wi - wp, 2)
        if ws < 0.05 or ws > 0.45: continue
        w3 = wi*p_ica8 + wp*p_pca5 + ws*p_std7
        chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*w3
        for rm in [0.27, 0.28]:
            ar = macro_auc((1-rm)*chk4 + rm*rank_norm)
            mname = f"wkt3c_wi{wi_int}_wp{wp_int}_ws{int(ws*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_c: best_c = ar
            if ar > best_loo - 0.00006:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  wi={wi:.2f} wp={wp:.2f} ws={ws:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best C: {best_c:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Score alpha re-tune with triple WKNN
# ═══════════════════════════════════════════════════════════════════════════════
print("=== D: Score alpha re-tune with triple WKNN ===", flush=True)
best_d = best_loo
for a_best in [round(x/100, 2) for x in range(16, 24)]:
    c3_new = apply_3way(double_best, alpha=a_best)
    for a_ica in [round(x/100, 2) for x in range(27, 36)]:
        i3_new = apply_3way(ica_ens_alt, alpha=a_ica)
        for a_std in [round(x/100, 2) for x in range(29, 38)]:
            s3_new = apply_3way(std_ens_ref, alpha=a_std)
            chk4_new = 0.74*c3_new + 0.16*i3_new + 0.06*s3_new + 0.04*wknn_ref
            final = 0.72*chk4_new + 0.28*rank_norm
            ar = macro_auc(final)
            mname = f"wkt3d_ab{int(a_best*100)}_ai{int(a_ica*100)}_as{int(a_std*100)}"
            delta = save_result(mname, ar)
            if ar > best_d: best_d = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  a_best={a_best:.2f} a_ica={a_ica:.2f} a_std={a_std:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best D: {best_d:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Quad WKNN — add NMF as 4th embedding space
# ═══════════════════════════════════════════════════════════════════════════════
print("=== E: Quad WKNN (ICA+PCA+STD+NMF) ===", flush=True)
best_e = best_loo
nmf_wknns = {}
for k_nmf in [5, 7, 8, 10]:
    nmf_wknns[k_nmf] = wknn(SIM_NMF, k=k_nmf)
    print(f"  NMF k={k_nmf} computed", flush=True)

for k_nmf, p_nmf in nmf_wknns.items():
    # Equal quad: 1/4 each
    w4_eq = 0.25*p_ica8 + 0.25*p_pca5 + 0.25*p_std7 + 0.25*p_nmf
    chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*w4_eq
    for rm in [0.27, 0.28]:
        ar = macro_auc((1-rm)*chk4 + rm*rank_norm)
        mname = f"wkq4_eq_kn{k_nmf}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_e: best_e = ar
        if ar > best_loo - 0.00006:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  quad_eq k_nmf={k_nmf} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

    # Weighted quad: ICA 0.4, PCA 0.25, STD 0.2, NMF 0.15
    w4_w1 = 0.40*p_ica8 + 0.25*p_pca5 + 0.20*p_std7 + 0.15*p_nmf
    chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*w4_w1
    for rm in [0.27, 0.28]:
        ar = macro_auc((1-rm)*chk4 + rm*rank_norm)
        mname = f"wkq4_w1_kn{k_nmf}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_e: best_e = ar
        if ar > best_loo - 0.00006:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  quad_w1 k_nmf={k_nmf} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

    # ICA dominant: 0.45, 0.30, 0.15, 0.10
    w4_w2 = 0.45*p_ica8 + 0.30*p_pca5 + 0.15*p_std7 + 0.10*p_nmf
    chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*w4_w2
    for rm in [0.27, 0.28]:
        ar = macro_auc((1-rm)*chk4 + rm*rank_norm)
        mname = f"wkq4_w2_kn{k_nmf}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_e: best_e = ar
        if ar > best_loo - 0.00006:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  quad_w2 k_nmf={k_nmf} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best E: {best_e:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: rm ultra-fine search (step 0.002) around 0.25–0.31
# ═══════════════════════════════════════════════════════════════════════════════
print("=== F: rm ultra-fine search (step 0.002) ===", flush=True)
best_f = best_loo
for rm_int in range(250, 315, 2):
    rm = rm_int / 1000.0
    final = (1-rm)*chk4_ref + rm*rank_norm
    ar = macro_auc(final)
    mname = f"wkt3f_rm{rm_int}"
    delta = save_result(mname, ar)
    if ar > best_f: best_f = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  rm={rm:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best F: {best_f:.6f}\n", flush=True)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 170]
print(f"Batch170 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
