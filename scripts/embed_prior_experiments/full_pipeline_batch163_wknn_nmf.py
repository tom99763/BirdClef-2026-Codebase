"""
batch163 — NMF window KNN + raw embedding KNN + multi-round iteration
===============================================================================
Current best: wfip_ki8_kp5_wi5_w25 LOO=0.995927
  chk = (1-0.025)*chk_ref + 0.025*(0.5*wknn_ica_k8 + 0.5*wknn_pca_k5)
  final = 0.72*chk + 0.28*rank_norm

batch162 findings:
- Plateau at 0.995927 across many (ki, kp, wi, w) combinations
- NMF window embeddings NOT yet tried
- Raw 1536-dim embeddings NOT yet tried

Directions:
 A: NMF window KNN at various k
 B: Raw 1536-dim embeddings (normalized) window KNN
 C: NMF + ICA combination
 D: NMF + PCA combination
 E: Multi-round iteration: apply window KNN output through 3way, then re-blend
 F: Per-species IDF-weighted window KNN (rare species get more weight)
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.preprocessing import normalize
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
emb_raw   = DATA["emb"].astype(np.float32)  # (739, 1536) raw Perch embeddings

with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

file_labels  = ep["file_labels"]
double_best  = ep["chain_double_best"]
ica_ens_alt  = ep["chain_ica_ens_alt"]
std_ens_ref  = ep["chain_std_ens_ref"]

emb_ica = ep["emb_win_ica_norm"]   # (739, 100)
emb_pca = ep["emb_win_pca_norm"]   # (739, 80)
emb_nmf = ep["emb_win_nmf_norm"]   # (739, 100)
labels_win  = ep["labels_win"]
win_file_id = ep["win_file_id"]

# Normalize raw embeddings for cosine similarity
emb_raw_norm = normalize(emb_raw, norm='l2')

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch163] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 163}
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
print(f"Verify chk_ref: {macro_auc(v):.6f}", flush=True)

# Best combination from batch162
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
fi_wins_list   = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list= [np.where(win_file_id != fi)[0] for fi in range(n_files)]

def wknn(SIM, k=7):
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

# Reference best blend
p_ica8 = wknn(SIM_ICA, k=8)
p_pca5 = wknn(SIM_PCA, k=5)
best_ref_comb = 0.5*p_ica8 + 0.5*p_pca5
print(f"Reference comb (ica8+pca5): {macro_auc(best_ref_comb):.6f}", flush=True)

t0 = time.time()

# Precompute NMF and raw similarity matrices
print("Pre-computing NMF similarity...", flush=True)
SIM_NMF = emb_nmf @ emb_nmf.T

print("Pre-computing raw 1536-dim similarity (this may take a moment)...", flush=True)
SIM_RAW = emb_raw_norm @ emb_raw_norm.T

# ═══════════════════════════════════════════════════════════════════════════════
# A: NMF window KNN at various k
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: NMF window KNN ===", flush=True)
best_a = best_loo
nmf_k = {}
for k in [3, 5, 7, 8, 10, 12, 15]:
    p_nmf = wknn(SIM_NMF, k=k)
    nmf_k[k] = p_nmf
    ar_sa = macro_auc(p_nmf)
    for w in [0.015, 0.020, 0.025, 0.030]:
        chk_new = (1-w)*chk_ref + w*p_nmf
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_nmf_k{k}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_a: best_a = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  NMF k={k} sa={ar_sa:.6f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section A: {best_a:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Raw 1536-dim embedding window KNN
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Raw 1536-dim embedding window KNN ===", flush=True)
best_b = best_loo
raw_k = {}
for k in [3, 5, 7, 8, 10]:
    p_raw = wknn(SIM_RAW, k=k)
    raw_k[k] = p_raw
    ar_sa = macro_auc(p_raw)
    for w in [0.015, 0.020, 0.025, 0.030]:
        chk_new = (1-w)*chk_ref + w*p_raw
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_raw_k{k}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_b: best_b = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  RAW k={k} sa={ar_sa:.6f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section B: {best_b:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: NMF + ICA combination
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: NMF + ICA combination ===", flush=True)
best_c = best_loo
for kn, ki in [(7, 8), (7, 7), (10, 8), (5, 8)]:
    for wn_frac in [0.3, 0.5, 0.7]:
        comb_ni = wn_frac*nmf_k[kn] + (1-wn_frac)*p_ica8
        for w in [0.020, 0.025, 0.030]:
            chk_new = (1-w)*chk_ref + w*comb_ni
            final = 0.72*chk_new + 0.28*rank_norm
            ar = macro_auc(final)
            mname = f"wknn_ni_kn{kn}_ki{ki}_wn{int(wn_frac*10)}_w{int(w*1000)}"
            delta = save_result(mname, ar)
            if ar > best_c: best_c = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  NI kn={kn} ki={ki} wn={wn_frac:.1f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: NMF + ICA + PCA triple
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: NMF + ICA + PCA triple ===", flush=True)
best_d = best_loo
for kn, ki, kp in [(7, 8, 5), (7, 8, 7), (5, 8, 5)]:
    for wi in [0.4, 0.5]:
        for wn in [0.2, 0.3]:
            wp = round(1.0 - wi - wn, 2)
            if wp < 0.1: continue
            comb = wi*p_ica8 + wn*nmf_k[kn] + wp*wknn(SIM_PCA, k=kp)
            for w in [0.020, 0.025]:
                chk_new = (1-w)*chk_ref + w*comb
                final = 0.72*chk_new + 0.28*rank_norm
                ar = macro_auc(final)
                mname = f"wknn_nip_kn{kn}_ki{ki}_kp{kp}_wi{int(wi*10)}_wn{int(wn*10)}_w{int(w*1000)}"
                delta = save_result(mname, ar)
                if ar > best_d: best_d = ar
                if ar > best_loo - 0.00005:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  NIP kn={kn} wi={wi:.1f} wn={wn:.1f} wp={wp:.1f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section D: {best_d:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Raw + ICA combination
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Raw + ICA combination ===", flush=True)
best_e = best_loo
for kr in [5, 7, 8]:
    p_r = raw_k.get(kr, wknn(SIM_RAW, k=kr))
    for wr_frac in [0.3, 0.5, 0.7]:
        comb_ri = wr_frac*p_r + (1-wr_frac)*p_ica8
        for w in [0.020, 0.025, 0.030]:
            chk_new = (1-w)*chk_ref + w*comb_ri
            final = 0.72*chk_new + 0.28*rank_norm
            ar = macro_auc(final)
            mname = f"wknn_ri_kr{kr}_wr{int(wr_frac*10)}_w{int(w*1000)}"
            delta = save_result(mname, ar)
            if ar > best_e: best_e = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  RI kr={kr} wr={wr_frac:.1f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section E: {best_e:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: Multi-round iteration on window KNN output
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== F: Multi-round iteration ===", flush=True)
best_f = best_loo

# Round 1: apply 3way smoothing on window KNN output
for alpha_iter in [0.10, 0.15, 0.20]:
    p_smooth = apply_3way(best_ref_comb, alpha=alpha_iter)
    for w in [0.015, 0.020, 0.025, 0.030]:
        chk_new = (1-w)*chk_ref + w*p_smooth
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_iter_a{int(alpha_iter*100)}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_f: best_f = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  iter a={alpha_iter:.2f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Two-round iteration: intermediate blend -> re-smooth
for alpha1, alpha2 in [(0.10, 0.15), (0.15, 0.20)]:
    s1 = apply_3way(best_ref_comb, alpha=alpha1)
    s2 = apply_3way(s1, alpha=alpha2)
    for w in [0.015, 0.020, 0.025]:
        chk_new = (1-w)*chk_ref + w*s2
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_iter2_a1{int(alpha1*100)}_a2{int(alpha2*100)}_w{int(w*1000)}"
        delta = save_result(mname, ar)
        if ar > best_f: best_f = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  iter2 a1={alpha1:.2f} a2={alpha2:.2f} w={w:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section F: {best_f:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 163]
print(f"Batch163 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
