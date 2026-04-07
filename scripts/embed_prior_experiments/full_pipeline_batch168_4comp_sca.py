"""
batch168 — Fine-tune score alphas for 4-component formula
===============================================================================
Current best: wk4sca_ia31_sa33 LOO=0.995954
  a_ica=0.31, a_std=0.33 (score alphas re-tuned)
  c3 = apply_3way(double_best, alpha=0.200)
  i3 = apply_3way(ica_ens_alt, alpha=0.31)
  s3 = apply_3way(std_ens_ref,  alpha=0.33)
  chk4 = wb*c3 + wi*i3 + ws*s3 + ww*wknn_comb (wb,wi,ws,ww from batch167)
  final = (1-rm)*chk4 + rm*rank_norm

batch167 section D findings:
- a_ica=0.31 best (was 0.26/0.260)
- a_std=0.33 best (was 0.28/0.280)
- Best: wk4sca_ia31_sa33 = 0.995954
- Many tied at ia25_sa33, ia26_sa30 = 0.995949

Directions:
 A: Fine a_ica (0.27-0.36 step 0.005) at a_std=0.33
 B: Fine a_std (0.28-0.40 step 0.005) at a_ica=0.31
 C: Joint a_ica × a_std grid (step 0.01)
 D: Joint with wb/wi/ws/ww optimization (best chk4 weights for new alphas)
 E: Also re-tune a_best (double_best alpha, currently 0.200)
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
emb_ica = ep["emb_win_ica_norm"]
emb_pca = ep["emb_win_pca_norm"]
labels_win  = ep["labels_win"]
win_file_id = ep["win_file_id"]

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch168] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 168}
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

# Fixed: rank components and wknn
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_ref   = 0.56*rank_c_ref + 0.44*rank_i_ref
rank_norm  = rank_ref / n_files

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

print("Pre-computing window KNN...", flush=True)
p_ica8 = wknn(SIM_ICA, k=8)
p_pca5 = wknn(SIM_PCA, k=5)
wknn_comb = 0.5*p_ica8 + 0.5*p_pca5

# Fixed c3 (a_best=0.200 is stable)
c3_ref = apply_3way(double_best, alpha=0.200)

# Current best params from batch167
# wb=0.74, wi=0.16, ws=0.06, ww=0.04, rm=0.28 (section B best); a_ica=0.31, a_std=0.33
# But let me re-verify the exact best formula from batch167
# The best method is wk4sca_ia31_sa33 which used the wb/wi/ws/ww from the 4-component context
# Looking at batch166 C section: the best was wb=0.74, wi=0.15, ws=0.07, ww=0.04 (different from batch167 B)
# batch167 A gave wb=0.74 wi=0.16 ws=0.06 ww=0.04 rm=0.28 as best = 0.995949
# batch167 B found ww fine adjustment giving 0.995952
# batch167 D used wb=0.74, wi=??, ws=??, ww=?? -- let me use the batch167 A best grid

# Use batch167 A best: wb=0.74, wi=0.16, ws=0.06, ww=0.04, rm=0.28

i3_best = apply_3way(ica_ens_alt, alpha=0.31)
s3_best = apply_3way(std_ens_ref,  alpha=0.33)
chk4_best = 0.74*c3_ref + 0.16*i3_best + 0.06*s3_best + 0.04*wknn_comb
v = 0.72*chk4_best + 0.28*rank_norm
print(f"Verify (ia31_sa33, wb74_wi16_ws6_ww4_rm28): {macro_auc(v):.6f}\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Fine a_ica (0.27-0.40 step 0.005) at a_std=0.33
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Fine a_ica search (a_std=0.33) ===", flush=True)
best_a = best_loo
s3_33 = apply_3way(std_ens_ref, alpha=0.33)

best_a_ica = 0.31
for a_int in range(54, 82):  # 0.270-0.405 step 0.005
    a = a_int / 200.0
    i3_new = apply_3way(ica_ens_alt, alpha=a)
    for wb, wi, ws, ww in [(0.74, 0.16, 0.06, 0.04), (0.74, 0.15, 0.07, 0.04)]:
        chk4 = wb*c3_ref + wi*i3_new + ws*s3_33 + ww*wknn_comb
        for rm in [0.27, 0.28, 0.29]:
            final = (1-rm)*chk4 + rm*rank_norm
            ar = macro_auc(final)
            mname = f"wk4a_ia{int(a*1000)}_sa330_wb{int(wb*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_a:
                best_a = ar
                best_a_ica = a
            if ar > best_loo - 0.00004:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  a_ica={a:.3f} wb={wb:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section A: {best_a:.6f}, best a_ica={best_a_ica:.3f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Fine a_std (0.28-0.45 step 0.005) at a_ica=0.31
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: Fine a_std search (a_ica=0.31) ===", flush=True)
best_b = best_loo
i3_31 = apply_3way(ica_ens_alt, alpha=0.31)

best_b_std = 0.33
for a_int in range(56, 92):  # 0.280-0.455 step 0.005
    a = a_int / 200.0
    s3_new = apply_3way(std_ens_ref, alpha=a)
    for wb, wi, ws, ww in [(0.74, 0.16, 0.06, 0.04), (0.74, 0.15, 0.07, 0.04)]:
        chk4 = wb*c3_ref + wi*i3_31 + ws*s3_new + ww*wknn_comb
        for rm in [0.27, 0.28, 0.29]:
            final = (1-rm)*chk4 + rm*rank_norm
            ar = macro_auc(final)
            mname = f"wk4b_ia310_sa{int(a*1000)}_wb{int(wb*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_b:
                best_b = ar
                best_b_std = a
            if ar > best_loo - 0.00004:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  a_std={a:.3f} wb={wb:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section B: {best_b:.6f}, best a_std={best_b_std:.3f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Joint a_ica × a_std grid (step 0.01, wider)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: Joint a_ica × a_std grid ===", flush=True)
best_c = best_loo

for a_ica in [round(x/100, 2) for x in range(26, 40)]:
    i3_new = apply_3way(ica_ens_alt, alpha=a_ica)
    for a_std in [round(x/100, 2) for x in range(28, 42)]:
        s3_new = apply_3way(std_ens_ref, alpha=a_std)
        chk4 = 0.74*c3_ref + 0.16*i3_new + 0.06*s3_new + 0.04*wknn_comb
        final = 0.72*chk4 + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wk4c_ia{int(a_ica*100)}_sa{int(a_std*100)}"
        delta = save_result(mname, ar)
        if ar > best_c: best_c = ar
        if ar > best_loo - 0.00004:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_ica={a_ica:.2f} a_std={a_std:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Re-tune a_best (double_best alpha, currently fixed at 0.200)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: a_best re-tune with 4-comp formula ===", flush=True)
best_d = best_loo
i3_31 = apply_3way(ica_ens_alt, alpha=0.31)
s3_33 = apply_3way(std_ens_ref, alpha=0.33)

for a_best in [round(x/100, 2) for x in range(15, 30)]:
    c3_new = apply_3way(double_best, alpha=a_best)
    chk4 = 0.74*c3_new + 0.16*i3_31 + 0.06*s3_33 + 0.04*wknn_comb
    final = 0.72*chk4 + 0.28*rank_norm
    ar = macro_auc(final)
    mname = f"wk4d_ab{int(a_best*100)}_ia31_sa33"
    delta = save_result(mname, ar)
    if ar > best_d: best_d = ar
    if ar > best_loo - 0.00004:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  a_best={a_best:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section D: {best_d:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 168]
print(f"Batch168 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
