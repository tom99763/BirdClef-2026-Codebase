"""
batch167 — Fine-tune 4-component blend (c3+i3+s3+wknn) + rm
===============================================================================
Current best: wk4j_wb74_wi15_ws7_ww40_rm28 LOO=0.995936
  chk4 = 0.74*c3 + 0.15*i3 + 0.07*s3 + 0.04*wknn
  final = 0.72*chk4 + 0.28*rank_norm

batch166 finding:
  4-component blend with wknn as explicit component (ww=0.04) in chk gave +0.000009
  But this means wknn is ~4% weight in chk, which corresponds to ~2.9% in final

Directions:
 A: Fine grid around best (wb=0.74, wi=0.15, ws=0.07, ww=0.04, rm=0.28)
 B: ww fine search (0.02-0.08 step 0.002)
 C: Joint (wb, wi, ws, ww) fine step 0.01
 D: rm fine step 0.005
 E: Score alpha re-tune (a_ica, a_std) with 4-comp formula
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
print(f"[batch167] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 167}
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

# Verify
best_chk4 = 0.74*c3_ref + 0.15*i3_ref + 0.07*s3_ref + 0.04*wknn_comb
v = 0.72*best_chk4 + 0.28*rank_norm
print(f"Verify: {macro_auc(v):.6f} (expect ~0.995936)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: Fine grid around best (step 0.01)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: Fine grid (step 0.01) ===", flush=True)
best_a = best_loo
for wb_int in range(70, 80):
    wb = wb_int / 100.0
    for wi_int in range(12, 19):
        wi = wi_int / 100.0
        for ws_int in range(5, 12):
            ws = ws_int / 100.0
            ww = round(1.0 - wb - wi - ws, 2)
            if ww < 0.01 or ww > 0.08: continue
            chk4 = wb*c3_ref + wi*i3_ref + ws*s3_ref + ww*wknn_comb
            for rm_int in range(26, 32):
                rm = rm_int / 100.0
                final = (1-rm)*chk4 + rm*rank_norm
                ar = macro_auc(final)
                mname = f"wk4f_wb{wb_int}_wi{wi_int}_ws{ws_int}_ww{int(ww*100)}_rm{rm_int}"
                delta = save_result(mname, ar)
                if ar > best_a: best_a = ar
                if ar > best_loo - 0.00005:
                    flag = " ← NEW BEST!" if ar > best_loo else ""
                    print(f"  wb={wb:.2f} wi={wi:.2f} ws={ws:.2f} ww={ww:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section A: {best_a:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: ww fine search (step 0.002) at best wb/wi/ws
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== B: ww fine search ===", flush=True)
best_b = best_loo
# Best: wb=0.74, wi=0.15, ws=0.07 (from batch166)
for ww_int in range(10, 80, 2):  # 0.010-0.078 step 0.002
    ww = ww_int / 1000.0
    wb_adj = 0.74 * (1 - ww) / (0.74 + 0.15 + 0.07)  # rescale to sum=1
    wi_adj = 0.15 * (1 - ww) / (0.74 + 0.15 + 0.07)
    ws_adj = 0.07 * (1 - ww) / (0.74 + 0.15 + 0.07)
    chk4 = wb_adj*c3_ref + wi_adj*i3_ref + ws_adj*s3_ref + ww*wknn_comb
    for rm in [0.27, 0.28, 0.29]:
        final = (1-rm)*chk4 + rm*rank_norm
        ar = macro_auc(final)
        mname = f"wk4ww_ww{ww_int}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_b: best_b = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  ww={ww:.3f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section B: {best_b:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: rm ultra-fine search (step 0.002)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== C: rm ultra-fine search ===", flush=True)
best_c = best_loo
chk4_best = 0.74*c3_ref + 0.15*i3_ref + 0.07*s3_ref + 0.04*wknn_comb
for rm_int in range(240, 330, 2):
    rm = rm_int / 1000.0
    final = (1-rm)*chk4_best + rm*rank_norm
    ar = macro_auc(final)
    mname = f"wk4rm_rm{rm_int}"
    delta = save_result(mname, ar)
    if ar > best_c: best_c = ar
    if ar > best_loo - 0.00005:
        flag = " ← NEW BEST!" if ar > best_loo else ""
        print(f"  rm={rm:.3f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Score alpha re-tune with 4-comp formula
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Score alpha re-tune with 4-comp ===", flush=True)
best_d = best_loo
for a_ica in [round(x/100, 2) for x in range(22, 32)]:
    i3_new = apply_3way(ica_ens_alt, alpha=a_ica)
    for a_std in [round(x/100, 2) for x in range(24, 34)]:
        s3_new = apply_3way(std_ens_ref, alpha=a_std)
        chk4_new = 0.74*c3_ref + 0.15*i3_new + 0.07*s3_new + 0.04*wknn_comb
        final = 0.72*chk4_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wk4sca_ia{int(a_ica*100)}_sa{int(a_std*100)}"
        delta = save_result(mname, ar)
        if ar > best_d: best_d = ar
        if ar > best_loo - 0.00004:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_ica={a_ica:.2f} a_std={a_std:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section D: {best_d:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 167]
print(f"Batch167 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
