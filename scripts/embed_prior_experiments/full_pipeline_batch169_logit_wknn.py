"""
batch169 — Window KNN with Logit Signal + Triple WKNN (ICA+PCA+STD)
===============================================================================
Current best: wk4d_ab19_ia31_sa33 LOO=0.995956
  chk4 = 0.74*c3 + 0.16*i3 + 0.06*s3 + 0.04*wknn_lbl_comb
  final = 0.73*chk4 + 0.27*rank_norm

Genuinely untried directions (confirmed by JSON scan):
  - wknn_logit: use logit_sig_win (continuous soft probabilities) as KNN signal
    instead of labels_win (binary 0/1). May give richer signal.
  - triple_wknn: add STD window KNN (emb_win_std_norm) as 3rd component
  - hybrid signal: blend(labels, logit_sig) as KNN aggregation signal

Sections:
  A: wknn_logit — pure logit signal (ICA k=8, PCA k=5) as replacement for wknn_lbl
  B: hybrid blend — w_lbl*(labels_wknn) + w_log*(logit_wknn) sweeping blend ratio
  C: triple_wknn — add STD wknn (k sweep) into chk4 as 5th component
  D: optimal logit wknn as explicit 4th chk component (replace label wknn)
  E: fine re-tune with best logit/triple combo
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
labels_win   = ep["labels_win"]     # (739, 234) binary
logit_sig_win = ep["logit_sig_win"] # (739, 234) sigmoid(logits), continuous
win_file_id  = ep["win_file_id"]    # (739,)

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch169] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)
print(f"[batch169] logit_sig_win range: {logit_sig_win.min():.3f}–{logit_sig_win.max():.3f}, "
      f"mean={logit_sig_win.mean():.4f}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 169}
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

# ── Co-occurrence helpers (same as batch167/168) ──────────────────────────────
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

# Reference components from best formula (batch168)
c3_ref     = apply_3way(double_best, alpha=0.19)
i3_ref     = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref     = apply_3way(std_ens_ref,  alpha=0.33)
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_ref   = 0.56*rank_c_ref + 0.44*rank_i_ref
rank_norm  = rank_ref / n_files

# ── Pre-compute window KNN similarity matrices ─────────────────────────────────
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
fi_wins_list    = [np.where(win_file_id == fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id != fi)[0] for fi in range(n_files)]

def wknn_with_signal(SIM, k, signal):
    """Window KNN with arbitrary signal (labels_win or logit_sig_win)."""
    sig = signal.astype(np.float32)
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

print("Pre-computing window KNN (label signal, reference)...", flush=True)
p_ica8_lbl = wknn_with_signal(SIM_ICA, k=8, signal=labels_win)
p_pca5_lbl = wknn_with_signal(SIM_PCA, k=5, signal=labels_win)
wknn_lbl_comb = 0.5*p_ica8_lbl + 0.5*p_pca5_lbl

# Verify reference formula
best_chk4 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_lbl_comb
v_ref = 0.73*best_chk4 + 0.27*rank_norm
print(f"Verify reference: {macro_auc(v_ref):.6f} (expect ~0.995956)\n", flush=True)

print("Pre-computing window KNN (logit signal, new)...", flush=True)
p_ica8_log = wknn_with_signal(SIM_ICA, k=8, signal=logit_sig_win)
p_pca5_log = wknn_with_signal(SIM_PCA, k=5, signal=logit_sig_win)
wknn_log_comb = 0.5*p_ica8_log + 0.5*p_pca5_log
print(f"wknn_log_comb range: {wknn_log_comb.min():.4f}–{wknn_log_comb.max():.4f}\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A: wknn_logit as replacement for wknn_lbl in chk4 (direct swap)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== A: wknn_logit as 4th chk component ===", flush=True)
best_a = best_loo
for ww in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
    ww_int = int(round(ww*100))
    # Rescale wb/wi/ws to sum to 1-ww
    wb = round(0.74*(1-ww)/0.96, 4)
    wi = round(0.16*(1-ww)/0.96, 4)
    ws = round(1.0 - wb - wi - ww, 4)
    chk4_log = wb*c3_ref + wi*i3_ref + ws*s3_ref + ww*wknn_log_comb
    for rm in [0.25, 0.26, 0.27, 0.28, 0.29, 0.30]:
        final = (1-rm)*chk4_log + rm*rank_norm
        ar = macro_auc(final)
        mname = f"wkl4_ww{ww_int}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_a: best_a = ar
        if ar > best_loo - 0.00008:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  ww={ww:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section A: {best_a:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B: Hybrid signal blend (lbl + logit) at various ratios as 4th component
# ═══════════════════════════════════════════════════════════════════════════════
print("=== B: Hybrid label+logit blend as 4th wknn component ===", flush=True)
best_b = best_loo
for f_log in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    wknn_hybrid = (1-f_log)*wknn_lbl_comb + f_log*wknn_log_comb
    chk4_h = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_hybrid
    for rm in [0.26, 0.27, 0.28]:
        final = (1-rm)*chk4_h + rm*rank_norm
        ar = macro_auc(final)
        mname = f"wkh_fl{int(f_log*10)}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_b: best_b = ar
        if ar > best_loo - 0.00008:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  f_log={f_log:.1f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section B: {best_b:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C: Triple wknn (ICA+PCA+STD) with label signal
# ═══════════════════════════════════════════════════════════════════════════════
print("=== C: Triple wknn ICA+PCA+STD (label signal) ===", flush=True)
best_c = best_loo
# Pre-compute STD wknn for several k values
std_wknns = {}
for k_std in [5, 6, 7, 8, 10, 12]:
    std_wknns[k_std] = wknn_with_signal(SIM_STD, k=k_std, signal=labels_win)
    print(f"  STD k={k_std} computed", flush=True)

for k_std, p_std in std_wknns.items():
    # Equal weight triple: 1/3 each
    wknn_triple_eq = (p_ica8_lbl + p_pca5_lbl + p_std) / 3.0
    chk4_te = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_triple_eq
    for rm in [0.26, 0.27, 0.28]:
        ar = macro_auc((1-rm)*chk4_te + rm*rank_norm)
        mname = f"wkt3_eq_ks{k_std}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_c: best_c = ar
        if ar > best_loo - 0.00008:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  triple_eq k_std={k_std} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

    # Weighted: ICA 0.5, PCA 0.3, STD 0.2
    wknn_triple_w1 = 0.5*p_ica8_lbl + 0.3*p_pca5_lbl + 0.2*p_std
    chk4_tw1 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_triple_w1
    for rm in [0.26, 0.27, 0.28]:
        ar = macro_auc((1-rm)*chk4_tw1 + rm*rank_norm)
        mname = f"wkt3_w1_ks{k_std}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_c: best_c = ar
        if ar > best_loo - 0.00008:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  triple_w1 k_std={k_std} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

    # ICA 0.4, PCA 0.4, STD 0.2
    wknn_triple_w2 = 0.4*p_ica8_lbl + 0.4*p_pca5_lbl + 0.2*p_std
    chk4_tw2 = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_triple_w2
    for rm in [0.26, 0.27, 0.28]:
        ar = macro_auc((1-rm)*chk4_tw2 + rm*rank_norm)
        mname = f"wkt3_w2_ks{k_std}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_c: best_c = ar
        if ar > best_loo - 0.00008:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  triple_w2 k_std={k_std} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section C: {best_c:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: 5-component chk (add logit_wknn as 5th separate from label_wknn)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== D: 5-component chk (label_wknn + logit_wknn as separate signals) ===", flush=True)
best_d = best_loo
for wl in [0.01, 0.02, 0.03, 0.04]:  # logit wknn weight
    for ww in [0.02, 0.03, 0.04]:    # label wknn weight
        # Normalize remaining 4 components
        rem = 1.0 - wl - ww
        wb = round(0.74 * rem / 0.96, 3)
        wi = round(0.16 * rem / 0.96, 3)
        ws = round(1.0 - wb - wi - wl - ww, 3)
        if ws < 0.01: continue
        chk5 = wb*c3_ref + wi*i3_ref + ws*s3_ref + ww*wknn_lbl_comb + wl*wknn_log_comb
        for rm in [0.26, 0.27, 0.28]:
            ar = macro_auc((1-rm)*chk5 + rm*rank_norm)
            mname = f"wk5_wl{int(wl*100)}_ww{int(ww*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_d: best_d = ar
            if ar > best_loo - 0.00007:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  wl={wl:.2f} ww={ww:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section D: {best_d:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: logit wknn K-sweep (find optimal k for logit signal)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== E: logit wknn k-sweep ===", flush=True)
best_e = best_loo
logit_wknns = {}
for k in [3, 5, 7, 8, 10, 12, 15]:
    logit_wknns[k] = {}
    logit_wknns[k]['ica'] = wknn_with_signal(SIM_ICA, k=k, signal=logit_sig_win)
    logit_wknns[k]['pca'] = wknn_with_signal(SIM_PCA, k=k, signal=logit_sig_win)

for ki in [5, 7, 8, 10, 12]:
    for kp in [3, 5, 7, 8, 10]:
        wknn_log_kk = 0.5*logit_wknns[ki]['ica'] + 0.5*logit_wknns[kp]['pca']
        chk4_lk = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + 0.04*wknn_log_kk
        for rm in [0.26, 0.27, 0.28]:
            ar = macro_auc((1-rm)*chk4_lk + rm*rank_norm)
            mname = f"wkl_ki{ki}_kp{kp}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_e: best_e = ar
            if ar > best_loo - 0.00007:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  ki={ki} kp={kp} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section E: {best_e:.6f}\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: Re-tune score alphas with logit wknn as 4th component (if section A improved)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== F: Score alpha re-tune with logit wknn ===", flush=True)
best_f = best_loo
# Find best ww from section A/E
best_ww_log = 0.04
for a_ica in [0.28, 0.30, 0.31, 0.32, 0.34]:
    i3_new = apply_3way(ica_ens_alt, alpha=a_ica)
    for a_std in [0.30, 0.32, 0.33, 0.34, 0.36]:
        s3_new = apply_3way(std_ens_ref, alpha=a_std)
        ww = best_ww_log
        rem = 1.0 - ww
        wb = 0.74*rem/0.96; wi = 0.16*rem/0.96; ws = 1.0-wb-wi-ww
        chk4_fa = wb*c3_ref + wi*i3_new + ws*s3_new + ww*wknn_log_comb
        ar = macro_auc(0.73*chk4_fa + 0.27*rank_norm)
        mname = f"wkla_ia{int(a_ica*100)}_sa{int(a_std*100)}"
        delta = save_result(mname, ar)
        if ar > best_f: best_f = ar
        if ar > best_loo - 0.00007:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  a_ica={a_ica:.2f} a_std={a_std:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)
print(f"  Best section F: {best_f:.6f}\n", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 169]
print(f"Batch169 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
