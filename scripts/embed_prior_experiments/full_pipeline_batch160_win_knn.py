"""
batch160 — Window-level KNN + cross-signal rank
===============================================================================
Current best: sca4_ia26_sa28 LOO=0.995800

Rationale:
  Current approach operates at file level (66 files). We have 739 windows with
  per-window embeddings (ICA/PCA/STD, 80-100 dims). Window-level KNN gives
  finer-grained similarity signals that file-level averaging might miss.

  Cross-signal rank: instead of ranking each signal separately, rank their
  product/geometric mean/max to get a combined signal.

Sections:
 A: Window-level ICA KNN — LOO prediction using window embeddings
 B: Window-level PCA KNN
 C: Window-level STD KNN
 D: Blend window-KNN with current chk reference
 E: Cross-signal rank: rank(c3*i3), rank(max(c3,i3)), rank(sqrt(c3*i3))
 F: Per-species rank mixing (rm varies by species rarity)
 G: Softmax-weighted rank (smooth alternative to argsort)
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

# Window-level data
emb_ica = ep["emb_win_ica_norm"]  # (739, 100)
emb_pca = ep["emb_win_pca_norm"]  # (739, 80)
emb_std = ep["emb_win_std_norm"]  # (739, 80)
labels_win  = ep["labels_win"]    # (739, 234)
win_file_id = ep["win_file_id"]   # (739,) int32
logit_sig   = ep["logit_sig_win"] # (739, 234)

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch160] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)
print(f"  Windows: {len(emb_ica)}, Files: {n_files}, Species: {n_species}", flush=True)

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
    entry = {"method": mname, "loo_auc": float(score), "config": config_dict or {}, "batch": 160}
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

# Standard COOC
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

# Reference
c3_ref     = apply_3way(double_best, alpha=0.200)
i3_ref     = apply_3way(ica_ens_alt, alpha=0.260)
s3_ref     = apply_3way(std_ens_ref,  alpha=0.280)
chk_ref    = 0.75*c3_ref + 0.15*i3_ref + 0.10*s3_ref
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_ref   = 0.56*rank_c_ref + 0.44*rank_i_ref
rank_norm  = rank_ref / n_files
v = 0.72*chk_ref + 0.28*rank_norm
print(f"Verify: {macro_auc(v):.6f} (expect ~0.995800)\n", flush=True)

t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# A-C: Window-level KNN function
# ═══════════════════════════════════════════════════════════════════════════════
def window_knn_loo(emb_win, k=3, use_logit=False, agg='mean'):
    """
    LOO window-level KNN prediction.
    For each file fi:
      - mask out all windows of fi
      - for each window wk of fi: find K nearest non-fi windows by cosine sim
      - get predictions for wk from neighbor windows' logits/labels
      - aggregate across all windows of fi to get file-level prediction

    emb_win: (739, D) normalized embeddings
    k: number of nearest neighbor windows
    use_logit: use logit_sig instead of labels_win
    agg: 'mean' or 'max' aggregation across windows of fi
    """
    SIM = emb_win @ emb_win.T  # (739, 739)
    signal = logit_sig if use_logit else labels_win.astype(np.float32)

    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for fi in range(n_files):
        fi_mask = (win_file_id == fi)          # windows of fi
        other_mask = ~fi_mask                  # non-fi windows
        fi_wins = np.where(fi_mask)[0]         # indices of fi windows
        other_wins = np.where(other_mask)[0]   # indices of other windows

        if len(fi_wins) == 0:
            continue

        win_preds = np.zeros((len(fi_wins), n_species), dtype=np.float32)
        for wi, wk in enumerate(fi_wins):
            sims_wk = SIM[wk, other_wins]  # similarity to all non-fi windows
            k_eff = min(k, len(sims_wk))
            top_local = np.argpartition(-sims_wk, k_eff - 1)[:k_eff]
            top_wins = other_wins[top_local]

            w = np.clip(sims_wk[top_local], 0, None)
            if w.sum() < EPS:
                w = np.ones(k_eff) / k_eff
            else:
                w = w / w.sum()

            win_preds[wi] = (w[:, None] * signal[top_wins]).sum(0)

        if agg == 'mean':
            preds[fi] = win_preds.mean(0)
        elif agg == 'max':
            preds[fi] = win_preds.max(0)
        elif agg == 'top2mean':
            if len(win_preds) > 1:
                preds[fi] = np.sort(win_preds, axis=0)[-2:].mean(0)
            else:
                preds[fi] = win_preds[0]

    return preds

print("=== A: Window-level ICA KNN ===", flush=True)
best_a = best_loo
win_knn_cache = {}
for k in [3, 5, 7, 10]:
    p = window_knn_loo(emb_ica, k=k, use_logit=False, agg='mean')
    ar = macro_auc(p)
    nm = f"wknn_ica_k{k}_lbl_mean"
    save_result(nm, ar)
    win_knn_cache[('ica', k, 'lbl', 'mean')] = p
    p_l = window_knn_loo(emb_ica, k=k, use_logit=True, agg='mean')
    ar_l = macro_auc(p_l)
    nm_l = f"wknn_ica_k{k}_lgit_mean"
    save_result(nm_l, ar_l)
    win_knn_cache[('ica', k, 'lgit', 'mean')] = p_l
    if max(ar, ar_l) > best_a: best_a = max(ar, ar_l)
    print(f"  ICA k={k}: lbl={ar:.6f} lgit={ar_l:.6f}", flush=True)
print(f"  Best section A: {best_a:.6f}", flush=True)

print(f"\n=== B: Window-level PCA KNN ===", flush=True)
best_b = best_loo
for k in [3, 5, 7, 10]:
    p = window_knn_loo(emb_pca, k=k, use_logit=False, agg='mean')
    ar = macro_auc(p)
    nm = f"wknn_pca_k{k}_lbl_mean"
    save_result(nm, ar)
    win_knn_cache[('pca', k, 'lbl', 'mean')] = p
    p_l = window_knn_loo(emb_pca, k=k, use_logit=True, agg='mean')
    ar_l = macro_auc(p_l)
    nm_l = f"wknn_pca_k{k}_lgit_mean"
    save_result(nm_l, ar_l)
    win_knn_cache[('pca', k, 'lgit', 'mean')] = p_l
    if max(ar, ar_l) > best_b: best_b = max(ar, ar_l)
    print(f"  PCA k={k}: lbl={ar:.6f} lgit={ar_l:.6f}", flush=True)
print(f"  Best section B: {best_b:.6f}", flush=True)

print(f"\n=== C: Window-level STD KNN ===", flush=True)
best_c = best_loo
for k in [3, 5, 7, 10]:
    p = window_knn_loo(emb_std, k=k, use_logit=False, agg='mean')
    ar = macro_auc(p)
    nm = f"wknn_std_k{k}_lbl_mean"
    save_result(nm, ar)
    win_knn_cache[('std', k, 'lbl', 'mean')] = p
    p_l = window_knn_loo(emb_std, k=k, use_logit=True, agg='mean')
    ar_l = macro_auc(p_l)
    nm_l = f"wknn_std_k{k}_lgit_mean"
    save_result(nm_l, ar_l)
    win_knn_cache[('std', k, 'lgit', 'mean')] = p_l
    if max(ar, ar_l) > best_c: best_c = max(ar, ar_l)
    print(f"  STD k={k}: lbl={ar:.6f} lgit={ar_l:.6f}", flush=True)
print(f"  Best section C: {best_c:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D: Blend best window-KNN into chk
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== D: Blend window-KNN into reference ===", flush=True)
best_d = best_loo

# Find best window KNN from sections A-C
all_wknn = []
for key, pred in win_knn_cache.items():
    ar = macro_auc(pred)
    all_wknn.append((ar, key, pred))
all_wknn.sort(key=lambda x: -x[0])
print(f"  Top-3 window KNN:", flush=True)
for ar, key, _ in all_wknn[:3]:
    print(f"    {key}: {ar:.6f}", flush=True)

# Blend top-3 with reference
for ar_wknn, key, p_wknn in all_wknn[:3]:
    emb_name = key[0]
    k = key[1]
    sig_type = key[2]
    for w_wk in [0.02, 0.05, 0.08, 0.10, 0.15]:
        # Blend into chk
        chk_new = (1-w_wk)*chk_ref + w_wk*p_wknn
        final = 0.72*chk_new + 0.28*rank_norm
        ar = macro_auc(final)
        mname = f"wknn_{emb_name}_k{k}_{sig_type}_wchk{int(w_wk*100)}"
        delta = save_result(mname, ar)
        if ar > best_d: best_d = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  {emb_name} k={k} {sig_type} w={w_wk:.2f} chk: {ar:.6f} {delta:+.6f}{flag}", flush=True)
        # Blend into rank
        rk_wknn = make_rank(p_wknn)
        rk_mix = (1-w_wk)*rank_ref + w_wk*rk_wknn
        final2 = 0.72*chk_ref + 0.28*(rk_mix/n_files)
        ar2 = macro_auc(final2)
        mname2 = f"wknn_{emb_name}_k{k}_{sig_type}_wrk{int(w_wk*100)}"
        delta2 = save_result(mname2, ar2)
        if ar2 > best_d: best_d = ar2
        if ar2 > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar2 > best_loo else ""
            print(f"  {emb_name} k={k} {sig_type} w={w_wk:.2f} rank: {ar2:.6f} {delta2:+.6f}{flag}", flush=True)

print(f"  Best section D: {best_d:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E: Cross-signal rank
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== E: Cross-signal rank ===", flush=True)
best_e = best_loo

# Product rank: rank(c3 * i3)
c3_alpha_vals = [0.200, 0.23, 0.25, 0.30]
i3_alpha_vals = [0.26, 0.30, 0.35, 0.40]

for ac in [0.20, 0.23]:
    for ai in [0.26, 0.30, 0.40]:
        c3 = apply_3way(double_best, alpha=ac)
        i3 = apply_3way(ica_ens_alt, alpha=ai)

        # Product rank
        rk_prod = make_rank(c3 * i3)
        for rm in [0.26, 0.28, 0.30]:
            final = (1-rm)*chk_ref + rm*(rk_prod/n_files)
            ar = macro_auc(final)
            mname = f"xrk_prod_ac{int(ac*100)}_ai{int(ai*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar, {"a_c": ac, "a_i": ai, "rm": rm, "type": "product"})
            if ar > best_e: best_e = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  prod ac={ac:.2f} ai={ai:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

        # Geometric mean rank
        rk_geom = make_rank(np.sqrt(np.clip(c3, 0, None) * np.clip(i3, 0, None)))
        for rm in [0.26, 0.28, 0.30]:
            final = (1-rm)*chk_ref + rm*(rk_geom/n_files)
            ar = macro_auc(final)
            mname = f"xrk_geom_ac{int(ac*100)}_ai{int(ai*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_e: best_e = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  geom ac={ac:.2f} ai={ai:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

        # Max rank (element-wise max of two signals)
        rk_max = make_rank(np.maximum(c3, i3))
        for rm in [0.26, 0.28, 0.30]:
            final = (1-rm)*chk_ref + rm*(rk_max/n_files)
            ar = macro_auc(final)
            mname = f"xrk_max_ac{int(ac*100)}_ai{int(ai*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_e: best_e = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  max ac={ac:.2f} ai={ai:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

        # Min rank (element-wise min = intersection signal)
        rk_min = make_rank(np.minimum(c3, i3))
        for rm in [0.26, 0.28, 0.30]:
            final = (1-rm)*chk_ref + rm*(rk_min/n_files)
            ar = macro_auc(final)
            mname = f"xrk_min_ac{int(ac*100)}_ai{int(ai*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_e: best_e = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  min ac={ac:.2f} ai={ai:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

        # Rank-of-rank: re-rank the reference rank blend using c3
        rk_of_rk = make_rank(rank_ref / n_files * c3)
        for rm in [0.26, 0.28, 0.30]:
            final = (1-rm)*chk_ref + rm*(rk_of_rk/n_files)
            ar = macro_auc(final)
            mname = f"xrk_rkc_ac{int(ac*100)}_rm{int(rm*100)}"
            delta = save_result(mname, ar)
            if ar > best_e: best_e = ar
            if ar > best_loo - 0.00005:
                flag = " ← NEW BEST!" if ar > best_loo else ""
                print(f"  rkc ac={ac:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section E: {best_e:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# F: Per-species rank mixing (rare species get more rank signal)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== F: Per-species rank mixing ===", flush=True)
best_f = best_loo

# Rare species (low count_i) might benefit from more rank signal
# Compute per-species rank mixing weight: rm_s = base_rm + idf_boost * (idf / max_idf)
idf_frac = raw_idf / (raw_idf.max() + EPS)  # (n_sp,) in [0,1]

for base_rm in [0.24, 0.26, 0.28]:
    for idf_boost in [0.02, 0.04, 0.06, 0.08]:
        rm_s = np.clip(base_rm + idf_boost * idf_frac, 0, 0.5)  # (n_sp,)
        # Apply per-species mixing
        final = (1-rm_s[None, :])*chk_ref + rm_s[None, :]*(rank_norm)
        ar = macro_auc(final)
        mname = f"ps_rm_base{int(base_rm*100)}_boost{int(idf_boost*100)}"
        delta = save_result(mname, ar, {"base_rm": base_rm, "idf_boost": idf_boost})
        if ar > best_f: best_f = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  ps_rm base={base_rm:.2f} boost={idf_boost:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section F: {best_f:.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# G: Softmax rank (smooth differentiable rank alternative)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== G: Softmax rank ===", flush=True)
best_g = best_loo

def softmax_rank(x, temperature=1.0):
    """
    Soft rank: expected rank position under softmax distribution.
    For each species s, the soft rank of file i is:
      soft_rank[i,s] = sum_j softmax(x[:,s]/T)[j] * rank(j)
    This is equivalent to: positions weighted by softmax probabilities.
    """
    out = np.zeros_like(x)
    n = x.shape[0]
    positions = np.arange(n, dtype=float)  # [0, 1, ..., n-1]
    for s in range(x.shape[1]):
        col = x[:, s]
        # Softmax
        col_t = col / temperature
        col_t = col_t - col_t.max()  # numerical stability
        sm = np.exp(col_t)
        sm = sm / (sm.sum() + EPS)
        # Soft rank: expected rank position
        hard_rank = np.argsort(np.argsort(col))  # [0..n-1]
        out[:, s] = np.sum(sm[:, None] * np.abs(positions[None, :] - hard_rank[:, None]), axis=1)
        # Actually simpler: just use softmax-weighted average of positions
        # sorted_pos = positions at sorted order; soft_rank[i] = sum_j sm[j] * rank[j]
        # where rank[j] = argsort rank of j
    return out

# Actually let me use a simpler smooth rank:
# Instead of argsort(argsort(x)), use sum of sigmoid comparisons
# smooth_rank[i, s] = sum_{j!=i} sigma((x[i,s] - x[j,s]) / T)
def smooth_rank(x, T=0.1):
    """
    Smooth rank via sigmoid pairwise comparisons.
    smooth_rank[i,s] = sum_j sigmoid((x[i,s]-x[j,s])/T)
    """
    out = np.zeros_like(x)
    for s in range(x.shape[1]):
        col = x[:, s]  # (n_files,)
        # (n_files, n_files) pairwise differences
        diff = col[:, None] - col[None, :]  # diff[i,j] = col[i] - col[j]
        diff_t = np.clip(diff / T, -88, 88)
        sig = 1.0 / (1.0 + np.exp(-diff_t))  # sigmoid
        out[:, s] = sig.sum(1)  # sum over j gives smooth rank
    return out

for T in [0.05, 0.10, 0.20, 0.30]:
    sr = smooth_rank(chk_ref, T=T) / n_files
    for rm in [0.26, 0.28, 0.30]:
        final = (1-rm)*chk_ref + rm*sr
        ar = macro_auc(final)
        mname = f"smrank_T{int(T*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_g: best_g = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  smooth_rank T={T:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

# Also try smooth rank of rank_ref
for T in [0.05, 0.10, 0.20]:
    # Use rank signal itself as input to smooth rank
    sr_of_rk = smooth_rank(rank_ref, T=T*n_files) / n_files
    for rm in [0.26, 0.28, 0.30]:
        final = (1-rm)*chk_ref + rm*sr_of_rk
        ar = macro_auc(final)
        mname = f"smrank_rk_T{int(T*100)}_rm{int(rm*100)}"
        delta = save_result(mname, ar)
        if ar > best_g: best_g = ar
        if ar > best_loo - 0.00005:
            flag = " ← NEW BEST!" if ar > best_loo else ""
            print(f"  smooth_rank_rk T={T:.2f} rm={rm:.2f}: {ar:.6f} {delta:+.6f}{flag}", flush=True)

print(f"  Best section G: {best_g:.6f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == 160]
print(f"Batch160 complete in {time.time()-t0:.1f}s: {len(exps_this)} experiments", flush=True)
print(f"Final best LOO: {best_loo:.6f}  method: {res['best']['method']}", flush=True)
top5 = sorted(exps_this, key=lambda x: -x["loo_auc"])[:5]
print("Top-5 this batch:")
for e in top5:
    print(f"  {e['method']}: {e['loo_auc']:.6f}")
