"""
batch183 — Novel logit feature transformations
===============================================
NEW BEST from batch182: ica128l3_al285_k5_wa68_ww5_rm28 LOO=0.996021
Pattern: n_logit=3, alpha≈0.285, wa≈0.68, k=5, ww=0.05, rm=0.28

batch183 tries fundamentally DIFFERENT logit feature transformations:
  A: Sigmoid(logit) before PCA — captures "soft detection" profile
  B: Logit deviation (logit - per-window-mean) — captures relative species salience
  C: Rank-normalized logit — ordinal transformation preserving order
  D: File-level aggregated logit (mean/max per file) — file-level detection profile
  E: Species-IDF weighted logit — rare species logits weighted more heavily
  F: Temporal variance logit — std of logit per file as complement to mean
  G: Signed log logit — log(|logit|)*sign(logit) for heavy-tail compression
"""
import numpy as np
import json, pickle, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import normalize, StandardScaler, QuantileTransformer
import warnings
warnings.filterwarnings('ignore')

EPS   = 1e-8
BATCH = 183
ROOT  = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

file_labels = ep["file_labels"]
double_best = ep["chain_double_best"]
ica_ens_alt = ep["chain_ica_ens_alt"]
std_ens_ref = ep["chain_std_ens_ref"]
emb_ica     = ep["emb_win_ica_norm"]
emb_pca     = ep["emb_win_pca_norm"]
emb_std     = ep["emb_win_std_norm"]
labels_win  = ep["labels_win"]
win_file_id = ep["win_file_id"]
logit_sig   = ep["logit_sig_win"]      # (739, 234) — sigmoid of raw logits

DATA    = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
raw_emb = DATA["emb"].astype(np.float32)
n_files = int(len(DATA["n_windows"]))
n_sp    = int(DATA["labels"].shape[1])
filenames   = DATA["filenames"]
file_list   = DATA["file_list"]

with open(RESULTS_PATH) as f:
    res = json.load(f)
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch{BATCH}] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)
print(f"[batch{BATCH}] Total tried: {len(tried)}", flush=True)

def macro_auc(s):
    aucs = []
    for si in range(n_sp):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try: aucs.append(roc_auc_score(y, s[:, si]))
            except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, cfg=None):
    global best_loo
    if mname in tried: return score - best_loo
    entry = {"method": mname, "loo_auc": float(score), "config": cfg or {}, "batch": BATCH}
    res["experiments"].append(entry)
    tried.add(mname)
    if score > best_loo + 1e-7:
        best_loo = score
        res["best"] = {"method": mname, "loo_auc": float(score)}
        with open(MODEL_PATH, "rb") as f_pkl:
            ep_up = pickle.load(f_pkl)
        ep_up["method"] = mname; ep_up["loo_auc"] = float(score)
        with open(MODEL_PATH, "wb") as f_pkl:
            pickle.dump(ep_up, f_pkl)
        print(f"  *** NEW BEST: {mname} LOO={score:.6f} ***", flush=True)
    with open(RESULTS_PATH, "w") as f_out:
        json.dump(res, f_out, indent=2)
    return score - best_loo

# ── Chain ─────────────────────────────────────────────────────────────────────
fl_hard = file_labels.astype(np.float32)
count_i = fl_hard.sum(0) + EPS
COOC    = (fl_hard.T @ fl_hard) / count_i[:, None]; np.fill_diagonal(COOC, 0)
raw_idf = np.clip(np.log(float(n_files)/(count_i+1.0-EPS)), 0, None)
IDF075  = raw_idf**0.75; IDF075 /= (IDF075.mean()+EPS)

def soft_cooc(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s  = scores[fi]; g = 1./(1.+np.exp(np.clip(-slope*(s-center),-88,88)))
        sg = s*g*(idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi]=s; continue
        c = COOC.T@sg; mc=np.abs(c).max()
        if mc>EPS: c/=mc
        out[fi] = (1-alpha)*s + alpha*np.clip(c,0,None)
    return out

def apply_3way(s, alpha=0.200, blend=0.55, r_idf=0.875, r_tr=0.125, a1=0.110, a2=0.030):
    sp    = np.clip(s,0,1)**2
    sc    = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = (1-blend)*s + blend*sc
    r1    = soft_cooc(s, center=0.54, slope=41.0, alpha=a1)
    tr    = soft_cooc(r1, center=0.53, slope=37.0, alpha=a2)
    return r_idf*idf_s + r_tr*tr

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

fi_wins = [np.where(win_file_id==fi)[0] for fi in range(n_files)]
ot_wins = [np.where(win_file_id!=fi)[0] for fi in range(n_files)]

c3 = apply_3way(double_best, alpha=0.19)
i3 = apply_3way(ica_ens_alt, alpha=0.31)
s3 = apply_3way(std_ens_ref,  alpha=0.33)
rk = (0.56*make_rank(apply_3way(double_best,0.23)) +
      0.44*make_rank(apply_3way(ica_ens_alt,0.40))) / n_files

def wknn(SIM, k=5):
    sig = labels_win.astype(np.float32)
    out = np.zeros((n_files,n_sp),dtype=np.float32)
    for fi in range(n_files):
        fw=fi_wins[fi]; ow=ot_wins[fi]
        if not len(fw): continue
        ke=min(k,len(ow)); wp=np.zeros((len(fw),n_sp),dtype=np.float32)
        for wi,wk in enumerate(fw):
            s=SIM[wk,ow]; tl=np.argpartition(-s,ke-1)[:ke]
            w=np.clip(s[tl],0,None); ws=w.sum()
            w=w/ws if ws>EPS else np.ones(ke)/ke
            wp[wi]=(w[:,None]*sig[ow[tl]]).sum(0)
        out[fi]=wp.mean(0)
    return out

def chain(triple, ww=0.04, rm=0.28):
    chk = 0.74*c3 + 0.16*i3 + 0.06*s3 + ww*triple
    return (1-rm)*chk + rm*rk

# ── Build base ICA-128 ────────────────────────────────────────────────────────
print("\n[batch183] Building ICA-128 (seed=42)...", flush=True)
t0 = time.time()
scaler = StandardScaler(); rs = scaler.fit_transform(raw_emb)
pca256 = PCA(n_components=256, random_state=42); rp256 = pca256.fit_transform(rs)
ica128 = FastICA(n_components=128, random_state=42, max_iter=500, tol=0.01)
e128   = normalize(ica128.fit_transform(rp256), norm='l2')
SIM128 = e128 @ e128.T

SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
p_pca3  = wknn(SIM_PCA, k=3)
p_std5  = wknn(SIM_STD, k=5)

# Best params from batch181/182
BEST_ND, BEST_AL, BEST_K, BEST_WA = 3, 0.285, 5, 0.68
BEST_WW, BEST_RM = 0.05, 0.28

# Verify current best
def make_fused_sim(logit_feat_norm, al, base_emb=None):
    b = base_emb if base_emb is not None else e128
    ef = normalize(np.concatenate([(1-al)*b, al*logit_feat_norm], axis=1), norm='l2')
    return ef @ ef.T

# Build standard logit-PCA-3 for verification
lg = logit_sig.astype(np.float32)
lg_sc = StandardScaler().fit_transform(lg)
lpca3 = normalize(PCA(n_components=3, random_state=42).fit_transform(lg_sc), norm='l2')
sv = make_fused_sim(lpca3, 0.285)
pv = wknn(sv, k=5)
tr_best = BEST_WA*pv + 0.17*p_pca3 + 0.15*p_std5
print(f"  Verify batch182 best (n3 al285 k5 wa68): {macro_auc(chain(tr_best, 0.05, 0.28)):.6f}", flush=True)
print(f"  Base done in {time.time()-t0:.1f}s", flush=True)

def run_sweep(logit_feat_norm, tag, nd_tag):
    """Sweep best params around known optimum for a given logit feature."""
    results = []
    for k in [4, 5, 6, 7]:
        for al in [0.24, 0.26, 0.28, 0.285, 0.29, 0.30, 0.32, 0.35]:
            sf = make_fused_sim(logit_feat_norm, al)
            p_f = wknn(sf, k=k)
            for wa in [0.60, 0.62, 0.65, 0.68, 0.70, 0.72]:
                wb = round(1.0 - wa - 0.15, 3)
                if wb < 0.05: continue
                triple = wa*p_f + wb*p_pca3 + (1-wa-wb)*p_std5
                for ww in [0.04, 0.05, 0.06]:
                    for rm in [0.26, 0.28, 0.30]:
                        al_i = int(round(al*1000))
                        wa_i = int(wa*100)
                        mname = f"{tag}{nd_tag}_al{al_i}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                        if mname in tried: continue
                        auc = macro_auc(chain(triple, ww, rm))
                        diff = save_result(mname, auc, {"tag": tag, "al": al, "k": k, "wa": wa})
                        results.append((mname, auc, diff))
                        if diff >= -0.000020:
                            marker = " ***NEW BEST***" if diff>1e-7 else ""
                            print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)
    if results:
        best_r = max(results, key=lambda x: x[1])
        print(f"  [{tag}{nd_tag}] best: {best_r[0]} = {best_r[1]:.6f}", flush=True)
    return results

# ════════════════════════════════════════════════════════════════════════════
# A: Sigmoid(logit) before PCA
# logit_sig is already sigmoid output. Use it directly with standardization.
# ════════════════════════════════════════════════════════════════════════════
print("\n=== A: Sigmoid-logit (direct) PCA ===", flush=True)
t0 = time.time()

# logit_sig is already sigmoid. Try with and without StandardScaler.
# Option 1: StandardScaler on sigmoid values (centers around 0.5)
for n_dim in [2, 3, 4, 5, 6, 8]:
    # Already z-scored via StandardScaler
    lpca_sig = normalize(PCA(n_components=n_dim, random_state=42).fit_transform(lg_sc), norm='l2')
    run_sweep(lpca_sig, "sigpca", str(n_dim))

# Option 2: No scaling (raw sigmoid in [0,1])
lg_raw_norm = normalize(lg, norm='l2')
for n_dim in [2, 3, 4, 5, 6, 8]:
    lpca_raw = normalize(PCA(n_components=n_dim, random_state=42).fit_transform(lg), norm='l2')
    for al in [0.24, 0.26, 0.28, 0.30, 0.32]:
        sf = make_fused_sim(lpca_raw, al)
        for k in [4, 5, 6]:
            p_f = wknn(sf, k=k)
            for wa in [0.62, 0.65, 0.68, 0.70]:
                wb = round(1.0-wa-0.15, 3)
                if wb < 0.05: continue
                triple = wa*p_f + wb*p_pca3 + (1-wa-wb)*p_std5
                for ww, rm in [(0.04,0.28),(0.05,0.28),(0.05,0.26)]:
                    mname = f"rawsig{n_dim}_al{int(al*1000)}_k{k}_wa{int(wa*100)}_ww{int(ww*100)}_rm{int(rm*100)}"
                    if mname in tried: continue
                    auc = macro_auc(chain(triple, ww, rm))
                    diff = save_result(mname, auc, {"type":"raw_sig","n_dim":n_dim,"al":al})
                    if diff >= -0.000020:
                        marker = " ***NEW BEST***" if diff>1e-7 else ""
                        print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  A done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# B: Logit deviation (logit - per-window mean) — relative salience
# Each window's logit relative to its own mean → suppresses baseline shift
# ════════════════════════════════════════════════════════════════════════════
print("\n=== B: Logit deviation (logit - row_mean) ===", flush=True)
t0 = time.time()

lg_dev = lg - lg.mean(axis=1, keepdims=True)  # (739, 234)
lg_dev_sc = StandardScaler().fit_transform(lg_dev)

for n_dim in [2, 3, 4, 5, 6, 8]:
    lpca_dev = normalize(PCA(n_components=n_dim, random_state=42).fit_transform(lg_dev_sc), norm='l2')
    run_sweep(lpca_dev, "devpca", str(n_dim))

# Also try deviation without PCA (direct concat of top-k deviation dims)
# Use top-3 highest-variance deviation dimensions
var_dev = lg_dev_sc.var(0)
top3_idx = np.argsort(-var_dev)[:3]
lg_dev_top3 = normalize(lg_dev_sc[:, top3_idx], norm='l2')
for al in [0.24, 0.26, 0.28, 0.30, 0.32]:
    sf = make_fused_sim(lg_dev_top3, al)
    for k in [4, 5, 6]:
        p_f = wknn(sf, k=k)
        for wa in [0.62, 0.65, 0.68]:
            wb = round(1.0-wa-0.15, 3)
            if wb < 0.05: continue
            triple = wa*p_f + wb*p_pca3 + (1-wa-wb)*p_std5
            for ww, rm in [(0.05,0.28),(0.04,0.28)]:
                mname = f"devtop3_al{int(al*1000)}_k{k}_wa{int(wa*100)}_ww{int(ww*100)}_rm{int(rm*100)}"
                if mname in tried: continue
                auc = macro_auc(chain(triple, ww, rm))
                diff = save_result(mname, auc, {"type":"dev_top3"})
                if diff >= -0.000020:
                    marker = " ***NEW BEST***" if diff>1e-7 else ""
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  B done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# C: Rank-normalized logit — ordinal transformation
# Per-window, rank species by logit value → captures relative order
# ════════════════════════════════════════════════════════════════════════════
print("\n=== C: Rank-normalized logit ===", flush=True)
t0 = time.time()

# Per-window species rank (ordinal) — normalized to [0,1]
lg_rank = np.argsort(np.argsort(lg, axis=1), axis=1).astype(np.float32)
lg_rank /= (lg_rank.max(1, keepdims=True) + EPS)  # normalize to [0,1]
lg_rank_sc = StandardScaler().fit_transform(lg_rank)

for n_dim in [2, 3, 4, 5, 6, 8]:
    lpca_rnk = normalize(PCA(n_components=n_dim, random_state=42).fit_transform(lg_rank_sc), norm='l2')
    run_sweep(lpca_rnk, "rankpca", str(n_dim))

print(f"  C done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# D: Species-IDF weighted logit
# Weight rare species' logits more (IDF weighting) before PCA
# ════════════════════════════════════════════════════════════════════════════
print("\n=== D: IDF-weighted logit PCA ===", flush=True)
t0 = time.time()

# IDF weights: log(N/df) for each species
label_counts = labels_win.sum(0) + EPS  # (234,) — window-level counts
idf_weights  = np.log(float(len(labels_win)) / (label_counts + 1.0))
idf_weights  = np.clip(idf_weights, 0, None)
idf_weights /= (idf_weights.mean() + EPS)

# Weight logit by IDF
lg_idf = lg_sc * idf_weights[None, :]  # (739, 234)
lg_idf = StandardScaler().fit_transform(lg_idf)  # re-normalize

for n_dim in [2, 3, 4, 5, 6, 8]:
    lpca_idf = normalize(PCA(n_components=n_dim, random_state=42).fit_transform(lg_idf), norm='l2')
    run_sweep(lpca_idf, "idfpca", str(n_dim))

print(f"  D done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# E: Temporal variance logit (file-level std as window feature)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== E: Temporal variance logit ===", flush=True)
t0 = time.time()

# For each window, add file-level statistics as features:
# Approach: file_mean_logit subtracted from window logit
# This highlights windows that deviate from their recording's average
file_mean_logit = np.zeros((len(lg), lg.shape[1]), dtype=np.float32)
for fi in range(n_files):
    mask = fi_wins[fi]
    if len(mask) > 0:
        file_mean_logit[mask] = lg[mask].mean(0)

lg_temporal_dev = lg - file_mean_logit  # deviation from file mean
lg_td_sc = StandardScaler().fit_transform(lg_temporal_dev)

for n_dim in [2, 3, 4, 5, 6]:
    lpca_td = normalize(PCA(n_components=n_dim, random_state=42).fit_transform(lg_td_sc), norm='l2')
    run_sweep(lpca_td, "tdpca", str(n_dim))

# Also try: concatenate logit-PCA-3 (best) + temporal-deviation-PCA-3
lpca3_ref = normalize(PCA(n_components=3, random_state=42).fit_transform(lg_sc), norm='l2')
lpca3_td  = normalize(PCA(n_components=3, random_state=42).fit_transform(lg_td_sc), norm='l2')

for al_main, al_td in [(0.22,0.06), (0.20,0.08), (0.25,0.05), (0.18,0.10)]:
    al_total = al_main + al_td
    ef = normalize(np.concatenate([(1-al_total)*e128, al_main*lpca3_ref, al_td*lpca3_td], axis=1), norm='l2')
    sf = ef @ ef.T
    for k in [4, 5, 6]:
        p_f = wknn(sf, k=k)
        for wa in [0.62, 0.65, 0.68]:
            wb = round(1.0-wa-0.15, 3)
            if wb < 0.05: continue
            triple = wa*p_f + wb*p_pca3 + (1-wa-wb)*p_std5
            for ww, rm in [(0.05,0.28),(0.04,0.28)]:
                al_mi = int(al_main*100); al_tdi = int(al_td*100)
                mname = f"ica128l3tdpca3_alm{al_mi}_alt{al_tdi}_k{k}_wa{int(wa*100)}_ww{int(ww*100)}_rm{int(rm*100)}"
                if mname in tried: continue
                auc = macro_auc(chain(triple, ww, rm))
                diff = save_result(mname, auc, {"type":"main_plus_td"})
                if diff >= -0.000020:
                    marker = " ***NEW BEST***" if diff>1e-7 else ""
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  E done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# F: Signed log logit — log(|logit|)*sign(logit)
# Compresses heavy tail, emphasizes extreme detections
# ════════════════════════════════════════════════════════════════════════════
print("\n=== F: Signed-log logit transformation ===", flush=True)
t0 = time.time()

# logit_sig is already sigmoid. Convert back to raw logit: logit = log(p/(1-p))
p_clip = np.clip(logit_sig, 1e-6, 1-1e-6).astype(np.float32)
raw_logit = np.log(p_clip / (1.0 - p_clip))  # (739, 234) raw logit values

# Signed log: sign(x) * log(1 + |x|)
signed_log = np.sign(raw_logit) * np.log1p(np.abs(raw_logit))
signed_log_sc = StandardScaler().fit_transform(signed_log)

for n_dim in [2, 3, 4, 5, 6, 8]:
    lpca_sl = normalize(PCA(n_components=n_dim, random_state=42).fit_transform(signed_log_sc), norm='l2')
    run_sweep(lpca_sl, "slogpca", str(n_dim))

print(f"  F done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# G: Quantile-normalized logit
# Map each species logit to uniform distribution before PCA
# ════════════════════════════════════════════════════════════════════════════
print("\n=== G: Quantile-normalized logit ===", flush=True)
t0 = time.time()

qt = QuantileTransformer(n_quantiles=min(500, len(lg)), output_distribution='normal', random_state=42)
lg_qt = qt.fit_transform(lg).astype(np.float32)
lg_qt_sc = StandardScaler().fit_transform(lg_qt)

for n_dim in [2, 3, 4, 5, 6, 8]:
    lpca_qt = normalize(PCA(n_components=n_dim, random_state=42).fit_transform(lg_qt_sc), norm='l2')
    run_sweep(lpca_qt, "qtpca", str(n_dim))

print(f"  G done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n[batch{BATCH}] Done. Final best: {res['best']['method']} LOO={res['best']['loo_auc']:.6f}",
      flush=True)
exps = [e for e in res["experiments"] if e.get("batch")==BATCH]
print(f"[batch{BATCH}] Experiments this batch: {len(exps)}", flush=True)
if exps:
    top = sorted(exps, key=lambda x: x["loo_auc"], reverse=True)[:10]
    print(f"[batch{BATCH}] Top-10:", flush=True)
    for e in top:
        print(f"  {e['method']}: {e['loo_auc']:.6f}", flush=True)
