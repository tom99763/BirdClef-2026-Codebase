"""
batch182 — Logit-fusion ultra-fine sweep + novel logit components
==================================================================
NEW BEST from batch181: ica128logit4_al28_k5_wa6_ww4_rm28 LOO=0.996020
Key params: alpha=0.28, wa=0.6, n_logit_comp=4-9 (all tie), k=5-6

batch182 probes the edges of this plateau:
  A: Ultra-fine alpha sweep (0.20..0.35 step 0.005) × n_dim=4,5,6 × wa=0.6
  B: Ultra-fine wa sweep (0.55..0.72 step 0.01) × best alpha=0.28 × n=4,5,6
  C: Very small n_logit_comp (1,2,3) — minimal logit signal
  D: Multi-seed ICA-128 with logit-fusion (does seed diversity help?)
  E: Logit-similarity KNN (use SIM_logit as explicit wkt4 component)
  F: Logit-fused ICA-128 at alt seeds × best alpha=0.28 × wa=0.6
  G: Triple concat: ICA-128 + logit-PCA + NMF
"""
import numpy as np
import json, pickle, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import normalize, StandardScaler
import warnings
warnings.filterwarnings('ignore')

EPS   = 1e-8
BATCH = 182
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
emb_nmf     = ep["emb_win_nmf_norm"]
labels_win  = ep["labels_win"]
win_file_id = ep["win_file_id"]
logit_sig   = ep["logit_sig_win"]

DATA    = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
raw_emb = DATA["emb"].astype(np.float32)
n_files = int(len(DATA["n_windows"]))
n_sp    = int(DATA["labels"].shape[1])
filenames   = DATA["filenames"]
file_list   = DATA["file_list"]
fname_to_id = {f: i for i, f in enumerate(file_list)}

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
        with open(MODEL_PATH, "rb") as f:
            ep_up = pickle.load(f)
        ep_up["method"] = mname; ep_up["loo_auc"] = float(score)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(ep_up, f)
        print(f"  *** NEW BEST: {mname} LOO={score:.6f} ***", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

# ── Chain setup ───────────────────────────────────────────────────────────────
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

def wknn(SIM, k=6):
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

# ── Build base embeddings ─────────────────────────────────────────────────────
print("\n[batch182] Building base embeddings...", flush=True)
t0 = time.time()

SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T
SIM_NMF = emb_nmf @ emb_nmf.T
p_pca3 = wknn(SIM_PCA, k=3)
p_std5 = wknn(SIM_STD, k=5)

# ICA-128 seed=42
scaler = StandardScaler(); raw_s = scaler.fit_transform(raw_emb)
pca256 = PCA(n_components=256, random_state=42)
rp256  = pca256.fit_transform(raw_s)
ica128 = FastICA(n_components=128, random_state=42, max_iter=500, tol=0.01)
e128   = normalize(ica128.fit_transform(rp256), norm='l2')
SIM128 = e128 @ e128.T

# Logit preprocessing
lg = logit_sig.astype(np.float32)
lg_sc = StandardScaler().fit_transform(lg)
lg_nrm = normalize(lg_sc.astype(np.float32), norm='l2')  # raw normalized logit

# Precompute logit PCA dims 1..32
lpcas = {}
for d in range(1, 33):
    lpcas[d] = normalize(PCA(n_components=d, random_state=42).fit_transform(lg_sc), norm='l2')

def sim_fused(nd, al, seed_emb=None):
    base = seed_emb if seed_emb is not None else e128
    ef = np.concatenate([(1-al)*base, al*lpcas[nd]], axis=1)
    return normalize(ef, norm='l2') @ normalize(ef, norm='l2').T

# Verify batch181 best
sv = sim_fused(4, 0.28)
pv = wknn(sv, k=5); trv = 0.6*pv + 0.25*p_pca3 + 0.15*p_std5
print(f"  Verify batch181 best (n4 al28 k5 wa6): {macro_auc(chain(trv, 0.04, 0.28)):.6f}", flush=True)
print(f"  Base embeddings done in {time.time()-t0:.1f}s", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# A: Ultra-fine alpha sweep × wa=0.6 (0.55..0.70)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== A: Ultra-fine alpha × wa sweep ===", flush=True)
t0 = time.time()

for nd in [3, 4, 5, 6, 8]:
    for al in [round(x,3) for x in np.arange(0.18, 0.40, 0.005)]:
        sf = sim_fused(nd, al)
        for k in [4, 5, 6, 7]:
            p_f = wknn(sf, k=k)
            for wa in [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]:
                wb = round(1.0 - wa - 0.15, 2)  # wa + wb + 0.15(STD) = 1
                if wb < 0.10: continue
                triple = wa*p_f + wb*p_pca3 + (1-wa-wb)*p_std5
                for ww in [0.03, 0.04, 0.05]:
                    for rm in [0.26, 0.28, 0.30]:
                        al_i = int(round(al*1000))
                        wa_i = int(wa*100)
                        mname = f"ica128l{nd}_al{al_i}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                        if mname in tried: continue
                        auc = macro_auc(chain(triple, ww, rm))
                        diff = save_result(mname, auc, {"nd":nd,"al":al,"k":k,"wa":wa})
                        if diff >= -0.000050:
                            marker = " ***NEW BEST***" if diff>1e-7 else ""
                            print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  A done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# B: Very small n_logit_comp (1,2,3) — minimal logit signal
# ════════════════════════════════════════════════════════════════════════════
print("\n=== B: n_logit=1,2,3 ===", flush=True)
t0 = time.time()

for nd in [1, 2, 3]:
    for al in [round(x,2) for x in np.arange(0.20, 0.50, 0.02)]:
        sf = sim_fused(nd, al)
        for k in [4, 5, 6, 7, 8]:
            p_f = wknn(sf, k=k)
            for wa in [0.55, 0.60, 0.65, 0.70]:
                wb = round(1.0 - wa - 0.15, 2)
                if wb < 0.05: continue
                triple = wa*p_f + wb*p_pca3 + (1-wa-wb)*p_std5
                for ww in [0.03, 0.04, 0.05]:
                    for rm in [0.26, 0.28, 0.30]:
                        mname = f"ica128l{nd}_al{int(al*100):02d}_k{k}_wa{int(wa*100)}_ww{int(ww*100)}_rm{int(rm*100)}"
                        if mname in tried: continue
                        auc = macro_auc(chain(triple, ww, rm))
                        diff = save_result(mname, auc, {"nd":nd,"al":al,"k":k,"wa":wa})
                        if diff >= -0.000050:
                            marker = " ***NEW BEST***" if diff>1e-7 else ""
                            print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  B done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# C: Logit SIM as explicit wkt4 component
# ════════════════════════════════════════════════════════════════════════════
print("\n=== C: Logit SIM as wkt4 component ===", flush=True)
t0 = time.time()

# Logit-PCA SIM matrices for multiple dims
for nd_sim in [4, 6, 8, 12, 16, 24]:
    SIM_logit_nd = lpcas[nd_sim] @ lpcas[nd_sim].T
    for k_l in [3, 4, 5, 6]:
        p_lg = wknn(SIM_logit_nd, k=k_l)
        # wkt3 with logit: ICA-fused + PCA + logit
        sf_best = sim_fused(4, 0.28)
        p_fused6 = wknn(sf_best, k=6)
        for wlf, wp, wlg, ws in [
            (5,2,2,1), (5,2,1,2), (4,3,2,1), (4,2,2,2),
            (6,2,1,1), (5,3,1,1), (6,1,2,1), (4,3,1,2),
        ]:
            total = wlf+wp+wlg+ws
            triple4 = (wlf*p_fused6 + wp*p_pca3 + wlg*p_lg + ws*p_std5) / total
            for ww in [0.03, 0.04, 0.05]:
                for rm in [0.26, 0.28, 0.30]:
                    mname = f"wkt4_lf{wlf}_pc{wp}_lg{wlg}_{nd_sim}_st{ws}_kl{k_l}_ww{int(ww*100)}_rm{int(rm*100)}"
                    if mname in tried: continue
                    auc = macro_auc(chain(triple4, ww, rm))
                    diff = save_result(mname, auc, {"type":"wkt4_logit_sim","nd_sim":nd_sim})
                    if diff >= -0.000050:
                        marker = " ***NEW BEST***" if diff>1e-7 else ""
                        print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

# Also try raw logit SIM
SIM_logit_raw = lg_nrm @ lg_nrm.T
for k_l in [3, 4, 5]:
    p_lg_r = wknn(SIM_logit_raw, k=k_l)
    sf_best = sim_fused(4, 0.28)
    p_fused6 = wknn(sf_best, k=6)
    for wlf, wp, wlg, ws in [(5,2,2,1), (4,3,2,1), (6,2,1,1)]:
        total = wlf+wp+wlg+ws
        triple = (wlf*p_fused6 + wp*p_pca3 + wlg*p_lg_r + ws*p_std5) / total
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.26, 0.28, 0.30]:
                mname = f"wkt4_lf{wlf}_pc{wp}_lgraw{wlg}_st{ws}_kl{k_l}_ww{int(ww*100)}_rm{int(rm*100)}"
                if mname in tried: continue
                auc = macro_auc(chain(triple, ww, rm))
                diff = save_result(mname, auc, {"type":"wkt4_logit_raw"})
                if diff >= -0.000050:
                    marker = " ***NEW BEST***" if diff>1e-7 else ""
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  C done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# D: Multi-seed ICA-128 logit-fused ensemble
# ════════════════════════════════════════════════════════════════════════════
print("\n=== D: Multi-seed logit-fused ensemble ===", flush=True)
t0 = time.time()

SEEDS_D = [0, 1, 7, 13, 17, 31, 42, 50, 77]
sims_by_seed = {42: sim_fused(4, 0.28)}  # seed=42 already computed

for seed in SEEDS_D:
    if seed == 42: continue
    ica_s = FastICA(n_components=128, random_state=seed, max_iter=500, tol=0.01)
    es = normalize(ica_s.fit_transform(rp256), norm='l2')
    # Fuse with best params: n=4, al=0.28
    ef_s = normalize(np.concatenate([0.72*es, 0.28*lpcas[4]], axis=1), norm='l2')
    sims_by_seed[seed] = ef_s @ ef_s.T
    print(f"  Built seed={seed}", flush=True)

# Average N seeds
for n_s in [2, 3, 4, 5, 7, 9]:
    sl = SEEDS_D[:n_s]
    SIM_avg = np.mean([sims_by_seed[s] for s in sl], axis=0)
    np.fill_diagonal(SIM_avg, 0.0)
    for k in [4, 5, 6, 7]:
        p_avg = wknn(SIM_avg, k=k)
        for wa in [0.55, 0.60, 0.65]:
            wb = round(1.0-wa-0.15, 2)
            if wb < 0.05: continue
            triple = wa*p_avg + wb*p_pca3 + (1-wa-wb)*p_std5
            for ww in [0.03, 0.04, 0.05]:
                for rm in [0.26, 0.28, 0.30]:
                    wa_i = int(wa*100)
                    mname = f"ica128lf_ms{n_s}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                    if mname in tried: continue
                    auc = macro_auc(chain(triple, ww, rm))
                    diff = save_result(mname, auc, {"n_seeds":n_s,"k":k,"wa":wa})
                    if diff >= -0.000050:
                        marker = " ***NEW BEST***" if diff>1e-7 else ""
                        print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  D done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# E: Single-seed variants with best alpha/wa
# ════════════════════════════════════════════════════════════════════════════
print("\n=== E: Single-seed variants (best params) ===", flush=True)
t0 = time.time()

for seed in SEEDS_D:
    if seed == 42: continue  # already in tried
    # Try seed with best params: n=4, al=0.28, k=5, wa=0.6
    SIM_s = sims_by_seed[seed]
    for k in [4, 5, 6, 7]:
        p_s = wknn(SIM_s, k=k)
        for wa in [0.55, 0.60, 0.65]:
            wb = round(1.0-wa-0.15, 2)
            if wb < 0.05: continue
            triple = wa*p_s + wb*p_pca3 + (1-wa-wb)*p_std5
            for ww in [0.03, 0.04, 0.05]:
                for rm in [0.26, 0.28, 0.30]:
                    wa_i = int(wa*100)
                    mname = f"ica128lf_s{seed}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                    if mname in tried: continue
                    auc = macro_auc(chain(triple, ww, rm))
                    diff = save_result(mname, auc, {"seed":seed,"k":k,"wa":wa})
                    if diff >= -0.000050:
                        marker = " ***NEW BEST***" if diff>1e-7 else ""
                        print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  E done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# F: Triple concat: ICA-128 + logit-PCA + NMF (three-way fusion)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== F: Triple concat ICA-128 + logit-PCA + NMF ===", flush=True)
t0 = time.time()

# NMF embedding: normalize existing NMF window embeddings
emb_nmf_n = normalize(emb_nmf, norm='l2')  # (739,100)

for nd in [4, 6, 8]:
    for al_l, al_n in [(0.20,0.10), (0.22,0.08), (0.25,0.10), (0.28,0.08), (0.25,0.05)]:
        al_main = 1.0 - al_l - al_n
        if al_main < 0.5: continue
        ef = normalize(np.concatenate([
            al_main*e128, al_l*lpcas[nd], al_n*emb_nmf_n[:,:nd]
        ], axis=1), norm='l2')
        SIM_tri = ef @ ef.T

        for k in [4, 5, 6]:
            p_tri = wknn(SIM_tri, k=k)
            for wa in [0.55, 0.60, 0.65]:
                wb = round(1.0-wa-0.15, 2)
                if wb < 0.05: continue
                triple = wa*p_tri + wb*p_pca3 + (1-wa-wb)*p_std5
                for ww in [0.03, 0.04, 0.05]:
                    for rm in [0.26, 0.28, 0.30]:
                        al_li = int(al_l*100); al_ni = int(al_n*100); wa_i = int(wa*100)
                        mname = f"ica128lf3_{nd}_all{al_li}_aln{al_ni}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                        if mname in tried: continue
                        auc = macro_auc(chain(triple, ww, rm))
                        diff = save_result(mname, auc, {"type":"triple_fused","nd":nd})
                        if diff >= -0.000050:
                            marker = " ***NEW BEST***" if diff>1e-7 else ""
                            print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  F done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# G: Logit-fused with ICA-100 instead of ICA-128 (cross-dim)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== G: Logit-fused ICA-100 (cross-dim check) ===", flush=True)
t0 = time.time()

e100 = emb_ica  # (739,100) existing ICA-100

for nd in [3, 4, 5, 6, 8]:
    for al in [0.24, 0.26, 0.28, 0.30, 0.32]:
        ef = normalize(np.concatenate([(1-al)*e100, al*lpcas[nd]], axis=1), norm='l2')
        SIM_l100 = ef @ ef.T
        for k in [5, 6, 7]:
            p_l100 = wknn(SIM_l100, k=k)
            for wa in [0.55, 0.60, 0.65]:
                wb = round(1.0-wa-0.15, 2)
                if wb < 0.05: continue
                triple = wa*p_l100 + wb*p_pca3 + (1-wa-wb)*p_std5
                for ww in [0.03, 0.04, 0.05]:
                    for rm in [0.26, 0.28, 0.30]:
                        wa_i = int(wa*100)
                        mname = f"ica100l{nd}_al{int(al*100)}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                        if mname in tried: continue
                        auc = macro_auc(chain(triple, ww, rm))
                        diff = save_result(mname, auc, {"type":"ica100_logitfused","nd":nd})
                        if diff >= -0.000050:
                            marker = " ***NEW BEST***" if diff>1e-7 else ""
                            print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  G done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ─── Final summary ────────────────────────────────────────────────────────────
print(f"\n[batch{BATCH}] Done. Final best: {res['best']['method']} LOO={res['best']['loo_auc']:.6f}",
      flush=True)
exps = [e for e in res["experiments"] if e.get("batch")==BATCH]
print(f"[batch{BATCH}] Experiments this batch: {len(exps)}", flush=True)
if exps:
    top = sorted(exps, key=lambda x: x["loo_auc"], reverse=True)[:10]
    print(f"[batch{BATCH}] Top-10:", flush=True)
    for e in top:
        print(f"  {e['method']}: {e['loo_auc']:.6f}", flush=True)
