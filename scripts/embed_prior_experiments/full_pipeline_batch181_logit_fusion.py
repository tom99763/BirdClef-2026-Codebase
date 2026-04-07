"""
batch181 — Deep Logit-Fusion Exploitation
==========================================
NEW BEST from batch180: ica128logit8_al30_k6_wa5_ww4_rm28 LOO=0.996004
Key insight: Concatenating ICA-128 emb + PCA(logit, 8-dim)*0.30 → better KNN geometry

batch181 systematically exploits this logit-fusion breakthrough:
  A: Fine alpha_logit sweep (0.20..0.40 step 0.02)
  B: Fine n_logit_comp sweep (4,5,6,7,8,9,10,12,14,16)
  C: k sweep around k=6 (k=4..10) for best logit configs
  D: wa/ww/rm sweep for best logit configs
  E: Logit-ICA (FastICA on logits instead of PCA)
  F: STD-normalized logit concat (instead of PCA on logits)
  G: Two-logit concat: both PCA-logit and direct logit contributions
  H: wkt4/wkt5 with logit-fused as new component
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
BATCH = 181
ROOT  = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

# ── Load PKL ──────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

file_labels = ep["file_labels"]
double_best = ep["chain_double_best"]
ica_ens_alt = ep["chain_ica_ens_alt"]
std_ens_ref = ep["chain_std_ens_ref"]
emb_ica     = ep["emb_win_ica_norm"]   # (739,100)
emb_pca     = ep["emb_win_pca_norm"]   # (739,80)
emb_std     = ep["emb_win_std_norm"]   # (739,80)
labels_win  = ep["labels_win"]
win_file_id = ep["win_file_id"]
logit_sig   = ep["logit_sig_win"]      # (739,234)

DATA      = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
raw_emb   = DATA["emb"].astype(np.float32)
n_files   = int(len(DATA["n_windows"]))
n_species = int(DATA["labels"].shape[1])
filenames = DATA["filenames"]
file_list = DATA["file_list"]
fname_to_id = {f: i for i, f in enumerate(file_list)}
win_file_id_raw = np.array([fname_to_id.get(str(f),-1) for f in filenames], dtype=np.int32)

with open(RESULTS_PATH) as f:
    res = json.load(f)
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch{BATCH}] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)
print(f"[batch{BATCH}] Total tried: {len(tried)}", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def macro_auc(s):
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try: aucs.append(roc_auc_score(y, s[:, si]))
            except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, cfg=None):
    global best_loo
    if mname in tried:
        return score - best_loo
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

# ── Co-occurrence chain ───────────────────────────────────────────────────────
fl_hard = file_labels.astype(np.float32)
count_i = fl_hard.sum(0) + EPS
COOC    = (fl_hard.T @ fl_hard) / count_i[:, None]; np.fill_diagonal(COOC, 0)
raw_idf = np.clip(np.log(float(n_files)/(count_i+1.0-EPS)), 0, None)
IDF075  = raw_idf**0.75; IDF075 /= (IDF075.mean()+EPS)

def soft_cooc(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s  = scores[fi]
        g  = 1.0/(1.0+np.exp(np.clip(-slope*(s-center),-88,88)))
        sg = s*g*(idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi]=s; continue
        c  = COOC.T@sg; mc=np.abs(c).max()
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

fi_wins_list    = [np.where(win_file_id==fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id!=fi)[0] for fi in range(n_files)]

c3_ref     = apply_3way(double_best, alpha=0.19)
i3_ref     = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref     = apply_3way(std_ens_ref, alpha=0.33)
rank_c_ref = make_rank(apply_3way(double_best, alpha=0.23))
rank_i_ref = make_rank(apply_3way(ica_ens_alt, alpha=0.40))
rank_norm  = (0.56*rank_c_ref + 0.44*rank_i_ref) / n_files

def wknn_single(SIM, k=7):
    signal = labels_win.astype(np.float32)
    preds  = np.zeros((n_files, n_species), dtype=np.float32)
    for fi in range(n_files):
        fi_w = fi_wins_list[fi]; ow = other_wins_list[fi]
        if len(fi_w)==0: continue
        ke = min(k, len(ow))
        wp = np.zeros((len(fi_w), n_species), dtype=np.float32)
        for wi, wkk in enumerate(fi_w):
            sims = SIM[wkk, ow]
            tl = np.argpartition(-sims, ke-1)[:ke]
            tw = ow[tl]; w = np.clip(sims[tl], 0, None)
            ws = w.sum(); w = w/ws if ws>EPS else np.ones(ke)/ke
            wp[wi] = (w[:,None]*signal[tw]).sum(0)
        preds[fi] = wp.mean(0)
    return preds

def run_chain(triple, ww=0.04, rm=0.28):
    chk4  = 0.74*c3_ref + 0.16*i3_ref + 0.06*s3_ref + ww*triple
    final = (1-rm)*chk4 + rm*rank_norm
    return final

# ── Build baseline embeddings ─────────────────────────────────────────────────
print("\n[batch181] Building base embeddings...", flush=True)
t0 = time.time()
SIM_ICA = emb_ica @ emb_ica.T
SIM_PCA = emb_pca @ emb_pca.T
SIM_STD = emb_std @ emb_std.T

p_ica6 = wknn_single(SIM_ICA, k=6)
p_pca3 = wknn_single(SIM_PCA, k=3)
p_std5 = wknn_single(SIM_STD, k=5)
wkt3_ref = 0.5*p_ica6 + 0.3*p_pca3 + 0.2*p_std5
base_chk = run_chain(wkt3_ref, ww=0.04, rm=0.28)
print(f"  Baseline wkt3: {macro_auc(base_chk):.6f}", flush=True)

# Build ICA-128 (seed=42) — new best embedding
scaler_raw = StandardScaler()
raw_scaled = scaler_raw.fit_transform(raw_emb)
pca_pre256 = PCA(n_components=256, random_state=42)
raw_pca256 = pca_pre256.fit_transform(raw_scaled)
ica128s42  = FastICA(n_components=128, random_state=42, max_iter=500, tol=0.01)
emb_ica128 = normalize(ica128s42.fit_transform(raw_pca256), norm='l2')
SIM_ICA128 = emb_ica128 @ emb_ica128.T

p_ica128_6 = wknn_single(SIM_ICA128, k=6)
wkt3_ica128 = 0.5*p_ica128_6 + 0.3*p_pca3 + 0.2*p_std5
chk_ica128 = run_chain(wkt3_ica128, ww=0.04, rm=0.28)
print(f"  ICA-128 s42 k6 ww4 rm28: {macro_auc(chk_ica128):.6f}", flush=True)

# Build logit PCA features (batch180 best: alpha=0.30, n_logit=8, k=6)
logit_arr    = logit_sig.astype(np.float32)
logit_scaler = StandardScaler()
logit_scaled = logit_scaler.fit_transform(logit_arr)

# Pre-compute logit PCA for all needed dimensions
logit_pcas = {}
for n_dim in [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32]:
    pca_l = PCA(n_components=n_dim, random_state=42)
    logit_pcas[n_dim] = normalize(pca_l.fit_transform(logit_scaled), norm='l2')

print(f"  Base embeddings done in {time.time()-t0:.1f}s", flush=True)

# Verify new best (batch180)
def make_fused_sim(n_dim, alpha):
    ef = np.concatenate([(1-alpha)*emb_ica128, alpha*logit_pcas[n_dim]], axis=1)
    ef = normalize(ef, norm='l2')
    return ef @ ef.T

SIM_best = make_fused_sim(8, 0.30)
p_best = wknn_single(SIM_best, k=6)
triple_best = 0.5*p_best + 0.3*p_pca3 + 0.2*p_std5
chk_best = run_chain(triple_best, ww=0.04, rm=0.28)
print(f"  Verify batch180 best (n8 al30 k6 wa5 ww4 rm28): {macro_auc(chk_best):.6f}", flush=True)
print(flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Section A: Fine alpha_logit sweep
# ════════════════════════════════════════════════════════════════════════════
print("=== A: Fine alpha_logit sweep ===", flush=True)
t0 = time.time()

# Fixed: n_dim=8, k=6, wa=0.5, ww=0.04, rm=0.28 (batch180 best params)
for n_dim in [6, 7, 8, 9, 10]:
    for alpha in np.arange(0.15, 0.55, 0.01):
        alpha = round(float(alpha), 2)
        SIM_f = make_fused_sim(n_dim, alpha)
        p_f = wknn_single(SIM_f, k=6)
        triple = 0.5*p_f + 0.3*p_pca3 + 0.2*p_std5
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.26, 0.28, 0.30]:
                mname = f"ica128logit{n_dim}_al{int(alpha*100):02d}_k6_wa5_ww{int(ww*100)}_rm{int(rm*100)}"
                if mname in tried: continue
                final = run_chain(triple, ww=ww, rm=rm)
                auc = macro_auc(final)
                diff = save_result(mname, auc, {"n_dim": n_dim, "alpha": alpha, "k": 6})
                if diff > -0.0005:
                    marker = " ***NEW BEST***" if diff > 1e-7 else ""
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section A done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Section B: Fine n_logit_comp sweep with best alpha range
# ════════════════════════════════════════════════════════════════════════════
print("\n=== B: Fine n_logit_comp sweep ===", flush=True)
t0 = time.time()

# Sweep n_dim 4..32, alpha in {0.25, 0.28, 0.30, 0.32, 0.35}
for n_dim in [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32]:
    for alpha in [0.25, 0.28, 0.30, 0.32, 0.35]:
        SIM_f = make_fused_sim(n_dim, alpha)
        for k in [5, 6, 7, 8]:
            p_f = wknn_single(SIM_f, k=k)
            for wa, wb, wc in [(0.5, 0.3, 0.2), (0.55, 0.25, 0.20), (0.6, 0.2, 0.2)]:
                triple = wa*p_f + wb*p_pca3 + wc*p_std5
                for ww in [0.03, 0.04, 0.05]:
                    for rm in [0.26, 0.28, 0.30]:
                        wa_i = int(wa*10)
                        mname = f"ica128logit{n_dim}_al{int(alpha*100):02d}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                        if mname in tried: continue
                        final = run_chain(triple, ww=ww, rm=rm)
                        auc = macro_auc(final)
                        diff = save_result(mname, auc, {"n_dim": n_dim, "alpha": alpha,
                                                         "k": k, "wa": wa})
                        if diff > -0.0003:
                            marker = " ***NEW BEST***" if diff > 1e-7 else ""
                            print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section B done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Section C: Logit ICA (FastICA on logits instead of PCA)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== C: Logit ICA (FastICA on logits) ===", flush=True)
t0 = time.time()

# Build FastICA on logit features
for n_ica_l in [8, 12, 16, 20, 24, 32]:
    ica_logit = FastICA(n_components=n_ica_l, random_state=42, max_iter=300, tol=0.01)
    try:
        logit_ica_emb = normalize(ica_logit.fit_transform(logit_scaled), norm='l2')
    except:
        print(f"  ICA logit {n_ica_l} failed, skip", flush=True)
        continue

    for alpha in [0.20, 0.25, 0.30, 0.35, 0.40]:
        ef = np.concatenate([(1-alpha)*emb_ica128, alpha*logit_ica_emb], axis=1)
        ef = normalize(ef, norm='l2')
        SIM_fica = ef @ ef.T

        for k in [5, 6, 7]:
            p_fica = wknn_single(SIM_fica, k=k)
            for wa, wb, wc in [(0.5, 0.3, 0.2), (0.55, 0.25, 0.20)]:
                triple = wa*p_fica + wb*p_pca3 + wc*p_std5
                for ww in [0.03, 0.04, 0.05]:
                    for rm in [0.26, 0.28, 0.30]:
                        wa_i = int(wa*10)
                        mname = f"ica128logitICA{n_ica_l}_al{int(alpha*100):02d}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                        if mname in tried: continue
                        final = run_chain(triple, ww=ww, rm=rm)
                        auc = macro_auc(final)
                        diff = save_result(mname, auc, {"type": "logit_ica", "n_ica_l": n_ica_l})
                        if diff > -0.0003:
                            marker = " ***NEW BEST***" if diff > 1e-7 else ""
                            print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section C done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Section D: STD-normalized logit concat (z-score each logit dim, then concat)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== D: STD-normalized logit concat ===", flush=True)
t0 = time.time()

# Use full 234-dim logit but z-score normalized + PCA to reduce dim
for n_dim_s in [8, 12, 16, 24, 32, 48, 64]:
    pca_s = PCA(n_components=min(n_dim_s, 100), random_state=42)
    # Use StandardScaler + PCA (same as existing std_emb approach but on logits)
    logit_std_emb = normalize(pca_s.fit_transform(logit_scaled), norm='l2')

    for alpha in [0.20, 0.25, 0.30, 0.35]:
        ef = np.concatenate([(1-alpha)*emb_ica128, alpha*logit_std_emb], axis=1)
        ef = normalize(ef, norm='l2')
        SIM_ls = ef @ ef.T

        for k in [5, 6, 7]:
            p_ls = wknn_single(SIM_ls, k=k)
            triple = 0.5*p_ls + 0.3*p_pca3 + 0.2*p_std5
            for ww in [0.03, 0.04, 0.05]:
                for rm in [0.26, 0.28, 0.30]:
                    mname = f"ica128logitSTD{n_dim_s}_al{int(alpha*100):02d}_k{k}_ww{int(ww*100)}_rm{int(rm*100)}"
                    if mname in tried: continue
                    final = run_chain(triple, ww=ww, rm=rm)
                    auc = macro_auc(final)
                    diff = save_result(mname, auc, {"type": "logit_std", "n_dim": n_dim_s})
                    if diff > -0.0003:
                        marker = " ***NEW BEST***" if diff > 1e-7 else ""
                        print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section D done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Section E: Two-logit concat (ICA-128 + logit-PCA + logit-STD)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== E: Two-logit (PCA + STD) concat ===", flush=True)
t0 = time.time()

# Best logit-PCA: n=8, alpha=0.30
# Best logit-STD: from section D results
logit_pca8  = logit_pcas[8]   # PCA(8)-normalized

for n_std in [8, 12, 16]:
    pca_std_l = PCA(n_components=n_std, random_state=42)
    logit_std8 = normalize(pca_std_l.fit_transform(logit_scaled), norm='l2')

    for al_p, al_s in [(0.20, 0.10), (0.25, 0.10), (0.30, 0.05), (0.25, 0.05)]:
        al_total = al_p + al_s
        if al_total > 0.6: continue
        al_main  = 1.0 - al_total
        ef = np.concatenate([
            al_main * emb_ica128,
            al_p    * logit_pca8,
            al_s    * logit_std8
        ], axis=1)
        ef = normalize(ef, norm='l2')
        SIM_2l = ef @ ef.T

        for k in [5, 6, 7]:
            p_2l = wknn_single(SIM_2l, k=k)
            triple = 0.5*p_2l + 0.3*p_pca3 + 0.2*p_std5
            for ww in [0.03, 0.04, 0.05]:
                for rm in [0.26, 0.28, 0.30]:
                    alp_i = int(al_p*100); als_i = int(al_s*100)
                    mname = f"ica128logit2x{n_std}_ap{alp_i}_as{als_i}_k{k}_ww{int(ww*100)}_rm{int(rm*100)}"
                    if mname in tried: continue
                    final = run_chain(triple, ww=ww, rm=rm)
                    auc = macro_auc(final)
                    diff = save_result(mname, auc, {"type": "two_logit", "n_std": n_std})
                    if diff > -0.0003:
                        marker = " ***NEW BEST***" if diff > 1e-7 else ""
                        print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section E done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Section F: wkt4/wkt5 with logit-fused as additional component
# ════════════════════════════════════════════════════════════════════════════
print("\n=== F: wkt4/wkt5 with logit-fused component ===", flush=True)
t0 = time.time()

# Build best logit-fused KNN predictions for multiple k values
SIM_best_fused = make_fused_sim(8, 0.30)
p_logfuse = {k: wknn_single(SIM_best_fused, k=k) for k in [4, 5, 6, 7, 8]}

# wkt4: ICA128-logit-fused + original ICA + PCA + STD
# wa=weight of logit-fused, wb=ICA100, wc=PCA, wd=STD
for wlf, wi, wp_w, ws_w in [
    (3, 3, 2, 2), (4, 2, 2, 2), (4, 3, 2, 1), (5, 2, 2, 1), (3, 4, 2, 1),
    (3, 2, 3, 2), (4, 3, 1, 2), (5, 3, 1, 1), (3, 3, 3, 1),
]:
    total = wlf + wi + wp_w + ws_w
    wa_lf = wlf/total; wa_ic = wi/total; wa_pc = wp_w/total; wa_st = ws_w/total
    for k_lf, k_ic in [(6, 6), (6, 5), (7, 6), (5, 6)]:
        p_ic = wknn_single(SIM_ICA, k=k_ic) if k_ic != 6 else p_ica6
        triple4 = wa_lf*p_logfuse[k_lf] + wa_ic*p_ic + wa_pc*p_pca3 + wa_st*p_std5
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.26, 0.28, 0.30]:
                mname = f"wkt4_lf{wlf}_ic{wi}_pc{wp_w}_st{ws_w}_klf{k_lf}_kic{k_ic}_ww{int(ww*100)}_rm{int(rm*100)}"
                if mname in tried: continue
                final = run_chain(triple4, ww=ww, rm=rm)
                auc = macro_auc(final)
                diff = save_result(mname, auc, {"type": "wkt4_logitfused"})
                if diff > -0.0003:
                    marker = " ***NEW BEST***" if diff > 1e-7 else ""
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

# wkt5: add NMF as 5th component
SIM_NMF = emb_nmf @ emb_nmf.T if hasattr(ep, 'emb_win_nmf_norm') else None
if ep.get('emb_win_nmf_norm') is not None:
    SIM_NMF = ep['emb_win_nmf_norm'] @ ep['emb_win_nmf_norm'].T
    p_nmf5  = wknn_single(SIM_NMF, k=5)
    for wlf, wi, wp_w, ws_w, wn in [(3,3,2,1,1), (3,2,2,2,1), (4,2,2,1,1)]:
        total = wlf+wi+wp_w+ws_w+wn
        triple5 = (wlf*p_logfuse[6] + wi*p_ica6 + wp_w*p_pca3
                   + ws_w*p_std5 + wn*p_nmf5) / total
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.26, 0.28, 0.30]:
                mname = f"wkt5_lf{wlf}_{wi}{wp_w}{ws_w}{wn}_ww{int(ww*100)}_rm{int(rm*100)}"
                if mname in tried: continue
                final = run_chain(triple5, ww=ww, rm=rm)
                auc = macro_auc(final)
                diff = save_result(mname, auc, {"type": "wkt5_logitfused"})
                if diff > -0.0003:
                    marker = " ***NEW BEST***" if diff > 1e-7 else ""
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section F done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Section G: Logit-fused ICA-128 with multiple seeds (ensemble logit fusion)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== G: Multi-seed logit-fused ICA-128 ===", flush=True)
t0 = time.time()

SEEDS_G = [0, 1, 7, 17, 31, 42]
SIM_logfuse_seeds = {42: SIM_best_fused}  # seed=42 already computed

for seed in SEEDS_G:
    if seed == 42: continue
    ica_s = FastICA(n_components=128, random_state=seed, max_iter=500, tol=0.01)
    emb_s = normalize(ica_s.fit_transform(raw_pca256), norm='l2')
    # Fuse with logit PCA-8 at alpha=0.30 (batch180 best params)
    ef_s = np.concatenate([(1-0.30)*emb_s, 0.30*logit_pcas[8]], axis=1)
    ef_s = normalize(ef_s, norm='l2')
    SIM_logfuse_seeds[seed] = ef_s @ ef_s.T
    print(f"  Built logit-fused seed={seed}", flush=True)

# Average SIM from N seeds (multi-seed ensemble)
for n_seeds in [2, 3, 4, 6]:
    seed_list = SEEDS_G[:n_seeds]
    SIM_ms_avg = np.mean([SIM_logfuse_seeds[s] for s in seed_list], axis=0)
    np.fill_diagonal(SIM_ms_avg, 0.0)
    for k in [5, 6, 7]:
        p_ms = wknn_single(SIM_ms_avg, k=k)
        for wa, wb, wc in [(0.5, 0.3, 0.2), (0.55, 0.25, 0.20)]:
            triple = wa*p_ms + wb*p_pca3 + wc*p_std5
            for ww in [0.03, 0.04, 0.05]:
                for rm in [0.26, 0.28, 0.30]:
                    wa_i = int(wa*10)
                    mname = f"ica128logitfused_ms{n_seeds}_k{k}_wa{wa_i}_ww{int(ww*100)}_rm{int(rm*100)}"
                    if mname in tried: continue
                    final = run_chain(triple, ww=ww, rm=rm)
                    auc = macro_auc(final)
                    diff = save_result(mname, auc, {"type": "ms_logitfused", "n_seeds": n_seeds})
                    if diff > -0.0003:
                        marker = " ***NEW BEST***" if diff > 1e-7 else ""
                        print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section G done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# Section H: Logit direct concat (no PCA — use raw logit as similarity)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== H: Raw logit direct concat (no dim reduction) ===", flush=True)
t0 = time.time()

# Normalize raw logit directly (no PCA)
logit_raw_norm = normalize(logit_scaled.astype(np.float32), norm='l2')

for alpha in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    ef = np.concatenate([(1-alpha)*emb_ica128, alpha*logit_raw_norm], axis=1)
    ef = normalize(ef, norm='l2')
    SIM_raw_l = ef @ ef.T

    for k in [5, 6, 7]:
        p_raw_l = wknn_single(SIM_raw_l, k=k)
        triple = 0.5*p_raw_l + 0.3*p_pca3 + 0.2*p_std5
        for ww in [0.03, 0.04, 0.05]:
            for rm in [0.26, 0.28, 0.30]:
                mname = f"ica128logitRAW_al{int(alpha*100):02d}_k{k}_ww{int(ww*100)}_rm{int(rm*100)}"
                if mname in tried: continue
                final = run_chain(triple, ww=ww, rm=rm)
                auc = macro_auc(final)
                diff = save_result(mname, auc, {"type": "logit_raw", "alpha": alpha})
                if diff > -0.0003:
                    marker = " ***NEW BEST***" if diff > 1e-7 else ""
                    print(f"  {mname}: {auc:.6f} ({diff:+.6f}){marker}", flush=True)

print(f"  Section H done in {time.time()-t0:.1f}s. Best: {best_loo:.6f}", flush=True)

# ─── Final summary ────────────────────────────────────────────────────────────
print(f"\n[batch{BATCH}] Done. Final best: {res['best']['method']} LOO={res['best']['loo_auc']:.6f}",
      flush=True)
exps_this = [e for e in res["experiments"] if e.get("batch") == BATCH]
print(f"[batch{BATCH}] Experiments this batch: {len(exps_this)}", flush=True)
if exps_this:
    top = sorted(exps_this, key=lambda x: x["loo_auc"], reverse=True)[:10]
    print(f"[batch{BATCH}] Top-10 this batch:", flush=True)
    for e in top:
        print(f"  {e['method']}: {e['loo_auc']:.6f}", flush=True)
