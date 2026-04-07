"""
batch175 — Resume under-explored methods + novel approaches
===============================================================================
Current best: wkt3_ki6_kp3_ks5_ww4_rm28 LOO=0.995986
Batch174 was killed after br_pca32 only. Continue:
  A: Bayesian Ridge PCA-{64,96,128,192,256} + blend search
  B: Nystroem+LogReg rbf/laplacian sweep (not started in batch174)
  C: Attention KNN deeper sweep (only 2 variants from earlier batches)
  D: Novel: Covariance Pooling KNN (2nd-order file representation)
  E: Novel: Random Subspace KNN ensemble (50 random projections to 128-dim)
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
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
emb_std = ep["emb_win_std_norm"]
emb_nmf = ep["emb_win_nmf_norm"]
logit_sig  = ep["logit_sig_win"]
labels_win = ep["labels_win"]
win_file_id= ep["win_file_id"]

# Raw data
raw_emb    = DATA["emb"].astype(np.float32)
raw_labels = DATA["labels"].astype(np.float32)
file_list  = DATA["file_list"]
filenames  = DATA["filenames"]
fname2idx  = {fn: i for i, fn in enumerate(file_list)}
file_ids_raw = np.array([fname2idx[fn] for fn in filenames], dtype=np.int32)
unique_files = np.arange(n_files)

with open(RESULTS_PATH, 'rb') as _f:
    res = json.loads(_f.read().decode('utf-8', errors='replace'))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch175] Best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)
print(f"[batch175] Total tried: {len(tried)}", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def macro_auc(s, fl=file_labels):
    aucs = []
    for si in range(n_species):
        y = fl[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            try: aucs.append(roc_auc_score(y, s[:, si]))
            except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(mname, score, batch_n=175, cfg=None):
    global best_loo
    if mname in tried: return score - best_loo
    res["experiments"].append({"method": mname, "loo_auc": float(score), "config": cfg or {}, "batch": batch_n})
    tried.add(mname)
    if score > best_loo + 1e-7:
        best_loo = score
        res["best"] = {"method": mname, "loo_auc": float(score)}
        with open(MODEL_PATH, "rb") as f:
            ep_up = pickle.load(f)
        ep_up["method"] = mname; ep_up["loo_auc"] = float(score)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(ep_up, f)
        print(f"  *** NEW BEST: {mname} = {score:.6f} ***", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    return score - best_loo

# ── Fixed baseline ─────────────────────────────────────────────────────────────
fl_hard = file_labels.astype(np.float32)
count_i = fl_hard.sum(0) + EPS
COOC = (fl_hard.T @ fl_hard) / count_i[:, None]; np.fill_diagonal(COOC, 0)
raw_idf = np.clip(np.log(float(n_files)/(count_i+1.0-EPS)), 0, None)
IDF075 = raw_idf**0.75; IDF075 /= (IDF075.mean()+EPS)

def soft_cooc(scores, center=0.55, slope=41.0, alpha=0.200, idf_w=None):
    out = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        gate = 1.0/(1.0+np.exp(np.clip(-slope*(s-center),-88,88)))
        sg = s*gate*(idf_w if idf_w is not None else 1.0)
        if np.abs(sg).sum() < EPS: out[fi]=s; continue
        c = COOC.T@sg; mc = np.abs(c).max()
        if mc > EPS: c /= mc
        out[fi] = (1-alpha)*s + alpha*np.clip(c,0,None)
    return out

def apply_3way(s, alpha=0.200):
    sp = np.clip(s,0,1)**2
    sc = soft_cooc(sp, alpha=alpha, idf_w=IDF075)
    idf_s = 0.45*s + 0.55*sc
    r1 = soft_cooc(s, center=0.54, slope=41.0, alpha=0.110)
    tr = soft_cooc(r1, center=0.53, slope=37.0, alpha=0.030)
    return 0.875*idf_s + 0.125*tr

def make_rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(float)

fi_wins_list    = [np.where(win_file_id==fi)[0] for fi in range(n_files)]
other_wins_list = [np.where(win_file_id!=fi)[0] for fi in range(n_files)]
c3_ref = apply_3way(double_best, alpha=0.19)
i3_ref = apply_3way(ica_ens_alt, alpha=0.31)
s3_ref = apply_3way(std_ens_ref,  alpha=0.33)
rank_norm = (0.56*make_rank(apply_3way(double_best,0.23)) + 0.44*make_rank(apply_3way(ica_ens_alt,0.40))) / n_files

# Best triple from batch173 (ki=6, kp=3, ks=5)
print("Pre-computing best wkt3 triple...", flush=True)
SIM_ICA = emb_ica@emb_ica.T; SIM_PCA = emb_pca@emb_pca.T; SIM_STD = emb_std@emb_std.T

def wknn_s(SIM, k=7):
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files,n_species),dtype=np.float32)
    for fi in range(n_files):
        fw=fi_wins_list[fi]; ow=other_wins_list[fi]
        if len(fw)==0: continue
        ke=min(k,len(ow)); wp=np.zeros((len(fw),n_species),dtype=np.float32)
        for wi,wkk in enumerate(fw):
            sims=SIM[wkk,ow]; tl=np.argpartition(-sims,ke-1)[:ke]; tw=ow[tl]
            w=np.clip(sims[tl],0,None); ws=w.sum()
            w=w/ws if ws>EPS else np.ones(ke)/ke
            wp[wi]=(w[:,None]*signal[tw]).sum(0)
        preds[fi]=wp.mean(0)
    return preds

p6=wknn_s(SIM_ICA,6); p3=wknn_s(SIM_PCA,3); p5=wknn_s(SIM_STD,5)
wknn_best = 0.5*p6+0.3*p3+0.2*p5
chk4_base = 0.74*c3_ref+0.16*i3_ref+0.06*s3_ref+0.04*wknn_best
print(f"  Verify: {macro_auc(0.72*chk4_base+0.28*rank_norm):.6f}", flush=True)

# Scaler for raw embeddings
scaler = StandardScaler(); X_all = scaler.fit_transform(raw_emb)

t0 = time.time()

# =============================================================================
# A: Bayesian Ridge — PCA-64 and above (batch174 only did PCA-32)
# =============================================================================
print("\n=== A: Bayesian Ridge PCA-64..256 ===", flush=True)
best_br = best_loo

for pca_dim in [64, 96, 128, 192, 256]:
    print(f"  BayesianRidge PCA-{pca_dim}...", flush=True)
    pca = PCA(n_components=pca_dim, random_state=42)
    Z = pca.fit_transform(X_all).astype(np.float32)

    file_preds = np.zeros((n_files,n_species),dtype=np.float32)
    for fi in range(n_files):
        tm = file_ids_raw==fi; trm = ~tm
        Z_tr,Z_te = Z[trm],Z[tm]; Y_tr = raw_labels[trm]
        wp = np.zeros((tm.sum(),n_species),dtype=np.float32)
        for si in range(n_species):
            y = Y_tr[:,si]
            if y.sum()==0 or y.sum()==len(y): wp[:,si]=y.mean(); continue
            clf = BayesianRidge(max_iter=300)
            clf.fit(Z_tr,y); wp[:,si]=np.clip(clf.predict(Z_te),0,1)
        file_preds[fi] = wp.mean(0)

    ar = macro_auc(file_preds)
    save_result(f"br_pca{pca_dim}", ar, cfg={"pca_dim":pca_dim})
    print(f"    br_pca{pca_dim}: {ar:.6f}", flush=True)

    for br_w in [0.05,0.08,0.10,0.12,0.15,0.20]:
        blended = (1-br_w)*chk4_base + br_w*file_preds
        for rm in [0.27,0.28,0.29]:
            final = (1-rm)*blended+rm*rank_norm
            ar2 = macro_auc(final)
            mn = f"br_pca{pca_dim}_w{int(br_w*100)}_rm{int(rm*100)}"
            d = save_result(mn, ar2)
            if ar2 > best_br:
                best_br = ar2
                print(f"    [BEST BR] {mn}: {ar2:.6f} (+{d:.6f})", flush=True)

print(f"  BR best: {best_br:.6f}", flush=True)

# =============================================================================
# B: Nystroem+LogReg (not started in batch174)
# =============================================================================
print("\n=== B: Nystroem+LogReg ===", flush=True)
best_nys = best_loo

pca128 = PCA(n_components=128, random_state=42)
Z128 = pca128.fit_transform(X_all).astype(np.float32)

for kernel in ['rbf','laplacian']:
    for gamma in [0.002,0.005,0.01,0.02,0.05,0.1,0.5]:
        for n_comp in [128,256]:
            mn_base = f"nys_{kernel[:3]}_g{int(gamma*1000)}_n{n_comp}"
            if mn_base in tried: continue
            print(f"  Nystroem {kernel} γ={gamma} n={n_comp}...", flush=True)
            try:
                nys = Nystroem(kernel=kernel,gamma=gamma,n_components=n_comp,random_state=42)
                Z_nys = nys.fit_transform(Z128).astype(np.float32)

                fp = np.zeros((n_files,n_species),dtype=np.float32)
                for fi in range(n_files):
                    tm=file_ids_raw==fi; trm=~tm
                    Ztr,Zte=Z_nys[trm],Z_nys[tm]; Ytr=raw_labels[trm]
                    wp=np.zeros((tm.sum(),n_species),dtype=np.float32)
                    for si in range(n_species):
                        y=Ytr[:,si]
                        if y.sum()==0 or y.sum()==len(y): wp[:,si]=y.mean(); continue
                        try:
                            clf=LogisticRegression(C=1.0,max_iter=200,solver='lbfgs')
                            clf.fit(Ztr,(y>0.5).astype(int))
                            wp[:,si]=clf.predict_proba(Zte)[:,1]
                        except: wp[:,si]=y.mean()
                    fp[fi]=wp.mean(0)

                ar = macro_auc(fp); save_result(mn_base,ar,cfg={"kernel":kernel,"gamma":gamma,"n":n_comp})
                print(f"    {mn_base}: {ar:.6f}", flush=True)

                for nw in [0.05,0.08,0.10,0.12,0.15]:
                    bl=(1-nw)*chk4_base+nw*fp
                    final=0.72*bl+0.28*rank_norm
                    ar2=macro_auc(final); mn2=f"nys_{kernel[:3]}_g{int(gamma*1000)}_n{n_comp}_w{int(nw*100)}"
                    d=save_result(mn2,ar2)
                    if ar2>best_nys:
                        best_nys=ar2
                        print(f"    [BEST Nys] {mn2}: {ar2:.6f} (+{d:.6f})", flush=True)
            except Exception as ex:
                print(f"    Nystroem error: {ex}", flush=True)

print(f"  Nystroem best: {best_nys:.6f}", flush=True)

# =============================================================================
# C: Attention KNN — comprehensive sweep (only 2 variants from earlier)
# =============================================================================
print("\n=== C: Attention KNN ===", flush=True)
best_attn = best_loo
SIM_NMF = emb_nmf@emb_nmf.T

def attn_wknn(SIM, k=8, temp=0.10, agg='softmax'):
    signal = labels_win.astype(np.float32)
    preds = np.zeros((n_files,n_species),dtype=np.float32)
    for fi in range(n_files):
        fw=fi_wins_list[fi]; ow=other_wins_list[fi]
        if len(fw)==0: continue
        ke=min(k,len(ow)); wp=np.zeros((len(fw),n_species),dtype=np.float32)
        for wi,wkk in enumerate(fw):
            sims=SIM[wkk,ow]; tl=np.argpartition(-sims,ke-1)[:ke]; tw=ow[tl]
            raw_w=sims[tl]
            if agg=='softmax':
                lw=raw_w/(temp+EPS); lw-=lw.max(); w=np.exp(lw); w/=(w.sum()+EPS)
            elif agg=='sigmoid':
                w=1.0/(1.0+np.exp(-raw_w/temp)); w/=(w.sum()+EPS)
            else:
                w=np.clip(raw_w,0,None); ws=w.sum(); w=w/ws if ws>EPS else np.ones(ke)/ke
            wp[wi]=(w[:,None]*signal[tw]).sum(0)
        preds[fi]=wp.mean(0)
    return preds

for emb_name, SIM in [('ica',SIM_ICA),('pca',SIM_PCA),('std',SIM_STD),('nmf',SIM_NMF)]:
    for agg in ['softmax','linear']:
        for temp in [0.05,0.10,0.15,0.20,0.30,0.50]:
            for k in [4,5,6,7,8,9,10,12]:
                mn_base = f"ak_{emb_name}_{agg[:3]}_T{int(temp*100):03d}_k{k}"
                if mn_base in tried: continue
                p_a = attn_wknn(SIM, k=k, temp=temp, agg=agg)
                for ww in [0.03,0.04,0.05]:
                    for mx in [0.0,0.3,0.5,0.7,1.0]:
                        pm = mx*p_a+(1-mx)*wknn_best
                        chk4 = 0.74*c3_ref+0.16*i3_ref+0.06*s3_ref+ww*pm
                        for rm in [0.27,0.28]:
                            final=(1-rm)*chk4+rm*rank_norm
                            ar=macro_auc(final)
                            mn=f"ak_{emb_name}_{agg[:3]}_T{int(temp*100):03d}_k{k}_ww{int(ww*100)}_mx{int(mx*10)}_rm{int(rm*100)}"
                            d=save_result(mn,ar)
                            if ar>best_attn:
                                best_attn=ar
                                print(f"  [BEST Attn] {mn}: {ar:.6f} (+{d:.6f})", flush=True)

print(f"  Attention KNN best: {best_attn:.6f}", flush=True)

# =============================================================================
# D: Novel — Covariance Pooling KNN (2nd-order file representation)
# =============================================================================
print("\n=== D: Covariance Pooling KNN ===", flush=True)
best_cov = best_loo

# Build file-level covariance features (mean + std + top eigenvalues)
pca32 = PCA(n_components=32, random_state=42)
Z32 = pca32.fit_transform(X_all).astype(np.float32)

def build_file_cov_features(Z, file_ids):
    """Per-file: mean (32) + std (32) + upper triangle cov (528) = 592-dim"""
    feats = []
    for fi in range(n_files):
        mask = file_ids==fi
        z = Z[mask]  # (n_win_in_file, 32)
        mu = z.mean(0)
        sg = z.std(0)+EPS
        cov = np.cov(z.T)  # (32,32)
        if cov.ndim==0: cov=np.array([[float(cov)]])
        upper = cov[np.triu_indices_from(cov,k=0)]  # 32*33/2=528
        feats.append(np.concatenate([mu, sg, upper]))
    return np.array(feats, dtype=np.float32)  # (66, 592)

print("  Building covariance features...", flush=True)
file_cov_feats = build_file_cov_features(Z32, file_ids_raw)

# Normalize cov features
cov_scaler = StandardScaler(); file_cov_norm = cov_scaler.fit_transform(file_cov_feats)

# LOO-KNN on cov features
def cov_knn_loo(k=5):
    preds = np.zeros((n_files,n_species),dtype=np.float32)
    for fi in range(n_files):
        other = [j for j in range(n_files) if j!=fi]
        sims = file_cov_norm[other]@file_cov_norm[fi]
        ke = min(k,len(other))
        tl = np.argpartition(-sims,ke-1)[:ke]
        w = np.clip(sims[tl],0,None); ws=w.sum()
        w = w/ws if ws>EPS else np.ones(ke)/ke
        top_files = np.array(other)[tl]
        preds[fi] = (w[:,None]*file_labels[top_files]).sum(0)
    return preds

for k in [3,4,5,6,7,8,10]:
    p_cov = cov_knn_loo(k=k)
    ar = macro_auc(p_cov)
    mn = f"cov_knn_k{k}"
    save_result(mn,ar,cfg={"k":k,"pca_dim":32})
    # Blend with baseline
    for cw in [0.05,0.08,0.10,0.12,0.15]:
        bl=(1-cw)*chk4_base+cw*p_cov
        final=0.72*bl+0.28*rank_norm
        ar2=macro_auc(final); mn2=f"cov_knn_k{k}_w{int(cw*100)}"
        d=save_result(mn2,ar2)
        if ar2>best_cov:
            best_cov=ar2
            print(f"  [BEST CovKNN] {mn2}: {ar2:.6f} (+{d:.6f})", flush=True)

print(f"  Covariance KNN best: {best_cov:.6f}", flush=True)

# =============================================================================
# E: Novel — Random Subspace KNN ensemble
# =============================================================================
print("\n=== E: Random Subspace KNN ===", flush=True)
best_rs = best_loo

rng = np.random.RandomState(42)
# Normalize raw embeddings
X_norm = raw_emb/(np.linalg.norm(raw_emb,axis=1,keepdims=True)+EPS)

def rs_knn_ensemble(n_proj=30, proj_dim=128, k=5, seed=42):
    """Ensemble of KNN over random projections."""
    rng2 = np.random.RandomState(seed)
    preds_all = []
    for _ in range(n_proj):
        # Random projection
        proj = rng2.randn(1536, proj_dim).astype(np.float32)
        proj /= (np.linalg.norm(proj,axis=0,keepdims=True)+EPS)
        Z_proj = X_norm@proj  # (739, proj_dim)
        # Normalize projected features
        Z_proj_n = Z_proj/(np.linalg.norm(Z_proj,axis=1,keepdims=True)+EPS)
        SIM_proj = Z_proj_n@Z_proj_n.T  # (739,739)

        p = wknn_s(SIM_proj, k=k)
        preds_all.append(p)

    return np.mean(preds_all, axis=0)

for n_proj, proj_dim, k in [(20,128,5),(30,128,6),(20,256,5),(30,256,6),(40,128,5)]:
    mn_base = f"rs_n{n_proj}_d{proj_dim}_k{k}"
    if mn_base in tried: continue
    print(f"  RandomSubspace n={n_proj} d={proj_dim} k={k}...", flush=True)
    p_rs = rs_knn_ensemble(n_proj=n_proj, proj_dim=proj_dim, k=k)
    ar = macro_auc(p_rs); save_result(mn_base,ar)
    for ww in [0.03,0.04,0.05]:
        chk4 = 0.74*c3_ref+0.16*i3_ref+0.06*s3_ref+ww*p_rs
        for rm in [0.27,0.28]:
            final=(1-rm)*chk4+rm*rank_norm
            ar2=macro_auc(final); mn2=f"rs_n{n_proj}_d{proj_dim}_k{k}_ww{int(ww*100)}_rm{int(rm*100)}"
            d=save_result(mn2,ar2)
            if ar2>best_rs:
                best_rs=ar2
                print(f"  [BEST RS] {mn2}: {ar2:.6f} (+{d:.6f})", flush=True)

    # Blend RS with best triple
    for mx in [0.3,0.5,0.7]:
        pm = mx*p_rs+(1-mx)*wknn_best
        chk4 = 0.74*c3_ref+0.16*i3_ref+0.06*s3_ref+0.04*pm
        final=0.72*chk4+0.28*rank_norm
        ar2=macro_auc(final); mn2=f"rs_blend_n{n_proj}_d{proj_dim}_k{k}_mx{int(mx*10)}"
        d=save_result(mn2,ar2)
        if ar2>best_rs:
            best_rs=ar2
            print(f"  [BEST RS-blend] {mn2}: {ar2:.6f} (+{d:.6f})", flush=True)

print(f"  Random Subspace best: {best_rs:.6f}", flush=True)

# =============================================================================
# Summary
# =============================================================================
elapsed = time.time()-t0
print(f"\n[batch175] Done in {elapsed/60:.1f} min", flush=True)
print(f"[batch175] Final best: {best_loo:.6f} ({res['best']['method']})", flush=True)
print(f"[batch175] Baseline: 0.995986  Improvement: {best_loo-0.995986:+.6f}", flush=True)
