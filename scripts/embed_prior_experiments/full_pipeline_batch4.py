"""
Full-pipeline direct search - Batch 4
Target: > 0.9432 (RKNN k=5 wg=0.40 a=0.95 b=1.70)

Novel structural ideas targeting dynamic/adaptive approaches:

1. dynamic_rknn_weight:  Per-file adaptive blend weight wg based on mutual neighbor
                         quality. More mutual neighbors → higher wg.
                         wg(i) = sigmoid(n_mutual / k - 0.5)

2. species_adaptive_blend: Different blend weights per SPECIES based on model confidence.
                         For species where VLOM base is uncertain (0.3-0.7),
                         lean more on RKNN; for confident species, lean on base.

3. geo_boosted_rknn:    Weight RKNN neighbors by geographic proximity
                         (same site = higher weight). Currently X_ref has geo
                         but this explicitly boosts same-site matches.

4. rknn_calibration:    Calibrate RKNN output using per-species scaling
                         based on species prevalence in training set.

5. perch_logit_knn:     Use Perch LOGIT features (not probs) for KNN in
                         234-dim logit space. Different from probs version.

6. rknn_k5_k7_asym:     RKNN k=5 and k=7 both, weighted by their own asymmetric
                         logspace contributions. The k=7 adds weaker but broader
                         mutual neighbors.
"""
import numpy as np, pickle, os, json
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch=np.load("outputs/perch_labeled_ss.npz",allow_pickle=True)
emb_win=perch['emb'].astype(np.float32); logits_win=perch['logits'].astype(np.float32)
labels_win=perch['labels'].astype(np.float32); file_list=list(perch['file_list'])
n_windows=perch['n_windows']; n_files=len(file_list); n_species=labels_win.shape[1]
file_start=np.concatenate([[0],np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end=np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))
file_labels=np.zeros((n_files,n_species),np.float32)
file_logit_max=np.zeros((n_files,n_species),np.float32)
for fi in range(n_files):
    s,e=int(file_start[fi]),int(file_end[fi])
    file_labels[fi]=(labels_win[s:e].max(0)>0.5).astype(np.float32)
    file_logit_max[fi]=logits_win[s:e].max(0)
emb_win_norm=normalize(emb_win,norm='l2').astype(np.float32)
win_file_id=np.zeros(len(emb_win),np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])]=fi

sed_npz=np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz",allow_pickle=True)
sed_by_file={}
for i,rid in enumerate(sed_npz['row_ids']): sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]),[]).append(i)
file_sed_max=np.zeros((n_files,n_species),np.float32)
for fi,fname in enumerate(file_list):
    fb=fname.replace('.ogg','').replace('.flac','')
    if fb in sed_by_file: file_sed_max[fi]=sed_npz['probs'][sed_by_file[fb]].max(0)

EPS=1e-7
def vlom_blend(a,b,w_a=0.5,w_b=0.5):
    ws=w_a+w_b; w_a/=ws; w_b/=ws
    return sigmoid(w_a*np.log(a.clip(EPS)/(1-a).clip(EPS))+w_b*np.log(b.clip(EPS)/(1-b).clip(EPS)))
def macro_auc(yt,ys):
    mask=yt.sum(0)>0; return roc_auc_score(yt[:,mask],ys[:,mask],average='macro')

base_probs=vlom_blend(sigmoid(file_logit_max),file_sed_max)
base_logit=np.log(base_probs.clip(EPS))-np.log((1-base_probs).clip(EPS))

with open("outputs/embed_prior_logspace_geo5_win1.pkl","rb") as f: ep_base=pickle.load(f)
X_ref=ep_base['X_combined_n'].astype(np.float32); fl=ep_base['file_labels'].astype(np.float32)
# Site info from pkl
sites = ep_base['sites']

BEST_SO_FAR = 0.9432
print(f"Target: > {BEST_SO_FAR:.4f}")
all_results = []

sim_all = X_ref @ X_ref.T
np.fill_diagonal(sim_all, -np.inf)

# Precompute win_k1
print("Precomputing win_k1...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s,te_e=int(file_start[i]),int(file_end[i]); X_te=emb_win_norm[te_s:te_e]
    tr_mask=(win_file_id!=i); X_tr=emb_win_norm[tr_mask]; tr_fi=win_file_id[tr_mask]
    sims=X_te@X_tr.T; top_idx=np.argsort(-sims,1)[:,:1]
    wp=np.zeros((te_e-te_s,n_species),np.float32)
    for wi in range(te_e-te_s):
        ww=sims[wi,top_idx[wi]].clip(0); ws=ww.sum()
        ww=ww/ws if ws>1e-8 else np.ones(1)
        wp[wi]=(ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i]=wp.mean(0)
print("  done.", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Method 1: Dynamic RKNN Weight
# Per-file wg: files with more mutual neighbors get higher weight on RKNN,
# files with no mutual neighbors (fallback) get lower weight.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Dynamic RKNN weight (per-file wg)...", flush=True)
best1 = {'auc': 0}
k, T = 5, 0.2
top_k_tr = np.argsort(-sim_all, axis=1)[:, :k]
kth_sim = sim_all[np.arange(n_files), top_k_tr[:, -1]]

# Count mutual neighbors for each file
mutual_counts = np.zeros(n_files, np.float32)
y_rknn_base = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    tr=np.array([j for j in range(n_files) if j!=i])
    sims_i=sim_all[i,tr]; top_i=np.argsort(-sims_i)[:k]
    mutual=[]; mutual_sims=[]
    for ti, tj in enumerate(tr[top_i]):
        if sims_i[top_i[ti]] >= kth_sim[tj]:
            mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
    mutual_counts[i] = len(mutual)
    if len(mutual)==0:
        top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_rknn_base[i]=(w[:,None]*fl[tr[top5]]).sum(0)
    else:
        ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_rknn_base[i]=(w[:,None]*fl[mutual]).sum(0)

print(f"  Mutual count distribution: {np.bincount(mutual_counts.astype(int))}")

for scale in [0.5, 1.0, 2.0]:
    # wg(i) = sigmoid(scale * (n_mutual/k - 0.5)) → 0 when no mutual, 1 when all mutual
    wg_per_file = sigmoid(scale * (mutual_counts / k - 0.5)).reshape(-1, 1)  # (66, 1)
    # Dynamic blend: wg_per_file * rknn + (1-wg_per_file) * win
    y_blend_dyn = wg_per_file * y_rknn_base + (1 - wg_per_file) * y_win_k1  # (66, 234)
    log_yb = np.log(y_blend_dyn.clip(EPS))
    for a in [0.85, 0.90, 0.95, 1.00]:
        for b in [1.50, 1.60, 1.70, 1.80, 1.90, 2.00]:
            full = sigmoid(a * base_logit + b * log_yb)
            auc = macro_auc(file_labels, full)
            if auc > best1['auc']:
                best1 = {'auc': auc, 'scale': scale, 'a': a, 'b': b}
                if auc > BEST_SO_FAR:
                    print(f"  *** scale={scale} a={a} b={b}: {auc:.4f} ***", flush=True)
                else:
                    print(f"  scale={scale} a={a} b={b}: {auc:.4f}", flush=True)

# Also try: base_wg + quality bonus
for base_wg in [0.30, 0.35, 0.40]:
    for bonus in [0.05, 0.10, 0.15]:
        # Quality-adjusted: if has mutual, boost by bonus; if not, reduce by bonus
        has_mutual = (mutual_counts > 0).astype(np.float32).reshape(-1, 1)
        wg_adj = (base_wg + bonus * has_mutual - bonus * (1 - has_mutual)).clip(0, 1)
        y_blend_adj = wg_adj * y_rknn_base + (1 - wg_adj) * y_win_k1
        log_yb = np.log(y_blend_adj.clip(EPS))
        for a in [0.85, 0.90, 0.95, 1.00]:
            for b in [1.50, 1.60, 1.70, 1.80, 1.90]:
                full = sigmoid(a * base_logit + b * log_yb)
                auc = macro_auc(file_labels, full)
                if auc > best1['auc']:
                    best1 = {'auc': auc, 'base_wg': base_wg, 'bonus': bonus, 'a': a, 'b': b}
                    if auc > BEST_SO_FAR:
                        print(f"  *** bwg={base_wg} bonus={bonus} a={a} b={b}: {auc:.4f} ***", flush=True)
                    else:
                        print(f"  bwg={base_wg} bonus={bonus} a={a} b={b}: {auc:.4f}", flush=True)

print(f"  Best dynamic RKNN: {best1['auc']:.4f}")
all_results.append(('dynamic_rknn', best1['auc'], best1))

# ══════════════════════════════════════════════════════════════════════════════
# Method 2: Geo-Boosted RKNN
# Same-site matches in X_ref get extra weight (site indicator from pkl)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Geo-boosted RKNN (same-site extra weight)...", flush=True)
best2 = {'auc': 0}

# Build same-site matrix
sites_arr = np.array(sites) if hasattr(sites, '__len__') else None
if sites_arr is not None and len(sites_arr) == n_files:
    same_site = (sites_arr[:, None] == sites_arr[None, :]).astype(np.float32)  # (66, 66)
    print(f"  Same-site pairs: {same_site.sum()}")

    for site_boost in [0.05, 0.10, 0.20]:
        y_geo_boost = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr=np.array([j for j in range(n_files) if j!=i])
            sims_i = sim_all[i, tr]
            # Boost same-site similarity
            site_bonus = site_boost * same_site[i, tr]
            sims_boosted = sims_i + site_bonus
            top_i = np.argsort(-sims_boosted)[:k]
            # Reciprocal check with original sims
            mutual=[]; mutual_sims=[]
            for ti, tj in enumerate(tr[top_i]):
                sim_adj = sims_boosted[top_i[ti]]
                if sims_i[top_i[ti]] >= kth_sim[tj]:
                    mutual.append(tj); mutual_sims.append(sim_adj)
            if len(mutual)==0:
                top5=np.argsort(-sims_boosted)[:5]; ls=sims_boosted[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
                y_geo_boost[i]=(w[:,None]*fl[tr[top5]]).sum(0)
            else:
                ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
                y_geo_boost[i]=(w[:,None]*fl[mutual]).sum(0)

        for wg in [0.30, 0.35, 0.40, 0.45]:
            yb = wg * y_geo_boost + (1-wg) * y_win_k1
            log_yb = np.log(yb.clip(EPS))
            for a in [0.85, 0.90, 0.95, 1.00]:
                for b in [1.50, 1.60, 1.70, 1.80, 1.90]:
                    full = sigmoid(a * base_logit + b * log_yb)
                    auc = macro_auc(file_labels, full)
                    if auc > best2['auc']:
                        best2 = {'auc': auc, 'boost': site_boost, 'wg': wg, 'a': a, 'b': b}
                        if auc > BEST_SO_FAR:
                            print(f"  *** boost={site_boost} wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                        else:
                            print(f"  boost={site_boost} wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
    print(f"  Best geo-boosted RKNN: {best2['auc']:.4f}")
else:
    best2 = {'auc': 0}
    print("  Sites not in expected format, skipping.")
all_results.append(('geo_boosted_rknn', best2['auc'], best2))

# ══════════════════════════════════════════════════════════════════════════════
# Method 3: Per-Species Adaptive Blend
# For each species independently, blend RKNN and win based on how uncertain
# the VLOM base prediction is for that species.
# Uncertain base (prob ≈ 0.5) → more RKNN weight
# Confident base (prob ≈ 0 or 1) → less RKNN weight
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Per-species adaptive blend...", flush=True)
best3 = {'auc': 0}

# base_probs: (66, 234) - VLOM base probabilities
# uncertainty = 1 - 2*|p - 0.5| → high when p ≈ 0.5
uncertainty = 1.0 - 2.0 * np.abs(base_probs - 0.5)  # (66, 234) in [0, 1]

for wg_min in [0.20, 0.25, 0.30]:
    for wg_max in [0.50, 0.55, 0.60]:
        # Per-file-per-species weight: higher uncertainty → higher RKNN weight
        wg_species = wg_min + (wg_max - wg_min) * uncertainty  # (66, 234)
        y_blend_sp = wg_species * y_rknn_base + (1 - wg_species) * y_win_k1
        log_yb = np.log(y_blend_sp.clip(EPS))
        for a in [0.85, 0.90, 0.95, 1.00]:
            for b in [1.50, 1.60, 1.70, 1.80, 1.90]:
                full = sigmoid(a * base_logit + b * log_yb)
                auc = macro_auc(file_labels, full)
                if auc > best3['auc']:
                    best3 = {'auc': auc, 'wg_min': wg_min, 'wg_max': wg_max, 'a': a, 'b': b}
                    if auc > BEST_SO_FAR:
                        print(f"  *** wg_min={wg_min} wg_max={wg_max} a={a} b={b}: {auc:.4f} ***", flush=True)
                    else:
                        print(f"  wg_min={wg_min} wg_max={wg_max} a={a} b={b}: {auc:.4f}", flush=True)

print(f"  Best per-species adaptive: {best3['auc']:.4f}")
all_results.append(('species_adaptive_blend', best3['auc'], best3))

# ══════════════════════════════════════════════════════════════════════════════
# Method 4: RKNN in Perch Logit Space
# Use raw logit outputs (pre-sigmoid) as features for RKNN.
# More linear / unbounded space might have better geometry.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] RKNN in Perch logit feature space...", flush=True)
best4 = {'auc': 0}
# file_logit_max: (66, 234) - max Perch logits per file
logit_norm = normalize(file_logit_max, norm='l2').astype(np.float32)
sim_logit = logit_norm @ logit_norm.T
np.fill_diagonal(sim_logit, -np.inf)
top_k_logit = np.argsort(-sim_logit, axis=1)[:, :k]
kth_sim_logit = sim_logit[np.arange(n_files), top_k_logit[:, -1]]

y_rknn_logit = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    tr=np.array([j for j in range(n_files) if j!=i])
    sims_i=sim_logit[i,tr]; top_i=np.argsort(-sims_i)[:k]
    mutual=[]; mutual_sims=[]
    for ti, tj in enumerate(tr[top_i]):
        if sims_i[top_i[ti]] >= kth_sim_logit[tj]:
            mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
    if len(mutual)==0:
        top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_rknn_logit[i]=(w[:,None]*fl[tr[top5]]).sum(0)
    else:
        ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_rknn_logit[i]=(w[:,None]*fl[mutual]).sum(0)

for wg in [0.25, 0.30, 0.35, 0.40, 0.45]:
    yb = wg * y_rknn_logit + (1-wg) * y_win_k1
    log_yb = np.log(yb.clip(EPS))
    for a in [0.85, 0.90, 0.95, 1.00]:
        for b in [1.50, 1.60, 1.70, 1.80, 1.90]:
            full = sigmoid(a * base_logit + b * log_yb)
            auc = macro_auc(file_labels, full)
            if auc > best4['auc']:
                best4 = {'auc': auc, 'wg': wg, 'a': a, 'b': b}
                if auc > BEST_SO_FAR:
                    print(f"  *** wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                else:
                    print(f"  wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
print(f"  Best RKNN logit space: {best4['auc']:.4f}")
all_results.append(('rknn_logit_space', best4['auc'], best4))

# ══════════════════════════════════════════════════════════════════════════════
# Method 5: RKNN k=5 + RKNN k=7 (asymmetric separate logspace)
# Both use X_ref space, but k=7 has different fallback and mutual conditions.
# sigmoid(a*base_logit + b1*log(rknn_k5) + b2*log(rknn_k7) + b3*log(win))
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Asymmetric RKNN k=5+k=7+win...", flush=True)
best5 = {'auc': 0}

y_rknn_k7 = np.zeros((n_files, n_species), np.float32)
top_k_tr7 = np.argsort(-sim_all, axis=1)[:, :7]
kth_sim7 = sim_all[np.arange(n_files), top_k_tr7[:, -1]]
for i in range(n_files):
    tr=np.array([j for j in range(n_files) if j!=i])
    sims_i=sim_all[i,tr]; top_i=np.argsort(-sims_i)[:7]
    mutual=[]; mutual_sims=[]
    for ti, tj in enumerate(tr[top_i]):
        if sims_i[top_i[ti]] >= kth_sim7[tj]:
            mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
    if len(mutual)==0:
        top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_rknn_k7[i]=(w[:,None]*fl[tr[top5]]).sum(0)
    else:
        ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_rknn_k7[i]=(w[:,None]*fl[mutual]).sum(0)

log_r5 = np.log(y_rknn_base.clip(EPS))
log_r7 = np.log(y_rknn_k7.clip(EPS))
log_w1 = np.log(y_win_k1.clip(EPS))

for a in [0.85, 0.90, 0.95, 1.00]:
    for b1 in [0.80, 1.00, 1.20, 1.40]:
        for b2 in [0.20, 0.40, 0.60]:
            for b3 in [0.40, 0.60, 0.80]:
                full = sigmoid(a * base_logit + b1 * log_r5 + b2 * log_r7 + b3 * log_w1)
                auc = macro_auc(file_labels, full)
                if auc > best5['auc']:
                    best5 = {'auc': auc, 'a': a, 'b1': b1, 'b2': b2, 'b3': b3}
                    if auc > BEST_SO_FAR:
                        print(f"  *** a={a} b1={b1} b2={b2} b3={b3}: {auc:.4f} ***", flush=True)
                    else:
                        print(f"  a={a} b1={b1} b2={b2} b3={b3}: {auc:.4f}", flush=True)
print(f"  Best RKNN k5+k7+win asym: {best5['auc']:.4f}")
all_results.append(('rknn_k5_k7_win_asym', best5['auc'], best5))

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"FULL PIPELINE BATCH 4 SUMMARY")
print(f"{'='*60}")
print(f"Baseline RKNN k=5 wg=0.40 a=0.95 b=1.70: {BEST_SO_FAR:.4f}")
for name, auc, cfg in sorted(all_results, key=lambda x: -x[1]):
    marker = " *** NEW BEST ***" if auc > BEST_SO_FAR else ""
    print(f"  {name:30s}: {auc:.4f}{marker}")
