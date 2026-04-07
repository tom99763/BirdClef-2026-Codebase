"""
Novel Methods Batch 3: Breakthrough designs beyond parameter sweeps.

Novel mechanisms (not previously explored):
A. Co-occurrence Prior: P(species_j | species_i in knn) from labeled soundscapes
B. Label Propagation: Graph-based label spreading (vs simple KNN voting)
C. Transductive PCA: Include test embeddings in PCA → better geometry
D. Multi-hop KNN: KNN of KNN with exponential decay
E. Platt-calibrated KNN: Fit sigmoid calibration on LOO predictions
F. Reciprocal KNN (Mutual KNN): Only count neighbors that also chose you back
G. SED-guided weighting: Weight windows by their SED confidence

All evaluated via full-pipeline LOO benchmark.
Best v14-ls2: 0.9408 (base+SED VLOM → logspace with geo_k5+win_k1)
"""
import numpy as np, pickle, re, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ─── Load data ────────────────────────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-88,88)))

file_embs      = np.zeros((n_files, emb_win.shape[1]), np.float32)
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)

emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

# Load SED
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_by_file = {}
for i, rid in enumerate(sed_npz['row_ids']):
    sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]), []).append(i)
file_sed_max = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fb = fname.replace('.ogg','').replace('.flac','')
    if fb in sed_by_file:
        file_sed_max[fi] = sed_npz['probs'][sed_by_file[fb]].max(0)

EPS = 1e-7
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:,mask], ys[:,mask], average='macro')
def vlom_blend(a, b, w_a=0.5, w_b=0.5):
    ws = w_a+w_b; w_a/=ws; w_b/=ws
    return sigmoid(w_a*np.log(a.clip(EPS)/(1-a).clip(EPS)) + w_b*np.log(b.clip(EPS)/(1-b).clip(EPS)))

base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))
base_auc   = macro_auc(file_labels, base_probs)

# Load best pkl (logspace_geo5_win1)
with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep = pickle.load(f)
X_ref = ep['X_combined_n'].astype(np.float32)
fl    = ep['file_labels'].astype(np.float32)

# Precompute geo-KNN (k=5) and win-KNN (k=1) LOO predictions
print("Precomputing geo-KNN and win-KNN LOO...", flush=True)
k_geo=5; T_geo=0.2; k_win=1

y_geo = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    tr = np.array([j for j in range(n_files) if j!=i])
    sims = (X_ref[[i]]@X_ref[tr].T).ravel(); top=np.argsort(-sims)[:k_geo]
    ls=sims[top]/T_geo; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
    y_geo[i]=(w[:,None]*fl[tr[top]]).sum(0)

y_win = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s,te_e=int(file_start[i]),int(file_end[i]); X_te=emb_win_norm[te_s:te_e]
    tr_mask=(win_file_id!=i); X_tr=emb_win_norm[tr_mask]; tr_fi=win_file_id[tr_mask]
    sims=X_te@X_tr.T; top_idx=np.argsort(-sims,1)[:,:k_win]
    wp=np.zeros((te_e-te_s,n_species),np.float32)
    for wi in range(te_e-te_s):
        ww=sims[wi,top_idx[wi]].clip(0); ws=ww.sum(); ww=ww/ws if ws>1e-8 else np.ones(k_win)/k_win
        wp[wi]=(ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win[i]=wp.mean(0)

y_blend50 = 0.5*y_geo + 0.5*y_win
# Best LS2 baseline
ls2_best = macro_auc(file_labels, sigmoid(0.90*base_logit + 1.55*np.log(y_blend50.clip(EPS))))
print(f"  LS2 baseline: {ls2_best:.4f}")

results = []

# ─── A. Co-occurrence Prior ────────────────────────────────────────────────────
# Build co-occurrence: C[i,j] = P(species_j in file | species_i in file)
# Then: y_cooc[file] = sum over detected species i: C[i,:] * y_knn[i]
# Insight: if KNN says species A is present, also boost species that co-occur with A
print("\n[A] Co-occurrence Prior...")
label_counts = file_labels.sum(0)  # per-species count
# Pairwise co-occurrence matrix (soft, from probabilities)
# C[i,j] = correlation between species i and j presence across files
C = (file_labels.T @ file_labels) / (n_files + 1e-6)  # (n_species, n_species)
# Normalize: P(j | i detected)
row_sum = file_labels.sum(0, keepdims=True).T + 1e-6
C_cond = C / row_sum  # C_cond[i,j] = P(j in file | i in file)
np.fill_diagonal(C_cond, 0)  # remove self

# Apply: given y_knn predictions, propagate through co-occurrence
# y_cooc = y_knn + alpha * (y_knn @ C_cond)
for alpha in [0.10, 0.20, 0.30, 0.50]:
    for src, y_knn, name in [(y_blend50, y_blend50, 'blend50'), (y_geo, y_geo, 'geo5')]:
        y_cooc = y_knn + alpha * (y_knn @ C_cond)
        y_cooc = y_cooc.clip(EPS, 1-EPS)
        for a, b in [(0.90, 1.55), (0.85, 1.50)]:
            auc = macro_auc(file_labels, sigmoid(a*base_logit + b*np.log(y_cooc.clip(EPS))))
            if auc > ls2_best:
                print(f"  BEAT: cooc_{name}_a{alpha:.2f}_ls{a:.2f}_{b:.2f}: {auc:.4f} (+{auc-ls2_best:.4f})")
                results.append(('cooc', f'cooc_{name}_a{alpha}', auc, {'alpha':alpha,'src':name,'a':a,'b':b}))

# ─── B. Label Propagation ────────────────────────────────────────────────────
# Build full similarity graph of 66 training files
# Propagate: F = alpha * S * F + (1-alpha) * Y
# where S = row-normalized similarity matrix, Y = initial KNN predictions
# This allows indirect connections: if A similar to B and B has species X, A gets partial X credit
print("\n[B] Label Propagation (graph spreading)...")
# Build normalized affinity matrix W from X_ref (66x66)
W_full = X_ref @ X_ref.T  # cosine similarity
np.fill_diagonal(W_full, 0)
W_full = np.maximum(W_full, 0)  # only positive similarities
# Row-normalize
D = W_full.sum(1, keepdims=True) + 1e-8
S = W_full / D  # row-normalized

for alpha_lp in [0.3, 0.5, 0.7, 0.8]:
    for n_iter in [5, 10, 20]:
        # LOO version: for each test file i, propagate on the 65-file training graph
        y_lp = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j!=i])
            # Sub-graph of 65 files
            W_sub = X_ref[tr] @ X_ref[tr].T; np.fill_diagonal(W_sub, 0)
            W_sub = np.maximum(W_sub, 0)
            D_sub = W_sub.sum(1, keepdims=True)+1e-8; S_sub = W_sub/D_sub
            # Test similarity to train
            sim_i = (X_ref[[i]] @ X_ref[tr].T).ravel()
            sim_i = np.maximum(sim_i, 0); sim_i /= sim_i.sum()+1e-8
            # Initialize F = file_labels for training files
            F = fl[tr].copy()
            Y0 = fl[tr].copy()
            for _ in range(n_iter):
                F = alpha_lp * (S_sub @ F) + (1-alpha_lp) * Y0
            # Predict for test: weighted avg of propagated labels
            y_lp[i] = sim_i @ F
        y_lp = y_lp.clip(EPS, 1-EPS)
        for a, b in [(0.90, 1.55), (0.85, 1.50), (0.80, 1.40)]:
            auc = macro_auc(file_labels, sigmoid(a*base_logit + b*np.log(y_lp.clip(EPS))))
            if auc > ls2_best:
                print(f"  BEAT: lp_a{alpha_lp}_it{n_iter}_ls{a}_{b}: {auc:.4f} (+{auc-ls2_best:.4f})")
                results.append(('label_prop', f'lp_a{alpha_lp}_it{n_iter}', auc, {'alpha_lp':alpha_lp,'n_iter':n_iter,'a':a,'b':b}))
            elif auc > ls2_best - 0.002:
                print(f"  close: lp_a{alpha_lp}_it{n_iter}: {auc:.4f}")

# ─── C. Transductive PCA ──────────────────────────────────────────────────────
# Instead of PCA fitted on 66 training files only,
# fit PCA on ALL 66 training files (same as current but we try different # components)
# The true transductive version would include test files — simulate by
# including the test file in PCA fitting each LOO round
print("\n[C] Transductive PCA (test file included in PCA)...")
file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)

# Precompute geo features
SITES=['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx={s:i for i,s in enumerate(SITES)}
file_sites=np.zeros(n_files,np.int32); file_hours=np.zeros(n_files,np.float32)
file_months=np.zeros(n_files,np.float32); file_days=np.zeros(n_files,np.float32)
for fi,fname in enumerate(file_list):
    m=re.match(r'BC2026_Train_\d+_(S\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})',str(fname))
    if m:
        site,yr,mo,dy,hr,mn=m.groups()
        file_sites[fi]=site2idx.get(site,0); file_hours[fi]=int(hr); file_months[fi]=int(mo)
        dpm=[0,31,28,31,30,31,30,31,31,30,31,30,31]; file_days[fi]=sum(dpm[:int(mo)])+int(dy)
site_oh=np.eye(len(SITES),dtype=np.float32)[file_sites]
hour_enc=np.stack([np.sin(2*np.pi*file_hours/24),np.cos(2*np.pi*file_hours/24)],1).astype(np.float32)
month_enc=np.stack([np.sin(2*np.pi*(file_months-1)/12),np.cos(2*np.pi*(file_months-1)/12)],1).astype(np.float32)
day_enc=np.stack([np.sin(2*np.pi*(file_days-1)/365),np.cos(2*np.pi*(file_days-1)/365)],1).astype(np.float32)
geo_all=np.concatenate([site_oh,hour_enc,month_enc,day_enc],1).astype(np.float32)

for n_pca in [16, 24, 32, 48]:
    y_trans = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        # Include test file in PCA (transductive)
        all_idx = list(range(n_files))  # all 66 files
        X_all = file_embs_norm[all_idx]
        pca_t = PCA(n_components=n_pca, random_state=42).fit(X_all)
        X_pca = pca_t.transform(X_all).astype(np.float32)
        X_pca_s = X_pca / (X_pca.std(0)+1e-6)
        X_nl = np.concatenate([X_pca_s, geo_all[all_idx]], 1).astype(np.float32)
        X_nl /= np.linalg.norm(X_nl,1,keepdims=True)+1e-8
        # KNN in transductive space (exclude i from reference)
        tr = np.array([j for j in range(n_files) if j!=i])
        sims = (X_nl[[i]] @ X_nl[tr].T).ravel()
        top = np.argsort(-sims)[:5]; ls=sims[top]/0.2; ls-=ls.max()
        w=np.exp(ls); w/=w.sum(); y_trans[i]=(w[:,None]*fl[tr[top]]).sum(0)
    y_trans=y_trans.clip(EPS,1-EPS)
    for a,b in [(0.90,1.55),(0.85,1.50),(0.80,1.40)]:
        auc=macro_auc(file_labels,sigmoid(a*base_logit+b*np.log(y_trans.clip(EPS))))
        if auc > ls2_best:
            print(f"  BEAT: trans_pca{n_pca}_ls{a}_{b}: {auc:.4f} (+{auc-ls2_best:.4f})")
            results.append(('trans_pca',f'trans_pca{n_pca}',auc,{'n_pca':n_pca,'a':a,'b':b}))
        elif auc > ls2_best - 0.001:
            print(f"  close: trans_pca{n_pca}: {auc:.4f}")

# ─── D. Multi-hop KNN ─────────────────────────────────────────────────────────
# Instead of just the k nearest neighbors,
# also aggregate their neighbors' labels with exponential decay
# y_2hop[i] = w1 * y_1hop[i] + w2 * sum_j(sim_j * y_1hop[j])
print("\n[D] Multi-hop KNN (2-hop aggregation)...")
for decay in [0.3, 0.5, 0.7]:
    y_2hop = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j!=i])
        sims_i = (X_ref[[i]]@X_ref[tr].T).ravel()
        top1 = np.argsort(-sims_i)[:5]
        ls=sims_i[top1]/T_geo; ls-=ls.max(); w1=np.exp(ls); w1/=w1.sum()
        y_1hop = (w1[:,None]*fl[tr[top1]]).sum(0)
        # 2-hop: for each 1-hop neighbor, get its neighbors' labels
        y_2hop_agg = np.zeros(n_species, np.float32)
        for ni, nn_idx in enumerate(tr[top1]):
            tr2 = np.array([j for j in range(n_files) if j!=i and j!=nn_idx])
            sims2 = (X_ref[[nn_idx]]@X_ref[tr2].T).ravel()
            top2 = np.argsort(-sims2)[:5]
            ls2=sims2[top2]/T_geo; ls2-=ls2.max(); w2=np.exp(ls2); w2/=w2.sum()
            y_2hop_agg += w1[ni] * (w2[:,None]*fl[tr2[top2]]).sum(0)
        y_2hop[i] = (1-decay)*y_1hop + decay*y_2hop_agg
    y_2hop=y_2hop.clip(EPS,1-EPS)
    for a,b in [(0.90,1.55),(0.85,1.50)]:
        auc=macro_auc(file_labels,sigmoid(a*base_logit+b*np.log(y_2hop.clip(EPS))))
        if auc > ls2_best:
            print(f"  BEAT: 2hop_d{decay}_ls{a}_{b}: {auc:.4f} (+{auc-ls2_best:.4f})")
            results.append(('2hop',f'2hop_d{decay}',auc,{'decay':decay,'a':a,'b':b}))
        elif auc > ls2_best - 0.002:
            print(f"  close: 2hop_d{decay}: {auc:.4f}")

# ─── E. Reciprocal KNN (Mutual KNN) ──────────────────────────────────────────
# Only count a neighbor if BOTH files consider each other as top-k neighbors
# This is more robust to noisy similarities
print("\n[E] Reciprocal KNN...")
for k in [5, 10, 15, 20]:
    y_rknn = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j!=i])
        sims_i = (X_ref[[i]]@X_ref[tr].T).ravel()
        top_i = set(tr[np.argsort(-sims_i)[:k]])
        mutual = []
        mutual_sims = []
        for tj in top_i:
            tr2 = np.array([jj for jj in range(n_files) if jj!=tj])
            sims_j = (X_ref[[tj]]@X_ref[tr2].T).ravel()
            top_j = set(tr2[np.argsort(-sims_j)[:k]])
            if i in top_j:  # reciprocal!
                mutual.append(tj)
                mutual_sims.append(sims_i[np.where(tr==tj)[0][0]])
        if len(mutual) == 0:
            # Fallback to standard KNN
            top = np.argsort(-sims_i)[:5]
            ls=sims_i[top]/T_geo; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y_rknn[i]=(w[:,None]*fl[tr[top]]).sum(0)
        else:
            mutual_arr = np.array(mutual); ms = np.array(mutual_sims)
            ls=ms/T_geo; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y_rknn[i]=(w[:,None]*fl[mutual_arr]).sum(0)
    y_rknn=y_rknn.clip(EPS,1-EPS)
    # Blend with win_k1
    for w_g in [0.5, 0.6, 0.7]:
        y_b = w_g*y_rknn + (1-w_g)*y_win
        for a,b in [(0.90,1.55),(0.85,1.50)]:
            auc=macro_auc(file_labels,sigmoid(a*base_logit+b*np.log(y_b.clip(EPS))))
            if auc > ls2_best:
                print(f"  BEAT: rknn_k{k}_wg{w_g}_ls{a}_{b}: {auc:.4f} (+{auc-ls2_best:.4f})")
                results.append(('rknn',f'rknn_k{k}_wg{w_g}',auc,{'k':k,'w_g':w_g,'a':a,'b':b}))
            elif auc > ls2_best - 0.001 and w_g==0.5 and a==0.9:
                print(f"  close: rknn_k{k}: {auc:.4f}")

# ─── F. SED-guided window weighting ───────────────────────────────────────────
# When doing window-KNN, weight each test window by its SED confidence
# (higher SED confidence → trust that window's KNN result more)
print("\n[F] SED-guided window weighting...")
# We need window-level SED predictions aligned to our windows
# Use file-level SED max as a proxy (weight all windows equally by file's SED score)
# Better: weight each window by its Perch logit confidence
y_win_sed = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s,te_e=int(file_start[i]),int(file_end[i])
    X_te=emb_win_norm[te_s:te_e]
    # Window confidence: max logit across species (how "confident" is Perch for this window)
    win_conf = sigmoid(logits_win[te_s:te_e]).max(1)  # (n_windows_i,)
    win_conf = win_conf / (win_conf.sum()+1e-8)  # normalize to weights
    tr_mask=(win_file_id!=i); X_tr=emb_win_norm[tr_mask]; tr_fi=win_file_id[tr_mask]
    sims=X_te@X_tr.T; top_idx=np.argsort(-sims,1)[:,:k_win]
    wp=np.zeros((te_e-te_s,n_species),np.float32)
    for wi in range(te_e-te_s):
        ww=sims[wi,top_idx[wi]].clip(0); ws=ww.sum(); ww=ww/ws if ws>1e-8 else np.ones(k_win)/k_win
        wp[wi]=(ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    # Weighted mean by SED confidence instead of uniform
    y_win_sed[i]=(win_conf[:,None]*wp).sum(0)
y_win_sed=y_win_sed.clip(EPS,1-EPS)
for w_g in [0.4, 0.5, 0.6]:
    y_b = w_g*y_geo + (1-w_g)*y_win_sed
    for a,b in [(0.90,1.55),(0.85,1.50),(0.80,1.40)]:
        auc=macro_auc(file_labels,sigmoid(a*base_logit+b*np.log(y_b.clip(EPS))))
        if auc > ls2_best:
            print(f"  BEAT: sed_win_wg{w_g}_ls{a}_{b}: {auc:.4f} (+{auc-ls2_best:.4f})")
            results.append(('sed_win',f'sed_win_wg{w_g}',auc,{'w_g':w_g,'a':a,'b':b}))
        elif auc > ls2_best - 0.001 and a==0.9:
            print(f"  close: sed_win_wg{w_g}: {auc:.4f}")

# ─── G. Platt-calibrated KNN ──────────────────────────────────────────────────
# Fit logistic regression on LOO KNN predictions → calibrated probabilities
# More accurate probability estimates than raw softmax weights
print("\n[G] Platt-calibrated KNN...")
# Train: use y_geo LOO as input, file_labels as target → per-species calibration
# Then apply calibration at test time
y_platt = np.zeros((n_files, n_species), np.float32)
# LOO: train calibrator on n_files-1, apply to left-out file
for i in range(n_files):
    tr = np.array([j for j in range(n_files) if j!=i])
    X_tr = y_geo[tr]; y_tr = file_labels[tr]
    X_te = y_geo[[i]]
    # Fit per-species Platt scaling
    y_cal = np.zeros(n_species, np.float32)
    for s in range(n_species):
        if y_tr[:,s].sum() < 2 or y_tr[:,s].sum() > len(tr)-2:
            y_cal[s] = y_geo[i,s]
            continue
        try:
            lr = LogisticRegression(C=1.0, max_iter=200)
            lr.fit(X_tr[:,s:s+1], y_tr[:,s])
            y_cal[s] = lr.predict_proba(X_te[:,s:s+1])[0,1]
        except:
            y_cal[s] = y_geo[i,s]
    y_platt[i] = y_cal
y_platt=y_platt.clip(EPS,1-EPS)
for w_g in [0.5, 0.6, 0.7]:
    y_b = w_g*y_platt + (1-w_g)*y_win
    for a,b in [(0.90,1.55),(0.85,1.50)]:
        auc=macro_auc(file_labels,sigmoid(a*base_logit+b*np.log(y_b.clip(EPS))))
        if auc > ls2_best:
            print(f"  BEAT: platt_wg{w_g}_ls{a}_{b}: {auc:.4f} (+{auc-ls2_best:.4f})")
            results.append(('platt',f'platt_wg{w_g}',auc,{'w_g':w_g,'a':a,'b':b}))
        elif auc > ls2_best - 0.001 and a==0.9:
            print(f"  close: platt_wg{w_g}: {auc:.4f}")

# ─── H. Asymmetric log blend (separate geo/win coefficients) ──────────────────
# Instead of blending first then log, apply log separately:
# sigmoid(a * base + b1 * log(geo) + b2 * log(win))
print("\n[H] Asymmetric log blend (separate geo/win log terms)...")
log_geo = np.log(y_geo.clip(EPS))
log_win = np.log(y_win.clip(EPS))
best_asym = {'auc': ls2_best}
for a in [0.85, 0.90, 0.95]:
    for b1 in [0.5, 0.7, 0.9, 1.0, 1.1, 1.2]:
        for b2 in [0.3, 0.5, 0.7, 0.9, 1.0, 1.1]:
            full = sigmoid(a*base_logit + b1*log_geo + b2*log_win)
            auc = macro_auc(file_labels, full)
            if auc > best_asym['auc']:
                best_asym = {'auc':auc,'a':a,'b1':b1,'b2':b2}
                print(f"  BEAT: asym_a{a}_b1{b1}_b2{b2}: {auc:.4f} (+{auc-ls2_best:.4f})")
                results.append(('asym_log',f'asym_a{a}_b1{b1}_b2{b2}',auc,{'a':a,'b1':b1,'b2':b2}))

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY — Novel Methods Batch 3")
print(f"{'='*60}")
print(f"  LS2 baseline (best): {ls2_best:.4f}")
print(f"  v7-geo-knn ref:      0.9246")
print(f"\nMethods beating LS2 baseline ({ls2_best:.4f}):")
if results:
    for method, name, auc, cfg in sorted(results, key=lambda x: -x[2]):
        print(f"  [{method}] {name}: {auc:.4f} (+{auc-ls2_best:.4f})  cfg={cfg}")
else:
    print("  None beat the baseline.")
print("\nDone.")
