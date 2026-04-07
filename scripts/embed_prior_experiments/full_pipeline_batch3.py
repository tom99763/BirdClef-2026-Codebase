"""
Full-pipeline direct search - Batch 3
Target: > 0.9432 (RKNN k=5 wg=0.40 a=0.95 b=1.70)

Key insight: RKNN currently uses X_combined_n (PCA24 + geo, 39-dim).
What if we use a richer feature space?

Variants:
1. rknn_perch_full:   RKNN using file-level L2-normalized Perch avg embedding (1536-dim)
2. rknn_perch_pca:    RKNN using PCA-compressed file-level Perch embeddings (48, 64, 96-dim)
3. rknn_concat_space: RKNN using concatenated X_ref + Perch_avg (39+1536=1575-dim)
4. rknn_sed_feature:  RKNN using SED probability predictions as feature (234-dim)
5. rknn_combined3:    Best of all: weighted combination of file-RKNN (X_ref) + win-RKNN (Perch)
"""
import numpy as np, pickle, os, json
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
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
file_embs=np.zeros((n_files,emb_win.shape[1]),np.float32)
for fi in range(n_files):
    s,e=int(file_start[fi]),int(file_end[fi])
    file_labels[fi]=(labels_win[s:e].max(0)>0.5).astype(np.float32)
    file_logit_max[fi]=logits_win[s:e].max(0)
    file_embs[fi]=emb_win[s:e].mean(0)
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

BEST_SO_FAR = 0.9432
print(f"Target: > {BEST_SO_FAR:.4f}")
all_results = []

# RKNN helper
def compute_rknn_space(X_space, k=5, T=0.2):
    """Compute RKNN in given feature space X_space (n_files, d)"""
    X_norm = normalize(X_space, norm='l2').astype(np.float32)
    sim_mat = X_norm @ X_norm.T
    np.fill_diagonal(sim_mat, -np.inf)
    top_k_tr = np.argsort(-sim_mat, axis=1)[:, :k]
    kth_sim = sim_mat[np.arange(n_files), top_k_tr[:, -1]]
    y = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr=np.array([j for j in range(n_files) if j!=i])
        sims_i=sim_mat[i,tr]; top_i=np.argsort(-sims_i)[:k]
        mutual=[]; mutual_sims=[]
        for ti, tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]] >= kth_sim[tj]:
                mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
        if len(mutual)==0:
            top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y[i]=(w[:,None]*fl[tr[top5]]).sum(0)
        else:
            ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y[i]=(w[:,None]*fl[mutual]).sum(0)
    return y

def sweep_blend(y_ep, y_win, label):
    best = {'auc': 0}
    for wg in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        yb = wg * y_ep + (1-wg) * y_win
        log_yb = np.log(yb.clip(EPS))
        for a in [0.85, 0.90, 0.95, 1.00]:
            for b in [1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]:
                full = sigmoid(a * base_logit + b * log_yb)
                auc = macro_auc(file_labels, full)
                if auc > best['auc']:
                    best = {'auc': auc, 'wg': wg, 'a': a, 'b': b}
                    if auc > BEST_SO_FAR:
                        print(f"  *** {label} wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                    else:
                        print(f"  {label} wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
    return best

# Precompute
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
# Method 1: RKNN in 1536-dim Perch avg embedding space
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] RKNN in 1536-dim Perch avg space...", flush=True)
y_rknn_perch = compute_rknn_space(file_embs, k=5)
best1 = sweep_blend(y_rknn_perch, y_win_k1, "rknn_perch_1536")
print(f"  Best: {best1['auc']:.4f}")
all_results.append(('rknn_perch_1536', best1['auc'], best1))

# ══════════════════════════════════════════════════════════════════════════════
# Method 2: RKNN in PCA-compressed Perch space (various dims)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] RKNN in PCA-Perch space...", flush=True)
for pca_dim in [32, 48, 64]:
    pca = PCA(n_components=pca_dim, random_state=42)
    file_embs_pca = pca.fit_transform(file_embs).astype(np.float32)
    y_rknn_pca = compute_rknn_space(file_embs_pca, k=5)
    best_pca = sweep_blend(y_rknn_pca, y_win_k1, f"rknn_perch_pca{pca_dim}")
    print(f"  PCA{pca_dim} Best: {best_pca['auc']:.4f}")
    all_results.append((f'rknn_perch_pca{pca_dim}', best_pca['auc'], best_pca))

# ══════════════════════════════════════════════════════════════════════════════
# Method 3: RKNN in SED feature space (234-dim probabilities)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] RKNN in SED probability space...", flush=True)
# Normalize SED features (log-probability space for better geometry)
sed_feat = np.log(file_sed_max.clip(EPS))  # log probabilities as features
y_rknn_sed = compute_rknn_space(sed_feat, k=5)
best3 = sweep_blend(y_rknn_sed, y_win_k1, "rknn_sed")
print(f"  Best: {best3['auc']:.4f}")
all_results.append(('rknn_sed_space', best3['auc'], best3))

# Also try raw SED probabilities
y_rknn_sed_raw = compute_rknn_space(file_sed_max, k=5)
best3b = sweep_blend(y_rknn_sed_raw, y_win_k1, "rknn_sed_raw")
print(f"  Best (raw): {best3b['auc']:.4f}")
all_results.append(('rknn_sed_raw', best3b['auc'], best3b))

# ══════════════════════════════════════════════════════════════════════════════
# Method 4: RKNN in concatenated space (X_ref + Perch_avg_pca32)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] RKNN in concatenated X_ref + Perch_PCA32 space...", flush=True)
pca32 = PCA(n_components=32, random_state=42)
perch_pca32 = pca32.fit_transform(file_embs).astype(np.float32)
X_concat = np.concatenate([X_ref, normalize(perch_pca32, norm='l2')], axis=1)
y_rknn_concat = compute_rknn_space(X_concat, k=5)
best4 = sweep_blend(y_rknn_concat, y_win_k1, "rknn_concat39+32")
print(f"  Best: {best4['auc']:.4f}")
all_results.append(('rknn_concat_Xref_perch32', best4['auc'], best4))

# ══════════════════════════════════════════════════════════════════════════════
# Method 5: Dual RKNN ensemble
# Compute RKNN in X_ref AND in Perch space, blend both signals
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Dual RKNN (X_ref + Perch) ensemble...", flush=True)
# Recompute standard RKNN
y_rknn_xref = compute_rknn_space(X_ref, k=5)

best5 = {'auc': 0}
for wr in [0.50, 0.60, 0.70, 0.80]:
    wp_ep = 1.0 - wr
    y_ep_dual = wr * y_rknn_xref + wp_ep * y_rknn_perch
    for wwin in [0.30, 0.35, 0.40, 0.45]:
        yb = (1-wwin) * y_ep_dual + wwin * y_win_k1
        log_yb = np.log(yb.clip(EPS))
        for a in [0.85, 0.90, 0.95, 1.00]:
            for b in [1.50, 1.60, 1.70, 1.80, 1.90]:
                full = sigmoid(a * base_logit + b * log_yb)
                auc = macro_auc(file_labels, full)
                if auc > best5['auc']:
                    best5 = {'auc': auc, 'wr': wr, 'wp': wp_ep, 'wwin': wwin, 'a': a, 'b': b}
                    if auc > BEST_SO_FAR:
                        print(f"  *** wr={wr} wp={wp_ep:.2f} wwin={wwin} a={a} b={b}: {auc:.4f} ***", flush=True)
                    else:
                        print(f"  wr={wr} wwin={wwin} a={a} b={b}: {auc:.4f}", flush=True)
print(f"  Best dual RKNN: {best5['auc']:.4f}")
all_results.append(('dual_rknn_xref_perch', best5['auc'], best5))

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"FULL PIPELINE BATCH 3 SUMMARY")
print(f"{'='*60}")
print(f"Baseline RKNN k=5 wg=0.40 a=0.95 b=1.70: {BEST_SO_FAR:.4f}")
for name, auc, cfg in sorted(all_results, key=lambda x: -x[1]):
    marker = " *** NEW BEST ***" if auc > BEST_SO_FAR else ""
    print(f"  {name:30s}: {auc:.4f}{marker}")
