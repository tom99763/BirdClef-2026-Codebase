"""
Full-pipeline Batch 5: Transductive Learning via Soundscape Bridge
Target: > 0.9432

Key insight: Use ALL 127,896 soundscape windows as "bridge" to compute
indirect similarity between the 66 labeled files.

Even if files A and B don't sound directly similar, if they both have
windows close to the same soundscape cluster, they likely share species.

Methods:
1. ss_bridge_jaccard:  Indirect sim via top-K soundscape overlap (Jaccard)
2. ss_bridge_weighted: Weighted indirect sim via soundscape k-NN
3. rknn_transductive:  RKNN on indirect similarity matrix (ss-bridged)
4. ss_feature_rknn:    Use soundscape co-occurrence features for RKNN
"""
import numpy as np, pickle, os
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
file_embs_avg=np.zeros((n_files,1536),np.float32)
for fi in range(n_files):
    s,e=int(file_start[fi]),int(file_end[fi])
    file_labels[fi]=(labels_win[s:e].max(0)>0.5).astype(np.float32)
    file_logit_max[fi]=logits_win[s:e].max(0)
    file_embs_avg[fi]=emb_win[s:e].mean(0)
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

# Load all soundscape embeddings
print("\nLoading all soundscape embeddings (127896 windows)...", flush=True)
ss_all = np.load("outputs/perch_emb_all_ss.npz", allow_pickle=True)
ss_emb = ss_all['emb'].astype(np.float32)    # (127896, 1536)
ss_emb_norm = normalize(ss_emb, norm='l2').astype(np.float32)
n_ss_wins = len(ss_emb_norm)
print(f"  Loaded {n_ss_wins} soundscape windows.", flush=True)

# File-level avg embeddings of labeled files
file_emb_avg_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)  # (66, 1536)

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

# Standard RKNN k=5 for comparison
print("Computing standard RKNN k=5...", flush=True)
sim_ref = X_ref @ X_ref.T
np.fill_diagonal(sim_ref, -np.inf)
k, T = 5, 0.2
top_k_tr = np.argsort(-sim_ref, axis=1)[:, :k]
kth_sim = sim_ref[np.arange(n_files), top_k_tr[:, -1]]
y_rknn_std = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    tr=np.array([j for j in range(n_files) if j!=i])
    sims_i=sim_ref[i,tr]; top_i=np.argsort(-sims_i)[:k]
    mutual=[]; mutual_sims=[]
    for ti, tj in enumerate(tr[top_i]):
        if sims_i[top_i[ti]] >= kth_sim[tj]:
            mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
    if len(mutual)==0:
        top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_rknn_std[i]=(w[:,None]*fl[tr[top5]]).sum(0)
    else:
        ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_rknn_std[i]=(w[:,None]*fl[mutual]).sum(0)
print("  done.", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Compute Soundscape Bridge Similarity
# For each labeled file, find its top-M soundscape windows.
# Two labeled files' indirect similarity = overlap of their SS neighborhoods.
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing soundscape bridge similarities...", flush=True)
CHUNK = 20000  # Process in chunks to avoid OOM

# For each labeled file (avg embedding), compute similarity to ALL ss windows
# Matrix: (66, 127896) - too large to store? 66 × 127896 × 4 = ~32 MB, OK
print("  Computing labeled vs all-soundscape similarity matrix...", flush=True)
# Process in chunks
sim_lab_ss = np.zeros((n_files, n_ss_wins), np.float32)
for chunk_start in range(0, n_ss_wins, CHUNK):
    chunk_end = min(chunk_start + CHUNK, n_ss_wins)
    sim_lab_ss[:, chunk_start:chunk_end] = file_emb_avg_norm @ ss_emb_norm[chunk_start:chunk_end].T
    if chunk_start % 40000 == 0:
        print(f"    chunk {chunk_start}/{n_ss_wins}...", flush=True)
print("  done.", flush=True)

# For each labeled file, get top-M ss window indices
for M in [50, 100, 200]:
    print(f"\n  Building Jaccard bridge with M={M}...", flush=True)
    # Top-M ss windows per labeled file
    top_M_idx = np.argsort(-sim_lab_ss, axis=1)[:, :M]  # (66, M)

    # Build intersection count matrix: (66, 66)
    sim_bridge = np.zeros((n_files, n_files), np.float32)
    for i in range(n_files):
        set_i = set(top_M_idx[i].tolist())
        for j in range(n_files):
            if i == j: continue
            set_j = set(top_M_idx[j].tolist())
            intersect = len(set_i & set_j)
            union = M + M - intersect
            sim_bridge[i, j] = intersect / union  # Jaccard similarity

    # Add to X_ref similarity (bridge as supplement)
    for alpha in [0.1, 0.2, 0.3, 0.5]:
        sim_combined = (1-alpha) * sim_ref.copy() + alpha * sim_bridge.copy()
        np.fill_diagonal(sim_combined, -np.inf)

        top_k_comb = np.argsort(-sim_combined, axis=1)[:, :k]
        kth_sim_comb = sim_combined[np.arange(n_files), top_k_comb[:, -1]]
        y_bridge = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr=np.array([j for j in range(n_files) if j!=i])
            sims_i=sim_combined[i,tr]; top_i=np.argsort(-sims_i)[:k]
            mutual=[]; mutual_sims=[]
            for ti, tj in enumerate(tr[top_i]):
                if sims_i[top_i[ti]] >= kth_sim_comb[tj]:
                    mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
            if len(mutual)==0:
                top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
                y_bridge[i]=(w[:,None]*fl[tr[top5]]).sum(0)
            else:
                ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
                y_bridge[i]=(w[:,None]*fl[mutual]).sum(0)

        best = {'auc': 0}
        for wg in [0.30, 0.35, 0.40, 0.45]:
            yb = wg * y_bridge + (1-wg) * y_win_k1
            log_yb = np.log(yb.clip(EPS))
            for a in [0.85, 0.90, 0.95, 1.00]:
                for b in [1.50, 1.60, 1.70, 1.80, 1.90]:
                    full = sigmoid(a * base_logit + b * log_yb)
                    auc = macro_auc(file_labels, full)
                    if auc > best['auc']:
                        best = {'auc': auc, 'M': M, 'alpha': alpha, 'wg': wg, 'a': a, 'b': b}
                        if auc > BEST_SO_FAR:
                            print(f"  *** M={M} alpha={alpha} wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                        elif auc > 0.9420:
                            print(f"  M={M} alpha={alpha} wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
        if best['auc'] > 0:
            print(f"  M={M} alpha={alpha} best: {best['auc']:.4f}", flush=True)
            all_results.append((f'ss_bridge_jaccard_M{M}_a{alpha}', best['auc'], best))

# ══════════════════════════════════════════════════════════════════════════════
# Method 2: Weighted Soundscape Bridge (not Jaccard, but weighted overlap)
# Instead of Jaccard, use SUM of top-M similarity scores as bridge signal.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Weighted soundscape bridge...", flush=True)

# Top-M similarity scores per labeled file
M = 100
top_M_idx = np.argsort(-sim_lab_ss, axis=1)[:, :M]
top_M_sims = np.sort(-sim_lab_ss, axis=1)[:, :M] * -1  # (66, M), sorted desc

# Bridge similarity: dot product in "SS indicator space"
# sim_bridge2[i,j] = sum over m of (sim_i_m × sim_j_m) for top shared windows
sim_bridge2 = np.zeros((n_files, n_files), np.float32)
for i in range(n_files):
    for j in range(n_files):
        if i == j: continue
        # Sum of products for shared windows in top-M
        sim_i_full = sim_lab_ss[i, top_M_idx[j]]  # i's similarity to j's top windows
        sim_j_full = top_M_sims[j]                  # j's own similarity to j's top windows
        sim_bridge2[i, j] = (sim_i_full * sim_j_full).sum()

# Normalize
sim_bridge2 /= (np.sqrt((sim_bridge2 * sim_bridge2).sum(1, keepdims=True)).clip(1e-8))

for alpha in [0.1, 0.2, 0.3]:
    sim_comb2 = (1-alpha) * sim_ref.copy() + alpha * sim_bridge2.copy()
    np.fill_diagonal(sim_comb2, -np.inf)
    top_k2 = np.argsort(-sim_comb2, axis=1)[:, :k]
    kth2 = sim_comb2[np.arange(n_files), top_k2[:, -1]]
    y_b2 = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr=np.array([j for j in range(n_files) if j!=i])
        sims_i=sim_comb2[i,tr]; top_i=np.argsort(-sims_i)[:k]
        mutual=[]; mutual_sims=[]
        for ti, tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]] >= kth2[tj]:
                mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
        if len(mutual)==0:
            top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y_b2[i]=(w[:,None]*fl[tr[top5]]).sum(0)
        else:
            ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y_b2[i]=(w[:,None]*fl[mutual]).sum(0)

    best2 = {'auc': 0}
    for wg in [0.30, 0.35, 0.40, 0.45]:
        yb = wg * y_b2 + (1-wg) * y_win_k1
        log_yb = np.log(yb.clip(EPS))
        for a in [0.85, 0.90, 0.95, 1.00]:
            for b in [1.50, 1.60, 1.70, 1.80, 1.90]:
                full = sigmoid(a * base_logit + b * log_yb)
                auc = macro_auc(file_labels, full)
                if auc > best2['auc']:
                    best2 = {'auc': auc, 'alpha': alpha, 'wg': wg, 'a': a, 'b': b}
                    if auc > BEST_SO_FAR:
                        print(f"  *** alpha={alpha} wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                    elif auc > 0.9420:
                        print(f"  alpha={alpha} wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
    print(f"  Weighted bridge alpha={alpha} best: {best2['auc']:.4f}", flush=True)
    all_results.append((f'ss_bridge_weighted_a{alpha}', best2['auc'], best2))

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"FULL PIPELINE BATCH 5 (Transductive) SUMMARY")
print(f"{'='*60}")
print(f"Baseline RKNN k=5 wg=0.40 a=0.95 b=1.70: {BEST_SO_FAR:.4f}")
for name, auc, cfg in sorted(all_results, key=lambda x: -x[1]):
    marker = " *** NEW BEST ***" if auc > BEST_SO_FAR else ""
    print(f"  {name:35s}: {auc:.4f}{marker}")
