"""
Full-pipeline direct search - Batch 2
Target: > 0.9432 (RKNN k=5 wg=0.40 a=0.95 b=1.70)

Novel approaches targeting RKNN's weakness:
1. window_rknn:       Reciprocal KNN at WINDOW level (739 windows, not 66 files)
                      More granular similarity, avoids file-level aggregation bias
2. weighted_mutual:   RKNN with GEOMETRIC MEAN of sim(i→j) and sim(j→i) as weight
                      Better utilizes both directions of similarity
3. soft_threshold:    Instead of hard reciprocal check, use SOFT version:
                      weight = exp(-dist_ij/tau) × exp(-dist_ji/tau)
                      Continuous credit for "near-reciprocal" neighbors
4. rknn_k5_win_rknn:  Window-level RKNN (reciprocal window neighbors)
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

BEST_SO_FAR = 0.9432
print(f"Full pipeline target: > {BEST_SO_FAR:.4f}")
all_results = []

# Precompute file-level sim
sim_file = X_ref @ X_ref.T  # (66, 66)
np.fill_diagonal(sim_file, -np.inf)

# Precompute window-level sim (739 × 739)
print("Computing win-win similarity matrix (739×739)...", flush=True)
sim_win = emb_win_norm @ emb_win_norm.T  # (739, 739)
np.fill_diagonal(sim_win, -np.inf)
print("  done.", flush=True)

# Precompute standard win_k1 for blending
print("Computing win_k1...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = win_file_id != i
    X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :1]
    wp = np.zeros((te_e - te_s, n_species), np.float32)
    for wi in range(te_e - te_s):
        ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
        ww = ww/ws if ws > 1e-8 else np.ones(1)
        wp[wi] = (ww[:, None] * file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)
print("  done.", flush=True)

# Standard RKNN k=5
def compute_rknn_k5():
    k=5; T=0.2
    top_k_tr = np.argsort(-sim_file, axis=1)[:, :k]
    kth_sim = sim_file[np.arange(n_files), top_k_tr[:, -1]]
    y_rknn = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr=np.array([j for j in range(n_files) if j!=i])
        sims_i=sim_file[i,tr]; top_i=np.argsort(-sims_i)[:k]
        mutual=[]; mutual_sims=[]
        for ti, tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]] >= kth_sim[tj]:
                mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
        if len(mutual)==0:
            top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y_rknn[i]=(w[:,None]*fl[tr[top5]]).sum(0)
        else:
            ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y_rknn[i]=(w[:,None]*fl[mutual]).sum(0)
    return y_rknn

print("Computing rknn_k5...", flush=True)
y_rknn_k5 = compute_rknn_k5()
print("  done.", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Method 1: Window-level Reciprocal KNN
# Each test window finds reciprocal training WINDOWS.
# More granular than file-level RKNN: catches specific acoustic moments.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Window-level RKNN (win-win reciprocal)...", flush=True)

for k_wr in [3, 5]:
    # Precompute top-k for each TRAINING window
    # For LOO: when processing test file i, exclude its own windows
    # "kth_sim_window[w]" = k-th neighbor similarity for window w among ALL windows
    # But we need to handle LOO: only neighbors from DIFFERENT FILES count
    top_k_win_all = np.argsort(-sim_win, axis=1)[:, :k_wr]  # (739, k)
    kth_sim_win_all = sim_win[np.arange(len(emb_win)), top_k_win_all[:, -1]]

    y_win_rknn = np.zeros((n_files, n_species), np.float32)
    T = 0.2
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        n_te = te_e - te_s
        # Training window mask: exclude windows from file i
        tr_mask = win_file_id != i
        tr_idx = np.where(tr_mask)[0]  # actual window indices

        y_per_test_win = np.zeros((n_te, n_species), np.float32)
        for wi, w in enumerate(range(te_s, te_e)):
            # For each test window w:
            # Find top-k_wr training windows by similarity
            sims_w_tr = sim_win[w, tr_idx]  # (n_train_windows,)
            top_tr = np.argsort(-sims_w_tr)[:k_wr]  # indices into tr_idx

            # Check reciprocal: does training window tw consider test window w as top-k?
            # Use kth_sim threshold from training window's perspective
            mutual_tw = []; mutual_sims_tw = []
            for t_local in range(len(top_tr)):
                tw = tr_idx[top_tr[t_local]]  # actual training window index
                sim_w_tw = sims_w_tr[top_tr[t_local]]
                # Check if sim(w, tw) >= kth-sim of tw (among all windows, LOO approximation)
                if sim_w_tw >= kth_sim_win_all[tw]:
                    mutual_tw.append(tw)
                    mutual_sims_tw.append(sim_w_tw)

            if len(mutual_tw) == 0:
                # Fallback: use top-5 training windows
                fallback = top_tr[:5]
                fb_sims = sims_w_tr[fallback]
                ls = fb_sims / T; ls -= ls.max(); w_fb = np.exp(ls); w_fb /= w_fb.sum()
                y_per_test_win[wi] = (w_fb[:, None] * file_labels[win_file_id[tr_idx[fallback]]]).sum(0)
            else:
                ms = np.array(mutual_sims_tw)
                ls = ms / T; ls -= ls.max(); w_m = np.exp(ls); w_m /= w_m.sum()
                y_per_test_win[wi] = (w_m[:, None] * file_labels[win_file_id[mutual_tw]]).sum(0)

        y_win_rknn[i] = y_per_test_win.mean(0)

    print(f"  Win-RKNN k={k_wr} computed.", flush=True)

    best = {'auc': 0}
    for wg in [0.30, 0.35, 0.40, 0.45, 0.50]:
        yb = wg * y_rknn_k5 + (1-wg) * y_win_rknn  # blend with file-RKNN
        log_yb = np.log(yb.clip(EPS))
        for a in [0.85, 0.90, 0.95, 1.00]:
            for b in [1.40, 1.50, 1.60, 1.70, 1.80, 1.90]:
                full = sigmoid(a * base_logit + b * log_yb)
                auc = macro_auc(file_labels, full)
                if auc > best['auc']:
                    best = {'auc': auc, 'wg': wg, 'a': a, 'b': b, 'k': k_wr}
                    if auc > BEST_SO_FAR:
                        print(f"  *** k={k_wr} wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                    else:
                        print(f"  k={k_wr} wg={wg} a={a} b={b}: {auc:.4f}", flush=True)

    # Also test win-RKNN alone with standard win_k1
    for wg in [0.50, 0.55, 0.60]:
        yb = wg * y_win_rknn + (1-wg) * y_win_k1
        log_yb = np.log(yb.clip(EPS))
        for a in [0.85, 0.90, 0.95]:
            for b in [1.50, 1.60, 1.70, 1.80]:
                full = sigmoid(a * base_logit + b * log_yb)
                auc = macro_auc(file_labels, full)
                if auc > best['auc']:
                    best = {'auc': auc, 'wg': wg, 'a': a, 'b': b, 'k': k_wr, 'mode': 'win-only'}
                    if auc > BEST_SO_FAR:
                        print(f"  *** win-only k={k_wr} wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                    else:
                        print(f"  win-only k={k_wr} wg={wg} a={a} b={b}: {auc:.4f}", flush=True)

    all_results.append((f'win_rknn_k{k_wr}', best['auc'], best))
    print(f"  Best win-RKNN k={k_wr}: {best['auc']:.4f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Method 2: Weighted Mutual RKNN
# Instead of just checking if i is in j's top-k,
# use GEOMETRIC MEAN of similarity in both directions as the weight:
# w(i,j) = sqrt(sim(i→j) × sim(j→i)) if j ∈ top-k(i)
# This gives higher weight to more "symmetric" neighbors.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Weighted Mutual RKNN (geometric mean of bidirectional sim)...", flush=True)
best2 = {'auc': 0}
k, T = 5, 0.2

y_wm_rknn = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    tr = np.array([j for j in range(n_files) if j != i])
    sims_i = sim_file[i, tr]
    top_i = np.argsort(-sims_i)[:k]  # top-k candidates

    # Compute geometric mean weight for each top-k neighbor
    gm_sims = []
    candidates = []
    for t_local, tj in enumerate(tr[top_i]):
        sim_ij = sims_i[top_i[t_local]]  # i→j similarity
        # j→i similarity: sim(j, i) = sim(i, j) (symmetric dot product)
        sim_ji = sim_file[tj, i]  # same as sim_ij in symmetric space
        # But looking at rank: find rank of i among j's neighbors
        tr_j = np.array([jj for jj in range(n_files) if jj != tj])
        sims_j = sim_file[tj, tr_j]
        rank_i_in_j = np.sum(sims_j > sim_ji)  # how many files j considers closer than i
        # Soft weight: exp(-rank_penalty)
        rank_weight = np.exp(-rank_i_in_j / k)  # higher rank = lower weight
        gm_weight = sim_ij * rank_weight  # use rank-adjusted weight
        candidates.append(tj)
        gm_sims.append(gm_weight)

    if len(candidates) == 0:
        top5 = np.argsort(-sims_i)[:5]
        ls = sims_i[top5]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
        y_wm_rknn[i] = (w[:, None] * fl[tr[top5]]).sum(0)
    else:
        gm_arr = np.array(gm_sims)
        ls = gm_arr / T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
        y_wm_rknn[i] = (w[:, None] * fl[candidates]).sum(0)

for wg in [0.30, 0.35, 0.40, 0.45, 0.50]:
    yb = wg * y_wm_rknn + (1-wg) * y_win_k1
    log_yb = np.log(yb.clip(EPS))
    for a in [0.85, 0.90, 0.95, 1.00]:
        for b in [1.40, 1.50, 1.60, 1.70, 1.80, 1.90]:
            full = sigmoid(a * base_logit + b * log_yb)
            auc = macro_auc(file_labels, full)
            if auc > best2['auc']:
                best2 = {'auc': auc, 'wg': wg, 'a': a, 'b': b}
                if auc > BEST_SO_FAR:
                    print(f"  *** wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                else:
                    print(f"  wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
print(f"  Best weighted-mutual RKNN: {best2['auc']:.4f}")
all_results.append(('weighted_mutual_rknn', best2['auc'], best2))

# ══════════════════════════════════════════════════════════════════════════════
# Method 3: Soft-Threshold RKNN
# Instead of binary reciprocal check (is i in top-k of j?),
# use SOFT credit: exp(-rank(i in j's neighbors) / tau)
# Every neighbor gets some credit, but closer-to-reciprocal ones get more.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Soft-threshold RKNN...", flush=True)
best3 = {'auc': 0}

# Precompute rank of each file in every other file's neighbor list
ranks = np.zeros((n_files, n_files), np.int32)  # ranks[i,j] = rank of i in j's neighbor list
for j in range(n_files):
    sorted_j = np.argsort(-sim_file[j])
    for r, s in enumerate(sorted_j):
        if s != j:
            ranks[s, j] = r  # rank of file s in j's sorted list

for k_soft in [5, 7, 10]:
    for tau_rank in [1.0, 2.0, 3.0, 5.0]:
        T = 0.2
        y_soft = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j != i])
            sims_i = sim_file[i, tr]
            top_i = np.argsort(-sims_i)[:k_soft]
            neighbors = tr[top_i]
            sims_top = sims_i[top_i]
            # For each neighbor j, get rank of i in j's list
            rank_i_in_j = np.array([ranks[i, j] for j in neighbors], dtype=np.float32)
            # Soft weight: sim × exp(-rank / tau)
            soft_weight = sims_top * np.exp(-rank_i_in_j / tau_rank)
            ls = soft_weight / T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y_soft[i] = (w[:, None] * fl[neighbors]).sum(0)

        for wg in [0.30, 0.35, 0.40, 0.45, 0.50]:
            yb = wg * y_soft + (1-wg) * y_win_k1
            log_yb = np.log(yb.clip(EPS))
            for a in [0.85, 0.90, 0.95, 1.00]:
                for b in [1.40, 1.50, 1.60, 1.70, 1.80, 1.90]:
                    full = sigmoid(a * base_logit + b * log_yb)
                    auc = macro_auc(file_labels, full)
                    if auc > best3['auc']:
                        best3 = {'auc': auc, 'k': k_soft, 'tau': tau_rank, 'wg': wg, 'a': a, 'b': b}
                        if auc > BEST_SO_FAR:
                            print(f"  *** k={k_soft} tau={tau_rank} wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                        else:
                            print(f"  k={k_soft} tau={tau_rank} wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
print(f"  Best soft-threshold RKNN: {best3['auc']:.4f}")
all_results.append(('soft_threshold_rknn', best3['auc'], best3))

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"FULL PIPELINE BATCH 2 SUMMARY")
print(f"{'='*60}")
print(f"Baseline RKNN k=5 wg=0.40 a=0.95 b=1.70: {BEST_SO_FAR:.4f}")
for name, auc, cfg in sorted(all_results, key=lambda x: -x[1]):
    marker = " *** NEW BEST ***" if auc > BEST_SO_FAR else ""
    print(f"  {name:30s}: {auc:.4f}{marker}")
