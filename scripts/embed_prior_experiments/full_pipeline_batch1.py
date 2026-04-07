"""
Full-pipeline direct search - Batch 1
直接在完整 pipeline 中搜索突破性方法
（不用 EP-only 代理，直接最大化完整 pipeline AUC）

Current best FULL pipeline: RKNN k=5 wg=0.40 a=0.95 b=1.70 = 0.9432

Novel RKNN variants to test:
1. rknn_k7:          RKNN with k=7 (more mutual neighbors)
2. rknn_k5_k3_mix:   Blend of RKNN k=5 and RKNN k=3 signals
3. rknn_win_k2:      RKNN k=5 + window KNN k=2 (instead of k=1)
4. rknn_geo_win:     RKNN k=5 + geo_k5 + win_k1 (3-way blend)
5. rknn_asymmetric:  RKNN and win have separate logspace coefficients (already tried: 0.9427)
6. rknn_soft_fallback: When no reciprocal neighbors, use soft decay instead of hard fallback
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

sim_train = X_ref @ X_ref.T
np.fill_diagonal(sim_train, -np.inf)

BEST_SO_FAR = 0.9432
print(f"Full pipeline target: > {BEST_SO_FAR:.4f} (RKNN k=5 wg=0.40 a=0.95 b=1.70)")
all_results = []

# ─── Window KNN precomputation ────────────────────────────────────────────────
def compute_win_knn(k_win=1):
    y_win = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s,te_e=int(file_start[i]),int(file_end[i]); X_te=emb_win_norm[te_s:te_e]
        tr_mask=(win_file_id!=i); X_tr=emb_win_norm[tr_mask]; tr_fi=win_file_id[tr_mask]
        sims=X_te@X_tr.T; top_idx=np.argsort(-sims,1)[:,:k_win]
        wp=np.zeros((te_e-te_s,n_species),np.float32)
        for wi in range(te_e-te_s):
            ww=sims[wi,top_idx[wi]].clip(0); ws=ww.sum()
            ww=ww/ws if ws>1e-8 else np.ones(k_win)/k_win
            wp[wi]=(ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
        y_win[i]=wp.mean(0)
    return y_win

# ─── RKNN computation ────────────────────────────────────────────────────────
def compute_rknn(k, T=0.2):
    top_k_train = np.argsort(-sim_train, axis=1)[:, :k]
    kth_sim_train = sim_train[np.arange(n_files), top_k_train[:, -1]]
    y_rknn = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr=np.array([j for j in range(n_files) if j!=i])
        sims_i=sim_train[i,tr]; top_i=np.argsort(-sims_i)[:k]
        mutual=[]; mutual_sims=[]
        for ti, tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]] >= kth_sim_train[tj]:
                mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
        if len(mutual)==0:
            top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y_rknn[i]=(w[:,None]*fl[tr[top5]]).sum(0)
        else:
            ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y_rknn[i]=(w[:,None]*fl[mutual]).sum(0)
    return y_rknn

# Precompute
print("Precomputing win_k1...", flush=True)
y_win_k1 = compute_win_knn(1)
print("Precomputing win_k2...", flush=True)
y_win_k2 = compute_win_knn(2)
print("Precomputing win_k3...", flush=True)
y_win_k3 = compute_win_knn(3)
print("Precomputing rknn_k5...", flush=True)
y_rknn_k5 = compute_rknn(5)
print("Precomputing rknn_k7...", flush=True)
y_rknn_k7 = compute_rknn(7)
print("Precomputing rknn_k3...", flush=True)
y_rknn_k3 = compute_rknn(3)
print("All precomputed.", flush=True)

# Geo KNN k=5
y_geo_k5 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    tr=np.array([j for j in range(n_files) if j!=i])
    sims=sim_train[i,tr]; top=np.argsort(-sims)[:5]
    ls=sims[top]/0.2; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
    y_geo_k5[i]=(w[:,None]*fl[tr[top]]).sum(0)

# ══════════════════════════════════════════════════════════════════════════════
# Test 1: RKNN k=7 + win_k1
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] RKNN k=7 + win_k1...", flush=True)
best1 = {'auc': 0}
for wg in [0.30, 0.35, 0.40, 0.45, 0.50]:
    yb = wg * y_rknn_k7 + (1-wg) * y_win_k1
    log_yb = np.log(yb.clip(EPS))
    for a in [0.85, 0.90, 0.95, 1.00]:
        for b in [1.40, 1.50, 1.55, 1.60, 1.70, 1.80, 1.90]:
            full = sigmoid(a * base_logit + b * log_yb)
            auc = macro_auc(file_labels, full)
            if auc > best1['auc']:
                best1 = {'auc': auc, 'wg': wg, 'a': a, 'b': b}
                if auc > BEST_SO_FAR:
                    print(f"  *** wg={wg} a={a} b={b}: {auc:.4f} > {BEST_SO_FAR} ***", flush=True)
                else:
                    print(f"  wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
print(f"  Best rknn_k7+win_k1: {best1['auc']:.4f}")
all_results.append(('rknn_k7_win_k1', best1['auc'], best1))

# ══════════════════════════════════════════════════════════════════════════════
# Test 2: RKNN k=5 + win_k2 (win with k=2 neighbors)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] RKNN k=5 + win_k2...", flush=True)
best2 = {'auc': 0}
for wg in [0.30, 0.35, 0.40, 0.45, 0.50]:
    yb = wg * y_rknn_k5 + (1-wg) * y_win_k2
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
print(f"  Best rknn_k5+win_k2: {best2['auc']:.4f}")
all_results.append(('rknn_k5_win_k2', best2['auc'], best2))

# ══════════════════════════════════════════════════════════════════════════════
# Test 3: Multi-RKNN blend (k=3 + k=5 + k=7 mixture)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Multi-RKNN k=3+5+7 blend + win_k1...", flush=True)
best3 = {'auc': 0}
for w3 in [0.20, 0.33]:
    for w5 in [0.40, 0.47]:
        w7 = 1.0 - w3 - w5
        if w7 < 0 or w7 > 0.50: continue
        y_multi = w3 * y_rknn_k3 + w5 * y_rknn_k5 + w7 * y_rknn_k7
        for wg in [0.30, 0.35, 0.40, 0.45]:
            yb = wg * y_multi + (1-wg) * y_win_k1
            log_yb = np.log(yb.clip(EPS))
            for a in [0.90, 0.95, 1.00]:
                for b in [1.50, 1.60, 1.70, 1.80, 1.90]:
                    full = sigmoid(a * base_logit + b * log_yb)
                    auc = macro_auc(file_labels, full)
                    if auc > best3['auc']:
                        best3 = {'auc': auc, 'w3': w3, 'w5': w5, 'w7': w7, 'wg': wg, 'a': a, 'b': b}
                        if auc > BEST_SO_FAR:
                            print(f"  *** w3={w3} w5={w5} w7={w7:.2f} wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                        else:
                            print(f"  w3={w3} w5={w5} wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
print(f"  Best multi-rknn: {best3['auc']:.4f}")
all_results.append(('multi_rknn_k357_win', best3['auc'], best3))

# ══════════════════════════════════════════════════════════════════════════════
# Test 4: RKNN k=5 + geo_k5 + win_k1 (3-way)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] RKNN k=5 + geo_k5 + win_k1 (3-way)...", flush=True)
best4 = {'auc': 0}
for wr in [0.40, 0.45, 0.50]:
    for wg2 in [0.10, 0.15, 0.20, 0.25]:
        ww = 1.0 - wr - wg2
        if ww < 0.15 or ww > 0.50: continue
        y_3way = wr * y_rknn_k5 + wg2 * y_geo_k5 + ww * y_win_k1
        log_y = np.log(y_3way.clip(EPS))
        for a in [0.90, 0.95, 1.00]:
            for b in [1.50, 1.60, 1.70, 1.80, 1.90]:
                full = sigmoid(a * base_logit + b * log_y)
                auc = macro_auc(file_labels, full)
                if auc > best4['auc']:
                    best4 = {'auc': auc, 'wr': wr, 'wg': wg2, 'ww': ww, 'a': a, 'b': b}
                    if auc > BEST_SO_FAR:
                        print(f"  *** wr={wr} wg={wg2} ww={ww:.2f} a={a} b={b}: {auc:.4f} ***", flush=True)
                    else:
                        print(f"  wr={wr} wg={wg2} a={a} b={b}: {auc:.4f}", flush=True)
print(f"  Best 3-way rknn+geo+win: {best4['auc']:.4f}")
all_results.append(('rknn_geo_win_3way', best4['auc'], best4))

# ══════════════════════════════════════════════════════════════════════════════
# Test 5: RKNN k=7 + win_k2 (larger k for both)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] RKNN k=7 + win_k2...", flush=True)
best5 = {'auc': 0}
for wg in [0.30, 0.35, 0.40, 0.45, 0.50]:
    yb = wg * y_rknn_k7 + (1-wg) * y_win_k2
    log_yb = np.log(yb.clip(EPS))
    for a in [0.85, 0.90, 0.95, 1.00]:
        for b in [1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]:
            full = sigmoid(a * base_logit + b * log_yb)
            auc = macro_auc(file_labels, full)
            if auc > best5['auc']:
                best5 = {'auc': auc, 'wg': wg, 'a': a, 'b': b}
                if auc > BEST_SO_FAR:
                    print(f"  *** wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                else:
                    print(f"  wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
print(f"  Best rknn_k7+win_k2: {best5['auc']:.4f}")
all_results.append(('rknn_k7_win_k2', best5['auc'], best5))

# ══════════════════════════════════════════════════════════════════════════════
# Test 6: RKNN k=5 + win_k3
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] RKNN k=5 + win_k3...", flush=True)
best6 = {'auc': 0}
for wg in [0.30, 0.35, 0.40, 0.45, 0.50]:
    yb = wg * y_rknn_k5 + (1-wg) * y_win_k3
    log_yb = np.log(yb.clip(EPS))
    for a in [0.85, 0.90, 0.95, 1.00]:
        for b in [1.40, 1.50, 1.60, 1.70, 1.80, 1.90]:
            full = sigmoid(a * base_logit + b * log_yb)
            auc = macro_auc(file_labels, full)
            if auc > best6['auc']:
                best6 = {'auc': auc, 'wg': wg, 'a': a, 'b': b}
                if auc > BEST_SO_FAR:
                    print(f"  *** wg={wg} a={a} b={b}: {auc:.4f} ***", flush=True)
                else:
                    print(f"  wg={wg} a={a} b={b}: {auc:.4f}", flush=True)
print(f"  Best rknn_k5+win_k3: {best6['auc']:.4f}")
all_results.append(('rknn_k5_win_k3', best6['auc'], best6))

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"FULL PIPELINE BATCH 1 SUMMARY")
print(f"{'='*60}")
print(f"Baseline RKNN k=5 wg=0.40 a=0.95 b=1.70: {BEST_SO_FAR:.4f}")
for name, auc, cfg in sorted(all_results, key=lambda x: -x[1]):
    marker = " *** NEW BEST ***" if auc > BEST_SO_FAR else ""
    print(f"  {name:30s}: {auc:.4f}{marker}")

# Update results.json if improved
best_full = max(all_results, key=lambda x: x[1])
if best_full[1] > BEST_SO_FAR:
    print(f"\nNEW FULL PIPELINE BEST: {best_full[0]} = {best_full[1]:.4f}")
    # Update json
    with open("outputs/embed_prior_results.json") as f: rd = json.load(f)
    rd['experiments'].append({
        'method': f"full_pipeline_{best_full[0]}",
        'loo_auc': best_full[1],
        'config': best_full[2]
    })
    if best_full[1] > rd.get('best_full', {}).get('auc', 0):
        rd['best_full'] = {'method': best_full[0], 'auc': best_full[1], **best_full[2]}
    with open("outputs/embed_prior_results.json", 'w') as f: json.dump(rd, f, indent=2)
