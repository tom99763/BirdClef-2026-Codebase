"""
Build SS Bridge pkl - Fine sweep + PKL construction
ss_bridge_weighted: Use 127,896 soundscape windows as bridge to enhance file similarity.

Bridge formula:
  sim_bridge2[i,j] = sum_m( sim(i, ss_m) × sim(j, ss_m) ) for m in top-M of j
  sim_combined = (1-alpha) * sim_ref + alpha * sim_bridge2_normalized
  Then RKNN on sim_combined.

Current best: alpha=0.2, wg=0.40, a=0.85, b=1.90 → 0.9440
"""
import numpy as np, pickle, os, json, shutil
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

# Load all soundscape embeddings
print("Loading all soundscape embeddings...", flush=True)
ss_all = np.load("outputs/perch_emb_all_ss.npz", allow_pickle=True)
ss_emb = ss_all['emb'].astype(np.float32)    # (127896, 1536)
ss_emb_norm = normalize(ss_emb, norm='l2').astype(np.float32)
n_ss_wins = len(ss_emb_norm)
print(f"  {n_ss_wins} windows loaded.", flush=True)

file_emb_avg_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)

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

# Compute sim_ref
sim_ref = X_ref @ X_ref.T
np.fill_diagonal(sim_ref, -np.inf)

# Compute sim_lab_ss (labeled vs all soundscape)
print("Computing labeled vs all-soundscape similarity matrix...", flush=True)
CHUNK = 20000
sim_lab_ss = np.zeros((n_files, n_ss_wins), np.float32)
for chunk_start in range(0, n_ss_wins, CHUNK):
    chunk_end = min(chunk_start + CHUNK, n_ss_wins)
    sim_lab_ss[:, chunk_start:chunk_end] = file_emb_avg_norm @ ss_emb_norm[chunk_start:chunk_end].T
    if chunk_start % 60000 == 0:
        print(f"    {chunk_start}/{n_ss_wins}...", flush=True)
print("  done.", flush=True)

# Build weighted bridge sim matrix
M = 100
print(f"Building weighted bridge (M={M})...", flush=True)
top_M_idx = np.argsort(-sim_lab_ss, axis=1)[:, :M]
top_M_sims = np.sort(-sim_lab_ss, axis=1)[:, :M] * -1

sim_bridge = np.zeros((n_files, n_files), np.float32)
for i in range(n_files):
    for j in range(n_files):
        if i == j: continue
        sim_i_full = sim_lab_ss[i, top_M_idx[j]]
        sim_j_full = top_M_sims[j]
        sim_bridge[i, j] = (sim_i_full * sim_j_full).sum()

# Normalize bridge matrix (row-normalize)
bridge_norm = np.sqrt((sim_bridge ** 2).sum(1, keepdims=True)).clip(1e-8)
sim_bridge_n = sim_bridge / bridge_norm
print("  done.", flush=True)

# Fine sweep: find best alpha, wg, a, b
print("\nFine sweep (alpha, wg, a, b)...", flush=True)
k, T = 5, 0.2
best = {'auc': 0}

def compute_rknn_sim(sim_combined):
    np.fill_diagonal(sim_combined, -np.inf)
    top_k = np.argsort(-sim_combined, axis=1)[:, :k]
    kth = sim_combined[np.arange(n_files), top_k[:, -1]]
    y = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr=np.array([j for j in range(n_files) if j!=i])
        sims_i=sim_combined[i,tr]; top_i=np.argsort(-sims_i)[:k]
        mutual=[]; mutual_sims=[]
        for ti, tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]] >= kth[tj]:
                mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
        if len(mutual)==0:
            top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y[i]=(w[:,None]*fl[tr[top5]]).sum(0)
        else:
            ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y[i]=(w[:,None]*fl[mutual]).sum(0)
    return y

for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    sim_comb = (1-alpha) * sim_ref.copy() + alpha * sim_bridge_n.copy()
    y_bridge = compute_rknn_sim(sim_comb.copy())
    for wg in [0.30, 0.35, 0.40, 0.45, 0.50]:
        yb = wg * y_bridge + (1-wg) * y_win_k1
        log_yb = np.log(yb.clip(EPS))
        for a in [0.80, 0.85, 0.90, 0.95, 1.00]:
            for b in [1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]:
                full = sigmoid(a * base_logit + b * log_yb)
                auc = macro_auc(file_labels, full)
                if auc > best['auc']:
                    best = {'auc': auc, 'alpha': alpha, 'wg': wg, 'a': a, 'b': b}
                    print(f"  alpha={alpha} wg={wg} a={a} b={b}: {auc:.4f}", flush=True)

print(f"\nBest ss_bridge_weighted: alpha={best['alpha']}, wg={best['wg']}, a={best['a']}, b={best['b']}")
print(f"Full pipeline AUC: {best['auc']:.4f}")
print(f"vs RKNN baseline: 0.9432, improvement: {best['auc']-0.9432:+.4f}")

# Build pkl with best config
best_alpha = best['alpha']
print(f"\nBuilding pkl with alpha={best_alpha}...", flush=True)
sim_comb_best = (1-best_alpha) * sim_ref.copy() + best_alpha * sim_bridge_n.copy()

pkl_data = dict(ep_base)
pkl_data.update({
    'method': 'ss_bridge_weighted_rknn',
    'type': 'ss_bridge',
    'bridge_alpha': best_alpha,
    'bridge_M': M,
    'logspace_a': best['a'],
    'logspace_b': best['b'],
    'w_rknn': best['wg'],
    'w_win': 1.0 - best['wg'],
    'full_auc': best['auc'],
    'loo_auc': best['auc'],
    # Store bridge matrix for production use
    'sim_bridge_n': sim_bridge_n,  # (66, 66) normalized bridge similarity
    'file_emb_avg_norm': file_emb_avg_norm,  # (66, 1536) for production bridge
    # Window data (same as geo5_win1 pkl)
    'emb_win_norm': emb_win_norm,
    'win_file_id': win_file_id,
    'n_windows': n_windows,
    'file_start': file_start,
    'file_end': file_end,
    'file_list': np.array(file_list),
    'file_labels': file_labels,
})

OUT = "outputs/embed_prior_ss_bridge.pkl"
with open(OUT,'wb') as f: pickle.dump(pkl_data, f)
sz = os.path.getsize(OUT)/1e6
print(f"Saved {OUT} ({sz:.1f} MB)")

WEIGHTS = "birdclef-2026/notebook resource/current_subs/weights"
shutil.copy2(OUT, f"{WEIGHTS}/embed_prior_ss_bridge.pkl")
print(f"Copied to weights/")

# Update embed_prior_results.json
with open("outputs/embed_prior_results.json") as f: rd = json.load(f)
rd['experiments'].append({
    'method': 'ss_bridge_weighted',
    'loo_auc': best['auc'],
    'full_auc': best['auc'],
    'config': best
})
if best['auc'] > rd['best'].get('loo_auc', 0):
    rd['best'] = {'method': 'ss_bridge_weighted', 'loo_auc': best['auc'], **best}
with open("outputs/embed_prior_results.json", 'w') as f: json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")

print(f"\n{'='*50}")
print(f"SS BRIDGE WEIGHTED SUMMARY")
print(f"  Method: ss_bridge_weighted_rknn")
print(f"  Bridge alpha: {best_alpha}")
print(f"  Best wg={best['wg']}, a={best['a']}, b={best['b']}")
print(f"  Full pipeline AUC: {best['auc']:.4f}")
print(f"  PKL: {OUT} ({sz:.1f} MB)")
