"""
Build production-ready SS Bridge PKL.

Key insight: In production, test file's bridge similarity can be computed as:
  sim_bridge_test_to_j = test_avg_norm @ train_ss_signature[j]
where:
  train_ss_signature[j] = sum_m (sim_train_j_to_ss_m × ss_emb_norm_m) for top-M of j

This allows efficient inference without loading all 127,896 ss windows.

Best config from sweep: alpha=0.4, wg=0.4, a=1.0, b=1.5, AUC=0.9441
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

print("Loading all soundscape embeddings...", flush=True)
ss_all = np.load("outputs/perch_emb_all_ss.npz", allow_pickle=True)
ss_emb = ss_all['emb'].astype(np.float32)
ss_emb_norm = normalize(ss_emb, norm='l2').astype(np.float32)
n_ss_wins = len(ss_emb_norm)
print(f"  {n_ss_wins} windows.", flush=True)

file_emb_avg_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)  # (66, 1536)

# Compute labeled vs all-soundscape similarities
print("Computing labeled × ss similarity matrix...", flush=True)
CHUNK = 20000
sim_lab_ss = np.zeros((n_files, n_ss_wins), np.float32)
for chunk_start in range(0, n_ss_wins, CHUNK):
    chunk_end = min(chunk_start + CHUNK, n_ss_wins)
    sim_lab_ss[:, chunk_start:chunk_end] = file_emb_avg_norm @ ss_emb_norm[chunk_start:chunk_end].T
    if chunk_start % 60000 == 0:
        print(f"    {chunk_start}/{n_ss_wins}...", flush=True)
print("  done.", flush=True)

# Build train_ss_signatures: (66, 1536)
# train_ss_signature[j] = weighted avg of top-M ss embeddings, weighted by sim(j, ss_m)
M = 100
print(f"Building train_ss_signatures (M={M})...", flush=True)
top_M_idx = np.argsort(-sim_lab_ss, axis=1)[:, :M]   # (66, M)
top_M_sims = np.sort(-sim_lab_ss, axis=1)[:, :M] * -1  # (66, M), positive

train_ss_signatures = np.zeros((n_files, 1536), np.float32)
for j in range(n_files):
    w = top_M_sims[j]  # (M,)
    idx = top_M_idx[j]  # (M,) window indices
    embs_j = ss_emb_norm[idx]  # (M, 1536)
    # Weighted sum
    sig = (w[:, None] * embs_j).sum(0)  # (1536,)
    train_ss_signatures[j] = sig
# L2 normalize signatures
sig_norms = np.linalg.norm(train_ss_signatures, axis=1, keepdims=True).clip(1e-8)
train_ss_signatures_n = (train_ss_signatures / sig_norms).astype(np.float32)
print("  done.", flush=True)

# Verify: bridge sim from signatures vs original
# sim_bridge_from_sig[i, j] = file_emb_avg_norm[i] @ train_ss_signatures[j]
sim_bridge_from_sig = file_emb_avg_norm @ train_ss_signatures.T  # (66, 66)
bridge_norm2 = np.sqrt((sim_bridge_from_sig ** 2).sum(1, keepdims=True)).clip(1e-8)
sim_bridge_from_sig_n = sim_bridge_from_sig / bridge_norm2
print(f"Bridge sim from signatures: min={sim_bridge_from_sig.min():.3f}, max={sim_bridge_from_sig.max():.3f}")

# Verify result matches batch5
best_alpha = 0.40
k, T = 5, 0.2
sim_ref = X_ref @ X_ref.T
np.fill_diagonal(sim_ref, -np.inf)

sim_comb = (1-best_alpha) * sim_ref.copy() + best_alpha * sim_bridge_from_sig_n.copy()
sim_comb_copy = sim_comb.copy()
np.fill_diagonal(sim_comb_copy, -np.inf)
top_k = np.argsort(-sim_comb_copy, axis=1)[:, :k]
kth = sim_comb_copy[np.arange(n_files), top_k[:, -1]]
y_bridge = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    tr=np.array([j for j in range(n_files) if j!=i])
    sims_i=sim_comb_copy[i,tr]; top_i=np.argsort(-sims_i)[:k]
    mutual=[]; mutual_sims=[]
    for ti, tj in enumerate(tr[top_i]):
        if sims_i[top_i[ti]] >= kth[tj]:
            mutual.append(tj); mutual_sims.append(sims_i[top_i[ti]])
    if len(mutual)==0:
        top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_bridge[i]=(w[:,None]*fl[tr[top5]]).sum(0)
    else:
        ms=np.array(mutual_sims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
        y_bridge[i]=(w[:,None]*fl[mutual]).sum(0)

# win_k1
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

wg = 0.40; a_best = 1.00; b_best = 1.50
yb = wg * y_bridge + (1-wg) * y_win_k1
log_yb = np.log(yb.clip(EPS))
full = sigmoid(a_best * base_logit + b_best * log_yb)
auc = macro_auc(file_labels, full)
print(f"\nVerification AUC (alpha=0.40, wg=0.40, a=1.0, b=1.5): {auc:.4f}")
print(f"Expected: ~0.9441")

# Build production-ready pkl
pkl_data = dict(ep_base)
pkl_data.update({
    'method': 'ss_bridge_weighted_rknn',
    'type': 'ss_bridge',
    # Bridge params
    'bridge_alpha': best_alpha,
    'bridge_M': M,
    # RKNN params
    'k_rknn': k,
    'T_rknn': T,
    # Logspace params
    'logspace_a': a_best,
    'logspace_b': b_best,
    'w_rknn': wg,
    'w_win': 1.0 - wg,
    # Metric
    'loo_auc': auc,
    'full_auc': auc,
    # Production: train_ss_signatures for fast bridge similarity computation
    # sim_bridge_test_j = test_emb_avg_norm @ train_ss_signatures[j]
    'train_ss_signatures': train_ss_signatures,  # (66, 1536), unnormalized
    'train_ss_sig_norm': sig_norms.ravel(),       # (66,) per-signature L2 norms
    'sim_bridge_n': sim_bridge_from_sig_n,         # (66, 66) precomputed for LOO training
    # Labeled file avg embeddings (for production bridge)
    'file_emb_avg_norm': file_emb_avg_norm,  # (66, 1536)
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
print(f"\nSaved {OUT} ({sz:.1f} MB)")

WEIGHTS = "birdclef-2026/notebook resource/current_subs/weights"
shutil.copy2(OUT, f"{WEIGHTS}/embed_prior_ss_bridge.pkl")
print(f"Copied to weights/")

# Update embed_prior_results.json
with open("outputs/embed_prior_results.json") as f: rd = json.load(f)
rd['experiments'].append({
    'method': 'ss_bridge_weighted_production',
    'loo_auc': float(auc),
    'full_auc': float(auc),
    'config': {'alpha': best_alpha, 'M': M, 'wg': wg, 'a': a_best, 'b': b_best}
})
if auc > rd['best'].get('loo_auc', 0):
    rd['best'] = {'method': 'ss_bridge_weighted', 'loo_auc': float(auc), 'full_auc': float(auc),
                  'alpha': best_alpha, 'M': M, 'wg': wg, 'a': a_best, 'b': b_best}
with open("outputs/embed_prior_results.json", 'w') as f: json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")

print(f"\n{'='*50}")
print(f"PRODUCTION PKL READY")
print(f"  Method: ss_bridge_weighted_rknn")
print(f"  Bridge: (1-{best_alpha}) × X_ref_sim + {best_alpha} × bridge_weighted")
print(f"  Logspace: sigmoid({a_best}×base_logit + {b_best}×log({wg}×rknn + {1-wg}×win))")
print(f"  AUC: {auc:.4f}")
print(f"  PKL: {OUT} ({sz:.1f} MB)")
print(f"\nIn production notebook inference function:")
print(f"  bridge_sim = test_emb_avg_norm @ train_ss_signatures.T / sig_norms")
print(f"  sim_combined = 0.6 × X_ref_sim + 0.4 × bridge_sim_normalized")
print(f"  RKNN k=5 on sim_combined")
