"""
Build production PKL for sed_species_bridge (NEW BEST = 0.9444).

Method: Weight top-M SS windows for file j by:
  combined_w[m] = perch_sim(j, m) * (1 + beta * max_SED(j.species, window_m))
Then build unnormalized signatures → row-normalize bridge matrix → RKNN k=5 + win_k1 + logspace.

Best config: beta=0.5, alpha=0.5, wg=0.45, a=0.85, b=1.70
"""
import numpy as np, pickle, os, json, shutil
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list = list(perch['file_list'])
n_windows = perch['n_windows']
n_files = len(file_list)
n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end = np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))
file_labels = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_embs_avg = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_embs_avg[fi] = emb_win[s:e].mean(0)
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])] = fi
file_emb_avg_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)

sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
ss_probs = sed_npz['probs'].astype(np.float32)
sed_by_file = {}
for i, rid in enumerate(sed_npz['row_ids']):
    sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]), []).append(i)
file_sed_max = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fb = fname.replace('.ogg','').replace('.flac','')
    if fb in sed_by_file: file_sed_max[fi] = sed_npz['probs'][sed_by_file[fb]].max(0)

EPS = 1e-7
def vlom_blend(a, b, w_a=0.5, w_b=0.5):
    ws=w_a+w_b; w_a/=ws; w_b/=ws
    return sigmoid(w_a*np.log(a.clip(EPS)/(1-a).clip(EPS))+w_b*np.log(b.clip(EPS)/(1-b).clip(EPS)))
def macro_auc(yt, ys):
    mask=yt.sum(0)>0; return roc_auc_score(yt[:,mask],ys[:,mask],average='macro')

base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit = np.log(base_probs.clip(EPS))-np.log((1-base_probs).clip(EPS))

with open("outputs/embed_prior_logspace_geo5_win1.pkl","rb") as f: ep_base=pickle.load(f)
X_ref = ep_base['X_combined_n'].astype(np.float32)
fl = ep_base['file_labels'].astype(np.float32)

print("Loading all soundscape embeddings...", flush=True)
ss_all = np.load("outputs/perch_emb_all_ss.npz", allow_pickle=True)
ss_emb = ss_all['emb'].astype(np.float32)
ss_emb_norm = normalize(ss_emb, norm='l2').astype(np.float32)
n_ss_wins = len(ss_emb_norm)
print(f"  {n_ss_wins} windows.", flush=True)

print("Computing labeled × ss similarity (perch)...", flush=True)
CHUNK = 20000
sim_lab_ss = np.zeros((n_files, n_ss_wins), np.float32)
for cs in range(0, n_ss_wins, CHUNK):
    ce = min(cs+CHUNK, n_ss_wins)
    sim_lab_ss[:, cs:ce] = file_emb_avg_norm @ ss_emb_norm[cs:ce].T
    if cs % 60000 == 0: print(f"    {cs}/{n_ss_wins}...", flush=True)
print("  done.", flush=True)

print("Precomputing win_k1...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s,te_e=int(file_start[i]),int(file_end[i]); X_te=emb_win_norm[te_s:te_e]
    tr_mask=(win_file_id!=i); X_tr=emb_win_norm[tr_mask]; tr_fi=win_file_id[tr_mask]
    sims=X_te@X_tr.T; top_idx=np.argsort(-sims,1)[:,:1]
    wp=np.zeros((te_e-te_s,n_species),np.float32)
    for wi in range(te_e-te_s):
        ww=sims[wi,top_idx[wi]].clip(0); ws_v=ww.sum()
        ww=ww/ws_v if ws_v>1e-8 else np.ones(1)
        wp[wi]=(ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i]=wp.mean(0)
print("  done.", flush=True)

sim_ref = X_ref @ X_ref.T
np.fill_diagonal(sim_ref, -np.inf)

# Best config
beta = 0.50
alpha = 0.50
wg = 0.45
a_best = 0.85
b_best = 1.70
M = 100
k_rknn, T = 5, 0.2

# Build SED-species weighted signatures
print(f"\nBuilding sed_species_bridge (beta={beta}, M={M})...", flush=True)
top_M_idx = np.argsort(-sim_lab_ss, axis=1)[:, :M]
top_M_sims = np.sort(-sim_lab_ss, axis=1)[:, :M] * -1

train_sigs_sp = np.zeros((n_files, 1536), np.float32)
for j in range(n_files):
    sp_j = file_labels[j].astype(bool)
    idx = top_M_idx[j]
    perch_w = top_M_sims[j]
    if sp_j.sum() > 0:
        sp_score = ss_probs[idx][:, sp_j].max(1)
        combined_w = perch_w * (1.0 + beta * sp_score)
    else:
        combined_w = perch_w
    train_sigs_sp[j] = (combined_w[:, None] * ss_emb_norm[idx]).sum(0)
print("  done.", flush=True)

# Compute bridge matrix (unnormalized sigs → row-normalize)
sb = file_emb_avg_norm @ train_sigs_sp.T  # (66, 66)
bridge_norm = np.sqrt((sb**2).sum(1, keepdims=True)).clip(1e-8)
sim_bridge_sp_n = sb / bridge_norm  # (66, 66)

# RKNN
sim_comb = (1-alpha)*sim_ref.copy() + alpha*sim_bridge_sp_n.copy()
sc = sim_comb.copy()
np.fill_diagonal(sc, -np.inf)
top_k = np.argsort(-sc, axis=1)[:, :k_rknn]
kth = sc[np.arange(n_files), top_k[:, -1]]
y_bridge = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    tr = np.array([j for j in range(n_files) if j != i])
    sims_i = sc[i, tr]; top_i = np.argsort(-sims_i)[:k_rknn]
    mutual, msims = [], []
    for ti, tj in enumerate(tr[top_i]):
        if sims_i[top_i[ti]] >= kth[tj]: mutual.append(tj); msims.append(sims_i[top_i[ti]])
    if len(mutual) == 0:
        top5 = np.argsort(-sims_i)[:5]; ls = sims_i[top5]/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
        y_bridge[i] = (w[:,None]*fl[tr[top5]]).sum(0)
    else:
        ms = np.array(msims); ls = ms/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
        y_bridge[i] = (w[:,None]*fl[mutual]).sum(0)

# Full prediction
yb = wg * y_bridge + (1-wg) * y_win_k1
full = sigmoid(a_best * base_logit + b_best * np.log(yb.clip(EPS)))
auc = macro_auc(file_labels, full)
print(f"\nVerification AUC (beta={beta}, alpha={alpha}, wg={wg}, a={a_best}, b={b_best}): {auc:.4f}")
print(f"Expected: ~0.9444")

# Build production PKL
# For production, we need:
# 1. train_sigs_sp (66, 1536) - unnormalized signatures for bridge computation
# 2. sig_norms (66,) - L2 norms of signatures
# 3. sim_bridge_n (66, 66) - precomputed for LOO training
# 4. emb_win_norm, win_file_id, etc. for win_k1

# sig norms for production normalization
sig_norms_sp = np.linalg.norm(train_sigs_sp, axis=1)  # (66,)

pkl_data = dict(ep_base)
pkl_data.update({
    'method': 'sed_species_bridge',
    'type': 'sed_species_bridge',
    # Bridge params
    'bridge_beta': beta,
    'bridge_alpha': alpha,
    'bridge_M': M,
    # RKNN params
    'k_rknn': k_rknn,
    'T_rknn': T,
    # Logspace params
    'logspace_a': a_best,
    'logspace_b': b_best,
    'w_rknn': wg,
    'w_win': 1.0 - wg,
    # Metric
    'loo_auc': auc,
    'full_auc': auc,
    # Production: train_sigs_sp for fast bridge similarity computation
    # sim_bridge_test_j = test_emb_avg_norm @ train_sigs_sp[j] (unnormalized)
    # then normalize by computing all sims and row-normalizing
    'train_ss_signatures': train_sigs_sp,   # (66, 1536), unnormalized
    'train_ss_sig_norm': sig_norms_sp,       # (66,) per-signature L2 norms
    'sim_bridge_n': sim_bridge_sp_n,         # (66, 66) precomputed
    # Labeled file avg embeddings
    'file_emb_avg_norm': file_emb_avg_norm,  # (66, 1536)
    # Window data for win_k1
    'emb_win_norm': emb_win_norm,
    'win_file_id': win_file_id,
    'n_windows': n_windows,
    'file_start': file_start,
    'file_end': file_end,
    'file_list': np.array(file_list),
    'file_labels': file_labels,
    # SED species info needed at inference: file species labels
    # (for test file: use Perch logit max to determine likely species)
    'ss_probs': ss_probs,  # (127896, 234) - SED probs for all SS windows (for offline LOO)
    # NOTE: for online inference, we pre-computed train_sigs_sp which already encodes
    # the species-weighted bridge. For test, we approximate using standard perch-only bridge.
})

OUT = "outputs/embed_prior_sed_species_bridge.pkl"
with open(OUT, 'wb') as f: pickle.dump(pkl_data, f)
sz = os.path.getsize(OUT)/1e6
print(f"\nSaved {OUT} ({sz:.1f} MB)")

WEIGHTS = "birdclef-2026/notebook resource/current_subs/weights"
if os.path.exists(WEIGHTS):
    shutil.copy2(OUT, f"{WEIGHTS}/embed_prior_sed_species_bridge.pkl")
    print(f"Copied to weights/")

# Update embed_prior_results.json
with open("outputs/embed_prior_results.json") as f: rd = json.load(f)
rd['experiments'].append({
    'method': 'sed_species_bridge_production',
    'loo_auc': float(auc),
    'full_auc': float(auc),
    'config': {'beta': beta, 'alpha': alpha, 'wg': wg, 'a': a_best, 'b': b_best, 'M': M}
})
if auc > rd['best'].get('loo_auc', 0):
    rd['best'] = {'method': 'sed_species_bridge', 'loo_auc': float(auc), 'full_auc': float(auc),
                  'beta': beta, 'alpha': alpha, 'wg': wg, 'a': a_best, 'b': b_best}
with open("outputs/embed_prior_results.json", 'w') as f: json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")

print(f"\n{'='*55}")
print(f"PRODUCTION PKL READY: sed_species_bridge")
print(f"  Bridge: (1-{alpha}) × X_ref_sim + {alpha} × sed_species_bridge_n")
print(f"  Beta: {beta} (SED species weight)")
print(f"  Logspace: sigmoid({a_best}×base_logit + {b_best}×log({wg}×rknn + {1-wg}×win))")
print(f"  AUC: {auc:.4f}")
print(f"  PKL: {OUT} ({sz:.1f} MB)")
print(f"\nInference: train_sigs_sp pre-encodes species-weighted bridge.")
print(f"  For test file: compute avg_emb, then bridge via @train_sigs_sp.T + row-norm")
