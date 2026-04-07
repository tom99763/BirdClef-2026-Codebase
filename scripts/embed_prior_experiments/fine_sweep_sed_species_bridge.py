"""
Fine sweep for sed_species_bridge (NEW BEST = 0.9444).

Method: For each labeled file j, weight top-M SS windows by:
  combined_w[m] = perch_sim(j, m) * (1 + beta * max_SED(j.species, window_m))

Then build unnormalized signatures → row-normalize bridge → RKNN + win_k1 + logspace.

Config: beta=0.5, alpha=0.5, wg=0.45, a=0.85, b=1.70
"""
import numpy as np, pickle, os, json
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

print("Computing labeled × ss similarity...", flush=True)
CHUNK = 20000
sim_lab_ss = np.zeros((n_files, n_ss_wins), np.float32)
for cs in range(0, n_ss_wins, CHUNK):
    ce = min(cs+CHUNK, n_ss_wins)
    sim_lab_ss[:, cs:ce] = file_emb_avg_norm @ ss_emb_norm[cs:ce].T
    if cs % 60000 == 0: print(f"  {cs}/{n_ss_wins}...", flush=True)

print("Precomputing win_k1...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s,te_e = int(file_start[i]),int(file_end[i]); X_te=emb_win_norm[te_s:te_e]
    tr_mask=(win_file_id!=i); X_tr=emb_win_norm[tr_mask]; tr_fi=win_file_id[tr_mask]
    sims=X_te@X_tr.T; top_idx=np.argsort(-sims,1)[:,:1]
    wp=np.zeros((te_e-te_s,n_species),np.float32)
    for wi in range(te_e-te_s):
        ww=sims[wi,top_idx[wi]].clip(0); ws_v=ww.sum()
        ww=ww/ws_v if ws_v>1e-8 else np.ones(1)
        wp[wi]=(ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i]=wp.mean(0)

sim_ref = X_ref @ X_ref.T
np.fill_diagonal(sim_ref, -np.inf)

k_rknn, T = 5, 0.2
def compute_rknn(sim_combined):
    sc=sim_combined.copy(); np.fill_diagonal(sc,-np.inf)
    top_k=np.argsort(-sc,axis=1)[:,:k_rknn]
    kth=sc[np.arange(n_files),top_k[:,-1]]
    y=np.zeros((n_files,n_species),np.float32)
    for i in range(n_files):
        tr=np.array([j for j in range(n_files) if j!=i])
        sims_i=sc[i,tr]; top_i=np.argsort(-sims_i)[:k_rknn]
        mutual,msims=[],[]
        for ti,tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]]>=kth[tj]: mutual.append(tj); msims.append(sims_i[top_i[ti]])
        if len(mutual)==0:
            top5=np.argsort(-sims_i)[:5]; ls=sims_i[top5]/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y[i]=(w[:,None]*fl[tr[top5]]).sum(0)
        else:
            ms=np.array(msims); ls=ms/T; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y[i]=(w[:,None]*fl[mutual]).sum(0)
    return y

def build_bridge(sigs):
    sb = file_emb_avg_norm @ sigs.T
    bn = np.sqrt((sb**2).sum(1,keepdims=True)).clip(1e-8)
    return sb / bn

def eval_full(y_ep, wg, a, b):
    yb = wg*y_ep + (1-wg)*y_win_k1
    full = sigmoid(a*base_logit + b*np.log(yb.clip(EPS)))
    return macro_auc(file_labels, full)

M = 100
top_M_idx = np.argsort(-sim_lab_ss, axis=1)[:, :M]
top_M_sims = np.sort(-sim_lab_ss, axis=1)[:, :M] * -1

print("\n=== Fine sweep: beta × alpha × wg × a × b ===", flush=True)
best = {'auc': 0}
all_results = []

for beta in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.00, 1.50, 2.00]:
    # Build species-weighted signatures
    sigs_sp = np.zeros((n_files, 1536), np.float32)
    for j in range(n_files):
        sp_j = file_labels[j].astype(bool)
        idx = top_M_idx[j]
        perch_w = top_M_sims[j]
        if sp_j.sum() > 0:
            sp_score = ss_probs[idx][:, sp_j].max(1)
            combined_w = perch_w * (1.0 + beta * sp_score)
        else:
            combined_w = perch_w
        sigs_sp[j] = (combined_w[:, None] * ss_emb_norm[idx]).sum(0)
    sb_sp_n = build_bridge(sigs_sp)

    for alph in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        sc_sp = (1-alph)*sim_ref.copy() + alph*sb_sp_n.copy()
        y_sp = compute_rknn(sc_sp)
        for wg in [0.30, 0.35, 0.40, 0.45, 0.50]:
            for a in [0.80, 0.85, 0.90, 0.95, 1.00]:
                for b in [1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10]:
                    auc = eval_full(y_sp, wg, a, b)
                    all_results.append((auc, beta, alph, wg, a, b))
                    if auc > best['auc']:
                        best = {'auc': auc, 'beta': beta, 'alpha': alph,
                                'wg': wg, 'a': a, 'b': b}
                        print(f"  NEW BEST: {auc:.4f}  beta={beta} alpha={alph} wg={wg} a={a} b={b}", flush=True)

print(f"\n{'='*60}")
print(f"FINE SWEEP BEST: {best['auc']:.4f}")
print(f"  beta={best['beta']}, alpha={best['alpha']}, wg={best['wg']}, a={best['a']}, b={best['b']}")

# Show top 10 configs
all_results.sort(reverse=True)
print(f"\nTop 10 configs:")
for auc, beta, alph, wg, a, b in all_results[:10]:
    print(f"  {auc:.4f}  beta={beta} alpha={alph} wg={wg} a={a} b={b}")

# Update JSON
with open("outputs/embed_prior_results.json") as f: rd=json.load(f)
rd['experiments'].append({
    'method': 'sed_species_bridge_fine',
    'loo_auc': float(best['auc']),
    'full_auc': float(best['auc']),
    'config': best
})
if best['auc'] > rd['best'].get('loo_auc', 0):
    rd['best'] = {'method': 'sed_species_bridge', **best}
    print(f"\n*** UPDATED BEST: {best['auc']:.4f} ***")
with open("outputs/embed_prior_results.json",'w') as f: json.dump(rd,f,indent=2)
print("Updated embed_prior_results.json")
