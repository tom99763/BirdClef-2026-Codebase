"""
Batch 62b: NT-Xent fast - reduced sweep, only most promising params
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[file_start[fi]:file_end[fi]] = fi

file_labels = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

EPS = 1e-7; mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9873

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def wl_ntxent(emb_wins_n, tau=0.5, k_neg=50, w_max_agg=0.90):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins_all = emb_wins_n[tr_mask]
        tr_lab_win_raw = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win_raw[:,si] > 0.5
            neg_win_mask = tr_lab_win_raw[:,si] < 0.1
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T
            best_pos = pos_sims.max(1)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg_sims = -np.sort(-neg_sims, axis=1)[:, :k_act]
                all_sims = np.concatenate([best_pos[:, None], top_neg_sims], axis=1) / tau
                log_denom = logsumexp(all_sims, axis=1)
                ws[:,si] = np.exp(best_pos / tau - log_denom)
            else:
                ws[:,si] = 1.0
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

print("Precomputing...", flush=True)
ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica100 = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
pca80 = PCA(n_components=80, random_state=42)
ew80 = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
ew80s = normalize(pca80s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
print("Done.", flush=True)

# Fast sweep: only best-guess params
print("\n=== NT-Xent fast sweep ===", flush=True)
t0 = time.time()
best_ntxent = 0; best_cfg_ntxent = None
for tau in [0.2, 0.3, 0.5, 1.0, 2.0]:
    for k_neg in [50, 80, 100]:
        for wma in [0.88, 0.90, 0.92]:
            for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80'), (ew80s, 'std80')]:
                out = wl_ntxent(emb, tau=tau, k_neg=k_neg, w_max_agg=wma)
                auc = eval_loo(out)
                if auc > best_ntxent: best_ntxent = auc; best_cfg_ntxent = (name, tau, k_neg, wma)
print(f"  NT-Xent: {best_ntxent:.4f}  cfg={best_cfg_ntxent}  ({time.time()-t0:.0f}s)", flush=True)
results['ntxent_fast'] = best_ntxent
flag = " *** NEW BEST ***" if best_ntxent > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# LogSumExp agg fast
def wl_lse_agg(emb_wins_n, k_neg=50, w_max_pos=0.80, temp_agg=3.0, w_lse=0.5):
    from sklearn.metrics import roc_auc_score
    from scipy.special import logsumexp as lse
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins_all = emb_wins_n[tr_mask]
        tr_lab_win_raw = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win_raw[:,si] > 0.5
            neg_win_mask = tr_lab_win_raw[:,si] < 0.1
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp = w_max_pos * pos_sims.max(1) + (1-w_max_pos) * (te_wins @ pp_mean)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        # LSE agg
        lse_score = lse(ws * temp_agg, axis=0) / temp_agg
        mean_score = ws.mean(0)
        max_score = ws.max(0)
        out[fi] = w_lse * lse_score + (1-w_lse) * mean_score
    return out

print("\n=== LogSumExp agg fast ===", flush=True)
t0 = time.time()
best_lse = 0; best_cfg_lse = None
for k_neg in [50, 80]:
    for wmp in [0.75, 0.80]:
        for temp in [2.0, 3.0, 5.0, 10.0, 20.0]:
            for w_lse in [0.3, 0.5, 0.7]:
                for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80')]:
                    out = wl_lse_agg(emb, k_neg=k_neg, w_max_pos=wmp, temp_agg=temp, w_lse=w_lse)
                    auc = eval_loo(out)
                    if auc > best_lse: best_lse = auc; best_cfg_lse = (name, k_neg, wmp, temp, w_lse)
print(f"  LSE agg: {best_lse:.4f}  cfg={best_cfg_lse}  ({time.time()-t0:.0f}s)", flush=True)
results['lse_agg_fast'] = best_lse
flag = " *** NEW BEST ***" if best_lse > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

print("\n=== Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1])[:10]:
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best_json:
        cur_best_json = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
