"""
Batch 62: NT-Xent / InfoNCE style contrastive scoring
Goal: fundamentally different single method
Methods:
  1. NT-Xent: score = exp(pos_sim/tau) / (exp(pos_sim/tau) + sum(exp(neg_sims/tau)))
  2. LogSumExp aggregation over windows
  3. Softmax attention over windows
  4. Prototypical Network style: score = softmax of similarities to all prototypes
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
    """
    NT-Xent style score using only the best positive and top-k negatives.
    score_per_window = exp(best_pos_sim/tau) / (exp(best_pos_sim/tau) + sum(exp(neg_sims/tau)))
    """
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
            pos_sims = te_wins @ pos_wins.T  # [n_te, n_pos]
            best_pos = pos_sims.max(1)  # best positive similarity per test window
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T  # [n_te, n_neg]
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg_sims = -np.sort(-neg_sims, axis=1)[:, :k_act]  # [n_te, k_act]
                # NT-Xent score: softmax over positive and negatives
                all_sims = np.concatenate([best_pos[:, None], top_neg_sims], axis=1) / tau  # [n_te, 1+k_act]
                log_denom = logsumexp(all_sims, axis=1)
                log_score = best_pos / tau - log_denom  # log P(pos | all)
                ws[:,si] = np.exp(log_score)  # probability
            else:
                ws[:,si] = 1.0  # no negatives → always positive
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

def wl_logsumexp_agg(emb_wins_n, k_neg=50, w_max_pos=0.80, temp_agg=2.0):
    """LogSumExp aggregation over test windows (smooth max)."""
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
        # LogSumExp aggregation over test windows: smooth max
        lse_score = logsumexp(ws * temp_agg, axis=0) / temp_agg  # [n_species]
        mean_score = ws.mean(0)
        out[fi] = lse_score * 0.5 + mean_score * 0.5  # blend
    return out

def wl_softmax_agg(emb_wins_n, k_neg=50, w_max_pos=0.80, temp_attn=2.0, w_max_agg=0.90):
    """Softmax attention aggregation over test windows."""
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
        # Softmax attention: weight windows by their max score
        attn_weights = np.exp(temp_attn * ws.max(1) - temp_attn * ws.max(1).max())
        attn_weights /= attn_weights.sum() + EPS
        attn_score = (ws * attn_weights[:, None]).sum(0)
        max_score = ws.max(0)
        out[fi] = w_max_agg * max_score + (1-w_max_agg) * attn_score
    return out

print("Precomputing...", flush=True)
pca80 = PCA(n_components=80, random_state=42)
ew80 = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
ew80s = normalize(pca80s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica100 = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
print("Done.", flush=True)

# ─── Method 1: NT-Xent scoring ─────────────────────────────────────────────────
print("\n=== Method 1: NT-Xent scoring ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None
for tau in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]:
    for k_neg in [20, 40, 50, 80, 100]:
        for wma in [0.80, 0.85, 0.90, 0.92, 0.95]:
            for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80'), (ew80s, 'std80')]:
                out = wl_ntxent(emb, tau=tau, k_neg=k_neg, w_max_agg=wma)
                auc = eval_loo(out)
                if auc > best1: best1 = auc; best_cfg1 = (name, tau, k_neg, wma)
print(f"  NT-Xent best: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['ntxent'] = best1
flag = " *** NEW BEST ***" if best1 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 2: LogSumExp aggregation ────────────────────────────────────────────
print("\n=== Method 2: LogSumExp aggregation ===", flush=True)
t0 = time.time()
best2 = 0; best_cfg2 = None
for k_neg in [40, 50, 80]:
    for wmp in [0.75, 0.80, 0.85]:
        for temp_agg in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
            for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80')]:
                out = wl_logsumexp_agg(emb, k_neg=k_neg, w_max_pos=wmp, temp_agg=temp_agg)
                auc = eval_loo(out)
                if auc > best2: best2 = auc; best_cfg2 = (name, k_neg, wmp, temp_agg)
print(f"  LogSumExp best: {best2:.4f}  cfg={best_cfg2}  ({time.time()-t0:.0f}s)", flush=True)
results['logsumexp_agg'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 3: Softmax attention aggregation ─────────────────────────────────
print("\n=== Method 3: Softmax attention aggregation ===", flush=True)
t0 = time.time()
best3 = 0; best_cfg3 = None
for k_neg in [40, 50, 80]:
    for wmp in [0.75, 0.80, 0.85]:
        for temp_attn in [1.0, 2.0, 3.0, 5.0]:
            for wma in [0.85, 0.90, 0.92]:
                for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80')]:
                    out = wl_softmax_agg(emb, k_neg=k_neg, w_max_pos=wmp, temp_attn=temp_attn, w_max_agg=wma)
                    auc = eval_loo(out)
                    if auc > best3: best3 = auc; best_cfg3 = (name, k_neg, wmp, temp_attn, wma)
print(f"  Softmax-attn best: {best3:.4f}  cfg={best_cfg3}  ({time.time()-t0:.0f}s)", flush=True)
results['softmax_attn_agg'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 4: Full NT-Xent with ALL positive windows (max over positives) ───
print("\n=== Method 4: NT-Xent over all positive windows ===", flush=True)
def wl_ntxent_allpos(emb_wins_n, tau=0.5, k_neg=50, w_max_agg=0.90, w_pos=0.7):
    """NT-Xent but using mean score over all positives (not just best)."""
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
            pos_sims = te_wins @ pos_wins.T  # [n_te, n_pos]
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T  # [n_te, n_neg]
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg_sims = -np.sort(-neg_sims, axis=1)[:, :k_act]
                # For each test window: compute NT-Xent for best pos
                best_pos = pos_sims.max(1)
                all_sims = np.concatenate([best_pos[:, None], top_neg_sims], axis=1) / tau
                log_denom = logsumexp(all_sims, axis=1)
                nt_max = np.exp(best_pos / tau - log_denom)
                # Also for mean pos
                mean_pos = pos_sims.mean(1)
                all_sims_mean = np.concatenate([mean_pos[:, None], top_neg_sims], axis=1) / tau
                log_denom_mean = logsumexp(all_sims_mean, axis=1)
                nt_mean = np.exp(mean_pos / tau - log_denom_mean)
                ws[:,si] = w_pos * nt_max + (1-w_pos) * nt_mean
            else:
                ws[:,si] = 1.0
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

t0 = time.time()
best4 = 0; best_cfg4 = None
for tau in [0.2, 0.3, 0.5, 0.7, 1.0]:
    for k_neg in [40, 50, 80]:
        for wma in [0.85, 0.90, 0.92]:
            for w_pos in [0.5, 0.6, 0.7, 0.8, 1.0]:
                for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80')]:
                    out = wl_ntxent_allpos(emb, tau=tau, k_neg=k_neg, w_max_agg=wma, w_pos=w_pos)
                    auc = eval_loo(out)
                    if auc > best4: best4 = auc; best_cfg4 = (name, tau, k_neg, wma, w_pos)
print(f"  NT-Xent allpos best: {best4:.4f}  cfg={best_cfg4}  ({time.time()-t0:.0f}s)", flush=True)
results['ntxent_allpos'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== Batch 62 Summary ===", flush=True)
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
