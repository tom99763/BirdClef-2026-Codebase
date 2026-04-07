"""
Focused experiments around per_species_alpha_knn3 = 0.9026

Methods:
  A) per_species_alpha knn4 (between k3=0.9026 and k5=unknown)
  B) per_species_alpha knn3 with logit_mean instead of logit_max
  C) per_species_alpha knn3 with log_prob_mean (= log(mean(sigmoid(logit_win))))
  D) Ensemble: blend A + knn3_logit_max (the 0.9026 method)
"""
import numpy as np, json, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
logits_win = raw['logits'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

def sigmoid(x): return 1 / (1 + np.exp(-x))

file_embs      = np.zeros((n_files, emb_win.shape[1]),  dtype=np.float32)
file_labels    = np.zeros((n_files, n_species),          dtype=np.float32)
file_logmax    = np.zeros((n_files, n_species),          dtype=np.float32)
file_logmean   = np.zeros((n_files, n_species),          dtype=np.float32)
file_logprobmn = np.zeros((n_files, n_species),          dtype=np.float32)  # log(mean(sigmoid))

idx = 0
for fi, nw in enumerate(n_windows):
    sl = slice(idx, idx+nw)
    file_embs[fi]      = emb_win[sl].mean(0)
    file_labels[fi]    = (labels_win[sl].max(0) > 0.5).astype(np.float32)
    file_logmax[fi]    = logits_win[sl].max(0)
    file_logmean[fi]   = logits_win[sl].mean(0)
    # log(mean(sigmoid(logit))) — smooth probability aggregation
    probs = sigmoid(logits_win[sl])
    file_logprobmn[fi] = np.log(probs.mean(0) + 1e-7)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
print(f"Files={n_files}, species={n_species}\n", flush=True)

# ── Precompute KNN (k=3,4,5) ─────────────────────────────────────────
print("Precomputing KNN scores...", flush=True)
SIM = file_embs_norm @ file_embs_norm.T
np.fill_diagonal(SIM, -2.0)

def knn_mat(k):
    out = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        sims_i = SIM[i].copy()
        top = np.argsort(-sims_i)[:k]
        w = sims_i[top].clip(0); w_sum = w.sum()
        if w_sum < 1e-8: w = np.ones(k)/k
        else: w /= w_sum
        out[i] = (w[:, None] * file_labels[top]).sum(0)
    return out

KNN = {k: knn_mat(k) for k in [3, 4, 5]}
print("  done.\n", flush=True)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

BEST = 0.9026
ALPHA_GRID = np.linspace(0, 1, 101)  # finer grid
results = {}

def ps_alpha_loo(logit_feat, knn_k, label=''):
    """Generic per-species alpha LOO-CV: blend logit_feat and KNN[knn_k]."""
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        tr = np.arange(n_files)[np.arange(n_files) != i]
        for s in range(n_species):
            y_s   = file_labels[tr, s]
            n_pos = int(y_s.sum())
            if n_pos < 2 or (len(y_s) - n_pos) < 2:
                preds[i, s] = np.clip(logit_feat[i, s], 0, 1)
                continue
            lf_tr = logit_feat[tr, s]
            kn_tr = KNN[knn_k][tr, s]
            best_a, best_auc = 0.5, -1
            for a in ALPHA_GRID:
                bl = a * lf_tr + (1-a) * kn_tr
                try:
                    v = roc_auc_score(y_s, bl)
                    if v > best_auc: best_auc, best_a = v, a
                except: pass
            preds[i, s] = np.clip(
                best_a * logit_feat[i, s] + (1-best_a) * KNN[knn_k][i, s], 0, 1)
    return preds

# ── A) per_species_alpha knn4 ─────────────────────────────────────────
print("A) per_species_alpha with knn4 (finer 101-pt grid)", flush=True)
p_k4 = ps_alpha_loo(file_logmax, 4, 'knn4')
a_k4 = macro_auc(file_labels, p_k4)
print(f"  knn4: {a_k4:.4f}  (Δ={a_k4-BEST:+.4f})", flush=True)
results['ps_alpha_knn4_lmax'] = a_k4

# ── B) per_species_alpha knn5 ─────────────────────────────────────────
print("B) per_species_alpha with knn5", flush=True)
p_k5 = ps_alpha_loo(file_logmax, 5, 'knn5')
a_k5 = macro_auc(file_labels, p_k5)
print(f"  knn5: {a_k5:.4f}  (Δ={a_k5-BEST:+.4f})", flush=True)
results['ps_alpha_knn5_lmax'] = a_k5

# ── C) per_species_alpha knn3 + logit_mean ────────────────────────────
print("C) per_species_alpha knn3 + logit_MEAN", flush=True)
p_lm = ps_alpha_loo(file_logmean, 3, 'knn3_logmean')
a_lm = macro_auc(file_labels, p_lm)
print(f"  knn3+logmean: {a_lm:.4f}  (Δ={a_lm-BEST:+.4f})", flush=True)
results['ps_alpha_knn3_logmean'] = a_lm

# ── D) per_species_alpha knn3 + log_prob_mean ─────────────────────────
print("D) per_species_alpha knn3 + log(mean(sigmoid(logit)))", flush=True)
p_lp = ps_alpha_loo(file_logprobmn, 3, 'knn3_logprobmn')
a_lp = macro_auc(file_labels, p_lp)
print(f"  knn3+logprobmn: {a_lp:.4f}  (Δ={a_lp-BEST:+.4f})", flush=True)
results['ps_alpha_knn3_logprobmn'] = a_lp

# ── E) Ensemble: logit_max_knn3 pred + logit_mean_knn3 pred ──────────
print("E) Ensemble: logit_max_knn3 (0.9026) + logit_mean_knn3", flush=True)
# We need the knn3+logmax predictions first → re-run with logmax
p_lx = ps_alpha_loo(file_logmax, 3, 'knn3_logmax_rerun')
for w in [0.5, 0.7, 0.8, 0.9]:
    ens = w * p_lx + (1-w) * p_lm
    a = macro_auc(file_labels, ens)
    nm = f'ens_lmax{w:.1f}_lmean{1-w:.1f}_k3'
    print(f"  {nm}: {a:.4f}  (Δ={a-BEST:+.4f})", flush=True)
    results[nm] = a

# ── SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {name:<40s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

best_auc = data['best']['loo_auc']
new_best = None
for name, auc in results.items():
    data['experiments'].append({'method': name, 'loo_auc': round(auc, 6), 'features': 'emb+logit'})
    if auc > best_auc:
        best_auc = auc; new_best = name
        data['best'] = {'method': name, 'loo_auc': round(auc, 6), 'note': 'NEW BEST'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

if new_best:
    print(f"\n*** NEW BEST: {new_best} AUC={best_auc:.4f} ***")
else:
    print(f"\n未超越 best ({BEST:.4f})，per_species_alpha_knn3 仍是最佳。")
print("done", flush=True)
