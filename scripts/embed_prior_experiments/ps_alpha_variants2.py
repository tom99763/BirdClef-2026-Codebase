"""
Per-species alpha variants — FIXED: use sigmoid(logit) NOT raw logit
Key insight: original 0.9026 used file_prob_max = sigmoid(logit_max)
Confirmed from logit_fusion_v3.py line 174: tr_logit = file_prob_max[mask]

Methods:
  A) per_species_alpha knn4 + prob_max  (original knn3=0.9026)
  B) per_species_alpha knn5 + prob_max
  C) per_species_alpha knn3 + prob_mean (sigmoid(logit_mean))
  D) Ensemble: (A knn3 rerun) + (C prob_mean knn3)
"""
import numpy as np, json, os
import scipy.special
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

file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_mean= np.zeros((n_files, n_species),         dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    sl = slice(idx, idx+nw)
    file_embs[fi]       = emb_win[sl].mean(0)
    file_labels[fi]     = (labels_win[sl].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi]  = logits_win[sl].max(0)
    file_logit_mean[fi] = logits_win[sl].mean(0)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
# *** KEY FIX: use sigmoid(logit) like original logit_fusion_v3.py ***
file_prob_max  = scipy.special.expit(file_logit_max)   # [0,1] range
file_prob_mean = scipy.special.expit(file_logit_mean)  # [0,1] range
print(f"Files={n_files}, species={n_species}\n", flush=True)

# Precompute KNN
print("Precomputing KNN...", flush=True)
SIM = file_embs_norm @ file_embs_norm.T
np.fill_diagonal(SIM, -2.0)

def knn_predict(k):
    out = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        sims_i = SIM[i].copy()
        mask = np.ones(n_files, bool); mask[i] = False
        nn = np.where(mask)[0]
        top_local = np.argsort(-sims_i[mask])[:k]
        top = nn[top_local]
        w = sims_i[top].clip(0); w_sum = w.sum()
        if w_sum < 1e-9: w = np.ones(k)/k
        else: w /= w_sum
        out[i] = (w[:, None] * file_labels[top]).sum(0)
    return out

KNN = {k: knn_predict(k) for k in [3, 4, 5]}
print("  done.\n", flush=True)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

BEST = 0.9026
ALPHA_GRID = np.arange(0.0, 1.01, 0.1)   # same as original
results = {}

def ps_alpha_loo(prob_feat, knn_k):
    """Per-species alpha LOO-CV: blend prob_feat [0,1] and KNN[knn_k]."""
    knn_all = KNN[knn_k]
    preds   = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        tr_knn    = knn_all[mask]        # (65, 234)
        tr_logit  = prob_feat[mask]      # (65, 234)
        tr_labels = file_labels[mask]    # (65, 234)

        for s in range(n_species):
            y_s = tr_labels[:, s]
            if y_s.sum() == 0:
                preds[i, s] = 0.0; continue
            if y_s.sum() == 65:
                preds[i, s] = 1.0; continue

            best_alpha_s, best_inner_auc = 0.30, -1.0
            for a in ALPHA_GRID:
                bl = a * tr_logit[:, s] + (1-a) * tr_knn[:, s]
                try:
                    v = roc_auc_score(y_s, bl)
                    if v > best_inner_auc:
                        best_inner_auc, best_alpha_s = v, a
                except: pass

            preds[i, s] = float(best_alpha_s * prob_feat[i, s] +
                                 (1-best_alpha_s) * knn_all[i, s])
    return preds

# ── A) knn3 rerun (should reproduce ~0.9026) ──────────────────────────
print("A) knn3 + prob_max (should match 0.9026)", flush=True)
p_k3 = ps_alpha_loo(file_prob_max, 3)
a_k3 = macro_auc(file_labels, p_k3)
print(f"  knn3 + prob_max: {a_k3:.4f}  (Δ={a_k3-BEST:+.4f})", flush=True)
results['ps_alpha_knn3_probmax_rerun'] = a_k3

# ── B) knn4 + prob_max ────────────────────────────────────────────────
print("B) knn4 + prob_max", flush=True)
p_k4 = ps_alpha_loo(file_prob_max, 4)
a_k4 = macro_auc(file_labels, p_k4)
print(f"  knn4 + prob_max: {a_k4:.4f}  (Δ={a_k4-BEST:+.4f})", flush=True)
results['ps_alpha_knn4_probmax'] = a_k4

# ── C) knn5 + prob_max ────────────────────────────────────────────────
print("C) knn5 + prob_max", flush=True)
p_k5 = ps_alpha_loo(file_prob_max, 5)
a_k5 = macro_auc(file_labels, p_k5)
print(f"  knn5 + prob_max: {a_k5:.4f}  (Δ={a_k5-BEST:+.4f})", flush=True)
results['ps_alpha_knn5_probmax'] = a_k5

# ── D) knn3 + prob_mean ───────────────────────────────────────────────
print("D) knn3 + prob_mean (sigmoid of logit_mean)", flush=True)
p_lm = ps_alpha_loo(file_prob_mean, 3)
a_lm = macro_auc(file_labels, p_lm)
print(f"  knn3 + prob_mean: {a_lm:.4f}  (Δ={a_lm-BEST:+.4f})", flush=True)
results['ps_alpha_knn3_probmean'] = a_lm

# ── E) Ensemble: prob_max_knn3 + prob_mean_knn3 ───────────────────────
print("E) Ensemble: prob_max_k3 + prob_mean_k3", flush=True)
for w in [0.6, 0.7, 0.8, 0.9]:
    ens = w * p_k3 + (1-w) * p_lm
    a   = macro_auc(file_labels, ens)
    nm  = f'ens_pmax{w:.1f}_pmean{1-w:.1f}_k3'
    print(f"  {nm}: {a:.4f}  (Δ={a-BEST:+.4f})", flush=True)
    results[nm] = a

# ── F) Ensemble: knn3 + knn4 prob_max ─────────────────────────────────
print("F) Ensemble: knn3 + knn4 (prob_max)", flush=True)
for w in [0.5, 0.6, 0.7]:
    ens = w * p_k3 + (1-w) * p_k4
    a   = macro_auc(file_labels, ens)
    nm  = f'ens_k3x{w:.1f}_k4x{1-w:.1f}_pmax'
    print(f"  {nm}: {a:.4f}  (Δ={a-BEST:+.4f})", flush=True)
    results[nm] = a

# ── SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {name:<45s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

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
    print(f"\n未超越 best ({BEST:.4f})。")
print("done", flush=True)
