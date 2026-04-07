"""
Per-species alpha improvements — FIXED edge cases to match original 0.9026

Root cause of 0.7847 vs 0.9026 discrepancy:
  - y_s.sum()==0 → original: file_prob_max[i,s], mine: 0.0
  - y_s.sum()<2  → original: default alpha=0.30 blend, mine: 0.0

Methods (all fixing edge cases):
  A) knn4 + prob_max (now should be close to knn3=0.9026)
  B) knn5 + prob_max
  C) knn3 + prob_mean
  D) Ensemble knn3_pmax + knn4_pmax
  E) 3-way: logit_max + knn3 + knn4
"""
import numpy as np, scipy.special, json, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
logits_win = raw['logits'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_embs       = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels     = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_max  = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_mean = np.zeros((n_files, n_species),         dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    wl = logits_win[idx:idx+nw]
    file_embs[fi]       = emb_win[idx:idx+nw].mean(0)
    file_labels[fi]     = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi]  = wl.max(0)
    file_logit_mean[fi] = wl.mean(0)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
file_prob_max  = scipy.special.expit(file_logit_max)
file_prob_mean = scipy.special.expit(file_logit_mean)
print(f"Files={n_files}, species={n_species}\n", flush=True)

# EXACT knn_predict matching logit_fusion_v3.py
def knn_predict(k=3, X=None):
    if X is None: X = file_embs_norm
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr = X[mask]; te = X[[i]]; y_tr = file_labels[mask]
        sims = (te @ tr.T).ravel()
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff)[:k_eff]
        weights = np.clip(sims[nn_idx], 0, None)
        if weights.sum() < 1e-9: weights = np.ones(k_eff)
        preds[i] = (weights[:, None] * y_tr[nn_idx]).sum(0) / weights.sum()
    return preds

print("Precomputing KNN...", flush=True)
KNN = {k: knn_predict(k) for k in [3, 4, 5]}
print("  done.\n", flush=True)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

BEST = 0.9026

def ps_alpha_loo(prob_feat, knn_k, default_alpha=0.30):
    """EXACT match to logit_fusion_v3.py per_species_alpha_loo."""
    knn_all = KNN[knn_k]
    preds   = np.zeros((n_files, n_species), dtype=np.float32)
    N_TRAIN = n_files - 1  # 65

    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_knn    = knn_all[mask]
        tr_logit  = prob_feat[mask]
        tr_labels = file_labels[mask]

        for s in range(n_species):
            y_s = tr_labels[:, s]
            if y_s.sum() == 0:
                # No positive training → fallback to logit
                preds[i, s] = prob_feat[i, s]
                continue
            if y_s.sum() == N_TRAIN:
                preds[i, s] = 1.0
                continue

            best_alpha_s, best_inner_auc = default_alpha, -1.0
            for a in np.arange(0.0, 1.01, 0.1):
                blend_s = a * tr_logit[:, s] + (1-a) * tr_knn[:, s]
                try:
                    v = roc_auc_score(y_s, blend_s)
                    if v > best_inner_auc:
                        best_inner_auc, best_alpha_s = v, a
                except Exception:
                    pass  # AUC fails for single-class → keep default

            preds[i, s] = float(best_alpha_s * prob_feat[i, s] +
                                 (1 - best_alpha_s) * knn_all[i, s])
    return preds

RESULTS = {}

# ── A) knn3 rerun (verify ~0.9026) ────────────────────────────────────
print("A) knn3 + prob_max rerun (verify)", flush=True)
p3 = ps_alpha_loo(file_prob_max, 3)
a3 = macro_auc(file_labels, p3)
print(f"  knn3 + prob_max: {a3:.4f}  (Δ={a3-BEST:+.4f})", flush=True)
RESULTS['ps_alpha_knn3_verify'] = a3

# ── B) knn4 + prob_max ────────────────────────────────────────────────
print("B) knn4 + prob_max", flush=True)
p4 = ps_alpha_loo(file_prob_max, 4)
a4 = macro_auc(file_labels, p4)
print(f"  knn4 + prob_max: {a4:.4f}  (Δ={a4-BEST:+.4f})", flush=True)
RESULTS['ps_alpha_knn4_probmax'] = a4

# ── C) knn5 + prob_max ────────────────────────────────────────────────
print("C) knn5 + prob_max", flush=True)
p5 = ps_alpha_loo(file_prob_max, 5)
a5 = macro_auc(file_labels, p5)
print(f"  knn5 + prob_max: {a5:.4f}  (Δ={a5-BEST:+.4f})", flush=True)
RESULTS['ps_alpha_knn5_probmax'] = a5

# ── D) knn3 + prob_mean ───────────────────────────────────────────────
print("D) knn3 + prob_mean (sigmoid of logit_mean)", flush=True)
p3m = ps_alpha_loo(file_prob_mean, 3)
a3m = macro_auc(file_labels, p3m)
print(f"  knn3 + prob_mean: {a3m:.4f}  (Δ={a3m-BEST:+.4f})", flush=True)
RESULTS['ps_alpha_knn3_probmean'] = a3m

# ── E) Ensemble blends ────────────────────────────────────────────────
print("E) Ensemble blends", flush=True)
for (wA, wB, pA, pB, tag) in [
    (0.7, 0.3, p3, p4, 'k3x0.7_k4x0.3'),
    (0.8, 0.2, p3, p4, 'k3x0.8_k4x0.2'),
    (0.7, 0.3, p3, p3m, 'k3pmx0.7_k3pmn0.3'),
    (0.8, 0.2, p3, p3m, 'k3pmx0.8_k3pmn0.2'),
    (0.9, 0.1, p3, p3m, 'k3pmx0.9_k3pmn0.1'),
]:
    ens = wA * pA + wB * pB
    a   = macro_auc(file_labels, ens)
    print(f"  ens_{tag}: {a:.4f}  (Δ={a-BEST:+.4f})", flush=True)
    RESULTS[f'ens_{tag}'] = a

# ── SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, auc in sorted(RESULTS.items(), key=lambda x: -x[1]):
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {name:<45s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

best_auc = data['best']['loo_auc']
new_best = None
for name, auc in RESULTS.items():
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
