"""
Ridge Stacking + Per-species alpha variants for Embed Prior LOO-CV
Best so far: per_species_alpha_knn3 = 0.9026

Methods:
  A) per_species_alpha with knn1, knn2 (k values not yet tried)
  B) 3-way blend: logit_max + knn1 + knn3 (per-species optimal)
  C) Ridge stack: [logit_max, logit_mean, knn1, knn2, knn3] per species
"""
import numpy as np, json, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load (same format as bayesian_ridge_rbf.py) ───────────────────────
raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
logits_win = raw['logits'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_embs   = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels = np.zeros((n_files, n_species), dtype=np.float32)
file_logmax = np.zeros((n_files, n_species), dtype=np.float32)
file_logmean= np.zeros((n_files, n_species), dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]    = emb_win[idx:idx+nw].mean(0)
    file_labels[fi]  = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    file_logmax[fi]  = logits_win[idx:idx+nw].max(0)
    file_logmean[fi] = logits_win[idx:idx+nw].mean(0)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
print(f"Files={n_files}, species={n_species}, best_so_far=0.9026\n", flush=True)

# ── Precompute pairwise KNN scores ────────────────────────────────────
print("Precomputing pairwise cosine KNN scores...", flush=True)
SIM = file_embs_norm @ file_embs_norm.T  # (66, 66)
np.fill_diagonal(SIM, -2.0)              # exclude self

def knn_scores_matrix(k):
    """(66, 234) KNN scores; each file uses all OTHER files as neighbors."""
    scores = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        sims_i = SIM[i].copy()
        top    = np.argsort(-sims_i)[:k]
        w      = sims_i[top].clip(0)
        w_sum  = w.sum()
        if w_sum < 1e-8: w = np.ones(k) / k
        else: w /= w_sum
        scores[i] = (w[:, None] * file_labels[top]).sum(0)
    return scores

KNN = {}
for k in [1, 2, 3, 5]:
    KNN[k] = knn_scores_matrix(k)
    print(f"  KNN k={k} done.", flush=True)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

BEST = 0.9026
results = {}

# ══════════════════════════════════════════════════════════════════════
# A) per_species_alpha with knn1 and knn2
# ══════════════════════════════════════════════════════════════════════
print("="*60)
print("A) per_species_alpha with knn1 and knn2")
print("="*60, flush=True)

def ps_alpha_knn_loo(knn_key):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    alpha_grid = np.linspace(0, 1, 41)
    for i in range(n_files):
        tr_idx = np.arange(n_files)[np.arange(n_files) != i]
        for s in range(n_species):
            y_s   = file_labels[tr_idx, s]
            n_pos = int(y_s.sum())
            if n_pos < 2 or (len(y_s) - n_pos) < 2:
                preds[i, s] = np.clip(file_logmax[i, s], 0, 1)
                continue
            lm_tr = file_logmax[tr_idx, s]
            kn_tr = KNN[knn_key][tr_idx, s]
            best_a, best_auc = 0.5, -1
            for a in alpha_grid:
                bl = a * lm_tr + (1-a) * kn_tr
                try:
                    auc = roc_auc_score(y_s, bl)
                    if auc > best_auc: best_auc, best_a = auc, a
                except: pass
            preds[i, s] = np.clip(
                best_a * file_logmax[i, s] + (1-best_a) * KNN[knn_key][i, s],
                0, 1)
    return preds

for k in [1, 2]:
    p = ps_alpha_knn_loo(k)
    a = macro_auc(file_labels, p)
    nm = f'per_species_alpha_knn{k}'
    print(f"  {nm}: {a:.4f}  (Δ={a-BEST:+.4f})", flush=True)
    results[nm] = a

# ══════════════════════════════════════════════════════════════════════
# B) 3-way blend: logit_max + knnA + knnB (per-species optimal)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) 3-way per-species: logit_max + knnA + knnB")
print("="*60, flush=True)

def ps_3way_loo(ka, kb):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    grid  = [(a, b) for a in np.linspace(0, 1, 11)
                    for b in np.linspace(0, 1-a, 11)]
    for i in range(n_files):
        tr_idx = np.arange(n_files)[np.arange(n_files) != i]
        for s in range(n_species):
            y_s = file_labels[tr_idx, s]
            n_pos = int(y_s.sum())
            if n_pos < 2 or (len(y_s) - n_pos) < 2:
                preds[i, s] = np.clip(file_logmax[i, s], 0, 1)
                continue
            lm_tr = file_logmax[tr_idx, s]
            ka_tr = KNN[ka][tr_idx, s]
            kb_tr = KNN[kb][tr_idx, s]
            best_a, best_b, best_auc = 0.33, 0.33, -1
            for (a, b) in grid:
                c = 1 - a - b
                if c < -1e-6: continue
                bl = a * lm_tr + b * ka_tr + max(c, 0) * kb_tr
                try:
                    auc = roc_auc_score(y_s, bl)
                    if auc > best_auc: best_auc, best_a, best_b = auc, a, b
                except: pass
            c = max(1 - best_a - best_b, 0)
            preds[i, s] = np.clip(
                best_a * file_logmax[i, s] + best_b * KNN[ka][i, s] + c * KNN[kb][i, s],
                0, 1)
    return preds

for (ka, kb) in [(1, 3), (2, 3), (1, 2)]:
    p = ps_3way_loo(ka, kb)
    a = macro_auc(file_labels, p)
    nm = f'ps_3way_lmax_k{ka}_k{kb}'
    print(f"  {nm}: {a:.4f}  (Δ={a-BEST:+.4f})", flush=True)
    results[nm] = a

# ══════════════════════════════════════════════════════════════════════
# C) Ridge stack: [logit_max, logit_mean, knn1, knn2, knn3] per species
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) Ridge stack: [logit_max, logit_mean, knn1, knn2, knn3]")
print("="*60, flush=True)

FEAT = np.stack([file_logmax, file_logmean,
                 KNN[1], KNN[2], KNN[3]], axis=2)  # (66, 234, 5)

def ridge_stack_loo():
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        tr_idx = np.arange(n_files)[np.arange(n_files) != i]
        X_te   = FEAT[i]         # (234, 5)
        X_tr   = FEAT[tr_idx]    # (65, 234, 5)
        for s in range(n_species):
            y_s   = file_labels[tr_idx, s]
            n_pos = int(y_s.sum())
            if n_pos < 2 or (len(y_s) - n_pos) < 2:
                preds[i, s] = np.clip(file_logmax[i, s], 0, 1)
                continue
            try:
                clf = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3)
                clf.fit(X_tr[:, s, :], y_s)
                p = float(clf.predict(X_te[[s], :])[0])
                preds[i, s] = np.clip(p, 0.0, 1.0)
            except:
                preds[i, s] = np.clip(file_logmax[i, s], 0, 1)
        if (i + 1) % 10 == 0:
            print(f"  fold {i+1}/66 done", flush=True)
    return preds

print("  Running Ridge Stack LOO-CV...", flush=True)
p_rs = ridge_stack_loo()
a_rs = macro_auc(file_labels, p_rs)
print(f"  Ridge Stack AUC: {a_rs:.4f}  (Δ={a_rs-BEST:+.4f})", flush=True)
results['ridge_stack_5feat'] = a_rs

# ══════════════════════════════════════════════════════════════════════
# SUMMARY + update JSON
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {name:<40s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

best_auc = data['best']['loo_auc']
new_best_method = None
for name, auc in results.items():
    data['experiments'].append({'method': name, 'loo_auc': round(auc, 6),
                                 'features': 'emb+logit'})
    if auc > best_auc:
        best_auc = auc
        new_best_method = name
        data['best'] = {'method': name, 'loo_auc': round(auc, 6), 'note': 'NEW BEST'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

if new_best_method:
    print(f"\n*** NEW BEST: {new_best_method} AUC={best_auc:.4f} ***")
else:
    print(f"\n未超越 best ({BEST:.4f})，per_species_alpha_knn3 仍是最佳。")
print("done", flush=True)
