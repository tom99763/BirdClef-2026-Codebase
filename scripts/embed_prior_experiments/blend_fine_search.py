"""
Finer search around the best blend (0.7×prob_max + 0.3×prob_mean, knn3 = 0.9051)

Methods:
  A) Finer blend ratios: 0.55, 0.60, 0.65, 0.75, 0.80
  B) prob_top2 (sigmoid of top-2 window mean logit) + knn3 ps_alpha
  C) 3-component ps_alpha: prob_max + prob_mean + knn3 (2-param optimization)
  D) Finer alpha grid (0.05 step) for knn3 + prob_max
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

file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_mean= np.zeros((n_files, n_species),         dtype=np.float32)
file_logit_top2= np.zeros((n_files, n_species),         dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    wl = logits_win[idx:idx+nw]
    file_embs[fi]       = emb_win[idx:idx+nw].mean(0)
    file_labels[fi]     = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi]  = wl.max(0)
    file_logit_mean[fi] = wl.mean(0)
    if nw >= 2:
        file_logit_top2[fi] = np.sort(wl, axis=0)[-2:].mean(0)
    else:
        file_logit_top2[fi] = wl.max(0)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
file_prob_max  = scipy.special.expit(file_logit_max)
file_prob_mean = scipy.special.expit(file_logit_mean)
file_prob_top2 = scipy.special.expit(file_logit_top2)
print(f"Files={n_files}, species={n_species}\n", flush=True)

# EXACT knn_predict
def knn_predict(k=3):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr = file_embs_norm[mask]; te = file_embs_norm[[i]]; y_tr = file_labels[mask]
        sims = (te @ tr.T).ravel()
        k_eff = min(k, len(sims))
        nn_idx = np.argpartition(-sims, k_eff)[:k_eff]
        weights = np.clip(sims[nn_idx], 0, None)
        if weights.sum() < 1e-9: weights = np.ones(k_eff)
        preds[i] = (weights[:, None] * y_tr[nn_idx]).sum(0) / weights.sum()
    return preds

print("Precomputing KNN-3...", flush=True)
knn3 = knn_predict(3)
print("  done.\n", flush=True)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

BEST = 0.905149
N_TRAIN = n_files - 1

def ps_alpha_loo(prob_feat, knn_all, alpha_step=0.1, default_alpha=0.30):
    """Exact match to logit_fusion_v3.py."""
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    alpha_grid = np.arange(0.0, 1.0 + alpha_step/2, alpha_step)
    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr_knn = knn_all[mask]; tr_logit = prob_feat[mask]; tr_labels = file_labels[mask]
        for s in range(n_species):
            y_s = tr_labels[:, s]
            if y_s.sum() == 0:
                preds[i, s] = prob_feat[i, s]; continue
            if y_s.sum() == N_TRAIN:
                preds[i, s] = 1.0; continue
            best_a, best_auc = default_alpha, -1.0
            for a in alpha_grid:
                bl = a * tr_logit[:, s] + (1-a) * tr_knn[:, s]
                try:
                    v = roc_auc_score(y_s, bl)
                    if v > best_auc: best_auc, best_a = v, a
                except: pass
            preds[i, s] = float(best_a * prob_feat[i, s] + (1-best_a) * knn_all[i, s])
    return preds

RESULTS = {}

# ── A) Finer blend ratios around 0.7 ─────────────────────────────────
print("="*60)
print("A) Finer blend ratios (prob_max + prob_mean, knn3)")
print("="*60, flush=True)

print("  Computing p_max and p_mean loo preds...", flush=True)
p_max  = ps_alpha_loo(file_prob_max, knn3)
p_mean = ps_alpha_loo(file_prob_mean, knn3)

for w in [0.55, 0.60, 0.65, 0.72, 0.75, 0.80, 0.85]:
    ens = w * p_max + (1-w) * p_mean
    a   = macro_auc(file_labels, ens)
    nm  = f'ens_pmx{w:.2f}_pmn{1-w:.2f}_k3'
    print(f"  {nm}: {a:.6f}  (Δ={a-BEST:+.6f})", flush=True)
    RESULTS[nm] = a

# ── B) prob_top2 + knn3 ps_alpha ─────────────────────────────────────
print("\n" + "="*60)
print("B) prob_top2 (top-2 window mean) + knn3 ps_alpha")
print("="*60, flush=True)

p_top2 = ps_alpha_loo(file_prob_top2, knn3)
a_top2 = macro_auc(file_labels, p_top2)
print(f"  knn3+prob_top2: {a_top2:.6f}  (Δ={a_top2-BEST:+.6f})", flush=True)
RESULTS['ps_alpha_knn3_probtop2'] = a_top2

# Blend top2 with max
for w in [0.6, 0.7, 0.8]:
    ens = w * p_max + (1-w) * p_top2
    a   = macro_auc(file_labels, ens)
    nm  = f'ens_pmx{w:.1f}_ptop2{1-w:.1f}_k3'
    print(f"  {nm}: {a:.6f}  (Δ={a-BEST:+.6f})", flush=True)
    RESULTS[nm] = a

# ── C) 3-component: prob_max + prob_mean + prob_top2 ─────────────────
print("\n" + "="*60)
print("C) 3-component ensemble (prob_max + prob_mean + prob_top2)")
print("="*60, flush=True)

best_3c, best_nm_3c = -1, ''
for wm in [0.5, 0.6, 0.65, 0.7, 0.75]:
    for wn in [0.1, 0.15, 0.2, 0.25, 0.3]:
        wt = max(1 - wm - wn, 0)
        if wt < 0.05: continue
        ens = wm * p_max + wn * p_mean + wt * p_top2
        a   = macro_auc(file_labels, ens)
        nm  = f'3c_mx{wm:.2f}_mn{wn:.2f}_t2{wt:.2f}'
        if a > best_3c: best_3c, best_nm_3c = a, nm
        if a > BEST:
            print(f"  {nm}: {a:.6f}  (Δ={a-BEST:+.6f}) ← BETTER", flush=True)
        RESULTS[nm] = a

print(f"  Best 3-component: {best_nm_3c} = {best_3c:.6f}  (Δ={best_3c-BEST:+.6f})", flush=True)

# ── D) Finer alpha grid (0.05 step) for knn3 + prob_max ──────────────
print("\n" + "="*60)
print("D) Finer alpha grid (0.05 step) for knn3+prob_max")
print("="*60, flush=True)

p_fine = ps_alpha_loo(file_prob_max, knn3, alpha_step=0.05)
a_fine = macro_auc(file_labels, p_fine)
print(f"  knn3+prob_max (0.05 step): {a_fine:.6f}  (Δ={a_fine-BEST:+.6f})", flush=True)
RESULTS['ps_alpha_knn3_pmax_fine05'] = a_fine

# Blend fine with prob_mean
for w in [0.65, 0.70, 0.75]:
    ens = w * p_fine + (1-w) * p_mean
    a   = macro_auc(file_labels, ens)
    nm  = f'ens_fine{w:.2f}_mn{1-w:.2f}'
    print(f"  {nm}: {a:.6f}  (Δ={a-BEST:+.6f})", flush=True)
    RESULTS[nm] = a

# ── SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (top 10)")
print("="*60)
top10 = sorted(RESULTS.items(), key=lambda x: -x[1])[:10]
for name, auc in top10:
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {name:<45s}  {auc:.6f}  {auc-BEST:+.6f}{marker}")

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
    print(f"\n*** NEW BEST: {new_best} AUC={best_auc:.6f} ***")
else:
    print(f"\n未超越 best ({BEST:.6f})，ens_k3pmx0.7_k3pmn0.3 仍是最佳。")
print("done", flush=True)
