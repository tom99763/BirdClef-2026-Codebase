"""
精調 logspace method:
score = a × logit_max + b × log(y_nl)  → sigmoid → AUC

最佳初始: a=0.7, b=1.5 → 0.9094
"""
import numpy as np, json, pickle, re, os, shutil
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
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

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])])
file_end   = np.cumsum(n_windows)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

file_embs        = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels      = np.zeros((n_files, n_species), dtype=np.float32)
file_logit_max   = np.zeros((n_files, n_species), dtype=np.float32)
file_logit_p90   = np.zeros((n_files, n_species), dtype=np.float32)
file_logit_mean  = np.zeros((n_files, n_species), dtype=np.float32)
file_prob_max    = np.zeros((n_files, n_species), dtype=np.float32)

for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]   = emb_win[s:e].mean(0)
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    lb = logits_win[s:e]
    file_logit_max[fi]  = lb.max(0)
    file_logit_p90[fi]  = np.percentile(lb, 90, axis=0)
    file_logit_mean[fi] = lb.mean(0)
    file_prob_max[fi]   = _sigmoid(lb.max(0))

file_embs_norm = normalize(file_embs, norm='l2')

SITES = ['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx = {s: i for i, s in enumerate(SITES)}
file_sites = np.zeros(n_files, dtype=np.int32)
file_hours = np.zeros(n_files, dtype=np.float32)
file_months= np.zeros(n_files, dtype=np.float32)
file_days  = np.zeros(n_files, dtype=np.float32)
for fi, fname in enumerate(file_list):
    m = re.match(r'BC2026_Train_\d+_(S\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})', str(fname))
    if m:
        site, yr, mo, dy, hr, mn = m.groups()
        file_sites[fi]  = site2idx.get(site, 0)
        file_hours[fi]  = int(hr)
        file_months[fi] = int(mo)
        dpm = [0,31,28,31,30,31,30,31,31,30,31,30,31]
        file_days[fi] = sum(dpm[:int(mo)]) + int(dy)
site_oh   = np.eye(len(SITES), dtype=np.float32)[file_sites]
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24), np.cos(2*np.pi*file_hours/24)], axis=1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12), np.cos(2*np.pi*(file_months-1)/12)], axis=1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365), np.cos(2*np.pi*(file_days-1)/365)], axis=1).astype(np.float32)

pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
X24  /= (X24.std(0) + 1e-6)
geo   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], axis=1).astype(np.float32)
X_comb = np.concatenate([X24, geo], axis=1).astype(np.float32)
X_nl = (X_comb / np.linalg.norm(X_comb, axis=1, keepdims=True)).astype(np.float32)

def attn_knn_loo(X, k=10, T=0.2):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        tr_idx = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr_idx].T).ravel()
        top = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr_idx[top]]).sum(0)
    return preds

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS  = 1e-7
BEST = 0.905463
results = {}

print(f"Files={n_files}, species={n_species}", flush=True)

# Precompute nologit
y_nl = attn_knn_loo(X_nl, k=10, T=0.2)
log_nl = np.log(y_nl.clip(EPS, 1-EPS))
print(f"nologit pca24+day k10: {macro_auc(file_labels, y_nl):.4f}", flush=True)

best_so_far = BEST

# ── A) Fine sweep around (a=0.7, b=1.5) ─────────────────────────────────
print("\n" + "="*60)
print("A) Fine sweep around (a=0.7, b=1.5)")
print("="*60, flush=True)

for a in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
    for b in [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 3.0]:
        score = a * file_logit_max + b * log_nl
        auc   = macro_auc(file_labels, _sigmoid(score))
        nm = f'ls_lmx_a{a:.2f}_b{b:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST - 0.003:
            print(f"  a={a:.2f} b={b:.2f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"  After fine sweep, best={best_so_far:.4f}", flush=True)

# ── B) 改用 logit_p90 (更穩定，比 max 雜訊少) ─────────────────────────────
print("\n" + "="*60)
print("B) logit_p90 instead of logit_max")
print("="*60, flush=True)

for a in [0.5, 0.7, 1.0, 1.2]:
    for b in [1.0, 1.5, 2.0, 2.5]:
        score = a * file_logit_p90 + b * log_nl
        auc   = macro_auc(file_labels, _sigmoid(score))
        nm = f'ls_p90_a{a:.1f}_b{b:.1f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST - 0.005:
            print(f"  p90 a={a:.1f} b={b:.1f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

# ── C) Blend: c1×logit_max + c2×logit_p90 + b×log_nl ───────────────────
print("\n" + "="*60)
print("C) Combo logit_max + logit_p90 + log_nl")
print("="*60, flush=True)

for c1, c2 in [(0.5, 0.3), (0.5, 0.2), (0.6, 0.2), (0.6, 0.3), (0.7, 0.1)]:
    for b in [1.3, 1.5, 1.7, 2.0]:
        score = c1 * file_logit_max + c2 * file_logit_p90 + b * log_nl
        auc   = macro_auc(file_labels, _sigmoid(score))
        nm = f'ls_mx{c1:.1f}_p90{c2:.1f}_nl{b:.1f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST - 0.003:
            print(f"  mx={c1:.1f} p90={c2:.1f} b={b:.1f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

# ── D) Use mean-centered logit (subtract per-file mean) ──────────────────
print("\n" + "="*60)
print("D) Mean-centered logit_max")
print("="*60, flush=True)

# Subtract per-file mean (removes file-level bias)
file_logit_mean_all = file_logit_mean.mean(1, keepdims=True)  # (66, 1)
logit_max_centered  = file_logit_max - file_logit_mean_all

for a in [0.5, 0.7, 1.0]:
    for b in [1.0, 1.5, 2.0]:
        score = a * logit_max_centered + b * log_nl
        auc   = macro_auc(file_labels, _sigmoid(score))
        nm = f'ls_centered_a{a:.1f}_b{b:.1f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST - 0.003:
            print(f"  centered a={a:.1f} b={b:.1f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

# ── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (top 15)")
print("="*60)
for nm, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {nm:<55s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\nOverall best: {global_best_name} = {global_best_auc:.4f}", flush=True)

# ── Update json + save pkl if new best ────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best = data['best']['loo_auc']
for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': 'logspace_finetune'})
    if auc > cur_best:
        cur_best = auc
        data['best'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'NEW BEST logspace_finetune'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\nbest: {data['best']['method']} = {data['best']['loo_auc']:.4f}", flush=True)

# ── Save best pkl ──────────────────────────────────────────────────────────
if global_best_auc > BEST:
    # Parse params from name
    import re as _re
    m = _re.search(r'_a([\d.]+)_b([\d.]+)', global_best_name)
    if m:
        a_best = float(m.group(1))
        b_best = float(m.group(2))
    else:
        a_best, b_best = 0.7, 1.5  # fallback

    print(f"\nSaving logspace pkl: a={a_best}, b={b_best}, AUC={global_best_auc:.4f}", flush=True)

    # Store all needed inference components
    pca24_fitted = PCA(n_components=24, random_state=42).fit(file_embs_norm)
    X24f = pca24_fitted.transform(file_embs_norm).astype(np.float32)
    pca24_std = (X24f.std(0) + 1e-6).astype(np.float32)

    pkl_data = {
        'method': global_best_name,
        'loo_auc': round(global_best_auc, 6),
        'type': 'logspace',
        'a': a_best,    # logit_max scaling
        'b': b_best,    # log(nologit) scaling
        # PCA components for test embedding projection
        'pca_dims': 24,
        'pca_mean': pca24_fitted.mean_.astype(np.float32),
        'pca_components': pca24_fitted.components_.astype(np.float32),
        'pca_std': pca24_std,
        'use_day': True,
        'SITES': SITES,
        'site2idx': site2idx,
        # Reference matrix
        'X_combined_n': X_nl,       # (66, 39) normalized
        'file_labels': file_labels,
        'file_list': file_list,
        'k': 10,
        'T': 0.2,
        'temperature': 0.2,
    }
    with open("outputs/embed_prior_logspace.pkl", "wb") as f:
        pickle.dump(pkl_data, f)
    shutil.copy("outputs/embed_prior_logspace.pkl",
                "birdclef-2026/notebook resource/current_subs/weights/embed_prior_logspace.pkl")
    print(f"Saved: outputs/embed_prior_logspace.pkl")

print("done", flush=True)
