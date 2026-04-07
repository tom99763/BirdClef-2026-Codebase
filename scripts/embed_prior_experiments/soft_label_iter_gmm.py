"""
三個新方法：
A) Soft-label Logspace：用 sigmoid(logit_max) 作為 KNN targets（非 binary labels）
   score = a × logit_max + b × log(y_soft_nl)  → sigmoid
B) Iterative Refinement：第一步用 binary labels，第二步用 step-1 預測作 soft labels，再跑一次
C) GMM per-species：對每個 species fit 2-component GMM，用 positive component 的 likelihood 作 prior

Current best: ls_lmx_a0.70_b1.50 = 0.9094
Current best_nologit: attn_k10_T02_pca24_day = 0.8758
"""
import numpy as np, json, pickle, re, os, shutil
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────
raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
logits_win = raw['logits'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species), dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species), dtype=np.float32)
file_prob_max  = np.zeros((n_files, n_species), dtype=np.float32)

for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    lb = logits_win[s:e]
    file_logit_max[fi] = lb.max(0)
    file_prob_max[fi]  = _sigmoid(lb.max(0))

file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)

# ── Nologit pca24+day combined space ────────────────────────────────────────
SITES = ['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx = {s: i for i, s in enumerate(SITES)}
file_sites  = np.zeros(n_files, dtype=np.int32)
file_hours  = np.zeros(n_files, dtype=np.float32)
file_months = np.zeros(n_files, dtype=np.float32)
file_days   = np.zeros(n_files, dtype=np.float32)
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
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24),
                       np.cos(2*np.pi*file_hours/24)], axis=1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12),
                       np.cos(2*np.pi*(file_months-1)/12)], axis=1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365),
                       np.cos(2*np.pi*(file_days-1)/365)], axis=1).astype(np.float32)

pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
X24  /= (X24.std(0) + 1e-6)
geo   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], axis=1).astype(np.float32)
X_comb = np.concatenate([X24, geo], axis=1).astype(np.float32)
X_nl   = (X_comb / np.linalg.norm(X_comb, axis=1, keepdims=True)).astype(np.float32)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS  = 1e-7
BEST = 0.909351
BEST_NL = 0.875789
results = {}

print(f"Files={n_files}, species={n_species}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# A) Soft-label Logspace
#    KNN targets = sigmoid(logit_max)  (continuous, not binary)
#    score = a × logit_max + b × log(y_soft_nl)  → sigmoid
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A) Soft-label Logspace: KNN targets = sigmoid(logit_max)")
print("="*60, flush=True)

soft_labels = file_prob_max.clip(EPS, 1-EPS)   # (66, 234) — continuous

def attn_knn_loo_soft(X, targets, k=10, T=0.2):
    """Attn-KNN LOO with soft label targets."""
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * targets[tr[top]]).sum(0)
    return preds

print("  Computing soft-label nologit KNN...", flush=True)
y_soft_nl = attn_knn_loo_soft(X_nl, soft_labels, k=10, T=0.2)
log_soft   = np.log(y_soft_nl.clip(EPS, 1-EPS))

# Also compute standard (binary) nologit for comparison
y_nl = attn_knn_loo_soft(X_nl, file_labels, k=10, T=0.2)
log_nl = np.log(y_nl.clip(EPS, 1-EPS))

print(f"  soft_nl base AUC: {macro_auc(file_labels, y_soft_nl):.4f}")
print(f"  hard_nl base AUC: {macro_auc(file_labels, y_nl):.4f}")

best_so_far = BEST
for a in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
    for b in [1.0, 1.2, 1.5, 1.7, 2.0, 2.5]:
        score = a * file_logit_max + b * log_soft
        auc   = macro_auc(file_labels, _sigmoid(score))
        nm = f'soft_ls_a{a:.2f}_b{b:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST - 0.003:
            print(f"  a={a:.2f} b={b:.2f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

# Blend soft and hard nologit
print("\n  Blend soft+hard nologit in logspace:", flush=True)
for pw_soft in [0.3, 0.5, 0.7]:
    log_blend = pw_soft * log_soft + (1-pw_soft) * log_nl
    for a in [0.65, 0.7, 0.75]:
        for b in [1.3, 1.5, 1.7]:
            score = a * file_logit_max + b * log_blend
            auc   = macro_auc(file_labels, _sigmoid(score))
            nm = f'soft_hard_blend_ps{pw_soft:.1f}_a{a:.2f}_b{b:.2f}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far: best_so_far = auc
            if auc > BEST - 0.003:
                print(f"  soft_pw={pw_soft:.1f} a={a:.2f} b={b:.2f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
            results[nm] = auc

print(f"  After A, best so far: {best_so_far:.4f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# B) Iterative Refinement
#    Step 1: logspace(binary) → soft predictions
#    Step 2: 用 step-1 predictions 作 nologit KNN targets，再跑一次 logspace
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) Iterative Refinement: 2-step logspace")
print("="*60, flush=True)

# Step 1 predictions: sigmoid(0.7 × logit_max + 1.5 × log(y_nl_hard))
step1_score = 0.7 * file_logit_max + 1.5 * log_nl
step1_preds = _sigmoid(step1_score).clip(EPS, 1-EPS)  # (66, 234)
auc_step1 = macro_auc(file_labels, step1_preds)
print(f"  Step 1 (logspace binary): {auc_step1:.4f}", flush=True)

# Step 2: use step1_preds as new nologit targets (LOO)
print("  Computing step-2 iterative nologit KNN...", flush=True)
y_iter_nl = attn_knn_loo_soft(X_nl, step1_preds, k=10, T=0.2)
log_iter   = np.log(y_iter_nl.clip(EPS, 1-EPS))

for a in [0.5, 0.6, 0.7, 0.75, 0.8]:
    for b in [1.0, 1.3, 1.5, 1.7, 2.0]:
        score = a * file_logit_max + b * log_iter
        auc   = macro_auc(file_labels, _sigmoid(score))
        nm = f'iter_ls_a{a:.2f}_b{b:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST - 0.003:
            print(f"  iter a={a:.2f} b={b:.2f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

# Blend step1 and step2 predictions
for alpha in [0.5, 0.6, 0.7, 0.8]:
    blend = alpha * step1_preds + (1-alpha) * y_iter_nl
    auc   = macro_auc(file_labels, blend)
    nm = f'iter_blend_a{alpha:.1f}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    if auc > BEST - 0.003:
        print(f"  iter_blend alpha={alpha:.1f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
    results[nm] = auc

print(f"  After B, best so far: {best_so_far:.4f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# C) GMM per species
#    For each species, fit GaussianMixture(2) on pca24 embs of all 66 files
#    Score = P(file belongs to positive component) → use as prior
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) GMM per-species prior (PCA-24 embedding space)")
print("="*60, flush=True)

# Use standardized pca24 (without geo) for GMM — more stable
pca24_raw = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24_raw   = pca24_raw.transform(file_embs_norm).astype(np.float32)
X24_raw  /= (X24_raw.std(0) + 1e-6)

def gmm_loo_prior(X, labels, n_components=2, n_pca=12):
    """
    LOO GMM: for each held-out file, fit GMM on training files,
    score the test file against positive component.
    """
    # Use top-n_pca dims only (most informative)
    Xr = X[:, :n_pca]
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    active_species = np.where(labels.sum(0) >= 2)[0]  # need ≥2 positives

    for fi in range(n_files):
        tr = [j for j in range(n_files) if j != fi]
        for s in active_species:
            pos_idx = [j for j in tr if labels[j, s] > 0.5]
            neg_idx = [j for j in tr if labels[j, s] <= 0.5]
            if len(pos_idx) < 1 or len(neg_idx) < 2:
                preds[fi, s] = labels[tr, s].mean()
                continue
            # Fit simple GMM: cluster all training points, identify positive cluster
            X_tr = Xr[tr]
            try:
                gmm = GaussianMixture(n_components=n_components, random_state=42,
                                      max_iter=50, n_init=1, covariance_type='diag')
                gmm.fit(X_tr)
                # Identify which component corresponds to positives
                # Use mean of positive files as reference
                pos_mean = Xr[pos_idx].mean(0)
                comp_means = gmm.means_  # (n_components, n_pca)
                dists = np.linalg.norm(comp_means - pos_mean, axis=1)
                pos_comp = np.argmin(dists)
                # Score test file: P(belongs to positive component)
                log_probs = gmm.predict_proba(Xr[[fi]])  # (1, n_components)
                preds[fi, s] = float(log_probs[0, pos_comp])
            except Exception:
                preds[fi, s] = labels[tr, s].mean()

        if (fi + 1) % 20 == 0:
            print(f"    GMM fold {fi+1}/{n_files} done", flush=True)

    return preds

print("  Running GMM LOO (may take a few minutes)...", flush=True)
y_gmm = gmm_loo_prior(X24_raw, file_labels, n_components=2, n_pca=12)
auc_gmm = macro_auc(file_labels, y_gmm)
print(f"  GMM prior LOO-AUC: {auc_gmm:.4f}  (Δ vs nologit best={auc_gmm-BEST_NL:+.4f})", flush=True)
results['gmm_prior_k12'] = auc_gmm

# Blend GMM prior with logit_max in logspace
log_gmm = np.log(y_gmm.clip(EPS, 1-EPS))
for a in [0.5, 0.7, 1.0]:
    for b in [1.0, 1.5, 2.0]:
        score = a * file_logit_max + b * log_gmm
        auc   = macro_auc(file_labels, _sigmoid(score))
        nm = f'gmm_ls_a{a:.1f}_b{b:.1f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST - 0.005:
            print(f"  gmm a={a:.1f} b={b:.1f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

# Blend GMM + nologit_hard + logit
for c_gmm in [0.3, 0.5]:
    log_fused = c_gmm * log_gmm + (1-c_gmm) * log_nl
    for a in [0.65, 0.7, 0.75]:
        for b in [1.3, 1.5, 1.7]:
            score = a * file_logit_max + b * log_fused
            auc   = macro_auc(file_labels, _sigmoid(score))
            nm = f'gmm_nl_fuse_cg{c_gmm:.1f}_a{a:.2f}_b{b:.2f}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far: best_so_far = auc
            if auc > BEST - 0.003:
                print(f"  gmm+nl c_gmm={c_gmm:.1f} a={a:.2f} b={b:.2f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
            results[nm] = auc

print(f"  After C, best so far: {best_so_far:.4f}", flush=True)

# ── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (top 15)")
print("="*60)
for nm, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {nm:<60s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\n整體最佳: {global_best_name} = {global_best_auc:.4f}", flush=True)

# ── Update results.json ────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best = data['best']['loo_auc']
new_best_found = False

for nm, auc in results.items():
    if 'gmm' in nm:
        note = 'gmm_prior'
    elif 'iter' in nm:
        note = 'iterative_refinement'
    else:
        note = 'soft_label_logspace'
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': note})
    if auc > cur_best:
        cur_best = auc
        data['best'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': f'NEW BEST {note}'}
        new_best_found = True

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\n已更新 embed_prior_results.json")
print(f"當前 best: {data['best']['method']} = {data['best']['loo_auc']:.4f}")

if new_best_found and global_best_auc > BEST:
    print(f"\nNEW BEST: {global_best_name} AUC={global_best_auc:.4f}")

print("done", flush=True)
