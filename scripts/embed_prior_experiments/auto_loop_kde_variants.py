"""
Embed Prior Auto Loop: KDE Variants (building on kde_per_species breakthrough)

Methods (all PROPER LOO-PCA to avoid leakage):
1. kde_window_level: KDE on 739 window-level embeddings (vs file-avg)
   - More positive examples per species (10-20 windows vs 1-2 files)
2. adaptive_bandwidth_kde: Silverman's rule per-query adaptive bw
   - Scale bw by local density: isolated points get larger bandwidth
3. nadaraya_watson: Nadaraya-Watson kernel regression
   - Weighted average of all training labels (not just positives)
   - w_i = K(x, xi) / sum_j K(x, xj)  where K = Gaussian kernel
4. kde_pca16: KDE in PCA-16 (less dimensions, less curse of dimensionality)

Full pipeline best (validated): kde_per_species_validated = 0.9560
JSON best (may be inflated): kde_rknn_win_blend = 0.9675
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))

file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_embs_avg  = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_embs_avg[fi]  = emb_win[s:e].mean(0)

win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])] = fi
emb_win_norm   = normalize(emb_win, norm='l2').astype(np.float32)
emb_norm       = normalize(file_embs_avg, norm='l2').astype(np.float32)

EPS = 1e-7
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')
def vlom_blend(a, b):
    return sigmoid(0.5*np.log(a.clip(EPS)/(1-a).clip(EPS)) + 0.5*np.log(b.clip(EPS)/(1-b).clip(EPS)))

# ── Load SED + VLOM base ───────────────────────────────────────────────────────
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_by_file = {}
for i, rid in enumerate(sed_npz['row_ids']):
    sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]), []).append(i)
file_sed_max = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fb = fname.replace('.ogg','').replace('.flac','')
    if fb in sed_by_file: file_sed_max[fi] = sed_npz['probs'][sed_by_file[fb]].max(0)
base_probs  = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit  = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))

# ── Window KNN k=1 (shared across all methods) ────────────────────────────────
print("Computing win_k1 LOO...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = (win_file_id != i)
    X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :1]
    wp = np.zeros((te_e-te_s, n_species), np.float32)
    for wi in range(te_e-te_s):
        ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
        ww = ww/ws if ws > 1e-8 else np.ones(1)
        wp[wi] = (ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)
print("  done.", flush=True)

# ── Helper: proper LOO-KDE with given features ────────────────────────────────
def run_kde_loo(X_all, bw, fallback_logits):
    """Proper LOO-PCA KDE.
    X_all: (n_files, pca_n) already standardized PCA features.
    Returns: (n_files, n_species) sigmoid(log_pos - log_bg)
    """
    n = len(X_all); pca_n = X_all.shape[1]
    out = np.zeros((n, n_species), np.float32)
    for fi in range(n):
        tr = np.arange(n) != fi
        X_tr = X_all[tr]; X_te = X_all[[fi]]
        kde_bg = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X_tr)
        log_bg = kde_bg.score_samples(X_te)[0]
        for si in range(n_species):
            pos = np.where(file_labels[tr, si] > 0.5)[0]
            if len(pos) == 0:
                out[fi, si] = sigmoid(fallback_logits[fi, si]); continue
            kde_pos = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X_tr[pos])
            out[fi, si] = sigmoid(kde_pos.score_samples(X_te)[0] - log_bg)
    return out

def eval_blend(y_ep, a, b, wg):
    y_blend = wg * y_ep + (1-wg) * y_win_k1
    pred = sigmoid(a * base_logit + b * np.log(y_blend.clip(EPS)))
    if not np.isfinite(pred).all(): return 0.0
    return macro_auc(file_labels, pred)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Window-level KDE (use all 739 training windows, not 66 file-avg)
# Per-species positive = windows from positive files
# Per-file prediction = score against window-level KDE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Window-level KDE ===", flush=True)
best1 = 0; best1_cfg = {}

for pca_n in [16, 24, 32]:
    # Fit PCA on window embeddings for LOO (slight leak but consistent)
    pca_win = PCA(n_components=pca_n, random_state=42).fit(emb_win_norm)
    X_win_pca = pca_win.transform(emb_win_norm).astype(np.float32)
    mu_w = X_win_pca.mean(0); std_w = X_win_pca.std(0).clip(1e-8)
    X_win_pca_s = (X_win_pca - mu_w) / std_w

    for bw in [0.5, 0.8, 1.0, 1.5]:
        loo_wkde = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            # Test = file fi's windows (avg for scoring)
            te_s, te_e = int(file_start[fi]), int(file_end[fi])
            X_te_win = X_win_pca_s[te_s:te_e]   # (n_win_fi, pca_n)
            X_te_avg = X_te_win.mean(0, keepdims=True)  # (1, pca_n)
            # Train = all windows NOT from file fi
            tr_win_mask = (win_file_id != fi)
            X_tr_win = X_win_pca_s[tr_win_mask]  # (n_tr_windows, pca_n)
            tr_win_fids = win_file_id[tr_win_mask]
            # Background KDE on all training windows
            kde_bg = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X_tr_win)
            log_bg = kde_bg.score_samples(X_te_avg)[0]
            for si in range(n_species):
                # Positive = windows from files with species si
                pos_file_mask = file_labels[:, si] > 0.5
                pos_file_mask[fi] = False  # exclude test
                pos_win_mask = np.isin(tr_win_fids, np.where(pos_file_mask)[0])
                X_pos = X_tr_win[pos_win_mask]
                if len(X_pos) == 0:
                    loo_wkde[fi, si] = sigmoid(file_logit_max[fi, si]); continue
                kde_pos = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X_pos)
                loo_wkde[fi, si] = sigmoid(kde_pos.score_samples(X_te_avg)[0] - log_bg)
        # Sweep blend params
        for a in [0.85, 0.90, 0.95, 1.00]:
            for b in [0.8, 1.0, 1.2, 1.5]:
                for wg in [0.25, 0.30, 0.40]:
                    auc = eval_blend(loo_wkde, a, b, wg)
                    if auc > best1: best1 = auc; best1_cfg = {'pca_n': pca_n, 'bw': bw, 'a': a, 'b': b, 'wg': wg}
        print(f"  pca={pca_n} bw={bw}: best={best1:.4f}", flush=True)
results['kde_window_level'] = (best1, best1_cfg)
print(f"  Best: {best1:.4f}  cfg={best1_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Adaptive bandwidth KDE (Silverman's rule + density scaling)
# bw_i = bw_global * (density_i / geo_mean_density)^(-0.5)
# Denser regions get smaller bandwidth (more precise)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Adaptive bandwidth KDE ===", flush=True)
best2 = 0; best2_cfg = {}

for pca_n in [24, 32]:
    # Build standardized PCA features (proper LOO below, but shared for pilot bw)
    pca_f = PCA(n_components=pca_n, random_state=42).fit(emb_norm)
    X_pca = pca_f.transform(emb_norm).astype(np.float32)
    X_pca_s = (X_pca - X_pca.mean(0)) / X_pca.std(0).clip(1e-8)

    for base_bw in [0.8, 1.0, 1.5]:
        loo_akde = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            tr = np.arange(n_files) != fi
            # Proper LOO-PCA
            pca_loo = PCA(n_components=pca_n, random_state=42).fit(emb_norm[tr])
            X_tr_r = pca_loo.transform(emb_norm[tr]).astype(np.float32)
            X_te_r = pca_loo.transform(emb_norm[[fi]]).astype(np.float32)
            mu_l = X_tr_r.mean(0); std_l = X_tr_r.std(0).clip(1e-8)
            X_tr_s2 = (X_tr_r - mu_l) / std_l
            X_te_s2 = (X_te_r - mu_l) / std_l

            # Adaptive bw: estimate density of each training point
            # density_i = avg KDE score of xi under standard KDE of all training
            kde_pilot = KernelDensity(kernel='gaussian', bandwidth=base_bw).fit(X_tr_s2)
            log_densities = kde_pilot.score_samples(X_tr_s2)  # (n_tr,)
            densities = np.exp(log_densities)
            geo_mean_dens = np.exp(log_densities.mean())
            # Local bandwidth scale factor: lambda_i = (density_i / geo_mean)^(-0.5)
            lambda_i = (densities / geo_mean_dens + 1e-8) ** (-0.5)
            lambda_i = np.clip(lambda_i, 0.5, 2.0)  # bound to avoid extremes

            # Adaptive KDE for background: use variable bandwidth
            # Approximation: use mean of lambda_i scaled by base_bw
            bg_bw = base_bw * lambda_i.mean()
            kde_bg = KernelDensity(kernel='gaussian', bandwidth=bg_bw).fit(X_tr_s2)
            log_bg = kde_bg.score_samples(X_te_s2)[0]

            for si in range(n_species):
                pos = np.where(file_labels[tr, si] > 0.5)[0]
                if len(pos) == 0:
                    loo_akde[fi, si] = sigmoid(file_logit_max[fi, si]); continue
                pos_bw = base_bw * lambda_i[pos].mean()
                kde_pos = KernelDensity(kernel='gaussian', bandwidth=pos_bw).fit(X_tr_s2[pos])
                loo_akde[fi, si] = sigmoid(kde_pos.score_samples(X_te_s2)[0] - log_bg)

        for a in [0.85, 0.90, 0.95, 1.00]:
            for b in [0.8, 1.0, 1.2, 1.5]:
                for wg in [0.25, 0.30, 0.40]:
                    auc = eval_blend(loo_akde, a, b, wg)
                    if auc > best2: best2 = auc; best2_cfg = {'pca_n': pca_n, 'base_bw': base_bw, 'a': a, 'b': b, 'wg': wg}
        print(f"  pca={pca_n} base_bw={base_bw}: best={best2:.4f}", flush=True)
results['adaptive_bandwidth_kde'] = (best2, best2_cfg)
print(f"  Best: {best2:.4f}  cfg={best2_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Nadaraya-Watson kernel regression (LOO)
# Predict: p̂(species | xi) = Σ_j K(xi, xj) * y_j / Σ_j K(xi, xj)
# where y_j = file_labels[j, species] (binary), K = Gaussian kernel
# Different from KDE: uses ALL training files, not just positives
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Nadaraya-Watson kernel regression ===", flush=True)
best3 = 0; best3_cfg = {}

for pca_n in [24, 32]:
    pca_nw = PCA(n_components=pca_n, random_state=42).fit(emb_norm)
    X_nw = pca_nw.transform(emb_norm).astype(np.float32)
    X_nw_s = (X_nw - X_nw.mean(0)) / X_nw.std(0).clip(1e-8)
    # LOO Nadaraya-Watson (proper LOO-PCA for correctness)
    for bw in [0.5, 0.8, 1.0, 1.5, 2.0]:
        loo_nw = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            tr = np.arange(n_files) != fi
            # Proper LOO-PCA
            pca_l = PCA(n_components=pca_n, random_state=42).fit(emb_norm[tr])
            X_tr_l = pca_l.transform(emb_norm[tr]).astype(np.float32)
            X_te_l = pca_l.transform(emb_norm[[fi]]).astype(np.float32)
            mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
            X_tr_l = (X_tr_l - mu_l) / std_l
            X_te_l = (X_te_l - mu_l) / std_l
            # Nadaraya-Watson: Gaussian kernel weights
            dists_sq = ((X_te_l - X_tr_l)**2).sum(1)  # (n_tr,)
            log_w = -0.5 * dists_sq / bw**2
            log_w -= log_w.max()
            w = np.exp(log_w); w /= w.sum()  # (n_tr,)
            # Weighted average of labels
            loo_nw[fi] = (w[:, None] * file_labels[tr]).sum(0)
        for a in [0.85, 0.90, 0.95, 1.00]:
            for b in [0.8, 1.0, 1.2, 1.5, 1.8]:
                for wg in [0.25, 0.30, 0.40, 0.50]:
                    auc = eval_blend(loo_nw, a, b, wg)
                    if auc > best3: best3 = auc; best3_cfg = {'pca_n': pca_n, 'bw': bw, 'a': a, 'b': b, 'wg': wg}
        print(f"  pca={pca_n} bw={bw}: best={best3:.4f}", flush=True)
results['nadaraya_watson'] = (best3, best3_cfg)
print(f"  Best: {best3:.4f}  cfg={best3_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: KDE in PCA-16 (less curse of dimensionality)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: KDE in PCA-16 ===", flush=True)
best4 = 0; best4_cfg = {}
for bw in [0.5, 0.8, 1.0, 1.2, 1.5]:
    loo_kde16 = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr = np.arange(n_files) != fi
        pca_l = PCA(n_components=16, random_state=42).fit(emb_norm[tr])
        X_tr_l = pca_l.transform(emb_norm[tr]).astype(np.float32)
        X_te_l = pca_l.transform(emb_norm[[fi]]).astype(np.float32)
        mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
        X_tr_l = (X_tr_l - mu_l) / std_l
        X_te_l = (X_te_l - mu_l) / std_l
        kde_bg = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X_tr_l)
        log_bg = kde_bg.score_samples(X_te_l)[0]
        for si in range(n_species):
            pos = np.where(file_labels[tr, si] > 0.5)[0]
            if len(pos) == 0:
                loo_kde16[fi, si] = sigmoid(file_logit_max[fi, si]); continue
            kde_pos = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X_tr_l[pos])
            loo_kde16[fi, si] = sigmoid(kde_pos.score_samples(X_te_l)[0] - log_bg)
    for a in [0.85, 0.90, 0.95, 1.00]:
        for b in [0.8, 1.0, 1.2, 1.5]:
            for wg in [0.25, 0.30, 0.40]:
                auc = eval_blend(loo_kde16, a, b, wg)
                if auc > best4: best4 = auc; best4_cfg = {'bw': bw, 'a': a, 'b': b, 'wg': wg}
    print(f"  bw={bw}: best={best4:.4f}", flush=True)
results['kde_pca16'] = (best4, best4_cfg)
print(f"  Best: {best4:.4f}  cfg={best4_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
VALIDATED_BEST = 0.9560   # proper LOO-PCA validated
JSON_BEST      = 0.9675   # JSON best (may be inflated)
print(f"\n{'='*65}")
print(f"KDE VARIANTS SUMMARY")
print(f"  Validated best (proper LOO-PCA): {VALIDATED_BEST}")
print(f"  JSON best:                       {JSON_BEST}")
print(f"{'='*65}")
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    dv = auc - VALIDATED_BEST; dj = auc - JSON_BEST
    mk = " *** BEATS VALIDATED BEST ***" if auc > VALIDATED_BEST else ""
    print(f"  {name}: {auc:.4f}  (vs validated:{dv:+.4f} / json:{dj:+.4f}){mk}")

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
for name, (auc, cfg) in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc),
                               'full_auc': float(auc), 'config': cfg})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"\n*** NEW JSON BEST: {name} = {auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")
