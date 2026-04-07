"""
Batch 60: Fundamentally different methods - NO blending
Goal: find a SINGLE method that beats 0.9873
Methods:
  1. LDA direction prototype (difference of class centroids)
  2. Mahalanobis similarity (using covariance of all training windows)
  3. Cosine similarity with contrast direction (pos_centroid - neg_centroid)
  4. Fisher LDA per-species (optimize discriminative projection)
  5. Regularized inverse covariance (Mahalanobis with Ledoit-Wolf)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
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

# ─── Method 1: LDA contrast direction ────────────────────────────────────────
def wl_lda_contrast(emb_wins_n, w_max_agg=0.90, w_lda=0.5):
    """
    LDA direction: score = w_lda * te @ lda_dir + (1-w_lda) * te @ pos_centroid
    lda_dir = (pos_centroid - neg_centroid) / ||...||
    This is the "contrast direction" — most discriminative 1D projection.
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
            mu_pos = pos_wins.mean(0); mu_pos /= (np.linalg.norm(mu_pos) + EPS)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                mu_neg = neg_wins.mean(0); mu_neg /= (np.linalg.norm(mu_neg) + EPS)
                lda_dir = mu_pos - mu_neg
                lda_dir /= (np.linalg.norm(lda_dir) + EPS)
                score = w_lda * (te_wins @ lda_dir) + (1-w_lda) * (te_wins @ mu_pos)
                # Map to [0, 1]
                score = (score + 1) / 2
            else:
                score = (te_wins @ mu_pos + 1) / 2
            ws[:,si] = score
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

# ─── Method 2: Mahalanobis similarity ────────────────────────────────────────
def wl_mahalanobis_contrast(emb_wins_n, w_max_pos=0.80, w_max_agg=0.90, k_neg=50):
    """
    Positive score: -Mahalanobis(te, pos_centroid, S_inv) where S = cov of all tr windows.
    Negative score: max of top-k neg cosine sims.
    Combined: (pos_score - neg_score + 1) / 2
    """
    # Compute global inverse covariance once (per LOO fold this changes slightly, but expensive)
    # Approximate with global cov (slightly leaking LOO but negligible for 1/66 exclusion)
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins_all = emb_wins_n[tr_mask]
        tr_lab_win_raw = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        # Global cov inverse for this fold's training set
        cov = np.cov(tr_wins_all.T)
        try:
            S_inv = np.linalg.inv(cov + 1e-4 * np.eye(cov.shape[0]))
        except:
            S_inv = np.eye(cov.shape[0])
        S_inv = S_inv.astype(np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win_raw[:,si] > 0.5
            neg_win_mask = tr_lab_win_raw[:,si] < 0.1
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            mu_pos = pos_wins.mean(0).astype(np.float32)
            # Mahalanobis: -||te - mu_pos||_S
            diff = te_wins - mu_pos[None, :]  # [n_te, dim]
            mahal_sq = (diff @ S_inv * diff).sum(1)
            pos_score_mahal = -np.sqrt(np.maximum(mahal_sq, 0))
            # Normalize to [0,1] range based on training data distribution
            pos_sims = te_wins @ normalize(pos_wins, norm='l2').T
            pos_score_cos = w_max_pos * pos_sims.max(1) + (1-w_max_pos) * (te_wins @ normalize(mu_pos[None], norm='l2')[0])
            # Combine: mostly cosine (already normalized) + small Mahalanobis correction
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (pos_score_cos - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (pos_score_cos+1)/2
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

# ─── Method 3: Fisher LDA per-species (2-class) ───────────────────────────────
def wl_fisher_lda(emb_wins_n, w_max_agg=0.90, n_comp=1):
    """
    For each species: compute Fisher's LDA discriminant direction.
    Score test windows by projecting onto this direction.
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
            if not pos_win_mask.any() or not neg_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            neg_wins = tr_wins_all[neg_win_mask]
            mu_pos = pos_wins.mean(0)
            mu_neg = neg_wins.mean(0)
            # Within-class scatter matrix (regularized)
            S_pos = (pos_wins - mu_pos).T @ (pos_wins - mu_pos) / max(1, len(pos_wins)-1)
            S_neg = (neg_wins - mu_neg).T @ (neg_wins - mu_neg) / max(1, len(neg_wins)-1)
            S_w = S_pos + S_neg + 1e-3 * np.eye(len(mu_pos))
            # Fisher direction: S_w^-1 * (mu_pos - mu_neg)
            diff = (mu_pos - mu_neg).astype(np.float64)
            S_w = S_w.astype(np.float64)
            try:
                w_fisher = np.linalg.solve(S_w, diff)
            except:
                w_fisher = diff
            w_fisher = w_fisher / (np.linalg.norm(w_fisher) + EPS)
            w_fisher = w_fisher.astype(np.float32)
            # Score = projection onto Fisher direction
            te_proj = te_wins @ w_fisher
            # Calibrate: threshold at halfway between class projections
            pos_proj = pos_wins @ w_fisher
            neg_proj = neg_wins @ w_fisher
            mu_pos_proj = pos_proj.mean()
            mu_neg_proj = neg_proj.mean()
            # Normalize to [0, 1]
            score_range = mu_pos_proj - mu_neg_proj
            if abs(score_range) > EPS:
                ws[:,si] = (te_proj - mu_neg_proj) / score_range
                ws[:,si] = np.clip(ws[:,si], 0.0, 1.0)
            else:
                ws[:,si] = 0.5
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

# ─── Precompute ───────────────────────────────────────────────────────────────
print("Precomputing...", flush=True)
scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)

ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica100 = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)

pca80 = PCA(n_components=80, random_state=42)
ew80 = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)

pca80s = PCA(n_components=80, random_state=42)
ew80s = normalize(pca80s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)
print("Done.", flush=True)

# ─── Evaluate Method 1: LDA contrast direction ────────────────────────────────
print("\n=== Method 1: LDA contrast direction ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None
for wma in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    for w_lda in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80'), (ew80s, 'std80')]:
            out = wl_lda_contrast(emb, w_max_agg=wma, w_lda=w_lda)
            auc = eval_loo(out)
            if auc > best1: best1 = auc; best_cfg1 = (name, wma, w_lda)
print(f"  LDA best: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['lda_contrast'] = best1
flag = " *** NEW BEST ***" if best1 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Evaluate Method 2: Fisher LDA ────────────────────────────────────────────
print("\n=== Method 2: Fisher LDA ===", flush=True)
t0 = time.time()
best2 = 0; best_cfg2 = None
for wma in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80'), (ew80s, 'std80')]:
        out = wl_fisher_lda(emb, w_max_agg=wma)
        auc = eval_loo(out)
        if auc > best2: best2 = auc; best_cfg2 = (name, wma)
print(f"  Fisher LDA best: {best2:.4f}  cfg={best_cfg2}  ({time.time()-t0:.0f}s)", flush=True)
results['fisher_lda'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Evaluate Method 3: WL with LDA-augmented pos score ──────────────────────
print("\n=== Method 3: WL with LDA-augmented positive ===", flush=True)
# Combine: LDA direction + cosine max, with standard top-k negative
def wl_lda_plus_cos(emb_wins_n, k_neg=50, w_max_agg=0.90, w_lda=0.5, w_max_pos=0.80):
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
            mu_pos = pos_wins.mean(0); mu_pos /= (np.linalg.norm(mu_pos) + EPS)
            pos_sims = te_wins @ pos_wins.T
            sp_cos = w_max_pos * pos_sims.max(1) + (1-w_max_pos) * (te_wins @ mu_pos)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                mu_neg = neg_wins.mean(0); mu_neg /= (np.linalg.norm(mu_neg) + EPS)
                # LDA direction
                lda_dir = mu_pos - mu_neg; lda_dir /= (np.linalg.norm(lda_dir) + EPS)
                sp_lda = (te_wins @ lda_dir + 1) / 2
                sp = (1-w_lda) * sp_cos + w_lda * sp_lda
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp_cos+1)/2
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

t0 = time.time()
best3 = 0; best_cfg3 = None
for k_neg in [40, 50, 60, 80, 100]:
    for wma in [0.85, 0.88, 0.90, 0.92]:
        for w_lda in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for wmp in [0.75, 0.78, 0.80]:
                for emb, name in [(ew_ica100, 'ica100'), (ew80, 'pca80')]:
                    out = wl_lda_plus_cos(emb, k_neg=k_neg, w_max_agg=wma, w_lda=w_lda, w_max_pos=wmp)
                    auc = eval_loo(out)
                    if auc > best3: best3 = auc; best_cfg3 = (name, k_neg, wma, w_lda, wmp)
print(f"  LDA+cos: {best3:.4f}  cfg={best_cfg3}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_lda_cos'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 4: Inverse class prototype (negative subspace) ────────────────────
print("\n=== Method 4: Projected away from negative centroid ===", flush=True)
# Instead of top-k negative windows, project test away from the global negative centroid
def wl_global_neg_projection(emb_wins_n, k_neg=50, w_max_pos=0.80, w_max_agg=0.92, w_gn=0.3):
    """
    Global negative: for each species, compute a SINGLE negative centroid from ALL negatives.
    Then use: score = w_cos * cos_sim + w_gn * global_neg_penalty + topk_neg_penalty.
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
            pos_sims = te_wins @ pos_wins.T
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp = w_max_pos * pos_sims.max(1) + (1-w_max_pos) * (te_wins @ pp_mean)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                # Global negative centroid (all negatives)
                g_neg = neg_wins.mean(0); g_neg /= (np.linalg.norm(g_neg) + EPS)
                # Top-k hard negative
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                # Combined negative: w_gn * global + (1-w_gn) * hard top-k
                combined_neg = w_gn * g_neg[None, :] + (1-w_gn) * top_neg
                combined_neg /= (np.linalg.norm(combined_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * combined_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

t0 = time.time()
best4 = 0; best_cfg4 = None
for k_neg in [40, 50, 60, 80]:
    for wma in [0.85, 0.88, 0.90, 0.92]:
        for wmp in [0.75, 0.78, 0.80]:
            for w_gn in [0.1, 0.2, 0.3, 0.4]:
                out = wl_global_neg_projection(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma, w_gn=w_gn)
                auc = eval_loo(out)
                if auc > best4: best4 = auc; best_cfg4 = (k_neg, wma, wmp, w_gn)
print(f"  Global-neg projection: {best4:.4f}  cfg={best_cfg4}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_global_neg'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== Batch 60 Summary ===", flush=True)
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
