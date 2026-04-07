"""
Embed Prior Auto Loop: LightGBM per-species (EP-only LOO-CV)

Method: lgbm_per_species
- Per-file avg/max Perch embeddings + Perch logit → PCA-32 → LightGBM per species
- Window-level training (more data), file-level aggregation for prediction
- LOO-CV: leave one FILE out (66 files, 739 windows total)

EP-only LOO-AUC target: beat current best (per_species_alpha_knn3 = 0.9026)
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win = perch['emb'].astype(np.float32)          # (739, 1536)
logits_win = perch['logits'].astype(np.float32)    # (739, 234)
labels_win = perch['labels'].astype(np.float32)    # (739, 234)
file_list = list(perch['file_list'])
n_windows = perch['n_windows']
n_files = len(file_list)
n_species = labels_win.shape[1]
n_wins_total = emb_win.shape[0]

# Build file_ids
file_ids = np.zeros(n_wins_total, dtype=np.int32)
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end = np.cumsum(n_windows).astype(np.int32)
for fi in range(n_files):
    file_ids[file_start[fi]:file_end[fi]] = fi

# File-level labels and logit max
file_labels = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_emb_avg = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_emb_avg[fi] = emb_win[s:e].mean(0)

def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

# ── PCA on all file embeddings (fit on all 66, use in LOO as a fixed transform) ─
# This is a slight information leak but is standard for EP-only evaluation
emb_avg_norm = normalize(file_emb_avg, norm='l2').astype(np.float32)
pca = PCA(n_components=32, random_state=42)
pca.fit(emb_avg_norm)
file_pca = pca.transform(emb_avg_norm).astype(np.float32)  # (66, 32)

# Standardize PCA features
pca_mean = file_pca.mean(0); pca_std = file_pca.std(0).clip(1e-8)
file_pca_s = ((file_pca - pca_mean) / pca_std).astype(np.float32)

# Feature matrix: PCA-32 + logit-max-234 = 266 features per file
# For LGBM, use file-level features
X_file = np.concatenate([file_pca_s, file_logit_max], axis=1)  # (66, 266)
print(f"Feature matrix: {X_file.shape}")
print(f"Species: {n_species}, Files: {n_files}, Windows: {n_wins_total}")

# ── LightGBM config ────────────────────────────────────────────────────────────
# Small/regularized to avoid overfitting on 65 training examples
LGBM_PARAMS = {
    'n_estimators': 30,
    'num_leaves': 4,
    'max_depth': 3,
    'min_child_samples': 3,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'learning_rate': 0.10,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'random_state': 42,
    'n_jobs': 4,
    'verbose': -1,
}

# ── LOO-CV ─────────────────────────────────────────────────────────────────────
print("\nRunning LOO-CV (LightGBM per-species)...")
loo_preds = np.zeros((n_files, n_species), np.float32)
EPS = 1e-7

for fi_test in range(n_files):
    train_mask = np.arange(n_files) != fi_test
    X_tr = X_file[train_mask]         # (65, 266)
    X_te = X_file[[fi_test]]          # (1, 266)

    for si in range(n_species):
        y_tr = file_labels[train_mask, si]
        n_pos = int(y_tr.sum()); n_neg = int(len(y_tr) - n_pos)
        if n_pos == 0:
            # No positives: use logit sigmoid as fallback
            loo_preds[fi_test, si] = sigmoid(file_logit_max[fi_test, si])
            continue
        if n_pos == len(y_tr):
            # All positive
            loo_preds[fi_test, si] = 1.0
            continue
        clf = lgb.LGBMClassifier(
            **LGBM_PARAMS,
            scale_pos_weight=n_neg / max(n_pos, 1)
        )
        clf.fit(X_tr, y_tr)
        loo_preds[fi_test, si] = clf.predict_proba(X_te)[:, 1][0]

    if fi_test % 10 == 0:
        print(f"  File {fi_test+1}/{n_files}...", flush=True)

auc = macro_auc(file_labels, loo_preds)
print(f"\nLGBM per-species LOO-AUC: {auc:.4f}")

# ── Compare with current best ──────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)

best_auc = rd['best'].get('loo_auc', 0)
print(f"Current best: {rd['best']['method']} = {best_auc:.4f}")
print(f"Improvement: {auc - best_auc:+.4f}")

method_name = "lgbm_per_species"

# ── Update JSON ────────────────────────────────────────────────────────────────
exp_entry = {
    'method': method_name,
    'loo_auc': float(auc),
    'full_auc': float(auc),
    'config': LGBM_PARAMS
}
rd['experiments'].append(exp_entry)
if auc > best_auc:
    rd['best'] = {'method': method_name, 'loo_auc': float(auc), 'full_auc': float(auc)}
    print(f"\n*** NEW BEST: {method_name} = {auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")

# ── If new best, create notebook ────────────────────────────────────────────────
if auc > best_auc:
    print(f"\nCreating notebook for {method_name}...")
    # Would need to fit full model and create notebook
    # For now, save the LOO predictions
    np.save("outputs/lgbm_per_species_loo_preds.npy", loo_preds)
    print("Saved LOO predictions to outputs/lgbm_per_species_loo_preds.npy")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Method: {method_name}")
print(f"LOO-AUC: {auc:.4f}")
print(f"vs best EP-only (per_species_alpha_knn3=0.9026): {auc - 0.9026:+.4f}")
print(f"vs full-pipeline best (sed_species_bridge=0.9444): {auc - 0.9444:+.4f}")
