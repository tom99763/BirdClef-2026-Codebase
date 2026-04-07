"""
Save best method: ICA-90 + Std-PCA-80 (k_neg=2) + PCA-80 triple blend
LOO-AUC = 0.9732 (vs knn5 baseline = 0.8402)
Blend: w_ica=0.4, w_std=0.08, w_base=0.52
"""
import numpy as np, pickle, os
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
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

print("Fitting components...", flush=True)

# 1. PCA-80 (base component)
print("  Fitting PCA-80...", flush=True)
pca80 = PCA(n_components=80, random_state=42)
emb_pca80 = pca80.fit_transform(emb_win).astype(np.float32)
ew_pca80 = normalize(emb_pca80, norm='l2').astype(np.float32)
print(f"  PCA-80 explained var: {pca80.explained_variance_ratio_.sum():.4f}", flush=True)

# 2. ICA-90
print("  Fitting ICA-90...", flush=True)
ica90 = FastICA(n_components=90, random_state=42, max_iter=500, tol=0.01)
emb_ica90 = ica90.fit_transform(emb_win).astype(np.float32)
ew_ica90 = normalize(emb_ica90, norm='l2').astype(np.float32)

# 3. Std-PCA-80 (StandardScaler → PCA-80)
print("  Fitting StandardScaler + PCA-80...", flush=True)
scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
emb_std_pca = pca80s.fit_transform(emb_std).astype(np.float32)
ew_std_pca = normalize(emb_std_pca, norm='l2').astype(np.float32)

model = {
    'method': 'ica90_std80kn2_pca80_triple',
    'loo_auc': 0.9732,
    'config': {
        'type': 'ica90_std80kn2_pca80_triple_blend',
        'w_ica': 0.40,         # ICA-90 weight
        'w_std': 0.08,         # Std-PCA-80 weight (k_neg=2)
        'w_base': 0.52,        # PCA-80 weight
        # Per-component params (all use maxmean_contrast)
        'pca_n_components': 80,
        'ica_n_components': 90,
        'std_pca_n_components': 80,
        'w_max_pos': 0.5,      # pos scoring weight
        'k_neg_base': 4,       # PCA-80 k_neg
        'k_neg_ica': 4,        # ICA-90 k_neg
        'k_neg_std': 2,        # Std-PCA-80 k_neg (KEY: k_neg=2 is optimal)
        'w_max_agg': 0.55,     # window aggregation: 0.55*max + 0.45*mean
    },
    # PCA-80 (base)
    'pca': pca80,
    'emb_win_pca_norm': ew_pca80,       # [739, 80] L2-normalized PCA-80 windows
    # ICA-90
    'ica': ica90,
    'emb_win_ica_norm': ew_ica90,       # [739, 90] L2-normalized ICA-90 windows
    # Std-PCA-80
    'scaler': scaler,
    'pca_std': pca80s,
    'emb_win_std_norm': ew_std_pca,     # [739, 80] L2-normalized Std-PCA-80 windows
    # Shared
    'file_labels': file_labels,          # [66, 234]
    'file_list': file_list,
    'win_file_id': win_file_id,          # [739]
}

out_path = "outputs/embed_prior_model.pkl"
with open(out_path, 'wb') as f:
    pickle.dump(model, f)

size_mb = os.path.getsize(out_path) / 1e6
print(f"\nSaved {out_path} ({size_mb:.1f} MB)", flush=True)
print(f"Method: ica90_std80kn2_pca80_triple  LOO-AUC=0.9732", flush=True)
print(f"  ICA-90: {ew_ica90.shape}  Std-PCA-80: {ew_std_pca.shape}  PCA-80: {ew_pca80.shape}", flush=True)
