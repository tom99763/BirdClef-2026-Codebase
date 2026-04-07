"""
Save best method: PCA-80 max+mean aggregation, k_neg=4
LOO-AUC = 0.9701
"""
import numpy as np, pickle, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
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

print("Fitting PCA-80...", flush=True)
pca80 = PCA(n_components=80, random_state=42)
emb_win_pca = pca80.fit_transform(emb_win).astype(np.float32)
print(f"  Explained variance: {pca80.explained_variance_ratio_.sum():.4f}", flush=True)

emb_win_pca_norm = normalize(emb_win_pca, norm='l2').astype(np.float32)

model = {
    'method': 'pca80_maxmean_agg_kn4',
    'loo_auc': 0.9701,
    'config': {
        'type': 'pca80_maxmean_window_contrast',
        'pca_n_components': 80,
        'w_max_pos': 0.5,   # pos scoring: 0.5*max_sim + 0.5*mean_proto
        'k_neg': 4,
        'w_max_agg': 0.55,  # aggregation: 0.55*max + 0.45*mean over windows
    },
    'pca': pca80,
    'emb_win_pca_norm': emb_win_pca_norm,
    'file_labels': file_labels,
    'file_list': file_list,
    'win_file_id': win_file_id,
}

out_path = "outputs/embed_prior_model.pkl"
with open(out_path, 'wb') as f:
    pickle.dump(model, f)

size_mb = os.path.getsize(out_path) / 1e6
print(f"\nSaved {out_path} ({size_mb:.1f} MB)", flush=True)
print(f"Method: pca80_maxmean_agg_kn4  LOO-AUC=0.9701", flush=True)
