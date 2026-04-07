"""
Save WL-triple (window-level label) model
WL-ICA-100 + WL-Std-PCA-80 + WL-PCA-80
LOO-AUC = 0.9853

Configs:
  WL-PCA-80:     k_neg=4, w_max_agg=0.60, w_max_pos=0.70
  WL-Std-PCA-80: k_neg=4, w_max_agg=0.65, w_max_pos=0.60
  WL-ICA-100:    k_neg=8, w_max_agg=0.75, w_max_pos=0.70
  Triple blend:  w_ica100=0.30, w_std=0.20, w_pca80=0.50
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

# 1. PCA-80
print("  Fitting PCA-80...", flush=True)
pca80 = PCA(n_components=80, random_state=42)
emb_pca80 = pca80.fit_transform(emb_win).astype(np.float32)
ew_pca80 = normalize(emb_pca80, norm='l2').astype(np.float32)

# 2. ICA-100
print("  Fitting ICA-100...", flush=True)
ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
emb_ica100 = ica100.fit_transform(emb_win).astype(np.float32)
ew_ica100 = normalize(emb_ica100, norm='l2').astype(np.float32)

# 3. StandardScaler + PCA-80
print("  Fitting Std-PCA-80...", flush=True)
scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
emb_std_pca = pca80s.fit_transform(emb_std).astype(np.float32)
ew_std_pca = normalize(emb_std_pca, norm='l2').astype(np.float32)

model = {
    'method': 'wl_ica100_std80_pca80_triple',
    'loo_auc': 0.9853,
    'config': {
        'type': 'winlabel_triple_blend',
        'description': 'Window-level labels: ICA-100 + Std-PCA-80 + PCA-80',
        # Triple blend weights
        'w_ica100': 0.30,
        'w_std': 0.20,
        'w_pca80': 0.50,
        # Per-component WL contrast params
        'pca80': {'k_neg': 4, 'w_max_agg': 0.60, 'w_max_pos': 0.70},
        'std_pca80': {'k_neg': 4, 'w_max_agg': 0.65, 'w_max_pos': 0.60},
        'ica100': {'k_neg': 8, 'w_max_agg': 0.75, 'w_max_pos': 0.70},
        # Architecture
        'pca_n_components': 80,
        'ica_n_components': 100,
        'std_pca_n_components': 80,
    },
    # PCA-80
    'pca': pca80,
    'emb_win_pca_norm': ew_pca80,    # [739, 80]
    # ICA-100
    'ica': ica100,
    'emb_win_ica_norm': ew_ica100,   # [739, 100]
    # Std-PCA-80
    'scaler': scaler,
    'pca_std': pca80s,
    'emb_win_std_norm': ew_std_pca,  # [739, 80]
    # Window-level labels (KEY: use window labels for prototype building)
    'labels_win': labels_win,        # [739, 234] window-level binary labels
    # Shared
    'file_labels': file_labels,      # [66, 234] file-level labels
    'file_list': file_list,
    'win_file_id': win_file_id,      # [739]
}

out_path = "outputs/embed_prior_model.pkl"
with open(out_path, 'wb') as f:
    pickle.dump(model, f)

size_mb = os.path.getsize(out_path) / 1e6
print(f"\nSaved {out_path} ({size_mb:.1f} MB)", flush=True)
print(f"Method: wl_ica100_std80_pca80_triple  LOO-AUC=0.9853", flush=True)
print(f"  Shapes: ICA-100={ew_ica100.shape} Std-PCA-80={ew_std_pca.shape} PCA-80={ew_pca80.shape}", flush=True)
print(f"  Window labels: {labels_win.shape}", flush=True)
