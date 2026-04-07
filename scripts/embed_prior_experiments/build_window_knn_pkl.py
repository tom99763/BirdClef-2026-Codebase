"""
Build embed_prior_window_knn.pkl for window-level KNN inference.

Method: window_knn_k5_mean
- For each test window, find k=5 nearest training windows by cosine sim
- Weight by cosine sim (clip ≥ 0, normalize)
- Aggregate per-file by mean
LOO-AUC: ~0.8615 on 66 labeled soundscapes
"""
import numpy as np, pickle, shutil, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)         # (739, 1536)
labels_win = raw['labels'].astype(np.float32)       # (739, 234)
file_list  = raw['file_list']
n_windows  = raw['n_windows'].astype(np.int32)
n_files    = len(file_list)
n_species  = labels_win.shape[1]
n_win_total = len(emb_win)

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

# Build file-level labels
file_labels = np.zeros((n_files, n_species), dtype=np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

# Normalize window embeddings
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)

# Build win_file_id
win_file_id = np.zeros(n_win_total, dtype=np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

# Compute LOO-AUC for window_knn k=5 mean
print("Computing LOO-AUC for window_knn k=5 mean ...", flush=True)
k = 5
preds = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]   # (nw_i, 1536)
    nw_te = te_e - te_s

    tr_mask_w = win_file_id != i
    X_tr  = emb_win_norm[tr_mask_w]
    tr_win_idx = np.where(tr_mask_w)[0]
    Y_tr  = file_labels[win_file_id[tr_win_idx]]   # (N_tr_win, 234)

    sims = X_te @ X_tr.T               # (nw_i, N_tr_win)
    top  = np.argsort(-sims, axis=1)[:, :k]

    win_preds = np.zeros((nw_te, n_species), dtype=np.float32)
    for wi in range(nw_te):
        w = sims[wi, top[wi]].clip(0)
        ws = w.sum()
        if ws > 1e-8:
            w /= ws
        else:
            w = np.ones(k) / k
        win_preds[wi] = (w[:, None] * Y_tr[top[wi]]).sum(0)

    preds[i] = win_preds.mean(0)   # file-level mean

    if (i + 1) % 20 == 0:
        print(f"  fold {i+1}/{n_files} done", flush=True)

loo_auc = macro_auc(file_labels, preds)
print(f"window_knn k=5 mean LOO-AUC: {loo_auc:.4f}", flush=True)

# Save pkl
pkl_data = {
    'method': 'window_knn_k5_mean',
    'loo_auc': round(float(loo_auc), 6),
    'type': 'window_knn',
    'k': k,
    'agg': 'mean',
    # Reference data for inference
    'emb_win_norm': emb_win_norm,      # (739, 1536) normalized training window embs
    'file_labels': file_labels,        # (66, 234) training file-level labels
    'win_file_id': win_file_id,        # (739,) window → file mapping
    'file_start': file_start,          # (66,)
    'file_end': file_end,              # (66,)
    'n_windows': n_windows,            # (66,)
    'file_list': file_list,            # (66,)
    'n_files': n_files,
    'n_species': n_species,
}

out_path = "outputs/embed_prior_window_knn.pkl"
with open(out_path, "wb") as f:
    pickle.dump(pkl_data, f)
print(f"Saved: {out_path}")

weights_path = "birdclef-2026/notebook resource/current_subs/weights/embed_prior_window_knn.pkl"
shutil.copy(out_path, weights_path)
print(f"Copied to: {weights_path}")

print("done", flush=True)
