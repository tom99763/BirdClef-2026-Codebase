"""
Build pkl for new best: ls2_geo_k5_T0.20_win_k1_wg0.50_a0.70_b1.45 (LOO=0.9164)
sigmoid(0.70 * logit_max + 1.45 * log(0.50 * geo_k5 + 0.50 * win_k1))
"""
import numpy as np, pickle, os, shutil
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

file_embs      = np.zeros((n_files, emb_win.shape[1]), np.float32)
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)

emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

# Load base pkl (X_combined_n)
with open("outputs/embed_prior_attn.pkl", "rb") as f:
    base_ep = pickle.load(f)
X_pkl  = base_ep['X_combined_n'].astype(np.float32)
fl_pkl = base_ep['file_labels'].astype(np.float32)

# Verify LOO-AUC
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS = 1e-7

def attn_knn_loo(X, k=5, T=0.2):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * fl_pkl[tr[top]]).sum(0)
    return preds

def window_knn_loo(k=1):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
        sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :k]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            w = sims[wi, top_idx[wi]].clip(0); ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k)/k
            wp[wi] = (w[:, None] * file_labels[tr_fi[top_idx[wi]]]).sum(0)
        preds[i] = wp.mean(0)
    return preds

print("Computing geo_k5 and win_k1 LOO predictions...")
y_geo_k5 = attn_knn_loo(X_pkl, k=5, T=0.2)
y_win_k1 = window_knn_loo(k=1)

# Verify LOO-AUC
W_GEO = 0.50; W_WIN = 0.50; A = 0.70; B = 1.45
y_blend = W_GEO * y_geo_k5 + W_WIN * y_win_k1
pred = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
loo_auc = macro_auc(file_labels, pred)
print(f"Verified LOO-AUC: {loo_auc:.6f} (expected ~0.9164)")

# Build pkl
pkl_data = dict(base_ep)  # copy base fields (X_combined_n, pca params, etc.)
pkl_data.update({
    'method':       'logspace_geo5_win1',
    'type':         'logspace_geo5_win1',
    'loo_auc':      loo_auc,
    # Logspace params
    'logspace_a':   A,     # coefficient for logit_max
    'logspace_b':   B,     # coefficient for log(knn_prob)
    'w_geo':        W_GEO, # weight for geo-KNN in blend
    'w_win':        W_WIN, # weight for win-KNN in blend
    'k_geo':        5,     # k for geo-KNN (X_combined_n space)
    'T_geo':        0.2,   # temperature for geo-KNN
    'k_win':        1,     # k for window-KNN
    # Window embedding data (for production inference)
    'emb_win_norm': emb_win_norm,  # (739, 1536) normalized window embeddings
    'win_file_id':  win_file_id,   # (739,) file index for each window
    'n_windows':    n_windows,
    'file_start':   file_start,
    'file_end':     file_end,
    'file_list':    np.array(file_list),
    'file_labels':  file_labels,  # override with current ground truth
})

OUT_PATH = "outputs/embed_prior_logspace_geo5_win1.pkl"
with open(OUT_PATH, 'wb') as f:
    pickle.dump(pkl_data, f)
size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"Saved {OUT_PATH} ({size_mb:.1f} MB)")

# Copy to weights/
WEIGHTS = "birdclef-2026/notebook resource/current_subs/weights"
shutil.copy2(OUT_PATH, f"{WEIGHTS}/embed_prior_logspace_geo5_win1.pkl")
print(f"Copied to {WEIGHTS}/")

print("done")
