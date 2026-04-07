"""
Build win-ensemble pkls with emb_win_norm for window-level KNN.
This allows window ensemble notebooks to work in production.
"""
import numpy as np, pickle, os, shutil
from sklearn.preprocessing import normalize
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# Load base pkl
with open("outputs/embed_prior_attn_k4.pkl", "rb") as f:
    base_ep = pickle.load(f)

# Load window-level embeddings from perch cache
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win = perch['emb'].astype(np.float32)
n_windows = perch['n_windows']
n_files = len(perch['file_list'])
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end = np.cumsum(n_windows).astype(np.int32)

# Window-level L2-normalized embeddings
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
print(f"Window embeddings: {emb_win_norm.shape}")

# Build win_file_id
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi
print(f"win_file_id: {win_file_id.shape}, unique={len(np.unique(win_file_id))}")

# Build LOO-AUC for reference
from sklearn.metrics import roc_auc_score
file_labels = base_ep['file_labels'].astype(np.float32)
n_species = file_labels.shape[1]

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

# Build window-level KNN pickle with emb_win_norm
def build_win_pkl(output_path):
    ep = dict(base_ep)  # copy all fields
    ep['emb_win_norm'] = emb_win_norm    # (total_windows, 1536)
    ep['win_file_id']  = win_file_id     # (total_windows,)
    ep['n_windows']    = n_windows       # (n_files,)
    ep['file_start']   = file_start
    ep['file_end']     = file_end
    ep['method'] = 'attn_k4_win_ensemble'
    ep['has_win_emb'] = True
    with open(output_path, 'wb') as f:
        pickle.dump(ep, f)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved {output_path} ({size_mb:.1f} MB)")

build_win_pkl("outputs/embed_prior_attn_k4_win.pkl")

# Copy to weights/
WEIGHTS_DIR = "birdclef-2026/notebook resource/current_subs/weights"
shutil.copy2("outputs/embed_prior_attn_k4_win.pkl", f"{WEIGHTS_DIR}/embed_prior_attn_k4_win.pkl")
print(f"  Copied to {WEIGHTS_DIR}/")

print("\ndone")
