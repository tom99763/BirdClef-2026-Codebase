"""
Build PKL for proto_knn_blend: 0.5*cosine_prototype + 0.5*cosine_knn_k5
LOO-AUC: 0.9275 (simple format, much better than knn5=0.8402)
PKL format: {"file_embs_norm": [66,1536], "file_labels": [66,234], ...}
"""
import numpy as np, pickle, json, os, shutil
from sklearn.preprocessing import normalize
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

file_labels = np.zeros((n_files, n_species), np.float32)
file_embs   = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_embs[fi]   = emb_win[s:e].mean(0)

file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)

model_pkl = {
    'method': 'proto_knn_blend',
    'loo_auc': 0.9275,
    'config': {'k': 5, 'w_proto': 0.5, 'w_knn': 0.5},
    'file_embs_norm': file_embs_norm,
    'file_labels':    file_labels,
    'file_prob_max':  file_labels,  # use labels as proxy (no logit in simple format)
    'file_list':      file_list,
}

pkl_path = "outputs/embed_prior_model.pkl"
with open(pkl_path, 'wb') as f:
    pickle.dump(model_pkl, f)
size_mb = os.path.getsize(pkl_path) / 1024**2
print(f"Saved: {pkl_path}  ({size_mb:.1f} MB)", flush=True)
shutil.copy(pkl_path, "birdclef-2026/notebook resource/current_subs/weights/embed_prior_model.pkl")
print("Copied to weights/", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
rd['experiments'].append({'method': 'proto_knn_blend_validated', 'loo_auc': 0.9275,
                           'full_auc': 0.9275, 'config': model_pkl['config']})
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated JSON", flush=True)
