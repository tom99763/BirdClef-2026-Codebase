"""
Build embed_prior_ens_nologit3.pkl
Method: 0.65 × attn_pca24+day_k10_T0.2 + 0.35 × window_knn_k1_mean
LOO-AUC: 0.8810
"""
import numpy as np, pickle, shutil, re, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

file_embs  = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels= np.zeros((n_files, n_species), dtype=np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]   = emb_win[s:e].mean(0)
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)
emb_win_norm   = normalize(emb_win,   norm='l2').astype(np.float32)

win_file_id = np.zeros(len(emb_win), dtype=np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

SITES = ['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx = {s: i for i, s in enumerate(SITES)}
file_sites  = np.zeros(n_files, dtype=np.int32)
file_hours  = np.zeros(n_files, dtype=np.float32)
file_months = np.zeros(n_files, dtype=np.float32)
file_days   = np.zeros(n_files, dtype=np.float32)
for fi, fname in enumerate(file_list):
    m = re.match(r'BC2026_Train_\d+_(S\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})', str(fname))
    if m:
        site, yr, mo, dy, hr, mn = m.groups()
        file_sites[fi]  = site2idx.get(site, 0)
        file_hours[fi]  = int(hr)
        file_months[fi] = int(mo)
        dpm = [0,31,28,31,30,31,30,31,31,30,31,30,31]
        file_days[fi] = sum(dpm[:int(mo)]) + int(dy)

site_oh   = np.eye(len(SITES), dtype=np.float32)[file_sites]
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24),
                       np.cos(2*np.pi*file_hours/24)], axis=1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12),
                       np.cos(2*np.pi*(file_months-1)/12)], axis=1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365),
                       np.cos(2*np.pi*(file_days-1)/365)], axis=1).astype(np.float32)

pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
pca24_std = X24.std(0) + 1e-6
X24s  = X24 / pca24_std
geo   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], axis=1).astype(np.float32)
X_comb = np.concatenate([X24s, geo], axis=1).astype(np.float32)
X_nl   = (X_comb / np.linalg.norm(X_comb, axis=1, keepdims=True)).astype(np.float32)

pkl_data = {
    'method': 'ens2_attn0.65_wink1_0.35',
    'loo_auc': 0.8810,
    'type': 'ens_nologit3',
    'note': '0.65×attn_pca24+day_k10_T0.2 + 0.35×window_knn_k1_mean',
    'w_attn': 0.65,
    'w_win':  0.35,
    'pca_dims': 24,
    'pca_mean': pca24.mean_.astype(np.float32),
    'pca_components': pca24.components_.astype(np.float32),
    'pca_std': pca24_std.astype(np.float32),
    'use_day': True,
    'SITES': SITES,
    'site2idx': site2idx,
    'X_combined_n': X_nl,
    'k_attn': 10,
    'T_attn': 0.2,
    'temperature': 0.2,
    'emb_win_norm': emb_win_norm,
    'win_file_id':  win_file_id,
    'k_win': 1,          # ← 關鍵：k=1
    'file_labels': file_labels,
    'file_list':   file_list,
}

out_path = "outputs/embed_prior_ens_nologit3.pkl"
with open(out_path, "wb") as f:
    pickle.dump(pkl_data, f)
print(f"Saved: {out_path}")

weights_path = "birdclef-2026/notebook resource/current_subs/weights/embed_prior_ens_nologit3.pkl"
shutil.copy(out_path, weights_path)
print(f"Copied to: {weights_path}")
print("done")
