"""
Rebuild embed_prior_attn.pkl with full PCA components for inference.
Best: pca24 + day-of-year + k=10 + T=0.2 → LOO-AUC=0.8758
"""
import numpy as np, pickle, re, shutil, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

os.chdir("/home/lab/BirdClef-2026-Codebase")

raw = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win, labels_win = raw['emb'].astype(np.float32), raw['labels'].astype(np.float32)
file_list, n_windows = raw['file_list'], raw['n_windows']
n_files = len(file_list)
n_species = labels_win.shape[1]

file_embs   = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels = np.zeros((n_files, n_species), dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]   = emb_win[idx:idx+nw].mean(0)
    file_labels[fi] = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    idx += nw
file_embs_norm = normalize(file_embs, norm='l2')

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
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24), np.cos(2*np.pi*file_hours/24)], axis=1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12), np.cos(2*np.pi*(file_months-1)/12)], axis=1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365), np.cos(2*np.pi*(file_days-1)/365)], axis=1).astype(np.float32)

# PCA-24 fitted on normalized embs
pca = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X_pca  = pca.transform(file_embs_norm).astype(np.float32)
pca_std = (X_pca.std(0) + 1e-6).astype(np.float32)
X_pca_s = X_pca / pca_std

geo = np.concatenate([site_oh, hour_enc, month_enc, day_enc], axis=1).astype(np.float32)
X = np.concatenate([X_pca_s, geo], axis=1).astype(np.float32)
X_combined_n = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
print(f"X_combined_n shape: {X_combined_n.shape}")  # (66, 47)

# Verify LOO-AUC
preds = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    mask = np.ones(n_files, bool); mask[i] = False
    tr_idx = np.where(mask)[0]
    sims = (X_combined_n[[i]] @ X_combined_n[tr_idx].T).ravel()
    top  = np.argsort(-sims)[:10]
    logits = sims[top] / 0.2; logits -= logits.max()
    w = np.exp(logits); w /= w.sum()
    preds[i] = (w[:, None] * file_labels[tr_idx[top]]).sum(0)
mask_sp = file_labels.sum(0) > 0
auc = roc_auc_score(file_labels[:, mask_sp], preds[:, mask_sp], average='macro')
print(f"LOO-AUC verified: {auc:.4f}")

pkl_data = {
    'method': 'attn_k10_T0.2_pca24_day',
    'loo_auc': round(auc, 4),
    'pca_dims': 24,
    'pca_mean': pca.mean_.astype(np.float32),           # (1536,)
    'pca_components': pca.components_.astype(np.float32), # (24, 1536)
    'pca_std': pca_std,                                  # (24,)
    'geo_w': 1.0,
    'use_day': True,
    'k': 10,
    'T': 0.2,
    'temperature': 0.2,   # alias for compatibility
    'SITES': SITES,
    'site2idx': site2idx,
    'X_combined_n': X_combined_n,   # (66, 47) reference
    'combined_norm': X_combined_n,  # alias for old code compatibility
    'file_labels': file_labels,     # (66, 234)
    'file_list': file_list,
    'sites': SITES,                 # alias
}
out = "outputs/embed_prior_attn.pkl"
with open(out, "wb") as f:
    pickle.dump(pkl_data, f)
dest = "birdclef-2026/notebook resource/current_subs/weights/embed_prior_attn.pkl"
shutil.copy(out, dest)
print(f"Saved: {out}")
print(f"Copied to: {dest}")
print("Keys:", list(pkl_data.keys()))
print("done")
