"""
Find exact best UH-triple params and save pkl (LOO=0.9873).
WL-ICA-100 ultra-high k_neg + WL-Std-PCA-80 + WL-PCA-80 triple.
"""
import numpy as np, json, pickle, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
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

EPS = 1e-7; mask = file_labels.sum(0) > 0

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')

def winlabel_contrast(emb_wins_n, k_neg=4, w_max_pos=0.5, w_max_agg=0.55):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins_all = emb_wins_n[tr_mask]
        tr_lab_win_raw = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win_raw[:,si] > 0.5
            neg_win_mask = tr_lab_win_raw[:,si] < 0.1
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp = w_max_pos * pos_sims.max(1) + (1-w_max_pos) * (te_wins @ pp_mean)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = w_max_agg * ws.max(0) + (1-w_max_agg) * ws.mean(0)
    return out

print("Fitting components...", flush=True)
pca80 = PCA(n_components=80, random_state=42)
ew80 = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)

scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
ew80s = normalize(pca80s.fit_transform(emb_std).astype(np.float32), norm='l2').astype(np.float32)

ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica100 = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2').astype(np.float32)
print("Done.", flush=True)

# Fixed baselines
out_wl80 = winlabel_contrast(ew80, k_neg=4, w_max_pos=0.70, w_max_agg=0.60)
out_wl_std = winlabel_contrast(ew80s, k_neg=4, w_max_pos=0.60, w_max_agg=0.65)
print(f"WL-PCA-80: {eval_loo(out_wl80):.4f}", flush=True)
print(f"WL-Std-PCA-80: {eval_loo(out_wl_std):.4f}", flush=True)

# Find best UH config for ICA-100
print("\nFinding best ICA-100 UH config...", flush=True)
t0 = time.time()
best_uh = 0; best_cfg_uh = None; best_out_uh = None
for k_neg in [40, 50, 60, 70, 80, 100, 120]:
    for wma in [0.80, 0.83, 0.85, 0.87, 0.88, 0.90, 0.92]:
        for wmp in [0.72, 0.73, 0.75, 0.77, 0.78, 0.80, 0.82]:
            out = winlabel_contrast(ew_ica100, k_neg=k_neg, w_max_pos=wmp, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_uh: best_uh = auc; best_cfg_uh = (k_neg, wma, wmp); best_out_uh = out
print(f"  ICA-100 UH best: {best_uh:.4f}  cfg={best_cfg_uh}  ({time.time()-t0:.0f}s)", flush=True)

# Find best triple blend with ultrafine grid
print("\nFinding best triple blend (ultrafine)...", flush=True)
best_trip = 0; best_cfg_trip = None; best_out_trip = None
for w_ica in np.arange(0.30, 0.70, 0.005):
    for w_std in np.arange(0.10, 0.50, 0.005):
        w_pca = 1.0 - w_ica - w_std
        if w_pca < 0.05 or w_pca > 0.55: continue
        blend = w_ica * best_out_uh + w_std * out_wl_std + w_pca * out_wl80
        auc = eval_loo(blend)
        if auc > best_trip: best_trip = auc; best_cfg_trip = (float(w_ica), float(w_std), float(w_pca)); best_out_trip = blend

print(f"  WL-UH-triple: {best_trip:.4f}  blend={best_cfg_trip}", flush=True)
print(f"  ICA-100 config: k_neg={best_cfg_uh[0]}, wma={best_cfg_uh[1]}, wmp={best_cfg_uh[2]}", flush=True)

# Save pkl
model = {
    'method': 'wl_ica100_uh_std80_pca80_triple',
    'loo_auc': float(best_trip),
    'config': {
        'type': 'winlabel_uh_triple_blend',
        'description': 'WL Window-label: ICA-100(uh) + Std-PCA-80 + PCA-80',
        # Triple blend weights
        'w_ica100': float(best_cfg_trip[0]),
        'w_std': float(best_cfg_trip[1]),
        'w_pca80': float(best_cfg_trip[2]),
        # Per-component WL contrast params
        'pca80': {'k_neg': 4, 'w_max_agg': 0.60, 'w_max_pos': 0.70},
        'std_pca80': {'k_neg': 4, 'w_max_agg': 0.65, 'w_max_pos': 0.60},
        'ica100': {'k_neg': best_cfg_uh[0], 'w_max_agg': best_cfg_uh[1], 'w_max_pos': best_cfg_uh[2]},
        # Architecture
        'pca_n_components': 80,
        'ica_n_components': 100,
        'std_pca_n_components': 80,
    },
    # PCA-80
    'pca': pca80,
    'emb_win_pca_norm': ew80,    # [739, 80]
    # ICA-100
    'ica': ica100,
    'emb_win_ica_norm': ew_ica100,   # [739, 100]
    # Std-PCA-80
    'scaler': scaler,
    'pca_std': pca80s,
    'emb_win_std_norm': ew80s,   # [739, 80]
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
print(f"Method: wl_ica100_uh_std80_pca80_triple  LOO-AUC={best_trip:.4f}", flush=True)
print(f"  Blend weights: w_ica100={best_cfg_trip[0]:.3f}, w_std={best_cfg_trip[1]:.3f}, w_pca80={best_cfg_trip[2]:.3f}", flush=True)
print(f"  ICA-100: k_neg={best_cfg_uh[0]}, wma={best_cfg_uh[1]}, wmp={best_cfg_uh[2]}", flush=True)
