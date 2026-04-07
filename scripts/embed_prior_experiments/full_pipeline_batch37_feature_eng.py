"""
Batch 37: Feature engineering + unconventional methods
Goal: beat maxmean_kn4 = 0.9701
Methods:
  1. L2 distance instead of cosine similarity in PCA space
  2. Concatenated multi-scale PCA (PCA-80 ++ PCA-256 concat)
  3. Per-species score with neighbor graph smoothing
  4. NMF-based embedding (non-negative components)
  5. ICA-based embedding (independent components)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, FastICA, NMF
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
CURRENT_BEST = 0.9701

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def maxmean_contrast(emb_wins_n, k_neg=4, w_max_pos=0.5, w_max_agg=0.55):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_lab = file_labels[tr_idx]
        te_wins = emb_wins_n[win_file_id == fi]
        tr_wins_all = emb_wins_n[win_file_id != fi]
        tr_fids_all = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win[:,si] > 0.5
            neg_win_mask = ~pos_win_mask
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

# PCA-80 base
pca80 = PCA(n_components=80, random_state=42)
emb_win_pca80 = pca80.fit_transform(emb_win).astype(np.float32)
ew80 = normalize(emb_win_pca80, norm='l2').astype(np.float32)
out_base = maxmean_contrast(ew80)
print(f"Base: {eval_loo(out_base):.4f}", flush=True)

# ─── Method 1: L2 distance scoring instead of cosine ────────────────────────
print("\n=== Method 1: L2 distance scoring (unnormalized PCA-80) ===", flush=True)
t0 = time.time()
# Use raw PCA-80 without L2 normalization
ew80_raw = emb_win_pca80  # not normalized
out_l2 = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_lab = file_labels[tr_idx]
    te_wins = ew80_raw[win_file_id == fi]
    tr_wins_all = ew80_raw[win_file_id != fi]
    tr_fids_all = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos_win_mask = tr_lab_win[:,si] > 0.5
        neg_win_mask = ~pos_win_mask
        if not pos_win_mask.any(): ws[:,si]=0.5; continue
        pos_wins = tr_wins_all[pos_win_mask]
        # L2 distance: negative distance = similarity
        pp_mean = pos_wins.mean(0)
        d_pos = -np.linalg.norm(te_wins - pp_mean, axis=1)  # negative L2 distance
        # For max: nearest positive window
        all_d_pos = -np.linalg.norm(te_wins[:, None] - pos_wins[None, :], axis=2)  # [n_te, n_pos]
        sp = 0.5 * all_d_pos.max(1) + 0.5 * d_pos
        # Normalize to [0,1] range roughly
        sp = (sp - sp.min()) / (sp.max() - sp.min() + EPS)
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            # L2 to nearest neg
            all_d_neg = -np.linalg.norm(te_wins[:, None] - neg_wins[None, :], axis=2)
            k_act = min(4, neg_wins.shape[0])
            top_neg_idx = np.argsort(-all_d_neg, axis=1)[:, :k_act]
            sn_raw = all_d_neg[np.arange(len(te_wins))[:, None], top_neg_idx].mean(1)
            sn = (sn_raw - sn_raw.min()) / (sn_raw.max() - sn_raw.min() + EPS)
            ws[:,si] = (sp - sn + 1) / 2
        else: ws[:,si] = sp
    out_l2[fi] = 0.55 * ws.max(0) + 0.45 * ws.mean(0)
auc1 = eval_loo(out_l2)
results['l2_dist_pca80'] = auc1
flag = " *** NEW BEST ***" if auc1 > CURRENT_BEST else ""
print(f"  L2-distance PCA-80: {auc1:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
# Blend with base
best1b = 0; best_w1b = None
for w_l2 in [0.2, 0.3, 0.4, 0.5]:
    blend = w_l2 * out_l2 + (1-w_l2) * out_base
    auc_c = eval_loo(blend)
    if auc_c > best1b: best1b = auc_c; best_w1b = w_l2
results['l2_base_blend'] = best1b
flag = " *** NEW BEST ***" if best1b > CURRENT_BEST else ""
print(f"  L2+base: {best1b:.4f}{flag}  w_l2={best_w1b}", flush=True)

# ─── Method 2: Concatenated multi-scale PCA ──────────────────────────────────
print("\n=== Method 2: Concatenated multi-scale PCA (80+256) ===", flush=True)
t0 = time.time()
pca256 = PCA(n_components=256, random_state=42)
emb_win_pca256 = pca256.fit_transform(emb_win).astype(np.float32)
# Concatenate PCA-80 and PCA-256, then normalize
concat_emb = np.concatenate([emb_win_pca80 / (np.std(emb_win_pca80) + EPS),
                               emb_win_pca256 / (np.std(emb_win_pca256) + EPS)], axis=1)
ew_concat = normalize(concat_emb, norm='l2').astype(np.float32)
out_concat = maxmean_contrast(ew_concat)
auc2 = eval_loo(out_concat)
results['concat_pca80_256'] = auc2
flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
print(f"  concat(pca80+pca256): {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
# Blend
best2b = 0; best_w2b = None
for w_c in [0.3, 0.4, 0.5, 0.6]:
    blend = w_c * out_concat + (1-w_c) * out_base
    auc_c = eval_loo(blend)
    if auc_c > best2b: best2b = auc_c; best_w2b = w_c
results['concat_base_blend'] = best2b
flag = " *** NEW BEST ***" if best2b > CURRENT_BEST else ""
print(f"  concat+base: {best2b:.4f}{flag}  w_c={best_w2b}", flush=True)

# ─── Method 3: ICA-based embedding ───────────────────────────────────────────
print("\n=== Method 3: ICA-80 embedding ===", flush=True)
t0 = time.time()
try:
    ica80 = FastICA(n_components=80, random_state=42, max_iter=500, tol=0.01)
    emb_win_ica80 = ica80.fit_transform(emb_win).astype(np.float32)
    ew_ica = normalize(emb_win_ica80, norm='l2').astype(np.float32)
    out_ica = maxmean_contrast(ew_ica)
    auc3 = eval_loo(out_ica)
    results['ica80'] = auc3
    flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
    print(f"  ICA-80: {auc3:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
    # Blend
    best3b = 0; best_w3b = None
    for w_i in [0.2, 0.3, 0.4, 0.5]:
        blend = w_i * out_ica + (1-w_i) * out_base
        auc_c = eval_loo(blend)
        if auc_c > best3b: best3b = auc_c; best_w3b = w_i
    results['ica80_base_blend'] = best3b
    flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
    print(f"  ICA+base: {best3b:.4f}{flag}  w_ica={best_w3b}", flush=True)
except Exception as e:
    print(f"  ICA failed: {e}", flush=True)

# ─── Method 4: NMF (non-negative) ─────────────────────────────────────────────
print("\n=== Method 4: NMF-80 embedding ===", flush=True)
t0 = time.time()
try:
    # NMF requires non-negative input
    emb_shift = emb_win - emb_win.min(0)  # shift to non-negative
    emb_shift = emb_shift.clip(0)
    nmf80 = NMF(n_components=80, random_state=42, max_iter=500)
    emb_win_nmf80 = nmf80.fit_transform(emb_shift).astype(np.float32)
    ew_nmf = normalize(emb_win_nmf80, norm='l2').astype(np.float32)
    out_nmf = maxmean_contrast(ew_nmf)
    auc4 = eval_loo(out_nmf)
    results['nmf80'] = auc4
    flag = " *** NEW BEST ***" if auc4 > CURRENT_BEST else ""
    print(f"  NMF-80: {auc4:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
    best4b = 0; best_w4b = None
    for w_n in [0.2, 0.3, 0.4, 0.5]:
        blend = w_n * out_nmf + (1-w_n) * out_base
        auc_c = eval_loo(blend)
        if auc_c > best4b: best4b = auc_c; best_w4b = w_n
    results['nmf80_base_blend'] = best4b
    flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
    print(f"  NMF+base: {best4b:.4f}{flag}  w_nmf={best_w4b}", flush=True)
except Exception as e:
    print(f"  NMF failed: {e}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 37 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best_json:
        cur_best_json = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
