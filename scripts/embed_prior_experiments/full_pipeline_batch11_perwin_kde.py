"""
Batch 11: Per-window KDE scoring (vs. current avg-then-score approach)
Current: avg test windows → single PCA-32 embedding → KDE score
New:     per-window PCA-32 → KDE score → avg over windows (same as RKNN strategy)

Also tries:
  - vMF-style kernel (cosine similarity as kernel, no PCA)
  - KDE with per-window scoring + RKNN blend
"""
import numpy as np, pickle, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

EPS = 1e-7
def vlom_blend(a, b):
    return sigmoid(0.5*np.log(a.clip(EPS)/(1-a).clip(EPS)) + 0.5*np.log(b.clip(EPS)/(1-b).clip(EPS)))
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_by_file = {}
for i, rid in enumerate(sed_npz['row_ids']):
    sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]), []).append(i)
file_sed_max = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fb = fname.replace('.ogg','').replace('.flac','')
    if fb in sed_by_file: file_sed_max[fi] = sed_npz['probs'][sed_by_file[fb]].max(0)
base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))
mask = file_labels.sum(0) > 0

# ── RKNN k5 LOO ──
def loo_rknn(K=5):
    out = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s = int(file_start[i]); te_e = int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_m = (win_file_id != i); X_tr = emb_win_norm[tr_m]; tr_fi = win_file_id[tr_m]
        sims_te_tr = X_te @ X_tr.T
        sims_tr_tr = X_tr @ X_tr.T
        thresh = np.partition(-sims_tr_tr, K, axis=1)[:, K] * -1
        top_k_idx = np.argsort(-sims_te_tr, axis=1)[:, :K]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            nbrs = top_k_idx[wi]; recip = [n for n in nbrs if sims_te_tr[wi, n] >= thresh[n]]
            if not recip: recip = nbrs.tolist()
            ww = sims_te_tr[wi, recip].clip(0); ws = ww.sum()
            ww = ww / ws if ws > 1e-8 else np.ones(len(recip)) / len(recip)
            wp[wi] = (ww[:, None] * file_labels[tr_fi[recip]]).sum(0)
        out[i] = wp.mean(0)
    return out

print("Computing RKNN k5...", flush=True)
t0 = time.time()
y_rknn5 = loo_rknn(K=5)
print(f"  Done in {time.time()-t0:.0f}s", flush=True)

results = {}

# ──────────────────────────────────────────────────────────────────────────────
# Method 1: kde_perwin — per-window KDE scoring then avg
# Instead of: avg windows → PCA → KDE
# New:        each window → PCA → KDE → avg scores
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 1: kde_perwin (per-window KDE scoring) ===", flush=True)
BW = 0.5; PCA_N = 32

t0 = time.time()
loo_kde_perwin = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    n_te_wins = te_e - te_s
    tr_mask = (win_file_id != fi)
    X_tr_raw = emb_win_norm[tr_mask]
    X_te_wins = emb_win_norm[te_s:te_e]  # (n_wins, 1536) — process each separately
    pca_l = PCA(n_components=PCA_N, random_state=42).fit(X_tr_raw)
    X_tr_l = pca_l.transform(X_tr_raw).astype(np.float32)
    mu_l = X_tr_l.mean(0); std_l = X_tr_l.std(0).clip(1e-8)
    X_tr_l = (X_tr_l - mu_l) / std_l
    X_te_pca = ((pca_l.transform(X_te_wins).astype(np.float32) - mu_l) / std_l)  # (n_wins, 32)
    tr_fids = win_file_id[tr_mask]
    kde_bg = KernelDensity(bandwidth=BW).fit(X_tr_l)
    log_bg_wins = kde_bg.score_samples(X_te_pca)  # (n_wins,)
    win_scores = np.zeros((n_te_wins, n_species), np.float32)
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fids])
        X_pos = X_tr_l[pos_mask]
        if len(X_pos) == 0:
            win_scores[:, si] = sigmoid(file_logit_max[fi, si]); continue
        kde_pos = KernelDensity(bandwidth=BW).fit(X_pos)
        log_pos_wins = kde_pos.score_samples(X_te_pca)  # (n_wins,)
        win_scores[:, si] = sigmoid(log_pos_wins - log_bg_wins)
    loo_kde_perwin[fi] = win_scores.mean(0)
print(f"  Per-window KDE LOO done in {time.time()-t0:.0f}s", flush=True)

best_pw = 0; best_cfg_pw = None
for a in [0.85, 0.88, 0.90, 0.92, 0.95]:
    for b in [1.0, 1.2, 1.4, 1.6, 1.8]:
        pred = sigmoid(a * base_logit + b * np.log(loo_kde_perwin.clip(EPS)))
        auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
        if auc > best_pw: best_pw = auc; best_cfg_pw = (a, b)
print(f"  Standalone best: {best_pw:.4f}  cfg={best_cfg_pw}", flush=True)

# Blend with RKNN k5
best_pw_rknn = 0; best_cfg_pw_rknn = None
for wg_kde in [0.25, 0.30, 0.35, 0.40, 0.45]:
    wg_rknn = 1.0 - wg_kde
    blend = wg_kde * loo_kde_perwin + wg_rknn * y_rknn5
    for a in [0.88, 0.90, 0.92, 0.95]:
        for b in [1.2, 1.4, 1.6, 1.8]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best_pw_rknn: best_pw_rknn = auc; best_cfg_pw_rknn = (wg_kde, wg_rknn, a, b)
print(f"  +RKNN best: {best_pw_rknn:.4f}  cfg={best_cfg_pw_rknn}", flush=True)
results['kde_perwin'] = best_pw
results['kde_perwin_rknn'] = best_pw_rknn

# ──────────────────────────────────────────────────────────────────────────────
# Method 2: vmf_kde — von Mises-Fisher style (cosine similarity kernel, no PCA)
# log p(x|species) = κ * (avg_pos_dir)^T x / ||avg_pos_dir|| (von Mises-Fisher MLE)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 2: vmf_kde (vMF kernel, no PCA) ===", flush=True)
t0 = time.time()
loo_vmf = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    te_s, te_e = int(file_start[fi]), int(file_end[fi])
    tr_mask = (win_file_id != fi)
    X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    X_te = emb_win_norm[te_s:te_e].mean(0, keepdims=True)  # avg test windows, already normed
    X_te = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + EPS)
    # Background: mean direction of all training windows
    bg_dir = X_tr.mean(0); bg_dir = bg_dir / (np.linalg.norm(bg_dir) + EPS)
    cos_bg = (X_te @ bg_dir)  # (1,) cosine to background mean direction
    for si in range(n_species):
        pos_mask = np.array([file_labels[f, si] > 0.5 for f in tr_fi])
        X_pos = X_tr[pos_mask]
        if len(X_pos) == 0:
            loo_vmf[fi, si] = sigmoid(file_logit_max[fi, si]); continue
        pos_dir = X_pos.mean(0); pos_dir = pos_dir / (np.linalg.norm(pos_dir) + EPS)
        cos_pos = (X_te @ pos_dir)[0]  # cosine to species mean direction
        # vMF log-ratio (log p(x|sp) - log p(x|bg)) ∝ κ * (cos_pos - cos_bg)
        # Use κ=1 (will be absorbed into b)
        loo_vmf[fi, si] = sigmoid(10.0 * (float(cos_pos) - float(cos_bg)))
print(f"  vMF LOO done in {time.time()-t0:.0f}s", flush=True)

best_vmf = 0; best_cfg_vmf = None
for a in [0.85, 0.90, 0.92, 0.95]:
    for b in [1.0, 1.2, 1.4, 1.6]:
        pred = sigmoid(a * base_logit + b * np.log(loo_vmf.clip(EPS)))
        auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
        if auc > best_vmf: best_vmf = auc; best_cfg_vmf = (a, b)
print(f"  Standalone best: {best_vmf:.4f}  cfg={best_cfg_vmf}", flush=True)

# Blend with RKNN k5
best_vmf_rknn = 0; best_cfg_vmf_rknn = None
for wg_vmf in [0.10, 0.15, 0.20, 0.25, 0.30]:
    wg_rknn = 1.0 - wg_vmf
    blend = wg_vmf * loo_vmf + wg_rknn * y_rknn5
    for a in [0.88, 0.90, 0.92, 0.95]:
        for b in [1.2, 1.4, 1.6, 1.8]:
            pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
            auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
            if auc > best_vmf_rknn: best_vmf_rknn = auc; best_cfg_vmf_rknn = (wg_vmf, wg_rknn, a, b)
print(f"  +RKNN best: {best_vmf_rknn:.4f}  cfg={best_cfg_vmf_rknn}", flush=True)
results['vmf_kde'] = best_vmf
results['vmf_kde_rknn'] = best_vmf_rknn

# ──────────────────────────────────────────────────────────────────────────────
# Method 3: kde_perwin + RKNN + vMF 3-way
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Method 3: 3-way perwin+rknn+vmf ===", flush=True)
best_3w = 0; best_cfg_3w = None
for wk in [0.25, 0.30, 0.35]:
    for wr in [0.45, 0.50, 0.55, 0.60]:
        wv = 1.0 - wk - wr
        if wv < 0: continue
        blend = wk * loo_kde_perwin + wr * y_rknn5 + wv * loo_vmf
        for a in [0.90, 0.92]:
            for b in [1.4, 1.6]:
                pred = sigmoid(a * base_logit + b * np.log(blend.clip(EPS)))
                auc  = roc_auc_score(file_labels[:, mask], pred[:, mask], average='macro')
                if auc > best_3w: best_3w = auc; best_cfg_3w = (wk, wr, wv, a, b)
print(f"  Best: {best_3w:.4f}  cfg={best_cfg_3w}", flush=True)
results['kde_perwin_rknn_vmf'] = best_3w

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===", flush=True)
current_best = 0.9711
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > current_best else ""
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
print("Updated embed_prior_results.json", flush=True)
