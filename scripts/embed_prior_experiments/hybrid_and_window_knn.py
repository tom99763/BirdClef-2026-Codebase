"""
兩種新方法：
A) hybrid_logit_nologit — 在同一 LOO loop 中同時計算：
   - logit-based: prob_max×0.76 + prob_mean×0.24, k=3 cosine KNN (= current best 0.9055)
   - nologit: Attn-KNN pca24+day, k=10, T=0.2 (= best nologit 0.8758)
   → blend α×logit + (1-α)×nologit，sweep α=0.70..0.98

B) window_knn_agg — 視窗層級 KNN（而非先 file-level 平均再 KNN）：
   - 對每個 test window，在全部 training windows 中找 k 個最近鄰
   - 聚合方式：file-level max / mean / 加權 max
   → 不使用 logit，純 embedding
"""
import numpy as np, json, pickle, re, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────
raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)         # (739, 1536)
labels_win = raw['labels'].astype(np.float32)       # (739, 234)
logits_win = raw['logits'].astype(np.float32)       # (739, 234)
file_list  = raw['file_list']                       # (66,)
n_windows  = raw['n_windows']                       # (66,)
n_files    = len(file_list)
n_species  = labels_win.shape[1]
n_win_total = len(emb_win)

# File boundary indices
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])])
file_end   = np.cumsum(n_windows)

# ── File-level aggregation ─────────────────────────────────────────────────
file_embs   = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels = np.zeros((n_files, n_species), dtype=np.float32)
file_prob_max  = np.zeros((n_files, n_species), dtype=np.float32)
file_prob_mean = np.zeros((n_files, n_species), dtype=np.float32)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]   = emb_win[s:e].mean(0)
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    logit_block = logits_win[s:e]          # (nw, 234)
    file_prob_max[fi]  = _sigmoid(logit_block.max(0))
    file_prob_mean[fi] = _sigmoid(logit_block.mean(0))

file_embs_norm = normalize(file_embs, norm='l2')
emb_win_norm   = normalize(emb_win,   norm='l2')

print(f"Files={n_files}, windows={n_win_total}, species={n_species}", flush=True)

# ── Geo metadata ────────────────────────────────────────────────────────────
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

# Build nologit combined space (pca24 + geo + day = 39 dims)
pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X_pca24 = pca24.transform(file_embs_norm).astype(np.float32)
pca24_std = X_pca24.std(0) + 1e-6
X_pca24_s = X_pca24 / pca24_std
geo_feats = np.concatenate([site_oh, hour_enc, month_enc, day_enc], axis=1).astype(np.float32)
X_combined = np.concatenate([X_pca24_s, geo_feats], axis=1).astype(np.float32)
X_nl = (X_combined / np.linalg.norm(X_combined, axis=1, keepdims=True)).astype(np.float32)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

BEST = 0.905463
BEST_NOLOGIT = 0.875789
results = {}

# ══════════════════════════════════════════════════════════════════════════════
# A) hybrid_logit_nologit — LOO 同時計算 logit-KNN 和 nologit-Attn-KNN，sweep blend
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("A) Hybrid Logit + Nologit KNN (同一 LOO loop)")
print("="*70, flush=True)

ALPHAS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 1.00]

preds_logit  = np.zeros((n_files, n_species), dtype=np.float32)
preds_nologit = np.zeros((n_files, n_species), dtype=np.float32)

for i in range(n_files):
    tr_idx = np.array([j for j in range(n_files) if j != i])

    # --- logit-based: k=3 cosine KNN, predict 0.76×prob_max + 0.24×prob_mean ---
    sims_l = (file_embs_norm[[i]] @ file_embs_norm[tr_idx].T).ravel()
    top3   = np.argsort(-sims_l)[:3]
    w3     = sims_l[top3].clip(0); w3 /= (w3.sum() + 1e-8)
    feat_l = 0.76 * file_prob_max[tr_idx[top3]] + 0.24 * file_prob_mean[tr_idx[top3]]
    preds_logit[i] = (w3[:, None] * feat_l).sum(0)

    # --- nologit: Attn-KNN pca24+day, k=10, T=0.2 ---
    sims_n = (X_nl[[i]] @ X_nl[tr_idx].T).ravel()
    top10  = np.argsort(-sims_n)[:10]
    logit_n = sims_n[top10] / 0.2; logit_n -= logit_n.max()
    w10    = np.exp(logit_n); w10 /= w10.sum()
    preds_nologit[i] = (w10[:, None] * file_labels[tr_idx[top10]]).sum(0)

# LOO-AUC for each component
auc_l = macro_auc(file_labels, preds_logit)
auc_n = macro_auc(file_labels, preds_nologit)
print(f"  Logit-KNN k=3 (0.76pmx+0.24pmn): {auc_l:.4f} (ref={BEST:.4f})")
print(f"  Nologit Attn-KNN pca24+day k=10: {auc_n:.4f} (ref={BEST_NOLOGIT:.4f})", flush=True)

best_so_far = BEST
for alpha in ALPHAS:
    blend = alpha * preds_logit + (1 - alpha) * preds_nologit
    a = macro_auc(file_labels, blend)
    nm = f'hybrid_logit_nologit_a{alpha:.2f}'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  α={alpha:.2f}: {a:.4f}  (Δ vs best={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# Also try nologit only with current best params (verify)
results['logit_knn_k3_verify'] = auc_l
results['nologit_attn_pca24_day_k10_verify'] = auc_n

# ══════════════════════════════════════════════════════════════════════════════
# B) window_knn_agg — 視窗層級 KNN，純 embedding（no logit）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("B) Window-level KNN with file aggregation (pure embedding)")
print("="*70, flush=True)

def window_knn_loo(k=5, agg='max'):
    """
    LOO-CV at window level:
    - For each test file i, training windows = all windows from files != i
    - For each test window, find top-k training windows by cosine sim
    - Aggregate per-window predictions to file-level using agg='max'/'mean'
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    # Pre-compute which window belongs to which file
    win_file_id = np.zeros(n_win_total, dtype=np.int32)
    for fi in range(n_files):
        win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

    for i in range(n_files):
        # Test windows for file i
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]     # (nw_i, 1536)
        nw_te = te_e - te_s

        # Training windows (exclude file i)
        tr_mask_w = win_file_id != i
        X_tr  = emb_win_norm[tr_mask_w]    # (N_tr_win, 1536)
        # Build label matrix for training windows (use max-label of the file)
        tr_win_idx = np.where(tr_mask_w)[0]
        Y_tr = file_labels[win_file_id[tr_win_idx]]  # (N_tr_win, 234)

        # Cosine similarity: (nw_te, N_tr_win)
        sims = X_te @ X_tr.T              # (nw_i, N_tr_win)
        top  = np.argsort(-sims, axis=1)[:, :k]  # (nw_i, k)

        # Per-window prediction
        win_preds = np.zeros((nw_te, n_species), dtype=np.float32)
        for wi in range(nw_te):
            top_sims = sims[wi, top[wi]]
            w = top_sims.clip(0); w_sum = w.sum()
            if w_sum > 1e-8:
                w /= w_sum
            else:
                w = np.ones(k) / k
            win_preds[wi] = (w[:, None] * Y_tr[top[wi]]).sum(0)

        # Aggregate across test windows
        if agg == 'max':
            preds[i] = win_preds.max(0)
        elif agg == 'mean':
            preds[i] = win_preds.mean(0)
        elif agg == 'attn_max':
            # Use window-level max over top-5 cosine sim as aggregation weight
            win_conf = sims.max(1)  # (nw_te,) — confidence of each window
            w_agg = np.exp(win_conf / 0.3); w_agg /= w_agg.sum()
            preds[i] = (w_agg[:, None] * win_preds).sum(0)

        if (i + 1) % 20 == 0:
            print(f"    fold {i+1}/{n_files} done", flush=True)

    return preds

print("  Running window_knn k=5 agg=max ...", flush=True)
p = window_knn_loo(k=5, agg='max')
a = macro_auc(file_labels, p)
print(f"  window_knn k=5 max: {a:.4f}  (Δ vs nologit best={a-BEST_NOLOGIT:+.4f})", flush=True)
results['window_knn_k5_max'] = a

print("  Running window_knn k=5 agg=mean ...", flush=True)
p = window_knn_loo(k=5, agg='mean')
a = macro_auc(file_labels, p)
print(f"  window_knn k=5 mean: {a:.4f}  (Δ={a-BEST_NOLOGIT:+.4f})", flush=True)
results['window_knn_k5_mean'] = a

print("  Running window_knn k=10 agg=max ...", flush=True)
p = window_knn_loo(k=10, agg='max')
a = macro_auc(file_labels, p)
print(f"  window_knn k=10 max: {a:.4f}  (Δ={a-BEST_NOLOGIT:+.4f})", flush=True)
results['window_knn_k10_max'] = a

print("  Running window_knn k=10 agg=attn_max ...", flush=True)
p = window_knn_loo(k=10, agg='attn_max')
a = macro_auc(file_labels, p)
print(f"  window_knn k=10 attn_max: {a:.4f}  (Δ={a-BEST_NOLOGIT:+.4f})", flush=True)
results['window_knn_k10_attn_max'] = a

print("  Running window_knn k=3 agg=max ...", flush=True)
p = window_knn_loo(k=3, agg='max')
a = macro_auc(file_labels, p)
print(f"  window_knn k=3 max: {a:.4f}  (Δ={a-BEST_NOLOGIT:+.4f})", flush=True)
results['window_knn_k3_max'] = a

# ── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY (top 12)")
print("="*70)
for nm, auc in sorted(results.items(), key=lambda x: -x[1])[:12]:
    marker = " ← NEW BEST" if auc > BEST else (" ← NOLOGIT BEST" if auc > BEST_NOLOGIT else "")
    print(f"  {nm:<50s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\nOverall best: {global_best_name} = {global_best_auc:.4f}", flush=True)

# ── Update results.json ────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best = data['best']['loo_auc']
cur_best_nl = data['best_nologit']['loo_auc']

new_best = False
new_best_nl = False

for nm, auc in results.items():
    note = 'hybrid_logit_nologit' if 'hybrid' in nm else 'window_knn_nologit'
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': note})
    if auc > cur_best:
        cur_best = auc
        data['best'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'NEW BEST'}
        new_best = True
    if 'nologit' in note and auc > cur_best_nl:
        cur_best_nl = auc
        data['best_nologit'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'no_logit NEW BEST'}
        new_best_nl = True

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\nbest 更新為: {data['best']['method']} = {data['best']['loo_auc']:.4f}")
if new_best_nl:
    print(f"best_nologit 更新為: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")
print("done", flush=True)
