"""
修正版：hybrid_logit_nologit

A) 正確複現 ens_pmx0.76_pmn0.24_k3 = 0.9055:
   - Test file 自身的 prob_max×0.76 + prob_mean×0.24（不是鄰居的）
   - k=3 KNN 鄰居 labels 作為第二個特徵（per_species_alpha 形式）
   - 確認能重現 ~0.9055

B) 正確 Hybrid: α × y_self_logit + (1-α) × y_nologit
   - y_self_logit = test file 自己的 0.76×prob_max + 0.24×prob_mean
   - y_nologit = Attn-KNN pca24+day k=10 T=0.2

C) Hybrid with KNN calibration:
   - Use k=3 KNN neighbors as calibration signal for nologit
   - knn_label_preds = weighted file_labels of k=3 nearest (by embedding)
   - Final = α × (0.76 prob_max + 0.24 prob_mean) + β × knn_label + γ × nologit
"""
import numpy as np, json, re, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
logits_win = raw['logits'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])])
file_end   = np.cumsum(n_windows)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species), dtype=np.float32)
file_prob_max  = np.zeros((n_files, n_species), dtype=np.float32)
file_prob_mean = np.zeros((n_files, n_species), dtype=np.float32)

for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]   = emb_win[s:e].mean(0)
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    lb = logits_win[s:e]
    file_prob_max[fi]  = _sigmoid(lb.max(0))
    file_prob_mean[fi] = _sigmoid(lb.mean(0))

file_embs_norm = normalize(file_embs, norm='l2')
print(f"Files={n_files}, species={n_species}", flush=True)

# Build nologit combined space (pca24 + geo + day = 39 dims)
SITES = ['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx = {s: i for i, s in enumerate(SITES)}
file_sites = np.zeros(n_files, dtype=np.int32)
file_hours = np.zeros(n_files, dtype=np.float32)
file_months= np.zeros(n_files, dtype=np.float32)
file_days  = np.zeros(n_files, dtype=np.float32)
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

pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
std24 = X24.std(0) + 1e-6
X24s  = X24 / std24
geo   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], axis=1).astype(np.float32)
X_combined = np.concatenate([X24s, geo], axis=1).astype(np.float32)
X_nl = (X_combined / np.linalg.norm(X_combined, axis=1, keepdims=True)).astype(np.float32)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

BEST       = 0.905463
BEST_NL    = 0.875789
results    = {}

# ── 預計算 LOO 預測 ─────────────────────────────────────────────────────────
print("\nPre-computing LOO predictions for all methods...", flush=True)

# 1) Self-logit score (test file 自己的 Perch logit)
y_self_pmx76_pmn24 = 0.76 * file_prob_max + 0.24 * file_prob_mean  # (66, 234)

# 2) Nologit Attn-KNN pca24+day k=10 T=0.2
y_nologit = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    tr_idx = np.array([j for j in range(n_files) if j != i])
    sims = (X_nl[[i]] @ X_nl[tr_idx].T).ravel()
    top10 = np.argsort(-sims)[:10]
    logit_n = sims[top10] / 0.2; logit_n -= logit_n.max()
    w10 = np.exp(logit_n); w10 /= w10.sum()
    y_nologit[i] = (w10[:, None] * file_labels[tr_idx[top10]]).sum(0)

# 3) KNN label prediction (k=3 cosine, using true labels of neighbors)
y_knn3_labels = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    tr_idx = np.array([j for j in range(n_files) if j != i])
    sims = (file_embs_norm[[i]] @ file_embs_norm[tr_idx].T).ravel()
    top3 = np.argsort(-sims)[:3]
    w3   = sims[top3].clip(0); w3 /= (w3.sum() + 1e-8)
    y_knn3_labels[i] = (w3[:, None] * file_labels[tr_idx[top3]]).sum(0)

# 4) KNN label prediction k=5
y_knn5_labels = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    tr_idx = np.array([j for j in range(n_files) if j != i])
    sims = (file_embs_norm[[i]] @ file_embs_norm[tr_idx].T).ravel()
    top5 = np.argsort(-sims)[:5]
    w5   = sims[top5].clip(0); w5 /= (w5.sum() + 1e-8)
    y_knn5_labels[i] = (w5[:, None] * file_labels[tr_idx[top5]]).sum(0)

# 個別 AUC
auc_self = macro_auc(file_labels, y_self_pmx76_pmn24)
auc_nl   = macro_auc(file_labels, y_nologit)
auc_k3   = macro_auc(file_labels, y_knn3_labels)
auc_k5   = macro_auc(file_labels, y_knn5_labels)
print(f"  self_logit 0.76pmx+0.24pmn: {auc_self:.4f} (ref={BEST:.4f})")
print(f"  nologit attn pca24+day k10:  {auc_nl:.4f} (ref={BEST_NL:.4f})")
print(f"  knn3_labels:                 {auc_k3:.4f}")
print(f"  knn5_labels:                 {auc_k5:.4f}", flush=True)

results['self_logit_pmx76_pmn24']   = auc_self
results['nologit_attn_pca24_day_k10'] = auc_nl

# ══════════════════════════════════════════════════════════════════════════════
# A) 正確 Hybrid: α × self_logit + (1-α) × nologit
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A) Hybrid: α×self_logit + (1-α)×nologit")
print("="*60, flush=True)

best_so_far = BEST
for alpha in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.99]:
    blend = alpha * y_self_pmx76_pmn24 + (1-alpha) * y_nologit
    a = macro_auc(file_labels, blend)
    nm = f'hybrid_self_logit_nl_a{alpha:.2f}'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  α={alpha:.2f}: {a:.4f}  (Δ vs best={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ══════════════════════════════════════════════════════════════════════════════
# B) 3-way: self_logit + nologit + knn3_labels
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) 3-way: α×self_logit + β×nologit + γ×knn3_labels")
print("="*60, flush=True)

combos = [
    (0.80, 0.10, 0.10),
    (0.80, 0.15, 0.05),
    (0.82, 0.10, 0.08),
    (0.85, 0.10, 0.05),
    (0.85, 0.08, 0.07),
    (0.88, 0.07, 0.05),
    (0.90, 0.05, 0.05),
    (0.90, 0.08, 0.02),
    (0.92, 0.05, 0.03),
    (0.92, 0.06, 0.02),
    (0.94, 0.04, 0.02),
    (0.95, 0.04, 0.01),
]
for (a_sl, a_nl, a_k3) in combos:
    blend = a_sl * y_self_pmx76_pmn24 + a_nl * y_nologit + a_k3 * y_knn3_labels
    a = macro_auc(file_labels, blend)
    nm = f'3way_sl{a_sl:.2f}_nl{a_nl:.2f}_k3{a_k3:.2f}'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  sl={a_sl:.2f} nl={a_nl:.2f} k3={a_k3:.2f}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ══════════════════════════════════════════════════════════════════════════════
# C) 使用 prob_max 自身 + nologit 的 per-species 自適應混合
#    （某些 species nologit 比 logit 更準）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) Per-species adaptive: if nologit[s] > threshold → use nologit more")
print("="*60, flush=True)

# 計算每個 species 的 nologit vs self_logit 相對優勢
# 用 LOO 比較哪些 species nologit 更準
# 簡單做法：threshold-based adaptive blend

for thr in [0.3, 0.4, 0.5]:
    # 若 nologit score > thr（說明有明確 geo 信號），用更多 nologit
    blend = y_self_pmx76_pmn24.copy()
    high_mask = y_nologit > thr
    blend[high_mask] = 0.80 * y_self_pmx76_pmn24[high_mask] + 0.20 * y_nologit[high_mask]
    a = macro_auc(file_labels, blend)
    nm = f'adaptive_nl_thr{thr:.1f}'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  adaptive thr={thr:.1f}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# D) Geometric mean of self_logit and nologit
print("\n" + "="*60)
print("D) Geometric mean blend")
print("="*60, flush=True)

for pw in [0.1, 0.2, 0.3]:
    # geo_blend = self_logit^(1-pw) × nologit^pw (in probability space)
    blend = (y_self_pmx76_pmn24.clip(1e-7, 1-1e-7) ** (1-pw)) * (y_nologit.clip(1e-7, 1-1e-7) ** pw)
    a = macro_auc(file_labels, blend)
    nm = f'geo_mean_pw{pw:.1f}'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  geo_mean pw={pw:.1f}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (top 15)")
print("="*60)
for nm, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {nm:<50s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\nOverall best: {global_best_name} = {global_best_auc:.4f}", flush=True)

# ── Update results.json ────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best = data['best']['loo_auc']
for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6),
                                  'note': 'hybrid_v2'})
    if auc > cur_best:
        cur_best = auc
        data['best'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'NEW BEST hybrid_v2'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\nbest 更新為: {data['best']['method']} = {data['best']['loo_auc']:.4f}")
print("done", flush=True)
