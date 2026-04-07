"""
No-logit embed prior improvement search.
Best so far: attn_knn_combined_k12_T02 = 0.8606
Combined space: [pca32 + site_oh9 + hour2 + month2] = 45 dims

New experiments:
  A) PCA dims sweep: 16, 24, 48, 64, 96
  B) Geo weight scaling: 1.0, 1.5, 2.0, 3.0
  C) K sweep: 6, 8, 10, 15, 20, 30
  D) Add day-of-year feature (sin/cos)
  E) Temperature fine search: 0.15, 0.20, 0.25, 0.30
"""
import numpy as np, json, pickle, os, re
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_embs   = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels = np.zeros((n_files, n_species), dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    file_embs[fi]   = emb_win[idx:idx+nw].mean(0)
    file_labels[fi] = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
print(f"Files={n_files}, species={n_species}\n", flush=True)

# ── Parse geo metadata ────────────────────────────────────────────────
SITES = ['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx = {s: i for i, s in enumerate(SITES)}
file_sites  = np.zeros(n_files, dtype=np.int32)
file_hours  = np.zeros(n_files, dtype=np.float32)
file_months = np.zeros(n_files, dtype=np.float32)
file_days   = np.zeros(n_files, dtype=np.float32)  # day of year 1-365

for fi, fname in enumerate(file_list):
    m = re.match(r'BC2026_Train_\d+_(S\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})', str(fname))
    if m:
        site, yr, mo, dy, hr, mn = m.groups()
        file_sites[fi]  = site2idx.get(site, 0)
        file_hours[fi]  = int(hr)
        file_months[fi] = int(mo)
        # Day of year (approximate)
        days_per_month = [0,31,28,31,30,31,30,31,31,30,31,30,31]
        file_days[fi] = sum(days_per_month[:int(mo)]) + int(dy)

site_oh   = np.eye(len(SITES), dtype=np.float32)[file_sites]
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24), np.cos(2*np.pi*file_hours/24)], axis=1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12), np.cos(2*np.pi*(file_months-1)/12)], axis=1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365), np.cos(2*np.pi*(file_days-1)/365)], axis=1).astype(np.float32)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

def attn_knn_loo(X_combined_n, k=12, T=0.2):
    """Attention-weighted KNN LOO-CV on normalized combined feature space."""
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, bool); mask[i] = False
        tr_idx = np.where(mask)[0]
        sims   = (X_combined_n[[i]] @ X_combined_n[tr_idx].T).ravel()
        top    = np.argsort(-sims)[:k]
        logits = sims[top] / T
        logits -= logits.max()
        w = np.exp(logits); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr_idx[top]]).sum(0)
    return preds

def build_combined(pca_dims=32, geo_w=1.0, use_day=False):
    """Build normalized combined feature space."""
    pca    = PCA(n_components=pca_dims, random_state=42).fit(file_embs_norm)
    X_pca  = pca.transform(file_embs_norm).astype(np.float32)
    X_pca /= (X_pca.std(0) + 1e-6)

    geo_parts = [site_oh, hour_enc, month_enc]
    if use_day: geo_parts.append(day_enc)
    geo = np.concatenate(geo_parts, axis=1).astype(np.float32) * geo_w

    X = np.concatenate([X_pca, geo], axis=1).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

BEST = 0.8606
results = {}

# ── A) PCA dims sweep ─────────────────────────────────────────────────
print("="*60)
print("A) PCA dims sweep (k=12, T=0.2, geo_w=1.0)")
print("="*60, flush=True)

for pca_d in [16, 24, 48, 64, 96]:
    X = build_combined(pca_dims=pca_d, geo_w=1.0)
    p = attn_knn_loo(X, k=12, T=0.2)
    a = macro_auc(file_labels, p)
    nm = f'attn_k12_T02_pca{pca_d}'
    marker = " ← NEW BEST" if a > BEST else ""
    print(f"  pca{pca_d}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── B) Geo weight sweep ───────────────────────────────────────────────
print("\n" + "="*60)
print("B) Geo weight sweep (pca32, k=12, T=0.2)")
print("="*60, flush=True)

for gw in [0.5, 1.5, 2.0, 3.0, 5.0]:
    X = build_combined(pca_dims=32, geo_w=gw)
    p = attn_knn_loo(X, k=12, T=0.2)
    a = macro_auc(file_labels, p)
    nm = f'attn_k12_T02_pca32_gw{gw:.1f}'
    marker = " ← NEW BEST" if a > BEST else ""
    print(f"  geo_w={gw}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── C) K sweep (best pca so far) ─────────────────────────────────────
print("\n" + "="*60)
print("C) K sweep (pca32, T=0.2, geo_w=1.0)")
print("="*60, flush=True)

X32 = build_combined(pca_dims=32, geo_w=1.0)
for k in [6, 8, 10, 15, 20, 30]:
    p = attn_knn_loo(X32, k=k, T=0.2)
    a = macro_auc(file_labels, p)
    nm = f'attn_k{k}_T02_pca32'
    marker = " ← NEW BEST" if a > BEST else ""
    print(f"  k={k}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── D) Add day-of-year feature ────────────────────────────────────────
print("\n" + "="*60)
print("D) Add day-of-year (sin/cos) feature")
print("="*60, flush=True)

X_day = build_combined(pca_dims=32, geo_w=1.0, use_day=True)
for k in [10, 12, 15]:
    p = attn_knn_loo(X_day, k=k, T=0.2)
    a = macro_auc(file_labels, p)
    nm = f'attn_k{k}_T02_pca32_day'
    marker = " ← NEW BEST" if a > BEST else ""
    print(f"  k={k}+day: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── E) Temperature fine search ────────────────────────────────────────
print("\n" + "="*60)
print("E) Temperature fine search (pca32, k=12, geo_w=1.0)")
print("="*60, flush=True)

for T in [0.10, 0.15, 0.25, 0.30, 0.50, 1.0]:
    p = attn_knn_loo(X32, k=12, T=T)
    a = macro_auc(file_labels, p)
    nm = f'attn_k12_T{T:.2f}_pca32'
    marker = " ← NEW BEST" if a > BEST else ""
    print(f"  T={T}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── F) Best combos ────────────────────────────────────────────────────
print("\n" + "="*60)
print("F) Best combos from A-E")
print("="*60, flush=True)

# Find best pca dims and geo_w from above
best_pca = max((results.get(f'attn_k12_T02_pca{d}', 0), d) for d in [16,24,32,48,64,96])[1]
best_gw  = max((results.get(f'attn_k12_T02_pca32_gw{w:.1f}', 0), w) for w in [0.5,1.0,1.5,2.0,3.0,5.0])[1]
best_k   = max((results.get(f'attn_k{k}_T02_pca32', results.get('knn_combined_geo_k12', 0)), k) 
               for k in [6,8,10,12,15,20,30])[1]
print(f"  Best pca={best_pca}, best geo_w={best_gw}, best k={best_k}", flush=True)

for (pca_d, gw, k) in [
    (best_pca, 1.0, 12),
    (32, best_gw, 12),
    (32, 1.0, best_k),
    (best_pca, best_gw, 12),
    (best_pca, 1.0, best_k),
]:
    X = build_combined(pca_dims=pca_d, geo_w=gw)
    p = attn_knn_loo(X, k=k, T=0.2)
    a = macro_auc(file_labels, p)
    nm = f'combo_pca{pca_d}_gw{gw:.1f}_k{k}'
    marker = " ← NEW BEST" if a > BEST else ""
    print(f"  {nm}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (top 10)")
print("="*60)
for name, auc in sorted(results.items(), key=lambda x: -x[1])[:10]:
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {name:<45s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

best_auc = data['best_nologit']['loo_auc']
new_best = None
for name, auc in results.items():
    data['experiments'].append({'method': name, 'loo_auc': round(auc, 4),
                                 'features': 'combined_geo', 'note': 'no_logit'})
    if auc > best_auc:
        best_auc = auc; new_best = name
        data['best_nologit'] = {'method': name, 'loo_auc': round(auc, 4), 'note': 'no_logit NEW BEST'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

if new_best:
    print(f"\n*** NEW BEST (no-logit): {new_best} AUC={best_auc:.4f} ***")
else:
    print(f"\n未超越 best_nologit ({BEST:.4f})，attn_knn_combined_k12_T02 仍是最佳。")
print("done", flush=True)
