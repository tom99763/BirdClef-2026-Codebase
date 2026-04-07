"""
Continuation of nologit_search.py after pca crash.
Known: pca24=0.8712 is new best.
Continue from B) onwards, plus add pca24-based combos.
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
print(f"Files={n_files}, species={n_species}", flush=True)

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
    pca_dims = min(pca_dims, n_files - 1)
    pca    = PCA(n_components=pca_dims, random_state=42).fit(file_embs_norm)
    X_pca  = pca.transform(file_embs_norm).astype(np.float32)
    X_pca /= (X_pca.std(0) + 1e-6)
    geo_parts = [site_oh, hour_enc, month_enc]
    if use_day: geo_parts.append(day_enc)
    geo = np.concatenate(geo_parts, axis=1).astype(np.float32) * geo_w
    X = np.concatenate([X_pca, geo], axis=1).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)

# Known from part A:
BEST = 0.8606
results = {
    'attn_k12_T02_pca16': 0.8534,
    'attn_k12_T02_pca24': 0.8712,
    'attn_k12_T02_pca48': 0.8207,
    'attn_k12_T02_pca64': 0.6927,
    'attn_k12_T02_pca32': 0.8606,  # baseline
}

# ── B) Geo weight sweep (pca24, k=12, T=0.2) ─────────────────────────
print("\n" + "="*60)
print("B) Geo weight sweep (pca24, k=12, T=0.2)")
print("="*60, flush=True)

best_so_far = 0.8712
for gw in [0.5, 1.5, 2.0, 3.0, 5.0]:
    X = build_combined(pca_dims=24, geo_w=gw)
    p = attn_knn_loo(X, k=12, T=0.2)
    a = macro_auc(file_labels, p)
    nm = f'attn_k12_T02_pca24_gw{gw:.1f}'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  geo_w={gw}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# Also sweep geo_w with pca32 for comparison
print("\n  (also pca32 for comparison)")
for gw in [0.5, 1.5, 2.0, 3.0]:
    X = build_combined(pca_dims=32, geo_w=gw)
    p = attn_knn_loo(X, k=12, T=0.2)
    a = macro_auc(file_labels, p)
    nm = f'attn_k12_T02_pca32_gw{gw:.1f}'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  pca32 geo_w={gw}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── C) K sweep (pca24, T=0.2, geo_w=1.0) ────────────────────────────
print("\n" + "="*60)
print("C) K sweep (pca24, T=0.2, geo_w=1.0)")
print("="*60, flush=True)

X24 = build_combined(pca_dims=24, geo_w=1.0)
for k in [6, 8, 10, 15, 20, 30]:
    p = attn_knn_loo(X24, k=k, T=0.2)
    a = macro_auc(file_labels, p)
    nm = f'attn_k{k}_T02_pca24'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  k={k}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── D) Add day-of-year feature ────────────────────────────────────────
print("\n" + "="*60)
print("D) Add day-of-year (sin/cos) — pca24")
print("="*60, flush=True)

X_day = build_combined(pca_dims=24, geo_w=1.0, use_day=True)
for k in [10, 12, 15]:
    p = attn_knn_loo(X_day, k=k, T=0.2)
    a = macro_auc(file_labels, p)
    nm = f'attn_k{k}_T02_pca24_day'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  k={k}+day: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── E) Temperature fine search (pca24, k=12) ─────────────────────────
print("\n" + "="*60)
print("E) Temperature fine search (pca24, k=12, geo_w=1.0)")
print("="*60, flush=True)

for T in [0.10, 0.15, 0.25, 0.30, 0.50, 1.0]:
    p = attn_knn_loo(X24, k=12, T=T)
    a = macro_auc(file_labels, p)
    nm = f'attn_k12_T{T:.2f}_pca24'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  T={T}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── F) Best combos ────────────────────────────────────────────────────
print("\n" + "="*60)
print("F) Best combos")
print("="*60, flush=True)

# Find best geo_w for pca24
best_gw_pca24 = max(
    [(results.get(f'attn_k12_T02_pca24_gw{w:.1f}', 0), w) for w in [0.5, 1.5, 2.0, 3.0, 5.0]] +
    [(results['attn_k12_T02_pca24'], 1.0)]
)[1]
best_k_pca24 = max(
    [(results.get(f'attn_k{k}_T02_pca24', 0), k) for k in [6,8,10,12,15,20,30]]
)[1]
best_T_pca24 = max(
    [(results.get(f'attn_k12_T{T:.2f}_pca24', 0), T) for T in [0.10,0.15,0.20,0.25,0.30,0.50,1.0]] +
    [(results['attn_k12_T02_pca24'], 0.2)]
)[1]

print(f"  Best: pca24, geo_w={best_gw_pca24}, k={best_k_pca24}, T={best_T_pca24}", flush=True)

for (pca_d, gw, k, T) in [
    (24, best_gw_pca24, 12, 0.2),
    (24, 1.0, best_k_pca24, 0.2),
    (24, 1.0, 12, best_T_pca24),
    (24, best_gw_pca24, best_k_pca24, 0.2),
    (24, best_gw_pca24, best_k_pca24, best_T_pca24),
]:
    X = build_combined(pca_dims=pca_d, geo_w=gw)
    p = attn_knn_loo(X, k=k, T=T)
    a = macro_auc(file_labels, p)
    nm = f'combo_pca{pca_d}_gw{gw:.1f}_k{k}_T{T:.2f}'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  {nm}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (top 15)")
print("="*60)
for name, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {name:<50s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

# Save best pkl
global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\nOverall best: {global_best_name} = {global_best_auc:.4f}", flush=True)

if global_best_auc > BEST:
    # Rebuild best X and save predictions + params
    import re as _re
    m = _re.match(r'(?:attn|combo)_k(\d+)_T([\d.]+)_pca(\d+)(?:_gw([\d.]+))?(?:_day)?', global_best_name)
    if m:
        k_b  = int(m.group(1))
        T_b  = float(m.group(2))
        pca_b = int(m.group(3))
        gw_b = float(m.group(4)) if m.group(4) else 1.0
    else:
        # parse combo
        m2 = _re.search(r'pca(\d+)_gw([\d.]+)_k(\d+)_T([\d.]+)', global_best_name)
        if m2:
            pca_b = int(m2.group(1)); gw_b = float(m2.group(2))
            k_b   = int(m2.group(3)); T_b  = float(m2.group(4))
        else:
            pca_b, gw_b, k_b, T_b = 24, 1.0, 12, 0.2  # fallback

    X_best = build_combined(pca_dims=pca_b, geo_w=gw_b)
    p_best = attn_knn_loo(X_best, k=k_b, T=T_b)

    pkl_data = {
        'method': global_best_name,
        'loo_auc': global_best_auc,
        'pca_dims': pca_b,
        'geo_w': gw_b,
        'k': k_b,
        'T': T_b,
        'X_combined_n': X_best,
        'file_labels': file_labels,
        'file_list': file_list,
    }
    with open("outputs/embed_prior_attn.pkl", "wb") as f:
        pickle.dump(pkl_data, f)
    import shutil
    shutil.copy("outputs/embed_prior_attn.pkl",
                "birdclef-2026/notebook resource/current_subs/weights/embed_prior_attn.pkl")
    print(f"Saved best pkl: pca{pca_b} geo_w={gw_b} k={k_b} T={T_b} AUC={global_best_auc:.4f}", flush=True)

# Update results.json
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best = data.get('best_nologit', {}).get('loo_auc', BEST)
for name, auc in results.items():
    data['experiments'].append({'method': name, 'loo_auc': round(auc, 6),
                                 'features': 'combined_geo', 'note': 'no_logit'})
    if auc > cur_best:
        cur_best = auc
        data['best_nologit'] = {'method': name, 'loo_auc': round(auc, 6), 'note': 'no_logit NEW BEST'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\nbest_nologit updated to: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")
print("done", flush=True)
