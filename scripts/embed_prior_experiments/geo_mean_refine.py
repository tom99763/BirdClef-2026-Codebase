"""
精調：幾何平均 blend (self_logit × nologit) 的 power weight
+ 多個 nologit 變體的幾何平均
+ knn3_labels 的三元組合
"""
import numpy as np, json, pickle, re, os
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

file_embs   = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels = np.zeros((n_files, n_species), dtype=np.float32)
file_prob_max  = np.zeros((n_files, n_species), dtype=np.float32)
file_prob_mean = np.zeros((n_files, n_species), dtype=np.float32)
file_prob_p75  = np.zeros((n_files, n_species), dtype=np.float32)

for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]   = emb_win[s:e].mean(0)
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    lb = logits_win[s:e]
    file_prob_max[fi]  = _sigmoid(lb.max(0))
    file_prob_mean[fi] = _sigmoid(lb.mean(0))
    file_prob_p75[fi]  = _sigmoid(np.percentile(lb, 75, axis=0))

file_embs_norm = normalize(file_embs, norm='l2')

# Nologit pca24+day space
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

def build_nl_space(pca_dims=24, use_day=True):
    pca = PCA(n_components=pca_dims, random_state=42).fit(file_embs_norm)
    X_p = pca.transform(file_embs_norm).astype(np.float32)
    X_p /= (X_p.std(0) + 1e-6)
    geo = np.concatenate([site_oh, hour_enc, month_enc] + ([day_enc] if use_day else []), axis=1).astype(np.float32)
    X = np.concatenate([X_p, geo], axis=1).astype(np.float32)
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)

def attn_knn_loo(X, k=10, T=0.2):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        tr_idx = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr_idx].T).ravel()
        top = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr_idx[top]]).sum(0)
    return preds

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS = 1e-7
BEST = 0.905463
BEST_NL = 0.875789
results = {}

print(f"Files={n_files}, species={n_species}", flush=True)
print(f"BEST={BEST:.4f}  BEST_NL={BEST_NL:.4f}", flush=True)

# ── 預計算各個基底預測 ─────────────────────────────────────────────────────
print("\nBuilding base predictions...", flush=True)

# Self-logit components
y_pmx  = file_prob_max
y_pmn  = file_prob_mean
y_pp75 = file_prob_p75
y_sl76 = 0.76 * y_pmx + 0.24 * y_pmn   # best self-logit blend

# Nologit variants
X_nl_pca24_day = build_nl_space(pca_dims=24, use_day=True)
X_nl_pca24     = build_nl_space(pca_dims=24, use_day=False)
X_nl_pca32_day = build_nl_space(pca_dims=32, use_day=True)

y_nl24d = attn_knn_loo(X_nl_pca24_day, k=10, T=0.2)
y_nl24  = attn_knn_loo(X_nl_pca24,     k=12, T=0.2)
y_nl32d = attn_knn_loo(X_nl_pca32_day, k=10, T=0.2)

auc_nl24d = macro_auc(file_labels, y_nl24d)
auc_nl24  = macro_auc(file_labels, y_nl24)
auc_nl32d = macro_auc(file_labels, y_nl32d)
auc_sl76  = macro_auc(file_labels, y_sl76)

print(f"  y_sl76 (0.76pmx+0.24pmn):    {auc_sl76:.4f}")
print(f"  y_nl24d (pca24+day k10 T02): {auc_nl24d:.4f}")
print(f"  y_nl24  (pca24 k12 T02):     {auc_nl24:.4f}")
print(f"  y_nl32d (pca32+day k10 T02): {auc_nl32d:.4f}", flush=True)

results['self_logit_sl76'] = auc_sl76
results['nologit_pca24_day_k10'] = auc_nl24d
results['nologit_pca24_k12'] = auc_nl24
results['nologit_pca32_day_k10'] = auc_nl32d

best_so_far = BEST

# ── A) 精調幾何平均 power weight (y_sl76 × y_nl24d) ────────────────────────
print("\n" + "="*60)
print("A) 精調 geo_mean pw: y_sl76^(1-pw) × y_nl24d^pw")
print("="*60, flush=True)

for pw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.60]:
    blend = (y_sl76.clip(EPS, 1-EPS) ** (1-pw)) * (y_nl24d.clip(EPS, 1-EPS) ** pw)
    a = macro_auc(file_labels, blend)
    nm = f'geo_sl76_nl24d_pw{pw:.2f}'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  pw={pw:.2f}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── B) 幾何平均：各種 y_logit 組合 × y_nl24d ────────────────────────────
print("\n" + "="*60)
print("B) 各種 logit 組合的幾何平均")
print("="*60, flush=True)

logit_combos = {
    'pmx':        y_pmx,
    'pmn':        y_pmn,
    'pp75':       y_pp75,
    'pmx90_pmn10': 0.90*y_pmx + 0.10*y_pmn,
    'pmx80_pmn20': 0.80*y_pmx + 0.20*y_pmn,
    'pmx76_pmn24': y_sl76,
    'pmx70_pmn30': 0.70*y_pmx + 0.30*y_pmn,
    'pmx_pp75':   0.60*y_pmx + 0.40*y_pp75,
    'sqrt_pmx':   np.sqrt(y_pmx.clip(EPS, 1-EPS)),  # sharper version
}
pw_best = 0.30  # use best pw from A
for nm_logit, y_logit in logit_combos.items():
    blend = (y_logit.clip(EPS, 1-EPS) ** (1-pw_best)) * (y_nl24d.clip(EPS, 1-EPS) ** pw_best)
    a = macro_auc(file_labels, blend)
    nm = f'geo_{nm_logit}_pw030'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  {nm_logit}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── C) 幾何平均：y_sl76 × best nologit 變體 ────────────────────────────
print("\n" + "="*60)
print("C) y_sl76 × 各種 nologit 變體 (pw=0.30)")
print("="*60, flush=True)

nologit_combos = {
    'nl24d': y_nl24d,
    'nl24':  y_nl24,
    'nl32d': y_nl32d,
    'avg_nl24d_nl24':  0.5*y_nl24d + 0.5*y_nl24,
    'avg_nl24d_nl32d': 0.5*y_nl24d + 0.5*y_nl32d,
}
for nm_nl, y_nl in nologit_combos.items():
    blend = (y_sl76.clip(EPS, 1-EPS) ** (1-pw_best)) * (y_nl.clip(EPS, 1-EPS) ** pw_best)
    a = macro_auc(file_labels, blend)
    nm = f'geo_sl76_{nm_nl}_pw030'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  {nm_nl}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── D) 三元幾何平均：y_logit × y_nl × y_knn ──────────────────────────────
print("\n" + "="*60)
print("D) 三元幾何平均 + knn3_labels")
print("="*60, flush=True)

y_knn3 = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    tr_idx = np.array([j for j in range(n_files) if j != i])
    sims = (file_embs_norm[[i]] @ file_embs_norm[tr_idx].T).ravel()
    top3 = np.argsort(-sims)[:3]
    w3   = sims[top3].clip(0); w3 /= (w3.sum() + 1e-8)
    y_knn3[i] = (w3[:, None] * file_labels[tr_idx[top3]]).sum(0)

auc_knn3 = macro_auc(file_labels, y_knn3)
print(f"  knn3_labels AUC: {auc_knn3:.4f}", flush=True)
results['knn3_labels'] = auc_knn3

# Three-way geo mean: y_sl76^a × y_nl24d^b × y_knn3^c  where a+b+c=1
for (pa, pb, pc) in [(0.65, 0.25, 0.10), (0.65, 0.28, 0.07), (0.65, 0.30, 0.05),
                      (0.68, 0.25, 0.07), (0.68, 0.27, 0.05), (0.70, 0.23, 0.07),
                      (0.70, 0.25, 0.05), (0.72, 0.23, 0.05), (0.72, 0.25, 0.03)]:
    blend = (y_sl76.clip(EPS,1-EPS)**pa) * (y_nl24d.clip(EPS,1-EPS)**pb) * (y_knn3.clip(EPS,1-EPS)**pc)
    a = macro_auc(file_labels, blend)
    nm = f'3geo_sl{pa:.2f}_nl{pb:.2f}_k3{pc:.2f}'
    marker = " ← NEW BEST" if a > best_so_far else ""
    if a > best_so_far: best_so_far = a
    print(f"  sl={pa:.2f} nl={pb:.2f} k3={pc:.2f}: {a:.4f}  (Δ={a-BEST:+.4f}){marker}", flush=True)
    results[nm] = a

# ── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (top 15)")
print("="*60)
for nm, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {nm:<55s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\nOverall best: {global_best_name} = {global_best_auc:.4f}", flush=True)

# ── Update results.json ────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best = data['best']['loo_auc']
for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': 'geo_mean_refine'})
    if auc > cur_best:
        cur_best = auc
        data['best'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'NEW BEST geo_mean'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

# ── Save best pkl if beats BEST_NL ────────────────────────────────────────
if global_best_auc > BEST_NL:
    import shutil
    # Rebuild geo_mean best predictions (for inference we'd need to store more)
    best_pw = None
    best_pw_auc = 0
    for pw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.60]:
        nm = f'geo_sl76_nl24d_pw{pw:.2f}'
        if results.get(nm, 0) > best_pw_auc:
            best_pw_auc = results[nm]; best_pw = pw
    if best_pw and best_pw_auc > BEST_NL:
        print(f"\nSaving best geo_mean pkl: pw={best_pw} AUC={best_pw_auc:.4f}", flush=True)
        # Note: geo_mean combines self_logit + nologit — for inference
        # we store both components' parameters
        pkl_data = {
            'method': f'geo_mean_sl76_nl24d_pw{best_pw:.2f}',
            'loo_auc': best_pw_auc,
            'type': 'geo_mean',
            'pw_nologit': best_pw,
            # nologit component
            'pca_dims': 24,
            'pca_mean': PCA(n_components=24, random_state=42).fit(file_embs_norm).mean_,
            'pca_components': PCA(n_components=24, random_state=42).fit(file_embs_norm).components_,
            'pca_std': (PCA(n_components=24, random_state=42).fit(file_embs_norm).transform(file_embs_norm).std(0) + 1e-6),
            'use_day': True,
            'k': 10,
            'T': 0.2,
            'SITES': SITES,
            'site2idx': site2idx,
            'X_combined_n': X_nl_pca24_day,
            'file_labels': file_labels,
            'file_list': file_list,
            # self-logit component
            'logit_blend': (0.76, 0.24),  # (prob_max_weight, prob_mean_weight)
        }
        with open("outputs/embed_prior_geomean.pkl", "wb") as f:
            pickle.dump(pkl_data, f)
        shutil.copy("outputs/embed_prior_geomean.pkl",
                    "birdclef-2026/notebook resource/current_subs/weights/embed_prior_geomean.pkl")
        print(f"Saved: outputs/embed_prior_geomean.pkl")

print(f"\nbest 更新為: {data['best']['method']} = {data['best']['loo_auc']:.4f}")
print("done", flush=True)
