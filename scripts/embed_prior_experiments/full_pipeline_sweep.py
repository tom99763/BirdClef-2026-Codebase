"""
Full Pipeline Sweep: Find methods that beat v7-geo-knn (0.9246)
Target: Full pipeline LOO-CV macro AUC > 0.9246

Sweep:
  A) λ sweep (0.10~0.50) on attn-KNN k10 T0.2
  B) k sweep (3~20) on attn-KNN T0.2 λ0.25
  C) T sweep (0.10~0.35) on attn-KNN k10 λ0.25
  D) VLOM weight sweep (ProtoSSM:SED from 30:70 to 70:30)
  E) SED agg: max vs mean vs (max+mean)/2
  F) PCA dims: 12, 18, 36, 48, 64
  G) Geo weight: 0.5, 1.0, 1.5, 2.0
  H) 3-way VLOM: ProtoSSM + SED + embed_prior
  I) Double space: PCA36 + geo, PCA48 + geo
  J) Top-k combo: joint λ+k+T best
"""
import numpy as np, pickle, re, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

file_embs      = np.zeros((n_files, emb_win.shape[1]), np.float32)
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_prob_mean = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_prob_mean[fi] = sigmoid(logits_win[s:e]).mean(0)

file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)

# ── Load SED ──────────────────────────────────────────────────────────────
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_row_ids = sed_npz['row_ids']
sed_probs_all = sed_npz['probs'].astype(np.float32)

sed_by_file = {}
for i, rid in enumerate(sed_row_ids):
    fname_base = '_'.join(str(rid).split('_')[:-1])
    if fname_base not in sed_by_file:
        sed_by_file[fname_base] = []
    sed_by_file[fname_base].append(i)

file_sed_max  = np.zeros((n_files, n_species), np.float32)
file_sed_mean = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fname_base = fname.replace('.ogg', '').replace('.flac', '')
    if fname_base in sed_by_file:
        idxs = sed_by_file[fname_base]
        win_probs = sed_probs_all[idxs]
        file_sed_max[fi]  = win_probs.max(0)
        file_sed_mean[fi] = win_probs.mean(0)

print(f"Files={n_files}, species={n_species}")

# ── Geo features ──────────────────────────────────────────────────────────
SITES = ['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx = {s: i for i, s in enumerate(SITES)}
file_sites  = np.zeros(n_files, np.int32)
file_hours  = np.zeros(n_files, np.float32)
file_months = np.zeros(n_files, np.float32)
file_days   = np.zeros(n_files, np.float32)
for fi, fname in enumerate(file_list):
    m = re.match(r'BC2026_Train_\d+_(S\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})', str(fname))
    if m:
        site, yr, mo, dy, hr, mn = m.groups()
        file_sites[fi]  = site2idx.get(site, 0)
        file_hours[fi]  = int(hr)
        file_months[fi] = int(mo)
        dpm = [0,31,28,31,30,31,30,31,31,30,31,30,31]
        file_days[fi]   = sum(dpm[:int(mo)]) + int(dy)

site_oh   = np.eye(len(SITES), dtype=np.float32)[file_sites]
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24), np.cos(2*np.pi*file_hours/24)], 1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12), np.cos(2*np.pi*(file_months-1)/12)], 1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365), np.cos(2*np.pi*(file_days-1)/365)], 1).astype(np.float32)
geo_all   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], 1).astype(np.float32)  # (66, 15)

def make_X_nl(pca_dims=24, geo_w=1.0):
    pca = PCA(n_components=pca_dims, random_state=42).fit(file_embs_norm)
    X_pca = pca.transform(file_embs_norm).astype(np.float32)
    X_pca /= (X_pca.std(0) + 1e-6)
    X_geo = geo_all * geo_w
    X = np.concatenate([X_pca, X_geo], 1).astype(np.float32)
    X /= np.linalg.norm(X, 1, keepdims=True) + 1e-8
    return X

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS = 1e-7

def vlom_blend(a, b, w_a=0.5, w_b=0.5):
    w_sum = w_a + w_b
    w_a /= w_sum; w_b /= w_sum
    log_a = np.log(a.clip(EPS) / (1-a).clip(EPS))
    log_b = np.log(b.clip(EPS) / (1-b).clip(EPS))
    return sigmoid(w_a * log_a + w_b * log_b)

def attn_knn_loo(X, k=10, T=0.2):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr[top]]).sum(0)
    return preds

def full_pipeline_auc(y_ep, lam=0.25, proto_w=0.5, sed_w=0.5, sed_agg='max'):
    """Compute full pipeline AUC given embed_prior predictions y_ep."""
    sed_prob = file_sed_max if sed_agg == 'max' else file_sed_mean
    proto_prob = sigmoid(file_logit_max)
    base_probs = vlom_blend(proto_prob, sed_prob, w_a=proto_w, w_b=sed_w)
    base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))
    ep_logit = np.log(y_ep.clip(EPS)) - np.log((1-y_ep).clip(EPS))
    full_logit = base_logit + lam * ep_logit
    full_probs = sigmoid(full_logit)
    return macro_auc(file_labels, full_probs)

# Reference: v7-geo-knn (pure attn-KNN k10 T0.2, λ=0.25, 50/50)
X_nl_24 = make_X_nl(pca_dims=24, geo_w=1.0)
y_ref = attn_knn_loo(X_nl_24, k=10, T=0.2)
ref_auc = full_pipeline_auc(y_ref, lam=0.25, proto_w=0.5, sed_w=0.5)
print(f"\nReference v7-geo-knn: {ref_auc:.4f}")
print(f"Target: > {ref_auc:.4f}\n")

BEST = ref_auc
results = []

def record(name, auc, params=None):
    global BEST
    delta = auc - BEST
    marker = " ★ NEW BEST" if auc > BEST else ""
    if auc > BEST:
        BEST = auc
    results.append({'name': name, 'auc': auc, 'params': params or {}})
    return marker

# ── A) λ sweep ────────────────────────────────────────────────────────────
print("="*60)
print("A) λ sweep (k=10, T=0.2, PCA24, 50/50)")
print("="*60)
for lam in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50]:
    auc = full_pipeline_auc(y_ref, lam=lam)
    mk = record(f"lam{lam:.2f}", auc, {'lam': lam})
    print(f"  λ={lam:.2f}: {auc:.4f}{mk}")

# ── B) k sweep ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("B) k sweep (T=0.2, PCA24, best-λ so far)")
print("="*60)
best_lam_A = sorted(results, key=lambda x: -x['auc'])[0]['params'].get('lam', 0.25)
print(f"  Using best λ={best_lam_A:.2f} from A)")
for k in [3, 5, 7, 10, 12, 15, 20]:
    y_k = attn_knn_loo(X_nl_24, k=k, T=0.2)
    auc = full_pipeline_auc(y_k, lam=best_lam_A)
    mk = record(f"k{k}", auc, {'k': k, 'lam': best_lam_A})
    print(f"  k={k}: {auc:.4f}{mk}")

# ── C) T sweep ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("C) T sweep (k=10, PCA24, best-λ so far)")
print("="*60)
for T in [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]:
    y_T = attn_knn_loo(X_nl_24, k=10, T=T)
    auc = full_pipeline_auc(y_T, lam=best_lam_A)
    mk = record(f"T{T:.2f}", auc, {'T': T, 'lam': best_lam_A})
    print(f"  T={T:.2f}: {auc:.4f}{mk}")

# ── D) VLOM weight sweep ───────────────────────────────────────────────────
print("\n" + "="*60)
print("D) VLOM weight sweep (k=10, T=0.2, λ=0.25)")
print("="*60)
for pw in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    sw = 1.0 - pw
    auc = full_pipeline_auc(y_ref, lam=0.25, proto_w=pw, sed_w=sw)
    mk = record(f"proto{pw:.0%}", auc, {'proto_w': pw, 'sed_w': sw})
    print(f"  proto_w={pw:.0%}: {auc:.4f}{mk}")

# ── E) SED aggregation ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("E) SED aggregation (k=10, T=0.2, λ=0.25, 50/50)")
print("="*60)
for agg in ['max', 'mean']:
    auc = full_pipeline_auc(y_ref, lam=0.25, sed_agg=agg)
    mk = record(f"sed_{agg}", auc, {'sed_agg': agg})
    print(f"  SED {agg}: {auc:.4f}{mk}")

# SED (max+mean)/2
file_sed_blend = (file_sed_max + file_sed_mean) / 2
proto_prob = sigmoid(file_logit_max)
base_tmp = vlom_blend(proto_prob, file_sed_blend)
base_logit_tmp = np.log(base_tmp.clip(EPS)) - np.log((1-base_tmp).clip(EPS))
ep_logit = np.log(y_ref.clip(EPS)) - np.log((1-y_ref).clip(EPS))
full_tmp = sigmoid(base_logit_tmp + 0.25 * ep_logit)
auc = macro_auc(file_labels, full_tmp)
mk = record("sed_maxmean", auc, {'sed_agg': 'maxmean'})
print(f"  SED (max+mean)/2: {auc:.4f}{mk}")

# ── F) PCA dims sweep ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("F) PCA dims sweep (k=10, T=0.2, λ=0.25)")
print("="*60)
for pca_d in [8, 12, 16, 18, 24, 30, 36, 48, 64]:
    X_tmp = make_X_nl(pca_dims=pca_d, geo_w=1.0)
    y_tmp = attn_knn_loo(X_tmp, k=10, T=0.2)
    auc = full_pipeline_auc(y_tmp, lam=0.25)
    mk = record(f"pca{pca_d}", auc, {'pca_dims': pca_d})
    print(f"  PCA{pca_d}: {auc:.4f}{mk}")

# ── G) Geo weight sweep ────────────────────────────────────────────────────
print("\n" + "="*60)
print("G) Geo weight (PCA24, k=10, T=0.2, λ=0.25)")
print("="*60)
for gw in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
    X_tmp = make_X_nl(pca_dims=24, geo_w=gw)
    y_tmp = attn_knn_loo(X_tmp, k=10, T=0.2)
    auc = full_pipeline_auc(y_tmp, lam=0.25)
    mk = record(f"geo_w{gw:.1f}", auc, {'geo_w': gw})
    print(f"  geo_w={gw:.1f}: {auc:.4f}{mk}")

# ── H) 3-way VLOM ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("H) 3-way VLOM: ProtoSSM + SED + EmbedPrior as 3rd component")
print("="*60)
proto_prob = sigmoid(file_logit_max)
ep_prob = y_ref
for pw, sw, ew in [
    (0.4, 0.4, 0.2),
    (0.35, 0.35, 0.30),
    (0.33, 0.33, 0.34),
    (0.4, 0.3, 0.3),
    (0.3, 0.4, 0.3),
    (0.30, 0.30, 0.40),
    (0.25, 0.25, 0.50),
]:
    w_sum = pw + sw + ew
    a, b, c = pw/w_sum, sw/w_sum, ew/w_sum
    la = np.log(proto_prob.clip(EPS)) - np.log((1-proto_prob).clip(EPS))
    lb = np.log(file_sed_max.clip(EPS)) - np.log((1-file_sed_max).clip(EPS))
    lc = np.log(ep_prob.clip(EPS)) - np.log((1-ep_prob).clip(EPS))
    full = sigmoid(a*la + b*lb + c*lc)
    auc = macro_auc(file_labels, full)
    mk = record(f"3way_{pw:.2f}_{sw:.2f}_{ew:.2f}", auc, {'proto_w': pw, 'sed_w': sw, 'ep_w': ew})
    print(f"  proto={pw:.2f} sed={sw:.2f} ep={ew:.2f}: {auc:.4f}{mk}")

# ── I) Joint λ+k grid (best PCA so far) ────────────────────────────────────
print("\n" + "="*60)
print("I) Joint k+T+λ grid (PCA24, 50/50)")
print("="*60)
best_grid = []
for k in [5, 7, 10, 15]:
    for T in [0.10, 0.15, 0.20, 0.25]:
        y_kt = attn_knn_loo(X_nl_24, k=k, T=T)
        for lam in [0.15, 0.20, 0.25, 0.30, 0.35]:
            auc = full_pipeline_auc(y_kt, lam=lam)
            best_grid.append((auc, k, T, lam))

best_grid.sort(reverse=True)
print("  Top 15:")
for auc, k, T, lam in best_grid[:15]:
    mk = record(f"grid_k{k}_T{T:.2f}_lam{lam:.2f}", auc, {'k': k, 'T': T, 'lam': lam})
    print(f"    k={k} T={T:.2f} λ={lam:.2f}: {auc:.4f}{mk}")

# ── J) PCA+geo+λ joint sweep (top PCA dims) ─────────────────────────────
print("\n" + "="*60)
print("J) Joint PCA+geo+λ (best k/T from I)")
print("="*60)
best_from_grid = best_grid[0]
best_k_I = best_from_grid[1]
best_T_I = best_from_grid[2]
print(f"  Using best k={best_k_I}, T={best_T_I:.2f}")
for pca_d in [18, 24, 30, 36]:
    for gw in [0.7, 1.0, 1.3, 1.5]:
        X_tmp = make_X_nl(pca_dims=pca_d, geo_w=gw)
        y_tmp = attn_knn_loo(X_tmp, k=best_k_I, T=best_T_I)
        for lam in [0.20, 0.25, 0.30]:
            auc = full_pipeline_auc(y_tmp, lam=lam)
            record(f"J_pca{pca_d}_gw{gw:.1f}_lam{lam:.2f}", auc,
                   {'pca_dims': pca_d, 'geo_w': gw, 'lam': lam, 'k': best_k_I, 'T': best_T_I})

# Best from J
results_J = [r for r in results if r['name'].startswith('J_')]
results_J.sort(key=lambda x: -x['auc'])
print("  Top 10 from J:")
for r in results_J[:10]:
    p = r['params']
    mk = " ★" if r['auc'] > ref_auc else ""
    print(f"    {r['name']}: {r['auc']:.4f}{mk}")

# ── K) No-emb geo only ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("K) Geo-only embedding (no Perch emb, pure geographic KNN)")
print("="*60)
geo_norm = geo_all / (np.linalg.norm(geo_all, 1, keepdims=True) + 1e-8)
for k in [3, 5, 7, 10]:
    for T in [0.1, 0.2, 0.3]:
        y_geo = attn_knn_loo(geo_norm, k=k, T=T)
        for lam in [0.15, 0.20, 0.25]:
            auc = full_pipeline_auc(y_geo, lam=lam)
            name = f"K_geo_k{k}_T{T:.1f}_lam{lam:.2f}"
            record(name, auc, {'k': k, 'T': T, 'lam': lam})
results_K = [r for r in results if r['name'].startswith('K_')]
results_K.sort(key=lambda x: -x['auc'])
print("  Top 5 geo-only:")
for r in results_K[:5]:
    mk = " ★" if r['auc'] > ref_auc else ""
    print(f"    {r['name']}: {r['auc']:.4f}{mk}")

# ── L) Proto weight + best config ──────────────────────────────────────────
print("\n" + "="*60)
print("L) VLOM weight × λ grid (PCA24, k=10, T=0.2)")
print("="*60)
best_vlom = []
for pw in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    sw = 1.0 - pw
    for lam in [0.15, 0.20, 0.25, 0.30, 0.35]:
        auc = full_pipeline_auc(y_ref, lam=lam, proto_w=pw, sed_w=sw)
        best_vlom.append((auc, pw, lam))

best_vlom.sort(reverse=True)
print("  Top 15 VLOM×λ:")
for auc, pw, lam in best_vlom[:15]:
    mk = record(f"L_pw{pw:.2f}_lam{lam:.2f}", auc, {'proto_w': pw, 'lam': lam})
    print(f"    proto={pw:.2f} λ={lam:.2f}: {auc:.4f}{mk}")

# ─── SUMMARY ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY — methods that beat reference (0.9246)")
print("="*70)
better = [r for r in results if r['auc'] > ref_auc]
better.sort(key=lambda x: -x['auc'])
print(f"  Reference: {ref_auc:.4f}")
print(f"  # methods beating reference: {len(better)}")
for r in better[:30]:
    print(f"    {r['name']:45s} {r['auc']:.4f}  (+{r['auc']-ref_auc:.4f})")

print(f"\n  Overall best: {results[0]['auc'] if results else 'N/A'}")
all_sorted = sorted(results, key=lambda x: -x['auc'])
print(f"  Top 5:")
for r in all_sorted[:5]:
    print(f"    {r['name']:45s} {r['auc']:.4f}")
print("done")
