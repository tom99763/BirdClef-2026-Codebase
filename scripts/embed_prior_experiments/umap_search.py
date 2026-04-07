"""
UMAP + Attn-KNN 搜尋（上次因路徑問題跳過）
+ 更細緻的 ensemble sweep
No-logit best to beat: 0.8779
"""
import numpy as np, json, re, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import umap
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

file_embs  = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels= np.zeros((n_files, n_species), dtype=np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]   = emb_win[s:e].mean(0)
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)

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
geo_all   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], axis=1).astype(np.float32)

# Current best space (pca24+day)
pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
X24  /= (X24.std(0) + 1e-6)
X_nl_pca24 = np.concatenate([X24, geo_all], axis=1).astype(np.float32)
X_nl_pca24 /= np.linalg.norm(X_nl_pca24, axis=1, keepdims=True) + 1e-8

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

def attn_knn_loo(X, k=10, T=0.2):
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr[top]]).sum(0)
    return preds

EPS     = 1e-7
BEST_NL = 0.8779
results = {}
best_so_far = BEST_NL

print(f"Files={n_files}, species={n_species}")
print(f"No-logit best to beat: {BEST_NL:.4f}\n", flush=True)

# ── E) UMAP + Attn-KNN ────────────────────────────────────────────────────
print("="*60)
print("E) UMAP reduction + Attn-KNN")
print("="*60, flush=True)

for n_comp in [8, 12, 16, 24]:
    for n_neighbors in [5, 10, 15]:
        try:
            reducer = umap.UMAP(n_components=n_comp, n_neighbors=n_neighbors,
                                 metric='cosine', random_state=42, verbose=False,
                                 min_dist=0.1)
            X_umap = reducer.fit_transform(file_embs_norm).astype(np.float32)
            X_umap_s = X_umap / (X_umap.std(0) + 1e-6)
            # with geo
            X_ug = np.concatenate([X_umap_s, geo_all], axis=1).astype(np.float32)
            X_ug /= np.linalg.norm(X_ug, axis=1, keepdims=True) + 1e-8
            # without geo
            X_uw = X_umap_s / (np.linalg.norm(X_umap_s, axis=1, keepdims=True) + 1e-8)

            for X_use, geo_tag in [(X_ug, '+geo'), (X_uw, '-geo')]:
                for k, T in [(10, 0.2), (10, 0.15), (15, 0.2), (7, 0.2)]:
                    p   = attn_knn_loo(X_use, k=k, T=T)
                    auc = macro_auc(file_labels, p)
                    nm  = f'umap{n_comp}_nn{n_neighbors}{geo_tag}_k{k}_T{T}'
                    marker = " ← NEW BEST" if auc > best_so_far else ""
                    if auc > best_so_far: best_so_far = auc
                    if auc > BEST_NL - 0.005:
                        print(f"  {nm}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
                    results[nm] = auc
        except Exception as ex:
            print(f"  UMAP n_comp={n_comp} nn={n_neighbors} failed: {ex}", flush=True)

print(f"After E, best so far: {best_so_far:.4f}\n", flush=True)

# ── F) Fine-sweep ensemble weights ────────────────────────────────────────
print("="*60)
print("F) Fine-sweep ens2 (attn + window_knn) weights")
print("="*60, flush=True)

# Load window knn predictions
y_attn = attn_knn_loo(X_nl_pca24, k=10, T=0.2)

# Recompute window KNN
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), dtype=np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

y_win = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = win_file_id != i
    X_tr = emb_win_norm[tr_mask]
    tr_wi = np.where(tr_mask)[0]
    Y_tr  = file_labels[win_file_id[tr_wi]]
    sims  = X_te @ X_tr.T
    top   = np.argsort(-sims, axis=1)[:, :5]
    win_p = np.zeros((te_e - te_s, n_species), dtype=np.float32)
    for wi in range(te_e - te_s):
        w = sims[wi, top[wi]].clip(0); ws = w.sum()
        w = w/ws if ws > 1e-8 else np.ones(5)/5
        win_p[wi] = (w[:, None] * Y_tr[top[wi]]).sum(0)
    y_win[i] = win_p.mean(0)

# Fine sweep
for wa in np.arange(0.60, 0.92, 0.02):
    wb = round(1 - wa, 4)
    blend = wa * y_attn + wb * y_win
    auc   = macro_auc(file_labels, blend)
    nm    = f'ens2_attn{wa:.2f}_win{wb:.2f}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    if auc > BEST_NL - 0.002:
        print(f"  wa={wa:.2f} wb={wb:.2f}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
    results[nm] = auc

# Try different k for window KNN
for k_w in [3, 7, 10]:
    y_w2 = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        Y_tr  = file_labels[win_file_id[tr_wi]]
        sims  = X_te @ X_tr.T; top = np.argsort(-sims, axis=1)[:, :k_w]
        wp = np.zeros((te_e - te_s, n_species), dtype=np.float32)
        for wi in range(te_e - te_s):
            w = sims[wi, top[wi]].clip(0); ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k_w)/k_w
            wp[wi] = (w[:, None] * Y_tr[top[wi]]).sum(0)
        y_w2[i] = wp.mean(0)
    print(f"  window_knn k={k_w}: {macro_auc(file_labels, y_w2):.4f}", flush=True)
    for wa in [0.70, 0.75, 0.80]:
        blend = wa * y_attn + (1-wa) * y_w2
        auc   = macro_auc(file_labels, blend)
        nm    = f'ens2_attn{wa:.2f}_wink{k_w}_{1-wa:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST_NL - 0.002:
            print(f"  wa={wa:.2f} wk={k_w}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

# Try different T for attn-KNN in ensemble
for T_a in [0.1, 0.15, 0.25, 0.3]:
    y_a2 = attn_knn_loo(X_nl_pca24, k=10, T=T_a)
    for wa in [0.70, 0.75, 0.80]:
        blend = wa * y_a2 + (1-wa) * y_win
        auc   = macro_auc(file_labels, blend)
        nm    = f'ens2_attnT{T_a}_wa{wa:.2f}_win{1-wa:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        if auc > BEST_NL - 0.002:
            print(f"  T_attn={T_a} wa={wa:.2f}: {auc:.4f}  (Δ={auc-BEST_NL:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"After F, best so far: {best_so_far:.4f}\n", flush=True)

# ── SUMMARY ────────────────────────────────────────────────────────────────
print("="*60)
print("SUMMARY (top 15)")
print("="*60)
for nm, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    marker = " ← NEW BEST" if auc > BEST_NL else ""
    print(f"  {nm:<65s}  {auc:.4f}  {auc-BEST_NL:+.4f}{marker}")

global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\n整體最佳: {global_best_name} = {global_best_auc:.4f}", flush=True)

# Update results.json
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best_nl = data.get('best_nologit', {}).get('loo_auc', BEST_NL)
for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': 'umap_ens_sweep'})
    if auc > cur_best_nl:
        cur_best_nl = auc
        data['best_nologit'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'no_logit NEW BEST'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"no-logit best: {data['best_nologit']['method']} = {data['best_nologit']['loo_auc']:.4f}")
print("done", flush=True)
