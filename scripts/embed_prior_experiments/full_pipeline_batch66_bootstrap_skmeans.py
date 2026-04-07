"""
Batch 66: Bootstrap Bagging + Spherical K-Means
兩個全新且從未試過的方向

Method 1: Bootstrap Bagging of WL scores
  - LOO-CV 每個 fold，對 training windows 做 bootstrap resample
  - 每次 resample 跑 WL contrast，平均 K 次的 scores
  - 降低 training set 有限性造成的 variance

Method 2: Spherical K-Means per species (多中心 prototype)
  - 現有方法只用一個 mean prototype + max 相似度
  - 改為每個 species 用 K 個 spherical k-means cluster centers
  - Score = max similarity to ANY center (捕捉多峰分布)

Current best: 0.9873025 (wl_uh_seedens_blend)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans
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

EPS = 1e-7
mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9873024930999804
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# ─── Precompute embeddings ────────────────────────────────────────────────────
print("Precomputing embeddings...", flush=True)
ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2')
pca80 = PCA(n_components=80, random_state=42)
ew_pca = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2')
scaler = StandardScaler()
ew_std = normalize(PCA(n_components=80, random_state=42).fit_transform(
    scaler.fit_transform(emb_win).astype(np.float32)).astype(np.float32), norm='l2')
W_ICA, W_STD, W_PCA = 0.655, 0.225, 0.120
ICA_K, ICA_WMA, ICA_WMP = 50, 0.92, 0.80
STD_K, STD_WMA, STD_WMP =  4, 0.65, 0.60
PCA_K, PCA_WMA, PCA_WMP =  4, 0.60, 0.70
print("Done.", flush=True)

# ─── Core WL contrast (returns window scores [n_te, n_species]) ──────────────
def wl_contrast_wins(emb_n, te, tr, tr_lab, k_neg, wmp):
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pm = tr_lab[:, si] > 0.5; nm = tr_lab[:, si] < 0.1
        if not pm.any(): ws[:, si] = 0.5; continue
        pw = tr[pm]; ps = te @ pw.T
        pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
        sp = wmp * ps.max(1) + (1-wmp) * (te @ pp)
        if nm.any():
            nw = tr[nm]; ns2 = te @ nw.T; k2 = min(k_neg, ns2.shape[1])
            tn = nw[np.argsort(-ns2, axis=1)[:, :k2]].mean(1)
            tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
            ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return ws

def agg(ws, wma): return wma * ws.max(0) + (1-wma) * ws.mean(0)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Bootstrap Bagging of WL scores
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Bootstrap Bagging ===", flush=True)
print("理念：對 training windows bootstrap resample，平均多次 WL 結果", flush=True)

def wl_bootstrap_loo(emb_n, k_neg, wmp, wma, n_boot=10, seed=0):
    rng = np.random.RandomState(seed)
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr_idx = np.where(win_file_id != fi)[0]
        tr_lab_all = labels_win[tr_idx]
        tr_all = emb_n[tr_idx]
        acc = np.zeros((len(te), n_species), np.float32)
        for _ in range(n_boot):
            boot_idx = rng.choice(len(tr_idx), len(tr_idx), replace=True)
            ws = wl_contrast_wins(emb_n, te, tr_all[boot_idx], tr_lab_all[boot_idx], k_neg, wmp)
            acc += ws
        out[fi] = agg(acc / n_boot, wma)
    return out

t0 = time.time()
best_boot = 0; best_cfg_boot = None
# Quick sweep: n_boot=5 (fast), then refine with n_boot=10 for best cfg
for n_boot in [5]:
    for wma in [0.88, 0.90, 0.92]:
        for emb, k, wmp, name in [
            (ew_ica, ICA_K, ICA_WMP, 'ica100'),
            (ew_pca, PCA_K, PCA_WMP, 'pca80'),
            (ew_std, STD_K, STD_WMP, 'std80'),
        ]:
            out = wl_bootstrap_loo(emb, k, wmp, wma, n_boot=n_boot)
            auc = eval_loo(out)
            if auc > best_boot: best_boot = auc; best_cfg_boot = (name, k, wmp, wma, n_boot)
            print(f"    boot({n_boot}) {name} wma={wma}: {auc:.4f}", flush=True)

print(f"  Bootstrap best: {best_boot:.4f}  cfg={best_cfg_boot}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_bootstrap'] = best_boot
flag = " *** NEW BEST ***" if best_boot > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# Refine with n_boot=15 for best single embedding
t0 = time.time()
best_boot2 = 0; best_cfg_boot2 = None
if best_cfg_boot:
    name_b, k_b, wmp_b, wma_b, _ = best_cfg_boot
    emb_b = {'ica100': ew_ica, 'pca80': ew_pca, 'std80': ew_std}[name_b]
    for n_boot in [10, 15, 20]:
        out = wl_bootstrap_loo(emb_b, k_b, wmp_b, wma_b, n_boot=n_boot)
        auc = eval_loo(out)
        if auc > best_boot2: best_boot2 = auc; best_cfg_boot2 = (name_b, n_boot)
        print(f"    boot({n_boot}) {name_b}: {auc:.4f}", flush=True)

print(f"  Bootstrap (more) best: {best_boot2:.4f}  cfg={best_cfg_boot2}  ({time.time()-t0:.0f}s)", flush=True)
if best_boot2 > best_boot:
    best_boot = best_boot2
    results['wl_bootstrap'] = best_boot

# Bootstrap triple blend
print("  Bootstrap triple blend...", flush=True)
t0 = time.time()
best_boot_triple = 0; best_cfg_boot_triple = None

def wl_bootstrap_triple_loo(n_boot=5, seed=0):
    rng = np.random.RandomState(seed)
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_ica = ew_ica[win_file_id == fi]
        te_std = ew_std[win_file_id == fi]
        te_pca = ew_pca[win_file_id == fi]
        tr_idx = np.where(win_file_id != fi)[0]
        tr_lab_all = labels_win[tr_idx]
        tr_ica_all = ew_ica[tr_idx]; tr_std_all = ew_std[tr_idx]; tr_pca_all = ew_pca[tr_idx]
        acc = np.zeros((len(te_ica), n_species), np.float32)
        for _ in range(n_boot):
            boot_idx = rng.choice(len(tr_idx), len(tr_idx), replace=True)
            w1 = wl_contrast_wins(ew_ica, te_ica, tr_ica_all[boot_idx], tr_lab_all[boot_idx], ICA_K, ICA_WMP)
            w2 = wl_contrast_wins(ew_std, te_std, tr_std_all[boot_idx], tr_lab_all[boot_idx], STD_K, STD_WMP)
            w3 = wl_contrast_wins(ew_pca, te_pca, tr_pca_all[boot_idx], tr_lab_all[boot_idx], PCA_K, PCA_WMP)
            acc += W_ICA * w1 + W_STD * w2 + W_PCA * w3
        out[fi] = ICA_WMA * (acc/n_boot).max(0) + (1-ICA_WMA) * (acc/n_boot).mean(0)
    return out

for n_boot in [5, 10]:
    out = wl_bootstrap_triple_loo(n_boot=n_boot)
    auc = eval_loo(out)
    if auc > best_boot_triple: best_boot_triple = auc; best_cfg_boot_triple = n_boot
    print(f"    boot-triple({n_boot}): {auc:.4f}", flush=True)

print(f"  Bootstrap-triple best: {best_boot_triple:.4f}  cfg=n_boot={best_cfg_boot_triple}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_bootstrap_triple'] = best_boot_triple
flag = " *** NEW BEST ***" if best_boot_triple > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Spherical K-Means per species (K=2,3 cluster centers)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Spherical K-Means per species ===", flush=True)
print("理念：用 K 個 cluster centers 取代單一 mean prototype，捕捉多峰分布", flush=True)

def wl_skmeans(emb_n, k_centers, k_neg, wmp, wma):
    """K spherical k-means cluster centers as positive prototypes."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr_m = win_file_id != fi
        tr = emb_n[tr_m]; tl = labels_win[tr_m]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]
            # Spherical K-Means cluster centers (or fallback to mean if too few positives)
            n_pos = len(pw)
            k_act = min(k_centers, n_pos)
            if k_act <= 1:
                pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
                centers = pp[None]  # [1, dim]
            else:
                # Run k-means on normalized vectors (spherical k-means)
                km = MiniBatchKMeans(n_clusters=k_act, random_state=42, n_init=3, max_iter=50)
                km.fit(pw)
                centers = normalize(km.cluster_centers_.astype(np.float32), norm='l2')
            # Score: max similarity to any center
            ctr_sims = te @ centers.T  # [n_te, k_act]
            sp = wmp * ctr_sims.max(1) + (1-wmp) * ctr_sims.mean(1)
            if nm.any():
                nw = tr[nm]; ns2 = te @ nw.T; k2 = min(k_neg, ns2.shape[1])
                tn = nw[np.argsort(-ns2, axis=1)[:, :k2]].mean(1)
                tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
                ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
            else:
                ws[:, si] = (sp + 1) / 2
        out[fi] = wma * ws.max(0) + (1-wma) * ws.mean(0)
    return out

t0 = time.time()
best_skm = 0; best_cfg_skm = None
for k_centers in [2, 3, 4, 5]:
    for wma in [0.88, 0.90, 0.92, 0.95]:
        for emb, k_neg, wmp, name in [
            (ew_ica, ICA_K, ICA_WMP, 'ica100'),
            (ew_pca, PCA_K, PCA_WMP, 'pca80'),
        ]:
            out = wl_skmeans(emb, k_centers, k_neg, wmp, wma)
            auc = eval_loo(out)
            if auc > best_skm: best_skm = auc; best_cfg_skm = (name, k_centers, wma)

print(f"  Spherical-KMeans best: {best_skm:.4f}  cfg={best_cfg_skm}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_spherical_kmeans'] = best_skm
flag = " *** NEW BEST ***" if best_skm > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# Blend spherical KMeans with UH triple
t0 = time.time()
best_skm_blend = 0; best_cfg_skm_blend = None
if best_cfg_skm:
    name_s, k_c, wma_s = best_cfg_skm
    emb_s = {'ica100': ew_ica, 'pca80': ew_pca}[name_s]
    k_neg_s = {'ica100': ICA_K, 'pca80': PCA_K}[name_s]
    wmp_s = {'ica100': ICA_WMP, 'pca80': PCA_WMP}[name_s]
    skm_all = wl_skmeans(emb_s, k_c, k_neg_s, wmp_s, wma_s)
    # UH triple reference
    def get_uh_triple():
        from sklearn.decomposition import FastICA as FICA
        # Reuse precomputed ew_ica, ew_std, ew_pca
        out = np.zeros((n_files, n_species), np.float32)
        for fi2 in range(n_files):
            te_i = ew_ica[win_file_id == fi2]; te_s = ew_std[win_file_id == fi2]; te_p = ew_pca[win_file_id == fi2]
            tr_m2 = win_file_id != fi2; tl2 = labels_win[tr_m2]
            w1 = wl_contrast_wins(ew_ica, te_i, ew_ica[tr_m2], tl2, ICA_K, ICA_WMP)
            w2 = wl_contrast_wins(ew_std, te_s, ew_std[tr_m2], tl2, STD_K, STD_WMP)
            w3 = wl_contrast_wins(ew_pca, te_p, ew_pca[tr_m2], tl2, PCA_K, PCA_WMP)
            bw = W_ICA * w1 + W_STD * w2 + W_PCA * w3
            out[fi2] = ICA_WMA * bw.max(0) + (1-ICA_WMA) * bw.mean(0)
        return out
    uh_triple = get_uh_triple()
    uh_auc = eval_loo(uh_triple)
    print(f"  UH-triple ref: {uh_auc:.4f}", flush=True)
    for w_skm in [0.05, 0.10, 0.15, 0.20, 0.25]:
        blend = (1-w_skm) * uh_triple + w_skm * skm_all
        auc = eval_loo(blend)
        if auc > best_skm_blend: best_skm_blend = auc; best_cfg_skm_blend = w_skm

print(f"  SKMeans+UH blend: {best_skm_blend:.4f}  cfg=w_skm={best_cfg_skm_blend}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_skmeans_blend'] = best_skm_blend
flag = " *** NEW BEST ***" if best_skm_blend > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 66 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
new_best_found = False
for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        new_best_found = True
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)

with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)

print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
if not new_best_found:
    print("未超越 0.9873，已 append 到 experiments。", flush=True)
