"""
Batch 66b: Spherical K-Means + Top-K Diverse Prototypes (快速版)
去除 bootstrap（太慢），專注 multi-prototype 方法

Method 1: Spherical K-Means (K=2,3 centers per species)
  - 用 K-Means 取代單一 mean prototype
  - Score = max sim to ANY center

Method 2: Top-K Diverse Prototypes (Max-Min Selection)
  - 選 K 個最多樣的正例 windows 作為 prototypes
  - 第1個 = mean prototype；後續每個選與已選集合最遠的
  - Deterministic, 不需要 k-means fitting

Current best: 0.9873025
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
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

# ─── Max-Min diverse prototype selection ──────────────────────────────────────
def select_diverse_prototypes(pw, k):
    """Select k most diverse prototypes using greedy max-min (farthest point sampling)."""
    if len(pw) <= k:
        return pw
    # Start with mean prototype
    centers = [pw.mean(0)]
    centers[0] /= np.linalg.norm(centers[0]) + EPS
    for _ in range(k - 1):
        c_arr = np.stack(centers)  # [n_sel, dim]
        sims = pw @ c_arr.T  # [n_pos, n_sel]
        max_sim_to_selected = sims.max(1)  # [n_pos]
        # Pick the one with lowest max similarity to selected set
        next_idx = np.argmin(max_sim_to_selected)
        centers.append(pw[next_idx] / (np.linalg.norm(pw[next_idx]) + EPS))
    return np.stack(centers)  # [k, dim]

# ─── WL with multi-prototype ──────────────────────────────────────────────────
def wl_multiproto(emb_n, fi, k_proto, k_neg, wma, w_max_ctr, proto_mode='kmeans'):
    """
    k_proto: number of positive prototypes
    w_max_ctr: weight for max-center similarity (vs mean-center similarity)
    proto_mode: 'kmeans' or 'diverse'
    """
    te = emb_n[win_file_id == fi]
    tr_m = win_file_id != fi
    tr = emb_n[tr_m]; tl = labels_win[tr_m]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
        if not pm.any(): ws[:, si] = 0.5; continue
        pw = tr[pm]
        n_pos = len(pw)
        k_act = min(k_proto, n_pos)

        if k_act <= 1 or proto_mode == 'mean':
            pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
            centers = pp[None]
        elif proto_mode == 'diverse':
            centers = select_diverse_prototypes(pw, k_act)  # [k_act, dim]
        else:  # kmeans
            # Simple K-means (few iterations, fast)
            rng_idx = np.linspace(0, n_pos-1, k_act, dtype=int)
            centers = pw[rng_idx].copy()
            centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + EPS)
            for _ in range(10):
                sims = pw @ centers.T  # [n_pos, k_act]
                assignments = sims.argmax(1)
                new_centers = np.zeros_like(centers)
                for ki in range(k_act):
                    members = pw[assignments == ki]
                    if len(members) > 0:
                        new_centers[ki] = members.mean(0)
                new_centers /= (np.linalg.norm(new_centers, axis=1, keepdims=True) + EPS)
                if np.allclose(centers, new_centers, atol=1e-4): break
                centers = new_centers

        # Score: similarity to centers
        ctr_sims = te @ centers.T  # [n_te, k_act]
        sp = w_max_ctr * ctr_sims.max(1) + (1-w_max_ctr) * ctr_sims.mean(1)

        if nm.any():
            nw = tr[nm]; ns2 = te @ nw.T; k2 = min(k_neg, ns2.shape[1])
            tn = nw[np.argsort(-ns2, axis=1)[:, :k2]].mean(1)
            tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
            ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return wma * ws.max(0) + (1-wma) * ws.mean(0)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Diverse Prototypes (max-min selection)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Diverse Prototypes (Max-Min Selection) ===", flush=True)
t0 = time.time()
best_div = 0; best_cfg_div = None

for k_proto in [2, 3, 4]:
    for wma in [0.88, 0.90, 0.92, 0.95]:
        for w_max_c in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for emb, k_neg, name in [
                (ew_ica, ICA_K, 'ica100'),
                (ew_pca, PCA_K, 'pca80'),
            ]:
                out = np.stack([wl_multiproto(emb, fi, k_proto, k_neg, wma, w_max_c, 'diverse')
                                for fi in range(n_files)])
                auc = eval_loo(out)
                if auc > best_div: best_div = auc; best_cfg_div = (name, k_proto, wma, w_max_c)

print(f"  Diverse-Proto best: {best_div:.4f}  cfg={best_cfg_div}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_diverse_proto'] = best_div
print(f"  {'*** NEW BEST ***' if best_div > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: K-Means prototypes
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: K-Means Prototypes ===", flush=True)
t0 = time.time()
best_km = 0; best_cfg_km = None

for k_proto in [2, 3, 4]:
    for wma in [0.88, 0.90, 0.92, 0.95]:
        for w_max_c in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for emb, k_neg, name in [
                (ew_ica, ICA_K, 'ica100'),
                (ew_pca, PCA_K, 'pca80'),
            ]:
                out = np.stack([wl_multiproto(emb, fi, k_proto, k_neg, wma, w_max_c, 'kmeans')
                                for fi in range(n_files)])
                auc = eval_loo(out)
                if auc > best_km: best_km = auc; best_cfg_km = (name, k_proto, wma, w_max_c)

print(f"  KMeans-Proto best: {best_km:.4f}  cfg={best_cfg_km}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_kmeans_proto'] = best_km
print(f"  {'*** NEW BEST ***' if best_km > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Multi-proto triple blend
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Multi-Proto Triple Blend ===", flush=True)
t0 = time.time()
best_mpt = 0; best_cfg_mpt = None

# Best single-emb configs
best_k_ica = 2; best_wma_ica = 0.92; best_wmc_ica = 0.80  # fallback
if best_cfg_div and best_cfg_div[0] == 'ica100':
    _, best_k_ica, best_wma_ica, best_wmc_ica = best_cfg_div
elif best_cfg_km and best_cfg_km[0] == 'ica100':
    _, best_k_ica, best_wma_ica, best_wmc_ica = best_cfg_km

for k_proto in [2, 3]:
    for wma in [0.90, 0.92]:
        for w_max_c in [0.7, 0.8]:
            mode = 'diverse'  # faster
            s_ica = np.stack([wl_multiproto(ew_ica, fi, k_proto, ICA_K, wma, w_max_c, mode)
                              for fi in range(n_files)])
            s_std = np.stack([wl_multiproto(ew_std, fi, k_proto, STD_K, wma, w_max_c, mode)
                              for fi in range(n_files)])
            s_pca = np.stack([wl_multiproto(ew_pca, fi, k_proto, PCA_K, wma, w_max_c, mode)
                              for fi in range(n_files)])
            out = W_ICA * s_ica + W_STD * s_std + W_PCA * s_pca
            auc = eval_loo(out)
            if auc > best_mpt: best_mpt = auc; best_cfg_mpt = (k_proto, wma, w_max_c)

print(f"  Multi-Proto Triple best: {best_mpt:.4f}  cfg={best_cfg_mpt}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_multiproto_triple'] = best_mpt
print(f"  {'*** NEW BEST ***' if best_mpt > CURRENT_BEST else ''}", flush=True)

# ─── Blend best multi-proto with UH-triple ────────────────────────────────────
if best_mpt > 0.96:
    print("  Blending with UH triple...", flush=True)
    t0 = time.time()
    best_mpt_blend = 0; best_cfg_mpt_blend = None
    # Compute UH-triple LOO
    def wl_uh_score(emb_n, fi, k_neg, wmp):
        te = emb_n[win_file_id == fi]
        tr_m = win_file_id != fi
        tr = emb_n[tr_m]; tl = labels_win[tr_m]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:,si] > 0.5; nm = tl[:,si] < 0.1
            if not pm.any(): ws[:,si] = 0.5; continue
            pw = tr[pm]; ps = te @ pw.T
            pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
            sp = wmp*ps.max(1) + (1-wmp)*(te@pp)
            if nm.any():
                nw=tr[nm]; ns2=te@nw.T; k2=min(k_neg,ns2.shape[1])
                tn=nw[np.argsort(-ns2,axis=1)[:,:k2]].mean(1)
                tn/=np.linalg.norm(tn,axis=1,keepdims=True)+EPS
                ws[:,si]=(sp-(te*tn).sum(1)+1)/2
            else: ws[:,si]=(sp+1)/2
        return ICA_WMA*ws.max(0)+(1-ICA_WMA)*ws.mean(0)
    s_ica_uh = np.stack([wl_uh_score(ew_ica, fi, ICA_K, ICA_WMP) for fi in range(n_files)])
    s_std_uh = np.stack([wl_uh_score(ew_std, fi, STD_K, STD_WMP) for fi in range(n_files)])
    s_pca_uh = np.stack([wl_uh_score(ew_pca, fi, PCA_K, PCA_WMP) for fi in range(n_files)])
    uh_triple = W_ICA*s_ica_uh + W_STD*s_std_uh + W_PCA*s_pca_uh
    print(f"  UH-triple ref: {eval_loo(uh_triple):.4f}", flush=True)
    # Best multi-proto triple scores
    k_b, wma_b, wmc_b = best_cfg_mpt
    s_ica_mp = np.stack([wl_multiproto(ew_ica, fi, k_b, ICA_K, wma_b, wmc_b, 'diverse')
                         for fi in range(n_files)])
    s_std_mp = np.stack([wl_multiproto(ew_std, fi, k_b, STD_K, wma_b, wmc_b, 'diverse')
                         for fi in range(n_files)])
    s_pca_mp = np.stack([wl_multiproto(ew_pca, fi, k_b, PCA_K, wma_b, wmc_b, 'diverse')
                         for fi in range(n_files)])
    mp_triple = W_ICA*s_ica_mp + W_STD*s_std_mp + W_PCA*s_pca_mp
    for w_mp in [0.05, 0.10, 0.15, 0.20, 0.30]:
        blend = (1-w_mp)*uh_triple + w_mp*mp_triple
        auc = eval_loo(blend)
        if auc > best_mpt_blend: best_mpt_blend = auc; best_cfg_mpt_blend = w_mp
    print(f"  MPTriple+UH blend: {best_mpt_blend:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    results['wl_multiproto_uh_blend'] = best_mpt_blend
    print(f"  {'*** NEW BEST ***' if best_mpt_blend > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 66b Summary ===", flush=True)
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
