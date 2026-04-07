"""
Batch 67b: WL in Logit Space + Species Subspace (精簡快速版)
預計算 sim cache（每 emb 只算一次 te@tr.T），大幅加速。

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
logit_win  = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
n_windows  = perch['n_windows']
file_list  = list(perch['file_list'])
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[file_start[fi]:file_end[fi]] = fi
file_labels = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

EPS = 1e-7; mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9873024930999804
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# ─── Precompute all embeddings ────────────────────────────────────────────────
print("Precomputing embeddings...", flush=True)
t0 = time.time()
ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2')
pca80 = PCA(n_components=80, random_state=42)
ew_pca = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2')
scaler = StandardScaler()
ew_std = normalize(PCA(n_components=80, random_state=42).fit_transform(
    scaler.fit_transform(emb_win).astype(np.float32)).astype(np.float32), norm='l2')

# Logit embeddings
logit_l2     = normalize(logit_win, norm='l2').astype(np.float32)
logit_std_l2 = normalize(StandardScaler().fit_transform(logit_win).astype(np.float32), norm='l2')
logit_sig    = (1.0 / (1.0 + np.exp(-logit_win))).astype(np.float32)
logit_sig_l2 = normalize(logit_sig, norm='l2')

W_ICA, W_STD, W_PCA = 0.655, 0.225, 0.120
ICA_K, ICA_WMA, ICA_WMP = 50, 0.92, 0.80
STD_K, STD_WMA, STD_WMP =  4, 0.65, 0.60
PCA_K, PCA_WMA, PCA_WMP =  4, 0.60, 0.70
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

# ─── Sim cache: te@tr.T for each LOO fold ────────────────────────────────────
def build_cache(emb_n):
    c = {}
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr_m = win_file_id != fi
        c[fi] = (te, emb_n[tr_m], labels_win[tr_m], te @ emb_n[tr_m].T)
    return c

# ─── Fast WL contrast using cache ────────────────────────────────────────────
def wl_from_cache(cache, fi, k_neg, wmp, wma):
    te, tr, tl, sims = cache[fi]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pos_idx = np.where(tl[:, si] > 0.5)[0]
        neg_idx = np.where(tl[:, si] < 0.1)[0]
        if len(pos_idx) == 0: ws[:, si] = 0.5; continue
        ps = sims[:, pos_idx]                      # (n_te, n_pos)
        pp = tr[pos_idx].mean(0); pp /= np.linalg.norm(pp) + EPS
        sp = wmp * ps.max(1) + (1-wmp) * (te @ pp)
        if len(neg_idx) > 0:
            ns2 = sims[:, neg_idx]                 # (n_te, n_neg)
            k2 = min(k_neg, len(neg_idx))
            top_idx = np.argsort(-ns2, axis=1)[:, :k2]
            tn_scores = np.array([
                (te[j] @ tr[neg_idx[top_idx[j]]].mean(0) /
                 (np.linalg.norm(tr[neg_idx[top_idx[j]]].mean(0)) + EPS))
                for j in range(len(te))], dtype=np.float32)
            ws[:, si] = (sp - tn_scores + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return wma * ws.max(0) + (1-wma) * ws.mean(0)

def wl_sweep(cache, k_neg_list, wmp_list, wma_list):
    best, cfg = 0, None
    for k in k_neg_list:
        for wmp in wmp_list:
            for wma in wma_list:
                out = np.stack([wl_from_cache(cache, fi, k, wmp, wma) for fi in range(n_files)])
                auc = eval_loo(out)
                if auc > best: best = auc; cfg = (k, wmp, wma)
    return best, cfg

print("Building sim caches...", flush=True)
t0 = time.time()
cache_ica = build_cache(ew_ica)
cache_std = build_cache(ew_std)
cache_pca = build_cache(ew_pca)
cache_ll2 = build_cache(logit_l2)
cache_ls  = build_cache(logit_std_l2)
cache_lsig = build_cache(logit_sig_l2)
print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

# ─── UH-triple reference ─────────────────────────────────────────────────────
print("Computing UH-triple reference...", flush=True)
t0 = time.time()
s_ica = np.stack([wl_from_cache(cache_ica, fi, ICA_K, ICA_WMP, ICA_WMA) for fi in range(n_files)])
s_std = np.stack([wl_from_cache(cache_std, fi, STD_K, STD_WMP, STD_WMA) for fi in range(n_files)])
s_pca = np.stack([wl_from_cache(cache_pca, fi, PCA_K, PCA_WMP, PCA_WMA) for fi in range(n_files)])
uh_triple = W_ICA * s_ica + W_STD * s_std + W_PCA * s_pca
uh_auc = eval_loo(uh_triple)
print(f"  UH-triple: {uh_auc:.4f}  ({time.time()-t0:.1f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: WL in Logit Space
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: WL in Logit Space ===", flush=True)
t0 = time.time()
best_wll, best_cfg_wll = 0, None

K_LIST  = [4, 8, 16, 32, 50]
WMP_LIST = [0.60, 0.70, 0.80, 0.90, 1.0]
WMA_LIST = [0.80, 0.85, 0.90, 0.92, 0.95]

for cache_log, lname in [(cache_ll2, 'logit_l2'), (cache_ls, 'logit_std'), (cache_lsig, 'logit_sig')]:
    b, cfg = wl_sweep(cache_log, K_LIST, WMP_LIST, WMA_LIST)
    print(f"  {lname}: {b:.4f}  cfg={cfg}", flush=True)
    if b > best_wll: best_wll = b; best_cfg_wll = (lname,) + cfg

print(f"  WL-logit best: {best_wll:.4f}  cfg={best_cfg_wll}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_logit_space'] = best_wll
print(f"  {'*** NEW BEST ***' if best_wll > CURRENT_BEST else ''}", flush=True)

# Blend logit-WL with UH-triple
print("  Logit-WL + UH-triple blend...", flush=True)
t0 = time.time()
best_lb, best_cfg_lb = 0, None
if best_cfg_wll:
    lname_b = best_cfg_wll[0]; kb, wmpb, wmab = best_cfg_wll[1:]
    cache_best_log = {'logit_l2': cache_ll2, 'logit_std': cache_ls, 'logit_sig': cache_lsig}[lname_b]
    logit_wl = np.stack([wl_from_cache(cache_best_log, fi, kb, wmpb, wmab) for fi in range(n_files)])
    for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]:
        blend = (1-w) * uh_triple + w * logit_wl
        auc = eval_loo(blend)
        if auc > best_lb: best_lb = auc; best_cfg_lb = w

print(f"  Logit-WL+UH blend: {best_lb:.4f}  w={best_cfg_lb}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_logit_uh_blend'] = best_lb
print(f"  {'*** NEW BEST ***' if best_lb > CURRENT_BEST else ''}", flush=True)

# Direct Perch logit predictions + UH-triple
print("  Direct sigmoid(logit) + UH-triple...", flush=True)
preds_direct = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    preds_direct[fi] = logit_sig[s:e].max(0)
direct_auc = eval_loo(preds_direct)
print(f"  Direct Perch logit AUC: {direct_auc:.4f}", flush=True)

best_db, best_cfg_db = 0, None
for w in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
    blend = (1-w) * uh_triple + w * preds_direct
    auc = eval_loo(blend)
    if auc > best_db: best_db = auc; best_cfg_db = w
results['direct_logit_uh_blend'] = best_db
print(f"  Direct+UH blend: {best_db:.4f}  w={best_cfg_db}", flush=True)
print(f"  {'*** NEW BEST ***' if best_db > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Species-Specific PCA Subspace (fast via precomputed sim cache)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Species-Specific PCA Subspace ===", flush=True)
t0 = time.time()
best_ss, best_cfg_ss = 0, None

def species_subspace_loo(emb_n, n_comp, wma):
    out = np.zeros((n_files, n_species), np.float32)
    dim = emb_n.shape[1]
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr_m = win_file_id != fi
        tr = emb_n[tr_m]; tl = labels_win[tr_m]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; n_pos = len(pos)
            k = min(n_comp, n_pos - 1, dim - 1)
            if k < 1:
                pp = pos.mean(0); pp /= np.linalg.norm(pp) + EPS
                ws[:, si] = np.clip((te @ pp + 1) / 2, 0, 1); continue
            try:
                pca_sp = PCA(n_components=k)
                pca_sp.fit(pos)
                te_proj = pca_sp.transform(te)
                te_recon = pca_sp.inverse_transform(te_proj)
                recon_err = np.linalg.norm(te - te_recon, axis=1)
                te_norm = np.linalg.norm(te, axis=1)
                ws[:, si] = np.clip(1 - recon_err / (te_norm + EPS), 0, 1)
            except Exception:
                ws[:, si] = 0.5
        out[fi] = wma * ws.max(0) + (1-wma) * ws.mean(0)
    return out

for n_comp in [1, 2, 3]:
    for wma in [0.88, 0.90, 0.92]:
        for emb, name in [(ew_ica, 'ica100'), (ew_pca, 'pca80')]:
            out = species_subspace_loo(emb, n_comp, wma)
            auc = eval_loo(out)
            if auc > best_ss: best_ss = auc; best_cfg_ss = (name, n_comp, wma)
    print(f"  n_comp={n_comp} done", flush=True)

print(f"  Species-PCA best: {best_ss:.4f}  cfg={best_cfg_ss}  ({time.time()-t0:.0f}s)", flush=True)
results['species_subspace'] = best_ss
print(f"  {'*** NEW BEST ***' if best_ss > CURRENT_BEST else ''}", flush=True)

# Blend with UH-triple
best_ssb, best_cfg_ssb = 0, None
if best_cfg_ss:
    name_ss, nc_ss, wma_ss = best_cfg_ss
    emb_ss = {'ica100': ew_ica, 'pca80': ew_pca}[name_ss]
    ss_scores = species_subspace_loo(emb_ss, nc_ss, wma_ss)
    for w in [0.05, 0.10, 0.15, 0.20]:
        blend = (1-w) * uh_triple + w * ss_scores
        auc = eval_loo(blend)
        if auc > best_ssb: best_ssb = auc; best_cfg_ssb = w
results['species_subspace_blend'] = best_ssb
print(f"  Subspace+UH blend: {best_ssb:.4f}  w={best_cfg_ssb}", flush=True)
print(f"  {'*** NEW BEST ***' if best_ssb > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Joint Emb+Logit WL (weighted concatenation)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Joint Emb+Logit WL ===", flush=True)
t0 = time.time()
best_joint, best_cfg_joint = 0, None

for w_emb in [0.7, 0.8, 0.9]:
    w_log = 1.0 - w_emb
    joint = normalize(
        np.concatenate([ew_ica * w_emb, logit_l2 * w_log], axis=1).astype(np.float32), norm='l2')
    c_joint = build_cache(joint)
    b, cfg = wl_sweep(c_joint, [20, 50], [0.70, 0.80], [0.90, 0.92])
    print(f"  w_emb={w_emb}: {b:.4f}  cfg={cfg}", flush=True)
    if b > best_joint: best_joint = b; best_cfg_joint = (w_emb,) + cfg

print(f"  Joint best: {best_joint:.4f}  cfg={best_cfg_joint}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_joint_emb_logit'] = best_joint
print(f"  {'*** NEW BEST ***' if best_joint > CURRENT_BEST else ''}", flush=True)

# Blend joint WL with UH-triple
best_jb, best_cfg_jb = 0, None
if best_cfg_joint:
    we = best_cfg_joint[0]; kj, wmpj, wmaj = best_cfg_joint[1:]
    wlj = 1.0 - we
    joint_best = normalize(
        np.concatenate([ew_ica * we, logit_l2 * wlj], axis=1).astype(np.float32), norm='l2')
    c_jb = build_cache(joint_best)
    joint_scores = np.stack([wl_from_cache(c_jb, fi, kj, wmpj, wmaj) for fi in range(n_files)])
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        blend = (1-w) * uh_triple + w * joint_scores
        auc = eval_loo(blend)
        if auc > best_jb: best_jb = auc; best_cfg_jb = w
results['wl_joint_uh_blend'] = best_jb
print(f"  Joint+UH blend: {best_jb:.4f}  w={best_cfg_jb}", flush=True)
print(f"  {'*** NEW BEST ***' if best_jb > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 67b Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)
print(f"  UH-triple ref: {uh_auc:.4f}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
new_best_found = False; best_new_method = None; best_new_auc = 0

for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        new_best_found = True
        if auc > best_new_auc: best_new_auc = auc; best_new_method = name

with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)

print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
if new_best_found:
    print(f"NEW BEST: {best_new_method} AUC={best_new_auc:.4f}", flush=True)
else:
    print("未超越 0.9873，已 append 到 experiments。", flush=True)
