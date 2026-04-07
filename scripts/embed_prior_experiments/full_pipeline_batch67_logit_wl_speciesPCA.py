"""
Batch 67: WL Contrast in Logit Space + Species-specific PCA Subspace

全新方法，兩個從未試過的方向：

Method 1: WL Contrast in Perch Logit Space (234-dim)
  - 現有方法全部在 1536-dim embedding 空間做 WL contrast
  - Perch logits 是 234-dim 語意壓縮空間，鳥種語意更集中
  - 在 L2-normalized logit vectors 上跑 WL contrast
  - 嘗試獨立使用 + 與 WL-emb triple 融合

Method 2: Species-Specific PCA Subspace Scoring
  - 對每個 species，只用 positive training windows 做 PCA
  - 重建誤差小 = 與正類子空間相似 → 高 score
  - 等效於「per-species subspace classifier」
  - 不同於 global PCA（ICA100/PCA80 是全資料 PCA）

Current best: 0.9873025 (wl_uh_seedens_blend)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logit_win  = perch['logits'].astype(np.float32)   # (739, 234) ← KEY
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

# Logit space embeddings (multiple normalization strategies)
# L2 normalized raw logits
logit_l2 = normalize(logit_win, norm='l2').astype(np.float32)
# Standardized then L2 normalized
sc_logit = StandardScaler()
logit_std_l2 = normalize(sc_logit.fit_transform(logit_win).astype(np.float32), norm='l2')
# Sigmoid + L2 normalize
logit_sig_l2 = normalize((1.0 / (1.0 + np.exp(-logit_win))).astype(np.float32), norm='l2')
# Softmax + L2 normalize
logit_sfm = logit_win - logit_win.max(1, keepdims=True)
logit_sfm = np.exp(logit_sfm) / np.exp(logit_sfm).sum(1, keepdims=True)
logit_sfm_l2 = normalize(logit_sfm.astype(np.float32), norm='l2')

# Best WL-UH-triple params
W_ICA, W_STD, W_PCA = 0.655, 0.225, 0.120
ICA_K, ICA_WMA, ICA_WMP = 50, 0.92, 0.80
STD_K, STD_WMA, STD_WMP =  4, 0.65, 0.60
PCA_K, PCA_WMA, PCA_WMP =  4, 0.60, 0.70
print("Done.", flush=True)

# ─── Core WL contrast ────────────────────────────────────────────────────────
def wl_contrast_loo(emb_n, k_neg, wmp, wma):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr_m = win_file_id != fi
        tr = emb_n[tr_m]; tl = labels_win[tr_m]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
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
        out[fi] = wma * ws.max(0) + (1-wma) * ws.mean(0)
    return out

# Precompute WL-UH-triple reference
print("Computing WL-UH-triple reference...", flush=True)
t0 = time.time()
s_ica = wl_contrast_loo(ew_ica, ICA_K, ICA_WMP, ICA_WMA)
s_std = wl_contrast_loo(ew_std, STD_K, STD_WMP, STD_WMA)
s_pca = wl_contrast_loo(ew_pca, PCA_K, PCA_WMP, PCA_WMA)
uh_triple = W_ICA * s_ica + W_STD * s_std + W_PCA * s_pca
uh_auc = eval_loo(uh_triple)
print(f"  UH-triple: {uh_auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: WL Contrast in Logit Space
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: WL Contrast in Logit Space (234-dim) ===", flush=True)
print("理念：logit 空間是 Perch 訓練得的語意壓縮空間，鳥種相似性更集中", flush=True)

t0 = time.time()
best_wl_logit = 0; best_cfg_wl_logit = None

for emb_log, log_name in [
    (logit_l2,      'logit_l2'),
    (logit_std_l2,  'logit_std'),
    (logit_sig_l2,  'logit_sig'),
    (logit_sfm_l2,  'logit_sfm'),
]:
    for k_neg in [4, 8, 16, 32, 50]:
        for wmp in [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
            for wma in [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]:
                out = wl_contrast_loo(emb_log, k_neg, wmp, wma)
                auc = eval_loo(out)
                if auc > best_wl_logit:
                    best_wl_logit = auc
                    best_cfg_wl_logit = (log_name, k_neg, wmp, wma)

print(f"  WL-logit best: {best_wl_logit:.4f}  cfg={best_cfg_wl_logit}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_logit_space'] = best_wl_logit
print(f"  {'*** NEW BEST ***' if best_wl_logit > CURRENT_BEST else ''}", flush=True)

# Blend logit-WL with UH-triple
print("  Logit-WL + UH-triple blend...", flush=True)
t0 = time.time()
best_logit_blend = 0; best_cfg_logit_blend = None

if best_cfg_wl_logit:
    log_n, k_n, wmp_n, wma_n = best_cfg_wl_logit
    emb_best_log = {'logit_l2': logit_l2, 'logit_std': logit_std_l2,
                    'logit_sig': logit_sig_l2, 'logit_sfm': logit_sfm_l2}[log_n]
    logit_wl_scores = wl_contrast_loo(emb_best_log, k_n, wmp_n, wma_n)
    for w_logit in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        blend = (1-w_logit) * uh_triple + w_logit * logit_wl_scores
        auc = eval_loo(blend)
        if auc > best_logit_blend:
            best_logit_blend = auc; best_cfg_logit_blend = w_logit

print(f"  Logit-WL+UH blend: {best_logit_blend:.4f}  w_logit={best_cfg_logit_blend}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_logit_uh_blend'] = best_logit_blend
print(f"  {'*** NEW BEST ***' if best_logit_blend > CURRENT_BEST else ''}", flush=True)

# Also: Direct Perch logit predictions blended with UH-triple
print("  Direct sigmoid(logit) + UH-triple blend...", flush=True)
t0 = time.time()
probs_direct = 1.0 / (1.0 + np.exp(-logit_win))
preds_direct = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    preds_direct[fi] = probs_direct[s:e].max(0)
direct_auc = eval_loo(preds_direct)
print(f"  Direct logit AUC (no LOO needed): {direct_auc:.4f}", flush=True)

best_direct_blend = 0; best_cfg_direct_blend = None
for w_dir in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
    blend = (1-w_dir) * uh_triple + w_dir * preds_direct
    auc = eval_loo(blend)
    if auc > best_direct_blend:
        best_direct_blend = auc; best_cfg_direct_blend = w_dir

print(f"  Direct-logit+UH blend: {best_direct_blend:.4f}  w_dir={best_cfg_direct_blend}  ({time.time()-t0:.1f}s)", flush=True)
results['direct_logit_uh_blend'] = best_direct_blend
print(f"  {'*** NEW BEST ***' if best_direct_blend > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Species-Specific PCA Subspace Scoring
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Species-Specific PCA Subspace Scoring ===", flush=True)
print("理念：每個 species 只用其正例 windows 做 PCA，重建誤差小 = 屬於該物種", flush=True)

def species_subspace_loo(emb_n, n_components, wma):
    """
    Per-species PCA subspace scorer.
    For each species: fit PCA on positive training windows,
    score test windows by their similarity to the subspace
    (1 - normalized reconstruction error = projection fidelity).
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]          # (n_te, dim)
        tr_m = win_file_id != fi
        tr = emb_n[tr_m]; tl = labels_win[tr_m]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]
            n_pos = len(pos_wins)
            k = min(n_components, n_pos - 1, pos_wins.shape[1] - 1)
            if k < 1:
                # Single positive: use cosine similarity to the prototype
                pp = pos_wins.mean(0); pp /= np.linalg.norm(pp) + EPS
                ws[:, si] = (te @ pp + 1) / 2
                continue
            # Fit PCA on positive windows
            pca_sp = PCA(n_components=k)
            try:
                pca_sp.fit(pos_wins)
                # Project test windows onto subspace
                te_proj = pca_sp.transform(te)
                te_recon = pca_sp.inverse_transform(te_proj)
                # Reconstruction fidelity: 1 - normalized reconstruction error
                recon_err = np.linalg.norm(te - te_recon, axis=1)
                te_norm = np.linalg.norm(te, axis=1)
                fidelity = 1 - recon_err / (te_norm + EPS)
                ws[:, si] = np.clip(fidelity, 0, 1)
            except Exception:
                ws[:, si] = 0.5
        out[fi] = wma * ws.max(0) + (1-wma) * ws.mean(0)
    return out

t0 = time.time()
best_subsp = 0; best_cfg_subsp = None

for n_comp in [1, 2, 3, 5]:
    for wma in [0.80, 0.85, 0.90, 0.92, 0.95]:
        for emb, name in [(ew_ica, 'ica100'), (ew_pca, 'pca80')]:
            out = species_subspace_loo(emb, n_comp, wma)
            auc = eval_loo(out)
            if auc > best_subsp:
                best_subsp = auc; best_cfg_subsp = (name, n_comp, wma)

print(f"  Species-PCA best: {best_subsp:.4f}  cfg={best_cfg_subsp}  ({time.time()-t0:.0f}s)", flush=True)
results['species_subspace'] = best_subsp
print(f"  {'*** NEW BEST ***' if best_subsp > CURRENT_BEST else ''}", flush=True)

# Blend with UH-triple
t0 = time.time()
best_subsp_blend = 0; best_cfg_subsp_blend = None
if best_cfg_subsp:
    name_s, nc_s, wma_s = best_cfg_subsp
    emb_s = {'ica100': ew_ica, 'pca80': ew_pca}[name_s]
    subsp_scores = species_subspace_loo(emb_s, nc_s, wma_s)
    for w_s in [0.05, 0.10, 0.15, 0.20, 0.25]:
        blend = (1-w_s) * uh_triple + w_s * subsp_scores
        auc = eval_loo(blend)
        if auc > best_subsp_blend:
            best_subsp_blend = auc; best_cfg_subsp_blend = w_s

print(f"  Subspace+UH blend: {best_subsp_blend:.4f}  ({time.time()-t0:.1f}s)", flush=True)
results['species_subspace_uh_blend'] = best_subsp_blend
print(f"  {'*** NEW BEST ***' if best_subsp_blend > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: WL contrast in JOINT embedding+logit space
# Concatenate ICA100 embedding with L2-normalized logits → richer feature
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Joint Emb+Logit WL Contrast ===", flush=True)
print("理念：將 ICA100 和 logit_l2 拼接，在更豐富空間做 WL contrast", flush=True)

t0 = time.time()
best_joint = 0; best_cfg_joint = None

for w_emb in [0.5, 0.6, 0.7, 0.8, 0.9]:
    w_log = 1.0 - w_emb
    # Weighted concatenation and re-normalize
    joint_emb = normalize(
        np.concatenate([ew_ica * w_emb, logit_l2 * w_log], axis=1).astype(np.float32),
        norm='l2'
    )
    for k_neg in [20, 50]:
        for wmp in [0.70, 0.80]:
            for wma in [0.88, 0.90, 0.92]:
                out = wl_contrast_loo(joint_emb, k_neg, wmp, wma)
                auc = eval_loo(out)
                if auc > best_joint:
                    best_joint = auc; best_cfg_joint = (w_emb, k_neg, wmp, wma)

print(f"  Joint Emb+Logit best: {best_joint:.4f}  cfg={best_cfg_joint}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_joint_emb_logit'] = best_joint
print(f"  {'*** NEW BEST ***' if best_joint > CURRENT_BEST else ''}", flush=True)

# Blend joint with UH-triple
t0 = time.time()
best_joint_blend = 0; best_cfg_joint_blend = None
if best_cfg_joint:
    we, kn, wmp_j, wma_j = best_cfg_joint
    wl_j = 1.0 - we
    joint_emb_best = normalize(
        np.concatenate([ew_ica * we, logit_l2 * wl_j], axis=1).astype(np.float32), norm='l2')
    joint_scores = wl_contrast_loo(joint_emb_best, kn, wmp_j, wma_j)
    for w_j in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        blend = (1-w_j) * uh_triple + w_j * joint_scores
        auc = eval_loo(blend)
        if auc > best_joint_blend:
            best_joint_blend = auc; best_cfg_joint_blend = w_j

print(f"  Joint+UH blend: {best_joint_blend:.4f}  ({time.time()-t0:.1f}s)", flush=True)
results['wl_joint_uh_blend'] = best_joint_blend
print(f"  {'*** NEW BEST ***' if best_joint_blend > CURRENT_BEST else ''}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 67 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

print(f"\n  比較 UH-triple 參考值: {uh_auc:.4f}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
new_best_found = False
best_new_method = None; best_new_auc = 0; best_new_cfg = None

for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        new_best_found = True
        if auc > best_new_auc:
            best_new_auc = auc; best_new_method = name

with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)

print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
if not new_best_found:
    print("未超越 0.9873，已 append 到 experiments。", flush=True)
else:
    print(f"NEW BEST: {best_new_method} AUC={best_new_auc:.4f}", flush=True)
