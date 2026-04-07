"""
Batch 64: Product-of-Experts (PoE) + Window-level Ridge Regression
兩個全新方法，從未在之前實驗中出現過

Method 1: WL-PoE
  - 現有 triple blend 用線性加權 (w_ica * score_ica + w_pca * score_pca)
  - PoE 改用乘法組合：p_final = p1*p2*p3 / Z (乘法組合每個 embedding space 的機率)
  - 等效於 log-sum: log_p_final = log_p1 + log_p2 + log_p3 - log_Z
  - 若三個空間對 class 條件獨立，PoE 理論上最優

Method 2: Window-level Ridge Regression (per species)
  - 對每個 species 在 window 層級 fit Ridge Classifier (或 Ridge Regression)
  - Train: 65 files × ~12 windows ≈ 780 windows (其中 5-20 正例)
  - 用 alpha 掃描控制正規化強度
  - Aggregation: 測試 window 預測分數 max / mean

Current best: 0.9873025 (wl_uh_seedens_blend)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import Ridge
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
ew_ica100 = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2')

pca80 = PCA(n_components=80, random_state=42)
ew_pca80 = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2')

scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
ew_std80 = normalize(pca80s.fit_transform(emb_std).astype(np.float32), norm='l2')

# ICA best params (from wl_uh_seedens_blend)
ICA_K_NEG = 50; ICA_WMA = 0.92; ICA_WMP = 0.80
STD_K_NEG = 4;  STD_WMA = 0.65; STD_WMP = 0.60
PCA_K_NEG = 4;  PCA_WMA = 0.60; PCA_WMP = 0.70
W_ICA = 0.655; W_STD = 0.225; W_PCA = 0.120
print("Done.", flush=True)


# ─── WL contrast (single embedding space) ────────────────────────────────────
def wl_contrast_single(emb_wins_n, fi, k_neg, w_max_pos, w_max_agg):
    """Compute WL scores for one test file. Returns [n_te_wins, n_species]."""
    te_wins = emb_wins_n[win_file_id == fi]
    tr_mask = win_file_id != fi
    tr_wins_all = emb_wins_n[tr_mask]
    tr_lab = labels_win[tr_mask]
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pm = tr_lab[:, si] > 0.5
        nm = tr_lab[:, si] < 0.1
        if not pm.any(): ws[:, si] = 0.5; continue
        pw = tr_wins_all[pm]
        ps = te_wins @ pw.T
        pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
        sp = w_max_pos * ps.max(1) + (1 - w_max_pos) * (te_wins @ pp)
        if nm.any():
            nw = tr_wins_all[nm]; ns2 = te_wins @ nw.T
            k2 = min(k_neg, ns2.shape[1])
            tn = nw[np.argsort(-ns2, axis=1)[:, :k2]].mean(1)
            tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
            ws[:, si] = (sp - (te_wins * tn).sum(1) + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return ws  # [n_te, n_species], values in ~[0,1]


# ══════════════════════════════════════════════════════════════════════════════
# Method 1: WL Product-of-Experts (PoE)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: WL Product-of-Experts (PoE) ===", flush=True)
print("理念：三個 embedding 空間各自計算機率，乘法組合而非線性加權", flush=True)

def poe_combine(scores_list, temp=1.0):
    """
    Product of Experts combination.
    scores: list of arrays [..., n_species] in [0,1]
    p_final = prod(p_i) / (prod(p_i) + prod(1-p_i))
    等效於 logit_final = sum(logit_i) where logit = log(p/(1-p))
    """
    # Convert to log-odds
    log_odds_sum = np.zeros_like(scores_list[0])
    for s in scores_list:
        s_clip = np.clip(s, EPS, 1 - EPS)
        log_odds_sum += np.log(s_clip / (1 - s_clip)) * temp
    # Convert back to probability
    return 1.0 / (1.0 + np.exp(-log_odds_sum))

t0 = time.time()
best_poe = 0; best_cfg_poe = None

for temp in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    for agg_mode in ['poe_max', 'poe_mean', 'poe_blend']:
        out = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            ws_ica = wl_contrast_single(ew_ica100, fi, ICA_K_NEG, ICA_WMP, ICA_WMA)
            ws_std = wl_contrast_single(ew_std80,  fi, STD_K_NEG, STD_WMP, STD_WMA)
            ws_pca = wl_contrast_single(ew_pca80,  fi, PCA_K_NEG, PCA_WMP, PCA_WMA)
            # PoE fusion at window level
            ws_poe = poe_combine([ws_ica, ws_std, ws_pca], temp=temp)  # [n_te, n_species]
            if agg_mode == 'poe_max':
                out[fi] = ws_poe.max(0)
            elif agg_mode == 'poe_mean':
                out[fi] = ws_poe.mean(0)
            else:  # blend 0.9 max + 0.1 mean
                out[fi] = 0.9 * ws_poe.max(0) + 0.1 * ws_poe.mean(0)
        auc = eval_loo(out)
        if auc > best_poe:
            best_poe = auc; best_cfg_poe = (temp, agg_mode)

print(f"  PoE best: {best_poe:.4f}  cfg={best_cfg_poe}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_poe'] = best_poe
flag = " *** NEW BEST ***" if best_poe > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# Also try: PoE only for ICA+std (strongest two), blend with PCA
print("  PoE 雙空間 + PCA 線性混合...", flush=True)
t0 = time.time()
best_poe2 = 0; best_cfg_poe2 = None
for temp in [0.5, 0.7, 1.0, 1.5]:
    for w_poe in [0.80, 0.85, 0.90, 0.92, 0.95]:
        for agg_w in [0.88, 0.90, 0.92]:
            out = np.zeros((n_files, n_species), np.float32)
            for fi in range(n_files):
                ws_ica = wl_contrast_single(ew_ica100, fi, ICA_K_NEG, ICA_WMP, ICA_WMA)
                ws_std = wl_contrast_single(ew_std80,  fi, STD_K_NEG, STD_WMP, STD_WMA)
                ws_pca = wl_contrast_single(ew_pca80,  fi, PCA_K_NEG, PCA_WMP, PCA_WMA)
                # PoE of ICA+std, then blend with PCA
                ws_poe2 = poe_combine([ws_ica, ws_std], temp=temp)
                ws_combined = w_poe * ws_poe2 + (1 - w_poe) * ws_pca
                out[fi] = agg_w * ws_combined.max(0) + (1 - agg_w) * ws_combined.mean(0)
            auc = eval_loo(out)
            if auc > best_poe2:
                best_poe2 = auc; best_cfg_poe2 = (temp, w_poe, agg_w)

print(f"  PoE2 best: {best_poe2:.4f}  cfg={best_cfg_poe2}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_poe2'] = best_poe2
flag = " *** NEW BEST ***" if best_poe2 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# Best overall PoE
best_poe_all = max(best_poe, best_poe2)
print(f"  PoE 最佳整體: {best_poe_all:.4f}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Method 2: Window-level Ridge Regression (per species)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Window-level Ridge Regression (per species) ===", flush=True)
print("理念：對每個 species 在 window 層級 fit Ridge 線性分類器", flush=True)

def ridge_win_loo(emb_wins_n, alpha=1.0, w_max_agg=0.90):
    """
    Per-species Ridge regression at window level.
    Train: windows from all files except held-out file.
    Predict: scores for held-out windows, aggregate to file score.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins = emb_wins_n[tr_mask]
        tr_lab  = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            y_tr = (tr_lab[:, si] > 0.5).astype(np.float32)
            if y_tr.sum() < 1:
                ws[:, si] = 0.5; continue
            if y_tr.sum() == len(y_tr):
                ws[:, si] = 1.0; continue
            reg = Ridge(alpha=alpha, fit_intercept=True)
            reg.fit(tr_wins, y_tr)
            raw = reg.predict(te_wins)
            # Normalize to [0,1] range using sigmoid-like scaling
            ws[:, si] = 1.0 / (1.0 + np.exp(-raw * 5.0))
        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

t0 = time.time()
best_ridge = 0; best_cfg_ridge = None

for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    for wma in [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
        for emb, name in [(ew_ica100, 'ica100'), (ew_pca80, 'pca80'), (ew_std80, 'std80')]:
            out = ridge_win_loo(emb, alpha=alpha, w_max_agg=wma)
            auc = eval_loo(out)
            if auc > best_ridge:
                best_ridge = auc; best_cfg_ridge = (name, alpha, wma)

print(f"  Ridge-win best: {best_ridge:.4f}  cfg={best_cfg_ridge}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_ridge_win'] = best_ridge
flag = " *** NEW BEST ***" if best_ridge > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# Also try: Ridge blend with WL contrast (UH triple)
print("  Ridge + WL-UH-triple 混合...", flush=True)
t0 = time.time()
best_ridge_blend = 0; best_cfg_ridge_blend = None

def wl_triple_score(fi):
    """Compute WL-UH-triple score for one file."""
    ws_ica = wl_contrast_single(ew_ica100, fi, ICA_K_NEG, ICA_WMP, ICA_WMA)
    ws_std = wl_contrast_single(ew_std80,  fi, STD_K_NEG, STD_WMP, STD_WMA)
    ws_pca = wl_contrast_single(ew_pca80,  fi, PCA_K_NEG, PCA_WMP, PCA_WMA)
    ws_blend = W_ICA * ws_ica + W_STD * ws_std + W_PCA * ws_pca
    return ICA_WMA * ws_blend.max(0) + (1 - ICA_WMA) * ws_blend.mean(0)

# Pre-compute WL triple scores for all files
wl_triple_all = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    wl_triple_all[fi] = wl_triple_score(fi)

# Best Ridge config from above
if best_cfg_ridge is not None:
    name_best, alpha_best, wma_best = best_cfg_ridge
    emb_best = {'ica100': ew_ica100, 'pca80': ew_pca80, 'std80': ew_std80}[name_best]
    ridge_all = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_best[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins = emb_best[tr_mask]
        tr_lab  = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            y_tr = (tr_lab[:, si] > 0.5).astype(np.float32)
            if y_tr.sum() < 1: ws[:, si] = 0.5; continue
            if y_tr.sum() == len(y_tr): ws[:, si] = 1.0; continue
            reg = Ridge(alpha=alpha_best, fit_intercept=True)
            reg.fit(tr_wins, y_tr)
            raw = reg.predict(te_wins)
            ws[:, si] = 1.0 / (1.0 + np.exp(-raw * 5.0))
        ridge_all[fi] = wma_best * ws.max(0) + (1 - wma_best) * ws.mean(0)

    for w_ridge in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        blend = (1 - w_ridge) * wl_triple_all + w_ridge * ridge_all
        auc = eval_loo(blend)
        if auc > best_ridge_blend:
            best_ridge_blend = auc; best_cfg_ridge_blend = w_ridge

print(f"  Ridge+WL blend: {best_ridge_blend:.4f}  cfg=w_ridge={best_cfg_ridge_blend}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_ridge_blend'] = best_ridge_blend
flag = " *** NEW BEST ***" if best_ridge_blend > CURRENT_BEST else ""
print(f"  {flag}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Method 3: PLDA-style (Probabilistic LDA) scoring
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: PLDA-style per-species discriminative scoring ===", flush=True)
print("理念：LDA 方向投影，per-species Fisher ratio scoring", flush=True)

def plda_style_loo(emb_wins_n, n_dim=32, w_max_agg=0.90):
    """
    PLDA-style: for each species, compute Fisher LDA direction using
    between-class and within-class scatter. Score test windows by projection.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te_wins = emb_wins_n[win_file_id == fi]
        tr_mask = win_file_id != fi
        tr_wins = emb_wins_n[tr_mask]
        tr_lab  = labels_win[tr_mask]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pm = tr_lab[:, si] > 0.5
            nm = tr_lab[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr_wins[pm]; neg = tr_wins[nm] if nm.any() else tr_wins[~pm]
            mu_p = pos.mean(0); mu_n = neg.mean(0)
            # Between-class scatter direction
            w_dir = mu_p - mu_n
            # Within-class scatter scaling (Tikhonov)
            S_w = np.cov(pos.T) + np.cov(neg.T) + 1e-4 * np.eye(pos.shape[1])
            try:
                w_lda = np.linalg.solve(S_w, w_dir)
            except Exception:
                w_lda = w_dir
            w_lda /= np.linalg.norm(w_lda) + EPS
            # Score: projection onto LDA direction
            proj_te = te_wins @ w_lda
            proj_p = pos @ w_lda; proj_n = neg @ w_lda
            mu_proj_p = proj_p.mean(); mu_proj_n = proj_n.mean()
            std_proj = np.sqrt(np.var(proj_p) + np.var(proj_n) + EPS)
            # Normalized score
            ws[:, si] = (proj_te - mu_proj_n) / (mu_proj_p - mu_proj_n + EPS * std_proj)
            ws[:, si] = np.clip(ws[:, si], 0, 1)
        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

t0 = time.time()
best_plda = 0; best_cfg_plda = None
for wma in [0.85, 0.88, 0.90, 0.92, 0.95]:
    for emb, name in [(ew_ica100, 'ica100'), (ew_pca80, 'pca80'), (ew_std80, 'std80')]:
        out = plda_style_loo(emb, w_max_agg=wma)
        auc = eval_loo(out)
        if auc > best_plda:
            best_plda = auc; best_cfg_plda = (name, wma)

print(f"  PLDA-style best: {best_plda:.4f}  cfg={best_cfg_plda}  ({time.time()-t0:.0f}s)", flush=True)
results['plda_style'] = best_plda
flag = " *** NEW BEST ***" if best_plda > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# PLDA blend with WL triple
t0 = time.time()
best_plda_blend = 0; best_cfg_plda_blend = None
if best_cfg_plda:
    name_p, wma_p = best_cfg_plda
    emb_p = {'ica100': ew_ica100, 'pca80': ew_pca80, 'std80': ew_std80}[name_p]
    plda_all = plda_style_loo(emb_p, w_max_agg=wma_p)
    for w_plda in [0.05, 0.10, 0.15, 0.20]:
        blend = (1 - w_plda) * wl_triple_all + w_plda * plda_all
        auc = eval_loo(blend)
        if auc > best_plda_blend:
            best_plda_blend = auc; best_cfg_plda_blend = w_plda

print(f"  PLDA+WL blend: {best_plda_blend:.4f}  cfg=w_plda={best_cfg_plda_blend}  ({time.time()-t0:.0f}s)", flush=True)
results['plda_wl_blend'] = best_plda_blend
flag = " *** NEW BEST ***" if best_plda_blend > CURRENT_BEST else ""
print(f"  {flag}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Batch 64 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
new_best_found = False
for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best_json:
        cur_best_json = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        new_best_found = True
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)

with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)

print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
if not new_best_found:
    print("未超越當前最佳 0.9873，已 append 到 experiments。", flush=True)
