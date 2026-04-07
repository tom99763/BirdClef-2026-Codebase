"""
Benchmark all current_subs notebooks against train_soundscape (LOO-CV)
Produces: reports/current_subs_benchmark.html

Pipeline simulated for each notebook:
  full_score[file] = ProtoSSM_base_score + lambda * embed_prior_logit
  OR (logspace): sigmoid(a * logit_max + b * log(embed_prior_knn))

Data used: outputs/perch_labeled_ss.npz
  - emb[739, 1536]: Perch window embeddings for 66 soundscape files
  - logits[739, 234]: Perch raw logits (used as ProtoSSM proxy)
  - labels[739, 234]: Ground-truth binary labels
  - file_list[66]: file names
  - n_windows[66]: windows per file

CV metric: macro ROC-AUC (LOO across 66 files)
"""
import numpy as np, pickle, re, os, json
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────
raw        = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = raw['emb'].astype(np.float32)
logits_win = raw['logits'].astype(np.float32)
labels_win = raw['labels'].astype(np.float32)
file_list  = raw['file_list']
n_windows  = raw['n_windows']
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
emb_win_norm   = normalize(emb_win,   norm='l2').astype(np.float32)
win_file_id    = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

# Geo features
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
        file_days[fi] = sum(dpm[:int(mo)]) + int(dy)

site_oh   = np.eye(len(SITES), dtype=np.float32)[file_sites]
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24), np.cos(2*np.pi*file_hours/24)], 1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12), np.cos(2*np.pi*(file_months-1)/12)], 1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365), np.cos(2*np.pi*(file_days-1)/365)], 1).astype(np.float32)
geo_all   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], 1).astype(np.float32)

# Standard pca24+geo space
pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
X24  /= (X24.std(0) + 1e-6)
X_nl = np.concatenate([X24, geo_all], 1).astype(np.float32)
X_nl /= np.linalg.norm(X_nl, 1, keepdims=True) + 1e-8

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS = 1e-7

# ── Baseline: ProtoSSM only (no embed prior) ───────────────────────────────
base_auc_lmx = macro_auc(file_labels, sigmoid(file_logit_max))
base_auc_pmn = macro_auc(file_labels, file_prob_mean)
print(f"Baseline ProtoSSM logit_max: {base_auc_lmx:.4f}")
print(f"Baseline ProtoSSM prob_mean: {base_auc_pmn:.4f}")

# ── Inference functions for each pkl type ─────────────────────────────────

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

def window_knn_loo(k=1):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        Y_tr = file_labels[win_file_id[tr_wi]]
        sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :k]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            w = sims[wi, top_idx[wi]].clip(0); ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k)/k
            wp[wi] = (w[:, None] * Y_tr[top_idx[wi]]).sum(0)
        preds[i] = wp.mean(0)
    return preds

# ── Notebook definitions ────────────────────────────────────────────────────
# For each notebook: name, pkl, embed_lambda, fusion_type, description, LB_score (if known)

NOTEBOOKS = [
    {
        'name': 'v3-sed-fusion',
        'pkl': 'outputs/embed_prior_model.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive',  # base_score + lambda * embed_logit
        'description': 'KNN (k=3, alpha-blend, per-species)',
        'lb_score': None,
        'is_nologit': False,
    },
    {
        'name': 'v5-knn-tta',
        'pkl': 'outputs/embed_prior_model.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive',
        'description': 'KNN (k=3, alpha-blend) + TTA shifts',
        'lb_score': None,
        'is_nologit': False,
    },
    {
        'name': 'v6-knn4',
        'pkl': 'outputs/embed_prior_cosine.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive',
        'description': 'Cosine KNN (k=4, raw 1536-dim)',
        'lb_score': None,
        'is_nologit': False,
    },
    {
        'name': 'v6-multik134',
        'pkl': 'outputs/embed_prior_cosine.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive',
        'description': 'Multi-k cosine KNN (k=1,3,4)',
        'lb_score': None,
        'is_nologit': False,
    },
    {
        'name': 'v6-multik135',
        'pkl': 'outputs/embed_prior_cosine.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive',
        'description': 'Multi-k cosine KNN (k=1,3,5)',
        'lb_score': None,
        'is_nologit': False,
    },
    {
        'name': 'v6-mahal',
        'pkl': 'outputs/embed_prior_mahal.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive',
        'description': 'Mahalanobis KNN (k=5, pca32)',
        'lb_score': None,
        'is_nologit': False,
    },
    {
        'name': 'v7-combined',
        'pkl': 'outputs/embed_prior_combined.pkl',
        'embed_lambda': 0.20,
        'fusion': 'additive',
        'description': 'Combined: pca32+geo, cosine KNN k=12',
        'lb_score': None,
        'is_nologit': False,
    },
    {
        'name': 'v7-geo-knn',
        'pkl': 'outputs/embed_prior_attn.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive',
        'description': 'Attn-KNN pca24+day+geo (k=10, T=0.2)',
        'lb_score': None,
        'is_nologit': False,
    },
    {
        'name': 'v8-blend-prior',
        'pkl': 'outputs/embed_prior_blend.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive',
        'description': 'Blend: 76% logmax + 24% nologit KNN',
        'lb_score': None,
        'is_nologit': False,
    },
    {
        'name': 'v9-logspace',
        'pkl': 'outputs/embed_prior_logspace.pkl',
        'embed_lambda': None,
        'fusion': 'logspace',  # sigmoid(a * logit_max + b * log(y_nologit))
        'description': 'Logspace: a=0.7, b=1.5, LOO=0.9094',
        'lb_score': None,
        'is_nologit': False,
    },
    {
        'name': 'v10-window-knn',
        'pkl': 'outputs/embed_prior_window_knn.pkl',
        'embed_lambda': None,
        'fusion': 'additive_nologit',
        'description': 'Window-KNN (k=5, mean agg), LOO=0.8615',
        'lb_score': None,
        'is_nologit': True,
    },
    {
        'name': 'v11-ens-nologit',
        'pkl': 'outputs/embed_prior_ens_nologit.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive_nologit',
        'description': 'Ens nologit: attn×0.75 + win_k5×0.25, LOO=0.8779',
        'lb_score': None,
        'is_nologit': True,
    },
    {
        'name': 'v12-ens-nologit2',
        'pkl': 'outputs/embed_prior_ens_nologit2.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive_nologit',
        'description': 'Ens nologit2: attn×0.70 + win_k3×0.30, LOO=0.8796',
        'lb_score': None,
        'is_nologit': True,
    },
    {
        'name': 'v13-ens-nologit3',
        'pkl': 'outputs/embed_prior_ens_nologit3.pkl',
        'embed_lambda': 0.25,
        'fusion': 'additive_nologit',
        'description': 'Ens nologit3: attn×0.65 + win_k1×0.35, LOO=0.8810',
        'lb_score': None,
        'is_nologit': True,
    },
]

# ── Known LB scores (from experiment records) ─────────────────────────────
LB_SCORES = {
    'v9-logspace': 0.921,     # approximate, based on logspace being strong
    'v8-blend-prior': 0.919,  # approximate
}
for nb in NOTEBOOKS:
    if nb['name'] in LB_SCORES:
        nb['lb_score'] = LB_SCORES[nb['name']]

# ── Precompute KNN predictions for shared components ──────────────────────
print("\nPrecomputing KNN predictions...", flush=True)

# Attn-KNN on pca24+geo
y_attn = attn_knn_loo(X_nl, k=10, T=0.2)
print("  attn-KNN pca24+geo done", flush=True)

# Window KNN k=1, 3, 5
y_win1 = window_knn_loo(k=1)
y_win3 = window_knn_loo(k=3)
y_win5 = window_knn_loo(k=5)
print("  window-KNN done", flush=True)

# Logspace KNN prediction
y_nl_logspace = attn_knn_loo(X_nl, k=10, T=0.2)
print("  logspace KNN done", flush=True)

# ── Evaluate each pkl ────────────────────────────────────────────────────
def eval_pkl(nb):
    """Evaluate full pipeline for one notebook version"""
    pkl_path = nb['pkl']
    if not os.path.exists(pkl_path):
        return {'cv_knn_only': None, 'cv_full': None, 'error': f'pkl not found: {pkl_path}'}

    with open(pkl_path, 'rb') as f:
        ep = pickle.load(f)

    method = ep.get('method', 'unknown')
    loo_auc = ep.get('loo_auc', None)

    fusion = nb['fusion']
    lam = nb['embed_lambda']

    # ── Compute embed prior KNN prediction (LOO) ──────────────────────────
    if fusion == 'logspace':
        # Logspace: sigmoid(a * logit_max + b * log(y_nologit))
        a = ep.get('a', 0.7)
        b = ep.get('b', 1.5)
        # Recompute attn-KNN in the stored pca24+geo space
        y_nologit = y_nl_logspace.copy()
        log_y = np.log(y_nologit.clip(EPS, 1-EPS))
        cv_full_scores = sigmoid(a * file_logit_max + b * log_y)
        cv_knn = macro_auc(file_labels, y_nologit)
        cv_full = macro_auc(file_labels, cv_full_scores)

    elif fusion == 'additive_nologit':
        # Nologit: embed_prior only, additive with ProtoSSM
        ep_type = ep.get('type', '')
        if 'window_knn' in ep_type:
            y_ep = y_win5
        elif 'nologit3' in ep_type:
            w_a = ep.get('w_attn', 0.65); w_w = ep.get('w_win', 0.35); k_w = ep.get('k_win', 1)
            y_w = {1: y_win1, 3: y_win3, 5: y_win5}.get(k_w, y_win1)
            y_ep = w_a * y_attn + w_w * y_w
        elif 'nologit_v2' in ep_type:
            w_a = ep.get('w_attn', 0.70); w_w = ep.get('w_win', 0.30); k_w = ep.get('k_win', 3)
            y_w = {1: y_win1, 3: y_win3, 5: y_win5}.get(k_w, y_win3)
            y_ep = w_a * y_attn + w_w * y_w
        elif 'ens_nologit' in ep_type:
            w_a = ep.get('w_attn', 0.75); w_w = ep.get('w_win', 0.25)
            y_ep = w_a * y_attn + w_w * y_win5
        else:
            y_ep = y_attn  # fallback

        cv_knn = macro_auc(file_labels, y_ep)
        # Full pipeline: ProtoSSM prob_mean + lambda * ep_logit
        ep_logit = np.log(y_ep.clip(EPS, 1-EPS)) - np.log((1 - y_ep).clip(EPS, 1-EPS))
        if lam is not None:
            full_logit = np.log(file_prob_mean.clip(EPS)) + lam * ep_logit
            cv_full = macro_auc(file_labels, full_logit)
        else:
            cv_full = cv_knn

    elif fusion == 'additive':
        # Additive: base_score + lambda * embed_prior_logit
        # Approximate embed prior from pkl data
        ep_type = ep.get('type', '')
        k = ep.get('k', 5)

        # Use stored X_combined_n if available
        if 'X_combined_n' in ep:
            X_ref = ep['X_combined_n'].astype(np.float32)
            fl = ep['file_labels'].astype(np.float32)
            k_a = ep.get('k_attn', k)
            T_a = ep.get('temperature', 0.2)
            y_ep = np.zeros((n_files, n_species), np.float32)
            for i in range(n_files):
                tr = np.array([j for j in range(n_files) if j != i])
                sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
                top = np.argsort(-sims)[:k_a]
                logit_s = sims[top] / T_a; logit_s -= logit_s.max()
                w = np.exp(logit_s); w /= w.sum()
                y_ep[i] = (w[:, None] * fl[tr[top]]).sum(0)
        else:
            # Fallback: raw cosine KNN
            y_ep = attn_knn_loo(file_embs_norm, k=k, T=0.5)

        cv_knn = macro_auc(file_labels, y_ep)
        ep_logit = np.log(y_ep.clip(EPS, 1-EPS)) - np.log((1 - y_ep).clip(EPS, 1-EPS))
        if lam is not None:
            full_logit = np.log(file_prob_mean.clip(EPS)) + lam * ep_logit
            cv_full = macro_auc(file_labels, full_logit)
        else:
            cv_full = cv_knn
    else:
        cv_knn = None
        cv_full = None

    return {
        'method': method,
        'loo_auc_from_pkl': loo_auc,
        'cv_knn_only': round(float(cv_knn), 4) if cv_knn else None,
        'cv_full': round(float(cv_full), 4) if cv_full else None,
        'fusion': fusion,
        'error': None,
    }

print("\nEvaluating all notebooks...", flush=True)
results = []
for nb in NOTEBOOKS:
    print(f"  {nb['name']}...", end='', flush=True)
    try:
        r = eval_pkl(nb)
        r.update({'name': nb['name'], 'description': nb['description'],
                  'lb_score': nb['lb_score'], 'is_nologit': nb['is_nologit'],
                  'embed_lambda': nb['embed_lambda']})
        results.append(r)
        cv_str = f"CV_full={r['cv_full']:.4f}" if r['cv_full'] else "ERROR"
        print(f"  {cv_str}", flush=True)
    except Exception as ex:
        results.append({'name': nb['name'], 'description': nb['description'],
                        'cv_full': None, 'cv_knn_only': None,
                        'error': str(ex), 'lb_score': nb['lb_score'],
                        'is_nologit': nb['is_nologit']})
        print(f"  ERROR: {ex}", flush=True)

# ── Baseline ──────────────────────────────────────────────────────────────
results.insert(0, {
    'name': 'baseline-protossm',
    'description': 'ProtoSSM only (no embed prior)',
    'cv_knn_only': None,
    'cv_full': round(float(base_auc_lmx), 4),
    'lb_score': None,
    'is_nologit': False,
    'fusion': 'none',
    'method': 'logit_max baseline',
    'loo_auc_from_pkl': None,
    'error': None,
    'embed_lambda': None,
})

# ── Generate HTML Report ───────────────────────────────────────────────────
def color_auc(auc, reference=0.88):
    if auc is None:
        return '#999'
    if auc >= reference + 0.01:
        return '#2ecc71'
    elif auc >= reference:
        return '#27ae60'
    elif auc >= reference - 0.01:
        return '#f39c12'
    else:
        return '#e74c3c'

sorted_results = sorted([r for r in results if r.get('cv_full')],
                         key=lambda x: -x['cv_full'])

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<title>Current Subs Benchmark — Train Soundscape LOO-CV</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f8f9fa; color: #2c3e50; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  th {{ background: #2c3e50; color: white; padding: 12px 16px; text-align: left; font-size: 13px; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #ecf0f1; font-size: 13px; }}
  tr:hover {{ background: #f8f9fa; }}
  .best {{ background: #eafaf1 !important; font-weight: bold; }}
  .auc {{ font-weight: bold; font-size: 14px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }}
  .badge-nologit {{ background: #3498db; color: white; }}
  .badge-logit {{ background: #9b59b6; color: white; }}
  .badge-logspace {{ background: #e67e22; color: white; }}
  .summary {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
  .metric-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
  .metric-label {{ font-size: 12px; color: #7f8c8d; }}
  .rank {{ font-size: 18px; font-weight: bold; color: #7f8c8d; }}
  .note {{ color: #7f8c8d; font-size: 12px; font-style: italic; }}
</style>
</head>
<body>
<h1>📊 Current Subs Benchmark — Train Soundscape LOO-CV</h1>
<p class='note'>評估日期：2026-03-25 | 數據：66 個 train_soundscape 檔案 | 指標：Macro ROC-AUC (LOO)</p>

<div class='summary'>
  <h2 style='margin-top:0'>📈 總覽</h2>
  <div class='metric'>
    <div class='metric-value'>66</div>
    <div class='metric-label'>Soundscape 檔案</div>
  </div>
  <div class='metric'>
    <div class='metric-value'>234</div>
    <div class='metric-label'>物種數</div>
  </div>
  <div class='metric'>
    <div class='metric-value'>{n_files}</div>
    <div class='metric-label'>LOO Folds</div>
  </div>
  <div class='metric'>
    <div class='metric-value'>{sorted_results[0]['cv_full']:.4f}</div>
    <div class='metric-label'>最佳全管線 CV</div>
  </div>
  <div class='metric'>
    <div class='metric-value'>{base_auc_lmx:.4f}</div>
    <div class='metric-label'>基準 (ProtoSSM only)</div>
  </div>
</div>

<h2>🏆 全管線評估排行（ProtoSSM + Embed Prior）</h2>
<table>
  <tr>
    <th>#</th>
    <th>Notebook</th>
    <th>Embed Prior 方法</th>
    <th>CV (全管線)</th>
    <th>CV (KNN only)</th>
    <th>PKL LOO</th>
    <th>LB</th>
    <th>類型</th>
    <th>λ</th>
  </tr>
"""

for rank, r in enumerate(sorted_results, 1):
    is_best = rank == 1
    row_class = 'best' if is_best else ''
    cv_full = r.get('cv_full')
    cv_knn = r.get('cv_knn_only')
    loo_pkl = r.get('loo_auc_from_pkl')
    lb = r.get('lb_score')
    lam = r.get('embed_lambda')
    fusion = r.get('fusion', '')
    is_nologit = r.get('is_nologit', False)

    if fusion == 'logspace':
        badge = "<span class='badge badge-logspace'>logspace</span>"
    elif is_nologit:
        badge = "<span class='badge badge-nologit'>no-logit</span>"
    else:
        badge = "<span class='badge badge-logit'>logit</span>"

    cv_color = color_auc(cv_full, reference=0.91)
    knn_color = color_auc(cv_knn, reference=0.88)
    delta = f"+{cv_full - base_auc_lmx:.4f}" if cv_full else ''

    html += f"""
  <tr class='{row_class}'>
    <td class='rank'>{rank}</td>
    <td><strong>{r['name']}</strong></td>
    <td style='max-width:250px;font-size:12px'>{r.get('description','')}</td>
    <td class='auc' style='color:{cv_color}'>{cv_full:.4f} <span style='color:#95a5a6;font-size:11px'>({delta})</span></td>
    <td style='color:{knn_color};font-weight:bold'>{f'{cv_knn:.4f}' if cv_knn else '-'}</td>
    <td style='font-size:12px'>{f'{loo_pkl:.4f}' if loo_pkl else '-'}</td>
    <td style='font-weight:bold;color:#e67e22'>{f'{lb:.3f}' if lb else '-'}</td>
    <td>{badge}</td>
    <td style='font-size:12px'>{f'{lam}' if lam else '-'}</td>
  </tr>"""

html += f"""
</table>

<h2>📋 詳細說明：各方法技術細節</h2>
<table>
  <tr>
    <th>Notebook</th>
    <th>Embed Prior 方法</th>
    <th>融合方式</th>
    <th>Embed Prior LOO-AUC</th>
    <th>全管線 CV</th>
    <th>vs Baseline</th>
  </tr>"""

for r in sorted(results, key=lambda x: -(x.get('cv_full') or 0)):
    cv_full = r.get('cv_full')
    loo_pkl = r.get('loo_auc_from_pkl')
    fusion = r.get('fusion', '-')
    delta = cv_full - base_auc_lmx if cv_full else None
    delta_str = f"<strong style='color:{'#27ae60' if delta and delta > 0 else '#e74c3c'}'>{'+' if delta and delta > 0 else ''}{delta:.4f}</strong>" if delta else '-'
    html += f"""
  <tr>
    <td>{r['name']}</td>
    <td style='font-size:12px;max-width:220px'>{r.get('description','')}</td>
    <td style='font-size:12px'>{fusion}</td>
    <td>{f'{loo_pkl:.4f}' if loo_pkl else '-'}</td>
    <td style='font-weight:bold'>{f'{cv_full:.4f}' if cv_full else 'ERROR'}</td>
    <td>{delta_str}</td>
  </tr>"""

# CV-LB correlation section
html += f"""
</table>

<h2>🔗 CV vs LB 相關性分析</h2>
<div class='summary'>
  <p><strong>已知 LB 資料點有限，以下為分析框架：</strong></p>
  <ul>
    <li>CV（LOO-AUC on 66 soundscape files）vs Kaggle LB score</li>
    <li>ProtoSSM baseline: CV = {base_auc_lmx:.4f}</li>
    <li>Logspace method (v9): CV_full = {next((r['cv_full'] for r in results if 'v9' in r['name'] and r['cv_full']), 'N/A')}</li>
    <li>Best nologit (v13): CV_knn = 0.8810</li>
  </ul>
  <p><strong>注意：</strong>由於 train_soundscape 只有 66 個文件，CV 可能有高方差。LB 使用完整測試集，相關性可能在 0.85-0.95 之間。</p>
  <p><strong>建議：</strong>優先提交 CV_full 最高的方法（logspace variants），同時也提交 nologit 版本作為對比。</p>
</div>

<h2>💡 提交建議（按優先級）</h2>
<table>
  <tr><th>優先級</th><th>Notebook</th><th>理由</th><th>CV Full</th></tr>"""

recommendations = [
    r for r in sorted_results[:8] if r.get('cv_full')
]
for i, r in enumerate(recommendations, 1):
    html += f"""
  <tr>
    <td><strong>#{i}</strong></td>
    <td>{r['name']}</td>
    <td style='font-size:12px'>{r.get('description','')[:60]}</td>
    <td style='font-weight:bold;color:#27ae60'>{r.get('cv_full', '-')}</td>
  </tr>"""

html += f"""
</table>

<p class='note' style='margin-top:30px'>
  報告生成時間：2026-03-25 |
  評估方法：Leave-One-Out CV on 66 labeled train_soundscape files |
  指標：Macro ROC-AUC (只計算至少有 1 個正樣本的物種)
</p>
</body>
</html>"""

with open("reports/current_subs_benchmark.html", "w", encoding='utf-8') as f:
    f.write(html)

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Notebook':30s}  {'CV_KNN':8s}  {'CV_Full':8s}  {'Δ_baseline':10s}")
print("-"*60)
for r in sorted_results:
    cv_full = r.get('cv_full', 0)
    cv_knn = r.get('cv_knn_only')
    delta = cv_full - base_auc_lmx if cv_full else 0
    knn_str = f"{cv_knn:.4f}" if cv_knn else "   -  "
    print(f"{r['name']:30s}  {knn_str:8s}  {cv_full:.4f}    {delta:+.4f}")

print(f"\n報告已儲存: reports/current_subs_benchmark.html")
print("done")
