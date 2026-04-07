"""
在對數空間中線性組合 raw logit 與 nologit：
score[i,s] = a × logit_max[i,s] + b × log(y_nologit[i,s])
             → final prob = sigmoid(a × logit_max + b × log_nl)

這等同於：prob = sigmoid(logit) × nologit^b (calibrated)
不同於幾何平均（後者在 prob 空間）

Another approach: per-window max logit aggregation variants
"""
import numpy as np, json, re, os
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

file_embs         = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels       = np.zeros((n_files, n_species), dtype=np.float32)
file_logit_max    = np.zeros((n_files, n_species), dtype=np.float32)  # raw max logit
file_logit_p90    = np.zeros((n_files, n_species), dtype=np.float32)  # raw p90 logit
file_logit_mean   = np.zeros((n_files, n_species), dtype=np.float32)  # raw mean logit
file_logit_sum    = np.zeros((n_files, n_species), dtype=np.float32)  # raw sum logit
file_prob_max     = np.zeros((n_files, n_species), dtype=np.float32)
file_prob_mean    = np.zeros((n_files, n_species), dtype=np.float32)
file_detect_frac  = np.zeros((n_files, n_species), dtype=np.float32)  # fraction windows > 0

for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]   = emb_win[s:e].mean(0)
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    lb = logits_win[s:e]              # (nw, 234)
    file_logit_max[fi]  = lb.max(0)
    file_logit_p90[fi]  = np.percentile(lb, 90, axis=0)
    file_logit_mean[fi] = lb.mean(0)
    file_logit_sum[fi]  = lb.sum(0)
    file_prob_max[fi]   = _sigmoid(lb.max(0))
    file_prob_mean[fi]  = _sigmoid(lb.mean(0))
    file_detect_frac[fi]= (lb > 0).mean(0).astype(np.float32)

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

pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
X24  /= (X24.std(0) + 1e-6)
geo   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], axis=1).astype(np.float32)
X_combined = np.concatenate([X24, geo], axis=1).astype(np.float32)
X_nl = (X_combined / np.linalg.norm(X_combined, axis=1, keepdims=True)).astype(np.float32)

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

BEST    = 0.905463
BEST_NL = 0.875789
EPS     = 1e-7
results = {}

print(f"Files={n_files}, species={n_species}", flush=True)

y_nl = attn_knn_loo(X_nl, k=10, T=0.2)
print(f"nologit base: {macro_auc(file_labels, y_nl):.4f}", flush=True)

# ── A) 對數空間線性組合: a×logit_max + b×log(y_nl) ────────────────────────
print("\n" + "="*60)
print("A) Log-space: a×logit_max + b×log(y_nl)")
print("="*60, flush=True)

log_nl = np.log(y_nl.clip(EPS, 1-EPS))
best_so_far = BEST

for a in [0.5, 0.7, 1.0, 1.2, 1.5]:
    for b in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        score = a * file_logit_max + b * log_nl
        # Convert to prob space for AUC
        prob  = _sigmoid(score)
        auc   = macro_auc(file_labels, prob)
        nm = f'logspace_a{a:.1f}_b{b:.1f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        print(f"  a={a:.1f} b={b:.1f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

# ── B) 用 logit_max 加上 detection fraction + nologit ────────────────────
print("\n" + "="*60)
print("B) logit_max + detection_frac + log(y_nl)")
print("="*60, flush=True)

log_df = np.log(file_detect_frac.clip(EPS, 1-EPS))
for a in [1.0, 1.2]:
    for c in [0.5, 1.0, 2.0]:
        for b in [0.5, 1.0]:
            score = a * file_logit_max + c * log_df + b * log_nl
            prob  = _sigmoid(score)
            auc   = macro_auc(file_labels, prob)
            nm = f'ldf_a{a:.1f}_c{c:.1f}_b{b:.1f}'
            marker = " ← NEW BEST" if auc > best_so_far else ""
            if auc > best_so_far: best_so_far = auc
            print(f"  a={a:.1f} c={c:.1f} b={b:.1f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
            results[nm] = auc

# ── C) 最佳 geo_mean pw 精調（用 raw logit → sigmoid） ─────────────────────
print("\n" + "="*60)
print("C) geo_mean精調: prob_max^(1-pw) × y_nl^pw, fine pw sweep")
print("="*60, flush=True)

for pw in np.arange(0.10, 0.55, 0.02):
    blend = (file_prob_max.clip(EPS, 1-EPS) ** (1-pw)) * (y_nl.clip(EPS, 1-EPS) ** pw)
    auc   = macro_auc(file_labels, blend)
    nm = f'geo_pmx_pw{pw:.2f}'
    marker = " ← NEW BEST" if auc > best_so_far else ""
    if auc > best_so_far: best_so_far = auc
    if pw in [0.10, 0.20, 0.30, 0.40, 0.50] or auc > BEST - 0.005:
        print(f"  pw={pw:.2f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
    results[nm] = auc

# ── D) prob_max × detect_frac → geo_mean with nologit ────────────────────
print("\n" + "="*60)
print("D) combined logit feature × nologit geo_mean")
print("="*60, flush=True)

# pmx × detect_frac^gamma
for gamma in [0.5, 1.0, 2.0]:
    y_combined = file_prob_max * (file_detect_frac.clip(EPS) ** gamma)
    for pw in [0.20, 0.25, 0.30]:
        blend = (y_combined.clip(EPS, 1-EPS) ** (1-pw)) * (y_nl.clip(EPS, 1-EPS) ** pw)
        auc = macro_auc(file_labels, blend)
        nm = f'geo_pmx_df{gamma:.1f}_pw{pw:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far: best_so_far = auc
        print(f"  pmx×df^{gamma:.1f} pw={pw:.2f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

# ── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (top 15)")
print("="*60)
for nm, auc in sorted(results.items(), key=lambda x: -x[1])[:15]:
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {nm:<50s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\nOverall best: {global_best_name} = {global_best_auc:.4f}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

cur_best = data['best']['loo_auc']
for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6), 'note': 'logspace_geo'})
    if auc > cur_best:
        cur_best = auc
        data['best'] = {'method': nm, 'loo_auc': round(auc, 6), 'note': 'NEW BEST logspace'}

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\nbest: {data['best']['method']} = {data['best']['loo_auc']:.4f}")
print("done", flush=True)
