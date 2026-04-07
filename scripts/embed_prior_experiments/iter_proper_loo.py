"""
Proper 2-level LOO for Iterative Refinement.

問題：naive iterative 的 step1_preds[j] 是用包含 file i 的訓練集算出的，
     所以 step2 的 KNN targets 實際上「知道」file i 的資訊 → data leakage。

修正：對每個 held-out file i，
  Step 1: 對每個 training file j，用「排除 i AND j」的資料算 logspace prediction
  Step 2: 用這些 proper step1 targets 對 file i 做 KNN

複雜度: O(n²) KNN = 66×65 次，仍然很快。

Current best: ls_lmx_a0.70_b1.50 = 0.9094
"""
import numpy as np, json, pickle, re, os, shutil
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

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

file_embs      = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels    = np.zeros((n_files, n_species), dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species), dtype=np.float32)

for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)

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

pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
X24  /= (X24.std(0) + 1e-6)
geo   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], axis=1).astype(np.float32)
X_comb = np.concatenate([X24, geo], axis=1).astype(np.float32)
X_nl   = (X_comb / np.linalg.norm(X_comb, axis=1, keepdims=True)).astype(np.float32)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS  = 1e-7
BEST = 0.909351
results = {}

print(f"Files={n_files}, species={n_species}", flush=True)
print("Running proper 2-level LOO iterative refinement...\n", flush=True)

# ── 預計算：普通 1-level LOO 的 nologit KNN 結果 (for baseline) ─────────────
# y_nl[i] = attn_knn 結果 (excluding file i)
y_nl_1level = np.zeros((n_files, n_species), dtype=np.float32)
for i in range(n_files):
    tr = np.array([j for j in range(n_files) if j != i])
    sims = (X_nl[[i]] @ X_nl[tr].T).ravel()
    top  = np.argsort(-sims)[:10]
    logit = sims[top] / 0.2; logit -= logit.max()
    w = np.exp(logit); w /= w.sum()
    y_nl_1level[i] = (w[:, None] * file_labels[tr[top]]).sum(0)

auc_1level = macro_auc(file_labels,
    _sigmoid(0.7 * file_logit_max + 1.5 * np.log(y_nl_1level.clip(EPS, 1-EPS))))
print(f"1-level logspace baseline: {auc_1level:.4f} (ref={BEST:.4f})", flush=True)

# ── Proper 2-level LOO ──────────────────────────────────────────────────────
def proper_iter_loo(a_step1=0.7, b_step1=1.5, a_final=0.7, b_final=1.5, k=10, T=0.2):
    """
    For each held-out file i:
      For each training file j (j ≠ i):
        Compute step1[j] = sigmoid(a1 × logit_max[j] + b1 × log(y_nl_{-i,-j}[j]))
        where y_nl_{-i,-j} = KNN of j excluding BOTH i and j
      Then:
        y_iter[i] = attn_knn of i using step1 targets (from tr files)
        final_score[i] = sigmoid(a2 × logit_max[i] + b2 × log(y_iter[i]))
    """
    preds = np.zeros((n_files, n_species), dtype=np.float32)

    for i in range(n_files):
        tr = [j for j in range(n_files) if j != i]   # training files (65)

        # Step 1: compute proper step1 targets for each training file j
        step1_targets = np.zeros((len(tr), n_species), dtype=np.float32)
        for ji, j in enumerate(tr):
            # Exclude both i and j from training set
            tr2 = np.array([k2 for k2 in tr if k2 != j])   # 64 files
            sims_j = (X_nl[[j]] @ X_nl[tr2].T).ravel()
            top_j  = np.argsort(-sims_j)[:k]
            logit_j = sims_j[top_j] / T; logit_j -= logit_j.max()
            w_j = np.exp(logit_j); w_j /= w_j.sum()
            y_nl_j = (w_j[:, None] * file_labels[tr2[top_j]]).sum(0)
            log_nl_j = np.log(y_nl_j.clip(EPS, 1-EPS))
            step1_targets[ji] = _sigmoid(a_step1 * file_logit_max[j] + b_step1 * log_nl_j)

        # Step 2: KNN of file i using step1_targets
        tr_arr = np.array(tr)
        sims_i = (X_nl[[i]] @ X_nl[tr_arr].T).ravel()
        top_i  = np.argsort(-sims_i)[:k]
        logit_i = sims_i[top_i] / T; logit_i -= logit_i.max()
        w_i = np.exp(logit_i); w_i /= w_i.sum()
        y_iter_i = (w_i[:, None] * step1_targets[top_i]).sum(0)

        # Final prediction
        log_iter_i = np.log(y_iter_i.clip(EPS, 1-EPS))
        preds[i] = _sigmoid(a_final * file_logit_max[i] + b_final * log_iter_i)

        if (i + 1) % 20 == 0:
            print(f"  fold {i+1}/{n_files} done", flush=True)

    return preds

# ── A) Sweep 最終 a, b（step1 固定用最佳 0.7/1.5）──────────────────────────
print("\n" + "="*60)
print("A) Proper 2-level LOO: step1=(0.7,1.5), sweep final (a,b)")
print("="*60, flush=True)

best_so_far = BEST
for a_final in [0.5, 0.6, 0.7, 0.75, 0.8]:
    for b_final in [1.0, 1.3, 1.5, 1.7, 2.0]:
        p = proper_iter_loo(a_step1=0.7, b_step1=1.5,
                            a_final=a_final, b_final=b_final)
        auc = macro_auc(file_labels, p)
        nm = f'proper_iter_s07_15_f{a_final:.2f}_{b_final:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_preds  = p.copy()
            best_nm     = nm
        print(f"  a={a_final:.2f} b={b_final:.2f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

print(f"\nAfter A, best so far: {best_so_far:.4f}", flush=True)

# ── B) 也嘗試 step1 用不同參數 ──────────────────────────────────────────────
print("\n" + "="*60)
print("B) Vary step1 params: a1=0.5, b1=2.0 (softer step1)")
print("="*60, flush=True)

for a_final in [0.5, 0.7]:
    for b_final in [1.5, 2.0]:
        p = proper_iter_loo(a_step1=0.5, b_step1=2.0,
                            a_final=a_final, b_final=b_final)
        auc = macro_auc(file_labels, p)
        nm = f'proper_iter_s05_20_f{a_final:.2f}_{b_final:.2f}'
        marker = " ← NEW BEST" if auc > best_so_far else ""
        if auc > best_so_far:
            best_so_far = auc
            best_preds  = p.copy()
            best_nm     = nm
        print(f"  step1=(0.5,2.0) final a={a_final:.2f} b={b_final:.2f}: {auc:.4f}  (Δ={auc-BEST:+.4f}){marker}", flush=True)
        results[nm] = auc

# ── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (top 10)")
print("="*60)
for nm, auc in sorted(results.items(), key=lambda x: -x[1])[:10]:
    marker = " ← NEW BEST" if auc > BEST else ""
    print(f"  {nm:<60s}  {auc:.4f}  {auc-BEST:+.4f}{marker}")

global_best_name = max(results, key=results.get)
global_best_auc  = results[global_best_name]
print(f"\n整體最佳: {global_best_name} = {global_best_auc:.4f}", flush=True)

# ── Update results.json ────────────────────────────────────────────────────
with open("outputs/embed_prior_results.json") as f:
    data = json.load(f)

# 移除之前 data leakage 的 iter_ 方法
data['experiments'] = [e for e in data['experiments']
                        if not e['method'].startswith('iter_')]
# 修正 best（如果當前 best 是 iter_ 的，回滾到 logspace best）
if data['best']['method'].startswith('iter_'):
    data['best'] = {'method': 'ls_lmx_a0.70_b1.50',
                    'loo_auc': 0.909351,
                    'note': 'logspace_finetune (restored after leakage fix)'}

cur_best = data['best']['loo_auc']
new_best_found = False

for nm, auc in results.items():
    data['experiments'].append({'method': nm, 'loo_auc': round(auc, 6),
                                'note': 'proper_2level_iter'})
    if auc > cur_best:
        cur_best = auc
        data['best'] = {'method': nm, 'loo_auc': round(auc, 6),
                        'note': 'NEW BEST proper_iter'}
        new_best_found = True

with open("outputs/embed_prior_results.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\n已更新 embed_prior_results.json（已清除 data leakage 的 iter_ 結果）")
print(f"當前 best: {data['best']['method']} = {data['best']['loo_auc']:.4f}")

if new_best_found:
    print(f"\nNEW BEST: {global_best_name} AUC={global_best_auc:.4f}")
    # Save pkl
    pkl_data = {
        'method': global_best_name,
        'loo_auc': round(global_best_auc, 6),
        'type': 'proper_iter_logspace',
        'note': '2-level LOO iterative refinement',
        'pca_dims': 24,
        'pca_mean': pca24.mean_.astype(np.float32),
        'pca_components': pca24.components_.astype(np.float32),
        'pca_std': (X24.std(0) + 1e-6).astype(np.float32),
        'use_day': True,
        'SITES': SITES,
        'site2idx': site2idx,
        'X_combined_n': X_nl,
        'file_labels': file_labels,
        'file_logit_max': file_logit_max,
        'file_list': file_list,
        'k': 10,
        'T': 0.2,
        'temperature': 0.2,
        'a_step1': 0.7, 'b_step1': 1.5,
    }
    with open("outputs/embed_prior_iter.pkl", "wb") as f:
        pickle.dump(pkl_data, f)
    shutil.copy("outputs/embed_prior_iter.pkl",
                "birdclef-2026/notebook resource/current_subs/weights/embed_prior_iter.pkl")
    print(f"Saved: outputs/embed_prior_iter.pkl")
else:
    print(f"\n未超越 best={BEST:.4f}，不儲存 pkl")

print("done", flush=True)
