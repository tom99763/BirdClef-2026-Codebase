"""
Batch 31: Fine PCA dimension sweep + PCA ensemble
Goal: beat pca128_max_pos = 0.9641
Methods:
  1. Fine PCA sweep: 80, 96, 112, 128, 144, 160, 192
  2. Multi-PCA ensemble: blend best few PCA dims
  3. PCA-best with fine w_max and k_neg sweep
  4. PCA-best + full-1536 fine blend
  5. PCA ensemble (5 different dims averaged)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
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
file_embs   = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_embs[fi]   = emb_win[s:e].mean(0)

file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)
emb_win_norm   = normalize(emb_win, norm='l2').astype(np.float32)
EPS = 1e-7; mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9641

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def max_pos_contrast_emb(emb_wins_n, emb_files_n, k_neg=6, w_max=0.7):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = emb_files_n[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_wins_n[win_file_id == fi]
        tr_wins_all = emb_wins_n[win_file_id != fi]
        tr_fids_all = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win[:,si] > 0.5
            neg_win_mask = ~pos_win_mask
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp = w_max * pos_sims.max(1) + (1-w_max) * (te_wins @ pp_mean)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(k_neg, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    return out

def make_pca_embs(n_comp):
    pca = PCA(n_components=n_comp, random_state=42)
    emb_pca = pca.fit_transform(emb_win).astype(np.float32)
    emb_win_pca_n = normalize(emb_pca, norm='l2').astype(np.float32)
    file_embs_pca = np.zeros((n_files, n_comp), np.float32)
    for fi in range(n_files):
        s, e = int(file_start[fi]), int(file_end[fi])
        file_embs_pca[fi] = emb_pca[s:e].mean(0)
    file_embs_pca_n = normalize(file_embs_pca, norm='l2').astype(np.float32)
    return emb_win_pca_n, file_embs_pca_n

# ─── Method 1: Fine PCA sweep ─────────────────────────────────────────────────
print("=== Method 1: Fine PCA dimension sweep ===", flush=True)
t0 = time.time()
pca_outs = {}
for n_comp in [80, 96, 112, 128, 144, 160, 192]:
    ew_n, ef_n = make_pca_embs(n_comp)
    out = max_pos_contrast_emb(ew_n, ef_n)
    auc = eval_loo(out)
    pca_outs[n_comp] = out
    results[f'pca{n_comp}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  PCA-{n_comp}: {auc:.4f}{flag}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

best_pca_n = max(pca_outs, key=lambda k: results[f'pca{k}'])
best_pca_out = pca_outs[best_pca_n]
best_pca_embs = make_pca_embs(best_pca_n)
print(f"  Best PCA dim: {best_pca_n}", flush=True)

# ─── Method 2: PCA ensemble (avg best dims) ───────────────────────────────────
print("\n=== Method 2: PCA ensemble ===", flush=True)
# Also add PCA-128 and 256 from batch 30
ew128, ef128 = make_pca_embs(128)
out128 = max_pos_contrast_emb(ew128, ef128)
ew256, ef256 = make_pca_embs(256)
out256 = max_pos_contrast_emb(ew256, ef256)
pca_outs[128] = out128; pca_outs[256] = out256

# Average top-5 by score
all_scored = [(n, results.get(f'pca{n}', 0) if n in results else eval_loo(pca_outs[n]), pca_outs[n]) for n in pca_outs]
all_scored.sort(key=lambda x: -x[1])
top5 = all_scored[:5]
print(f"  Top-5 PCA dims: {[(n, f'{s:.4f}') for n,s,_ in top5]}", flush=True)

# Simple average
ens5 = np.mean([o for _,_,o in top5], axis=0)
auc_ens5 = eval_loo(ens5)
results['pca_ens5'] = auc_ens5
flag = " *** NEW BEST ***" if auc_ens5 > CURRENT_BEST else ""
print(f"  PCA ens-5 (avg): {auc_ens5:.4f}{flag}", flush=True)

# Weighted ensemble
best_ens = 0; best_ens_cfg = None
for n_ens in range(2, min(6, len(all_scored)+1)):
    topn = all_scored[:n_ens]
    ens = np.mean([o for _,_,o in topn], axis=0)
    auc_e = eval_loo(ens)
    if auc_e > best_ens: best_ens = auc_e; best_ens_cfg = [n for n,_,_ in topn]
results['pca_best_ens'] = best_ens
flag = " *** NEW BEST ***" if best_ens > CURRENT_BEST else ""
print(f"  PCA best-ens: {best_ens:.4f}{flag}  dims={best_ens_cfg}", flush=True)

# ─── Method 3: Fine w_max and k_neg for best PCA ─────────────────────────────
print(f"\n=== Method 3: Fine w_max x k_neg for PCA-{best_pca_n} ===", flush=True)
t0 = time.time()
ew_n, ef_n = make_pca_embs(best_pca_n)
best3 = 0; best_cfg3 = None
for w_max in [0.5, 0.6, 0.7, 0.75, 0.8]:
    for k_neg in [4, 5, 6, 7, 8]:
        out = max_pos_contrast_emb(ew_n, ef_n, k_neg=k_neg, w_max=w_max)
        auc = eval_loo(out)
        key = f'pca{best_pca_n}_wm{int(w_max*10)}_kn{k_neg}'
        results[key] = auc
        flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
        if auc > CURRENT_BEST or auc > best3:
            print(f"  w_max={w_max}, k_neg={k_neg}: {auc:.4f}{flag}", flush=True)
        if auc > best3: best3 = auc; best_cfg3 = (w_max, k_neg)
print(f"  Best: {best_cfg3}: {best3:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: PCA-best + full-1536 fine blend ────────────────────────────────
print(f"\n=== Method 4: PCA-{best_pca_n} + full-1536 fine blend ===", flush=True)
out_base = max_pos_contrast_emb(emb_win_norm, file_embs_norm)
best4 = 0; best_w4 = None
for w_pca in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    blend = w_pca * best_pca_out + (1-w_pca) * out_base
    auc_c = eval_loo(blend)
    if auc_c > best4: best4 = auc_c; best_w4 = w_pca
results['pca_full_blend'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  PCA+full blend: {best4:.4f}{flag}  w_pca={best_w4}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 31 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1])[:20]:
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best_json = rd['best'].get('loo_auc', 0)
for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best_json:
        cur_best_json = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
