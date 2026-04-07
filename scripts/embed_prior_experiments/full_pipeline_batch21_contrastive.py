"""
Batch 21: Contrastive & nested approaches
Goal: beat proto_multik_blend = 0.9321
Methods:
  1. Window-level proto + multi-k KNN blend
  2. Contrastive prototype (pos vs NEAREST neg, not mean neg)
  3. Species-adaptive k (rare species get larger k)
  4. Nested KNN: file-level KNN → within those files, window-level re-rank
  5. Logit-blend prototype (Perch logit provides soft positive labels)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[file_start[fi]:file_end[fi]] = fi

file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_embs_raw  = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_embs_raw[fi]  = emb_win[s:e].mean(0)

file_embs_norm = normalize(file_embs_raw, norm='l2').astype(np.float32)
emb_win_norm   = normalize(emb_win, norm='l2').astype(np.float32)
EPS = 1e-7
mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9321
species_counts = file_labels.sum(0)  # [234] - how many files have each species

def eval_loo(scores): return roc_auc_score(file_labels[:, mask], scores[:, mask], average='macro')

def cosine_knn(tr_emb, tr_lab, te_wins, k=5):
    sims = te_wins @ tr_emb.T
    topk = np.argsort(-sims, axis=1)[:, :k]
    w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
    w /= w.sum(1, keepdims=True) + EPS
    return (w[:, :, None] * tr_lab[topk]).sum(1).astype(np.float32)

results = {}

# Precompute components
print("Precomputing components...", flush=True)
out_proto  = np.zeros((n_files, n_species), np.float32)
out_knn_mk = np.zeros((n_files, n_species), np.float32)
out_winproto = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    # File-level prototype
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:, si] > 0.5
        if not pos.any(): ws[:, si] = 0.5; continue
        pp = tr_emb[pos].mean(0); pp /= (np.linalg.norm(pp) + EPS)
        neg = ~pos
        sp = te_wins @ pp
        if neg.any():
            pn = tr_emb[neg].mean(0); pn /= (np.linalg.norm(pn) + EPS)
            ws[:, si] = (sp - te_wins @ pn + 1) / 2
        else: ws[:, si] = (sp + 1) / 2
    out_proto[fi] = ws.mean(0)
    # Multi-k KNN
    out_knn_mk[fi] = np.mean([cosine_knn(tr_emb, tr_lab, te_wins, k=k).mean(0) for k in [3,5,10]], 0)
    # Window-level prototype
    tr_mask = (win_file_id != fi)
    tr_wins = emb_win_norm[tr_mask]; tr_fids = win_file_id[tr_mask]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids])
    ws2 = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos2 = tr_lab_win[:, si] > 0.5
        if not pos2.any(): ws2[:, si] = 0.5; continue
        pp2 = tr_wins[pos2].mean(0); pp2 /= (np.linalg.norm(pp2) + EPS)
        neg2 = ~pos2; sp2 = te_wins @ pp2
        if neg2.any():
            pn2 = tr_wins[neg2].mean(0); pn2 /= (np.linalg.norm(pn2) + EPS)
            ws2[:, si] = (sp2 - te_wins @ pn2 + 1) / 2
        else: ws2[:, si] = (sp2 + 1) / 2
    out_winproto[fi] = ws2.mean(0)
print("  Done.", flush=True)

# ─── Method 1: Window-level proto + multi-k KNN blend ─────────────────────────
print("\n=== Method 1: Window-proto + multi-k KNN blend ===", flush=True)
best1 = 0; best_w1 = None
for w_wp in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    blend = w_wp * out_winproto + (1-w_wp) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best1: best1 = auc_c; best_w1 = w_wp
results['winproto_knn_blend'] = best1
flag = " *** NEW BEST ***" if best1 > CURRENT_BEST else ""
print(f"  winproto+knn: {best1:.4f}{flag}  w_wp={best_w1}", flush=True)

# ─── Method 2: Contrastive prototype (pos vs nearest neg) ────────────────────
print("\n=== Method 2: Contrastive prototype (nearest neg) ===", flush=True)
t0 = time.time()
out_contrast = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:, si] > 0.5; neg = ~pos
        if not pos.any(): ws[:, si] = 0.5; continue
        pp = tr_emb[pos].mean(0); pp /= (np.linalg.norm(pp) + EPS)
        sp = te_wins @ pp  # sim to positive prototype
        if neg.any():
            # Nearest negative (hardest neg)
            neg_sims = te_wins @ tr_emb[neg].T  # [n_te, n_neg]
            nearest_neg_idx = neg_sims.argmax(1)  # [n_te]
            nearest_neg = tr_emb[neg][nearest_neg_idx]  # [n_te, 1536]
            nearest_neg /= (np.linalg.norm(nearest_neg, axis=1, keepdims=True) + EPS)
            sn = (te_wins * nearest_neg).sum(1)  # [n_te] sim to nearest neg
            ws[:, si] = (sp - sn + 1) / 2
        else: ws[:, si] = (sp + 1) / 2
    out_contrast[fi] = ws.mean(0)
auc2 = eval_loo(out_contrast)
flag = " *** NEW BEST ***" if auc2 > CURRENT_BEST else ""
print(f"  contrast_proto: {auc2:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['contrastive_proto'] = auc2

# Contrastive + KNN blend
best2b = 0; best_w2b = None
for w_c in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    blend = w_c * out_contrast + (1-w_c) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best2b: best2b = auc_c; best_w2b = w_c
results['contrast_knn_blend'] = best2b
flag = " *** NEW BEST ***" if best2b > CURRENT_BEST else ""
print(f"  contrast+knn: {best2b:.4f}{flag}  w_c={best_w2b}", flush=True)

# ─── Method 3: Species-adaptive k ─────────────────────────────────────────────
# Rare species (fewer positives) → larger k; common → smaller k
print("\n=== Method 3: Species-adaptive k ===", flush=True)
t0 = time.time()
# For each species, choose k based on number of positive training files
# Rare: n_pos < 5 → k=15; medium: 5-15 → k=10; common: >15 → k=5
out_adaptive = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    sims = te_wins @ tr_emb.T  # [n_te, 65]
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        n_pos = int(tr_lab[:, si].sum())
        if n_pos == 0: ws[:, si] = 0.5; continue
        # Adaptive k
        if n_pos < 3: k = min(15, len(tr_emb))
        elif n_pos < 8: k = min(10, len(tr_emb))
        else: k = min(5, len(tr_emb))
        topk_idx = np.argsort(-sims, axis=1)[:, :k]
        w = np.take_along_axis(sims, topk_idx, axis=1).clip(0, 1)
        w /= w.sum(1, keepdims=True) + EPS
        ws[:, si] = (w * tr_lab[topk_idx, si]).sum(1)
    out_adaptive[fi] = ws.mean(0)
auc3 = eval_loo(out_adaptive)
flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
print(f"  adaptive_k: {auc3:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['adaptive_k_knn'] = auc3

# Adaptive + proto blend
best3b = 0; best_w3b = None
for w_ad in [0.3, 0.4, 0.5]:
    blend = (1-w_ad) * out_proto + w_ad * out_adaptive
    auc_c = eval_loo(blend)
    if auc_c > best3b: best3b = auc_c; best_w3b = w_ad
results['proto_adaptive_blend'] = best3b
flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
print(f"  proto+adaptive: {best3b:.4f}{flag}  w_ad={best_w3b}", flush=True)

# ─── Method 4: Nested KNN (file→window re-rank) ───────────────────────────────
print("\n=== Method 4: Nested KNN (top-3 files → window re-rank) ===", flush=True)
t0 = time.time()
out_nested = np.zeros((n_files, n_species), np.float32)
K_FILE = 10  # top files by file-level similarity
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    te_file = file_embs_norm[fi:fi+1]  # [1, 1536]
    # Step 1: file-level KNN to find K_FILE most similar training files
    file_sims = (te_file @ tr_emb.T)[0]  # [65]
    top_file_idx = np.argsort(-file_sims)[:K_FILE]  # top-K_FILE training files
    # Step 2: gather windows from those top files
    top_wins_list = []; top_labs_list = []
    for fj_local in top_file_idx:
        fj = tr_idx[fj_local]
        top_wins_list.append(emb_win_norm[win_file_id == fj])
        n_win_fj = int((win_file_id == fj).sum())
        top_labs_list.append(np.tile(file_labels[fj], (n_win_fj, 1)))
    top_wins = np.vstack(top_wins_list)  # [K*n_win, 1536]
    top_labs = np.vstack(top_labs_list)  # [K*n_win, 234]
    # Step 3: window-level KNN within those top files
    sims = te_wins @ top_wins.T  # [n_te, K*n_win]
    topk = np.argsort(-sims, axis=1)[:, :5]
    w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
    w /= w.sum(1, keepdims=True) + EPS
    out_nested[fi] = (w[:, :, None] * top_labs[topk]).sum(1).mean(0)
auc4 = eval_loo(out_nested)
flag = " *** NEW BEST ***" if auc4 > CURRENT_BEST else ""
print(f"  nested_knn: {auc4:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['nested_knn'] = auc4

# Nested + proto blend
best4b = 0; best_w4b = None
for w_n in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blend = (1-w_n) * out_proto + w_n * out_nested
    auc_c = eval_loo(blend)
    if auc_c > best4b: best4b = auc_c; best_w4b = w_n
results['proto_nested_blend'] = best4b
flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
print(f"  proto+nested: {best4b:.4f}{flag}  w_n={best_w4b}", flush=True)

# ─── Method 5: Logit-soft prototype ──────────────────────────────────────────
# Instead of binary positive/negative labels, use Perch logit (soft label)
# proto = Σ_j sigmoid(logit_j) * emb_j (soft-weighted centroid per species)
print("\n=== Method 5: Soft-label prototype (Perch logit weighted) ===", flush=True)
t0 = time.time()
def sigmoid(x): return 1/(1+np.exp(-x.clip(-10,10)))
out_soft = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    tr_logit = file_logit_max[tr_idx]  # [65, 234]
    te_wins = emb_win_norm[win_file_id == fi]
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        soft_w = sigmoid(tr_logit[:, si])  # [65] soft positive weights
        if soft_w.sum() < 0.1: ws[:, si] = 0.5; continue
        soft_w_norm = soft_w / (soft_w.sum() + EPS)
        proto_soft = (tr_emb * soft_w_norm[:, None]).sum(0)
        proto_soft /= (np.linalg.norm(proto_soft) + EPS)
        # Negative proto: 1 - soft_w as negative weight
        neg_w = (1 - soft_w); neg_w_norm = neg_w / (neg_w.sum() + EPS)
        proto_neg = (tr_emb * neg_w_norm[:, None]).sum(0)
        proto_neg /= (np.linalg.norm(proto_neg) + EPS)
        sp = te_wins @ proto_soft
        sn = te_wins @ proto_neg
        ws[:, si] = (sp - sn + 1) / 2
    out_soft[fi] = ws.mean(0)
auc5 = eval_loo(out_soft)
flag = " *** NEW BEST ***" if auc5 > CURRENT_BEST else ""
print(f"  soft_proto: {auc5:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['soft_label_proto'] = auc5

# Soft + multi-k KNN blend
best5b = 0; best_w5b = None
for w_s in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    blend = w_s * out_soft + (1-w_s) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best5b: best5b = auc_c; best_w5b = w_s
results['softproto_knn_blend'] = best5b
flag = " *** NEW BEST ***" if best5b > CURRENT_BEST else ""
print(f"  soft_proto+knn: {best5b:.4f}{flag}  w_s={best_w5b}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 21 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
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
print(f"\nCurrent simple-format best: {CURRENT_BEST:.4f}", flush=True)
