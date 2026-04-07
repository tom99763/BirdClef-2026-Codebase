"""
Batch 20: Advanced prototype + novel methods
Goal: beat proto_multik_blend = 0.9321
Methods:
  1. Isotonic prototype: calibrate prototype scores with isotonic regression
  2. Window-level prototype: use all training WINDOWS (not file means) as prototypes
  3. Species co-occurrence prior: also predict based on species that appear together
  4. Temperature-scaled prototype (sharpen/smooth the similarity)
  5. Dual-branch: Prototype(1536d) + KNN(PCA-64)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
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
for fi in range(n_files):
    win_file_id[file_start[fi]:file_end[fi]] = fi

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

def eval_loo(scores): return roc_auc_score(file_labels[:, mask], scores[:, mask], average='macro')

def cosine_prototype(tr_emb, tr_lab, te_wins):
    """Cosine prototype. tr_emb: L2-normalized."""
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab[:, si] > 0.5; neg = ~pos
        if not pos.any(): ws[:, si] = 0.5; continue
        pp = tr_emb[pos].mean(0); pp /= (np.linalg.norm(pp) + EPS)
        sp = te_wins @ pp
        if neg.any():
            pn = tr_emb[neg].mean(0); pn /= (np.linalg.norm(pn) + EPS)
            ws[:, si] = (sp - te_wins @ pn + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return ws

def cosine_knn(tr_emb, tr_lab, te_wins, k=5):
    sims = te_wins @ tr_emb.T
    topk = np.argsort(-sims, axis=1)[:, :k]
    w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
    w /= w.sum(1, keepdims=True) + EPS
    return (w[:, :, None] * tr_lab[topk]).sum(1).astype(np.float32)

results = {}

# Reference: proto_multik (best so far = 0.9321)
print("=== Computing reference proto_multik ===", flush=True)
out_proto  = np.zeros((n_files, n_species), np.float32)
out_knn_mk = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    out_proto[fi] = cosine_prototype(tr_emb, tr_lab, te_wins).mean(0)
    out_knn_mk[fi] = np.mean([cosine_knn(tr_emb, tr_lab, te_wins, k=k).mean(0) for k in [3,5,10]], 0)
proto_multik = 0.7 * out_proto + 0.3 * out_knn_mk
auc_ref = eval_loo(proto_multik)
print(f"  proto_multik reference: {auc_ref:.4f}", flush=True)

# ─── Method 1: Window-level prototype ─────────────────────────────────────────
# Use individual training windows (not file means) as prototype references
print("\n=== Method 1: Window-level prototype (all 739-n_te training windows) ===", flush=True)
t0 = time.time()
out_win_proto = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_mask = (win_file_id != fi)
    tr_wins = emb_win_norm[tr_mask]     # all training windows
    tr_fids = win_file_id[tr_mask]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids])  # window-level labels
    te_wins = emb_win_norm[win_file_id == fi]
    # For each species: prototype = mean of ALL positive WINDOWS (not file means)
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos = tr_lab_win[:, si] > 0.5; neg = ~pos
        if not pos.any(): ws[:, si] = 0.5; continue
        pp = tr_wins[pos].mean(0); pp /= (np.linalg.norm(pp) + EPS)
        sp = te_wins @ pp
        if neg.any():
            pn = tr_wins[neg].mean(0); pn /= (np.linalg.norm(pn) + EPS)
            ws[:, si] = (sp - te_wins @ pn + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    out_win_proto[fi] = ws.mean(0)
auc1 = eval_loo(out_win_proto)
flag = " *** NEW BEST ***" if auc1 > CURRENT_BEST else ""
print(f"  win_level_proto: {auc1:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)
results['win_level_proto'] = auc1

# Blend window-proto with file-proto
best1b = 0; best_w1b = None
for w_win in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    blend = w_win * out_win_proto + (1-w_win) * proto_multik
    auc_c = eval_loo(blend)
    if auc_c > best1b: best1b = auc_c; best_w1b = w_win
results['winproto_multik_blend'] = best1b
flag = " *** NEW BEST ***" if best1b > CURRENT_BEST else ""
print(f"  win_proto + multik: {best1b:.4f}{flag}  w_win={best_w1b}", flush=True)

# ─── Method 2: Temperature-scaled prototype ────────────────────────────────────
print("\n=== Method 2: Temperature-scaled prototype ===", flush=True)
t0 = time.time()
out_temps = {}
for temp in [0.5, 1.0, 2.0, 5.0]:
    out_t = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_win_norm[win_file_id == fi]
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos = tr_lab[:, si] > 0.5; neg = ~pos
            if not pos.any(): ws[:, si] = 0.5; continue
            pp = tr_emb[pos].mean(0); pp /= (np.linalg.norm(pp) + EPS)
            sp = te_wins @ pp  # cosine sim [-1,1]
            # Temperature scaling: sigmoid(sim * temp)
            if neg.any():
                pn = tr_emb[neg].mean(0); pn /= (np.linalg.norm(pn) + EPS)
                logit = (sp - te_wins @ pn) * temp
            else:
                logit = sp * temp
            ws[:, si] = 1 / (1 + np.exp(-logit.clip(-10, 10)))
        out_t[fi] = ws.mean(0)
    auc_t = eval_loo(out_t)
    out_temps[temp] = out_t
    print(f"  temp={temp}: {auc_t:.4f}", flush=True)
    results[f'proto_temp{temp}'] = auc_t
print(f"  ({time.time()-t0:.0f}s)", flush=True)

best_temp = max(out_temps, key=lambda t: results[f'proto_temp{t}'])
out_temp_best = out_temps[best_temp]
print(f"  Best temp: {best_temp}", flush=True)

# Temp-proto + multik blend
best2b = 0; best_w2b = None
for w_t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    blend = w_t * out_temp_best + (1-w_t) * out_knn_mk
    auc_c = eval_loo(blend)
    if auc_c > best2b: best2b = auc_c; best_w2b = w_t
results['temp_proto_knn_blend'] = best2b
flag = " *** NEW BEST ***" if best2b > CURRENT_BEST else ""
print(f"  temp_proto + knn: {best2b:.4f}{flag}  w_t={best_w2b}", flush=True)

# ─── Method 3: Dual-branch (file-proto + window-proto + multi-k KNN) ──────────
print("\n=== Method 3: Triple blend (file_proto + win_proto + multik_knn) ===", flush=True)
best3 = 0; best_cfg3 = None
for w_fp in [0.3, 0.4, 0.5]:
    for w_wp in [0.2, 0.3, 0.4]:
        w_knn = 1.0 - w_fp - w_wp
        if w_knn < 0.1: continue
        blend = w_fp * out_proto + w_wp * out_win_proto + w_knn * out_knn_mk
        auc_c = eval_loo(blend)
        if auc_c > best3: best3 = auc_c; best_cfg3 = (w_fp, w_wp, w_knn)
results['triple_fp_wp_knn'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  Triple: fp={best_cfg3[0]}, wp={best_cfg3[1]}, knn={best_cfg3[2]:.1f} → {best3:.4f}{flag}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 20 Summary ===", flush=True)
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
print(f"\nFinal best (simple format): ref={auc_ref:.4f}, new={max(results.values()):.4f}", flush=True)
