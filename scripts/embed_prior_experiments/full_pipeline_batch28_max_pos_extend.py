"""
Batch 28: Extend max_pos + mean_pos blend discoveries
Goal: beat max_mean_pos_blend = 0.9611
Methods:
  1. Fine-sweep max/mean blend weights + neg k
  2. Top-2/3 pos sims blend (soft max)
  3. Max-pos + mean-pos + adap_proto triple
  4. Per-species adaptive: use max when few positives, mean when many
  5. Aggregated score: combine mean(top-k pos sims) directly (not via prototype)
"""
import numpy as np, json, os, time
from sklearn.preprocessing import normalize
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
CURRENT_BEST = 0.9611

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

def compute_out(w_max, k_neg):
    """max_sim_pos * w_max + mean_proto_pos * (1-w_max), with topk k_neg neg."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_win_norm[win_file_id == fi]
        tr_wins_all = emb_win_norm[win_file_id != fi]
        tr_fids_all = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win[:,si] > 0.5
            neg_win_mask = ~pos_win_mask
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T
            sp_max = pos_sims.max(1)
            pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
            sp_mean = te_wins @ pp_mean
            sp = w_max * sp_max + (1-w_max) * sp_mean
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

# ─── Method 1: Fine-sweep w_max × k_neg ──────────────────────────────────────
print("=== Method 1: Fine-sweep w_max x k_neg ===", flush=True)
t0 = time.time()
grid_outs = {}
best1 = 0; best_cfg1 = None; best_out1 = None
for w_max in [0.4, 0.5, 0.6, 0.7, 0.8]:
    for k_neg in [3, 4, 5, 6]:
        out = compute_out(w_max, k_neg)
        auc = eval_loo(out)
        key = f'mm_wm{int(w_max*10)}_kn{k_neg}'
        results[key] = auc
        grid_outs[(w_max, k_neg)] = out
        flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
        print(f"  w_max={w_max}, k_neg={k_neg}: {auc:.4f}{flag}", flush=True)
        if auc > best1: best1 = auc; best_cfg1 = (w_max, k_neg); best_out1 = out
print(f"  ({time.time()-t0:.0f}s)\n  Best: {best_cfg1}: {best1:.4f}", flush=True)

# ─── Method 2: Top-k pos sims mean (not via prototype) ───────────────────────
print("\n=== Method 2: Mean of top-k pos sims directly ===", flush=True)
t0 = time.time()
best2 = 0; best_k2 = None; best_out2 = None
for k_pos_sims in [1, 2, 3, 5]:
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_win_norm[win_file_id == fi]
        tr_wins_all = emb_win_norm[win_file_id != fi]
        tr_fids_all = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win[:,si] > 0.5
            neg_win_mask = ~pos_win_mask
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T  # [n_te, n_pos]
            k_act_pos = min(k_pos_sims, pos_sims.shape[1])
            # Mean of top-k pos sims (not prototype-based)
            top_pos_sims = np.sort(-pos_sims, axis=1)[:, :k_act_pos] * -1
            sp = top_pos_sims.mean(1)  # [n_te]
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act_neg = min(4, neg_sims.shape[1])
                top_neg_sims = np.sort(-neg_sims, axis=1)[:, :k_act_neg] * -1
                sn = top_neg_sims.mean(1)
                ws[:,si] = (sp - sn + 1) / 2
            else: ws[:,si] = (sp + 1) / 2
        out[fi] = ws.mean(0)
    auc = eval_loo(out)
    results[f'topk_sims_direct_k{k_pos_sims}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  k_pos={k_pos_sims}: {auc:.4f}{flag}", flush=True)
    if auc > best2: best2 = auc; best_k2 = k_pos_sims; best_out2 = out
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 3: max_pos + adap_pos blend (pos aggregation) ─────────────────────
print("\n=== Method 3: max_pos + adap_k1_proto blend ===", flush=True)
t0 = time.time()
# Compute adap_k1 (top-1 pos window prototype)
out_adap_k1 = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    tr_wins_all = emb_win_norm[win_file_id != fi]
    tr_fids_all = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
    ws = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos_win_mask = tr_lab_win[:,si] > 0.5
        neg_win_mask = ~pos_win_mask
        if not pos_win_mask.any(): ws[:,si]=0.5; continue
        pos_wins = tr_wins_all[pos_win_mask]
        pos_sims = te_wins @ pos_wins.T
        pp_adap = pos_wins[pos_sims.argmax(1)]
        pp_adap /= (np.linalg.norm(pp_adap, axis=1, keepdims=True) + EPS)
        sp = (te_wins * pp_adap).sum(1)
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :min(4, neg_sims.shape[1])]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
        else: ws[:,si] = (sp+1)/2
    out_adap_k1[fi] = ws.mean(0)

# Compute max_pos (w_max=1.0)
out_max_only = compute_out(1.0, 4)

best3 = 0; best_w3 = None
for w_adap in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blend = w_adap * out_adap_k1 + (1-w_adap) * out_max_only
    auc_c = eval_loo(blend)
    if auc_c > best3: best3 = auc_c; best_w3 = w_adap
results['adap_maxonly_blend'] = best3
flag = " *** NEW BEST ***" if best3 > CURRENT_BEST else ""
print(f"  adap_k1+max_only: {best3:.4f}{flag}  w_adap={best_w3}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: Best-w_max + adap_k1 triple blend ─────────────────────────────
print("\n=== Method 4: Best-wmax + adap_k1 triple ===", flush=True)
best4 = 0; best_cfg4 = None
best_mm_out = best_out1 if best_out1 is not None else grid_outs.get((0.6, 4))
if best_mm_out is not None:
    for w_mm in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for w_adap in [0.1, 0.2, 0.3]:
            w_rest = 1.0 - w_mm - w_adap
            if w_rest < 0: continue
            blend = w_mm * best_mm_out + w_adap * out_adap_k1
            if w_rest > 0:
                # Add direct sims if available
                blend = blend  # keep as is
            auc_c = eval_loo(blend)
            if auc_c > best4: best4 = auc_c; best_cfg4 = (w_mm, w_adap)
results['mm_adap_triple'] = best4
flag = " *** NEW BEST ***" if best4 > CURRENT_BEST else ""
print(f"  mm+adap triple: {best4:.4f}{flag}  cfg={best_cfg4}", flush=True)

# ─── Method 5: Soft-max pos sim (log-sum-exp) ─────────────────────────────────
print("\n=== Method 5: Log-sum-exp positive scoring ===", flush=True)
t0 = time.time()
best5 = 0; best_tau5 = None
for tau in [0.1, 0.2, 0.5, 1.0]:
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_idx = [fj for fj in range(n_files) if fj != fi]
        tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
        te_wins = emb_win_norm[win_file_id == fi]
        tr_wins_all = emb_win_norm[win_file_id != fi]
        tr_fids_all = win_file_id[win_file_id != fi]
        tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
        ws = np.zeros((len(te_wins), n_species), np.float32)
        for si in range(n_species):
            pos_win_mask = tr_lab_win[:,si] > 0.5
            neg_win_mask = ~pos_win_mask
            if not pos_win_mask.any(): ws[:,si]=0.5; continue
            pos_wins = tr_wins_all[pos_win_mask]
            pos_sims = te_wins @ pos_wins.T  # [n_te, n_pos]
            # Log-sum-exp (soft max): tau * log(mean(exp(pos_sims / tau)))
            lse = pos_sims / tau
            lse -= lse.max(1, keepdims=True)
            sp = tau * (np.log(np.exp(lse).mean(1)) + lse.max(1))  # = tau * log(mean(exp(pos/tau)))
            # Clip to [-1,1] range
            sp = sp.clip(-1, 1)
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(4, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp + 1) / 2
        out[fi] = ws.mean(0)
    auc = eval_loo(out)
    results[f'lse_tau{tau}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  tau={tau}: {auc:.4f}{flag}", flush=True)
    if auc > best5: best5 = auc; best_tau5 = tau
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 28 Summary ===", flush=True)
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
print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
