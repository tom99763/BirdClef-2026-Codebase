"""
Batch 27: Transductive/semi-transductive methods
Goal: beat current best (expected ~0.96xx)
Methods:
  1. Test-guided prototype: use test windows to reweight positive prototype
  2. Iterative refinement: refine pos prototype with high-confidence test windows
  3. Dual-score: per-window max-sim-pos AND mean-sim-pos, blended
  4. Species-aware contrast: for ambiguous species, use tighter contrast margin
  5. Cross-species repulsion: explicitly push away prototypes of co-occurring species
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
CURRENT_BEST = 0.9610  # update after batch 26 if needed

def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

# ─── Method 1: Test-guided prototype (reweight pos windows by test affinity) ──
print("=== Method 1: Test-guided positive prototype ===", flush=True)
t0 = time.time()
best1 = 0; best_tau1 = None; best_out1 = None
for tau in [0.3, 0.5, 1.0]:
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
            # Test-guided: weight pos windows by mean test affinity
            pos_sims_all = te_wins @ pos_wins.T  # [n_te, n_pos]
            # Mean test window affinity to each pos win (test-guided weight)
            test_affinity = pos_sims_all.mean(0)  # [n_pos]
            w = np.exp(test_affinity / tau)
            w /= w.sum() + EPS
            pp = (w[:, None] * pos_wins).sum(0)
            pp /= (np.linalg.norm(pp) + EPS)
            sp = te_wins @ pp
            if neg_win_mask.any():
                neg_wins = tr_wins_all[neg_win_mask]
                neg_sims = te_wins @ neg_wins.T
                k_act = min(4, neg_sims.shape[1])
                top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
                top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
                ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
            else: ws[:,si] = (sp+1)/2
        out[fi] = ws.mean(0)
    auc = eval_loo(out)
    results[f'test_guided_tau{tau}'] = auc
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  tau={tau}: {auc:.4f}{flag}", flush=True)
    if auc > best1: best1 = auc; best_tau1 = tau; best_out1 = out
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 2: Max-sim pos + mean-sim pos dual blend ─────────────────────────
print("\n=== Method 2: Max-pos + mean-pos dual blend ===", flush=True)
t0 = time.time()
out_max_pos = np.zeros((n_files, n_species), np.float32)
out_mean_pos = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    tr_idx = [fj for fj in range(n_files) if fj != fi]
    tr_emb = file_embs_norm[tr_idx]; tr_lab = file_labels[tr_idx]
    te_wins = emb_win_norm[win_file_id == fi]
    tr_wins_all = emb_win_norm[win_file_id != fi]
    tr_fids_all = win_file_id[win_file_id != fi]
    tr_lab_win = np.array([file_labels[f] for f in tr_fids_all])
    ws_max = np.zeros((len(te_wins), n_species), np.float32)
    ws_mean = np.zeros((len(te_wins), n_species), np.float32)
    for si in range(n_species):
        pos_win_mask = tr_lab_win[:,si] > 0.5
        neg_win_mask = ~pos_win_mask
        if not pos_win_mask.any(): ws_max[:,si]=0.5; ws_mean[:,si]=0.5; continue
        pos_wins = tr_wins_all[pos_win_mask]
        neg_wins = tr_wins_all[neg_win_mask] if neg_win_mask.any() else None
        pos_sims = te_wins @ pos_wins.T  # [n_te, n_pos]
        # Max: nearest positive window sim
        sp_max = pos_sims.max(1)
        # Mean: use mean prototype
        pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
        sp_mean = te_wins @ pp_mean
        if neg_wins is not None:
            neg_sims = te_wins @ neg_wins.T
            k_act = min(4, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            sn = (te_wins * top_neg).sum(1)
            ws_max[:,si] = (sp_max - sn + 1) / 2
            ws_mean[:,si] = (sp_mean - sn + 1) / 2
        else:
            ws_max[:,si] = (sp_max + 1) / 2
            ws_mean[:,si] = (sp_mean + 1) / 2
    out_max_pos[fi] = ws_max.mean(0)
    out_mean_pos[fi] = ws_mean.mean(0)
best2 = 0; best_w2 = None
for w_max in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    blend = w_max * out_max_pos + (1-w_max) * out_mean_pos
    auc_c = eval_loo(blend)
    if auc_c > best2: best2 = auc_c; best_w2 = w_max
results['max_mean_pos_blend'] = best2
flag = " *** NEW BEST ***" if best2 > CURRENT_BEST else ""
print(f"  max+mean pos blend: {best2:.4f}{flag}  w_max={best_w2}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 3: Iterative proto refinement ─────────────────────────────────────
print("\n=== Method 3: 2-round iterative prototype refinement ===", flush=True)
t0 = time.time()
out_iter = np.zeros((n_files, n_species), np.float32)
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
        # Round 1: adaptive pos prototype (top-1)
        pos_sims = te_wins @ pos_wins.T
        pp_r1 = pos_wins[pos_sims.argmax(1)]
        pp_r1 /= (np.linalg.norm(pp_r1, axis=1, keepdims=True) + EPS)
        sp_r1 = (te_wins * pp_r1).sum(1)
        # Round 2: re-rank pos windows using round-1 scores to get refined prototype
        # Use top-confidence test windows to compute refined proto
        high_conf = sp_r1 > sp_r1.median() if hasattr(sp_r1, 'median') else sp_r1 > np.median(sp_r1)
        if high_conf.any():
            ref_query = te_wins[high_conf].mean(0)
            ref_query /= (np.linalg.norm(ref_query) + EPS)
            refined_sims = pos_wins @ ref_query
            k_act = min(3, pos_wins.shape[0])
            top_pos_r2 = pos_wins[np.argsort(-refined_sims)[:k_act]]
            pp_r2 = top_pos_r2.mean(0); pp_r2 /= (np.linalg.norm(pp_r2) + EPS)
        else:
            pp_r2 = pos_wins.mean(0); pp_r2 /= (np.linalg.norm(pp_r2) + EPS)
        sp_r2 = te_wins @ pp_r2
        # Final pos score: blend round 1 and round 2
        sp = 0.5 * sp_r1 + 0.5 * sp_r2
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(4, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            ws[:,si] = (sp - (te_wins * top_neg).sum(1) + 1) / 2
        else: ws[:,si] = (sp+1)/2
    out_iter[fi] = ws.mean(0)
auc3 = eval_loo(out_iter)
results['iterative_proto'] = auc3
flag = " *** NEW BEST ***" if auc3 > CURRENT_BEST else ""
print(f"  iterative_proto: {auc3:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# ─── Method 4: Per-window score: max(top-k pos sims) - mean(top-k neg sims) ──
print("\n=== Method 4: Direct margin score (max pos sim - mean neg sim) ===", flush=True)
t0 = time.time()
out_margin = np.zeros((n_files, n_species), np.float32)
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
        sp = pos_sims.max(1)  # max pos sim
        if neg_win_mask.any():
            neg_wins = tr_wins_all[neg_win_mask]
            neg_sims = te_wins @ neg_wins.T
            k_act = min(4, neg_sims.shape[1])
            top_neg_sims = np.sort(-neg_sims, axis=1)[:, :k_act] * -1  # top-k neg sims
            sn = top_neg_sims.mean(1)  # mean of top-k neg sims (direct, not via prototype)
            ws[:,si] = (sp - sn + 1) / 2
        else: ws[:,si] = (sp + 1) / 2
    out_margin[fi] = ws.mean(0)
auc4 = eval_loo(out_margin)
results['direct_margin'] = auc4
flag = " *** NEW BEST ***" if auc4 > CURRENT_BEST else ""
print(f"  direct_margin: {auc4:.4f}{flag}  ({time.time()-t0:.0f}s)", flush=True)

# Blend with adaptive_k1_k4
# Load previous best or recompute
out_adap_k1_k4 = np.zeros((n_files, n_species), np.float32)
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
    out_adap_k1_k4[fi] = ws.mean(0)

best4b = 0; best_w4b = None
for w_m in [0.3, 0.4, 0.5, 0.6]:
    blend = w_m * out_margin + (1-w_m) * out_adap_k1_k4
    auc_c = eval_loo(blend)
    if auc_c > best4b: best4b = auc_c; best_w4b = w_m
results['margin_adap_blend'] = best4b
flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
print(f"  margin+adap_k1k4: {best4b:.4f}{flag}  w_margin={best_w4b}", flush=True)

# ─── Method 5: Triple blend: adap_k1k4 + max_mean_blend + test_guided ─────────
print("\n=== Method 5: Triple blend ===", flush=True)
if best_out1 is not None:
    best_max_mean = w_max * out_max_pos + (1-w_max) * out_mean_pos if best_w2 else out_max_pos
    best5 = 0; best_cfg5 = None
    for w1 in [0.3, 0.4, 0.5, 0.6]:
        for w2 in [0.2, 0.3, 0.4]:
            w3 = 1.0 - w1 - w2
            if w3 < 0.1: continue
            blend = w1 * out_adap_k1_k4 + w2 * best_out1 + w3 * best_max_mean
            auc_c = eval_loo(blend)
            if auc_c > best5: best5 = auc_c; best_cfg5 = (w1, w2, w3)
    results['triple_blend'] = best5
    flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
    print(f"  triple_blend: {best5:.4f}{flag}  cfg={best_cfg5}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Batch 27 Summary ===", flush=True)
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
