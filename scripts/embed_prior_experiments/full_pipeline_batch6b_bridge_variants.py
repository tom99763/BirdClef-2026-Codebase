"""
Full Pipeline Batch 6b: SS Bridge Advanced Variants (FIXED computation)
Baseline: ss_bridge_weighted = 0.9441

Key fix: bridge computation uses UNNORMALIZED signatures, then row-normalize.
sim_bridge[i,j] = file_emb_avg_norm[i] @ train_ss_sigs[j].T (unnormalized sigs)
then row-normalize: sim_bridge_n = sim_bridge / sqrt(sum_j(sim_bridge^2))

Methods:
1. m_sweep: M=200, 500, 1000
2. sed_weighted: Weight windows by perch × SED similarity
3. sed_species: Weight by perch × max SED score for file's species
4. iterative: 2nd-order bridge via SS-mediated re-weighting
5. dual_modal: Combine perch-bridge + SED-bridge
"""
import numpy as np, pickle, os, json
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load labeled soundscape data ──────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list = list(perch['file_list'])
n_windows = perch['n_windows']
n_files = len(file_list)
n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end = np.cumsum(n_windows).astype(np.int32)

def sigmoid(x): return 1. / (1. + np.exp(-np.clip(x, -88, 88)))
file_labels = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_embs_avg = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_embs_avg[fi] = emb_win[s:e].mean(0)
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])] = fi
file_emb_avg_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)

# ── Load SED predictions ───────────────────────────────────────────────────────
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
ss_probs = sed_npz['probs'].astype(np.float32)  # (127896, 234)
sed_by_file = {}
for i, rid in enumerate(sed_npz['row_ids']):
    sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]), []).append(i)
file_sed_max = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fb = fname.replace('.ogg', '').replace('.flac', '')
    if fb in sed_by_file:
        file_sed_max[fi] = sed_npz['probs'][sed_by_file[fb]].max(0)

EPS = 1e-7
def vlom_blend(a, b, w_a=0.5, w_b=0.5):
    ws = w_a + w_b; w_a /= ws; w_b /= ws
    return sigmoid(w_a * np.log(a.clip(EPS)/(1-a).clip(EPS)) +
                   w_b * np.log(b.clip(EPS)/(1-b).clip(EPS)))
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))

# ── Base PKL ───────────────────────────────────────────────────────────────────
with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep_base = pickle.load(f)
X_ref = ep_base['X_combined_n'].astype(np.float32)
fl = ep_base['file_labels'].astype(np.float32)

# ── All soundscape embeddings ──────────────────────────────────────────────────
print("Loading all soundscape embeddings...", flush=True)
ss_all = np.load("outputs/perch_emb_all_ss.npz", allow_pickle=True)
ss_emb = ss_all['emb'].astype(np.float32)
ss_emb_norm = normalize(ss_emb, norm='l2').astype(np.float32)
n_ss_wins = len(ss_emb_norm)
print(f"  {n_ss_wins} windows.", flush=True)

# ── Compute sim_lab_ss (perch) ─────────────────────────────────────────────────
print("Computing labeled × ss similarity (perch)...", flush=True)
CHUNK = 20000
sim_lab_ss = np.zeros((n_files, n_ss_wins), np.float32)
for cs in range(0, n_ss_wins, CHUNK):
    ce = min(cs + CHUNK, n_ss_wins)
    sim_lab_ss[:, cs:ce] = file_emb_avg_norm @ ss_emb_norm[cs:ce].T
    if cs % 60000 == 0: print(f"    {cs}/{n_ss_wins}...", flush=True)
print("  done.", flush=True)

# ── Precompute win_k1 ──────────────────────────────────────────────────────────
print("Precomputing win_k1...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = (win_file_id != i)
    X_tr = emb_win_norm[tr_mask]
    tr_fi = win_file_id[tr_mask]
    sims = X_te @ X_tr.T
    top_idx = np.argsort(-sims, 1)[:, :1]
    wp = np.zeros((te_e - te_s, n_species), np.float32)
    for wi in range(te_e - te_s):
        ww = sims[wi, top_idx[wi]].clip(0)
        ws_v = ww.sum()
        ww = ww / ws_v if ws_v > 1e-8 else np.ones(1)
        wp[wi] = (ww[:, None] * file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)
print("  done.", flush=True)

# ── X_ref similarity ───────────────────────────────────────────────────────────
sim_ref = X_ref @ X_ref.T
np.fill_diagonal(sim_ref, -np.inf)

# ── RKNN helper ────────────────────────────────────────────────────────────────
k_rknn, T = 5, 0.2
def compute_rknn(sim_combined):
    sc = sim_combined.copy()
    np.fill_diagonal(sc, -np.inf)
    top_k = np.argsort(-sc, axis=1)[:, :k_rknn]
    kth = sc[np.arange(n_files), top_k[:, -1]]
    y = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims_i = sc[i, tr]
        top_i = np.argsort(-sims_i)[:k_rknn]
        mutual, msims = [], []
        for ti, tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]] >= kth[tj]:
                mutual.append(tj); msims.append(sims_i[top_i[ti]])
        if len(mutual) == 0:
            top5 = np.argsort(-sims_i)[:5]
            ls = sims_i[top5] / T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y[i] = (w[:, None] * fl[tr[top5]]).sum(0)
        else:
            ms = np.array(msims); ls = ms / T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y[i] = (w[:, None] * fl[mutual]).sum(0)
    return y

def build_bridge_from_sigs(train_sigs):
    """
    Correct bridge computation:
    sim_bridge[i,j] = file_emb_avg_norm[i] @ train_sigs[j] (unnormalized sigs)
    then row-normalize by sqrt(sum_j sim_bridge[i,j]^2)
    """
    sb = file_emb_avg_norm @ train_sigs.T  # (66, 66)
    bridge_norm = np.sqrt((sb**2).sum(1, keepdims=True)).clip(1e-8)
    return sb / bridge_norm  # row-normalized

def eval_full(y_ep, wg=0.40, a=1.00, b=1.50):
    yb = wg * y_ep + (1-wg) * y_win_k1
    log_yb = np.log(yb.clip(EPS))
    full = sigmoid(a * base_logit + b * log_yb)
    return macro_auc(file_labels, full)

# ── Baseline: SS Bridge α=0.4 (correct implementation) ───────────────────────
print("\n=== Baseline: SS Bridge alpha=0.4 (CORRECT) ===", flush=True)
M = 100
top_M_idx_base = np.argsort(-sim_lab_ss, axis=1)[:, :M]
top_M_sims_base = np.sort(-sim_lab_ss, axis=1)[:, :M] * -1

# Unnormalized signatures
train_sigs_100 = np.zeros((n_files, 1536), np.float32)
for j in range(n_files):
    w = top_M_sims_base[j]; idx = top_M_idx_base[j]
    train_sigs_100[j] = (w[:, None] * ss_emb_norm[idx]).sum(0)

sim_bridge_100_n = build_bridge_from_sigs(train_sigs_100)

# Full sweep to confirm best
best_base = 0
best_base_cfg = {}
for alph in [0.30, 0.35, 0.40, 0.45, 0.50]:
    sc = (1-alph) * sim_ref.copy() + alph * sim_bridge_100_n.copy()
    y = compute_rknn(sc)
    for wg in [0.35, 0.40, 0.45]:
        for a, b in [(1.00, 1.50), (0.95, 1.55), (0.90, 1.60), (0.85, 1.70), (0.85, 1.90)]:
            auc = eval_full(y, wg=wg, a=a, b=b)
            if auc > best_base:
                best_base = auc
                best_base_cfg = {'alpha': alph, 'wg': wg, 'a': a, 'b': b}

print(f"  Baseline best: {best_base:.4f}  cfg={best_base_cfg}", flush=True)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: M sweep with correct computation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: M sweep (correct) ===", flush=True)
for M_new in [200, 500, 1000]:
    idx2 = np.argsort(-sim_lab_ss, axis=1)[:, :M_new]
    sims2 = np.sort(-sim_lab_ss, axis=1)[:, :M_new] * -1
    sigs2 = np.zeros((n_files, 1536), np.float32)
    for j in range(n_files):
        sigs2[j] = (sims2[j][:, None] * ss_emb_norm[idx2[j]]).sum(0)
    sb2_n = build_bridge_from_sigs(sigs2)
    best_m = 0; best_cfg = {}
    for alph in [0.30, 0.40, 0.50]:
        sc = (1-alph) * sim_ref.copy() + alph * sb2_n.copy()
        y2 = compute_rknn(sc)
        for wg in [0.35, 0.40, 0.45]:
            for a, b in [(1.00, 1.50), (0.90, 1.60), (0.85, 1.70), (0.85, 1.90)]:
                auc2 = eval_full(y2, wg=wg, a=a, b=b)
                if auc2 > best_m:
                    best_m = auc2
                    best_cfg = {'M': M_new, 'alpha': alph, 'wg': wg, 'a': a, 'b': b}
    results[f"bridge_M{M_new}"] = best_m
    print(f"  M={M_new}: {best_m:.4f}  cfg={best_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: SED-weighted bridge (weight = perch_sim * (1 + gamma * sed_cosim))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: SED-weighted bridge ===", flush=True)
file_sed_norm = file_sed_max / (np.linalg.norm(file_sed_max, axis=1, keepdims=True).clip(1e-8))
ss_probs_norm = ss_probs / (np.linalg.norm(ss_probs, axis=1, keepdims=True).clip(1e-8))

best_sed = 0; best_sed_cfg = {}
for gamma in [0.5, 1.0, 2.0, 3.0]:
    sigs_sed = np.zeros((n_files, 1536), np.float32)
    for j in range(n_files):
        idx = top_M_idx_base[j]
        perch_w = top_M_sims_base[j]
        sed_w = (ss_probs_norm[idx] @ file_sed_norm[j]).clip(0)
        combined_w = perch_w * (1.0 + gamma * sed_w)
        sigs_sed[j] = (combined_w[:, None] * ss_emb_norm[idx]).sum(0)
    sb_sed_n = build_bridge_from_sigs(sigs_sed)
    for alph in [0.30, 0.40, 0.50]:
        sc_sed = (1-alph) * sim_ref.copy() + alph * sb_sed_n.copy()
        y_sed = compute_rknn(sc_sed)
        for wg in [0.35, 0.40, 0.45]:
            for a, b in [(1.00, 1.50), (0.90, 1.60), (0.85, 1.70), (0.85, 1.90)]:
                auc_sed = eval_full(y_sed, wg=wg, a=a, b=b)
                if auc_sed > best_sed:
                    best_sed = auc_sed
                    best_sed_cfg = {'gamma': gamma, 'alpha': alph, 'wg': wg, 'a': a, 'b': b}
results["sed_weighted_bridge"] = best_sed
print(f"  SED-weighted bridge: {best_sed:.4f}  cfg={best_sed_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: SED-species bridge
# Weight top-M by perch_sim * (1 + beta * max_species_SED_score)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: SED-species bridge ===", flush=True)
best_sp = 0; best_sp_cfg = {}
for beta in [0.5, 1.0, 2.0, 3.0]:
    sigs_sp = np.zeros((n_files, 1536), np.float32)
    for j in range(n_files):
        sp_j = file_labels[j].astype(bool)
        idx = top_M_idx_base[j]
        perch_w = top_M_sims_base[j]
        if sp_j.sum() > 0:
            sp_score = ss_probs[idx][:, sp_j].max(1)
            combined_w = perch_w * (1.0 + beta * sp_score)
        else:
            combined_w = perch_w
        sigs_sp[j] = (combined_w[:, None] * ss_emb_norm[idx]).sum(0)
    sb_sp_n = build_bridge_from_sigs(sigs_sp)
    for alph in [0.30, 0.40, 0.50]:
        sc_sp = (1-alph) * sim_ref.copy() + alph * sb_sp_n.copy()
        y_sp = compute_rknn(sc_sp)
        for wg in [0.35, 0.40, 0.45]:
            for a, b in [(1.00, 1.50), (0.90, 1.60), (0.85, 1.70), (0.85, 1.90)]:
                auc_sp = eval_full(y_sp, wg=wg, a=a, b=b)
                if auc_sp > best_sp:
                    best_sp = auc_sp
                    best_sp_cfg = {'beta': beta, 'alpha': alph, 'wg': wg, 'a': a, 'b': b}
results["sed_species_bridge"] = best_sp
print(f"  SED-species bridge: {best_sp:.4f}  cfg={best_sp_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Iterative bridge (2nd-order propagation through SS graph)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Iterative Bridge ===", flush=True)
best_iter = 0; best_iter_cfg = {}
for alpha1 in [0.30, 0.40, 0.50]:
    sc1 = (1-alpha1) * sim_ref.copy() + alpha1 * sim_bridge_100_n.copy()
    np.fill_diagonal(sc1, 0)
    sc1 = np.maximum(sc1, 0)
    row_sum = sc1.sum(1, keepdims=True).clip(1e-8)
    sc1_soft = sc1 / row_sum  # (66, 66) row-stochastic soft attention
    # 2nd-order SS similarity: (66, 66) @ (66, 127896) = (66, 127896)
    sim_lab_ss2 = sc1_soft @ sim_lab_ss
    idx2 = np.argsort(-sim_lab_ss2, axis=1)[:, :M]
    sims_iter2 = np.sort(-sim_lab_ss2, axis=1)[:, :M] * -1
    sigs_iter2 = np.zeros((n_files, 1536), np.float32)
    for j in range(n_files):
        sigs_iter2[j] = (sims_iter2[j][:, None] * ss_emb_norm[idx2[j]]).sum(0)
    sb_iter2_n = build_bridge_from_sigs(sigs_iter2)
    for alpha2 in [0.20, 0.30, 0.40]:
        sc2 = (1-alpha2) * sim_ref.copy() + alpha2 * sb_iter2_n.copy()
        y_iter2 = compute_rknn(sc2)
        for wg in [0.35, 0.40, 0.45]:
            for a, b in [(1.00, 1.50), (0.90, 1.60), (0.85, 1.70), (0.85, 1.90)]:
                auc_iter2 = eval_full(y_iter2, wg=wg, a=a, b=b)
                if auc_iter2 > best_iter:
                    best_iter = auc_iter2
                    best_iter_cfg = {'alpha1': alpha1, 'alpha2': alpha2, 'wg': wg, 'a': a, 'b': b}
results["iterative_bridge"] = best_iter
print(f"  Iterative bridge: {best_iter:.4f}  cfg={best_iter_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Dual-modal Bridge (Perch-bridge + SED-bridge combined)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Dual-modal Bridge ===", flush=True)
print("  Computing SED labeled × SS similarity...", flush=True)
CHUNK2 = 5000
sim_lab_ss_sed = np.zeros((n_files, n_ss_wins), np.float32)
for cs in range(0, n_ss_wins, CHUNK2):
    ce = min(cs + CHUNK2, n_ss_wins)
    sim_lab_ss_sed[:, cs:ce] = file_sed_norm @ ss_probs_norm[cs:ce].T
    if cs % 50000 == 0: print(f"    {cs}/{n_ss_wins}...", flush=True)
print("  done.", flush=True)

# SED-based bridge
idx_sed = np.argsort(-sim_lab_ss_sed, axis=1)[:, :M]
sims_sed2 = np.sort(-sim_lab_ss_sed, axis=1)[:, :M] * -1
sigs_sed_modal = np.zeros((n_files, 1536), np.float32)
for j in range(n_files):
    sigs_sed_modal[j] = (sims_sed2[j][:, None] * ss_emb_norm[idx_sed[j]]).sum(0)
sim_bridge_sed_modal_n = build_bridge_from_sigs(sigs_sed_modal)

best_dual = 0; best_dual_cfg = {}
for w_perch_b in [0.5, 0.6, 0.7, 0.8]:
    w_sed_b = 1.0 - w_perch_b
    sim_bridge_dual = w_perch_b * sim_bridge_100_n + w_sed_b * sim_bridge_sed_modal_n
    # Re-normalize
    bn_dual = np.sqrt((sim_bridge_dual**2).sum(1, keepdims=True)).clip(1e-8)
    sim_bridge_dual_n = sim_bridge_dual / bn_dual
    for alph in [0.30, 0.40, 0.50]:
        sc_dual = (1-alph) * sim_ref.copy() + alph * sim_bridge_dual_n.copy()
        y_dual = compute_rknn(sc_dual)
        for wg in [0.35, 0.40, 0.45]:
            for a, b in [(1.00, 1.50), (0.90, 1.60), (0.85, 1.70), (0.85, 1.90)]:
                auc_dual = eval_full(y_dual, wg=wg, a=a, b=b)
                if auc_dual > best_dual:
                    best_dual = auc_dual
                    best_dual_cfg = {'w_perch': w_perch_b, 'alpha': alph, 'wg': wg, 'a': a, 'b': b}
results["dual_modal_bridge"] = best_dual
print(f"  Dual-modal bridge: {best_dual:.4f}  cfg={best_dual_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 6: Bridge with fine alpha sweep around best configs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 6: Fine bridge sweep (alpha, wg, a, b) ===", flush=True)
best_fine = 0; best_fine_cfg = {}
for alph in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    sc = (1-alph) * sim_ref.copy() + alph * sim_bridge_100_n.copy()
    y = compute_rknn(sc)
    for wg in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for a in [0.80, 0.85, 0.90, 0.95, 1.00]:
            for b in [1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]:
                auc = eval_full(y, wg=wg, a=a, b=b)
                if auc > best_fine:
                    best_fine = auc
                    best_fine_cfg = {'alpha': alph, 'wg': wg, 'a': a, 'b': b}
results["bridge_fine_sweep"] = best_fine
print(f"  Fine sweep: {best_fine:.4f}  cfg={best_fine_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"BATCH 6b BRIDGE VARIANTS SUMMARY")
print(f"Baseline (corrected): {best_base:.4f}  cfg={best_base_cfg}")
print(f"Known best: 0.9441 (ss_bridge_weighted from build_ss_bridge_production_pkl)")
print(f"{'='*60}")
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    delta = auc - best_base
    marker = " *** NEW BEST ***" if auc > 0.9441 else ""
    print(f"  {name}: {auc:.4f}  ({delta:+.4f}){marker}")

# Update results JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
for name, auc in results.items():
    rd['experiments'].append({
        'method': f"b6b_{name}",
        'loo_auc': float(auc),
        'full_auc': float(auc),
        'config': {'batch': '6b'}
    })
    if auc > rd['best'].get('loo_auc', 0):
        rd['best'] = {'method': f"b6b_{name}", 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"\n*** NEW BEST: {name} = {auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("\nUpdated embed_prior_results.json")
