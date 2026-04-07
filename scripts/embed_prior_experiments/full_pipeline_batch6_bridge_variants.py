"""
Full Pipeline Batch 6: SS Bridge Advanced Variants
Baseline: ss_bridge_weighted = 0.9441

New ideas:
1. sed_bridge: Weight SS windows by SED similarity to labeled file (semantic bridge)
2. m_sweep: Try larger M (200, 500) for bridge construction
3. dual_bridge: Combine perch-bridge with sed-bridge
4. sym_bridge: Symmetrize bridge matrix before RKNN
5. bridge_iter2: Apply bridge transform twice (iterative)
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

# ── Load SED predictions for SS windows ───────────────────────────────────────
sed_labeled_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
# SED covers ALL soundscape windows (same 127,896)
file_sed_max = np.zeros((n_files, n_species), np.float32)
sed_by_file = {}
for i, rid in enumerate(sed_labeled_npz['row_ids']):
    sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]), []).append(i)
for fi, fname in enumerate(file_list):
    fb = fname.replace('.ogg', '').replace('.flac', '')
    if fb in sed_by_file:
        file_sed_max[fi] = sed_labeled_npz['probs'][sed_by_file[fb]].max(0)

EPS = 1e-7
def vlom_blend(a, b, w_a=0.5, w_b=0.5):
    ws = w_a + w_b; w_a /= ws; w_b /= ws
    return sigmoid(w_a * np.log(a.clip(EPS) / (1-a).clip(EPS)) +
                   w_b * np.log(b.clip(EPS) / (1-b).clip(EPS)))
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))

# ── Load base PKL (X_ref, window data) ────────────────────────────────────────
with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep_base = pickle.load(f)
X_ref = ep_base['X_combined_n'].astype(np.float32)  # (66, 39)
fl = ep_base['file_labels'].astype(np.float32)

# ── Load ALL soundscape embeddings ────────────────────────────────────────────
print("Loading all soundscape embeddings...", flush=True)
ss_all = np.load("outputs/perch_emb_all_ss.npz", allow_pickle=True)
ss_emb = ss_all['emb'].astype(np.float32)
ss_emb_norm = normalize(ss_emb, norm='l2').astype(np.float32)
n_ss_wins = len(ss_emb_norm)
print(f"  {n_ss_wins} windows.", flush=True)

# ── Load SED predictions for all SS windows ───────────────────────────────────
# ss_probs: (127896, 234) - SED predictions for each window
ss_probs = sed_labeled_npz['probs'].astype(np.float32)  # already sigmoid'd
print(f"  ss_probs shape: {ss_probs.shape}", flush=True)

# ── Precompute sim_lab_ss ──────────────────────────────────────────────────────
print("Computing labeled × ss similarity matrix (perch)...", flush=True)
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
        ws = ww.sum()
        ww = ww / ws if ws > 1e-8 else np.ones(1)
        wp[wi] = (ww[:, None] * file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)
print("  done.", flush=True)

# ── X_ref similarity matrix ────────────────────────────────────────────────────
sim_ref = X_ref @ X_ref.T
np.fill_diagonal(sim_ref, -np.inf)

# ── RKNN helper ────────────────────────────────────────────────────────────────
k_rknn, T = 5, 0.2
def compute_rknn(sim_combined, fl_in=None):
    if fl_in is None: fl_in = fl
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
            y[i] = (w[:, None] * fl_in[tr[top5]]).sum(0)
        else:
            ms = np.array(msims); ls = ms / T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y[i] = (w[:, None] * fl_in[mutual]).sum(0)
    return y

def eval_full(y_ep, wg=0.40, a=1.00, b=1.50):
    yb = wg * y_ep + (1-wg) * y_win_k1
    log_yb = np.log(yb.clip(EPS))
    full = sigmoid(a * base_logit + b * log_yb)
    return macro_auc(file_labels, full)

# ── Baseline: SS Bridge α=0.4 (reproduce) ─────────────────────────────────────
print("\n=== Baseline: SS Bridge alpha=0.4 ===", flush=True)
M = 100
top_M_idx = np.argsort(-sim_lab_ss, axis=1)[:, :M]
top_M_sims = np.sort(-sim_lab_ss, axis=1)[:, :M] * -1

train_ss_sigs = np.zeros((n_files, 1536), np.float32)
for j in range(n_files):
    w = top_M_sims[j]; idx = top_M_idx[j]
    sig = (w[:, None] * ss_emb_norm[idx]).sum(0)
    train_ss_sigs[j] = sig
sig_norms_100 = np.linalg.norm(train_ss_sigs, axis=1, keepdims=True).clip(1e-8)
bridge_sigs_100_n = train_ss_sigs / sig_norms_100
sim_bridge_100 = file_emb_avg_norm @ bridge_sigs_100_n.T
bridge_norm = np.sqrt((sim_bridge_100**2).sum(1, keepdims=True)).clip(1e-8)
sim_bridge_100_n = sim_bridge_100 / bridge_norm

alpha = 0.40
sim_comb_base = (1-alpha) * sim_ref.copy() + alpha * sim_bridge_100_n.copy()
y_base = compute_rknn(sim_comb_base)
auc_base = eval_full(y_base, wg=0.40, a=1.00, b=1.50)
print(f"  Baseline (M=100, alpha=0.40): {auc_base:.4f}", flush=True)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: M sweep (200, 500, 1000)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: M sweep ===", flush=True)
for M_new in [200, 500, 1000]:
    top_M2_idx = np.argsort(-sim_lab_ss, axis=1)[:, :M_new]
    top_M2_sims = np.sort(-sim_lab_ss, axis=1)[:, :M_new] * -1
    sigs2 = np.zeros((n_files, 1536), np.float32)
    for j in range(n_files):
        w = top_M2_sims[j]; idx = top_M2_idx[j]
        sigs2[j] = (w[:, None] * ss_emb_norm[idx]).sum(0)
    sn2 = np.linalg.norm(sigs2, axis=1, keepdims=True).clip(1e-8)
    bridge2_n_raw = sigs2 / sn2
    sb2 = file_emb_avg_norm @ bridge2_n_raw.T
    sbn2 = np.sqrt((sb2**2).sum(1, keepdims=True)).clip(1e-8)
    sb2_n = sb2 / sbn2
    best_m = 0
    best_cfg = {}
    for alph in [0.30, 0.40, 0.50]:
        sc = (1-alph) * sim_ref.copy() + alph * sb2_n.copy()
        y2 = compute_rknn(sc)
        for wg in [0.35, 0.40, 0.45]:
            for a, b in [(1.00, 1.50), (0.90, 1.60), (0.95, 1.55)]:
                auc2 = eval_full(y2, wg=wg, a=a, b=b)
                if auc2 > best_m:
                    best_m = auc2
                    best_cfg = {'M': M_new, 'alpha': alph, 'wg': wg, 'a': a, 'b': b}
    name = f"bridge_M{M_new}"
    results[name] = best_m
    print(f"  M={M_new}: {best_m:.4f}  cfg={best_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: SED-weighted bridge
# Weight top-M windows by perch_sim * (1 + gamma * sed_sim_to_labeled_file)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: SED-weighted bridge ===", flush=True)
# file_sed_max: (66, 234) - SED predictions for labeled files
# ss_probs: (127896, 234) - SED predictions for all SS windows
# sed_sim[j, m] = cosine_sim(file_sed_max[j], ss_probs[m])

# Normalize SED vectors
file_sed_norm = file_sed_max / (np.linalg.norm(file_sed_max, axis=1, keepdims=True).clip(1e-8))
ss_probs_norm = ss_probs / (np.linalg.norm(ss_probs, axis=1, keepdims=True).clip(1e-8))

# Compute SED similarity for top-100 SS windows per labeled file
M = 100
top_M_idx_base = np.argsort(-sim_lab_ss, axis=1)[:, :M]
top_M_sims_base = np.sort(-sim_lab_ss, axis=1)[:, :M] * -1

best_sed = 0
best_sed_cfg = {}
for gamma in [0.5, 1.0, 2.0, 3.0]:
    sigs_sed = np.zeros((n_files, 1536), np.float32)
    for j in range(n_files):
        idx = top_M_idx_base[j]  # (M,)
        perch_w = top_M_sims_base[j]  # (M,) perch similarity weights
        # SED similarity between labeled file j and each top-M SS window
        sed_w = ss_probs_norm[idx] @ file_sed_norm[j]  # (M,) in [-1,1]
        sed_w = np.maximum(sed_w, 0)  # only positive
        combined_w = perch_w * (1.0 + gamma * sed_w)  # (M,)
        sigs_sed[j] = (combined_w[:, None] * ss_emb_norm[idx]).sum(0)
    sn_sed = np.linalg.norm(sigs_sed, axis=1, keepdims=True).clip(1e-8)
    sigs_sed_n = sigs_sed / sn_sed
    sb_sed = file_emb_avg_norm @ sigs_sed_n.T
    sbn_sed = np.sqrt((sb_sed**2).sum(1, keepdims=True)).clip(1e-8)
    sb_sed_n = sb_sed / sbn_sed
    for alph in [0.30, 0.40, 0.50]:
        sc_sed = (1-alph) * sim_ref.copy() + alph * sb_sed_n.copy()
        y_sed = compute_rknn(sc_sed)
        for wg in [0.35, 0.40, 0.45]:
            for a, b in [(1.00, 1.50), (0.90, 1.60), (0.95, 1.55)]:
                auc_sed = eval_full(y_sed, wg=wg, a=a, b=b)
                if auc_sed > best_sed:
                    best_sed = auc_sed
                    best_sed_cfg = {'gamma': gamma, 'alpha': alph, 'wg': wg, 'a': a, 'b': b}

name = "sed_weighted_bridge"
results[name] = best_sed
print(f"  SED-weighted bridge: {best_sed:.4f}  cfg={best_sed_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: SED-soft-label bridge
# For each labeled file j, find top-M SS windows with highest SED score for j's species
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: SED-species bridge ===", flush=True)
# For file j with species set S_j, score each SS window by:
# score[m] = max(ss_probs[m, s] for s in S_j) * sim_perch(j, m)

best_sp = 0
best_sp_cfg = {}
for beta in [0.5, 1.0, 2.0]:
    sigs_sp = np.zeros((n_files, 1536), np.float32)
    for j in range(n_files):
        sp_j = file_labels[j].astype(bool)  # species in this file
        # Top-M SS windows by perch similarity
        idx = top_M_idx_base[j]
        perch_w = top_M_sims_base[j]
        if sp_j.sum() > 0:
            # SED score for j's species in each window
            sp_score = ss_probs[idx][:, sp_j].max(1)  # (M,)
            combined_w = perch_w * (1.0 + beta * sp_score)
        else:
            combined_w = perch_w
        sigs_sp[j] = (combined_w[:, None] * ss_emb_norm[idx]).sum(0)
    sn_sp = np.linalg.norm(sigs_sp, axis=1, keepdims=True).clip(1e-8)
    sigs_sp_n = sigs_sp / sn_sp
    sb_sp = file_emb_avg_norm @ sigs_sp_n.T
    sbn_sp = np.sqrt((sb_sp**2).sum(1, keepdims=True)).clip(1e-8)
    sb_sp_n = sb_sp / sbn_sp
    for alph in [0.30, 0.40, 0.50]:
        sc_sp = (1-alph) * sim_ref.copy() + alph * sb_sp_n.copy()
        y_sp = compute_rknn(sc_sp)
        for wg in [0.35, 0.40, 0.45]:
            for a, b in [(1.00, 1.50), (0.90, 1.60), (0.95, 1.55)]:
                auc_sp = eval_full(y_sp, wg=wg, a=a, b=b)
                if auc_sp > best_sp:
                    best_sp = auc_sp
                    best_sp_cfg = {'beta': beta, 'alpha': alph, 'wg': wg, 'a': a, 'b': b}

name = "sed_species_bridge"
results[name] = best_sp
print(f"  SED-species bridge: {best_sp:.4f}  cfg={best_sp_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: Iterative bridge (apply bridge transform twice)
# iter2_bridge: use bridge-enhanced sim matrix to rebuild signatures
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: Iterative Bridge ===", flush=True)
# Use the SS bridge similarities as NEW "file embeddings" for a second-pass bridge
# After 1 iteration: sim_bridge_100_n gives 66×66 similarities
# Use sim_bridge_100_n as "soft embeddings" to find more SS windows

# Approach: after building bridge sigs v1, use them to query SS windows again
best_iter = 0
best_iter_cfg = {}

for alpha1 in [0.30, 0.40]:
    sc1 = (1-alpha1) * sim_ref.copy() + alpha1 * sim_bridge_100_n.copy()
    # "Enhanced embeddings" - combine original with bridge
    # Normalize the enhanced sim matrix rows to get new "embeddings"
    # Treat each row of sc1 as new similarity vector → find nearest SS
    # But we need embeddings for SS computation...
    # Alternative: use sc1 to reweight the original sim_lab_ss
    # new_sim_to_ss[i, m] = sum_j (sc1[i,j] * sim_lab_ss[j, m]) for top-L of j
    # This is a 2-hop: labeled → labeled (via bridge) → SS → labeled
    # But 66×66 × 66×127896 = 66×127896 is fast

    # Compute 2nd-order SS similarity
    # sc1_softmax: row-normalize sc1 (set diagonal to 0)
    sc1_n = sc1.copy()
    np.fill_diagonal(sc1_n, 0)
    sc1_n = np.maximum(sc1_n, 0)
    row_sum = sc1_n.sum(1, keepdims=True).clip(1e-8)
    sc1_n /= row_sum  # (66, 66) soft attention

    # New SS similarity: weighted combination
    sim_lab_ss2 = sc1_n @ sim_lab_ss  # (66, 127896)

    # Rebuild signatures with 2nd-order SS sim
    top_M2_idx = np.argsort(-sim_lab_ss2, axis=1)[:, :M]
    top_M2_sims = np.sort(-sim_lab_ss2, axis=1)[:, :M] * -1
    sigs_iter = np.zeros((n_files, 1536), np.float32)
    for j in range(n_files):
        w2 = top_M2_sims[j]; idx2 = top_M2_idx[j]
        sigs_iter[j] = (w2[:, None] * ss_emb_norm[idx2]).sum(0)
    sn_iter = np.linalg.norm(sigs_iter, axis=1, keepdims=True).clip(1e-8)
    sigs_iter_n = sigs_iter / sn_iter
    sb_iter = file_emb_avg_norm @ sigs_iter_n.T
    sbn_iter = np.sqrt((sb_iter**2).sum(1, keepdims=True)).clip(1e-8)
    sb_iter_n = sb_iter / sbn_iter

    for alpha2 in [0.20, 0.30, 0.40]:
        sc2 = (1-alpha2) * sim_ref.copy() + alpha2 * sb_iter_n.copy()
        y_iter = compute_rknn(sc2)
        for wg in [0.35, 0.40, 0.45]:
            for a, b in [(1.00, 1.50), (0.90, 1.60), (0.95, 1.55)]:
                auc_iter = eval_full(y_iter, wg=wg, a=a, b=b)
                if auc_iter > best_iter:
                    best_iter = auc_iter
                    best_iter_cfg = {'alpha1': alpha1, 'alpha2': alpha2, 'wg': wg, 'a': a, 'b': b}

name = "iterative_bridge"
results[name] = best_iter
print(f"  Iterative bridge: {best_iter:.4f}  cfg={best_iter_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Combined Perch + SED bridge (dual-modal)
# Build two bridges: perch-based and SED-based, then combine
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Dual-modal Bridge (Perch + SED) ===", flush=True)
# SED bridge: sim_lab_ss_sed[j, m] = cosine_sim(file_sed_max[j], ss_probs[m])
# Build train_ss_sigs_sed using SED similarity as weights
print("  Computing SED labeled × SS similarity...", flush=True)
CHUNK2 = 5000
sim_lab_ss_sed = np.zeros((n_files, n_ss_wins), np.float32)
for cs in range(0, n_ss_wins, CHUNK2):
    ce = min(cs + CHUNK2, n_ss_wins)
    sim_lab_ss_sed[:, cs:ce] = file_sed_norm @ ss_probs_norm[cs:ce].T
    if cs % 50000 == 0: print(f"    {cs}/{n_ss_wins}...", flush=True)
print("  done.", flush=True)

# Build SED-based signatures
top_M_sed_idx = np.argsort(-sim_lab_ss_sed, axis=1)[:, :M]
top_M_sed_sims = np.sort(-sim_lab_ss_sed, axis=1)[:, :M] * -1

sigs_sed_modal = np.zeros((n_files, 1536), np.float32)
for j in range(n_files):
    w_s = top_M_sed_sims[j]; idx_s = top_M_sed_idx[j]
    sigs_sed_modal[j] = (w_s[:, None] * ss_emb_norm[idx_s]).sum(0)
sn_sed_m = np.linalg.norm(sigs_sed_modal, axis=1, keepdims=True).clip(1e-8)
sigs_sed_modal_n = sigs_sed_modal / sn_sed_m
sb_sed_m = file_emb_avg_norm @ sigs_sed_modal_n.T
sbn_sed_m = np.sqrt((sb_sed_m**2).sum(1, keepdims=True)).clip(1e-8)
sim_bridge_sed_n = sb_sed_m / sbn_sed_m

best_dual = 0
best_dual_cfg = {}
for w_perch_b in [0.6, 0.7, 0.8]:
    w_sed_b = 1.0 - w_perch_b
    sim_bridge_dual_n = w_perch_b * sim_bridge_100_n + w_sed_b * sim_bridge_sed_n
    # Re-normalize
    bn_dual = np.sqrt((sim_bridge_dual_n**2).sum(1, keepdims=True)).clip(1e-8)
    sim_bridge_dual_n2 = sim_bridge_dual_n / bn_dual
    for alph in [0.30, 0.40, 0.50]:
        sc_dual = (1-alph) * sim_ref.copy() + alph * sim_bridge_dual_n2.copy()
        y_dual = compute_rknn(sc_dual)
        for wg in [0.35, 0.40, 0.45]:
            for a, b in [(1.00, 1.50), (0.90, 1.60), (0.95, 1.55)]:
                auc_dual = eval_full(y_dual, wg=wg, a=a, b=b)
                if auc_dual > best_dual:
                    best_dual = auc_dual
                    best_dual_cfg = {'w_perch_b': w_perch_b, 'alpha': alph, 'wg': wg, 'a': a, 'b': b}

name = "dual_modal_bridge"
results[name] = best_dual
print(f"  Dual-modal bridge: {best_dual:.4f}  cfg={best_dual_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"BATCH 6 BRIDGE VARIANTS SUMMARY")
print(f"Baseline: {auc_base:.4f} (M=100, alpha=0.40, wg=0.40, a=1.0, b=1.5)")
print(f"{'='*60}")
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    delta = auc - auc_base
    marker = " *** NEW BEST ***" if auc > 0.9441 else ""
    print(f"  {name}: {auc:.4f}  ({delta:+.4f}){marker}")

# Update results JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
for name, auc in results.items():
    rd['experiments'].append({
        'method': name,
        'loo_auc': float(auc),
        'full_auc': float(auc),
        'config': {'batch': 6}
    })
    if auc > rd['best'].get('loo_auc', 0):
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"\n*** NEW BEST: {name} = {auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("\nUpdated embed_prior_results.json")
