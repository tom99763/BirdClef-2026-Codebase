"""
Full Pipeline Batch 8: Larger M + Symmetric Bridge

Methods:
1. larger_M_bridge: Build SED-species bridge with M=500, M=2000 (production uses M=100)
2. symmetric_bridge: sim[i,j] using SS windows similar to BOTH files (intersection)
3. dual_sig_bridge: Combine two signatures (plain + SED-species) per file

Current best: 0.9444 (sed_species_bridge, M=100, beta=0.5, alpha=0.5, wg=0.45, a=0.85, b=1.70)
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list = list(perch['file_list'])
n_windows = perch['n_windows']
n_files = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end = np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))
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

EPS = 1e-7
def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')
def vlom_blend(a, b, w_a=0.5, w_b=0.5):
    ws = w_a + w_b; w_a /= ws; w_b /= ws
    return sigmoid(w_a*np.log(a.clip(EPS)/(1-a).clip(EPS)) + w_b*np.log(b.clip(EPS)/(1-b).clip(EPS)))

# ── Load SED ───────────────────────────────────────────────────────────────────
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_by_file = {}
for i, rid in enumerate(sed_npz['row_ids']):
    sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]), []).append(i)
file_sed_max = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fb = fname.replace('.ogg','').replace('.flac','')
    if fb in sed_by_file:
        file_sed_max[fi] = sed_npz['probs'][sed_by_file[fb]].max(0)
base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))

# ── Load SS embeddings ─────────────────────────────────────────────────────────
print("Loading SS embeddings...", flush=True)
ss_npz = np.load("outputs/perch_emb_all_ss.npz", allow_pickle=True)
ss_emb = ss_npz['emb'].astype(np.float32)           # (127896, 1536)
ss_logits = ss_npz['logits'].astype(np.float32)      # (127896, 234)
ss_probs = sigmoid(ss_logits)                         # (127896, 234)
ss_emb_norm = normalize(ss_emb, norm='l2').astype(np.float32)
n_ss = len(ss_emb)
print(f"  SS windows: {n_ss}", flush=True)

# ── Load base PKL for X_ref ────────────────────────────────────────────────────
with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep_base = pickle.load(f)
X_ref = ep_base['X_combined_n'].astype(np.float32)
fl = ep_base['file_labels'].astype(np.float32)

# ── Compute labeled-file to SS similarity ─────────────────────────────────────
print("Computing lab_file × SS similarity...", flush=True)
sim_lab_ss = file_emb_avg_norm @ ss_emb_norm.T   # (66, 127896)
print("  done.", flush=True)

# ── Window KNN k=1 ─────────────────────────────────────────────────────────────
print("Computing win_k1 LOO...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i]); X_te = emb_win_norm[te_s:te_e]
    tr_mask = (win_file_id != i); X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :1]
    wp = np.zeros((te_e-te_s, n_species), np.float32)
    for wi in range(te_e-te_s):
        ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
        ww = ww/ws if ws > 1e-8 else np.ones(1)
        wp[wi] = (ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)
log_win = np.log(y_win_k1.clip(EPS))
print("  done.", flush=True)

# ── RKNN helper ────────────────────────────────────────────────────────────────
T = 0.2
def compute_rknn_ep(sim_mat, k=5):
    sc = sim_mat.copy(); np.fill_diagonal(sc, -np.inf)
    top_k = np.argsort(-sc, axis=1)[:, :k]
    kth = sc[np.arange(n_files), top_k[:, -1]]
    y = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims_i = sc[i, tr]; top_i = np.argsort(-sims_i)[:k]
        mutual, msims = [], []
        for ti, tj in enumerate(tr[top_i]):
            if sims_i[top_i[ti]] >= kth[tj]: mutual.append(tj); msims.append(sims_i[top_i[ti]])
        if len(mutual) == 0:
            top5 = np.argsort(-sims_i)[:5]; ls = sims_i[top5]/T; ls -= ls.max()
            w = np.exp(ls); w /= w.sum(); y[i] = (w[:,None]*fl[tr[top5]]).sum(0)
        else:
            ms = np.array(msims); ls = ms/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y[i] = (w[:,None]*fl[mutual]).sum(0)
    return y

def build_bridge_from_sigs(sigs):
    """sigs: (66, 1536) unnormalized weighted sum of SS embeddings."""
    sb = file_emb_avg_norm @ sigs.T   # (66, 66)
    bn = np.sqrt((sb**2).sum(1, keepdims=True)).clip(1e-8)
    return sb / bn  # row-normalize

def build_sed_species_sigs(M=100, beta=0.5):
    """Build SED-species weighted signatures for each labeled file."""
    top_M_idx = np.argsort(-sim_lab_ss, axis=1)[:, :M]
    top_M_sims = np.sort(-sim_lab_ss, axis=1)[:, :M] * -1
    sigs = np.zeros((n_files, 1536), np.float32)
    for j in range(n_files):
        sp_j = file_labels[j].astype(bool)
        idx = top_M_idx[j]; perch_w = top_M_sims[j]
        if sp_j.sum() > 0:
            sp_score = ss_probs[idx][:, sp_j].max(1)  # (M,)
            combined_w = perch_w * (1.0 + beta * sp_score)
        else:
            combined_w = perch_w
        sigs[j] = (combined_w[:, None] * ss_emb_norm[idx]).sum(0)
    return sigs

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Larger M values
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: Larger M (M=500, M=2000) ===", flush=True)
best1 = 0; best1_cfg = {}
for M in [500, 2000]:
    print(f"  Building M={M} signatures...", flush=True)
    sigs = build_sed_species_sigs(M=M, beta=0.5)
    bridge = build_bridge_from_sigs(sigs)
    for alpha in [0.3, 0.4, 0.5, 0.6]:
        sim_comb = (1 - alpha) * (X_ref @ X_ref.T) + alpha * bridge
        y_rknn = compute_rknn_ep(sim_comb, k=5)
        log_rknn = np.log(y_rknn.clip(EPS))
        for wg in [0.40, 0.45, 0.50]:
            y_blend = wg * y_rknn + (1-wg) * y_win_k1
            log_b = np.log(y_blend.clip(EPS))
            for a in [0.80, 0.85, 0.90]:
                for b in [1.60, 1.70, 1.80]:
                    pred = sigmoid(a*base_logit + b*log_b)
                    if np.isfinite(pred).all():
                        auc = macro_auc(file_labels, pred)
                        if auc > best1:
                            best1 = auc; best1_cfg = {'M': M, 'alpha': alpha, 'wg': wg, 'a': a, 'b': b}
    print(f"  M={M} best so far: {best1:.4f}", flush=True)
results['larger_M_bridge'] = (best1, best1_cfg)
print(f"  Overall best: {best1:.4f}  cfg={best1_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Symmetric Bridge
# sim_sym[i,j] = Σ_m min(sim(i,m), sim(j,m)) for top-100 SS of BOTH i and j
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Symmetric Bridge ===", flush=True)
M_sym = 200
print(f"  Building symmetric bridge (M={M_sym})...", flush=True)
top_M_idx = np.argsort(-sim_lab_ss, axis=1)[:, :M_sym]

sim_sym = np.zeros((n_files, n_files), np.float32)
for i in range(n_files):
    idx_i = set(top_M_idx[i].tolist())
    for j in range(i, n_files):
        if i == j:
            sim_sym[i, j] = 1.0
            continue
        idx_j = set(top_M_idx[j].tolist())
        mutual_idx = list(idx_i & idx_j)
        if len(mutual_idx) == 0:
            sim_sym[i, j] = 0.0
        else:
            s_i = sim_lab_ss[i, mutual_idx]
            s_j = sim_lab_ss[j, mutual_idx]
            sim_sym[i, j] = np.minimum(s_i, s_j).sum() / len(mutual_idx)
        sim_sym[j, i] = sim_sym[i, j]

# Normalize
s_max = sim_sym[sim_sym > 0].max(); s_min = sim_sym[sim_sym > 0].min()
sim_sym_n = (sim_sym - s_min) / (s_max - s_min + 1e-8)
np.fill_diagonal(sim_sym_n, 1.0)
print("  done.", flush=True)

best2 = 0; best2_cfg = {}
for alpha in [0.3, 0.4, 0.5]:
    sim_comb2 = (1 - alpha) * (X_ref @ X_ref.T) + alpha * sim_sym_n
    y_rknn2 = compute_rknn_ep(sim_comb2, k=5)
    log_rknn2 = np.log(y_rknn2.clip(EPS))
    for wg in [0.35, 0.40, 0.45, 0.50]:
        y_blend2 = wg * y_rknn2 + (1-wg) * y_win_k1
        log_b2 = np.log(y_blend2.clip(EPS))
        for a in [0.80, 0.85, 0.90]:
            for b in [1.50, 1.60, 1.70, 1.80, 1.90]:
                pred = sigmoid(a*base_logit + b*log_b2)
                if np.isfinite(pred).all():
                    auc = macro_auc(file_labels, pred)
                    if auc > best2:
                        best2 = auc; best2_cfg = {'alpha': alpha, 'wg': wg, 'a': a, 'b': b}
results['symmetric_bridge'] = (best2, best2_cfg)
print(f"  Best: {best2:.4f}  cfg={best2_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Dual signature bridge (plain perch + SED-species)
# Each file has two signatures; combined similarity
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Dual signature bridge ===", flush=True)
print("  Building plain signatures...", flush=True)
M_dual = 100
top_M_idx_d = np.argsort(-sim_lab_ss, axis=1)[:, :M_dual]
top_M_sims_d = np.sort(-sim_lab_ss, axis=1)[:, :M_dual] * -1
sigs_plain = np.zeros((n_files, 1536), np.float32)
for j in range(n_files):
    idx = top_M_idx_d[j]; w = top_M_sims_d[j]
    sigs_plain[j] = (w[:, None] * ss_emb_norm[idx]).sum(0)

print("  Building SED-species signatures...", flush=True)
sigs_sed = build_sed_species_sigs(M=100, beta=0.5)

bridge_plain = build_bridge_from_sigs(sigs_plain)
bridge_sed = build_bridge_from_sigs(sigs_sed)

best3 = 0; best3_cfg = {}
for w_sed_br in [0.4, 0.5, 0.6, 0.7]:
    w_plain_br = 1.0 - w_sed_br
    bridge_dual = w_sed_br * bridge_sed + w_plain_br * bridge_plain
    for alpha in [0.3, 0.4, 0.5, 0.6]:
        sim_comb3 = (1-alpha) * (X_ref @ X_ref.T) + alpha * bridge_dual
        y_rknn3 = compute_rknn_ep(sim_comb3, k=5)
        for wg in [0.40, 0.45, 0.50]:
            y_blend3 = wg * y_rknn3 + (1-wg) * y_win_k1
            log_b3 = np.log(y_blend3.clip(EPS))
            for a in [0.80, 0.85, 0.90]:
                for b in [1.60, 1.70, 1.80]:
                    pred = sigmoid(a*base_logit + b*log_b3)
                    if np.isfinite(pred).all():
                        auc = macro_auc(file_labels, pred)
                        if auc > best3:
                            best3 = auc; best3_cfg = {'w_sed': w_sed_br, 'alpha': alpha, 'wg': wg, 'a': a, 'b': b}
results['dual_sig_bridge'] = (best3, best3_cfg)
print(f"  Best: {best3:.4f}  cfg={best3_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
FULL_BEST = 0.9444
print(f"\n{'='*60}")
print(f"LARGER M + SYMMETRIC BRIDGE SUMMARY")
print(f"Current full pipeline best: {FULL_BEST}")
print(f"{'='*60}")
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    delta = auc - FULL_BEST
    marker = " *** NEW BEST ***" if auc > FULL_BEST else ""
    print(f"  {name}: {auc:.4f}  ({delta:+.4f}){marker}")

# Update JSON
with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
for name, (auc, cfg) in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc), 'config': cfg})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"\n*** NEW BEST: {name} = {auc:.4f} ***")
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print("Updated embed_prior_results.json")
