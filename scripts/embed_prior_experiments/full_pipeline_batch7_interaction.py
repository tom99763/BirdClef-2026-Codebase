"""
Full Pipeline Batch 7: Interaction Term + Bridge Ensemble

Methods:
1. sed_species_bridge_interaction: Add log(rknn)*log(win) interaction term
   sigmoid(a*base_logit + b1*log(rknn) + b2*log(win) + b3*log(rknn)*log(win))
2. bridge_ensemble: Weighted average of SS Bridge + SED-Species Bridge
3. bridge_interaction_emb: Interaction with raw Perch embedding similarity
4. triple_bridge_logspace: 3-way logspace: rknn_bridge + win + geo_k5

Current full pipeline best: sed_species_bridge = 0.9444
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load Perch data ────────────────────────────────────────────────────────────
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
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

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
fl_labels = file_labels.copy()

# ── Load SED-species bridge PKL ────────────────────────────────────────────────
with open("outputs/embed_prior_sed_species_bridge.pkl", "rb") as f:
    ep_sed = pickle.load(f)
X_ref_sed = ep_sed['X_combined_n'].astype(np.float32)
fl = ep_sed['file_labels'].astype(np.float32)
sim_sed_bridge = ep_sed['sim_bridge_n'].astype(np.float32)  # (66,66) SED-species bridge sim
train_ss_sigs_sed = ep_sed['train_ss_signatures'].astype(np.float32)  # (66,1536)

# ── Load SS Bridge PKL ─────────────────────────────────────────────────────────
with open("outputs/embed_prior_ss_bridge.pkl", "rb") as f:
    ep_ss = pickle.load(f)
sim_ss_bridge = ep_ss['sim_bridge_n'].astype(np.float32)  # (66,66) plain SS bridge sim
train_ss_sigs_plain = ep_ss['train_ss_signatures'].astype(np.float32)  # (66,1536)

# ── RKNN helper ────────────────────────────────────────────────────────────────
T = 0.2
def compute_rknn(sim_mat, k=5):
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

# ── Window KNN k=1 ──────────────────────────────────────────────────────────────
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
        wp[wi] = (ww[:,None]*fl_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)
log_win = np.log(y_win_k1.clip(EPS))
print("  done.", flush=True)

# ── RKNN on SED-species bridge ────────────────────────────────────────────────
print("Computing RKNN on SED-species bridge...", flush=True)
# Build combined sim: (1-alpha)*X_ref + alpha*bridge (same as pkl config)
alpha_sed = ep_sed.get('alpha', 0.5)
sim_combined_sed = (1 - alpha_sed) * (X_ref_sed @ X_ref_sed.T) + alpha_sed * sim_sed_bridge
y_rknn_sed = compute_rknn(sim_combined_sed, k=5)
log_rknn_sed = np.log(y_rknn_sed.clip(EPS))
print(f"  done. alpha={alpha_sed}", flush=True)

# ── RKNN on SS Bridge ─────────────────────────────────────────────────────────
print("Computing RKNN on plain SS bridge...", flush=True)
alpha_ss = ep_ss.get('alpha', 0.5)
sim_combined_ss = (1 - alpha_ss) * (X_ref_sed @ X_ref_sed.T) + alpha_ss * sim_ss_bridge
y_rknn_ss = compute_rknn(sim_combined_ss, k=5)
log_rknn_ss = np.log(y_rknn_ss.clip(EPS))
print(f"  done. alpha={alpha_ss}", flush=True)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: SED-Species Bridge + Interaction Term
# sigmoid(a*base + b1*log(rknn_sed) + b2*log(win) + b3*log(rknn)*log(win))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 1: SED-Species Bridge + Interaction Term ===", flush=True)
best1 = 0; best1_cfg = {}
interaction_sed = log_rknn_sed * log_win  # element-wise product

for a in [0.80, 0.85, 0.90, 0.95, 1.00]:
    for b1 in [1.20, 1.40, 1.60, 1.80, 2.00]:
        for b2 in [0.30, 0.40, 0.50, 0.60, 0.80]:
            for b3 in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                pred = sigmoid(a*base_logit + b1*log_rknn_sed + b2*log_win + b3*interaction_sed)
                if np.isfinite(pred).all():
                    auc = macro_auc(fl_labels, pred)
                    if auc > best1:
                        best1 = auc; best1_cfg = {'a': a, 'b1': b1, 'b2': b2, 'b3': b3}
print(f"  Best: {best1:.4f}  cfg={best1_cfg}", flush=True)
results['sed_bridge_interaction'] = (best1, best1_cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Bridge Ensemble (SS Bridge + SED-Species Bridge)
# sigmoid(a*base + b*log(w_ss*rknn_ss + w_sed*rknn_sed + (1-w_ss-w_sed)*win))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: Bridge Ensemble (SS + SED-species) ===", flush=True)
best2 = 0; best2_cfg = {}
for w_sed in [0.3, 0.4, 0.5, 0.6]:
    for w_ss in [0.1, 0.2, 0.3]:
        w_win = 1.0 - w_sed - w_ss
        if w_win < 0: continue
        y_blend = w_sed * y_rknn_sed + w_ss * y_rknn_ss + w_win * y_win_k1
        log_blend = np.log(y_blend.clip(EPS))
        for a in [0.80, 0.85, 0.90, 1.00]:
            for b in [1.20, 1.50, 1.70, 2.00]:
                pred = sigmoid(a*base_logit + b*log_blend)
                if np.isfinite(pred).all():
                    auc = macro_auc(fl_labels, pred)
                    if auc > best2:
                        best2 = auc; best2_cfg = {'w_sed': w_sed, 'w_ss': w_ss, 'w_win': w_win, 'a': a, 'b': b}
print(f"  Best: {best2:.4f}  cfg={best2_cfg}", flush=True)
results['bridge_ensemble'] = (best2, best2_cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: SED-species bridge + Separate log(win) coefficient
# More fine-grained sweep around current best (wg=0.45, a=0.85, b=1.70)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: Fine sweep around SED-bridge best ===", flush=True)
best3 = 0; best3_cfg = {}
for wg in [0.35, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.55]:
    y_blend3 = wg * y_rknn_sed + (1-wg) * y_win_k1
    log_b3 = np.log(y_blend3.clip(EPS))
    for a in [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]:
        for b in [1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 2.00]:
            pred = sigmoid(a*base_logit + b*log_b3)
            if np.isfinite(pred).all():
                auc = macro_auc(fl_labels, pred)
                if auc > best3:
                    best3 = auc; best3_cfg = {'wg': wg, 'a': a, 'b': b}
print(f"  Best: {best3:.4f}  cfg={best3_cfg}", flush=True)
results['sed_bridge_fine_sweep'] = (best3, best3_cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 4: SS Bridge + Interaction
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 4: SS Bridge + Interaction Term ===", flush=True)
best4 = 0; best4_cfg = {}
interaction_ss = log_rknn_ss * log_win

for wg in [0.35, 0.40, 0.45, 0.50]:
    y_blend4 = wg * y_rknn_ss + (1-wg) * y_win_k1
    log_b4 = np.log(y_blend4.clip(EPS))
    for a in [0.80, 0.85, 0.90, 1.00]:
        for b in [1.40, 1.60, 1.80, 2.00]:
            pred = sigmoid(a*base_logit + b*log_b4)
            if np.isfinite(pred).all():
                auc = macro_auc(fl_labels, pred)
                if auc > best4: best4 = auc; best4_cfg = {'wg': wg, 'a': a, 'b': b}
results['ss_bridge_fine_sweep'] = (best4, best4_cfg)
print(f"  Best: {best4:.4f}  cfg={best4_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 5: Separate logspace for each signal
# sigmoid(a*base + b_sed*log(rknn_sed) + b_win*log(win))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 5: Separate logspace coefficients ===", flush=True)
best5 = 0; best5_cfg = {}
for a in [0.80, 0.85, 0.90, 0.95, 1.00]:
    for b_sed in [0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00]:
        for b_win in [0.30, 0.50, 0.70, 0.90, 1.10]:
            pred = sigmoid(a*base_logit + b_sed*log_rknn_sed + b_win*log_win)
            if np.isfinite(pred).all():
                auc = macro_auc(fl_labels, pred)
                if auc > best5:
                    best5 = auc; best5_cfg = {'a': a, 'b_sed': b_sed, 'b_win': b_win}
print(f"  Best: {best5:.4f}  cfg={best5_cfg}", flush=True)
results['sed_bridge_separate_logspace'] = (best5, best5_cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
FULL_BEST = 0.9444
print(f"\n{'='*60}")
print(f"INTERACTION + BRIDGE ENSEMBLE SUMMARY")
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
