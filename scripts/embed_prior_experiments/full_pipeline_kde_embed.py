"""
Full Pipeline: KDE per-species Embed Prior
EP-only breakthrough: kde_per_species = 0.9373 (new EP-only best, beats interaction_knn 0.9199)

Now: integrate into full pipeline (VLOM base + logspace correction)
Target: beat full pipeline best 0.9444 (sed_species_bridge)
"""
import numpy as np, pickle, json, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load Perch data ────────────────────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_embs_avg  = np.zeros((n_files, 1536), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_embs_avg[fi]  = emb_win[s:e].mean(0)
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

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
base_probs  = vlom_blend(sigmoid(file_logit_max), file_sed_max)
base_logit  = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))

# ── PCA features ──────────────────────────────────────────────────────────────
emb_norm = normalize(file_embs_avg, norm='l2').astype(np.float32)
pca32 = PCA(n_components=32, random_state=42).fit(emb_norm)
X_pca32 = pca32.transform(emb_norm).astype(np.float32)
X_pca32 = (X_pca32 - X_pca32.mean(0)) / X_pca32.std(0).clip(1e-8)

pca64 = PCA(n_components=64, random_state=42).fit(emb_norm)
X_pca64 = pca64.transform(emb_norm).astype(np.float32)
X_pca64 = (X_pca64 - X_pca64.mean(0)) / X_pca64.std(0).clip(1e-8)

# ── Window KNN k=1 ─────────────────────────────────────────────────────────────
print("Computing win_k1 LOO...", flush=True)
y_win_k1 = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = (win_file_id != i)
    X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :1]
    wp = np.zeros((te_e-te_s, n_species), np.float32)
    for wi in range(te_e-te_s):
        ww = sims[wi, top_idx[wi]].clip(0); ws = ww.sum()
        ww = ww/ws if ws > 1e-8 else np.ones(1)
        wp[wi] = (ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i] = wp.mean(0)
log_win = np.log(y_win_k1.clip(EPS))
print("  done.", flush=True)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: KDE full pipeline sweep (bw fine sweep around best: 1.2)
# EP-only: a=1.0, b=0.8 → 0.9373
# Full pipeline: base_logit = VLOM(ProtoSSM+SED)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Method 1: KDE full pipeline sweep ===", flush=True)
best1 = 0; best1_cfg = {}

for bw in [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
    for pca_n, X_pca in [(32, X_pca32), (64, X_pca64)]:
        loo_kde = np.zeros((n_files, n_species), np.float32)
        for fi_test in range(n_files):
            tr_mask = np.arange(n_files) != fi_test
            X_tr = X_pca[tr_mask]
            X_te = X_pca[[fi_test]]
            kde_bg = KernelDensity(kernel='gaussian', bandwidth=bw)
            kde_bg.fit(X_tr)
            log_bg = kde_bg.score_samples(X_te)[0]
            for si in range(n_species):
                pos_idx = np.where(file_labels[tr_mask, si] > 0.5)[0]
                if len(pos_idx) == 0:
                    loo_kde[fi_test, si] = sigmoid(file_logit_max[fi_test, si])
                    continue
                kde_pos = KernelDensity(kernel='gaussian', bandwidth=bw)
                kde_pos.fit(X_tr[pos_idx])
                log_pos = kde_pos.score_samples(X_te)[0]
                loo_kde[fi_test, si] = sigmoid(log_pos - log_bg)
        log_kde = np.log(loo_kde.clip(EPS))
        for a in [0.80, 0.85, 0.90, 0.95, 1.00]:
            for b in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]:
                pred = sigmoid(a*base_logit + b*log_kde)
                if np.isfinite(pred).all():
                    auc = macro_auc(file_labels, pred)
                    if auc > best1:
                        best1 = auc; best1_cfg = {'bw': bw, 'pca_n': pca_n, 'a': a, 'b': b}
        print(f"  bw={bw} pca={pca_n}: best so far={best1:.4f}", flush=True)
results['kde_full_pipeline'] = (best1, best1_cfg)
print(f"  Best: {best1:.4f}  cfg={best1_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: KDE + win_k1 blend (logspace fusion)
# sigmoid(a*base + b*log(wg*kde + (1-wg)*win))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 2: KDE + win_k1 blend ===", flush=True)
best2 = 0; best2_cfg = {}

# Use best KDE config from Method 1 and sweep wg
best_bw = 1.2; best_pca_n = 32  # from EP-only best, refine after M1
for bw in [1.0, 1.2, 1.5, 2.0]:
    loo_kde2 = np.zeros((n_files, n_species), np.float32)
    for fi_test in range(n_files):
        tr_mask = np.arange(n_files) != fi_test
        X_tr = X_pca32[tr_mask]; X_te = X_pca32[[fi_test]]
        kde_bg2 = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde_bg2.fit(X_tr); log_bg2 = kde_bg2.score_samples(X_te)[0]
        for si in range(n_species):
            pos_idx = np.where(file_labels[tr_mask, si] > 0.5)[0]
            if len(pos_idx) == 0:
                loo_kde2[fi_test, si] = sigmoid(file_logit_max[fi_test, si]); continue
            kde_pos2 = KernelDensity(kernel='gaussian', bandwidth=bw)
            kde_pos2.fit(X_tr[pos_idx]); log_pos2 = kde_pos2.score_samples(X_te)[0]
            loo_kde2[fi_test, si] = sigmoid(log_pos2 - log_bg2)

    for wg in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        y_blend = wg * loo_kde2 + (1-wg) * y_win_k1
        log_blend = np.log(y_blend.clip(EPS))
        for a in [0.80, 0.85, 0.90, 0.95]:
            for b in [1.2, 1.4, 1.6, 1.8, 2.0]:
                pred2 = sigmoid(a*base_logit + b*log_blend)
                if np.isfinite(pred2).all():
                    auc2 = macro_auc(file_labels, pred2)
                    if auc2 > best2:
                        best2 = auc2; best2_cfg = {'bw': bw, 'wg': wg, 'a': a, 'b': b}
    print(f"  bw={bw}: best so far={best2:.4f}", flush=True)
results['kde_win_blend'] = (best2, best2_cfg)
print(f"  Best: {best2:.4f}  cfg={best2_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: KDE + RKNN (SED-species bridge) blend
# sigmoid(a*base + b*log(wg_kde*kde + wg_rknn*rknn_sed + wg_win*win))
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Method 3: KDE + SED-bridge RKNN blend ===", flush=True)
# Load SED bridge pkl
with open("outputs/embed_prior_sed_species_bridge.pkl", "rb") as f:
    ep_sed = pickle.load(f)
X_ref_sed = ep_sed['X_combined_n'].astype(np.float32)
fl_ep = ep_sed['file_labels'].astype(np.float32)
sim_sed_bridge = ep_sed['sim_bridge_n'].astype(np.float32)
alpha_sed = ep_sed.get('alpha', 0.5)
sim_combined_sed = (1-alpha_sed)*(X_ref_sed @ X_ref_sed.T) + alpha_sed*sim_sed_bridge

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
            w = np.exp(ls); w /= w.sum(); y[i] = (w[:,None]*fl_ep[tr[top5]]).sum(0)
        else:
            ms = np.array(msims); ls = ms/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
            y[i] = (w[:,None]*fl_ep[mutual]).sum(0)
    return y

y_rknn_sed = compute_rknn(sim_combined_sed, k=5)
print("  RKNN computed.", flush=True)

# Recompute KDE with best bw=1.2
loo_kde3 = np.zeros((n_files, n_species), np.float32)
for fi_test in range(n_files):
    tr_mask = np.arange(n_files) != fi_test
    X_tr = X_pca32[tr_mask]; X_te = X_pca32[[fi_test]]
    kde_bg3 = KernelDensity(kernel='gaussian', bandwidth=1.2)
    kde_bg3.fit(X_tr); log_bg3 = kde_bg3.score_samples(X_te)[0]
    for si in range(n_species):
        pos_idx = np.where(file_labels[tr_mask, si] > 0.5)[0]
        if len(pos_idx) == 0:
            loo_kde3[fi_test, si] = sigmoid(file_logit_max[fi_test, si]); continue
        kde_pos3 = KernelDensity(kernel='gaussian', bandwidth=1.2)
        kde_pos3.fit(X_tr[pos_idx]); log_pos3 = kde_pos3.score_samples(X_te)[0]
        loo_kde3[fi_test, si] = sigmoid(log_pos3 - log_bg3)

best3 = 0; best3_cfg = {}
for wg_kde in [0.1, 0.2, 0.3]:
    for wg_rknn in [0.3, 0.4, 0.5]:
        wg_win = 1.0 - wg_kde - wg_rknn
        if wg_win < 0: continue
        y_blend3 = wg_kde*loo_kde3 + wg_rknn*y_rknn_sed + wg_win*y_win_k1
        log_b3 = np.log(y_blend3.clip(EPS))
        for a in [0.80, 0.85, 0.90]:
            for b in [1.4, 1.6, 1.7, 1.8, 2.0]:
                pred3 = sigmoid(a*base_logit + b*log_b3)
                if np.isfinite(pred3).all():
                    auc3 = macro_auc(file_labels, pred3)
                    if auc3 > best3:
                        best3 = auc3; best3_cfg = {'wg_kde': wg_kde, 'wg_rknn': wg_rknn, 'wg_win': wg_win, 'a': a, 'b': b}
results['kde_rknn_win_blend'] = (best3, best3_cfg)
print(f"  Best: {best3:.4f}  cfg={best3_cfg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary & JSON update
# ═══════════════════════════════════════════════════════════════════════════════
FULL_BEST = 0.9444
print(f"\n{'='*60}")
print(f"KDE FULL PIPELINE SUMMARY")
print(f"Full pipeline best: {FULL_BEST}")
print(f"{'='*60}")
for name, (auc, cfg) in sorted(results.items(), key=lambda x: -x[1][0]):
    delta = auc - FULL_BEST
    marker = " *** NEW BEST ***" if auc > FULL_BEST else ""
    print(f"  {name}: {auc:.4f}  ({delta:+.4f}){marker}")

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
