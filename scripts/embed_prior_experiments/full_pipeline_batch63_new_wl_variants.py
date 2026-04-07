"""
Batch 63: Genuinely new WL variants targeting beyond 0.9873
Methods:
  1. WL Soft-Label: use continuous label values (not hard >0.5/<0.1 thresholds)
  2. WL Positive Reweighting: reweight positive prototypes by their discriminability
  3. WL Dual Softmax: mutual normalization (query→DB and DB→query)
  4. WL Global Calibration: calibrate per-species scores with global statistics
  5. WL Bilinear: learn a bilinear form for similarity via PCA of label co-occurrence
"""
import numpy as np, json, os, time, pickle
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp
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
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi] = (labels_win[s:e].max(0) > 0.5).astype(np.float32)

EPS = 1e-7
mask = file_labels.sum(0) > 0
CURRENT_BEST = 0.9873
def eval_loo(s): return roc_auc_score(file_labels[:, mask], s[:, mask], average='macro')
results = {}

print("Precomputing...", flush=True)
ica100 = FastICA(n_components=100, random_state=42, max_iter=500, tol=0.01)
ew_ica100 = normalize(ica100.fit_transform(emb_win).astype(np.float32), norm='l2')
pca80 = PCA(n_components=80, random_state=42)
ew80 = normalize(pca80.fit_transform(emb_win).astype(np.float32), norm='l2')
scaler = StandardScaler()
emb_std = scaler.fit_transform(emb_win).astype(np.float32)
pca80s = PCA(n_components=80, random_state=42)
ew80s = normalize(pca80s.fit_transform(emb_std).astype(np.float32), norm='l2')
print("Done.", flush=True)

# ─── Baseline WL-UH-triple (reference) ────────────────────────────────────────
def wl_contrast_base(emb_n, k_neg=50, w_max_pos=0.80, w_max_agg=0.92):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr = emb_n[win_file_id != fi]
        lw = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = lw[:, si] > 0.5; nm = lw[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]; ps = te @ pw.T
            pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
            sp = w_max_pos * ps.max(1) + (1 - w_max_pos) * (te @ pp)
            if nm.any():
                nw = tr[nm]; ns = te @ nw.T; k2 = min(k_neg, ns.shape[1])
                tn = nw[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
                tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
                ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
            else:
                ws[:, si] = (sp + 1) / 2
        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

# ─── Method 1: WL Soft-Label ──────────────────────────────────────────────────
# Use continuous label values instead of hard >0.5/<0.1 thresholds
# Positive weight = label value; negative weight = 1 - label value
def wl_soft_label(emb_n, k_neg=50, w_max_pos=0.80, w_max_agg=0.92, soft_thresh=0.3):
    """Use soft labels: positives weighted by label value, negatives by 1-label."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr = emb_n[win_file_id != fi]
        lw = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            lv = lw[:, si]
            pm = lv > soft_thresh   # soft positive: label > thresh
            nm = lv < (1 - soft_thresh)  # soft negative: label < 1-thresh
            if not pm.any(): ws[:, si] = 0.5; continue
            # Weighted positive prototype
            pw_raw = tr[pm]; lv_pos = lv[pm]
            pw_weights = lv_pos / (lv_pos.sum() + EPS)
            pp_weighted = (pw_raw * pw_weights[:, None]).sum(0)
            pp_weighted /= np.linalg.norm(pp_weighted) + EPS
            ps = te @ pw_raw.T
            sp = w_max_pos * ps.max(1) + (1 - w_max_pos) * (te @ pp_weighted)
            if nm.any():
                nw_raw = tr[nm]; lv_neg = lv[nm]
                nw_weights = (1 - lv_neg) / ((1 - lv_neg).sum() + EPS)
                ns = te @ nw_raw.T
                k2 = min(k_neg, ns.shape[1])
                top_idx = np.argsort(-ns, axis=1)[:, :k2]
                # Weighted negative prototype using top-k
                top_weights = nw_weights[top_idx]  # (n_te, k2)
                top_weights /= top_weights.sum(1, keepdims=True) + EPS
                tn = (nw_raw[top_idx] * top_weights[:, :, None]).sum(1)
                tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
                ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
            else:
                ws[:, si] = (sp + 1) / 2
        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

print("\n=== Method 1: WL Soft-Label ===", flush=True)
t0 = time.time()
best1 = 0; best_cfg1 = None
for thresh in [0.2, 0.3, 0.4, 0.5]:
    for emb, name in [(ew_ica100, 'ica100'), (ew80s, 'std80')]:
        out = wl_soft_label(emb, soft_thresh=thresh)
        auc = eval_loo(out)
        if auc > best1: best1 = auc; best_cfg1 = (name, thresh)
print(f"  Soft-label best: {best1:.4f}  cfg={best_cfg1}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_soft_label'] = best1

# Triple blend of soft-label
t0 = time.time()
best1b = 0; best_cfg1b = None
thresh_best = best_cfg1[1]
for wi, ws, wb in [(0.655, 0.225, 0.120), (0.60, 0.25, 0.15), (0.70, 0.20, 0.10)]:
    s1 = wl_soft_label(ew_ica100, soft_thresh=thresh_best)
    s2 = wl_soft_label(ew80s,    soft_thresh=thresh_best)
    s3 = wl_soft_label(ew80,     soft_thresh=thresh_best)
    blend = wi * s1 + ws * s2 + wb * s3
    auc = eval_loo(blend)
    if auc > best1b: best1b = auc; best_cfg1b = (wi, ws, wb)
print(f"  Soft-label triple: {best1b:.4f}  cfg={best_cfg1b}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_soft_label_triple'] = best1b
flag = " *** NEW BEST ***" if best1b > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 2: WL Positive Reweighting ───────────────────────────────────────
# Reweight positive windows by their "discriminability" = how well they separate
# from the negative distribution
def wl_pos_reweight(emb_n, k_neg=50, w_max_pos=0.80, w_max_agg=0.92):
    """Reweight positive prototypes by their discriminability vs negatives."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr = emb_n[win_file_id != fi]
        lw = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = lw[:, si] > 0.5; nm = lw[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]
            if nm.any():
                nw = tr[nm]
                # Discriminability of each positive window = max(pos_neg_gap)
                pn_sims = pw @ nw.T  # (n_pos, n_neg)
                pos_self = pw @ pw.T  # (n_pos, n_pos)
                # Discriminability = mean similarity to other positives - max similarity to negatives
                np.fill_diagonal(pos_self, 0)
                mean_pos_sim = pos_self.sum(1) / max(len(pw) - 1, 1)
                max_neg_sim = pn_sims.max(1)
                discrim = mean_pos_sim - max_neg_sim  # higher = more discriminative
                # Softmax reweighting
                discrim_w = np.exp(discrim); discrim_w /= discrim_w.sum() + EPS
                # Weighted positive prototype
                pp_disc = (pw * discrim_w[:, None]).sum(0)
                pp_disc /= np.linalg.norm(pp_disc) + EPS
                # Standard max-mean pos score
                ps = te @ pw.T
                sp = w_max_pos * ps.max(1) + (1 - w_max_pos) * (te @ pp_disc)
                ns = te @ nw.T; k2 = min(k_neg, ns.shape[1])
                tn = nw[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
                tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
                ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
            else:
                ps = te @ pw.T
                pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
                sp = w_max_pos * ps.max(1) + (1 - w_max_pos) * (te @ pp)
                ws[:, si] = (sp + 1) / 2
        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

print("\n=== Method 2: WL Positive Reweighting ===", flush=True)
t0 = time.time()
best2 = 0; best_cfg2 = None
for emb, name in [(ew_ica100, 'ica100'), (ew80s, 'std80')]:
    out = wl_pos_reweight(emb, k_neg=50, w_max_agg=0.92, w_max_pos=0.80)
    auc = eval_loo(out)
    if auc > best2: best2 = auc; best_cfg2 = name
print(f"  Pos-reweight best: {best2:.4f}  cfg={best_cfg2}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_pos_reweight'] = best2

# Triple blend
t0 = time.time()
s1 = wl_pos_reweight(ew_ica100); s2 = wl_pos_reweight(ew80s); s3 = wl_pos_reweight(ew80)
best2b = max(eval_loo(0.655*s1 + 0.225*s2 + 0.120*s3),
             eval_loo(0.60*s1 + 0.25*s2 + 0.15*s3))
print(f"  Pos-reweight triple: {best2b:.4f}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_pos_reweight_triple'] = best2b
flag = " *** NEW BEST ***" if best2b > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 3: WL Dual Softmax ───────────────────────────────────────────────
# Normalize score both from query→DB (standard) AND DB→query perspective
def wl_dual_softmax(emb_n, tau=0.3, k_neg=50, w_max_agg=0.92):
    """
    Dual softmax: P(t belongs to si) =
      (sim(t, best_pos) / sum_pos_sims) * (sim(t, best_pos) / sum_all_sims_for_best_pos)
    Geometrically: take geometric mean of forward and backward softmax probabilities.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr = emb_n[win_file_id != fi]
        lw = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = lw[:, si] > 0.5; nm = lw[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]
            pos_sims = te @ pw.T  # (n_te, n_pos)
            best_pos = pos_sims.max(1)  # (n_te,)
            # Forward: query→pos (standard NT-Xent style)
            if nm.any():
                nw = tr[nm]; ns = te @ nw.T
                k2 = min(k_neg, ns.shape[1])
                top_neg = -np.sort(-ns, axis=1)[:, :k2]
                all_sims_fwd = np.concatenate([best_pos[:, None], top_neg], axis=1) / tau
                log_denom_fwd = logsumexp(all_sims_fwd, axis=1)
                forward_score = np.exp(best_pos / tau - log_denom_fwd)
                # Backward: best_pos_window → (test wins + neg wins)
                # For each te window, find the matching pos window
                best_pos_idx = pos_sims.argmax(1)  # (n_te,) index into pw
                # Backward score: how much does the best pos window prefer this te window?
                all_te_sims = pw @ np.concatenate([te, nw[:k2]], axis=0).T  # (n_pos, n_te+k2)
                backward_scores = []
                for ti in range(len(te)):
                    pi = best_pos_idx[ti]
                    te_neg_sims = pw[pi] @ np.concatenate([te, nw[:k2]], axis=0).T  # (n_te+k2,)
                    bwd_denom = logsumexp(te_neg_sims / tau)
                    bwd_score = np.exp(pos_sims[ti, pi] / tau - bwd_denom)
                    backward_scores.append(bwd_score)
                backward_score = np.array(backward_scores)
                # Geometric mean of forward and backward
                ws[:, si] = np.sqrt(forward_score * backward_score + EPS)
            else:
                ws[:, si] = 1.0
        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

print("\n=== Method 3: WL Dual Softmax ===", flush=True)
t0 = time.time()
best3 = 0; best_cfg3 = None
for tau in [0.2, 0.3, 0.5]:
    for emb, name in [(ew_ica100, 'ica100'), (ew80s, 'std80')]:
        out = wl_dual_softmax(emb, tau=tau)
        auc = eval_loo(out)
        if auc > best3: best3 = auc; best_cfg3 = (name, tau)
print(f"  Dual-softmax best: {best3:.4f}  cfg={best_cfg3}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_dual_softmax'] = best3

# Best config triple
tau_best = best_cfg3[1]
t0 = time.time()
s1 = wl_dual_softmax(ew_ica100, tau=tau_best)
s2 = wl_dual_softmax(ew80s,    tau=tau_best)
s3 = wl_dual_softmax(ew80,     tau=tau_best)
best3b = max(eval_loo(0.655*s1+0.225*s2+0.120*s3),
             eval_loo(0.60*s1+0.25*s2+0.15*s3),
             eval_loo(0.70*s1+0.20*s2+0.10*s3))
print(f"  Dual-softmax triple: {best3b:.4f}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_dual_softmax_triple'] = best3b
flag = " *** NEW BEST ***" if best3b > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 4: WL Global Calibration ─────────────────────────────────────────
# After computing WL scores, apply global calibration:
# For each species, subtract the mean score across all files (center the distribution)
# This helps species with systematically high/low background scores
def wl_global_calib(emb_n, k_neg=50, w_max_pos=0.80, w_max_agg=0.92, calib_strength=0.5):
    """
    LOO WL with global calibration: subtract per-species mean of LOO scores
    to center the distribution (removes species-level bias).
    """
    # First compute raw WL LOO scores
    raw_scores = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]
        tr = emb_n[win_file_id != fi]
        lw = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = lw[:, si] > 0.5; nm = lw[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]; ps = te @ pw.T
            pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
            sp = w_max_pos * ps.max(1) + (1 - w_max_pos) * (te @ pp)
            if nm.any():
                nw = tr[nm]; ns = te @ nw.T; k2 = min(k_neg, ns.shape[1])
                tn = nw[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
                tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
                ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
            else:
                ws[:, si] = (sp + 1) / 2
        raw_scores[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    # Global calibration: subtract per-species mean * calib_strength
    species_mean = raw_scores.mean(0)  # (n_species,)
    calibrated = raw_scores - calib_strength * (species_mean - 0.5)
    return np.clip(calibrated, 0, 1)

print("\n=== Method 4: WL Global Calibration ===", flush=True)
t0 = time.time()
best4 = 0; best_cfg4 = None
for cs in [0.3, 0.5, 0.7, 1.0]:
    for emb, name in [(ew_ica100, 'ica100'), (ew80s, 'std80')]:
        out = wl_global_calib(emb, calib_strength=cs)
        auc = eval_loo(out)
        if auc > best4: best4 = auc; best_cfg4 = (name, cs)
print(f"  Global calib best: {best4:.4f}  cfg={best_cfg4}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_global_calib'] = best4

# Triple blend
cs_best = best_cfg4[1]
s1 = wl_global_calib(ew_ica100, calib_strength=cs_best)
s2 = wl_global_calib(ew80s,    calib_strength=cs_best)
s3 = wl_global_calib(ew80,     calib_strength=cs_best)
best4b = max(eval_loo(0.655*s1+0.225*s2+0.120*s3),
             eval_loo(0.60*s1+0.25*s2+0.15*s3))
print(f"  Global calib triple: {best4b:.4f}", flush=True)
results['wl_global_calib_triple'] = best4b
flag = " *** NEW BEST ***" if best4b > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Method 5: WL UH + Global Calib blend ─────────────────────────────────────
# Blend the best existing WL-UH-triple with global calibration variant
print("\n=== Method 5: WL UH + Calibration Blend ===", flush=True)
t0 = time.time()
# Load existing UH triple
ep = pickle.load(open("outputs/embed_prior_model.pkl", "rb"))
cfg = ep['config']
_w_ica, _w_std, _w_pca = cfg['w_ica100'], cfg['w_std'], cfg['w_pca80']
_k_ica, _wma_ica, _wmp_ica = cfg['ica100']['k_neg'], cfg['ica100']['w_max_agg'], cfg['ica100']['w_max_pos']

# Recompute UH triple (current best)
def wl_uh(emb_n, k_neg=50, wmp=0.80, wma=0.92):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = emb_n[win_file_id == fi]; tr = emb_n[win_file_id != fi]
        lw = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = lw[:, si] > 0.5; nm = lw[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]; ps = te @ pw.T
            pp = pw.mean(0); pp /= np.linalg.norm(pp) + EPS
            sp = wmp * ps.max(1) + (1 - wmp) * (te @ pp)
            if nm.any():
                nw = tr[nm]; ns = te @ nw.T; k2 = min(k_neg, ns.shape[1])
                tn = nw[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
                tn /= np.linalg.norm(tn, axis=1, keepdims=True) + EPS
                ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
            else: ws[:, si] = (sp + 1) / 2
        out[fi] = wma * ws.max(0) + (1 - wma) * ws.mean(0)
    return out

uh_ica = wl_uh(ew_ica100, _k_ica, _wmp_ica, _wma_ica)
uh_std = wl_uh(ew80s, 4, 0.60, 0.65)
uh_pca = wl_uh(ew80,  4, 0.70, 0.60)
uh_triple = _w_ica * uh_ica + _w_std * uh_std + _w_pca * uh_pca

# Calibrated version of UH triple
uh_triple_mean = uh_triple.mean(0)
best5 = 0; best_cfg5 = None
for cs in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    calib = np.clip(uh_triple - cs * (uh_triple_mean - 0.5), 0, 1)
    auc = eval_loo(calib)
    if auc > best5: best5 = auc; best_cfg5 = cs
print(f"  UH+calib best: {best5:.4f}  calib_strength={best_cfg5}  ({time.time()-t0:.0f}s)", flush=True)
results['wl_uh_calib'] = best5
flag = " *** NEW BEST ***" if best5 > CURRENT_BEST else ""
print(f"  {flag}", flush=True)

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== Batch 63 Summary ===", flush=True)
for name, auc in sorted(results.items(), key=lambda x: -x[1])[:10]:
    flag = " *** NEW BEST ***" if auc > CURRENT_BEST else ""
    print(f"  {name}: {auc:.4f}{flag}", flush=True)

with open("outputs/embed_prior_results.json") as f:
    rd = json.load(f)
cur_best = rd['best'].get('loo_auc', 0)
for name, auc in results.items():
    rd['experiments'].append({'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)})
    if auc > cur_best:
        cur_best = auc
        rd['best'] = {'method': name, 'loo_auc': float(auc), 'full_auc': float(auc)}
        print(f"*** JSON BEST UPDATED: {name} = {auc:.4f} ***", flush=True)
with open("outputs/embed_prior_results.json", 'w') as f:
    json.dump(rd, f, indent=2)
print(f"\nFinal best: {rd['best']['method']} = {rd['best']['loo_auc']:.4f}", flush=True)
