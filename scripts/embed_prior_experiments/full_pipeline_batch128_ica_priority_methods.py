"""
batch128: Priority methods with ICA/PCA features (not raw embeddings)
Previously tried with raw [739,1536] embeddings → 0.82-0.90
Now retry with ICA-100 / PCA-80 / NMF-80 from PKL → targeting >0.975

M1: Mahalanobis KNN in ICA-100 space (LedoitWolf covariance)
M2: GMM per species in ICA-100 space
M3: Bayesian Ridge per species in ICA-100 space
M4: Attention-weighted KNN in ICA-100 space (logit-gated)
M5: Dual softmax with NMF / STD features (new feature combos)
M6: Ensemble of best ICA-based methods
"""
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import BayesianRidge
from numpy.linalg import norm
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-8
ROOT = Path("/home/lab/BirdClef-2026-Codebase")

# ── data ──────────────────────────────────────────────────────────────────────
DATA       = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
labels_win = DATA["labels"].astype(np.float32)
logit_win  = DATA["logits"].astype(np.float32)
n_windows  = DATA["n_windows"]
n_files    = len(n_windows)
n_species  = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(739, np.int32)
for fi in range(n_files):
    win_file_id[file_start[fi]:file_end[fi]] = fi

with open(ROOT / "outputs" / "embed_prior_model.pkl", "rb") as f:
    ep = pickle.load(f)

ew_ica = ep["emb_win_ica_norm"]   # [739, 100] ICA-100
ew_pca = ep["emb_win_pca_norm"]   # [739,  80] PCA-80
ew_std = ep["emb_win_std_norm"]   # [739,  80] Std-PCA-80
ew_nmf = ep["emb_win_nmf_norm"]   # [739, 100] NMF-100
file_labels = ep["file_labels"]   # [66, 234]
logit_sig_win = ep["logit_sig_win"]  # [739, 234] sigmoid(logits)

print(f"[batch128] ICA{ew_ica.shape}, n_files={n_files}", flush=True)

RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
res      = json.load(open(RESULTS_PATH))
best_loo = res["best"]["loo_auc"]
tried    = {e["method"] for e in res["experiments"]}
print(f"[batch128] Current best: {res['best']['method']} LOO={best_loo:.6f}", flush=True)

def macro_auc(s):
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            aucs.append(roc_auc_score(y, s[:, si]))
    return float(np.mean(aucs)) if aucs else 0.0

def save_result(method, score, config, note=''):
    global best_loo
    delta = score - best_loo
    res['experiments'].append({'method': method, 'loo_auc': score, 'config': config, 'note': note})
    if score > best_loo:
        best_loo = score
        res['best'] = {'method': method, 'loo_auc': score}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(res, f, indent=2)
    return delta

# ═════════════════════════════════════════════════════════════════════════════
# M1: Mahalanobis KNN in ICA space
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M1] Mahalanobis KNN in ICA space...", flush=True)

def mahal_ica_loo(ew, k=5, w_max_agg=0.8):
    """
    Mahalanobis KNN: for each positive species prototype window, compute
    Mahalanobis distance (LedoitWolf covariance) vs test windows.
    Aggregate with max+mean.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]
        tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]

        # Fit LedoitWolf on training data
        try:
            lw = LedoitWolf().fit(tr)
            VI = lw.precision_  # [d, d]
        except Exception:
            VI = np.eye(tr.shape[1])

        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            nm = tl[:, si] < 0.1
            if not pm.any():
                ws[:, si] = 0.5; continue

            pos = tr[pm]  # [n_pos, d]

            # Mahalanobis distances: test vs positive examples
            # d^2(x,y) = (x-y)^T VI (x-y)
            XV_pos = pos @ VI         # [n_pos, d]
            diag_pos = (XV_pos * pos).sum(1)  # [n_pos]

            mahal_sq = np.zeros((len(te), len(pos)), np.float64)
            for ti in range(len(te)):
                x = te[ti]
                xVI = x @ VI
                mahal_sq[ti] = diag_pos - 2*(XV_pos*x).sum(1) + (xVI*x).sum()
            mahal_sq = np.clip(mahal_sq, 0, None)

            # Convert to similarity (negative Mahal distance, normalized)
            neg_mahal = -np.sqrt(mahal_sq + EPS)
            neg_mahal -= neg_mahal.max(1, keepdims=True)
            pos_sims = np.exp(neg_mahal)
            pos_sims /= (pos_sims.sum(1, keepdims=True) + EPS)

            score = pos_sims.max(1)

            # Optionally subtract negative reference
            if nm.any():
                neg = tr[nm]
                XV_neg = neg @ VI
                diag_neg = (XV_neg * neg).sum(1)
                mahal_neg = np.zeros((len(te), len(neg)), np.float64)
                for ti in range(len(te)):
                    x = te[ti]
                    xVI = x @ VI
                    mahal_neg[ti] = diag_neg - 2*(XV_neg*x).sum(1) + (xVI*x).sum()
                k_neg = min(k, len(neg))
                nn_neg = np.sort(mahal_neg, axis=1)[:, :k_neg].mean(1)
                nn_pos = np.sort(mahal_sq, axis=1)[:, :k_neg].mean(1)
                score = 1.0 / (1.0 + nn_pos / (nn_neg + EPS))

            ws[:, si] = score

        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

m1_configs = [
    ('ica', ew_ica, 5, 0.80),
    ('ica', ew_ica, 5, 0.70),
    ('ica', ew_ica, 3, 0.80),
    ('pca', ew_pca, 5, 0.80),
    ('std', ew_std, 5, 0.80),
]

m1_best = 0.0
for feat_name, ew, k, wma in m1_configs:
    mname = f'mahal_ica_{feat_name}_k{k}_wma{int(wma*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    t0 = time.time()
    result = mahal_ica_loo(ew, k=k, w_max_agg=wma)
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'feat': feat_name, 'k': k, 'wma': wma})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag} [{time.time()-t0:.0f}s]', flush=True)
    m1_best = max(m1_best, score)
print(f"  M1 done, best={m1_best:.6f}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M2: GMM per species in ICA space
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M2] GMM per species in ICA space...", flush=True)

def gmm_ica_loo(ew, n_components=2, w_max_agg=0.8):
    """Fit per-species GMM in ICA space; score = positive log-likelihood ratio."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]
        tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]

        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any():
                ws[:, si] = 0.5; continue
            pos = tr[pm]

            if len(pos) < n_components:
                # Fallback to cosine similarity
                proto = pos.mean(0); proto /= (norm(proto) + EPS)
                ws[:, si] = np.clip((te @ proto + 1) / 2, 0, 1)
                continue

            try:
                gmm_pos = GaussianMixture(
                    n_components=min(n_components, len(pos)),
                    covariance_type='diag', max_iter=50, random_state=42)
                gmm_pos.fit(pos)
                ll_pos = gmm_pos.score_samples(te)
            except Exception:
                proto = pos.mean(0); proto /= (norm(proto) + EPS)
                ws[:, si] = np.clip((te @ proto + 1) / 2, 0, 1)
                continue

            # Background: ALL training windows
            try:
                gmm_bg = GaussianMixture(
                    n_components=min(n_components*2, len(tr)//2),
                    covariance_type='diag', max_iter=50, random_state=42)
                gmm_bg.fit(tr)
                ll_bg = gmm_bg.score_samples(te)
            except Exception:
                ws[:, si] = 0.5; continue

            llr = ll_pos - ll_bg
            ws[:, si] = 1.0 / (1.0 + np.exp(-np.clip(llr, -10, 10)))

        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

m2_configs = [
    ('ica', ew_ica, 2, 0.80),
    ('ica', ew_ica, 2, 0.70),
    ('ica', ew_ica, 1, 0.80),
    ('nmf', ew_nmf, 2, 0.80),
]

m2_best = 0.0
for feat_name, ew, nc, wma in m2_configs:
    mname = f'gmm_ica_{feat_name}_c{nc}_wma{int(wma*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    t0 = time.time()
    result = gmm_ica_loo(ew, n_components=nc, w_max_agg=wma)
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'feat': feat_name, 'nc': nc, 'wma': wma})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag} [{time.time()-t0:.0f}s]', flush=True)
    m2_best = max(m2_best, score)
print(f"  M2 done, best={m2_best:.6f}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M3: Bayesian Ridge per species in ICA space
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M3] Bayesian Ridge per species in ICA space...", flush=True)

def bayridge_ica_loo(ew, w_max_agg=0.8):
    """Fit per-species BayesianRidge in ICA space."""
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]
        tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]

        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            y_sp = tl[:, si]
            if y_sp.sum() < 1 or y_sp.mean() > 0.999:
                ws[:, si] = y_sp.mean(); continue
            try:
                br = BayesianRidge(max_iter=100)
                br.fit(tr, y_sp)
                p = br.predict(te)
                ws[:, si] = np.clip(p, 0, 1)
            except Exception:
                ws[:, si] = y_sp.mean()

        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

m3_configs = [
    ('ica', ew_ica, 0.80),
    ('ica', ew_ica, 0.70),
    ('pca', ew_pca, 0.80),
    ('nmf', ew_nmf, 0.80),
]

m3_best = 0.0
for feat_name, ew, wma in m3_configs:
    mname = f'bayridge_ica_{feat_name}_wma{int(wma*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    t0 = time.time()
    result = bayridge_ica_loo(ew, w_max_agg=wma)
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'feat': feat_name, 'wma': wma})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag} [{time.time()-t0:.0f}s]', flush=True)
    m3_best = max(m3_best, score)
print(f"  M3 done, best={m3_best:.6f}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M4: Attention KNN in ICA space (logit-gated)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M4] Attention KNN in ICA space (logit-gated)...", flush=True)

def attn_ica_loo(ew, tau=0.3, w_max_agg=0.8):
    """
    For each species, find positive windows in ICA space.
    Weight them by their logit attention score (forward × backward).
    Cosine similarity score.
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]
        tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        tl_logit = logit_win[win_file_id != fi]

        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            nm = tl[:, si] < 0.1
            if not pm.any():
                ws[:, si] = 0.5; continue

            pos = tr[pm]
            pos_logit = tl_logit[pm, si]
            # Attention weight from logit
            attn = 1.0 / (1.0 + np.exp(-np.clip(pos_logit / tau, -10, 10)))
            attn /= (attn.sum() + EPS)

            # Forward similarity: test → positive
            sims = te @ pos.T   # [n_te, n_pos]
            forward = (sims * attn[None, :]).sum(1)  # [n_te]

            # Optional negative suppression
            if nm.any():
                neg = tr[nm]
                neg_sims = te @ neg.T
                neg_mean = neg_sims.mean(1)
                ws[:, si] = np.clip((forward - neg_mean + 1) / 2, 0, 1)
            else:
                ws[:, si] = np.clip((forward + 1) / 2, 0, 1)

        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

m4_configs = [
    ('ica', ew_ica, 0.3, 0.80),
    ('ica', ew_ica, 0.5, 0.80),
    ('ica', ew_ica, 0.3, 0.70),
    ('ica', ew_ica, 0.2, 0.80),
    ('std', ew_std, 0.3, 0.80),
]

m4_best = 0.0
for feat_name, ew, tau, wma in m4_configs:
    mname = f'attn_ica_{feat_name}_tau{int(tau*10):d}_wma{int(wma*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    t0 = time.time()
    result = attn_ica_loo(ew, tau=tau, w_max_agg=wma)
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'feat': feat_name, 'tau': tau, 'wma': wma})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag} [{time.time()-t0:.0f}s]', flush=True)
    m4_best = max(m4_best, score)
print(f"  M4 done, best={m4_best:.6f}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M5: Dual softmax with NMF/STD80 features (new feature combos)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M5] Dual softmax new feature combos...", flush=True)

def wl_dual_softmax(ew, tau=0.3, w_max_agg=0.8):
    """
    Dual softmax: forward sim × backward sim (geometric mean).
    """
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]
        tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]

        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            nm = tl[:, si] < 0.1
            if not pm.any():
                ws[:, si] = 1.0; continue

            pos = tr[pm]
            nw = tr[nm] if nm.any() else tr[~pm]
            k2 = min(5, len(nw))

            pos_sims = te @ pos.T   # [n_te, n_pos]
            # Forward: softmax over positives
            fwd_sm = pos_sims / tau
            fwd_sm -= fwd_sm.max(1, keepdims=True)
            fwd_e = np.exp(fwd_sm)
            fwd = fwd_e / (fwd_e.sum(1, keepdims=True) + EPS)
            best_pos_idx = fwd.argmax(1)  # [n_te]
            forward_score = fwd[np.arange(len(te)), best_pos_idx]

            # Backward: for best matching positive, score = exp(sim/tau) / sum over all
            backward_score = np.zeros(len(te), np.float32)
            all_others = np.concatenate([te, nw[:k2]], axis=0) if k2 > 0 else te
            for ti in range(len(te)):
                pi = best_pos_idx[ti]
                bwd_sims = pos[pi] @ all_others.T
                bwd = np.exp(pos_sims[ti, pi] / tau - np.log(np.exp(bwd_sims / tau).sum() + EPS))
                backward_score[ti] = bwd

            ws[:, si] = np.sqrt(forward_score * backward_score + EPS)

        out[fi] = w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)
    return out

m5_configs = [
    ('nmf', ew_nmf, 0.3, 0.80),
    ('nmf', ew_nmf, 0.2, 0.80),
    ('std', ew_std, 0.3, 0.80),
    ('std', ew_std, 0.2, 0.80),
    ('pca', ew_pca, 0.3, 0.80),
]

m5_best = 0.0
for feat_name, ew, tau, wma in m5_configs:
    mname = f'dualsoft_ica_{feat_name}_tau{int(tau*10):d}_wma{int(wma*100):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    t0 = time.time()
    result = wl_dual_softmax(ew, tau=tau, w_max_agg=wma)
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'feat': feat_name, 'tau': tau, 'wma': wma})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag} [{time.time()-t0:.0f}s]', flush=True)
    m5_best = max(m5_best, score)

# Also try ensemble: best known (ica dual_softmax) + new NMF/STD
# Re-compute the best dual softmax (ica)
print("  Computing reference ICA dual softmax...", flush=True)
ref_ds_ica = wl_dual_softmax(ew_ica, tau=0.3, w_max_agg=0.8)
ref_ds_std = wl_dual_softmax(ew_std, tau=0.3, w_max_agg=0.8)
ref_ds_nmf = wl_dual_softmax(ew_nmf, tau=0.3, w_max_agg=0.8)

for w_ica, w_nmf, w_std in [
    (0.6, 0.2, 0.2),
    (0.5, 0.3, 0.2),
    (0.5, 0.25, 0.25),
    (0.7, 0.15, 0.15),
]:
    mname = f'dualsoft_ens_i{int(w_ica*10):d}_n{int(w_nmf*10):d}_s{int(w_std*10):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    result = w_ica * ref_ds_ica + w_nmf * ref_ds_nmf + w_std * ref_ds_std
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'w_ica': w_ica, 'w_nmf': w_nmf, 'w_std': w_std})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)
    m5_best = max(m5_best, score)

print(f"  M5 done, best={m5_best:.6f}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# M6: ICA Mahal + Dual softmax ensemble
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M6] ICA ensemble: mahal + attn + dual softmax...", flush=True)

# Use best from M1/M4 and blend with reference dual softmax
best_mahal_ica = mahal_ica_loo(ew_ica, k=5, w_max_agg=0.80)
best_attn_ica  = attn_ica_loo(ew_ica, tau=0.3, w_max_agg=0.80)

for w_ds, w_mahal, w_attn in [
    (0.7, 0.2, 0.1),
    (0.6, 0.2, 0.2),
    (0.8, 0.1, 0.1),
    (0.7, 0.15, 0.15),
]:
    mname = f'ica_ens_ds{int(w_ds*10):d}_m{int(w_mahal*10):d}_a{int(w_attn*10):d}'
    if mname in tried:
        print(f'  {mname}: skip', flush=True); continue
    result = w_ds * ref_ds_ica + w_mahal * best_mahal_ica + w_attn * best_attn_ica
    score  = macro_auc(result)
    delta  = save_result(mname, score, {'w_ds': w_ds, 'w_mahal': w_mahal, 'w_attn': w_attn})
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {delta:+.6f}{flag}', flush=True)

print(f"  M6 done", flush=True)

print("\n" + "="*60, flush=True)
print(f"[batch128] SUMMARY", flush=True)
print(f"  Global best: {res['best']['method']} LOO={res['best']['loo_auc']:.6f}", flush=True)
print(f"  M1 Mahal ICA:       {m1_best:.6f}", flush=True)
print(f"  M2 GMM ICA:         {m2_best:.6f}", flush=True)
print(f"  M3 BayRidge ICA:    {m3_best:.6f}", flush=True)
print(f"  M4 Attn ICA:        {m4_best:.6f}", flush=True)
print(f"  M5 DualSoft new:    {m5_best:.6f}", flush=True)
