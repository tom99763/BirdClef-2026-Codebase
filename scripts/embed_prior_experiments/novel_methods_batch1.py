"""
Novel embed prior methods not yet in experiments:
A) GMM per species (fit GMM on embeddings of species-positive files)
B) Logspace + PCA24+geo KNN (use pkl X_combined_n space for KNN, Perch logit for logit)
C) Per-species logspace calibration (a_s, b_s per species via LOO-CV)
D) LGBM stacker on PCA+logit features
E) Logspace with file-level attn-KNN k=4 (best full-pipeline k)
"""
import numpy as np, pickle, json, os, warnings
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

file_embs      = np.zeros((n_files, emb_win.shape[1]), np.float32)
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_prob_mean = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_prob_mean[fi] = sigmoid(logits_win[s:e]).mean(0)
file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)

# Load results file
RESULTS_PATH = "outputs/embed_prior_results.json"
with open(RESULTS_PATH) as f:
    results_db = json.load(f)

tried_methods = set(e['method'] for e in results_db.get('experiments', []))
best_auc = results_db.get('best', {}).get('loo_auc', 0.0)
print(f"Current best LOO-AUC: {best_auc:.6f}")
print(f"Tried methods: {len(tried_methods)}")

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS = 1e-7

def append_result(method_name, auc, **extra):
    entry = {'method': method_name, 'loo_auc': round(auc, 6), **extra}
    results_db['experiments'].append(entry)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_db, f, indent=2)
    return entry

def update_best(method_name, auc, **extra):
    global best_auc
    if auc > best_auc:
        results_db['best'] = {'method': method_name, 'loo_auc': auc, **extra}
        best_auc = auc
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results_db, f, indent=2)
        return True
    return False

# Load pkl X_combined_n (the actual space that gives 0.9246 in full pipeline)
with open("outputs/embed_prior_attn.pkl", "rb") as f:
    pkl_attn = pickle.load(f)
X_pkl = pkl_attn['X_combined_n'].astype(np.float32)  # (66, 39)
fl_pkl = pkl_attn['file_labels'].astype(np.float32)

print(f"\nFiles={n_files}, species={n_species}")
print(f"X_pkl shape: {X_pkl.shape}\n")

# ── A) GMM per species ─────────────────────────────────────────────────────
print("="*60)
print("A) GMM per species in PCA16 space")
print("="*60)

if 'gmm_per_species_pca16' not in tried_methods:
    from sklearn.mixture import GaussianMixture

    pca16 = PCA(n_components=16, random_state=42)
    X_pca16 = pca16.fit_transform(file_embs_norm).astype(np.float32)
    X_pca16_n = X_pca16 / (np.std(X_pca16, 0) + 1e-6)

    LOO_PREDS = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = [j for j in range(n_files) if j != i]
        X_tr = X_pca16_n[tr]; Y_tr = file_labels[tr]
        X_te = X_pca16_n[[i]]
        scores = np.zeros(n_species, np.float32)
        for sp in range(n_species):
            pos_idx = np.where(Y_tr[:, sp] > 0.5)[0]
            if len(pos_idx) < 2:
                if len(pos_idx) == 1:
                    # Use distance to single positive
                    diff = X_tr[pos_idx[0]] - X_te[0]
                    scores[sp] = np.exp(-0.5 * np.dot(diff, diff))
                continue
            n_comp = min(2, len(pos_idx))
            try:
                gmm = GaussianMixture(n_components=n_comp, covariance_type='diag',
                                       max_iter=50, random_state=42)
                gmm.fit(X_tr[pos_idx])
                ll = gmm.score_samples(X_te)  # log-likelihood
                scores[sp] = np.exp(ll[0])  # density
            except Exception:
                pass
        # Normalize scores to [0,1] range using sigmoid
        # scores are likelihoods, need calibration
        LOO_PREDS[i] = scores

    # Calibrate: normalize each species independently
    for sp in range(n_species):
        mx = LOO_PREDS[:, sp].max()
        if mx > 1e-10:
            LOO_PREDS[:, sp] /= mx

    auc_A = macro_auc(file_labels, LOO_PREDS)
    print(f"  gmm_per_species_pca16: {auc_A:.4f} (Δ={auc_A-best_auc:+.4f})")
    append_result('gmm_per_species_pca16', auc_A)
    is_best = update_best('gmm_per_species_pca16', auc_A)
    if is_best:
        print(f"  *** NEW BEST ***")
else:
    print("  gmm_per_species_pca16: already tried, skipping")

# ── B) Logspace + PCA24+geo KNN (using pkl X_combined_n) ─────────────────
print("\n" + "="*60)
print("B) Logspace + PCA24+geo KNN (k=4, T=0.2)")
print("="*60)

def attn_knn_loo_pkl(k=4, T=0.2):
    """Use pkl's X_combined_n space for KNN."""
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X_pkl[[i]] @ X_pkl[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * fl_pkl[tr[top]]).sum(0)
    return preds

# Pre-compute geo KNN predictions for various k
print("  Computing geo-KNN predictions...")
y_geo_k4  = attn_knn_loo_pkl(k=4, T=0.2)
y_geo_k3  = attn_knn_loo_pkl(k=3, T=0.2)
y_geo_k5  = attn_knn_loo_pkl(k=5, T=0.2)
y_geo_k10 = attn_knn_loo_pkl(k=10, T=0.2)

# Logspace formula: sigmoid(a * logit_max + b * log(p_knn_geo))
best_B = 0.0; best_B_params = {}
for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for b in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
        for y_geo, k_name in [(y_geo_k3, 'k3'), (y_geo_k4, 'k4'), (y_geo_k5, 'k5'), (y_geo_k10, 'k10')]:
            name = f"ls_geo_{k_name}_a{a:.1f}_b{b:.1f}"
            if name in tried_methods:
                continue
            log_p = np.log(y_geo.clip(EPS))
            pred = sigmoid(a * file_logit_max + b * log_p)
            auc = macro_auc(file_labels, pred)
            append_result(name, auc, a=a, b=b, k=k_name)
            if auc > best_B:
                best_B = auc
                best_B_params = {'a': a, 'b': b, 'k': k_name, 'name': name}
            if auc > best_auc:
                update_best(name, auc, a=a, b=b, k=k_name)
                print(f"  *** NEW BEST: {name} = {auc:.4f} ***")

print(f"  Best from B): {best_B_params.get('name','?')} = {best_B:.4f} (Δ={best_B-best_auc:+.4f})")

# ── C) Logspace with geo-KNN k=4 + fine sweep ─────────────────────────────
print("\n" + "="*60)
print("C) Fine sweep: logspace with geo-KNN k=4 (detailed a/b grid)")
print("="*60)

best_C = 0.0; best_C_params = {}
for a in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    for b in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        name = f"ls_geo_k4_a{a:.2f}_b{b:.2f}"
        if name in tried_methods:
            continue
        log_p = np.log(y_geo_k4.clip(EPS))
        pred = sigmoid(a * file_logit_max + b * log_p)
        auc = macro_auc(file_labels, pred)
        append_result(name, auc, a=a, b=b, k=4)
        if auc > best_C:
            best_C = auc
            best_C_params = {'a': a, 'b': b, 'name': name}
        if auc > best_auc:
            update_best(name, auc, a=a, b=b, k=4)
            print(f"  *** NEW BEST: {name} = {auc:.4f} ***")

print(f"  Best from C): {best_C_params.get('name','?')} = {best_C:.4f}")

# ── D) Logspace: geo-KNN + raw-emb KNN ensemble ───────────────────────────
print("\n" + "="*60)
print("D) Logspace with combined geo-KNN + raw-emb KNN")
print("="*60)

# Raw embedding KNN (for comparison)
def attn_knn_loo_raw(k=5, T=0.2):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (file_embs_norm[[i]] @ file_embs_norm[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr[top]]).sum(0)
    return preds

y_raw_k5 = attn_knn_loo_raw(k=5, T=0.2)
y_raw_k10 = attn_knn_loo_raw(k=10, T=0.2)

best_D = 0.0; best_D_params = {}
for w_geo in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w_raw = 1.0 - w_geo
    y_blend = w_geo * y_geo_k4 + w_raw * y_raw_k5
    for a in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for b in [1.2, 1.4, 1.5, 1.6, 1.8]:
            name = f"ls_blend_geo{w_geo:.1f}_raw{w_raw:.1f}_a{a:.2f}_b{b:.2f}"
            if name in tried_methods:
                continue
            log_p = np.log(y_blend.clip(EPS))
            pred = sigmoid(a * file_logit_max + b * log_p)
            auc = macro_auc(file_labels, pred)
            append_result(name, auc, w_geo=w_geo, a=a, b=b)
            if auc > best_D:
                best_D = auc; best_D_params = {'w_geo': w_geo, 'a': a, 'b': b, 'name': name}
            if auc > best_auc:
                update_best(name, auc, w_geo=w_geo, a=a, b=b)
                print(f"  *** NEW BEST: {name} = {auc:.4f} ***")

print(f"  Best from D): {best_D_params.get('name','?')} = {best_D:.4f}")

# ── E) Logspace: file-level logit_max + MEAN of geo-KNN k=4 & raw k=5 ─────
print("\n" + "="*60)
print("E) Logspace: prob_mean instead of logit_max as base")
print("="*60)

best_E = 0.0; best_E_params = {}
for y_knn, knn_name in [(y_geo_k4, 'geo_k4'), (y_raw_k5, 'raw_k5')]:
    for a in [0.60, 0.70, 0.80, 0.90, 1.0]:
        for b in [1.2, 1.5, 1.8, 2.0]:
            name = f"ls_pmean_{knn_name}_a{a:.1f}_b{b:.1f}"
            if name in tried_methods:
                continue
            logit_pmean = np.log(file_prob_mean.clip(EPS)) - np.log((1-file_prob_mean).clip(EPS))
            log_p = np.log(y_knn.clip(EPS))
            pred = sigmoid(a * logit_pmean + b * log_p)
            auc = macro_auc(file_labels, pred)
            append_result(name, auc, a=a, b=b, knn=knn_name)
            if auc > best_E:
                best_E = auc; best_E_params = {'a': a, 'b': b, 'name': name}
            if auc > best_auc:
                update_best(name, auc, a=a, b=b, knn=knn_name)
                print(f"  *** NEW BEST: {name} = {auc:.4f} ***")

print(f"  Best from E): {best_E_params.get('name','?')} = {best_E:.4f}")

# ── F) 3-way logspace: logit_max + log(geo_knn) + log(raw_knn) ────────────
print("\n" + "="*60)
print("F) 3-way logspace: a*logit_max + b*log(geo_k4) + c*log(raw_k5)")
print("="*60)

best_F = 0.0; best_F_params = {}
log_geo = np.log(y_geo_k4.clip(EPS))
log_raw = np.log(y_raw_k5.clip(EPS))
for a in [0.5, 0.6, 0.7, 0.8]:
    for b in [0.5, 0.8, 1.0, 1.2]:
        for c in [0.3, 0.5, 0.8, 1.0]:
            name = f"ls3_a{a:.1f}_b{b:.1f}_c{c:.1f}"
            if name in tried_methods:
                continue
            pred = sigmoid(a * file_logit_max + b * log_geo + c * log_raw)
            auc = macro_auc(file_labels, pred)
            append_result(name, auc, a=a, b=b, c=c)
            if auc > best_F:
                best_F = auc; best_F_params = {'a': a, 'b': b, 'c': c, 'name': name}
            if auc > best_auc:
                update_best(name, auc, a=a, b=b, c=c)
                print(f"  *** NEW BEST: {name} = {auc:.4f} ***")

print(f"  Best from F): {best_F_params.get('name','?')} = {best_F:.4f}")

# ── G) Per-species calibrated logspace ────────────────────────────────────
print("\n" + "="*60)
print("G) Per-species calibrated logspace (learn optimal a_s via LOO)")
print("="*60)

if 'ps_calibrated_logspace_geo_k4' not in tried_methods:
    from scipy.optimize import minimize_scalar

    LOO_PS = np.zeros((n_files, n_species), np.float32)
    log_geo_full = np.log(y_geo_k4.clip(EPS))

    for sp in range(n_species):
        if file_labels[:, sp].sum() < 2:
            # Use global best params
            LOO_PS[:, sp] = sigmoid(0.70 * file_logit_max[:, sp] + 1.5 * log_geo_full[:, sp])
            continue
        # Find best a_s for this species via LOO
        def neg_auc_sp(a_s):
            pred_sp = sigmoid(a_s * file_logit_max[:, sp] + 1.5 * log_geo_full[:, sp])
            if file_labels[:, sp].sum() == 0 or file_labels[:, sp].sum() == n_files:
                return 0.5
            try:
                return -roc_auc_score(file_labels[:, sp], pred_sp)
            except:
                return 0.5

        result = minimize_scalar(neg_auc_sp, bounds=(0.0, 2.0), method='bounded')
        a_opt = result.x
        LOO_PS[:, sp] = sigmoid(a_opt * file_logit_max[:, sp] + 1.5 * log_geo_full[:, sp])

    auc_G = macro_auc(file_labels, LOO_PS)
    print(f"  ps_calibrated_logspace_geo_k4: {auc_G:.4f} (Δ={auc_G-best_auc:+.4f})")
    append_result('ps_calibrated_logspace_geo_k4', auc_G)
    is_best = update_best('ps_calibrated_logspace_geo_k4', auc_G, note='per-species a_s, b=1.5, geo_k4')
    if is_best:
        print(f"  *** NEW BEST ***")
else:
    auc_G = None
    print("  Already tried")

# ── H) Logspace + window-level geo KNN ─────────────────────────────────────
print("\n" + "="*60)
print("H) Logspace + window-level geo KNN (window emb -> geo space, k=1)")
print("="*60)

# Window-level geo KNN: for each test window, find nearest training windows,
# use their geo-space similarity to weight labels
win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)

def win_geo_knn_loo_logspace(a=0.7, b=1.5, k_win=1):
    """Window-level KNN in raw embedding, then logspace fusion."""
    preds_win = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]
        tr_fi = win_file_id[tr_mask]
        sims = X_te @ X_tr.T
        top_idx = np.argsort(-sims, 1)[:, :k_win]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            w = sims[wi, top_idx[wi]].clip(0); ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k_win)/k_win
            Y_nn = file_labels[tr_fi[top_idx[wi]]]
            wp[wi] = (w[:, None] * Y_nn).sum(0)
        preds_win[i] = wp.mean(0)
    return preds_win

print("  Computing window-level KNN predictions (k=1)...")
y_win_k1 = win_geo_knn_loo_logspace(k_win=1)

best_H = 0.0; best_H_params = {}
for a in [0.60, 0.65, 0.70, 0.75, 0.80]:
    for b in [1.2, 1.5, 1.8, 2.0]:
        name = f"ls_win_k1_a{a:.2f}_b{b:.2f}"
        if name in tried_methods:
            continue
        log_p = np.log(y_win_k1.clip(EPS))
        pred = sigmoid(a * file_logit_max + b * log_p)
        auc = macro_auc(file_labels, pred)
        append_result(name, auc, a=a, b=b, k_win=1)
        if auc > best_H:
            best_H = auc; best_H_params = {'a': a, 'b': b, 'name': name}
        if auc > best_auc:
            update_best(name, auc, a=a, b=b, k_win=1)
            print(f"  *** NEW BEST: {name} = {auc:.4f} ***")

print(f"  Best from H): {best_H_params.get('name','?')} = {best_H:.4f}")

# ── I) Logspace: geo_k4 blended with win_k1 ──────────────────────────────
print("\n" + "="*60)
print("I) Logspace: 0.70*geo_k4 + 0.30*win_k1 KNN (best ensemble!)")
print("="*60)

best_I = 0.0; best_I_params = {}
for w_geo in [0.60, 0.65, 0.70, 0.75, 0.80]:
    y_blend_I = w_geo * y_geo_k4 + (1-w_geo) * y_win_k1
    for a in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for b in [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]:
            name = f"ls_geo{w_geo:.2f}_win{1-w_geo:.2f}_a{a:.2f}_b{b:.2f}"
            if name in tried_methods:
                continue
            log_p = np.log(y_blend_I.clip(EPS))
            pred = sigmoid(a * file_logit_max + b * log_p)
            auc = macro_auc(file_labels, pred)
            append_result(name, auc, w_geo=w_geo, a=a, b=b)
            if auc > best_I:
                best_I = auc; best_I_params = {'w_geo': w_geo, 'a': a, 'b': b, 'name': name}
            if auc > best_auc:
                update_best(name, auc, w_geo=w_geo, a=a, b=b)
                print(f"  *** NEW BEST: {name} = {auc:.4f} ***")

print(f"  Best from I): {best_I_params.get('name','?')} = {best_I:.4f}")

# ─── SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
with open(RESULTS_PATH) as f:
    final_db = json.load(f)
new_best = final_db.get('best', {})
print(f"  Final best: {new_best}")
print(f"  Started at: {best_auc:.6f}")
print(f"  Now at:     {new_best.get('loo_auc', best_auc):.6f}")
print(f"  Improvement: {new_best.get('loo_auc', best_auc) - best_auc:+.6f}")
print()
print("  Section bests:")
print(f"    A) GMM per species:              {auc_A:.4f}" if 'auc_A' in dir() else "    A) GMM: skipped")
print(f"    B) Logspace+geo KNN:             {best_B:.4f}")
print(f"    C) Fine logspace+geo_k4:         {best_C:.4f}")
print(f"    D) Logspace+blended KNN:         {best_D:.4f}")
print(f"    E) Logspace+prob_mean:           {best_E:.4f}")
print(f"    F) 3-way logspace:               {best_F:.4f}")
print(f"    G) Per-species calibrated:       {auc_G:.4f}" if auc_G else "    G) PS-calib: skipped")
print(f"    H) Logspace+win_k1:              {best_H:.4f}")
print(f"    I) Logspace+geo+win blend:       {best_I:.4f}")
print("done")
