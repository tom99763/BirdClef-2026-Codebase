"""
batch126: Novel embedding-based methods beyond priority list
M1: Prototype Network (per-species positive centroid cosine similarity)
M2: Per-species LDA (binary LinearDiscriminantAnalysis)
M3: KNN in logit space (use logit cosine similarity as distance)
M4: Gaussian Naive Bayes per species
M5: Ensemble of best batch125 methods
M6: Proto+IDF blend (prototype scores blended with IDF co-occurrence)
"""
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-9

# ── data ──────────────────────────────────────────────────────────────────────
data     = np.load('outputs/perch_labeled_ss.npz', allow_pickle=True)
EMB      = data['emb'].astype(np.float32)
LOGITS   = data['logits'].astype(np.float32)
LABELS   = data['labels'].astype(np.float32)
fnames   = data['filenames']
file_list = data['file_list']
file_ids = np.array([np.where(file_list == fn)[0][0] for fn in fnames])
n_files  = len(file_list)
N_SP     = LABELS.shape[1]
sp_present = (LABELS.max(0) > 0)

EMB_NORM   = normalize(EMB, norm='l2')
LOGIT_NORM = normalize(LOGITS, norm='l2')

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

LOGIT_SIG = sigmoid(LOGITS)

print(f"[batch126] EMB={EMB.shape}, n_files={n_files}, sp_present={sp_present.sum()}")

# ── JSON store ─────────────────────────────────────────────────────────────────
results_path = Path('outputs/embed_prior_results.json')
with open(results_path) as f:
    store = json.load(f)
tried     = {e['method'] for e in store.get('experiments', [])}
best_loo  = store['best']['loo_auc']
best_method = store['best']['method']
print(f"[batch126] Current best: {best_method} LOO={best_loo:.6f}")

def loo_auc(pred_probs):
    auc_list = []
    for fi in range(n_files):
        mask = (file_ids == fi)
        file_score = pred_probs[mask].mean(0)
        file_true  = LABELS[mask].max(0)
        sp = sp_present
        if sp.sum() < 2: continue
        try:
            auc_list.append(roc_auc_score(file_true[sp], file_score[sp]))
        except Exception:
            pass
    return float(np.mean(auc_list))

def save_result(method, score, config, note=''):
    global best_loo, best_method
    delta = score - best_loo
    r = {'method': method, 'loo_auc': score, 'config': config, 'note': note}
    store['experiments'].append(r)
    if score > best_loo:
        best_loo = score
        best_method = method
        store['best'] = {'method': method, 'loo_auc': score}
    with open(results_path, 'w') as f:
        json.dump(store, f, indent=2)
    return delta

# ═════════════════════════════════════════════════════════════════════════════
# M1: Prototype Network (per-species positive centroid)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M1] Prototype Network...")

def proto_net_loo(pca_dim=128, use_idf_weight=False, power=1.0):
    """
    Per-species prototype = mean of positive training windows.
    Score = cosine similarity to prototype.
    """
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)

    # IDF weights for species
    n_pos_files = np.zeros(N_SP)
    for fi in range(n_files):
        mask = (file_ids == fi)
        has_sp = LABELS[mask].max(0) > 0.5
        n_pos_files += has_sp
    idf = np.clip(np.log((n_files + 1) / (n_pos_files + 1)), 0, None)
    idf_w = idf ** 0.75 / (idf.mean() + EPS)

    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = normalize(pca.fit_transform(X_tr), norm='l2')
        X_te_pca = normalize(pca.transform(X_te), norm='l2')

        sp_scores = np.zeros((len(X_te), N_SP), dtype=np.float32)
        for sp in range(N_SP):
            pos_mask = y_tr[:, sp] > 0.5
            if pos_mask.sum() == 0:
                continue
            pos_emb = X_tr_pca[pos_mask]
            if use_idf_weight:
                # weight positive examples by their IDF score
                w = idf_w[sp] * np.ones(pos_mask.sum())
                prototype = (pos_emb * w[:, None]).sum(0)
            else:
                prototype = pos_emb.mean(0)
            prototype = prototype / (np.linalg.norm(prototype) + EPS)
            sims = X_te_pca @ prototype
            if power != 1.0:
                sims = np.sign(sims) * np.abs(sims) ** power
            sp_scores[:, sp] = (sims + 1) / 2  # map [-1,1] → [0,1]

        pred[test_mask] = sp_scores

    return pred

m1_configs = [
    {'pca_dim': 128, 'use_idf_weight': False, 'power': 1.0},
    {'pca_dim': 64,  'use_idf_weight': False, 'power': 1.0},
    {'pca_dim': 256, 'use_idf_weight': False, 'power': 1.0},
    {'pca_dim': 128, 'use_idf_weight': True,  'power': 1.0},
    {'pca_dim': 128, 'use_idf_weight': False, 'power': 2.0},
    {'pca_dim': 64,  'use_idf_weight': True,  'power': 2.0},
]

m1_best = 0.0
for cfg in m1_configs:
    mname = f'proto_p{cfg["pca_dim"]}_idf{int(cfg["use_idf_weight"])}_pw{int(cfg["power"]*10):02d}'
    if mname in tried:
        print(f'  {mname}: skip')
        continue
    pred = proto_net_loo(**cfg)
    score = loo_auc(pred)
    delta = save_result(mname, score, cfg)
    flag  = ' ← NEW BEST!' if score > best_loo - (score - best_loo) else ''
    tag   = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m1_best = max(m1_best, score)
print(f"  M1 done, best={m1_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M2: Per-species LDA
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M2] Per-species LDA...")

def lda_loo(pca_dim=64, shrinkage='auto'):
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        sp_scores = np.zeros((len(X_te), N_SP), dtype=np.float32)
        for sp in range(N_SP):
            y_sp = (y_tr[:, sp] > 0.5).astype(int)
            if y_sp.sum() < 2 or (1 - y_sp).sum() < 2:
                sp_scores[:, sp] = y_sp.mean()
                continue
            try:
                lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
                lda.fit(X_tr_pca, y_sp)
                p = lda.predict_proba(X_te_pca)
                sp_scores[:, sp] = p[:, 1]
            except Exception:
                sp_scores[:, sp] = y_sp.mean()

        pred[test_mask] = sp_scores
    return pred

m2_configs = [
    {'pca_dim': 64,  'shrinkage': 'auto'},
    {'pca_dim': 128, 'shrinkage': 'auto'},
    {'pca_dim': 32,  'shrinkage': 'auto'},
    {'pca_dim': 64,  'shrinkage': 0.5},
]

m2_best = 0.0
for cfg in m2_configs:
    sh = cfg['shrinkage']
    mname = f'lda_p{cfg["pca_dim"]}_sh{sh}'.replace('.', 'p')
    if mname in tried:
        print(f'  {mname}: skip')
        continue
    pred = lda_loo(**cfg)
    score = loo_auc(pred)
    delta = save_result(mname, score, cfg)
    tag   = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag  = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m2_best = max(m2_best, score)
print(f"  M2 done, best={m2_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M3: KNN in logit space
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M3] KNN in logit space...")

def logit_knn_loo(k=40, temp=10.0):
    """Use logit cosine similarity to find K nearest training windows."""
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        L_tr = LOGIT_NORM[train_mask]
        L_te = LOGIT_NORM[test_mask]
        y_tr = LABELS[train_mask]

        sim = L_te @ L_tr.T   # [n_te, n_tr]
        nn_idx = np.argsort(-sim, axis=1)[:, :k]
        nn_sim = sim[np.arange(len(L_te))[:, None], nn_idx]
        # softmax weights
        exp_s = np.exp(nn_sim * temp - (nn_sim * temp).max(1, keepdims=True))
        weights = exp_s / (exp_s.sum(1, keepdims=True) + EPS)
        nn_labels = y_tr[nn_idx]
        pred[test_mask] = np.einsum('tk,tkc->tc', weights, nn_labels)
    return pred

m3_configs = [
    {'k': 40, 'temp': 10.0},
    {'k': 20, 'temp': 10.0},
    {'k': 60, 'temp': 10.0},
    {'k': 40, 'temp':  5.0},
    {'k': 40, 'temp': 20.0},
    {'k': 20, 'temp':  5.0},
]

m3_best = 0.0
for cfg in m3_configs:
    mname = f'logit_knn_k{cfg["k"]}_t{int(cfg["temp"]):02d}'
    if mname in tried:
        print(f'  {mname}: skip')
        continue
    pred = logit_knn_loo(**cfg)
    score = loo_auc(pred)
    delta = save_result(mname, score, cfg)
    tag   = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag  = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m3_best = max(m3_best, score)
print(f"  M3 done, best={m3_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M4: Gaussian Naive Bayes per species
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M4] Gaussian Naive Bayes...")

def gnb_loo(pca_dim=64, var_smoothing=1e-9):
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        sp_scores = np.zeros((len(X_te), N_SP), dtype=np.float32)
        for sp in range(N_SP):
            y_sp = (y_tr[:, sp] > 0.5).astype(int)
            if y_sp.sum() < 1 or y_sp.mean() > 0.999:
                sp_scores[:, sp] = y_sp.mean()
                continue
            try:
                gnb = GaussianNB(var_smoothing=var_smoothing)
                gnb.fit(X_tr_pca, y_sp)
                p = gnb.predict_proba(X_te_pca)
                sp_scores[:, sp] = p[:, 1] if p.shape[1] > 1 else 1.0 - p[:, 0]
            except Exception:
                sp_scores[:, sp] = y_sp.mean()

        pred[test_mask] = sp_scores
    return pred

m4_configs = [
    {'pca_dim': 64,  'var_smoothing': 1e-9},
    {'pca_dim': 32,  'var_smoothing': 1e-9},
    {'pca_dim': 64,  'var_smoothing': 1e-6},
    {'pca_dim': 128, 'var_smoothing': 1e-9},
]

m4_best = 0.0
for cfg in m4_configs:
    mname = f'gnb_p{cfg["pca_dim"]}_vs{int(-np.log10(cfg["var_smoothing"])):d}'
    if mname in tried:
        print(f'  {mname}: skip')
        continue
    pred = gnb_loo(**cfg)
    score = loo_auc(pred)
    delta = save_result(mname, score, cfg)
    tag   = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag  = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m4_best = max(m4_best, score)
print(f"  M4 done, best={m4_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M5: Ensemble blend of best batch125 methods
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M5] Ensemble of best prior methods...")

def attn_knn_loo(k=20, pca_dim=64, temp=1.0):
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        L_tr = LOGIT_NORM[train_mask]
        L_te = LOGIT_NORM[test_mask]
        y_tr = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = normalize(pca.fit_transform(X_tr), norm='l2')
        X_te_pca = normalize(pca.transform(X_te), norm='l2')

        sim = X_te_pca @ X_tr_pca.T
        nn_idx = np.argsort(-sim, axis=1)[:, :k]
        nn_labels = y_tr[nn_idx]
        sp_scores = np.zeros((len(X_te), N_SP), dtype=np.float32)
        for ti in range(len(X_te)):
            k_idx = nn_idx[ti]
            l_sim = L_te[ti] @ L_tr[k_idx].T
            exp_s = np.exp(l_sim / temp - (l_sim / temp).max())
            attn = exp_s / (exp_s.sum() + EPS)
            sp_scores[ti] = attn @ y_tr[k_idx]
        pred[test_mask] = sp_scores
    return pred

def emb_knn_loo(k=40, pca_dim=128, temp=10.0):
    pred = np.zeros((len(EMB_NORM), N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = normalize(pca.fit_transform(X_tr), norm='l2')
        X_te_pca = normalize(pca.transform(X_te), norm='l2')

        sim = X_te_pca @ X_tr_pca.T
        nn_idx = np.argsort(-sim, axis=1)[:, :k]
        nn_sim = sim[np.arange(len(X_te))[:, None], nn_idx]
        exp_s = np.exp(nn_sim * temp - (nn_sim * temp).max(1, keepdims=True))
        weights = exp_s / (exp_s.sum(1, keepdims=True) + EPS)
        pred[test_mask] = np.einsum('tk,tkc->tc', weights, y_tr[nn_idx])
    return pred

print("  Pre-computing component predictions...")
attn_pred = attn_knn_loo(k=20, pca_dim=64, temp=1.0)
emb_pred  = emb_knn_loo(k=40, pca_dim=128, temp=10.0)
logit_knn_pred = logit_knn_loo(k=40, temp=10.0)
print("  Done pre-computing.")

ens_configs = [
    {'wa': 0.5,  'we': 0.5,  'wl': 0.0,  'name': 'ens_attn50_emb50'},
    {'wa': 0.4,  'we': 0.4,  'wl': 0.2,  'name': 'ens_a40_e40_l20'},
    {'wa': 0.6,  'we': 0.3,  'wl': 0.1,  'name': 'ens_a60_e30_l10'},
    {'wa': 0.33, 'we': 0.33, 'wl': 0.34, 'name': 'ens_equal3'},
    {'wa': 0.5,  'we': 0.3,  'wl': 0.2,  'name': 'ens_a50_e30_l20'},
    {'wa': 0.7,  'we': 0.2,  'wl': 0.1,  'name': 'ens_a70_e20_l10'},
]

m5_best = 0.0
for cfg in ens_configs:
    mname = cfg['name']
    if mname in tried:
        print(f'  {mname}: skip')
        continue
    pred = cfg['wa'] * attn_pred + cfg['we'] * emb_pred + cfg['wl'] * logit_knn_pred
    score = loo_auc(pred)
    delta = save_result(mname, score, {'wa': cfg['wa'], 'we': cfg['we'], 'wl': cfg['wl']})
    tag   = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag  = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m5_best = max(m5_best, score)
print(f"  M5 done, best={m5_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M6: Proto + IDF co-occurrence hybrid blend
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M6] Proto + co-occurrence blend...")

# Soft-gate co-occurrence (from main chain)
COOC_NORM = None
fl = np.zeros((n_files, N_SP), dtype=np.float32)
for fi in range(n_files):
    mask = (file_ids == fi)
    fl[fi] = (LABELS[mask].max(0) > 0.5).astype(np.float32)

count_i = fl.sum(0) + EPS
cooc = fl.T @ fl / count_i[:, None]
np.fill_diagonal(cooc, 0)
COOC_NORM = cooc / (cooc.sum(1, keepdims=True) + EPS)

n_pos_files = fl.sum(0)
IDF = np.clip(np.log((n_files + 1) / (n_pos_files + 1)), 0, None)
IDF_W075 = IDF ** 0.75 / (IDF.mean() + EPS)

def soft_cooc(scores, center=0.53, slope=37.0, alpha=0.086, idf_w=None):
    smoothed = np.zeros_like(scores)
    for fi in range(n_files):
        s = scores[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate
        if idf_w is not None:
            s_gated = s_gated * idf_w
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS:
            contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def idf_cooc(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    s_pow = np.clip(scores, 0, 1) ** 2.0
    s_cooc = soft_cooc(s_pow, center=center, slope=slope, alpha=alpha, idf_w=IDF_W075)
    return (1 - blend) * scores + blend * s_cooc

def two_round(scores):
    r1 = soft_cooc(scores, center=0.54, slope=41.0, alpha=0.089)
    r2 = soft_cooc(r1,     center=0.53, slope=37.0, alpha=0.040)
    return r2

# Compute file-level proto scores
def compute_proto_file_scores(pca_dim=128):
    """Compute prototype-based LOO scores at FILE level [n_files, N_SP]."""
    file_scores = np.zeros((n_files, N_SP), dtype=np.float32)
    for fi in range(n_files):
        test_mask  = (file_ids == fi)
        train_mask = ~test_mask
        X_tr = EMB_NORM[train_mask]
        X_te = EMB_NORM[test_mask]
        y_tr = LABELS[train_mask]

        pca = PCA(n_components=min(pca_dim, X_tr.shape[0]-1, X_tr.shape[1]))
        X_tr_pca = normalize(pca.fit_transform(X_tr), norm='l2')
        X_te_pca = normalize(pca.transform(X_te), norm='l2')

        sp_scores = np.zeros((len(X_te), N_SP), dtype=np.float32)
        for sp in range(N_SP):
            pos_mask = y_tr[:, sp] > 0.5
            if pos_mask.sum() == 0:
                continue
            prototype = X_tr_pca[pos_mask].mean(0)
            prototype /= (np.linalg.norm(prototype) + EPS)
            sims = X_te_pca @ prototype
            sp_scores[:, sp] = (sims + 1) / 2

        file_scores[fi] = sp_scores.mean(0)
    return file_scores

# We need double_best file-level scores as our base
# Load from existing PKL - use file_prob_max as approximation
pkl = pickle.load(open('outputs/embed_prior_model.pkl', 'rb'))
double_best_file = pkl['file_prob_max'].astype(np.float32)  # [66, 234]

print("  Computing proto file scores (LOO)...")
proto_file = compute_proto_file_scores(pca_dim=128)

# Run IDF co-occurrence on double_best_file
idf_result   = idf_cooc(double_best_file)
tworound_res = two_round(double_best_file)
base_blend   = 0.85 * idf_result + 0.15 * tworound_res

# Evaluate base_blend AUC
def file_loo_auc(file_scores):
    auc_list = []
    for fi in range(n_files):
        file_score = file_scores[fi]
        file_true  = fl[fi]
        sp = sp_present
        if sp.sum() < 2: continue
        try:
            auc_list.append(roc_auc_score(file_true[sp], file_score[sp]))
        except Exception:
            pass
    return float(np.mean(auc_list))

print(f"  base_blend (3way_i85_r15_d00) AUC: {file_loo_auc(base_blend):.6f}")
print(f"  proto_file AUC: {file_loo_auc(proto_file):.6f}")

m6_configs = [
    {'w_proto': 0.02},
    {'w_proto': 0.05},
    {'w_proto': 0.10},
    {'w_proto': 0.15},
    {'w_proto': 0.20},
]

m6_best = 0.0
for cfg in m6_configs:
    wp = cfg['w_proto']
    mname = f'proto_cooc_wp{int(wp*100):02d}'
    if mname in tried:
        print(f'  {mname}: skip')
        continue
    blended = (1 - wp) * base_blend + wp * proto_file
    score = file_loo_auc(blended)
    delta = save_result(mname, score, cfg, 'proto blend on top of 3way_i85_r15_d00')
    tag   = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag  = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m6_best = max(m6_best, score)
print(f"  M6 done, best={m6_best:.6f}")

# ── final summary ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"[batch126] SUMMARY")
print(f"  Global best: {store['best']['method']} LOO={store['best']['loo_auc']:.6f}")
print(f"  M1 Proto Network:     {m1_best:.6f}")
print(f"  M2 LDA per species:   {m2_best:.6f}")
print(f"  M3 Logit-space KNN:   {m3_best:.6f}")
print(f"  M4 Gaussian NB:       {m4_best:.6f}")
print(f"  M5 Ensemble:          {m5_best:.6f}")
print(f"  M6 Proto+Cooc blend:  {m6_best:.6f}")
