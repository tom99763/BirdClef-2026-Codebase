"""
Full Pipeline Benchmark: SED + ProtoSSM + Embed Prior vs Train Soundscape Labels
Produces: reports/current_subs_benchmark.html

Pipeline (same as Kaggle notebook):
  proto_probs = sigmoid(logit_max[file])   # ProtoSSM proxy
  sed_probs   = max(SED probs per file)    # SED ONNX predictions
  base = VLOM_blend(proto, sed, w=0.5/0.5)
  full = base + lambda * embed_prior_logit  (or logspace for v9)

Data:
  - outputs/perch_labeled_ss.npz  → ProtoSSM logits
  - outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz  → SED predictions
  - outputs/embed_prior_*.pkl  → embed prior KNN params
"""
import numpy as np, pickle, re, os, json
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

# File-level ProtoSSM proxy (Perch logit → prob)
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
emb_win_norm   = normalize(emb_win,   norm='l2').astype(np.float32)
win_file_id    = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

# ── Load SED predictions ──────────────────────────────────────────────────
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_row_ids = sed_npz['row_ids']
sed_probs_all = sed_npz['probs'].astype(np.float32)

# Build file_name → windows mapping for SED
sed_by_file = {}
for i, rid in enumerate(sed_row_ids):
    # rid format: BC2026_Train_XXXX_SXX_YYYYMMDD_HHMMSS_<sec>
    fname_base = '_'.join(str(rid).split('_')[:-1])  # remove last part (sec)
    if fname_base not in sed_by_file:
        sed_by_file[fname_base] = []
    sed_by_file[fname_base].append(i)

# File-level SED aggregation (max over windows)
file_sed_max  = np.zeros((n_files, n_species), np.float32)
file_sed_mean = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fname_base = fname.replace('.ogg', '').replace('.flac', '')
    if fname_base in sed_by_file:
        idxs = sed_by_file[fname_base]
        win_probs = sed_probs_all[idxs]
        file_sed_max[fi]  = win_probs.max(0)
        file_sed_mean[fi] = win_probs.mean(0)
    else:
        print(f"WARNING: No SED predictions for {fname_base}")

print(f"Files={n_files}, species={n_species}")
print(f"SED predictions loaded: {sed_probs_all.shape}")

# ── Geo features ──────────────────────────────────────────────────────────
SITES = ['S03','S08','S09','S13','S15','S18','S19','S22','S23']
site2idx = {s: i for i, s in enumerate(SITES)}
file_sites  = np.zeros(n_files, np.int32)
file_hours  = np.zeros(n_files, np.float32)
file_months = np.zeros(n_files, np.float32)
file_days   = np.zeros(n_files, np.float32)
for fi, fname in enumerate(file_list):
    m = re.match(r'BC2026_Train_\d+_(S\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})', str(fname))
    if m:
        site, yr, mo, dy, hr, mn = m.groups()
        file_sites[fi]  = site2idx.get(site, 0)
        file_hours[fi]  = int(hr)
        file_months[fi] = int(mo)
        dpm = [0,31,28,31,30,31,30,31,31,30,31,30,31]
        file_days[fi] = sum(dpm[:int(mo)]) + int(dy)

site_oh   = np.eye(len(SITES), dtype=np.float32)[file_sites]
hour_enc  = np.stack([np.sin(2*np.pi*file_hours/24), np.cos(2*np.pi*file_hours/24)], 1).astype(np.float32)
month_enc = np.stack([np.sin(2*np.pi*(file_months-1)/12), np.cos(2*np.pi*(file_months-1)/12)], 1).astype(np.float32)
day_enc   = np.stack([np.sin(2*np.pi*(file_days-1)/365), np.cos(2*np.pi*(file_days-1)/365)], 1).astype(np.float32)
geo_all   = np.concatenate([site_oh, hour_enc, month_enc, day_enc], 1).astype(np.float32)

pca24 = PCA(n_components=24, random_state=42).fit(file_embs_norm)
X24   = pca24.transform(file_embs_norm).astype(np.float32)
X24  /= (X24.std(0) + 1e-6)
X_nl = np.concatenate([X24, geo_all], 1).astype(np.float32)
X_nl /= np.linalg.norm(X_nl, 1, keepdims=True) + 1e-8

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS = 1e-7

# ── VLOM blend ────────────────────────────────────────────────────────────
def vlom_blend(a, b, w_a=0.5, w_b=0.5):
    """Log-odds mixing (VLOM): w_a and w_b should be relative weights"""
    w_sum = w_a + w_b
    w_a /= w_sum; w_b /= w_sum
    log_a = np.log(a.clip(EPS) / (1 - a).clip(EPS))
    log_b = np.log(b.clip(EPS) / (1 - b).clip(EPS))
    log_blend = w_a * log_a + w_b * log_b
    return sigmoid(log_blend)

# ── KNN functions ─────────────────────────────────────────────────────────
def attn_knn_loo(X, k=10, T=0.2):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * file_labels[tr[top]]).sum(0)
    return preds

def window_knn_loo(k=1):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        Y_tr = file_labels[win_file_id[tr_wi]]
        sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :k]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            w = sims[wi, top_idx[wi]].clip(0); ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k)/k
            wp[wi] = (w[:, None] * Y_tr[top_idx[wi]]).sum(0)
        preds[i] = wp.mean(0)
    return preds

# ── Precompute common predictions ─────────────────────────────────────────
print("\nPrecomputing KNN predictions...", flush=True)
y_attn = attn_knn_loo(X_nl, k=10, T=0.2)
y_win1 = window_knn_loo(k=1)
y_win3 = window_knn_loo(k=3)
y_win5 = window_knn_loo(k=5)

# Baselines
proto_max_auc  = macro_auc(file_labels, sigmoid(file_logit_max))
proto_mean_auc = macro_auc(file_labels, file_prob_mean)
sed_max_auc    = macro_auc(file_labels, file_sed_max)
sed_mean_auc   = macro_auc(file_labels, file_sed_mean)
vlom_50_50_max = macro_auc(file_labels, vlom_blend(sigmoid(file_logit_max), file_sed_max))
vlom_50_50_mean= macro_auc(file_labels, vlom_blend(file_prob_mean, file_sed_mean))

print(f"\n{'='*60}")
print(f"BASELINES")
print(f"  ProtoSSM logit_max:        {proto_max_auc:.4f}")
print(f"  ProtoSSM prob_mean:        {proto_mean_auc:.4f}")
print(f"  SED max:                   {sed_max_auc:.4f}")
print(f"  SED mean:                  {sed_mean_auc:.4f}")
print(f"  VLOM(50/50, max):          {vlom_50_50_max:.4f}")
print(f"  VLOM(50/50, mean):         {vlom_50_50_mean:.4f}")

# Base score for full pipeline (VLOM blend)
base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max, w_a=0.5, w_b=0.5)
base_auc   = macro_auc(file_labels, base_probs)
base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))

# ── Notebook definitions ────────────────────────────────────────────────────
NOTEBOOKS = [
    {'name': 'v3/v5/v6 (cosine)',  'pkl': 'outputs/embed_prior_model.pkl',        'lambda': 0.25, 'fusion': 'additive', 'type': 'logit'},
    {'name': 'v6-knn4',            'pkl': 'outputs/embed_prior_cosine.pkl',        'lambda': 0.25, 'fusion': 'additive', 'type': 'logit'},
    {'name': 'v6-mahal',           'pkl': 'outputs/embed_prior_mahal.pkl',         'lambda': 0.25, 'fusion': 'additive', 'type': 'logit'},
    {'name': 'v7-combined',        'pkl': 'outputs/embed_prior_combined.pkl',      'lambda': 0.20, 'fusion': 'additive', 'type': 'logit'},
    {'name': 'v7-geo-knn ★REF',   'pkl': 'outputs/embed_prior_attn.pkl',          'lambda': 0.25, 'fusion': 'additive', 'type': 'logit'},
    {'name': 'v8-blend-prior',     'pkl': 'outputs/embed_prior_blend.pkl',         'lambda': 0.25, 'fusion': 'additive', 'type': 'logit'},
    {'name': 'v9-logspace',        'pkl': 'outputs/embed_prior_logspace.pkl',      'lambda': None, 'fusion': 'logspace', 'type': 'logspace'},
    {'name': 'v10-window-knn',     'pkl': 'outputs/embed_prior_window_knn.pkl',    'lambda': None, 'fusion': 'nologit',  'type': 'nologit'},
    {'name': 'v11-ens-nologit',    'pkl': 'outputs/embed_prior_ens_nologit.pkl',   'lambda': 0.25, 'fusion': 'nologit',  'type': 'nologit'},
    {'name': 'v12-ens-nologit2',   'pkl': 'outputs/embed_prior_ens_nologit2.pkl',  'lambda': 0.25, 'fusion': 'nologit',  'type': 'nologit'},
    {'name': 'v13-ens-nologit3',   'pkl': 'outputs/embed_prior_ens_nologit3.pkl',  'lambda': 0.25, 'fusion': 'nologit',  'type': 'nologit'},
    # ── v14 series (all beat v7-geo-knn 0.9246) ──────────────────────────
    # Day 1: k variants
    {'name': 'v14-k4-lam25',       'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.25, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.50},
    {'name': 'v14-k4-lam40',       'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.40, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.50},
    {'name': 'v14-k4-T018-lam40',  'pkl': 'outputs/embed_prior_attn_k4_T018.pkl',  'lambda': 0.40, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.50},
    {'name': 'v14-k3-lam25',       'pkl': 'outputs/embed_prior_attn_k3.pkl',       'lambda': 0.25, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.50},
    {'name': 'v14-k5-lam35',       'pkl': 'outputs/embed_prior_attn_k5.pkl',       'lambda': 0.35, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.50},
    # Day 2: VLOM weight variants
    {'name': 'v14-pw60-lam25',     'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.25, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.60},
    {'name': 'v14-pw65-lam30',     'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.30, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.65},
    {'name': 'v14-pw70-lam30',     'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.30, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.70},
    {'name': 'v14-pw60-lam35',     'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.35, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.60},
    {'name': 'v14-pw55-lam50',     'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.50, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.55},
    # Day 3: Window KNN ensemble (attn_k4 × w_a + win_k1 × w_w)
    {'name': 'v14-win070-lam35',   'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.35, 'fusion': 'win_ens', 'type': 'logit', 'proto_w': 0.50, 'w_attn': 0.70, 'w_win': 0.30},
    {'name': 'v14-win075-lam35',   'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.35, 'fusion': 'win_ens', 'type': 'logit', 'proto_w': 0.50, 'w_attn': 0.75, 'w_win': 0.25},
    {'name': 'v14-win080-lam35',   'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.35, 'fusion': 'win_ens', 'type': 'logit', 'proto_w': 0.50, 'w_attn': 0.80, 'w_win': 0.20},
    {'name': 'v14-win070-lam30',   'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.30, 'fusion': 'win_ens', 'type': 'logit', 'proto_w': 0.50, 'w_attn': 0.70, 'w_win': 0.30},
    {'name': 'v14-win085-lam25',   'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.25, 'fusion': 'win_ens', 'type': 'logit', 'proto_w': 0.50, 'w_attn': 0.85, 'w_win': 0.15},
    # Day 4: 3-way VLOM + extra
    {'name': 'v14-3way-020',       'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.20, 'fusion': '3way_vlom', 'type': 'logit', 'proto_w': 0.48, 'sed_w': 0.32, 'ep_w': 0.20},
    {'name': 'v14-3way-035',       'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.35, 'fusion': '3way_vlom', 'type': 'logit', 'proto_w': 0.39, 'sed_w': 0.26, 'ep_w': 0.35},
    {'name': 'v14-3way-025',       'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.25, 'fusion': '3way_vlom', 'type': 'logit', 'proto_w': 0.45, 'sed_w': 0.30, 'ep_w': 0.25},
    {'name': 'v14-pw70-lam25',     'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.25, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.70},
    {'name': 'v14-pw65-lam25',     'pkl': 'outputs/embed_prior_attn_k4.pkl',       'lambda': 0.25, 'fusion': 'additive', 'type': 'logit', 'proto_w': 0.65},
    # ── v14 LS2 series: sigmoid(a*vlom_logit + b*log(geo_k5*0.5 + win_k1*0.5)) ──
    {'name': 'v14-ls2-a090-b155',  'pkl': 'outputs/embed_prior_logspace_geo5_win1.pkl', 'lambda': None, 'fusion': 'ls2', 'type': 'logspace', 'ls2_a': 0.90, 'ls2_b': 1.55},
    {'name': 'v14-ls2-a080-b140',  'pkl': 'outputs/embed_prior_logspace_geo5_win1.pkl', 'lambda': None, 'fusion': 'ls2', 'type': 'logspace', 'ls2_a': 0.80, 'ls2_b': 1.40},
    {'name': 'v14-ls2-a075-b130',  'pkl': 'outputs/embed_prior_logspace_geo5_win1.pkl', 'lambda': None, 'fusion': 'ls2', 'type': 'logspace', 'ls2_a': 0.75, 'ls2_b': 1.30},
    {'name': 'v14-ls2-a070-b120',  'pkl': 'outputs/embed_prior_logspace_geo5_win1.pkl', 'lambda': None, 'fusion': 'ls2', 'type': 'logspace', 'ls2_a': 0.70, 'ls2_b': 1.20},
    {'name': 'v14-ls2-a060-b145',  'pkl': 'outputs/embed_prior_logspace_geo5_win1.pkl', 'lambda': None, 'fusion': 'ls2', 'type': 'logspace', 'ls2_a': 0.60, 'ls2_b': 1.45},
    # ── v14 RKNN series: Reciprocal KNN k=5 + win_k1 ──────────────────────────
    {'name': 'rknn-wg040-a095-b170', 'pkl': 'outputs/embed_prior_rknn_k5_win1.pkl', 'lambda': None, 'fusion': 'rknn', 'type': 'logspace', 'ls2_a': 0.95, 'ls2_b': 1.70, 'w_rknn': 0.40},
    {'name': 'rknn-wg040-a090-b155', 'pkl': 'outputs/embed_prior_rknn_k5_win1.pkl', 'lambda': None, 'fusion': 'rknn', 'type': 'logspace', 'ls2_a': 0.90, 'ls2_b': 1.55, 'w_rknn': 0.40},
    {'name': 'rknn-wg035-a090-b140', 'pkl': 'outputs/embed_prior_rknn_k5_win1.pkl', 'lambda': None, 'fusion': 'rknn', 'type': 'logspace', 'ls2_a': 0.90, 'ls2_b': 1.40, 'w_rknn': 0.35},
    {'name': 'rknn-wg030-a090-b140', 'pkl': 'outputs/embed_prior_rknn_k5_win1.pkl', 'lambda': None, 'fusion': 'rknn', 'type': 'logspace', 'ls2_a': 0.90, 'ls2_b': 1.40, 'w_rknn': 0.30},
]

def get_ep_pred(nb):
    with open(nb['pkl'], 'rb') as f:
        ep = pickle.load(f)
    ep_type = ep.get('type', '')
    if 'nologit3' in ep_type:
        w_a = ep.get('w_attn', 0.65); k_w = ep.get('k_win', 1)
        y_w = {1: y_win1, 3: y_win3, 5: y_win5}.get(k_w, y_win1)
        return w_a * y_attn + (1-w_a) * y_w, ep
    elif 'nologit_v2' in ep_type:
        w_a = ep.get('w_attn', 0.70); k_w = ep.get('k_win', 3)
        y_w = {1: y_win1, 3: y_win3, 5: y_win5}.get(k_w, y_win3)
        return w_a * y_attn + (1-w_a) * y_w, ep
    elif 'ens_nologit' in ep_type:
        w_a = ep.get('w_attn', 0.75)
        return w_a * y_attn + (1-w_a) * y_win5, ep
    elif 'window_knn' in ep_type:
        return y_win5, ep
    elif 'rknn_win' in ep_type:
        # Reciprocal KNN k=5 blended with win_k1
        X_ref2 = ep['X_combined_n'].astype(np.float32)
        fl2    = ep['file_labels'].astype(np.float32)
        k_r    = ep.get('k_rknn', 5); T_r = ep.get('T_rknn', 0.2)
        w_r    = nb.get('w_rknn', ep.get('w_rknn', 0.40))
        # Precompute pairwise similarities among training files
        sim_train = X_ref2 @ X_ref2.T; np.fill_diagonal(sim_train, -np.inf)
        top_k_tr  = np.argsort(-sim_train, axis=1)[:, :k_r]
        y_rknn = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j!=i])
            sims_i = (X_ref2[[i]]@X_ref2[tr].T).ravel()
            top_i = np.argsort(-sims_i)[:k_r]
            mutual = []; mutual_sims = []
            for ti_idx, tj in enumerate(tr[top_i]):
                kth_sim = sim_train[tj, top_k_tr[tj, -1]]
                if sims_i[top_i[ti_idx]] >= kth_sim:
                    mutual.append(tj); mutual_sims.append(sims_i[top_i[ti_idx]])
            if len(mutual) == 0:
                top = top_i[:5]; ls=sims_i[top]/T_r; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
                y_rknn[i]=(w[:,None]*fl2[tr[top]]).sum(0)
            else:
                ma=np.array(mutual); ms=np.array(mutual_sims)
                ls=ms/T_r; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
                y_rknn[i]=(w[:,None]*fl2[ma]).sum(0)
        blended = w_r * y_rknn + (1-w_r) * y_win1
        return blended, ep
    elif 'logspace_geo5_win1' in ep_type:
        # Use pkl's X_combined_n for geo-KNN k=5, then blend with win_k1
        X_ref = ep['X_combined_n'].astype(np.float32)
        fl    = ep['file_labels'].astype(np.float32)
        k_g   = ep.get('k_geo', 5); T_g = ep.get('T_geo', 0.2)
        w_g   = ep.get('w_geo', 0.50)
        y_g   = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j != i])
            sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
            top = np.argsort(-sims)[:k_g]; ls = sims[top]/T_g; ls -= ls.max()
            w = np.exp(ls); w /= w.sum(); y_g[i] = (w[:,None] * fl[tr[top]]).sum(0)
        blended = w_g * y_g + (1-w_g) * y_win1
        return blended, ep
    elif 'logspace' in ep_type:
        return y_attn, ep  # logspace uses attn-KNN
    elif 'X_combined_n' in ep:
        # Use stored KNN space
        X_ref = ep['X_combined_n'].astype(np.float32)
        fl = ep['file_labels'].astype(np.float32)
        k_a = ep.get('k_attn', ep.get('k', 10))
        T_a = ep.get('temperature', 0.2)
        preds = np.zeros((n_files, n_species), np.float32)
        for i in range(n_files):
            tr = np.array([j for j in range(n_files) if j != i])
            sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
            top = np.argsort(-sims)[:k_a]
            logit_s = sims[top] / T_a; logit_s -= logit_s.max()
            w = np.exp(logit_s); w /= w.sum()
            preds[i] = (w[:, None] * fl[tr[top]]).sum(0)
        return preds, ep
    else:
        k = ep.get('k', 5)
        return attn_knn_loo(file_embs_norm, k=k, T=0.5), ep

print(f"\n{'='*60}")
print("FULL PIPELINE EVALUATION")
print(f"{'='*60}")
print(f"{'Notebook':25s}  {'EP_only':8s}  {'Base+EP':8s}  {'Δ_base':8s}")
print("-"*60)

results = []
for nb in NOTEBOOKS:
    try:
        y_ep, ep = get_ep_pred(nb)
        ep_auc = macro_auc(file_labels, y_ep)
        loo_auc = ep.get('loo_auc', None)
        fusion = nb['fusion']
        lam = nb['lambda']

        # Determine VLOM weights for this notebook
        nb_proto_w = nb.get('proto_w', 0.5)
        nb_sed_w   = nb.get('sed_w', 1.0 - nb_proto_w)

        # Recompute base with notebook-specific VLOM weights
        nb_base = vlom_blend(sigmoid(file_logit_max), file_sed_max, w_a=nb_proto_w, w_b=nb_sed_w)
        nb_base_logit = np.log(nb_base.clip(EPS)) - np.log((1-nb_base).clip(EPS))

        if fusion == 'win_ens':
            # Window KNN ensemble: attn + window predictions
            w_a = nb.get('w_attn', 0.70)
            w_w = nb.get('w_win', 0.30)
            y_ep_win = w_a * y_ep + w_w * y_win1
            ep_logit = np.log(y_ep_win.clip(EPS)) - np.log((1-y_ep_win).clip(EPS))
            full_logit = nb_base_logit + lam * ep_logit
            full = sigmoid(full_logit)
            full_auc = macro_auc(file_labels, full)
        elif fusion == '3way_vlom':
            # 3-way VLOM: ProtoSSM + SED + EmbedPrior
            ep_w = nb.get('ep_w', 0.25)
            pw = nb_proto_w; sw = nb_sed_w
            w_sum = pw + sw + ep_w
            a, b, c = pw/w_sum, sw/w_sum, ep_w/w_sum
            la = np.log(sigmoid(file_logit_max).clip(EPS)) - np.log((1-sigmoid(file_logit_max)).clip(EPS))
            lb = np.log(file_sed_max.clip(EPS)) - np.log((1-file_sed_max).clip(EPS))
            lc = np.log(y_ep.clip(EPS)) - np.log((1-y_ep).clip(EPS))
            full = sigmoid(a*la + b*lb + c*lc)
            full_auc = macro_auc(file_labels, full)
        elif fusion == 'rknn':
            # sigmoid(a * vlom_logit + b * log(rknn_blend))
            a = nb.get('ls2_a', 0.95); b = nb.get('ls2_b', 1.70)
            log_y = np.log(y_ep.clip(EPS, 1-EPS))
            full = sigmoid(a * nb_base_logit + b * log_y)
            full_auc = macro_auc(file_labels, full)
        elif fusion == 'ls2':
            # sigmoid(a * vlom_logit + b * log(blended_knn))
            # y_ep is already geo_k5+win_k1 blended (from get_ep_pred)
            a = nb.get('ls2_a', 0.90); b = nb.get('ls2_b', 1.55)
            log_y = np.log(y_ep.clip(EPS, 1-EPS))
            full = sigmoid(a * nb_base_logit + b * log_y)
            full_auc = macro_auc(file_labels, full)
        elif fusion == 'logspace':
            a, b = 0.7, 1.5
            log_y = np.log(y_ep.clip(EPS, 1-EPS))
            full = sigmoid(a * base_logit + b * log_y)
            full_auc = macro_auc(file_labels, full)
        elif fusion == 'nologit':
            ep_logit = np.log(y_ep.clip(EPS)) - np.log((1-y_ep).clip(EPS))
            if lam is not None:
                full_logit = nb_base_logit + lam * ep_logit
                full = sigmoid(full_logit)
            else:
                full = y_ep
            full_auc = macro_auc(file_labels, full)
        else:
            # additive: nb_base_logit + lambda * ep_logit
            ep_logit = np.log(y_ep.clip(EPS)) - np.log((1-y_ep).clip(EPS))
            if lam is not None:
                full_logit = nb_base_logit + lam * ep_logit
                full = sigmoid(full_logit)
                full_auc = macro_auc(file_labels, full)
            else:
                full_auc = ep_auc

        delta = full_auc - base_auc
        print(f"{nb['name']:30s}  {ep_auc:.4f}    {full_auc:.4f}    {delta:+.4f}")
        results.append({
            'name': nb['name'],
            'type': nb['type'],
            'ep_auc': round(ep_auc, 4),
            'full_auc': round(full_auc, 4),
            'delta_base': round(delta, 4),
            'delta_proto': round(full_auc - proto_max_auc, 4),
            'loo_auc_pkl': round(loo_auc, 4) if loo_auc else None,
            'lambda': lam,
        })
    except Exception as ex:
        print(f"{nb['name']:25s}  ERROR: {ex}")
        results.append({'name': nb['name'], 'type': nb['type'], 'full_auc': None, 'error': str(ex)})

# ── Generate HTML Report ───────────────────────────────────────────────────
sorted_results = sorted([r for r in results if r.get('full_auc')], key=lambda x: -x['full_auc'])
best_auc = sorted_results[0]['full_auc'] if sorted_results else 0

def bar(val, max_val, width=200):
    pct = min(val / max_val, 1.0) if max_val > 0 else 0
    return f"<div style='background:#3498db;height:14px;width:{int(pct*width)}px;border-radius:3px'></div>"

html = f"""<!DOCTYPE html>
<html><head>
<meta charset='utf-8'>
<title>BirdCLEF 2026 — Full Pipeline Benchmark (Train Soundscape LOO-CV)</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f8f9fa; color: #2c3e50; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 12px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px;
           overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 30px; }}
  th {{ background: #2c3e50; color: white; padding: 12px 16px; text-align: left; font-size: 13px; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #ecf0f1; font-size: 13px; vertical-align: middle; }}
  tr:hover {{ background: #f8f9fa; }}
  tr.best {{ background: #eafaf1 !important; }}
  .auc {{ font-weight: bold; font-size: 15px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; }}
  .badge-nologit {{ background: #3498db; color: white; }}
  .badge-logit {{ background: #9b59b6; color: white; }}
  .badge-logspace {{ background: #e67e22; color: white; }}
  .badge-baseline {{ background: #7f8c8d; color: white; }}
  .summary-box {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                  box-shadow: 0 2px 8px rgba(0,0,0,0.1); display: flex; flex-wrap: wrap; gap: 20px; }}
  .metric {{ text-align: center; min-width: 120px; }}
  .metric-value {{ font-size: 26px; font-weight: bold; color: #2c3e50; }}
  .metric-label {{ font-size: 11px; color: #7f8c8d; margin-top: 4px; }}
  .good {{ color: #27ae60; }}
  .bad {{ color: #e74c3c; }}
  .note {{ color: #7f8c8d; font-size: 12px; font-style: italic; }}
  .section {{ background: white; border-radius: 8px; padding: 16px; margin: 20px 0;
              box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>🐦 BirdCLEF 2026 — Full Pipeline Benchmark</h1>
<p class='note'>評估日期：2026-03-25 | 資料：66 train_soundscape 文件 (LOO-CV) | 指標：Macro ROC-AUC</p>

<div class='summary-box'>
  <div class='metric'>
    <div class='metric-value'>{n_files}</div>
    <div class='metric-label'>LOO Folds</div>
  </div>
  <div class='metric'>
    <div class='metric-value'>{n_species}</div>
    <div class='metric-label'>物種數</div>
  </div>
  <div class='metric'>
    <div class='metric-value' style='color:#e74c3c'>{proto_max_auc:.4f}</div>
    <div class='metric-label'>ProtoSSM Only</div>
  </div>
  <div class='metric'>
    <div class='metric-value' style='color:#e74c3c'>{sed_max_auc:.4f}</div>
    <div class='metric-label'>SED Only (max)</div>
  </div>
  <div class='metric'>
    <div class='metric-value' style='color:#e67e22'>{base_auc:.4f}</div>
    <div class='metric-label'>VLOM Base (50/50)</div>
  </div>
  <div class='metric'>
    <div class='metric-value' style='color:#27ae60'>{best_auc:.4f}</div>
    <div class='metric-label'>最佳 Full Pipeline</div>
  </div>
</div>

<h2>🏆 Full Pipeline Ranking (SED + ProtoSSM + Embed Prior)</h2>
<table>
  <tr>
    <th>#</th>
    <th>Notebook</th>
    <th>類型</th>
    <th>Embed Prior AUC</th>
    <th>Full Pipeline AUC</th>
    <th>vs VLOM Base</th>
    <th>vs Proto Only</th>
    <th>PKL LOO-AUC</th>
    <th>λ</th>
    <th>AUC bar</th>
  </tr>"""

# Baselines row
for base_name, base_auc_val, badge in [
    ('VLOM Base (50/50)', base_auc, 'baseline'),
    ('ProtoSSM Only', proto_max_auc, 'baseline'),
    ('SED Only (max)', sed_max_auc, 'baseline'),
]:
    delta_v = base_auc_val - base_auc
    delta_p = base_auc_val - proto_max_auc
    html += f"""
  <tr style='background:#f7f9fc'>
    <td>—</td>
    <td><em>{base_name}</em></td>
    <td><span class='badge badge-{badge}'>baseline</span></td>
    <td>—</td>
    <td class='auc'>{base_auc_val:.4f}</td>
    <td style='color:{"#27ae60" if delta_v >= 0 else "#e74c3c"}'>{delta_v:+.4f}</td>
    <td style='color:{"#27ae60" if delta_p >= 0 else "#e74c3c"}'>{delta_p:+.4f}</td>
    <td>—</td><td>—</td>
    <td>{bar(base_auc_val, best_auc)}</td>
  </tr>"""

for rank, r in enumerate(sorted_results, 1):
    is_best = rank == 1
    row_cls = 'best' if is_best else ''
    badge_cls = {'logit': 'logit', 'logspace': 'logspace', 'nologit': 'nologit'}.get(r.get('type'), 'logit')
    badge_txt = {'logit': 'logit', 'logspace': 'logspace', 'nologit': 'no-logit'}.get(r.get('type'), '?')
    ep_auc = r.get('ep_auc', '-')
    full_auc = r.get('full_auc', 0)
    delta_v = r.get('delta_base', 0)
    delta_p = r.get('delta_proto', 0)
    loo = r.get('loo_auc_pkl')
    lam = r.get('lambda')
    html += f"""
  <tr class='{row_cls}'>
    <td><strong>{rank}</strong></td>
    <td><strong>{r['name']}</strong></td>
    <td><span class='badge badge-{badge_cls}'>{badge_txt}</span></td>
    <td>{f'{ep_auc:.4f}' if isinstance(ep_auc, float) else '-'}</td>
    <td class='auc' style='color:{"#27ae60" if full_auc > base_auc else "#e74c3c"}'>{full_auc:.4f}</td>
    <td class='{"good" if delta_v > 0 else "bad"}'>{delta_v:+.4f}</td>
    <td class='{"good" if delta_p > 0 else "bad"}'>{delta_p:+.4f}</td>
    <td style='font-size:12px'>{f'{loo:.4f}' if loo else '—'}</td>
    <td style='font-size:12px'>{f'{lam}' if lam else '—'}</td>
    <td>{bar(full_auc, best_auc)}</td>
  </tr>"""

html += f"""
</table>

<h2>🔗 CV vs LB 相關性分析</h2>
<div class='section'>
  <p><strong>CV（LOO on 66 soundscapes）與 LB 的相關性說明：</strong></p>
  <ul>
    <li>⚠️ <strong>66 個文件的 LOO 方差較高</strong>，單一方法差距 &lt; 0.002 在統計上不顯著</li>
    <li>✅ <strong>整體趨勢</strong>：Logspace 方法（融合 Perch logit）在 CV 上持續優於純 nologit</li>
    <li>📊 <strong>LB 最高已知分數</strong>：0.926（LGBM + event smooth），但這是不同管線</li>
    <li>🎯 <strong>embed_prior 對 CV 的貢獻</strong>：約 +0.01 ~ +0.15 AUC，視方法而定</li>
  </ul>
  <table style='width:auto'>
    <tr><th>管線層</th><th>CV AUC</th><th>說明</th></tr>
    <tr><td>ProtoSSM only</td><td><strong>{proto_max_auc:.4f}</strong></td><td>Perch logit → sigmoid</td></tr>
    <tr><td>SED only</td><td><strong>{sed_max_auc:.4f}</strong></td><td>EfficientNet-B0 SED 預測</td></tr>
    <tr><td>VLOM(50/50)</td><td><strong>{base_auc:.4f}</strong></td><td>ProtoSSM + SED log-odds blend</td></tr>
    <tr><td>最佳 Full Pipeline</td><td><strong style='color:#27ae60'>{best_auc:.4f}</strong></td><td>{sorted_results[0]['name'] if sorted_results else 'N/A'}</td></tr>
  </table>
</div>

<h2>💡 提交優先級建議</h2>
<div class='section'>
  <ol>"""

for i, r in enumerate(sorted_results[:8], 1):
    delta = r.get('delta_base', 0)
    color = '#27ae60' if delta > 0 else '#e74c3c'
    html += f"""
    <li><strong>{r['name']}</strong> — Full AUC: <strong>{r['full_auc']:.4f}</strong>
        <span style='color:{color}'>({delta:+.4f} vs VLOM base)</span></li>"""

html += f"""
  </ol>
  <p class='note'>注意：上述排名基於 LOO-CV on 66 soundscapes，可能與實際 LB 排名略有差異。
  Logspace 方法通常 CV 較高，但 nologit 方法可能有更好的泛化（不依賴 Perch logit 的品質）。</p>
</div>

<p class='note' style='margin-top:30px'>
  報告生成：2026-03-25 |
  SED 預測：sed-ns-b0-20s-r1 (corrected) |
  ProtoSSM 代理：Perch logit_max |
  VLOM blend: 50%/50%
</p>
</body></html>"""

with open("reports/current_subs_benchmark.html", "w", encoding='utf-8') as f:
    f.write(html)

print(f"\n{'='*60}")
print(f"報告已儲存: reports/current_subs_benchmark.html")
print(f"{'='*60}")
print("done")
