#!/usr/bin/env python3
"""
cv_params_sweep.py
==================
Local OOF parameter sweep for the lgbm-proto-family-feat-nssed pipeline.
Uses pre-computed Perch cache + NS SED r1 all_ss_probs to sweep:
  - PERCH_W / SED_W blend weights
  - LGBM hyperparameters
  - FROZEN_FUSION lambda_event / lambda_texture
  - SMOOTH_EVENT_ALPHA / smooth_texture
  - HEAD_BLEND_ALPHA (if head tflite available)
  - proto_sim / family_mean features on/off

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/cv_params_sweep.py

Output:
    cv_sweep_results.csv  (sorted by OOF AUC desc)
"""

import re, warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / 'birdclef-2026'
MODEL_DIR  = ROOT / 'models/bird-vocalization-classifier-tensorflow2-perch_v2_cpu-v1'
CACHE_NPZ  = ROOT / 'outputs/perch_cache_extended.npz'
SED_NPZ    = ROOT / 'outputs/sed-ns-b0-r1/all_ss_probs.npz'

N_WINDOWS   = 12
NUM_CLASSES = 234

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading Perch cache...")
cache = np.load(CACHE_NPZ, allow_pickle=True)
scores_full_raw = cache['scores_full_raw'].astype(np.float32)  # (N, 234) logits
emb_full        = cache['emb_full'].astype(np.float32)         # (N, 1536)
filenames_cache = cache['filenames']                            # (N,)
row_ids_cache   = cache['row_ids']                             # (N,)

print(f"  Perch cache: {scores_full_raw.shape[0]} rows ({scores_full_raw.shape[0]//N_WINDOWS} files)")

print("Loading SED probs...")
sed_data    = np.load(SED_NPZ, allow_pickle=True)
sed_row_ids = sed_data['row_ids']
sed_probs   = sed_data['probs'].astype(np.float32)

# Filter SED to only labeled soundscapes (matching row_ids)
cache_row_id_set = set(row_ids_cache.tolist())
sed_mask = np.array([r in cache_row_id_set for r in sed_row_ids])
sed_row_ids_f = sed_row_ids[sed_mask]
sed_probs_f   = sed_probs[sed_mask]

# Align SED to Perch cache order
sed_rid_to_idx = {r: i for i, r in enumerate(sed_row_ids_f)}
sed_aligned = np.zeros_like(scores_full_raw)
for i, rid in enumerate(row_ids_cache):
    j = sed_rid_to_idx.get(rid)
    if j is not None:
        sed_aligned[i] = sed_probs_f[j]

print(f"  SED aligned: {(sed_aligned.sum(axis=1) > 0).sum()} / {len(row_ids_cache)} rows matched")

# ── Taxonomy + mapping setup ───────────────────────────────────────────────────
print("Loading taxonomy...")
taxonomy          = pd.read_csv(DATA_DIR / 'taxonomy.csv')
soundscape_labels = pd.read_csv(DATA_DIR / 'train_soundscapes_labels.csv')
sample_sub        = pd.read_csv(DATA_DIR / 'sample_submission.csv')
PRIMARY_LABELS    = sample_sub.columns[1:].tolist()
label_to_idx      = {c: i for i, c in enumerate(PRIMARY_LABELS)}

taxonomy['primary_label']          = taxonomy['primary_label'].astype(str)
soundscape_labels['primary_label'] = soundscape_labels['primary_label'].astype(str)

bc_labels_df = (
    pd.read_csv(MODEL_DIR / 'assets' / 'labels.csv')
    .reset_index()
    .rename(columns={'index': 'bc_index', 'inat2024_fsd50k': 'scientific_name'})
)

taxonomy_copy = taxonomy.copy()
taxonomy_copy['scientific_name_lookup'] = taxonomy_copy['scientific_name']
bc_lookup = bc_labels_df.rename(columns={'scientific_name': 'scientific_name_lookup'})
mapping = taxonomy_copy.merge(bc_lookup[['scientific_name_lookup','bc_index']],
                               on='scientific_name_lookup', how='left')
NO_LABEL_INDEX  = len(bc_labels_df)
mapping['bc_index'] = mapping['bc_index'].fillna(NO_LABEL_INDEX).astype(int)
label_to_bc_index   = mapping.set_index('primary_label')['bc_index']
BC_INDICES          = np.array([int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK         = BC_INDICES != NO_LABEL_INDEX

CLASS_NAME_MAP = taxonomy.set_index('primary_label')['class_name'].to_dict()
TEXTURE_TAXA   = {'Amphibia', 'Insecta'}

# ── Parse soundscape labels ───────────────────────────────────────────────────
FNAME_RE = re.compile(r'BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg')

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {'file_id': None, 'site': None, 'hour_utc': -1}
    file_id, site, ymd, hms = m.groups()
    return {'file_id': file_id, 'site': site, 'hour_utc': int(hms[:2])}

def parse_soundscape_labels(x):
    if pd.isna(x): return []
    return [t.strip() for t in str(x).split(';') if t.strip()]

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))

sc_clean = (
    soundscape_labels
    .groupby(['filename','start','end'])['primary_label']
    .apply(union_labels)
    .reset_index(name='label_list')
)
sc_clean['start_sec'] = pd.to_timedelta(sc_clean['start']).dt.total_seconds().astype(int)
sc_clean['end_sec']   = pd.to_timedelta(sc_clean['end']).dt.total_seconds().astype(int)
sc_clean['row_id']    = (sc_clean['filename'].str.replace('.ogg','',regex=False)
                         + '_' + sc_clean['end_sec'].astype(str))

meta_sc = sc_clean['filename'].apply(lambda fn: parse_soundscape_filename(fn)).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta_sc], axis=1)

windows_per_file  = sc_clean.groupby('filename').size()
full_files        = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc_clean['file_fully_labeled'] = sc_clean['filename'].isin(full_files)

Y_SC = np.zeros((len(sc_clean), NUM_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean['label_list']):
    idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
    if idxs: Y_SC[i, idxs] = 1

# Build full_truth aligned to cache
full_truth = (
    sc_clean[sc_clean['file_fully_labeled']]
    .sort_values(['filename','end_sec'])
    .reset_index(drop=False)
)

# Align cache to full_truth row order
cache_rid_to_i = {r: i for i, r in enumerate(row_ids_cache)}
full_truth_aligned_idx = [cache_rid_to_i.get(rid, -1) for rid in full_truth['row_id']]
valid_mask = np.array(full_truth_aligned_idx) >= 0
print(f"  full_truth rows: {len(full_truth)}, matched to cache: {valid_mask.sum()}")

full_truth = full_truth[valid_mask].reset_index(drop=True)
align_idx  = np.array([cache_rid_to_i[rid] for rid in full_truth['row_id']])

scores_full_raw_aligned = scores_full_raw[align_idx]
emb_full_aligned        = emb_full[align_idx]
sed_aligned_full        = sed_aligned[align_idx]
Y_FULL                  = Y_SC[full_truth['index'].to_numpy()]

# ACTIVE_CLASSES
ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]

idx_active_texture = np.array(
    [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA],
    dtype=np.int32)
idx_active_event = np.array(
    [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA],
    dtype=np.int32)
idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
idx_mapped_active_event   = idx_active_event[MAPPED_MASK[idx_active_event]]
idx_unmapped_inactive     = np.array(
    [i for i in np.where(~MAPPED_MASK)[0] if PRIMARY_LABELS[i] not in ACTIVE_CLASSES],
    dtype=np.int32)

print(f"  ACTIVE={len(ACTIVE_CLASSES)}  texture={len(idx_active_texture)}  event={len(idx_active_event)}")

# Family mapping
FAMILY_MAP    = taxonomy.set_index('primary_label')['class_name'].to_dict()
FAMILY_GROUPS = {}
for ci, label in enumerate(PRIMARY_LABELS):
    family = FAMILY_MAP.get(label, 'Unknown')
    FAMILY_GROUPS.setdefault(family, []).append(ci)
FAMILY_IDX_MAP = {fam: np.array(idxs, dtype=np.int32) for fam, idxs in FAMILY_GROUPS.items()}
CLASS_FAMILY   = {ci: FAMILY_MAP.get(label, 'Unknown') for ci, label in enumerate(PRIMARY_LABELS)}

# ── Helper functions ───────────────────────────────────────────────────────────

def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0: return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro')

def smooth_cols_fixed12(scores, cols, alpha=0.35):
    if alpha <= 0 or len(cols) == 0: return scores.copy()
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:,:1,:], x[:,:-1,:]], axis=1)
    next_x = np.concatenate([x[:,1:,:], x[:,-1:,:]], axis=1)
    view[:, :, cols] = (1.0 - alpha)*x + 0.5*alpha*(prev_x + next_x)
    return s

def smooth_events_fixed12(scores, cols, alpha=0.15):
    if alpha <= 0 or len(cols) == 0: return scores.copy()
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev = np.concatenate([x[:,:1,:], x[:,:-1,:]], axis=1)
    nxt  = np.concatenate([x[:,1:,:], x[:,-1:,:]], axis=1)
    local_max = np.maximum(x, np.maximum(prev, nxt))
    view[:, :, cols] = (1.0 - alpha)*x + alpha*local_max
    return s

def cosine_sim_to_prototype(Z, prototype):
    Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    p_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
    return (Z_norm @ p_norm).astype(np.float32)

def seq_features_1d(v):
    x = v.reshape(-1, N_WINDOWS)
    prev_v  = np.concatenate([x[:,:1], x[:,:-1]], axis=1).reshape(-1)
    next_v  = np.concatenate([x[:,1:], x[:,-1:]], axis=1).reshape(-1)
    mean_v  = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v   = np.repeat(x.max(axis=1),  N_WINDOWS)
    min_v   = np.repeat(x.min(axis=1),  N_WINDOWS)
    range_v = max_v - min_v
    return prev_v, next_v, mean_v, max_v, min_v, range_v

def build_class_features(emb_proj, raw_col, prior_col, base_col,
                          proto_sim_col=None, family_mean_col=None):
    prev_base, next_base, mean_base, max_base, min_base, range_base = seq_features_1d(base_col)
    parts = [emb_proj, raw_col[:,None], prior_col[:,None], base_col[:,None],
             prev_base[:,None], next_base[:,None], mean_base[:,None], max_base[:,None],
             min_base[:,None], range_base[:,None],
             (raw_col*prior_col)[:,None], (raw_col*base_col)[:,None], (prior_col*base_col)[:,None]]
    if proto_sim_col is not None: parts.append(proto_sim_col[:,None])
    if family_mean_col is not None: parts.append(family_mean_col[:,None])
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

def fit_prior_tables(prior_df, Y_prior):
    prior_df  = prior_df.reset_index(drop=True)
    global_p  = Y_prior.mean(axis=0).astype(np.float32)
    site_keys = sorted(prior_df['site'].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    site_p = np.zeros((len(site_keys), Y_prior.shape[1]), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]; mask = prior_df['site'].astype(str).values == s
        site_n[i] = mask.sum(); site_p[i] = Y_prior[mask].mean(axis=0)
    hour_keys = sorted(prior_df['hour_utc'].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    hour_p = np.zeros((len(hour_keys), Y_prior.shape[1]), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]; mask = prior_df['hour_utc'].astype(int).values == h
        hour_n[i] = mask.sum(); hour_p[i] = Y_prior[mask].mean(axis=0)
    sh_to_i = {}; sh_n_list = []; sh_p_list = []
    for (s, h), idx_g in prior_df.groupby(['site','hour_utc']).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx_g = np.array(list(idx_g))
        sh_n_list.append(len(idx_g)); sh_p_list.append(Y_prior[idx_g].mean(axis=0))
    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = (np.stack(sh_p_list).astype(np.float32) if sh_p_list
            else np.zeros((0, Y_prior.shape[1]), dtype=np.float32))
    return {'global_p': global_p, 'site_to_i': site_to_i, 'site_n': site_n, 'site_p': site_p,
            'hour_to_i': hour_to_i, 'hour_n': hour_n, 'hour_p': hour_p,
            'sh_to_i': sh_to_i, 'sh_n': sh_n, 'sh_p': sh_p}

def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables['global_p'][None,:], n, axis=0).astype(np.float32, copy=True)
    site_idx = np.fromiter((tables['site_to_i'].get(str(s), -1) for s in sites), dtype=np.int32, count=n)
    hour_idx = np.fromiter((tables['hour_to_i'].get(int(h), -1) if int(h) >= 0 else -1 for h in hours), dtype=np.int32, count=n)
    sh_idx   = np.fromiter((tables['sh_to_i'].get((str(s), int(h)), -1) if int(h) >= 0 else -1
                            for s, h in zip(sites, hours)), dtype=np.int32, count=n)
    valid = hour_idx >= 0
    if valid.any():
        nh = tables['hour_n'][hour_idx[valid]][:,None]; wh = nh/(nh+8.0)
        p[valid] = wh*tables['hour_p'][hour_idx[valid]] + (1.0-wh)*p[valid]
    valid = site_idx >= 0
    if valid.any():
        ns = tables['site_n'][site_idx[valid]][:,None]; ws = ns/(ns+8.0)
        p[valid] = ws*tables['site_p'][site_idx[valid]] + (1.0-ws)*p[valid]
    valid = sh_idx >= 0
    if valid.any():
        nsh = tables['sh_n'][sh_idx[valid]][:,None]; wsh = nsh/(nsh+4.0)
        p[valid] = wsh*tables['sh_p'][sh_idx[valid]] + (1.0-wsh)*p[valid]
    np.clip(p, eps, 1.0-eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)

def fuse_scores(base_scores, sites, hours, tables,
                lambda_event=0.4, lambda_texture=1.0,
                smooth_texture=0.35, smooth_event=0.15):
    scores = base_scores.copy()
    prior  = prior_logits_from_tables(sites, hours, tables)
    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event]   += lambda_event   * prior[:, idx_mapped_active_event]
    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += lambda_texture * prior[:, idx_mapped_active_texture]
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0
    scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
    scores = smooth_events_fixed12(scores, idx_active_event, alpha=smooth_event)
    return scores.astype(np.float32, copy=False), prior

def vlom_blend(a, b, w_a=0.5, w_b=0.5):
    eps = 1e-9; total = w_a+w_b; pa = w_a/total; pb = w_b/total
    geomean = np.exp(pa*np.log(a+eps) + pb*np.log(b+eps))
    rms     = np.sqrt(pa*a**2 + pb*b**2)
    return ((geomean+rms)/2.0).astype(np.float32)

# ── OOF pipeline ──────────────────────────────────────────────────────────────

def run_oof(cfg):
    """Run full OOF pipeline with given config, return macro AUC."""
    pca_dim        = cfg.get('pca_dim', 64)
    min_pos        = cfg.get('min_pos', 8)
    lambda_event   = cfg.get('lambda_event', 0.4)
    lambda_texture = cfg.get('lambda_texture', 1.0)
    smooth_texture = cfg.get('smooth_texture', 0.35)
    smooth_event   = cfg.get('smooth_event', 0.15)
    perch_w        = cfg.get('perch_w', 0.5)
    sed_w          = cfg.get('sed_w', 0.5)
    use_proto      = cfg.get('use_proto', True)
    use_family     = cfg.get('use_family', True)
    lgbm_params    = cfg.get('lgbm_params', {
        'n_estimators': 100, 'max_depth': 3, 'num_leaves': 7,
        'learning_rate': 0.05, 'min_child_samples': 5,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42,
        'verbose': -1, 'n_jobs': 4,
    })

    N = len(scores_full_raw_aligned)
    groups = full_truth['filename'].to_numpy()
    gkf    = GroupKFold(n_splits=5)

    # PCA on full embeddings (fit on all, since we're evaluating probe config)
    scaler  = StandardScaler()
    emb_sc  = scaler.fit_transform(emb_full_aligned)
    pca     = PCA(n_components=pca_dim, random_state=42)
    Z_FULL  = pca.fit_transform(emb_sc).astype(np.float32)

    # Class prototypes
    CLASS_PROTOTYPES = {}
    if use_proto:
        for ci in range(NUM_CLASSES):
            pos = Y_FULL[:, ci] == 1
            if pos.sum() >= min_pos:
                CLASS_PROTOTYPES[ci] = Z_FULL[pos].mean(axis=0)

    # OOF base + prior
    oof_base  = np.zeros_like(scores_full_raw_aligned)
    oof_prior = np.zeros_like(scores_full_raw_aligned)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(scores_full_raw_aligned, groups=groups)):
        val_files  = set(full_truth.iloc[va_idx]['filename'].tolist())
        prior_mask = ~sc_clean['filename'].isin(val_files).values
        tables     = fit_prior_tables(
            sc_clean.loc[prior_mask].reset_index(drop=True),
            Y_SC[prior_mask]
        )
        va_base, va_prior = fuse_scores(
            scores_full_raw_aligned[va_idx],
            sites=full_truth.iloc[va_idx]['site'].to_numpy(),
            hours=full_truth.iloc[va_idx]['hour_utc'].to_numpy(),
            tables=tables,
            lambda_event=lambda_event, lambda_texture=lambda_texture,
            smooth_texture=smooth_texture, smooth_event=smooth_event,
        )
        oof_base[va_idx]  = va_base
        oof_prior[va_idx] = va_prior

    # Sigmoid
    oof_base_prob  = 1.0 / (1.0 + np.exp(-oof_base))
    oof_prior_prob = 1.0 / (1.0 + np.exp(-oof_prior))

    # LGBM OOF probe
    oof_probe = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    full_pos_counts = Y_FULL.sum(axis=0)
    PROBE_CLASS_IDX = np.where(full_pos_counts >= min_pos)[0].astype(np.int32)

    for cls_idx in PROBE_CLASS_IDX:
        y = Y_FULL[:, cls_idx]
        if y.sum() == 0 or y.sum() == len(y): continue

        n_rows = len(Z_FULL)
        if use_proto and cls_idx in CLASS_PROTOTYPES:
            _proto_sim = cosine_sim_to_prototype(Z_FULL, CLASS_PROTOTYPES[cls_idx])
        else:
            _proto_sim = np.zeros(n_rows, dtype=np.float32)

        if use_family:
            _fam_idxs = FAMILY_IDX_MAP.get(CLASS_FAMILY.get(cls_idx,'Unknown'), np.array([]))
            _other    = _fam_idxs[_fam_idxs != cls_idx]
            _fam_mean = oof_base_prob[:, _other].mean(axis=1) if len(_other) > 0 else np.zeros(n_rows, dtype=np.float32)
        else:
            _fam_mean = np.zeros(n_rows, dtype=np.float32)

        X_cls = build_class_features(
            Z_FULL,
            raw_col=scores_full_raw_aligned[:, cls_idx],
            prior_col=oof_prior_prob[:, cls_idx],
            base_col=oof_base_prob[:, cls_idx],
            proto_sim_col=_proto_sim,
            family_mean_col=_fam_mean,
        )

        n_pos = y.sum(); n_neg = len(y) - n_pos
        spw   = float(n_neg) / max(float(n_pos), 1.0)
        params = dict(lgbm_params, scale_pos_weight=spw)
        clf = LGBMClassifier(**params)
        clf.fit(X_cls, y)
        oof_probe[:, cls_idx] = clf.predict_proba(X_cls)[:, 1]

    # Blend: probe × Perch + SED
    probe_auc = macro_auc(Y_FULL, oof_probe)

    # Perch sigmoid base
    perch_probs = oof_base_prob

    # Blend Perch + SED
    if sed_w > 0 and perch_w > 0:
        blend = vlom_blend(perch_probs, sed_aligned_full, w_a=perch_w, w_b=sed_w)
    elif sed_w > 0:
        blend = sed_aligned_full
    else:
        blend = perch_probs

    blend_auc = macro_auc(Y_FULL, blend)

    # Combine probe logit + blend
    probe_logit = np.log(np.clip(oof_probe, 1e-7, 1-1e-7)) - np.log(np.clip(1-oof_probe, 1e-7, 1-1e-7))
    blend_logit = np.log(np.clip(blend, 1e-7, 1-1e-7)) - np.log(np.clip(1-blend, 1e-7, 1-1e-7))
    final_logit = probe_logit + blend_logit
    final_prob  = 1.0 / (1.0 + np.exp(-final_logit))
    final_auc   = macro_auc(Y_FULL, final_prob)

    return {'probe_auc': probe_auc, 'blend_auc': blend_auc, 'final_auc': final_auc}


# ── Parameter grid ─────────────────────────────────────────────────────────────
BASE_LGBM = dict(n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.05,
                 min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=4)

SWEEP = [
    # Baseline
    {'name': 'baseline_50_50',          'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': True,  'use_family': True},
    # Blend weight variants
    {'name': 'perch60_sed40',           'perch_w': 0.6,  'sed_w': 0.4,  'use_proto': True,  'use_family': True},
    {'name': 'perch40_sed60',           'perch_w': 0.4,  'sed_w': 0.6,  'use_proto': True,  'use_family': True},
    {'name': 'perch70_sed30',           'perch_w': 0.7,  'sed_w': 0.3,  'use_proto': True,  'use_family': True},
    {'name': 'perch30_sed70',           'perch_w': 0.3,  'sed_w': 0.7,  'use_proto': True,  'use_family': True},
    {'name': 'perch_only',              'perch_w': 1.0,  'sed_w': 0.0,  'use_proto': True,  'use_family': True},
    {'name': 'sed_only',                'perch_w': 0.0,  'sed_w': 1.0,  'use_proto': True,  'use_family': True},
    # Feature ablations
    {'name': 'no_proto_no_family',      'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': False, 'use_family': False},
    {'name': 'proto_only',              'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': True,  'use_family': False},
    {'name': 'family_only',             'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': False, 'use_family': True},
    # Prior fusion variants
    {'name': 'lambda_event_02',         'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': True,  'use_family': True, 'lambda_event': 0.2},
    {'name': 'lambda_event_06',         'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': True,  'use_family': True, 'lambda_event': 0.6},
    {'name': 'smooth_event_010',        'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': True,  'use_family': True, 'smooth_event': 0.10},
    {'name': 'smooth_event_020',        'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': True,  'use_family': True, 'smooth_event': 0.20},
    # LGBM depth variants
    {'name': 'lgbm_depth4',             'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': True,  'use_family': True,
     'lgbm_params': dict(BASE_LGBM, max_depth=4, num_leaves=15)},
    {'name': 'lgbm_depth2',             'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': True,  'use_family': True,
     'lgbm_params': dict(BASE_LGBM, max_depth=2, num_leaves=4)},
    {'name': 'lgbm_200est',             'perch_w': 0.5,  'sed_w': 0.5,  'use_proto': True,  'use_family': True,
     'lgbm_params': dict(BASE_LGBM, n_estimators=200)},
]

# Add default lgbm_params if not specified
for cfg in SWEEP:
    cfg.setdefault('lgbm_params', BASE_LGBM)
    cfg.setdefault('lambda_event', 0.4)
    cfg.setdefault('lambda_texture', 1.0)
    cfg.setdefault('smooth_texture', 0.35)
    cfg.setdefault('smooth_event', 0.15)
    cfg.setdefault('pca_dim', 64)
    cfg.setdefault('min_pos', 8)

# ── Run sweep ──────────────────────────────────────────────────────────────────
results = []
for cfg in tqdm(SWEEP, desc='Sweeping params'):
    name = cfg['name']
    try:
        r = run_oof(cfg)
        results.append({'name': name, **r,
                        'perch_w': cfg['perch_w'], 'sed_w': cfg['sed_w'],
                        'use_proto': cfg['use_proto'], 'use_family': cfg['use_family'],
                        'lambda_event': cfg['lambda_event'],
                        'smooth_event': cfg['smooth_event'],
                        'lgbm_depth': cfg['lgbm_params']['max_depth'],
                        'lgbm_n_est': cfg['lgbm_params']['n_estimators']})
        print(f"  {name:35s}  probe={r['probe_auc']:.4f}  blend={r['blend_auc']:.4f}  final={r['final_auc']:.4f}")
    except Exception as e:
        print(f"  {name}: ERROR {e}")

df = pd.DataFrame(results).sort_values('final_auc', ascending=False)
out_path = ROOT / 'outputs/cv_sweep_results.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print("\n=== Top 10 by final_auc ===")
print(df[['name','perch_w','sed_w','use_proto','use_family','probe_auc','blend_auc','final_auc']].head(10).to_string(index=False))
