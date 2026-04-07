#!/usr/bin/env python3
"""
precompute_probe_cache.py
=========================
Pre-fit all few-shot probe components using GroupKFold-5 (true k-fold: PCA,
global_mean, TIP keys/values, LogReg, and prototypes are all fit on the
**training fold only** — the validation fold is never seen during fitting).

At Kaggle inference time the notebook loads probe_cache.pkl and averages the
predictions of all 5 fold models.

Saved objects:
  FROZEN_PROBE   – config dict (for version-check at load time)
  fold_models    – list of 5 dicts, each containing:
      global_mean     (1536,)  float32 — L2-norm centering mean (train fold)
      pca             sklearn PCA(128, whiten=True) fit on train fold
      tip_keys        (N_tr, 1536) float32 — L2-normalised processed embeddings
      tip_values      (N_tr, 234)  float32 — multi-hot support labels
      prior_tables    dict — site/hour prior tables from train-fold files
      probe_models    dict[cls_idx → LogisticRegression]
      proto_prototypes dict[cls_idx → (128,) float32 centroid]
      full_pos_counts  (234,) int — positives per class in train fold
  PRIMARY_LABELS – list of 234 class names
  n_support      – total support clips (all folds, i.e. N=708)

Usage:
  python scripts/precompute_probe_cache.py \\
      --meta   "birdclef-2026/notebook resource/best perch/perch meta/full_perch_meta.parquet" \\
      --npz    "birdclef-2026/notebook resource/best perch/perch meta/full_perch_arrays.npz" \\
      --data   birdclef-2026 \\
      --out    "birdclef-2026/notebook resource/best perch/perch meta/probe_cache.pkl"
"""

import argparse
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ── Defaults (edit if needed) ──────────────────────────────────────────────────
DEFAULT_META = "birdclef-2026/notebook resource/best perch/perch meta/full_perch_meta.parquet"
DEFAULT_NPZ  = "birdclef-2026/notebook resource/best perch/perch meta/full_perch_arrays.npz"
DEFAULT_DATA = "birdclef-2026"
DEFAULT_OUT  = "birdclef-2026/notebook resource/best perch/perch meta/probe_cache.pkl"

FROZEN_PROBE = {
    'pca_dim':        128,
    'pca_whiten':     True,
    'l2_norm':        True,
    'min_pos_logreg': 8,
    'min_pos_proto':  2,
    'C':              0.50,
    'alpha':          0.40,
    'proto_alpha':    0.30,
    'tip_alpha':      0.35,
    'tip_tau':        0.50,
    'graph_alpha':    0.25,
    'n_folds':        5,         # GroupKFold-5
}

# Distribution Calibration (Yang et al., ICLR 2021 — "Free Lunch for Few-Shot")
# Applied to rare classes (min_pos_proto ≤ n_pos < min_pos_logreg) instead of
# falling back to prototype-only. Borrows variance from K nearest base classes.
DISTRIB_CAL_K    = 3      # K nearest base classes to borrow variance from
DISTRIB_CAL_ALPHA= 0.5    # blend: 0=use rare class var only, 1=neighbor var only
DISTRIB_CAL_NAUG = 50     # synthetic positive samples per class
DISTRIB_CAL_ENABLED = True  # set False to disable (falls back to prototype only)
FROZEN_FUSION = {
    'lambda_event': 0.4, 'lambda_texture': 1.0,
    'lambda_proxy_texture': 0.8, 'smooth_texture': 0.35,
}

N_WINDOWS   = 12
MANUAL_SCIENTIFIC_NAME_MAP = {}


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions (mirrored from notebook cells 5 / 6 / 7)
# ══════════════════════════════════════════════════════════════════════════════

FNAME_RE = re.compile(r'BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg')

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {'file_id': None, 'site': None, 'hour_utc': -1, 'month': -1}
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format='%Y%m%d', errors='coerce')
    return {
        'file_id': file_id, 'site': site,
        'hour_utc': int(hms[:2]),
        'month': int(dt.month) if pd.notna(dt) else -1,
    }

def parse_soundscape_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(';') if t.strip()]

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))


def seq_features_1d(v):
    assert len(v) % N_WINDOWS == 0
    x      = v.reshape(-1, N_WINDOWS)
    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v  = np.repeat(x.max(axis=1),  N_WINDOWS)
    return prev_v, next_v, mean_v, max_v


def build_class_features(emb_proj, raw_col, prior_col, base_col):
    prev_base, next_base, mean_base, max_base = seq_features_1d(base_col)
    return np.concatenate([
        emb_proj,
        raw_col[:, None], prior_col[:, None], base_col[:, None],
        prev_base[:, None], next_base[:, None],
        mean_base[:, None], max_base[:, None],
    ], axis=1).astype(np.float32, copy=False)


def fit_prior_tables(prior_df, Y_prior):
    prior_df  = prior_df.reset_index(drop=True)
    global_p  = Y_prior.mean(axis=0).astype(np.float32)

    site_keys = sorted(prior_df['site'].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n    = np.zeros(len(site_keys), dtype=np.float32)
    site_p    = np.zeros((len(site_keys), Y_prior.shape[1]), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]; mask = prior_df['site'].astype(str).values == s
        site_n[i] = mask.sum(); site_p[i] = Y_prior[mask].mean(axis=0)

    hour_keys = sorted(prior_df['hour_utc'].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n    = np.zeros(len(hour_keys), dtype=np.float32)
    hour_p    = np.zeros((len(hour_keys), Y_prior.shape[1]), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]; mask = prior_df['hour_utc'].astype(int).values == h
        hour_n[i] = mask.sum(); hour_p[i] = Y_prior[mask].mean(axis=0)

    sh_to_i = {}; sh_n_list = []; sh_p_list = []
    for (s, h), idx in prior_df.groupby(['site', 'hour_utc']).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx)); sh_p_list.append(Y_prior[idx].mean(axis=0))

    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = (np.stack(sh_p_list).astype(np.float32)
            if len(sh_p_list) else np.zeros((0, Y_prior.shape[1]), dtype=np.float32))
    return dict(global_p=global_p, site_to_i=site_to_i, site_n=site_n, site_p=site_p,
                hour_to_i=hour_to_i, hour_n=hour_n, hour_p=hour_p,
                sh_to_i=sh_to_i, sh_n=sh_n, sh_p=sh_p)


def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    n   = len(sites)
    p   = np.repeat(tables['global_p'][None, :], n, axis=0).astype(np.float32, copy=True)
    site_idx = np.fromiter((tables['site_to_i'].get(str(s), -1) for s in sites),
                           dtype=np.int32, count=n)
    hour_idx = np.fromiter((tables['hour_to_i'].get(int(h), -1) if int(h) >= 0 else -1
                            for h in hours), dtype=np.int32, count=n)
    sh_idx   = np.fromiter((tables['sh_to_i'].get((str(s), int(h)), -1)
                            if int(h) >= 0 else -1 for s, h in zip(sites, hours)),
                           dtype=np.int32, count=n)
    for idx_arr, n_arr, p_arr in [
        (hour_idx, tables['hour_n'], tables['hour_p']),
        (site_idx, tables['site_n'], tables['site_p']),
    ]:
        valid = idx_arr >= 0
        if valid.any():
            nn = n_arr[idx_arr[valid]][:, None]; w = nn / (nn + 8.0)
            p[valid] = w * p_arr[idx_arr[valid]] + (1.0 - w) * p[valid]
    valid = sh_idx >= 0
    if valid.any():
        nn = tables['sh_n'][sh_idx[valid]][:, None]; w = nn / (nn + 4.0)
        p[valid] = w * tables['sh_p'][sh_idx[valid]] + (1.0 - w) * p[valid]
    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)


def fuse_scores_with_tables(base_scores, sites, hours, tables,
                             idx_mapped_active_event, idx_mapped_active_texture,
                             idx_selected_proxy_active_texture,
                             idx_selected_prioronly_active_event):
    scores = base_scores.copy()
    prior  = prior_logits_from_tables(sites, hours, tables)
    lam_e, lam_t = FROZEN_FUSION['lambda_event'], FROZEN_FUSION['lambda_texture']
    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += lam_e * prior[:, idx_mapped_active_event]
    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += lam_t * prior[:, idx_mapped_active_texture]
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += (
            FROZEN_FUSION['lambda_proxy_texture'] * prior[:, idx_selected_proxy_active_texture]
        )
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = (
            lam_e * prior[:, idx_selected_prioronly_active_event]
        )
    return scores, prior


def l2_normalize(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-8)


def distrib_calibrate_augment(pos_emb, all_protos, all_vars, K=3, alpha=0.5, N_aug=50,
                               rng=None):
    """
    Distribution Calibration (Yang et al., ICLR 2021).

    For a rare class with few positive embeddings, calibrate its distribution by
    borrowing variance statistics from the K nearest base classes (by prototype
    cosine similarity). Sample N_aug synthetic embeddings from the calibrated Gaussian.

    pos_emb:    (n_pos, D) — positive PCA embeddings for this rare class
    all_protos: (C, D)     — per-class prototype vectors (pre-computed from train fold)
    all_vars:   (C, D)     — per-class per-dimension variance (pre-computed)
    K:          number of nearest base classes to borrow from
    alpha:      variance blend weight (0=own var, 1=neighbor var)
    N_aug:      number of synthetic samples to return
    Returns:    (N_aug, D) synthetic embeddings
    """
    if rng is None:
        rng = np.random.default_rng(42)

    D        = pos_emb.shape[1]
    rare_proto = pos_emb.mean(axis=0)  # (D,)
    rare_var   = pos_emb.var(axis=0) if len(pos_emb) > 1 else np.zeros(D, dtype=np.float32)

    # Find K nearest base-class prototypes by cosine similarity
    norm_rare  = rare_proto / (np.linalg.norm(rare_proto) + 1e-8)
    norm_protos = all_protos / (np.linalg.norm(all_protos, axis=1, keepdims=True) + 1e-8)
    cos_sims   = norm_protos @ norm_rare                    # (C,)
    nn_idx     = np.argsort(cos_sims)[::-1][:K + 1]        # top-K+1 (exclude self)
    # exclude exact self match (if rare class is in all_protos)
    nn_idx     = [i for i in nn_idx if not np.allclose(all_protos[i], rare_proto)][:K]

    if len(nn_idx) == 0:
        # Fallback: sample from rare Gaussian without calibration
        calibrated_var = np.maximum(rare_var, 1e-6)
    else:
        neighbor_var   = all_vars[nn_idx].mean(axis=0)  # (D,)
        calibrated_var = (1.0 - alpha) * rare_var + alpha * neighbor_var
        calibrated_var = np.maximum(calibrated_var, 1e-6)

    # Sample from calibrated Gaussian N(rare_proto, calibrated_var)
    synthetic = (rng.standard_normal((N_aug, D)).astype(np.float32)
                 * np.sqrt(calibrated_var)[None, :]
                 + rare_proto[None, :])
    return synthetic.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    DATA_DIR  = Path(args.data)
    META_PATH = Path(args.meta)
    NPZ_PATH  = Path(args.npz)
    OUT_PATH  = Path(args.out)

    print(f'Loading perch meta from: {META_PATH}')
    meta_full       = pd.read_parquet(META_PATH)
    arr             = np.load(NPZ_PATH)
    emb_full        = arr['emb_full'].astype(np.float32)
    scores_full_raw = arr['scores_full_raw'].astype(np.float32)
    N_support       = len(meta_full)
    print(f'  Support clips: {N_support}  Files: {meta_full["filename"].nunique()}')
    print(f'  emb_full: {emb_full.shape}  scores_full_raw: {scores_full_raw.shape}')

    # ── Load labels ────────────────────────────────────────────────────────────
    print('Loading taxonomy + soundscape labels ...')
    taxonomy          = pd.read_csv(DATA_DIR / 'taxonomy.csv')
    soundscape_labels = pd.read_csv(DATA_DIR / 'train_soundscapes_labels.csv')
    sample_sub        = pd.read_csv(DATA_DIR / 'sample_submission.csv')

    taxonomy['primary_label']          = taxonomy['primary_label'].astype(str)
    soundscape_labels['primary_label'] = soundscape_labels['primary_label'].astype(str)

    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    NUM_CLASSES    = len(PRIMARY_LABELS)
    label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

    # Build sc_clean with row_id
    sc_clean = (
        soundscape_labels
        .groupby(['filename', 'start', 'end'])['primary_label']
        .apply(union_labels)
        .reset_index(name='label_list')
    )
    sc_clean['start_sec'] = pd.to_timedelta(sc_clean['start']).dt.total_seconds().astype(int)
    sc_clean['end_sec']   = pd.to_timedelta(sc_clean['end']).dt.total_seconds().astype(int)
    sc_clean['row_id']    = (sc_clean['filename'].str.replace('.ogg', '', regex=False)
                             + '_' + sc_clean['end_sec'].astype(str))
    meta_sc   = sc_clean['filename'].apply(lambda fn: parse_soundscape_filename(fn)).apply(pd.Series)
    sc_clean  = pd.concat([sc_clean, meta_sc], axis=1)

    # Multi-hot matrix
    Y_SC = np.zeros((len(sc_clean), NUM_CLASSES), dtype=np.uint8)
    for i, labels in enumerate(sc_clean['label_list']):
        idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
        if idxs:
            Y_SC[i, idxs] = 1

    # ── Taxonomy mapping (for fuse_scores) ─────────────────────────────────────
    print('Building taxonomy mapping ...')
    CLASS_NAME_MAP = taxonomy.set_index('primary_label')['class_name'].to_dict()
    TEXTURE_TAXA   = {'Amphibia', 'Insecta'}

    ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]
    idx_active_texture = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA],
        dtype=np.int32)
    idx_active_event = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA],
        dtype=np.int32)

    # BC mapping for MAPPED_MASK
    bc_labels_path = DATA_DIR / 'assets' / 'labels.csv'
    if not bc_labels_path.exists():
        bc_labels_path = DATA_DIR.parent / 'models' / 'labels.csv'
    if bc_labels_path.exists():
        bc_labels_df = pd.read_csv(bc_labels_path).reset_index().rename(
            columns={'index': 'bc_index', 'inat2024_fsd50k': 'scientific_name'})
        NO_LABEL_INDEX = len(bc_labels_df)
        taxonomy_copy  = taxonomy.copy()
        taxonomy_copy['scientific_name_lookup'] = taxonomy_copy['scientific_name'].replace(
            MANUAL_SCIENTIFIC_NAME_MAP)
        bc_lookup = bc_labels_df.rename(columns={'scientific_name': 'scientific_name_lookup'})
        mapping   = taxonomy_copy.merge(
            bc_lookup[['scientific_name_lookup', 'bc_index']],
            on='scientific_name_lookup', how='left')
        mapping['bc_index'] = mapping['bc_index'].fillna(NO_LABEL_INDEX).astype(int)
        BC_INDICES   = np.array([int(mapping.set_index('primary_label')['bc_index'].loc[c])
                                  for c in PRIMARY_LABELS], dtype=np.int32)
        MAPPED_MASK  = BC_INDICES != NO_LABEL_INDEX
    else:
        print('  WARNING: labels.csv not found — using dummy MAPPED_MASK (all True)')
        MAPPED_MASK = np.ones(NUM_CLASSES, dtype=bool)

    idx_mapped_active_texture          = idx_active_texture[MAPPED_MASK[idx_active_texture]]
    idx_mapped_active_event            = idx_active_event[MAPPED_MASK[idx_active_event]]
    idx_unmapped_active_texture        = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
    idx_unmapped_active_event          = idx_active_event[~MAPPED_MASK[idx_active_event]]
    idx_selected_proxy_active_texture  = np.array([], dtype=np.int32)  # simplified
    idx_selected_prioronly_active_event = idx_unmapped_active_event

    # ── Align meta to labels ───────────────────────────────────────────────────
    print('Aligning support embeddings to labels ...')
    rid_to_row = {rid: i for i, rid in enumerate(sc_clean['row_id'])}
    aligned    = [rid_to_row[rid] for rid in meta_full['row_id']]
    Y_FULL     = Y_SC[aligned]
    print(f'  Y_FULL: {Y_FULL.shape}  '
          f'classes>=8: {(Y_FULL.sum(0)>=8).sum()}  '
          f'classes 2-7: {((Y_FULL.sum(0)>=2)&(Y_FULL.sum(0)<8)).sum()}')

    # ── GroupKFold-5 OOF base/prior (for training meta-features) ───────────────
    # These OOF scores are proper: each clip's base/prior are computed without
    # its file in the prior tables. They are reused as training features for
    # the per-fold LogReg (using tr_idx subset of these arrays).
    print('Building OOF base/prior features (GroupKFold-5) ...')
    groups_full = meta_full['filename'].to_numpy()
    gkf         = GroupKFold(n_splits=5)
    oof_base    = np.zeros_like(scores_full_raw, dtype=np.float32)
    oof_prior   = np.zeros_like(scores_full_raw, dtype=np.float32)

    sc_idx = sc_clean.reset_index(drop=True)
    splits = list(gkf.split(scores_full_raw, groups=groups_full))

    for fold_i, (tr_idx, va_idx) in enumerate(
            tqdm(splits, desc='OOF folds')):
        tr_idx = np.sort(tr_idx); va_idx = np.sort(va_idx)
        val_files     = set(meta_full.iloc[va_idx]['filename'].tolist())
        prior_mask    = ~sc_idx['filename'].isin(val_files).values
        prior_df_fold = sc_idx.loc[prior_mask].reset_index(drop=True)
        Y_prior_fold  = Y_SC[prior_mask]
        tables        = fit_prior_tables(prior_df_fold, Y_prior_fold)

        va_sites = meta_full.iloc[va_idx]['site'].to_numpy()
        va_hours = meta_full.iloc[va_idx]['hour_utc'].to_numpy()
        va_base, va_prior = fuse_scores_with_tables(
            scores_full_raw[va_idx], va_sites, va_hours, tables,
            idx_mapped_active_event, idx_mapped_active_texture,
            idx_selected_proxy_active_texture, idx_selected_prioronly_active_event)
        oof_base[va_idx]  = va_base
        oof_prior[va_idx] = va_prior

    # ── Per-fold fitting (GroupKFold-5, train fold only) ───────────────────────
    print('\nFitting 5 fold models (PCA + TIP + LogReg + Proto each on train fold) ...')
    n_comp = min(int(FROZEN_PROBE['pca_dim']), emb_full.shape[0] - 1, emb_full.shape[1])

    fold_models = []
    for fold_i, (tr_idx, va_idx) in enumerate(
            tqdm(splits, desc='Fold models')):
        tr_idx = np.sort(tr_idx); va_idx = np.sort(va_idx)
        print(f'\n  Fold {fold_i+1}/5  |  train={len(tr_idx)}  val={len(va_idx)}')

        # ── 1. PCA / global_mean on train fold ──────────────────────────────
        emb_tr     = emb_full[tr_idx]
        emb_tr_l2  = l2_normalize(emb_tr)
        global_mean_k = emb_tr_l2.mean(axis=0).astype(np.float32)
        emb_tr_proc   = (emb_tr_l2 - global_mean_k).astype(np.float32)

        pca_k = PCA(n_components=n_comp, whiten=FROZEN_PROBE['pca_whiten'], random_state=42)
        Z_tr  = pca_k.fit_transform(emb_tr_proc).astype(np.float32)
        print(f'    PCA({n_comp}) expl_var={pca_k.explained_variance_ratio_.sum():.4f}')

        # ── 2. TIP-Adapter keys/values from train fold ──────────────────────
        tip_keys_k   = l2_normalize(emb_tr_proc).astype(np.float32)   # (N_tr, 1536)
        tip_values_k = Y_FULL[tr_idx].astype(np.float32)               # (N_tr, 234)

        # ── 3. Prior tables from train-fold files only ───────────────────────
        tr_files      = set(meta_full.iloc[tr_idx]['filename'].tolist())
        prior_mask_k  = sc_idx['filename'].isin(tr_files).values
        prior_df_k    = sc_idx.loc[prior_mask_k].reset_index(drop=True)
        Y_prior_k     = Y_SC[prior_mask_k]
        tables_k      = fit_prior_tables(prior_df_k, Y_prior_k)

        # ── 4. LogReg on train fold (using OOF meta-features for tr_idx) ────
        pos_counts_k  = Y_FULL[tr_idx].sum(axis=0)
        logreg_idx_k  = np.where(pos_counts_k >= int(FROZEN_PROBE['min_pos_logreg']))[0]
        probe_models_k = {}
        for cls_idx in tqdm(logreg_idx_k, desc=f'    LogReg fold {fold_i+1}', leave=False):
            y = Y_FULL[tr_idx, cls_idx]
            if y.sum() == 0 or y.sum() == len(tr_idx):
                continue
            X_cls = build_class_features(
                Z_tr,
                raw_col=scores_full_raw[tr_idx, cls_idx],
                prior_col=oof_prior[tr_idx, cls_idx],
                base_col=oof_base[tr_idx, cls_idx],
            )
            # ── LP++ (Huang et al., CVPR 2024): prototype-initialized LogReg ──
            # Compute class prototype in PCA space for warm initialization
            proto_lp = Z_tr[y.astype(bool)].mean(axis=0)               # (128,)
            proto_lp = proto_lp / (np.linalg.norm(proto_lp) + 1e-8)

            clf = LogisticRegression(C=float(FROZEN_PROBE['C']), max_iter=300,
                                     solver='lbfgs', warm_start=True,
                                     class_weight='balanced')
            clf.fit(X_cls, y)                                           # initial fit
            # Override coef_ with prototype (PCA dims) + keep meta-feat dims near 0
            coef_lp = np.zeros((1, X_cls.shape[1]), dtype=np.float64)
            coef_lp[0, :n_comp] = proto_lp.astype(np.float64)
            clf.coef_ = coef_lp
            clf.intercept_ = np.zeros(1, dtype=np.float64)
            clf.set_params(max_iter=200)                                # fewer iters from good init
            clf.fit(X_cls, y)                                           # re-fit from prototype init
            probe_models_k[cls_idx] = clf
        print(f'    LogReg: {len(probe_models_k)} probes')

        # ── 5. Prototype fallback on train fold ──────────────────────────────
        rare_idx_k = np.where(
            (pos_counts_k >= int(FROZEN_PROBE['min_pos_proto'])) &
            (pos_counts_k <  int(FROZEN_PROBE['min_pos_logreg']))
        )[0]
        proto_prototypes_k = {}

        if DISTRIB_CAL_ENABLED and len(rare_idx_k) > 0:
            # ── Distribution Calibration (Yang et al., ICLR 2021) ───────────
            # Pre-compute per-class prototypes & variances in PCA space (train fold)
            all_protos_k = np.zeros((Y_FULL.shape[1], Z_tr.shape[1]), dtype=np.float32)
            all_vars_k   = np.zeros_like(all_protos_k)
            for ci in range(Y_FULL.shape[1]):
                pm = Y_FULL[tr_idx, ci].astype(bool)
                if pm.sum() >= 1:
                    all_protos_k[ci] = Z_tr[pm].mean(axis=0)
                if pm.sum() >= 2:
                    all_vars_k[ci]   = Z_tr[pm].var(axis=0)

            rng_cal = np.random.default_rng(42 + fold_i)
            distcal_logreg = 0

            for cls_idx in tqdm(rare_idx_k, desc=f'    DistCal+LogReg fold {fold_i+1}', leave=False):
                pos_mask = Y_FULL[tr_idx, cls_idx].astype(bool)
                if pos_mask.sum() == 0:
                    continue
                pos_emb_real = Z_tr[pos_mask]        # (n_pos, 128) real positives

                # Synthesize N_aug extra positives via distribution calibration
                syn_pos = distrib_calibrate_augment(
                    pos_emb_real, all_protos_k, all_vars_k,
                    K=DISTRIB_CAL_K, alpha=DISTRIB_CAL_ALPHA,
                    N_aug=DISTRIB_CAL_NAUG, rng=rng_cal,
                )

                # Build combined training set: real_pos + syn_pos + negatives
                neg_mask  = ~pos_mask
                neg_emb   = Z_tr[neg_mask]           # all real negatives
                aug_emb   = np.vstack([pos_emb_real, syn_pos, neg_emb])
                aug_y     = np.hstack([
                    np.ones(len(pos_emb_real) + len(syn_pos)),
                    np.zeros(len(neg_emb)),
                ]).astype(np.int32)

                # Build class features for augmented set
                # (only PCA dims — raw/prior/base features not meaningful for synthetic)
                # Use simplified features: [Z_128] only for rare-class LogReg
                aug_raw   = np.zeros(len(aug_y), dtype=np.float32)
                aug_prior = np.zeros(len(aug_y), dtype=np.float32)
                aug_base  = np.zeros(len(aug_y), dtype=np.float32)
                X_aug = build_class_features(
                    aug_emb, raw_col=aug_raw, prior_col=aug_prior, base_col=aug_base
                )

                clf = LogisticRegression(C=float(FROZEN_PROBE['C']), max_iter=600,
                                         solver='liblinear', class_weight='balanced')
                clf.fit(X_aug, aug_y)
                probe_models_k[cls_idx] = clf
                distcal_logreg += 1

            print(f'    DistCal LogReg: {distcal_logreg} rare-class probes added')
        else:
            # Original prototype fallback (no distribution calibration)
            for cls_idx in rare_idx_k:
                pos_mask = Y_FULL[tr_idx, cls_idx].astype(bool)
                if pos_mask.sum() == 0:
                    continue
                proto = Z_tr[pos_mask].mean(axis=0)
                proto_prototypes_k[cls_idx] = (
                    proto / (np.linalg.norm(proto) + 1e-8)
                ).astype(np.float32)

        # Prototype for classes that didn't get DistCal LogReg (n_pos < min_pos_proto)
        for cls_idx in rare_idx_k:
            if cls_idx not in probe_models_k:
                pos_mask = Y_FULL[tr_idx, cls_idx].astype(bool)
                if pos_mask.sum() == 0:
                    continue
                proto = Z_tr[pos_mask].mean(axis=0)
                proto_prototypes_k[cls_idx] = (
                    proto / (np.linalg.norm(proto) + 1e-8)
                ).astype(np.float32)
        print(f'    Proto: {len(proto_prototypes_k)} fallbacks')

        fold_models.append({
            'global_mean':      global_mean_k,       # (1536,)
            'pca':              pca_k,                # sklearn PCA
            'tip_keys':         tip_keys_k,           # (N_tr, 1536)
            'tip_values':       tip_values_k,         # (N_tr, 234)
            'prior_tables':     tables_k,             # dict
            'probe_models':     probe_models_k,       # dict int→LogReg
            'proto_prototypes': proto_prototypes_k,   # dict int→(128,)
            'full_pos_counts':  pos_counts_k,         # (234,)
        })

    # ── Save cache ─────────────────────────────────────────────────────────────
    full_pos_counts_all = Y_FULL.sum(axis=0)
    cache = {
        'FROZEN_PROBE':    FROZEN_PROBE,
        'fold_models':     fold_models,          # list of 5 dicts
        'PRIMARY_LABELS':  PRIMARY_LABELS,
        'n_support':       N_support,
        'full_pos_counts': full_pos_counts_all,  # (234,) over all clips
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(cache, f, protocol=4)

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f'\n✓ Saved probe_cache.pkl → {OUT_PATH}  ({size_mb:.1f} MB)')
    total_logreg = sum(len(fm['probe_models']) for fm in fold_models)
    total_proto  = sum(len(fm['proto_prototypes']) for fm in fold_models)
    print(f'  Folds: {len(fold_models)}  '
          f'Total LogReg: {total_logreg}  Total Proto: {total_proto}  '
          f'Support: {N_support} clips')
    for i, fm in enumerate(fold_models):
        n_tr = fm['tip_keys'].shape[0]
        print(f'  Fold {i+1}: N_tr={n_tr}  '
              f'LogReg={len(fm["probe_models"])}  '
              f'Proto={len(fm["proto_prototypes"])}  '
              f'TIP_KEYS={fm["tip_keys"].shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', default=DEFAULT_META)
    parser.add_argument('--npz',  default=DEFAULT_NPZ)
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--out',  default=DEFAULT_OUT)
    main(parser.parse_args())
