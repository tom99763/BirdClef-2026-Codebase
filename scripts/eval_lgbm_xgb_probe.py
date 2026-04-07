#!/usr/bin/env python3
"""
eval_lgbm_xgb_probe.py
=======================
Local OOF evaluation: LGBM vs XGB (multiple param sets) vs ensemble.
Uses same features as v3-lgbm-infer (74-dim: PCA64 + raw + prior + base + seq + interactions).
Outputs best XGB params and optimal ensemble weight.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_lgbm_xgb_probe.py
"""

import re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / 'birdclef-2026'
CACHE_META = ROOT / 'submissions_v2/few_shot/full_perch_meta.parquet'
CACHE_NPZ  = ROOT / 'submissions_v2/few_shot/full_perch_arrays.npz'

N_WINDOWS  = 12
NUM_CLASSES = 234

# ── Frozen probe params (same as v3-lgbm-infer) ───────────────────────────────
FROZEN_PROBE  = {'pca_dim': 64, 'min_pos': 8, 'C': 0.50, 'alpha': 0.40}
FROZEN_FUSION = {'lambda_event': 0.4, 'lambda_texture': 1.0,
                 'lambda_proxy_texture': 0.8, 'smooth_texture': 0.35}

# ── LGBM baseline params (from v3-lgbm-infer) ────────────────────────────────
N_THREADS = 4   # limit per-model threads to avoid contention (20 CPUs / 6 models ≈ 3)

LGBM_PARAMS = {
    'n_estimators': 100, 'max_depth': 3, 'num_leaves': 7,
    'learning_rate': 0.05, 'min_child_samples': 5,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'random_state': 42, 'verbose': -1, 'n_jobs': N_THREADS,
}

# ── XGB param grids to test ───────────────────────────────────────────────────
XGB_GRIDS = {
    'xgb_v1_baseline': {
        'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05,
        'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': 42, 'verbosity': 0, 'tree_method': 'hist',
        'nthread': N_THREADS,
    },
    'xgb_v2_deeper': {
        'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05,
        'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': 42, 'verbosity': 0, 'tree_method': 'hist',
        'nthread': N_THREADS,
    },
    'xgb_v3_more_trees': {
        'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.03,
        'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': 42, 'verbosity': 0, 'tree_method': 'hist',
        'nthread': N_THREADS,
    },
    'xgb_v4_strong_reg': {
        'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05,
        'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 0.5, 'reg_lambda': 2.0,
        'random_state': 42, 'verbosity': 0, 'tree_method': 'hist',
        'nthread': N_THREADS,
    },
    'xgb_v5_gamma': {
        'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05,
        'min_child_weight': 3, 'gamma': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': 42, 'verbosity': 0, 'tree_method': 'hist',
        'nthread': N_THREADS,
    },
}


# ── Data loading helpers ──────────────────────────────────────────────────────

FNAME_RE = re.compile(r'BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg')

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {'site': None, 'hour_utc': -1, 'month': -1}
    _, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format='%Y%m%d', errors='coerce')
    return {'site': site, 'hour_utc': int(hms[:2]),
            'month': int(dt.month) if pd.notna(dt) else -1}

def parse_soundscape_labels(x):
    if pd.isna(x): return []
    return [t.strip() for t in str(x).split(';') if t.strip()]

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))


def fit_prior_tables(prior_df, Y_prior):
    prior_df = prior_df.reset_index(drop=True)
    global_p = Y_prior.mean(axis=0).astype(np.float32)

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
    for (s, h), idx in prior_df.groupby(['site', 'hour_utc']).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx)); sh_p_list.append(Y_prior[idx].mean(axis=0))
    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = (np.stack(sh_p_list).astype(np.float32)
            if sh_p_list else np.zeros((0, Y_prior.shape[1]), dtype=np.float32))

    return {'global_p': global_p, 'site_to_i': site_to_i, 'site_n': site_n, 'site_p': site_p,
            'hour_to_i': hour_to_i, 'hour_n': hour_n, 'hour_p': hour_p,
            'sh_to_i': sh_to_i, 'sh_n': sh_n, 'sh_p': sh_p}


def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables['global_p'][None, :], n, axis=0).astype(np.float32, copy=True)
    site_idx = np.fromiter((tables['site_to_i'].get(str(s), -1) for s in sites), dtype=np.int32, count=n)
    hour_idx = np.fromiter((tables['hour_to_i'].get(int(h), -1) if int(h) >= 0 else -1 for h in hours), dtype=np.int32, count=n)
    sh_idx   = np.fromiter((tables['sh_to_i'].get((str(s), int(h)), -1) if int(h) >= 0 else -1
                            for s, h in zip(sites, hours)), dtype=np.int32, count=n)
    valid = hour_idx >= 0
    if valid.any():
        nh = tables['hour_n'][hour_idx[valid]][:, None]; wh = nh / (nh + 8.0)
        p[valid] = wh * tables['hour_p'][hour_idx[valid]] + (1.0 - wh) * p[valid]
    valid = site_idx >= 0
    if valid.any():
        ns = tables['site_n'][site_idx[valid]][:, None]; ws = ns / (ns + 8.0)
        p[valid] = ws * tables['site_p'][site_idx[valid]] + (1.0 - ws) * p[valid]
    valid = sh_idx >= 0
    if valid.any():
        nsh = tables['sh_n'][sh_idx[valid]][:, None]; wsh = nsh / (nsh + 4.0)
        p[valid] = wsh * tables['sh_p'][sh_idx[valid]] + (1.0 - wsh) * p[valid]
    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)


def seq_features_1d(v):
    x = v.reshape(-1, N_WINDOWS)
    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v  = np.repeat(x.max(axis=1),  N_WINDOWS)
    return prev_v, next_v, mean_v, max_v


def build_class_features(emb_proj, raw_col, prior_col, base_col):
    """74-dim features: PCA64 + raw + prior + base + seq(4) + interactions(3)"""
    prev_base, next_base, mean_base, max_base = seq_features_1d(base_col)
    return np.concatenate([
        emb_proj,
        raw_col[:, None], prior_col[:, None], base_col[:, None],
        prev_base[:, None], next_base[:, None], mean_base[:, None], max_base[:, None],
        (raw_col * prior_col)[:, None],
        (raw_col * base_col)[:, None],
        (prior_col * base_col)[:, None],
    ], axis=1).astype(np.float32, copy=False)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    print('Loading Perch cache...')
    meta_full       = pd.read_parquet(CACHE_META)
    arr             = np.load(CACHE_NPZ)
    scores_full_raw = arr['scores_full_raw'].astype(np.float32)
    emb_full        = arr['emb_full'].astype(np.float32)

    print('Loading soundscape labels...')
    soundscape_labels = pd.read_csv(DATA_DIR / 'train_soundscapes_labels.csv')
    soundscape_labels['primary_label'] = soundscape_labels['primary_label'].astype(str)

    sc_clean = (
        soundscape_labels
        .groupby(['filename', 'start', 'end'])['primary_label']
        .apply(union_labels)
        .reset_index(name='label_list')
    )
    sc_clean['start_sec'] = pd.to_timedelta(sc_clean['start']).dt.total_seconds().astype(int)
    sc_clean['end_sec']   = pd.to_timedelta(sc_clean['end']).dt.total_seconds().astype(int)
    sc_clean['row_id']    = sc_clean['filename'].str.replace('.ogg','',regex=False) + '_' + sc_clean['end_sec'].astype(str)

    meta_sc = sc_clean['filename'].apply(lambda fn: parse_soundscape_filename(fn)).apply(pd.Series)
    sc_clean = pd.concat([sc_clean.reset_index(drop=True), meta_sc], axis=1)

    windows_per_file = sc_clean.groupby('filename').size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    sc_clean['file_fully_labeled'] = sc_clean['filename'].isin(full_files)

    # Load primary labels
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

    Y_SC = np.zeros((len(sc_clean), NUM_CLASSES), dtype=np.uint8)
    for i, labels in enumerate(sc_clean['label_list']):
        idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
        if idxs: Y_SC[i, idxs] = 1

    full_truth = (
        sc_clean[sc_clean['file_fully_labeled']]
        .sort_values(['filename', 'end_sec'])
        .reset_index(drop=False)
    )

    full_truth_aligned = full_truth.set_index('row_id').loc[meta_full['row_id']].reset_index()
    Y_FULL = Y_SC[full_truth_aligned['index'].to_numpy()]

    print(f'Clips: {len(meta_full)}  Classes w/ positives: {(Y_FULL.sum(0)>0).sum()}/{NUM_CLASSES}')
    return meta_full, scores_full_raw, emb_full, Y_FULL, sc_clean, Y_SC, PRIMARY_LABELS


def build_oof_features(meta_full, scores_full_raw, emb_full, Y_FULL,
                       sc_clean, Y_SC, n_splits=5):
    """Build oof_base, oof_prior via GroupKFold — same as notebook."""
    print('Building OOF base+prior...')
    groups = meta_full['filename'].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    # Need a rough fuse function (event+texture combined, simplified)
    oof_base  = np.zeros_like(scores_full_raw)
    oof_prior = np.zeros_like(scores_full_raw)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(scores_full_raw, groups=groups)):
        val_files = set(meta_full.iloc[va_idx]['filename'].tolist())
        prior_mask = ~sc_clean['filename'].isin(val_files).values
        tables = fit_prior_tables(
            sc_clean.loc[prior_mask].reset_index(drop=True), Y_SC[prior_mask])

        sites_va = meta_full.iloc[va_idx]['site'].to_numpy()
        hours_va = meta_full.iloc[va_idx]['hour_utc'].to_numpy()
        prior_logits = prior_logits_from_tables(sites_va, hours_va, tables)

        # base = raw + prior (lambda=0.4 for all, simplified)
        base = scores_full_raw[va_idx].copy()
        base += 0.4 * prior_logits

        oof_base[va_idx]  = base
        oof_prior[va_idx] = prior_logits

    return oof_base, oof_prior


def fit_pca_emb(emb_full, pca_dim=64):
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_full)
    pca = PCA(n_components=pca_dim)
    Z = pca.fit_transform(emb_scaled).astype(np.float32)
    print(f'PCA({pca_dim}) explained variance: {pca.explained_variance_ratio_.sum():.3f}')
    return Z, scaler, pca


# ── OOF probe evaluation ──────────────────────────────────────────────────────

def run_oof_probe(meta_full, scores_full_raw, Z_FULL, Y_FULL, oof_prior, oof_base,
                  lgbm_params, xgb_params_dict, n_splits=5):
    """
    Run OOF for LGBM + each XGB variant.
    Returns dict of {model_name: oof_preds array}.
    """
    groups = meta_full['filename'].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    full_pos_counts = Y_FULL.sum(axis=0)
    probe_cls_idx = np.where(full_pos_counts >= FROZEN_PROBE['min_pos'])[0].astype(np.int32)

    model_names = ['lgbm'] + list(xgb_params_dict.keys())
    oof_preds = {name: np.zeros((len(Z_FULL), NUM_CLASSES), dtype=np.float32)
                 for name in model_names}

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(Z_FULL, groups=groups)):
        n_cls = len(probe_cls_idx)
        print(f'Fold {fold+1}/5  ({n_cls} classes)', flush=True)
        for ci, cls_idx in enumerate(probe_cls_idx):
            y    = Y_FULL[:, cls_idx]
            y_tr = y[tr_idx]
            if y_tr.sum() == 0 or (len(y_tr) - y_tr.sum()) == 0:
                continue
            if ci % 10 == 0:
                print(f'  class {ci}/{n_cls}', flush=True)

            X_tr = build_class_features(
                Z_FULL[tr_idx],
                raw_col=scores_full_raw[tr_idx, cls_idx],
                prior_col=oof_prior[tr_idx, cls_idx],
                base_col=oof_base[tr_idx, cls_idx],
            )
            X_va = build_class_features(
                Z_FULL[va_idx],
                raw_col=scores_full_raw[va_idx, cls_idx],
                prior_col=oof_prior[va_idx, cls_idx],
                base_col=oof_base[va_idx, cls_idx],
            )

            spw = float(len(y_tr) - y_tr.sum()) / max(float(y_tr.sum()), 1.0)

            # LGBM
            lgbm_f = LGBMClassifier(**lgbm_params, scale_pos_weight=spw)
            lgbm_f.fit(X_tr, y_tr)
            oof_preds['lgbm'][va_idx, cls_idx] = lgbm_f.predict_proba(X_va)[:, 1]

            # XGB variants
            for name, xgb_p in xgb_params_dict.items():
                xgb_f = XGBClassifier(**xgb_p, scale_pos_weight=spw)
                xgb_f.fit(X_tr, y_tr)
                oof_preds[name][va_idx, cls_idx] = xgb_f.predict_proba(X_va)[:, 1]

    return oof_preds, probe_cls_idx


def print_results(oof_preds, Y_FULL):
    keep = Y_FULL.sum(axis=0) > 0
    results = {}
    print('\n' + '='*62)
    print('  OOF Probe AUC Results')
    print('='*62)
    for name, preds in oof_preds.items():
        auc = roc_auc_score(Y_FULL[:, keep], preds[:, keep], average='macro')
        results[name] = auc
        print(f'  {name:<30} {auc:.5f}')

    print('\n  --- Ensemble (LGBM + best XGB) ---')
    lgbm_preds = oof_preds['lgbm']
    best_xgb_name = max((n for n in results if n != 'lgbm'), key=lambda n: results[n])
    best_xgb_preds = oof_preds[best_xgb_name]

    best_ens_auc = 0.0
    best_w = 0.5
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ens = w * lgbm_preds[:, keep] + (1-w) * best_xgb_preds[:, keep]
        a = roc_auc_score(Y_FULL[:, keep], ens, average='macro')
        marker = ' ←' if a > best_ens_auc else ''
        print(f'  LGBM={w:.1f} {best_xgb_name}={1-w:.1f}: {a:.5f}{marker}')
        if a > best_ens_auc:
            best_ens_auc = a
            best_w = w

    print(f'\n  Best XGB variant : {best_xgb_name}  AUC={results[best_xgb_name]:.5f}')
    print(f'  Best ensemble    : LGBM={best_w:.1f}  XGB={1-best_w:.1f}  AUC={best_ens_auc:.5f}')
    print(f'  LGBM baseline    : {results["lgbm"]:.5f}')
    delta = best_ens_auc - results['lgbm']
    print(f'  Ensemble gain    : {delta:+.5f}')
    print('='*62)
    return best_xgb_name, best_w, results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()

    meta_full, scores_full_raw, emb_full, Y_FULL, sc_clean, Y_SC, PRIMARY_LABELS = load_data()
    oof_base, oof_prior = build_oof_features(meta_full, scores_full_raw, emb_full,
                                              Y_FULL, sc_clean, Y_SC)
    Z_FULL, scaler, pca = fit_pca_emb(emb_full, pca_dim=64)

    print(f'\nRunning OOF for LGBM + {len(XGB_GRIDS)} XGB variants...')
    oof_preds, probe_cls_idx = run_oof_probe(
        meta_full, scores_full_raw, Z_FULL, Y_FULL, oof_prior, oof_base,
        lgbm_params=LGBM_PARAMS,
        xgb_params_dict=XGB_GRIDS,
    )

    best_xgb_name, best_w_lgbm, results = print_results(oof_preds, Y_FULL)

    print(f'\nTotal time: {(time.time()-t0)/60:.1f} min')
    print(f'\n→ Best XGB params to write into notebook: XGB_GRIDS["{best_xgb_name}"]')
    print(f'→ ENSEMBLE_W_LGBM = {best_w_lgbm:.1f}')
    print(f'→ ENSEMBLE_W_XGB  = {1-best_w_lgbm:.1f}')
