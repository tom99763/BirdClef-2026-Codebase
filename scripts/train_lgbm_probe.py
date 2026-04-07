"""
Train LGBM per-class Perch probe and save pkl to submissions_v3/weights/.

Usage:
    python scripts/train_lgbm_probe.py

Generates: submissions_v3/weights/lgbm_probe_models.pkl
"""
import pickle, re
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = Path("birdclef-2026")
PERCH_CACHE    = Path("birdclef-2026/notebook resource/best perch/perch meta")
OUT_PKL        = Path("submissions_v3/weights/lgbm_probe_models.pkl")

# ── Frozen probe params (same as v3 notebook) ─────────────────────────────────
FROZEN_PROBE   = {"pca_dim": 64, "min_pos": 8, "alpha": 0.40}
FROZEN_FUSION  = {"lambda_event": 0.4, "lambda_texture": 1.0,
                  "lambda_proxy_texture": 0.8, "smooth_texture": 0.35}
LGBM_PARAMS    = {
    "n_estimators": 100, "max_depth": 3, "num_leaves": 7,
    "learning_rate": 0.05, "min_child_samples": 5,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "verbosity": -1, "n_jobs": -1,
}

N_WINDOWS = 12

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading Perch cache...")
arr             = np.load(PERCH_CACHE / "full_perch_arrays.npz")
scores_full_raw = arr["scores_full_raw"].astype(np.float32)
emb_full        = arr["emb_full"].astype(np.float32)
meta_full       = pd.read_parquet(PERCH_CACHE / "full_perch_meta.parquet")

sample_sub      = pd.read_csv(BASE / "sample_submission.csv")
taxonomy        = pd.read_csv(BASE / "taxonomy.csv")
sc_labels_raw   = pd.read_csv(BASE / "train_soundscapes_labels.csv")

PRIMARY_LABELS  = sample_sub.columns[1:].tolist()
NUM_CLASSES     = len(PRIMARY_LABELS)
label_to_idx    = {c: i for i, c in enumerate(PRIMARY_LABELS)}

print(f"scores: {scores_full_raw.shape}  emb: {emb_full.shape}  meta: {meta_full.shape}")
print(f"Classes: {NUM_CLASSES}")

# ── Parse soundscape labels ───────────────────────────────────────────────────
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def parse_soundscape_labels(x):
    if pd.isna(x): return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"site": None, "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))

sc_clean = (
    sc_labels_raw
    .groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels)
    .reset_index(name="label_list")
)
sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
sc_clean["end_sec"]   = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"]    = (sc_clean["filename"].str.replace(".ogg", "", regex=False)
                         + "_" + sc_clean["end_sec"].astype(str))
meta_sc = sc_clean["filename"].apply(lambda fn: parse_soundscape_filename(fn)).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta_sc], axis=1)

windows_per_file = sc_clean.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

Y_SC = np.zeros((len(sc_clean), NUM_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean["label_list"]):
    for lbl in labels:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1

full_truth = (sc_clean[sc_clean["file_fully_labeled"]]
              .sort_values(["filename", "end_sec"]).reset_index(drop=False))
full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index()
Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]

print(f"Fully-labeled files: {len(full_files)}  Full truth rows: {len(Y_FULL)}")

# ── Prior tables ──────────────────────────────────────────────────────────────
def fit_prior_tables(prior_df, Y_prior):
    prior_df  = prior_df.reset_index(drop=True)
    global_p  = Y_prior.mean(axis=0).astype(np.float32)
    site_keys = sorted(prior_df["site"].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n    = np.zeros(len(site_keys), np.float32)
    site_p    = np.zeros((len(site_keys), Y_prior.shape[1]), np.float32)
    for s in site_keys:
        i = site_to_i[s]; m = prior_df["site"].astype(str).values == s
        site_n[i] = m.sum(); site_p[i] = Y_prior[m].mean(axis=0)
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n    = np.zeros(len(hour_keys), np.float32)
    hour_p    = np.zeros((len(hour_keys), Y_prior.shape[1]), np.float32)
    for h in hour_keys:
        i = hour_to_i[h]; m = prior_df["hour_utc"].astype(int).values == h
        hour_n[i] = m.sum(); hour_p[i] = Y_prior[m].mean(axis=0)
    sh_to_i, sh_n_list, sh_p_list = {}, [], []
    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx)); sh_p_list.append(Y_prior[idx].mean(axis=0))
    sh_n = np.array(sh_n_list, np.float32)
    sh_p = (np.stack(sh_p_list).astype(np.float32) if sh_p_list
            else np.zeros((0, Y_prior.shape[1]), np.float32))
    return dict(global_p=global_p, site_to_i=site_to_i, site_n=site_n, site_p=site_p,
                hour_to_i=hour_to_i, hour_n=hour_n, hour_p=hour_p,
                sh_to_i=sh_to_i, sh_n=sh_n, sh_p=sh_p)

def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)
    site_idx = [tables["site_to_i"].get(str(s), -1) for s in sites]
    hour_idx = [tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours]
    sh_idx   = [tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1
                for s, h in zip(sites, hours)]
    for kind, idx_arr, n_arr, p_arr, ps in [
        ("hour", hour_idx, tables["hour_n"], tables["hour_p"], 8.0),
        ("site", site_idx, tables["site_n"], tables["site_p"], 8.0),
        ("sh",   sh_idx,   tables["sh_n"],   tables["sh_p"],   4.0),
    ]:
        valid = [i for i, x in enumerate(idx_arr) if x >= 0]
        if valid:
            vi = np.array(valid); ki = np.array([idx_arr[i] for i in valid])
            nk = n_arr[ki][:, None]; w = nk / (nk + ps)
            p[vi] = w * p_arr[ki] + (1.0 - w) * p[vi]
    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)

def smooth_cols_fixed12(scores, cols, alpha=0.35):
    if alpha <= 0 or len(cols) == 0: return scores.copy()
    s = scores.copy(); view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s

# ── Active class indices for fusion ──────────────────────────────────────────
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}

active_mask           = Y_SC.sum(axis=0) > 0
active_classes        = [PRIMARY_LABELS[i] for i in np.where(active_mask)[0]]
idx_active_texture    = np.array([label_to_idx[c] for c in active_classes
                                   if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA], dtype=np.int32)
idx_active_event      = np.array([label_to_idx[c] for c in active_classes
                                   if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA], dtype=np.int32)

# ── Simplified fusion (no proxy mapping for now — same as v3) ─────────────────
def fuse_scores(base_scores, sites, hours, tables):
    scores = base_scores.copy()
    prior  = prior_logits_from_tables(sites, hours, tables)
    lam_e  = FROZEN_FUSION["lambda_event"]
    lam_t  = FROZEN_FUSION["lambda_texture"]
    if len(idx_active_event):
        scores[:, idx_active_event]   += lam_e * prior[:, idx_active_event]
    if len(idx_active_texture):
        scores[:, idx_active_texture] += lam_t * prior[:, idx_active_texture]
    scores = smooth_cols_fixed12(scores, idx_active_texture,
                                  alpha=FROZEN_FUSION["smooth_texture"])
    return scores.astype(np.float32), prior

# ── Sequential features ───────────────────────────────────────────────────────
def seq_features_1d(v):
    assert len(v) % N_WINDOWS == 0
    x      = v.reshape(-1, N_WINDOWS)
    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v  = np.repeat(x.max(axis=1),  N_WINDOWS)
    return prev_v, next_v, mean_v, max_v

def build_class_features(emb_proj, raw_col, prior_col, base_col):
    """74-dim: PCA-64 + 7 meta + 3 interactions"""
    prev_b, next_b, mean_b, max_b = seq_features_1d(base_col)
    return np.concatenate([
        emb_proj,
        raw_col[:, None], prior_col[:, None], base_col[:, None],
        prev_b[:, None], next_b[:, None], mean_b[:, None], max_b[:, None],
        (raw_col * prior_col)[:, None],
        (raw_col * base_col)[:, None],
        (prior_col * base_col)[:, None],
    ], axis=1).astype(np.float32)

# ── OOF base/prior meta-features ─────────────────────────────────────────────
print("\nBuilding OOF base/prior features (5-fold)...")
groups_full = meta_full["filename"].to_numpy()
gkf         = GroupKFold(n_splits=5)

oof_base    = np.zeros_like(scores_full_raw)
oof_prior   = np.zeros_like(scores_full_raw)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(scores_full_raw, groups=groups_full), 1):
    va_idx    = np.sort(va_idx)
    val_files = set(meta_full.iloc[va_idx]["filename"].tolist())
    pm        = ~sc_clean["filename"].isin(val_files).values
    tables    = fit_prior_tables(sc_clean.loc[pm].reset_index(drop=True), Y_SC[pm])
    va_base, va_prior = fuse_scores(
        scores_full_raw[va_idx],
        sites=meta_full.iloc[va_idx]["site"].tolist(),
        hours=meta_full.iloc[va_idx]["hour_utc"].tolist(),
        tables=tables,
    )
    oof_base[va_idx]  = va_base
    oof_prior[va_idx] = va_prior
    print(f"  fold {fold} done")

# ── Final prior tables + scaler + PCA (fit on all data) ──────────────────────
print("\nFitting final prior tables, scaler, PCA...")
final_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)

emb_scaler = StandardScaler()
emb_scaled = emb_scaler.fit_transform(emb_full)
n_comp     = min(int(FROZEN_PROBE["pca_dim"]), emb_scaled.shape[0] - 1, emb_scaled.shape[1])
emb_pca    = PCA(n_components=n_comp)
Z_FULL     = emb_pca.fit_transform(emb_scaled).astype(np.float32)
print(f"Z_FULL: {Z_FULL.shape}  explained var: {emb_pca.explained_variance_ratio_.sum():.3f}")

# ── OOF AUC (baseline) ────────────────────────────────────────────────────────
keep = Y_FULL.sum(axis=0) > 0
baseline_auc = roc_auc_score(Y_FULL[:, keep], oof_base[:, keep], average="macro")
print(f"\nOOF baseline AUC: {baseline_auc:.6f}")

# ── Train final LGBM probes ───────────────────────────────────────────────────
print(f"\nTraining LGBM probes (min_pos={FROZEN_PROBE['min_pos']})...")
full_pos_counts = Y_FULL.sum(axis=0)
PROBE_IDX       = np.where(full_pos_counts >= int(FROZEN_PROBE["min_pos"]))[0].astype(np.int32)
probe_models    = {}

for cls_idx in tqdm(PROBE_IDX, desc="LGBM probes"):
    y = Y_FULL[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X_cls = build_class_features(
        Z_FULL,
        raw_col=scores_full_raw[:, cls_idx],
        prior_col=oof_prior[:, cls_idx],
        base_col=oof_base[:, cls_idx],
    )
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    clf   = LGBMClassifier(scale_pos_weight=max(1.0, n_neg / max(n_pos, 1)), **LGBM_PARAMS)
    clf.fit(X_cls, y)
    probe_models[cls_idx] = clf

print(f"Trained {len(probe_models)} LGBM probe models")

# ── OOF probe AUC (skip — too slow locally; run on Kaggle for diagnostics) ───
SKIP_OOF_PROBE_AUC = True
probe_auc = float("nan")
print("\n[OOF probe AUC skipped — see oof_baseline_auc in pkl for reference]")
if not SKIP_OOF_PROBE_AUC:
 oof_final = oof_base.copy()
 groups_full2 = meta_full["filename"].to_numpy()
 for fold, (tr_idx, va_idx) in enumerate(gkf.split(scores_full_raw, groups=groups_full2), 1):
    va_idx    = np.sort(va_idx)
    val_files = set(meta_full.iloc[va_idx]["filename"].tolist())
    pm        = ~sc_clean["filename"].isin(val_files).values
    tables_f  = fit_prior_tables(sc_clean.loc[pm].reset_index(drop=True), Y_SC[pm])
    base_tr, prior_tr = fuse_scores(
        scores_full_raw[tr_idx],
        meta_full.iloc[tr_idx]["site"].tolist(),
        meta_full.iloc[tr_idx]["hour_utc"].tolist(), tables_f,
    )
    base_va, prior_va = fuse_scores(
        scores_full_raw[va_idx],
        meta_full.iloc[va_idx]["site"].tolist(),
        meta_full.iloc[va_idx]["hour_utc"].tolist(), tables_f,
    )
    sc_fold = StandardScaler()
    ez_tr   = sc_fold.fit_transform(emb_full[tr_idx])
    ez_va   = sc_fold.transform(emb_full[va_idx])
    nc      = min(int(FROZEN_PROBE["pca_dim"]), ez_tr.shape[0]-1, ez_tr.shape[1])
    pca_f   = PCA(n_components=nc)
    Ztr     = pca_f.fit_transform(ez_tr).astype(np.float32)
    Zva     = pca_f.transform(ez_va).astype(np.float32)
    cls_f   = np.where(Y_FULL[tr_idx].sum(axis=0) >= int(FROZEN_PROBE["min_pos"]))[0]
    for ci in cls_f:
        y_tr = Y_FULL[tr_idx, ci]
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr): continue
        Xtr_c = build_class_features(Ztr, scores_full_raw[tr_idx, ci],
                                      prior_tr[:, ci], base_tr[:, ci])
        Xva_c = build_class_features(Zva, scores_full_raw[va_idx, ci],
                                      prior_va[:, ci], base_va[:, ci])
        n_p = int(y_tr.sum()); n_n = len(y_tr) - n_p
        clf = LGBMClassifier(scale_pos_weight=max(1.0, n_n / max(n_p, 1)), **LGBM_PARAMS)
        clf.fit(Xtr_c, y_tr)
        prob = clf.predict_proba(Xva_c)[:, 1].astype(np.float32)
        pred = np.log(prob + 1e-7) - np.log(1.0 - prob + 1e-7)
        oof_final[va_idx, ci] = (
            (1.0 - FROZEN_PROBE["alpha"]) * base_va[:, ci] + FROZEN_PROBE["alpha"] * pred
        )

if not SKIP_OOF_PROBE_AUC:
    probe_auc = roc_auc_score(Y_FULL[:, keep], oof_final[:, keep], average="macro")
    print(f"OOF baseline AUC : {baseline_auc:.6f}")
    print(f"OOF LGBM probe AUC: {probe_auc:.6f}  (delta={probe_auc - baseline_auc:+.6f})")

# ── Save pkl ──────────────────────────────────────────────────────────────────
OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PKL, "wb") as f:
    pickle.dump({
        "probe_models":       probe_models,
        "emb_scaler":         emb_scaler,
        "emb_pca":            emb_pca,
        "final_prior_tables": final_prior_tables,
        "frozen_probe":       FROZEN_PROBE,
        "oof_baseline_auc":   float(baseline_auc),
        "oof_probe_auc":      float(probe_auc),
    }, f, protocol=4)

import os
size_mb = os.path.getsize(OUT_PKL) / 1e6
print(f"\nSaved: {OUT_PKL}  ({size_mb:.1f} MB)")
print(f"Models: {len(probe_models)}  Classes probed: {[PRIMARY_LABELS[i] for i in list(probe_models.keys())[:10]]}...")
