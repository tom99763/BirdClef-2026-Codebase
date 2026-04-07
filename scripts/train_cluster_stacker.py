#!/usr/bin/env python3
"""
train_cluster_stacker.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cluster-based stacker using HDBSCAN on the 5-model prediction space
(and optionally Perch embeddings).

Core idea:
  In prediction space, windows from the same acoustic scene cluster together.
  HDBSCAN finds these dense "scene clusters"; within each cluster the
  mean label distribution corrects individual-window predictions.

Score-Weighted MST Distance (key innovation):
  d(x,y) = cosine_dist(x,y) × (1 + β × disagreement(x,y))
  disagreement = mean inter-model variance (how much the 5 models disagree)
  → uncertain windows are kept farther apart → cleaner cluster boundaries

5 Experiments:
  A1: Scores (1170-dim),  Cosine,         no pseudo
  A2: Perch Emb (1536),   Cosine,         no pseudo
  A3: Emb+Scores (2706),  Score-weighted, no pseudo
  B1: Scores (1170),      Cosine,         + pseudo anchors (5K subsample)
  B2: Emb+Scores (2706),  Score-weighted, + pseudo anchors (5K subsample)

Output artifacts:
  stacker_weights/stacker_cluster_{exp}_auc{auc:.4f}.pkl
  stacker_weights/stacker_results_cluster.xlsx
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, sys, warnings, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
import hdbscan

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
OUTPUTS    = ROOT / "outputs"
PERCH_META = ROOT / "birdclef-2026" / "notebook resource" / "current_subs 2" / "perch meta"
OUT_DIR    = ROOT / "birdclef-2026" / "notebook resource" / "current_subs 2" / "stacker_weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED      = 42
N_CLASSES = 234
N_WINDOWS = 12
FEAT_DIM  = 1170   # 5 × 234

# ── W&B ───────────────────────────────────────────────────────────────────────
try:
    import wandb
    wandb.init(project="birdclef-2026", name="cluster-stacker", config={
        "feat_dim": FEAT_DIM, "n_classes": N_CLASSES, "pseudo_n": 5000
    })
    USE_WANDB = True
except Exception:
    USE_WANDB = False
    print("W&B not available — skipping")


# ═══════════════════════════════════════════════════════════════════════════════
# [1/6] Load labeled features (708 rows, 59 files × 12 windows)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading labeled features (708 rows) …")

# Meta: row_id + filename per window
meta = pd.read_parquet(PERCH_META / "full_perch_meta.parquet")
assert len(meta) == 708, f"Expected 708, got {len(meta)}"
filenames_708 = meta["filename"].values   # (708,)
row_ids_708   = meta["row_id"].values     # (708,)

unique_files  = list(dict.fromkeys(filenames_708))
assert len(unique_files) == 59, f"Expected 59 unique files, got {len(unique_files)}"
file_to_idx   = {f: i for i, f in enumerate(unique_files)}
groups        = np.array([file_to_idx[f] for f in filenames_708], dtype=np.int32)

# Perch raw scores + embeddings
perch_arr = np.load(PERCH_META / "full_perch_arrays.npz")
perch_raw_prob = perch_arr["scores_full_raw"].astype(np.float32)  # (708, 234) prob
emb_lab        = perch_arr["emb_full"].astype(np.float32)         # (708, 1536)
print(f"  perch_raw_prob : {perch_raw_prob.shape}  range=[{perch_raw_prob.min():.3f},{perch_raw_prob.max():.3f}]")
print(f"  emb_lab        : {emb_lab.shape}")

# Perch prior-fused + MLP probe (OOF logit space)
oof_data   = np.load(PERCH_META / "full_oof_meta_features.npz")
perch_prior = oof_data["oof_base"].astype(np.float32)    # (708, 234) logit
mlp_probe   = oof_data["oof_prior"].astype(np.float32)   # (708, 234) logit (fallback)
fold_id     = oof_data["fold_id"].astype(np.int32)       # (708,)
print(f"  perch_prior    : {perch_prior.shape}  logit range=[{perch_prior.min():.2f},{perch_prior.max():.2f}]")
print(f"  mlp_probe      : {mlp_probe.shape}")

# Check for actual mlp_probe OOF file
mlp_probe_path = OUTPUTS / "mlp_probe_oof.npy"
if mlp_probe_path.exists():
    mlp_probe = np.load(mlp_probe_path).astype(np.float32)
    print(f"  mlp_probe      : loaded from mlp_probe_oof.npy")

# Proto SSM (59 files → broadcast to 708 rows)
proto_preds_59 = np.load(OUTPUTS / "proto_ssm_oof_preds.npy").astype(np.float32)   # (59, 234)
proto_files_59 = np.load(OUTPUTS / "proto_ssm_oof_file_list.npy", allow_pickle=True)  # (59,)
proto_708 = np.zeros((708, N_CLASSES), dtype=np.float32)
for wi, fname in enumerate(filenames_708):
    mask = proto_files_59 == fname
    if mask.any():
        proto_708[wi] = proto_preds_59[np.where(mask)[0][0]]
print(f"  proto_ssm      : {proto_708.shape}  range=[{proto_708.min():.2f},{proto_708.max():.2f}]")

# SED BranchEns→cSEBBs (pre-computed)
sed_csebbs = np.load(OUTPUTS / "stacker_train_sed_csebbs_v3.npy").astype(np.float32)  # (708, 234)
print(f"  sed_csebbs     : {sed_csebbs.shape}  range=[{sed_csebbs.min():.3f},{sed_csebbs.max():.3f}]")

# Ground-truth labels (align by row_id)
label_data  = np.load(OUTPUTS / "perch_labeled_ss.npz", allow_pickle=True)
rid_to_label = dict(zip(label_data["row_ids"], range(len(label_data["row_ids"]))))
Y_lab = np.zeros((708, N_CLASSES), dtype=np.float32)
for i, rid in enumerate(row_ids_708):
    if rid in rid_to_label:
        Y_lab[i] = label_data["labels"][rid_to_label[rid]]
print(f"  Y_lab          : {Y_lab.shape}  pos_rate={Y_lab.mean():.4f}")

# Convert to uniform logit space for stacker input
EPS = 1e-7
def safe_logit(p):
    p = np.clip(p.astype(np.float32), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

# Determine if each array is prob or logit
def to_logit(arr, name):
    mn, mx = float(arr.min()), float(arr.max())
    if mn >= -0.05 and mx <= 1.05:
        print(f"  → {name}: prob → logit")
        return safe_logit(arr)
    print(f"  → {name}: already logit [{mn:.2f},{mx:.2f}]")
    return arr.astype(np.float32)

perch_raw_l  = to_logit(perch_raw_prob, "perch_raw")
perch_prior_l = to_logit(perch_prior,   "perch_prior")
mlp_probe_l  = to_logit(mlp_probe,      "mlp_probe")
proto_l      = to_logit(proto_708,       "proto_ssm")
sed_l        = to_logit(sed_csebbs,      "sed_csebbs")

# Build 1170-dim feature matrix
X_lab_raw = np.concatenate([perch_raw_l, perch_prior_l, mlp_probe_l, proto_l, sed_l], axis=1)
assert X_lab_raw.shape == (708, FEAT_DIM)

# Load norm stats from v3
norm_data  = np.load(OUT_DIR / "stacker_norm_v3.npz", allow_pickle=True)
X_mean, X_std = norm_data["mean"], norm_data["std"]
X_lab_norm = ((X_lab_raw - X_mean) / (X_std + 1e-8)).astype(np.float32)
emb_lab_norm = normalize(emb_lab, norm='l2')   # L2-norm for cosine distances
print(f"\n  X_lab_norm     : {X_lab_norm.shape}  (normalized, 1170-dim)")
print(f"  emb_lab_norm   : {emb_lab_norm.shape}  (L2-normalized, 1536-dim)")


# ═══════════════════════════════════════════════════════════════════════════════
# [2/6] Load pseudo anchors (5K subsample for B-experiments)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Loading pseudo anchors (5K subsample) …")

PSEUDO_N = 5000

pseudo_cache   = np.load(OUTPUTS / "stacker_pseudo_features.npz", allow_pickle=True)
X_pseudo_raw   = pseudo_cache["X_pseudo_raw"]    # (127188, 1170) raw (unnormalized)
Y_pseudo_all   = pseudo_cache["Y_pseudo"]        # (127188, 234)
pseudo_fns_all = pseudo_cache["pseudo_filenames"] # (127188,)

rng = np.random.default_rng(SEED)
sub_idx = rng.choice(len(X_pseudo_raw), size=PSEUDO_N, replace=False)
X_pseudo_sub  = X_pseudo_raw[sub_idx]   # (5000, 1170)
Y_pseudo_sub  = Y_pseudo_all[sub_idx]   # (5000, 234)
X_pseudo_norm = ((X_pseudo_sub - X_mean) / (X_std + 1e-8)).astype(np.float32)

# Perch embeddings for pseudo anchors
perch_all_data = np.load(OUTPUTS / "perch_emb_all_ss.npz", allow_pickle=True)
all_emb_flat   = perch_all_data["emb"]        # (127896, 1536)
all_fns_flat   = perch_all_data["filenames"]  # (127896,)

# Build stem→rows map
from pathlib import Path as _P
fn_to_rows = {}
for i, fn in enumerate(all_fns_flat):
    stem = _P(fn).stem
    fn_to_rows.setdefault(stem, []).append(i)

emb_pseudo_sub = np.zeros((PSEUDO_N, 1536), dtype=np.float32)
pseudo_fns_sub = pseudo_fns_all[sub_idx]
for j, fn in enumerate(pseudo_fns_sub):
    stem = _P(fn).stem
    if stem in fn_to_rows:
        rows = fn_to_rows[stem]
        emb_pseudo_sub[j] = all_emb_flat[rows[j % len(rows)]]

emb_pseudo_norm = normalize(emb_pseudo_sub, norm='l2')

print(f"  X_pseudo_norm  : {X_pseudo_norm.shape}")
print(f"  emb_pseudo_norm: {emb_pseudo_norm.shape}")
print(f"  Y_pseudo_sub   : {Y_pseudo_sub.shape}  pos_rate={Y_pseudo_sub.mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# [3/6] Distance functions
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Defining distance/feature functions …")


def score_weighted_dist(X_scores_norm: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """
    Pairwise distance matrix with inter-model disagreement weighting.

    Split X (1170-dim) into 5 model blocks × 234 classes.
    d(x,y) = cosine_dist(x,y) × (1 + β × (var_x + var_y))
    var_x = mean variance across 5 models for sample x

    High model disagreement → inflated distance → less likely same cluster.
    """
    n = len(X_scores_norm)
    preds = X_scores_norm.reshape(n, 5, N_CLASSES)    # (n, 5, 234)
    model_var = preds.var(axis=1).mean(axis=1)          # (n,) per-sample disagreement
    D_cos = cosine_distances(X_scores_norm).astype(np.float64)
    uncertainty = model_var[:, None] + model_var[None, :]  # (n, n) pairwise
    D_w = D_cos * (1.0 + beta * uncertainty)
    np.fill_diagonal(D_w, 0.0)
    return D_w.astype(np.float32)


def build_emb_score_feat(X_norm, emb_norm, emb_w=0.5):
    """Concatenate Perch embedding + score features with relative weighting."""
    return np.concatenate([emb_norm * emb_w, X_norm * (1.0 - emb_w)], axis=1).astype(np.float32)


def macro_auc(y_true, y_score):
    keep = y_true.sum(0) > 0
    if keep.sum() == 0:
        return 0.0
    try:
        return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro'))
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# [4/6] ClusterStacker class
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Defining ClusterStacker …")


class ClusterStacker:
    """
    HDBSCAN-based cluster stacker.

    Fit:
      1. Cluster training windows in feature space.
      2. Store per-cluster label mean (species distribution).

    Predict:
      For each test window, find its cluster via approximate_predict,
      blend the direct model output with the cluster-conditional mean:
        final = (1 - α*s) * base_pred + (α*s) * cluster_mean
      where s = soft membership strength (0=noise, 1=core).
    """

    def __init__(self, min_cluster_size=8, min_samples=3,
                 blend_alpha=0.4, cluster_selection_method='eom',
                 use_precomputed=False):
        self.min_cluster_size = min_cluster_size
        self.min_samples      = min_samples
        self.blend_alpha      = blend_alpha
        self.csel             = cluster_selection_method
        self.use_precomputed  = use_precomputed
        self.clusterer        = None
        self.cluster_means    = {}
        self.global_mean      = None
        self._X_train         = None   # for nearest-cluster fallback

    def fit(self, X_feat, Y_labels, D_pre=None):
        # HDBSCAN BallTree does not support 'cosine' metric directly.
        # Always precompute cosine distance matrix and use metric='precomputed'.
        if D_pre is not None:
            D_input = np.array(D_pre, dtype=np.float64)
        else:
            D_input = cosine_distances(X_feat).astype(np.float64)

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='precomputed',
            cluster_selection_method=self.csel,
            prediction_data=False,  # not supported with precomputed
        )
        self.clusterer.fit(D_input)
        self._X_train = X_feat  # for soft-predict fallback

        lbl = self.clusterer.labels_
        n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_no = (lbl == -1).sum()
        print(f"    clusters={n_cl}  noise={n_no}/{len(lbl)}")

        for cid in set(lbl[lbl >= 0]):
            self.cluster_means[int(cid)] = Y_labels[lbl == cid].mean(0).astype(np.float32)
        self.global_mean = Y_labels.mean(0).astype(np.float32)
        return self

    def predict(self, X_feat_test, base_pred):
        """base_pred: (n, 234) probabilities."""
        # Always use nearest-cluster fallback (precomputed metric doesn't support approximate_predict)
        cids, strengths = self._nearest_cluster(X_feat_test)

        out = base_pred.copy().astype(np.float32)
        for i, (cid, s) in enumerate(zip(cids, strengths)):
            cid = int(cid)
            if cid >= 0 and cid in self.cluster_means:
                alpha = self.blend_alpha * float(s)
                out[i] = (1.0 - alpha) * base_pred[i] + alpha * self.cluster_means[cid]
        return out

    def _nearest_cluster(self, X_test):
        D = cosine_distances(X_test, self._X_train)   # (n_test, n_train)
        nn = np.argmin(D, axis=1)
        cids     = np.array([self.clusterer.labels_[i]       for i in nn], dtype=np.int32)
        strengths = np.array([self.clusterer.probabilities_[i] for i in nn], dtype=np.float32)
        return cids, strengths


# ═══════════════════════════════════════════════════════════════════════════════
# [5/6] Cross-validation — 5 experiments
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Cross-validation (5 experiments) …")

# Base prediction in prob space: mean of sigmoid(5 logit blocks)
base_prob_lab = sigmoid(X_lab_norm.reshape(708, 5, N_CLASSES)).mean(1)   # (708, 234)

gkf = GroupKFold(n_splits=5)
results = []


def cv_run(exp_name, feat_fn, dist_fn=None, use_pseudo=False,
           min_cs=8, blend_alpha=0.4, use_precomputed=False):
    """
    feat_fn(X_norm, emb_norm) → X_feat
    dist_fn(X_feat) → D_precomputed  (only used when use_precomputed=True)
    """
    print(f"\n  [{exp_name}]")
    fold_aucs = []

    for fold, (tr, va) in enumerate(gkf.split(X_lab_norm, groups=groups)):
        # Train features
        X_tr_feat = feat_fn(X_lab_norm[tr], emb_lab_norm[tr])
        Y_tr      = Y_lab[tr]
        base_tr   = base_prob_lab[tr]

        # Optionally append pseudo anchors
        if use_pseudo:
            X_ps_feat = feat_fn(X_pseudo_norm, emb_pseudo_norm)
            X_tr_feat = np.vstack([X_tr_feat, X_ps_feat])
            Y_tr      = np.vstack([Y_tr, Y_pseudo_sub])

        D_tr = dist_fn(X_tr_feat) if use_precomputed and dist_fn else None

        cs = ClusterStacker(min_cluster_size=min_cs, blend_alpha=blend_alpha,
                            use_precomputed=use_precomputed)
        cs.fit(X_tr_feat, Y_tr, D_pre=D_tr)

        # Validation features (labeled only)
        X_va_feat = feat_fn(X_lab_norm[va], emb_lab_norm[va])
        pred_va   = cs.predict(X_va_feat, base_prob_lab[va])
        auc       = macro_auc(Y_lab[va], pred_va)
        fold_aucs.append(auc)
        print(f"    fold {fold+1}: AUC={auc:.4f}  clusters={len(cs.cluster_means)}")

    mean_auc = float(np.mean(fold_aucs))
    print(f"  → OOF mean={mean_auc:.4f}  folds={[f'{a:.4f}' for a in fold_aucs]}")
    results.append({"exp": exp_name, "oof_auc": mean_auc,
                    "fold_aucs": str([round(a, 4) for a in fold_aucs])})
    return mean_auc


# Baseline: 5-model mean, no clustering
baseline_aucs = [macro_auc(Y_lab[va], base_prob_lab[va])
                 for _, va in gkf.split(X_lab_norm, groups=groups)]
b_mean = float(np.mean(baseline_aucs))
print(f"\n  [baseline_5model_mean] {[f'{a:.4f}' for a in baseline_aucs]}  → {b_mean:.4f}")
results.append({"exp": "baseline_5model_mean", "oof_auc": b_mean, "fold_aucs": str(baseline_aucs)})

# A1: Scores (1170), Cosine, no pseudo
cv_run("A1_scores_cosine_nopseudo",
       feat_fn=lambda X, E: X,
       use_pseudo=False, min_cs=8, blend_alpha=0.4)

# A2: Perch Embedding (1536), Cosine, no pseudo
cv_run("A2_emb_cosine_nopseudo",
       feat_fn=lambda X, E: E,
       use_pseudo=False, min_cs=8, blend_alpha=0.4)

# A3: Emb+Scores (2706), Score-weighted distance, no pseudo
cv_run("A3_emb_scores_scoreweighted_nopseudo",
       feat_fn=lambda X, E: build_emb_score_feat(X, E, emb_w=0.5),
       dist_fn=lambda F: score_weighted_dist(F[:, 1536:] / 0.5, beta=2.0),
       use_pseudo=False, min_cs=8, blend_alpha=0.4,
       use_precomputed=True)

# B1: Scores (1170), Cosine, + 5K pseudo anchors
cv_run("B1_scores_cosine_pseudo5k",
       feat_fn=lambda X, E: X,
       use_pseudo=True, min_cs=10, blend_alpha=0.4)

# B2: Emb+Scores (2706), Score-weighted, + 5K pseudo anchors
cv_run("B2_emb_scores_scoreweighted_pseudo5k",
       feat_fn=lambda X, E: build_emb_score_feat(X, E, emb_w=0.5),
       dist_fn=lambda F: score_weighted_dist(F[:, 1536:] / 0.5, beta=2.0),
       use_pseudo=True, min_cs=10, blend_alpha=0.4,
       use_precomputed=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  Cluster Stacker — OOF Summary")
print("="*65)
print(f"  {'Experiment':<45} {'OOF AUC':>8}")
print("-"*65)
for r in sorted(results, key=lambda x: -x["oof_auc"]):
    print(f"  {r['exp']:<45} {r['oof_auc']:>8.4f}")
print("="*65)


# ═══════════════════════════════════════════════════════════════════════════════
# [6/6] Fit final models on ALL labeled data + save
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Fitting final models and saving artifacts …")

exp_configs = [
    ("A1", lambda X, E: X,                            None,
     False, False, 8),
    ("A2", lambda X, E: E,                            None,
     False, False, 8),
    ("A3", lambda X, E: build_emb_score_feat(X,E,.5),
     lambda F: score_weighted_dist(F[:,1536:]/.5, 2.), True, False, 8),
    ("B1", lambda X, E: X,                            None,
     False, True, 10),
    ("B2", lambda X, E: build_emb_score_feat(X,E,.5),
     lambda F: score_weighted_dist(F[:,1536:]/.5, 2.), True, True, 10),
]

oof_map = {r["exp"].split("_")[0]: r["oof_auc"] for r in results if r["exp"] != "baseline_5model_mean"}

for tag, feat_fn, dist_fn, use_pre, use_ps, min_cs in exp_configs:
    auc = oof_map.get(tag, 0.0)
    print(f"\n  Fitting final {tag}  (OOF={auc:.4f}) …")

    X_all_feat = feat_fn(X_lab_norm, emb_lab_norm)
    Y_all      = Y_lab.copy()

    if use_ps:
        X_ps = feat_fn(X_pseudo_norm, emb_pseudo_norm)
        X_all_feat = np.vstack([X_all_feat, X_ps])
        Y_all      = np.vstack([Y_all, Y_pseudo_sub])

    D_all = dist_fn(X_all_feat) if use_pre and dist_fn else None

    cs_final = ClusterStacker(min_cluster_size=min_cs, blend_alpha=0.4,
                               use_precomputed=use_pre)
    cs_final.fit(X_all_feat, Y_all, D_pre=D_all)

    out_path = OUT_DIR / f"stacker_cluster_{tag.lower()}_auc{auc:.4f}.pkl"
    with open(out_path, "wb") as fh:
        import pickle as _pk
        _pk.dump(cs_final, fh)
    print(f"  Saved → {out_path}")

# Excel
df = pd.DataFrame(results)
excel_path = OUT_DIR / "stacker_results_cluster.xlsx"
df.to_excel(excel_path, index=False)
print(f"\nExcel saved → {excel_path}")

# W&B
if USE_WANDB:
    import wandb as _w
    _w.log({r["exp"]: r["oof_auc"] for r in results})
    _w.finish()

print("\n[done] Cluster stacker training complete.")
