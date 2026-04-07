"""Plan C: Soundscape Embedding Prior

Trains P(species | perch_embedding) on 66 labeled soundscapes.
Uses file-level labels (OR of window labels) as training target.

Two models compared:
  1. KNN  — nearest-neighbour retrieval in embedding space (no training)
  2. PCA + LogReg — dimensionality reduction + multi-label linear model

Output:
  outputs/embed_prior_model.pkl   — sklearn pipeline (PCA + LogReg)
  outputs/embed_prior_eval.txt    — LOO-CV AUC report
  outputs/embed_prior_preds.npz   — LOO-CV soft predictions for 66 files

Usage:
  python3 scripts/train_embed_prior.py
  python3 scripts/train_embed_prior.py --alpha 0.3 --eval_only
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

NPZ_PATH   = "outputs/perch_labeled_ss.npz"
OUT_MODEL  = "outputs/embed_prior_model.pkl"
OUT_EVAL   = "outputs/embed_prior_eval.txt"
OUT_PREDS  = "outputs/embed_prior_preds.npz"

PCA_DIM    = 64
LR_C       = 0.05      # strong regularisation (66 samples, 234 classes)
LR_ITER    = 2000


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_meta(filename: str) -> dict:
    """BC2026_Train_0001_S08_20250606_030007.ogg → site/hour/month."""
    m = re.match(r"BC2026_Train_\d+_S(\d+)_(\d{4})(\d{2})\d{2}_(\d{2})", filename)
    if not m:
        return {"site": "00", "hour": 0, "month": 6}
    site, _, month, hour = m.groups()
    return {"site": site, "hour": int(hour), "month": int(month)}


def cyclic(val: float, period: float) -> tuple:
    """Encode cyclic feature as (sin, cos)."""
    rad = 2 * np.pi * val / period
    return np.sin(rad), np.cos(rad)


def build_features(mean_embs: np.ndarray, metas: list[dict],
                   site_list: list[str]) -> np.ndarray:
    """Concatenate embedding + site one-hot + hour/month cyclic."""
    site2idx = {s: i for i, s in enumerate(site_list)}
    n = len(mean_embs)
    n_sites = len(site_list)
    feats = []
    for i in range(n):
        emb = mean_embs[i]                             # (1536,)
        meta = metas[i]
        # site one-hot (9-dim)
        site_oh = np.zeros(n_sites, dtype=np.float32)
        site_oh[site2idx.get(meta["site"], 0)] = 1.0
        # hour cyclic (2-dim, period=24)
        h_sin, h_cos = cyclic(meta["hour"], 24)
        # month cyclic (2-dim, period=12)
        m_sin, m_cos = cyclic(meta["month"], 12)
        extra = np.array([h_sin, h_cos, m_sin, m_cos], dtype=np.float32)
        feats.append(np.concatenate([emb, site_oh, extra]))
    return np.stack(feats)   # (n, 1536 + n_sites + 4)


def auc_present(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro AUC over species present in at least one true label."""
    mask = y_true.sum(0) > 0
    if mask.sum() < 2:
        return float("nan")
    try:
        return roc_auc_score(y_true[:, mask], y_score[:, mask], average="macro")
    except Exception:
        return float("nan")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    d = np.load(NPZ_PATH, allow_pickle=True)
    emb       = d["emb"]          # (739, 1536)
    labels    = d["labels"]       # (739, 234)
    filenames = d["filenames"]    # (739,)   per-window filename
    file_list = d["file_list"]    # (66,)    unique files
    n_windows = d["n_windows"]    # (66,)

    # Build file-level mean embeddings and OR labels
    file_embs   = np.zeros((len(file_list), emb.shape[1]),   dtype=np.float32)
    file_labels = np.zeros((len(file_list), labels.shape[1]), dtype=np.float32)
    file_metas  = []

    idx = 0
    for fi, (fname, nw) in enumerate(zip(file_list, n_windows)):
        ws = emb[idx: idx + nw]
        lb = labels[idx: idx + nw]
        file_embs[fi]   = ws.mean(0)
        file_labels[fi] = (lb.max(0) > 0.5).astype(np.float32)
        file_metas.append(parse_meta(fname))
        idx += nw

    site_list = sorted({m["site"] for m in file_metas})
    X = build_features(file_embs, file_metas, site_list)
    Y = file_labels   # (66, 234)

    return X, Y, file_list, file_embs, site_list


# ── Models ────────────────────────────────────────────────────────────────────

def make_logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=PCA_DIM, random_state=42)),
        ("clf",    OneVsRestClassifier(
            LogisticRegression(C=LR_C, max_iter=LR_ITER,
                               solver="lbfgs", random_state=42),
            n_jobs=-1,
        )),
    ])


def knn_predict(train_emb: np.ndarray, train_labels: np.ndarray,
                test_emb: np.ndarray, k: int = 5) -> np.ndarray:
    """Weighted KNN in embedding space; weights = cosine similarity."""
    # L2-normalise for cosine similarity via euclidean distance
    tr = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-8)
    te = test_emb  / (np.linalg.norm(test_emb,  axis=1, keepdims=True) + 1e-8)
    nn = NearestNeighbors(n_neighbors=min(k, len(tr)), metric="euclidean")
    nn.fit(tr)
    dists, idxs = nn.kneighbors(te)
    # similarity = 1 - dist/2 (cosine)
    sims = np.clip(1.0 - dists / 2.0, 0, 1)
    preds = np.zeros((len(te), train_labels.shape[1]), dtype=np.float32)
    for i in range(len(te)):
        w = sims[i]
        if w.sum() < 1e-8:
            w = np.ones_like(w)
        preds[i] = (w[:, None] * train_labels[idxs[i]]).sum(0) / w.sum()
    return preds


# ── LOO-CV evaluation ─────────────────────────────────────────────────────────

def loo_cv(X: np.ndarray, Y: np.ndarray, file_embs: np.ndarray):
    n = len(X)
    loo_logreg = np.zeros_like(Y, dtype=np.float32)
    loo_knn    = np.zeros_like(Y, dtype=np.float32)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, Y_tr = X[mask], Y[mask]
        X_te       = X[[i]]
        emb_tr     = file_embs[mask]
        emb_te     = file_embs[[i]]

        # LogReg — only train on classes with >=1 positive in train fold
        valid = Y_tr.sum(0) > 0
        pipe = make_logreg_pipeline()
        pipe.fit(X_tr, Y_tr[:, valid])
        # predict_proba returns (n_samples, n_valid_classes)
        prob = pipe.predict_proba(X_te)   # (1, n_valid)
        probs_full = np.zeros((1, Y.shape[1]), dtype=np.float32)
        probs_full[0, valid] = prob[0]
        loo_logreg[i] = probs_full[0]

        # KNN (raw embeddings, k=5)
        loo_knn[i] = knn_predict(emb_tr, Y_tr, emb_te, k=5)[0]

        if (i + 1) % 10 == 0:
            print(f"  LOO {i+1}/{n} done")

    return loo_logreg, loo_knn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Prior blend weight for demo fusion")
    args = p.parse_args()

    print("Loading data ...")
    X, Y, file_list, file_embs, site_list = load_data()
    print(f"  X={X.shape}  Y={Y.shape}  sites={site_list}")
    print(f"  species present in >=1 file: {(Y.sum(0)>0).sum()}")

    # ── LOO-CV ────────────────────────────────────────────────────────────────
    print("\nRunning LOO-CV ...")
    loo_lr, loo_knn = loo_cv(X, Y, file_embs)

    auc_lr  = auc_present(Y, loo_lr)
    auc_knn = auc_present(Y, loo_knn)

    # Uniform baseline (no prior info)
    baseline_pred = np.full_like(Y, Y.mean(0)[None, :], dtype=np.float32)
    auc_base = auc_present(Y, baseline_pred)

    lines = [
        "=== Plan C: Embedding Prior — LOO-CV Results ===",
        f"  n_files={len(file_list)}  n_species_present={int((Y.sum(0)>0).sum())}",
        f"  Baseline (prevalence)   AUC: {auc_base:.4f}",
        f"  KNN (k=5, cosine)       AUC: {auc_knn:.4f}  Δ={auc_knn-auc_base:+.4f}",
        f"  PCA({PCA_DIM})+LogReg   AUC: {auc_lr:.4f}  Δ={auc_lr-auc_base:+.4f}",
        "",
        "Best model: " + ("LogReg" if auc_lr >= auc_knn else "KNN"),
    ]
    report = "\n".join(lines)
    print("\n" + report)

    Path(OUT_EVAL).write_text(report + "\n")
    np.savez(OUT_PREDS, loo_logreg=loo_lr, loo_knn=loo_knn, labels=Y,
             file_list=file_list)
    print(f"\nEval saved → {OUT_EVAL}")

    if args.eval_only:
        return

    # ── Train final model on all data ─────────────────────────────────────────
    print("\nTraining final model on all 66 files ...")
    pipe = make_logreg_pipeline()
    pipe.fit(X, Y)

    model_data = {
        "pipeline":   pipe,
        "site_list":  site_list,
        "pca_dim":    PCA_DIM,
        "lr_C":       LR_C,
        "file_list":  file_list,
        "file_labels": Y,
        "file_embs":  file_embs,
        "best_model": "logreg" if auc_lr >= auc_knn else "knn",
        "loo_auc":    {"logreg": auc_lr, "knn": auc_knn, "baseline": auc_base},
    }
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model saved → {OUT_MODEL}")

    # ── Inference demo ────────────────────────────────────────────────────────
    print(f"\nDemo: predict prior for first 3 files (alpha={args.alpha})")
    demo_X = X[:3]
    # Final model trained on all data, predict_proba → (n_samples, 234)
    probs_arr = pipe.predict_proba(demo_X).astype(np.float32)
    for i in range(3):
        top5 = np.argsort(probs_arr[i])[::-1][:5]
        print(f"  {file_list[i]}")
        print(f"    top-5 predicted species idx: {top5.tolist()}")
        print(f"    true species idx: {np.where(Y[i]>0)[0].tolist()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
