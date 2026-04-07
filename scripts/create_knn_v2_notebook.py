"""
Build pantanal-distill-birdclef2026-knn-v2.ipynb
Changes vs improvement.ipynb:
  1. CFG["embed_knn"] config block
  2. New cell after Cell 39: build/save KNN reference pkl (train) or load it (both)
  3. Cell 48: enhanced Step 2b (CFG lambda, OOF AUC diagnostic)
  4. Cell 51: KNN contribution log
"""
import json, copy
from pathlib import Path

SRC = Path("birdclef-2026/notebook resource/current_subs 2/clustering/pantanal-distill-birdclef2026-improvement.ipynb")
DST = Path("birdclef-2026/notebook resource/current_subs 2/clustering/pantanal-distill-birdclef2026-knn-v2.ipynb")

with open(SRC) as f:
    nb = json.load(f)

cells = nb["cells"]


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src,
    }


# ── 1. Modify Cell 6: add CFG["embed_knn"] ────────────────────────────────────
KNN_CFG_ADDITION = '''
# ── KNN Embed Prior Config ──
CFG["embed_knn"] = {
    "enabled": True,
    "k": 5,
    "lambda": 0.25,
    "pkl_train_path": "/kaggle/working/knn_reference.pkl",
    "pkl_submit_path": "/kaggle/input/datasets/tom99763/birdclef2026-claude/weights_with_prior/knn_reference.pkl",
}
print("✅ KNN embed prior config loaded")
'''
src6 = "".join(cells[6]["source"])
cells[6]["source"] = src6 + KNN_CFG_ADDITION


# ── 2. Insert new cell after Cell 39 (index 39): build/load KNN reference ─────
KNN_BUILD_CODE = '''\
# ── KNN Reference: Build (train) / Load (submit) ────────────────────────────
# PKL format: {"file_embs": (n_files, 1536), "file_labels": (n_files, 234), "file_list": [...]}
import pickle as _pkl

KNN_REFERENCE = None
_knn_cfg = CFG.get("embed_knn", {})

if _knn_cfg.get("enabled", False):
    if MODE == "train":
        # Build per-file mean embeddings + OR labels from all labeled soundscapes
        _file_groups = meta_full["filename"].values        # (n_windows,)
        _uniq_files  = sorted(set(_file_groups))
        _n_files = len(_uniq_files)

        _file_embs   = np.zeros((_n_files, emb_full.shape[1]), dtype=np.float32)
        _file_labels = np.zeros((_n_files, Y_FULL.shape[1]),   dtype=np.float32)

        for fi, fn in enumerate(_uniq_files):
            mask = _file_groups == fn
            _file_embs[fi]   = emb_full[mask].mean(axis=0)
            # OR of binary presence across windows in this file
            _file_labels[fi] = (Y_FULL[mask] > 0.5).any(axis=0).astype(np.float32)

        KNN_REFERENCE = {
            "file_embs":   _file_embs,
            "file_labels": _file_labels,
            "file_list":   _uniq_files,
        }

        _pkl_path = Path(_knn_cfg["pkl_train_path"])
        _pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_pkl_path, "wb") as _f:
            _pkl.dump(KNN_REFERENCE, _f)

        print(f"✅ KNN reference built & saved → {_pkl_path}")
        print(f"   file_embs  : {_file_embs.shape}")
        print(f"   file_labels: {_file_labels.shape}")
        print(f"   n_files    : {_n_files}")

        # ── OOF KNN contribution diagnostic ──────────────────────────────────
        # Leave-one-file-out cosine KNN: for each file predict using other files
        from sklearn.metrics import roc_auc_score

        def _cosine_knn(tr_emb, tr_labels, te_emb, k=5):
            tr_n = tr_emb / (np.linalg.norm(tr_emb, axis=1, keepdims=True) + 1e-8)
            te_n = te_emb / (np.linalg.norm(te_emb, axis=1, keepdims=True) + 1e-8)
            sims = te_n @ tr_n.T
            topk = np.argsort(-sims, axis=1)[:, :k]
            w    = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
            w    = w / (w.sum(1, keepdims=True) + 1e-8)
            return (w[:, :, None] * tr_labels[topk]).sum(1).astype(np.float32)

        # Window-level LOO AUC (each window leaves its file out)
        oof_knn = np.zeros_like(emb_full[:, :Y_FULL.shape[1]])
        _file_idx_map = {fn: i for i, fn in enumerate(_uniq_files)}

        for fi, fn in enumerate(_uniq_files):
            win_mask   = _file_groups == fn                         # test windows
            other_mask = np.array([_file_idx_map[g] != fi for g in _file_groups])
            # build reference from other files' mean embeddings
            oth_files  = [j for j in range(_n_files) if j != fi]
            tr_emb     = _file_embs[oth_files]
            tr_labels  = _file_labels[oth_files]
            te_emb     = emb_full[win_mask]
            oof_knn[win_mask] = _cosine_knn(tr_emb, tr_labels, te_emb,
                                            k=min(_knn_cfg["k"], len(oth_files)))

        # Only evaluate species with ≥1 positive window
        valid = Y_FULL.sum(0) > 0
        knn_oof_auc = roc_auc_score(Y_FULL[:, valid], oof_knn[:, valid],
                                     average="macro")
        print(f"\\n  KNN LOO OOF AUC (window-level): {knn_oof_auc:.4f}")
        LOGS["knn_oof_auc"] = knn_oof_auc

    else:  # submit
        _pkl_paths = [
            Path(_knn_cfg.get("pkl_train_path", "")),
            Path(_knn_cfg.get("pkl_submit_path", "")),
        ]
        for _p in _pkl_paths:
            if _p.exists():
                with open(_p, "rb") as _f:
                    KNN_REFERENCE = _pkl.load(_f)
                print(f"✅ KNN reference loaded from {_p}")
                print(f"   file_embs  : {KNN_REFERENCE['file_embs'].shape}")
                print(f"   file_labels: {KNN_REFERENCE['file_labels'].shape}")
                break
        if KNN_REFERENCE is None:
            print("⚠️  KNN reference pkl not found — KNN will be skipped")
else:
    print("KNN embed prior: disabled")
'''

knn_build_cell = code_cell(KNN_BUILD_CODE)
knn_build_md   = md_cell("## KNN Embed Prior\nBuild per-file cosine-KNN reference from labeled soundscapes (train) or load pkl (submit).")

# Insert after Cell 39 (index 39)
cells.insert(40, knn_build_cell)
cells.insert(40, knn_build_md)
# Now Cell 48 becomes Cell 50 (shifted by 2)


# ── 3. Modify Cell 50 (was 48): enhanced Step 2b ─────────────────────────────
STEP2B_OLD = '''\
# --- Step 2: Prior-fused base scores ---
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw,
    sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)'''

STEP2B_NEW = '''\
# --- Step 2: Prior-fused base scores ---
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw,
    sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)

# --- Step 2b: KNN Embed Prior ---
_knn_applied = False
if CFG.get("embed_knn", {}).get("enabled", False) and KNN_REFERENCE is not None:
    _lam = CFG["embed_knn"]["lambda"]
    _k   = CFG["embed_knn"]["k"]

    def _cosine_knn_scores(tr_emb, tr_labels, te_emb, k):
        tr_n = tr_emb / (np.linalg.norm(tr_emb, axis=1, keepdims=True) + 1e-8)
        te_n = te_emb / (np.linalg.norm(te_emb, axis=1, keepdims=True) + 1e-8)
        sims = te_n @ tr_n.T
        topk = np.argsort(-sims, axis=1)[:, :k]
        w    = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
        w    = w / (w.sum(1, keepdims=True) + 1e-8)
        return (w[:, :, None] * tr_labels[topk]).sum(1).astype(np.float32)

    _knn_probs  = _cosine_knn_scores(
        KNN_REFERENCE["file_embs"],
        KNN_REFERENCE["file_labels"],
        emb_test, k=_k,
    )
    _knn_logits = np.log(_knn_probs + 1e-6) - np.log(1.0 - _knn_probs + 1e-6)

    test_base_scores = test_base_scores + _lam * _knn_logits
    _knn_applied = True

    print(f"KNN embed prior applied: k={_k}, lambda={_lam}")
    print(f"  mean_abs_logit={np.abs(_knn_logits).mean():.3f}, "
          f"max_logit={np.abs(_knn_logits).max():.3f}")
    print(f"  n_train_files={len(KNN_REFERENCE['file_embs'])}")
    LOGS["knn_applied"]         = True
    LOGS["knn_lambda"]          = _lam
    LOGS["knn_k"]               = _k
    LOGS["knn_mean_abs_logit"]  = float(np.abs(_knn_logits).mean())
else:
    print("KNN embed prior: SKIPPED")
    LOGS["knn_applied"] = False'''

# Find and replace in cell 50
cell_50 = cells[50]
src_50 = "".join(cell_50["source"])
if STEP2B_OLD in src_50:
    cells[50]["source"] = src_50.replace(STEP2B_OLD, STEP2B_NEW)
    print("✅ Patched Cell 50 Step 2b")
else:
    # Try the original improvement notebook's Step 2 (no Step 2b)
    ORIG_STEP2 = '''\
# --- Step 2: Prior-fused base scores ---
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw,
    sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)'''
    if ORIG_STEP2 in src_50:
        cells[50]["source"] = src_50.replace(ORIG_STEP2, STEP2B_NEW)
        print("✅ Patched Cell 50 Step 2b (from improvement base)")
    else:
        print(f"⚠️  Step 2 pattern not found in Cell 50!")
        print(f"Cell 50 starts with: {src_50[:200]}")


# ── 4. Modify Cell 53 (was 51): add KNN diagnostics ──────────────────────────
KNN_DIAG = '''
# ── KNN Contribution Diagnostics ──────────────────────────────
if LOGS.get("knn_applied"):
    print(f"\\nKNN Embed Prior Contribution:")
    print(f"  k={LOGS['knn_k']}, lambda={LOGS['knn_lambda']}")
    print(f"  mean_abs_logit={LOGS['knn_mean_abs_logit']:.4f}")
    if "knn_oof_auc" in LOGS:
        print(f"  LOO OOF AUC (train mode): {LOGS['knn_oof_auc']:.4f}")
'''
cells[53]["source"] = "".join(cells[53]["source"]) + KNN_DIAG


# ── Save ──────────────────────────────────────────────────────────────────────
nb["cells"] = cells
with open(DST, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Saved → {DST}")
print(f"   Total cells: {len(nb['cells'])}")
