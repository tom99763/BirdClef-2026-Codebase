"""
Create v6-kde-rknn notebook: KDE-Window + RKNN k5 blend
Formula: sigmoid(0.92 * vlom_logit + 1.4 * log(0.35*kde_win + 0.65*rknn_k5))
LOO-AUC: 0.9711 (proper LOO-window PCA validated)
Based on: dual-foundation-protossm-v6-kde-win.ipynb
"""
import json, copy, os
os.chdir("/home/lab/BirdClef-2026-Codebase")

SRC_NB = "birdclef-2026/notebook resource/current_subs/dual-foundation-protossm-v6-kde-win.ipynb"
DST_NB = "birdclef-2026/notebook resource/current_subs/dual-foundation-protossm-v6-kde-rknn.ipynb"

with open(SRC_NB) as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Source: {len(cells)} cells")

# Find the main inference cell (cell 51 in kde-win)
target_cell_idx = None
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    if '_KDE_A' in src and '_kde_embed_prior' in src and '_win_k1_prior_flat' in src:
        target_cell_idx = i
        break

if target_cell_idx is None:
    print("ERROR: Could not find KDE inference cell!")
    exit(1)

print(f"Found inference cell at index {target_cell_idx}")

# ── New cell content ──────────────────────────────────────────────────────────
NEW_CELL = '''# Score Fusion: ProtoSSM v2 + MLP Probes + KDE+RKNN Embed Prior (OOF-optimized weight)
# KDE+RKNN Embed Prior: kde_win_rknn_blend, LOO-AUC=0.9711 (proper LOO-window PCA validated)
# Formula: sigmoid(0.92 × vlom_logit + 1.4 × log(0.35 × kde_win + 0.65 × rknn_k5))
# Reference: KDE-win best=0.9701, KDE+RKNN=+0.0010 improvement

_KDE_A    = 0.92    # base logit coefficient
_KDE_B    = 1.4     # embed prior log coefficient
_KDE_WG   = 0.35    # KDE weight
_KDE_WRKNN = 0.65   # RKNN weight (KDE + RKNN = 1.0)

def _kde_log_prob_np(X_test, X_train, bw):
    """Pure-numpy Gaussian KDE log probability.
    X_test:  (n_test, d)
    X_train: (n_train, d)
    Returns: (n_test,) log-probability under the KDE
    """
    d = X_train.shape[1]
    sq_test  = (X_test**2).sum(1, keepdims=True)    # (n_test, 1)
    sq_train = (X_train**2).sum(1)                  # (n_train,)
    cross    = X_test @ X_train.T                   # (n_test, n_train)
    dists_sq = sq_test + sq_train - 2 * cross       # (n_test, n_train)
    dists_sq = np.maximum(dists_sq, 0)
    log_dens = -0.5 * dists_sq / (bw**2)
    max_ld   = log_dens.max(1, keepdims=True)
    log_sum  = np.log(np.exp(log_dens - max_ld).sum(1)) + max_ld[:, 0]
    log_norm = np.log(len(X_train)) + 0.5 * d * np.log(2 * np.pi * (bw**2))
    return log_sum - log_norm   # (n_test,)


def _kde_win_embed_prior(ep, test_emb_file, file_row_ids):
    """Window-level KDE per-species embed prior.

    Args:
        ep:             pkl dict with KDE model data
        test_emb_file:  dict mapping file_id -> (n_windows, 1536) Perch embeddings
        file_row_ids:   dict mapping file_id -> list of row indices in output

    Returns:
        out: (n_rows, 234) per-row KDE predictions (file-level value broadcast to all rows)
    """
    EPS = 1e-7
    pca_comp  = ep['pca_components']    # (32, 1536)
    pca_mr    = ep['pca_mean_raw']      # (1536,)
    pca_mean  = ep['pca_mean']          # (32,)
    pca_std   = ep['pca_std']           # (32,)
    bw        = ep['kde_bandwidth']     # float
    X_bg      = ep['kde_bg_train_X']    # (739, 32) standardized bg training features
    sp_pos    = ep['species_pos_X']     # dict {si: (n_pos, 32)}
    fl_logmax = ep['file_logit_max']    # (66, 234) fallback logits
    def sigmoid_np(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))

    n_cls = fl_logmax.shape[1]
    all_row_ids = []
    for rids in file_row_ids.values():
        all_row_ids.extend(rids)
    n_rows = max(all_row_ids) + 1 if all_row_ids else 0
    out = np.zeros((n_rows, n_cls), np.float32)

    for fid, emb_rows in test_emb_file.items():
        rids = file_row_ids[fid]
        # Average windows, then normalize
        avg_emb = emb_rows.mean(0)                          # (1536,)
        avg_norm = avg_emb / (np.linalg.norm(avg_emb) + EPS)
        # PCA-32 transform (production: fit on all 739 windows)
        x_pca = ((avg_norm - pca_mr) @ pca_comp.T - pca_mean) / pca_std  # (32,)
        x_pca = x_pca[None, :]                              # (1, 32)
        # Background KDE score
        log_bg = _kde_log_prob_np(x_pca, X_bg, bw)[0]
        # Per-species KDE score
        kde_scores = np.zeros(n_cls, np.float32)
        for si in range(n_cls):
            if si in sp_pos:
                log_pos = _kde_log_prob_np(x_pca, sp_pos[si], bw)[0]
                kde_scores[si] = sigmoid_np(log_pos - log_bg)
            else:
                kde_scores[si] = sigmoid_np(fl_logmax[:, si].mean())
        for ri in rids:
            out[ri] = kde_scores
    return out


def _rknn_k5_prior_flat(ep, test_emb_flat):
    """Window RKNN k=5: reciprocal nearest neighbors with fallback softmax.

    Args:
        ep:             pkl dict
        test_emb_flat:  (n_rows, 1536) Perch embeddings (one per soundscape row)
    Returns:
        (n_rows, 234) per-row RKNN predictions
    """
    train_win_norm = ep['emb_win_norm']     # (739, 1536)
    win_file_id    = ep['win_file_id']      # (739,)
    file_labels    = ep['file_labels']      # (66, 234)
    K_RKNN = ep['config'].get('k_rknn', 5)
    EPS = 1e-7

    # Precompute training-side k-th similarity threshold for reciprocity check
    sims_tr_tr = train_win_norm @ train_win_norm.T  # (739, 739)
    np.fill_diagonal(sims_tr_tr, -np.inf)           # exclude self
    thresh = np.partition(-sims_tr_tr, K_RKNN, axis=1)[:, K_RKNN] * -1  # (739,)

    te_norm = test_emb_flat / (np.linalg.norm(test_emb_flat, axis=1, keepdims=True) + EPS)
    n_rows = len(te_norm); n_cls = file_labels.shape[1]
    out = np.zeros((n_rows, n_cls), np.float32)

    BSZ = 128
    for s in range(0, n_rows, BSZ):
        Xb = te_norm[s:s+BSZ]                           # (nb, 1536)
        sims = Xb @ train_win_norm.T                    # (nb, 739)
        top_k_idx = np.argsort(-sims, axis=1)[:, :K_RKNN]  # (nb, K)
        for bi in range(len(Xb)):
            nbrs = top_k_idx[bi]
            recip = [n for n in nbrs if sims[bi, n] >= thresh[n]]
            if not recip:
                recip = nbrs.tolist()   # fallback: use all k neighbors
            ww = sims[bi, recip].clip(0)
            ws = ww.sum()
            ww = ww / ws if ws > 1e-8 else np.ones(len(recip)) / len(recip)
            fi_lbls = file_labels[win_file_id[np.array(recip)]]  # (len(recip), 234)
            out[s + bi] = (ww[:, None] * fi_lbls).sum(0)
    return out


# ─── Load PKL ──────────────────────────────────────────────────────────────────
import pickle, pathlib
_ep_path = pathlib.Path("/kaggle/input/birdclef-embed-prior/embed_prior_model.pkl")
if not _ep_path.exists():
    _ep_path = pathlib.Path("outputs/embed_prior_model.pkl")

_ep_model = None
if _ep_path.exists():
    with open(_ep_path, "rb") as f:
        _ep_model = pickle.load(f)
    print(f"[KDE+RKNN EmbedPrior] Loaded pkl: method={_ep_model.get(\'method\',\'?\')}, loo_auc={_ep_model.get(\'loo_auc\',0):.4f}")
else:
    print("[KDE+RKNN EmbedPrior] WARNING: pkl not found, skipping")


# --- Step 1: ProtoSSM v2 inference on test ---
emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)

# Build test metadata
test_site_ids, test_hours = get_file_metadata(meta_test, test_file_list, site_to_idx, CFG["proto_ssm"]["n_sites"])

emb_test_tensor = torch.tensor(emb_test_files, dtype=torch.float32)
logits_test_tensor = torch.tensor(logits_test_files, dtype=torch.float32)
test_site_tensor = torch.tensor(test_site_ids, dtype=torch.long)
test_hour_tensor = torch.tensor(test_hours, dtype=torch.long)

USE_TEMPORAL_TTA = True  # Set False to disable
TTA_SHIFTS = CFG.get("tta_shifts", [0])

if USE_TEMPORAL_TTA and len(TTA_SHIFTS) > 1:
    proto_scores = temporal_shift_tta(
        emb_test_tensor, logits_test_tensor, model,
        test_site_tensor, test_hour_tensor,
        shifts=TTA_SHIFTS
    )
    print(f"Temporal TTA applied with shifts={TTA_SHIFTS}")
else:
    model.eval()
    with torch.no_grad():
        proto_out, _, h_test = model(
            emb_test_tensor, logits_test_tensor,
            site_ids=test_site_tensor, hours=test_hour_tensor
        )
        proto_scores = proto_out.numpy()  # (n_files, n_windows, n_classes)

# Flatten back to (n_rows, n_classes)
proto_scores_flat = proto_scores.reshape(-1, N_CLASSES).astype(np.float32)

print(f"ProtoSSM v2 test scores: {proto_scores_flat.shape}")
print(f"Score range: {proto_scores_flat.min():.3f} to {proto_scores_flat.max():.3f}")

# --- Step 2: Prior-fused base scores ---
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw,
    sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)

# --- Step 2b: KDE+RKNN Embed Prior (LOO-AUC=0.9711) ─────────────────────────
_RKNN_ACTIVE = False   # will be set True if model loaded successfully
_y_blend_rknn = None

if _ep_model is not None:
    # Group rows by unique soundscape file
    import re as _re_kde
    _fn_base_pat = _re_kde.compile(r'^(.+?)(?:_chunk\\d+|_\\d+)?\\.(?:ogg|flac|wav)$')
    _fn_to_fid = {}
    _fid_to_rows = {}
    _fid_to_embs = {}
    _next_fid = [0]
    for ri, fname in enumerate(meta_test['filename'].values):
        m = _fn_base_pat.match(str(fname))
        base = m.group(1) if m else str(fname)
        if base not in _fn_to_fid:
            fid = _next_fid[0]; _next_fid[0] += 1
            _fn_to_fid[base] = fid
            _fid_to_rows[fid] = []
            _fid_to_embs[fid] = []
        fid = _fn_to_fid[base]
        _fid_to_rows[fid].append(ri)
        _fid_to_embs[fid].append(emb_test[ri])

    for fid in _fid_to_embs:
        _fid_to_embs[fid] = np.stack(_fid_to_embs[fid], axis=0)  # (n_win, 1536)

    n_rows_total = len(meta_test)

    # Compute window-level KDE prior (file-level, broadcast to rows)
    y_kde = _kde_win_embed_prior(_ep_model, _fid_to_embs, _fid_to_rows)
    if len(y_kde) < n_rows_total:
        _y_kde_full = np.zeros((n_rows_total, N_CLASSES), np.float32)
        _y_kde_full[:len(y_kde)] = y_kde
        y_kde = _y_kde_full
    y_kde = y_kde[:n_rows_total]
    print(f"KDE-Win prior: {y_kde.shape}, mean={y_kde.mean():.4f}", flush=True)

    # Compute RKNN k5 prior (row-level)
    y_rknn = _rknn_k5_prior_flat(_ep_model, emb_test)
    print(f"RKNN k5 prior: {y_rknn.shape}, mean={y_rknn.mean():.4f}", flush=True)

    # Blend KDE + RKNN
    EPS_e = 1e-7
    y_ep_blended = _KDE_WG * y_kde + _KDE_WRKNN * y_rknn  # (n, 234)

    # Activate post-VLOM logspace correction
    _RKNN_ACTIVE = True
    _RKNN_A = _KDE_A
    _RKNN_B = _KDE_B
    _y_blend_rknn = y_ep_blended
    print(f"KDE+RKNN embed prior computed. Activating post-VLOM correction (a={_KDE_A}, b={_KDE_B})")
else:
    print("[KDE+RKNN EmbedPrior] Skipped (model not found)")

# --- Step 3: MLP probe scores ---
emb_test_scaled = emb_scaler.transform(emb_test)
Z_TEST = emb_pca.transform(emb_test_scaled).astype(np.float32)

mlp_scores = test_base_scores.copy()

for cls_idx, clf in probe_models.items():
    X_cls_test = build_class_features(
        Z_TEST,
        raw_col=scores_test_raw[:, cls_idx],
        prior_col=test_prior_scores[:, cls_idx],
        base_col=test_base_scores[:, cls_idx],
    )

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_cls_test)[:, 1].astype(np.float32)
        pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
    else:
        pred = clf.decision_function(X_cls_test).astype(np.float32)

    alpha = float(CFG["frozen_best_probe"]["alpha"])
    mlp_scores[:, cls_idx] = (1.0 - alpha) * test_base_scores[:, cls_idx] + alpha * pred

# --- Step 4: Ensemble fusion with OOF-optimized weight ---
print(f"\\nUsing OOF-optimized ensemble weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")

final_test_scores = (
    ENSEMBLE_WEIGHT_PROTO * proto_scores_flat +
    (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
).astype(np.float32)

# --- Step 5: Residual SSM correction ---
if res_model is not None and CORRECTION_WEIGHT > 0:
    first_pass_test_files, _ = reshape_to_files(final_test_scores, meta_test)
first_pass_test_t = torch.tensor(first_pass_test_files if res_model is not None and CORRECTION_WEIGHT > 0 else np.zeros((1,1,1)), dtype=torch.float32)

if res_model is not None and CORRECTION_WEIGHT > 0:
    first_pass_test_t = torch.tensor(first_pass_test_files, dtype=torch.float32)

    res_model.eval()
    with torch.no_grad():
        test_correction = res_model(
            emb_test_tensor, first_pass_test_t,
            site_ids=test_site_tensor, hours=test_hour_tensor
        ).numpy()

    test_correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)

    print(f"\\nResidual correction: mean_abs={np.abs(test_correction_flat).mean():.4f}, "
          f"max={np.abs(test_correction_flat).max():.4f}")

    final_test_scores = final_test_scores + CORRECTION_WEIGHT * test_correction_flat
    print(f"Final scores (after residual): range [{final_test_scores.min():.3f}, {final_test_scores.max():.3f}]")
else:
    print("\\nResidual correction: SKIPPED")

print(f"Final scores: {final_test_scores.shape}")

# ── VLOM blend: ProtoSSM final scores + SED BranchEns→cSEBBs ─────────────────
def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

TEMP_SCALE_PROTO = 1.10

if USE_SED and sed_preds_all is not None:
    proto_probs = _sigmoid_np(final_test_scores / TEMP_SCALE_PROTO)
    final_blended = vlom_blend(proto_probs, sed_preds_all,
                               w_a=PERCH_PROTO_W, w_b=SED_W)
    print(f"VLOM blend (ProtoSSM x{PERCH_PROTO_W} + SED x{SED_W}): "
          f"range [{final_blended.min():.3f}, {final_blended.max():.3f}]")
    final_test_scores_blended = final_blended
else:
    final_test_scores_blended = _sigmoid_np(final_test_scores / TEMP_SCALE_PROTO)
    print("SED blend SKIPPED — using ProtoSSM-only scores.")

# ── Post-VLOM KDE+RKNN logspace correction ───────────────────────────────────
# Full-pipeline CV AUC (LOO-validated with proper LOO-window PCA): 0.9711
if _RKNN_ACTIVE:
    EPS_R = 1e-7
    _vlom_logit = np.log(final_test_scores_blended.clip(EPS_R)) - np.log((1-final_test_scores_blended).clip(EPS_R))
    _log_blend = np.log(_y_blend_rknn.clip(EPS_R))
    final_test_scores_blended = _sigmoid_np(_RKNN_A * _vlom_logit + _RKNN_B * _log_blend)
    print(f"KDE+RKNN correction applied (a={_RKNN_A}, b={_RKNN_B}): range [{final_test_scores_blended.min():.3f}, {final_test_scores_blended.max():.3f}]")

print(f"Final blended scores: {final_test_scores_blended.shape}")
'''

# Also update the markdown cell (cell 49 in source)
MARKDOWN_CELL = '''## Embed Prior: KDE+RKNN Embed Prior (CV=0.9711)

**Method**: Window-level KDE per species + RKNN k=5 blend
**Formula**: `sigmoid(0.92 × vlom_logit + 1.4 × log(0.35 × kde_win + 0.65 × rknn_k5))`
**Validation**: Proper LOO-window PCA (LOO-AUC = 0.9711 confirmed)
**Improvement**: +0.0010 over KDE-win alone (0.9701), +0.0267 over SS-Bridge (0.9444)
**PKL**: `embed_prior_model.pkl` (5.4 MB), method=`kde_win_rknn_blend`

**Key insight**: RKNN k=5 mutual nearest neighbors provides complementary signal to KDE density
- KDE captures global density distribution (density-based prior)
- RKNN captures local geometric structure (similarity-based prior)
- Optimal blend: 35% KDE + 65% RKNN (pure KDE+RKNN, no win_k1)
'''

# Find and update markdown cell
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    if 'Embed Prior: KDE Embed Prior RKNN' in src or 'kde_per_species' in src:
        cells[i] = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [MARKDOWN_CELL]
        }
        print(f"Updated markdown cell at index {i}")
        break

# Replace inference cell
new_cell = copy.deepcopy(cells[target_cell_idx])
new_cell['source'] = [NEW_CELL]
new_cell['outputs'] = []
new_cell['execution_count'] = None
cells[target_cell_idx] = new_cell
print(f"Replaced inference cell at index {target_cell_idx}")

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"\nCreated: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
