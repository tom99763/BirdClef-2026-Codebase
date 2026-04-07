"""
Create v6-kde notebook from SED-species bridge template.
Formula: sigmoid(0.95 * vlom_logit + 1.2 * log(0.30*kde + 0.70*win_k1))
LOO-AUC: 0.9560 (proper LOO-PCA validated)
"""
import json, copy, os
os.chdir("/home/lab/BirdClef-2026-Codebase")

SRC_NB = "birdclef-2026/notebook resource/current_subs/dual-foundation-protossm-sed-species-bridge-b050-a050-wg045-a085-b170.ipynb"
DST_NB = "birdclef-2026/notebook resource/current_subs/dual-foundation-protossm-v6-kde.ipynb"

with open(SRC_NB) as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Source: {len(cells)} cells")

# Find the main inference cell (cell 51 in SED-bridge)
target_cell_idx = None
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    if '_SSBRIDGE_ALPHA' in src and '_ss_bridge_embed_prior' in src:
        target_cell_idx = i
        break

if target_cell_idx is None:
    print("ERROR: Could not find SED-bridge inference cell!")
    exit(1)

print(f"Found inference cell at index {target_cell_idx}")

# ── Build the new cell content ─────────────────────────────────────────────────
KDE_CELL = '''# Score Fusion: ProtoSSM v2 + MLP Probes + KDE Embed Prior (OOF-optimized weight)
# KDE Embed Prior: kde_per_species, LOO-AUC=0.9560 (proper LOO-PCA validated)
# Formula: sigmoid(0.95 × vlom_logit + 1.2 × log(0.30 × kde + 0.70 × win_k1))
# Reference: SS-Bridge best=0.9444, KDE=+0.0116 improvement

_KDE_A = 0.95    # base logit coefficient
_KDE_B = 1.2     # embed prior log coefficient
_KDE_WG = 0.30   # KDE weight (0.70 goes to win_k1)

def _kde_log_prob_np(X_test, X_train, bw):
    """Pure-numpy Gaussian KDE log probability.
    X_test:  (n_test, d)
    X_train: (n_train, d)
    Returns: (n_test,) log-probability under the KDE
    """
    d = X_train.shape[1]
    # Squared Euclidean distances: ||x - xi||^2 = ||x||^2 + ||xi||^2 - 2*x^T xi
    sq_test  = (X_test**2).sum(1, keepdims=True)    # (n_test, 1)
    sq_train = (X_train**2).sum(1)                  # (n_train,)
    cross    = X_test @ X_train.T                   # (n_test, n_train)
    dists_sq = sq_test + sq_train - 2 * cross       # (n_test, n_train)
    dists_sq = np.maximum(dists_sq, 0)
    log_dens = -0.5 * dists_sq / (bw**2)            # unnormalized log density
    # logsumexp for numerical stability
    max_ld   = log_dens.max(1, keepdims=True)
    log_sum  = np.log(np.exp(log_dens - max_ld).sum(1)) + max_ld[:, 0]
    log_norm = np.log(len(X_train)) + 0.5 * d * np.log(2 * np.pi * (bw**2))
    return log_sum - log_norm   # (n_test,)


def _kde_embed_prior(ep, test_emb_file, file_row_ids):
    """KDE per-species embed prior.

    Args:
        ep:             pkl dict with KDE model data
        test_emb_file:  dict mapping file_id -> (n_windows, 1536) Perch embeddings
        file_row_ids:   dict mapping file_id -> list of row indices in output

    Returns:
        out: (n_rows, 234) per-row KDE predictions (file-level value broadcast to all rows)
    """
    EPS = 1e-7
    # PCA transform params
    pca_comp  = ep['pca_components']    # (32, 1536)
    pca_mr    = ep['pca_mean_raw']      # (1536,)
    pca_mean  = ep['pca_mean']          # (32,)
    pca_std   = ep['pca_std']           # (32,)
    bw        = ep['kde_bandwidth']     # float
    X_bg      = ep['kde_bg_train_X']    # (66, 32) standardized bg training features
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
        # Average windows to file-level embedding
        avg_emb = emb_rows.mean(0)                         # (1536,)
        avg_norm = avg_emb / (np.linalg.norm(avg_emb) + EPS)
        # PCA-32 transform
        x_pca = ((avg_norm - pca_mr) @ pca_comp.T - pca_mean) / pca_std  # (32,)
        x_pca = x_pca[None, :]                             # (1, 32)
        # Background KDE score
        log_bg = _kde_log_prob_np(x_pca, X_bg, bw)[0]    # scalar
        # Per-species KDE score
        kde_scores = np.zeros(n_cls, np.float32)
        for si in range(n_cls):
            if si in sp_pos:
                log_pos = _kde_log_prob_np(x_pca, sp_pos[si], bw)[0]
                kde_scores[si] = sigmoid_np(log_pos - log_bg)
            else:
                # No positive training files → use fallback (mean of training logit)
                kde_scores[si] = sigmoid_np(fl_logmax[:, si].mean())
        # Broadcast to all rows for this file
        for ri in rids:
            out[ri] = kde_scores
    return out


def _win_k1_prior_flat(ep, test_emb_flat):
    """Window KNN k=1: compare each test window to training windows.

    Args:
        ep:             pkl dict
        test_emb_flat:  (n_rows, 1536) Perch embeddings (one per soundscape row)
    Returns:
        (n_rows, 234) per-row win_k1 predictions
    """
    train_win_norm = ep['emb_win_norm']     # (739, 1536)
    win_file_id    = ep['win_file_id']      # (739,)
    file_labels    = ep['file_labels']      # (66, 234)
    EPS = 1e-7

    te_norm = test_emb_flat / (np.linalg.norm(test_emb_flat, axis=1, keepdims=True) + EPS)
    n_rows = len(te_norm); n_cls = file_labels.shape[1]
    out = np.zeros((n_rows, n_cls), np.float32)

    BSZ = 256
    for s in range(0, n_rows, BSZ):
        Xb = te_norm[s:s+BSZ]
        sims = Xb @ train_win_norm.T        # (nb, 739)
        top1 = np.argmax(sims, axis=1)      # (nb,)
        for bi in range(len(Xb)):
            fi = win_file_id[top1[bi]]
            w  = max(sims[bi, top1[bi]], 0)
            out[s+bi] = file_labels[fi] * w
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
    print(f"[KDE EmbedPrior] Loaded pkl: method={_ep_model.get('method','?')}, loo_auc={_ep_model.get('loo_auc',0):.4f}")
else:
    print("[KDE EmbedPrior] WARNING: pkl not found, skipping")


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

# --- Step 2b: KDE Embed Prior (kde_per_species, LOO-AUC=0.9560) ─────────────
_RKNN_ACTIVE = False   # will be set True if model loaded successfully
_y_blend_rknn = None

if _ep_model is not None:
    # Build file-level embedding map for KDE
    test_file_emb_map = {}
    test_file_row_map = {}
    for ri, fname in enumerate(meta_test['filename'].values):
        # file id = just use row in file list
        pass

    # Rebuild: group rows by unique file (soundscape filename without chunk suffix)
    import re as _re_kde
    _fn_base_pat = _re_kde.compile(r'^(.+?)(?:_chunk\d+|_\d+)?\.(?:ogg|flac|wav)$')
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

    # Convert to numpy arrays
    for fid in _fid_to_embs:
        _fid_to_embs[fid] = np.stack(_fid_to_embs[fid], axis=0)  # (n_win, 1536)

    n_rows_total = len(meta_test)

    # Compute KDE prior (file-level, broadcast to rows)
    y_kde = _kde_embed_prior(_ep_model, _fid_to_embs, _fid_to_rows)
    # Resize if needed
    if len(y_kde) < n_rows_total:
        _y_kde_full = np.zeros((n_rows_total, N_CLASSES), np.float32)
        _y_kde_full[:len(y_kde)] = y_kde
        y_kde = _y_kde_full
    y_kde = y_kde[:n_rows_total]
    print(f"KDE prior: {y_kde.shape}, mean={y_kde.mean():.4f}", flush=True)

    # Compute win_k1 prior (row-level)
    y_win_kde = _win_k1_prior_flat(_ep_model, emb_test)
    print(f"Win K1 shape: {y_win_kde.shape}, mean={y_win_kde.mean():.4f}")

    # Blend KDE + win_k1
    EPS_e = 1e-7
    y_ep_blended = _KDE_WG * y_kde + (1 - _KDE_WG) * y_win_kde  # (n, 234)
    log_ep = np.log(y_ep_blended.clip(EPS_e))

    # Activate post-VLOM logspace correction
    _RKNN_ACTIVE = True
    _RKNN_A = _KDE_A
    _RKNN_B = _KDE_B
    _y_blend_rknn = y_ep_blended
    print(f"KDE embed prior computed. Activating post-VLOM correction (a={_KDE_A}, b={_KDE_B})")
else:
    print("[KDE EmbedPrior] Skipped (model not found)")

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

# ── Post-VLOM KDE logspace correction ────────────────────────────────────────
# Full-pipeline CV AUC (LOO-validated with proper LOO-PCA): 0.9560
if _RKNN_ACTIVE:
    EPS_R = 1e-7
    _vlom_logit = np.log(final_test_scores_blended.clip(EPS_R)) - np.log((1-final_test_scores_blended).clip(EPS_R))
    _log_blend = np.log(_y_blend_rknn.clip(EPS_R))
    final_test_scores_blended = _sigmoid_np(_RKNN_A * _vlom_logit + _RKNN_B * _log_blend)
    print(f"KDE correction applied (a={_RKNN_A}, b={_RKNN_B}): range [{final_test_scores_blended.min():.3f}, {final_test_scores_blended.max():.3f}]")

print(f"Final blended scores: {final_test_scores_blended.shape}")
'''

# ── Create new notebook ───────────────────────────────────────────────────────
nb_new = copy.deepcopy(nb)
cells_new = nb_new['cells']

# Replace the target cell
old_src = ''.join(cells_new[target_cell_idx].get('source', []))
cells_new[target_cell_idx]['source'] = KDE_CELL.splitlines(keepends=True)

# Also update any markdown cells that describe the method
for i, c in enumerate(cells_new):
    if c.get('cell_type') == 'markdown':
        src = ''.join(c.get('source', []))
        if 'SED-Species Bridge' in src or 'sed-species-bridge' in src.lower():
            new_src = src.replace('SED-Species Bridge', 'KDE Embed Prior')
            new_src = new_src.replace('sed-species-bridge', 'kde-embed-prior')
            new_src = new_src.replace('sed_species_bridge', 'kde_per_species')
            new_src = new_src.replace('0.9444', '0.9560')
            cells_new[i]['source'] = new_src.splitlines(keepends=True)

with open(DST_NB, 'w') as f:
    json.dump(nb_new, f, indent=1)

print(f"\nCreated: {DST_NB}")
print(f"  - Source cell {target_cell_idx} replaced with KDE inference")
print(f"  - LOO-AUC: 0.9560 (proper LOO-PCA validated)")
print(f"  - Formula: sigmoid(0.95*vlom + 1.2*log(0.30*kde + 0.70*win_k1))")
