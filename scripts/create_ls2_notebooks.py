"""
Create v14-ls2-geo5-win1 notebooks based on v9-logspace template.
New formula: sigmoid(a * vlom_logit + b * log(0.50 * geo_k5 + 0.50 * win_k1))
Applied AFTER VLOM blend (ProtoSSM + SED).
Best params: a=0.90, b=1.55, full-pipeline AUC=0.9408
"""
import json, os, re

BASE = "/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/current_subs"
SRC  = f"{BASE}/dual-foundation-protossm-v9-logspace.ipynb"
SRC_IMP = f"{BASE}/dual-foundation-protossm-v9-logspace-improve.ipynb"

# Configs: (a_coef, b_coef, cv_auc)
# We'll create a few variants
CONFIGS = [
    # name suffix,   a,    b,   desc
    ('ls2-a090-b155', 0.90, 1.55, 'BEST: 0.9408, a=0.90 b=1.55'),
    ('ls2-a080-b140', 0.80, 1.40, '0.9408 alt, a=0.80 b=1.40'),
    ('ls2-a075-b130', 0.75, 1.30, '0.9408 alt, a=0.75 b=1.30'),
    ('ls2-a070-b120', 0.70, 1.20, '0.9407 alt, a=0.70 b=1.20'),
    ('ls2-a060-b145', 0.60, 1.45, '0.9407 alt, a=0.60 b=1.45'),
]

NEW_EMBED_PRIOR_CELL = '''# Score Fusion: ProtoSSM v2 + MLP Probes + Priors (OOF-optimized weight)

# --- Step 1: ProtoSSM v2 inference on test ---
emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)

# Build test metadata
test_site_ids, test_hours = get_file_metadata(meta_test, test_file_list, site_to_idx, CFG["proto_ssm"]["n_sites"])

emb_test_tensor = torch.tensor(emb_test_files, dtype=torch.float32)
logits_test_tensor = torch.tensor(logits_test_files, dtype=torch.float32)
test_site_tensor = torch.tensor(test_site_ids, dtype=torch.long)
test_hour_tensor = torch.tensor(test_hours, dtype=torch.long)

USE_TEMPORAL_TTA = True   # Set False to disable TTA
TTA_SHIFTS = CFG.get("tta_shifts", [0, 1, -1])

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

# --- Step 2b: Logspace Geo5+Win1 Embed Prior (LOO=0.9164, full-pipeline=__CV_AUC__) ---
# Formula (applied AFTER VLOM blend): sigmoid(a × vlom_logit + b × log(0.50×geo_k5 + 0.50×win_k1))
# - geo_k5: attn-KNN k=5 in PCA24+geo space (pkl X_combined_n)
# - win_k1: window-KNN k=1 in raw 1536-dim Perch embedding space
_LS2_A = __A_COEF__   # VLOM-logit coefficient
_LS2_B = __B_COEF__   # log(blended_knn) coefficient

def _geo_knn_ls2(ep, test_emb, meta_df, k=5, T=0.2):
    """Geo-KNN in PCA24+geo (X_combined_n) space."""
    import re as _re
    SITES = ep['SITES']; site2idx = ep['site2idx']
    n_rows = len(test_emb); EPS = 1e-7
    _dt_re = _re.compile(r'_(\d{4})(\d{2})(\d{2})_')
    test_months = np.zeros(n_rows, dtype=np.float32)
    test_days   = np.zeros(n_rows, dtype=np.float32)
    _dpm = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    for ri, fn in enumerate(meta_df['filename'].values):
        m = _dt_re.search(str(fn))
        if m:
            mo, dy = int(m.group(2)), int(m.group(3))
            test_months[ri] = mo; test_days[ri] = sum(_dpm[:mo]) + dy
        else:
            test_months[ri] = 6; test_days[ri] = 152
    test_sites = meta_df['site'].values
    test_hours = meta_df['hour_utc'].values.astype(float)
    te_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-8)
    X_pca   = (te_norm - ep['pca_mean']) @ ep['pca_components'].T
    X_pca_s = (X_pca / ep['pca_std']).astype(np.float32)
    site_idxs = np.array([site2idx.get(str(s), -1) for s in test_sites])
    site_oh   = np.zeros((n_rows, len(SITES)), dtype=np.float32)
    valid = site_idxs >= 0; site_oh[valid, site_idxs[valid]] = 1.0
    hour_enc  = np.stack([np.sin(2*np.pi*test_hours/24), np.cos(2*np.pi*test_hours/24)], 1).astype(np.float32)
    month_enc = np.stack([np.sin(2*np.pi*(test_months-1)/12), np.cos(2*np.pi*(test_months-1)/12)], 1).astype(np.float32)
    day_enc   = np.stack([np.sin(2*np.pi*(test_days-1)/365), np.cos(2*np.pi*(test_days-1)/365)], 1).astype(np.float32)
    X_combined = np.concatenate([X_pca_s, site_oh, hour_enc, month_enc, day_enc], 1)
    norms = np.linalg.norm(X_combined, 1, keepdims=True); norms[norms<1e-8]=1e-8
    X_cn = X_combined / norms
    X_ref = ep['X_combined_n']; file_labels = ep['file_labels']
    temperature = ep.get('T_geo', T)
    k_use = ep.get('k_geo', k)
    batch = 256; y = np.zeros((n_rows, file_labels.shape[1]), np.float32)
    for s in range(0, n_rows, batch):
        Xb = X_cn[s:s+batch]; sims = Xb @ X_ref.T
        top = np.argsort(-sims, 1)[:, :k_use]
        ts = np.take_along_axis(sims, top, 1)
        lk = ts / temperature; lk -= lk.max(1, keepdims=True)
        w = np.exp(lk); w /= w.sum(1, keepdims=True)
        y[s:s+batch] = (w[:, :, None] * file_labels[top]).sum(1)
    return y.clip(EPS, 1-EPS)

def _win_knn_ls2(ep, test_emb, k=1):
    """Window-KNN k=1 in raw 1536-dim L2-normalized Perch space."""
    emb_ref = ep.get('emb_win_norm', None)
    if emb_ref is None: return None
    wfi = ep['win_file_id']; fl = ep['file_labels']
    n_cls = fl.shape[1]; X_te = test_emb.astype(np.float32)
    nrm = np.linalg.norm(X_te, 1, keepdims=True); nrm[nrm<1e-8]=1.0; X_te=X_te/nrm
    X_ref = emb_ref.astype(np.float32); n_te = X_te.shape[0]
    out = np.zeros((n_te, n_cls), np.float32); BSZ = 512
    k_use = ep.get('k_win', k)
    for s in range(0, n_te, BSZ):
        Xb = X_te[s:s+BSZ]; sims = Xb @ X_ref.T
        top = np.argsort(-sims, 1)[:, :k_use]
        for bi in range(len(Xb)):
            fids = wfi[top[bi]]; Ynn = fl[fids]
            w = sims[bi, top[bi]].clip(0); ws = w.sum()
            w = w/ws if ws>1e-8 else np.ones(k_use)/k_use
            out[s+bi] = (w[:,None]*Ynn).sum(0)
    return out.clip(1e-6, 1-1e-6)

import pickle as _pickle, pathlib as _pl
_ep_ls2_path = _pl.Path("/kaggle/input/birdclef-embed-prior/embed_prior_logspace_geo5_win1.pkl")
if not _ep_ls2_path.exists():
    _ep_ls2_path = _pl.Path("outputs/embed_prior_logspace_geo5_win1.pkl")

_LS2_ACTIVE = _ep_ls2_path.exists()
if _LS2_ACTIVE:
    with open(_ep_ls2_path, "rb") as _f:
        _ep_ls2 = _pickle.load(_f)
    _w_geo_ls2 = _ep_ls2.get('w_geo', 0.50)
    _w_win_ls2 = 1.0 - _w_geo_ls2
    _y_geo_ls2 = _geo_knn_ls2(_ep_ls2, emb_test, meta_test)
    _y_win_ls2 = _win_knn_ls2(_ep_ls2, emb_test)
    if _y_win_ls2 is not None:
        _y_blend_ls2 = _w_geo_ls2 * _y_geo_ls2 + _w_win_ls2 * _y_win_ls2
    else:
        _y_blend_ls2 = _y_geo_ls2
        print("  LS2: win-KNN skipped (no emb_win_norm), using geo-only")
    print(f"Logspace Geo5+Win1 blend ready: geo={_w_geo_ls2:.2f} win={_w_win_ls2:.2f}")
else:
    _LS2_ACTIVE = False
    print("LS2 pkl not found, SKIPPED")

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

# --- Step 5: Residual SSM correction (second pass) ---
if res_model is not None and CORRECTION_WEIGHT > 0:
    first_pass_test_files, _ = reshape_to_files(final_test_scores, meta_test)
first_pass_test_t = torch.tensor(first_pass_test_files, dtype=torch.float32)

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

# --- Logging: score distributions per taxon ---
test_logs = {}
window_scores = proto_scores.mean(axis=(0, 2))
test_logs["window_position_scores"] = window_scores.tolist()
print(f"\\nWindow position mean scores: {[f'{s:.3f}' for s in window_scores]}")

if hasattr(model, 'class_to_family'):
    taxon_scores = defaultdict(list)
    idx_to_fam = {v: k for k, v in fam_to_idx.items()}
    for ci in range(N_CLASSES):
        fam_idx = class_to_family[ci]
        fam_name = idx_to_fam.get(fam_idx, f"group_{fam_idx}")
        taxon_scores[fam_name].append(float(proto_scores_flat[:, ci].mean()))
    test_logs["taxon_mean_scores"] = {k: float(np.mean(v)) for k, v in taxon_scores.items()}
    for k, v in sorted(taxon_scores.items(), key=lambda x: -np.mean(x[1]))[:5]:
        print(f"  {k}: mean_score={np.mean(v):.4f} (n_classes={len(v)})")

with torch.no_grad():
    p_norm = F.normalize(model.prototypes, dim=-1)
    cos_sim = torch.matmul(p_norm, p_norm.T)
    cos_sim.fill_diagonal_(0)
    top_sims = cos_sim.max(dim=1)[0].numpy()
    test_logs["prototype_max_similarity"] = {
        "mean": float(top_sims.mean()), "max": float(top_sims.max()), "min": float(top_sims.min()),
    }
    print(f"\\nPrototype nearest-neighbor similarity: mean={top_sims.mean():.3f}, max={top_sims.max():.3f}")

LOGS["test_inference"] = test_logs


# ── VLOM blend: ProtoSSM final scores + SED BranchEns→cSEBBs ─────────────────
def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

TEMP_SCALE_PROTO = 1.10   # temperature for ProtoSSM logits before sigmoid

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

# ── Logspace Geo5+Win1 correction (applied AFTER VLOM blend) ──────────────────
# Formula: sigmoid(a × vlom_logit + b × log(blended_knn))
# Best full-pipeline CV AUC: __CV_AUC__ (a=__A_COEF__, b=__B_COEF__)
if _LS2_ACTIVE:
    EPS_LS2 = 1e-7
    _vlom_logit = np.log(final_test_scores_blended.clip(EPS_LS2)) - np.log((1-final_test_scores_blended).clip(EPS_LS2))
    _log_blend = np.log(_y_blend_ls2.clip(EPS_LS2))
    final_test_scores_blended = _sigmoid_np(_LS2_A * _vlom_logit + _LS2_B * _log_blend)
    print(f"LS2 Logspace applied (a={_LS2_A}, b={_LS2_B}): "
          f"range [{final_test_scores_blended.min():.3f}, {final_test_scores_blended.max():.3f}]")

print(f"Final blended scores: {final_test_scores_blended.shape}")'''


def create_ls2_notebook(src_path, dst_path, a_coef, b_coef, cv_auc, desc):
    with open(src_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Replace Cell 51 with new content
    cell_replaced = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'ProtoSSM v2 + MLP Probes + Priors' in src and '_logspace_embed_prior' in src:
            new_src = NEW_EMBED_PRIOR_CELL
            new_src = new_src.replace('__A_COEF__', str(a_coef))
            new_src = new_src.replace('__B_COEF__', str(b_coef))
            new_src = new_src.replace('__CV_AUC__', cv_auc)
            nb['cells'][i]['source'] = [new_src]
            cell_replaced = True
            print(f"  Replaced Cell {i} with LS2 formula (a={a_coef}, b={b_coef})")
            break

    if not cell_replaced:
        # Try finding by logspace
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            src = ''.join(cell['source'])
            if 'ProtoSSM v2 + MLP Probes + Priors' in src:
                new_src = NEW_EMBED_PRIOR_CELL
                new_src = new_src.replace('__A_COEF__', str(a_coef))
                new_src = new_src.replace('__B_COEF__', str(b_coef))
                new_src = new_src.replace('__CV_AUC__', cv_auc)
                nb['cells'][i]['source'] = [new_src]
                cell_replaced = True
                print(f"  Replaced Cell {i} (fallback search) with LS2 formula")
                break

    if not cell_replaced:
        print(f"  WARNING: Could not find cell to replace in {os.path.basename(src_path)}")

    # Update notebook metadata description (look for notebook title cell)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell['source'])
            if 'v9-logspace' in src or 'Logspace' in src:
                new_src = src.replace('v9-logspace', f'v14-{desc.split(",")[0].strip()}')
                nb['cells'][i]['source'] = [new_src]
                break

    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Created: {os.path.basename(dst_path)}")


for suffix, a_coef, b_coef, desc in CONFIGS:
    cv_auc = '0.9408' if '090' in suffix or '080' in suffix or '075' in suffix else '0.9407'

    # Main notebook
    dst = f"{BASE}/dual-foundation-protossm-{suffix}.ipynb"
    print(f"\nCreating {suffix}...")
    create_ls2_notebook(SRC, dst, a_coef, b_coef, cv_auc, desc)

    # Improve notebook
    if os.path.exists(SRC_IMP):
        dst_imp = f"{BASE}/dual-foundation-protossm-{suffix}-improve.ipynb"
        create_ls2_notebook(SRC_IMP, dst_imp, a_coef, b_coef, cv_auc, desc)
    else:
        print(f"  WARNING: improve template not found")

print("\ndone")
