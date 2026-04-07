"""
Create v14-rknn notebooks based on v9-logspace template.
Method: Reciprocal KNN (Mutual KNN) + Window KNN blend
Formula: sigmoid(a * vlom_logit + b * log(wg*rknn_k5 + (1-wg)*win_k1))
Applied AFTER VLOM blend.
Best config: wg=0.40, a=0.95, b=1.70, full-pipeline CV=0.9430
"""
import json, os

BASE = "/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/current_subs"
SRC  = f"{BASE}/dual-foundation-protossm-v9-logspace.ipynb"
SRC_IMP = f"{BASE}/dual-foundation-protossm-v9-logspace-improve.ipynb"

CONFIGS = [
    # name,            wg,   a,    b,    cv_auc
    ('rknn-wg040-a095-b170', 0.40, 0.95, 1.70, '0.9430'),
    ('rknn-wg040-a090-b155', 0.40, 0.90, 1.55, '0.9430'),
    ('rknn-wg035-a090-b140', 0.35, 0.90, 1.40, '0.9429'),
    ('rknn-wg035-a080-b130', 0.35, 0.80, 1.30, '0.9428'),
    ('rknn-wg030-a090-b140', 0.30, 0.90, 1.40, '0.9427'),
]

NEW_CELL = '''# Score Fusion: ProtoSSM v2 + MLP Probes + Priors (OOF-optimized weight)

# --- Step 1: ProtoSSM v2 inference on test ---
emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)

test_site_ids, test_hours = get_file_metadata(meta_test, test_file_list, site_to_idx, CFG["proto_ssm"]["n_sites"])

emb_test_tensor = torch.tensor(emb_test_files, dtype=torch.float32)
logits_test_tensor = torch.tensor(logits_test_files, dtype=torch.float32)
test_site_tensor = torch.tensor(test_site_ids, dtype=torch.long)
test_hour_tensor = torch.tensor(test_hours, dtype=torch.long)

USE_TEMPORAL_TTA = True
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
        proto_scores = proto_out.numpy()

proto_scores_flat = proto_scores.reshape(-1, N_CLASSES).astype(np.float32)
print(f"ProtoSSM v2 test scores: {proto_scores_flat.shape}")

# --- Step 2: Prior-fused base scores ---
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw,
    sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)

# --- Step 2b: Reciprocal KNN + Win1 Embed Prior (full-pipeline CV=__CV_AUC__) ---
# Formula (applied AFTER VLOM blend):
#   rknn_k5: Mutual/Reciprocal KNN — only count neighbors that also consider us a neighbor
#   win_k1:  Window-level KNN in raw 1536-dim Perch space
#   sigmoid(a × vlom_logit + b × log(__WG__ × rknn_k5 + __WW__ × win_k1))
_RKNN_A = __A_COEF__
_RKNN_B = __B_COEF__
_RKNN_WG = __WG__   # weight for reciprocal KNN
_RKNN_WW = __WW__   # weight for window KNN

def _build_geo_features_rknn(ep, test_emb, meta_df):
    """Build X_combined_n for test rows (same as geo-KNN)."""
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
    return (X_combined / norms).astype(np.float32)

def _rknn_embed_prior(ep, test_emb, meta_df, k=5, T=0.2):
    """Reciprocal KNN: only count neighbors that also consider us a top-k neighbor."""
    X_te = _build_geo_features_rknn(ep, test_emb, meta_df)
    X_ref = ep['X_combined_n']        # (66, 39) training files
    file_labels = ep['file_labels']   # (66, 234)
    n_rows = len(X_te); n_cls = file_labels.shape[1]; n_train = len(X_ref); EPS = 1e-7
    # Precompute pairwise similarities among training files
    sim_train = X_ref @ X_ref.T  # (66, 66)
    np.fill_diagonal(sim_train, -np.inf)
    top_k_train = np.argsort(-sim_train, axis=1)[:, :k]  # (66, k) top-k for each training file
    out = np.zeros((n_rows, n_cls), np.float32)
    BSZ = 256
    for s in range(0, n_rows, BSZ):
        Xb = X_te[s:s+BSZ]; nb = len(Xb)
        # sim of test batch to all training files
        sims_te = Xb @ X_ref.T  # (nb, 66)
        for bi in range(nb):
            sims_i = sims_te[bi]
            # top-k training neighbors for this test row
            top_i = np.argsort(-sims_i)[:k]
            # Reciprocal check: for each candidate training file tj,
            # check if this test row would be in tj's top-k
            # (i.e., test_sim[tj, this_row] > tj's k-th nearest training neighbor sim)
            mutual = []; mutual_sims = []
            for tj in top_i:
                # tj's k-th nearest training neighbor sim (threshold for reciprocal)
                kth_sim = sim_train[tj, top_k_train[tj, -1]]
                # test row is reciprocal if its similarity to tj >= tj's k-th sim
                if sims_i[tj] >= kth_sim:
                    mutual.append(tj); mutual_sims.append(sims_i[tj])
            if len(mutual) == 0:
                # Fallback to standard attn-KNN
                top = top_i[:5]; ls = sims_i[top]/T; ls -= ls.max()
                w = np.exp(ls); w /= w.sum()
                out[s+bi] = (w[:,None] * file_labels[top]).sum(0)
            else:
                ma = np.array(mutual); ms = np.array(mutual_sims)
                ls = ms/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
                out[s+bi] = (w[:,None] * file_labels[ma]).sum(0)
    return out.clip(EPS, 1-EPS)

def _win_knn_rknn(ep, test_emb, k=1):
    """Window-KNN k=1 in raw 1536-dim L2-normalized Perch space."""
    emb_ref = ep.get('emb_win_norm', None)
    if emb_ref is None: return None
    wfi = ep['win_file_id']; fl = ep['file_labels']
    n_cls = fl.shape[1]; X_te = test_emb.astype(np.float32)
    nrm = np.linalg.norm(X_te, 1, keepdims=True); nrm[nrm<1e-8]=1.0; X_te=X_te/nrm
    X_ref = emb_ref.astype(np.float32); n_te = X_te.shape[0]
    out = np.zeros((n_te, n_cls), np.float32); BSZ = 512
    for s in range(0, n_te, BSZ):
        Xb = X_te[s:s+BSZ]; sims = Xb @ X_ref.T
        top = np.argsort(-sims, 1)[:, :k]
        for bi in range(len(Xb)):
            fids = wfi[top[bi]]; Ynn = fl[fids]
            w = sims[bi, top[bi]].clip(0); ws = w.sum()
            w = w/ws if ws>1e-8 else np.ones(k)/k
            out[s+bi] = (w[:,None]*Ynn).sum(0)
    return out.clip(1e-6, 1-1e-6)

import pickle as _pickle, pathlib as _pl
_ep_rknn_path = _pl.Path("/kaggle/input/birdclef-embed-prior/embed_prior_rknn_k5_win1.pkl")
if not _ep_rknn_path.exists():
    _ep_rknn_path = _pl.Path("outputs/embed_prior_rknn_k5_win1.pkl")

_RKNN_ACTIVE = _ep_rknn_path.exists()
if _RKNN_ACTIVE:
    with open(_ep_rknn_path, "rb") as _f:
        _ep_rknn = _pickle.load(_f)
    _y_rknn = _rknn_embed_prior(_ep_rknn, emb_test, meta_test, k=5, T=0.2)
    _y_win_rknn = _win_knn_rknn(_ep_rknn, emb_test, k=1)
    if _y_win_rknn is not None:
        _y_blend_rknn = _RKNN_WG * _y_rknn + _RKNN_WW * _y_win_rknn
    else:
        _y_blend_rknn = _y_rknn
        print("  RKNN: win-KNN skipped, using rknn-only")
    print(f"Reciprocal KNN blend ready: rknn={_RKNN_WG:.2f} win={_RKNN_WW:.2f}")
else:
    _RKNN_ACTIVE = False
    print("RKNN pkl not found, SKIPPED")

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

# --- Step 4: Ensemble fusion ---
print(f"\\nUsing OOF-optimized ensemble weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")
final_test_scores = (
    ENSEMBLE_WEIGHT_PROTO * proto_scores_flat +
    (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
).astype(np.float32)

# --- Step 5: Residual SSM correction ---
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
    final_test_scores = final_test_scores + CORRECTION_WEIGHT * test_correction_flat
    print(f"Residual correction applied.")
else:
    print("Residual correction: SKIPPED")

# --- Logging ---
test_logs = {}
window_scores = proto_scores.mean(axis=(0, 2))
test_logs["window_position_scores"] = window_scores.tolist()
print(f"Window position mean scores: {[f'{s:.3f}' for s in window_scores]}")
if hasattr(model, 'class_to_family'):
    taxon_scores = defaultdict(list)
    idx_to_fam = {v: k for k, v in fam_to_idx.items()}
    for ci in range(N_CLASSES):
        fam_idx = class_to_family[ci]
        fam_name = idx_to_fam.get(fam_idx, f"group_{fam_idx}")
        taxon_scores[fam_name].append(float(proto_scores_flat[:, ci].mean()))
    test_logs["taxon_mean_scores"] = {k: float(np.mean(v)) for k, v in taxon_scores.items()}
with torch.no_grad():
    p_norm = F.normalize(model.prototypes, dim=-1)
    cos_sim = torch.matmul(p_norm, p_norm.T); cos_sim.fill_diagonal_(0)
    top_sims = cos_sim.max(dim=1)[0].numpy()
    test_logs["prototype_max_similarity"] = {"mean": float(top_sims.mean()), "max": float(top_sims.max()), "min": float(top_sims.min())}
    print(f"Prototype nearest-neighbor similarity: mean={top_sims.mean():.3f}")
LOGS["test_inference"] = test_logs

# ── VLOM blend: ProtoSSM + SED ─────────────────────────────────────────────────
def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

TEMP_SCALE_PROTO = 1.10

if USE_SED and sed_preds_all is not None:
    proto_probs = _sigmoid_np(final_test_scores / TEMP_SCALE_PROTO)
    final_blended = vlom_blend(proto_probs, sed_preds_all, w_a=PERCH_PROTO_W, w_b=SED_W)
    print(f"VLOM blend (ProtoSSM x{PERCH_PROTO_W} + SED x{SED_W}): range [{final_blended.min():.3f}, {final_blended.max():.3f}]")
    final_test_scores_blended = final_blended
else:
    final_test_scores_blended = _sigmoid_np(final_test_scores / TEMP_SCALE_PROTO)
    print("SED blend SKIPPED.")

# ── Reciprocal KNN correction (applied AFTER VLOM blend) ─────────────────────
# Method: Reciprocal (Mutual) KNN k=5 + Window KNN k=1
# Only counts neighbors that mutually consider each other as top-k neighbors
# Full-pipeline CV AUC: __CV_AUC__
if _RKNN_ACTIVE:
    EPS_R = 1e-7
    _vlom_logit = np.log(final_test_scores_blended.clip(EPS_R)) - np.log((1-final_test_scores_blended).clip(EPS_R))
    _log_blend = np.log(_y_blend_rknn.clip(EPS_R))
    final_test_scores_blended = _sigmoid_np(_RKNN_A * _vlom_logit + _RKNN_B * _log_blend)
    print(f"RKNN applied (a={_RKNN_A}, b={_RKNN_B}): range [{final_test_scores_blended.min():.3f}, {final_test_scores_blended.max():.3f}]")

print(f"Final blended scores: {final_test_scores_blended.shape}")'''


def create_rknn_notebook(src_path, dst_path, wg, a_coef, b_coef, cv_auc):
    with open(src_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    ww = round(1.0 - wg, 2)
    cell_src = NEW_CELL
    cell_src = cell_src.replace('__A_COEF__', str(a_coef))
    cell_src = cell_src.replace('__B_COEF__', str(b_coef))
    cell_src = cell_src.replace('__WG__', str(wg))
    cell_src = cell_src.replace('__WW__', str(ww))
    cell_src = cell_src.replace('__CV_AUC__', cv_auc)
    replaced = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code': continue
        src = ''.join(cell['source'])
        if 'ProtoSSM v2 + MLP Probes + Priors' in src:
            nb['cells'][i]['source'] = [cell_src]
            replaced = True
            print(f"  Replaced Cell {i}")
            break
    if not replaced:
        print(f"  WARNING: cell not found in {os.path.basename(src_path)}")
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Created: {os.path.basename(dst_path)}")


for name, wg, a, b, cv_auc in CONFIGS:
    print(f"\nCreating {name}...")
    dst = f"{BASE}/dual-foundation-protossm-{name}.ipynb"
    create_rknn_notebook(SRC, dst, wg, a, b, cv_auc)
    if os.path.exists(SRC_IMP):
        dst_imp = f"{BASE}/dual-foundation-protossm-{name}-improve.ipynb"
        create_rknn_notebook(SRC_IMP, dst_imp, wg, a, b, cv_auc)

print("\ndone")
