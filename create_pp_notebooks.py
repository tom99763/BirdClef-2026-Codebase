#!/usr/bin/env python3
"""
Create 20 post-processing experiment notebooks for BirdCLEF 2026.
Each is a copy of pantanal-distill-birdclef2026-improvement.ipynb with a new PP method.
"""
import json
import copy
import os

SRC = "/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/new direction/pantanal-distill-birdclef2026-improvement.ipynb"
OUT_DIR = "/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/new direction/"

with open(SRC, "r") as f:
    BASE_NB = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a notebook with modified cells 4, 6, 19, 50, 51
# ─────────────────────────────────────────────────────────────────────────────

def make_notebook(title_md, cfg_extra_lines, helper_code, postproc_code, filename):
    nb = copy.deepcopy(BASE_NB)

    # ── Cell 4: markdown title ────────────────────────────────────────────────
    nb["cells"][4]["source"] = [title_md]

    # ── Cell 6: CFG upgrades (append new params after existing code) ──────────
    orig_cfg = "".join(BASE_NB["cells"][6]["source"])
    nb["cells"][6]["source"] = [orig_cfg + "\n" + cfg_extra_lines]

    # ── Cell 19: helpers (append after existing helpers) ─────────────────────
    orig_helpers = "".join(BASE_NB["cells"][19]["source"])
    nb["cells"][19]["source"] = [orig_helpers + "\n\n" + helper_code]

    # ── Cell 50: full postproc replacement ───────────────────────────────────
    nb["cells"][50]["source"] = [postproc_code]

    # ── Cell 51: fix LOGS["temperature"] ─────────────────────────────────────
    cell51 = "".join(nb["cells"][51]["source"])
    cell51 = cell51.replace(
        'LOGS["temperature"] = CFG["temperature"]',
        'LOGS["temperature"] = {"aves": T_AVES, "texture": T_TEXTURE}'
    )
    nb["cells"][51]["source"] = [cell51]

    out_path = os.path.join(OUT_DIR, filename)
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Saved: {filename}")

# ══════════════════════════════════════════════════════════════════════════════
# COMMON HEADER used in every cell 50
# ══════════════════════════════════════════════════════════════════════════════

COMMON_HEADER = """
# Per-class threshold optimization (OOF only)
PER_CLASS_THRESHOLDS = np.full(N_CLASSES, 0.5, dtype=np.float32)
if MODE == "train" and oof_proto_flat is not None:
    best_thresholds, best_scores = optimize_per_class_thresholds(
        oof_proto_flat, Y_FULL, n_windows=N_WINDOWS, thresholds=CFG["threshold_grid"]
    )
    PER_CLASS_THRESHOLDS = best_thresholds.astype(np.float32)
    print(f"  Thresholds: mean={best_thresholds.mean():.3f}, range=[{best_thresholds.min():.2f},{best_thresholds.max():.2f}]")
else:
    print("Using default thresholds (0.5)")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# Step 1: Per-taxon temperature
temp_cfg = CFG["temperature"]
T_AVES = temp_cfg["aves"]
T_TEXTURE = temp_cfg["texture"]
class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * T_AVES
for ci, label in enumerate(PRIMARY_LABELS):
    if CLASS_NAME_MAP.get(label, "Aves") in TEXTURE_TAXA:
        class_temperatures[ci] = T_TEXTURE
probs = sigmoid(final_test_scores / class_temperatures[None, :])

# Step 2: File-level confidence scaling (top_k=2)
probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=2)
probs = np.clip(probs, 0.0, 1.0)

# Step 3: Rank-aware scaling (power=0.4)
probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=CFG.get("rank_aware_power", 0.4))
probs = np.clip(probs, 0.0, 1.0)
""".lstrip()

COMMON_FOOTER = """
# Final: per-class threshold sharpening
probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)

# Build submission
submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)
assert len(submission) == len(test_paths) * N_WINDOWS
assert not submission.isna().any().any()
submission.to_csv("submission.csv", index=False)
print(f"Saved submission.csv  shape={submission.shape}")
print(f"Score range: {probs.min():.6f} – {probs.max():.6f}, mean={probs.mean():.4f}")
print(submission.iloc[:3, :8])
""".lstrip()

# ══════════════════════════════════════════════════════════════════════════════
# pp-v4: Causal EMA
# ══════════════════════════════════════════════════════════════════════════════
v4_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v4 Causal EMA

## PP-v4 — Causal Exponential Moving Average

Forward-only EMA across 12 windows. Birds tend to keep calling once detected.
Sweeps γ in {0.2, 0.3, 0.5, 0.7, 0.9} (train mode).
"""

v4_cfg = """# PP-v4: Causal EMA config
CFG["causal_ema_gammas"] = [0.2, 0.3, 0.5, 0.7, 0.9]
CFG["causal_ema_gamma"]  = 0.5   # default; overridden by sweep in train mode
print("PP-v4 CFG loaded")"""

v4_helpers = """# PP-v4 helpers

def causal_ema_smooth(probs, n_windows=12, gamma=0.5):
    \"\"\"Forward-only EMA across temporal windows.
    ema[t] = (1-gamma)*ema[t-1] + gamma*probs[t]
    gamma close to 1 => fast tracking; close to 0 => heavy smoothing.
    \"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C).copy()
    for t in range(1, n_windows):
        view[:, t, :] = (1.0 - gamma) * view[:, t - 1, :] + gamma * view[:, t, :]
    return view.reshape(N, C)

print("PP-v4 helpers defined: causal_ema_smooth")"""

v4_postproc = "# Cell 18 — PP-v4: Causal EMA\n\n" + COMMON_HEADER + """
# Step 4: Sweep causal EMA gamma (train) or apply best (submit)
if MODE == "train" and oof_proto_flat is not None:
    from sklearn.metrics import roc_auc_score
    best_gamma = 0.5
    best_auc   = -1.0
    oof_p = sigmoid(oof_proto_flat / class_temperatures[None, :])
    n_files_oof = oof_p.shape[0] // N_WINDOWS
    Y_rep = np.repeat(Y_FULL, N_WINDOWS, axis=0) if Y_FULL.shape[0] == n_files_oof else Y_FULL

    for gamma in CFG["causal_ema_gammas"]:
        p_smooth = causal_ema_smooth(oof_p, n_windows=N_WINDOWS, gamma=gamma)
        p_smooth = np.clip(p_smooth, 0.0, 1.0)
        # aggregate to file level for AUC
        p_file = p_smooth.reshape(n_files_oof, N_WINDOWS, -1).max(axis=1)
        try:
            auc = roc_auc_score(Y_FULL, p_file, average="macro")
        except Exception:
            auc = 0.0
        print(f"  gamma={gamma:.1f}  AUC={auc:.4f}")
        if auc > best_auc:
            best_auc   = auc
            best_gamma = gamma
    print(f"  Best gamma={best_gamma:.1f}  AUC={best_auc:.4f}")
    CFG["causal_ema_gamma"] = best_gamma

gamma = CFG["causal_ema_gamma"]
print(f"Applying causal EMA (gamma={gamma})")
probs = causal_ema_smooth(probs, n_windows=N_WINDOWS, gamma=gamma)
probs = np.clip(probs, 0.0, 1.0)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v5: Median Filter
# ══════════════════════════════════════════════════════════════════════════════
v5_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v5 Median Filter

## PP-v5 — Median Temporal Filter

Median filter with window k across 12 temporal windows. Robust to noise spikes.
Sweeps k in {3, 5, 7} (train mode). Boundary: reflect padding.
"""

v5_cfg = """# PP-v5: Median filter config
CFG["median_filter_ks"] = [3, 5, 7]
CFG["median_filter_k"]  = 3   # default
print("PP-v5 CFG loaded")"""

v5_helpers = """# PP-v5 helpers

def median_filter_temporal(probs, n_windows=12, k=3):
    \"\"\"Per-species median filter across the 12 temporal windows.
    Reflect-pads at boundaries.  k must be odd.
    \"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)
    half = k // 2
    # reflect pad
    left  = view[:, :half, :][:, ::-1, :]
    right = view[:, -half:, :][:, ::-1, :]
    padded = np.concatenate([left, view, right], axis=1)  # (F, n_windows+k-1, C)
    out = np.empty_like(view)
    for t in range(n_windows):
        window_slice = padded[:, t:t + k, :]   # (F, k, C)
        out[:, t, :] = np.median(window_slice, axis=1)
    return out.reshape(N, C)

print("PP-v5 helpers defined: median_filter_temporal")"""

v5_postproc = "# Cell 18 — PP-v5: Median Filter\n\n" + COMMON_HEADER + """
# Step 4: Sweep median filter k (train) or apply best (submit)
if MODE == "train" and oof_proto_flat is not None:
    from sklearn.metrics import roc_auc_score
    best_k   = 3
    best_auc = -1.0
    oof_p = sigmoid(oof_proto_flat / class_temperatures[None, :])
    n_files_oof = oof_p.shape[0] // N_WINDOWS

    for k in CFG["median_filter_ks"]:
        p_smooth = median_filter_temporal(oof_p, n_windows=N_WINDOWS, k=k)
        p_smooth = np.clip(p_smooth, 0.0, 1.0)
        p_file   = p_smooth.reshape(n_files_oof, N_WINDOWS, -1).max(axis=1)
        try:
            auc = roc_auc_score(Y_FULL, p_file, average="macro")
        except Exception:
            auc = 0.0
        print(f"  k={k}  AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_k   = k
    print(f"  Best k={best_k}  AUC={best_auc:.4f}")
    CFG["median_filter_k"] = best_k

k = CFG["median_filter_k"]
print(f"Applying median filter (k={k})")
probs = median_filter_temporal(probs, n_windows=N_WINDOWS, k=k)
probs = np.clip(probs, 0.0, 1.0)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v6: Savitzky-Golay
# ══════════════════════════════════════════════════════════════════════════════
v6_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v6 Savitzky-Golay Smooth

## PP-v6 — Savitzky-Golay Polynomial Smoothing

Fit polynomial of degree d over a window of size w to smooth temporal predictions.
Preserves peak shape better than mean-based smoothing.
Sweeps (window_length, polyorder) in {(5,2),(7,2),(7,3),(9,2),(11,3)}.
"""

v6_cfg = """# PP-v6: Savitzky-Golay config
CFG["savgol_params"] = [(5, 2), (7, 2), (7, 3), (9, 2), (11, 3)]
CFG["savgol_best"]   = (7, 2)
print("PP-v6 CFG loaded")"""

v6_helpers = """# PP-v6 helpers

def savgol_smooth(probs, n_windows=12, window_length=7, polyorder=2):
    \"\"\"Savitzky-Golay filter across temporal windows using numpy polynomial fit.
    Reflect-pads at boundaries. window_length must be odd and > polyorder.
    \"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)
    half = window_length // 2

    # Build SG coefficients via least-squares polynomial fit
    x = np.arange(window_length) - half
    A = np.vstack([x ** i for i in range(polyorder + 1)]).T   # (w, d+1)
    # coefficient for the center point
    coef = np.linalg.pinv(A)[0]   # shape (w,)

    # reflect pad
    left  = view[:, :half, :][:, ::-1, :]
    right = view[:, -half:, :][:, ::-1, :]
    padded = np.concatenate([left, view, right], axis=1)

    out = np.empty_like(view)
    for t in range(n_windows):
        segment = padded[:, t:t + window_length, :]  # (F, w, C)
        out[:, t, :] = np.einsum("fwc,w->fc", segment, coef)

    return np.clip(out, 0.0, 1.0).reshape(N, C)

print("PP-v6 helpers defined: savgol_smooth")"""

v6_postproc = "# Cell 18 — PP-v6: Savitzky-Golay Polynomial Smoothing\n\n" + COMMON_HEADER + """
# Step 4: Sweep SavGol params (train) or apply best (submit)
if MODE == "train" and oof_proto_flat is not None:
    from sklearn.metrics import roc_auc_score
    best_params = (7, 2)
    best_auc    = -1.0
    oof_p = sigmoid(oof_proto_flat / class_temperatures[None, :])
    n_files_oof = oof_p.shape[0] // N_WINDOWS

    for wl, po in CFG["savgol_params"]:
        if wl > N_WINDOWS:
            continue
        p_smooth = savgol_smooth(oof_p, n_windows=N_WINDOWS, window_length=wl, polyorder=po)
        p_file   = p_smooth.reshape(n_files_oof, N_WINDOWS, -1).max(axis=1)
        try:
            auc = roc_auc_score(Y_FULL, p_file, average="macro")
        except Exception:
            auc = 0.0
        print(f"  (w={wl}, d={po})  AUC={auc:.4f}")
        if auc > best_auc:
            best_auc    = auc
            best_params = (wl, po)
    print(f"  Best params={best_params}  AUC={best_auc:.4f}")
    CFG["savgol_best"] = best_params

wl, po = CFG["savgol_best"]
print(f"Applying Savitzky-Golay (window={wl}, polyorder={po})")
probs = savgol_smooth(probs, n_windows=N_WINDOWS, window_length=wl, polyorder=po)
probs = np.clip(probs, 0.0, 1.0)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v7: HMM Temporal Smoothing
# ══════════════════════════════════════════════════════════════════════════════
v7_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v7 HMM Smoothing

## PP-v7 — HMM Temporal Smoothing

2-state HMM (present/absent) per species. Forward-backward posterior decoding.
p_on=0.1, p_off=0.2. Species tends to stay absent; once present, stays.
"""

v7_cfg = """# PP-v7: HMM config
CFG["hmm_p_on"]  = 0.1   # absent->present transition probability
CFG["hmm_p_off"] = 0.2   # present->absent transition probability
print("PP-v7 CFG loaded")"""

v7_helpers = """# PP-v7 helpers

def hmm_forward_backward(probs_seq, p_on=0.1, p_off=0.2, eps=1e-10):
    \"\"\"
    2-state HMM forward-backward for a single species over T windows.
    States: 0=absent, 1=present.
    Emission: Bernoulli(probs_seq[t]).
    Transition: A[0,0]=1-p_on, A[0,1]=p_on, A[1,0]=p_off, A[1,1]=1-p_off.
    Returns posterior P(state=1 | observations).
    \"\"\"
    T = len(probs_seq)
    A = np.array([[1 - p_on, p_on], [p_off, 1 - p_off]])
    # emission probability for each state
    def emit(t, s):
        p = probs_seq[t]
        return (p if s == 1 else (1.0 - p)) + eps

    # forward pass
    alpha = np.zeros((T, 2))
    alpha[0, 0] = (1 - p_on) * emit(0, 0)
    alpha[0, 1] = p_on       * emit(0, 1)
    alpha[0] /= (alpha[0].sum() + eps)
    for t in range(1, T):
        for s in range(2):
            alpha[t, s] = emit(t, s) * np.dot(alpha[t-1], A[:, s])
        alpha[t] /= (alpha[t].sum() + eps)

    # backward pass
    beta = np.ones((T, 2))
    for t in range(T - 2, -1, -1):
        for s in range(2):
            beta[t, s] = np.sum(A[s, :] * np.array([emit(t+1, sp) for sp in range(2)]) * beta[t+1])
        beta[t] /= (beta[t].sum() + eps)

    # posterior
    gamma = alpha * beta
    gamma /= (gamma.sum(axis=1, keepdims=True) + eps)
    return gamma[:, 1]   # P(present | obs)


def hmm_smooth_all(probs, n_windows=12, p_on=0.1, p_off=0.2):
    \"\"\"Apply HMM forward-backward to each (file, species) pair.\"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)
    out  = np.empty_like(view)
    for f in range(n_files):
        for c in range(C):
            out[f, :, c] = hmm_forward_backward(view[f, :, c], p_on=p_on, p_off=p_off)
    return out.reshape(N, C)

print("PP-v7 helpers defined: hmm_forward_backward, hmm_smooth_all")"""

v7_postproc = "# Cell 18 — PP-v7: HMM Temporal Smoothing\n\n" + COMMON_HEADER + """
# Step 4: HMM forward-backward smoothing
p_on  = CFG["hmm_p_on"]
p_off = CFG["hmm_p_off"]
print(f"Applying HMM smoothing (p_on={p_on}, p_off={p_off})")
probs = hmm_smooth_all(probs, n_windows=N_WINDOWS, p_on=p_on, p_off=p_off)
probs = np.clip(probs, 0.0, 1.0)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v8: Co-occurrence Score Propagation
# ══════════════════════════════════════════════════════════════════════════════
v8_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v8 Co-occurrence Propagation

## PP-v8 — Co-occurrence Score Propagation

Build P(b|a) from training labels. If species A is detected, boost correlated
species B proportionally. beta=0.1 boost strength.
"""

v8_cfg = """# PP-v8: Co-occurrence propagation config
CFG["cooc_beta"] = 0.1   # boost strength
print("PP-v8 CFG loaded")"""

v8_helpers = """# PP-v8 helpers

def build_cooc_matrix(Y_full):
    \"\"\"Build conditional co-occurrence matrix P(b present | a present).
    Y_full: (n_files, n_classes) binary.
    Returns (n_classes, n_classes) matrix.
    \"\"\"
    C = Y_full.shape[1]
    cooc = np.zeros((C, C), dtype=np.float32)
    for a in range(C):
        mask = Y_full[:, a] > 0
        if mask.sum() == 0:
            continue
        cooc[a] = Y_full[mask].mean(axis=0)
    # zero diagonal (no self-boost)
    np.fill_diagonal(cooc, 0.0)
    return cooc


def cooc_propagate(probs, cooc_matrix, beta=0.1):
    \"\"\"Boost each species by the weighted sum of co-occurring detected species.
    probs: (N, C), cooc_matrix: (C, C).
    \"\"\"
    boost = probs @ cooc_matrix   # (N, C)
    new_probs = probs + beta * boost
    return np.clip(new_probs, 0.0, 1.0)

print("PP-v8 helpers defined: build_cooc_matrix, cooc_propagate")"""

v8_postproc = "# Cell 18 — PP-v8: Co-occurrence Score Propagation\n\n" + COMMON_HEADER + """
# Step 4: Build co-occurrence matrix (train) and apply propagation
if MODE == "train" and Y_FULL is not None:
    COOC_MATRIX = build_cooc_matrix(Y_FULL)
    print(f"Built co-occurrence matrix: shape={COOC_MATRIX.shape}, mean={COOC_MATRIX.mean():.4f}")
else:
    # submit: uniform co-occurrence (no boost)
    COOC_MATRIX = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float32)
    print("Submit mode: co-occurrence matrix unavailable, using zeros (no boost)")

beta = CFG["cooc_beta"]
print(f"Applying co-occurrence propagation (beta={beta})")
probs = cooc_propagate(probs, COOC_MATRIX, beta=beta)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v9: Taxonomic Family Smoothing
# ══════════════════════════════════════════════════════════════════════════════
v9_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v9 Taxonomic Family Smoothing

## PP-v9 — Taxonomic Family Smoothing

Within each window, blend each species' score with its family's mean score.
new[c] = (1-gamma)*probs[c] + gamma*family_mean[family(c)].
Sweeps gamma in {0.05, 0.10, 0.15, 0.20}.
"""

v9_cfg = """# PP-v9: Taxonomic family smoothing config
CFG["family_smooth_gammas"] = [0.05, 0.10, 0.15, 0.20]
CFG["family_smooth_gamma"]  = 0.10
print("PP-v9 CFG loaded")"""

v9_helpers = """# PP-v9 helpers

def build_family_index(taxonomy, primary_labels):
    \"\"\"Return family_ids (list[int]) parallel to primary_labels.
    Uses 'family' column from taxonomy if available, else 'order'.
    \"\"\"
    lbl2fam = {}
    col = "family" if "family" in taxonomy.columns else ("order" if "order" in taxonomy.columns else None)
    if col:
        for _, row in taxonomy.iterrows():
            lbl = row.get("primary_label") or row.get("ebird_code") or row.get("label")
            if lbl and pd.notna(row.get(col)):
                lbl2fam[lbl] = row[col]
    # assign integer ids
    families = sorted(set(lbl2fam.values()))
    fam2id  = {f: i for i, f in enumerate(families)}
    unk_id  = len(families)
    ids = [fam2id.get(lbl2fam.get(lbl, "UNK"), unk_id) for lbl in primary_labels]
    n_families = unk_id + 1
    return np.array(ids, dtype=np.int32), n_families


def family_smooth(probs, family_ids, n_families, gamma=0.10):
    \"\"\"Blend each species score with its family mean (per window).\"\"\"
    N, C = probs.shape
    out = np.empty_like(probs)
    for fid in range(n_families):
        mask = (family_ids == fid)
        if mask.sum() == 0:
            continue
        fam_mean = probs[:, mask].mean(axis=1, keepdims=True)   # (N,1)
        out[:, mask] = (1.0 - gamma) * probs[:, mask] + gamma * fam_mean
    # handle species with no family
    no_fam = np.ones(n_families, dtype=bool)
    for fid in np.unique(family_ids):
        no_fam[fid] = False
    # fallback: keep as-is for any uncovered species
    out = np.where(np.isnan(out), probs, out)
    return np.clip(out, 0.0, 1.0)

print("PP-v9 helpers defined: build_family_index, family_smooth")"""

v9_postproc = "# Cell 18 — PP-v9: Taxonomic Family Smoothing\n\n" + COMMON_HEADER + """
# Step 4: Build family index and sweep gamma
FAMILY_IDS, N_FAMILIES = build_family_index(taxonomy, PRIMARY_LABELS)
print(f"Family index built: {N_FAMILIES} families for {N_CLASSES} species")

if MODE == "train" and oof_proto_flat is not None:
    from sklearn.metrics import roc_auc_score
    best_gamma = 0.10
    best_auc   = -1.0
    oof_p = sigmoid(oof_proto_flat / class_temperatures[None, :])
    n_files_oof = oof_p.shape[0] // N_WINDOWS

    for gamma in CFG["family_smooth_gammas"]:
        p_smooth = family_smooth(oof_p, FAMILY_IDS, N_FAMILIES, gamma=gamma)
        p_file   = p_smooth.reshape(n_files_oof, N_WINDOWS, -1).max(axis=1)
        try:
            auc = roc_auc_score(Y_FULL, p_file, average="macro")
        except Exception:
            auc = 0.0
        print(f"  gamma={gamma:.2f}  AUC={auc:.4f}")
        if auc > best_auc:
            best_auc   = auc
            best_gamma = gamma
    print(f"  Best gamma={best_gamma:.2f}  AUC={best_auc:.4f}")
    CFG["family_smooth_gamma"] = best_gamma

gamma = CFG["family_smooth_gamma"]
print(f"Applying family smoothing (gamma={gamma})")
probs = family_smooth(probs, FAMILY_IDS, N_FAMILIES, gamma=gamma)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v10: Platt Scaling
# ══════════════════════════════════════════════════════════════════════════════
v10_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v10 Platt Scaling Calibration

## PP-v10 — Platt Scaling (Per-species Logistic Calibration)

Fit LogisticRegression on OOF per species to correct miscalibration.
calibrated[c] = sigmoid(a[c]*logit(probs[c]) + b[c]).
"""

v10_cfg = """# PP-v10: Platt scaling config
CFG["platt_C"] = 1e5   # high C = minimal regularisation
print("PP-v10 CFG loaded")"""

v10_helpers = """# PP-v10 helpers

def fit_platt_scaling(oof_probs, y_true, C=1e5, eps=1e-7):
    \"\"\"Fit per-species logistic calibration on OOF probabilities.
    Returns (a, b) arrays of shape (n_classes,).
    oof_probs: (n_samples, n_classes), y_true: (n_files, n_classes).
    y_true is repeated N_WINDOWS times to match oof_probs rows.
    \"\"\"
    from sklearn.linear_model import LogisticRegression
    n_samples, C_count = oof_probs.shape
    n_files = y_true.shape[0]
    n_windows = n_samples // n_files
    y_rep = np.repeat(y_true, n_windows, axis=0)

    a_arr = np.ones(C_count,  dtype=np.float32)
    b_arr = np.zeros(C_count, dtype=np.float32)

    for c in range(C_count):
        yc = y_rep[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        xc = np.clip(oof_probs[:, c], eps, 1 - eps)
        logit_x = np.log(xc / (1 - xc)).reshape(-1, 1)
        try:
            lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
            lr.fit(logit_x, yc.astype(int))
            a_arr[c] = float(lr.coef_[0, 0])
            b_arr[c] = float(lr.intercept_[0])
        except Exception:
            pass
    return a_arr, b_arr


def apply_platt_scaling(probs, a_arr, b_arr, eps=1e-7):
    \"\"\"Apply per-species Platt calibration.\"\"\"
    probs_c = np.clip(probs, eps, 1 - eps)
    logit_p = np.log(probs_c / (1 - probs_c))
    calib = 1.0 / (1.0 + np.exp(-(a_arr[None, :] * logit_p + b_arr[None, :])))
    return np.clip(calib, 0.0, 1.0)

print("PP-v10 helpers defined: fit_platt_scaling, apply_platt_scaling")"""

v10_postproc = "# Cell 18 — PP-v10: Platt Scaling Calibration\n\n" + COMMON_HEADER + """
# Step 4: Platt scaling calibration
if MODE == "train" and oof_proto_flat is not None:
    oof_p = sigmoid(oof_proto_flat / class_temperatures[None, :])
    print("Fitting Platt scaling from OOF...")
    PLATT_A, PLATT_B = fit_platt_scaling(oof_p, Y_FULL, C=CFG["platt_C"])
    print(f"  Platt a: mean={PLATT_A.mean():.3f}, std={PLATT_A.std():.3f}")
    print(f"  Platt b: mean={PLATT_B.mean():.3f}, std={PLATT_B.std():.3f}")
else:
    PLATT_A = np.ones(N_CLASSES,  dtype=np.float32)
    PLATT_B = np.zeros(N_CLASSES, dtype=np.float32)
    print("Submit mode: no OOF available, using identity Platt (a=1, b=0)")

print("Applying Platt scaling calibration")
probs = apply_platt_scaling(probs, PLATT_A, PLATT_B)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v11: Isotonic Regression Calibration
# ══════════════════════════════════════════════════════════════════════════════
v11_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v11 Isotonic Regression Calibration

## PP-v11 — Isotonic Regression Calibration

Non-parametric monotone calibration per species, fitted on OOF predictions.
More flexible than Platt scaling; can fix arbitrary miscalibration curves.
"""

v11_cfg = """# PP-v11: Isotonic calibration config (no extra hyperparams needed)
print("PP-v11 CFG loaded")"""

v11_helpers = """# PP-v11 helpers

def fit_isotonic_calibration(oof_probs, y_true):
    \"\"\"Fit per-species isotonic regression on OOF.
    Returns list of fitted IsotonicRegression models (length n_classes).
    \"\"\"
    from sklearn.isotonic import IsotonicRegression
    n_samples, C = oof_probs.shape
    n_files = y_true.shape[0]
    n_windows = n_samples // n_files
    y_rep = np.repeat(y_true, n_windows, axis=0)
    models = []
    for c in range(C):
        yc = y_rep[:, c]
        xc = oof_probs[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc):
            models.append(None)
            continue
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(xc, yc.astype(float))
        models.append(ir)
    return models


def apply_isotonic_calibration(probs, models):
    \"\"\"Apply per-species isotonic calibration.\"\"\"
    out = probs.copy()
    for c, ir in enumerate(models):
        if ir is None:
            continue
        out[:, c] = ir.transform(probs[:, c])
    return np.clip(out, 0.0, 1.0)

print("PP-v11 helpers defined: fit_isotonic_calibration, apply_isotonic_calibration")"""

v11_postproc = "# Cell 18 — PP-v11: Isotonic Regression Calibration\n\n" + COMMON_HEADER + """
# Step 4: Isotonic calibration
if MODE == "train" and oof_proto_flat is not None:
    oof_p = sigmoid(oof_proto_flat / class_temperatures[None, :])
    print("Fitting isotonic calibration from OOF...")
    ISO_MODELS = fit_isotonic_calibration(oof_p, Y_FULL)
    n_fitted = sum(1 for m in ISO_MODELS if m is not None)
    print(f"  Fitted {n_fitted}/{N_CLASSES} species isotonic models")
else:
    ISO_MODELS = [None] * N_CLASSES
    print("Submit mode: no OOF available, isotonic calibration skipped (identity)")

print("Applying isotonic calibration")
probs = apply_isotonic_calibration(probs, ISO_MODELS)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v12: Window Position Deweighting
# ══════════════════════════════════════════════════════════════════════════════
v12_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v12 Window Position Deweighting

## PP-v12 — Window Position Deweighting

First and last 2 windows may contain cut-off calls or startup noise.
Apply position-based weights to down-weight edge windows.
Sweeps edge weight in {0.6, 0.7, 0.8, 0.9}.
"""

v12_cfg = """# PP-v12: Window position deweighting config
CFG["position_edge_weights"] = [0.6, 0.7, 0.8, 0.9]
CFG["position_edge_weight"]  = 0.7   # default
print("PP-v12 CFG loaded")"""

v12_helpers = """# PP-v12 helpers

def build_position_weights(n_windows=12, edge_weight=0.7):
    \"\"\"Build window-position weight vector.
    First 2 and last 2 windows get edge_weight; middle windows get 1.0.
    \"\"\"
    w = np.ones(n_windows, dtype=np.float32)
    w[0]  = edge_weight
    w[1]  = 0.5 * (1.0 + edge_weight)   # interpolate
    w[-2] = 0.5 * (1.0 + edge_weight)
    w[-1] = edge_weight
    return w


def position_deweight(probs, n_windows=12, edge_weight=0.7):
    \"\"\"Scale window predictions by position-based weights.\"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    weights = build_position_weights(n_windows, edge_weight)   # (T,)
    view = probs.reshape(n_files, n_windows, C)
    out  = view * weights[None, :, None]
    return np.clip(out.reshape(N, C), 0.0, 1.0)

print("PP-v12 helpers defined: build_position_weights, position_deweight")"""

v12_postproc = "# Cell 18 — PP-v12: Window Position Deweighting\n\n" + COMMON_HEADER + """
# Step 4: Sweep edge weight (train) or apply best (submit)
if MODE == "train" and oof_proto_flat is not None:
    from sklearn.metrics import roc_auc_score
    best_ew  = 0.7
    best_auc = -1.0
    oof_p = sigmoid(oof_proto_flat / class_temperatures[None, :])
    n_files_oof = oof_p.shape[0] // N_WINDOWS

    for ew in CFG["position_edge_weights"]:
        p_scaled = position_deweight(oof_p, n_windows=N_WINDOWS, edge_weight=ew)
        p_file   = p_scaled.reshape(n_files_oof, N_WINDOWS, -1).max(axis=1)
        try:
            auc = roc_auc_score(Y_FULL, p_file, average="macro")
        except Exception:
            auc = 0.0
        print(f"  edge_weight={ew:.1f}  AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_ew  = ew
    print(f"  Best edge_weight={best_ew:.1f}  AUC={best_auc:.4f}")
    CFG["position_edge_weight"] = best_ew

ew = CFG["position_edge_weight"]
print(f"Applying position deweighting (edge_weight={ew})")
probs = position_deweight(probs, n_windows=N_WINDOWS, edge_weight=ew)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v13: Texture-Species Temporal Integration
# ══════════════════════════════════════════════════════════════════════════════
v13_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v13 Texture-Species Temporal Integration

## PP-v13 — Taxonomy-Aware Gaussian Temporal Integration

Aves: sigma=1.0 (event-based, narrow). Texture (Insecta/Amphibia): sigma=3.0 (wide).
Each species gets a taxon-appropriate Gaussian smoothing sigma.
"""

v13_cfg = """# PP-v13: Texture integration config
CFG["taxon_sigma"] = {"aves": 1.0, "texture": 3.0}
print("PP-v13 CFG loaded")"""

v13_helpers = """# PP-v13 helpers

def gaussian_kernel_1d(sigma, truncate=4.0):
    \"\"\"1D Gaussian kernel with given sigma.\"\"\"
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def taxon_gaussian_smooth(probs, primary_labels, class_name_map, texture_taxa,
                          n_windows=12, sigma_aves=1.0, sigma_texture=3.0):
    \"\"\"Apply species-specific Gaussian smoothing based on taxon type.\"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)
    out  = np.empty_like(view)

    k_aves    = gaussian_kernel_1d(sigma_aves)
    k_texture = gaussian_kernel_1d(sigma_texture)

    def convolve1d_reflect(seq, k):
        \"\"\"1D convolution with reflect padding.\"\"\"
        half = len(k) // 2
        left  = seq[:half][::-1]
        right = seq[-half:][::-1]
        padded = np.concatenate([left, seq, right])
        result = np.convolve(padded, k, mode="valid")
        return result[:len(seq)]

    for c in range(C):
        lbl = primary_labels[c] if c < len(primary_labels) else ""
        taxon = class_name_map.get(lbl, "Aves")
        k = k_texture if taxon in texture_taxa else k_aves
        for f in range(n_files):
            out[f, :, c] = convolve1d_reflect(view[f, :, c], k)

    return np.clip(out.reshape(N, C), 0.0, 1.0)

print("PP-v13 helpers defined: gaussian_kernel_1d, taxon_gaussian_smooth")"""

v13_postproc = "# Cell 18 — PP-v13: Texture-Species Temporal Integration\n\n" + COMMON_HEADER + """
# Step 4: Taxon-aware Gaussian smoothing
sigma_aves    = CFG["taxon_sigma"]["aves"]
sigma_texture = CFG["taxon_sigma"]["texture"]
print(f"Applying taxon-aware Gaussian smooth (sigma_aves={sigma_aves}, sigma_texture={sigma_texture})")
probs = taxon_gaussian_smooth(
    probs, PRIMARY_LABELS, CLASS_NAME_MAP, TEXTURE_TAXA,
    n_windows=N_WINDOWS, sigma_aves=sigma_aves, sigma_texture=sigma_texture
)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v14: Two-Pass Smoothing
# ══════════════════════════════════════════════════════════════════════════════
v14_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v14 Two-Pass Smoothing

## PP-v14 — Two-Pass Smoothing (Smooth → Gate → Re-smooth)

Pass 1: Gaussian σ=1.0 denoising.
Gate: soft threshold at thr=0.15, temp=0.05.
Pass 2: Gaussian σ=0.5 to refine call boundaries.
"""

v14_cfg = """# PP-v14: Two-pass smoothing config
CFG["twopass_sigma1"] = 1.0
CFG["twopass_sigma2"] = 0.5
CFG["twopass_thr"]    = 0.15
CFG["twopass_temp"]   = 0.05
print("PP-v14 CFG loaded")"""

v14_helpers = """# PP-v14 helpers

def gaussian_smooth_temporal(probs, n_windows=12, sigma=1.0):
    \"\"\"Uniform Gaussian smoothing across temporal windows (all species same sigma).\"\"\"
    from math import exp
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)

    radius = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / (sigma + 1e-8)) ** 2)
    k /= k.sum()
    half = len(k) // 2

    out = np.empty_like(view)
    for c in range(C):
        for f in range(n_files):
            seq = view[f, :, c]
            left  = seq[:half][::-1]
            right = seq[-half:][::-1]
            padded = np.concatenate([left, seq, right])
            conv = np.convolve(padded, k, mode="valid")
            out[f, :, c] = conv[:n_windows]
    return np.clip(out.reshape(N, C), 0.0, 1.0)


def soft_gate(probs, thr=0.15, temp=0.05):
    \"\"\"Soft threshold gate: probs * sigmoid((probs - thr) / temp).\"\"\"
    gate = 1.0 / (1.0 + np.exp(-np.clip((probs - thr) / (temp + 1e-8), -30, 30)))
    return np.clip(probs * gate, 0.0, 1.0)

print("PP-v14 helpers defined: gaussian_smooth_temporal, soft_gate")"""

v14_postproc = "# Cell 18 — PP-v14: Two-Pass Smoothing\n\n" + COMMON_HEADER + """
# Step 4: Two-pass smoothing
s1  = CFG["twopass_sigma1"]
s2  = CFG["twopass_sigma2"]
thr = CFG["twopass_thr"]
tmp = CFG["twopass_temp"]
print(f"Pass 1: Gaussian sigma={s1}")
probs = gaussian_smooth_temporal(probs, n_windows=N_WINDOWS, sigma=s1)
print(f"Gate: soft threshold thr={thr}, temp={tmp}")
probs = soft_gate(probs, thr=thr, temp=tmp)
print(f"Pass 2: Gaussian sigma={s2}")
probs = gaussian_smooth_temporal(probs, n_windows=N_WINDOWS, sigma=s2)
probs = np.clip(probs, 0.0, 1.0)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v15: Intra-File Score Normalization
# ══════════════════════════════════════════════════════════════════════════════
v15_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v15 Intra-File Score Normalization

## PP-v15 — Intra-File Score Normalization

Per-file z-score normalization across windows×species, then sigmoid re-scale.
Files at different locations have different baseline noise — normalization
creates consistent thresholds across files.
"""

v15_cfg = """# PP-v15: Intra-file normalization config
CFG["intrafile_norm_temp"] = 1.0   # sigmoid temperature after z-score
print("PP-v15 CFG loaded")"""

v15_helpers = """# PP-v15 helpers

def intrafile_normalize(probs, n_windows=12, temp=1.0, eps=1e-6):
    \"\"\"Per-file z-score normalization across all windows and species,
    followed by sigmoid re-scaling with temperature.
    \"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)

    # Per-file mean and std across all windows and species
    flat = view.reshape(n_files, -1)   # (F, T*C)
    mu  = flat.mean(axis=1)            # (F,)
    std = flat.std(axis=1)             # (F,)

    z = (view - mu[:, None, None]) / (std[:, None, None] + eps)
    # sigmoid re-scale
    out = 1.0 / (1.0 + np.exp(-np.clip(z / (temp + eps), -30, 30)))
    return np.clip(out.reshape(N, C), 0.0, 1.0)

print("PP-v15 helpers defined: intrafile_normalize")"""

v15_postproc = "# Cell 18 — PP-v15: Intra-File Score Normalization\n\n" + COMMON_HEADER + """
# Step 4: Intra-file normalization
temp_norm = CFG["intrafile_norm_temp"]
print(f"Applying intra-file z-score normalization (sigmoid temp={temp_norm})")
probs = intrafile_normalize(probs, n_windows=N_WINDOWS, temp=temp_norm)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v16: Peak Detection and Sharpening
# ══════════════════════════════════════════════════════════════════════════════
v16_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v16 Peak Detection and Sharpening

## PP-v16 — Peak Detection and Sharpening

Find local maxima in temporal sequence per species. Amplify peaks, leave non-peaks.
sharpened[t,c] = probs[t,c] * (1 + alpha * peak_strength[t,c]).
alpha=0.3 (tunable).
"""

v16_cfg = """# PP-v16: Peak sharpening config
CFG["peak_sharpen_alpha"] = 0.3
print("PP-v16 CFG loaded")"""

v16_helpers = """# PP-v16 helpers

def peak_sharpen(probs, n_windows=12, alpha=0.3):
    \"\"\"Soft peak detection: peak strength proportional to how much t exceeds neighbours.
    sharpened[t] = probs[t] * (1 + alpha * relu((probs[t] - max(probs[t-1], probs[t+1]))/probs[t]))
    Boundary windows are left unchanged.
    \"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C).copy()

    prev_v = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)
    next_v = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)
    neighbour_max = np.maximum(prev_v, next_v)
    # soft peak strength in [0,1]
    peak_strength = np.maximum(0.0, (view - neighbour_max) / (view + 1e-8))
    out = view * (1.0 + alpha * peak_strength)
    return np.clip(out.reshape(N, C), 0.0, 1.0)

print("PP-v16 helpers defined: peak_sharpen")"""

v16_postproc = "# Cell 18 — PP-v16: Peak Detection and Sharpening\n\n" + COMMON_HEADER + """
# Step 4: Peak sharpening
alpha = CFG["peak_sharpen_alpha"]
print(f"Applying peak sharpening (alpha={alpha})")
probs = peak_sharpen(probs, n_windows=N_WINDOWS, alpha=alpha)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v17: Negative Detection Suppression
# ══════════════════════════════════════════════════════════════════════════════
v17_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v17 Negative Detection Suppression

## PP-v17 — Negative Detection Suppression

If a species never exceeds suppress_thr=0.10 in any window of a file,
actively suppress it to near-zero (suppress_strength=0.7).
Reduces noise floor of absent species.
"""

v17_cfg = """# PP-v17: Negative suppression config
CFG["suppress_thr"]      = 0.10
CFG["suppress_strength"] = 0.70
print("PP-v17 CFG loaded")"""

v17_helpers = """# PP-v17 helpers

def negative_suppress(probs, n_windows=12, suppress_thr=0.10, suppress_strength=0.70):
    \"\"\"Suppress species where file-level max score < suppress_thr.
    probs_new[f,:,c] *= (1 - suppress_strength) if file_max[f,c] < suppress_thr.
    \"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)
    file_max = view.max(axis=1)   # (F, C)
    suppress_mask = (file_max < suppress_thr).astype(np.float32)   # (F, C)
    scale = 1.0 - suppress_strength * suppress_mask   # (F, C)
    out = view * scale[:, None, :]
    return np.clip(out.reshape(N, C), 0.0, 1.0)

print("PP-v17 helpers defined: negative_suppress")"""

v17_postproc = "# Cell 18 — PP-v17: Negative Detection Suppression\n\n" + COMMON_HEADER + """
# Step 4: Negative suppression
s_thr = CFG["suppress_thr"]
s_str = CFG["suppress_strength"]
print(f"Applying negative suppression (thr={s_thr}, strength={s_str})")
probs = negative_suppress(probs, n_windows=N_WINDOWS,
                          suppress_thr=s_thr, suppress_strength=s_str)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v18: Weighted Mean by Window Entropy
# ══════════════════════════════════════════════════════════════════════════════
v18_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v18 Entropy-Weighted Windows

## PP-v18 — Weighted Mean by Window Entropy

Weight windows by inverse entropy: confident windows (low entropy) get higher weight.
broadcast_scale[f,w] = weight[f,w] * N_WINDOWS (mean=1 per file).
tau=1.0 (entropy temperature, tunable).
"""

v18_cfg = """# PP-v18: Entropy weighting config
CFG["entropy_weight_tau"] = 1.0
print("PP-v18 CFG loaded")"""

v18_helpers = """# PP-v18 helpers

def entropy_weight_windows(probs, n_windows=12, tau=1.0, eps=1e-8):
    \"\"\"Weight windows by exp(-entropy/tau). Low-entropy (confident) windows
    get higher weights, scaled so mean weight per file = 1.
    \"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)

    p_safe = np.clip(view, eps, 1.0)
    # Shannon entropy per window per file
    H = -np.sum(p_safe * np.log(p_safe), axis=2)   # (F, T)
    # exp weighting
    raw_w = np.exp(-H / (tau + eps))               # (F, T)
    norm_w = raw_w / (raw_w.sum(axis=1, keepdims=True) + eps)   # (F, T)
    broadcast_scale = norm_w * n_windows            # (F, T), mean=1

    out = view * broadcast_scale[:, :, None]
    return np.clip(out.reshape(N, C), 0.0, 1.0)

print("PP-v18 helpers defined: entropy_weight_windows")"""

v18_postproc = "# Cell 18 — PP-v18: Entropy-Weighted Window Scaling\n\n" + COMMON_HEADER + """
# Step 4: Entropy-weighted window scaling
tau = CFG["entropy_weight_tau"]
print(f"Applying entropy-weighted window scaling (tau={tau})")
probs = entropy_weight_windows(probs, n_windows=N_WINDOWS, tau=tau)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v19: Score Clipping and Re-scaling
# ══════════════════════════════════════════════════════════════════════════════
v19_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v19 Score Clipping & Re-scaling

## PP-v19 — Per-species Score Clipping and Re-scaling

Clip at per-species 99th-percentile (OOF) + small margin, then rescale to [0,1].
Prevents overconfident outlier predictions from dominating AUC.
"""

v19_cfg = """# PP-v19: Score clipping config
CFG["clip_percentile"] = 99.0   # OOF percentile for upper clip
CFG["clip_margin"]     = 0.02   # margin added to percentile clip
print("PP-v19 CFG loaded")"""

v19_helpers = """# PP-v19 helpers

def compute_per_species_clip(oof_probs, percentile=99.0, margin=0.02):
    \"\"\"Compute per-species upper clip value from OOF predictions.
    Returns clip array of shape (n_classes,).
    \"\"\"
    clips = np.percentile(oof_probs, percentile, axis=0).astype(np.float32)
    clips = np.clip(clips + margin, 0.05, 1.0)
    return clips


def clip_rescale(probs, clips):
    \"\"\"Clip and rescale to [0,1] using per-species upper bounds.\"\"\"
    clipped = np.minimum(probs, clips[None, :])
    rescaled = clipped / (clips[None, :] + 1e-8)
    return np.clip(rescaled, 0.0, 1.0)

print("PP-v19 helpers defined: compute_per_species_clip, clip_rescale")"""

v19_postproc = "# Cell 18 — PP-v19: Score Clipping and Re-scaling\n\n" + COMMON_HEADER + """
# Step 4: Per-species clipping and re-scaling
if MODE == "train" and oof_proto_flat is not None:
    oof_p = sigmoid(oof_proto_flat / class_temperatures[None, :])
    CLIP_VALUES = compute_per_species_clip(
        oof_p, percentile=CFG["clip_percentile"], margin=CFG["clip_margin"]
    )
    print(f"Per-species clips: mean={CLIP_VALUES.mean():.3f}, min={CLIP_VALUES.min():.3f}, max={CLIP_VALUES.max():.3f}")
else:
    CLIP_VALUES = np.ones(N_CLASSES, dtype=np.float32)
    print("Submit mode: using clip=1.0 (no clipping)")

print("Applying per-species clip and rescale")
probs = clip_rescale(probs, CLIP_VALUES)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v20: Iterative Temporal Smoothing
# ══════════════════════════════════════════════════════════════════════════════
v20_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v20 Iterative Temporal Smoothing

## PP-v20 — Iterative Temporal Smoothing (3 rounds, geometric alpha decay)

Apply delta-shift smoothing 3 times with geometric decay: 0.20 → 0.12 → 0.06.
Converges to stable state faster than one heavy pass while preserving local structure.
"""

v20_cfg = """# PP-v20: Iterative smoothing config
CFG["iter_smooth_alphas"] = [0.20, 0.12, 0.06]
print("PP-v20 CFG loaded")"""

v20_helpers = """# PP-v20 helpers — reuses delta_shift_smooth from cell 19
print("PP-v20: using delta_shift_smooth from V17 utilities")"""

v20_postproc = "# Cell 18 — PP-v20: Iterative Temporal Smoothing\n\n" + COMMON_HEADER + """
# Step 4: 3-round iterative delta-shift smoothing
alphas = CFG["iter_smooth_alphas"]
print(f"Iterative temporal smoothing ({len(alphas)} rounds): {alphas}")
for i, alpha in enumerate(alphas):
    probs = delta_shift_smooth(probs, n_windows=N_WINDOWS, alpha=alpha)
    probs = np.clip(probs, 0.0, 1.0)
    print(f"  Round {i+1}: alpha={alpha:.2f} done")

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v21: Bilateral Temporal Filtering
# ══════════════════════════════════════════════════════════════════════════════
v21_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v21 Bilateral Temporal Filter

## PP-v21 — Bilateral Temporal Filtering

Smoothes temporally only when adjacent windows have similar score profiles.
sigma_s=1.5 (temporal distance), sigma_r=0.15 (score similarity).
Preserves call onset/offset edges unlike plain Gaussian.
"""

v21_cfg = """# PP-v21: Bilateral filter config
CFG["bilateral_sigma_s"] = 1.5
CFG["bilateral_sigma_r"] = 0.15
print("PP-v21 CFG loaded")"""

v21_helpers = """# PP-v21 helpers

def bilateral_temporal_filter(probs, n_windows=12, sigma_s=1.5, sigma_r=0.15, eps=1e-8):
    \"\"\"Bilateral filter across temporal windows.
    For each window t and species c:
      weight[t'] = Gaussian(|t-t'|, sigma_s) * Gaussian(|p[t,c]-p[t',c]|, sigma_r)
      bilateral[t,c] = sum_t'(weight[t'] * p[t',c]) / sum(weight)
    \"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)
    T = n_windows

    # Precompute spatial kernel (independent of data)
    t_idx  = np.arange(T, dtype=np.float32)
    sp_dist = (t_idx[:, None] - t_idx[None, :]) ** 2   # (T, T)
    K_s = np.exp(-0.5 * sp_dist / (sigma_s ** 2 + eps))

    out = np.empty_like(view)
    for f in range(n_files):
        for c in range(C):
            seq = view[f, :, c]  # (T,)
            range_dist = (seq[:, None] - seq[None, :]) ** 2   # (T, T)
            K_r = np.exp(-0.5 * range_dist / (sigma_r ** 2 + eps))
            W = K_s * K_r    # (T, T)
            W_norm = W / (W.sum(axis=1, keepdims=True) + eps)
            out[f, :, c] = W_norm @ seq
    return np.clip(out.reshape(N, C), 0.0, 1.0)

print("PP-v21 helpers defined: bilateral_temporal_filter")"""

v21_postproc = "# Cell 18 — PP-v21: Bilateral Temporal Filtering\n\n" + COMMON_HEADER + """
# Step 4: Bilateral temporal filter
sig_s = CFG["bilateral_sigma_s"]
sig_r = CFG["bilateral_sigma_r"]
print(f"Applying bilateral temporal filter (sigma_s={sig_s}, sigma_r={sig_r})")
probs = bilateral_temporal_filter(probs, n_windows=N_WINDOWS, sigma_s=sig_s, sigma_r=sig_r)
probs = np.clip(probs, 0.0, 1.0)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v22: Species Rarity Boosting
# ══════════════════════════════════════════════════════════════════════════════
v22_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v22 Species Rarity Boosting

## PP-v22 — Species Rarity Boosting

Rare species (few training examples) tend to be under-predicted due to class imbalance.
Gentle logit-space boost scaled by rarity: rarity_weight[c] = (1/freq[c])^0.3.
beta=0.5, max_boost=3.0.
"""

v22_cfg = """# PP-v22: Rarity boosting config
CFG["rarity_power"]    = 0.3
CFG["rarity_beta"]     = 0.5
CFG["rarity_max_boost"]= 3.0
print("PP-v22 CFG loaded")"""

v22_helpers = """# PP-v22 helpers

def compute_rarity_weights(y_full, rarity_power=0.3, max_boost=3.0, eps=1e-6):
    \"\"\"Compute per-species rarity weight from training soundscape labels.
    rarity_weight[c] = (1 / freq[c]) ^ rarity_power, clipped at max_boost.
    \"\"\"
    freq = y_full.mean(axis=0).astype(np.float32)  # (C,)
    freq = np.clip(freq, eps, 1.0)
    raw  = (1.0 / freq) ** rarity_power
    return np.clip(raw, 1.0, max_boost)


def rarity_boost(probs, rarity_weights, beta=0.5, eps=1e-7):
    \"\"\"Boost in logit space: logit_new = logit + log(w) * beta.\"\"\"
    probs_c = np.clip(probs, eps, 1 - eps)
    logit_p = np.log(probs_c / (1 - probs_c))
    boost   = np.log(rarity_weights[None, :]) * beta
    calib   = 1.0 / (1.0 + np.exp(-np.clip(logit_p + boost, -30, 30)))
    return np.clip(calib, 0.0, 1.0)

print("PP-v22 helpers defined: compute_rarity_weights, rarity_boost")"""

v22_postproc = "# Cell 18 — PP-v22: Species Rarity Boosting\n\n" + COMMON_HEADER + """
# Step 4: Rarity boosting
if MODE == "train" and Y_FULL is not None:
    RARITY_WEIGHTS = compute_rarity_weights(
        Y_FULL, rarity_power=CFG["rarity_power"], max_boost=CFG["rarity_max_boost"]
    )
    print(f"Rarity weights: mean={RARITY_WEIGHTS.mean():.3f}, max={RARITY_WEIGHTS.max():.3f}")
    n_boosted = (RARITY_WEIGHTS > 1.1).sum()
    print(f"  Species with boost>1.1: {n_boosted}")
else:
    RARITY_WEIGHTS = np.ones(N_CLASSES, dtype=np.float32)
    print("Submit mode: no Y_FULL available, rarity weights = 1 (no boost)")

beta = CFG["rarity_beta"]
print(f"Applying rarity boost (beta={beta})")
probs = rarity_boost(probs, RARITY_WEIGHTS, beta=beta)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# pp-v23: Ensemble Post-proc (Gaussian + Max-mean + Consensus)
# ══════════════════════════════════════════════════════════════════════════════
v23_title = """# BirdCLEF+ 2026 -- ProtoSSM v5: PP-v23 Ensemble Post-processing

## PP-v23 — Ensemble Post-processing (Gaussian + Max-mean + Consensus)

Combines three complementary methods:
1. Gaussian temporal smoothing (σ=1.5)
2. Max-mean blend (alpha=0.3)
3. Soft voting consensus gate (thr=0.15, beta=0.4)

Order: temperature → rank-aware → Gaussian → max-mean blend → consensus → threshold.
"""

v23_cfg = """# PP-v23: Ensemble post-proc config
CFG["pp23_gauss_sigma"]     = 1.5
CFG["pp23_maxmean_alpha"]   = 0.3
CFG["pp23_consensus_thr"]   = 0.15
CFG["pp23_consensus_beta"]  = 0.4
print("PP-v23 CFG loaded")"""

v23_helpers = """# PP-v23 helpers

def maxmean_blend(probs, n_windows=12, alpha=0.3):
    \"\"\"Blend per-window scores with the file-level mean.
    new[f,w,c] = (1-alpha)*probs[f,w,c] + alpha*file_mean[f,c]
    \"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view     = probs.reshape(n_files, n_windows, C)
    file_mean = view.mean(axis=1, keepdims=True)   # (F,1,C)
    out = (1.0 - alpha) * view + alpha * file_mean
    return np.clip(out.reshape(N, C), 0.0, 1.0)


def consensus_gate(probs, n_windows=12, thr=0.15, beta=0.4):
    \"\"\"Soft consensus: boost species that exceed thr in multiple windows.
    consensus_frac[f,c] = fraction of windows where probs>thr
    new[f,w,c] = probs[f,w,c] * (1 + beta * consensus_frac[f,c])
    \"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)
    frac = (view > thr).mean(axis=1, keepdims=True)   # (F,1,C)
    out  = view * (1.0 + beta * frac)
    return np.clip(out.reshape(N, C), 0.0, 1.0)


def gaussian_smooth_uniform(probs, n_windows=12, sigma=1.5, eps=1e-8):
    \"\"\"Uniform Gaussian smooth for all species (same sigma).\"\"\"
    N, C = probs.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = probs.reshape(n_files, n_windows, C)

    radius = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / (sigma + eps)) ** 2)
    k /= k.sum()
    half = len(k) // 2

    out = np.empty_like(view)
    for f in range(n_files):
        for c in range(C):
            seq = view[f, :, c]
            left  = seq[:half][::-1]
            right = seq[-half:][::-1]
            padded = np.concatenate([left, seq, right])
            conv = np.convolve(padded, k, mode="valid")
            out[f, :, c] = conv[:n_windows]
    return np.clip(out.reshape(N, C), 0.0, 1.0)

print("PP-v23 helpers defined: maxmean_blend, consensus_gate, gaussian_smooth_uniform")"""

v23_postproc = "# Cell 18 — PP-v23: Ensemble Post-processing\n\n" + COMMON_HEADER + """
# Step 4a: Gaussian temporal smoothing
sigma = CFG["pp23_gauss_sigma"]
print(f"Step 4a: Gaussian smooth (sigma={sigma})")
probs = gaussian_smooth_uniform(probs, n_windows=N_WINDOWS, sigma=sigma)
probs = np.clip(probs, 0.0, 1.0)

# Step 4b: Max-mean blend
alpha_mm = CFG["pp23_maxmean_alpha"]
print(f"Step 4b: Max-mean blend (alpha={alpha_mm})")
probs = maxmean_blend(probs, n_windows=N_WINDOWS, alpha=alpha_mm)
probs = np.clip(probs, 0.0, 1.0)

# Step 4c: Consensus gate
c_thr  = CFG["pp23_consensus_thr"]
c_beta = CFG["pp23_consensus_beta"]
print(f"Step 4c: Consensus gate (thr={c_thr}, beta={c_beta})")
probs = consensus_gate(probs, n_windows=N_WINDOWS, thr=c_thr, beta=c_beta)
probs = np.clip(probs, 0.0, 1.0)

""" + COMMON_FOOTER

# ══════════════════════════════════════════════════════════════════════════════
# BUILD ALL 20 NOTEBOOKS
# ══════════════════════════════════════════════════════════════════════════════
print("Creating 20 post-processing notebooks...")

make_notebook(v4_title,  v4_cfg,  v4_helpers,  v4_postproc,  "pantanal-pp-v4-causal-ema.ipynb")
make_notebook(v5_title,  v5_cfg,  v5_helpers,  v5_postproc,  "pantanal-pp-v5-median-filter.ipynb")
make_notebook(v6_title,  v6_cfg,  v6_helpers,  v6_postproc,  "pantanal-pp-v6-savgol-smooth.ipynb")
make_notebook(v7_title,  v7_cfg,  v7_helpers,  v7_postproc,  "pantanal-pp-v7-hmm-smooth.ipynb")
make_notebook(v8_title,  v8_cfg,  v8_helpers,  v8_postproc,  "pantanal-pp-v8-cooc-propagate.ipynb")
make_notebook(v9_title,  v9_cfg,  v9_helpers,  v9_postproc,  "pantanal-pp-v9-family-smooth.ipynb")
make_notebook(v10_title, v10_cfg, v10_helpers, v10_postproc, "pantanal-pp-v10-platt-calib.ipynb")
make_notebook(v11_title, v11_cfg, v11_helpers, v11_postproc, "pantanal-pp-v11-isotonic-calib.ipynb")
make_notebook(v12_title, v12_cfg, v12_helpers, v12_postproc, "pantanal-pp-v12-position-deweight.ipynb")
make_notebook(v13_title, v13_cfg, v13_helpers, v13_postproc, "pantanal-pp-v13-texture-integrate.ipynb")
make_notebook(v14_title, v14_cfg, v14_helpers, v14_postproc, "pantanal-pp-v14-twopass-smooth.ipynb")
make_notebook(v15_title, v15_cfg, v15_helpers, v15_postproc, "pantanal-pp-v15-intrafile-norm.ipynb")
make_notebook(v16_title, v16_cfg, v16_helpers, v16_postproc, "pantanal-pp-v16-peak-sharpen.ipynb")
make_notebook(v17_title, v17_cfg, v17_helpers, v17_postproc, "pantanal-pp-v17-neg-suppress.ipynb")
make_notebook(v18_title, v18_cfg, v18_helpers, v18_postproc, "pantanal-pp-v18-entropy-weight.ipynb")
make_notebook(v19_title, v19_cfg, v19_helpers, v19_postproc, "pantanal-pp-v19-clip-rescale.ipynb")
make_notebook(v20_title, v20_cfg, v20_helpers, v20_postproc, "pantanal-pp-v20-iterative-smooth.ipynb")
make_notebook(v21_title, v21_cfg, v21_helpers, v21_postproc, "pantanal-pp-v21-bilateral-filter.ipynb")
make_notebook(v22_title, v22_cfg, v22_helpers, v22_postproc, "pantanal-pp-v22-rarity-boost.ipynb")
make_notebook(v23_title, v23_cfg, v23_helpers, v23_postproc, "pantanal-pp-v23-ensemble-postproc.ipynb")

print("\nDone! 20 notebooks created.")
