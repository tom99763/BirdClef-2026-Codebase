"""Save the current best temporal post-processing pipeline to event_smooth/.

Reads outputs/smooth_experiments_results.json, identifies the best method,
and writes:
  event_smooth/best_postproc.py   — standalone apply_best_postproc() function
  event_smooth/best_postproc.json — metadata (name, AUC, vs Gaussian)

Run periodically to keep event_smooth in sync with latest experiments.
"""

import json
import os
import textwrap
from datetime import datetime

RESULTS_JSON = "outputs/smooth_experiments_results.json"
OUT_DIR = "event_smooth"
N_WINDOWS = 12
NUM_CLASSES = 234
TEMP_SCALE = 1.0

# ── Pipeline implementations (copy from eval_smooth_experiments.py) ──────────
# Keys are matched as substrings of the best method name (first match wins).
# Most-specific keys should come first.
PIPELINE_IMPLEMENTATIONS = {
    # ── MeanMax family (current best) ────────────────────────────────────────
    "MeanMax": """\
def apply_best_postproc(probs_file_T_C):
    \"\"\"Best temporal post-processing: SoftEntrWt(T=0.2) → LSE(β=4.5) → MeanMax anchor → cSEBBs.
    MeanMax anchor = max_w * file_max + (1-max_w) * file_mean   (max_w≈0.5-1.0 optimal).

    Args:
        probs_file_T_C: np.ndarray shape (N_WINDOWS, NUM_CLASSES) in [0,1]
    Returns:
        smoothed: np.ndarray shape (N_WINDOWS, NUM_CLASSES) in [0,1]
    \"\"\"
    import numpy as np
    import re as _re
    X = probs_file_T_C.copy()
    T, C = X.shape
    eps = 1e-9

    # Parse max_w and entr_temp from method name if available (defaults: max_w=0.5, temp=0.2)
    # These are set as module-level constants below; override as needed.
    entr_temp = ENTR_TEMP
    max_w     = MAX_W
    gm_alpha  = GM_ALPHA
    cp_thr    = CP_THR

    # Step 0: Soft entropy weighting
    H = -(X * np.log(X + eps) + (1 - X) * np.log(1 - X + eps))
    H_clip = H.mean(axis=1)                   # (T,) — avg binary entropy per clip
    w = np.exp(-H_clip / entr_temp)
    w = w / w.sum() * T                       # normalize

    # Step 1: Entropy-weighted logits → LSE pooling (β=4.5, r=1)
    logits = np.log(np.clip(X, eps, 1-eps) / np.clip(1-X, eps, 1-eps))
    logits_w = logits * w[:, None]
    beta, radius = 4.5, 1
    pad = np.pad(logits_w, ((radius, radius), (0, 0)), mode="edge")
    out_lse = np.zeros_like(logits_w)
    for t in range(T):
        win = pad[t:t + 2*radius+1, :]
        out_lse[t] = (1.0/beta)*(beta*win).max(axis=0) + \\
                     (1.0/beta)*np.log(np.exp(beta*(win - win.max(axis=0))).sum(axis=0))
    lse_probs = 1.0 / (1.0 + np.exp(-out_lse))   # (T, C)

    # Step 2: MeanMax anchor blend  — anchor = max_w*file_max + (1-max_w)*file_mean
    file_max  = lse_probs.max(axis=0)             # (C,)
    file_mean = lse_probs.mean(axis=0)            # (C,)
    anchor = max_w * file_max + (1 - max_w) * file_mean
    blended = (1 - gm_alpha) * lse_probs + gm_alpha * anchor[None, :]  # (T, C)

    # Step 3: cSEBBs-lite (thr, blend=0.4) — clean segment boundaries
    out = blended.copy()
    diff = np.abs(np.diff(blended, axis=0))       # (T-1, C)
    for t in range(T - 1):
        for c in np.where(diff[t] > cp_thr)[0]:
            seg = blended[max(0, t-2):min(T, t+3), c]
            out[t, c] = 0.6 * blended[t, c] + 0.4 * seg.mean()
    return np.clip(out, 0.0, 1.0)
""",

    # ── DualAnchor family (current best) ─────────────────────────────────────
    "DualAnchor": """\
def apply_best_postproc(probs_file_T_C):
    \"\"\"Best post-processing: SoftEntrWt → LSE → DualAnchor(NOR+Max) → cSEBBs.
    DualAnchor = NOR_W * NoisyOR + (1-NOR_W) * file_max   (combines MIL + extreme pooling).
    \"\"\"
    import numpy as np
    X = probs_file_T_C.copy()
    T, C = X.shape
    eps = 1e-9
    entr_temp = ENTR_TEMP
    alpha     = GM_ALPHA
    nor_w     = NOR_W
    cp_thr    = CP_THR

    # Step 0: Soft entropy weighting
    H = -(X * np.log(X + eps) + (1 - X) * np.log(1 - X + eps))
    H_clip = H.mean(axis=1)
    w = np.exp(-H_clip / entr_temp)
    w = w / w.sum() * T

    # Step 1: LSE pooling (β=4.5, r=1)
    logits = np.log(np.clip(X, eps, 1-eps) / np.clip(1-X, eps, 1-eps))
    logits_w = logits * w[:, None]
    beta, radius = 4.5, 1
    pad = np.pad(logits_w, ((radius, radius), (0, 0)), mode="edge")
    out_lse = np.zeros_like(logits_w)
    for t in range(T):
        win = pad[t:t + 2*radius+1, :]
        out_lse[t] = (1.0/beta)*(beta*win).max(axis=0) + \\
                     (1.0/beta)*np.log(np.exp(beta*(win - win.max(axis=0))).sum(axis=0))
    lse_probs = 1.0 / (1.0 + np.exp(-out_lse))

    # Step 2: Dual anchor = nor_w * NoisyOR + (1-nor_w) * file_max
    nor_anchor = 1.0 - np.prod(1.0 - lse_probs, axis=0)   # (C,)
    max_anchor = lse_probs.max(axis=0)                      # (C,)
    anchor = nor_w * nor_anchor + (1 - nor_w) * max_anchor
    blended = (1 - alpha) * lse_probs + alpha * anchor[None, :]

    # Step 3: cSEBBs-lite
    out = blended.copy()
    diff = np.abs(np.diff(blended, axis=0))
    for t in range(T - 1):
        for c in np.where(diff[t] > cp_thr)[0]:
            seg = blended[max(0, t-2):min(T, t+3), c]
            out[t, c] = 0.6 * blended[t, c] + 0.4 * seg.mean()
    return np.clip(out, 0.0, 1.0)
""",

    # ── NoisyOR family ───────────────────────────────────────────────────────
    "NoisyOR": """\
def apply_best_postproc(probs_file_T_C):
    \"\"\"Best post-processing: SoftEntrWt(T=0.1) → LSE(β=4.5) → NoisyOR anchor → cSEBBs.
    NoisyOR anchor = 1 - ∏_{t}(1 - p(t,c))  (MIL multiplicative pooling).
    \"\"\"
    import numpy as np
    X = probs_file_T_C.copy()
    T, C = X.shape
    eps = 1e-9
    entr_temp = ENTR_TEMP
    alpha     = GM_ALPHA
    cp_thr    = CP_THR

    # Step 0: Soft entropy weighting
    H = -(X * np.log(X + eps) + (1 - X) * np.log(1 - X + eps))
    H_clip = H.mean(axis=1)
    w = np.exp(-H_clip / entr_temp)
    w = w / w.sum() * T

    # Step 1: LSE pooling (β=4.5, r=1)
    logits = np.log(np.clip(X, eps, 1-eps) / np.clip(1-X, eps, 1-eps))
    logits_w = logits * w[:, None]
    beta, radius = 4.5, 1
    pad = np.pad(logits_w, ((radius, radius), (0, 0)), mode="edge")
    out_lse = np.zeros_like(logits_w)
    for t in range(T):
        win = pad[t:t + 2*radius+1, :]
        out_lse[t] = (1.0/beta)*(beta*win).max(axis=0) + \\
                     (1.0/beta)*np.log(np.exp(beta*(win - win.max(axis=0))).sum(axis=0))
    lse_probs = 1.0 / (1.0 + np.exp(-out_lse))

    # Step 2: NoisyOR anchor blend
    anchor = 1.0 - np.prod(1.0 - lse_probs, axis=0)   # (C,)
    blended = (1 - alpha) * lse_probs + alpha * anchor[None, :]

    # Step 3: cSEBBs-lite
    out = blended.copy()
    diff = np.abs(np.diff(blended, axis=0))
    for t in range(T - 1):
        for c in np.where(diff[t] > cp_thr)[0]:
            seg = blended[max(0, t-2):min(T, t+3), c]
            out[t, c] = 0.6 * blended[t, c] + 0.4 * seg.mean()
    return np.clip(out, 0.0, 1.0)
""",

    # ── SoftEntrWt family ────────────────────────────────────────────────────
    "SoftEntrWt(T=0.5)": """\
def apply_best_postproc(probs_file_T_C):
    \"\"\"Best temporal post-processing: SoftEntrWt(T=0.5) → LSE → GlobalMean → cSEBBs.

    Args:
        probs_file_T_C: np.ndarray shape (N_WINDOWS, NUM_CLASSES) in [0,1]
    Returns:
        smoothed: np.ndarray shape (N_WINDOWS, NUM_CLASSES) in [0,1]
    \"\"\"
    import numpy as np
    X = probs_file_T_C.copy()  # (T, C)
    T, C = X.shape

    # Step 0: Soft entropy weighting (T=0.5 — sharp discrimination of noisy clips)
    eps = 1e-9
    H = -(X * np.log(X + eps) + (1 - X) * np.log(1 - X + eps))
    H_clip = H.mean(axis=1)               # (T,) avg binary entropy per clip
    w = np.exp(-H_clip / 0.5)             # soft weight — low entropy → high weight
    w = w / w.sum() * T                   # normalize: mean weight = 1

    # Apply weights to logits
    logits = np.log(np.clip(X, eps, 1 - eps) / np.clip(1 - X, eps, 1 - eps))
    logits_w = logits * w[:, None]        # (T, C)

    # Step 1: LogSumExp pooling (β=4.5, r=1) — soft temporal max-dilation
    beta = 4.5
    radius = 1
    pad = np.pad(logits_w, ((radius, radius), (0, 0)), mode="edge")
    out_lse = np.zeros_like(logits_w)
    for t in range(T):
        window = pad[t:t + 2 * radius + 1, :]   # (2r+1, C)
        out_lse[t] = (1.0 / beta) * (beta * window).max(axis=0) + \\
                     (1.0 / beta) * np.log(np.exp(beta * (window - window.max(axis=0))).sum(axis=0))
    lse_probs = 1.0 / (1.0 + np.exp(-out_lse))  # (T, C)

    # Step 2: Global mean blend (α=0.175) — file-level prior
    file_mean = lse_probs.mean(axis=0)    # (C,)
    alpha = 0.175
    gm_probs = (1 - alpha) * lse_probs + alpha * file_mean[None, :]

    # Step 3: cSEBBs-lite (thr=0.06, blend=0.4) — clean segment boundaries
    out = gm_probs.copy()
    diff = np.abs(np.diff(gm_probs, axis=0))  # (T-1, C)
    for t in range(T - 1):
        change = diff[t]                   # (C,)
        is_boundary = change > 0.06
        # Find segment mean around each boundary
        for c in np.where(is_boundary)[0]:
            # Segment from last boundary to this one
            seg_start = max(0, t - 2)
            seg_end = min(T, t + 3)
            seg_mean = gm_probs[seg_start:seg_end, c].mean()
            out[t, c] = 0.6 * gm_probs[t, c] + 0.4 * seg_mean
    return np.clip(out, 0.0, 1.0)
""",

    "RDP→LSE→GM→cSEBBs": """\
def apply_best_postproc(probs_file_T_C):
    \"\"\"Best temporal post-processing: RDP → LSE → GlobalMean → cSEBBs.

    Args:
        probs_file_T_C: np.ndarray shape (N_WINDOWS, NUM_CLASSES) in [0,1]
    Returns:
        smoothed: np.ndarray shape (N_WINDOWS, NUM_CLASSES) in [0,1]
    \"\"\"
    import numpy as np
    X = probs_file_T_C.copy()  # (T, C)
    T, C = X.shape
    eps = 1e-9

    # Step 0: RDP weighting — upweight clips that deviate from file mean
    file_mean = X.mean(axis=0, keepdims=True)   # (1, C)
    file_std  = X.std(axis=0, keepdims=True) + 1e-6
    rdp = (np.abs(X - file_mean) / file_std).mean(axis=1)  # (T,) — deviation score
    w = np.exp(rdp)
    w = w / w.sum() * T                          # normalize

    logits = np.log(np.clip(X, eps, 1-eps) / np.clip(1-X, eps, 1-eps))
    logits_w = logits * w[:, None]

    # Step 1: LSE pooling (β=4.5, r=1)
    beta, radius = 4.5, 1
    pad = np.pad(logits_w, ((radius, radius), (0, 0)), mode="edge")
    out_lse = np.zeros_like(logits_w)
    for t in range(T):
        window = pad[t:t + 2*radius+1, :]
        out_lse[t] = (1.0/beta)*(beta*window).max(axis=0) + \\
                     (1.0/beta)*np.log(np.exp(beta*(window-window.max(axis=0))).sum(axis=0))
    lse_probs = 1.0 / (1.0 + np.exp(-out_lse))

    # Step 2: GlobalMean blend (α=0.175)
    gm = lse_probs.mean(axis=0)
    gm_probs = 0.825 * lse_probs + 0.175 * gm[None, :]

    # Step 3: cSEBBs-lite (thr=0.06, blend=0.4)
    out = gm_probs.copy()
    diff = np.abs(np.diff(gm_probs, axis=0))
    for t in range(T - 1):
        for c in np.where(diff[t] > 0.06)[0]:
            seg = gm_probs[max(0,t-2):min(T,t+3), c]
            out[t, c] = 0.6 * gm_probs[t, c] + 0.4 * seg.mean()
    return np.clip(out, 0.0, 1.0)
""",

    "SoftEntrWt(T=1)": """\
def apply_best_postproc(probs_file_T_C):
    \"\"\"Best temporal post-processing: SoftEntrWt(T=1) → LSE → GlobalMean → cSEBBs.\"\"\"
    import numpy as np
    X = probs_file_T_C.copy()
    T, C = X.shape
    eps = 1e-9
    H = -(X * np.log(X + eps) + (1 - X) * np.log(1 - X + eps))
    H_clip = H.mean(axis=1)
    w = np.exp(-H_clip / 1.0)
    w = w / w.sum() * T
    logits = np.log(np.clip(X, eps, 1-eps) / np.clip(1-X, eps, 1-eps))
    logits_w = logits * w[:, None]
    beta, radius = 4.5, 1
    pad = np.pad(logits_w, ((radius, radius), (0, 0)), mode="edge")
    out_lse = np.zeros_like(logits_w)
    for t in range(T):
        window = pad[t:t + 2*radius+1, :]
        out_lse[t] = (1.0/beta)*(beta*window).max(axis=0) + \\
                     (1.0/beta)*np.log(np.exp(beta*(window-window.max(axis=0))).sum(axis=0))
    lse_probs = 1.0 / (1.0 + np.exp(-out_lse))
    gm = lse_probs.mean(axis=0)
    gm_probs = 0.825 * lse_probs + 0.175 * gm[None, :]
    out = gm_probs.copy()
    diff = np.abs(np.diff(gm_probs, axis=0))
    for t in range(T-1):
        for c in np.where(diff[t] > 0.06)[0]:
            seg = gm_probs[max(0,t-2):min(T,t+3), c]
            out[t, c] = 0.6 * gm_probs[t, c] + 0.4 * seg.mean()
    return np.clip(out, 0.0, 1.0)
""",
}

# Default fallback implementation (LSE→GM→cSEBBs baseline)
DEFAULT_IMPL = """\
def apply_best_postproc(probs_file_T_C):
    \"\"\"Best temporal post-processing: LSE(β=4.5) → GlobalMean(α=0.175) → cSEBBs(thr=0.06).\"\"\"
    import numpy as np
    X = probs_file_T_C.copy()
    T, C = X.shape
    eps = 1e-9
    logits = np.log(np.clip(X, eps, 1-eps) / np.clip(1-X, eps, 1-eps))
    beta, radius = 4.5, 1
    pad = np.pad(logits, ((radius, radius), (0, 0)), mode="edge")
    out_lse = np.zeros_like(logits)
    for t in range(T):
        window = pad[t:t+2*radius+1, :]
        out_lse[t] = (1.0/beta)*(beta*window).max(axis=0) + \\
                     (1.0/beta)*np.log(np.exp(beta*(window-window.max(axis=0))).sum(axis=0))
    lse_probs = 1.0 / (1.0 + np.exp(-out_lse))
    gm = lse_probs.mean(axis=0)
    gm_probs = 0.825 * lse_probs + 0.175 * gm[None, :]
    out = gm_probs.copy()
    diff = np.abs(np.diff(gm_probs, axis=0))
    for t in range(T-1):
        for c in np.where(diff[t] > 0.06)[0]:
            seg = gm_probs[max(0,t-2):min(T,t+3), c]
            out[t, c] = 0.6 * gm_probs[t, c] + 0.4 * seg.mean()
    return np.clip(out, 0.0, 1.0)
"""


def find_best_method():
    if not os.path.exists(RESULTS_JSON):
        return None, 0.0, 0.0
    d = json.load(open(RESULTS_JSON))
    r = d["results"]
    gauss = r.get("2.Gaussian (fixed)", {}).get("mean", 0.0)
    ranked = sorted(r.items(), key=lambda x: -x[1].get("mean", 0))
    if not ranked:
        return None, gauss, 0.0
    best_name, best_data = ranked[0]
    return best_name, gauss, best_data.get("mean", 0.0)


def get_implementation(method_name):
    for key, impl in PIPELINE_IMPLEMENTATIONS.items():
        if key in method_name:
            return impl
    return DEFAULT_IMPL


def make_safe_filename(method_name):
    """Convert method name to a filesystem-safe string."""
    # e.g. "R13.01.SoftEntr(T=0.2)→LSE→GM→cSEBBs" → "R13.01_SoftEntr_T0.2_LSE_GM_cSEBBs"
    s = method_name
    s = s.replace("→", "_")
    s = s.replace("(", "_").replace(")", "").replace("=", "")
    s = s.replace(".", "_", 1)   # keep first dot as _ (round number)
    s = s.replace(" ", "")
    # collapse multiple underscores
    import re
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _parse_method_constants(method_name):
    """Parse hyperparameter constants from method name string.
    e.g. 'R19.12.NoisyOR(a=0.30)→cSEBBs' → entr_temp=0.1, gm_alpha=0.30
    e.g. 'R18.10.MeanMax(T=0.1,w=1.0,a=0.30)→cSEBBs' → entr_temp=0.1, max_w=1.0, gm_alpha=0.30
    """
    import re
    # Default: entr_temp=0.1 for NoisyOR/AvgTopK/PowerMean (all use 0.1),
    #          entr_temp=0.2 for older MeanMax without explicit T=
    entr_temp = 0.1   # safe default (matches R19 best methods)
    max_w     = 1.0   # default (w=1.0 wins in R17+)
    gm_alpha  = 0.30  # default (a=0.30 wins in R18+)
    cp_thr    = 0.06  # fixed

    m = re.search(r'T=([\d.]+)', method_name)
    if m:
        entr_temp = float(m.group(1))
    m = re.search(r'[,\(]w=([\d.]+)', method_name)
    if m:
        max_w = float(m.group(1))
    m = re.search(r'[,\(]a=([\d.]+)', method_name)
    if m:
        gm_alpha = float(m.group(1))
    m = re.search(r'thr=([\d.]+)', method_name)
    if m:
        cp_thr = float(m.group(1))
    # nor_w: weight of NoisyOR in DualAnchor (default 0.5 = R20.11 best)
    nor_w = 0.5
    m = re.search(r'nw=([\d.]+)', method_name)
    if m:
        nor_w = float(m.group(1))
    return entr_temp, max_w, gm_alpha, cp_thr, nor_w


def save_best_postproc():
    best_name, gauss_auc, best_auc = find_best_method()
    if best_name is None:
        print("No results found.")
        return

    delta = best_auc - gauss_auc
    impl = get_implementation(best_name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    safe_name = make_safe_filename(best_name)
    auc_str = f"{best_auc:.4f}".replace(".", "p")  # e.g. 0p7913

    # Check if named .ipynb already exists and metadata matches — skip if so
    nb_filename = f"best_ensemble_postproc_{safe_name}_auc{auc_str}.ipynb"
    nb_named_path = os.path.join(OUT_DIR, nb_filename)
    json_path = os.path.join(OUT_DIR, "best_postproc.json")
    if os.path.exists(nb_named_path) and os.path.exists(json_path):
        try:
            meta_existing = json.load(open(json_path))
            if meta_existing.get("best_method") == best_name and abs(meta_existing.get("best_auc", 0) - best_auc) < 1e-6:
                print(f"Already up to date: {nb_filename}  (AUC={best_auc:.4f})")
                return
        except Exception:
            pass

    # Named file: postproc_<method>_auc<score>.py
    named_filename = f"postproc_{safe_name}_auc{auc_str}.py"
    named_path = os.path.join(OUT_DIR, named_filename)

    # Canonical pointer: best_postproc.py (always the latest best)
    best_path = os.path.join(OUT_DIR, "best_postproc.py")

    # Parse method-specific constants
    entr_temp, max_w, gm_alpha, cp_thr, nor_w = _parse_method_constants(best_name)
    constants_block = f"""\
ENTR_TEMP  = {entr_temp}   # soft entropy temperature
MAX_W      = {max_w}   # MeanMax anchor weight (max_w*file_max + (1-max_w)*file_mean)
GM_ALPHA   = {gm_alpha}  # global anchor blend weight (alpha)
CP_THR     = {cp_thr}   # cSEBBs change-point threshold
NOR_W      = {nor_w}   # DualAnchor: weight of NoisyOR (1-NOR_W = weight of GlobalMax)
"""

    header = f'''\
"""Best temporal post-processing pipeline for BirdCLEF 2026 soundscape inference.
Auto-generated by scripts/save_best_postproc.py — {timestamp}

Method:  {best_name}
OOF AUC: {best_auc:.4f}  (vs Gaussian {gauss_auc:.4f},  delta {delta:+.4f})

Usage:
    from best_postproc import apply_best_postproc
    # probs: np.ndarray (N_WINDOWS, NUM_CLASSES) in [0, 1]
    smoothed = apply_best_postproc(probs)
"""
import numpy as np

N_WINDOWS  = 12
NUM_CLASSES = 234
{constants_block}
'''
    content = header + textwrap.dedent(impl)

    # Write named file (permanent history)
    with open(named_path, "w") as f:
        f.write(content)
    print(f"Wrote {named_path}")

    # Write/overwrite best_postproc.py (current best pointer)
    with open(best_path, "w") as f:
        f.write(content)
    print(f"Wrote {best_path}")

    # Write metadata JSON
    meta = {
        "best_method": best_name,
        "best_auc": best_auc,
        "gaussian_auc": gauss_auc,
        "delta_vs_gaussian": delta,
        "timestamp": timestamp,
        "results_source": RESULTS_JSON,
    }
    json_path = os.path.join(OUT_DIR, "best_postproc.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {json_path}")

    # Write .ipynb — copy best_ensemble.ipynb and inject post-processing
    nb_path = save_notebook(best_name, best_auc, gauss_auc, delta, impl, safe_name, auc_str, timestamp, constants_block)
    if nb_path:
        print(f"Wrote {nb_path}")

    print(f"Best: {best_name}  AUC={best_auc:.4f}  ({delta:+.4f} vs Gaussian)")


# ── Notebook generation ────────────────────────────────────────────────────────
TEMPLATE_NB = "submissions/best_ensemble.ipynb"

# The apply_best_postproc function is injected as a standalone cell.
# It takes probs (12, NUM_CLASSES) in [0,1] and returns smoothed (12, NUM_CLASSES).
POSTPROC_CELL_HEADER = """\
# ═══════════════════════════════════════════════════════════════════
# TEMPORAL POST-PROCESSING  —  auto-generated by save_best_postproc.py
# Method : {method}
# OOF AUC: {auc:.4f}  (Gaussian {gauss:.4f},  delta {delta:+.4f})
# Updated: {ts}
# ═══════════════════════════════════════════════════════════════════
import numpy as np

N_WINDOWS  = 12
NUM_CLASSES = 234
{constants_block}
"""

# Patch applied to the inference cell: add apply_best_postproc(preds) call
INFERENCE_PATCH_BEFORE = "    return row_ids, preds"
INFERENCE_PATCH_AFTER  = "    preds = apply_best_postproc(preds)  # temporal post-processing\n    return row_ids, preds"


def make_cell(source, cell_type="code"):
    """Create a notebook cell dict."""
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    }


def save_notebook(best_name, best_auc, gauss_auc, delta, impl, safe_name, auc_str, timestamp, constants_block=""):
    """Clone best_ensemble.ipynb, inject post-processing, save to event_smooth/."""
    if not os.path.exists(TEMPLATE_NB):
        print(f"Template not found: {TEMPLATE_NB}")
        return None

    nb = json.load(open(TEMPLATE_NB))

    # 1. Update title cell (cell 0)
    old_title = "".join(nb["cells"][0]["source"])
    new_title = old_title.replace(
        "# BirdCLEF 2026 — Best Ensemble",
        f"# BirdCLEF 2026 — Best Ensemble + PostProc: {best_name}"
    )
    new_title = new_title.replace(
        "**Holdout AUC:",
        f"**PostProc OOF AUC: {best_auc:.4f} ({delta:+.4f} vs Gaussian)** | Generated: {timestamp}\n\n**Holdout AUC:"
    )
    nb["cells"][0]["source"] = [new_title]

    # 2. Build post-processing code cell
    pp_source = (
        POSTPROC_CELL_HEADER.format(
            method=best_name, auc=best_auc, gauss=gauss_auc, delta=delta, ts=timestamp,
            constants_block=constants_block,
        )
        + textwrap.dedent(impl)
    )
    pp_cell = make_cell(pp_source, cell_type="code")

    # 3. Find "## Inference" markdown cell and insert pp_cell right before it
    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and "Inference" in src:
            insert_idx = i
            break

    if insert_idx is not None:
        nb["cells"].insert(insert_idx, pp_cell)
    else:
        # Fallback: append before last cell
        nb["cells"].insert(-1, pp_cell)

    # 4. Patch inference cell: add apply_best_postproc(preds) before return
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if INFERENCE_PATCH_BEFORE in src:
            patched = src.replace(INFERENCE_PATCH_BEFORE, INFERENCE_PATCH_AFTER, 1)
            cell["source"] = [patched]
            break

    # 5. Write to event_smooth/
    nb_filename = f"best_ensemble_postproc_{safe_name}_auc{auc_str}.ipynb"
    nb_path = os.path.join(OUT_DIR, nb_filename)
    best_nb_path = os.path.join(OUT_DIR, "best_ensemble_postproc.ipynb")

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(best_nb_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return nb_path


if __name__ == "__main__":
    save_best_postproc()
