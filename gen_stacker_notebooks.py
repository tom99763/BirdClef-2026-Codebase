#!/usr/bin/env python3
"""Generate all 18 stacker notebooks from the reference notebook."""

import json
import copy
import os

REF_PATH = "/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/current_subs 2/correct-sed-perch-probe-birdclef2026.ipynb"
OUT_DIR  = "/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/current_subs 2"

with open(REF_PATH) as f:
    ref_nb = json.load(f)

# ── Identify the VLOM blend boundary inside cell 49 ──────────────────────────
VLOM_MARKER = "# ── VLOM blend: ProtoSSM final scores + SED BranchEns→cSEBBs ─────────────────"

def trim_cell49(cell):
    """Return cell 49 with everything from the VLOM blend comment removed."""
    cell = copy.deepcopy(cell)
    src = cell["source"]
    # Find the line index of the VLOM marker
    cut = None
    for i, line in enumerate(src):
        if "VLOM blend" in line and "ProtoSSM final scores" in line:
            cut = i
            break
    if cut is not None:
        # Remove trailing blank lines before cut
        while cut > 0 and src[cut - 1].strip() == "":
            cut -= 1
        cell["source"] = src[:cut]
    return cell


def make_code_cell(source_str):
    """Create a notebook code cell from a source string."""
    # Split into lines, preserving newlines on each (except last)
    lines = source_str.split("\n")
    source_lines = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


# ── Cell A: Build Stacker Features ───────────────────────────────────────────
CELL_A_SRC = """\
# ── Stacker Input: concat 5 model predictions ────────────────────────────────
import numpy as np

# Ensure all feature arrays are (n_rows, 234) float32
_feat_perch_raw    = scores_test_raw.astype(np.float32)        # (n_rows, 234)
_feat_perch_prior  = test_base_scores.astype(np.float32)       # (n_rows, 234)
_feat_mlp          = mlp_scores.astype(np.float32)             # (n_rows, 234)
_feat_proto        = proto_scores_flat.astype(np.float32)      # (n_rows, 234)

if sed_preds_all is not None:
    _feat_sed = sed_preds_all.astype(np.float32)               # (n_rows, 234)
else:
    _feat_sed = np.zeros_like(_feat_perch_raw)                 # fallback

X_test_raw = np.concatenate([
    _feat_perch_raw,
    _feat_perch_prior,
    _feat_mlp,
    _feat_proto,
    _feat_sed,
], axis=1)  # (n_rows, 1170)

print(f"Stacker input X_test_raw: {X_test_raw.shape}")

# Load normalization stats (same stats used during training)
import pickle
_norm = np.load(f"{STACKER_DIR}/stacker_norm_v3.npz", allow_pickle=True)
_mean, _std = _norm["mean"], _norm["std"]
X_test_norm = (X_test_raw - _mean) / (_std + 1e-8)
print(f"X_test_norm: mean={X_test_norm.mean():.4f} std={X_test_norm.std():.4f}")\
"""

# ── Cell B templates ──────────────────────────────────────────────────────────

def cell_b_sequence(arch, weight_file):
    return f"""\
# ── Stacker Inference: {arch} ─────────────────────────────────────────────────
import onnxruntime as ort
import numpy as np

STACKER_ARCH = "{arch}"
STACKER_WEIGHT_FILE = "{weight_file}"

# Reshape to sequences: (n_files, 12, 1170)
n_rows = X_test_norm.shape[0]
n_files = len(test_paths)
n_windows = 12

X_seq = X_test_norm.reshape(n_files, n_windows, -1).astype(np.float32)
print(f"X_seq: {{X_seq.shape}}")

# Run ONNX inference
_sess_opts2 = ort.SessionOptions()
_sess_opts2.intra_op_num_threads = 4
_stacker_sess = ort.InferenceSession(
    f"{{STACKER_DIR}}/{{STACKER_WEIGHT_FILE}}",
    _sess_opts2,
    providers=["CPUExecutionProvider"]
)

_stacker_out = _stacker_sess.run(None, {{_stacker_sess.get_inputs()[0].name: X_seq}})[0]
# Output: (n_files, 12, 234) or (n_files, n_windows, 234)
stacker_final_probs = _stacker_out.reshape(n_rows, 234).astype(np.float32)
print(f"Stacker final probs: {{stacker_final_probs.shape}}  range [{{stacker_final_probs.min():.3f}}, {{stacker_final_probs.max():.3f}}]")\
"""

def cell_b_sequence_ss(arch, weight_file):
    return f"""\
# ── Stacker Inference: {arch} ─────────────────────────────────────────────────
# NOTE: _ss weights pending from train_stacker_v3_ss.py — update STACKER_WEIGHT_FILE when training completes
import onnxruntime as ort
import numpy as np

STACKER_ARCH = "{arch}"
STACKER_WEIGHT_FILE = "{weight_file}"

# Reshape to sequences: (n_files, 12, 1170)
n_rows = X_test_norm.shape[0]
n_files = len(test_paths)
n_windows = 12

X_seq = X_test_norm.reshape(n_files, n_windows, -1).astype(np.float32)
print(f"X_seq: {{X_seq.shape}}")

# Run ONNX inference
_sess_opts2 = ort.SessionOptions()
_sess_opts2.intra_op_num_threads = 4
_stacker_sess = ort.InferenceSession(
    f"{{STACKER_DIR}}/{{STACKER_WEIGHT_FILE}}",
    _sess_opts2,
    providers=["CPUExecutionProvider"]
)

_stacker_out = _stacker_sess.run(None, {{_stacker_sess.get_inputs()[0].name: X_seq}})[0]
# Output: (n_files, 12, 234) or (n_files, n_windows, 234)
stacker_final_probs = _stacker_out.reshape(n_rows, 234).astype(np.float32)
print(f"Stacker final probs: {{stacker_final_probs.shape}}  range [{{stacker_final_probs.min():.3f}}, {{stacker_final_probs.max():.3f}}]")\
"""

def cell_b_context_pkl(arch, weight_file, is_ss=False):
    note = "# NOTE: _ss weights pending from train_stacker_v3_ss.py — update STACKER_WEIGHT_FILE when training completes\n" if is_ss else ""
    return f"""\
# ── Stacker Inference: {arch} ─────────────────────────────────────────────────
{note}import numpy as np

STACKER_ARCH = "{arch}"
STACKER_WEIGHT_FILE = "{weight_file}"

# Build context features: context_k=1 → [t-1, t, t+1] windows padded with zeros
n_rows = X_test_norm.shape[0]
n_files = len(test_paths)
n_windows = 12
feat_dim = X_test_norm.shape[1]  # 1170

# Reshape to (n_files, 12, 1170)
X_win = X_test_norm.reshape(n_files, n_windows, feat_dim)

# Build context features
context_k = 1
ctx_dim = (2 * context_k + 1) * feat_dim  # 3 * 1170 = 3510
X_ctx = np.zeros((n_rows, ctx_dim), dtype=np.float32)
for fi in range(n_files):
    for ti in range(n_windows):
        row_i = fi * n_windows + ti
        chunks = []
        for dt in range(-context_k, context_k + 1):
            t2 = ti + dt
            if 0 <= t2 < n_windows:
                chunks.append(X_win[fi, t2])
            else:
                chunks.append(np.zeros(feat_dim, dtype=np.float32))
        X_ctx[row_i] = np.concatenate(chunks)

print(f"X_ctx (context features): {{X_ctx.shape}}")

# Load and run model
import pickle
with open(f"{{STACKER_DIR}}/{{STACKER_WEIGHT_FILE}}", "rb") as f:
    _stacker_model = pickle.load(f)

stacker_final_probs = _stacker_model.predict_proba(X_ctx).astype(np.float32)
print(f"Stacker final probs: {{stacker_final_probs.shape}}  range [{{stacker_final_probs.min():.3f}}, {{stacker_final_probs.max():.3f}}]")\
"""

def cell_b_mlp_onnx(arch, weight_file, is_ss=False):
    note = "# NOTE: _ss weights pending from train_stacker_v3_ss.py — update STACKER_WEIGHT_FILE when training completes\n" if is_ss else ""
    return f"""\
# ── Stacker Inference: {arch} ─────────────────────────────────────────────────
{note}import onnxruntime as ort
import numpy as np

STACKER_ARCH = "{arch}"
STACKER_WEIGHT_FILE = "{weight_file}"

# Build context features: context_k=1 → [t-1, t, t+1] windows padded with zeros
n_rows = X_test_norm.shape[0]
n_files = len(test_paths)
n_windows = 12
feat_dim = X_test_norm.shape[1]  # 1170

X_win = X_test_norm.reshape(n_files, n_windows, feat_dim)
context_k = 1
ctx_dim = (2 * context_k + 1) * feat_dim  # 3 * 1170 = 3510
X_ctx = np.zeros((n_rows, ctx_dim), dtype=np.float32)
for fi in range(n_files):
    for ti in range(n_windows):
        row_i = fi * n_windows + ti
        chunks = []
        for dt in range(-context_k, context_k + 1):
            t2 = ti + dt
            if 0 <= t2 < n_windows:
                chunks.append(X_win[fi, t2])
            else:
                chunks.append(np.zeros(feat_dim, dtype=np.float32))
        X_ctx[row_i] = np.concatenate(chunks)

print(f"X_ctx (context features): {{X_ctx.shape}}")

_sess_opts2 = ort.SessionOptions()
_sess_opts2.intra_op_num_threads = 4
_stacker_sess = ort.InferenceSession(
    f"{{STACKER_DIR}}/{{STACKER_WEIGHT_FILE}}",
    _sess_opts2,
    providers=["CPUExecutionProvider"]
)

stacker_final_probs = _stacker_sess.run(None, {{_stacker_sess.get_inputs()[0].name: X_ctx}})[0]
stacker_final_probs = stacker_final_probs.astype(np.float32)
print(f"Stacker final probs: {{stacker_final_probs.shape}}  range [{{stacker_final_probs.min():.3f}}, {{stacker_final_probs.max():.3f}}]")\
"""

def cell_stacker_dir():
    return """\
STACKER_DIR = '/kaggle/input/datasets/tom99763/birdclef2026-claude/weights/stacker_weights'\
"""

def cell_submission(arch):
    return f"""\
# Cell 18 — Build submission (Stacker: {arch})
submission = pd.DataFrame(stacker_final_probs, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

expected_rows = len(test_paths) * N_WINDOWS
assert len(submission) == expected_rows, f"Expected {{expected_rows}}, got {{len(submission)}}"
assert submission.columns.tolist() == ["row_id"] + PRIMARY_LABELS
assert not submission.isna().any().any()

submission.to_csv("submission.csv", index=False)
print(f"Saved submission.csv (stacker: {arch})")
print("Submission shape:", submission.shape)
print(submission.iloc[:3, :8])\
"""

# ── Notebook spec table ───────────────────────────────────────────────────────
# (notebook_stem, arch, weight_file, type)
NOTEBOOKS_V3 = [
    ("stacker_lgbm",           "lgbm",           "stacker_lgbm_v3_auc0.8826.pkl",              "context_pkl"),
    ("stacker_xgb",            "xgb",            "stacker_xgb_v3_auc0.8767.pkl",               "context_pkl"),
    ("stacker_mlp",            "mlp",            "stacker_mlp_v3_auc0.9594.onnx",              "mlp_onnx"),
    ("stacker_bigru",          "bigru",          "stacker_bigru_v3_auc0.8770.onnx",            "sequence"),
    ("stacker_tcn",            "tcn",            "stacker_tcn_v3_auc0.9007.onnx",              "sequence"),
    ("stacker_transformer",    "transformer",    "stacker_transformer_v3_auc0.9072.onnx",      "sequence"),
    ("stacker_ssm",            "ssm",            "stacker_ssm_v3_auc0.8782.onnx",              "sequence"),
    ("stacker_ft_transformer", "ft_transformer", "stacker_ft_transformer_v3_auc0.8936.onnx",   "sequence"),
    ("stacker_cnn1d",          "cnn1d",          "stacker_cnn1d_v3_auc0.8853.onnx",            "sequence"),
]

NOTEBOOKS_SS = [
    ("stacker_lgbm_ss",           "lgbm_ss",           "stacker_lgbm_ss_auc0.8826.pkl",              "context_pkl"),
    ("stacker_xgb_ss",            "xgb_ss",            "stacker_xgb_ss_auc0.8767.pkl",               "context_pkl"),
    ("stacker_mlp_ss",            "mlp_ss",            "stacker_mlp_ss_auc0.9594.onnx",              "mlp_onnx"),
    ("stacker_bigru_ss",          "bigru_ss",          "stacker_bigru_ss_auc0.8770.onnx",            "sequence"),
    ("stacker_tcn_ss",            "tcn_ss",            "stacker_tcn_ss_auc0.9007.onnx",              "sequence"),
    ("stacker_transformer_ss",    "transformer_ss",    "stacker_transformer_ss_auc0.9072.onnx",      "sequence"),
    ("stacker_ssm_ss",            "ssm_ss",            "stacker_ssm_ss_auc0.8782.onnx",              "sequence"),
    ("stacker_ft_transformer_ss", "ft_transformer_ss", "stacker_ft_transformer_ss_auc0.8936.onnx",   "sequence"),
    ("stacker_cnn1d_ss",          "cnn1d_ss",          "stacker_cnn1d_ss_auc0.8853.onnx",            "sequence"),
]


def build_cell_b(arch, weight_file, kind, is_ss):
    if kind == "sequence":
        if is_ss:
            return cell_b_sequence_ss(arch, weight_file)
        else:
            return cell_b_sequence(arch, weight_file)
    elif kind == "context_pkl":
        return cell_b_context_pkl(arch, weight_file, is_ss=is_ss)
    elif kind == "mlp_onnx":
        return cell_b_mlp_onnx(arch, weight_file, is_ss=is_ss)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def build_notebook(stem, arch, weight_file, kind, is_ss):
    nb = copy.deepcopy(ref_nb)
    cells = nb["cells"]

    # Step 1: Trim VLOM blend from cell 49
    cells[49] = trim_cell49(cells[49])

    # Step 2: Build the 3 new stacker cells
    stacker_dir_cell = make_code_cell(cell_stacker_dir())
    cell_a           = make_code_cell(CELL_A_SRC)
    cell_b_src       = build_cell_b(arch, weight_file, kind, is_ss)
    cell_b           = make_code_cell(cell_b_src)

    # Step 3: Replace cell 51 (submission)
    cells[51] = make_code_cell(cell_submission(arch))

    # Step 4: Insert stacker cells between cell 49 and cell 50 (markdown "Submission")
    # After insertion: ..., cell49, stacker_dir, cell_a, cell_b, cell50(markdown), cell51(new sub), cell52
    cells.insert(50, cell_b)
    cells.insert(50, cell_a)
    cells.insert(50, stacker_dir_cell)

    nb["cells"] = cells
    return nb


# ── Generate all 18 notebooks ─────────────────────────────────────────────────
all_specs = [(spec, False) for spec in NOTEBOOKS_V3] + [(spec, True) for spec in NOTEBOOKS_SS]

created = []
for (stem, arch, weight_file, kind), is_ss in all_specs:
    nb = build_notebook(stem, arch, weight_file, kind, is_ss)
    out_path = os.path.join(OUT_DIR, f"{stem}.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    created.append(out_path)
    print(f"Created: {stem}.ipynb  ({len(nb['cells'])} cells, arch={arch})")

print(f"\nDone — created {len(created)} notebooks.")
