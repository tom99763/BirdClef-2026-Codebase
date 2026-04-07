"""
train_stacker_v3.py — Extended stacking ensemble meta-learner for BirdCLEF 2026.

Trains 9 stacker architectures that combine predictions from 5 models:
  perch_raw | perch_prior_fused | mlp_probe | proto_ssm | sed_csebbs
  Feature dim: 5 × 234 = 1170

Architectures:
  1. LGBM          (context window, 3510 dim)
  2. XGBoost       (context window, 3510 dim)
  3. MLP           (context window, reshaped to (3, 5, 234))
  4. BiGRU         (sequence, (12, 1170))
  5. TCN           (sequence, (12, 1170))
  6. Transformer   (sequence, (12, 1170))
  7. SSM           (sequence, (12, 1170))
  8. FT-Transformer (sequence, (12, 1170))
  9. CNN1D         (sequence, (12, 1170))

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_stacker_v3.py

Output dir: birdclef-2026/notebook resource/current_subs 2/stacker_weights/
"""

import os
import gc
import json
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import onnxruntime as ort

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import lightgbm as lgb
from xgboost import XGBClassifier

import torchaudio
import torchaudio.transforms as T

import wandb
import openpyxl

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/lab/BirdClef-2026-Codebase")
NB_DIR      = BASE_DIR / "birdclef-2026" / "notebook resource" / "current_subs 2"
PERCH_META  = NB_DIR / "perch meta"
WEIGHTS_DIR = NB_DIR / "weights"
OUT_DIR     = NB_DIR / "stacker_weights"
OUTPUTS     = BASE_DIR / "outputs"
AUDIO_DIR   = BASE_DIR / "birdclef-2026" / "train_soundscapes"
CACHE_SED   = OUTPUTS / "stacker_train_sed_csebbs_v3.npy"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
SEED        = 42
SR          = 32_000
N_WIN       = 12           # 12 × 5s = 60s
WIN_SAMPLES = SR * 5
N_CLASSES   = 234
N_MODELS    = 5            # perch_raw | perch_prior_fused | mlp_probe | proto_ssm | sed_csebbs
EPS         = 1e-6
FEAT_DIM    = N_MODELS * N_CLASSES   # 1170

# Multi-chunk context window
CONTEXT_K    = 1            # use [t-1, t, t+1] → 3 windows
CONTEXT_SIZE = 2 * CONTEXT_K + 1   # 3
CTX_FEAT_DIM = CONTEXT_SIZE * FEAT_DIM   # 3510

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] device={DEVICE}  out={OUT_DIR}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── W&B init ──────────────────────────────────────────────────────────────────
wandb.init(
    project="birdclef-2026",
    name="stacker-v3",
    config={
        "n_models": N_MODELS,
        "n_classes": N_CLASSES,
        "n_windows": N_WIN,
        "context_k": CONTEXT_K,
        "context_size": CONTEXT_SIZE,
        "feat_dim": FEAT_DIM,
        "seed": SEED,
        "feature_layout": ["perch_raw", "perch_prior_fused", "mlp_probe", "proto_ssm", "sed_csebbs"],
        "architectures": ["lgbm", "xgb", "mlp", "bigru", "tcn", "transformer", "ssm",
                          "ft_transformer", "cnn1d"],
    },
    tags=["stacker", "meta-learner", "v3"],
)
print("[wandb] run initialized")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def safe_logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clip-safe logit transform."""
    p = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro AUC, skipping classes with no positive samples."""
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            try:
                aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# [1/7] LOAD FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/7] Loading features …")

# --- Meta: row_id + filename for each of 708 windows
meta = pd.read_parquet(PERCH_META / "full_perch_meta.parquet")   # (708, 4)
assert len(meta) == 708, f"Expected 708 rows, got {len(meta)}"
filenames_708 = meta["filename"].values       # (708,)
row_ids_708   = meta["row_id"].values         # (708,)

# Unique ordered files (59 files)
unique_files = list(dict.fromkeys(filenames_708))   # preserve order
assert len(unique_files) == 59, f"Expected 59 unique files, got {len(unique_files)}"
file_to_idx  = {f: i for i, f in enumerate(unique_files)}

# 1. perch_raw: full_perch_arrays.npz → 'scores_full_raw' (708, 234)
perch_raw_data = np.load(PERCH_META / "full_perch_arrays.npz")
perch_raw = perch_raw_data["scores_full_raw"].astype(np.float32)  # (708, 234)
perch_raw_logit = safe_logit(perch_raw)
print(f"  perch_raw   : {perch_raw.shape}  logit range [{perch_raw_logit.min():.2f}, {perch_raw_logit.max():.2f}]")

# 2. perch_prior_fused: full_oof_meta_features.npz → 'oof_base' (708, 234)
oof_data = np.load(PERCH_META / "full_oof_meta_features.npz")
perch_prior = oof_data["oof_base"].astype(np.float32)   # (708, 234) — already logit space
fold_id     = oof_data["fold_id"].astype(np.int32)       # (708,)
print(f"  perch_prior : {perch_prior.shape}  range [{perch_prior.min():.2f}, {perch_prior.max():.2f}]")

# 3. mlp_probe: try outputs/mlp_probe_oof.npy first, else use oof_data["oof_prior"]
mlp_probe_path = OUTPUTS / "mlp_probe_oof.npy"
if mlp_probe_path.exists():
    mlp_probe = np.load(mlp_probe_path).astype(np.float32)
    print(f"  mlp_probe   : {mlp_probe.shape}  (loaded from mlp_probe_oof.npy)")
else:
    # Fallback: oof_prior or just use perch_prior again
    if "oof_prior" in oof_data:
        mlp_probe = oof_data["oof_prior"].astype(np.float32)
        print(f"  mlp_probe   : {mlp_probe.shape}  (fallback: oof_prior)")
    else:
        mlp_probe = perch_prior.copy()
        print(f"  mlp_probe   : {mlp_probe.shape}  (fallback: perch_prior)")

# 4. proto_ssm: outputs/proto_ssm_oof_preds.npy (59, 234) → broadcast to (708, 234)
proto_preds_59  = np.load(OUTPUTS / "proto_ssm_oof_preds.npy").astype(np.float32)
proto_files_59  = np.load(OUTPUTS / "proto_ssm_oof_file_list.npy", allow_pickle=True)

proto_logit_708 = np.zeros((708, N_CLASSES), dtype=np.float32)
for win_i, fname in enumerate(filenames_708):
    mask = proto_files_59 == fname
    if mask.any():
        fi = np.where(mask)[0][0]
        proto_logit_708[win_i] = proto_preds_59[fi]
    # else stays 0 (neutral logit)
print(f"  proto_ssm   : {proto_logit_708.shape}  range [{proto_logit_708.min():.2f}, {proto_logit_708.max():.2f}]")

# --- Ground-truth labels: align perch_labeled_ss to 708 windows by row_id
label_data  = np.load(OUTPUTS / "perch_labeled_ss.npz", allow_pickle=True)
label_y_raw = label_data["labels"].astype(np.float32)       # (N, 234)
label_row_ids = label_data["row_ids"]                        # (N,) strings

rid_to_label = dict(zip(label_row_ids, range(len(label_row_ids))))
Y = np.zeros((708, N_CLASSES), dtype=np.float32)
missing = 0
for i, rid in enumerate(row_ids_708):
    if rid in rid_to_label:
        Y[i] = label_y_raw[rid_to_label[rid]]
    else:
        missing += 1
print(f"  labels      : {Y.shape}  missing_rows={missing}  pos_rate={Y.mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# [2/7] SED INFERENCE WITH BranchEns→cSEBBs
# ═══════════════════════════════════════════════════════════════════════════════

def build_mel_eff(audio: np.ndarray) -> np.ndarray:
    """EfficientNet mel spectrogram. Returns (3, 224, time_frames) float32 in [0,1]."""
    wav = torch.from_numpy(audio).unsqueeze(0)
    mel_tf = T.MelSpectrogram(
        sample_rate=SR, n_fft=2048, hop_length=512, n_mels=224,
        f_min=0, f_max=16000, power=2.0, norm="slaney", mel_scale="htk",
    )
    mel = mel_tf(wav)
    db_tf = T.AmplitudeToDB(top_db=80)
    mel = db_tf(mel)
    mel = mel - mel.min()
    mx  = mel.max()
    if mx > 0:
        mel = mel / mx
    mel = mel.repeat(3, 1, 1)
    return mel.numpy()


def run_onnx_batch(session: ort.InferenceSession,
                   batch: np.ndarray,
                   input_name: str) -> np.ndarray:
    return session.run(None, {input_name: batch})[0]


def apply_branchens_csebbs(probs_12: np.ndarray) -> np.ndarray:
    """
    BranchEns→cSEBBs post-processing for SED outputs.
    Input:  probs_12 (T, C) — raw sigmoid probabilities
    Output: (T, C) — smoothed probabilities
    """
    eps = 1e-7
    p = np.clip(probs_12.astype(np.float32), eps, 1.0 - eps)
    T, C = p.shape
    H = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).mean(axis=1)
    w = np.exp(-H / 0.1); w = w / w.sum() * T
    wl = np.log(p / (1.0 - p)) * w[:, None]

    def _lse_pool(wl_in: np.ndarray, beta: float) -> np.ndarray:
        out = np.zeros_like(wl_in)
        for t in range(T):
            win = wl_in[max(0, t-1):min(T, t+2)]
            mx = win.max(axis=0)
            out[t] = mx + (1.0/beta) * np.log(np.exp(beta*(win-mx)).sum(axis=0))
        return 1.0 / (1.0 + np.exp(-out))

    def _dual_anchor(lp: np.ndarray, nw: float, alpha: float) -> np.ndarray:
        anc = nw * (1.0 - np.prod(1.0-lp, axis=0)) + (1.0-nw) * lp.max(axis=0)
        return (1.0 - alpha) * lp + alpha * anc[None, :]

    out_a = _dual_anchor(_lse_pool(wl, 5.15), 0.40, 0.38)
    out_b = _dual_anchor(_lse_pool(wl, 6.0),  0.30, 0.40)
    ens = np.clip(0.55*out_a + 0.45*out_b, eps, 1.0-eps)
    out = ens.copy()
    diff = np.abs(np.diff(ens, axis=0))
    for t in range(T - 1):
        cols = np.where(diff[t] > 0.06)[0]
        if len(cols):
            seg = ens[max(0, t-2):min(T, t+3)]
            out[t, cols] = seg[:, cols].mean(axis=0)
    return out.astype(np.float32)


def infer_file_sed(audio_path: Path,
                   sess_eff: list) -> np.ndarray:
    """
    Infer SED on a 60-second soundscape, returns (12, 234) float32 probs.
    Averages predictions from multiple ONNX sessions.
    """
    import soundfile as sf
    audio, sr_in = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr_in != SR:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), sr_in, SR
        ).numpy()
    target = SR * 60
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    sed_wins = []
    for w in range(N_WIN):
        clip = audio[w * WIN_SAMPLES: (w + 1) * WIN_SAMPLES]
        mel_e = build_mel_eff(clip)[None]   # (1, 3, 224, T)
        eff_preds = []
        for sess in sess_eff:
            inp_name = sess.get_inputs()[0].name
            eff_preds.append(run_onnx_batch(sess, mel_e, inp_name))
        sed_wins.append(np.mean(eff_preds, axis=0).squeeze(0))   # (234,)

    return np.stack(sed_wins)   # (12, 234)


def build_sed_csebbs_features(unique_files_list: list) -> np.ndarray:
    """
    Run SED ONNX on 59 soundscapes, apply BranchEns→cSEBBs, return (708, 234).
    Caches result to CACHE_SED.
    """
    if CACHE_SED.exists():
        print("  [cache] loading SED cSEBBs from cache …")
        arr = np.load(CACHE_SED)
        assert arr.shape == (708, N_CLASSES), f"Bad cache shape: {arr.shape}"
        return arr

    print("  [inference] running SED + cSEBBs on 59 soundscapes …")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_eff = [
        ort.InferenceSession(str(WEIGHTS_DIR / "best_sed_b0_v5.onnx"),      providers=providers),
        ort.InferenceSession(str(WEIGHTS_DIR / "competitor_sed_fold0.onnx"), providers=providers),
    ]

    all_csebbs = []
    for fname in tqdm(unique_files_list, desc="SED+cSEBBs inference"):
        audio_path = AUDIO_DIR / fname
        raw_probs_12 = infer_file_sed(audio_path, sess_eff)   # (12, 234)
        csebbs_12    = apply_branchens_csebbs(raw_probs_12)   # (12, 234)
        all_csebbs.append(csebbs_12)

    arr = np.concatenate(all_csebbs, axis=0).astype(np.float32)   # (708, 234)
    np.save(CACHE_SED, arr)
    print(f"  cached → {CACHE_SED}")
    return arr


print("\n[2/7] Loading / running SED + cSEBBs inference …")
sed_csebbs_probs = build_sed_csebbs_features(unique_files)
sed_csebbs_logit = safe_logit(sed_csebbs_probs)
print(f"  sed_csebbs  : {sed_csebbs_logit.shape}  range [{sed_csebbs_logit.min():.2f}, {sed_csebbs_logit.max():.2f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# [3/7] BUILD FEATURE MATRIX X (708, 1170) + CONTEXT X_ctx (708, 3510)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3/7] Building feature matrix …")

# Layout: [perch_raw(0:234) | perch_prior(234:468) | mlp_probe(468:702) | proto_ssm(702:936) | sed_csebbs(936:1170)]
# perch_raw → logit space
# perch_prior → already logit
# mlp_probe → treat as logit (may already be logit or prob; safe_logit handles prob)
# proto_ssm → already logit
# sed_csebbs → logit applied above

# Determine if mlp_probe is in prob or logit space (check range)
mlp_probe_min = mlp_probe.min()
mlp_probe_max = mlp_probe.max()
if mlp_probe_min >= -0.1 and mlp_probe_max <= 1.1:
    print(f"  mlp_probe appears to be prob (range [{mlp_probe_min:.2f}, {mlp_probe_max:.2f}]) → converting to logit")
    mlp_probe_logit = safe_logit(mlp_probe)
else:
    print(f"  mlp_probe appears to be logit (range [{mlp_probe_min:.2f}, {mlp_probe_max:.2f}]) → using as-is")
    mlp_probe_logit = mlp_probe.astype(np.float32)

X = np.concatenate([
    perch_raw_logit,     # (708, 234)
    perch_prior,         # (708, 234)  — already logit
    mlp_probe_logit,     # (708, 234)
    proto_logit_708,     # (708, 234)
    sed_csebbs_logit,    # (708, 234)
], axis=1).astype(np.float32)
assert X.shape == (708, FEAT_DIM), f"Bad X shape: {X.shape}"
print(f"  X shape     : {X.shape}")

# Feature normalisation stats (save for inference)
X_mean = X.mean(axis=0, keepdims=True).astype(np.float32)   # (1, 1170)
X_std  = X.std(axis=0, keepdims=True).astype(np.float32)    # (1, 1170)
X_std[X_std < 1e-8] = 1.0

X_norm = (X - X_mean) / X_std   # (708, 1170)

np.savez(
    OUT_DIR / "stacker_norm_v3.npz",
    mean=X_mean,   # (1, 1170)
    std=X_std,     # (1, 1170)
)
print(f"  norm stats saved → {OUT_DIR / 'stacker_norm_v3.npz'}")

# Groups for GroupKFold (group = filename, 12 rows per file)
groups = np.array([file_to_idx[f] for f in filenames_708], dtype=np.int32)


def build_context_features(X_win: np.ndarray,
                           filenames: np.ndarray,
                           unique_files_list: list,
                           context_k: int = 1) -> np.ndarray:
    """
    Build context-windowed features.
    For each window t, concat [t-k, ..., t, ..., t+k]. Pad edges with zeros.

    Args:
        X_win           : (N, feat_dim)
        filenames       : (N,)
        unique_files_list: list of unique filenames
        context_k       : int
    Returns:
        X_ctx : (N, (2*context_k+1)*feat_dim)
    """
    N, F = X_win.shape
    ctx_size = 2 * context_k + 1
    X_ctx = np.zeros((N, ctx_size * F), dtype=np.float32)
    for fname in unique_files_list:
        mask = filenames == fname
        rows = np.where(mask)[0]
        Tf = len(rows)
        for local_t, row_idx in enumerate(rows):
            parts = []
            for offset in range(-context_k, context_k + 1):
                t_nb = local_t + offset
                if 0 <= t_nb < Tf:
                    parts.append(X_win[rows[t_nb]])
                else:
                    parts.append(np.zeros(F, dtype=np.float32))
            X_ctx[row_idx] = np.concatenate(parts)
    return X_ctx


# Build context features
X_ctx = build_context_features(X_norm, filenames_708, unique_files, context_k=CONTEXT_K)
print(f"  X_ctx shape : {X_ctx.shape}")   # (708, 3510)

# Prepare file-level sequences for sequence models
def win_to_file_seq(X_w: np.ndarray, files: list, filenames: np.ndarray) -> np.ndarray:
    n_files = len(files)
    F = X_w.shape[1]
    out = np.zeros((n_files, N_WIN, F), dtype=np.float32)
    for fi, fname in enumerate(files):
        mask = filenames == fname
        rows = np.where(mask)[0]
        assert len(rows) == N_WIN, f"Expected {N_WIN} windows for {fname}, got {len(rows)}"
        out[fi] = X_w[rows]
    return out


def win_to_file_labels(Y_w: np.ndarray, files: list, filenames: np.ndarray) -> np.ndarray:
    n_files = len(files)
    out = np.zeros((n_files, N_WIN, N_CLASSES), dtype=np.float32)
    for fi, fname in enumerate(files):
        mask = filenames == fname
        rows = np.where(mask)[0]
        out[fi] = Y_w[rows]
    return out


X_norm_seq = win_to_file_seq(X_norm, unique_files, filenames_708)   # (59, 12, 1170)
Y_seq      = win_to_file_labels(Y, unique_files, filenames_708)      # (59, 12, 234)

# fold_id per file (use first window's fold)
file_fold = np.array(
    [fold_id[np.where(filenames_708 == f)[0][0]] for f in unique_files],
    dtype=np.int32
)  # (59,)

print(f"  X_norm_seq  : {X_norm_seq.shape}")
print(f"  Y_seq       : {Y_seq.shape}")
print(f"  file_fold   : unique folds = {np.unique(file_fold)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CV EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def cv_eval_flat(name: str, pred_fn,
                 X_in: np.ndarray, Y_in: np.ndarray,
                 fold_id_in: np.ndarray) -> tuple:
    """CV for flat (window-level) models. Uses pre-assigned fold_id."""
    oof = np.zeros_like(Y_in, dtype=np.float32)
    fold_aucs = []
    for fv in np.unique(fold_id_in):
        va_mask = fold_id_in == fv
        tr_mask = ~va_mask
        preds = pred_fn(X_in[tr_mask], Y_in[tr_mask], X_in[va_mask])
        oof[va_mask] = preds
        fa = macro_auc(Y_in[va_mask], preds)
        fold_aucs.append(fa)
        wandb.log({"arch": name, "fold": int(fv), "fold_val_auc": fa})
    mean_auc = float(np.mean(fold_aucs))
    print(f"  [{name}] fold aucs: {[f'{a:.4f}' for a in fold_aucs]}  → mean={mean_auc:.4f}")
    wandb.log({f"{name}_oof_auc": mean_auc})
    return mean_auc, oof


def cv_eval_seq(name: str, model_cls,
                X_seq: np.ndarray, Y_seq_in: np.ndarray,
                file_fold_in: np.ndarray, **train_kwargs) -> tuple:
    """CV for file-level sequence models → OOF in (708, 234) window space."""
    oof_file  = np.zeros((len(unique_files), N_WIN, N_CLASSES), dtype=np.float32)
    fold_aucs = []
    for fv in tqdm(np.unique(file_fold_in), desc=f"{name} CV"):
        va_mask = file_fold_in == fv
        tr_mask = ~va_mask
        model = train_seq_model(
            model_cls(), X_seq[tr_mask], Y_seq_in[tr_mask],
            X_seq[va_mask], Y_seq_in[va_mask], **train_kwargs
        )
        model.eval()
        with torch.no_grad():
            preds = model(
                torch.from_numpy(X_seq[va_mask]).float().to(DEVICE)
            ).cpu().numpy()   # (n_va_files, T, 234)
        oof_file[va_mask] = preds
        yva_flat = Y_seq_in[va_mask].reshape(-1, N_CLASSES)
        pva_flat = preds.reshape(-1, N_CLASSES)
        fa = macro_auc(yva_flat, pva_flat)
        fold_aucs.append(fa)
        wandb.log({"arch": name, "fold": int(fv), "fold_val_auc": fa})

    mean_auc = float(np.mean(fold_aucs))
    print(f"  [{name}] fold aucs: {[f'{a:.4f}' for a in fold_aucs]}  → mean={mean_auc:.4f}")
    wandb.log({f"{name}_oof_auc": mean_auc})

    # Flatten back to (708, 234) window order
    oof_win = np.zeros((708, N_CLASSES), dtype=np.float32)
    for fi, fname in enumerate(unique_files):
        mask = filenames_708 == fname
        rows = np.where(mask)[0]
        oof_win[rows] = oof_file[fi]
    return mean_auc, oof_win


# ═══════════════════════════════════════════════════════════════════════════════
# [4/7] CV EVALUATION — ALL 9 ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/7] Cross-validation (9 architectures) …")
oof_aucs = {}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: equal-weight average of 5 model logits
# ─────────────────────────────────────────────────────────────────────────────

def baseline_avg_logit(X_w: np.ndarray) -> np.ndarray:
    parts = [X_w[:, i*N_CLASSES:(i+1)*N_CLASSES] for i in range(N_MODELS)]
    return np.mean(parts, axis=0)

def baseline_fn(X_tr, Y_tr, X_va):
    return baseline_avg_logit(X_va)

baseline_auc, _ = cv_eval_flat("baseline", baseline_fn, X, Y, fold_id)
oof_aucs["baseline"] = baseline_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 1: LGBM — per-class LightGBM, context features (3510 dim)
# ─────────────────────────────────────────────────────────────────────────────

def lgbm_fn(X_ctx_tr: np.ndarray, Y_tr: np.ndarray, X_ctx_va: np.ndarray) -> np.ndarray:
    """Per-class LGBMClassifier. Returns (n_va, 234) logit-like preds."""
    preds = np.zeros((len(X_ctx_va), N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        if Y_tr[:, c].sum() == 0:
            continue
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(X_ctx_tr, Y_tr[:, c])
        prob = clf.predict_proba(X_ctx_va)[:, 1].astype(np.float32)
        preds[:, c] = safe_logit(prob)
    return preds


def cv_eval_lgbm(X_norm_in: np.ndarray, Y_in: np.ndarray,
                 fold_id_in: np.ndarray,
                 filenames_in: np.ndarray,
                 unique_files_in: list) -> tuple:
    oof = np.zeros_like(Y_in, dtype=np.float32)
    fold_aucs_lgbm = []
    for fv in tqdm(np.unique(fold_id_in), desc="LGBM CV"):
        va_mask = fold_id_in == fv
        tr_mask = ~va_mask
        fn_tr = filenames_in[tr_mask]
        fn_va = filenames_in[va_mask]
        uf_tr = list(dict.fromkeys(fn_tr))
        uf_va = list(dict.fromkeys(fn_va))
        X_ctx_tr = build_context_features(X_norm_in[tr_mask], fn_tr, uf_tr, CONTEXT_K)
        X_ctx_va = build_context_features(X_norm_in[va_mask], fn_va, uf_va, CONTEXT_K)
        preds = lgbm_fn(X_ctx_tr, Y_in[tr_mask], X_ctx_va)
        oof[va_mask] = preds
        fa = macro_auc(Y_in[va_mask], preds)
        fold_aucs_lgbm.append(fa)
        wandb.log({"arch": "lgbm", "fold": int(fv), "fold_val_auc": fa})
    mean_auc = float(np.mean(fold_aucs_lgbm))
    print(f"  [lgbm] fold aucs: {[f'{a:.4f}' for a in fold_aucs_lgbm]}  → mean={mean_auc:.4f}")
    wandb.log({"lgbm_oof_auc": mean_auc})
    return mean_auc, oof


lgbm_auc, lgbm_oof = cv_eval_lgbm(X_norm, Y, fold_id, filenames_708, unique_files)
oof_aucs["lgbm"] = lgbm_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 2: XGBoost — per-class XGBClassifier, context features (3510 dim)
# ─────────────────────────────────────────────────────────────────────────────

def xgb_fn(X_ctx_tr: np.ndarray, Y_tr: np.ndarray, X_ctx_va: np.ndarray) -> np.ndarray:
    """Per-class XGBClassifier. Returns (n_va, 234) logit-like preds."""
    preds = np.zeros((len(X_ctx_va), N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        if Y_tr[:, c].sum() == 0:
            continue
        clf = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            tree_method="hist",
            device="cuda",
            random_state=SEED,
            eval_metric="logloss",
            verbosity=0,
        )
        clf.fit(X_ctx_tr, Y_tr[:, c])
        prob = clf.predict_proba(X_ctx_va)[:, 1].astype(np.float32)
        preds[:, c] = safe_logit(prob)
    return preds


def cv_eval_xgb(X_norm_in: np.ndarray, Y_in: np.ndarray,
                fold_id_in: np.ndarray,
                filenames_in: np.ndarray,
                unique_files_in: list) -> tuple:
    oof = np.zeros_like(Y_in, dtype=np.float32)
    fold_aucs_xgb = []
    for fv in tqdm(np.unique(fold_id_in), desc="XGB CV"):
        va_mask = fold_id_in == fv
        tr_mask = ~va_mask
        fn_tr = filenames_in[tr_mask]
        fn_va = filenames_in[va_mask]
        uf_tr = list(dict.fromkeys(fn_tr))
        uf_va = list(dict.fromkeys(fn_va))
        X_ctx_tr = build_context_features(X_norm_in[tr_mask], fn_tr, uf_tr, CONTEXT_K)
        X_ctx_va = build_context_features(X_norm_in[va_mask], fn_va, uf_va, CONTEXT_K)
        preds = xgb_fn(X_ctx_tr, Y_in[tr_mask], X_ctx_va)
        oof[va_mask] = preds
        fa = macro_auc(Y_in[va_mask], preds)
        fold_aucs_xgb.append(fa)
        wandb.log({"arch": "xgb", "fold": int(fv), "fold_val_auc": fa})
    mean_auc = float(np.mean(fold_aucs_xgb))
    print(f"  [xgb] fold aucs: {[f'{a:.4f}' for a in fold_aucs_xgb]}  → mean={mean_auc:.4f}")
    wandb.log({"xgb_oof_auc": mean_auc})
    return mean_auc, oof


xgb_auc, xgb_oof = cv_eval_xgb(X_norm, Y, fold_id, filenames_708, unique_files)
oof_aucs["xgb"] = xgb_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 3: MLP — per-class shared MLP with context window
# Input: (B, context_size=3, N_MODELS=5, N_CLASSES=234)
# ─────────────────────────────────────────────────────────────────────────────

class StackerMLP(nn.Module):
    """
    Per-class shared MLP stacker with context window.
    Input : (B, context_size=3, N_MODELS=5, N_CLASSES=234)
    Output: (B, N_CLASSES=234)
    """
    def __init__(self, n_models: int = N_MODELS, n_classes: int = N_CLASSES,
                 context_size: int = CONTEXT_SIZE, hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.n_models     = n_models
        self.n_classes    = n_classes
        self.context_size = context_size
        in_dim = context_size * n_models   # 3*5=15
        self.fc1     = nn.Linear(in_dim, hidden)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, context_size, N_MODELS, N_CLASSES)
        returns: (B, N_CLASSES)
        """
        B, S, M, C = x.shape
        # (B, S, M, C) → (B, C, S*M)
        x = x.permute(0, 3, 1, 2).reshape(B, C, S * M)   # (B, C, 15)
        # → (B*C, 15)
        x = x.reshape(B * C, S * M)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)        # (B*C, 1)
        return x.reshape(B, C) # (B, C)


def X_ctx_to_mlp_input(X_ctx: np.ndarray) -> torch.Tensor:
    """
    X_ctx: (B, CONTEXT_SIZE * FEAT_DIM) = (B, 3510)
    → (B, CONTEXT_SIZE=3, N_MODELS=5, N_CLASSES=234)
    """
    B = X_ctx.shape[0]
    t = torch.from_numpy(X_ctx).float()              # (B, 3510)
    t = t.reshape(B, CONTEXT_SIZE, FEAT_DIM)         # (B, 3, 1170)
    t = t.reshape(B, CONTEXT_SIZE, N_MODELS, N_CLASSES)  # (B, 3, 5, 234)
    return t


def train_mlp(X_ctx_tr: np.ndarray, Y_tr: np.ndarray,
              X_ctx_va: np.ndarray, Y_va: np.ndarray,
              epochs: int = 100, patience: int = 15,
              lr: float = 1e-3, wd: float = 1e-3) -> StackerMLP:

    model = StackerMLP().to(DEVICE)
    pos_rate = Y_tr.mean(axis=0)
    pos_w    = np.clip((1.0 - pos_rate) / np.maximum(pos_rate, 1e-6), 0, 20).astype(np.float32)
    pw_tensor = torch.from_numpy(pos_w).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optim     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    Xtr_t = X_ctx_to_mlp_input(X_ctx_tr).to(DEVICE)
    Ytr_t = torch.from_numpy(Y_tr).float().to(DEVICE)
    Xva_t = X_ctx_to_mlp_input(X_ctx_va).to(DEVICE)

    ds     = TensorDataset(Xtr_t, Ytr_t)
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)

    best_auc  = 0.0
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                va_preds = model(Xva_t).cpu().numpy()
            auc = macro_auc(Y_va, va_preds)
            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


def cv_eval_mlp(X_norm_in: np.ndarray, Y_in: np.ndarray,
                fold_id_in: np.ndarray,
                filenames_in: np.ndarray,
                unique_files_in: list) -> tuple:
    oof = np.zeros_like(Y_in, dtype=np.float32)
    fold_aucs_mlp = []
    for fv in tqdm(np.unique(fold_id_in), desc="MLP CV"):
        va_mask = fold_id_in == fv
        tr_mask = ~va_mask
        fn_tr = filenames_in[tr_mask]
        fn_va = filenames_in[va_mask]
        uf_tr = list(dict.fromkeys(fn_tr))
        uf_va = list(dict.fromkeys(fn_va))
        X_ctx_tr = build_context_features(X_norm_in[tr_mask], fn_tr, uf_tr, CONTEXT_K)
        X_ctx_va = build_context_features(X_norm_in[va_mask], fn_va, uf_va, CONTEXT_K)
        model = train_mlp(X_ctx_tr, Y_in[tr_mask], X_ctx_va, Y_in[va_mask])
        model.eval()
        with torch.no_grad():
            preds = model(X_ctx_to_mlp_input(X_ctx_va).to(DEVICE)).cpu().numpy()
        oof[va_mask] = preds
        fa = macro_auc(Y_in[va_mask], preds)
        fold_aucs_mlp.append(fa)
        wandb.log({"arch": "mlp", "fold": int(fv), "fold_val_auc": fa})
    mean_auc = float(np.mean(fold_aucs_mlp))
    print(f"  [mlp] fold aucs: {[f'{a:.4f}' for a in fold_aucs_mlp]}  → mean={mean_auc:.4f}")
    wandb.log({"mlp_oof_auc": mean_auc})
    return mean_auc, oof


mlp_auc, mlp_oof = cv_eval_mlp(X_norm, Y, fold_id, filenames_708, unique_files)
oof_aucs["mlp"] = mlp_auc


# ─────────────────────────────────────────────────────────────────────────────
# Generic sequence model trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_seq_model(model: nn.Module,
                    X_tr_seq: np.ndarray, Y_tr_seq: np.ndarray,
                    X_va_seq: np.ndarray, Y_va_seq: np.ndarray,
                    epochs: int = 150, patience: int = 15,
                    lr: float = 5e-4, wd: float = 1e-3,
                    batch_size: int = 16) -> nn.Module:
    model = model.to(DEVICE)
    pos_rate  = Y_tr_seq.mean(axis=(0, 1))
    pos_w     = np.clip((1.0 - pos_rate) / np.maximum(pos_rate, 1e-6), 0, 20).astype(np.float32)
    pw_tensor = torch.from_numpy(pos_w).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optim_    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=epochs)

    Xtr = torch.from_numpy(X_tr_seq).float()
    Ytr = torch.from_numpy(Y_tr_seq).float()
    Xva = torch.from_numpy(X_va_seq).float().to(DEVICE)
    Yva_np = Y_va_seq

    ds     = TensorDataset(Xtr, Ytr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_auc   = 0.0
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim_.zero_grad()
            pred = model(xb)   # (B, T, 234)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                va_pred = model(Xva).cpu().numpy()   # (n_va, T, 234)
            va_flat  = va_pred.reshape(-1, N_CLASSES)
            yva_flat = Yva_np.reshape(-1, N_CLASSES)
            auc = macro_auc(yva_flat, va_flat)
            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Arch 4: BiGRU
# ─────────────────────────────────────────────────────────────────────────────

class StackerBiGRU(nn.Module):
    """File-level BiGRU stacker. Input: (B, T=12, 1170) → Output: (B, T, 234)"""
    def __init__(self, in_features: int = FEAT_DIM, hidden: int = 128,
                 n_layers: int = 2, n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden)
        self.gru = nn.GRU(hidden, hidden // 2, num_layers=n_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1170) → (B, T, 234)"""
        x = self.in_proj(x)          # (B, T, 128)
        x, _ = self.gru(x)           # (B, T, 128)
        return self.out_proj(x)      # (B, T, 234)


bigru_auc, bigru_oof = cv_eval_seq(
    "bigru", StackerBiGRU, X_norm_seq, Y_seq, file_fold,
    epochs=150, patience=15, lr=5e-4, wd=1e-3, batch_size=16
)
oof_aucs["bigru"] = bigru_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 5: TCN (Temporal Convolutional Network)
# ─────────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    """Causal dilated TCN block."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self._pad = pad
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.norm1    = nn.LayerNorm(out_ch)
        self.norm2    = nn.LayerNorm(out_ch)
        self.drop     = nn.Dropout(dropout)
        self.residual = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C)"""
        res = self.residual(x)
        T   = x.shape[1]
        h   = x.transpose(1, 2)              # (B, C, T)
        h   = self.conv1(h)[..., :T]         # causal: remove right padding
        h   = h.transpose(1, 2)              # (B, T, C)
        h   = self.drop(F.gelu(self.norm1(h)))
        h   = h.transpose(1, 2)              # (B, C, T)
        h   = self.conv2(h)[..., :T]
        h   = h.transpose(1, 2)              # (B, T, C)
        h   = self.drop(F.gelu(self.norm2(h)))
        return h + res


class StackerTCN(nn.Module):
    """File-level TCN stacker. Input: (B, T=12, 1170) → Output: (B, T, 234)"""
    def __init__(self, in_features: int = FEAT_DIM, channels: int = 128,
                 n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_features, channels)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(channels, channels, kernel_size=3, dilation=d, dropout=dropout)
            for d in [1, 2, 4]
        ])
        self.out_proj = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1170) → (B, T, 234)"""
        x = self.in_proj(x)            # (B, T, 128)
        for block in self.tcn_blocks:
            x = block(x)               # (B, T, 128)
        return self.out_proj(x)        # (B, T, 234)


tcn_auc, tcn_oof = cv_eval_seq(
    "tcn", StackerTCN, X_norm_seq, Y_seq, file_fold,
    epochs=150, patience=15, lr=5e-4, wd=1e-3, batch_size=16
)
oof_aucs["tcn"] = tcn_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 6: Transformer
# ─────────────────────────────────────────────────────────────────────────────

class StackerTransformer(nn.Module):
    """File-level Transformer stacker. Input: (B, T=12, 1170) → Output: (B, T, 234)"""
    def __init__(self, in_features: int = FEAT_DIM, d_model: int = 128,
                 nhead: int = 4, dim_ff: int = 256, n_layers: int = 2,
                 n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, N_WIN, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1170) → (B, T, 234)"""
        x = self.in_proj(x) + self.pos_emb   # (B, T, 128)
        x = self.encoder(x)
        return self.out_proj(x)              # (B, T, 234)


tfm_auc, tfm_oof = cv_eval_seq(
    "transformer", StackerTransformer, X_norm_seq, Y_seq, file_fold,
    epochs=150, patience=15, lr=5e-4, wd=1e-3, batch_size=16
)
oof_aucs["transformer"] = tfm_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 7: SSM (Mamba-style selective SSM)
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveSSMLayer(nn.Module):
    """Simplified selective SSM (Mamba-inspired) layer. Input/output: (B, T, d_model)"""
    def __init__(self, d_model: int = 128, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, 2 * d_model)
        self.conv1d   = nn.Conv1d(d_model, d_model, kernel_size=3,
                                   padding=1, groups=d_model)
        self.x_proj   = nn.Linear(d_model, d_state * 2 + 1)
        self.dt_proj  = nn.Linear(1, d_model)
        self.A_log    = nn.Parameter(torch.randn(d_model, d_state))
        self.D        = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

    def ssm_scan(self, x: torch.Tensor, B_p: torch.Tensor,
                 C_p: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Causal discretised SSM scan. x:(B,T,d), B_p/C_p:(B,T,d_state), dt:(B,T,d)."""
        batch, T, d = x.shape
        A = -torch.exp(self.A_log.float())
        h = torch.zeros(batch, d, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = F.softplus(dt[:, t, :])
            dA   = torch.exp(dt_t.unsqueeze(-1) * A)
            dB   = dt_t.unsqueeze(-1) * B_p[:, t, :].unsqueeze(1)
            h    = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y    = (h * C_p[:, t, :].unsqueeze(1)).sum(-1)
            ys.append(y)
        return torch.stack(ys, dim=1)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_in = self.norm(x_in)
        xz   = self.in_proj(x_in)
        x, z = xz.chunk(2, dim=-1)
        x    = x.transpose(1, 2)
        x    = self.conv1d(x).transpose(1, 2)
        params = self.x_proj(x)
        B_p = params[..., :self.d_state]
        C_p = params[..., self.d_state:2 * self.d_state]
        dt  = self.dt_proj(params[..., -1:])
        y   = self.ssm_scan(x, B_p, C_p, dt)
        y   = y * F.silu(z)
        y   = self.out_proj(y)
        return x_in + self.drop(y)


class StackerSSM(nn.Module):
    """File-level SSM stacker. Input: (B, T=12, 1170) → Output: (B, T, 234)"""
    def __init__(self, in_features: int = FEAT_DIM, d_model: int = 128,
                 d_state: int = 16, n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj  = nn.Linear(in_features, d_model)
        self.ssm1     = SelectiveSSMLayer(d_model, d_state, dropout)
        self.ssm2     = SelectiveSSMLayer(d_model, d_state, dropout)
        self.out_proj = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1170) → (B, T, 234)"""
        x = self.in_proj(x)     # (B, T, 128)
        x = self.ssm1(x)
        x = self.ssm2(x)
        return self.out_proj(x) # (B, T, 234)


ssm_auc, ssm_oof = cv_eval_seq(
    "ssm", StackerSSM, X_norm_seq, Y_seq, file_fold,
    epochs=150, patience=15, lr=5e-4, wd=1e-3, batch_size=16
)
oof_aucs["ssm"] = ssm_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 8: FT-Transformer (Feature Tokenizer Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class StackerFTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular data.
    Each of the 5 model predictions is tokenized separately.
    Input: (B, T=12, 1170) → treat as (B*T, 5, 234) feature tokens.
    Each model's 234-dim vector is projected to d_model via a learned tokenizer.
    Then Transformer encoder on 5 tokens → pooled → (B*T, 234) output.
    """
    def __init__(self, n_models: int = N_MODELS, n_classes: int = N_CLASSES,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_models  = n_models
        self.n_classes = n_classes
        self.d_model   = d_model
        # Shared tokenizer: maps each model's (n_classes,) → (d_model,)
        self.tokenizer = nn.Linear(n_classes, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 2, dropout, batch_first=True, activation="gelu"
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, n_layers)
        # Concat all token outputs
        self.out_proj = nn.Linear(d_model * n_models, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1170) → (B, T, 234)"""
        B, T, _ = x.shape
        # Reshape to (B*T, 5, 234)
        x = x.view(B * T, self.n_models, self.n_classes)
        # Tokenize: (B*T, 5, d_model)
        tokens = self.tokenizer(x)
        # Transformer: (B*T, 5, d_model)
        out = self.encoder(tokens)
        # Concat all tokens: (B*T, 5*d_model)
        out = out.reshape(B * T, -1)
        # Project: (B*T, 234)
        out = self.out_proj(out)
        # Reshape: (B, T, 234)
        return out.view(B, T, self.n_classes)


ft_tfm_auc, ft_tfm_oof = cv_eval_seq(
    "ft_transformer", StackerFTTransformer, X_norm_seq, Y_seq, file_fold,
    epochs=150, patience=15, lr=3e-4, wd=1e-3, batch_size=16
)
oof_aucs["ft_transformer"] = ft_tfm_auc


# ─────────────────────────────────────────────────────────────────────────────
# Arch 9: CNN1D (Lightweight 1D CNN)
# ─────────────────────────────────────────────────────────────────────────────

class StackerCNN1D(nn.Module):
    """
    1D CNN over temporal dimension. Lightweight baseline.
    Input: (B, T=12, 1170) → Output: (B, T, 234)
    """
    def __init__(self, in_features: int = FEAT_DIM, channels: int = 128,
                 n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_features, channels)
        self.conv1   = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm    = nn.LayerNorm(channels)
        self.drop    = nn.Dropout(dropout)
        self.out_proj = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1170) → (B, T, 234)"""
        res = self.in_proj(x)           # (B, T, 128)
        h   = res.transpose(1, 2)       # (B, 128, T)
        h   = F.gelu(self.conv1(h) + self.conv2(h))  # (B, 128, T)
        h   = h.transpose(1, 2)         # (B, T, 128)
        h   = self.drop(self.norm(h))
        return self.out_proj(res + h)   # residual + (B, T, 234)


cnn1d_auc, cnn1d_oof = cv_eval_seq(
    "cnn1d", StackerCNN1D, X_norm_seq, Y_seq, file_fold,
    epochs=150, patience=15, lr=5e-4, wd=1e-3, batch_size=16
)
oof_aucs["cnn1d"] = cnn1d_auc


# ═══════════════════════════════════════════════════════════════════════════════
# [5/7] FINAL FIT ON ALL DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5/7] Final fit on all data …")

best_arch = max(
    [k for k in oof_aucs if k != "baseline"],
    key=lambda k: oof_aucs[k]
)
print(f"  best architecture: {best_arch}  (AUC={oof_aucs[best_arch]:.4f})")


# ── LGBM (full fit) ───────────────────────────────────────────────────────────
print("  Fitting LGBM (full) …")
lgbm_models = []
for c in tqdm(range(N_CLASSES), desc="LGBM full fit"):
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        n_jobs=4,
        verbose=-1,
    )
    if Y[:, c].sum() > 0:
        clf.fit(X_ctx, Y[:, c])
    lgbm_models.append(clf)
with open(OUT_DIR / "stacker_lgbm_v3.pkl", "wb") as f:
    pickle.dump(lgbm_models, f)
print(f"  LGBM saved → {OUT_DIR / 'stacker_lgbm_v3.pkl'}")


# ── XGBoost (full fit) ────────────────────────────────────────────────────────
print("  Fitting XGBoost (full) …")
xgb_models = []
for c in tqdm(range(N_CLASSES), desc="XGB full fit"):
    clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        eval_metric="logloss",
        verbosity=0,
    )
    if Y[:, c].sum() > 0:
        clf.fit(X_ctx, Y[:, c])
    xgb_models.append(clf)
with open(OUT_DIR / "stacker_xgb_v3.pkl", "wb") as f:
    pickle.dump(xgb_models, f)
print(f"  XGB saved → {OUT_DIR / 'stacker_xgb_v3.pkl'}")


# ── MLP (full fit + ONNX export) ─────────────────────────────────────────────
print("  Fitting MLP (full) …")
mlp_model = train_mlp(X_ctx, Y, X_ctx, Y, epochs=100, patience=100)
mlp_model.eval()
torch.save(mlp_model.state_dict(), OUT_DIR / "stacker_mlp_v3.pt")

dummy_mlp = torch.zeros(1, CONTEXT_SIZE, N_MODELS, N_CLASSES)
torch.onnx.export(
    mlp_model.cpu(), dummy_mlp,
    str(OUT_DIR / "stacker_mlp_v3.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=14,
)
print(f"  MLP saved → {OUT_DIR / 'stacker_mlp_v3.onnx'}")
mlp_model = mlp_model.to(DEVICE)


# ── BiGRU (full fit + ONNX export) ────────────────────────────────────────────
print("  Fitting BiGRU (full) …")
bigru_model = train_seq_model(
    StackerBiGRU(), X_norm_seq, Y_seq, X_norm_seq, Y_seq,
    epochs=150, patience=150, lr=5e-4, wd=1e-3, batch_size=16
)
bigru_model.eval()
torch.save(bigru_model.state_dict(), OUT_DIR / "stacker_bigru_v3.pt")
dummy_seq = torch.zeros(1, N_WIN, FEAT_DIM)
torch.onnx.export(
    bigru_model.cpu(), dummy_seq,
    str(OUT_DIR / "stacker_bigru_v3.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  BiGRU saved → {OUT_DIR / 'stacker_bigru_v3.onnx'}")
bigru_model = bigru_model.to(DEVICE)


# ── TCN (full fit + ONNX export) ─────────────────────────────────────────────
print("  Fitting TCN (full) …")
tcn_model = train_seq_model(
    StackerTCN(), X_norm_seq, Y_seq, X_norm_seq, Y_seq,
    epochs=150, patience=150, lr=5e-4, wd=1e-3, batch_size=16
)
tcn_model.eval()
torch.save(tcn_model.state_dict(), OUT_DIR / "stacker_tcn_v3.pt")
torch.onnx.export(
    tcn_model.cpu(), dummy_seq,
    str(OUT_DIR / "stacker_tcn_v3.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  TCN saved → {OUT_DIR / 'stacker_tcn_v3.onnx'}")
tcn_model = tcn_model.to(DEVICE)


# ── Transformer (full fit + ONNX export) ─────────────────────────────────────
print("  Fitting Transformer (full) …")
tfm_model = train_seq_model(
    StackerTransformer(), X_norm_seq, Y_seq, X_norm_seq, Y_seq,
    epochs=150, patience=150, lr=5e-4, wd=1e-3, batch_size=16
)
tfm_model.eval()
torch.save(tfm_model.state_dict(), OUT_DIR / "stacker_transformer_v3.pt")
torch.onnx.export(
    tfm_model.cpu(), dummy_seq,
    str(OUT_DIR / "stacker_transformer_v3.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  Transformer saved → {OUT_DIR / 'stacker_transformer_v3.onnx'}")
tfm_model = tfm_model.to(DEVICE)


# ── SSM (full fit + ONNX export) ─────────────────────────────────────────────
print("  Fitting SSM (full) …")
ssm_model = train_seq_model(
    StackerSSM(), X_norm_seq, Y_seq, X_norm_seq, Y_seq,
    epochs=150, patience=150, lr=5e-4, wd=1e-3, batch_size=16
)
ssm_model.eval()
torch.save(ssm_model.state_dict(), OUT_DIR / "stacker_ssm_v3.pt")
torch.onnx.export(
    ssm_model.cpu(), dummy_seq,
    str(OUT_DIR / "stacker_ssm_v3.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  SSM saved → {OUT_DIR / 'stacker_ssm_v3.onnx'}")
ssm_model = ssm_model.to(DEVICE)


# ── FT-Transformer (full fit + ONNX export) ──────────────────────────────────
print("  Fitting FT-Transformer (full) …")
ft_tfm_model = train_seq_model(
    StackerFTTransformer(), X_norm_seq, Y_seq, X_norm_seq, Y_seq,
    epochs=150, patience=150, lr=3e-4, wd=1e-3, batch_size=16
)
ft_tfm_model.eval()
torch.save(ft_tfm_model.state_dict(), OUT_DIR / "stacker_ft_transformer_v3.pt")
torch.onnx.export(
    ft_tfm_model.cpu(), dummy_seq,
    str(OUT_DIR / "stacker_ft_transformer_v3.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  FT-Transformer saved → {OUT_DIR / 'stacker_ft_transformer_v3.onnx'}")
ft_tfm_model = ft_tfm_model.to(DEVICE)


# ── CNN1D (full fit + ONNX export) ────────────────────────────────────────────
print("  Fitting CNN1D (full) …")
cnn1d_model = train_seq_model(
    StackerCNN1D(), X_norm_seq, Y_seq, X_norm_seq, Y_seq,
    epochs=150, patience=150, lr=5e-4, wd=1e-3, batch_size=16
)
cnn1d_model.eval()
torch.save(cnn1d_model.state_dict(), OUT_DIR / "stacker_cnn1d_v3.pt")
torch.onnx.export(
    cnn1d_model.cpu(), dummy_seq,
    str(OUT_DIR / "stacker_cnn1d_v3.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  CNN1D saved → {OUT_DIR / 'stacker_cnn1d_v3.onnx'}")


# ═══════════════════════════════════════════════════════════════════════════════
# [6/7] SAVE WEIGHTS + META JSON
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[6/7] Saving meta JSON …")
meta_dict = {
    "best_arch"       : best_arch,
    "oof_aucs"        : {k: round(v, 6) for k, v in oof_aucs.items()},
    "n_models"        : N_MODELS,
    "n_classes"       : N_CLASSES,
    "n_windows"       : N_WIN,
    "context_k"       : CONTEXT_K,
    "context_size"    : CONTEXT_SIZE,
    "feature_layout"  : ["perch_raw", "perch_prior_fused", "mlp_probe", "proto_ssm", "sed_csebbs"],
    "feature_dim"     : FEAT_DIM,
    "context_feat_dim": CTX_FEAT_DIM,
    "temperature"     : 1.5,
    "architectures"   : ["lgbm", "xgb", "mlp", "bigru", "tcn", "transformer",
                         "ssm", "ft_transformer", "cnn1d"],
    "trained_date"    : time.strftime("%Y-%m-%d"),
}
with open(OUT_DIR / "stacker_meta_v3.json", "w") as f:
    json.dump(meta_dict, f, indent=2)
print(f"  meta saved → {OUT_DIR / 'stacker_meta_v3.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# [7/7] W&B + EXCEL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  OOF AUC SUMMARY")
print("=" * 65)
print(f"  {'Model':<20} {'OOF macro AUC':>14}")
print("-" * 45)
arch_list = ["baseline", "lgbm", "xgb", "mlp", "bigru", "tcn",
             "transformer", "ssm", "ft_transformer", "cnn1d"]
for k in arch_list:
    marker = " ◀ best" if k == best_arch else ""
    auc_val = oof_aucs.get(k, float("nan"))
    print(f"  {k:<20} {auc_val:>14.4f}{marker}")
print("=" * 65)
print(f"\n[done]  All artifacts saved to: {OUT_DIR}")

# ─── Excel export ──────────────────────────────────────────────────────────────
rows = []
for k in arch_list:
    rows.append({
        "arch"         : k,
        "oof_macro_auc": round(oof_aucs.get(k, float("nan")), 6),
        "best"         : k == best_arch,
        "trained_date" : time.strftime("%Y-%m-%d"),
    })
df_results = pd.DataFrame(rows)
excel_path = OUT_DIR / "stacker_results_v3.xlsx"

if excel_path.exists():
    with pd.ExcelWriter(str(excel_path), engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        df_results.to_excel(writer, sheet_name="OOF_AUC", index=False)
else:
    df_results.to_excel(str(excel_path), index=False, sheet_name="OOF_AUC")
print(f"  Excel saved → {excel_path}")

# ─── W&B final summary + finish ───────────────────────────────────────────────
wandb.log({
    "summary/best_arch"    : best_arch,
    "summary/best_oof_auc" : oof_aucs[best_arch],
    **{f"summary/oof_{k}": oof_aucs.get(k, float("nan")) for k in arch_list},
})
wandb.finish()
print("[wandb] run finished")
