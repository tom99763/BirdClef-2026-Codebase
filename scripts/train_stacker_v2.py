"""
train_stacker_v2.py — Stacking ensemble meta-learner for BirdCLEF 2026.

Trains 4 stacker architectures (Ridge, MLP, SSM, Transformer) that combine
per-5s-window predictions from Perch, ProtoSSM, SED ensemble, and HGNet.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_stacker_v2.py

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

from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import torchaudio
import torchaudio.transforms as T

import wandb
import openpyxl

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/home/lab/BirdClef-2026-Codebase")
NB_DIR     = BASE_DIR / "birdclef-2026" / "notebook resource" / "current_subs 2"
PERCH_META = NB_DIR / "perch meta"
WEIGHTS_DIR = NB_DIR / "weights"
OUT_DIR    = NB_DIR / "stacker_weights"
OUTPUTS    = BASE_DIR / "outputs"
AUDIO_DIR  = BASE_DIR / "birdclef-2026" / "train_soundscapes"
CACHE_SED  = OUTPUTS / "stacker_train_sed_win.npy"
CACHE_HGNET = OUTPUTS / "stacker_train_hgnet_win.npy"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
SEED       = 42
SR         = 32_000
N_WIN      = 12           # 12 × 5s = 60s
WIN_SAMPLES = SR * 5
N_CLASSES  = 234
N_MODELS   = 4            # perch | proto_ssm | sed_ens | hgnet
EPS        = 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] device={DEVICE}  out={OUT_DIR}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── W&B init ──────────────────────────────────────────────────────────────────
wandb.init(
    project="birdclef-2026",
    name="stacker-v2",
    config={
        "n_models": N_MODELS,
        "n_classes": N_CLASSES,
        "n_windows": N_WIN,
        "seed": SEED,
        "feature_layout": ["perch_oof", "proto_ssm_oof", "sed_ensemble", "hgnet"],
    },
    tags=["stacker", "meta-learner"],
)
print("[wandb] run initialized")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD STATIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def safe_logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clip-safe logit transform."""
    p = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


print("\n[1/6] Loading static features …")

# --- Perch OOF (708, 234) already in logit space
oof_data  = np.load(PERCH_META / "full_oof_meta_features.npz")
perch_oof = oof_data["oof_base"].astype(np.float32)   # (708, 234)
fold_id   = oof_data["fold_id"].astype(np.int32)       # (708,)
print(f"  perch_oof   : {perch_oof.shape}  range [{perch_oof.min():.2f}, {perch_oof.max():.2f}]")

# --- Meta: row_id + filename for each of 708 windows
meta = pd.read_parquet(PERCH_META / "full_perch_meta.parquet")   # (708, 4)
assert len(meta) == 708, f"Expected 708 rows, got {len(meta)}"
filenames_708 = meta["filename"].values       # (708,) e.g. "BC2026_…ogg"
row_ids_708   = meta["row_id"].values         # (708,)

# Unique ordered files (59 files)
unique_files = list(dict.fromkeys(filenames_708))   # preserve order
assert len(unique_files) == 59, f"Expected 59 unique files, got {len(unique_files)}"
file_to_idx  = {f: i for i, f in enumerate(unique_files)}

# --- ProtoSSM OOF (59, 234) → logit → broadcast to (708, 234)
proto_preds_59  = np.load(OUTPUTS / "proto_ssm_oof_preds.npy").astype(np.float32)  # logit-range already
proto_files_59  = np.load(OUTPUTS / "proto_ssm_oof_file_list.npy", allow_pickle=True)

# Build proto logit array aligned to 708 windows
proto_logit_708 = np.zeros((708, N_CLASSES), dtype=np.float32)
for win_i, fname in enumerate(filenames_708):
    # find matching file in proto list
    mask = proto_files_59 == fname
    if mask.any():
        fi = np.where(mask)[0][0]
        proto_logit_708[win_i] = proto_preds_59[fi]
    # else stays 0 (neutral logit)

print(f"  proto_logit : {proto_logit_708.shape}  range [{proto_logit_708.min():.2f}, {proto_logit_708.max():.2f}]")

# --- Ground-truth labels: align perch_labeled_ss to 708 windows by row_id
label_data  = np.load(OUTPUTS / "perch_labeled_ss.npz", allow_pickle=True)
label_y_raw = label_data["labels"].astype(np.float32)       # (739, 234)
label_row_ids = label_data["row_ids"]                        # (739,) strings

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
# 2.  SED + HGNet INFERENCE ON 59 TRAIN SOUNDSCAPES
# ═══════════════════════════════════════════════════════════════════════════════

def build_mel_eff(audio: np.ndarray) -> np.ndarray:
    """
    EfficientNet mel spectrogram.
    Returns (3, 224, time_frames) float32 in [0,1].
    """
    wav = torch.from_numpy(audio).unsqueeze(0)   # (1, samples)
    mel_tf = T.MelSpectrogram(
        sample_rate=SR, n_fft=2048, hop_length=512, n_mels=224,
        f_min=0, f_max=16000, power=2.0, norm="slaney", mel_scale="htk",
    )
    mel = mel_tf(wav)                             # (1, 224, T)
    db_tf = T.AmplitudeToDB(top_db=80)
    mel = db_tf(mel)                              # (1, 224, T) in dB
    mel = mel - mel.min()
    mx  = mel.max()
    if mx > 0:
        mel = mel / mx
    mel = mel.repeat(3, 1, 1)                    # (3, 224, T)
    return mel.numpy()


def build_mel_hgnet(audio: np.ndarray) -> np.ndarray:
    """
    HGNet mel spectrogram.
    Returns (1, 256, 256) float32 in [0,1].
    """
    wav = torch.from_numpy(audio).unsqueeze(0)
    mel_tf = T.MelSpectrogram(
        sample_rate=SR, n_fft=2048, win_length=626, hop_length=313,
        n_mels=256, f_min=20, power=2.0,
    )
    mel = mel_tf(wav)                             # (1, 256, T)
    db_tf = T.AmplitudeToDB(top_db=80)
    mel = db_tf(mel)
    mel = mel - mel.min()
    mx  = mel.max()
    if mx > 0:
        mel = mel / mx
    # Resize to (1, 256, 256)
    mel = F.interpolate(
        mel.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
    ).squeeze(0)
    return mel.numpy()


def run_onnx_batch(session: ort.InferenceSession,
                   batch: np.ndarray,
                   input_name: str) -> np.ndarray:
    return session.run(None, {input_name: batch})[0]


def infer_file(audio_path: Path,
               sess_eff: list[ort.InferenceSession],
               sess_hgnet: ort.InferenceSession,
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer on a 60-second soundscape.
    Returns:
        sed_preds  : (12, 234) float32 – averaged EfficientNet probs
        hgnet_preds: (12, 234) float32 – HGNet probs
    """
    import soundfile as sf
    audio, sr_in = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr_in != SR:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), sr_in, SR
        ).numpy()
    # Pad or trim to exactly 60 s
    target = SR * 60
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    sed_wins   = []
    hgnet_wins = []

    for w in range(N_WIN):
        clip = audio[w * WIN_SAMPLES: (w + 1) * WIN_SAMPLES]

        # EfficientNet SED (averaged across multiple sessions)
        mel_e = build_mel_eff(clip)[None]   # (1, 3, 224, T)
        eff_preds = []
        for sess in sess_eff:
            inp_name = sess.get_inputs()[0].name
            eff_preds.append(run_onnx_batch(sess, mel_e, inp_name))
        sed_wins.append(np.mean(eff_preds, axis=0).squeeze(0))   # (234,)

        # HGNet
        mel_h = build_mel_hgnet(clip)[None]  # (1, 1, 256, 256)
        inp_name_h = sess_hgnet.get_inputs()[0].name
        hgnet_wins.append(
            run_onnx_batch(sess_hgnet, mel_h, inp_name_h).squeeze(0)
        )

    return np.stack(sed_wins), np.stack(hgnet_wins)   # each (12, 234)


def build_sed_hgnet_features(unique_files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SED + HGNet on 59 soundscapes → (708, 234) each.
    Caches results to disk.
    """
    if CACHE_SED.exists() and CACHE_HGNET.exists():
        print("  [cache] loading SED + HGNet from cache …")
        sed   = np.load(CACHE_SED)
        hgnet = np.load(CACHE_HGNET)
        assert sed.shape == (708, N_CLASSES) and hgnet.shape == (708, N_CLASSES)
        return sed, hgnet

    print("  [inference] running SED + HGNet on 59 soundscapes …")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_eff = [
        ort.InferenceSession(str(WEIGHTS_DIR / "best_sed_b0_v5.onnx"),     providers=providers),
        ort.InferenceSession(str(WEIGHTS_DIR / "competitor_sed_fold0.onnx"), providers=providers),
    ]
    sess_hgnet = ort.InferenceSession(
        str(WEIGHTS_DIR / "hgnet_ss_v1_fold0.onnx"), providers=providers
    )

    all_sed   = []
    all_hgnet = []
    for fname in tqdm(unique_files, desc="SED+HGNet inference"):
        audio_path = AUDIO_DIR / fname
        s, h = infer_file(audio_path, sess_eff, sess_hgnet)   # (12, 234) each
        all_sed.append(s)
        all_hgnet.append(h)

    sed_arr   = np.concatenate(all_sed,   axis=0).astype(np.float32)   # (708, 234)
    hgnet_arr = np.concatenate(all_hgnet, axis=0).astype(np.float32)   # (708, 234)

    np.save(CACHE_SED,   sed_arr)
    np.save(CACHE_HGNET, hgnet_arr)
    print(f"  cached → {CACHE_SED}")
    return sed_arr, hgnet_arr


print("\n[2/6] Running / loading SED + HGNet inference …")
sed_probs, hgnet_probs = build_sed_hgnet_features(unique_files)
sed_logit   = safe_logit(sed_probs)
hgnet_logit = safe_logit(hgnet_probs)
print(f"  sed_logit   : {sed_logit.shape}  range [{sed_logit.min():.2f}, {sed_logit.max():.2f}]")
print(f"  hgnet_logit : {hgnet_logit.shape}  range [{hgnet_logit.min():.2f}, {hgnet_logit.max():.2f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD FEATURE MATRIX X  (708, 936)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3/6] Building feature matrix …")

# Layout: [perch(0:234) | proto(234:468) | sed(468:702) | hgnet(702:936)]
X = np.concatenate([perch_oof, proto_logit_708, sed_logit, hgnet_logit], axis=1).astype(np.float32)
assert X.shape == (708, N_MODELS * N_CLASSES), f"Bad X shape: {X.shape}"
print(f"  X shape     : {X.shape}")

# --- Feature normalisation stats (save for inference)
X_mean = X.mean(axis=0, keepdims=True).astype(np.float32)   # (1, 936)
X_std  = X.std(axis=0, keepdims=True).astype(np.float32)    # (1, 936)
X_std[X_std < 1e-8] = 1.0

X_norm = (X - X_mean) / X_std   # normalised copy used by MLP / SSM / Transformer

np.savez(OUT_DIR / "stacker_feature_stats.npz", mean=X_mean, std=X_std)
print(f"  feature stats saved → {OUT_DIR / 'stacker_feature_stats.npz'}")

# Groups for GroupKFold (group = filename, 12 rows per file)
groups = np.array([file_to_idx[f] for f in filenames_708], dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

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


def baseline_avg_logit(X: np.ndarray) -> np.ndarray:
    """Equal-weight average of 4 model logits."""
    p = X[:, :N_CLASSES]       # perch
    r = X[:, N_CLASSES:2*N_CLASSES]    # proto
    s = X[:, 2*N_CLASSES:3*N_CLASSES]  # sed
    h = X[:, 3*N_CLASSES:]    # hgnet
    return (p + r + s + h) / 4.0


def cv_eval(name: str,
            pred_fn,        # fn(X_tr, Y_tr, X_va) -> np.ndarray (n_va, 234)
            X: np.ndarray,
            Y: np.ndarray,
            fold_id: np.ndarray,
            n_folds: int = 5) -> tuple[float, np.ndarray]:
    """
    Group-file cross-validation using pre-assigned fold_id.
    Returns (mean_oof_auc, oof_preds).
    """
    oof = np.zeros_like(Y, dtype=np.float32)
    fold_aucs = []

    unique_folds = np.unique(fold_id)
    for fv in unique_folds:
        va_mask = fold_id == fv
        tr_mask = ~va_mask
        X_tr, Y_tr = X[tr_mask], Y[tr_mask]
        X_va, Y_va = X[va_mask], Y[va_mask]

        preds = pred_fn(X_tr, Y_tr, X_va)
        oof[va_mask] = preds
        fa = macro_auc(Y_va, preds)
        fold_aucs.append(fa)
        wandb.log({"arch": name, "fold": int(fv), "fold_val_auc": fa})

    mean_auc = float(np.mean(fold_aucs))
    print(f"  [{name}] fold aucs: {[f'{a:.4f}' for a in fold_aucs]}  → mean={mean_auc:.4f}")
    wandb.log({f"{name}_oof_auc": mean_auc})
    return mean_auc, oof


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  STACKER ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/6] Cross-validation …")

oof_aucs = {}

# ─────────────────────────────────────────────────────────────────────────────
# 5a.  BASELINE  (equal-weight average)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_fn(X_tr, Y_tr, X_va):
    return baseline_avg_logit(X_va)

baseline_auc, _ = cv_eval("baseline", baseline_fn, X, Y, fold_id)
oof_aucs["baseline"] = baseline_auc


# ─────────────────────────────────────────────────────────────────────────────
# 5b.  RIDGE  (per-class Ridge regression)
# ─────────────────────────────────────────────────────────────────────────────

def ridge_fn(X_tr, Y_tr, X_va):
    """
    Per-class Ridge: features for class c = [perch_c, proto_c, sed_c, hgnet_c].
    """
    preds = np.zeros((len(X_va), N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        feat_idx = [c, N_CLASSES + c, 2 * N_CLASSES + c, 3 * N_CLASSES + c]
        x_tr_c = X_tr[:, feat_idx]
        x_va_c = X_va[:, feat_idx]
        y_tr_c = Y_tr[:, c]
        if y_tr_c.sum() == 0:
            continue
        reg = Ridge(alpha=0.5, fit_intercept=True)
        reg.fit(x_tr_c, y_tr_c)
        preds[:, c] = reg.predict(x_va_c)
    return preds

ridge_auc, ridge_oof = cv_eval("ridge", ridge_fn, X, Y, fold_id)
oof_aucs["ridge"] = ridge_auc


# ─────────────────────────────────────────────────────────────────────────────
# 5c.  MLP  (shared-weight window-level)
# ─────────────────────────────────────────────────────────────────────────────

class StackerMLP(nn.Module):
    """
    Per-class shared MLP stacker.

    Input : (B, N_MODELS=4, N_CLASSES=234) — stack of 4 model logits
    Output: (B, N_CLASSES=234) — blended logit
    """
    def __init__(self, n_models: int = 4, n_classes: int = N_CLASSES,
                 hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.n_models  = n_models
        self.n_classes = n_classes
        # Maps (n_models,) → hidden per class, applied identically across classes
        self.fc1     = nn.Linear(n_models, hidden)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N_MODELS, N_CLASSES)
        returns: (B, N_CLASSES)
        """
        B, M, C = x.shape
        # (B, M, C) → (B*C, M)
        x = x.permute(0, 2, 1).reshape(B * C, M)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)                          # (B*C, 1)
        return x.reshape(B, C)                   # (B, C)


def X_to_mlp_input(X: np.ndarray) -> torch.Tensor:
    """
    X: (B, 4*234) → (B, 4, 234)
    layout: perch|proto|sed|hgnet each block of 234
    """
    B = X.shape[0]
    t = torch.from_numpy(X).float()              # (B, 936)
    blocks = [t[:, i * N_CLASSES:(i + 1) * N_CLASSES] for i in range(N_MODELS)]
    return torch.stack(blocks, dim=1)            # (B, 4, 234)


def train_mlp(X_tr: np.ndarray, Y_tr: np.ndarray,
              X_va: np.ndarray, Y_va: np.ndarray,
              epochs: int = 80, patience: int = 15,
              lr: float = 1e-3, wd: float = 1e-3) -> StackerMLP:

    model = StackerMLP().to(DEVICE)

    # pos_weight for training set (capped at 20)
    pos_rate = Y_tr.mean(axis=0)
    pos_w    = np.clip(
        (1.0 - pos_rate) / np.maximum(pos_rate, 1e-6), 0, 20
    ).astype(np.float32)
    pw_tensor = torch.from_numpy(pos_w).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optim     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    Xtr_t = X_to_mlp_input(X_tr).to(DEVICE)        # (N_tr, 4, 234)
    Ytr_t = torch.from_numpy(Y_tr).float().to(DEVICE)
    Xva_t = X_to_mlp_input(X_va).to(DEVICE)
    Yva_t = torch.from_numpy(Y_va).float()

    ds    = TensorDataset(Xtr_t, Ytr_t)
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)

    best_auc = 0.0
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
                best_auc  = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


def mlp_fn(X_tr, Y_tr, X_va):
    model = train_mlp(X_tr, Y_tr, X_va, Y_va=Y[fold_id == fold_id[0]])
    # Use actual validation labels from Y
    model.eval()
    with torch.no_grad():
        preds = model(X_to_mlp_input(X_va).to(DEVICE)).cpu().numpy()
    return preds


# CV for MLP (needs Y_va per fold — refactor loop)
def cv_eval_mlp(X: np.ndarray, Y: np.ndarray,
                fold_id: np.ndarray) -> tuple[float, np.ndarray]:
    oof   = np.zeros_like(Y, dtype=np.float32)
    fold_aucs = []
    unique_folds = np.unique(fold_id)
    for fv in tqdm(unique_folds, desc="MLP CV"):
        va_mask = fold_id == fv
        tr_mask = ~va_mask
        model = train_mlp(X[tr_mask], Y[tr_mask], X[va_mask], Y[va_mask])
        model.eval()
        with torch.no_grad():
            preds = model(X_to_mlp_input(X[va_mask]).to(DEVICE)).cpu().numpy()
        oof[va_mask] = preds
        fa = macro_auc(Y[va_mask], preds)
        fold_aucs.append(fa)
        wandb.log({"arch": "mlp", "fold": int(fv), "fold_val_auc": fa})
    mean_auc = float(np.mean(fold_aucs))
    print(f"  [mlp] fold aucs: {[f'{a:.4f}' for a in fold_aucs]}  → mean={mean_auc:.4f}")
    wandb.log({"mlp_oof_auc": mean_auc})
    return mean_auc, oof

mlp_auc, mlp_oof = cv_eval_mlp(X_norm, Y, fold_id)
oof_aucs["mlp"] = mlp_auc


# ─────────────────────────────────────────────────────────────────────────────
# 5d.  SSM  (Mamba-style selective SSM, file-level sequence)
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveSSMLayer(nn.Module):
    """
    Simplified selective SSM (Mamba-inspired) layer.
    Input / output: (B, T, d_model)
    """
    def __init__(self, d_model: int = 128, d_state: int = 16,
                 dropout: float = 0.1):
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

    def ssm_scan(self, x: torch.Tensor,
                 B: torch.Tensor,
                 C: torch.Tensor,
                 dt: torch.Tensor) -> torch.Tensor:
        """Causal discretised SSM scan. x:(B,T,d), B/C:(B,T,d_state), dt:(B,T,d)."""
        batch, T, d = x.shape
        A = -torch.exp(self.A_log.float())                # (d, d_state)
        h = torch.zeros(batch, d, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = F.softplus(dt[:, t, :])                # (B, d)
            dA   = torch.exp(dt_t.unsqueeze(-1) * A)      # (B, d, d_state)
            dB   = dt_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # (B, d, d_state)
            h    = dA * h + dB * x[:, t, :].unsqueeze(-1) # (B, d, d_state)
            y    = (h * C[:, t, :].unsqueeze(1)).sum(-1)   # (B, d)
            ys.append(y)
        return torch.stack(ys, dim=1)                     # (B, T, d)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_in = self.norm(x_in)                            # (B, T, d)
        xz   = self.in_proj(x_in)                         # (B, T, 2d)
        x, z = xz.chunk(2, dim=-1)                        # each (B, T, d)

        # Depthwise conv
        x = x.transpose(1, 2)                             # (B, d, T)
        x = self.conv1d(x).transpose(1, 2)                # (B, T, d)

        # SSM params
        params = self.x_proj(x)                           # (B, T, 2*d_state+1)
        B_p = params[..., :self.d_state]
        C_p = params[..., self.d_state:2 * self.d_state]
        dt  = self.dt_proj(params[..., -1:])              # (B, T, d)

        y = self.ssm_scan(x, B_p, C_p, dt)               # (B, T, d)
        y = y * F.silu(z)                                 # gating
        y = self.out_proj(y)
        return x_in + self.drop(y)                        # residual


class StackerSSM(nn.Module):
    """
    File-level SSM stacker.

    Input : (B_files, T=12, 936)
    Output: (B_files, T=12, 234)
    """
    def __init__(self, in_features: int = N_MODELS * N_CLASSES,
                 d_model: int = 128, d_state: int = 16,
                 n_classes: int = N_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.ssm1    = SelectiveSSMLayer(d_model, d_state, dropout)
        self.ssm2    = SelectiveSSMLayer(d_model, d_state, dropout)
        self.out_proj = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 936) → (B, T, 234)"""
        x = self.in_proj(x)     # (B, T, 128)
        x = self.ssm1(x)
        x = self.ssm2(x)
        return self.out_proj(x) # (B, T, 234)


def win_to_file_seq(X: np.ndarray, files: list[str],
                    filenames_708: np.ndarray) -> np.ndarray:
    """
    Rearrange (708, F) → (n_files, 12, F) in file order.
    """
    n_files = len(files)
    F = X.shape[1]
    out = np.zeros((n_files, N_WIN, F), dtype=np.float32)
    for fi, fname in enumerate(files):
        mask = filenames_708 == fname
        rows = np.where(mask)[0]
        assert len(rows) == N_WIN, f"Expected 12 windows for {fname}, got {len(rows)}"
        out[fi] = X[rows]
    return out


def win_to_file_labels(Y: np.ndarray, files: list[str],
                       filenames_708: np.ndarray) -> np.ndarray:
    n_files = len(files)
    out = np.zeros((n_files, N_WIN, N_CLASSES), dtype=np.float32)
    for fi, fname in enumerate(files):
        mask = filenames_708 == fname
        rows = np.where(mask)[0]
        out[fi] = Y[rows]
    return out


def train_seq_model(model: nn.Module,
                    X_tr_seq: np.ndarray, Y_tr_seq: np.ndarray,
                    X_va_seq: np.ndarray, Y_va_seq: np.ndarray,
                    epochs: int = 100, patience: int = 12,
                    lr: float = 5e-4, wd: float = 1e-3,
                    batch_size: int = 16) -> nn.Module:
    """Generic trainer for file-level sequence stackers (SSM / Transformer)."""
    model = model.to(DEVICE)
    pos_rate = Y_tr_seq.mean(axis=(0, 1))          # (234,)
    pos_w    = np.clip((1.0 - pos_rate) / np.maximum(pos_rate, 1e-6), 0, 20).astype(np.float32)
    pw_tensor = torch.from_numpy(pos_w).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optim     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    Xtr = torch.from_numpy(X_tr_seq).float()
    Ytr = torch.from_numpy(Y_tr_seq).float()
    Xva = torch.from_numpy(X_va_seq).float().to(DEVICE)
    Yva_np = Y_va_seq

    ds     = TensorDataset(Xtr, Ytr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_auc  = 0.0
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            pred = model(xb)                         # (B, T, 234)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                va_pred = model(Xva).cpu().numpy()   # (n_va, T, 234)
            va_flat  = va_pred.reshape(-1, N_CLASSES)
            yva_flat = Yva_np.reshape(-1, N_CLASSES)
            auc = macro_auc(yva_flat, va_flat)
            if auc > best_auc:
                best_auc  = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


# Prepare file-level sequences
X_norm_seq = win_to_file_seq(X_norm, unique_files, filenames_708)   # (59, 12, 936)
Y_seq      = win_to_file_labels(Y,      unique_files, filenames_708) # (59, 12, 234)
# fold_id per file (use first window's fold)
file_fold  = np.array([fold_id[np.where(filenames_708 == f)[0][0]] for f in unique_files],
                      dtype=np.int32)  # (59,)


def cv_eval_seq(name: str, model_cls, X_seq: np.ndarray,
                Y_seq: np.ndarray, file_fold: np.ndarray,
                **train_kwargs) -> tuple[float, np.ndarray]:
    """CV for file-level sequence models → OOF in (708, 234) window space."""
    oof_file  = np.zeros((len(unique_files), N_WIN, N_CLASSES), dtype=np.float32)
    fold_aucs = []
    for fv in tqdm(np.unique(file_fold), desc=f"{name} CV"):
        va_mask = file_fold == fv
        tr_mask = ~va_mask
        model   = train_seq_model(
            model_cls(), X_seq[tr_mask], Y_seq[tr_mask],
            X_seq[va_mask], Y_seq[va_mask], **train_kwargs
        )
        model.eval()
        with torch.no_grad():
            preds = model(
                torch.from_numpy(X_seq[va_mask]).float().to(DEVICE)
            ).cpu().numpy()   # (n_va_files, T, 234)
        oof_file[va_mask] = preds
        yva_flat  = Y_seq[va_mask].reshape(-1, N_CLASSES)
        pva_flat  = preds.reshape(-1, N_CLASSES)
        fa = macro_auc(yva_flat, pva_flat)
        fold_aucs.append(fa)
        wandb.log({"arch": name, "fold": int(fv), "fold_val_auc": fa})

    mean_auc = float(np.mean(fold_aucs))
    print(f"  [{name}] fold aucs: {[f'{a:.4f}' for a in fold_aucs]}  → mean={mean_auc:.4f}")
    wandb.log({f"{name}_oof_auc": mean_auc})

    # Flatten back to (708, 234) window order matching filenames_708
    oof_win = np.zeros((708, N_CLASSES), dtype=np.float32)
    for fi, fname in enumerate(unique_files):
        mask = filenames_708 == fname
        rows = np.where(mask)[0]
        oof_win[rows] = oof_file[fi]

    return mean_auc, oof_win


# SSM CV
ssm_auc, ssm_oof = cv_eval_seq(
    "ssm", StackerSSM, X_norm_seq, Y_seq, file_fold,
    epochs=100, patience=12, lr=5e-4, wd=1e-3, batch_size=16
)
oof_aucs["ssm"] = ssm_auc


# ─────────────────────────────────────────────────────────────────────────────
# 5e.  TRANSFORMER  (temporal attention, file-level)
# ─────────────────────────────────────────────────────────────────────────────

class StackerTransformer(nn.Module):
    """
    File-level Transformer stacker.

    Input : (B_files, T=12, 936)
    Output: (B_files, T=12, 234)
    """
    def __init__(self, in_features: int = N_MODELS * N_CLASSES,
                 d_model: int = 128, nhead: int = 4,
                 dim_ff: int = 256, n_layers: int = 2,
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
        """x: (B, T, 936) → (B, T, 234)"""
        x = self.in_proj(x) + self.pos_emb   # (B, T, 128)
        x = self.encoder(x)
        return self.out_proj(x)              # (B, T, 234)


tfm_auc, tfm_oof = cv_eval_seq(
    "transformer", StackerTransformer, X_norm_seq, Y_seq, file_fold,
    epochs=100, patience=12, lr=5e-4, wd=1e-3, batch_size=16
)
oof_aucs["transformer"] = tfm_auc


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  FINAL FIT ON ALL DATA + EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5/6] Final fit on all data …")

best_arch = max(
    [k for k in oof_aucs if k not in ("baseline",)],
    key=lambda k: oof_aucs[k]
)
print(f"  best architecture: {best_arch}  (AUC={oof_aucs[best_arch]:.4f})")

# ── Ridge (full fit) ──────────────────────────────────────────────────────────
print("  Fitting Ridge (full) …")
ridge_models = []
for c in tqdm(range(N_CLASSES), desc="Ridge full fit"):
    feat_idx = [c, N_CLASSES + c, 2 * N_CLASSES + c, 3 * N_CLASSES + c]
    reg = Ridge(alpha=0.5, fit_intercept=True)
    if Y[:, c].sum() > 0:
        reg.fit(X[:, feat_idx], Y[:, c])
    ridge_models.append(reg)

with open(OUT_DIR / "stacker_ridge.pkl", "wb") as f:
    pickle.dump(ridge_models, f)
print(f"  Ridge saved → {OUT_DIR / 'stacker_ridge.pkl'}")


# ── MLP (full fit + ONNX export) ─────────────────────────────────────────────
print("  Fitting MLP (full) …")
mlp_model = train_mlp(X_norm, Y, X_norm, Y, epochs=80, patience=80)
mlp_model.eval()
torch.save(mlp_model.state_dict(), OUT_DIR / "stacker_mlp.pt")

# ONNX export
dummy_mlp = torch.zeros(1, N_MODELS, N_CLASSES)
torch.onnx.export(
    mlp_model.cpu(), dummy_mlp,
    str(OUT_DIR / "stacker_mlp.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=14,
)
print(f"  MLP saved → {OUT_DIR / 'stacker_mlp.onnx'}")


# ── SSM (full fit + ONNX export) ─────────────────────────────────────────────
print("  Fitting SSM (full) …")
ssm_model = train_seq_model(
    StackerSSM(), X_norm_seq, Y_seq, X_norm_seq, Y_seq,
    epochs=100, patience=100, lr=5e-4, wd=1e-3, batch_size=16
)
ssm_model.eval()
torch.save(ssm_model.state_dict(), OUT_DIR / "stacker_ssm.pt")

dummy_ssm = torch.zeros(1, N_WIN, N_MODELS * N_CLASSES)
torch.onnx.export(
    ssm_model.cpu(), dummy_ssm,
    str(OUT_DIR / "stacker_ssm.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  SSM saved → {OUT_DIR / 'stacker_ssm.onnx'}")


# ── Transformer (full fit + ONNX export) ─────────────────────────────────────
print("  Fitting Transformer (full) …")
tfm_model = train_seq_model(
    StackerTransformer(), X_norm_seq, Y_seq, X_norm_seq, Y_seq,
    epochs=100, patience=100, lr=5e-4, wd=1e-3, batch_size=16
)
tfm_model.eval()
torch.save(tfm_model.state_dict(), OUT_DIR / "stacker_transformer.pt")

dummy_tfm = torch.zeros(1, N_WIN, N_MODELS * N_CLASSES)
torch.onnx.export(
    tfm_model.cpu(), dummy_tfm,
    str(OUT_DIR / "stacker_transformer.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  Transformer saved → {OUT_DIR / 'stacker_transformer.onnx'}")


# ── Meta JSON ─────────────────────────────────────────────────────────────────
meta_dict = {
    "best_arch"      : best_arch,
    "oof_aucs"       : {k: round(v, 6) for k, v in oof_aucs.items()},
    "n_models"       : N_MODELS,
    "n_classes"      : N_CLASSES,
    "n_windows"      : N_WIN,
    "feature_layout" : ["perch_oof", "proto_ssm_oof", "sed_ensemble", "hgnet"],
    "feature_dim"    : N_MODELS * N_CLASSES,
    "temperature"    : 1.5,
    "trained_date"   : time.strftime("%Y-%m-%d"),
}
with open(OUT_DIR / "stacker_meta.json", "w") as f:
    json.dump(meta_dict, f, indent=2)
print(f"  meta saved → {OUT_DIR / 'stacker_meta.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  OOF AUC SUMMARY")
print("=" * 60)
print(f"  {'Model':<20} {'OOF macro AUC':>14}")
print("-" * 40)
for k in ["baseline", "ridge", "mlp", "ssm", "transformer"]:
    marker = " ◀ best" if k == best_arch else ""
    print(f"  {k:<20} {oof_aucs.get(k, float('nan')):>14.4f}{marker}")
print("=" * 60)
print(f"\n[done]  All artifacts saved to: {OUT_DIR}")

# ─── Excel export ──────────────────────────────────────────────────────────────
rows = []
for k in ["baseline", "ridge", "mlp", "ssm", "transformer"]:
    rows.append({
        "arch"         : k,
        "oof_macro_auc": round(oof_aucs.get(k, float("nan")), 6),
        "best"         : k == best_arch,
        "trained_date" : time.strftime("%Y-%m-%d"),
    })
df_results = pd.DataFrame(rows)
excel_path  = OUT_DIR / "stacker_results.xlsx"

# Append to existing file if it exists, otherwise create
if excel_path.exists():
    with pd.ExcelWriter(str(excel_path), engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        df_results.to_excel(writer, sheet_name="OOF_AUC", index=False)
else:
    df_results.to_excel(str(excel_path), index=False, sheet_name="OOF_AUC")

print(f"  Excel saved → {excel_path}")

# ─── W&B final summary + finish ───────────────────────────────────────────────
wandb.log({
    "summary/best_arch"        : best_arch,
    "summary/best_oof_auc"     : oof_aucs[best_arch],
    **{f"summary/oof_{k}": oof_aucs.get(k, float("nan"))
       for k in ["baseline", "ridge", "mlp", "ssm", "transformer"]},
})
wandb.finish()
print("[wandb] run finished")
