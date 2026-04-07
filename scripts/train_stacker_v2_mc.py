"""
train_stacker_v2_mc.py — Stacking ensemble meta-learner for BirdCLEF 2026.

Feature layout (5 × 234 = 1170 dims per window):
  X = [perch_raw | perch_prior_fused | mlp_probe | proto_ssm | sed_csebbs]
       [0:234]     [234:468]           [468:702]   [702:936]   [936:1170]

1. perch_raw        (234): Raw Perch logits from full_perch_arrays.npz 'scores_full_raw'
2. perch_prior_fused(234): Prior-fused Perch logits from full_oof_meta_features.npz 'oof_base'
3. mlp_probe        (234): MLP probe OOF from full_oof_meta_features.npz 'oof_prior'
                           (or outputs/mlp_probe_oof.npy if it exists)
4. proto_ssm        (234): ProtoSSM OOF preds (59,234) broadcast to (708,234)
5. sed_csebbs       (234): SED ONNX -> BranchEns->cSEBBs post-proc -> logit space

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_stacker_v2_mc.py

Output dir: birdclef-2026/notebook resource/current_subs 2/stacker_weights/
"""

import os, gc, json, pickle, time, warnings
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

import torchaudio
import torchaudio.transforms as T

import wandb
import openpyxl

# ─── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/lab/BirdClef-2026-Codebase")
NB_DIR      = BASE_DIR / "birdclef-2026" / "notebook resource" / "current_subs 2"
PERCH_META  = NB_DIR / "perch meta"
WEIGHTS_DIR = NB_DIR / "weights"
OUT_DIR     = NB_DIR / "stacker_weights"
OUTPUTS     = BASE_DIR / "outputs"
AUDIO_DIR   = BASE_DIR / "birdclef-2026" / "train_soundscapes"
CACHE_SED_CSEBBS = OUTPUTS / "stacker_train_sed_csebbs_win.npy"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────────────────
SEED        = 42
SR          = 32_000
N_WIN       = 12
WIN_SAMPLES = SR * 5
N_CLASSES   = 234
N_MODELS    = 5    # perch_raw | perch_prior_fused | mlp_probe | proto_ssm | sed_csebbs
FEATURE_DIM = N_MODELS * N_CLASSES   # 1170
CONTEXT_K   = 1    # context window half-size for LGBM: 2*1+1=3 windows => 3510 feats
EPS         = 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] device={DEVICE}  out={OUT_DIR}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── W&B init ──────────────────────────────────────────────────────────────
wandb.init(
    project="birdclef-2026",
    name="stacker-v2-mc",
    config={
        "n_models": N_MODELS,
        "n_classes": N_CLASSES,
        "n_windows": N_WIN,
        "feature_dim": FEATURE_DIM,
        "context_k": CONTEXT_K,
        "seed": SEED,
        "feature_layout": ["perch_raw", "perch_prior_fused", "mlp_probe", "proto_ssm", "sed_csebbs"],
    },
    tags=["stacker", "meta-learner", "v2-mc"],
)
print("[wandb] run initialized")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def safe_logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            try:
                aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# BranchEns->cSEBBs post-processing
# ═══════════════════════════════════════════════════════════════════════════

def apply_branchens_csebbs(probs_12: np.ndarray) -> np.ndarray:
    """BranchEns->cSEBBs temporal post-processing. Input/Output: (12, C) in [0,1]."""
    eps = 1e-7
    p = np.clip(probs_12.astype(np.float32), eps, 1.0 - eps)
    T, C = p.shape

    H  = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).mean(axis=1)
    w  = np.exp(-H / 0.1)
    w  = w / w.sum() * T
    wl = np.log(p / (1.0 - p)) * w[:, None]

    def _lse_pool(wl_in, beta):
        out = np.zeros_like(wl_in)
        for t in range(T):
            win = wl_in[max(0, t - 1):min(T, t + 2)]
            mx  = win.max(axis=0)
            out[t] = mx + (1.0 / beta) * np.log(np.exp(beta * (win - mx)).sum(axis=0))
        return 1.0 / (1.0 + np.exp(-out))

    def _dual_anchor(lp, nw, alpha):
        anc = nw * (1.0 - np.prod(1.0 - lp, axis=0)) + (1.0 - nw) * lp.max(axis=0)
        return (1.0 - alpha) * lp + alpha * anc[None, :]

    out_a = _dual_anchor(_lse_pool(wl, 5.15), 0.40, 0.38)
    out_b = _dual_anchor(_lse_pool(wl, 6.0),  0.30, 0.40)
    ens   = np.clip(0.55 * out_a + 0.45 * out_b, eps, 1.0 - eps)

    out  = ens.copy()
    diff = np.abs(np.diff(ens, axis=0))
    for t in range(T - 1):
        cols = np.where(diff[t] > 0.06)[0]
        if len(cols):
            seg = ens[max(0, t - 2):min(T, t + 3)]
            out[t, cols] = seg[:, cols].mean(axis=0)
    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD STATIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n[1/6] Loading static features ...")

# perch_raw: (708, 234) raw Perch logits
perch_arrays   = np.load(PERCH_META / "full_perch_arrays.npz")
perch_raw      = perch_arrays["scores_full_raw"].astype(np.float32)
print(f"  perch_raw        : {perch_raw.shape}  [{perch_raw.min():.2f}, {perch_raw.max():.2f}]")

# perch_prior_fused: oof_base (708, 234)
oof_data          = np.load(PERCH_META / "full_oof_meta_features.npz")
perch_prior_fused = oof_data["oof_base"].astype(np.float32)
fold_id           = oof_data["fold_id"].astype(np.int32)
print(f"  perch_prior_fused: {perch_prior_fused.shape}  [{perch_prior_fused.min():.2f}, {perch_prior_fused.max():.2f}]")

# mlp_probe: prefer dedicated file, else oof_prior
mlp_probe_path = OUTPUTS / "mlp_probe_oof.npy"
if mlp_probe_path.exists():
    mlp_probe = np.load(str(mlp_probe_path)).astype(np.float32)
    print(f"  mlp_probe        : {mlp_probe.shape} (from mlp_probe_oof.npy)")
else:
    mlp_probe = oof_data["oof_prior"].astype(np.float32)
    print(f"  mlp_probe        : {mlp_probe.shape} (from oof_prior)")
print(f"                     [{mlp_probe.min():.2f}, {mlp_probe.max():.2f}]")

# Meta: filename/row_id per 708 windows
meta          = pd.read_parquet(PERCH_META / "full_perch_meta.parquet")
assert len(meta) == 708
filenames_708 = meta["filename"].values
row_ids_708   = meta["row_id"].values

unique_files = list(dict.fromkeys(filenames_708))
assert len(unique_files) == 59
file_to_idx  = {f: i for i, f in enumerate(unique_files)}

# proto_ssm: (59,234) -> broadcast to (708,234)
proto_preds_59  = np.load(OUTPUTS / "proto_ssm_oof_preds.npy").astype(np.float32)
proto_files_59  = np.load(OUTPUTS / "proto_ssm_oof_file_list.npy", allow_pickle=True)
proto_logit_708 = np.zeros((708, N_CLASSES), dtype=np.float32)
for win_i, fname in enumerate(filenames_708):
    mask = proto_files_59 == fname
    if mask.any():
        proto_logit_708[win_i] = proto_preds_59[np.where(mask)[0][0]]
print(f"  proto_ssm        : {proto_logit_708.shape}  [{proto_logit_708.min():.2f}, {proto_logit_708.max():.2f}]")

# Ground-truth labels (708, 234)
label_data    = np.load(OUTPUTS / "perch_labeled_ss.npz", allow_pickle=True)
label_y_raw   = label_data["labels"].astype(np.float32)
label_row_ids = label_data["row_ids"]
rid_to_label  = dict(zip(label_row_ids, range(len(label_row_ids))))
Y = np.zeros((708, N_CLASSES), dtype=np.float32)
missing = 0
for i, rid in enumerate(row_ids_708):
    if rid in rid_to_label:
        Y[i] = label_y_raw[rid_to_label[rid]]
    else:
        missing += 1
print(f"  labels           : {Y.shape}  missing={missing}  pos_rate={Y.mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. SED INFERENCE + BranchEns->cSEBBs ON 59 TRAIN SOUNDSCAPES
# ═══════════════════════════════════════════════════════════════════════════

def build_mel_eff(clip: np.ndarray) -> np.ndarray:
    """clip (160000,) -> (1,3,224,T) float32 in [0,1]."""
    wav   = torch.from_numpy(clip).unsqueeze(0)
    mel_t = T.MelSpectrogram(
        sample_rate=SR, n_fft=2048, hop_length=512, n_mels=224,
        f_min=0, f_max=16000, power=2.0, norm="slaney", mel_scale="htk",
    )
    db_t = T.AmplitudeToDB(top_db=80)
    mel  = db_t(mel_t(wav))
    mel  = mel - mel.min()
    mx   = mel.max()
    if mx > 0:
        mel = mel / mx
    mel = mel.repeat(3, 1, 1)
    return mel.numpy()[None]   # (1,3,224,T)


def infer_sed_file(audio_path: Path, sess_list: list) -> np.ndarray:
    """
    Run SED ensemble on a 60s soundscape -> apply BranchEns->cSEBBs.
    Returns: (12, 234) float32 logits.
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
        clip  = audio[w * WIN_SAMPLES:(w + 1) * WIN_SAMPLES].astype(np.float32)
        mel_e = build_mel_eff(clip)
        preds = []
        for sess in sess_list:
            inp_name = sess.get_inputs()[0].name
            preds.append(sess.run(None, {inp_name: mel_e})[0].squeeze(0))
        sed_wins.append(np.mean(preds, axis=0).astype(np.float32))

    raw_probs = np.stack(sed_wins)            # (12, 234) in [0,1]
    csebbs    = apply_branchens_csebbs(raw_probs)
    return safe_logit(csebbs)                 # (12, 234) logits


def build_sed_csebbs_features(unique_files: list) -> np.ndarray:
    """Run SED+cSEBBs on 59 soundscapes -> (708, 234). Caches to disk."""
    if CACHE_SED_CSEBBS.exists():
        print("  [cache] loading SED cSEBBs from cache ...")
        arr = np.load(str(CACHE_SED_CSEBBS))
        assert arr.shape == (708, N_CLASSES)
        return arr

    print("  [inference] running SED+cSEBBs on 59 soundscapes ...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_list = []
    for onnx_name in ["best_sed_b0_v5.onnx", "competitor_sed_fold0.onnx"]:
        p = WEIGHTS_DIR / onnx_name
        if p.exists():
            sess_list.append(ort.InferenceSession(str(p), providers=providers))
            print(f"    loaded: {onnx_name}")
        else:
            print(f"    WARNING: {onnx_name} not found - skipping")

    if not sess_list:
        print("  WARNING: no SED sessions found - using zeros")
        return np.zeros((708, N_CLASSES), dtype=np.float32)

    all_wins = []
    for fname in tqdm(unique_files, desc="SED+cSEBBs inference"):
        all_wins.append(infer_sed_file(AUDIO_DIR / fname, sess_list))

    arr = np.concatenate(all_wins, axis=0).astype(np.float32)
    np.save(str(CACHE_SED_CSEBBS), arr)
    print(f"  cached -> {CACHE_SED_CSEBBS}")
    return arr


print("\n[2/6] Running / loading SED+cSEBBs inference ...")
sed_csebbs_logit = build_sed_csebbs_features(unique_files)
print(f"  sed_csebbs       : {sed_csebbs_logit.shape}  [{sed_csebbs_logit.min():.2f}, {sed_csebbs_logit.max():.2f}]")


# ═══════════════════════════════════════════════════════════════════════════
# 3. BUILD FEATURE MATRIX X  (708, 1170)
# ═══════════════════════════════════════════════════════════════════════════

print("\n[3/6] Building feature matrix ...")

# [perch_raw(0:234) | perch_prior(234:468) | mlp_probe(468:702) |
#  proto_ssm(702:936) | sed_csebbs(936:1170)]
X = np.concatenate([
    perch_raw,
    perch_prior_fused,
    mlp_probe,
    proto_logit_708,
    sed_csebbs_logit,
], axis=1).astype(np.float32)
assert X.shape == (708, FEATURE_DIM), f"Bad X shape: {X.shape}"
print(f"  X shape : {X.shape}")

X_mean = X.mean(axis=0, keepdims=True).astype(np.float32)
X_std  = X.std(axis=0, keepdims=True).astype(np.float32)
X_std[X_std < 1e-8] = 1.0
X_norm = (X - X_mean) / X_std

np.savez(OUT_DIR / "stacker_feature_stats.npz", mean=X_mean, std=X_std)
np.savez(OUT_DIR / "stacker_norm.npz",          mean=X_mean, std=X_std)
print(f"  feature stats saved -> {OUT_DIR / 'stacker_feature_stats.npz'}")

# Save probe artifacts for inference
probe_cache_path = PERCH_META / "probe_cache.pkl"
if probe_cache_path.exists():
    with open(str(probe_cache_path), "rb") as fh:
        probe_cache = pickle.load(fh)
    with open(str(OUT_DIR / "probe_cache.pkl"), "wb") as fh:
        pickle.dump(probe_cache, fh)
    fm0 = probe_cache["fold_models"][0]
    with open(str(OUT_DIR / "prior_tables.pkl"), "wb") as fh:
        pickle.dump(fm0["prior_tables"], fh)
    with open(str(OUT_DIR / "pca_model.pkl"), "wb") as fh:
        pickle.dump(fm0["pca"], fh)
    with open(str(OUT_DIR / "mlp_probes.pkl"), "wb") as fh:
        pickle.dump(fm0["probe_models"], fh)
    print(f"  probe artifacts saved -> {OUT_DIR}")
else:
    print("  WARNING: probe_cache.pkl not found")


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def baseline_avg_logit(X_in: np.ndarray) -> np.ndarray:
    blocks = [X_in[:, i * N_CLASSES:(i + 1) * N_CLASSES] for i in range(N_MODELS)]
    return np.mean(blocks, axis=0)


def cv_eval(name, pred_fn, X_in, Y_in, fid):
    oof = np.zeros_like(Y_in, dtype=np.float32)
    fold_aucs = []
    for fv in np.unique(fid):
        va = fid == fv
        tr = ~va
        preds = pred_fn(X_in[tr], Y_in[tr], X_in[va])
        oof[va] = preds
        fa = macro_auc(Y_in[va], preds)
        fold_aucs.append(fa)
        wandb.log({"arch": name, "fold": int(fv), "fold_val_auc": fa})
    m = float(np.mean(fold_aucs))
    print(f"  [{name}] {[f'{a:.4f}' for a in fold_aucs]}  mean={m:.4f}")
    wandb.log({f"{name}_oof_auc": m})
    return m, oof


# ═══════════════════════════════════════════════════════════════════════════
# 5. STACKER ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n[4/6] Cross-validation ...")
oof_aucs = {}

# ── 5a. Baseline ────────────────────────────────────────────────────────────
baseline_auc, _ = cv_eval("baseline",
                           lambda Xtr, Ytr, Xva: baseline_avg_logit(Xva),
                           X, Y, fold_id)
oof_aucs["baseline"] = baseline_auc

# ── 5b. LGBM (per-class, context window k=1) ────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("  [lgbm] lightgbm not installed - skipping")


def build_context_features(X_in: np.ndarray, filenames: np.ndarray, k: int) -> np.ndarray:
    """(N, F) -> (N, (2k+1)*F) with temporal context per file."""
    N, F = X_in.shape
    W    = 2 * k + 1
    out  = np.zeros((N, W * F), dtype=np.float32)
    for i in range(N):
        fname     = filenames[i]
        file_rows = np.where(filenames == fname)[0]
        pos       = np.where(file_rows == i)[0][0]
        for slot, offset in enumerate(range(-k, k + 1)):
            j = pos + offset
            if 0 <= j < len(file_rows):
                out[i, slot * F:(slot + 1) * F] = X_in[file_rows[j]]
    return out


if HAS_LGBM:
    print("\n  Building context features for LGBM ...")
    X_ctx = build_context_features(X, filenames_708, CONTEXT_K)
    print(f"    X_ctx shape: {X_ctx.shape}")

    def cv_eval_lgbm(X_c, Y_in, fid):
        oof  = np.zeros((len(Y_in), N_CLASSES), dtype=np.float32)
        fas  = []
        for fv in np.unique(fid):
            va = fid == fv
            tr = ~va
            preds = np.zeros((va.sum(), N_CLASSES), dtype=np.float32)
            for c in range(N_CLASSES):
                if Y_in[tr, c].sum() == 0:
                    continue
                m = lgb.LGBMClassifier(
                    objective="binary", num_leaves=31, learning_rate=0.05,
                    n_estimators=300, min_child_samples=5,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    verbose=-1, n_jobs=4,
                )
                m.fit(X_c[tr], Y_in[tr, c], callbacks=[lgb.log_evaluation(-1)])
                preds[:, c] = m.predict_proba(X_c[va])[:, 1]
            oof[va] = safe_logit(preds)
            fas.append(macro_auc(Y_in[va], safe_logit(preds)))
            wandb.log({"arch": "lgbm", "fold": int(fv), "fold_val_auc": fas[-1]})
        m_auc = float(np.mean(fas))
        print(f"  [lgbm] {[f'{a:.4f}' for a in fas]}  mean={m_auc:.4f}")
        wandb.log({"lgbm_oof_auc": m_auc})
        return m_auc, oof

    lgbm_auc, lgbm_oof = cv_eval_lgbm(X_ctx, Y, fold_id)
    oof_aucs["lgbm"] = lgbm_auc
else:
    oof_aucs["lgbm"] = 0.0
    lgbm_oof = np.zeros((708, N_CLASSES), dtype=np.float32)

# ── 5c. Ridge ───────────────────────────────────────────────────────────────
def ridge_fn(X_tr, Y_tr, X_va):
    preds = np.zeros((len(X_va), N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        feat_idx = [i * N_CLASSES + c for i in range(N_MODELS)]
        if Y_tr[:, c].sum() == 0:
            continue
        reg = Ridge(alpha=0.5, fit_intercept=True)
        reg.fit(X_tr[:, feat_idx], Y_tr[:, c])
        preds[:, c] = reg.predict(X_va[:, feat_idx])
    return preds

ridge_auc, ridge_oof = cv_eval("ridge", ridge_fn, X, Y, fold_id)
oof_aucs["ridge"] = ridge_auc

# ── 5d. MLP ─────────────────────────────────────────────────────────────────
class StackerMLP(nn.Module):
    """Per-class shared MLP. Input: (B,5,234) -> (B,234)."""
    def __init__(self, n_models=N_MODELS, n_classes=N_CLASSES, hidden=32, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_models, hidden)
        self.act = nn.GELU()
        self.drp = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        B, M, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, M)
        x = self.drp(self.act(self.fc1(x)))
        return self.fc2(x).reshape(B, C)


def X_to_mlp_input(X_in):
    t = torch.from_numpy(X_in).float()
    return torch.stack([t[:, i * N_CLASSES:(i + 1) * N_CLASSES] for i in range(N_MODELS)], dim=1)


def train_mlp(X_tr, Y_tr, X_va, Y_va, epochs=80, patience=15, lr=1e-3, wd=1e-3):
    model  = StackerMLP().to(DEVICE)
    pos_w  = np.clip((1 - Y_tr.mean(0)) / np.maximum(Y_tr.mean(0), 1e-6), 0, 20).astype(np.float32)
    crit   = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_w).to(DEVICE))
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtr    = X_to_mlp_input(X_tr).to(DEVICE)
    Ytr    = torch.from_numpy(Y_tr).float().to(DEVICE)
    Xva    = X_to_mlp_input(X_va).to(DEVICE)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=256, shuffle=True)
    best_auc, best_st, no_imp = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                p = model(Xva).cpu().numpy()
            a = macro_auc(Y_va, p)
            if a > best_auc:
                best_auc, best_st, no_imp = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                no_imp += 5
            if no_imp >= patience:
                break
    model.load_state_dict(best_st)
    return model


def cv_eval_mlp(X_in, Y_in, fid):
    oof, fas = np.zeros_like(Y_in, dtype=np.float32), []
    for fv in tqdm(np.unique(fid), desc="MLP CV"):
        va, tr = fid == fv, ~(fid == fv)
        m = train_mlp(X_in[tr], Y_in[tr], X_in[va], Y_in[va])
        m.eval()
        with torch.no_grad():
            p = m(X_to_mlp_input(X_in[va]).to(DEVICE)).cpu().numpy()
        oof[va] = p
        fas.append(macro_auc(Y_in[va], p))
        wandb.log({"arch": "mlp", "fold": int(fv), "fold_val_auc": fas[-1]})
    mean_auc = float(np.mean(fas))
    print(f"  [mlp] {[f'{a:.4f}' for a in fas]}  mean={mean_auc:.4f}")
    wandb.log({"mlp_oof_auc": mean_auc})
    return mean_auc, oof


mlp_auc, mlp_oof = cv_eval_mlp(X_norm, Y, fold_id)
oof_aucs["mlp"] = mlp_auc

# ── 5e/5f. SSM + Transformer (file-level) ──────────────────────────────────

class SelectiveSSMLayer(nn.Module):
    def __init__(self, d_model=128, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, 2 * d_model)
        self.conv1d   = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.x_proj   = nn.Linear(d_model, d_state * 2 + 1)
        self.dt_proj  = nn.Linear(1, d_model)
        self.A_log    = nn.Parameter(torch.randn(d_model, d_state))
        self.D        = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

    def ssm_scan(self, x, Bp, Cp, dt):
        batch, T, d = x.shape
        A  = -torch.exp(self.A_log.float())
        h  = torch.zeros(batch, d, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = F.softplus(dt[:, t, :])
            dA   = torch.exp(dt_t.unsqueeze(-1) * A)
            dB   = dt_t.unsqueeze(-1) * Bp[:, t, :].unsqueeze(1)
            h    = dA * h + dB * x[:, t, :].unsqueeze(-1)
            ys.append((h * Cp[:, t, :].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1)

    def forward(self, x_in):
        x_in  = self.norm(x_in)
        xz    = self.in_proj(x_in)
        x, z  = xz.chunk(2, dim=-1)
        x     = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        p     = self.x_proj(x)
        Bp, Cp, dtp = p[..., :self.d_state], p[..., self.d_state:2*self.d_state], p[..., -1:]
        y = self.ssm_scan(x, Bp, Cp, self.dt_proj(dtp))
        y = y * F.silu(z)
        return x_in + self.drop(self.out_proj(y))


class StackerSSM(nn.Module):
    """(B,12,1170) -> (B,12,234)"""
    def __init__(self, in_features=FEATURE_DIM, d_model=128, d_state=16,
                 n_classes=N_CLASSES, dropout=0.1):
        super().__init__()
        self.in_proj  = nn.Linear(in_features, d_model)
        self.ssm1     = SelectiveSSMLayer(d_model, d_state, dropout)
        self.ssm2     = SelectiveSSMLayer(d_model, d_state, dropout)
        self.out_proj = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.out_proj(self.ssm2(self.ssm1(self.in_proj(x))))


class StackerTransformer(nn.Module):
    """(B,12,1170) -> (B,12,234)"""
    def __init__(self, in_features=FEATURE_DIM, d_model=128, nhead=4,
                 dim_ff=256, n_layers=2, n_classes=N_CLASSES, dropout=0.1):
        super().__init__()
        self.in_proj  = nn.Linear(in_features, d_model)
        self.pos_emb  = nn.Parameter(torch.zeros(1, N_WIN, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                         batch_first=True, activation="gelu")
        self.encoder  = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.in_proj(x) + self.pos_emb
        return self.out_proj(self.encoder(x))


def win_to_file_seq(X_in, files, filenames):
    F   = X_in.shape[1]
    out = np.zeros((len(files), N_WIN, F), dtype=np.float32)
    for fi, fname in enumerate(files):
        rows = np.where(filenames == fname)[0]
        assert len(rows) == N_WIN
        out[fi] = X_in[rows]
    return out


def win_to_file_labels(Y_in, files, filenames):
    out = np.zeros((len(files), N_WIN, N_CLASSES), dtype=np.float32)
    for fi, fname in enumerate(files):
        out[fi] = Y_in[np.where(filenames == fname)[0]]
    return out


def train_seq_model(model, X_tr_s, Y_tr_s, X_va_s, Y_va_s,
                    epochs=100, patience=12, lr=5e-4, wd=1e-3, batch_size=16):
    model   = model.to(DEVICE)
    pos_r   = Y_tr_s.mean(axis=(0, 1))
    pos_w   = np.clip((1 - pos_r) / np.maximum(pos_r, 1e-6), 0, 20).astype(np.float32)
    crit    = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_w).to(DEVICE))
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtr     = torch.from_numpy(X_tr_s).float()
    Ytr     = torch.from_numpy(Y_tr_s).float()
    Xva     = torch.from_numpy(X_va_s).float().to(DEVICE)
    loader  = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    best_a, best_st, no_imp = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                p = model(Xva).cpu().numpy()
            a = macro_auc(Y_va_s.reshape(-1, N_CLASSES), p.reshape(-1, N_CLASSES))
            if a > best_a:
                best_a, best_st, no_imp = a, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                no_imp += 5
            if no_imp >= patience:
                break
    model.load_state_dict(best_st)
    return model


def cv_eval_seq(name, model_cls, X_s, Y_s, ff, **kw):
    oof_f = np.zeros((len(unique_files), N_WIN, N_CLASSES), dtype=np.float32)
    fas   = []
    for fv in tqdm(np.unique(ff), desc=f"{name} CV"):
        va, tr = ff == fv, ~(ff == fv)
        m = train_seq_model(model_cls(), X_s[tr], Y_s[tr], X_s[va], Y_s[va], **kw)
        m.eval()
        with torch.no_grad():
            p = m(torch.from_numpy(X_s[va]).float().to(DEVICE)).cpu().numpy()
        oof_f[va] = p
        fas.append(macro_auc(Y_s[va].reshape(-1, N_CLASSES), p.reshape(-1, N_CLASSES)))
        wandb.log({"arch": name, "fold": int(fv), "fold_val_auc": fas[-1]})
    mean_auc = float(np.mean(fas))
    print(f"  [{name}] {[f'{a:.4f}' for a in fas]}  mean={mean_auc:.4f}")
    wandb.log({f"{name}_oof_auc": mean_auc})
    oof_w = np.zeros((708, N_CLASSES), dtype=np.float32)
    for fi, fname in enumerate(unique_files):
        oof_w[np.where(filenames_708 == fname)[0]] = oof_f[fi]
    return mean_auc, oof_w


X_norm_seq = win_to_file_seq(X_norm, unique_files, filenames_708)
Y_seq      = win_to_file_labels(Y, unique_files, filenames_708)
file_fold  = np.array([fold_id[np.where(filenames_708 == f)[0][0]] for f in unique_files],
                      dtype=np.int32)

ssm_auc, ssm_oof = cv_eval_seq(
    "ssm", StackerSSM, X_norm_seq, Y_seq, file_fold,
    epochs=100, patience=12, lr=5e-4, wd=1e-3, batch_size=16
)
oof_aucs["ssm"] = ssm_auc

tfm_auc, tfm_oof = cv_eval_seq(
    "transformer", StackerTransformer, X_norm_seq, Y_seq, file_fold,
    epochs=100, patience=12, lr=5e-4, wd=1e-3, batch_size=16
)
oof_aucs["transformer"] = tfm_auc


# ═══════════════════════════════════════════════════════════════════════════
# 6. FINAL FIT ON ALL DATA + EXPORT
# ═══════════════════════════════════════════════════════════════════════════

print("\n[5/6] Final fit on all data ...")

best_arch = max([k for k in oof_aucs if k != "baseline"], key=lambda k: oof_aucs[k])
print(f"  best architecture: {best_arch}  (AUC={oof_aucs[best_arch]:.4f})")

# LGBM full fit
if HAS_LGBM:
    print("  Fitting LGBM (full) ...")
    lgbm_models = []
    for c in tqdm(range(N_CLASSES), desc="LGBM full fit"):
        m = lgb.LGBMClassifier(
            objective="binary", num_leaves=31, learning_rate=0.05,
            n_estimators=300, min_child_samples=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            verbose=-1, n_jobs=4,
        )
        if Y[:, c].sum() > 0:
            m.fit(X_ctx, Y[:, c], callbacks=[lgb.log_evaluation(-1)])
        lgbm_models.append(m)
    with open(OUT_DIR / "stacker_lgbm.pkl", "wb") as fh:
        pickle.dump(lgbm_models, fh)
    print(f"  LGBM saved -> {OUT_DIR / 'stacker_lgbm.pkl'}")

# Ridge full fit
print("  Fitting Ridge (full) ...")
ridge_models = []
for c in tqdm(range(N_CLASSES), desc="Ridge full fit"):
    feat_idx = [i * N_CLASSES + c for i in range(N_MODELS)]
    reg = Ridge(alpha=0.5, fit_intercept=True)
    if Y[:, c].sum() > 0:
        reg.fit(X[:, feat_idx], Y[:, c])
    ridge_models.append(reg)
with open(OUT_DIR / "stacker_ridge.pkl", "wb") as fh:
    pickle.dump(ridge_models, fh)
print(f"  Ridge saved -> {OUT_DIR / 'stacker_ridge.pkl'}")

# MLP full fit + ONNX
print("  Fitting MLP (full) ...")
mlp_model = train_mlp(X_norm, Y, X_norm, Y, epochs=80, patience=80)
mlp_model.eval()
torch.save(mlp_model.state_dict(), OUT_DIR / "stacker_mlp.pt")
dummy_mlp = torch.zeros(1, N_MODELS, N_CLASSES)
torch.onnx.export(
    mlp_model.cpu(), dummy_mlp,
    str(OUT_DIR / "stacker_mlp.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=14,
)
print(f"  MLP saved -> {OUT_DIR / 'stacker_mlp.onnx'}")

# SSM full fit + ONNX
print("  Fitting SSM (full) ...")
ssm_model = train_seq_model(
    StackerSSM(), X_norm_seq, Y_seq, X_norm_seq, Y_seq,
    epochs=100, patience=100, lr=5e-4, wd=1e-3, batch_size=16
)
ssm_model.eval()
torch.save(ssm_model.state_dict(), OUT_DIR / "stacker_ssm.pt")
dummy_ssm = torch.zeros(1, N_WIN, FEATURE_DIM)
torch.onnx.export(
    ssm_model.cpu(), dummy_ssm,
    str(OUT_DIR / "stacker_ssm.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  SSM saved -> {OUT_DIR / 'stacker_ssm.onnx'}")

# Transformer full fit + ONNX
print("  Fitting Transformer (full) ...")
tfm_model = train_seq_model(
    StackerTransformer(), X_norm_seq, Y_seq, X_norm_seq, Y_seq,
    epochs=100, patience=100, lr=5e-4, wd=1e-3, batch_size=16
)
tfm_model.eval()
torch.save(tfm_model.state_dict(), OUT_DIR / "stacker_transformer.pt")
dummy_tfm = torch.zeros(1, N_WIN, FEATURE_DIM)
torch.onnx.export(
    tfm_model.cpu(), dummy_tfm,
    str(OUT_DIR / "stacker_transformer.onnx"),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_files"}, "output": {0: "batch_files"}},
    opset_version=14,
)
print(f"  Transformer saved -> {OUT_DIR / 'stacker_transformer.onnx'}")

# Meta JSON
meta_dict = {
    "best_arch"      : best_arch,
    "oof_aucs"       : {k: round(v, 6) for k, v in oof_aucs.items()},
    "n_models"       : N_MODELS,
    "n_classes"      : N_CLASSES,
    "n_windows"      : N_WIN,
    "feature_layout" : ["perch_raw", "perch_prior_fused", "mlp_probe", "proto_ssm", "sed_csebbs"],
    "feature_dim"    : FEATURE_DIM,
    "context_k"      : CONTEXT_K,
    "temperature"    : 1.5,
    "trained_date"   : time.strftime("%Y-%m-%d"),
}
with open(OUT_DIR / "stacker_meta.json", "w") as fh:
    json.dump(meta_dict, fh, indent=2)
print(f"  meta saved -> {OUT_DIR / 'stacker_meta.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. SUMMARY + EXCEL + W&B
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  OOF AUC SUMMARY")
print("=" * 60)
for k in ["baseline", "lgbm", "ridge", "mlp", "ssm", "transformer"]:
    marker = " <- best" if k == best_arch else ""
    print(f"  {k:<20} {oof_aucs.get(k, float('nan')):>10.4f}{marker}")
print("=" * 60)

rows_xl = [
    {"arch": k, "oof_macro_auc": round(oof_aucs.get(k, float("nan")), 6),
     "best": k == best_arch, "trained_date": time.strftime("%Y-%m-%d")}
    for k in ["baseline", "lgbm", "ridge", "mlp", "ssm", "transformer"]
]
df_results = pd.DataFrame(rows_xl)
excel_path  = OUT_DIR / "stacker_results.xlsx"
if excel_path.exists():
    with pd.ExcelWriter(str(excel_path), engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        df_results.to_excel(writer, sheet_name="OOF_AUC", index=False)
else:
    df_results.to_excel(str(excel_path), index=False, sheet_name="OOF_AUC")
print(f"  Excel saved -> {excel_path}")

wandb.log({
    "summary/best_arch"    : best_arch,
    "summary/best_oof_auc" : oof_aucs[best_arch],
    **{f"summary/oof_{k}": oof_aucs.get(k, float("nan"))
       for k in ["baseline", "lgbm", "ridge", "mlp", "ssm", "transformer"]},
})
wandb.finish()
print(f"\n[done]  Artifacts -> {OUT_DIR}")
