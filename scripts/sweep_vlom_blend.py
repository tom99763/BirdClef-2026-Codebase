"""Sweep PERCH_W / SED_W blend on soundscape val set (59 files, 4-fold OOF).

Uses the same Perch OOF as eval_smooth_experiments.py (with LGBM probe, best postproc).
Runs SED inference on those 59 soundscape files for each model in SED_CHECKPOINTS.
Evaluates macro-AUC at each (perch_w, sed_w) combination.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/sweep_vlom_blend.py
    CUDA_VISIBLE_DEVICES=1 python scripts/sweep_vlom_blend.py --sed_smooth r51  # apply R51 postproc to SED
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio.transforms as T
import librosa
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.ndimage import convolve1d
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import build_species_mapping
from src.model.sed_model import SEDModel

# ── Paths (same as eval_smooth_experiments.py) ────────────────────────────────
BASE         = Path("birdclef-2026")
SS_DIR       = BASE / "train_soundscapes"
CACHE_EXT    = Path("outputs/perch_cache_extended.npz")
META_PARQUET = Path("birdclef-2026/notebook resource/best perch/perch meta/full_perch_meta.parquet")
OOF_NPZ      = Path("birdclef-2026/notebook resource/best perch/perch meta/full_oof_meta_features.npz")
PROBE_PKL    = Path("submissions_v3/weights/lgbm_probe_models.pkl")
FOLDS_DIR    = Path("configs/ss_folds")

SR           = 32_000
CLIP_SAMPLES = SR * 5
N_WINDOWS    = 12
TEMP_SCALE   = 1.15   # matches eval_smooth_experiments.py

# ── SED checkpoints to evaluate ───────────────────────────────────────────────
SED_CHECKPOINTS = [
    {
        "name":     "v9-asl-soup",
        "path":     "submissions/weights/soup_sed-b0-v9-asl.pt",
        "config":   "configs/sed_b0_v33_warmrestart.yaml",   # same arch: b0, n_mels=224, in_chans=3
        "weight":   1.0,
    },
    {
        "name":     "v30-multipseu-soup",
        "path":     "submissions/weights/soup_sed-b0-v30-multipseu.pt",
        "config":   "configs/sed_b0_v33_warmrestart.yaml",
        "weight":   1.0,
    },
]

# Optional: add competitor
COMPETITOR = {
    "name":   "competitor",
    "path":   "submissions/weights/competitor_sed_fold0.pt",
    "config": "configs/sed_b0_v33_warmrestart.yaml",
    "weight": 1.0,
}


# ── Model helpers ─────────────────────────────────────────────────────────────
class MelTransform(nn.Module):
    def __init__(self, n_mels=224, in_chans=3):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=2048, hop_length=512,
            n_mels=n_mels, f_min=0, f_max=16000,
            power=2.0, norm="slaney", mel_scale="htk",
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=80.0)

    @torch.no_grad()
    def forward(self, waveforms):
        peak = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
        waveforms = waveforms / peak
        mel = self.db(self.mel(waveforms))
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mn) / (mx - mn + 1e-7)
        return mel.unsqueeze(1).repeat(1, self.in_chans, 1, 1)

    @property
    def in_chans(self):
        return 3


def load_sed_model(ckpt_path, config_path, device):
    cfg = load_config(config_path)
    n_classes = 234
    model = SEDModel(
        backbone=cfg.model.backbone,
        num_classes=n_classes,
        pretrained=False,
        drop_rate=cfg.model.get("dropout", 0.1),
        in_chans=cfg.model.get("in_chans", 3),
        use_gem=cfg.model.get("use_gem", True),
        gem_p_init=cfg.model.get("gem_p_init", 3.0),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    n_mels = cfg.mel.get("n_mels", 224)
    in_chans = cfg.model.get("in_chans", 3)
    mel_fn = MelTransform(n_mels=n_mels, in_chans=in_chans).to(device)
    return model, mel_fn


@torch.no_grad()
def run_sed_on_file(ogg_path, model, mel_fn, device, n_classes=234):
    try:
        wav, _ = librosa.load(str(ogg_path), sr=SR, mono=True)
    except Exception:
        return np.zeros((N_WINDOWS, n_classes), dtype=np.float32)

    clips = np.zeros((N_WINDOWS, CLIP_SAMPLES), dtype=np.float32)
    for i in range(N_WINDOWS):
        s = i * CLIP_SAMPLES
        e = s + CLIP_SAMPLES
        chunk = wav[s:e]
        if len(chunk) < CLIP_SAMPLES:
            chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))
        clips[i] = chunk

    wav_t = torch.tensor(clips, dtype=torch.float32).to(device)
    mel = mel_fn(wav_t)                     # (N_WINDOWS, C, H, W)
    clip_pred, _ = model(mel)               # (N_WINDOWS, 234)
    return clip_pred.cpu().numpy().astype(np.float32)


# ── Post-processing helpers ────────────────────────────────────────────────────
GAUSSIAN_KERN = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)

def gaussian_smooth_logits(logits):
    """Gaussian smooth in logit space, per file."""
    n_files = logits.shape[0] // N_WINDOWS
    X = logits.reshape(n_files, N_WINDOWS, -1)
    out = np.stack([convolve1d(X[i], GAUSSIAN_KERN, axis=0, mode="nearest") for i in range(n_files)])
    return out.reshape(-1, logits.shape[-1])


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def logit(p, eps=1e-6): return np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps))


def lmax_pre_aves(logits, alpha=0.1, radius=1):
    """Local-max propagation in logit space, Aves-only (indices 72-233). R50/R51 best."""
    n_files = logits.shape[0] // N_WINDOWS
    X = logits.reshape(n_files, N_WINDOWS, logits.shape[-1]).copy()
    aves_idx = list(range(72, logits.shape[-1]))
    for f in range(n_files):
        for t in range(N_WINDOWS):
            t0 = max(0, t - radius)
            t1 = min(N_WINDOWS, t + radius + 1)
            nbr_max = X[f, t0:t1, :][:, aves_idx].max(axis=0)
            X[f, t, aves_idx] = (1 - alpha) * X[f, t, aves_idx] + alpha * nbr_max
    return X.reshape(-1, logits.shape[-1])


# ── Main ──────────────────────────────────────────────────────────────────────
def build_ground_truth(n_classes=234):
    ss_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
    ss_labels["primary_label"] = ss_labels["primary_label"].astype(str)

    def parse_labels(x):
        return [t.strip() for t in str(x).split(";") if t.strip()] if not pd.isna(x) else []

    sc = (
        ss_labels.groupby(["filename", "start", "end"])["primary_label"]
        .apply(lambda s: sorted(set(lbl for x in s for lbl in parse_labels(x))))
        .reset_index(name="label_list")
    )
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"]  = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    full_files = set(sc.groupby("filename").filter(lambda g: len(g) == N_WINDOWS)["filename"].unique())
    sc = sc[sc["filename"].isin(full_files)].sort_values(["filename", "end_sec"]).reset_index(drop=True)

    # Load species mapping to get n_classes=234 index
    (PRIMARY_LABELS, label_to_idx, *_) = _load_species_mapping(n_classes)

    Y = np.zeros((len(sc), n_classes), dtype=np.float32)
    for i, labels in enumerate(sc["label_list"]):
        for lbl in labels:
            if lbl in label_to_idx:
                Y[i, label_to_idx[lbl]] = 1.0
    return sc, Y, sorted(full_files)


def _load_species_mapping(n_classes=234):
    taxonomy = pd.read_csv(BASE / "taxonomy.csv")
    sample_sub = pd.read_csv(BASE / "sample_submission.csv")
    PRIMARY_LABELS = [c for c in sample_sub.columns if c != "row_id"]
    label_to_idx = {lbl: i for i, lbl in enumerate(PRIMARY_LABELS)}
    return PRIMARY_LABELS, label_to_idx, None, None, None, None, None, None


def macro_auc(Y, preds, fold_mask=None):
    if fold_mask is not None:
        Y = Y[fold_mask]; preds = preds[fold_mask]
    keep = Y.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    aucs = []
    for c in range(Y.shape[1]):
        if keep[c] and len(np.unique(Y[:, c])) > 1:
            aucs.append(roc_auc_score(Y[:, c], preds[:, c]))
    return float(np.mean(aucs)) if aucs else 0.0


def build_perch_oof(sc_df, full_files_sorted, Y, n_classes=234):
    """Rebuild Perch OOF (LGBM probe) matching eval_smooth_experiments.py exactly.

    Key fixes vs previous broken version:
    1. Fold splits loaded from configs/ss_folds/ text files (not np.array_split)
    2. Scaler + PCA loaded from PROBE_PKL (not re-fit on 59-file subset)
    3. LGBM hyperparams match eval_smooth (n_est=100, max_depth=4, num_leaves=15)
    """
    import pickle
    import lightgbm as lgb
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # ── Load cache ────────────────────────────────────────────────────────────
    cache = np.load(str(CACHE_EXT))
    scores_raw = cache["scores_full_raw"]   # (N_windows, 234) — Perch logits
    emb_full   = cache["emb_full"]          # (N_windows, 1536)
    row_ids    = cache["row_ids"]           # (N_windows,)

    # Match ordering to sc_df (preserves sc_df row order → aligned with Y)
    row_id_to_idx = {rid: i for i, rid in enumerate(row_ids)}
    sel = [row_id_to_idx[rid] for rid in sc_df["row_id"] if rid in row_id_to_idx]
    if len(sel) != len(sc_df):
        print(f"Warning: matched {len(sel)}/{len(sc_df)} row_ids in Perch cache")
    scores_raw_sel = scores_raw[sel]   # (N_windows, 234)
    emb_sel        = emb_full[sel]     # (N_windows, 1536)

    # ── Load hyperparams from PROBE_PKL; fit fresh PCA on soundscape embs ────
    # IMPORTANT: do NOT use probe_w["emb_scaler"/"emb_pca"] — those were fitted
    # on train_audio data, applying them to soundscape embeddings causes domain
    # shift and anti-correlated LGBM predictions. Fit fresh on emb_sel instead,
    # exactly as eval_smooth_experiments.py does.
    probe_w = pickle.load(open(str(PROBE_PKL), "rb"))
    FROZEN  = probe_w["frozen_probe"]
    PCA_DIM = int(FROZEN.get("pca_dim", 64))
    ALPHA   = float(FROZEN.get("alpha", 0.40))
    MIN_POS = int(FROZEN.get("min_pos", 8))
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    pca    = PCA(n_components=PCA_DIM, whiten=True, random_state=42)
    Z_all  = pca.fit_transform(scaler.fit_transform(emb_sel)).astype(np.float32)
    print(f"  PCA fit+transform: emb (N,1536) → Z (N,{Z_all.shape[1]})  [fresh on soundscape, alpha={ALPHA}]")

    # ── Fold assignment from text files (same as eval_smooth_experiments.py) ──
    files_arr  = np.array([sc_df["filename"].iloc[i] for i in range(len(sc_df))])
    fold_id    = np.full(len(files_arr), -1, dtype=np.int32)
    for k in range(4):
        fold_file = FOLDS_DIR / f"ss_fold{k}_val.txt"
        val_files = set(fold_file.read_text().splitlines()) if fold_file.exists() else set()
        mask = np.array([f in val_files for f in files_arr])
        fold_id[mask] = k
    n_assigned = (fold_id >= 0).sum()
    print(f"  Fold assignment: {n_assigned}/{len(files_arr)} windows assigned to folds 0-3")

    # ── OOF LGBM probe (same params as eval_smooth_experiments.py) ────────────
    lgbm_params = dict(n_estimators=100, max_depth=4, num_leaves=15,
                       learning_rate=0.05, min_child_samples=5,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=4, verbose=-1)

    oof_base  = scores_raw_sel.copy()
    oof_final = oof_base.copy()
    active    = np.where(Y.sum(axis=0) >= MIN_POS)[0]
    print(f"  Training OOF LGBM probe: {len(active)} active classes (≥{MIN_POS} positives)...")

    for cls_idx in tqdm(active, desc="OOF LGBM", leave=False):
        y_cls         = Y[:, cls_idx]
        prior         = oof_base[:, cls_idx]
        X_all         = np.column_stack([Z_all, prior])
        oof_pred_logit = np.zeros(len(y_cls), dtype=np.float32)

        for k in range(4):
            val_mask   = fold_id == k
            train_mask = (fold_id >= 0) & (~val_mask)
            # Skip if too few training examples or not enough positives to learn from
            if train_mask.sum() < 5 or val_mask.sum() == 0:
                continue
            if y_cls[train_mask].sum() < 2:
                # Not enough positives to train — leave oof_pred_logit=0 for this fold
                # (blending with 0 scales oof_base by (1-ALPHA), preserving rank)
                continue
            clf = lgb.LGBMClassifier(**lgbm_params)
            clf.fit(X_all[train_mask], y_cls[train_mask])
            proba = np.clip(clf.predict_proba(X_all[val_mask])[:, 1], 1e-7, 1 - 1e-7)
            oof_pred_logit[val_mask] = np.log(proba / (1 - proba))

        val_windows = fold_id >= 0
        blended     = (1 - ALPHA) * oof_base[:, cls_idx] + ALPHA * oof_pred_logit
        oof_final[val_windows, cls_idx] = blended[val_windows]

    return oof_final, fold_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",         default="1")
    parser.add_argument("--sed_smooth",  default="r51",
                        choices=["none","gauss","r51"],
                        help="Post-processing on SED probs before blend")
    parser.add_argument("--add_competitor", action="store_true",
                        help="Include competitor SED in SED branch")
    parser.add_argument("--perch_smooth", default="r51",
                        choices=["none","gauss","r51"],
                        help="Post-processing on Perch logits")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Ground truth ──────────────────────────────────────────────────────────
    print("Building ground truth...")
    sc_df, Y, full_files = build_ground_truth()
    print(f"  {len(full_files)} soundscape files, {len(sc_df)} windows, Y={Y.shape}")

    # ── Perch OOF ─────────────────────────────────────────────────────────────
    print("Building Perch OOF (LGBM probe)...")
    perch_logits, fold_id = build_perch_oof(sc_df, full_files, Y)
    print(f"  perch_logits: {perch_logits.shape}")

    # Apply Perch post-processing
    if args.perch_smooth == "r51":
        perch_logits_pp = lmax_pre_aves(perch_logits, alpha=0.1, radius=1)
        print("  Perch post-proc: lmax_pre_aves(alpha=0.1)")
    elif args.perch_smooth == "gauss":
        perch_logits_pp = gaussian_smooth_logits(perch_logits)
        print("  Perch post-proc: Gaussian")
    else:
        perch_logits_pp = perch_logits
        print("  Perch post-proc: none")

    perch_probs = sigmoid(perch_logits_pp / TEMP_SCALE)

    # Baseline: Perch only
    val_mask = fold_id >= 0
    perch_auc = macro_auc(Y, perch_probs, val_mask)
    print(f"\nPerch-only AUC: {perch_auc:.4f}")

    # ── SED inference on soundscapes ──────────────────────────────────────────
    checkpoints = list(SED_CHECKPOINTS)
    if args.add_competitor and os.path.exists(COMPETITOR["path"]):
        checkpoints.append(COMPETITOR)

    sed_preds_list = []
    for ckpt_info in checkpoints:
        if not os.path.exists(ckpt_info["path"]):
            print(f"  SKIP {ckpt_info['name']}: not found at {ckpt_info['path']}")
            continue
        print(f"\nRunning SED inference: {ckpt_info['name']}")
        model, mel_fn = load_sed_model(ckpt_info["path"], ckpt_info["config"], device)

        # Infer on each soundscape file in sc_df order
        all_preds = []
        unique_fnames = list(dict.fromkeys(sc_df["filename"].tolist()))
        for fname in tqdm(unique_fnames, desc=ckpt_info["name"]):
            ogg_path = SS_DIR / fname
            preds = run_sed_on_file(ogg_path, model, mel_fn, device)  # (12, 234)
            all_preds.append(preds)
        sed_preds = np.concatenate(all_preds, axis=0)   # (N_windows, 234)
        print(f"  SED preds: {sed_preds.shape}  mean={sed_preds.mean():.4f}")

        # Apply SED post-processing
        if args.sed_smooth == "r51":
            sed_logits = logit(np.clip(sed_preds, 1e-6, 1-1e-6))
            sed_logits = lmax_pre_aves(sed_logits, alpha=0.1, radius=1)
            sed_probs  = sigmoid(sed_logits)
            print("  SED post-proc: lmax_pre_aves(alpha=0.1)")
        elif args.sed_smooth == "gauss":
            sed_logits = logit(np.clip(sed_preds, 1e-6, 1-1e-6))
            sed_logits = gaussian_smooth_logits(sed_logits)
            sed_probs  = sigmoid(sed_logits)
            print("  SED post-proc: Gaussian")
        else:
            sed_probs = sed_preds
            print("  SED post-proc: none (raw probs)")

        sed_auc = macro_auc(Y, sed_probs, val_mask)
        print(f"  SED-only AUC ({ckpt_info['name']}): {sed_auc:.4f}")
        sed_preds_list.append((ckpt_info["name"], ckpt_info["weight"], sed_probs))

    if not sed_preds_list:
        print("No SED models loaded. Exiting.")
        return

    # Average SED branch
    total_w = sum(w for _, w, _ in sed_preds_list)
    sed_avg = sum(w * p for _, w, p in sed_preds_list) / total_w
    sed_avg_auc = macro_auc(Y, sed_avg, val_mask)
    print(f"\nSED-branch avg AUC ({[n for n,_,_ in sed_preds_list]}): {sed_avg_auc:.4f}")

    # ── Blend sweep ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PERCH_W / SED_W blend sweep (in PROBABILITY space)")
    print("="*60)
    print(f"{'perch_w':>8}  {'sed_w':>6}  {'AUC':>8}")
    print("-"*30)

    best_auc = 0.0
    best_weights = (0.5, 0.5)
    results = []

    perch_weights = np.arange(0.0, 1.05, 0.05)
    for pw in perch_weights:
        sw = 1.0 - pw
        blended = pw * perch_probs + sw * sed_avg
        auc = macro_auc(Y, blended, val_mask)
        marker = " <-- BEST" if auc > best_auc else ""
        print(f"  {pw:6.2f}    {sw:6.2f}   {auc:.4f}{marker}")
        results.append((pw, sw, auc))
        if auc > best_auc:
            best_auc = auc
            best_weights = (pw, sw)

    print(f"\nBEST: PERCH_W={best_weights[0]:.2f}  SED_W={best_weights[1]:.2f}  AUC={best_auc:.4f}")
    print(f"Perch-only: {perch_auc:.4f}  |  SED-only: {sed_avg_auc:.4f}")

    # Save results
    out_path = "outputs/vlom_blend_sweep.csv"
    df_res = pd.DataFrame(results, columns=["perch_w", "sed_w", "auc"])
    df_res.to_csv(out_path, index=False)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
