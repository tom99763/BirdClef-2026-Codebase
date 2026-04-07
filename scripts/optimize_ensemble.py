"""Optuna ensemble weight optimization.

Finds optimal linear combination weights for Perch×3 + SED models on holdout set.
Uses cached predictions (saves/loads .npy) to avoid re-running inference each trial.

Usage:
    # Full run — generates predictions then optimizes
    python scripts/optimize_ensemble.py --gpu 0

    # Use specific SED checkpoints
    python scripts/optimize_ensemble.py \
        --sed_checkpoints checkpoints/sed-b0-v5/best_sed.pt checkpoints/sed-b0-v6/best_sed.pt \
        --n_trials 200

    # Re-use cached predictions (skip inference)
    python scripts/optimize_ensemble.py --cache_dir outputs/optuna_preds_cache --skip_inference
"""

import argparse
import json
import os
import sys

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import build_species_mapping
from src.model.sed_model import SEDModel

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: optuna not installed — run: pip install optuna")
    sys.exit(1)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

HOLDOUT_CSV    = "configs/holdout_val_files.csv"
CONFIG         = "configs/default.yaml"
AUDIO_DIR      = "birdclef-2026/train_audio"
SR             = 32_000
CLIP_SAMPLES   = SR * 5

PERCH_RUNS = [
    ("nohuman-label-pseudo",           "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-label-soundscape-train", "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-embedding-soundscape",   "embedding_head", "embeddings_cache_nohuman"),
]


class MelTransform(nn.Module):
    def __init__(self, n_mels=224):
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
        B    = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn   = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx   = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel  = (mel - mn) / (mx - mn + 1e-7)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",             default=None)
    p.add_argument("--holdout_csv",     default=HOLDOUT_CSV)
    p.add_argument("--config",          default=CONFIG)
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--audio_dir",       default=AUDIO_DIR)
    p.add_argument("--cache_dir",       default="outputs/optuna_preds_cache",
                   help="Directory to save/load prediction .npy files")
    p.add_argument("--skip_inference",  action="store_true",
                   help="Skip model inference — load predictions from cache only")
    p.add_argument("--sed_checkpoints", nargs="+",
                   default=["checkpoints/sed-b0-v5/best_sed.pt",
                             "checkpoints/sed-b0-v6/best_sed.pt"],
                   help="SED checkpoints to include in ensemble")
    p.add_argument("--sed_configs",     nargs="+",
                   default=["configs/sed_b0_v5.yaml",
                             "configs/sed_b0_v6.yaml"],
                   help="SED configs matching --sed_checkpoints")
    p.add_argument("--n_trials",   type=int, default=200,
                   help="Number of Optuna trials (default: 200)")
    p.add_argument("--output",          default="outputs/ensemble_weights_optuna.json")
    return p.parse_args()


# ── Perch prediction helpers ───────────────────────────────────────────────────

def load_perch_predictions(run_name, mode, cache_name, holdout_csv,
                            species_to_idx, num_classes, checkpoints_dir,
                            config, outputs_dir="outputs"):
    """Load Perch model predictions at file level. Returns (N_files, C), y, files."""
    from src.model.classifier import PerchClassifier

    holdout    = pd.read_csv(holdout_csv)
    holdout_fs = set(holdout["filename"].unique())
    f2label    = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))

    mcsv = f"outputs/{cache_name}/manifest.csv"
    if not os.path.isfile(mcsv):
        print(f"  [{run_name}] manifest not found: {mcsv} — skipping")
        return None, None, None

    mf = pd.read_csv(mcsv)
    mf = mf[mf["source_file"].isin(holdout_fs) & (mf["split"] == "holdout")].copy()
    mf["primary_label"] = mf["source_file"].map(f2label)
    mf = mf.dropna(subset=["primary_label"])

    embs, labs, fnames = [], [], []
    for _, row in mf.iterrows():
        if not os.path.isfile(row["npy_path"]):
            continue
        embs.append(np.load(row["npy_path"]))
        labs.append(str(row["primary_label"]))
        fnames.append(row["source_file"])

    if not embs:
        print(f"  [{run_name}] no embeddings found — skipping")
        return None, None, None

    X         = np.stack(embs).astype(np.float32)
    ckpt_path = os.path.join(checkpoints_dir, run_name, "best_head")
    cfg_path  = os.path.join(outputs_dir, run_name, "config.yaml")
    run_cfg   = load_config(cfg_path) if os.path.isfile(cfg_path) else config

    model = PerchClassifier(
        perch_dir    = config.model.perch_dir,
        num_classes  = num_classes,
        mode         = mode,
        hidden_dim   = run_cfg.model.hidden_dim,
        dropout      = 0.0,
        embedding_dim = X.shape[1],
    )
    model.load_head(ckpt_path)

    clip_preds = []
    for start in range(0, len(X), 512):
        batch  = tf.constant(X[start: start + 512])
        logits = model.head(batch, training=False)
        out    = logits[0] if isinstance(logits, tuple) else logits
        clip_preds.append(tf.sigmoid(out).numpy())
    clip_preds = np.concatenate(clip_preds)

    del model
    tf.keras.backend.clear_session()

    # Aggregate to file level
    df = pd.DataFrame({"fname": fnames, "label": labs, "idx": range(len(fnames))})
    files = df["fname"].unique()
    file_preds = np.zeros((len(files), num_classes), dtype=np.float32)
    y          = np.zeros((len(files), num_classes), dtype=np.float32)
    for i, fname in enumerate(files):
        rows = df[df["fname"] == fname]["idx"].tolist()
        file_preds[i] = clip_preds[rows].mean(axis=0)
        sp = df[df["fname"] == fname]["label"].iloc[0]
        if sp in species_to_idx:
            y[i, species_to_idx[sp]] = 1.0

    print(f"  [{run_name}] {len(files)} files  "
          f"species_with_pos={(y.sum(0)>0).sum()}")
    return file_preds, y, files


# ── SED prediction helpers ─────────────────────────────────────────────────────

def load_sed_predictions(checkpoint, sed_config_path, holdout_csv,
                          audio_dir, device, mel_tf_obj, species_to_idx, num_classes):
    """Run SED inference on holdout files. Returns (N_files, C), y, files."""
    holdout    = pd.read_csv(holdout_csv)
    files      = holdout["filename"].unique()
    f2label    = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))

    sed_config = load_config(sed_config_path)
    ckpt       = torch.load(checkpoint, map_location=device, weights_only=False)
    state      = ckpt.get("model_state_dict", ckpt)
    if any("gem_pool" in k for k in state):
        state = {k.replace("gem_pool", "freq_pool"): v for k, v in state.items()}

    model = SEDModel(
        backbone    = sed_config.model.backbone,
        num_classes = num_classes,
        in_chans    = sed_config.model.get("in_chans", 3),
        pretrained  = False,
        drop_rate   = sed_config.model.get("dropout", 0.1),
        use_gem     = sed_config.model.get("use_gem", True),
        gem_p_init  = sed_config.model.get("gem_p_init", 3.0),
        n_mels      = sed_config.mel.n_mels,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"  SED ep={ckpt.get('epoch','?')} "
          f"auc={ckpt.get('metrics',{}).get('macro_auc','?')}")

    preds_list = []
    y          = np.zeros((len(files), num_classes), dtype=np.float32)

    for i, fname in enumerate(tqdm(files, desc="SED holdout", ncols=80)):
        audio_path = os.path.join(audio_dir, fname)
        try:
            audio, _ = librosa.load(audio_path, sr=SR, mono=True)
            audio    = audio.astype(np.float32)
        except Exception:
            preds_list.append(np.zeros(num_classes, dtype=np.float32))
            continue

        n_clips    = max(1, int(np.ceil(len(audio) / CLIP_SAMPLES)))
        clip_preds = []
        for ci in range(n_clips):
            clip = audio[ci * CLIP_SAMPLES: (ci + 1) * CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES:
                clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            t = torch.from_numpy(clip).unsqueeze(0).to(device)
            with torch.no_grad():
                mel = mel_tf_obj(t)
                out = model(mel)
            prob = (out[0] if isinstance(out, tuple) else out).cpu().numpy()[0]
            clip_preds.append(prob)

        # Max-pool over clips
        preds_list.append(np.stack(clip_preds).max(axis=0))
        sp = f2label.get(fname, "")
        if sp in species_to_idx:
            y[i, species_to_idx[sp]] = 1.0

    del model
    return np.stack(preds_list).astype(np.float32), y, files


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config         = load_config(args.config)
    target_species, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes    = len(target_species)
    print(f"Classes: {num_classes}")

    os.makedirs(args.cache_dir, exist_ok=True)

    all_preds = {}   # name → (N, C)
    y_ref     = None
    files_ref = None

    # ── Perch models ─────────────────────────────────────────────────────────
    if TF_AVAILABLE and not args.skip_inference:
        for run_name, mode, cache_name in PERCH_RUNS:
            cache_path = os.path.join(args.cache_dir, f"{run_name}.npy")
            y_path     = os.path.join(args.cache_dir, f"{run_name}_y.npy")
            f_path     = os.path.join(args.cache_dir, f"{run_name}_files.npy")
            if os.path.isfile(cache_path):
                print(f"[{run_name}] loading from cache …")
                preds = np.load(cache_path)
                y_arr = np.load(y_path)
                files = np.load(f_path, allow_pickle=True)
            else:
                print(f"\n[{run_name}] generating predictions …")
                preds, y_arr, files = load_perch_predictions(
                    run_name, mode, cache_name, args.holdout_csv,
                    species_to_idx, num_classes, args.checkpoints_dir, config,
                )
                if preds is None:
                    continue
                np.save(cache_path, preds)
                np.save(y_path, y_arr)
                np.save(f_path, np.array(files))

            all_preds[run_name] = preds
            if y_ref is None:
                y_ref     = y_arr
                files_ref = files
    else:
        # Try loading Perch from cache
        for run_name, _, _ in PERCH_RUNS:
            cache_path = os.path.join(args.cache_dir, f"{run_name}.npy")
            y_path     = os.path.join(args.cache_dir, f"{run_name}_y.npy")
            f_path     = os.path.join(args.cache_dir, f"{run_name}_files.npy")
            if os.path.isfile(cache_path):
                print(f"[{run_name}] loading from cache")
                preds = np.load(cache_path)
                y_arr = np.load(y_path)
                files = np.load(f_path, allow_pickle=True)
                all_preds[run_name] = preds
                if y_ref is None:
                    y_ref     = y_arr
                    files_ref = files

    # ── SED models ───────────────────────────────────────────────────────────
    mel_tf = MelTransform(n_mels=224).to(device)
    mel_tf.eval()

    for i, (ckpt, cfg_path) in enumerate(zip(args.sed_checkpoints, args.sed_configs)):
        run_name = os.path.basename(os.path.dirname(ckpt))
        if not run_name:
            run_name = f"sed_{i}"
        cache_path = os.path.join(args.cache_dir, f"{run_name}.npy")
        y_path     = os.path.join(args.cache_dir, f"{run_name}_y.npy")
        f_path     = os.path.join(args.cache_dir, f"{run_name}_files.npy")

        if os.path.isfile(cache_path) and args.skip_inference:
            print(f"[{run_name}] loading from cache")
            preds = np.load(cache_path)
            y_arr = np.load(y_path)
            files = np.load(f_path, allow_pickle=True)
        elif not os.path.isfile(ckpt):
            print(f"[{run_name}] checkpoint not found: {ckpt} — skipping")
            continue
        elif not os.path.isfile(cache_path):
            print(f"\n[{run_name}] generating SED predictions …")
            preds, y_arr, files = load_sed_predictions(
                ckpt, cfg_path, args.holdout_csv, args.audio_dir,
                device, mel_tf, species_to_idx, num_classes,
            )
            np.save(cache_path, preds)
            np.save(y_path, y_arr)
            np.save(f_path, np.array(files, dtype=object))
        else:
            print(f"[{run_name}] loading from cache")
            preds = np.load(cache_path)
            y_arr = np.load(y_path)
            files = np.load(f_path, allow_pickle=True)

        all_preds[run_name] = preds
        if y_ref is None:
            y_ref     = y_arr
            files_ref = files

    if not all_preds:
        print("ERROR: no model predictions available — nothing to optimize")
        return

    model_names = list(all_preds.keys())
    preds_arr   = np.stack([all_preds[n] for n in model_names], axis=0)  # (M, N, C)
    species_with_pos = np.where(y_ref.sum(0) > 0)[0]
    print(f"\nModels: {model_names}")
    print(f"Shape:  {preds_arr.shape}  y={y_ref.shape}  "
          f"species_with_pos={len(species_with_pos)}")

    # ── Baseline (equal weights) ───────────────────────────────────────────
    equal_w    = np.ones(len(model_names)) / len(model_names)
    equal_pred = (preds_arr * equal_w[:, None, None]).sum(0)
    equal_auc  = roc_auc_score(
        y_ref[:, species_with_pos], equal_pred[:, species_with_pos], average="macro"
    )
    print(f"\nBaseline (equal weights): {equal_auc:.4f}")

    # ── Optuna ────────────────────────────────────────────────────────────
    def objective(trial):
        # Sample weights using Dirichlet-like approach via softmax of log-uniforms
        raw = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(model_names))]
        w   = np.array(raw, dtype=np.float32)
        w   = w / w.sum()
        combined = (preds_arr * w[:, None, None]).sum(0)
        try:
            return roc_auc_score(
                y_ref[:, species_with_pos], combined[:, species_with_pos], average="macro"
            )
        except Exception:
            return 0.0

    print(f"\nRunning Optuna ({args.n_trials} trials) …")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    raw  = np.array([best.params[f"w_{i}"] for i in range(len(model_names))])
    best_w = raw / raw.sum()

    print(f"\n{'='*60}")
    print(f"  Optuna Best AUC : {best.value:.4f}  (vs equal: {equal_auc:.4f}  "
          f"gain={best.value-equal_auc:+.4f})")
    print(f"  Best weights:")
    for name, w in zip(model_names, best_w):
        print(f"    {name:<40} {w:.4f}")
    print(f"{'='*60}")

    # ── Save results ──────────────────────────────────────────────────────
    result = {
        "models":       model_names,
        "best_weights": {n: round(float(w), 6) for n, w in zip(model_names, best_w)},
        "best_auc":     round(float(best.value), 6),
        "equal_w_auc":  round(float(equal_auc), 6),
        "gain":         round(float(best.value - equal_auc), 6),
        "n_trials":     args.n_trials,
        "n_files":      int(y_ref.shape[0]),
        "species_with_pos": int(len(species_with_pos)),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
