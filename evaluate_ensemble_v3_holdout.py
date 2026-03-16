"""Ensemble holdout eval v3: Perch×3 (label-pseudo, label-soundscape, embedding) + SED (sed-b0-v5).

Evaluates combinations at FILE level (mean-pool clips → file prediction).
SED runs on raw holdout audio; Perch models use pre-cached embeddings.

Usage:
    python evaluate_ensemble_v3_holdout.py
    python evaluate_ensemble_v3_holdout.py --gpu 0
    python evaluate_ensemble_v3_holdout.py --sed_checkpoint checkpoints/sed-b0-v5/best_sed.pt
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torchaudio.transforms as T
import librosa
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataset import build_species_mapping
from src.model.classifier import PerchClassifier
from src.model.sed_model import SEDModel, GEMFreqPool, AttentionSEDHead


PERCH_RUNS = [
    ("nohuman-label-pseudo",          "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-label-soundscape-train", "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-embedding-soundscape",   "embedding_head", "embeddings_cache_nohuman"),
]
HOLDOUT_CSV     = "configs/holdout_val_files.csv"
CONFIG          = "configs/default.yaml"
SED_CHECKPOINT  = "checkpoints/sed-b0-v5/best_sed.pt"
AUDIO_DIR       = "birdclef-2026/train_audio"
SR              = 32_000
CLIP_SAMPLES    = SR * 5


# ── Mel transform matching train_sed.py apply_gpu_mel ─────────────────────────

class MelTransform(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, n_mels=224, fmin=0, fmax=16000, sr=32000):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax,
            power=2.0, norm="slaney", mel_scale="htk",
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=80.0)

    @torch.no_grad()
    def forward(self, waveforms):
        # Per-sample peak normalization (matches train_sed.py)
        peak = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
        waveforms = waveforms / peak
        mel = self.db(self.mel(waveforms))
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        mel_min = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mel_max = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, n_mels, T)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",             default=None)
    p.add_argument("--holdout_csv",     default=HOLDOUT_CSV)
    p.add_argument("--config",          default=CONFIG)
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--outputs_dir",     default="outputs")
    p.add_argument("--sed_checkpoint",  default=SED_CHECKPOINT)
    p.add_argument("--audio_dir",       default=AUDIO_DIR)
    return p.parse_args()


def load_holdout_embeddings_file_level(holdout_csv, cache_name, species_to_idx, num_classes):
    """Load cached embeddings, return file-level mean predictions after model inference."""
    holdout = pd.read_csv(holdout_csv)
    holdout_files = set(holdout["filename"].unique())
    file_to_label = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))

    mcsv = f"outputs/{cache_name}/manifest.csv"
    mf   = pd.read_csv(mcsv)
    mf   = mf[mf["source_file"].isin(holdout_files) & (mf["split"] == "holdout")].copy()
    mf["primary_label"] = mf["source_file"].map(file_to_label)
    mf   = mf.dropna(subset=["primary_label"])
    print(f"  [{cache_name}] {len(mf)} clips  files={mf['source_file'].nunique()}")

    embs, labs, fnames = [], [], []
    for _, row in mf.iterrows():
        if not os.path.isfile(row["npy_path"]):
            continue
        embs.append(np.load(row["npy_path"]))
        labs.append(str(row["primary_label"]))
        fnames.append(row["source_file"])

    X = np.stack(embs).astype(np.float32)
    species_with_pos_all = None
    return X, labs, fnames, species_with_pos_all


def predict_clip_level(model, X, batch_size=512):
    preds = []
    for start in range(0, len(X), batch_size):
        batch  = tf.constant(X[start: start + batch_size])
        logits = model.head(batch, training=False)
        out    = logits[0] if isinstance(logits, tuple) else logits
        preds.append(tf.sigmoid(out).numpy())
    return np.concatenate(preds, axis=0)


def aggregate_to_file_level(clip_preds, fnames, labs, species_to_idx, num_classes):
    """Mean-pool clip predictions per file, build y matrix."""
    df = pd.DataFrame({"fname": fnames, "label": labs})
    df["idx"] = range(len(fnames))
    files = df["fname"].unique()

    file_preds = np.zeros((len(files), num_classes), dtype=np.float32)
    y          = np.zeros((len(files), num_classes), dtype=np.float32)

    for i, fname in enumerate(files):
        rows = df[df["fname"] == fname]["idx"].tolist()
        file_preds[i] = clip_preds[rows].mean(axis=0)
        sp = df[df["fname"] == fname]["label"].iloc[0]
        if sp in species_to_idx:
            y[i, species_to_idx[sp]] = 1.0

    species_with_pos = np.where(y.sum(0) > 0)[0]
    print(f"  file-level: {len(files)} files  species_with_pos={len(species_with_pos)}/234")
    return file_preds, y, species_with_pos, files


def predict_sed_file_level(sed_model, mel_tf, audio_dir, files, num_classes, species_to_idx, file_to_label, batch_size=16):
    """Run SED on raw audio for all holdout files. Returns (N_files, 234) predictions."""
    sed_model.eval()
    preds_list = []

    for fname in tqdm(files, desc="SED inference", ncols=80):
        audio_path = os.path.join(audio_dir, fname)
        try:
            audio, _ = librosa.load(audio_path, sr=SR, mono=True)
            audio = audio.astype(np.float32)
        except Exception as e:
            print(f"  WARN: cannot load {fname}: {e}")
            preds_list.append(np.zeros(num_classes, dtype=np.float32))
            continue

        # Split into 5-second clips
        n_clips = max(1, int(np.ceil(len(audio) / CLIP_SAMPLES)))
        clips = []
        for i in range(n_clips):
            clip = audio[i * CLIP_SAMPLES: (i + 1) * CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES:
                clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            clips.append(clip)

        clip_preds = []
        for b_start in range(0, len(clips), batch_size):
            batch_np = np.stack(clips[b_start: b_start + batch_size])
            t = torch.from_numpy(batch_np)
            with torch.no_grad():
                mel = mel_tf(t)
                out = sed_model(mel)
            clip_prob = out[0] if isinstance(out, tuple) else out
            clip_preds.append(clip_prob.cpu().numpy())

        clip_preds = np.concatenate(clip_preds, axis=0)
        # Max-pool over clips (matches competition notebook inference)
        preds_list.append(clip_preds.max(axis=0))

    return np.stack(preds_list).astype(np.float32)


def roc_auc(y, preds, species_with_pos):
    try:
        return roc_auc_score(
            y[:, species_with_pos],
            preds[:, species_with_pos],
            average="macro",
        )
    except Exception as e:
        print(f"  Scoring error: {e}")
        return None


def pp_threshold(preds, threshold=0.02):
    out = preds.copy()
    out[out < threshold] = 0.0
    return out


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)
    target_species, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(target_species)
    print(f"Species: {num_classes}\n")

    holdout = pd.read_csv(args.holdout_csv)
    file_to_label = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))

    all_preds = {}          # run_name → (N_files, 234) file-level preds
    y_ref, swp_ref = None, None
    files_ref = None

    # ── Load Perch models ────────────────────────────────────────────────────
    for run_name, mode, cache_name in PERCH_RUNS:
        ckpt_path    = os.path.join(args.checkpoints_dir, run_name, "best_head")
        run_cfg_path = os.path.join(args.outputs_dir, run_name, "config.yaml")

        if not (os.path.isfile(ckpt_path + ".weights.h5") or os.path.isfile(ckpt_path)):
            print(f"[{run_name}] checkpoint not found — skipping")
            continue

        print(f"\n[{run_name}] mode={mode}  loading embeddings …")
        X, labs, fnames, _ = load_holdout_embeddings_file_level(
            args.holdout_csv, cache_name, species_to_idx, num_classes
        )

        run_config = load_config(run_cfg_path) if os.path.isfile(run_cfg_path) else config
        emb_dim    = X.shape[1]

        print(f"[{run_name}] loading model …")
        model = PerchClassifier(
            perch_dir=config.model.perch_dir,
            num_classes=num_classes,
            mode=mode,
            hidden_dim=run_config.model.hidden_dim,
            dropout=0.0,
            embedding_dim=emb_dim,
        )
        model.load_head(ckpt_path)
        clip_preds = predict_clip_level(model, X)
        del model
        tf.keras.backend.clear_session()

        file_preds, y, swp, files = aggregate_to_file_level(
            clip_preds, fnames, labs, species_to_idx, num_classes
        )
        all_preds[run_name] = file_preds

        if y_ref is None:
            y_ref, swp_ref, files_ref = y, swp, files

    # ── Load SED model ───────────────────────────────────────────────────────
    sed_preds = None
    sed_label = "sed-b0-v5"
    if os.path.isfile(args.sed_checkpoint):
        print(f"\n[{sed_label}] loading SED checkpoint: {args.sed_checkpoint}")
        ckpt = torch.load(args.sed_checkpoint, map_location="cpu", weights_only=False)
        sed_cfg_ep = ckpt.get("epoch", "?")
        sed_val    = ckpt.get("metrics", {}).get("macro_auc", "?")
        print(f"  epoch={sed_cfg_ep}  val_macro_auc={sed_val}")

        sed_model = SEDModel(
            backbone="tf_efficientnet_b0.ns_jft_in1k",
            num_classes=num_classes,
            in_chans=3,
            pretrained=False,
            drop_rate=0.1,
            use_gem=True,
            gem_p_init=3.0,
            n_mels=224,
        )
        sed_model.load_state_dict(ckpt["model_state_dict"])
        sed_model.eval()

        mel_tf = MelTransform()
        mel_tf.eval()

        if files_ref is None:
            # Build file list from holdout CSV directly
            files_ref = holdout["filename"].unique()
            y_ref = np.zeros((len(files_ref), num_classes), dtype=np.float32)
            for i, fname in enumerate(files_ref):
                sp = file_to_label.get(fname, "")
                if sp in species_to_idx:
                    y_ref[i, species_to_idx[sp]] = 1.0
            swp_ref = np.where(y_ref.sum(0) > 0)[0]

        sed_preds = predict_sed_file_level(
            sed_model, mel_tf, args.audio_dir, files_ref,
            num_classes, species_to_idx, file_to_label,
        )
        all_preds[sed_label] = sed_preds
        print(f"  SED pred_range=[{sed_preds.min():.4f}, {sed_preds.max():.4f}]")
    else:
        print(f"\n[{sed_label}] checkpoint not found ({args.sed_checkpoint}) — SED skipped")

    if not all_preds:
        print("No models loaded — aborting.")
        return

    # ── Build ensemble combinations ──────────────────────────────────────────
    label_pseudo = all_preds.get("nohuman-label-pseudo")
    label_ss     = all_preds.get("nohuman-label-soundscape-train")
    emb_ss       = all_preds.get("nohuman-embedding-soundscape")
    sed_p        = all_preds.get(sed_label)

    results = []

    # Individual models
    for name, preds in [("label-pseudo", label_pseudo),
                        ("label-soundscape-train", label_ss),
                        ("embedding-soundscape", emb_ss),
                        (sed_label, sed_p)]:
        if preds is not None:
            results.append((name, preds, "raw"))

    # 2-model
    if label_pseudo is not None and label_ss is not None:
        ens_2 = (label_pseudo + label_ss) / 2.0
        results.append(("ensemble(label×2)",    ens_2, "raw"))
        results.append(("ensemble(label×2)+PP", pp_threshold(ens_2), "+PP(0.02)"))

    # 3-model Perch only
    if label_pseudo is not None and label_ss is not None and emb_ss is not None:
        ens_3 = (label_pseudo + label_ss + emb_ss) / 3.0
        results.append(("ensemble(label×2+emb)",    ens_3, "raw"))
        results.append(("ensemble(label×2+emb)+PP", pp_threshold(ens_3), "+PP(0.02)"))

    # 3-model with SED
    if label_pseudo is not None and label_ss is not None and sed_p is not None:
        ens_3s = (label_pseudo + label_ss + sed_p) / 3.0
        results.append(("ensemble(label×2+SED)",    ens_3s, "raw"))
        results.append(("ensemble(label×2+SED)+PP", pp_threshold(ens_3s), "+PP(0.02)"))

    # 4-model full
    if label_pseudo is not None and label_ss is not None and emb_ss is not None and sed_p is not None:
        ens_4 = (label_pseudo + label_ss + emb_ss + sed_p) / 4.0
        results.append(("ensemble(label×2+emb+SED)",    ens_4, "raw"))
        results.append(("ensemble(label×2+emb+SED)+PP", pp_threshold(ens_4), "+PP(0.02)"))

    # ── Score ────────────────────────────────────────────────────────────────
    scored = []
    for name, preds, pp in results:
        score = roc_auc(y_ref, preds, swp_ref)
        scored.append((name, score, pp))

    # ── Print ────────────────────────────────────────────────────────────────
    valid_scores = [s for _, s, _ in scored if s]
    best_score = max(valid_scores) if valid_scores else None

    print(f"\n{'='*72}")
    print(f"  {'Model':<42}  {'Holdout AUC':>11}  PP")
    print(f"{'='*72}")
    for name, score, pp in scored:
        s      = f"{score:.4f}" if score else "  N/A"
        marker = " ★" if score and score == best_score else ""
        print(f"  {name:<42}  {s:>11}  {pp}{marker}")
    print(f"{'='*72}")
    print(f"\nBaseline (label-pseudo, LB=0.849+PP):        holdout=0.9453")
    print(f"Prev best ensemble (label×2):                 holdout=0.9595")
    print(f"Competitor SED (best_fold0.pt, LB=0.862):     holdout=0.9883")
    print(f"\n{len(y_ref)} holdout files  {len(swp_ref)}/234 species with positives")

    # ── Save ─────────────────────────────────────────────────────────────────
    log_path = "outputs/ensemble_v3_holdout_eval.log"
    with open(log_path, "w") as f:
        f.write(f"{'='*72}\n")
        f.write(f"  {'Model':<42}  {'Holdout AUC':>11}  PP\n")
        f.write(f"{'='*72}\n")
        for name, score, pp in scored:
            s      = f"{score:.4f}" if score else "  N/A"
            marker = " ★" if score and score == best_score else ""
            f.write(f"  {name:<42}  {s:>11}  {pp}{marker}\n")
        f.write(f"{'='*72}\n")
        f.write(f"\n{len(y_ref)} holdout files  {len(swp_ref)}/234 species\n")
    print(f"\nResults saved → {log_path}")


if __name__ == "__main__":
    main()
