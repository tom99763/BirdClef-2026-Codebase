"""BirdClef 2026 — SED Model Inference / Submission Generation

Processes soundscape files using the PyTorch SED model and writes submission.csv.
Supports TTA with 2.5-second temporal shifts (BirdCLEF 2025 2nd place).

Usage:
    # Basic inference
    python inference_sed.py \\
        --config configs/sed_default.yaml \\
        --checkpoint checkpoints/sed-v1/best_sed

    # With TTA
    python inference_sed.py \\
        --config configs/sed_default.yaml \\
        --checkpoint checkpoints/sed-v1/best_sed \\
        --tta

    # Ensemble with Perch submission (average predictions)
    python inference_sed.py \\
        --config configs/sed_default.yaml \\
        --checkpoint checkpoints/sed-v1/best_sed \\
        --tta \\
        --ensemble_with submission_perch.csv \\
        --output submission_ensemble.csv
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
from src.utils.audio import load_audio
from src.data.dataset import build_species_mapping
from src.data.mel_dataset import compute_mel, normalize_mel
from src.model.sed_model import SEDModel


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="BirdClef 2026 SED Inference")
    parser.add_argument("--config", default="configs/sed_default.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to SED checkpoint (without .pt extension)")
    parser.add_argument("--soundscapes_dir", default=None)
    parser.add_argument("--output", default="submission_sed.csv")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--tta", action="store_true",
                        help="TTA with 2.5-second temporal shifts (BirdCLEF25 2nd place)")
    parser.add_argument("--ensemble_with", default=None,
                        help="Path to another submission CSV to average with (e.g. Perch)")
    return parser.parse_args()


# ── Per-soundscape inference ─────────────────────────────────────────────────

def _mel_clips(audio: np.ndarray, clip_length: int, mel_kwargs: dict,
               shift: int = 0) -> np.ndarray:
    """Extract evenly-spaced clips starting at `shift` samples, compute mel."""
    a = audio[shift:]
    n = len(a) // clip_length
    if n == 0:
        return np.empty((0,))
    mels = []
    for i in range(n):
        clip = a[i * clip_length: (i + 1) * clip_length]
        mel = compute_mel(clip, **mel_kwargs)
        mel = normalize_mel(mel)
        mels.append(mel[np.newaxis])            # (1, n_mels, T)
    return np.stack(mels)                       # (n, 1, n_mels, T)


def _batched_predict(model: SEDModel, mels: np.ndarray,
                     batch_size: int, device: torch.device) -> np.ndarray:
    """Run batched inference and return clip probabilities (n, num_classes)."""
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(mels), batch_size):
            batch = torch.tensor(
                mels[start: start + batch_size], dtype=torch.float32, device=device
            )
            clip_pred, _ = model(batch)
            preds.append(clip_pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def process_soundscape(
    filepath: str,
    model: SEDModel,
    sample_rate: int,
    clip_duration: int,
    batch_size: int,
    device: torch.device,
    mel_kwargs: dict,
    tta: bool = False,
) -> tuple:
    """
    Split soundscape into 5-second clips, compute mel spectrograms, run inference.

    Args:
        tta: Average predictions from 0-sec and 2.5-sec shifted clips.

    Returns:
        row_ids    : List[str]
        predictions: np.ndarray (n_segments, num_classes)
    """
    ss_id = re.sub(r"\.ogg$", "", os.path.basename(filepath), flags=re.IGNORECASE)
    audio = load_audio(filepath, sample_rate)
    if audio is None:
        return [], np.empty((0,))

    clip_length = clip_duration * sample_rate
    n_segments = len(audio) // clip_length
    if n_segments == 0:
        return [], np.empty((0,))

    row_ids = [f"{ss_id}_{(i + 1) * clip_duration}" for i in range(n_segments)]

    mels_normal = _mel_clips(audio, clip_length, mel_kwargs, shift=0)
    preds = _batched_predict(model, mels_normal, batch_size, device)

    if tta:
        half = clip_length // 2
        mels_shifted = _mel_clips(audio, clip_length, mel_kwargs, shift=half)
        if len(mels_shifted) > 0:
            preds_shifted = _batched_predict(model, mels_shifted, batch_size, device)
            n_use = min(len(preds), len(preds_shifted))
            preds[:n_use] = 0.5 * preds[:n_use] + 0.5 * preds_shifted[:n_use]

    return row_ids, preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Soundscapes directory
    if args.soundscapes_dir:
        soundscapes_dir = args.soundscapes_dir
    else:
        data_root = os.path.dirname(config.data.train_soundscapes_dir)
        test_dir = os.path.join(data_root, "test_soundscapes")
        if os.path.isdir(test_dir) and glob.glob(os.path.join(test_dir, "*.ogg")):
            soundscapes_dir = test_dir
        else:
            soundscapes_dir = config.data.train_soundscapes_dir
    print(f"Soundscapes: {soundscapes_dir}")

    ogg_files = sorted(glob.glob(os.path.join(soundscapes_dir, "*.ogg")))
    if args.max_files:
        ogg_files = ogg_files[:args.max_files]
    print(f"Files to process: {len(ogg_files)}")

    # Species mapping
    target_species, species_to_idx = build_species_mapping(
        config.data.sample_submission_csv
    )
    num_classes = len(target_species)

    # Model
    print("\nLoading SED model …")
    model = SEDModel(
        backbone=config.model.backbone,
        num_classes=num_classes,
        in_chans=1,
        pretrained=False,          # weights loaded from checkpoint
        drop_rate=config.model.dropout,
    ).to(device)
    model.load(args.checkpoint)

    batch_size = args.batch_size or config.training.batch_size * 2
    mel_kwargs = dict(
        sample_rate=config.audio.sample_rate,
        n_fft=config.mel.n_fft,
        hop_length=config.mel.hop_length,
        n_mels=config.mel.n_mels,
        fmin=config.mel.fmin,
        fmax=config.mel.fmax,
    )

    # Inference
    all_row_ids, all_preds = [], []
    for filepath in ogg_files:
        print(f"  {os.path.basename(filepath)}")
        row_ids, preds = process_soundscape(
            filepath=filepath,
            model=model,
            sample_rate=config.audio.sample_rate,
            clip_duration=config.audio.clip_duration,
            batch_size=batch_size,
            device=device,
            mel_kwargs=mel_kwargs,
            tta=args.tta,
        )
        if len(row_ids) > 0:
            all_row_ids.extend(row_ids)
            all_preds.append(preds)

    if not all_preds:
        print("ERROR: No predictions generated.")
        return

    predictions = np.concatenate(all_preds, axis=0)

    # Optional: ensemble with another submission (e.g., Perch predictions)
    if args.ensemble_with and os.path.isfile(args.ensemble_with):
        print(f"\nEnsembling with: {args.ensemble_with}")
        other = pd.read_csv(args.ensemble_with)
        other_preds = other[target_species].values.astype(np.float32)
        n_use = min(len(predictions), len(other_preds))
        predictions[:n_use] = 0.5 * predictions[:n_use] + 0.5 * other_preds[:n_use]
        print(f"  Averaged {n_use} rows.")

    # Write submission
    submission = pd.DataFrame(predictions, columns=target_species)
    submission.insert(0, "row_id", all_row_ids)
    submission.to_csv(args.output, index=False)

    print(f"\nSED submission saved → {args.output}  "
          f"({submission.shape[0]} rows × {num_classes} species)")
    print(submission.head())


if __name__ == "__main__":
    main()
