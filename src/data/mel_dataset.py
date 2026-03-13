"""Mel-spectrogram dataset classes for SED training (BirdClef 2026).

Provides mel-spectrogram versions of ClipDataset and SoundscapeDataset
for use with the PyTorch-based SEDModel pipeline.

Audio → Mel spectrogram (n_mels × T) → normalized → (1, n_mels, T) tensor
"""

import glob
import os

import librosa
import numpy as np
import pandas as pd
from typing import Dict, Generator, List, Optional, Tuple

from src.utils.audio import load_audio, random_crop, center_crop, parse_time_str
from src.data.augment import apply_augmentations
from src.data.dataset import _parse_secondary_labels


# ── Mel spectrogram helpers ───────────────────────────────────────────────────

def compute_mel(
    audio: np.ndarray,
    sample_rate: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float = 16000.0,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Compute log-mel spectrogram from a waveform.

    Returns float32 array of shape (n_mels, T) where
    T = ceil(len(audio) / hop_length).

    Args:
        audio      : Float32 mono waveform.
        sample_rate: Audio sample rate (Hz).
        n_fft      : FFT window size.
        hop_length : Hop size between frames. 320 @ 32 kHz → 100 frames/sec.
        n_mels     : Number of mel filter banks.
        fmin / fmax: Mel filterbank frequency range.
        top_db     : Dynamic range clamp in dB (80 dB is standard).

    Returns:
        Log-mel spectrogram in dB, shape (n_mels, T), dtype float32.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
    return mel_db.astype(np.float32)


def normalize_mel(mel: np.ndarray) -> np.ndarray:
    """Instance normalization: zero mean, unit std per spectrogram."""
    mean = mel.mean()
    std = mel.std() + 1e-6
    return (mel - mean) / std


# ── MelClipDataset ────────────────────────────────────────────────────────────

class MelClipDataset:
    """
    Dataset built from individual recordings in train_audio/.
    Yields (mel_spectrogram, label) pairs where mel has shape (1, n_mels, T).

    Mirrors ClipDataset but outputs mel spectrograms for the PyTorch SED pipeline.
    Supports all BirdCLEF 2025 improvements:
      - Sqrt inverse-frequency class weighting (2nd place)
      - Time masking augmentation
      - Background noise injection
    """

    def __init__(
        self,
        train_csv: str,
        audio_dir: str,
        species_to_idx: Dict[str, int],
        num_classes: int,
        sample_rate: int = 32000,
        clip_duration: int = 5,
        n_clips_per_file: int = 3,
        is_train: bool = True,
        use_secondary_labels: bool = True,
        min_rating: float = 0.0,
        max_files: Optional[int] = None,
        augment_config: Optional[dict] = None,
        class_weights: Optional[np.ndarray] = None,
        noise_dir: Optional[str] = None,
        # Mel spectrogram parameters
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float = 16000.0,
    ):
        self.audio_dir = audio_dir
        self.species_to_idx = species_to_idx
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_length = clip_duration * sample_rate
        self.n_clips_per_file = n_clips_per_file
        self.is_train = is_train
        self.use_secondary_labels = use_secondary_labels
        self.augment_config = augment_config or {"enabled": False}
        self.class_weights = class_weights
        self.mel_kwargs = dict(
            sample_rate=sample_rate,
            n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax,
        )

        # Background noise files
        self.noise_files: List[str] = []
        if noise_dir and os.path.isdir(noise_dir):
            self.noise_files = (
                glob.glob(os.path.join(noise_dir, "**/*.ogg"), recursive=True)
                + glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
            )
            print(f"[MelClipDataset] {len(self.noise_files)} background noise files")

        df = pd.read_csv(train_csv)
        if min_rating > 0:
            df = df[df["rating"] >= min_rating]
        if max_files:
            df = df.head(max_files)
        self.metadata = df.reset_index(drop=True)

        mode_str = "train" if is_train else "val"
        print(
            f"[MelClipDataset] {len(self.metadata)} recordings | "
            f"{mode_str} | {n_clips_per_file} clips/file | "
            f"mel({n_mels}×{clip_duration * sample_rate // hop_length + 1})"
        )

    def _make_label(self, primary_label: str, secondary_labels_str: str) -> np.ndarray:
        label = np.zeros(self.num_classes, dtype=np.float32)
        primary = str(primary_label).strip()
        if primary in self.species_to_idx:
            label[self.species_to_idx[primary]] = 1.0
        if self.use_secondary_labels:
            for sp in _parse_secondary_labels(str(secondary_labels_str)):
                if sp in self.species_to_idx:
                    label[self.species_to_idx[sp]] = 1.0
        return label

    def _sample_weight(self, primary_label: str) -> float:
        if self.class_weights is None:
            return 1.0
        idx = self.species_to_idx.get(str(primary_label).strip())
        return float(self.class_weights[idx]) if idx is not None else 1.0

    def generate_samples(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (mel, label) where mel has shape (1, n_mels, T)."""
        indices = np.arange(len(self.metadata))
        if self.is_train:
            if self.class_weights is not None:
                weights = np.array([
                    self._sample_weight(self.metadata.iloc[i]["primary_label"])
                    for i in range(len(self.metadata))
                ], dtype=np.float32)
                weights /= weights.sum()
                indices = np.random.choice(
                    len(self.metadata), size=len(self.metadata),
                    replace=True, p=weights,
                )
            else:
                np.random.shuffle(indices)

        for idx in indices:
            row = self.metadata.iloc[idx]
            filepath = os.path.join(self.audio_dir, str(row["filename"]))
            audio = load_audio(filepath, self.sample_rate)
            if audio is None:
                continue

            label = self._make_label(
                row["primary_label"], row.get("secondary_labels", "[]")
            )

            n_clips = self.n_clips_per_file if self.is_train else 1
            for _ in range(n_clips):
                clip = (
                    random_crop(audio, self.clip_length) if self.is_train
                    else center_crop(audio, self.clip_length)
                )

                if self.is_train:
                    clip = apply_augmentations(clip, self.augment_config, self.noise_files)

                mel = compute_mel(clip, **self.mel_kwargs)   # (n_mels, T)
                mel = normalize_mel(mel)
                yield mel[np.newaxis], label                 # (1, n_mels, T), (num_classes,)


# ── MelSoundscapeDataset ──────────────────────────────────────────────────────

class MelSoundscapeDataset:
    """
    Dataset for labeled soundscape segments from train_soundscapes/.
    Used as validation data — matches test-time conditions.
    Yields mel spectrograms of shape (1, n_mels, T).
    """

    def __init__(
        self,
        soundscapes_dir: str,
        labels_csv: str,
        species_to_idx: Dict[str, int],
        num_classes: int,
        sample_rate: int = 32000,
        clip_duration: int = 5,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float = 16000.0,
    ):
        self.soundscapes_dir = soundscapes_dir
        self.species_to_idx = species_to_idx
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_length = clip_duration * sample_rate
        self.mel_kwargs = dict(
            sample_rate=sample_rate,
            n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax,
        )

        self.labels_df = pd.read_csv(labels_csv)
        print(f"[MelSoundscapeDataset] {len(self.labels_df)} labeled segments")

    def _make_label(self, labels_str: str) -> np.ndarray:
        label = np.zeros(self.num_classes, dtype=np.float32)
        for sp in str(labels_str).split(";"):
            sp = sp.strip()
            if sp in self.species_to_idx:
                label[self.species_to_idx[sp]] = 1.0
        return label

    def get_all_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all segments into memory. Returns (mels, labels)."""
        mels, labels = [], []

        for filename, group in self.labels_df.groupby("filename"):
            filepath = os.path.join(self.soundscapes_dir, str(filename))
            audio = load_audio(filepath, self.sample_rate)
            if audio is None:
                continue

            for _, row in group.iterrows():
                start_sec = parse_time_str(str(row["start"]))
                start_sample = int(start_sec * self.sample_rate)
                clip = audio[start_sample: start_sample + self.clip_length]
                if len(clip) < self.clip_length:
                    clip = np.pad(clip, (0, self.clip_length - len(clip)))

                mel = compute_mel(clip, **self.mel_kwargs)
                mel = normalize_mel(mel)
                mels.append(mel[np.newaxis])              # (1, n_mels, T)
                labels.append(self._make_label(str(row["primary_label"])))

        if not mels:
            raise RuntimeError("MelSoundscapeDataset: no valid samples found.")

        return np.stack(mels), np.stack(labels)
