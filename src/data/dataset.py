"""Dataset classes for BirdClef 2026.

Two data sources:
  - ClipDataset       : individual recordings from train_audio/
  - SoundscapeDataset : labeled 5-second segments from train_soundscapes/

Both yield (audio_clip, multi_hot_label) pairs.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Generator, List, Optional, Tuple

from src.utils.audio import load_audio, random_crop, center_crop, parse_time_str
from src.data.augment import apply_augmentations


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_species_mapping(sample_submission_csv: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Read target species list from sample_submission.csv.

    Returns:
        target_species: Ordered list of 234 species labels.
        species_to_idx: Mapping from label string → index in [0, 233].
    """
    df = pd.read_csv(sample_submission_csv)
    target_species = list(df.columns[1:])          # skip 'row_id'
    species_to_idx = {sp: i for i, sp in enumerate(target_species)}
    return target_species, species_to_idx


def _parse_secondary_labels(raw: str) -> List[str]:
    """Parse secondary_labels field which looks like \"['sp1', 'sp2']\" or empty."""
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    if raw in ("[]", "", "nan"):
        return []
    if raw.startswith("["):
        items = raw.strip("[]").split(",")
        return [s.strip().strip("'\"") for s in items if s.strip().strip("'\"")]
    return [s.strip() for s in raw.split(";") if s.strip()]


# ── ClipDataset ──────────────────────────────────────────────────────────────

class ClipDataset:
    """
    Dataset built from individual recordings in train_audio/.

    Each file contributes n_clips_per_file random (train) or center (val) clips.
    Labels come from train.csv primary_label + optional secondary_labels.
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

        df = pd.read_csv(train_csv)
        if min_rating > 0:
            df = df[df["rating"] >= min_rating]
        if max_files:
            df = df.head(max_files)
        self.metadata = df.reset_index(drop=True)

        mode_str = "train" if is_train else "val"
        print(
            f"[ClipDataset] {len(self.metadata)} recordings | "
            f"{mode_str} | {n_clips_per_file} clips/file"
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

    def generate_samples(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (clip, multi_hot_label) pairs; shuffles file order each epoch."""
        indices = np.arange(len(self.metadata))
        if self.is_train:
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
                clip = random_crop(audio, self.clip_length) if self.is_train \
                    else center_crop(audio, self.clip_length)

                if self.is_train:
                    clip = apply_augmentations(clip, self.augment_config)

                yield clip, label


# ── SoundscapeDataset ────────────────────────────────────────────────────────

class SoundscapeDataset:
    """
    Dataset for labeled soundscape segments from train_soundscapes/.

    Each row in train_soundscapes_labels.csv describes a 5-second segment;
    primary_label is a semicolon-separated list of species present.

    Used primarily as validation data to match test-time conditions.
    """

    def __init__(
        self,
        soundscapes_dir: str,
        labels_csv: str,
        species_to_idx: Dict[str, int],
        num_classes: int,
        sample_rate: int = 32000,
        clip_duration: int = 5,
    ):
        self.soundscapes_dir = soundscapes_dir
        self.species_to_idx = species_to_idx
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_length = clip_duration * sample_rate

        self.labels_df = pd.read_csv(labels_csv)
        print(f"[SoundscapeDataset] {len(self.labels_df)} labeled segments")

    def _make_label(self, labels_str: str) -> np.ndarray:
        label = np.zeros(self.num_classes, dtype=np.float32)
        for sp in str(labels_str).split(";"):
            sp = sp.strip()
            if sp in self.species_to_idx:
                label[self.species_to_idx[sp]] = 1.0
        return label

    def generate_samples(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (clip, multi_hot_label) pairs grouped by soundscape file."""
        for filename, group in self.labels_df.groupby("filename"):
            filepath = os.path.join(self.soundscapes_dir, str(filename))
            audio = load_audio(filepath, self.sample_rate)
            if audio is None:
                continue

            for _, row in group.iterrows():
                start_sec = parse_time_str(str(row["start"]))
                start_sample = int(start_sec * self.sample_rate)

                clip = audio[start_sample : start_sample + self.clip_length]
                if len(clip) < self.clip_length:
                    clip = np.pad(clip, (0, self.clip_length - len(clip)))

                yield clip, self._make_label(str(row["primary_label"]))

    def get_all_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all segments into memory and return (clips, labels) arrays."""
        clips, labels = [], []
        for clip, label in self.generate_samples():
            clips.append(clip)
            labels.append(label)

        if not clips:
            raise RuntimeError("SoundscapeDataset: no valid samples found.")

        return np.stack(clips), np.stack(labels)
