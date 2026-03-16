"""Dataset classes for BirdClef 2026.

Two data sources:
  - ClipDataset       : individual recordings from train_audio/
  - SoundscapeDataset : labeled 5-second segments from train_soundscapes/

Both yield (audio_clip, multi_hot_label) pairs.

BirdCLEF 2025 improvement: sqrt inverse-frequency sample weighting (2nd place).
"""

import glob
import os
import numpy as np
import pandas as pd
from typing import Callable, Dict, Generator, List, Optional, Tuple

from src.utils.audio import load_audio, random_crop, center_crop, parse_time_str
from src.data.augment import apply_augmentations


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_class_weights(
    train_csv: str,
    species_to_idx: Dict[str, int],
    num_classes: int,
    mode: str = "sqrt",
) -> np.ndarray:
    """
    Compute per-class sample weights from training metadata.

    BirdCLEF 2025 2nd place used sqrt inverse-frequency weighting so that
    rare species recordings are sampled more frequently.

    Args:
        train_csv     : Path to train.csv.
        species_to_idx: Species label → class index mapping.
        num_classes   : Total number of classes.
        mode          : "sqrt"   → weight ∝ 1/sqrt(freq)  [2nd place]
                        "linear" → weight ∝ 1/freq
                        "none"   → uniform weights (all 1.0)

    Returns:
        Float32 array of shape (num_classes,) with per-class weights.
    """
    df = pd.read_csv(train_csv)
    counts = np.ones(num_classes, dtype=np.float32)  # Laplace smoothing
    for label in df["primary_label"].astype(str):
        if label in species_to_idx:
            counts[species_to_idx[label]] += 1.0

    if mode == "sqrt":
        weights = 1.0 / np.sqrt(counts)
    elif mode == "linear":
        weights = 1.0 / counts
    else:
        return np.ones(num_classes, dtype=np.float32)

    # Normalise so mean weight = 1 (preserves effective learning rate scale)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def compute_taxon_weights(
    train_csv: str,
    taxonomy_csv: str,
    species_to_idx: Dict[str, int],
    num_classes: int,
    nonbird_boost: float = 3.0,
) -> np.ndarray:
    """
    Per-class weights that additionally boost non-bird taxa.

    Applies sqrt inverse-frequency first, then multiplies non-Aves species
    by nonbird_boost so that Amphibia/Reptilia/Insecta/Mammalia species
    receive proportionally more gradient updates.

    Args:
        nonbird_boost: Multiplier applied to non-Aves species (default 3.0).
    """
    # Base sqrt weights
    weights = compute_class_weights(train_csv, species_to_idx, num_classes, mode="sqrt")

    # Load taxonomy
    tax_df = pd.read_csv(taxonomy_csv)[["primary_label", "class_name"]]
    tax_df["primary_label"] = tax_df["primary_label"].astype(str)
    taxon_map = dict(zip(tax_df["primary_label"], tax_df["class_name"]))

    # Apply boost to non-bird species
    for sp, idx in species_to_idx.items():
        class_name = taxon_map.get(str(sp), "Aves")
        if class_name != "Aves":
            weights[idx] *= nonbird_boost

    # Re-normalise
    weights = weights / weights.mean()
    return weights.astype(np.float32)


TAXON_CLASSES = ["Aves", "Amphibia", "Reptilia", "Insecta", "Mammalia"]
TAXON_TO_IDX  = {t: i for i, t in enumerate(TAXON_CLASSES)}


def build_taxon_label_fn(
    taxonomy_csv: str,
    species_to_idx: Dict[str, int],
) -> "Callable[[str], int]":
    """
    Returns a function that maps species label → taxon class index.
    Used for the multi-task taxonomy auxiliary loss.
    """
    tax_df = pd.read_csv(taxonomy_csv)[["primary_label", "class_name"]]
    tax_df["primary_label"] = tax_df["primary_label"].astype(str)
    taxon_map = dict(zip(tax_df["primary_label"], tax_df["class_name"]))

    def label_fn(species: str) -> int:
        return TAXON_TO_IDX.get(taxon_map.get(str(species), "Aves"), 0)

    return label_fn


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
        noise_dir: Optional[str] = None,
        class_weights: Optional[np.ndarray] = None,
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
        self.class_weights = class_weights  # (num_classes,) or None
        # Background noise files (BirdCLEF 2025 multi-team technique)
        self.noise_files: List[str] = []
        if noise_dir and os.path.isdir(noise_dir):
            self.noise_files = (
                glob.glob(os.path.join(noise_dir, "**/*.ogg"), recursive=True)
                + glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
                + glob.glob(os.path.join(noise_dir, "**/*.flac"), recursive=True)
            )
            print(f"[ClipDataset] Loaded {len(self.noise_files)} background noise files from {noise_dir}")

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

    def _sample_weight(self, primary_label: str) -> float:
        """Return the per-sample weight based on primary species frequency."""
        if self.class_weights is None:
            return 1.0
        idx = self.species_to_idx.get(str(primary_label).strip())
        if idx is None:
            return 1.0
        return float(self.class_weights[idx])

    def generate_samples(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (clip, multi_hot_label) pairs; shuffles file order each epoch.

        When class_weights are provided the file order is shuffled with
        probability proportional to the primary-species weight (sqrt inverse
        frequency, BirdCLEF 2025 2nd place) so rare species appear more often.
        """
        indices = np.arange(len(self.metadata))
        if self.is_train:
            if self.class_weights is not None:
                # Weighted shuffle: sample indices with replacement proportional
                # to each file's primary-species weight.
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
                clip = random_crop(audio, self.clip_length) if self.is_train \
                    else center_crop(audio, self.clip_length)

                if self.is_train:
                    clip = apply_augmentations(clip, self.augment_config, self.noise_files)

                yield clip, label


# ── SoundscapeDataset ────────────────────────────────────────────────────────

class SoundscapeDataset:
    """
    Dataset for labeled soundscape segments from train_soundscapes/.

    Each row in train_soundscapes_labels.csv describes a 5-second segment;
    primary_label is a semicolon-separated list of species present.

    Pass `split_csv` (birdclef-2026/soundscapes_split.csv) and `split`
    ("train" or "val") to use the held-out validation set correctly.
    Without split_csv all segments are returned (legacy behaviour).
    """

    def __init__(
        self,
        soundscapes_dir: str,
        labels_csv: str,
        species_to_idx: Dict[str, int],
        num_classes: int,
        sample_rate: int = 32000,
        clip_duration: int = 5,
        split_csv: Optional[str] = None,
        split: Optional[str] = None,       # "train" | "val" | None (all)
    ):
        self.soundscapes_dir = soundscapes_dir
        self.species_to_idx = species_to_idx
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_length = clip_duration * sample_rate

        df = pd.read_csv(labels_csv)

        if split_csv and split:
            sc_split = pd.read_csv(split_csv)
            split_files = set(sc_split[sc_split["split"] == split]["filename"].tolist())
            df = df[df["filename"].isin(split_files)].reset_index(drop=True)
            print(f"[SoundscapeDataset] {len(df)} segments (split={split}, {len(split_files)} files)")
        else:
            print(f"[SoundscapeDataset] {len(df)} labeled segments (no split)")

        self.labels_df = df

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


# ── CachedEmbeddingDataset ───────────────────────────────────────────────────

class CachedEmbeddingDataset:
    """
    Load pre-computed Perch embeddings from disk (output of extract_embeddings.py).

    Skips the heavy Perch backbone entirely; only the lightweight MLP head is
    trained, making each epoch orders-of-magnitude faster.
    """

    def __init__(
        self,
        manifest_csv: str,
        species_to_idx: Dict[str, int],
        num_classes: int,
        split: str = "train",
        class_weights: Optional[np.ndarray] = None,
        taxon_label_fn=None,
        soundscape_split_csv: Optional[str] = None,
        soundscape_split: Optional[str] = None,   # "train" | "val" | None (all)
    ):
        df = pd.read_csv(manifest_csv)
        rows = df[df["split"] == split].reset_index(drop=True)

        # For soundscape embeddings, optionally filter to train or val files
        if split == "soundscape" and soundscape_split_csv and soundscape_split:
            sc_split = pd.read_csv(soundscape_split_csv)
            split_files = set(sc_split[sc_split["split"] == soundscape_split]["filename"].tolist())
            # source_file column contains the original ogg filename
            rows = rows[rows["source_file"].isin(split_files)].reset_index(drop=True)
            print(f"[CachedEmbeddingDataset] {len(rows)} embeddings "
                  f"(split=soundscape/{soundscape_split}, {len(split_files)} files)")
        else:
            print(f"[CachedEmbeddingDataset] {len(rows)} embeddings (split={split})")

        self.df = rows
        self.species_to_idx = species_to_idx
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.taxon_label_fn = taxon_label_fn

    @property
    def embedding_dim(self) -> int:
        return int(np.load(self.df.iloc[0]["npy_path"]).shape[-1])

    def _make_label(self, label_str: str) -> np.ndarray:
        label = np.zeros(self.num_classes, dtype=np.float32)
        for sp in str(label_str).split(";"):
            sp = sp.strip()
            if sp in self.species_to_idx:
                label[self.species_to_idx[sp]] = 1.0
        return label

    def generate_samples(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        indices = np.arange(len(self.df))
        if self.class_weights is not None:
            weights = np.array([
                float(self.class_weights[self.species_to_idx[str(self.df.iloc[i]["label"]).strip()]])
                if str(self.df.iloc[i]["label"]).strip() in self.species_to_idx else 1.0
                for i in range(len(self.df))
            ], dtype=np.float32)
            weights /= weights.sum()
            indices = np.random.choice(len(self.df), size=len(self.df), replace=True, p=weights)
        else:
            np.random.shuffle(indices)
        for idx in indices:
            row = self.df.iloc[idx]
            emb = np.load(row["npy_path"]).astype(np.float32)
            if self.taxon_label_fn is not None:
                taxon_idx = np.int32(self.taxon_label_fn(row["label"]))
                yield emb, self._make_label(row["label"]), taxon_idx
            else:
                yield emb, self._make_label(row["label"])

    def get_all_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        embs, labels = [], []
        for emb, label in self.generate_samples():
            embs.append(emb)
            labels.append(label)
        if not embs:
            raise RuntimeError("CachedEmbeddingDataset: no embeddings found.")
        return np.stack(embs), np.stack(labels)


# ── PseudoSoundscapeDataset ───────────────────────────────────────────────────

class PseudoSoundscapeDataset:
    """
    Dataset built from pseudo-labeled soundscape segments.

    Reads pseudo_labels.csv (output of pseudo_label.py generate), parses each
    row_id to recover (filename, start_sec), loads audio, and yields
    (clip, soft_label) pairs.

    row_id format: <soundscape_basename_without_ext>_<end_seconds>
    e.g.  BC2026_Train_0001_5   → BC2026_Train_0001.ogg, start=0s
          BC2026_Train_0001_10  → BC2026_Train_0001.ogg, start=5s
    """

    def __init__(
        self,
        pseudo_csv: str,
        soundscapes_dir: str,
        species_to_idx: Dict[str, int],
        target_species: List[str],
        num_classes: int,
        sample_rate: int = 32000,
        clip_duration: int = 5,
        use_soft_labels: bool = True,
    ):
        self.soundscapes_dir = soundscapes_dir
        self.species_to_idx = species_to_idx
        self.target_species = target_species
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.clip_length = clip_duration * sample_rate
        self.clip_duration = clip_duration
        self.use_soft_labels = use_soft_labels

        df = pd.read_csv(pseudo_csv)
        self.df = df.reset_index(drop=True)

        # Detect soft-label columns (species codes among column names)
        species_set = set(target_species)
        self.soft_cols = [c for c in df.columns if c in species_set]
        self.has_soft = len(self.soft_cols) == num_classes and use_soft_labels

        print(f"[PseudoSoundscapeDataset] {len(self.df)} pseudo-labeled segments "
              f"({'soft' if self.has_soft else 'hard'} labels)")

    def _parse_row_id(self, row_id: str):
        """Return (filename_with_ext, start_sec)."""
        parts = row_id.rsplit("_", 1)
        if len(parts) != 2:
            return None, None
        basename, end_str = parts
        try:
            end_sec = int(end_str)
        except ValueError:
            return None, None
        start_sec = max(0, end_sec - self.clip_duration)
        filename = basename + ".ogg"
        return filename, start_sec

    def _make_hard_label(self, primary_label: str, secondary_str: str = "") -> np.ndarray:
        label = np.zeros(self.num_classes, dtype=np.float32)
        primary = str(primary_label).strip()
        if primary in self.species_to_idx:
            label[self.species_to_idx[primary]] = 1.0
        for sp in _parse_secondary_labels(str(secondary_str)):
            if sp in self.species_to_idx:
                label[self.species_to_idx[sp]] = 1.0
        return label

    def generate_samples(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (clip, label) pairs. Audio is loaded on demand."""
        # Group by filename to avoid re-loading the same ogg file
        current_file, current_audio = None, None

        indices = np.arange(len(self.df))
        np.random.shuffle(indices)

        for idx in indices:
            row = self.df.iloc[idx]
            filename, start_sec = self._parse_row_id(str(row["row_id"]))
            if filename is None:
                continue

            if filename != current_file:
                current_file = filename
                filepath = os.path.join(self.soundscapes_dir, filename)
                current_audio = load_audio(filepath, self.sample_rate)

            if current_audio is None:
                continue

            start_sample = int(start_sec * self.sample_rate)
            clip = current_audio[start_sample: start_sample + self.clip_length]
            if len(clip) < self.clip_length:
                clip = np.pad(clip, (0, self.clip_length - len(clip)))

            if self.has_soft:
                label = np.array(row[self.soft_cols].values, dtype=np.float32)
            else:
                label = self._make_hard_label(
                    row.get("primary_label", ""),
                    row.get("secondary_labels", ""),
                )

            yield clip, label
