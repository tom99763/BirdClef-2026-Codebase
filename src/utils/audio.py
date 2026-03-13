"""Audio loading and processing utilities."""

import numpy as np
import librosa
from typing import Optional


def load_audio(
    filepath: str,
    sample_rate: int = 32000,
    mono: bool = True,
) -> Optional[np.ndarray]:
    """
    Load an audio file and resample to target sample rate.

    Returns:
        Float32 numpy array of shape (n_samples,), or None on error.
    """
    try:
        audio, _ = librosa.load(filepath, sr=sample_rate, mono=mono)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Warning: failed to load {filepath}: {e}")
        return None


def random_crop(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Extract a uniformly random crop of exactly target_length samples."""
    if len(audio) <= target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    start = np.random.randint(0, len(audio) - target_length)
    return audio[start : start + target_length]


def center_crop(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Extract a center crop of exactly target_length samples."""
    if len(audio) <= target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    start = (len(audio) - target_length) // 2
    return audio[start : start + target_length]


def parse_time_str(time_str: str) -> float:
    """Convert 'HH:MM:SS' or 'MM:SS' string to seconds (float)."""
    parts = str(time_str).split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])
