"""Audio augmentation functions applied during training.

Techniques included:
  - Gaussian noise          : original baseline
  - Random gain             : original baseline
  - Time masking            : SpecAugment-style waveform masking (BirdCLEF 2025 top solutions)
  - Background noise inject : mix in ambient noise files (BirdCLEF 2025 multiple teams)
"""

import os
import glob
import numpy as np
from typing import List, Optional, Tuple


# ── Per-sample augmentations ────────────────────────────────────────────────

def add_gaussian_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    noise = np.random.randn(*audio.shape).astype(np.float32)
    return audio + noise_level * noise


def random_gain(audio: np.ndarray, gain_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
    gain = np.random.uniform(*gain_range)
    return audio * gain


def time_masking(
    audio: np.ndarray,
    max_mask_ratio: float = 0.1,
    n_masks: int = 1,
) -> np.ndarray:
    """
    SpecAugment-style time masking applied to the raw waveform.

    Zeroes out up to `n_masks` contiguous time windows, each at most
    `max_mask_ratio * len(audio)` samples long. Adapted from BirdCLEF 2025
    top solutions that applied time/frequency masking universally.

    Args:
        audio         : Float32 waveform of shape (n_samples,).
        max_mask_ratio: Max fraction of the clip that a single mask covers.
        n_masks       : Number of independent masks to apply.

    Returns:
        Masked audio (same shape).
    """
    audio = audio.copy()
    n = len(audio)
    max_width = max(1, int(n * max_mask_ratio))
    for _ in range(n_masks):
        width = np.random.randint(1, max_width + 1)
        start = np.random.randint(0, max(1, n - width))
        audio[start : start + width] = 0.0
    return audio


def add_background_noise(
    audio: np.ndarray,
    noise_files: List[str],
    snr_db_range: Tuple[float, float] = (5.0, 30.0),
) -> np.ndarray:
    """
    Mix a random background noise file into the audio at a random SNR.

    Inspired by multiple BirdCLEF 2025 teams that injected ambient / background
    noise to improve model robustness in soundscape conditions.

    Args:
        audio         : Target waveform (n_samples,) float32.
        noise_files   : List of pre-loaded noise waveform paths.
        snr_db_range  : (min_snr_db, max_snr_db) — higher = cleaner signal.

    Returns:
        Mixed audio (same shape, clipped to [-1, 1]).
    """
    if not noise_files:
        return audio

    # Load a random noise file lazily (keep simple; caching done at dataset level)
    import soundfile as sf
    noise_path = noise_files[np.random.randint(len(noise_files))]
    try:
        noise, _ = sf.read(noise_path, dtype="float32", always_2d=False)
        # Convert stereo → mono
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
    except Exception:
        return audio

    # Tile or crop noise to match audio length
    n = len(audio)
    if len(noise) < n:
        repeats = int(np.ceil(n / len(noise)))
        noise = np.tile(noise, repeats)
    start = np.random.randint(0, max(1, len(noise) - n))
    noise = noise[start : start + n].astype(np.float32)

    # Scale noise to achieve desired SNR
    signal_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-8
    snr_db = np.random.uniform(*snr_db_range)
    desired_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    noise = noise * (desired_noise_rms / noise_rms)

    return np.clip(audio + noise, -1.0, 1.0)


def apply_augmentations(
    audio: np.ndarray,
    aug_config: dict,
    noise_files: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Apply configured augmentations to a single clip.

    Args:
        audio      : Float32 array of shape (n_samples,).
        aug_config : Dict with augmentation settings (see configs/default.yaml).
        noise_files: Optional list of background noise file paths.
    """
    if not aug_config.get("enabled", False):
        return audio

    audio = add_gaussian_noise(audio, noise_level=aug_config.get("noise_level", 0.005))
    audio = random_gain(audio, gain_range=aug_config.get("gain_range", [0.7, 1.3]))

    if aug_config.get("time_masking", False):
        audio = time_masking(
            audio,
            max_mask_ratio=aug_config.get("time_mask_ratio", 0.1),
            n_masks=aug_config.get("time_mask_n", 2),
        )

    if aug_config.get("background_noise", False) and noise_files:
        audio = add_background_noise(
            audio,
            noise_files=noise_files,
            snr_db_range=aug_config.get("snr_db_range", [5.0, 30.0]),
        )

    return audio


# ── Batch augmentations ──────────────────────────────────────────────────────

def apply_mixup_batch(
    audios: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixup augmentation on a batch.

    Blends each sample with a randomly permuted sample using a Beta(alpha, alpha)
    mixing coefficient. Operates in-place equivalent (returns new arrays).

    Args:
        audios: Shape (batch, n_samples).
        labels: Shape (batch, n_classes).
        alpha: Beta distribution parameter; higher → more aggressive mixing.

    Returns:
        Tuple of (mixed_audios, mixed_labels) with the same shapes.
    """
    batch_size = len(audios)
    if alpha <= 0 or batch_size < 2:
        return audios, labels

    lam = np.random.beta(alpha, alpha, size=(batch_size, 1)).astype(np.float32)
    perm = np.random.permutation(batch_size)

    mixed_audios = lam * audios + (1.0 - lam) * audios[perm]
    mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
    return mixed_audios, mixed_labels
