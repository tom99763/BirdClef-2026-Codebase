"""Audio augmentation functions applied during training."""

import numpy as np
from typing import Tuple


# ── Per-sample augmentations ────────────────────────────────────────────────

def add_gaussian_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    noise = np.random.randn(*audio.shape).astype(np.float32)
    return audio + noise_level * noise


def random_gain(audio: np.ndarray, gain_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
    gain = np.random.uniform(*gain_range)
    return audio * gain


def apply_augmentations(audio: np.ndarray, aug_config: dict) -> np.ndarray:
    """
    Apply configured augmentations to a single clip.

    Args:
        audio: Float32 array of shape (n_samples,).
        aug_config: Dict with keys: enabled, noise_level, gain_range.
    """
    if not aug_config.get("enabled", False):
        return audio

    audio = add_gaussian_noise(audio, noise_level=aug_config.get("noise_level", 0.005))
    audio = random_gain(audio, gain_range=aug_config.get("gain_range", [0.7, 1.3]))
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
