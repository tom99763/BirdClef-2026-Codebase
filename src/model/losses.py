"""Loss functions from BirdCLEF 2025 top solutions.

Implementations:
  - FocalBCELoss     : Focal loss for class imbalance (2nd, 5th place)
  - power_transform  : Sharpen pseudo-label probabilities (1st place)
"""

import tensorflow as tf
import numpy as np


class FocalBCELoss:
    """
    Focal Binary Cross-Entropy loss.

    Reduces loss weight for easy (high-confidence) samples so the model
    focuses on hard / rare-class examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma          : Focusing parameter. 0 = standard BCE. Typical: 2.0.
        alpha          : Class balance weight. 0.25 is common (down-weights negatives).
                         Set to -1 to disable (no alpha weighting).
        from_logits    : Whether inputs are raw logits (recommended).
        label_smoothing: Smoothing applied before focal weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        from_logits: bool = True,
        label_smoothing: float = 0.0,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: Float tensor of shape (batch, num_classes) in [0, 1].
            y_pred: Float tensor same shape. Raw logits if from_logits=True.

        Returns:
            Scalar mean loss.
        """
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.from_logits:
            # Numerically stable sigmoid + log-sigmoid
            p = tf.sigmoid(y_pred)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        else:
            p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            bce = -(y_true * tf.math.log(p) + (1 - y_true) * tf.math.log(1 - p))

        # p_t: probability of the true class
        p_t = p * y_true + (1.0 - p) * (1.0 - y_true)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * y_true + (1.0 - self.alpha) * (1.0 - y_true)
            focal_bce = alpha_t * focal_weight * bce
        else:
            focal_bce = focal_weight * bce

        return tf.reduce_mean(focal_bce)


def power_transform(probs: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    PowerTransform for pseudo-label sharpening (1st place BirdCLEF 2025).

    Amplifies confident predictions and suppresses uncertain ones, enabling
    multiple rounds of stable pseudo-label refinement without collapse.

    y_sharp = y^power / sum(y^power)  [per-sample normalisation]

    For multi-label: apply element-wise without normalisation:
        y_sharp = y^power

    Args:
        probs : Float array of shape (n_samples, n_classes) in [0, 1].
        power : Exponent. >1 sharpens, <1 softens. Typical: 1.5–3.0.

    Returns:
        Sharpened probabilities, same shape.
    """
    return np.power(np.clip(probs, 0.0, 1.0), power).astype(np.float32)
