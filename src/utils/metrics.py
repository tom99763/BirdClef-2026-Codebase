"""Competition metrics for BirdClef 2026."""

import numpy as np
from sklearn.metrics import average_precision_score
from typing import List, Optional


def padded_cmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    padding_factor: float = 0.5,
) -> float:
    """
    Padded class-wise Mean Average Precision (padded cMAP) — the primary
    BirdClef competition metric.

    For each species:
      - If it has at least one positive label in y_true → compute AP.
      - If it has NO positive labels → assign padding_factor as the AP.
    Final score = mean over all species.

    Args:
        y_true: Binary ground truth, shape (n_samples, n_classes).
        y_pred: Prediction scores (higher = more confident), same shape.
        padding_factor: Default AP for classes with no positive examples.

    Returns:
        Scalar padded cMAP score in [0, 1].
    """
    n_classes = y_true.shape[1]
    ap_scores = []

    for cls_idx in range(n_classes):
        true_cls = y_true[:, cls_idx]
        pred_cls = y_pred[:, cls_idx]

        if true_cls.sum() == 0:
            ap_scores.append(padding_factor)
        else:
            ap_scores.append(float(average_precision_score(true_cls, pred_cls)))

    return float(np.mean(ap_scores))


def per_class_ap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Compute per-class Average Precision.

    Returns:
        Dict mapping class name (or str index) → AP score (None if no positives).
    """
    n_classes = y_true.shape[1]
    results: dict = {}

    for cls_idx in range(n_classes):
        true_cls = y_true[:, cls_idx]
        pred_cls = y_pred[:, cls_idx]
        name = class_names[cls_idx] if class_names else str(cls_idx)

        if true_cls.sum() == 0:
            results[name] = None
        else:
            results[name] = float(average_precision_score(true_cls, pred_cls))

    return results
