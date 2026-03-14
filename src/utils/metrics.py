"""Competition metrics for BirdClef 2026."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import List, Optional


def competition_roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Macro-averaged ROC-AUC — the official BirdCLEF 2026 competition metric.

    Only species that have at least one positive label in y_true are scored.
    This matches the official Kaggle scoring script exactly.

    Args:
        y_true: Binary ground truth, shape (n_samples, n_classes).
        y_pred: Prediction scores (sigmoid outputs), same shape.

    Returns:
        Scalar macro ROC-AUC in [0, 1].
    """
    scored_mask = y_true.sum(axis=0) > 0
    if scored_mask.sum() == 0:
        return 0.0
    return float(roc_auc_score(
        y_true[:, scored_mask],
        y_pred[:, scored_mask],
        average="macro",
    ))


def padded_cmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    padding_factor: float = 0.5,
) -> float:
    """Padded class-wise Mean Average Precision.

    Note: NOT the official BirdCLEF 2026 metric (use competition_roc_auc).
    Kept for reference / cross-checking.
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
    """Per-class Average Precision. Returns None for classes with no positives."""
    n_classes = y_true.shape[1]
    results: dict = {}
    for cls_idx in range(n_classes):
        true_cls = y_true[:, cls_idx]
        pred_cls = y_pred[:, cls_idx]
        name = class_names[cls_idx] if class_names else str(cls_idx)
        results[name] = (
            None if true_cls.sum() == 0
            else float(average_precision_score(true_cls, pred_cls))
        )
    return results
