"""Evaluation metrics, curve generation, and interactive threshold calculations.

Provides rigorous, leakage-safe evaluation calculations for the HiggsEvent dashboard:
- ROC curve points and area calculation
- Precision-Recall curve points and average precision
- Calibration reliability points and Brier score
- Interactive threshold operating point calculations with robust edge-case handling
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _validate_binary_arrays(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate 1D ground truth and probability score vectors."""
    y = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)

    if y.ndim != 1 or probs.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional arrays")
    if y.shape[0] != probs.shape[0]:
        raise ValueError("y_true and probabilities must have the same length")
    if y.shape[0] == 0:
        raise ValueError("Input arrays must not be empty")

    unique_y = set(np.unique(y))
    if not unique_y.issubset({0, 1}):
        raise ValueError(f"y_true must only contain binary values 0 and 1, got {unique_y}")

    if not np.isfinite(probs).all():
        raise ValueError("probabilities must contain only finite numbers")
    if (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError("probabilities must be bounded within [0.0, 1.0]")

    return y, probs


def calculate_operating_metrics(
    y_true: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
    threshold: float,
) -> dict[str, Any]:
    """Calculate confusion matrix and operating point metrics for a decision threshold.

    Parameters
    ----------
    y_true : 1D array of binary ground-truth labels (0 or 1).
    probabilities : 1D array of predicted class-1 probability scores in [0, 1].
    threshold : Decision cutoff in [0.0, 1.0]. Events with score >= threshold are
        classified as signal (class 1).

    Returns
    -------
    dict[str, Any]
        Dictionary with confusion matrix counts, signal selection metrics,
        rates, and edge-case status flags.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

    y, probs = _validate_binary_arrays(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )

    total_events = int(y.shape[0])
    total_signal = int(np.sum(y == 1))
    total_background = int(np.sum(y == 0))

    # Binary decision at cutoff
    y_pred = (probs >= threshold).astype(np.int64)

    tp = int(np.sum((y_pred == 1) & (y == 1)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    tn = int(np.sum((y_pred == 0) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))

    signal_selections = tp + fp
    selection_rate = signal_selections / total_events if total_events > 0 else 0.0

    # Precision: TP / (TP + FP)
    precision_undefined = signal_selections == 0
    precision = float(tp / signal_selections) if signal_selections > 0 else 0.0

    # Recall (Signal Efficiency / TPR): TP / (TP + FN)
    recall = float(tp / total_signal) if total_signal > 0 else 0.0

    # False Positive Rate (FPR): FP / (FP + TN)
    fpr = float(fp / total_background) if total_background > 0 else 0.0

    # Background Rejection (1 - FPR): TN / (TN + FP)
    background_rejection = float(tn / total_background) if total_background > 0 else 0.0

    # Overall accuracy
    accuracy = float((tp + tn) / total_events) if total_events > 0 else 0.0

    # F1 score: 2 * P * R / (P + R)
    f1_score = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "total_events": total_events,
        "total_signal": total_signal,
        "total_background": total_background,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "signal_selections": signal_selections,
        "selection_rate": float(selection_rate),
        "precision": float(precision),
        "precision_undefined": precision_undefined,
        "recall": float(recall),
        "signal_efficiency": float(recall),
        "false_positive_rate": float(fpr),
        "background_rejection": float(background_rejection),
        "accuracy": float(accuracy),
        "f1_score": float(f1_score),
    }


def compute_roc_points(
    y_true: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute False Positive Rates, True Positive Rates, and ROC-AUC.

    Returns: (fpr, tpr, roc_auc)
    """
    y, probs = _validate_binary_arrays(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("ROC curve requires both binary classes (0 and 1) in the test set")

    fpr, tpr, _ = roc_curve(y, probs)
    auc_val = float(roc_auc_score(y, probs))
    return fpr, tpr, auc_val


def compute_pr_points(
    y_true: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute Precision, Recall, and Average Precision (PR-AUC).

    Returns: (precision, recall, pr_auc)
    """
    y, probs = _validate_binary_arrays(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("PR curve requires both binary classes (0 and 1) in the test set")

    precision, recall, _ = precision_recall_curve(y, probs)
    ap_val = float(average_precision_score(y, probs))
    return precision, recall, ap_val


def compute_calibration_points(
    y_true: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute reliability curve coordinates and Brier score.

    Returns: (fraction_positives, mean_predicted_value, brier_score)
    """
    y, probs = _validate_binary_arrays(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Calibration curve requires both binary classes (0 and 1)")

    fraction_positives, mean_predicted = calibration_curve(
        y, probs, n_bins=n_bins, strategy="uniform"
    )
    brier = float(brier_score_loss(y, probs))
    return fraction_positives, mean_predicted, brier
