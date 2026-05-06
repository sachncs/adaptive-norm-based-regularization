"""Evaluation metrics for regression and classification."""

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy for multi-class classification.

    Args:
        y_true: True integer labels of shape (n_samples,).
        y_pred: Predicted integer labels of shape (n_samples,).

    Returns:
        Balanced accuracy.
    """
    classes = np.unique(y_true)
    per_class_acc = []
    for c in classes:
        mask = y_true == c
        if np.any(mask):
            acc = np.mean(y_pred[mask] == c)
            per_class_acc.append(acc)
    return float(np.mean(per_class_acc)) if per_class_acc else 0.0
