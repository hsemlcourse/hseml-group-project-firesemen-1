"""Regression metrics used across experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute project metrics for next-minute close prediction."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100)
    true_direction = np.sign(np.diff(y_true, prepend=y_true[0]))
    pred_direction = np.sign(np.diff(y_pred, prepend=y_pred[0]))
    directional_accuracy = float(np.mean(true_direction == pred_direction))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_percent": mape,
        "directional_accuracy": directional_accuracy,
    }
