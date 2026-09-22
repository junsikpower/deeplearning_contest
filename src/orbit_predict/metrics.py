"""Competition metrics used by both model paths."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_float_arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {actual.shape} and {predicted.shape}"
        )
    if actual.ndim != 1:
        actual = actual.reshape(-1)
        predicted = predicted.reshape(-1)
    if actual.size == 0:
        raise ValueError("metrics require at least one row")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("metrics require finite y_true and y_pred values")
    return actual, predicted


def rmse(y_true: object, y_pred: object) -> float:
    """Return root mean squared error."""

    actual, predicted = _as_float_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def competition_huber_score(y_true: object, y_pred: object, delta: float = 5.0) -> float:
    """Return the competition's scaled Huber score.

    The competition applies 0.5 * error^2 up to ``delta`` and
    ``delta * (abs(error) - delta / 2)`` above it, then averages and scales
    the value by 1,000.
    """

    if delta <= 0:
        raise ValueError("delta must be positive")
    actual, predicted = _as_float_arrays(y_true, y_pred)
    absolute_error = np.abs(actual - predicted)
    loss = np.where(
        absolute_error <= delta,
        0.5 * np.square(actual - predicted),
        delta * (absolute_error - delta / 2.0),
    )
    return float(np.mean(loss) * 1_000.0)


def metric_dict(y_true: object, y_pred: object) -> dict[str, float]:
    """Compute every metric reported for a prediction stage."""

    actual, predicted = _as_float_arrays(y_true, y_pred)
    return {
        "rmse": rmse(actual, predicted),
        "huber_score": competition_huber_score(actual, predicted),
        "max_absolute_error": float(np.max(np.abs(actual - predicted))),
    }


def merge_metric_dicts(*metrics: Mapping[str, float]) -> dict[str, float]:
    """Merge metric dictionaries while preserving a stable key order."""

    merged: dict[str, float] = {}
    for metric in metrics:
        merged.update({str(key): float(value) for key, value in metric.items()})
    return merged

