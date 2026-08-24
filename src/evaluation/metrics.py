"""Regression metrics for Remaining Useful Life prediction.

MAE and RMSE are the headline numbers, but they treat a 10-cycle optimistic
error the same as a 10-cycle pessimistic one. In maintenance that is wrong:
predicting more life than an engine has left means it fails in service. The
asymmetric NASA/PHM08 score in :func:`nasa_score` captures that cost, so it is
reported alongside the symmetric metrics.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    "evaluate_model",
    "mae",
    "mape",
    "nasa_score",
    "r2",
    "regression_metrics",
    "rmse",
    "within_tolerance",
]


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to matching 1-D float arrays."""
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    return true, pred


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in cycles."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(mean_absolute_error(true, pred))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in cycles."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(true, pred)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(r2_score(true, pred))


def mape(y_true, y_pred, epsilon: float = 1.0) -> float:
    """Mean absolute percentage error, in percent.

    RUL legitimately reaches zero, so the denominator is floored at ``epsilon``
    cycles instead of dividing by zero.
    """
    true, pred = _as_arrays(y_true, y_pred)
    denominator = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs((true - pred) / denominator)) * 100.0)


def nasa_score(y_true, y_pred, late_penalty: float = 10.0, early_penalty: float = 13.0) -> float:
    """Asymmetric PHM08 prognostics score - lower is better.

    For each error ``d = predicted - actual``:

    * ``d > 0`` (late, over-estimated life) costs ``exp(d / late_penalty) - 1``
    * ``d <= 0`` (early, conservative) costs ``exp(-d / early_penalty) - 1``

    The steeper late branch reflects that unplanned failures cost more than
    early maintenance.
    """
    true, pred = _as_arrays(y_true, y_pred)
    d = pred - true
    penalties = np.where(
        d > 0,
        np.expm1(np.clip(d, None, 700) / late_penalty),
        np.expm1(-np.clip(d, -700, None) / early_penalty),
    )
    return float(penalties.sum())


def within_tolerance(y_true, y_pred, tolerance: float = 10.0) -> float:
    """Share of predictions (0-1) whose absolute error is within ``tolerance``."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred) <= tolerance))


def regression_metrics(
    y_true, y_pred, tolerance: float = 10.0, include_score: bool = True
) -> dict[str, float]:
    """Compute the full metric set as a flat dictionary.

    Keys are capitalised (``MAE``, ``RMSE``, ...) so the result drops straight
    into a leaderboard DataFrame.
    """
    true, pred = _as_arrays(y_true, y_pred)
    errors = pred - true
    metrics: dict[str, float] = {
        "MAE": mae(true, pred),
        "RMSE": rmse(true, pred),
        "R2": r2(true, pred),
        "MAPE": mape(true, pred),
        f"Within{int(tolerance)}": within_tolerance(true, pred, tolerance),
        "Bias": float(errors.mean()),
        "MaxError": float(np.abs(errors).max()),
        "N": int(true.size),
    }
    if include_score:
        metrics["NASAScore"] = nasa_score(true, pred)
    return metrics


def evaluate_model(
    model_name: str, y_true, y_pred, tolerance: float = 10.0
) -> dict[str, object]:
    """Metrics for one model as a leaderboard row, labelled by ``Model``."""
    return {"Model": model_name, **regression_metrics(y_true, y_pred, tolerance)}
