"""Turning sequences into flat feature vectors.

Tree and linear models cannot consume a ``(window, n_sensors)`` matrix, so each
window is summarised by per-sensor aggregations. The default set - mean, std,
min, max, last value and trend (last minus first) - is what the committed
Random Forest and Linear Regression baselines were trained on, and the block
order below must not change or those artifacts stop matching their inputs.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.config import Config, get_config
from src.features.lag_features import add_diff_features, add_lag_features
from src.features.rolling_features import add_rolling_features

logger = logging.getLogger(__name__)

__all__ = [
    "STAT_FUNCTIONS",
    "build_tabular_frame",
    "create_statistical_features",
    "statistical_feature_names",
]

#: Aggregations available for sequence flattening, applied over the time axis.
STAT_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": lambda X: X.mean(axis=1),
    "std": lambda X: X.std(axis=1),
    "min": lambda X: X.min(axis=1),
    "max": lambda X: X.max(axis=1),
    "last": lambda X: X[:, -1, :],
    "first": lambda X: X[:, 0, :],
    "trend": lambda X: X[:, -1, :] - X[:, 0, :],
    "range": lambda X: X.max(axis=1) - X.min(axis=1),
    "median": lambda X: np.median(X, axis=1),
}

DEFAULT_STATS: tuple[str, ...] = ("mean", "std", "min", "max", "last", "trend")


def create_statistical_features(
    X: np.ndarray,
    stats: Sequence[str] | None = None,
    config: Config | None = None,
) -> np.ndarray:
    """Flatten ``(n, window, n_sensors)`` sequences into ``(n, n_stats * n_sensors)``.

    Blocks are concatenated in the order given by ``stats``; each block holds one
    value per sensor.
    """
    array = np.asarray(X, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D array (n, window, features), got {array.shape}")
    if array.shape[0] == 0:
        n_stats = len(_resolve_stats(stats, config))
        return np.empty((0, n_stats * array.shape[2]), dtype=float)

    names = _resolve_stats(stats, config)
    blocks = [STAT_FUNCTIONS[name](array) for name in names]
    features = np.concatenate(blocks, axis=1)
    logger.debug("Flattened %s -> %s using %s", array.shape, features.shape, names)
    return features


def statistical_feature_names(
    feature_columns: Sequence[str],
    stats: Sequence[str] | None = None,
    config: Config | None = None,
) -> list[str]:
    """Column names matching :func:`create_statistical_features` output order."""
    names = _resolve_stats(stats, config)
    return [f"{column}_{stat}" for stat in names for column in feature_columns]


def _resolve_stats(
    stats: Sequence[str] | None, config: Config | None = None
) -> list[str]:
    """Validate a stat list, falling back to config then to the default set."""
    if stats is None:
        try:
            configured = (config or get_config()).features.get("statistical")
        except Exception:  # config is optional for pure-array use
            configured = None
        stats = configured or DEFAULT_STATS

    names = list(stats)
    unknown = [name for name in names if name not in STAT_FUNCTIONS]
    if unknown:
        raise ValueError(
            f"Unknown aggregation(s) {unknown}; choose from {sorted(STAT_FUNCTIONS)}"
        )
    if not names:
        raise ValueError("At least one aggregation is required")
    return names


def build_tabular_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    config: Config | None = None,
    windows: Sequence[int] | None = None,
    lags: Sequence[int] | None = None,
    include_diff: bool | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Enrich a cycle-level frame with rolling, lag and difference features.

    This is the row-wise alternative to sequence models: instead of feeding a
    window to an RNN, each cycle carries its own recent history as columns.

    Returns
    -------
    The enriched frame and the list of engineered feature column names.
    """
    config = config or get_config()
    features_config = config.features
    data = config.data

    windows = windows if windows is not None else features_config.get("rolling_windows", [])
    lags = lags if lags is not None else features_config.get("lags", [])
    if include_diff is None:
        include_diff = bool(features_config.get("include_diff", True))
    stats = features_config.get("rolling_stats", ["mean", "std"])

    result = frame.copy()
    before = set(result.columns)

    if windows:
        result = add_rolling_features(
            result,
            columns=list(feature_columns),
            windows=list(windows),
            stats=list(stats),
            id_column=data.id_column,
            time_column=data.time_column,
        )
    if lags:
        result = add_lag_features(
            result,
            columns=list(feature_columns),
            lags=list(lags),
            id_column=data.id_column,
            time_column=data.time_column,
        )
    if include_diff:
        result = add_diff_features(
            result,
            columns=list(feature_columns),
            id_column=data.id_column,
            time_column=data.time_column,
        )

    engineered = [column for column in result.columns if column not in before]
    if dropna and engineered:
        n_before = len(result)
        result = result.dropna(subset=engineered).reset_index(drop=True)
        logger.info(
            "Dropped %d warm-up row(s) with incomplete history", n_before - len(result)
        )

    logger.info("Engineered %d additional feature column(s)", len(engineered))
    return result, engineered
