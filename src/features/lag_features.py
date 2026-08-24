"""Lag and difference features computed per engine.

Degradation shows up as *change*: the absolute value of a temperature sensor
matters far less than how fast it is climbing. Lags expose earlier readings to
the model directly; differences expose the slope between them.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["add_diff_features", "add_lag_features"]


def add_lag_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int] = (1, 2, 5),
    id_column: str = "engine_id",
    time_column: str = "cycle",
) -> pd.DataFrame:
    """Append ``<column>_lag<k>`` features.

    The first ``max(lags)`` cycles of every engine necessarily hold NaN; drop or
    impute them before fitting.
    """
    _validate(frame, columns, lags)
    result = frame.sort_values([id_column, time_column]).copy()
    grouped = result.groupby(id_column, sort=False)[list(columns)]

    new_columns: dict[str, pd.Series] = {}
    for lag in lags:
        shifted = grouped.shift(lag)
        for column in columns:
            new_columns[f"{column}_lag{lag}"] = shifted[column]

    result = result.assign(**new_columns)
    logger.debug("Added %d lag feature(s)", len(new_columns))
    return result


def add_diff_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    periods: Sequence[int] = (1,),
    id_column: str = "engine_id",
    time_column: str = "cycle",
) -> pd.DataFrame:
    """Append ``<column>_diff<k>`` features - the change over ``k`` cycles."""
    _validate(frame, columns, periods)
    result = frame.sort_values([id_column, time_column]).copy()
    grouped = result.groupby(id_column, sort=False)[list(columns)]

    new_columns: dict[str, pd.Series] = {}
    for period in periods:
        differenced = grouped.diff(period)
        for column in columns:
            new_columns[f"{column}_diff{period}"] = differenced[column]

    result = result.assign(**new_columns)
    logger.debug("Added %d difference feature(s)", len(new_columns))
    return result


def _validate(
    frame: pd.DataFrame, columns: Sequence[str], offsets: Sequence[int]
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    invalid = [offset for offset in offsets if offset < 1]
    if invalid:
        raise ValueError(f"Offsets must be >= 1, got {invalid}")
