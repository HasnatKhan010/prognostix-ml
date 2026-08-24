"""Rolling-window statistics computed per engine.

Sensor noise in CMAPSS is substantial; rolling means smooth it while rolling
standard deviations expose the growing instability that precedes failure. Every
window is grouped by engine so history never bleeds between machines, and only
past cycles contribute - no look-ahead.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["add_expanding_features", "add_rolling_features"]

_ROLLING_STATS = ("mean", "std", "min", "max", "median")


def add_rolling_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    windows: Sequence[int] = (5, 10, 20),
    stats: Sequence[str] = ("mean", "std"),
    id_column: str = "engine_id",
    time_column: str = "cycle",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Append ``<column>_roll<window>_<stat>`` features.

    Parameters
    ----------
    min_periods:
        Cycles required before a value is produced. The default of 1 keeps early
        cycles usable (the window simply covers fewer rows); ``std`` still
        yields NaN on the very first cycle of an engine.
    """
    _validate(frame, columns, stats, windows)
    result = frame.sort_values([id_column, time_column]).copy()
    grouped = result.groupby(id_column, sort=False)[list(columns)]

    new_columns: dict[str, pd.Series] = {}
    for window in windows:
        rolling = grouped.rolling(window=window, min_periods=min_periods)
        for stat in stats:
            computed = getattr(rolling, stat)().reset_index(level=0, drop=True)
            for column in columns:
                new_columns[f"{column}_roll{window}_{stat}"] = computed[column]

    result = result.assign(**new_columns)
    logger.debug("Added %d rolling feature(s)", len(new_columns))
    return result


def add_expanding_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    stats: Sequence[str] = ("mean", "std"),
    id_column: str = "engine_id",
    time_column: str = "cycle",
) -> pd.DataFrame:
    """Append ``<column>_expanding_<stat>`` features over an engine's whole past.

    Useful as a slow-moving reference: comparing a fast rolling mean against the
    expanding mean highlights drift from an engine's own baseline.
    """
    _validate(frame, columns, stats, windows=())
    result = frame.sort_values([id_column, time_column]).copy()
    expanding = result.groupby(id_column, sort=False)[list(columns)].expanding(
        min_periods=1
    )

    new_columns: dict[str, pd.Series] = {}
    for stat in stats:
        computed = getattr(expanding, stat)().reset_index(level=0, drop=True)
        for column in columns:
            new_columns[f"{column}_expanding_{stat}"] = computed[column]

    return result.assign(**new_columns)


def _validate(
    frame: pd.DataFrame,
    columns: Sequence[str],
    stats: Sequence[str],
    windows: Sequence[int],
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    unknown = [stat for stat in stats if stat not in _ROLLING_STATS]
    if unknown:
        raise ValueError(
            f"Unsupported statistic(s) {unknown}; choose from {list(_ROLLING_STATS)}"
        )
    invalid = [window for window in windows if window < 1]
    if invalid:
        raise ValueError(f"Window sizes must be >= 1, got {invalid}")
