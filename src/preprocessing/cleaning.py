"""Cleaning steps applied before scaling and windowing.

The CMAPSS sensor set contains several channels that never move within a
sub-dataset (for FD001: sensors 1, 5, 10, 16, 18 and 19). They carry no signal,
break variance-based scalers and inflate model input width, so they are dropped
here rather than in every downstream consumer.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger(__name__)

__all__ = [
    "clip_rul",
    "drop_constant_sensors",
    "find_constant_sensors",
    "prepare_frame",
    "remove_duplicate_cycles",
    "select_feature_columns",
]


def find_constant_sensors(
    frame: pd.DataFrame,
    sensor_columns: list[str] | None = None,
    config: Config | None = None,
) -> list[str]:
    """Return sensor columns with a single distinct value."""
    config = config or get_config()
    sensor_columns = sensor_columns or config.sensor_columns
    present = [column for column in sensor_columns if column in frame.columns]
    return [column for column in present if frame[column].nunique(dropna=False) <= 1]


def select_feature_columns(
    frame: pd.DataFrame, config: Config | None = None
) -> list[str]:
    """Resolve the sensor columns that feed the models, in a stable order.

    Order matters: it defines the feature axis of every sequence array, the
    column order the scaler was fitted on, and therefore the layout the API
    must send at inference time.
    """
    config = config or get_config()
    data = config.data
    columns = [column for column in config.sensor_columns if column in frame.columns]

    dropped: list[str] = []
    if bool(data.get("drop_constant_sensors", True)):
        dropped += find_constant_sensors(frame, columns, config)
    dropped += [column for column in (data.get("drop_sensors") or []) if column in columns]

    selected = [column for column in columns if column not in set(dropped)]
    if not selected:
        raise ValueError("No usable sensor columns remain after cleaning")
    logger.info(
        "Selected %d/%d sensors (dropped: %s)",
        len(selected),
        len(columns),
        sorted(set(dropped)) or "none",
    )
    return selected


def drop_constant_sensors(
    frame: pd.DataFrame, config: Config | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Drop zero-variance sensor columns.

    Returns the trimmed frame and the names that were removed.
    """
    constant = find_constant_sensors(frame, config=config)
    if not constant:
        return frame.copy(), []
    return frame.drop(columns=constant), constant


def remove_duplicate_cycles(
    frame: pd.DataFrame, config: Config | None = None
) -> pd.DataFrame:
    """Drop repeated ``(engine, cycle)`` rows, keeping the first occurrence."""
    config = config or get_config()
    keys = [config.data.id_column, config.data.time_column]
    n_duplicates = int(frame.duplicated(subset=keys).sum())
    if not n_duplicates:
        return frame
    logger.warning("Dropping %d duplicate cycle row(s)", n_duplicates)
    return frame.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


def clip_rul(
    frame: pd.DataFrame,
    cap: float | None,
    target_column: str = "RUL",
) -> pd.DataFrame:
    """Clip the RUL target at ``cap`` (no-op when ``cap`` is ``None``)."""
    if cap is None:
        return frame
    frame = frame.copy()
    n_clipped = int((frame[target_column] > cap).sum())
    frame[target_column] = frame[target_column].clip(upper=cap)
    logger.info("Capped RUL at %s (%d row(s) affected)", cap, n_clipped)
    return frame


def prepare_frame(
    frame: pd.DataFrame,
    config: Config | None = None,
    cap: float | None = None,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the full cleaning chain and report the surviving feature columns.

    Sorts by ``(engine, cycle)``, removes duplicate cycles, optionally caps the
    target and selects the informative sensors.
    """
    config = config or get_config()
    data = config.data
    target_column = target_column or data.target_column

    frame = frame.sort_values([data.id_column, data.time_column]).reset_index(drop=True)
    frame = remove_duplicate_cycles(frame, config)
    if target_column in frame.columns:
        frame = clip_rul(frame, cap, target_column)

    feature_columns = select_feature_columns(frame, config)
    return frame, feature_columns
