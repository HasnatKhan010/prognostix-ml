"""Leaderboard handling: merge run results into one comparable table.

``artifacts/baseline_results.csv`` was written by the baseline notebook with the
columns ``Model,MAE,RMSE``. The richer metric set added later must not break it,
so :func:`update_leaderboard` merges by model name and tolerates missing
columns on either side.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger(__name__)

__all__ = [
    "build_leaderboard",
    "load_leaderboard",
    "save_leaderboard",
    "update_leaderboard",
]

DEFAULT_FILENAME = "model_comparison.csv"
SORT_COLUMN = "RMSE"


def build_leaderboard(
    rows: list[dict[str, object]], sort_by: str = SORT_COLUMN
) -> pd.DataFrame:
    """Assemble metric dictionaries into a sorted table."""
    if not rows:
        return pd.DataFrame(columns=["Model", "MAE", "RMSE"])
    frame = pd.DataFrame(rows)
    return _sorted(frame, sort_by)


def load_leaderboard(
    path: str | Path | None = None,
    config: Config | None = None,
    filename: str = DEFAULT_FILENAME,
) -> pd.DataFrame:
    """Load a leaderboard CSV, returning an empty frame when absent."""
    path = _resolve(path, config, filename)
    if not path.exists():
        return pd.DataFrame(columns=["Model", "MAE", "RMSE"])
    return pd.read_csv(path)


def save_leaderboard(
    frame: pd.DataFrame,
    path: str | Path | None = None,
    config: Config | None = None,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """Write a leaderboard to CSV."""
    path = _resolve(path, config, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Saved leaderboard (%d row(s)) -> %s", len(frame), path)
    return path


def update_leaderboard(
    rows: list[dict[str, object]] | dict[str, object] | pd.DataFrame,
    path: str | Path | None = None,
    config: Config | None = None,
    filename: str = DEFAULT_FILENAME,
    sort_by: str = SORT_COLUMN,
    save: bool = True,
) -> pd.DataFrame:
    """Insert or replace rows in the leaderboard, keyed by ``Model``.

    Re-running a training script overwrites that model's row instead of
    appending a duplicate.
    """
    if isinstance(rows, dict):
        rows = [rows]
    incoming = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if incoming.empty:
        return load_leaderboard(path, config, filename)
    if "Model" not in incoming.columns:
        raise ValueError("Every leaderboard row needs a 'Model' column")

    existing = load_leaderboard(path, config, filename)
    if not existing.empty and "Model" in existing.columns:
        existing = existing[~existing["Model"].isin(incoming["Model"])]
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = _sorted(merged, sort_by)

    if save:
        save_leaderboard(merged, path, config, filename)
    return merged


def _sorted(frame: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    """Sort ascending by ``sort_by`` when that column exists."""
    if sort_by in frame.columns:
        frame = frame.sort_values(sort_by, na_position="last")
    return frame.reset_index(drop=True)


def _resolve(
    path: str | Path | None, config: Config | None, filename: str
) -> Path:
    if path is not None:
        return Path(path)
    config = config or get_config()
    return config.path("artifacts") / filename
