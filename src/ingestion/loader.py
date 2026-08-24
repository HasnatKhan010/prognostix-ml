"""Loaders for the CMAPSS turbofan degradation dataset.

The raw files are whitespace-separated with no header. Column names follow the
dataset documentation: unit number, time in cycles, three operational settings
and twenty-one sensor measurements.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger(__name__)

__all__ = [
    "add_rul",
    "load_cmapss",
    "load_processed",
    "load_raw_split",
    "load_rul_truth",
    "load_sequences",
    "save_sequences",
]


def load_cmapss(
    path: str | Path,
    columns: list[str] | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Read a single raw CMAPSS ``*.txt`` file into a DataFrame.

    Parameters
    ----------
    path:
        Path to e.g. ``data/raw/CMAPSS/train_FD001.txt``.
    columns:
        Column names to apply. Defaults to the standard 26-column layout.
    """
    config = config or get_config()
    columns = list(columns) if columns is not None else config.raw_columns
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )

    frame = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    logger.info("Loaded %s -> %s", path.name, frame.shape)
    return frame


def load_raw_split(
    split: str = "train",
    dataset: str | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Load the ``train`` or ``test`` file of a CMAPSS sub-dataset."""
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    config = config or get_config()
    dataset = dataset or config.data.dataset
    return load_cmapss(
        config.path("data_raw") / f"{split}_{dataset}.txt", config=config
    )


def load_rul_truth(
    dataset: str | None = None, config: Config | None = None
) -> pd.DataFrame:
    """Load the ground-truth RUL vector that accompanies a test file.

    One row per test engine: the number of cycles it survives *after* its last
    recorded cycle.
    """
    config = config or get_config()
    dataset = dataset or config.data.dataset
    path = config.path("data_raw") / f"RUL_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    truth = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])
    truth.insert(0, "engine_id", np.arange(1, len(truth) + 1))
    return truth


def add_rul(
    frame: pd.DataFrame,
    id_column: str = "engine_id",
    time_column: str = "cycle",
    target_column: str = "RUL",
    cap: float | None = None,
    final_rul: pd.Series | None = None,
) -> pd.DataFrame:
    """Attach a Remaining Useful Life column.

    For training runs each engine is observed until failure, so RUL is simply
    ``max(cycle) - cycle``. For test runs the series is truncated before
    failure, so pass ``final_rul`` (indexed by engine id) to offset each engine
    by the cycles it still had left.

    Parameters
    ----------
    cap:
        Optional upper clip. Capping (commonly at 125) reflects that
        degradation is not observable while an engine is healthy.
    """
    frame = frame.copy()
    last_cycle = frame.groupby(id_column)[time_column].transform("max")
    frame[target_column] = last_cycle - frame[time_column]

    if final_rul is not None:
        offsets = frame[id_column].map(final_rul)
        if offsets.isna().any():
            missing = sorted(frame.loc[offsets.isna(), id_column].unique())
            raise ValueError(f"No ground-truth RUL for engines: {missing}")
        frame[target_column] = frame[target_column] + offsets.to_numpy()

    if cap is not None:
        frame[target_column] = frame[target_column].clip(upper=cap)
    return frame


def load_processed(
    name: str, config: Config | None = None, **read_csv_kwargs
) -> pd.DataFrame:
    """Read a CSV from ``data/processed``.

    ``name`` may be given with or without the ``.csv`` suffix.
    """
    config = config or get_config()
    filename = name if name.endswith(".csv") else f"{name}.csv"
    path = config.path("data_processed") / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_data.py` first."
        )
    return pd.read_csv(path, **read_csv_kwargs)


def load_sequences(
    split: str, config: Config | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load a windowed split saved by :func:`save_sequences`.

    Returns
    -------
    tuple of ``(X, y)`` with shapes ``(n_windows, window_size, n_features)``
    and ``(n_windows,)``.
    """
    config = config or get_config()
    path = config.path("data_processed") / f"{split}_sequences.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_data.py` first."
        )
    with np.load(path) as payload:
        X, y = payload["X"], payload["y"]
    logger.info("Loaded %s sequences: X=%s y=%s", split, X.shape, y.shape)
    return X, y


def save_sequences(
    split: str,
    X: np.ndarray,
    y: np.ndarray,
    config: Config | None = None,
    **extra: np.ndarray,
) -> Path:
    """Write a windowed split to ``data/processed/<split>_sequences.npz``."""
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    config = config or get_config()
    directory = config.path("data_processed")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}_sequences.npz"
    np.savez_compressed(path, X=X, y=y, **extra)
    logger.info("Saved %s (X=%s)", path.name, X.shape)
    return path
