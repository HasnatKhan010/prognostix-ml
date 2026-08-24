"""Engine-level splitting and sliding-window sequence construction.

Two rules govern this module:

* **Split by engine, never by row.** Consecutive cycles of one engine are almost
  identical, so a random row split leaks the answer into the validation set and
  produces optimistic scores.
* **Window inside an engine, never across engines.** Sequences are built per
  engine after sorting by cycle, so no window straddles two machines.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

__all__ = [
    "create_sequences",
    "last_window_per_engine",
    "split_by_engine",
    "split_engines",
]


def split_engines(
    engine_ids: np.ndarray | list[int],
    test_size: float = 0.30,
    val_ratio: float = 0.50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split engine ids into train / validation / test groups.

    ``test_size`` is held out first, then divided between validation and test
    according to ``val_ratio``. The defaults give a 70 / 15 / 15 split and
    reproduce the arrays committed under ``data/processed/``.
    """
    engine_ids = np.asarray(engine_ids)
    if len(engine_ids) < 3:
        raise ValueError(f"Need at least 3 engines to split, got {len(engine_ids)}")

    train_engines, held_out = train_test_split(
        engine_ids, test_size=test_size, random_state=random_state
    )
    val_engines, test_engines = train_test_split(
        held_out, test_size=val_ratio, random_state=random_state
    )
    logger.info(
        "Split engines: train=%d val=%d test=%d",
        len(train_engines),
        len(val_engines),
        len(test_engines),
    )
    return train_engines, val_engines, test_engines


def split_by_engine(
    frame: pd.DataFrame,
    engine_ids: np.ndarray | list[int],
    id_column: str = "engine_id",
) -> pd.DataFrame:
    """Select every row belonging to ``engine_ids``."""
    return frame[frame[id_column].isin(engine_ids)].copy()


def create_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str | None = "RUL",
    window_size: int = 30,
    id_column: str = "engine_id",
    time_column: str = "cycle",
    return_ids: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a fixed window over each engine's history.

    Window ``i`` spans cycles ``[i - window_size, i)`` and its label is the RUL
    at cycle ``i`` - i.e. the model predicts the RUL of the cycle immediately
    following the observed window. Engines shorter than ``window_size + 1``
    cycles contribute nothing.

    Parameters
    ----------
    return_ids:
        Also return the engine id each window came from, which is what makes
        per-engine evaluation and monitoring possible.

    Returns
    -------
    ``(X, y)`` with shapes ``(n_windows, window_size, n_features)`` and
    ``(n_windows,)``. When ``target_column`` is ``None`` the returned ``y`` is
    empty.
    """
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if target_column is not None and target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column!r}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    windows: list[np.ndarray] = []
    labels: list[float] = []
    engines: list[int] = []

    for engine_id, engine_frame in frame.groupby(id_column, sort=True):
        engine_frame = engine_frame.sort_values(time_column)
        features = engine_frame[feature_columns].to_numpy(dtype=float)
        targets = (
            engine_frame[target_column].to_numpy(dtype=float)
            if target_column is not None
            else None
        )

        for index in range(window_size, len(engine_frame)):
            windows.append(features[index - window_size : index])
            if targets is not None:
                labels.append(targets[index])
            engines.append(int(engine_id))

    n_features = len(feature_columns)
    if windows:
        X = np.stack(windows)
    else:
        logger.warning(
            "No engine is longer than the %d-cycle window; returning empty arrays",
            window_size,
        )
        X = np.empty((0, window_size, n_features), dtype=float)

    y = np.asarray(labels, dtype=float)
    ids = np.asarray(engines, dtype=int)

    logger.info("Built %d sequence(s) of shape (%d, %d)", len(X), window_size, n_features)
    return (X, y, ids) if return_ids else (X, y)


def last_window_per_engine(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = 30,
    id_column: str = "engine_id",
    time_column: str = "cycle",
    pad: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one window per engine: its most recent ``window_size`` cycles.

    This is the CMAPSS test protocol - each test engine is scored on a single
    prediction made from its final observation - and the same shape the API
    receives from a live machine.

    Parameters
    ----------
    pad:
        Engines with fewer than ``window_size`` cycles are front-padded with
        their earliest reading. With ``pad=False`` they are skipped instead.

    Returns
    -------
    ``(X, engine_ids)`` with ``X`` of shape ``(n_engines, window_size, n_features)``.
    """
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    windows: list[np.ndarray] = []
    engines: list[int] = []

    for engine_id, engine_frame in frame.groupby(id_column, sort=True):
        features = (
            engine_frame.sort_values(time_column)[feature_columns].to_numpy(dtype=float)
        )
        if len(features) >= window_size:
            windows.append(features[-window_size:])
        elif pad:
            padding = np.repeat(features[:1], window_size - len(features), axis=0)
            windows.append(np.concatenate([padding, features]))
            logger.debug(
                "Engine %s padded from %d to %d cycles",
                engine_id,
                len(features),
                window_size,
            )
        else:
            logger.warning(
                "Skipping engine %s: %d cycles < window %d",
                engine_id,
                len(features),
                window_size,
            )
            continue
        engines.append(int(engine_id))

    n_features = len(feature_columns)
    X = (
        np.stack(windows)
        if windows
        else np.empty((0, window_size, n_features), dtype=float)
    )
    return X, np.asarray(engines, dtype=int)
