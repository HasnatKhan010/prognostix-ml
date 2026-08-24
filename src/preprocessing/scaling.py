"""Feature scaling, persisted together with the metadata inference needs.

The notebooks fitted a ``StandardScaler`` on the training engines and threw it
away, which makes a trained model unusable in production: raw sensor readings
arriving at the API must be transformed with the *same* statistics. Scalers are
therefore saved as a :class:`ScalerBundle` that also records the feature order
and window size the model expects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import Config, get_config

logger = logging.getLogger(__name__)

__all__ = [
    "ScalerBundle",
    "apply_scaler",
    "build_scaler",
    "fit_scaler",
    "load_scaler",
    "save_scaler",
]

_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}


@dataclass
class ScalerBundle:
    """A fitted scaler plus the contract the models were trained against."""

    scaler: StandardScaler | MinMaxScaler
    feature_columns: list[str]
    kind: str = "standard"
    window_size: int | None = None
    rul_cap: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def transform(self, values: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Scale a 2-D array/frame whose columns follow ``feature_columns``."""
        if isinstance(values, pd.DataFrame):
            missing = [c for c in self.feature_columns if c not in values.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            values = values[self.feature_columns]
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2-D array, got shape {array.shape}")
        if array.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {array.shape[1]}"
            )
        return self.scaler.transform(array)

    def transform_window(self, window: np.ndarray) -> np.ndarray:
        """Scale one ``(cycles, n_features)`` window."""
        return self.transform(np.asarray(window, dtype=float))

    def transform_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Scale a batch of ``(n, window, n_features)`` sequences."""
        array = np.asarray(sequences, dtype=float)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3-D array, got shape {array.shape}")
        n, window, n_features = array.shape
        flat = self.transform(array.reshape(-1, n_features))
        return flat.reshape(n, window, n_features)


def build_scaler(kind: str = "standard"):
    """Instantiate a scaler by name (``standard`` or ``minmax``)."""
    try:
        return _SCALERS[kind]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown scaler {kind!r}; choose from {sorted(_SCALERS)}"
        ) from exc


def fit_scaler(
    frame: pd.DataFrame,
    feature_columns: list[str],
    kind: str = "standard",
    window_size: int | None = None,
    rul_cap: float | None = None,
    metadata: dict[str, object] | None = None,
) -> ScalerBundle:
    """Fit a scaler on the training split only and wrap it in a bundle."""
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    scaler = build_scaler(kind)
    scaler.fit(frame[feature_columns])
    logger.info("Fitted %s scaler on %d rows", kind, len(frame))
    return ScalerBundle(
        scaler=scaler,
        feature_columns=list(feature_columns),
        kind=kind,
        window_size=window_size,
        rul_cap=rul_cap,
        metadata=dict(metadata or {}),
    )


def apply_scaler(
    frame: pd.DataFrame, bundle: ScalerBundle, copy: bool = True
) -> pd.DataFrame:
    """Return ``frame`` with its feature columns scaled in place."""
    result = frame.copy() if copy else frame
    result[bundle.feature_columns] = bundle.transform(result[bundle.feature_columns])
    return result


def save_scaler(
    bundle: ScalerBundle,
    path: str | Path | None = None,
    config: Config | None = None,
) -> Path:
    """Persist a bundle to ``artifacts/<scaler_filename>``."""
    config = config or get_config()
    if path is None:
        path = config.path("artifacts") / config.preprocessing.scaler_filename
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Saved scaler bundle -> %s", path)
    return path


def load_scaler(
    path: str | Path | None = None, config: Config | None = None
) -> ScalerBundle:
    """Load a bundle written by :func:`save_scaler`."""
    config = config or get_config()
    if path is None:
        path = config.path("artifacts") / config.preprocessing.scaler_filename
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_data.py` first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, ScalerBundle):
        raise TypeError(f"{path} does not hold a ScalerBundle (got {type(bundle)})")
    return bundle
