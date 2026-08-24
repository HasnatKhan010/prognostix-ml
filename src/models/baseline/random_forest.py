"""Tabular baselines: Random Forest and Linear Regression over window statistics.

Tree and linear models cannot read a ``(window, n_sensors)`` matrix, so each
window is flattened into per-sensor aggregations first. :class:`TabularRULModel`
bundles that transform with the estimator, which means the same object used in
training is the one served at inference - the flattening can never drift out of
sync with the weights.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config import Config, get_config
from src.features.engineering import (
    DEFAULT_STATS,
    create_statistical_features,
    statistical_feature_names,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TabularRULModel",
    "build_linear_regression",
    "build_random_forest",
    "build_tabular_estimator",
]


def build_random_forest(
    config: Config | None = None, **overrides: Any
) -> RandomForestRegressor:
    """Random Forest configured from ``models.random_forest``."""
    config = config or get_config()
    params = dict(config.models.get("random_forest", {}) or {})
    params.update(overrides)
    return RandomForestRegressor(**params)


def build_linear_regression(
    config: Config | None = None, **overrides: Any
) -> LinearRegression:
    """Ordinary least squares - the cheapest reference point that uses the sensors."""
    config = config or get_config()
    params = dict(config.models.get("linear", {}) or {})
    params.update(overrides)
    return LinearRegression(**params)


def build_tabular_estimator(name: str, config: Config | None = None, **overrides: Any):
    """Construct ``random_forest`` or ``linear`` by name."""
    builders = {
        "random_forest": build_random_forest,
        "linear": build_linear_regression,
    }
    try:
        return builders[name](config=config, **overrides)
    except KeyError as exc:
        raise ValueError(
            f"Unknown tabular model {name!r}; choose from {sorted(builders)}"
        ) from exc


@dataclass
class TabularRULModel:
    """A scikit-learn estimator plus the sequence-flattening step it expects.

    Attributes
    ----------
    estimator:
        Any fitted or unfitted regressor with ``fit``/``predict``.
    stats:
        Aggregations applied over the time axis, in output-block order.
    feature_columns:
        Sensor names behind the feature axis, recorded for traceability.
    """

    estimator: Any
    stats: tuple[str, ...] = DEFAULT_STATS
    feature_columns: list[str] | None = None
    window_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- transform -------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Flatten sequences into the estimator's tabular input."""
        array = np.asarray(X, dtype=float)
        if array.ndim == 2:  # a single window
            array = array[None, ...]
        return create_statistical_features(array, stats=self.stats)

    def feature_names(self) -> list[str] | None:
        """Names of the flattened features, when sensor names are known."""
        if self.feature_columns is None:
            return None
        return statistical_feature_names(self.feature_columns, self.stats)

    # -- estimator API ---------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabularRULModel:
        """Flatten ``X`` and fit the estimator."""
        features = self.transform(X)
        targets = np.asarray(y, dtype=float).ravel()
        if len(features) != len(targets):
            raise ValueError(f"X/y length mismatch: {len(features)} vs {len(targets)}")
        if self.window_size is None and np.asarray(X).ndim == 3:
            self.window_size = int(np.asarray(X).shape[1])
        logger.info(
            "Fitting %s on %s tabular features",
            type(self.estimator).__name__,
            features.shape,
        )
        self.estimator.fit(features, targets)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL for one window or a batch of windows."""
        return np.asarray(self.estimator.predict(self.transform(X)), dtype=float).ravel()

    def feature_importances(self) -> dict[str, float] | None:
        """Estimator importances (or absolute coefficients) keyed by feature name.

        Returns ``None`` for estimators that expose neither.
        """
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            coefficients = getattr(self.estimator, "coef_", None)
            if coefficients is None:
                return None
            importances = np.abs(np.asarray(coefficients, dtype=float).ravel())

        names = self.feature_names() or [
            f"f{index}" for index in range(len(importances))
        ]
        if len(names) != len(importances):
            names = [f"f{index}" for index in range(len(importances))]
        pairs = sorted(
            zip(names, (float(value) for value in importances), strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
        return dict(pairs)

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist the whole bundle with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved tabular model -> %s", path)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        stats: Sequence[str] | None = None,
        feature_columns: list[str] | None = None,
    ) -> TabularRULModel:
        """Load a bundle, or adopt a bare estimator saved by the notebooks.

        The committed ``*_baseline.joblib`` files hold raw scikit-learn
        estimators. They are wrapped here with the default aggregation set they
        were trained on, so old and new artifacts load through one code path.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Train it first: "
                "python scripts/train.py --model random_forest"
            )

        loaded = joblib.load(path)
        if isinstance(loaded, cls):
            return loaded
        if not hasattr(loaded, "predict"):
            raise TypeError(f"{path} holds {type(loaded)}, which has no predict()")

        logger.info(
            "Wrapping bare %s from %s with default window statistics",
            type(loaded).__name__,
            path.name,
        )
        return cls(
            estimator=loaded,
            stats=tuple(stats or DEFAULT_STATS),
            feature_columns=feature_columns,
            metadata={"source": "bare_estimator", "path": str(path)},
        )
