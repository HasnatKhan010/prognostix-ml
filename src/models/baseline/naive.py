"""Constant-prediction baselines.

A model that ignores the sensors entirely and always answers with the training
mean still reaches roughly 58 cycles RMSE on FD001. Any architecture that cannot
beat that number by a wide margin is not learning degradation - it is learning
the RUL distribution. These baselines make that comparison explicit.

Unlike ``sklearn.dummy.DummyRegressor`` they accept the 3-D sequence arrays used
throughout this project, so they slot into the same evaluation code as the RNNs.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ConstantBaseline",
    "MeanBaseline",
    "MedianBaseline",
    "QuantileBaseline",
    "build_naive",
]


class ConstantBaseline:
    """Predict one fixed value for every input, learned from the targets.

    Parameters
    ----------
    strategy:
        ``mean``, ``median`` or ``quantile``.
    quantile:
        Used when ``strategy="quantile"``. A low quantile makes the baseline
        deliberately conservative, which matters when a late prediction means an
        unplanned failure.
    """

    strategy: str = "mean"

    def __init__(self, strategy: str = "mean", quantile: float = 0.5):
        if strategy not in {"mean", "median", "quantile"}:
            raise ValueError(
                f"Unknown strategy {strategy!r}; use 'mean', 'median' or 'quantile'"
            )
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")
        self.strategy = strategy
        self.quantile = float(quantile)
        self.constant_: float | None = None

    def fit(self, X=None, y=None) -> "ConstantBaseline":
        """Learn the constant from ``y``. ``X`` is accepted and ignored."""
        if y is None:
            raise ValueError("y is required to fit a constant baseline")
        targets = np.asarray(y, dtype=float).ravel()
        if targets.size == 0:
            raise ValueError("Cannot fit on an empty target array")

        if self.strategy == "mean":
            self.constant_ = float(np.mean(targets))
        elif self.strategy == "median":
            self.constant_ = float(np.median(targets))
        else:
            self.constant_ = float(np.quantile(targets, self.quantile))

        logger.info("%s baseline constant: %.4f", self.strategy, self.constant_)
        return self

    def predict(self, X) -> np.ndarray:
        """Return the learned constant, once per row of ``X``."""
        if self.constant_ is None:
            raise RuntimeError("Baseline is not fitted; call fit(X, y) first")
        n_samples = len(X) if X is not None and len(np.shape(X)) else 1
        return np.full(n_samples, self.constant_, dtype=float)

    def get_params(self, deep: bool = False) -> dict[str, object]:
        """scikit-learn style parameter access."""
        return {"strategy": self.strategy, "quantile": self.quantile}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        fitted = "unfitted" if self.constant_ is None else f"{self.constant_:.3f}"
        return f"{type(self).__name__}(strategy={self.strategy!r}, constant={fitted})"


class MeanBaseline(ConstantBaseline):
    """Always predict the mean training RUL."""

    def __init__(self):
        super().__init__(strategy="mean")


class MedianBaseline(ConstantBaseline):
    """Always predict the median training RUL - robust to the long RUL tail."""

    def __init__(self):
        super().__init__(strategy="median")


class QuantileBaseline(ConstantBaseline):
    """Always predict a chosen quantile of the training RUL."""

    def __init__(self, quantile: float = 0.25):
        super().__init__(strategy="quantile", quantile=quantile)


def build_naive(strategy: str = "mean", quantile: float = 0.5) -> ConstantBaseline:
    """Construct a constant baseline by strategy name."""
    return ConstantBaseline(strategy=strategy, quantile=quantile)
