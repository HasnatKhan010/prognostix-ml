"""Data quality checks for raw frames, windowed arrays and API payloads.

Every check returns a :class:`ValidationReport` so callers can decide whether a
problem is fatal (``report.raise_for_errors()``) or merely worth logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationError",
    "ValidationReport",
    "validate_raw_frame",
    "validate_sequences",
    "validate_window",
]


class ValidationError(ValueError):
    """Raised when data does not satisfy a mandatory expectation."""


@dataclass
class ValidationReport:
    """Outcome of a validation pass."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no blocking error was recorded."""
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def raise_for_errors(self) -> ValidationReport:
        """Raise :class:`ValidationError` if any error was recorded."""
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise ValidationError(f"Data validation failed:\n  - {joined}")
        return self

    def log(self, logger_: logging.Logger | None = None) -> ValidationReport:
        """Emit warnings and errors through a logger."""
        log = logger_ or logger
        for message in self.warnings:
            log.warning(message)
        for message in self.errors:
            log.error(message)
        if self.ok and not self.warnings:
            log.info("Validation passed (%s)", self.stats.get("n_rows", "n/a"))
        return self

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return (
            f"[{status}] {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )


def validate_raw_frame(
    frame: pd.DataFrame,
    config: Config | None = None,
    expect_target: bool = False,
) -> ValidationReport:
    """Validate a freshly loaded CMAPSS frame.

    Checks the schema, missing values, duplicate ``(engine, cycle)`` keys,
    per-engine cycle monotonicity, non-finite values and constant sensors.
    """
    config = config or get_config()
    data = config.data
    report = ValidationReport()

    id_column = data.id_column
    time_column = data.time_column
    expected = list(config.raw_columns)
    if expect_target:
        expected.append(data.target_column)

    missing_columns = [column for column in expected if column not in frame.columns]
    if missing_columns:
        report.error(f"Missing columns: {missing_columns}")
        return report  # nothing else can be trusted

    report.stats["n_rows"] = len(frame)
    report.stats["n_columns"] = int(frame.shape[1])

    if frame.empty:
        report.error("Frame is empty")
        return report

    null_counts = frame.isna().sum()
    non_null = null_counts[null_counts > 0]
    if not non_null.empty:
        report.error(f"Null values found: {non_null.to_dict()}")

    numeric = frame.select_dtypes(include=[np.number])
    non_numeric = [c for c in expected if c not in numeric.columns]
    if non_numeric:
        report.error(f"Non-numeric columns: {non_numeric}")
    else:
        n_non_finite = int((~np.isfinite(frame[expected].to_numpy())).sum())
        if n_non_finite:
            report.error(f"{n_non_finite} non-finite value(s) (inf/NaN)")

    n_engines = int(frame[id_column].nunique())
    report.stats["n_engines"] = n_engines
    if n_engines == 0:
        report.error("No engines present")

    duplicates = int(frame.duplicated(subset=[id_column, time_column]).sum())
    if duplicates:
        report.error(f"{duplicates} duplicate ({id_column}, {time_column}) pair(s)")

    if (frame[time_column] < 1).any():
        report.error(f"{time_column} must be >= 1")

    non_monotonic = [
        int(engine)
        for engine, group in frame.groupby(id_column, sort=False)[time_column]
        if not group.is_monotonic_increasing
    ]
    if non_monotonic:
        report.error(
            f"{time_column} is not increasing for engine(s): {non_monotonic[:10]}"
        )

    lengths = frame.groupby(id_column)[time_column].size()
    report.stats["min_engine_length"] = int(lengths.min())
    report.stats["max_engine_length"] = int(lengths.max())
    window = int(data.window_size)
    too_short = lengths[lengths <= window]
    if not too_short.empty:
        report.warn(
            f"{len(too_short)} engine(s) shorter than the {window}-cycle window "
            "produce no sequences"
        )

    constant = [
        column for column in config.sensor_columns if frame[column].nunique() <= 1
    ]
    report.stats["constant_sensors"] = constant
    if constant:
        report.warn(f"Constant sensors (no signal): {constant}")

    if expect_target:
        target = frame[data.target_column]
        report.stats["rul_min"] = float(target.min())
        report.stats["rul_max"] = float(target.max())
        if (target < 0).any():
            report.error(f"{data.target_column} contains negative values")

    return report


def validate_sequences(
    X: np.ndarray,
    y: np.ndarray | None = None,
    window_size: int | None = None,
    n_features: int | None = None,
    config: Config | None = None,
) -> ValidationReport:
    """Validate a windowed dataset produced by the preprocessing pipeline."""
    config = config or get_config()
    window_size = window_size or int(config.data.window_size)
    report = ValidationReport()

    X = np.asarray(X)
    if X.ndim != 3:
        report.error(f"X must be 3-D (n, window, features), got shape {X.shape}")
        return report

    report.stats["shape"] = tuple(int(dim) for dim in X.shape)
    if X.shape[0] == 0:
        report.error("X contains no windows")
        return report
    if X.shape[1] != window_size:
        report.error(f"Expected window size {window_size}, got {X.shape[1]}")
    if n_features is not None and X.shape[2] != n_features:
        report.error(f"Expected {n_features} features, got {X.shape[2]}")
    if not np.isfinite(X).all():
        report.error(f"X holds {int((~np.isfinite(X)).sum())} non-finite value(s)")

    if y is not None:
        y = np.asarray(y)
        if y.ndim != 1:
            report.error(f"y must be 1-D, got shape {y.shape}")
        elif len(y) != len(X):
            report.error(f"X/y length mismatch: {len(X)} vs {len(y)}")
        elif not np.isfinite(y).all():
            report.error("y holds non-finite value(s)")
        elif (y < 0).any():
            report.error("y holds negative RUL values")
        else:
            report.stats["y_min"] = float(y.min())
            report.stats["y_max"] = float(y.max())
            report.stats["y_mean"] = float(y.mean())

    return report


def validate_window(
    window: np.ndarray,
    window_size: int,
    n_features: int,
    strict_length: bool = True,
) -> ValidationReport:
    """Validate a single inference window of shape ``(window_size, n_features)``."""
    report = ValidationReport()
    array = np.asarray(window, dtype=float)

    if array.ndim != 2:
        report.error(f"Window must be 2-D (cycles, features), got shape {array.shape}")
        return report

    n_cycles, seen_features = array.shape
    report.stats["shape"] = (int(n_cycles), int(seen_features))

    if seen_features != n_features:
        report.error(f"Expected {n_features} features per cycle, got {seen_features}")
    if strict_length and n_cycles != window_size:
        report.error(f"Expected exactly {window_size} cycles, got {n_cycles}")
    elif n_cycles < window_size:
        report.error(f"Need at least {window_size} cycles, got {n_cycles}")
    if array.size and not np.isfinite(array).all():
        report.error("Window holds non-finite value(s)")

    return report
