"""Data drift detection between a reference and a live sensor distribution.

A RUL model is only valid while incoming sensors look like the ones it trained
on. New hardware, a recalibrated probe or a different operating regime shifts the
inputs, and accuracy degrades long before anyone notices ground truth (which for
RUL only arrives when the machine actually fails).

Two complementary tests run per feature:

* **PSI** (Population Stability Index) quantifies *how much* a distribution moved.
  The industry convention - < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant -
  is configurable under ``monitoring.drift``.
* **KS test** asks whether the shift is statistically significant at all, which
  keeps small samples from looking dramatic.

Run as ``python -m monitoring.drift``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config, get_config, setup_logging

logger = logging.getLogger(__name__)

__all__ = [
    "DriftReport",
    "FeatureDrift",
    "detect_drift",
    "flatten_sequences",
    "population_stability_index",
    "reference_distribution",
]

EPSILON = 1e-6


@dataclass
class FeatureDrift:
    """Drift verdict for one feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float
    mean_shift: float
    drifted: bool
    severity: str  # stable | warning | critical

    def to_dict(self) -> dict[str, Any]:
        return {
            key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in asdict(self).items()
        }


@dataclass
class DriftReport:
    """Fleet-level drift verdict across all features."""

    status: str  # stable | warning | critical
    n_samples: int
    n_features: int
    features: list[FeatureDrift] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def drifted_features(self) -> list[str]:
        """Names of features that crossed the PSI warning threshold."""
        return [feature.feature for feature in self.features if feature.drifted]

    @property
    def feature_share(self) -> float:
        """Share of features flagged as drifted (0-1)."""
        return len(self.drifted_features) / self.n_features if self.n_features else 0.0

    @property
    def max_psi(self) -> float:
        return max((feature.psi for feature in self.features), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "drifted_features": self.drifted_features,
            "feature_share": round(self.feature_share, 4),
            "max_psi": round(self.max_psi, 6),
            "thresholds": self.thresholds,
            "details": [feature.to_dict() for feature in self.features],
        }

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved drift report -> %s", path)
        return path

    def summary(self) -> str:
        return (
            f"[{self.status.upper()}] {len(self.drifted_features)}/{self.n_features} "
            f"feature(s) drifted (max PSI {self.max_psi:.4f}, n={self.n_samples})"
        )


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between two 1-D samples.

    Bin edges come from the reference quantiles, so bins hold roughly equal mass
    and the statistic is not dominated by outliers. Empty bins are floored at
    ``EPSILON`` to keep the logarithm finite.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if edges.size < 2:  # constant reference - nothing to compare against
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_share = np.maximum(ref_counts / ref.size, EPSILON)
    cur_share = np.maximum(cur_counts / cur.size, EPSILON)

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def flatten_sequences(sequences: np.ndarray) -> np.ndarray:
    """Collapse ``(n, window, features)`` into ``(n * window, features)``.

    Drift is measured per sensor across all observed cycles, so the window axis
    is irrelevant here.
    """
    array = np.asarray(sequences, dtype=float)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D array, got shape {array.shape}")
    return array.reshape(-1, array.shape[2])


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: Sequence[str] | None = None,
    config: Config | None = None,
    bins: int | None = None,
) -> DriftReport:
    """Compare two feature matrices and return a per-feature drift report.

    Both inputs may be 2-D ``(n, features)`` or 3-D ``(n, window, features)``.
    """
    from scipy.stats import ks_2samp

    config = config or get_config()
    settings = config.monitoring.drift
    bins = int(bins or settings.get("bins", 10))
    psi_warning = float(settings.get("psi_warning", 0.1))
    psi_critical = float(settings.get("psi_critical", 0.25))
    ks_alpha = float(settings.get("ks_alpha", 0.05))
    share_warning = float(settings.get("feature_share_warning", 0.2))
    share_critical = float(settings.get("feature_share_critical", 0.4))

    ref_matrix = flatten_sequences(reference)
    cur_matrix = flatten_sequences(current)
    if ref_matrix.shape[1] != cur_matrix.shape[1]:
        raise ValueError(
            f"Feature count mismatch: reference has {ref_matrix.shape[1]}, "
            f"current has {cur_matrix.shape[1]}"
        )

    n_features = ref_matrix.shape[1]
    names = list(feature_names) if feature_names else [f"feature_{i}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError(
            f"Got {len(names)} feature name(s) for {n_features} feature column(s)"
        )

    results: list[FeatureDrift] = []
    for index, name in enumerate(names):
        ref_column = ref_matrix[:, index]
        cur_column = cur_matrix[:, index]
        psi = population_stability_index(ref_column, cur_column, bins)
        ks_statistic, ks_pvalue = ks_2samp(ref_column, cur_column)

        severity = (
            "critical" if psi >= psi_critical
            else "warning" if psi >= psi_warning
            else "stable"
        )
        # PSI decides magnitude; the KS test guards against calling noise drift.
        drifted = severity != "stable" and float(ks_pvalue) < ks_alpha

        results.append(
            FeatureDrift(
                feature=name,
                psi=float(psi),
                ks_statistic=float(ks_statistic),
                ks_pvalue=float(ks_pvalue),
                reference_mean=float(np.nanmean(ref_column)),
                current_mean=float(np.nanmean(cur_column)),
                mean_shift=float(np.nanmean(cur_column) - np.nanmean(ref_column)),
                drifted=bool(drifted),
                severity=severity,
            )
        )

    n_drifted = sum(1 for feature in results if feature.drifted)
    share = n_drifted / n_features if n_features else 0.0
    any_critical = any(
        feature.drifted and feature.severity == "critical" for feature in results
    )
    status = (
        "critical" if share >= share_critical or any_critical
        else "warning" if share >= share_warning
        else "stable"
    )

    return DriftReport(
        status=status,
        n_samples=int(cur_matrix.shape[0]),
        n_features=n_features,
        features=results,
        thresholds={
            "psi_warning": psi_warning,
            "psi_critical": psi_critical,
            "ks_alpha": ks_alpha,
            "feature_share_warning": share_warning,
            "feature_share_critical": share_critical,
        },
    )


def reference_distribution(
    config: Config | None = None, split: str = "train"
) -> tuple[np.ndarray, list[str] | None]:
    """Load the training distribution the live data is compared against.

    Returns the flattened feature matrix and, when the scaler bundle is present,
    the sensor names behind its columns.
    """
    from src.ingestion.loader import load_sequences

    config = config or get_config()
    X, _ = load_sequences(split, config)

    feature_names: list[str] | None = None
    try:
        from src.preprocessing.scaling import load_scaler

        feature_names = load_scaler(config=config).feature_columns
    except Exception:
        logger.debug("No scaler bundle; drift report will use positional names")

    return flatten_sequences(X), feature_names


def main(argv: list[str] | None = None) -> int:
    """CLI: compare a live split against the training distribution."""
    parser = argparse.ArgumentParser(
        description="Detect sensor drift against the training distribution."
    )
    parser.add_argument(
        "--current",
        default="test",
        help="Split or .npz path to test for drift (default: test).",
    )
    parser.add_argument(
        "--reference", default="train", help="Reference split (default: train)."
    )
    parser.add_argument("--bins", type=int, default=None, help="PSI bin count.")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the JSON report (default: artifacts/reports/drift_report.json).",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Features to print, worst PSI first."
    )
    parser.add_argument(
        "--no-alerts", action="store_true", help="Skip raising alerts on drift."
    )
    args = parser.parse_args(argv)

    setup_logging()
    config = get_config()

    reference, feature_names = reference_distribution(config, args.reference)
    current = _load_current(args.current, config)

    report = detect_drift(reference, current, feature_names, config, args.bins)
    logger.info(report.summary())

    ranked = sorted(report.features, key=lambda feature: feature.psi, reverse=True)
    print(f"\n{'feature':<16}{'PSI':>10}{'KS p':>12}{'mean shift':>14}  status")
    print("-" * 66)
    for feature in ranked[: args.top]:
        print(
            f"{feature.feature:<16}{feature.psi:>10.4f}{feature.ks_pvalue:>12.4g}"
            f"{feature.mean_shift:>14.4f}  {feature.severity}"
        )

    output = Path(args.output) if args.output else config.path("reports") / "drift_report.json"
    report.save(output)

    if not args.no_alerts and report.status != "stable":
        from monitoring.alerts import Alert, AlertManager, Severity

        AlertManager(config).emit(
            Alert(
                title=f"Sensor drift detected ({report.status})",
                message=(
                    f"{len(report.drifted_features)} of {report.n_features} features "
                    f"drifted vs the {args.reference} distribution: "
                    f"{', '.join(report.drifted_features[:8]) or 'none'}"
                ),
                severity=Severity.CRITICAL if report.status == "critical" else Severity.WARNING,
                source="drift_monitor",
                metric="max_psi",
                value=report.max_psi,
                threshold=report.thresholds["psi_warning"],
                metadata={"drifted_features": report.drifted_features},
            )
        )

    return 0 if report.status == "stable" else 1


def _load_current(target: str, config: Config) -> np.ndarray:
    """Resolve ``--current`` as either a split name or an ``.npz`` path."""
    path = Path(target)
    if path.suffix == ".npz":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        with np.load(path) as payload:
            return payload["X"]

    from src.ingestion.loader import load_sequences

    X, _ = load_sequences(target, config)
    return X


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
