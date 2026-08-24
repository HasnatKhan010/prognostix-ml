"""Model performance tracking and degradation detection.

Drift watches the inputs; this module watches the outputs once ground truth
arrives. Each evaluation is appended to ``artifacts/reports/performance_log.csv``
so the trend is visible over time, and current error is compared against the
baseline recorded at training. Crossing the configured degradation thresholds
raises an alert - that is the signal to retrain.

Run as ``python -m monitoring.performance``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, get_config, setup_logging
from src.evaluation.metrics import regression_metrics

logger = logging.getLogger(__name__)

__all__ = [
    "PerformanceReport",
    "append_history",
    "baseline_metrics",
    "load_history",
    "track_performance",
]

LOG_FILENAME = "performance_log.csv"


@dataclass
class PerformanceReport:
    """Current error, the baseline it is judged against, and the verdict."""

    status: str  # ok | warning | critical
    model: str
    metrics: dict[str, float] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    degradation_pct: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    thresholds: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    note: str | None = None

    @property
    def degraded(self) -> bool:
        return self.status != "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "status": self.status,
            "n_samples": self.n_samples,
            **{key: round(value, 6) for key, value in self.metrics.items() if isinstance(value, (int, float))},
            **{f"baseline_{k}": round(v, 6) for k, v in self.baseline.items()},
            **{f"degradation_{k}_pct": round(v, 3) for k, v in self.degradation_pct.items()},
            **({"note": self.note} if self.note else {}),
        }

    def summary(self) -> str:
        current = self.metrics.get("RMSE", float("nan"))
        reference = self.baseline.get("RMSE")
        delta = self.degradation_pct.get("RMSE")
        text = f"[{self.status.upper()}] {self.model}: RMSE {current:.3f}"
        if reference is not None:
            text += f" vs baseline {reference:.3f}"
        if delta is not None:
            text += f" ({delta:+.1f}%)"
        return f"{text} on {self.n_samples} sample(s)"


def baseline_metrics(
    model: str, config: Config | None = None
) -> dict[str, float]:
    """Reference metrics for a model.

    Configured values under ``monitoring.performance`` win; otherwise the model's
    row in the leaderboard is used, and failing that the best row available.
    """
    config = config or get_config()
    settings = config.monitoring.performance

    configured = {
        "RMSE": settings.get("baseline_rmse"),
        "MAE": settings.get("baseline_mae"),
    }
    resolved = {key: float(value) for key, value in configured.items() if value is not None}
    if resolved:
        return resolved

    from src.evaluation.compare import load_leaderboard

    leaderboard = load_leaderboard(config=config)
    if leaderboard.empty or "RMSE" not in leaderboard.columns:
        return {}

    matches = leaderboard[
        leaderboard["Model"].astype(str).str.lower() == model.lower()
    ]
    row = (
        matches.iloc[0]
        if not matches.empty
        else leaderboard.loc[leaderboard["RMSE"].idxmin()]
    )
    return {
        key: float(row[key])
        for key in ("RMSE", "MAE")
        if key in row.index and pd.notna(row[key])
    }


def track_performance(
    y_true,
    y_pred,
    model: str = "gru",
    config: Config | None = None,
    baseline: dict[str, float] | None = None,
    save: bool = True,
    raise_alerts: bool = True,
) -> PerformanceReport:
    """Score a batch of predictions against ground truth and judge the trend.

    Parameters
    ----------
    baseline:
        Overrides the resolved reference metrics.
    save:
        Append the result to the performance log.
    raise_alerts:
        Emit an alert when a degradation threshold is crossed.
    """
    config = config or get_config()
    settings = config.monitoring.performance
    warning_pct = float(settings.get("degradation_warning_pct", 10.0))
    critical_pct = float(settings.get("degradation_critical_pct", 25.0))
    min_samples = int(settings.get("min_samples", 50))

    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    metrics = regression_metrics(true, pred)
    reference = baseline if baseline is not None else baseline_metrics(model, config)

    degradation: dict[str, float] = {}
    for key in ("RMSE", "MAE"):
        if key in reference and reference[key]:
            degradation[key] = (metrics[key] - reference[key]) / reference[key] * 100.0

    note: str | None = None
    worst = max(degradation.values(), default=0.0)
    if len(true) < min_samples:
        status = "ok"
        note = (
            f"Only {len(true)} sample(s); {min_samples} required before a "
            "degradation verdict is meaningful"
        )
    elif not degradation:
        status = "ok"
        note = "No baseline available for comparison"
    elif worst >= critical_pct:
        status = "critical"
    elif worst >= warning_pct:
        status = "warning"
    else:
        status = "ok"

    report = PerformanceReport(
        status=status,
        model=model,
        metrics=metrics,
        baseline=reference,
        degradation_pct=degradation,
        n_samples=int(len(true)),
        thresholds={
            "degradation_warning_pct": warning_pct,
            "degradation_critical_pct": critical_pct,
            "min_samples": min_samples,
        },
        note=note,
    )

    logger.info(report.summary())
    if note:
        logger.info(note)
    if save:
        append_history(report, config=config)
    if raise_alerts and report.degraded:
        _raise_alert(report, config)

    return report


def append_history(
    report: PerformanceReport,
    config: Config | None = None,
    filename: str = LOG_FILENAME,
) -> Path:
    """Append one report row to the performance log CSV."""
    config = config or get_config()
    path = config.path("reports") / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([report.to_dict()])
    header = not path.exists()
    row.to_csv(path, mode="a", header=header, index=False)
    logger.debug("Appended performance row -> %s", path)
    return path


def load_history(
    config: Config | None = None, filename: str = LOG_FILENAME
) -> pd.DataFrame:
    """Read the performance log, or an empty frame when absent."""
    config = config or get_config()
    path = config.path("reports") / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _raise_alert(report: PerformanceReport, config: Config) -> None:
    """Emit a degradation alert."""
    from monitoring.alerts import Alert, AlertManager, Severity

    AlertManager(config).emit(
        Alert(
            title=f"Model performance degraded ({report.status})",
            message=(
                f"{report.model} RMSE {report.metrics['RMSE']:.3f} vs baseline "
                f"{report.baseline.get('RMSE', float('nan')):.3f} "
                f"({report.degradation_pct.get('RMSE', 0.0):+.1f}%). Consider retraining."
            ),
            severity=Severity.CRITICAL if report.status == "critical" else Severity.WARNING,
            source="performance_monitor",
            metric="RMSE",
            value=report.metrics["RMSE"],
            threshold=report.baseline.get("RMSE"),
            entity=report.model,
            metadata={"n_samples": report.n_samples},
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: score a model on a labelled split and log the result."""
    parser = argparse.ArgumentParser(
        description="Track model performance against its training baseline."
    )
    parser.add_argument("--model", default=None, help="Model to evaluate.")
    parser.add_argument(
        "--split", default="test", help="Labelled split to score (default: test)."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not append to the performance log."
    )
    parser.add_argument(
        "--no-alerts", action="store_true", help="Do not raise degradation alerts."
    )
    args = parser.parse_args(argv)

    setup_logging()
    config = get_config()

    from src.ingestion.loader import load_sequences
    from src.inference.predictor import RULPredictor

    model_name = args.model or str(config.inference.default_model)
    X, y = load_sequences(args.split, config)
    predictor = RULPredictor(model_name=model_name, config=config)

    # Sequences on disk are already scaled by the preprocessing pipeline.
    predictions = predictor.predict(X, scaled=True)

    report = track_performance(
        y,
        predictions,
        model=model_name,
        config=config,
        save=not args.no_save,
        raise_alerts=not args.no_alerts,
    )
    print(report.summary())
    return 0 if not report.degraded else 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
