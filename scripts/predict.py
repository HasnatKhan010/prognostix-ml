#!/usr/bin/env python
"""Score machines with a trained model and write a maintenance worklist.

Default behaviour scores every engine in the processed test frame from its most
recent window - the same shape a live deployment sends - and writes one row per
engine with predicted RUL, health score, risk band and recommended action.

Usage::

    python scripts/predict.py                          # official test frame
    python scripts/predict.py --input data/live.csv --alerts
    python scripts/predict.py --split test --model lstm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, setup_logging
from src.inference.health_score import assess_health
from src.inference.predictor import RULPredictor
from src.ingestion.loader import load_processed, load_sequences

logger = logging.getLogger("prognostix.predict")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Model to serve with.")
    parser.add_argument(
        "--input",
        default=None,
        help="Cycle-level CSV with engine_id, cycle and sensor columns.",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "val", "test"],
        help="Score prepared windows from data/processed instead of a CSV.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination CSV (default: artifacts/reports/predictions.csv).",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Rows to print, most urgent first."
    )
    parser.add_argument(
        "--alerts", action="store_true", help="Raise alerts for at-risk machines."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    args = parser.parse_args(argv)

    setup_logging()
    config = get_config()

    try:
        predictor = RULPredictor(model_name=args.model, config=config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    try:
        frame = (
            _predict_split(predictor, args.split, config)
            if args.split
            else _predict_frame(predictor, args.input, config)
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if frame.empty:
        logger.error("Nothing to score - no engine had enough cycles")
        return 1

    output = Path(args.output) if args.output else config.path("reports") / "predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    logger.info("Wrote %d prediction(s) -> %s", len(frame), output)

    ranked = frame.sort_values("rul")
    if args.json:
        print(json.dumps(ranked.head(args.top).to_dict(orient="records"), indent=2, default=str))
    else:
        _print_report(ranked, args.top, predictor.model_name)

    if args.alerts:
        _raise_alerts(ranked, config)

    return 0


def _predict_frame(
    predictor: RULPredictor, input_path: str | None, config
) -> pd.DataFrame:
    """Score every engine in a cycle-level frame using its latest window."""
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        frame = pd.read_csv(path)
        source = path.name
    else:
        frame = load_processed(f"test_{config.data.dataset}_processed", config)
        source = f"test_{config.data.dataset}_processed.csv"

    logger.info("Scoring %s (%d rows)", source, len(frame))
    result = predictor.predict_frame(frame, scaled=False)

    truth_column = config.data.target_column
    if truth_column in frame.columns and not result.empty:
        # The last recorded cycle of each engine carries its true remaining life.
        last = (
            frame.sort_values(config.data.time_column)
            .groupby(config.data.id_column)[truth_column]
            .last()
        )
        result["actual_rul"] = result["engine_id"].map(last)
        result["error"] = (result["rul"] - result["actual_rul"]).round(2)

    result.insert(0, "source", source)
    return result


def _predict_split(predictor: RULPredictor, split: str, config) -> pd.DataFrame:
    """Score prepared windows from ``data/processed`` (already scaled)."""
    X, y = load_sequences(split, config)
    path = config.path("data_processed") / f"{split}_sequences.npz"
    with np.load(path) as payload:
        engine_ids = payload.get("engine_ids", None)

    predictions = predictor.predict(X, scaled=True)
    assessments = [
        assess_health(value, config=config, model=predictor.model_name)
        for value in predictions
    ]

    return pd.DataFrame(
        {
            "source": f"{split}_sequences.npz",
            "engine_id": engine_ids if engine_ids is not None else np.arange(len(predictions)),
            "window_index": np.arange(len(predictions)),
            "rul": np.round(predictions, 2),
            "health_score": [round(a.health_score, 2) for a in assessments],
            "risk_level": [a.risk_level.value for a in assessments],
            "recommended_action": [a.recommended_action for a in assessments],
            "actual_rul": y,
            "error": np.round(predictions - y, 2),
            "model": predictor.model_name,
        }
    )


def _print_report(ranked: pd.DataFrame, top: int, model: str) -> None:
    """Print the fleet rollup and the most urgent machines."""
    counts = ranked["risk_level"].value_counts().to_dict()
    order = ["critical", "warning", "watch", "healthy"]

    print("\n" + "=" * 72)
    print(f"MAINTENANCE WORKLIST - model: {model} | {len(ranked)} machine(s)")
    print("=" * 72)
    print("  ".join(f"{level}: {counts.get(level, 0)}" for level in order))

    has_truth = "actual_rul" in ranked.columns
    header = f"\n{'engine':>8}{'RUL':>10}{'health':>9}  {'risk':<10}"
    if has_truth:
        header += f"{'actual':>9}{'error':>9}"
    print(header)
    print("-" * (len(header) + 4))

    for _, row in ranked.head(top).iterrows():
        line = (
            f"{row.get('engine_id', '-')!s:>8}{row['rul']:>10.1f}"
            f"{row['health_score']:>9.1f}  {row['risk_level']:<10}"
        )
        if has_truth and pd.notna(row.get("actual_rul")):
            line += f"{row['actual_rul']:>9.1f}{row['error']:>9.1f}"
        print(line)

    if has_truth:
        errors = ranked["error"].dropna()
        if not errors.empty:
            rmse = float(np.sqrt((errors**2).mean()))
            print(f"\nMAE {errors.abs().mean():.2f} | RMSE {rmse:.2f} cycles")


def _raise_alerts(ranked: pd.DataFrame, config) -> None:
    """Emit alerts for every machine in the warning band or worse."""
    from monitoring.alerts import Alert, AlertManager, Severity

    manager = AlertManager(config)
    at_risk = ranked[ranked["risk_level"].isin(["warning", "critical"])]

    for _, row in at_risk.iterrows():
        manager.emit(
            Alert(
                title=f"{str(row['risk_level']).title()} RUL for engine {row.get('engine_id', '?')}",
                message=str(row["recommended_action"]),
                severity=Severity.CRITICAL if row["risk_level"] == "critical" else Severity.WARNING,
                source="predict_cli",
                metric="rul_cycles",
                value=float(row["rul"]),
                entity=row.get("engine_id"),
                metadata={"health_score": float(row["health_score"])},
            )
        )

    logger.info("Raised %d alert(s) for %d at-risk machine(s)", len(manager.sent), len(at_risk))


if __name__ == "__main__":
    raise SystemExit(main())
