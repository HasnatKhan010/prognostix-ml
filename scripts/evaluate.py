#!/usr/bin/env python
"""Evaluate every trained model and refresh the leaderboard.

Two evaluation modes:

* **windowed** (default) - score every sliding window in a split. Many
  predictions per engine, so the metrics are stable and comparable across models.
* **official** (``--official``) - the CMAPSS competition protocol: one prediction
  per test engine from its final window, scored against ``RUL_FD001.txt``. Fewer
  points and harder, but it is the number the literature quotes.

Usage::

    python scripts/evaluate.py                    # all models, windowed test split
    python scripts/evaluate.py --split val
    python scripts/evaluate.py --official --model gru
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, setup_logging  # noqa: E402
from src.evaluation.compare import build_leaderboard, save_leaderboard  # noqa: E402
from src.evaluation.metrics import evaluate_model  # noqa: E402
from src.evaluation.plots import (  # noqa: E402
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_residuals,
)
from src.ingestion.loader import load_processed, load_sequences  # noqa: E402
from src.inference.predictor import ModelRegistry  # noqa: E402

logger = logging.getLogger("prognostix.evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Evaluate a single model instead of everything on disk.",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"], help="Split to score."
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Score one prediction per test engine against the ground-truth RUL file.",
    )
    parser.add_argument(
        "--output",
        default="model_comparison.csv",
        help="Leaderboard filename under artifacts/ (default: model_comparison.csv).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip figures.")
    args = parser.parse_args(argv)

    setup_logging()
    config = get_config()
    registry = ModelRegistry(config)

    available = registry.available()
    if not available:
        logger.error(
            "No trained model in %s. Run `python scripts/train.py --model all` first.",
            config.path("models"),
        )
        return 1

    targets = [args.model] if args.model else available
    unknown = [name for name in targets if name not in available]
    if unknown:
        logger.error("Not trained: %s. Available: %s", unknown, available)
        return 1

    X, y_true, engine_ids = _load_evaluation_data(args, config)
    logger.info(
        "Evaluating %d model(s) on %d %s window(s)",
        len(targets),
        len(X),
        "official test" if args.official else args.split,
    )

    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}

    for name in targets:
        try:
            predictor = registry.get(name)
            # Windows from data/processed are already scaled; the official test
            # frame is raw and must go through the model's scaler.
            y_pred = predictor.predict(X, scaled=not args.official)
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.error("Skipping %s: %s", name, exc)
            continue

        predictions[name] = y_pred
        row = evaluate_model(name, y_true, y_pred)
        row["Split"] = "official_test" if args.official else args.split
        rows.append(row)
        logger.info(
            "%-14s MAE %7.3f | RMSE %7.3f | R2 %6.3f | NASA %10.1f",
            name,
            row["MAE"],
            row["RMSE"],
            row["R2"],
            row["NASAScore"],
        )

    if not rows:
        logger.error("Every model failed to evaluate")
        return 1

    leaderboard = build_leaderboard(rows)
    save_leaderboard(leaderboard, config=config, filename=args.output)

    best = str(leaderboard.iloc[0]["Model"])
    print("\n" + "=" * 74)
    print(f"EVALUATION - {'official test protocol' if args.official else args.split + ' split'}")
    print("=" * 74)
    columns = [c for c in ("Model", "MAE", "RMSE", "R2", "MAPE", "Within10", "NASAScore") if c in leaderboard.columns]
    print(leaderboard[columns].round(3).to_string(index=False))
    print(f"\nBest by RMSE: {best}")

    if not args.no_plots:
        _write_figures(leaderboard, predictions, y_true, best, engine_ids, config, args)

    return 0


def _load_evaluation_data(args: argparse.Namespace, config):
    """Return ``(X, y_true, engine_ids)`` for the requested evaluation mode."""
    data = config.data

    if not args.official:
        path = config.path("data_processed") / f"{args.split}_sequences.npz"
        with np.load(path) as payload:
            engine_ids = payload["engine_ids"] if "engine_ids" in payload else None
        X, y = load_sequences(args.split, config)
        return X, y, engine_ids

    from src.preprocessing.sequences import last_window_per_engine

    frame = load_processed(f"test_{data.dataset}_processed")
    truth = load_processed(f"RUL_{data.dataset}_processed")

    try:
        from src.preprocessing.scaling import load_scaler

        feature_columns = load_scaler(config=config).feature_columns
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nRun `python scripts/prepare_data.py` before the official evaluation."
        ) from exc

    X, engine_ids = last_window_per_engine(
        frame,
        feature_columns=feature_columns,
        window_size=int(data.window_size),
        id_column=data.id_column,
        time_column=data.time_column,
    )

    if "engine_id" in truth.columns:
        y_true = truth.set_index("engine_id")["RUL"].reindex(engine_ids).to_numpy(dtype=float)
    else:  # ground-truth file is ordered by engine number
        y_true = truth["RUL"].to_numpy(dtype=float)[: len(engine_ids)]

    if np.isnan(y_true).any():
        missing = engine_ids[np.isnan(y_true)]
        raise SystemExit(f"No ground-truth RUL for engine(s): {missing.tolist()}")

    cap = data.get("rul_cap")
    if cap is not None:
        y_true = np.clip(y_true, None, float(cap))
    return X, y_true, engine_ids


def _write_figures(
    leaderboard: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    best: str,
    engine_ids,
    config,
    args: argparse.Namespace,
) -> None:
    """Save the comparison chart plus diagnostics for the winning model."""
    import matplotlib.pyplot as plt

    figures = config.path("figures")
    suffix = "official" if args.official else args.split
    try:
        plot_model_comparison(
            leaderboard,
            metric="RMSE",
            title=f"Model comparison - {suffix}",
            save_path=figures / f"model_comparison_{suffix}.png",
        )
        best_key = best.lower()
        if best_key in predictions:
            plot_actual_vs_predicted(
                y_true,
                predictions[best_key],
                title=f"{best} - actual vs predicted RUL ({suffix})",
                save_path=figures / f"{best_key}_actual_vs_predicted_{suffix}.png",
            )
            plot_residuals(
                y_true,
                predictions[best_key],
                title=f"{best} residuals ({suffix})",
                save_path=figures / f"{best_key}_residuals_{suffix}.png",
            )
        logger.info("Figures written to %s", figures)
    finally:
        plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
